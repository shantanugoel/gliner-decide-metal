import Dispatch
import Foundation
import MLX
import MLXFast
import MLXNN

public struct ProfileEvent: Sendable, Codable {
    public let name: String
    public let nanoseconds: UInt64

    public var milliseconds: Double { Double(nanoseconds) / 1_000_000 }
}

public struct InferenceResult {
    public let logits: MLXArray
    public let wallNanoseconds: UInt64
    public let events: [ProfileEvent]

    public var wallMilliseconds: Double { Double(wallNanoseconds) / 1_000_000 }
}

public struct DecisionInput {
    public let inputIDs: MLXArray
    public let attentionMask: MLXArray
    public let markerIndices: MLXArray
    public let markerMask: MLXArray

    public init(
        inputIDs: MLXArray,
        attentionMask: MLXArray,
        markerIndices: MLXArray,
        markerMask: MLXArray
    ) {
        self.inputIDs = inputIDs
        self.attentionMask = attentionMask
        self.markerIndices = markerIndices
        self.markerMask = markerMask
    }

    public init(url: URL) throws {
        let arrays = try loadArrays(url: url)
        guard let inputIDs = arrays["input_ids"],
            let attentionMask = arrays["attention_mask"],
            let markerIndices = arrays["marker_indices"],
            let markerMask = arrays["marker_mask"]
        else {
            throw WeightStoreError.invalidInput(
                "Input safetensors must contain input_ids, attention_mask, marker_indices, and marker_mask"
            )
        }
        self.init(
            inputIDs: inputIDs,
            attentionMask: attentionMask,
            markerIndices: markerIndices,
            markerMask: markerMask
        )
    }
}

private struct LinearWeight {
    let weight: MLXArray
    let bias: MLXArray

    init(_ store: WeightStore, _ prefix: String) throws {
        weight = try store.array("\(prefix).weight")
        bias = try store.array("\(prefix).bias")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        addMM(bias, x, weight.T)
    }
}

private struct FusedProjectionWeight {
    let weight: MLXArray
    let bias: MLXArray

    init(_ store: WeightStore, prefixes: [String]) throws {
        weight = try concatenated(
            prefixes.map { try store.array("\($0).weight") },
            axis: 0
        )
        bias = try concatenated(
            prefixes.map { try store.array("\($0).bias") },
            axis: 0
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        addMM(bias, x, weight.T)
    }
}

private struct LayerNormWeight {
    let weight: MLXArray
    let bias: MLXArray

    init(_ store: WeightStore, _ prefix: String) throws {
        weight = try store.array("\(prefix).weight")
        bias = try store.array("\(prefix).bias")
    }

    func callAsFunction(_ x: MLXArray, eps: Float) -> MLXArray {
        MLXFast.layerNorm(x, weight: weight, bias: bias, eps: eps)
    }
}

private struct RelativeProjection {
    let query: MLXArray
    let key: MLXArray
}

private struct RelativeData {
    let c2pIndices: MLXArray
    let p2cIndices: MLXArray
}

private struct DebertaLayerWeight {
    let qkv: FusedProjectionWeight
    let relativeQK: FusedProjectionWeight
    let attentionOutput: LinearWeight
    let attentionLayerNorm: LayerNormWeight
    let intermediate: LinearWeight
    let output: LinearWeight
    let outputLayerNorm: LayerNormWeight

    init(_ store: WeightStore, index: Int) throws {
        let p = "encoder.encoder.layer.\(index)"
        qkv = try FusedProjectionWeight(
            store,
            prefixes: [
                "\(p).attention.self.query_proj",
                "\(p).attention.self.key_proj",
                "\(p).attention.self.value_proj",
            ]
        )
        relativeQK = try FusedProjectionWeight(
            store,
            prefixes: [
                "\(p).attention.self.query_proj",
                "\(p).attention.self.key_proj",
            ]
        )
        attentionOutput = try LinearWeight(store, "\(p).attention.output.dense")
        attentionLayerNorm = try LayerNormWeight(store, "\(p).attention.output.LayerNorm")
        intermediate = try LinearWeight(store, "\(p).intermediate.dense")
        output = try LinearWeight(store, "\(p).output.dense")
        outputLayerNorm = try LayerNormWeight(store, "\(p).output.LayerNorm")
    }
}

/// Classification-only DeBERTa-v2 encoder used by GLiNER2.5-Decide.
///
/// This intentionally omits GLiNER's span, relation, and count heads. The
/// upstream model is still exactly the same encoder and classification MLP.
public final class GLiNERDecideClassifier: @unchecked Sendable {
    public let configuration: DeBERTaV2Configuration

    private let wordEmbedding: MLXArray
    private let embeddingLayerNorm: LayerNormWeight
    private let relativeEmbedding: MLXArray
    private let relativeLayerNorm: LayerNormWeight
    private let normalizedRelativeEmbedding: MLXArray
    private let relativeProjections: [RelativeProjection]
    private var relativeCache: [Int: RelativeData] = [:]
    private let layers: [DebertaLayerWeight]
    private let classifier0: LinearWeight
    private let classifier2: LinearWeight
    private let fusedKernels: FusedKernels?
    private let useCompiledGraph: Bool
    private let useFastAttention: Bool
    private let useCustomRelativeKernel: Bool
    private let useFusedAttentionKernel: Bool

    private lazy var compiledForward: @Sendable ([MLXArray]) -> [MLXArray] = {
        compile { [self] arrays in
            let input = DecisionInput(
                inputIDs: arrays[0],
                attentionMask: arrays[1],
                markerIndices: arrays[2],
                markerMask: arrays[3]
            )
            var ignoredEvents: [ProfileEvent] = []
            return [forward(input, collectProfile: false, events: &ignoredEvents)]
        }
    }()

    public init(
        weightsURL: URL,
        configuration: DeBERTaV2Configuration = .default,
        useFusedKernels: Bool = false,
        useCompiledGraph: Bool = false,
        useFastAttention: Bool = true,
        useCustomRelativeKernel: Bool = false,
        useFusedAttentionKernel: Bool = false
    ) throws {
        let store = try WeightStore(url: weightsURL)
        self.configuration = configuration
        wordEmbedding = try store.array(
            "encoder.embeddings.word_embeddings.weight",
            shape: [128011, configuration.hiddenSize]
        )
        embeddingLayerNorm = try LayerNormWeight(store, "encoder.embeddings.LayerNorm")
        relativeEmbedding = try store.array(
            "encoder.encoder.rel_embeddings.weight",
            shape: [512, configuration.hiddenSize]
        )
        relativeLayerNorm = try LayerNormWeight(store, "encoder.encoder.LayerNorm")
        let normalizedRelative = relativeLayerNorm(
            relativeEmbedding.reshaped([1, 512, configuration.hiddenSize]),
            eps: configuration.layerNormEps
        )
        normalizedRelativeEmbedding = normalizedRelative
        let loadedLayers = try (0 ..< configuration.numHiddenLayers).map {
            try DebertaLayerWeight(store, index: $0)
        }
        layers = loadedLayers
        let loadedRelativeProjections = loadedLayers.map { layer in
            let parts = layer.relativeQK(normalizedRelative)
                .reshaped([1, 512, 2, configuration.numAttentionHeads, configuration.headSize])
                .split(parts: 2, axis: 2)
            return RelativeProjection(
                query: parts[0].squeezed(axis: 2).transposed(0, 2, 1, 3),
                key: parts[1].squeezed(axis: 2).transposed(0, 2, 1, 3)
            )
        }
        relativeProjections = loadedRelativeProjections
        classifier0 = try LinearWeight(store, "classifier.0")
        classifier2 = try LinearWeight(store, "classifier.2")
        fusedKernels = (useFusedKernels || useCustomRelativeKernel || useFusedAttentionKernel)
            ? FusedKernels()
            : nil
        self.useCompiledGraph = useCompiledGraph
        self.useFastAttention = useFastAttention
        self.useCustomRelativeKernel = useCustomRelativeKernel
        self.useFusedAttentionKernel = useFusedAttentionKernel
        normalizedRelative.eval()
        for projection in loadedRelativeProjections {
            projection.query.eval()
            projection.key.eval()
        }
    }

    public func classify(
        _ input: DecisionInput,
        collectProfile: Bool = false
    ) -> InferenceResult {
        let start = DispatchTime.now().uptimeNanoseconds
        if !collectProfile && useCompiledGraph {
            let logits = compiledForward([
                input.inputIDs,
                input.attentionMask,
                input.markerIndices,
                input.markerMask,
            ])[0]
            logits.eval()
            return InferenceResult(
                logits: logits,
                wallNanoseconds: DispatchTime.now().uptimeNanoseconds - start,
                events: []
            )
        }

        var events: [ProfileEvent] = []
        let logits = forward(input, collectProfile: collectProfile, events: &events)
        logits.eval()
        return InferenceResult(
            logits: logits,
            wallNanoseconds: DispatchTime.now().uptimeNanoseconds - start,
            events: events
        )
    }

    private func forward(
        _ input: DecisionInput,
        collectProfile: Bool,
        events: inout [ProfileEvent]
    ) -> MLXArray {

        let batch = input.inputIDs.shape[0]
        let sequenceLength = input.inputIDs.shape[1]
        let maxHeads = input.markerIndices.shape[1]
        let maxOptions = input.markerIndices.shape[2]

        var hidden = stage("embedding", collectProfile, &events) {
            let ids = input.inputIDs
            let mask = input.attentionMask.asType(.float16)
            var x = wordEmbedding[ids]
            x = embeddingLayerNorm(x, eps: configuration.layerNormEps)
            x = x * mask.reshaped([batch, sequenceLength, 1])
            return x
        }

        let relative = cachedRelativeData(sequenceLength: sequenceLength)

        for (index, layer) in layers.enumerated() {
            let relativeProjection = relativeProjections[index]
            let projected = stage3(
                "layer_\(index)_attention_projection",
                collectProfile,
                &events
            ) {
                let parts = layer.qkv(hidden)
                    .reshaped([
                        batch,
                        sequenceLength,
                        3,
                        configuration.numAttentionHeads,
                        configuration.headSize,
                    ])
                    .split(parts: 3, axis: 2)
                return (
                    parts[0].squeezed(axis: 2).transposed(0, 2, 1, 3),
                    parts[1].squeezed(axis: 2).transposed(0, 2, 1, 3),
                    parts[2].squeezed(axis: 2).transposed(0, 2, 1, 3)
                )
            }
            let q = projected.0
            let k = projected.1
            let v = projected.2

            let attentionContext = stage("layer_\(index)_attention", collectProfile, &events) {
                let scale = MLXArray(
                    Float(configuration.headSize * 3).squareRoot(),
                    dtype: .float16
                )
                let positionQuery = relativeProjection.query
                let positionKey = relativeProjection.key
                let valid = input.attentionMask.asType(.float16) .> MLXArray(0, dtype: .float16)
                let keyMask = valid.reshaped([batch, 1, 1, sequenceLength])
                let additiveKeyMask = `where`(
                    keyMask,
                    MLXArray(0.0, dtype: .float16),
                    MLXArray(-65_504.0, dtype: .float16)
                )

                if useFusedAttentionKernel {
                    return fusedKernels!.fusedAttention(
                        queries: q,
                        keys: k,
                        values: v,
                        positionKey: positionKey,
                        positionQuery: positionQuery,
                        c2pIndices: relative.c2pIndices,
                        p2cIndices: relative.p2cIndices,
                        keyMask: additiveKeyMask,
                        scale: 1.0 / Float(configuration.headSize * 3).squareRoot(),
                        batch: batch,
                        heads: configuration.numAttentionHeads,
                        sequenceLength: sequenceLength,
                        headDimension: configuration.headSize,
                        positionCount: 512
                    ).reshaped([batch, sequenceLength, configuration.hiddenSize])
                }

                let relativeBias: MLXArray
                if useCustomRelativeKernel {
                    relativeBias = fusedKernels!.relativeBias(
                        q: q,
                        k: k,
                        positionKey: positionKey,
                        positionQuery: positionQuery,
                        c2pIndices: relative.c2pIndices,
                        p2cIndices: relative.p2cIndices,
                        keyMask: additiveKeyMask,
                        scale: 1.0 / Float(configuration.headSize * 3).squareRoot(),
                        batch: batch,
                        heads: configuration.numAttentionHeads,
                        sequenceLength: sequenceLength,
                        headDimension: configuration.headSize,
                        positionCount: 512
                    )
                } else {
                    let c2pRaw = q.matmul(positionKey.transposed(0, 1, 3, 2))
                    let c2p = takeAlong(c2pRaw, relative.c2pIndices, axis: -1)
                    let p2cRaw = k.matmul(positionQuery.transposed(0, 1, 3, 2))
                    let p2c = takeAlong(p2cRaw, relative.p2cIndices, axis: -1)
                        .transposed(0, 1, 3, 2)
                    relativeBias = (c2p + p2c) / scale
                }

                if useFastAttention {
                    let additiveMask = useCustomRelativeKernel
                        ? relativeBias
                        : relativeBias + additiveKeyMask
                    return MLXFast.scaledDotProductAttention(
                        queries: q,
                        keys: k,
                        values: v,
                        scale: 1.0 / Float(configuration.headSize * 3).squareRoot(),
                        mask: .array(additiveMask),
                        forceFused: false
                    ).transposed(0, 2, 1, 3)
                        .reshaped([batch, sequenceLength, configuration.hiddenSize])
                }

                let content = (q.matmul(k.transposed(0, 1, 3, 2))) / scale
                let scores = content + relativeBias
                let masked = useCustomRelativeKernel
                    ? scores
                    : `where`(
                        keyMask,
                        scores,
                        MLXArray(-65_504.0, dtype: scores.dtype)
                    )
                let probabilities = softmax(masked, axis: -1)
                let context = probabilities.matmul(v)
                return context.transposed(0, 2, 1, 3)
                    .reshaped([batch, sequenceLength, configuration.hiddenSize])
            }

            hidden = stage("layer_\(index)_attention_output", collectProfile, &events) {
                residualLayerNorm(
                    layer.attentionOutput(attentionContext),
                    hidden,
                    norm: layer.attentionLayerNorm
                )
            }

            hidden = stage("layer_\(index)_feed_forward", collectProfile, &events) {
                let intermediate = MLXNN.gelu(layer.intermediate(hidden))
                return residualLayerNorm(
                    layer.output(intermediate),
                    hidden,
                    norm: layer.outputLayerNorm
                )
            }
        }

        let logits = stage("classifier", collectProfile, &events) {
            let offsets = MLXArray(0 ..< batch, [batch, 1, 1])
                * MLXArray(sequenceLength, dtype: .int32)
            let flatIndices = (input.markerIndices + offsets).reshaped([-1])
            let flatHidden = hidden.reshaped([batch * sequenceLength, configuration.hiddenSize])
            let markers = take(flatHidden, flatIndices, axis: 0)
                .reshaped([batch, maxHeads, maxOptions, configuration.hiddenSize])
            let values = classifier2(MLXNN.relu(classifier0(markers))).squeezed(axis: -1)
            let markerMask = input.markerMask.asType(values.dtype)
            return `where`(
                markerMask .> MLXArray(0, dtype: values.dtype),
                values,
                MLXArray(Float(-10_000), dtype: values.dtype)
            )
        }

        return logits
    }

    private func residualLayerNorm(
        _ x: MLXArray,
        _ residual: MLXArray,
        norm: LayerNormWeight
    ) -> MLXArray {
        if let fusedKernels {
            return fusedKernels.residualLayerNorm(
                x,
                residual,
                weight: norm.weight,
                bias: norm.bias,
                eps: configuration.layerNormEps
            )
        }
        return norm(x + residual, eps: configuration.layerNormEps)
    }

    private func stage(
        _ name: String,
        _ enabled: Bool,
        _ events: inout [ProfileEvent],
        _ operation: () -> MLXArray
    ) -> MLXArray {
        guard enabled else { return operation() }
        let start = DispatchTime.now().uptimeNanoseconds
        let result = operation()
        result.eval()
        let elapsed = DispatchTime.now().uptimeNanoseconds - start
        events.append(ProfileEvent(name: name, nanoseconds: elapsed))
        return result
    }

    private func stage3(
        _ name: String,
        _ enabled: Bool,
        _ events: inout [ProfileEvent],
        _ operation: () -> (MLXArray, MLXArray, MLXArray)
    ) -> (MLXArray, MLXArray, MLXArray) {
        guard enabled else { return operation() }
        let start = DispatchTime.now().uptimeNanoseconds
        let result = operation()
        result.0.eval()
        result.1.eval()
        result.2.eval()
        let elapsed = DispatchTime.now().uptimeNanoseconds - start
        events.append(ProfileEvent(name: name, nanoseconds: elapsed))
        return result
    }

    private func cachedRelativeData(sequenceLength: Int) -> RelativeData {
        if let cached = relativeCache[sequenceLength] {
            return cached
        }
        let positions = makeRelativePositions(
            sequenceLength: sequenceLength,
            positionBuckets: configuration.positionBuckets,
            maxRelativePositions: 512
        )
        let c2p = MLX.broadcast(
            clip(positions + 256, min: 0, max: 511),
            to: [1, configuration.numAttentionHeads, sequenceLength, sequenceLength]
        )
        let p2cBase = positions[0].squeezed(axis: 0).transposed(0, 1)
        let p2c = MLX.broadcast(
            clip(-p2cBase + 256, min: 0, max: 511),
            to: [1, configuration.numAttentionHeads, sequenceLength, sequenceLength]
        )
        let result = RelativeData(c2pIndices: c2p, p2cIndices: p2c)
        relativeCache[sequenceLength] = result
        return result
    }

    private func makeRelativePositions(
        sequenceLength: Int,
        positionBuckets: Int,
        maxRelativePositions: Int
    ) -> MLXArray {
        let mid = positionBuckets / 2
        var values: [Int32] = []
        values.reserveCapacity(sequenceLength * sequenceLength)
        for q in 0 ..< sequenceLength {
            for k in 0 ..< sequenceLength {
                let relative = q - k
                let absPosition: Int
                if relative > -mid && relative < mid {
                    absPosition = mid - 1
                } else {
                    absPosition = abs(relative)
                }
                let logPosition =
                    Int(ceil(
                        log(Double(absPosition) / Double(mid))
                            / log(Double(maxRelativePositions - 1) / Double(mid))
                            * Double(mid - 1)
                    )) + mid
                let bucket = absPosition <= mid
                    ? relative
                    : logPosition * (relative < 0 ? -1 : 1)
                values.append(Int32(bucket))
            }
        }
        return MLXArray(values, [1, 1, sequenceLength, sequenceLength])
    }
}
