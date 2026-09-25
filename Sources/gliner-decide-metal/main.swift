import Dispatch
import Foundation
import GLiNERDecideCore
import MLX
import MLXFast

struct Options {
    var weights = "Artifacts/decide-classification-fp16.safetensors"
    var inputs = "Artifacts/sample-inputs.safetensors"
    var warmup = 3
    var iterations = 20
    var profile = false
    var fused = false
    var kernelTest = false
    var relativeKernelTest = false
    var compiled = false
    var fastAttention = true
    var customRelative = false
    var batchSize = 0
    var output: String?

    init(_ arguments: [String]) {
        var index = 1
        while index < arguments.count {
            let argument = arguments[index]
            func nextValue() -> String {
                index += 1
                guard index < arguments.count else {
                    fatalError("Missing value for \(argument)")
                }
                return arguments[index]
            }
            switch argument {
            case "--weights": weights = nextValue()
            case "--inputs": inputs = nextValue()
            case "--warmup": warmup = Int(nextValue()) ?? warmup
            case "--iterations": iterations = Int(nextValue()) ?? iterations
            case "--output": output = nextValue()
            case "--profile": profile = true
            case "--fused": fused = true
            case "--kernel-test": kernelTest = true
            case "--relative-kernel-test": relativeKernelTest = true
            case "--compiled": compiled = true
            case "--fast-attention": fastAttention = true
            case "--generic-attention": fastAttention = false
            case "--custom-relative": customRelative = true
            case "--batch-size": batchSize = Int(nextValue()) ?? batchSize
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                fatalError("Unknown argument: \(argument)")
            }
            index += 1
        }
    }
}

func printUsage() {
    print("""
    GLiNER2.5-Decide classification-only MLX/Metal benchmark

    --weights PATH     FP16 classification weights
    --inputs PATH      safetensors containing input_ids, attention_mask,
                        marker_indices, marker_mask
    --warmup N         warmup iterations (default: 3)
    --iterations N     measured iterations (default: 20)
    --profile          evaluate and print per-stage timings
    --fused            use the custom residual+LayerNorm Metal kernel
    --kernel-test      compare the custom kernel with MLXFast.layerNorm
    --relative-kernel-test
                       compare the custom relative-bias kernel with MLX ops
    --compiled        compile and cache the full MLX graph
    --fast-attention  use MLX's fused scaled-dot-product-attention (default)
    --generic-attention
                       use the explicit DeBERTa score/softmax implementation
    --custom-relative
                       use the experimental custom relative-bias kernel
    --batch-size N     process all rows in input file in chunks of N
    --output PATH      write logits as safetensors
    """)
}

let options = Options(CommandLine.arguments)
setvbuf(stdout, nil, _IONBF, 0)

do {
    if options.relativeKernelTest {
        let b: Int = 1
        let h: Int = 2
        let l: Int = 8
        let d: Int = 4
        let p: Int = 8
        let q = MLXArray((0 ..< (b * h * l * d)).map { Float($0 % 13) / 13.0 })
            .asType(.float16).reshaped([b, h, l, d])
        let k = MLXArray((0 ..< (b * h * l * d)).map { Float(($0 * 7) % 17) / 17.0 })
            .asType(.float16).reshaped([b, h, l, d])
        let posK = MLXArray((0 ..< (h * p * d)).map { Float(($0 * 3) % 19) / 19.0 })
            .asType(.float16).reshaped([1, h, p, d])
        let posQ = MLXArray((0 ..< (h * p * d)).map { Float(($0 * 5) % 23) / 23.0 })
            .asType(.float16).reshaped([1, h, p, d])
        let c2p = MLXArray((0 ..< (h * l * l)).map { Int32($0 % p) }).reshaped([1, h, l, l])
        let p2c = MLXArray((0 ..< (h * l * l)).map { Int32(($0 * 3) % p) }).reshaped([1, h, l, l])
        let mask = MLXArray((0 ..< l).map { $0 < 6 ? Float(0) : Float(-65504) })
            .asType(.float16).reshaped([1, 1, 1, l])
        let scale = Float(Double(d * 3).squareRoot())
        let rawC2P = q.matmul(posK.transposed(0, 1, 3, 2))
        let rawP2C = k.matmul(posQ.transposed(0, 1, 3, 2))
        let expectedC2P = takeAlong(rawC2P, c2p, axis: -1)
        let expectedP2C = takeAlong(rawP2C, p2c, axis: -1)
            .transposed(0, 1, 3, 2)
        let expectedRelative = (expectedC2P + expectedP2C) / MLXArray(scale, dtype: .float16)
        let expected = `where`(
            mask .> MLXArray(-60000.0, dtype: .float16),
            expectedRelative,
            MLXArray(-65504.0, dtype: .float16)
        )
        let actual = FusedKernels().relativeBias(
            q: q,
            k: k,
            positionKey: posK,
            positionQuery: posQ,
            c2pIndices: c2p,
            p2cIndices: p2c,
            keyMask: mask,
            scale: 1.0 / scale,
            batch: b,
            heads: h,
            sequenceLength: l,
            headDimension: d,
            positionCount: p
        )
        expected.eval()
        actual.eval()
        print("expected: \(expected)")
        print("actual:   \(actual)")
        print("max_abs:  \(abs(expected - actual).max().item(Float.self))")
        exit(0)
    }

    if options.kernelTest {
        let values = (0 ..< 1024).map { Float($0) / 1024.0 }
        let x = MLXArray(values).asType(.float16).reshaped([1, 1, 1024])
        let residual = MLXArray(Array(repeating: Float(0), count: 1024)).asType(.float16).reshaped([1, 1, 1024])
        let weight = MLXArray(Array(repeating: Float(1), count: 1024)).asType(.float16)
        let bias = MLXArray(Array(repeating: Float(0), count: 1024)).asType(.float16)
        let summed = x + residual
        summed.eval()
        print("input:    \(x)")
        print("residual: \(residual)")
        print("summed:   \(summed)")
        let expected = MLXFast.layerNorm(
            summed,
            weight: weight,
            bias: bias,
            eps: 1e-7
        )
        let actual = FusedKernels().residualLayerNorm(
            x,
            residual,
            weight: weight,
            bias: bias,
            eps: 1e-7
        )
        expected.eval()
        actual.eval()
        let difference = abs(expected - actual)
        print("expected: \(expected)")
        print("actual:   \(actual)")
        print("max_abs:  \(difference.max().item(Float.self))")
        exit(0)
    }

    let weightsURL = URL(fileURLWithPath: options.weights)
    let inputsURL = URL(fileURLWithPath: options.inputs)
    let model = try GLiNERDecideClassifier(
        weightsURL: weightsURL,
        useFusedKernels: options.fused,
        useCompiledGraph: options.compiled,
        useFastAttention: options.fastAttention,
        useCustomRelativeKernel: options.customRelative
    )
    let input = try DecisionInput(url: inputsURL)

    print("device: \(Device.defaultDevice())")
    print("weights: \(weightsURL.path)")
    print("inputs: \(inputsURL.path)")

    if options.batchSize > 0 {
        let count = input.inputIDs.shape[0]
        precondition(count > 0, "batch input must contain at least one row")
        // Warm the shape-specific compiled graph before measuring throughput.
        if max(0, options.warmup) > 0 {
            let warmEnd = min(options.batchSize, count)
            let warmChunk = DecisionInput(
                inputIDs: input.inputIDs[0 ..< warmEnd],
                attentionMask: input.attentionMask[0 ..< warmEnd],
                markerIndices: input.markerIndices[0 ..< warmEnd],
                markerMask: input.markerMask[0 ..< warmEnd]
            )
            for _ in 0 ..< max(1, options.warmup) {
                _ = model.classify(warmChunk)
            }
        }
        let started = DispatchTime.now().uptimeNanoseconds
        var pieces: [MLXArray] = []
        var offset = 0
        while offset < count {
            let end = min(offset + options.batchSize, count)
            let chunk = DecisionInput(
                inputIDs: input.inputIDs[offset ..< end],
                attentionMask: input.attentionMask[offset ..< end],
                markerIndices: input.markerIndices[offset ..< end],
                markerMask: input.markerMask[offset ..< end]
            )
            let result = model.classify(chunk)
            pieces.append(result.logits[0 ..< (end - offset)])
            offset = end
        }
        let logits = concatenated(pieces, axis: 0)
        logits.eval()
        let elapsed = DispatchTime.now().uptimeNanoseconds - started
        print("batch_count: \(count)")
        print(String(format: "batch_total_ms: %.3f", Double(elapsed) / 1_000_000))
        print(String(format: "batch_rows_per_sec: %.3f", Double(count) / (Double(elapsed) / 1_000_000_000)))
        if let output = options.output {
            try save(arrays: ["logits": logits], url: URL(fileURLWithPath: output))
            print("output: \(output)")
        }
        exit(0)
    }

    for _ in 0 ..< max(0, options.warmup) {
        _ = model.classify(input)
    }

    var timings: [Double] = []
    var last: InferenceResult?
    for _ in 0 ..< max(1, options.iterations) {
        let result = model.classify(input, collectProfile: options.profile)
        timings.append(result.wallMilliseconds)
        last = result
    }

    let sorted = timings.sorted()
    let p50 = sorted[sorted.count / 2]
    let p95 = sorted[min(sorted.count - 1, Int(Double(sorted.count - 1) * 0.95))]
    print(String(format: "p50_ms: %.3f", p50))
    print(String(format: "p95_ms: %.3f", p95))
    print(String(format: "mean_ms: %.3f", timings.reduce(0, +) / Double(timings.count)))

    if let last, options.profile {
        print("profile:")
        for event in last.events {
            let name = event.name.padding(toLength: 42, withPad: " ", startingAt: 0)
            print(String(format: "  %@ %8.3f ms", name, event.milliseconds))
        }
    }

    if let last, let output = options.output {
        try save(
            arrays: ["logits": last.logits],
            url: URL(fileURLWithPath: output)
        )
        print("output: \(output)")
    }
} catch {
    fputs("error: \(error)\n", stderr)
    exit(1)
}
