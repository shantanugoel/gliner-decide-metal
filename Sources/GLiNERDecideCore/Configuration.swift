import Foundation

public struct DeBERTaV2Configuration: Codable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let layerNormEps: Float
    public let positionBuckets: Int
    public let maxRelativePositions: Int
    public let hiddenAct: String
    public let padTokenId: Int

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        numHiddenLayers: Int,
        numAttentionHeads: Int,
        layerNormEps: Float,
        positionBuckets: Int,
        maxRelativePositions: Int,
        hiddenAct: String,
        padTokenId: Int
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.layerNormEps = layerNormEps
        self.positionBuckets = positionBuckets
        self.maxRelativePositions = maxRelativePositions
        self.hiddenAct = hiddenAct
        self.padTokenId = padTokenId
    }

    public var headSize: Int { hiddenSize / numAttentionHeads }

    public static let `default` = DeBERTaV2Configuration(
        hiddenSize: 1024,
        intermediateSize: 4096,
        numHiddenLayers: 24,
        numAttentionHeads: 16,
        layerNormEps: 1e-7,
        positionBuckets: 256,
        maxRelativePositions: -1,
        hiddenAct: "gelu",
        padTokenId: 0
    )
}
