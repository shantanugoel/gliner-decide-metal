import Testing
@testable import GLiNERDecideCore

@Test func decideConfigurationMatchesCheckpoint() {
    let configuration = DeBERTaV2Configuration.default
    #expect(configuration.hiddenSize == 1024)
    #expect(configuration.intermediateSize == 4096)
    #expect(configuration.numHiddenLayers == 24)
    #expect(configuration.numAttentionHeads == 16)
    #expect(configuration.headSize == 64)
}
