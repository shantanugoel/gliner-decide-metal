import Foundation
import MLX

public enum WeightStoreError: Error, LocalizedError {
    case missingWeight(String)
    case invalidShape(key: String, expected: [Int], actual: [Int])
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .missingWeight(let key):
            return "Missing model weight: \(key)"
        case .invalidShape(let key, let expected, let actual):
            return "Weight \(key) has shape \(actual), expected \(expected)"
        case .invalidInput(let message):
            return message
        }
    }
}

/// Classification-only weights loaded from a FP16 safetensors file prepared by
/// `Tools/prepare_model.py`. The store deliberately omits GLiNER span/count heads.
public struct WeightStore {
    public let arrays: [String: MLXArray]

    public init(url: URL) throws {
        arrays = try loadArrays(url: url)
    }

    public init(arrays: [String: MLXArray]) {
        self.arrays = arrays
    }

    public func array(
        _ key: String,
        shape expectedShape: [Int]? = nil,
        dtype expectedDType: DType? = nil
    ) throws -> MLXArray {
        guard let value = arrays[key] else {
            throw WeightStoreError.missingWeight(key)
        }
        if let expectedShape, value.shape != expectedShape {
            throw WeightStoreError.invalidShape(
                key: key,
                expected: expectedShape,
                actual: value.shape
            )
        }
        if let expectedDType, value.dtype != expectedDType {
            throw WeightStoreError.invalidShape(
                key: key,
                expected: [expectedDType == .float16 ? 16 : 32],
                actual: [value.itemSize * 8]
            )
        }
        return value
    }
}
