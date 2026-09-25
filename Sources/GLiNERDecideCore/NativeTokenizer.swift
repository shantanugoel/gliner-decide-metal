import Foundation
import MLX
import Tokenizers

public enum NativeTokenizerError: Error, LocalizedError {
    case missingTokenizerFiles(URL)
    case invalidTokenizerConfiguration
    case noTaskLabels
    case inputDoesNotFit([Int])

    public var errorDescription: String? {
        switch self {
        case .missingTokenizerFiles(let url):
            return "Tokenizer directory is missing tokenizer.json/tokenizer_config.json: \(url.path)"
        case .invalidTokenizerConfiguration:
            return "Could not decode tokenizer_config.json"
        case .noTaskLabels:
            return "Every decision task must contain at least one label"
        case .inputDoesNotFit(let buckets):
            return "Encoded schema/text does not fit any requested bucket: \(buckets)"
        }
    }
}

public struct DecisionTask: Codable, Sendable, Equatable {
    public let name: String
    public let labels: [String]
    public let multiLabel: Bool
    public let prompt: String?
    public let labelDescriptions: [String: String]?

    public init(
        name: String,
        labels: [String],
        multiLabel: Bool = false,
        prompt: String? = nil,
        labelDescriptions: [String: String]? = nil
    ) {
        self.name = name
        self.labels = labels
        self.multiLabel = multiLabel
        self.prompt = prompt
        self.labelDescriptions = labelDescriptions
    }
}

public struct DecisionSchema: Codable, Sendable {
    public let tasks: [DecisionTask]

    public init(tasks: [DecisionTask]) {
        self.tasks = tasks
    }
}

/// Swift-native tokenizer and GLiNER classification-schema formatter for the
/// exact Decide checkpoint. It reproduces the upstream inference collator for
/// classification-only inputs, including schema markers, lower-cased whitespace
/// splitting, sentence-final punctuation, and the smallest length bucket.
public final class NativeDecisionTokenizer: @unchecked Sendable {
    private let tokenizer: Tokenizer
    private let specialTokenIDs: [String: Int]
    private let padTokenID: Int
    private let wordRegex: NSRegularExpression

    public init(directory: URL) async throws {
        let preparedDirectory = try Self.prepareTokenizerDirectory(directory)
        tokenizer = try await AutoTokenizer.from(modelFolder: preparedDirectory)
        specialTokenIDs = try Self.loadSpecialTokenIDs(from: preparedDirectory)
        padTokenID = specialTokenIDs["[PAD]"] ?? tokenizer.convertTokenToId("[PAD]") ?? 0
        wordRegex = try NSRegularExpression(
            pattern: #"(?:https?://[^\s]+|www\.[^\s]+)|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|@[a-z0-9_]+|\w+(?:[-_]\w+)*|\S"#,
            options: [.caseInsensitive]
        )
    }

    public func makeInput(
        text: String,
        tasks: [DecisionTask],
        buckets: [Int] = [32, 64, 96, 128, 256, 512]
    ) throws -> DecisionInput {
        guard !tasks.isEmpty else { throw NativeTokenizerError.noTaskLabels }
        guard tasks.allSatisfy({ !$0.labels.isEmpty }) else {
            throw NativeTokenizerError.noTaskLabels
        }
        let orderedBuckets = Array(Set(buckets)).sorted()
        guard !orderedBuckets.isEmpty else {
            throw NativeTokenizerError.inputDoesNotFit(buckets)
        }

        var schemaGroups: [[SchemaEntry]] = []
        for (taskIndex, task) in tasks.enumerated() {
            let prompt = task.prompt.map { "\(task.name): \($0)" } ?? task.name
            let descriptions = task.labels.compactMap { label -> String? in
                guard let description = task.labelDescriptions?[label] else { return nil }
                return " [DESCRIPTION] \(label): \(description)"
            }.joined()
            var entries: [SchemaEntry] = [
                SchemaEntry(text: "(", marker: nil),
                SchemaEntry(text: "[P]", marker: .prompt(taskIndex)),
                SchemaEntry(text: prompt + descriptions, marker: nil),
                SchemaEntry(text: "(", marker: nil),
            ]
            for label in task.labels {
                entries.append(SchemaEntry(text: "[L]", marker: .label(taskIndex)))
                entries.append(SchemaEntry(text: label, marker: nil))
            }
            entries.append(SchemaEntry(text: ")", marker: nil))
            entries.append(SchemaEntry(text: ")", marker: nil))
            schemaGroups.append(entries)
        }

        var combined: [SchemaEntry] = []
        for (taskIndex, entries) in schemaGroups.enumerated() {
            combined.append(contentsOf: entries)
            if taskIndex < tasks.count - 1 {
                combined.append(SchemaEntry(text: "[SEP_STRUCT]", marker: nil))
            }
        }
        combined.append(SchemaEntry(text: "[SEP_TEXT]", marker: nil))
        combined.append(contentsOf: splitWords(text).map { SchemaEntry(text: $0, marker: nil) })

        var ids: [Int] = []
        var markerPositions = Array(
            repeating: [Int](),
            count: tasks.count
        )
        for entry in combined {
            let start = ids.count
            appendTokenized(entry.text, to: &ids)
            if case .label(let taskIndex) = entry.marker {
                markerPositions[taskIndex].append(start)
            }
        }
        let unpaddedLength = ids.count
        guard let selectedBucket = orderedBuckets.first(where: { $0 >= unpaddedLength }) else {
            throw NativeTokenizerError.inputDoesNotFit(orderedBuckets)
        }

        let paddedIDs = ids + Array(repeating: padTokenID, count: selectedBucket - unpaddedLength)
        let realMask = (0 ..< selectedBucket).map { Float($0 < unpaddedLength ? 1 : 0) }
        let maxHeads = max(tasks.count, 1)
        let maxOptions = max(tasks.map(\.labels.count).max() ?? 1, 1)
        var flatMarkers = [Int](repeating: 0, count: maxHeads * maxOptions)
        var flatMask = [Float](repeating: 0, count: maxHeads * maxOptions)
        for (head, positions) in markerPositions.enumerated() {
            for (option, position) in positions.enumerated() where option < maxOptions {
                flatMarkers[head * maxOptions + option] = position
                flatMask[head * maxOptions + option] = 1
            }
        }

        return DecisionInput(
            inputIDs: MLXArray(paddedIDs, [1, selectedBucket]),
            attentionMask: MLXArray(realMask, [1, selectedBucket]).asType(.float16),
            markerIndices: MLXArray(flatMarkers, [1, maxHeads, maxOptions]),
            markerMask: MLXArray(flatMask, [1, maxHeads, maxOptions]).asType(.float32)
        )
    }

    private func appendTokenized(_ text: String, to ids: inout [Int]) {
        var cursor = text.startIndex
        while cursor < text.endIndex {
            var nextRange: Range<String.Index>?
            var nextToken: String?
            for token in specialTokenIDs.keys {
                guard let range = text.range(of: token, range: cursor ..< text.endIndex) else {
                    continue
                }
                if nextRange == nil || range.lowerBound < nextRange!.lowerBound ||
                    (range.lowerBound == nextRange!.lowerBound &&
                        text.distance(from: range.lowerBound, to: range.upperBound) >
                            text.distance(from: nextRange!.lowerBound, to: nextRange!.upperBound))
                {
                    nextRange = range
                    nextToken = token
                }
            }
            guard let range = nextRange, let token = nextToken else {
                appendPlainTokenized(text[cursor...], to: &ids)
                return
            }
            if range.lowerBound > cursor {
                appendPlainTokenized(text[cursor ..< range.lowerBound], to: &ids)
            }
            ids.append(specialTokenIDs[token]!)
            cursor = range.upperBound
        }
    }

    private func appendPlainTokenized(_ text: String.SubSequence, to ids: inout [Int]) {
        guard !text.isEmpty else { return }
        let pieces = tokenizer.tokenize(text: String(text))
        for tokenID in tokenizer.convertTokensToIds(pieces) {
            ids.append(tokenID ?? tokenizer.unknownTokenId ?? 0)
        }
    }

    private func splitWords(_ text: String) -> [String] {
        let normalized: String
        if text.isEmpty {
            normalized = "."
        } else if text.last == "." || text.last == "!" || text.last == "?" {
            normalized = text
        } else {
            normalized = text + "."
        }
        let range = NSRange(normalized.startIndex ..< normalized.endIndex, in: normalized)
        return wordRegex.matches(in: normalized, options: [], range: range).compactMap { match in
            guard let swiftRange = Range(match.range, in: normalized) else { return nil }
            return String(normalized[swiftRange]).lowercased()
        }
    }

    private static func loadSpecialTokenIDs(from directory: URL) throws -> [String: Int] {
        let data = try Data(contentsOf: directory.appendingPathComponent("tokenizer.json"))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let added = root["added_tokens"] as? [[String: Any]]
        else {
            return [:]
        }
        return added.reduce(into: [String: Int]()) { result, token in
            if let content = token["content"] as? String,
                let id = token["id"] as? Int
            {
                result[content] = id
            }
        }
    }

    private static func prepareTokenizerDirectory(_ source: URL) throws -> URL {
        let configURL = source.appendingPathComponent("tokenizer_config.json")
        let tokenizerURL = source.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: configURL.path),
            FileManager.default.fileExists(atPath: tokenizerURL.path)
        else {
            throw NativeTokenizerError.missingTokenizerFiles(source)
        }

        let prepared = source.appendingPathComponent(".swift-unigram-tokenizer", isDirectory: true)
        try FileManager.default.createDirectory(at: prepared, withIntermediateDirectories: true)
        let tokenizerDestination = prepared.appendingPathComponent("tokenizer.json")
        if !FileManager.default.fileExists(atPath: tokenizerDestination.path) {
            try FileManager.default.copyItem(at: tokenizerURL, to: tokenizerDestination)
        }
        let specialURL = source.appendingPathComponent("special_tokens_map.json")
        if FileManager.default.fileExists(atPath: specialURL.path) {
            let destination = prepared.appendingPathComponent("special_tokens_map.json")
            if !FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.copyItem(at: specialURL, to: destination)
            }
        }

        let data = try Data(contentsOf: configURL)
        guard var config = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NativeTokenizerError.invalidTokenizerConfiguration
        }
        // Decide's upstream class is DebertaV2Tokenizer (a Unigram model),
        // but swift-transformers exposes the same Unigram implementation under
        // the registered XLMRoberta class. The tokenizer JSON remains exact.
        config["tokenizer_class"] = "XLMRobertaTokenizer"
        let output = try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted])
        try output.write(to: prepared.appendingPathComponent("tokenizer_config.json"))
        return prepared
    }
}

private struct SchemaEntry {
    enum Marker {
        case prompt(Int)
        case label(Int)
    }

    let text: String
    let marker: Marker?
}
