import Foundation
import GLiNERDecideCore
import MLX

private struct Prediction: Codable {
    let task: String
    let labels: [String]
    let probabilities: [Double]
}

private struct RawExample: Codable {
    let id: String
    let text: String
    let tasks: [DecisionTask]
}

private struct RawResult: Codable {
    let id: String
    let predictions: [Prediction]
    let preprocessingMilliseconds: Double
    let inferenceMilliseconds: Double
}

@main
struct RawCLI {
    static func main() async {
        do {
            let options = try Options(CommandLine.arguments)
            let frontend = try await NativeDecisionTokenizer(directory: options.tokenizer)
            let model = try GLiNERDecideClassifier(
                weightsURL: options.weights,
                useCompiledGraph: true,
                useFastAttention: !options.genericAttention,
                useCustomRelativeKernel: false,
                useFusedAttentionKernel: options.fusedAttentionKernel
            )

            if let jsonl = options.jsonl {
                try await runJSONL(
                    jsonl,
                    frontend: frontend,
                    model: model,
                    buckets: options.buckets,
                    batchSize: options.batchSize,
                    output: options.output
                )
            } else {
                try runSingle(
                    options,
                    frontend: frontend,
                    model: model
                )
            }
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            exit(1)
        }
    }

    private static func runSingle(
        _ options: Options,
        frontend: NativeDecisionTokenizer,
        model: GLiNERDecideClassifier
    ) throws {
        let schemaData = try Data(contentsOf: options.tasks)
        let schema = try JSONDecoder().decode(DecisionSchema.self, from: schemaData)
        guard let text = options.text else {
            throw NSError(
                domain: "RawCLI",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "--text is required in single mode"]
            )
        }
        let started = DispatchTime.now().uptimeNanoseconds
        let input = try frontend.makeInput(
            text: text,
            tasks: schema.tasks,
            buckets: options.buckets
        )
        let preprocessingMilliseconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
        if let dumpInput = options.dumpInput {
            try save(
                arrays: [
                    "input_ids": input.inputIDs,
                    "attention_mask": input.attentionMask,
                    "marker_indices": input.markerIndices,
                    "marker_mask": input.markerMask,
                ],
                url: dumpInput
            )
        }
        let result = model.classify(input)
        let predictions = decode(result.logits, tasks: schema.tasks)
        let output = try JSONEncoder().encode(predictions)
        print(String(decoding: output, as: UTF8.self))
        FileHandle.standardError.write(
            Data(String(format: "preprocessing_ms: %.3f\n", preprocessingMilliseconds).utf8)
        )
        FileHandle.standardError.write(
            Data(String(format: "inference_ms: %.3f\n", result.wallMilliseconds).utf8)
        )
    }

    private static func runJSONL(
        _ url: URL,
        frontend: NativeDecisionTokenizer,
        model: GLiNERDecideClassifier,
        buckets: [Int],
        batchSize: Int,
        output: URL?
    ) async throws {
        let text = try String(contentsOf: url, encoding: .utf8)
        let lines = text.split(whereSeparator: \.isNewline).map(String.init)
        var pending: [Pending] = []
        pending.reserveCapacity(lines.count)
        for (index, line) in lines.enumerated() {
            let example = try JSONDecoder().decode(RawExample.self, from: Data(line.utf8))
            let preprocessingStart = DispatchTime.now().uptimeNanoseconds
            let input = try frontend.makeInput(
                text: example.text,
                tasks: example.tasks,
                buckets: buckets
            )
            let preprocessingMilliseconds = Double(
                DispatchTime.now().uptimeNanoseconds - preprocessingStart
            ) / 1_000_000
            pending.append(
                Pending(index: index, example: example, input: input, preprocessingMilliseconds: preprocessingMilliseconds)
            )
            if (index + 1) % 100 == 0 || index + 1 == lines.count {
                FileHandle.standardError.write(Data("tokenized \(index + 1)/\(lines.count)\n".utf8))
            }
        }

        var groups: [String: [Pending]] = [:]
        for item in pending {
            let key = "\(item.input.inputIDs.shape[1])-\(item.input.markerIndices.shape[1])-\(item.input.markerIndices.shape[2])"
            groups[key, default: []].append(item)
        }
        var results = [RawResult?](repeating: nil, count: pending.count)
        for (_, group) in groups {
            for start in stride(from: 0, to: group.count, by: batchSize) {
                let end = min(start + batchSize, group.count)
                let batch = Array(group[start ..< end])
                let input = DecisionInput(
                    inputIDs: concatenated(batch.map { $0.input.inputIDs }, axis: 0),
                    attentionMask: concatenated(batch.map { $0.input.attentionMask }, axis: 0),
                    markerIndices: concatenated(batch.map { $0.input.markerIndices }, axis: 0),
                    markerMask: concatenated(batch.map { $0.input.markerMask }, axis: 0)
                )
                let result = model.classify(input)
                let logits = result.logits.asType(.float32)
                logits.eval()
                for (row, item) in batch.enumerated() {
                    results[item.index] = RawResult(
                        id: item.example.id,
                        predictions: decode(logits, row: row, tasks: item.example.tasks),
                        preprocessingMilliseconds: item.preprocessingMilliseconds,
                        inferenceMilliseconds: result.wallMilliseconds / Double(batch.count)
                    )
                }
            }
        }
        let finalResults = results.compactMap { $0 }
        let data = try JSONEncoder().encode(finalResults)
        if let output {
            try data.write(to: output)
        } else {
            print(String(decoding: data, as: UTF8.self))
        }
    }

    private static func decode(
        _ rawLogits: MLXArray,
        tasks: [DecisionTask]
    ) -> [Prediction] {
        let logits = rawLogits.asType(.float32)
        logits.eval()
        return decode(logits, row: 0, tasks: tasks)
    }

    private static func decode(
        _ logits: MLXArray,
        row: Int,
        tasks: [DecisionTask]
    ) -> [Prediction] {
        var predictions: [Prediction] = []
        for (head, task) in tasks.enumerated() {
            var probabilities = (0 ..< task.labels.count).map { option in
                Double(logits[row, head, option].item(Float.self))
            }
            if task.multiLabel {
                probabilities = probabilities.map { 1.0 / (1.0 + exp(-$0)) }
                var selected = probabilities.enumerated()
                    .filter { $0.element >= 0.5 }
                    .map(\.offset)
                if selected.isEmpty {
                    selected = [probabilities.enumerated().max { $0.element < $1.element }!.offset]
                }
                predictions.append(
                    Prediction(
                        task: task.name,
                        labels: selected.map { task.labels[$0] },
                        probabilities: probabilities
                    )
                )
            } else {
                let maxLogit = probabilities.max() ?? 0
                let exponentials = probabilities.map { exp($0 - maxLogit) }
                let total = exponentials.reduce(0, +)
                probabilities = exponentials.map { $0 / total }
                let best = probabilities.enumerated().max { $0.element < $1.element }!.offset
                predictions.append(
                    Prediction(
                        task: task.name,
                        labels: [task.labels[best]],
                        probabilities: probabilities
                    )
                )
            }
        }
        return predictions
    }

    private struct Pending {
        let index: Int
        let example: RawExample
        let input: DecisionInput
        let preprocessingMilliseconds: Double
    }
}

private struct Options {
    let text: String?
    let tasks: URL
    let jsonl: URL?
    let tokenizer: URL
    let weights: URL
    let buckets: [Int]
    let batchSize: Int
    let dumpInput: URL?
    let output: URL?
    let genericAttention: Bool
    let fusedAttentionKernel: Bool

    init(_ arguments: [String]) throws {
        var text: String?
        var tasks = URL(fileURLWithPath: "Artifacts/tasks.json")
        var jsonl: URL?
        var tokenizer = URL(fileURLWithPath: "Artifacts/coreml-runtime")
        var weights = URL(fileURLWithPath: "Artifacts/decide-classification-fp16.safetensors")
        var buckets = [32, 64, 96, 128, 256, 512]
        var batchSize = 32
        var dumpInput: URL?
        var output: URL?
        var genericAttention = false
        var fusedAttentionKernel = false
        var index = 1
        while index < arguments.count {
            func next() -> String {
                index += 1
                guard index < arguments.count else { fatalError("Missing option value") }
                return arguments[index]
            }
            switch arguments[index] {
            case "--text": text = next()
            case "--tasks": tasks = URL(fileURLWithPath: next())
            case "--jsonl": jsonl = URL(fileURLWithPath: next())
            case "--tokenizer": tokenizer = URL(fileURLWithPath: next())
            case "--weights": weights = URL(fileURLWithPath: next())
            case "--buckets": buckets = next().split(separator: ",").compactMap { Int($0) }
            case "--batch-size": batchSize = Int(next()) ?? batchSize
            case "--dump-input": dumpInput = URL(fileURLWithPath: next())
            case "--output": output = URL(fileURLWithPath: next())
            case "--generic-attention": genericAttention = true
            case "--fused-attention-kernel": fusedAttentionKernel = true
            case "--help", "-h":
                print("""
                gliner-decide-raw --text TEXT --tasks schema.json
                gliner-decide-raw --jsonl examples.jsonl --output results.json
                  [--tokenizer DIR] [--weights FILE] [--buckets 32,64,...]
                """)
                exit(0)
            default:
                throw NSError(
                    domain: "RawCLI",
                    code: 2,
                    userInfo: [NSLocalizedDescriptionKey: "Unknown option \(arguments[index])"]
                )
            }
            index += 1
        }
        guard text != nil || jsonl != nil else {
            throw NSError(
                domain: "RawCLI",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "--text or --jsonl is required"]
            )
        }
        self.text = text
        self.tasks = tasks
        self.jsonl = jsonl
        self.tokenizer = tokenizer
        self.weights = weights
        self.buckets = buckets
        self.batchSize = batchSize
        self.dumpInput = dumpInput
        self.output = output
        self.genericAttention = genericAttention
        self.fusedAttentionKernel = fusedAttentionKernel
    }
}
