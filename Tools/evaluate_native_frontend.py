#!/usr/bin/env python3
"""Evaluate the Swift-native tokenizer/schema frontend on fast-decisions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from evaluate_fast_decisions import load_examples, score_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path(".build/DerivedData/Build/Products/Debug/gliner-decide-raw"))
    parser.add_argument("--tokenizer", type=Path, default=Path("Artifacts/coreml-runtime"))
    parser.add_argument("--weights", type=Path, default=Path("Artifacts/decide-classification-fp16.safetensors"))
    parser.add_argument("--artifacts", type=Path, default=Path("Artifacts/fast-decisions-frontend"))
    parser.add_argument("--native-results", type=Path, default=Path("Artifacts/fast-decisions-batch32/results.json"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)

    examples = load_examples(args.limit)
    input_path = args.artifacts / "inputs.jsonl"
    with input_path.open("w") as handle:
        for example in examples:
            record = {
                "id": example["uid"],
                "text": example["text"],
                "tasks": [
                    {
                        "name": example["task"],
                        "labels": example["labels"],
                        "multiLabel": example["multi_label"],
                    }
                ],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(examples)} examples to {input_path}", flush=True)

    output_path = args.artifacts / "results.json"
    command = [
        str(args.binary),
        "--jsonl",
        str(input_path),
        "--tokenizer",
        str(args.tokenizer),
        "--weights",
        str(args.weights),
        "--output",
        str(output_path),
    ]
    started = time.perf_counter()
    subprocess.run(command, check=True)
    elapsed = time.perf_counter() - started
    raw_results = json.loads(output_path.read_text())
    predictions = {row["id"]: row["predictions"][0]["labels"] for row in raw_results}
    accuracy = score_predictions(examples, predictions)
    preprocessing = sum(row["preprocessingMilliseconds"] for row in raw_results) / 1000
    inference = sum(row["inferenceMilliseconds"] for row in raw_results) / 1000
    result = {
        "count": len(examples),
        "accuracy": accuracy,
        "preprocessing_seconds": preprocessing,
        "inference_seconds": inference,
        "wall_seconds": elapsed,
        "examples_per_second": len(examples) / elapsed,
        "results_path": str(output_path),
    }
    if args.native_results.exists():
        native = json.loads(args.native_results.read_text())
        native_predictions = native.get("native", {}).get("predictions", {})
        result["native_accuracy"] = native.get("native", {}).get("accuracy", {}).get("accuracy")
        result["native_agreement"] = sum(
            native_predictions.get(example["uid"]) == predictions[example["uid"]]
            for example in examples
        ) / len(examples)
    (args.artifacts / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
