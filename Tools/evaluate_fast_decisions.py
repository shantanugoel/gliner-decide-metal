#!/usr/bin/env python3
"""Evaluate native Decide and the Swift/MLX runtime on fast-decisions.

The default run evaluates the complete public development split (17 domains ×
100 examples). It reports exact-set accuracy and throughput for both paths.
The Swift path receives the same Core ML-compatible preprocessing used by the
reference export, grouped into the smallest fixed length bucket that fits.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = "fastino/GLiNER2.5-Decide"
MODEL_REVISION = "65624f1a0265b3f612bae66a2685a06b94a68a9d"
DATASET_ID = "fastino/fast-decisions"
DATASET_REVISION = "1a33070c"
COREML_ID = "FluidInference/gliner2-5-decide-coreml"
COREML_REVISION = "cd0d7b1ef32b10e1e3a5a73c9d9ac8411d819c5a"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", type=Path, default=Path("Artifacts/fast-decisions-eval"))
    p.add_argument("--binary", type=Path, default=Path(".build/DerivedData/Build/Products/Debug/gliner-decide-metal"))
    p.add_argument("--weights", type=Path, default=Path("Artifacts/decide-classification-fp16.safetensors"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--length", type=int, default=256, choices=(128, 256, 512))
    p.add_argument("--max-options", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="Limit examples per domain; 0 means all 100")
    p.add_argument("--skip-native", action="store_true")
    p.add_argument("--skip-swift", action="store_true")
    p.add_argument("--native-device", choices=("cpu", "mps"), default="mps")
    p.add_argument("--output", type=Path)
    return p.parse_args()


def load_examples(limit: int) -> list[dict[str, Any]]:
    from datasets import get_dataset_config_names, load_dataset

    examples: list[dict[str, Any]] = []
    for config_index, config in enumerate(get_dataset_config_names(DATASET_ID)):
        rows = load_dataset(DATASET_ID, config, revision=DATASET_REVISION)["train"]
        if limit:
            rows = rows.select(range(min(limit, len(rows))))
        for row_index, row in enumerate(rows):
            classification = row["output"]["classifications"][0]
            task = classification["task"]
            examples.append(
                {
                    "uid": f"{config}:{row_index}",
                    "config": config,
                    "row_index": row_index,
                    "text": row["input"],
                    "task": task,
                    "labels": list(classification["labels"]),
                    "multi_label": bool(classification.get("multi_label", False)),
                    "true_label": list(classification["true_label"]),
                }
            )
    return examples


def task_dict(example: dict[str, Any]) -> dict[str, Any]:
    return {
        example["task"]: {
            "labels": example["labels"],
            "multi_label": example["multi_label"],
        }
    }


def native_prediction(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item["label"] if isinstance(item, dict) else str(item) for item in value]
    if isinstance(value, dict):
        return [str(value["label"])]
    return [str(value)]


def score_predictions(examples: list[dict[str, Any]], predictions: dict[str, list[str]]) -> dict[str, Any]:
    by_config: dict[str, list[int]] = defaultdict(list)
    correct = 0
    for example in examples:
        expected = set(example["true_label"])
        got = set(predictions[example["uid"]])
        ok = expected == got
        correct += int(ok)
        by_config[example["config"]].append(int(ok))
    per_config = {
        config: sum(values) / len(values) for config, values in sorted(by_config.items())
    }
    return {
        "count": len(examples),
        "correct": correct,
        "accuracy": correct / len(examples),
        "macro_accuracy": sum(per_config.values()) / len(per_config),
        "per_config_accuracy": per_config,
    }


def evaluate_native(examples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from gliner2 import AutoExtractor

    model = AutoExtractor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        map_location=args.native_device,
        quantize=args.native_device == "mps",
    )
    model.eval()

    # Warm the MPS/CPU backend without including it in the measured window.
    warm = examples[: min(args.batch_size, len(examples))]
    if warm:
        model.batch_classify_text(
            [x["text"] for x in warm],
            task_dict(warm[0]),
            batch_size=args.batch_size,
            max_len=512,
            include_confidence=True,
        )
    if args.native_device == "mps":
        torch.mps.synchronize()

    predictions: dict[str, list[str]] = {}
    started = time.perf_counter()
    for config in sorted({x["config"] for x in examples}):
        group = [x for x in examples if x["config"] == config]
        for start in range(0, len(group), args.batch_size):
            chunk = group[start : start + args.batch_size]
            results = model.batch_classify_text(
                [x["text"] for x in chunk],
                task_dict(chunk[0]),
                batch_size=args.batch_size,
                max_len=512,
                include_confidence=True,
            )
            for example, result in zip(chunk, results):
                predictions[example["uid"]] = native_prediction(result[example["task"]])
    if args.native_device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started

    del model
    if args.native_device == "mps":
        torch.mps.empty_cache()
    return {
        "accuracy": score_predictions(examples, predictions),
        "seconds": elapsed,
        "examples_per_second": len(examples) / elapsed,
        "predictions": predictions,
    }


def ensure_runtime(runtime_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    runtime_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        COREML_ID,
        revision=COREML_REVISION,
        local_dir=runtime_dir,
        allow_patterns=[
            "preprocessing.py",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    )


def prepare_swift_groups(
    examples: list[dict[str, Any]], args: argparse.Namespace
) -> dict[int, dict[str, Any]]:
    from huggingface_hub import snapshot_download
    from safetensors.numpy import save_file

    runtime_dir = args.artifacts / "coreml-runtime"
    ensure_runtime(runtime_dir)
    sys.path.insert(0, str(runtime_dir.resolve()))
    from preprocessing import prepare_decision  # type: ignore

    from gliner2.models.base import load_extractor_tokenizer
    from gliner2.processor import SchemaTransformer

    tokenizer = load_extractor_tokenizer(str(runtime_dir))
    processor = SchemaTransformer(tokenizer=tokenizer, token_pooling="first")
    groups: dict[int, list[tuple[dict[str, Any], dict[str, np.ndarray]]]] = defaultdict(list)

    for index, example in enumerate(examples):
        chosen_length = args.length
        while True:
            try:
                arrays = prepare_decision(
                    processor,
                    example["text"],
                    task_dict(example),
                    chosen_length,
                    4,
                    args.max_options,
                )
                break
            except ValueError as exc:
                if chosen_length == 512:
                    raise
                chosen_length = 256 if chosen_length == 128 else 512
                if "require" not in str(exc) and chosen_length == args.length:
                    # A schema can exceed the requested bucket even when text
                    # does not; the larger bucket is the safe fallback.
                    chosen_length = 512

        arrays = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
        groups[chosen_length].append((example, arrays))
        if (index + 1) % 100 == 0:
            print(f"prepared {index + 1}/{len(examples)} examples", flush=True)

    result: dict[int, dict[str, Any]] = {}
    for length, rows in sorted(groups.items()):
        input_path = args.artifacts / f"inputs-L{length}.safetensors"
        output_path = args.artifacts / f"swift-outputs-L{length}.safetensors"
        arrays = {
            name: np.concatenate([item[1][name] for item in rows], axis=0)
            for name in ("input_ids", "attention_mask", "marker_indices", "marker_mask")
        }
        save_file(arrays, str(input_path), metadata={"source_revision": MODEL_REVISION})
        metadata = [item[0] for item in rows]
        (args.artifacts / f"metadata-L{length}.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        result[length] = {
            "input_path": input_path,
            "output_path": output_path,
            "metadata": metadata,
        }
    return result


def decode_swift_logits(logits: np.ndarray, example: dict[str, Any]) -> list[str]:
    row = logits[0, : len(example["labels"])].astype(np.float32)
    if example["multi_label"]:
        shifted = row - row.max()
        probs = 1.0 / (1.0 + np.exp(-shifted))
        chosen = [i for i, p in enumerate(probs) if p >= 0.5]
        if not chosen:
            chosen = [int(row.argmax())]
    else:
        shifted = row - row.max()
        probs = np.exp(shifted)
        probs /= probs.sum()
        chosen = [int(row.argmax())]
    return [example["labels"][i] for i in chosen]


def evaluate_swift(
    examples: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    from safetensors import safe_open

    groups = prepare_swift_groups(examples, args)
    predictions: dict[str, list[str]] = {}
    total_rows = 0
    total_seconds = 0.0
    per_length: dict[str, Any] = {}

    for length, group in groups.items():
        command = [
            str(args.binary),
            "--weights",
            str(args.weights),
            "--inputs",
            str(group["input_path"]),
            "--batch-size",
            str(args.batch_size),
            "--output",
            str(group["output_path"]),
            "--compiled",
            "--fast-attention",
        ]
        print("$", " ".join(command), flush=True)
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        print(completed.stdout, end="", flush=True)
        match = re.search(r"batch_total_ms:\s*([0-9.]+)", completed.stdout)
        seconds = float(match.group(1)) / 1000.0 if match else 0.0
        count = len(group["metadata"])
        total_rows += count
        total_seconds += seconds
        with safe_open(group["output_path"], framework="numpy") as handle:
            logits = handle.get_tensor("logits")
        if logits.shape[0] != count:
            raise RuntimeError(f"Swift output count mismatch for L{length}: {logits.shape[0]} != {count}")
        for index, example in enumerate(group["metadata"]):
            predictions[example["uid"]] = decode_swift_logits(logits[index], example)
        per_length[str(length)] = {
            "count": count,
            "seconds": seconds,
            "examples_per_second": count / seconds if seconds else None,
        }

    return {
        "accuracy": score_predictions(examples, predictions),
        "seconds": total_seconds,
        "examples_per_second": total_rows / total_seconds if total_seconds else None,
        "per_length": per_length,
        "predictions": predictions,
        "groups": {str(k): {kk: str(vv) for kk, vv in v.items() if kk != "metadata"} for k, v in groups.items()},
    }


def main() -> None:
    args = parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    if not args.weights.exists():
        raise SystemExit(
            f"Missing {args.weights}. Run Tools/prepare_model.py before evaluation."
        )
    examples = load_examples(args.limit)
    print(f"loaded {len(examples)} examples from {DATASET_ID}")

    result: dict[str, Any] = {
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "examples": len(examples),
        "batch_size": args.batch_size,
    }
    if not args.skip_native:
        print("running native baseline...", flush=True)
        result["native"] = evaluate_native(examples, args)
        print(json.dumps(result["native"]["accuracy"], indent=2), flush=True)
    if not args.skip_swift:
        print("running Swift/MLX runtime...", flush=True)
        result["swift"] = evaluate_swift(examples, args)
        print(json.dumps(result["swift"]["accuracy"], indent=2), flush=True)

    if "native" in result and "swift" in result:
        result["prediction_agreement"] = sum(
            result["native"]["predictions"].get(uid) == pred
            for uid, pred in result["swift"]["predictions"].items()
        ) / len(examples)

    output = args.output or (args.artifacts / "results.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"results: {output}")


if __name__ == "__main__":
    main()
