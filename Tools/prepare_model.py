#!/usr/bin/env python3
"""Prepare the exact GLiNER2.5-Decide checkpoint for the Swift/MLX runtime.

This script intentionally downloads only encoder and classification weights. Span,
relation, and count heads are not part of the fast Decide classification path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

MODEL_ID = "fastino/GLiNER2.5-Decide"
MODEL_REVISION = "65624f1a0265b3f612bae66a2685a06b94a68a9d"
COREML_ID = "FluidInference/gliner2-5-decide-coreml"
COREML_REVISION = "cd0d7b1ef32b10e1e3a5a73c9d9ac8411d819c5a"

DEFAULT_TEXT = (
    "My transfer is still pending and I used the wrong sort code. "
    "Can you stop it?"
)
DEFAULT_TASKS = {
    "intent": [
        "transfer_pending",
        "transfer_cancel",
        "beneficiary_add",
        "card_lost",
    ],
    "urgency": ["low", "normal", "high"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("Artifacts"))
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--max-heads", type=int, default=4)
    parser.add_argument("--max-options", type=int, default=8)
    parser.add_argument(
        "--reference-device",
        choices=("cpu", "mps"),
        default="mps",
        help="Generate native logits for correctness checks.",
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Do not load the full native model after conversion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = DEFAULT_TASKS
    if args.tasks:
        tasks = json.loads(args.tasks.read_text())

    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors import safe_open
    from safetensors.numpy import save_file

    source_path = Path(
        hf_hub_download(MODEL_ID, "model.safetensors", revision=MODEL_REVISION)
    )
    config_path = Path(
        hf_hub_download(
            MODEL_ID, "encoder_config/config.json", revision=MODEL_REVISION
        )
    )

    converted: dict[str, np.ndarray] = {}
    with safe_open(source_path, framework="numpy", device="cpu") as source:
        for key in source.keys():
            if not (key.startswith("encoder.") or key.startswith("classifier.")):
                continue
            value = source.get_tensor(key)
            if np.issubdtype(value.dtype, np.floating):
                value = value.astype(np.float16)
            converted[key] = np.ascontiguousarray(value)

    required = {
        "encoder.embeddings.word_embeddings.weight",
        "encoder.encoder.rel_embeddings.weight",
        "encoder.encoder.LayerNorm.weight",
        "classifier.0.weight",
        "classifier.2.weight",
    }
    missing = sorted(required.difference(converted))
    if missing:
        raise RuntimeError(f"Checkpoint is missing required weights: {missing}")

    weights_path = args.output / "decide-classification-fp16.safetensors"
    save_file(
        converted,
        str(weights_path),
        metadata={
            "source_model": MODEL_ID,
            "source_revision": MODEL_REVISION,
            "precision": "float16",
            "scope": "encoder+classification-only",
        },
    )
    (args.output / "encoder-config.json").write_text(config_path.read_text())
    del converted

    runtime_dir = args.output / "coreml-runtime"
    runtime_dir.mkdir(exist_ok=True)
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

    sys.path.insert(0, str(runtime_dir.resolve()))
    from preprocessing import prepare_decision  # type: ignore

    from gliner2.models.base import load_extractor_tokenizer
    from gliner2.processor import SchemaTransformer

    tokenizer = load_extractor_tokenizer(str(runtime_dir))
    processor = SchemaTransformer(tokenizer=tokenizer, token_pooling="first")
    arrays = prepare_decision(
        processor,
        args.text,
        tasks,
        args.length,
        args.max_heads,
        args.max_options,
    )
    save_file(
        {name: np.ascontiguousarray(value) for name, value in arrays.items()},
        str(args.output / "sample-inputs.safetensors"),
        metadata={"source_revision": MODEL_REVISION},
    )
    (args.output / "sample.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "text": args.text,
                "tasks": tasks,
                "length": args.length,
                "max_heads": args.max_heads,
                "max_options": args.max_options,
                "shapes": {name: list(value.shape) for name, value in arrays.items()},
            },
            indent=2,
        )
        + "\n"
    )

    if not args.skip_reference:
        reference_path = generate_reference(
            arrays=arrays,
            output_path=args.output / "reference-outputs.safetensors",
            device=args.reference_device,
        )
        print(f"reference: {reference_path}")

    print(f"weights:  {weights_path}")
    print(f"inputs:   {args.output / 'sample-inputs.safetensors'}")
    print(f"metadata: {args.output / 'sample.json'}")


def generate_reference(
    *, arrays: dict[str, np.ndarray], output_path: Path, device: str
) -> Path:
    import torch
    from gliner2 import AutoExtractor
    from safetensors.numpy import save_file

    model = AutoExtractor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        map_location=device,
        quantize=device == "mps",
    )
    model.eval()

    input_ids = torch.from_numpy(arrays["input_ids"]).to(device)
    attention_mask = torch.from_numpy(arrays["attention_mask"]).to(device)
    marker_indices = torch.from_numpy(arrays["marker_indices"])
    marker_mask = torch.from_numpy(arrays["marker_mask"])

    with torch.inference_mode():
        encoded = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        gathered = []
        for batch_index in range(encoded.shape[0]):
            for head in range(marker_indices.shape[1]):
                positions = marker_indices[batch_index, head]
                gathered.append(encoded[batch_index, positions.long()])
        marker_embeddings = torch.stack(gathered).unsqueeze(0)
        logits = model.classifier(marker_embeddings).squeeze(-1)
        logits = logits * torch.from_numpy(arrays["marker_mask"]).to(logits.device)
        logits = torch.where(
            torch.from_numpy(arrays["marker_mask"]).to(logits.device) > 0,
            logits,
            torch.full_like(logits, -10_000),
        )

    save_file(
        {
            "logits": np.ascontiguousarray(logits.float().cpu().numpy()),
        },
        str(output_path),
        metadata={"source_revision": MODEL_REVISION, "device": device},
    )
    return output_path


if __name__ == "__main__":
    main()
