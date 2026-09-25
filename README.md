# GLiNER2.5-Decide Metal runtime

A classification-only Swift/MLX runtime for the exact
[`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide)
checkpoint. It does not substitute `gliner2.5-small`, `base`, `multi`, or
`Decide-1B`.

The implementation ports the checkpoint's DeBERTa-v3-large encoder and GLiNER
classification MLP. GLiNER span, relation, and count heads are intentionally not
loaded because the Core ML reference path used as the baseline exports only the
classification path.

## Quick start

This runtime is an encoder-based decision model: it returns one or more typed
labels for a supplied task schema. It does not generate text and currently
supports GLiNER's classification path, not span/relation extraction.

### Requirements

- Apple Silicon Mac
- macOS 14 or newer
- Xcode with the Metal toolchain
- [`uv`](https://docs.astral.sh/uv/)

### 1. Download and prepare the exact model

```bash
git clone https://github.com/shantanugoel/gliner-decide-metal.git
cd gliner-decide-metal

uv run --python 3.12 --with-requirements Tools/requirements.txt \
  python Tools/prepare_model.py \
  --output Artifacts \
  --skip-reference
```

This downloads the tokenizer and the classification-only FP16 weights from
`fastino/GLiNER2.5-Decide`. Model weights are not committed to this repository.

### 2. Build the Swift runner

```bash
xcodebuild build \
  -scheme gliner-decide-raw \
  -destination 'platform=macOS' \
  -clonedSourcePackagesDirPath .build/SourcePackages \
  -derivedDataPath .build/DerivedData
```

### 3. Describe the allowed decisions

Create `tasks.json`:

```json
{
  "tasks": [
    {
      "name": "intent",
      "labels": [
        "order_status",
        "refund_request",
        "cancel_subscription",
        "speak_to_human",
        "other"
      ],
      "multiLabel": false
    }
  ]
}
```

A request may contain multiple task heads. Set `multiLabel` to `true` and
optionally provide `prompt` and `labelDescriptions` to match the schemas
supported by the original model.

### 4. Run raw text

```bash
.build/DerivedData/Build/Products/Debug/gliner-decide-raw \
  --text 'Please cancel my subscription and refund the latest payment.' \
  --tasks tasks.json \
  --tokenizer Artifacts/coreml-runtime \
  --weights Artifacts/decide-classification-fp16.safetensors
```

Example output:

```json
[
  {
    "task": "intent",
    "labels": ["refund_request"],
    "probabilities": [0.01, 0.97, 0.01, 0.005, 0.005]
  }
]
```

The exact probabilities depend on the input. The runner chooses the smallest
supported sequence bucket automatically.

For repeated or batched requests, pass JSONL input and write JSONL results:

```bash
.build/DerivedData/Build/Products/Debug/gliner-decide-raw \
  --jsonl examples.jsonl \
  --output results.json \
  --tokenizer Artifacts/coreml-runtime \
  --weights Artifacts/decide-classification-fp16.safetensors
```

Each input line has this form:

```json
{"id":"example-1","text":"...","tasks":[{"name":"intent","labels":["yes","no"],"multiLabel":false}]}
```

### Python usage

The optimized MLX model does **not** yet have a direct Python import binding.
Python can still run the original model without Swift:

```bash
uv run --python 3.12 --with 'gliner2[local]==2.0.0' python - <<'PY'
from gliner2 import AutoExtractor

model = AutoExtractor.from_pretrained(
    "fastino/GLiNER2.5-Decide",
    map_location="mps",
    quantize=True,
)
print(model.classify_text(
    "Please cancel my subscription and refund the latest payment.",
    {"intent": ["refund_request", "cancel_subscription", "other"]},
    include_confidence=True,
))
PY
```

Alternatively, the repository's Python evaluators can prepare inputs and invoke
the Swift runner, as described below.

## Baseline comparison

The first full comparison uses the public `fastino/fast-decisions` development
split: 17 domains × 100 examples = 1,700 examples, dataset revision
`1a33070c`. Accuracy is exact-set match against the reference label set.

| Runtime | Batch | Attention path | Accuracy | Throughput |
|---|---:|---|---:|---:|
| Python native `gliner2`/PyTorch MPS (no Swift) | 8 | native PyTorch | **64.41%** | 13.95 rows/s |
| Python native `gliner2`/PyTorch MPS (no Swift) | 32 | native PyTorch | **64.41%** | **14.57 rows/s** |
| Swift/MLX final build | 8 | built-in fused SDPA | 64.24% | 23.72 rows/s |
| Swift/MLX final build | 16 | built-in fused SDPA | 64.24% | 27.21 rows/s |
| Swift/MLX final build | 32 | built-in fused SDPA | 64.24% | **29.02 rows/s** |
| Swift/MLX custom-attention experiment | 8 | custom fused Metal | 64.24% | 19.65 rows/s |
| Swift/MLX custom-attention experiment | 16 | custom fused Metal | 64.24% | 18.41 rows/s |
| Swift/MLX custom-attention experiment | 32 | custom fused Metal | 64.24% | 17.47 rows/s |

These are fresh full-split runs of the current build. The Python-native rows
run `gliner2[local]` directly on MPS and do not load Swift or MLX. The Swift rows
are launched by the Python evaluator, which prepares the exact same arrays and
invokes `gliner-decide-metal`; the table therefore includes both “no Swift” and
“Python driving the Swift runner” paths. Throughput excludes model initialization
and one warmup batch. The default Swift path uses dynamic
L32/L64/L96/L128/L256/L512 buckets and does **not** enable the experimental
custom fully fused attention kernel. Native/Swift agreement for the final default
path is **99.12%**; the three differing examples account for the 0.18-point
accuracy difference.

There is not yet a direct Python import binding for `GLiNERDecideCore`: Python
can run the native model, or orchestrate the Swift executable, but the optimized
MLX graph itself is currently a Swift library/binary.

Reproduce with:

```bash
uv run --python 3.12 --with-requirements Tools/requirements.txt \
  python Tools/evaluate_fast_decisions.py \
  --batch-size 32 \
  --length 256 \
  --buckets 32,64,96,128,256,512 \
  --artifacts Artifacts/fast-decisions-eval \
  --output Artifacts/fast-decisions-eval/baseline-results.json
```

For the no-Swift Python-native row, add `--skip-swift`. For the custom
attention row, add `--skip-native --fused-attention-kernel`.

The complete per-domain results and predictions are written by the evaluator;
large model/data artifacts are ignored by git. The phase tables below are
historical checkpoints; the table above is the latest full comparison.

### Phase 1 — dynamic length buckets

The Swift evaluator now tries `L32, L64, L96, L128, L256, L512` and sends each
request to the smallest bucket that fits its schema and text. On the same
1,700-example comparison:

| Runtime | Accuracy | Throughput | Change vs. previous Swift path |
|---|---:|---:|---:|
| Previous fixed L256/L512 Swift path | 64.24% | 28.72 rows/s | — |
| Dynamic-bucket Swift path | **64.24%** | **29.89 rows/s** | **+4.1%** |

Bucket distribution: 7 examples at L96, 145 at L128, 1,424 at L256, and 124 at
L512. Accuracy and native/Swift agreement were unchanged at 99.12%.

### Phase 2 — batch scheduling

The Swift batch runner now supports fixed-size chunks, optional final-chunk
padding, and multiple rows per compiled graph invocation. A full 1,700-example
sweep selected batch 32 as the best throughput point on this M1 Max:

| Runtime at batch 32 | Accuracy | Throughput |
|---|---:|---:|
| Native MPS baseline | **64.41%** | 14.55 rows/s |
| Swift/MLX dynamic buckets | **64.24%** | **31.39 rows/s** |

Native/Swift prediction agreement remains **99.12%**. Batch 32 improves Swift
throughput by about 5% over batch 8; batch 1 remains the latency-optimized mode.
The phase-2 comparison is reproduced by adding `--batch-size 32` to the
baseline command above.

### Phase 3 — fully fused DeBERTa attention

A custom Metal kernel now owns the complete DeBERTa attention operation:

```text
QK + relative C2P/P2C bias + mask + softmax + V accumulation
```

It is correct within FP16 tolerance, including the full public split:

| Runtime at batch 32 | Accuracy | Throughput |
|---|---:|---:|
| Native MPS baseline | **64.41%** | 14.44 rows/s |
| Swift/MLX built-in fused SDPA | **64.24%** | **31.39 rows/s** |
| Swift custom fully fused attention | **64.24%** | 20.10 rows/s |

Native/Swift agreement remains **99.12%**, but the custom attention kernel is
slower than MLX's built-in fused SDPA on this M1 Max. It remains opt-in via
`--fused-attention-kernel`; the default stays on the built-in path.

### Phase 4 — native Swift tokenizer/schema frontend

`NativeDecisionTokenizer` now loads the exact Decide Unigram tokenizer from
local `tokenizer.json`/`tokenizer_config.json` assets through
`swift-transformers`. It reproduces the inference collator's classification
schema formatting, marker positions, lower-cased word splitting, sentence-final
punctuation, padding, and dynamic length selection without Python at runtime.

A new raw-text executable accepts either one request or JSONL:

```bash
xcodebuild build \
  -scheme gliner-decide-raw \
  -destination 'platform=macOS' \
  -clonedSourcePackagesDirPath .build/SourcePackages \
  -derivedDataPath .build/DerivedData

.build/DerivedData/Build/Products/Debug/gliner-decide-raw \
  --text 'My transfer is still pending and I used the wrong sort code. Can you stop it?' \
  --tasks /path/to/tasks.json \
  --tokenizer Artifacts/coreml-runtime \
  --weights Artifacts/decide-classification-fp16.safetensors
```

Full 1,700-example frontend comparison:

| Runtime | Accuracy | Throughput | Notes |
|---|---:|---:|---|
| Native MPS base model | 64.41% | 14.55 rows/s | Reference |
| Swift frontend + MLX runtime | **64.47%** | **19.86 rows/s** | Includes native tokenization/schema formatting |

The frontend agrees with the native base model on **99.94%** of predictions.
Its 15.0 seconds of preprocessing and 65.4 seconds of inference produce an
85.6-second end-to-end run; the lower throughput than the preprocessed-array
benchmark is the expected cost of moving tokenization/schema construction into
Swift.



## Short-request microbenchmark

Short 128-token request, 4 heads × 8 labels, batch 1, warm process:

| Path | p50 |
|---|---:|
| Core ML FP16 reference, CPU+GPU | ~24.7 ms |
| MLX, cached static projections, compiled graph, fused SDPA | **~24.6 ms** |
| MLX, cached static projections, compiled graph | ~25.1 ms |
| MLX eager graph | ~32–35 ms |
| MLX compiled + custom residual/LayerNorm kernel | ~26 ms |
| MLX compiled + custom relative-bias kernel | ~28 ms |

The custom residual/LayerNorm kernel is retained as an experiment, but is not
enabled by default because it was slower than MLX's built-in LayerNorm on this
machine. All correctness checks are against native Decide logits.

## Prepare the exact checkpoint

The preparation script downloads only the encoder and classification weights,
converts them to FP16 safetensors, creates a reproducible preprocessed input, and
optionally creates native MPS reference logits:

```bash
uv run --python 3.12 --with-requirements Tools/requirements.txt \
  python Tools/prepare_model.py --output Artifacts --reference-device mps
```

`Tools/requirements.txt` pins the GLiNER2/Transformers versions used for the
reference run. The source checkpoint is pinned to revision
`65624f1a0265b3f612bae66a2685a06b94a68a9d`.

## Build

MLX Swift must be built through Xcode so its Metal shaders are available:

```bash
xcodebuild build \
  -scheme gliner-decide-metal \
  -destination 'platform=macOS' \
  -clonedSourcePackagesDirPath .build/SourcePackages \
  -derivedDataPath .build/DerivedData
```

## Benchmark

The recommended path is the compiled graph with MLX's fused
scaled-dot-product-attention and the precomputed relative projections:

```bash
.build/DerivedData/Build/Products/Debug/gliner-decide-metal \
  --compiled \
  --fast-attention \
  --warmup 8 \
  --iterations 100
```

Other useful modes:

```bash
# Eager MLX graph
.build/DerivedData/Build/Products/Debug/gliner-decide-metal --warmup 8 --iterations 100

# Explicit per-stage profile (forces synchronization and is intentionally slower)
.build/DerivedData/Build/Products/Debug/gliner-decide-metal --profile --warmup 1 --iterations 1

# Experimental custom residual + LayerNorm kernel
.build/DerivedData/Build/Products/Debug/gliner-decide-metal --compiled --fused

# Experimental custom relative-bias kernel
.build/DerivedData/Build/Products/Debug/gliner-decide-metal \
  --compiled --fast-attention --custom-relative

# Kernel correctness smoke tests
.build/DerivedData/Build/Products/Debug/gliner-decide-metal --kernel-test
```

## Metal System Trace

Record a warmed run without the intrusive stage profiler:

```bash
xcrun xctrace record \
  --template 'Metal System Trace' \
  --time-limit 15s \
  --no-prompt \
  --output Artifacts/metal.trace \
  --launch -- \
  .build/DerivedData/Build/Products/Debug/gliner-decide-metal \
    --compiled --fast-attention --warmup 5 --iterations 50
```

Inspect the trace with Instruments or export a table:

```bash
xcrun xctrace export --input Artifacts/metal.trace --toc
xcrun xctrace export \
  --input Artifacts/metal.trace \
  --xpath "/trace-toc/run[@number='1']/data/table[@schema='metal-gpu-intervals']" \
  --output Artifacts/metal-gpu-intervals.xml
```

## What has been optimized so far

1. Classification-only graph; no span/count work.
2. FP16 weights.
3. Q/K/V projection fusion into one matrix multiplication per layer.
4. Relative Q/K projections precomputed once, since relative embeddings are static.
5. Relative index tensors cached by sequence length.
6. Full MLX graph compilation and MLX fused SDPA.
7. Optional custom residual/LayerNorm and relative-bias kernels, both rejected
   by the current benchmark despite passing small correctness tests.

At this point the remaining gains are more likely to come from smaller dynamic
sequence buckets, batching for throughput, or a fully fused attention kernel
that also owns the QK product. A custom kernel should only be retained if it
beats the compiled MLX/Core ML baseline on a real request distribution.
