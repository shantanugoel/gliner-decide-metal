# GLiNER2.5-Decide Metal runtime

A classification-only Swift/MLX runtime for the exact
[`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide)
checkpoint. It does not substitute `gliner2.5-small`, `base`, `multi`, or
`Decide-1B`.

The implementation ports the checkpoint's DeBERTa-v3-large encoder and GLiNER
classification MLP. GLiNER span, relation, and count heads are intentionally not
loaded because the Core ML reference path used as the baseline exports only the
classification path.

## Fast-decisions baseline

The first full comparison uses the public `fastino/fast-decisions` development
split: 17 domains × 100 examples = 1,700 examples, dataset revision
`1a33070c`. Accuracy is exact-set match against the reference label set.

| Runtime | Accuracy | Macro accuracy | Throughput |
|---|---:|---:|---:|
| Native `fastino/GLiNER2.5-Decide`, MPS, batch 8 | **64.41%** | 64.41% | 13.46 rows/s |
| Swift/MLX runtime, compiled GPU graph, batch 8 | **64.24%** | 64.24% | **28.72 rows/s** |

The native run is the base-model reference. The Swift run groups inputs into
L256/L512 buckets, evaluates all heads/labels through the same preprocessing,
and excludes model initialization plus one warmup batch from throughput.
Native/Swift prediction agreement is **99.12%**; the three differing examples
account for the 0.18-point accuracy difference.

Reproduce with:

```bash
uv run --python 3.12 --with-requirements Tools/requirements.txt \
  python Tools/evaluate_fast_decisions.py \
  --batch-size 8 \
  --length 256 \
  --artifacts Artifacts/fast-decisions-eval \
  --output Artifacts/fast-decisions-eval/baseline-results.json
```

The complete per-domain results and predictions are written by the evaluator;
large model/data artifacts are ignored by git.


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
