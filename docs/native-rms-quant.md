# Native post-attention RMSNorm + residual-A8 packing (SM70)

The opt-in `q8_rms_quant` build flag combines the **primary rank's**
post-attention RMSNorm and the two-pass activation pack used by column-parallel
MLP gate/up projections. It does not replace model-weight Q8 quantization.

## Numerical and dispatch contract

- Same 256-thread FP32 RMS reduction order as the unfused kernel.
- Round the normalized value to FP16 **before** both activation-quantization
  passes. Preserve codes/scales and FP32 TP partials, not just output tokens.
- Materialize the same FP16 normalized tensor for the peer's existing
  host-staged transport. The peer still packs its received FP16 values normally.
- No pointer-based quantization cache. The packed input is consumed immediately
  under the same model lock/stream; existing TP event fences remain in place.
- Selection requires SM70, resident FP16 two-pass column-MLP, hidden size 5120,
  and 1–32 rows. Other shapes/profiles keep the original path. Input/final
  norms and attention/GDN input packing are not fused by this change.
- Eight-group chunks repeat RMS statistics to expose more CTAs for small
  decode batches. This trades extra reads/arithmetic for parallelism, so it
  must not be described as fewer operations or universally lower bandwidth.
- Disabled by default. Compare otherwise identical OFF/ON libraries before
  opting in; a faster isolated producer is not proof of faster generation.

## Explicit remote build

Run on the V100 host, not the development Mac:

```sh
python -m venturi.ops.build --component blocks --architecture sm70 \
  --q8-row-pair --q8-swiglu-quant --q8-rms-quant \
  --nvcc /usr/local/cuda/bin/nvcc --output-dir /absolute/path/build
```

Select that library with the existing `--blocks-library` option and explicit
`--dtype fp16 --weight-mode resident`, host TP on the second V100, and the
`q8-residual-column-mlp-head-gdn` projection profile. Build without
`--q8-rms-quant` for the matched control. Build identity hashes include the
header and flag; neither import nor generation compiles automatically.

## Qwen3.8 checkpoint compatibility

The inspected Qwen3.8 checkpoint declares the existing `qwen3_5` /
`qwen3_5_text` schema. Its BF16 GDN norm parameters are widened exactly to the
existing FP32 payload, just as BF16 A_log/dt_bias can be widened; existing FP32
parameters are never narrowed. Shape/config validation remains strict.

The extra `output_gate_type: "swish"` field does **not** justify changing the
attention formula: the inspected [official HF implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
uses sigmoid attention output gating and SiLU for gated GDN normalization.
This integration preserves those formulas. Support is for the checked schema,
not arbitrary models sharing a marketing name.

## Evidence boundaries

`benchmarks/v100-rms-native/` retains the failed transfer/acquisition attempts,
fixed-revision source hashes, exact executed source/build records, correctness
logs, raw unprofiled timing reports and separate profiler data. The full-model
comparison is native RMS OFF versus ON using one identical official-source Q8
artifact, **not** production Uncensored GGUF weights or a llama.cpp comparison.
Fixed-work short-context results do not establish long-context quality,
BF16-versus-Q8 quality, MTP performance, or HTTP serving latency.
