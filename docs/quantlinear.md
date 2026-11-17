# Canonical row QuantLinear

This is an owned operator/runtime mechanism, not an external model engine.
Model adapters bind roles; they do not select bit-specific CUDA kernels per step.

```
artifact matrix descriptor (format, shape, planes)
  -> storage admission -> architecture execution qualification
  -> numerical policy + implementation -> prepared LinearPlan
  -> direct kernel / explicit chunked dense reference / explicit activation INT8
```

## Formats and mixed checkpoints

`q4_g64_fp16`, `q5_g64_fp16`, `q6_g64_fp16`, `q8_g32_fp16` retain their canonical
row-split K128 layout. Every ABI-2 matrix carries bits/group, including embedding
and output head. `venturi.matrix_format` remains the default for old artifacts;
optional `venturi.matrix_formats` is a JSON object mapping **logical matrix roles**
to canonical codecs, e.g. `{"text/layers/0/mlp/down":"q6_g64_fp16"}`.
Unknown roles, float-parameter overrides, unsupported groups/layouts and conflicting
tied embedding/head formats fail closed. Nothing is re-encoded during loading.

`quantization/linear.py` holds reusable load-time admission/diagnostics;
`ops/csrc/row_codec_dispatch.cuh` holds stateless kernel capabilities. Native plans
are prepared at load and indexed by validated codec, without a map search in the
projection hot path. Another model adapter can reuse both without importing Qwen.
Capabilities are not execution/quality qualification; the architecture gate in
`quantization/codecs.py` is separate and remains closed without evidence.

## Explicit numerical policies

- `projection_profile="fp16"` (existing default): activations are never quantized
  as a function of batch size. Q4/Q5/Q6 decode each integer*FP16-scale into FP16
  **before** FP32 dot accumulation, including scalar and WMMA paths. Q8 retains
  its established FP32 scaled-integer dot contract, with FP16 or BF16 activations.
  Kernel association may differ: do not promise bit equality across GEMM families.
- `projection_profile="int8"`: explicit one-pass group32 activation INT8 with
  FP32 activation scales; existing Q8 weights or SM70 Q5 weights only. This changes
  numerical arithmetic/precision, is never an automatic fallback, and is not an
  A16-equivalent quality claim. Its separate model qualification gate currently
  remains **closed**: an independent tiny fixture measured relative L2 0.0553,
  exceeding the unchanged 0.02 gate. Kernel registration/plumbing tests do not
  open this gate. Workspace is `tokens*ceil(K/32)*36`, shared within
  an explicitly scoped group of projections, not 36 bytes per activation column.
- Existing `q8-residual*` profiles retain their explicit two-pass Q8 policy and
  existing TP/column/head-GDN constraints. They reject non-Q8 projections.

`linear_implementation="dequant"` selects a bounded chunked FP16 decode+cuBLAS
reference for Q4/Q5/Q6 without storing an entire FP16 model. Mixed Q8 projections
retain their existing A16 kernel. At least one Q4/Q5/Q6 matrix is required.
`dequant_resident=True` remains the separate explicit all-Q4/Q5/Q6 FP16-shadow
mode. Neither option changes automatically on OOM.

Native MTP remains all-Q8-only: both Python and the native ABI reject row-codec/mixed targets before the draft can misinterpret their vocabulary planes.

Row-codec/mixed execution currently requires resident, single-device FP16 without
TP/PP; unsupported combinations reject before weight allocation. This refactor
is not a claim of Q4/Q5 TP, arbitrary GPU architectures, or cross-host execution.
SM89 Q5 WMMA is not selected merely because an SM70 kernel exists. Q6 keeps its
working chunked-dense implementation instead of requiring a nonexistent fused
kernel. Qwen runtime ABI is now 2; old DSOs are rejected explicitly, and native
Config/Matrix/Layer size checks remain mandatory. PD/PP reject mismatched numerical
implementation/shadow settings before state transfer.

## Qualification and performance

Operator checks must include independent decode/math, odd K tails, partial row
and token tiles, explicit error returns and workspace bounds. Integrated checks
compare logits, private GDN/KV/convolution state and greedy tokens across chunks
and the WMMA crossover. Real-model checks use the same pinned artifact; synthetic
fixtures and capacity estimates are not full-model generation or TPS evidence.

`tools/night_blast.py` emits `valid=false`, `aggregate_tps=null`, and exits nonzero
if any request fails or has invalid/incomplete output. Partial successful work is
retained only as a separately named diagnostic. The loopback router has an explicit
128-connection accept backlog and reports upstream transport failures as 502,
without silently retrying generation. These are benchmark tools, not production
routing defaults. Historical campaign JSON is preserved, including failed runs;
its advertised historical TPS must not be treated as qualified current results.
