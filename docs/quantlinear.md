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
row-split K128 layout. Every ABI-3 matrix carries bits/group, including embedding
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
  as a function of batch size. Q5/Q6 decode each integer*FP16-scale into FP16
  **before** FP32 dot accumulation, including scalar and WMMA paths. Q8 retains
  its established FP32 scaled-integer dot contract, with FP16 or BF16 activations.
  Q4 canonical auto/reference also materializes FP16 decoded weights. The older
  Q4 fused research kernel retains FP32 group-scale arithmetic and is not admitted
  as the same qualified implementation.
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
Q4/Q5/Q6 `auto` use the safe chunked-dense path on both SM70/SM89. The fused
Q5 full-model state campaign failed the unchanged elementwise state gate, even
though logits/top-1 passed; `linear_implementation="fused"` is a separate
research capability and remains closed by `FUSED_LINEAR_QUALIFIED`. Codec
qualification alone never admits a new implementation. Actual native selected
families are exposed in `projection_info["linear_contracts"]`. ABI 3 rejects
size-compatible ABI-2 runtimes that still selected unqualified Q5 fusion.
SM89 Q5 WMMA is not selected merely because an SM70 kernel exists. Q6 keeps its
working chunked-dense implementation instead of requiring a nonexistent fused
kernel. Qwen runtime ABI is now 3 (operator ABI 18); old DSOs are rejected explicitly, and native
Config/Matrix/Layer size checks remain mandatory. PD/PP reject mismatched numerical
implementation/shadow settings before state transfer.

## Opt-in SM70 Q5 single-slot prefill unpack reuse

For a resident canonical Q5-G64 FP16 model prepared with `max_tokens=32`,
`model.create_batch([capacity], q5_prefill_reuse=True)` opts one batch into
sharing each decoded weight shard across the 16-row cuBLAS calls of a 17–32
row prefill step. Use a 32-row budget and `prefill_chunk_size=32` to reach
the path in `ContinuousScheduler`. This changes reuse only: it does not
substitute a new GEMM implementation, create a resident FP16 shadow or
change the scheduler.

The flag is off by default. The native constructor rejects a multi-slot batch
even if most slots would be idle. Decode, the vocabulary head and prefill at
most 16 rows are unchanged. Existing ABI 3 libraries still load for the default;
the optional path requires the additive constructor from a rebuilt Qwen DSO.
See [the V100 correctness and performance record](q5-prefill-reuse-sm70.md).

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

### Qualification outcome and compatibility

The real SM70 fused-Q5 campaign retained the unchanged logit gates (top-1,
relative L2 <= 0.02, cosine >= 0.999, KL <= 0.002) and state gates
(`rtol=0.01`, `atol=0.005`). The first trained case passed logits but failed
50 state components, so **fusion is not a qualified model default**. `auto`
uses chunked dense for Q4/Q5/Q6 on both supported architectures. A later L40
Q4 fixture also failed the unchanged private-state comparison (2/320 elements;
max absolute error 0.0074463), so its direct family is research-only as well. Its throughput is
reported as the safe native baseline, not as a fused-kernel speedup.

A16 describes activation precision, not one universal weight-rounding rule.
Q4/Q5/Q6 auto/reference materialize FP16 decoded weights. The older Q4 fused
research capability preserves FP32 group-scale association, is separately gated
and is not a qualified substitute for canonical FP16 decode. The native plan query reports the selected
family and shadow status, and loader admission validates that actual family.
Codec, activation-policy and implementation qualification are separate gates.

Rebuild the explicit library pair: **operator ABI 18, Qwen runtime ABI 3**.
ABI-2 runtimes are rejected even though their descriptor sizes match, because
those runtimes could still auto-select unqualified Q5 fusion. Weight-file
formats are unchanged; no checkpoint re-download/requantization is required.

### Qualified safe SM70 Q5 campaign (2026-11-17)

Canonical A16 Q5 `auto`/`dequant` is now admitted on SM70 after the real 64-layer
27B artifact passed four scalar trajectories/native replays and C4/8/16/32
grouped output and byte-identical private-state checks (7680 components).
This does not open fused/A8, low-bit TP/PP/MTP, Q4/Q6 on SM70, or establish
original-BF16/corpus quality. The frozen reference uses the **same Q5 artifact**.

The shared Q4/Q5/Q6 decoder now coalesces adjacent octets with aligned packed
reads/vector stores and safe offset/tail fallbacks. Arithmetic is unchanged:
integer times FP16 scale, rounded once to FP16. Both contiguous and indexed rows
use it. Native multi-token chunk output uses a strided 2D copy instead of one
copy per token; vocabulary-size-8193 tail fixtures match a frozen runtime pair
byte-for-byte. No full-model FP16 shadow is retained.

Two V100 independent native replicas, 16 input IDs / 32 forced outputs, three
clean repetitions, common end-to-end wall-time TPS (including prefill and HTTP):

| Concurrency | Frozen safe control | Optimized safe path |
|---:|---:|---:|
| 1 | 0.750 | 5.385 |
| 4 | 2.777 | 19.155 |
| 8 | 5.307 | 34.606 |
| 16 | 5.208 | 28.973 |
| 32 | 6.717 | 30.433 |

These are not decode-only TPS, TP2, a llama.cpp comparison or historical fused
results. The complete reference arm was reused from an earlier protected window
with matching artifact/DSO/workload/config checks; both windows and the earlier
campaign's incomplete overall status are preserved. C16 repeat variance and
non-monotonic C16/C32 scaling remain unresolved, not filtered out.

Separate one-step C16 Nsight attribution showed row decode 1280.687→91.544ms;
GEMM is now the largest kernel family. Counter occupancy and DRAM bandwidth
were not measured. CPU/latency/first-token availability/RAM/VRAM, sampled GPU
busy/power/clocks/temperature, trace gaps and production restoration receipts
are in [the complete campaign](../operations/pr14-quantlinear-fix-20261117-v1/README.md).
