# SM70 Q5 prefill unpack reuse

This option is deliberately not a continuous-batching policy. Equivalent
prompt text and greedy IDs do not imply bitwise state equality after regrouping
GEMM rows: on WZU the earlier multi-request strategy moved the first difference
into layer-0 projections with identical inputs and propagated differences into
GDN/KV. It is not admitted by this option.

Given an already loaded canonical resident FP16 Q5-G64 SM70 model with
`max_tokens=32`:

```python
from venturi.runtime import ContinuousScheduler

batch = model.create_batch([1024], q5_prefill_reuse=True)
scheduler = ContinuousScheduler(batch, token_budget=32, prefill_chunk_size=32)
```

The native batch constructor requires a single-slot lease, FP32 GDN side
weights and aligned Q5-G64 trunk projections. The opt-in cannot be applied to
a multi-slot batch even when only one of its slots is active. The default
constructor, scheduler defaults, decode, vocabulary projection and resident
weight layout remain unchanged. Ops/Qwen ABIs stay 18/3; existing libraries
work for the ordinary constructor, and opt-in requires a rebuild. No serving
flag is added.

Prefill segments of 17–32 rows use the existing bounded weight shard scratch.
Each shard is decoded once and reused on the same stream by the same cuBLAS
16-row and remainder calls used by two independent fixed-16 steps. State
operators still process each request causally. No input/output math mode
change, persisted FP16 weights, new decoder kernel or cross-request state reuse
is introduced. The bit-equality claim refers to the fixed-16 single-request
reference on the tested stack, not arbitrary model shapes, regrouping, driver
versions or prepared states.

## Reproduce

Run on a dedicated V100 with the same Q5-G64 checkpoint already available.
Build the candidate and frozen reference Qwen DSOs against the same ops
ABI 18 build. Pin the CUDA-visible device to the supplied UUID.

```sh
PYTHONPATH=src CUDA_VISIBLE_DEVICES="$GPU_UUID" \
  python tools/qualify_qwen_q5_prefill_reuse.py \
    --artifact "$Q5_ARTIFACT" --ops "$OPS_SO" --qwen "$QWEN_SO" \
    --reference-qwen "$FROZEN_QWEN_SO" --gpu-uuid "$GPU_UUID" \
    --cases benchmarks/v100-q5-prefill-reuse/cases.json --out result.json
```

The checked-in cases retain the exact synthetic stimulus used for each arm
and truncate the unused rows and labels from the larger original input file.
The output records the supplied input-file digest. The shape-only
`--profile fixed16` / `--profile reuse32` modes bracket 128 tokens with
CUDA profiler APIs for a separate Nsight Systems pass.

## WZU V100 qualification (2026-11-21)

The published [results](../benchmarks/v100-q5-prefill-reuse/results.json)
are for the additive constructor in this PR, not the earlier diagnostic
build. One V100-PCIE-32GB, driver 580.178.04, CUDA 12.9.86, PyTorch
2.7.1+cu126; same resident Qwen 27B Q5-G64 A16 checkpoint, original ops ABI 18
and frozen `main` `790163af` as the C4 reference. The checkpoint SHA-256
in the results was carried from the earlier complete hash; file size/mtime
were checked unchanged.

- **C1**: 48/127/320/768/768 input tokens, reference-forced trajectory. All
  30 logit tensors, 640 private GDN/KV tensors and greedy IDs were bitwise
  equal to fixed16 (including the unaligned 127-token tail).
- **C4 unchanged**: 48/128/320/768 inputs, 16 outputs each, chunk 16 and
  global budget 32. The ordinary four-slot constructor with the PR DSO
  produced the same 64 greedy IDs and 512 bit-identical state tensors as
  the frozen main DSO. The opt-in is rejected for four slots in both
  Python and native tests. This does not qualify the earlier mixed scheduler.
- Synthetic SM70 fixture: 16/17/31/32/65-token inputs, nonconstant weight
  planes and a projection shard past N=8192; exact FP16/FP32 bytes,
  greedy tokens and state checked.

Unprofiled native replay, 768 input / 16 output tokens, one 1024-token
slot and a 32-row workspace, no EOS early stop; separate 128-token warmups
per route, order fixed16/reuse32/reuse32/fixed16. TTFT, wall, CPU and the
single-request largest inter-token gap are medians of two waves. TPS uses
total generated tokens divided by total elapsed wall time. This excludes
loading, warmup, capture, HTTP, MTP and prefix reuse.

| Single-slot route | TTFT (s) | Max ITL (ms) | Output TPS | Wall (s) | Host CPU (core-s) |
|---|---:|---:|---:|---:|---:|
| fixed16 | 16.264 | 195.1 | 0.8339 | 19.188 | 19.333 |
| reuse32 | 13.859 | 194.8 | 0.9536 | 16.779 | 16.903 |

On this replay, TTFT fell by 14.8% and output
TPS rose by 14.4%, with all measured greedy
sequences matching. A separate 128-token Nsight Systems capture showed
5,760 → 2,880 unpack launches and
699.8 → 350.2 ms of unpack activity.
GEMM and CUDA timeline gaps are in the results. GPU telemetry (busy,
VRAM, power, clocks and temperature) is sampled separately. Achieved SM
occupancy/DRAM/stall counters were not collected under
`RmProfilingAdminOnly=1`; timeline and NVML are not replacements for them.
