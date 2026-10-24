# V100 Q8 projection scheduling: guarded T8 retained; naive WMMA rejected

## Decision

Retain the SM70-only, guarded DP4A T8 schedule. It preserves the existing
arithmetic profile's output bytes and improves the eight-request fixed cohort
by **11.86%**. Do **not** select the standalone WMMA prototype in model execution:
it loses the relevant projection microbenchmarks. Single-request post-first
throughput is essentially unchanged. **The external performance goal is not met.**

All six owned workers finished successfully. Tests, commands, trial outputs,
telemetry, unsuccessful variants, source deltas and remote retention hashes are
preserved here. Nothing was committed or pushed. No local Rust/CUDA compilation
or driver privilege/policy change was used.

## What changed

`q8_dp4a.cuh` selects `dot<8,Passes>` only on SM70, with at least eight tokens,
at least 1,024 output rows, and a shard's padded weight-code plane larger than
6 MiB. T1/T2 and the unqualified SM89 dispatch are unchanged; smaller/cache-resident
shards retain T4. The per-output quantization and FP32 reduction order are
unchanged. Unguarded T8 regressed the small 512-by-5,120 shard at T8 by about
10%; that experiment is retained rather than hidden. This is not the earlier
rejected FP16-kernel T8 variant.

A separate `q8_wmma.cuh`/`q8_wmma.py` research operator uses SM70 FP16 WMMA,
exact FP16 conversion of Q8 codes, unmodified FP16 input operands, per-group
FP32 accumulation and explicit scale application. Split-K warps 1/2/4/8 change
reduction arithmetic and are separately labelled. It is **not** dispatched by
native generation, and has no trained full-model quality qualification.

No native batching, MTP, TP transport, model ABI, checkpoint layout or cache
ownership semantics changed. ABI 16 remains intact. Model quantization scratch
is model-owned and mutex protected; private request state remains separate.

## Matched unprofiled results

Two V100-PCIE-32GB GPUs, logical devices `1,0`, explicit pinned-host TP;
23 formatted prompt IDs, 32 fixed greedy outputs, ignore EOS for this speed
case, no prompt cache reuse, FP16 KV, eight admitted slots, context capacity
256, maximum projection rows **32**, and 25 ms cohort admission included in
HTTP wall time. One screening repeat preceded promotion. Every promoted cell
below has three unprofiled timed repeats after warmup; all emitted IDs match
the pinned case. Aggregates use all output tokens / global cohort elapsed
wall, never summed concurrent request durations.

| Concurrent requests | Qualified T4 control HTTP TPS | Guarded T8 HTTP TPS | Gain | Fresh llama.cpp non-spec HTTP TPS |
|---:|---:|---:|---:|---:|
| 1 | 19.051 | 19.410 | 1.89% | 24.536 |
| 2 | 27.722 | 28.575 | 3.08% | 37.907 |
| 4 | 34.344 | 35.568 | 3.56% | 50.254 |
| 8 | 38.603 | 43.181 | 11.86% | 67.287 |

The C1/C2/C4 gains are prefill improvements, not decode improvements. Native
post-first cohort TPS is 23.083 -> 23.103 (C1), 37.210 -> 37.218 (C2),
49.515 -> 49.568 (C4), and **58.820 -> 66.055** (C8). This denominator is
31 outputs per request / (last native finish minus the shared first-token
timestamp); it is not HTTP throughput, streaming client TTFT, or directly the
competitor's timing metric.

| Requests | Wall seconds old -> new | Server CPU core-seconds old -> new | Mean client completion seconds old -> new | Native TTFT seconds old -> new |
|---:|---:|---:|---:|---:|
| 1 | 1.67969 -> 1.64860 | 1.657 -> 1.627 | 1.67935 -> 1.64793 | .29727 -> .26656 |
| 2 | 2.30862 -> 2.23971 | 2.283 -> 2.213 | 2.30737 -> 2.23864 | .59912 -> .53207 |
| 4 | 3.72698 -> 3.59879 | 3.703 -> 3.577 | 3.72500 -> 3.59722 | 1.17573 -> 1.05200 |
| 8 | 6.63162 -> 5.92857 | 6.633 -> 5.927 | 6.62753 -> 5.92425 | 2.37915 -> 2.13585 |

Native TTFT starts inside the native call; it excludes HTTP/prior queue/reset
work. Batch TTFT includes the whole cohort prefill. Full trial values and client
latency ranges are in `metrics.json` and the raw timed reports.

One-proposal C1 MTP: HTTP 23.465 -> 23.992 TPS, post-first **30.415 -> 30.423**
TPS (effectively flat), native TTFT .30789 -> .27771 s, CPU 1.367 -> 1.337
core-seconds, wall 1.36372 -> 1.33377 s. Acceptance is 14/16 (87.5%), with two
rollback/replays and no forced proposals. Fresh llama.cpp MTP1 is HTTP 29.117
TPS and reported post-first 46.681 TPS (31 / mean reported `predicted_ms`).
Its HTTP and internal decode denominators are kept separate.

The pinned production llama binary SHA256 is
`e5b1b5782cd5b4acec5316ac5bd40123e0b0c9826643ae4148b0ec6e359f5fa2`.
The new non-spec C8 baseline is stronger than the prior valid 66.83 result;
both are retained, with no selection of the slower control. Competitor best
supported configuration/depth and a complete successful parallel speculative
comparison are still outstanding. Earlier failed configurations are not zeros
or native wins.

## Checkpoints, precision and correctness

Official Qwen3.5-27B revision:
`fc05daec18b0a78c049392ed2e771dde82bdf654`.
Target ID `3d635b87db004f42aefd6730e372e530`, SHA256
`b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8`.
MTP ID `1a4307f5e7954a929d7ad70539875915`, SHA256
`a1a18b5f828d9b1cc1028fdc8463564aa4ffdbfe527311e1c0baa48cccfc1641`.
Original-source GGUF SHA256
`eb07277161915232e227a38608fc3ea80c0c17dc1c1583a9f3c5671edb6b5513`.
These are reused, previously verified payload hashes, not fresh scans here.

Speed runs use the explicit `q8-residual` target profile: original group-32
Q8/F16-scale weights, FP16 external activation/KV storage, two group-32 A8
high/residual activation passes with FP32 scales/reduction. The original
`fp16` projection profile remains the default; the MTP draft stays `fp16`.
Native/GGUF Q8 encodings, rounding, small projections and activation arithmetic
are not identical. No hidden precision change was made to obtain the speedup.

Qualification before performance:

- 150 standalone WMMA tests (numeric/tail/subnormal/cancellation, graph/alias,
  native-only, nonfinite, scalar/batch/permutation and CPU preflight coverage).
- 20 independent old/new DP4A variant cases, both pass counts, T1/2/8/23/32,
  inactive small and active threshold/tail geometries: exact output/workspace
  bytes, FP64 reference gates and untouched row bounds.
- Both trained `fp16` and `q8-residual` profiles: 98 complete-vocabulary vectors
  each, four original greedy trajectories, final scalar private caches,
  eight mixed batched requests and selected-reset continuations byte-exact
  to the pinned qualified control; native generation disallows Torch math.
- The fixed independent CPU/quality gates are reused through these exact bytes:
  top-1, relative L2 <= .02, centered cosine >= .999, KL <= .002. This is not a
  fresh independent BF16 reference run or broad corpus-quality qualification.
- Trained MTP committed-state reference is explicitly reused and hash-pinned
  from the prior capsule: MTP/draft code and T1/T2 dispatch did not change,
  and fresh trained qualification includes the changed prefill. Fresh MTP
  timing checks pinned IDs/counters; the final suite retests transaction edges.
- Final V100 suite: **1,206 passed / 440 skipped**; local non-GPU suite:
  **761 passed / 885 skipped**. Local skips are not counted as GPU passes.

The qualification report's inherited prose calls the control "unshared";
that word is stale. Its control SHA256 below is the **shared-input T4** binary;
report fields, commands, loaded maps and exact comparisons identify it correctly.

Control library SHA256:
`861b528b0d7464c53ce51a4f29140984bda1ebfccf55b539aa82d5492f0f1ed4`.
Candidate:
`/home/data/wangyue/artifacts/venturi-wmma-guarded-v4/build/8537a2e208c36c23/libventuri_blocks.so`,
SHA256 `ccf03d8caa89ced089b80cd5b8eb49cb494e53baf855c78c99e9c8f5ed9627c5`.

## Standalone Tensor Core result (not inference TPS)

Eight TP-first real projection shapes at T1/2/8/32, six arms and three timed
repeats per case. Graph waves contain nine operator calls. Fixed split-4 WMMA
median speed relative to residual DP4A is approximately .199x/.240x/.573x/.832x
for T1/2/8/32: slower, including at T32. Across all 32 cases sampled top-1
matches; max FP64 relative L2 is .000261285, max relative L2 versus the original
FP16 projection is .0000319372. This does not qualify a full WMMA model.
Only this naive group-scaled WMMA design is rejected, not Tensor Cores generally.

## GPU diagnostics and remaining bottlenecks

Unprofiled C8 NVML, primary physical1 / peer physical0:

- GPU busy 85.72/63.94 -> 83.74/59.76%; memory-controller busy
  36.61/34.03 -> 22.61/19.90% (**not GB/s**).
- Power 135.25/106.65 -> 132.83/101.73 W; temperature
  43.70/40.45 -> 43.84/40.48 C. SM clock 1,380 MHz, memory 877 MHz;
  no clock/policy changes.
- Native device used memory 20,235/17,227 MiB unchanged; MTP
  22,435/17,227 unchanged. Fresh llama C8 peak 17,521/17,983 MiB;
  MTP peak 18,373/18,835. These are whole-device samples including unrelated
  resident allocations, not isolated allocator peaks.

Separate 23-prompt/**8-output** model traces, never used for throughput:
control traces reused from the hash-pinned preceding capsule; candidate freshly
profiled with CUDA event tracing disabled. C1 kernel count 15,808 unchanged;
DP4A dot summed device time 625.29 -> 564.92 ms, quantize 11.57 -> 11.56,
GDN 56.34 -> 56.36, timeline gaps 103.71 -> 102.74 ms. C8 kernel count
37,306 unchanged; dot 3,926.86 -> 3,239.06 ms, quantize 32.14 -> 32.28,
GDN 490.65 -> 492.04, gaps 214.49 -> 214.40 ms. Projection is still
84.01% of summed C8 kernel time; that is not achieved compute utilization.
H2D/D2H traffic is unchanged: C8 about 1.990 GB each direction.
No WMMA kernels run in these model traces. C8 has 9,584 T8 and 832 T4 dot calls;
C1 has 736 T8 prefill, 5,616 T1 and 64 T4 calls.

DP4A T8: 124 registers/thread, 128 threads/block, no spill, four resident blocks,
**25% theoretical** occupancy vs T4's 78 registers and 37.5%. Lower theoretical
occupancy nevertheless runs faster. WMMA uses 71 registers, 4/8/16/32 KiB shared
memory and 37.5% theoretical occupancy. Static SASS has 32 HMMA instructions
per WMMA function and separate primitive traces confirm dispatch; static
instructions are not dynamic tensor utilization. Achieved occupancy, compute,
DRAM bandwidth, cache/stall counters remain unavailable under
`RmProfilingAdminOnly: 1`; no repeated denied counter campaign was attempted.

Read-only copy attribution (`copy-audit.json`, `copy-api-partition.json`) finds
5,208 primary D2D `cudaMemcpy2DAsync` calls inside the C8 native range: summed
CPU API duration **498.68 ms**, compared with 16.89 ms summed device duration
for **all** D2D copies. CPU API time includes dependency waits/instrumentation;
it is neither entirely removable overhead nor a speedup prediction. Source
inspection shows the primary DP4A shard writes compact scratch, then copies
into the full-N output pitch. Next test: direct full-N-pitched primary output,
with exact sentinel/geometry tests and all peer fences preserved. Keep the old
contiguous ABI and original FP16 path. Screen C1/C8 before promotion. This is
a hypothesis, not an external-win claim.

See `metrics.json` for recomputed rates/telemetry, `v100/` for raw evidence,
`launchers/` for execution commands, and the manifests for frozen source and
previous-capsule integrity. Large profiler/binary artifacts remain at the
hash-recorded remote paths. All owned processes/port were absent at terminal
audit, and unrelated baseline allocations returned to 3,651/3,189 MiB idle.
