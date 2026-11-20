# V100 projection-budget tuning: explicit opt-in, overall goal incomplete

## Result

Same native binary and numerical profile; three unprofiled repeats per cell:

| Requests | Rows32 HTTP TPS | Rows256 HTTP TPS | Native change | Reused llama.cpp HTTP TPS |
|---:|---:|---:|---:|---:|
| 1 | 22.952 | 22.925 | -0.12% | 24.536 |
| 2 | 35.837 | 36.291 | +1.27% | 37.907 |
| 4 | 52.038 | 53.248 | +2.33% | 50.254 |
| 8 | 63.235 | 65.407 | +3.43% | 67.287 |

C4 exceeds the pinned external HTTP control by **5.96%** in this fixed-work
comparison. **C1/C2/C8 still trail**; the full >=5% single/parallel goal remains
active. Native/GGUF Q8 encodings/internal arithmetic are not identical, and one
winning cell does not prove universal superiority or complete acceptance.

This is primarily a **prefill improvement, not a decode improvement**. Ordinary
single native post-first is 26.274 -> 26.259 TPS (-.06%); C8 native post-first
cohort is 94.322 -> 94.198 TPS (-.13%). These small regressions are retained.
C2/C4 post-first cohort rates are 44.491 -> 44.553 / 71.780 -> 72.145 TPS.
Native rates exclude first outputs and HTTP admission, unlike the HTTP table.

MTP serving stays at rows32 and is unchanged. Its existing compact-GDN results
are explicitly reused, not remeasured: native post-first 35.691 TPS / HTTP
29.381 TPS. Best successfully tested external MTP2 remains reported post-first
49.803 TPS / HTTP 30.173 TPS. No new MTP2 or 256-row MTP claim.

## What changed

`NativeQwenModel.from_q8(max_tokens=...)` already supports this allocation/
scheduling parameter and defaults to 128. The benchmark adapter had explicitly
restricted it to 32 rows. New `--projection-rows` options in the HTTP adapter,
profiler and trained qualifier expose and record the 32/128/256 controls.
**This is configuration tuning, not a new native kernel or architecture.**
The public native API and adapter default32 remain unchanged; larger budgets
are explicit opt-in because they consume substantially more memory.

For the fixed 23-token prompts, C1/2/4/8 prefill group calls fall from
**1/2/3/6 -> 1/1/1/1**. Maximum actual grouped rows are 23/46/92/184 at budget256.
Total native input rows stay **54/108/216/432**, including consumed generated
inputs, with identical positions and output IDs. Context capacity remains 256
per slot; a projection-row budget is not a KV/context size. No prompt caching,
precision reduction, omitted work or external forward engine is introduced.

Single-repeat C1/C8 screens, before promotion:

| Budget | C1 HTTP TPS | C8 HTTP TPS |
|---:|---:|---:|
| 32 | 22.925 | 63.205 |
| 128 | 22.923 | 64.734 |
| 256 | 22.982 | 65.372 |

Budget256 has the highest eligible C8 ratio. The internal screen required C8
>3% gain and no >1% C1 regression, then a complete three-repeat matrix. This
screen rule does not replace the external >=5% acceptance requirement.

To use the measured configuration, explicitly pass `max_tokens=256` when creating
the native resident model before `create_batch(...)`; for the loopback benchmark
adapter add `--batch-target --projection-rows 256 --projection-profile q8-residual`.
Scalar/MTP and historical rows32 runs retain their explicit old geometry.
No production-server, automatic universal tuning or larger-context claim.

## Correctness and reused native code

35 added tests pass: large-budget full hybrid-model original-FP16/residual,
single/host-TP ragged eight-slot logits and all private cache bytes; selected
reset/continuation, uneven generation/EOS, native no-Torch generation, row/position
accounting and increased workspace accounting. Large fused operators cover
T56/127/128/184/255/256, short/large mixed rows, K tails and pitched sentinels,
with exact original output/workspace bytes and sampled independent FP64 gates.

Both budget128 and256 execute fresh residual trained checks against complete
hash-pinned rows32 signatures: **98 full-vocabulary vectors**, four greedy
trajectories/final states, eight heterogeneous batch slots and selected reset/
continuation logits/positions/caches. Every byte matches. Qualifier metadata
separates control32 from the candidate budget. Original independent target
thresholds and unexecuted trained FP16 references remain unchanged and explicitly
reused. No broad original-BF16/corpus-quality claim or fresh large payload scan.

Unchanged MTP qualification/timing are reused, not rerun. Source inspection
clarifies that the prior MTP qualification helper itself used max_tokens=128,
whereas its HTTP speed campaign used32. Neither proves new 256-row MTP support.

Full V100 suite: **1679 passed / 440 skipped**. Local CPU suite:
**769 passed / 1350 skipped**; skips are not GPU passes. The initial CPU test
import-path error was corrected before GPU qualification; no numerical failure
was suppressed. All 27 previous immutable capsules are preserved, including the
existing legacy unavailable-path exception. No commit/push or local compilation.

There is **no native/CUDA build change**. Every arm uses
`/home/data/wangyue/artifacts/venturi-gdn-compact-model-v2/build/0849ea2437e65c8a/libventuri_blocks.so`,
SHA256 `e8090c0531b3b794f34ca267baf1b61fcb84ba6d58c1e4550795ad330ae6965a`.
All nine compiled-source hashes are checked; blocks ABI16 and all kernels,
quantization/FMA/reduction rules and FP16 KV remain unchanged.

## Timing, CPU and memory tradeoff

Same official-original-derived Q8 target, declared two-pass residual profile,
resident host TP on physical GPUs1,0, eight slots/capacity256, identical formatted
23 prompt IDs and 32 fixed greedy outputs. EOS is ignored only for the fixed-work
speed case; correctness cases test EOS independently. HTTP includes the same
25-ms admission window; loading, warmup/reset, diagnostics and profiling are
excluded. Aggregate TPS uses emitted outputs/global cohort wall, never summed
concurrent latencies or average request rates.

| Requests | Native TTFT seconds, rows32 -> 256 | Server CPU core-s/cohort | Mean client completion seconds |
|---:|---:|---:|---:|
| 1 | .174351 -> .174417 | 1.370 -> 1.370 | 1.39351 -> 1.39514 |
| 2 | .350821 -> .328222 | 1.760 -> 1.737 | 1.78498 -> 1.76242 |
| 4 | .684764 -> .636319 | 2.433 -> 2.380 | 2.45770 -> 2.40176 |
| 8 | 1.382975 -> 1.245949 | 4.047 -> 3.913 | 4.04411 -> 3.91013 |

Native batch TTFT includes cohort prefill, not streamed client TTFT. Raw reports
retain all trial/min/max latency and row/cursor/counter denominators.

**Memory cost is real:** non-spec primary/peer NVML peaks rise
**20235/17227 -> 21957/17297 MiB**, including unrelated allocations and CUDA
context. Primary +1722 MiB buys only a 3.43% C8 HTTP gain in this workload; this
is why the adapter's historical default is not silently raised. Projection
quantization scratch grows **1253376 -> 10027008 bytes/device**, plus other model/
request workspaces. No memory saving or expanded slot context is claimed.

C8 timed-window primary/peer NVML busy is 79.21/54.40 -> 79.85/55.80%; these
percentages are not achieved occupancy/bandwidth. Raw power, temperature, clocks,
controller activity and sample-window counts remain separately recorded.

## Separate profiling

Fresh rows256 candidates versus explicitly reused, hash-pinned rows32 controls;
all use **23 prompt / 8 output** diagnostics, not the 32-output HTTP cells:

| Trace | Host span ms, rows32 -> 256 | Launches | Inter-activity gaps ms |
|---|---:|---:|---:|
| C1 | 468.30 -> 469.44 | 13504 -> 13504 | 80.12 -> 80.16 |
| C8 | 2034.38 -> 1871.46 | 33562 -> 20672 | 174.74 -> 107.32 |

C8 summed projection time across GPUs is 1746.86 -> 1707.83 ms; compact GDN
263.81 -> 243.79 ms. These sums overlap across devices, and gaps concern this
process's activities, not whole-GPU idleness. Fewer group forwards/launches help
prefill; the unprofiled decode rates still do not improve.

C8 H2D/D2H counts fall **8557/6680 -> 5272/4120**, but payloads remain
**1990494080 / 1990493184 bytes** in the two directions. D2H activity sums
actually rise 223.49 -> 234.22 ms; do not infer bandwidth savings from fewer
calls. C1 counts/payloads stay unchanged. No paging or transport redesign.

The same exact library has unchanged compiled registers/shared-memory limits;
only grids and per-call row shapes differ. Driver API ceilings are **not achieved
occupancy**. SM/DRAM/cache/stall counters remain unavailable under unchanged
`RmProfilingAdminOnly: 1`; NVML busy/controller percentages are not substitutes.
No privilege modification or repeated denied counter campaign.

Raw Nsight/SQLite files remain remotely retained with size/SHA256 records.
All three owned workers are terminal, the benchmark port is closed, and original
unrelated physical-GPU allocations **3651/3189 MiB** are restored. No unrelated
process/service was stopped.

## Evidence and unfinished acceptance

`design.md` explains scheduling and test scopes; `metrics.json`/`summarize.py`
retain/recompute HTTP, native timing, CPU, telemetry and reused external controls.
`v100/model-v1` contains exact reused-build and fresh budget qualifications;
`campaign-v2` contains screens/full repeats; `final-v3` contains separate timelines
and compiled resource queries. Terminal/source/evidence manifests pin executed
source and cleanup. Strongest external controls are explicitly reused, not
remeasured or selected weaker; failed prior configurations are not zero-speed
controls. See `next-bottlenecks.md` for follow-up hypotheses, not implemented gains.

Overall external acceptance, single-request/deeper MTP performance, paging/
continuous batching, sampling, whole-model graphs and broad/long-context quality
remain unfinished. Do not turn the C4 result into a complete-goal claim.
