# V100 compact GDN: retained, external acceptance incomplete

## Measured result

Three-repeat unprofiled, fixed-work native engine measurements on two V100s:

| Requests | Previous HTTP TPS | Compact GDN HTTP TPS | Native gain | Reused llama.cpp HTTP TPS |
|---:|---:|---:|---:|---:|
| 1 | 22.128 | 22.964 | 3.78% | 24.536 |
| 2 | 33.824 | 35.812 | 5.88% | 37.907 |
| 4 | 47.836 | 52.050 | 8.81% | 50.254 |
| 8 | 57.053 | 63.184 | 10.74% | 67.287 |

C4 exceeds the pinned external HTTP control by **3.57%, not the required 5%**.
C1/C2/C8 still trail. No complete external acceptance or universal speed claim.
The goal remains active, with declared native/GGUF encoding/arithmetic differences.

Ordinary single-request native post-first improves **25.557 -> 26.295 TPS**.
C2/C4/C8 native post-first cohort rates improve **42.366/66.366/85.230 ->
44.470/71.762/94.205 TPS**. These exclude first outputs and HTTP admission;
concurrent rates divide all post-first outputs by the common cohort interval,
never summed client durations. They are not competitor HTTP throughput.

MTP1 C1 native post-first improves **34.430 -> 35.691 TPS**; HTTP
**28.096 -> 29.381 TPS (+4.57%)**. All three repeats keep **14/16 (87.5%)**
accepted proposals, 16 verifications, two replayed inputs and 32 emitted outputs;
no forced proposals. Best successfully tested external MTP2 remains 30.173 HTTP
and 49.803 reported post-first TPS, in its separately labelled native timing
scope. Native MTP2 remains unimplemented; do not handicap the competitor by
comparing only its depth-one result.

## Implementation and correctness

The original GDN operator launches 256 threads/value row even at DK=64/128.
The new SM70 compact operator uses DK threads but preserves the original **eight
logical warp partials and their reduction order**. Missing warps get correctly
synchronized virtual inactive-lane contributions, including their normalized
zero/state/decay/FMA behavior. No uninitialized shared partials, changed FMA,
FP16 state conversion, altered recurrence or merged request state. Shared memory
stays 32 bytes; no new persistent buffer, allocation or projection workspace.

Only the explicit residual target profile selects it in scalar/batch native
forwards. Default FP16 targets, the legacy public primitive, PP, SM89 and DK256
retain their old paths. MTP changes the target only; draft remains FP16. Public
ABI 16, nine native build sources, no new precision/CLI switch or Python forward.
Use the existing `q8-residual` option with the rebuilt library.

The explicit standalone research operator passes **125** initial tests, followed
by matched timing and separate profiling before integration. The integrated
snapshot adds signed-zero/dynamic-graph tests: **127** operator tests pass.
Packed/unpacked, normalization on/off, DK64/128/256, T1/2/4/8/23/32, nonzero
initial states, guards, in-place chunks, graph/no-fallback, aliases and raw
geometry are covered. Outputs and FP32 state bytes match the pinned old library
and candidate legacy operator, with an independent FP64 matrix recurrence.
Invalid/nonfinite cases compare classifications and finite-value bytes, not
unspecified NaN payloads. SM70 BF16 rejection remains unchanged.

Fresh trained residual checks match complete pinned control signatures:
**98 full-vocabulary vectors**, four original greedy trajectories/final private
states, eight heterogeneous batch slots and selected reset/continuation logits,
positions and caches. Fresh four-case MTP checks preserve ordinary-target output
and committed state, including rejection/replay/EOS. No numerical Torch/Python
fallback in native generation.

Control records, unchanged trained FP16 and independent FP16 draft references
are explicitly reused, not freshly executed. The qualifier now retains inherited
unexecuted-profile provenance and refuses selected profiles without complete
prior signatures. Prior independent target thresholds are unchanged (top-1,
relative L2<=.02, centered cosine>=.999, KL<=.002). No broad original-BF16 corpus
quality claim, weakened tolerance or fresh 27-GB payload hash scan.

Final V100 suite: **1644 passed / 440 skipped**; local CPU suite:
**766 passed / 1318 skipped**. Skips are not GPU passes. All 26 earlier immutable
capsules remain unchanged, with the existing legacy unavailable-path exception
preserved rather than repaired. No commit/push or local Rust/CUDA compilation.

Retained library:
`/home/data/wangyue/artifacts/venturi-gdn-compact-model-v2/build/0849ea2437e65c8a/libventuri_blocks.so`,
SHA256 `e8090c0531b3b794f34ca267baf1b61fcb84ba6d58c1e4550795ad330ae6965a`.
Control SHA256:
`698452e2b2a39f3a5bb4d08d58e37a3330857f6444cc462f96147f53a7ab5e3d`.

## Primitive performance is not serving performance

At Qwen's Hqk=16/Hv=48/DK=128/DV=128, compact GDN improves median warmed
primitive time **1.81–1.87x** at T1/2/4/8/23/32. Each arm uses the same immutable
FP32 input state and distinct output state, three repeats and 32 graph-contained
calls/wave with reversed alternate order. Candidate legacy matches the pinned
256-thread control. DK64 diagnostic gains are 3.03–3.33x; DK256 fallback is flat.
Every output/state byte and independent FP64 gate passes before timing.

This is not an engine gain of 1.87x. The model still spends time on projections,
transfers, other kernels and CPU scheduling. Short event windows also cannot
supply reliable per-wave NVML utilization; broad raw samples are retained.

## Fixed work, CPU, latency and memory

Same official-original-derived Q8 target and explicit two-pass residual profile;
FP16 activation/KV boundary, resident host TP on physical GPUs 1,0. Eight slots,
capacity 256/context per slot and 32 projection rows. Same formatted 23 prompt
IDs and 32 fixed greedy outputs; EOS ignored only for the fixed-work speed case.
Correctness cases exercise EOS independently. HTTP includes the 25-ms admission
window; loading, warmup/reset and profiling are outside timed waves.

| Requests | Native TTFT seconds, old -> new | Server CPU core-s/cohort | Mean client completion seconds |
|---:|---:|---:|---:|
| 1 | .192532 -> .174365 | 1.420 -> 1.367 | 1.44545 -> 1.39279 |
| 2 | .386883 -> .350572 | 1.870 -> 1.763 | 1.89123 -> 1.78577 |
| 4 | .760259 -> .684200 | 2.650 -> 2.433 | 2.67380 -> 2.45712 |
| 8 | 1.542422 -> 1.383139 | 4.490 -> 4.050 | 4.48303 -> 4.04777 |

Native batch TTFT includes common-cohort prefill, not streamed client TTFT.
MTP1 native TTFT .203297 -> .185166 s, CPU 1.140 -> 1.090 core-s, client
completion 1.13814 -> 1.08848 s. Trial distributions, min/max latencies, all
counter denominators and telemetry windows are retained in raw/derived JSON.

C8 primary/peer timed-window NVML busy is **81.19/49.34 -> 79.09/54.56%**;
these percentages are not occupancy or bandwidth. Device-used peaks remain
**20235/17227 MiB**, including unrelated allocations and CUDA context; MTP1
primary remains 22435 MiB. No VRAM saving is claimed. Projection scratch stays
1,253,376 bytes/device. Power, temperature, clocks, controller busy and sample
counts are retained for each physical GPU.

## Separate timeline and compiled-resource evidence

Fresh candidates versus explicitly reused, hash-pinned unchanged control traces,
all **23 prompt / 8 output** diagnostics, not the 32-output HTTP speed cells:

| Trace | Old host span ms | New host span ms | Old GDN ms | Compact GDN ms |
|---|---:|---:|---:|---:|
| C1 ordinary | 496.42 | 468.30 | 56.35 | 30.06 |
| C8 cohort | 2262.15 | 2034.38 | 492.02 | 263.81 |
| C1 MTP1 | 416.32 | 390.85 | 53.50 | 28.42 |

GDN improves approximately **1.87x** in the model traces, consistent with the
primitive study. Projection sums stay approximately unchanged. Kernel launch
counts remain 13504/33562/8621. C1/C8 inter-activity gaps remain about
81 -> 80 / 174 -> 175 ms; this optimization does not remove CPU/transport gaps.
Gaps concern this process's recorded activities, not proof of whole-GPU idleness.

C8 still copies approximately **1.990 GB in each host direction**, with the same
8557 H2D / 6680 D2H count. No transport or paging benefit is claimed. Dot/kernel
sums overlap across devices and must not be added to derive elapsed time saved.
Only unprofiled repeats support the serving improvement above.

Exact compiled/trace joins: old GDN **256 threads, 42 registers, 32 shared bytes**;
compact **128 threads, 44 registers, 32 shared bytes**. Both have theoretical
**62.5%** residency on the tested V100, with zero reported local bytes. Thus
this is not evidence of higher achieved occupancy: the unchanged theoretical
ceiling hides the old inactive-state lanes. Actual SM/DRAM/cache/stall counters
remain unavailable under unchanged `RmProfilingAdminOnly: 1`. No policy change
or repeated denied counter campaign; NVML busy is not a substitute.

Raw SQLite/Nsight binaries remain remotely retained with size/SHA256 records;
local attribution/resource/reuse metadata is frozen. All owned workers, including
the failed metadata-only v4 launcher, are terminal; the benchmark port is closed
and original unrelated physical-GPU allocations **3651/3189 MiB** are restored.
No unrelated process/service was stopped.

## Failures, provenance and remaining work

The first final-profile launcher (v4) failed before any GPU profiling because it
expected an optional old API-audit JSON on the server; that audit had been saved
locally only. The failed directory/exit/traceback are preserved. The successor
v5 removes only that unused reference, reuses identical candidate source/library,
and runs separate profiles without repeating successful qualification or timing.
A transient SSH observation failure was recovered on the same verified host;
the already-running study completed and was inspected, not restarted.

`design.md` explains kernel/ownership and evidence scopes. `metrics.json` and
`summarize.py` retain/recompute HTTP, native timing, CPU and timed-window telemetry.
`v100/study-v1`, `model-v2`, `campaign-v3`, failed `final-v4` and successful
`final-v5` preserve build/test/command/profiling provenance. Executed source deltas,
terminal audit and source/evidence manifests pin exact variants and controls.
External non-spec and successful MTP depth controls are explicitly reused from
prior immutable capsules; failed configurations are not zero-speed controls.

The full external goal, deeper/batched MTP, paging/continuous batching, sampling,
whole-model graphs and broad/long-context quality remain unfinished. See
`next-bottlenecks.md` for bounded follow-up hypotheses, not implemented gains.
