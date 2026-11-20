# T3 projection schedules: correct, but not promoted

2026-10-20, two V100-PCIE-32GB GPUs. **Negative/limited-gain experiment; goal active.**
Model dispatch is restored. Keep the qualified three-repeat 42.83 single-decode TPS
and 65.41 ordinary-batch C8 TPS milestones; do not replace them with a one-repeat
prototype point. No precision, workspace, KV, draft protocol or transport change.

## What was tested

The prior full 32-output MTP2-prefix profile attributed about 85.5% of summed CUDA
kernel time to Q8 projections. T3 verification used a padded token4/row8 K tile.
This study tests token3 full activation sharing and token3 K tiles at row 4/8/16:
**six research schedules**, not an unbounded tuning search.

Eight distinct original-Q8 checkpoint matrix shapes are tested through one
representative binding per shape, first TP row shards, plus the actual fused MLP,
GDN and Q/K/V groups. Inputs/outputs are FP16, weight scales FP16, activation scales
FP32 and both original residual A8 passes remain. Each output bit must equal the
pinned control; sampled rows pass independent represented-weight FP64 relative-
L2<.001. No split-K reordering or weaker arithmetic is introduced.

Three alternating microbenchmark repeats use three-node graphs, three replays/
wave (nine calls); event, wall and CPU times are retained. These operator graphs
are not whole-model graphs or serving TPS. Control and candidate-unchanged T4 paths
are both retained to detect unrelated build/measurement effects.

| Actual fused group, K=5120 | Old T4/r8 ms | Full T3/r16 ms | K-tiled T3/r8 ms |
|---|---:|---:|---:|
| MLP gate/up | 0.14220 | 0.15901 | 0.13602 |
| GDN QKV/z | 0.07874 | 0.08494 | 0.07791 |
| Attention Q/K/V | 0.06673 | 0.07429 | 0.06454 |

Full sharing **regresses about 8–12%** on these large fused groups. T3 K-tiled/r8
improves about 1.1–4.5% at the operator level. The standalone 512-row V projection's
best-arm 20% gain is not an engine gain: deployed Q/K/V are fused, and that small
standalone matrix can have different cache behavior. All other shapes, strategies,
unsupported full-plane sizes and measured trial values remain in `metrics.json`.

## Full-model screen and decision

Candidate source-v2 changes only residual T3 plain/fused dispatch to token3/row8
K tiles. All qualification gates pass before timing. Screen: 23 formatted prompt
IDs, 32 fixed greedy outputs/request, ignore EOS, FP16 KV, resident host TP on
physical GPUs 1,0, eight prepared slots/capacity256. MTP2 prefix uses rows32;
ordinary cohort batch uses rows256, both unchanged from their controls.

**Each cell below has one timed repeat**, not repeatable-performance acceptance.
Complete-request TPS is all outputs / global cohort wall; concurrent request
latencies are never summed. HTTP wall includes handler reset/queueing/replies and
client archival/validation/future collection. Native same-call timers exclude reset
and HTTP. Loading, slot/prefix allocation, warmup and profiling are outside waves.

| Path / requests | Control HTTP TPS | Candidate HTTP TPS | Ratio |
|---|---:|---:|---:|
| MTP2 prefix C1 | 34.021 | 34.114 | 1.002729 |
| MTP2 prefix C8 | 34.136 | 34.524 | 1.011391 |
| Ordinary batch256 C1 | 23.076 | 23.084 | 1.000333 |
| Ordinary batch256 C8 | 65.785 | 65.797 | 1.000185 |

Single native post-first screen (31 outputs / same-call total-minus-first-token)
is 42.8406 -> 43.1897 TPS (+0.81%), **not a new qualified decode result**. It remains
below the best tested pinned external MTP2 reported 49.80 TPS, before the required
5% margin. Ordinary parallel batching remains much stronger than scalar MTP.

Predeclared internal rule: C1 HTTP>1% or C8>2% triggers a full three-repeat MTP
matrix; ordinary batch requires C8>2%. Neither passes. **No full 1/2/4/8 matrix is
launched**, no fresh external campaign is warranted, and no speed superiority is
claimed. The stronger existing native and external three-repeat controls are
explicitly reused, not replaced by the small one-wave changes above.

The model dispatch changes are reverted. All nine current native source hashes
match the full-suite-tested research source-v1: existing model paths are unchanged,
while explicit research entry points remain available for reproduction. Immutable
source-v2 deltas, compiled candidate, passed gates, screens and profiles preserve
the rejected experiment. `decision.json` and `restored-dispatch-proof.json` record
this decision. No commit/push or external service change.

## Correctness

- Both research and integrated-candidate V100 suites: **1906 passed /448 skipped**;
  selected operator suites 445 passed. Local CPU suite 776 passed /1578 skipped.
- New 134 checks include plain/fused row guards, packed workspace byte equality,
  one/two activation passes, narrow/odd/real K shapes, independent FP64 references,
  graph replay, nonfinite handling, alias rejection and malformed raw geometry.
  Oversized full-sharing combinations are tested expected rejections, not executed
  kernels or skipped successes.
- Fresh candidate residual target qualification matches all 98 teacher-step logit
  byte signatures, four generated trajectories, final private GDN/history/KV,
  eight mixed batch caches and reset/continuation signatures to the pinned control.
  Existing independent CPU/quality thresholds are reused, not relaxed/relabelled
  as a fresh original-BF16/corpus campaign. Original-FP16 model references are
  explicitly reused; the candidate changes only residual T3 dispatch.
- Four fresh trained MTP2-prefix cases match ordinary target output/state and
  repaired draft KV vs MTP1; three recursive draft reference steps pass unchanged
  represented-weight FP64 thresholds. No forced proposals or replayed target
  inputs in trained/timed prefix runs.
- Restored source-v1 has the same nine sources as its already tested binary, so
  no redundant rebuild/qualification is performed after reverting dispatch.

This is not broad original-BF16/corpus quality, other-GPU qualification, sampling,
arbitrary continuation, long-context or production continuous batching/paging.

## CPU, latency, memory and sampled telemetry

One-repeat MTP2 screen:

| Requests | CPU core-s old -> new | Mean client completion s old -> new | Native TTFT s old -> new |
|---:|---:|---:|---:|
| 1 | 0.94 ->0.94 | 0.939748 ->0.937346 | 0.184649 ->0.184669 |
| 8 | 7.97 ->7.91 | 6.936154 ->6.999561 | 1.010363 ->1.129347 |

C8 maximum client completion 7.498512 ->7.414098 s, but mean latency and mean
internal readiness worsen. This illustrates why a small aggregate-throughput
change is not a universal efficiency/latency improvement. Ordinary batch C1/C8
CPU is 1.36/3.89 core-s in both arms. Full per-cell/trial denominators and responses
are retained; native TTFT is internal readiness, not streaming client TTFT.

Primary/secondary NVML device usage is unchanged: MTP 23993/17227 MiB;
ordinary batch 21957/17297 MiB (includes existing unrelated allocations).
T3 dynamic shared storage is not additional persistent VRAM or a KV-format change.
Timed-window MTP busy means (physical 1/0): control C1 73.0/52.1%, candidate C1
73.2/50.7%; control C8 74.0/51.7%, candidate C8 75.1/52.5%. Power means W:
control C1 139.6/116.7, candidate 141.1/107.0; control C8 145.3/114.5,
candidate 146.5/114.3. SM clocks 1380 MHz; observed temperatures 36–44 C. Memory
clocks/controller busy and all raw samples are retained. No achieved occupancy,
DRAM bandwidth or energy-efficiency claim follows from these coarse samples.

## Separate full 32-output profiling

Fresh candidate C1/C8 traces exercise actual later rejections. The exact prior
full 23/32 baseline profiles, tool/case/library and raw hashes are reused explicitly;
short 8-output traces without rejections are not substituted. Profiles are **not**
unprofiled throughput repeats.

| Profile | Host cohort ms | Summed kernel ms | Gaps between this process's captured activities ms |
|---|---:|---:|---:|
| Reused control C1 | 954.381 | 968.449 | 160.388 |
| Candidate C1 | 946.916 | 959.169 | 158.696 |
| Reused control C8 | 8452.085 | 7747.506 | 1817.264 |
| Candidate C8 | 8437.893 | 7672.843 | 1819.451 |

Summed kernels can exceed wall due to two-GPU overlap. Gaps are process activity
gaps, not device-wide idle or a prediction of graph speedup. C1 H2D/D2H bytes remain
535,484,944 /535,484,976; C8 4,283,879,552 /4,283,879,808. All profiles keep seven
recovery calls/request with zero target-input replay. Workload/acceptance is not
silently reduced to manufacture a kernel gain.

Compiled launch/resource evidence:

| Kernel | Registers/thread old -> T3 | Threads | Dynamic shared bytes old -> T3 | Theoretical blocks/SM old -> T3 |
|---|---:|---:|---:|---:|
| Fused T4/r8 ->T3/r8 | 53 ->51 | 256 | 9216 ->6912 | 4 ->4 (50%) |
| Plain T4/r8 ->T3/r8 | 48 ->45 | 256 | 9216 ->6912 | 5 ->5 (62.5%) |

No local-memory spill is reported for these launched kernels. Reduced registers/
shared payload does **not** increase their theoretical block residency. Achieved
SM occupancy, compute/DRAM/cache/stall counters remain unavailable under unchanged
`RmProfilingAdminOnly: 1`; no driver privilege/settings change or repeated blocked
counter campaign. NVML busy/theoretical residency/kernel share are not substitutes.

## Next and reproducibility

Prioritize native batched one-proposal target verification, per-request GDN/history
prefix recovery and corrected draft KV; retain scalar MTP2 as the stronger single
option. `next-bottlenecks.md` identifies concrete ownership, hidden/head capture,
lock order, tail/EOS and failure/reset requirements. That feature is **not yet
implemented** here; fixed cohorts must not be mislabeled continuous batching.

`design.md`, `summarize.py`, raw reports, launch commands, executed-source deltas,
compiled metadata and `metrics.json` reproduce this bounded experiment. Remote
libraries and Nsight raw traces stay retained by exact path/SHA256. Four owned
workers are terminal; port 18953 is closed and GPU footprint restored to original
3651/3189 MiB. All 29 earlier capsules remain unchanged, with the legacy decoder
manifest's previously recorded missing paths still explicit.

Pinned production performance library:
`9a4d8655c9f781746ab2240c86e51c79d02a3653dc09c7262a51130d661c0894`.
Research/restored-source library:
`853e2949bf83cde59ad9db3346b8d7139b153400eb3c019778966528f16c6af2`.
Rejected integrated candidate:
`f6ef262cec6e14676496dcac9a4cb9e2aa0cbd7f70f71d4a4338c560a0906894`.

Official source revision `fc05daec18b0a78c049392ed2e771dde82bdf654` and original-
derived target/draft/GGUF artifacts are unchanged. Native and external Q8 encoding,
rounding, small parameters and activation arithmetic still differ; this study
changes none of them and does not claim identical precision from Q8 labels.
The overall external-performance goal remains **active and incomplete**.
