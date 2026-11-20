# Token-tile ordering: correct, but not promoted

2026-10-20, two Tesla V100-PCIE-32GB GPUs. **Negative/limited-gain engine study;
full speed goal remains active. All nine qualified native source files are
restored.** Do not replace previous three-repeat milestones with these small
one-repeat screen points. The candidate SO/snapshots remain preserved for research.

## Change and fixed work

The hypothesis is weight reuse between adjacent token CTAs. Wide residual
projections process eight token rows per CTA; the old 2D grid completes a row
plane before the next token tile. The candidate flattens
`row_block * token_tiles + token_tile`, recovering row block and token origin in
the kernel. It changes **ordering only**, not quantization, tile size, lane
K/FMA/reduction order, scales, masks, workspace, private state or TP fences.
A separate ordered specialization is selected only for tile8, >8 tokens and a
flattened grid within INT32_MAX. Single-tile decode and fallback retain the old
specialization; tile4/T1/T2/T3 and the original FP16 profile remain unchanged.
This is distinct from the rejected T3 body/row schedule study.

Same official source revision `fc05daec18b0a78c049392ed2e771dde82bdf654`, original-
derived Q8 target/draft, resident FP16 activations/KV and host TP on physical GPUs
1,0. Target projections keep two A8 residual passes/F32 FMA; draft keeps its prior
FP16 profile. Native group32 Q8 and external Q8_0/Q8_1/scales/small parameters are
not identical precision; previous encoding/quality audits remain applicable.

Engine case: 23 formatted prompt IDs, 32 fixed greedy outputs/request, ignore EOS,
eight prepared slots of capacity 256, no cache reuse, one warmup per cell. MTP rows 32;
stronger ordinary batch rows 256. HTTP throughput includes reset, admission,
queueing/reply and client archival/validation/future collection. Native same-call
TTFT/generation excludes those. Never sum concurrent request durations. No loading,
allocation, warmup, profiling or client-side scope change is used to claim a win.

## Correctness and bounded operator study

All 104 new checks pass (102 GPU +2 host geometry diagnostics); full candidate
V100 suite **2,096 passed /448 skipped**. Local and restored-source local suites
both **785 passed /1,759 skipped**; the added explicit research tests skip without
the candidate library. Tails, pitched fused descriptor boundaries, packed inputs,
nonfinite/zero graph replay, no Torch numerical fallback and INT32 grid-limit
fallback are checked without massive allocations.

The first 100 numerical/pitch tests already passed in the initial attempt, but
its graph fixture read a missing public wrapper attribute before execution.
The corrected fixture takes the source plane's declared row count. Same native
SO/nine dependencies are reused; all 104 checks/full suite are rerun before timing.
The failed test and two stopped dependent preflights are retained, not passing
runs or failed numerical comparisons.

The operator study covers **77 cells**: eight distinct actual checkpoint matrix
shapes through first TP row shards plus fused MLP/GDN/QKV groups, at 8/16/23/32/46/
92/184 tokens. All full output bytes match control; sampled rows/first-middle-last
tokens meet represented-weight FP64 relative-L2<.001. Three alternating unprofiled
repeats use three graph nodes × three replays =9 calls/wave. GPU event, wall/CPU,
all negatives and NVML telemetry are retained. These are operator graphs, not
whole-model graphs or engine TPS; repeated hot weights are not engine cache data.

| Fused group | Tokens | Old event ms/call | Adjacent event ms/call | Rate ratio |
|---|---:|---:|---:|---:|
| MLP | 8 | 0.226468 | 0.227417 | 0.9958× |
| MLP | 16 | 0.438980 | 0.429493 | 1.0221× |
| MLP | 23 | 0.625344 | 0.606635 | 1.0308× |
| MLP | 184 | 4.793187 | 4.592992 | 1.0436× |
| GDN QKV/z | 16 | 0.213294 | 0.208935 | 1.0209× |
| GDN QKV/z | 184 | 2.250347 | 2.152405 | 1.0455× |
| Attention Q/K/V | 16 | 0.193877 | 0.189828 | 1.0213× |
| Attention Q/K/V | 184 | 1.957614 | 1.867691 | 1.0481× |

Median fused multi-tile rate ratio is 1.03712× (~3.6% event-time reduction);
minimum unchanged fused T8 ratio is 0.99583×. The best head cell is 1.06241×.
This passes the predeclared operator gate for **measurement**, not model adoption.

Fresh trained qualification follows: 98 teacher-logit byte signatures, four target
trajectories, eight mixed/reset private states remain exact to pinned controls.
Four trained MTP1 cohorts/scalar references, mixed-eight/ragged/EOS/reversed slots,
corrected draft KV and untouched selected-reset slots pass. Stronger scalar MTP2
prefix also passes four output/target/draft-state cases and three independent
recurrence steps. Unchanged target/teacher and draft-block reference work is
explicitly reused; no new original-BF16/corpus-quality claim is made.

## End-to-end screens: promotion failed

Each table cell below has **one timed repeat**, not repeatable-performance
acceptance. Fresh controls use the stronger qualified scalar MTP2-prefix and
ordinary256 library, or the qualified MTP1-cohort library with the optimized SDK.
All output IDs match the same pinned expected sequence.

| Path / requests | Control HTTP TPS | Candidate HTTP TPS | Ratio |
|---|---:|---:|---:|
| Scalar MTP2 C1 | 33.9550 | 33.9499 | 0.99985× |
| Scalar MTP2 C8 | 34.0153 | 34.1140 | 1.00290× |
| Ordinary256 C1 | 22.9665 | 22.9802 | 1.00059× |
| Ordinary256 C8 | 65.3513 | 65.7830 | 1.00661× |
| MTP1 cohort C1 | 31.4789 | 31.4515 | 0.99913× |
| MTP1 cohort C2 | 45.4865 | 45.6685 | 1.00400× |
| MTP1 cohort C8 | 60.0058 | 60.6436 | 1.01063× |

Predeclared rule: per-path C1 HTTP>1% or C8>2%, plus known MTP-cohort C2>2%,
triggers a complete three-repeat 1/2/4/8 matrix. **No path passes.** No expensive
engine matrix, new external run, new qualified TPS milestone or universal router
is launched/claimed. All nine source dependencies are restored to the qualified
`f07bf75a...` default; tested candidate headers remain in executed-source deltas.

Single MTP2 native post-first screen is 42.7265→42.7804 TPS (+0.13%), not a new
qualified result. The existing three-repeat 42.77–42.83 remains below external
reported 49.80 before the required 5% margin. Qualified best native HTTP 1/2/4/8
remains 33.81/45.40/55.22/65.43, using separate scalar-MTP2/cohort/ordinary paths;
external best-tested controls are 30.17/37.91/50.25/67.29. Reused reports are pinned,
including failed external parallel MTP configurations; no exhaustive best-config
claim or conversion of failures into zero-TPS controls.

### CPU and latency (one-repeat screens)

| Path | Cohort wall s old→new | Server CPU core-s old→new | Mean client completion s old→new | Mean native TTFT s old→new |
|---|---:|---:|---:|---:|
| MTP2 C1 | 0.94242→0.94256 | 0.94→0.95 | 0.94158→0.94190 | 0.18474→0.18261 |
| Ordinary256 C8 | 3.91729→3.89158 | 3.91→3.90 | 3.91356→3.88751 | 1.24737→1.22015 |
| MTP1 cohort C2 | 1.40701→1.40140 | 1.38→1.39 | 1.40546→1.39994 | 0.27711→0.27390 |
| MTP1 cohort C8 | 4.26625→4.22139 | 4.27→4.22 | 4.26095→4.21623 | 0.83177→0.82181 |

Native TTFT is not streaming client TTFT. Cohort MTP prefill remains serial;
its per-slot timestamps are staggered. MTP1 story acceptance remains 14/16 (87.5%)
with zero target replays. MTP2 prefix's exact counters/acceptance are preserved in
raw/derived reports. Shared verification/cohort times count once, not per client.

## GPU data and limits

All unprofiled arms sample NVML every 100 ms, filtered to timed waves. At C8,
ordinary control/candidate use 21,957/17,297 MiB on physical GPUs 1/0; MTP1 cohort
control/candidate use 22,459/17,227 MiB. These include preexisting allocations and
are not isolated process allocation or KV-compression measurements. Workspace
and cache accounting are unchanged by this scheduling-only candidate.

Ordinary C8 mean GPU busy is 79.66/55.79%→79.92/55.41%; power 153.16/121.94 W→
149.83/117.46 W, temperatures 42.16/40.08°C→42.11/39.95°C. MTP1 C8 busy is
81.20/55.49%→81.53/54.58%; power 168.24/125.29 W→165.14/119.88 W. Both GPUs hold
1380 MHz SM/877 MHz memory clocks in these windows. Full per-cell sampled
min/mean/max, memory-controller busy, request latency and CPU values are in
`metrics.json`. These one-wave power differences are not energy-efficiency wins.

Separate full 32-output Nsight captures explicitly reuse the stronger frozen
control profiles; they are not fresh unprofiled control trials. Event tracing is
disabled. One native range counts one logical C1 call or an eight-request cohort;
metadata—not the attribution range count—defines logical request count.

| Full-call profile | Summed kernel ms old→candidate | Process activity gaps ms old→candidate | Two-device kernel overlap ms old→candidate |
|---|---:|---:|---:|
| MTP2 C1 | 968.45→963.65 | 160.39→162.82 | 298.32→278.82 |
| Ordinary256 C8 | 4,197.58→4,138.17 | 423.20→426.57 | 1,275.01→1,248.28 |
| MTP1 cohort C8 | 4,693.76→4,600.74 | 362.72→372.36 | 1,428.85→1,380.72 |

Launch counts stay 27,026/82,688/69,592 respectively. Summed kernel time shrinks
only ~0.5/1.4/2.0%; changed overlap and persistent host/transfer gaps illustrate why
operator event gains cannot be substituted for engine throughput. Instrumented
variation/reused control profiles do not establish a causal cache or wall-time
speedup. No whole-model graph is introduced.

`grid-layout-proof.json` verifies actual native launches, not merely source intent:
fused MLP prefill 184 changes `(1088,23)`→`(25024,1)`; MTP verifier 16 changes
`(1088,2)`→`(2176,1)`. Single-tile decode stays `(1088,1)` with the unordered
specialization. Fused T8 ordered/unordered both use 56 registers/thread,
512 threads, 18,432 dynamic shared bytes and no local memory; driver resource
queries give two resident blocks/SM, **50% theoretical**, unchanged. The single
T3/T4 body remains 53 registers, 256 threads, 9,216 shared bytes, four blocks/SM,
50% theoretical and no spills. This is not achieved occupancy progress.

Achieved occupancy/compute/DRAM/cache/stall counters remain unavailable under
unchanged `RmProfilingAdminOnly: 1`. NVML busy, kernel shares, process activity gaps
and theoretical residency are not replacements; no driver privilege change.

## Disposition and evidence

- Candidate SHA256: `ecec6e10052e47eb75fc783f4bfc59e80f8829279037975e784c4294b58a1461`.
- Candidate was built once in the first attempt; fixture recovery reuses that
  exact SO and all nine dependencies. No numerical threshold or CUDA code repair.
- Target/teacher/private-state, MTP1 and stronger MTP2 qualification reports,
  all 77 operator cells/repeats, full screen failures/non-promotion and raw telemetry
  remain preserved. There is no new default performance result.
- Both executed snapshots are reconstructible from restored source plus deltas;
  remote binaries/SQLite/Nsight files remain hash-pinned. The geometry diagnostic
  is research-only in that preserved candidate, not a new production API.
- All seven owned workers are terminal, including failed setup/preflights;
  port 18953 is closed and prior GPU footprint 3,651/3,189 MiB is restored. No
  unrelated services, local Rust/CUDA builds, commit/push or old-capsule changes.
- Source/evidence manifests verify all 31 prior capsules. `summarize.py` recomputes
  derived metrics without editing any old report.

**Decision:** reject default promotion, preserve qualified native milestones and
keep the full external objective active. Next test a distinct single-verifier
shared-word/register-layout hypothesis and measure native host-source/launch costs;
parallel draft/head batching remains a separate ownership task. See
`next-bottlenecks.md` for specific gates and already-implemented TP event fences.
