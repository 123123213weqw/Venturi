# V100 paired activation words: operator benefit, no default adoption

**Disposition:** all ten owned workers completed successfully; model correctness
passed. The guarded experiment improves single post-first decode, but its
three-repeat HTTP improvement misses the predeclared adoption threshold. All
nine qualified default native dependency hashes are restored. The experimental
library and all ablations remain available; no commit, push, permanent service,
precision reduction or driver-policy change was made. The full goal is unfinished.

## Controlled implementation and attribution

On SM70, token-tile4/pass2/row8 stages corresponding activation words into four
low/high `int2` planes. Integer chains, their final addition, FP32 scale/FMA and
warp reduction order remain identical. The quantizer, global workspace format,
FP16 KV, source checkpoint, target two-pass residual arithmetic, FP16 draft,
TP host bridge and both cross-device ownership fences are unchanged.

Four separately built, immutable sources separate layout from a compiler bound:

| Variant | Median fused T3 rate ratio vs original | Outcome |
|---|---:|---|
| Natural paired planes | 1.00365 | Fails operator gate |
| Paired planes, minimum five blocks | 1.03388 | Passes median; attention regression retained |
| Original AoS, minimum five blocks | 1.02349 | Passes median; attention regression retained |
| Guarded paired-five, original attention/plain paths | 1.03760 | Advances to trained/model tests |

The final guard uses a separate kernel only for two-descriptor SM70 T4/P2/R8
groups with at least one projection having >=1024 output rows. It preserves the
original three-descriptor attention, plain, T8, tiny-group and FP16 functions.
The guard is based on reusable geometry, not a model name. Three alternating
unprofiled repeats in each of33 checkpoint-shape cells preserve all arms,
including regressions. The final fused MLP T3 event time is0.13600→0.13108 ms;
fused GDN is0.07451→0.07005 ms. Unchanged attention is0.06310→0.06353 ms, within
this study's noise, rather than an adopted slower constrained attention body.
The minimum unchanged fused T8 ratio is0.99234, above the declared0.98 floor.
These are operator measurements, not model/HTTP throughput.

The selected fused specialization uses47 registers/thread versus53 originally,
with256 threads and9,216 dynamic shared bytes/CTA; driver-reported local memory
is zero. Theoretical residency rises from4 to5 CTAs/SM (50%→62.5%). Original
plain and attention functions retain48 and53 registers respectively. Static
SASS shows the additional pair-load/store instruction cost. These are compiled
resource facts, **not achieved occupancy or bandwidth**. The initial unbounded
layout reduces plain registers without improving its residency and leaves
fused registers unchanged; that failed hypothesis remains preserved.

## Correctness before speed

Each variant passed92 explicit projection checks and the full GPU fixture suite:
**2,085 passed,552 skipped**. The guarded fixtures exercise eligible two-descriptor
and preserved three-descriptor groups, token/K/row tails, pitches/sentinels,
identical packed workspaces, graph replay, zero/nonfinite input and no Python
Torch math/allocation in dispatch. Historical tile-order-specific tests remain
unbound/skipped, not incorrectly pointed at these libraries. Each operator
cell checks complete output bytes and sampled represented-weight FP64 relative
L2<.001. This is not broad BF16/corpus quality qualification.

Fresh guarded-model checks preserve98 teacher-logit byte signatures across four
trained trajectories, eight mixed private target caches and reset/continuation
states. Independent teacher/control work is reused only by pinned report/source
hashes. Fresh MTP1-cohort checks preserve target/corrected draft states and
selective-reset isolation; fresh scalar MTP2-prefix checks preserve output IDs,
committed target state and corrected draft state, including independent draft
recurrence references. All benchmark output IDs match the pinned case.
Local gates pass **786 tests,1,851 skipped** before integration and after restore.

## End-to-end: three repeats, not a one-wave win

Same two V100-PCIE-32GB devices, logical order physical1,0; original official
checkpoint revision, native group32 Q8/F16 scales, target two A8 residual passes,
FP16 KV/draft,23 formatted prompt IDs,32 fixed greedy outputs, eight slots with
capacity256, no prefix reuse. Both native sources use host TP; P2P byte validation
is still unsupported. Native and external Q8 encodings/internal arithmetic are
not identical; existing comparison disclosures remain in force.

Fresh C1/C8 screens advance scalar MTP2 only (C1 HTTP +1.30%). Ordinary256 and
MTP1-target-cohort screens miss the declared gates; their results remain one-wave
screens, not promoted three-repeat improvements. MTP1-cohort C2 is +1.44%, below
its2% gate. All stronger parallel controls are retained.

Scalar MTP2-prefix results below are **serialized inference behind concurrent
HTTP clients**, not native batched parallel scaling:

| HTTP clients | Control TPS | Guarded TPS | Change | Server CPU core-seconds, control→guard |
|---:|---:|---:|---:|---:|
| 1 | 33.855 | 34.123 | +0.79% | 0.947→0.933 |
| 2 | 33.583 | 33.626 | +0.13% | 1.937→1.923 |
| 4 | 33.934 | 34.371 | +1.29% | 3.927→3.897 |
| 8 | 33.683 | 33.962 | +0.83% | 8.050→7.967 |

TPS is32×clients divided by mean **global complete-cohort HTTP wall**, including
reset/admission/queue/reply and client archival/validation/future collection;
concurrent request durations are never summed. All cells have three unprofiled
repeats, with fresh control then candidate ordering retained. Full trial values,
client completion latency, TTFT, CPU and telemetry are in `metrics.json`.

Single native post-first decode is **42.747→43.296 TPS (+1.28%)**; native TTFT is
184.797→184.717 ms. C1 HTTP cohort wall is945.207→937.783 ms. Native TTFT is a
same-call measurement, not streaming client TTFT; concurrent native timers can
include native serialization/queue effects. Acceptance remains17/26 proposed
tokens per request (65.38%), with13 verification calls, zero target replay and
zero forced proposals. No acceptance/precision/length advantage creates the gain.

No repeated qualifying cell clears the declared C1 HTTP>1% or C8>2% threshold.
The C4 improvement cannot retroactively replace that gate. Therefore the guarded
source is **not adopted**, despite its consistent small single-decode benefit.

## GPU diagnostics and remaining gap

Separate full-call Nsight profiles show2,912 affected paired-kernel calls in the
single MTP2 case, but **zero** in either C8 ordinary or C8 MTP1-cohort traces,
confirming the guard's narrow scope. The two C8 profiles each contain one native
submitting range for eight logical requests; that is not one logical request.
The affected kernel accounts for289.38/954.87 ms of summed kernel durations in
the instrumented single call. That sum overlaps two GPUs and is not a wall-time
or critical-path share. Improving one family cannot be extrapolated to a3.8%
whole-engine win. Captured process-activity gaps are not device-wide idle time.

C1 candidate samples show primary/secondary GPU busy means74.0%/51.0%, compared
with73.6%/51.8% control; device-used VRAM is unchanged at23,993/17,227 MiB. This
includes unrelated retained baseline allocations of3,189/3,651 MiB. Candidate
sampled SM/memory clocks are1,380/877 MHz on both devices; power means are
139.9/111.8 W and temperatures40.9/39.0 C. Full count/arm distributions and sample
counts are retained;100 ms NVML sampling is not an instruction-level measurement.
`RmProfilingAdminOnly:1` remains unchanged. Achieved occupancy, DRAM throughput,
cache/bank conflicts and stall counters are unavailable, not estimated from busy
percentage or static residency. Nsight/SASS observations are separate from TPS.

The pinned external MTP2 single reference remains49.803 post-first TPS; even the
experimental43.296 is below it, and the >=5% goal requires52.293 TPS. The stronger
qualified native C8 ordinary control remains65.430 HTTP TPS versus67.287 external
non-speculative TPS; the goal floor is70.651. Qualified native C2/C4 MTP1 cohorts
remain45.402/55.215 HTTP TPS versus37.907/50.254 external non-speculative controls.
These are pinned reused references, not new external runs or exhaustive
configuration-best claims. A complete single+parallel win is not established.

**Next:** stop this small T3/register-bound series. Test the distinct T16/row8
weight-reuse hypothesis for the C8 verifier, with explicit register/shared-memory
tradeoffs and the same byte/state/C1-C8 gates; see `next-bottlenecks.md`.
