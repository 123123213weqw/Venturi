# Next evidence-driven candidates (not implemented or claimed as speedups)

## Single-request projection: shared activation layout/live registers

The token-order experiment cannot accelerate the T3 verifier's main body: it
changes only T8 multi-tile scheduling. The previous T3 schedule study reduced
registers modestly (fused53→51) without crossing the four-block static residency
limit; full sharing was slower and a small tile-body gain did not pass engine
promotion. Do not repeat that identical schedule/row sweep.

A distinct bounded hypothesis is **SoA shared activation words + short-lived
scalar load pairs**. Current AoS int4 staging loads two int4 activation vectors
(eight words) before their integer dots. For a single-request T3 verifier, keeping
fewer activation words live might cross a register allocation boundary without
spilling or changing any FMA/reduction/quantization order. A naive scalar load
from the existing AoS layout risks shared-bank conflicts; it is not enough to
replace vector loads alone. Consider eight shared word planes (same36 bytes per
scale group including FP32 scale), then consume low/high word pairs into the
same two four-step integer accumulators. Across a warp, each scalar plane read
should be contiguous. Staging/load instruction cost and lost ILP can negate any
register benefit; inspect the actual generated resources and measure it.

First compare old AoS versus one SoA schedule on actual T3 plain/fused shapes,
checking exact outputs/packed inputs and no spills. Theoretical residency is only
a diagnostic, not the objective or achieved occupancy. Do not force a high
launch-bound residency and call it progress; previous six/eight-block variants
spilled and regressed. Only qualify/promote with actual model performance and
unchanged accuracy. T2 full sharing may later be another explicitly measured
case, not an unbounded expansion of the initial experiment.

## Native host gaps and source checks

Host TP **already uses cross-device CUDA events**, with a producer/scratch reuse
fence and a consumer/output gather fence. There is no per-projection host stream
synchronize to remove. Do not rediscover event transport or drop either ownership
fence. Process activity gaps are not device-wide GPU idle or guaranteed graph
speedup.

`Model::weight` still calls source identity verification per projection. Before
changing this, measure syscall/API costs separately (instrumentation is not an
unprofiled TPS result). A resident-scoped check requires start/end mutation
detection, exception/poison cleanup and explicit guard-equivalence tests; never
silently remove source checks. A future CUDA graph also needs dynamic position/
KV bounds, stable host staging and selection buffers, per-request prefix owners,
CUDA cross-device event/capture semantics and separate tail shapes. Fixed
positions or omitted reset/failure tests are not an acceptable shortcut.

## Parallel native MTP

Serial shifted prefill/draft blocks/proposal heads/catch-up remain. If measured
cost justifies it, batch the shared proposal head across independently normalized
draft outputs first, preserving every proposal/logit and corrected draft KV byte.
Expose shared counters/time once per cohort; do not hide it in overlapping
per-request statistics or allocate duplicate permanent target caches. Full draft
block batching is a separate later ownership task.

## Restored-source research test binding

`VENTURI_TILE_ORDER_LIBRARY` is an explicit historical research-library binding.
After rejection, production headers are restored and do not export the candidate's
host geometry diagnostic. Do not bind that variable blindly to a future ordinary/
SoA build and interpret a missing export as a numerical failure. Either leave
these unchanged research checks skipped/reused by hash, or explicitly point them
to the preserved `ecec6e1005...` candidate. The new experiment's own numerical and
model gates must use its actual new library, not that historical candidate.
