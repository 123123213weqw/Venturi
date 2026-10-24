# Subsequent targets (hypotheses, not implemented speed claims)

## Parallel MTP: verifier token tile versus emitted work

The current eight-request MTP1 verifier packs16 target rows, while wide residual
projection CTAs process8 token rows. Thus it logically reads two weight planes
per verification round. The story emits~1.875 outputs/request/round (14/16
proposals accepted), so weight planes per emitted output are about2/1.875—not
less than ordinary's one plane/output. Draft work and prefix snapshots add costs.
These are logical accesses/work from source, **not measured DRAM traffic**; cache
may change physical traffic. The earlier adjacent-CTA order study improved the
engine by only~1%, so do not repeat that same ordering hypothesis.

A distinct bounded candidate is **T16/row8** for >=16 packed rows on wide SM70
residual projections. It keeps128 output elements/CTA, like current T8/row16,
but reuses weights across16 token rows instead of8. Shared data doubles18,432→
36,864 bytes and each thread holds more sums; theoretical occupancy may fall
from50% to25%. That is a tradeoff to measure, not an automatic rejection or a
speedup. Fewer active warps can still carry the same number of independent output
accumulators; extra staging/register/issue cost can negate weight reuse.

First test exact bytes, tails/nonfinite/graph behavior and generated resources,
then actual16/23/32/46/92/184-row checkpoint plain/fused cases with unchanged T8
controls. Preserve tiny/cache-resident row guards. Only a qualified real-model
C1/C8 comparison can establish serving benefit. No changed quantization/FMA or
KV precision, no new permanent caches. The target's independent per-request
GDN/history/KV/cursor/EOS/failure semantics remain mandatory. This is not the old
naive WMMA or T3 body/row sweep.

## Draft work and single request

Shared proposal-head batching can remove repeated target-head reads across active
draft requests; full draft-block batching is a separate ownership task. Preserve
per-request normalized hidden, corrected draft KV, target prefix owners and all
selection/logit bytes. Reuse existing private caches rather than duplicate them.
Expose shared head/cohort time once, not repeated as additive per-slot timing.

For single request, attribute source checks and CUDA launch costs before graphs
or scoped guards. Host TP already has cross-device CUDA events and both scratch/
consumer fences; do not drop them or rediscover them as a new implementation.
The paired layout and bounded ablation below are finite experiments, not a basis
for endless register/block-count sweeps. Keep the full single+parallel external
speed goal intact and preserve stronger scalar MTP2 and ordinary C8 controls.

## Closed experiment and next priority

The four paired-word / five-block arms are now closed. The guarded arm gives
+3.76% median fused T3 rate and +1.28% single post-first decode, but only +0.79%
whole HTTP in its three-repeat C1 cell, below the declared adoption gate. All
qualified default native dependency hashes are restored. Whole-model traces
show zero guarded-kernel calls in the two C8 paths, so further tuning that guard
cannot directly fix the parallel gap. Do not repeat that small-bound series or
rename the C4 serialized-HTTP gain as the predeclared C1/C8 gate.

The next distinct priority is the T16/row8 packed-verifier experiment above.
Single request still needs a substantially larger improvement than this isolated
kernel family can provide; inspect draft depth / verification algorithm and
critical-path launch/host-transfer costs separately, without conflating summed
two-GPU kernel time with wall time or weakening arithmetic/source checks.
