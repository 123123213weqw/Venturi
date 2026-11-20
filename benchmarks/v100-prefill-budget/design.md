# Explicit batched-prefill projection budget

## What this is—and is not

This is tuning of an **existing native scheduling/allocation parameter**, not a
new GPU kernel or model architecture. `NativeQwenModel.from_q8(max_tokens=...)`
already supports 1..1024 rows and defaults to 128. The fixed-work HTTP adapter
had explicitly capped it at 32. The new adapter/profile/qualification option
`--projection-rows` exposes the tested 32/128/256 budgets without changing the
public native API, kernel binary, arithmetic or per-slot context capacity.
The adapter default stays 32 for historical reproducibility; recommended larger
budgets must be explicit. No silent change to default FP16/SM89/PP/MTP behavior.

Batch prefill divides the model row budget among active requests. Eight equal
23-token prompts need six group forwards at rows32, two at rows128, and one at
rows256. The same 184 logical prompt rows, output IDs, 32 generated tokens per
request, FP16 KV, positions and greedy policy are evaluated. This reduces some
launch/fence/state traffic, not the logical workload. Native projections still
use the previously qualified group/tile kernels. A bigger budget consumes more
memory; it is not free VRAM or a larger KV/context promise.

The same exact `e8090c0531b3b794f34ca267baf1b61fcb84ba6d58c1e4550795ad330ae6965a`
library is used for every arm. All nine compiled-source hashes are checked.
There is **no new CUDA build**, local or remote, in this tuning study. New tools
record the allocation budget separately from the 256-token slot context.

## Correctness before serving comparisons

35 added tests include full hybrid-model budgets128/256 versus32, original FP16
and residual profiles, single/host-TP, heterogeneous eight-slot prompts and
inactive/reset/continuation behavior. Logit and all private cache bytes must
match. Unequal generation limits/EOS, native no-Torch generation, position/input-
row accounting, fewer prefill calls and increased request workspace accounting
are checked. Large fused projections test T56/127/128/184/255/256 with short/large
mixed matrices, K tails, pitched sentinels, exact old output/workspace bytes and
sampled independent FP64 represented-weight references. CPU tests check parser
rejection before runtime setup. The initial CPU test import-path issue was fixed
before GPU qualification; no numerical counterexample was suppressed.

Each candidate budget executes fresh trained residual checks against complete
hash-pinned rows32 control signatures: 98 full-vocabulary vectors, four greedy
trajectories/final private states, eight heterogeneous batch slots and selected
reset/continuation state. Context, profiles, checkpoint and tolerances remain
fixed. `fixed_work` explicitly records candidate versus control projection rows.
Original independent target gates and unexecuted FP16 reference provenance are
reused, not weakened or freshly re-executed. No broad original-BF16 corpus-quality
claim or fresh large payload SHA scan.

The selected tuning applies to ordinary cohort batching only. MTP serving stays
at rows32 with the same code/library/configuration, so its successful prior
qualification/performance is explicitly reused rather than repeated. Source
inspection also shows the old MTP qualification helper used max_tokens=128;
that is provenance of existing tests, not a new 256-row MTP claim.

## Performance and diagnostics

Fresh rows32,128,256 C1/C8 screens precede conditional three-repeat whole C1/2/4/8
HTTP comparisons against rows32. All arms use resident host TP on physical GPUs
1,0, eight slots/capacity256, the same original-derived Q8 checkpoint and explicit
two-pass residual profile, FP16 KV, 23 formatted prompt IDs and 32 greedy outputs.
The fixed-work speed case ignores EOS; separate correctness tests cover EOS.
HTTP includes the same 25-ms admission window. Loading, warmup/reset and profiling
are excluded. CPU core-seconds, native TTFT/post-first and client completion are
separate; cohort TPS uses emitted outputs/global cohort wall, never summed
concurrent durations.

The highest eligible C8 screen ratio is selected only if C1 has no >1% regression
and C8 improves >3%. This internal tuning threshold does not replace the >=5%
external objective. Larger allocations, flat/decode regressions and all screen
results remain recorded. Previous strongest external controls are hash-pinned
and reused, not selected weaker or relabelled as fresh measurements.

Separate fresh selected-budget C1/C8 23-prompt/8-output timelines are compared
with identical unchanged rows32 traces explicitly reused from the compact-GDN
capsule. Resource queries report compiled ceilings, not achieved occupancy.
GPU busy/controller percentages are not bandwidth or occupancy. Hardware counters
remain unavailable under unchanged restricted driver policy; no repeated denied
campaign or privilege modification. Raw Nsight/SQLite binaries stay remotely
retained with size/SHA256 metadata; local attribution/resource/source manifests
preserve the audit trail.
