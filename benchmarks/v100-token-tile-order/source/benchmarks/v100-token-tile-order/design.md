# Adjacent token-tile CTA ordering

Previous turn: verified progress. Native target-batched MTP1 plus a same-library
SDK optimization reached 45.40 TPS at C2 and 55.22 at C4. Best C8 ordinary remains
65.43 vs pinned external 67.29; single post-first 42.77 vs 49.80. Goal remains active.
The preceding capsule is frozen and is not modified by this experiment.

## Hypothesis and bounded change

SM70 wide residual projections process 8 token rows per CTA, 16 output rows per
block. With multiple token tiles, the previous 2D grid emits a whole row plane
before moving to the next token tile. Adjacent token tiles of the same weight
rows may reuse those weights through L2 before eviction. Hardware counters are
restricted, so even a measured gain is not proof of L2 hit rate/DRAM traffic.

Flatten `row_block * token_tiles + token_tile`, then recover row-block and token
origin within each CTA. Apply to plain K-tiled dots and complete-row fused dots,
including descriptor boundary mapping. Preserve K/FMA/reduction/scale order,
quantizer, masks, pitched outputs, all barriers and both host-TP ownership fences.
This is not the rejected T3 body/row schedule, arithmetic change or tile-size sweep.

Only token_tile8 with >8 tokens and a flattened grid <=INT32_MAX uses a separate
ordered kernel specialization. One-tile decode and overflow fallback keep the
old specialization; do not add the division/branch/register path to T8 decode.
Tile4, T1/T2/T3 and the original FP16 profile are not changed. The host-only
geometry diagnostic checks both the bijection and grid-limit fallback without
huge allocations or CUDA kernel launches. Public ABI16 sizes stay unchanged.

## Gates and measurements

New checks exercise 8/9/15/16/17/23/32/46/92/184 tokens, both A8 passes, tiny/large
row and K tails, fused two/three-descriptor boundaries and pitched sentinels,
quantizer bytes, graph replay/nonfinite inputs and no Torch numerical fallback.
Every output bit must match the unchanged original DP4A reference. Geometry tests
cover 262140 tokens and the INT32_MAX grid fallback. Complete GPU fixtures follow.

Operator study: eight distinct checkpoint shapes through first TP row shards,
plus actual fused MLP/GDN/QKV groups; T8 is a no-layout-change control, then
16/23/32/46/92/184 tokens. Two libraries, same Q8/F16-scale weights and two A8
residual passes/F32 scales; exact full outputs and sampled rows/tokens of the
independent represented-weight FP64 reference (<.001 relative L2). Three
alternating unprofiled repeats, three graph nodes × three replays =9 calls/wave.
GPU events, host wall/CPU and sampled GPU telemetry are retained. Operator graphs
are not whole-model graphs or serving TPS; hot repeated weights are not engine
cache behavior.

Proceed to fresh trained/state qualification and C1/C8 engine screens only if
the median control/candidate event-time ratio over fused mult-tile cases is
>=1.01, or a checkpoint head mult-tile case exceeds 1.02, and the unchanged fused
T8 controls do not regress >2%. Preserve all tested shapes, repeats and negatives.
Qualification reuses the pinned unchanged independent teacher reference only
when the new target logits/trajectories/private states remain byte-identical;
then qualify trained MTP1 cohorts and the stronger scalar MTP2 protocol.

Fresh engine screens retain scalar MTP2-prefix C1/C8, ordinary256 C1/C8 and
MTP1-cohort C1/C8 (plus C2 for the known geometry). Same case/capacity/prompt IDs,
32 greedy outputs/FP16 KV/profiles/host TP/admission and HTTP/native timing scopes.
A path gets a full three-repeat 1/2/4/8 matrix only for C1 HTTP>1% or C8>2%
(C2 MTP cohort >2% also permits its matrix). A screen or microbenchmark alone
cannot replace a stronger qualified result. No automatic universal router.

Profiles remain separate, use full 32 outputs and preserve the exact SO/snapshot.
Unchanged `RmProfilingAdminOnly: 1` means no achieved occupancy/DRAM/cache/stall
claims; static resources, NVML busy and process gaps are not substitutes.
No local CUDA/Rust builds, no unrelated GPU/process changes, no commit/push.

## Retained fixture setup failure

The first 100 new byte/reference/pitch cases pass. The next graph fixture fails
before execution because it reads `vocab_size` on the public `ResidentLinear`
wrapper. Its output rows are already declared by the source plane. Correct that
fixture construction, keep every numerical/graph/nonfinite threshold unchanged,
and reuse the exact native SO by all nine dependency hashes. Rerun all 104 new
checks and the complete suite before timing; retain the failed test and stopped
qualification/campaign preflights. No CUDA rebuild or numerical change is needed.
