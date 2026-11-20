# T16/row8 weight reuse for packed verification

Previous goal turn: **verified progress**. Four paired-word/compiler-bound arms
passed correctness but gave only +0.79% repeated single HTTP throughput, below
the declared adoption gate. All nine qualified native source hashes are restored;
the33 earlier capsules remain immutable. Do not repeat that small-bound series.
The full >=5% single-decode and parallel external objective remains unfulfilled.

## One new geometry, unchanged arithmetic

For SM70 residual projections with >=16 packed token rows, >=1024 output rows
and encoded row-plane size >6 MiB, use T16/row8 instead of T8/row16. Apply to both
plain projections and two/three-descriptor fused groups, not only a convenient
subcase. Fused dispatch uses the largest descriptor row-plane for its guard;
small descriptors may share its selected CTA grid, as before. Single-pass model
projection, T1/T2/T3/T4/T8 and tiny/cache-resident weight paths remain unchanged.
A separate explicit T16 research export accepts either activation-pass count,
including smaller token tails; other row sizes fail before dispatch. The old
tile4/8 API and ABI16 structures stay unchanged. Python research preparation
checks for the additional export instead of silently using a different kernel.

Each CTA still produces128 output elements for full tiles, but reuses each weight
across16 token rows instead of8. Row CTAs double while token CTAs halve at16 rows;
logical weight words per complete output tile halve, not necessarily DRAM bytes.
Shared staging doubles18,432→36,864 bytes and each lane holds16 sums. More
registers, fewer resident warps, activation staging and instruction pressure may
outweigh reuse. No forced block bound or multidimensional autotuning sweep.

The existing templated dot body keeps each output's original two integer chains,
FP32 scale/FMA sequence, K order, warp reduction and FP16 cast. Quantizer/workspace,
Q8 encoding/scales, FP16 KV, draft profile, target/draft state ownership and both
host-TP CUDA-event fences are unchanged. No graph execution or new persistent
cache is introduced in the model. Projection graphs below are microbenchmarks.

## Gates declared before GPU timing

-124 fixtures: plain/fused3-shape grids, T8 control plus16/17/23/31/32/33/46/92/184
  tokens, both passes, tails/pitches/guards/workspace bytes, graph replay with
  nonfinite/zero inputs and no Python math, plus CPU geometry/missing-export preflight. Compare
  exact original global-DP4A output bytes and sampled represented-weight FP64
  relative L2<.001. Run the full GPU suite; keep historical tile-order checks
  unbound rather than point them at a library lacking their geometry export.
-77 actual checkpoint first-TP-row-shard cells: eight unique shapes and three
  fused groups × T8/16/23/32/46/92/184, three alternating unprofiled repeats,
  three graph nodes × three replays (nine calls/wave), exact full output bytes
  and sampled FP64. Archive CPU/wall/event timing and NVML. No serving TPS claim.
- Promote to trained/model gates only if median eligible fused control/candidate
  rate >=1.05, median eligible plain rate >=.98 and minimum unchanged fused T8
  rate >=.98. Preserve all shapes and adverse results, not best-arm selection.
  Report registers/local memory and theoretical residency with actual dynamic
  shared bytes. A theoretical occupancy reduction is not an automatic failure.
- Fresh target-logit/trajectory/private-state and MTP1/MTP2 checks precede C1/C8
  scalar MTP2, stronger ordinary256 and MTP1-cohort screens (known cohort C2 too).
  Reuse independent teacher/control references only by exact pinned hashes.
- Per-path C1 HTTP>1% or C8>2% (MTP1 cohort C2>2% also permits) triggers a full
  three-repeat1/2/4/8 matrix. Default adoption requires that gain to persist, at
  least two improving paired-index trials, and no matrix cell >2% slower. An
  unchanged single post-first decode cannot be substituted for the full goal.
- Separate eager projection and full-call profiles from unprofiled timings.
  Counter permissions stay unchanged: report achieved occupancy/DRAM/cache/stall
  counters unavailable, never replace them with static residency or NVML busy.

Sync complete current sources to a fresh remote snapshot and assert a full
pre-build manifest before compilation. All compilation stays remote on the V100
host; no local Rust/CUDA build, unrelated process changes, commit/push, permanent
service or earlier-capsule edits. Restore all native dependencies if gates fail;
keep explicit research API support only if independently qualified and labeled.

## One evidence-driven follow-up: staged T16

The natural T16 variant completes all124 fixtures/full2,209-test GPU suite and77
operator cells, but median eligible fused/plain rates are0.93143/0.93055×. It
fails the declared gate; no trained-model or serving campaign is run for that
arm. Both target functions have zero reported local memory. Shared staging is
36,864 bytes with64/71 registers in plain/fused kernels and two256-thread resident
CTAs (25% theoretical occupancy), versus two512-thread T8 CTAs (50%). These facts
do not prove that occupancy is the sole cause of the slowdown.

One bounded dataflow follow-up retains T16/row8 and original per-output math,
but stages two sets of eight activation rows sequentially, holding each weight
and scale across both stages. Shared storage falls to18,432 bytes, while barriers
double per K tile. More live weight registers, lower ILP or barriers can erase
any residency benefit; no forced compiler block bound or further geometry sweep.
All threads, including invalid row/tail lanes, participate in both stage barriers.
Fresh source/build/fixtures/operators compare original T8, new staged T16 and the
pinned natural T16 ablation in three alternating repeats. Only staged T16 versus
original T8 may meet the unchanged fused>=1.05/plain>=.98/T8>=.98 gate. Its new
resource-parser CPU check distinguishes eight-token shared staging from sixteen
output rows. Preserve the natural failure and original immutable snapshot.

## Next-action prerequisite, not a T16 performance arm

After the staged run is terminal, a separate two-V100 transport probe uses the
unchanged qualified library to test CUDA Graph capture across pinned host
D2H/H2D bridges with both producer/consumer fences. Its synthetic T3/K128/17-row
projection shapes, alternating eager/graph trials, exact bytes and changed-input
replays are capability diagnostics only. No model, dynamic KV/GDN position or
source/failure/reset graph support is claimed. Keep all trials, including any
slower graph trial; do not extrapolate a serving speedup. GPU samplers and owned
runtime/stream/event/graph handles are cleaned up before the terminal audit.
