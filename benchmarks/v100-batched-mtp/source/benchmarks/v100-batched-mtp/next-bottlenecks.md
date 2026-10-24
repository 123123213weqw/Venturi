# Follow-up priorities (not implemented or claimed as speedups)

## Token-tile scheduling / weight reuse

The current wide residual projection uses token tiles of 8 and CUDA grid
`(row_blocks, token_tiles)`. A 16-row MTP verification therefore has two token
partitions of the same weights. Before more instruction/row-tile sweeping, test
row-major blocks with **token tiles adjacent** in the flattened block order:

- `flat = row_block * token_tiles + token_tile`;
- recover `row_block = flat / token_tiles`, token origin `(flat % token_tiles)*8`;
- apply to plain `vdt::dot` and fused `vqf::tiled_dot`/descriptor selection;
- preserve every lane's K/FMA/reduction order, all tails and both TP fences;
- retain old 2D geometry when the flattened grid would exceed device/grid integer
  limits, and do not perturb the one-token-tile decode path without testing it.

Hypothesis: neighboring CTAs may reuse a weight tile through cache before it is
evicted. This may help 16-row verification and 23/46/92/184-row prefill. It is not
proof of L2 hit rate, DRAM traffic or bandwidth; hardware counters are restricted.
Measure real shapes, exact bytes and model C1/C8 performance before promotion.
The earlier rejected T3 body/row search does not test this block-order change.

## Draft work

The first implementation batches the expensive target verifier but keeps native
per-request shifted prefill and drafting/head/catch-up. If profiling shows draft
cost dominates, first consider collecting independently normalized draft outputs
and running the shared proposal head once across active requests. That must keep
proposal logits/selection, draft KV and verified-hidden catch-up correct; add
shared-head counters rather than hiding its time in repeated per-slot totals.
A subsequent full draft-block cohort can reuse request-owned draft caches, but
must not duplicate permanent caches or invoke an unbound draft embedding/head.

## Single request

Scalar MTP2 remains the stronger existing single path. Preserve its qualified
control. Separately attribute native host gaps and cross-GPU overlap before
claiming a graph/CPU optimization. A request-scoped resident source-identity check
could remove repeated metadata work only with start/end mutation detection,
exception cleanup and unchanged source-guard semantics; measure syscalls first.
A fixed-shape graph must preserve dynamic positions, cache/prefix ownership,
host-TP transport/fences and failure/reset behavior. Process timeline gaps are
not device-wide idle or a prediction of graph speedup.

## What the new cohort measurements rule out

The C8 loss is not only staggered serial prefill. In the initial three-repeat
cohort study, common-prefill completion is about 1.48 s (ordinary batch about
1.25 s), and the native remaining-248-output interval is also slightly longer
than ordinary's. Preserve both denominators. The complete 32-output C8 profile
has fewer kernels (69,592 vs 82,688) but *more* summed kernel time (~4.69 vs
4.20 GPU-seconds); projection kernels account for roughly 83% vs 79%. Fewer
launches therefore do not prove lower execution cost. The 16-row verifier's two
weight-reading token tiles and serial draft/head work remain separate hypotheses
for attribution, not established DRAM/cache counter results.

The SDK revision in this capsule changes repeated public metadata checks only.
It must not be presented as a native decoding-kernel speedup, fixed-shape graph,
batched draft implementation, or an exemption from source mutation guards.
