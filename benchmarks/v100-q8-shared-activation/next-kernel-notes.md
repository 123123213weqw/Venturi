# Next hypothesis: coalesced half-group loads with byte-exact reduction

Not implemented or measured. Do not treat this design as a performance result.

The present Q8 dot gives each lane one 32-byte scale group, loaded as two
16-byte vectors at a 32-byte lane stride. Hypothesis: adjacent half-group lanes
can reduce sector requests without changing checkpoint layout or arithmetic.
No hardware counters currently establish that sector traffic is the bottleneck.

Use two warps per output row. Within a warp, lane pair `(2g,2g+1)` loads the
first/second 16 bytes of a group, yielding contiguous 16-byte lane accesses.
Warp index within the row chooses group base 0 or 16; logical old lane is
`warp_in_row*16 + lane/2`. Iterate K in **1024-column** chunks, preserving each
old logical lane's group order. In each residual pass, four integer DP4A ops
per lane produce its half-dot; shuffle-XOR by 1 adds the halves **as integers**.
Only even lanes perform the original `fmul_rn(weight_scale, activation_scale)`
and `fmaf_rn(integer_full_group_dot, scale, sum)`. Do not scale each half
separately, which would change rounding.

All lanes execute a common ceil(K/1024) loop. Guard invalid group loads, but
keep the integer shuffle outside lane-divergent guards with a valid full-warp
mask. A group's two halves both exist in padded storage even for K tails;
quantizer padding is zero. Invalid groups must skip their floating FMA.

For final reduction, warp 1's even lanes store their 16 partial sums per token
in small CTA shared memory. After a CTA barrier (including invalid-row warps),
warp 0's even lanes add those corresponding sums, reproducing the original
shuffle-down-16 stage. Finish logical reductions by 8/4/2/1 (physical lane
deltas 16/8/4/2, even-lane mask); lane 0 stores. This should reproduce the old
output tree rather than introduce a new reduction/arithmetic profile. Prove it
with output/workspace bytes, tails, nonfinite and independent FP64 gates.

Ablate 2/4/8 output rows per CTA, T1/T2/T4/T8 and both activation pass counts.
Regard masks, shared barriers, register pressure and small row grids as open
risks. Keep the original default and SM89 untouched until qualified. Screen
actual TP-first shapes against the strongest pinned T8/shared-T2 control before
trained integration; then C1/C8 plus targeted MTP/C2 screens before promotion.
If byte parity fails, fix it or explicitly qualify changed arithmetic rather
than silently relabel it. Do not repeat the unchanged rejected copy/T1 shared/
naive WMMA variants.

For unchanged-profile scheduling/storage changes, a future qualifier may safely
compare candidate hashes against the fully hash-pinned prior control's complete
logit/state signatures, rather than rerunning all unchanged FP16/control work.
It must pin source/artifact/library/work/case metadata, compare complete private
cache and reset records, and clearly mark reused reference evidence. Do not
weaken gates or use only greedy-token equality.

## Higher-priority extensions suggested by this study

These are hypotheses, not implementations or guaranteed wins:

1. **T4/T8 K-tiled shared staging.** Full T8/K5120 activation data is 92,160
   bytes (and larger K exceeds even opt-in shared capacity). Stage 32 scale
   groups / 1024 columns at a time instead: T8/P2 needs 18,432 shared bytes.
   Preload all token/pass groups for that tile in separate int4-half planes,
   barrier, compute exactly the old lane's group for that K iteration, barrier,
   advance K by 1024. All row warps, including invalid-row/tail warps, must
   participate in barriers. Guard token/group loads; preserve group FMA order.
   Ablate row-block shapes and measure barrier/register costs. No function-local
   shared-memory opt-in is required for this tiled design. This addresses the
   fixed C4/C8/prefill paths that the current T2-only integration does not touch.
2. **Fuse scoped same-input projection groups, not just copies.** Gate/up,
   Q/K/V and GDN QKV/z already share packing. A fused descriptor-aware complete
   row-dot kernel plus one producer/completion fence per group could reduce
   kernel/API/event count. Keep separate output pitches, per-device descriptors,
   weight/source identity and peer scratch lifetimes. Validate worst-case sum
   of group shard sizes for scratch admission rather than assuming max(single
   projection) is sufficient on every model. The rejected pitched-store code
   may be useful inside this broader fusion, but does not itself establish gain.
3. The paired-half-group load design above remains a finite T1 experiment.
   Large-head T1 micro timing already streams the nominal code/scale payload
   rapidly; don't assume a 2x transaction reduction gives 2x elapsed speed.
   Event/copy/launch overhead and GDN work remain separate measured costs.

Prioritize actual speed screens over elaborate unmeasured redesigns. Future
native MTP2 still needs correct speculative draft recurrence and rewind/catchup
with true verified target hidden states; cursor rollback alone is not a complete
GDN transaction or proof that speculative draft KV is correctly conditioned.
