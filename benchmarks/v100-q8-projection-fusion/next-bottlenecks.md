# Next bounded experiments, not implemented performance claims

The fused/grouped path is retained, but every external HTTP cell still loses.
Do not repeat unchanged row-output-copy removal, shared-T1 or WMMA experiments;
their negative serving/operator evidence remains in previous immutable capsules.

## 1. GDN launch geometry / inactive lanes

Source inspection: `gdn_recurrent` launches 256 threads per value row even at
DK=64/128, with one state component per thread. It performs Q/K normalization,
state-key and state-query reductions through the fixed eight-warp `sum_sync`
path. Native batch invokes this separately for each segment's private state.
Fusion does not change GDN; separate new traces still attribute about 56 ms
at C1 and 492 ms at C8 to it for 23-prompt/8-output generations. These are
profiled sums, not serving TPS or hardware-counter proof
of a particular stall or a promised speedup.

A narrow next prototype can specialize DK=64/128 launch width while preserving
the original logical reduction and state-update order. **Simply changing
blockDim is incorrect**: `block_sum` unconditionally reads eight partials.
Missing virtual warp partials must be defined, with correct synchronization,
and invalid/nonfinite propagation must still poison/reset correctly. Do not
apply a new block size globally to other kernels using `block_sum`.

First compare one-layer old/new outputs **and FP32 state bytes**, packed/unpacked,
FP16/BF16, DK=64/128/256, T1/2/4/8/23/32, nonzero initial states, nonfinite/zero,
alias/graph and tails. Use an independent recurrent reference, not only the old
CUDA kernel. Query exact compiled registers/shared memory and launch ceilings;
actual occupancy/counters remain unavailable under unchanged driver policy.
Only after operator gates, run the same full-model 98-vector/private-state and
MTP gates, C1/C8 screens, then conditional three-repeat promotion. Changed FP32
association is not a byte-preserving optimization and must be declared and
qualified separately, never silently folded into the existing profile.

A separate ragged descriptor-grid experiment could merge independent-request
GDN launches without merging their state, history or cursor. It must preserve
per-request token chronology, inactive slots, heterogeneous segment lengths,
EOS compaction, failure/reset and scalar-equivalent state. Launch fusion alone
is not assumed to cure GDN's per-value work.

## 2. Native longer-chain MTP

MTP1 now improves to 34.40 native post-first TPS, but the best tested external
MTP2 reports 49.80 TPS in its own post-first scope. Reusing MTP1's draft-cache
shortcut at depth two is incorrect. A longer native chain needs a speculative
normalized-draft-hidden recurrence, bounded draft cursor/KV rollback, and draft
catch-up using **verified target hidden states**, not stale speculative states.
The target must restore GDN/history and replay exactly the accepted input prefix
on partial rejection/EOS/budget boundaries. Committed target state must match
ordinary native decode, including mixed accept/reject and failure/reset. No
sampling or batched-MTP claim follows from an initial greedy chain prototype.

## 3. Larger TP communication redesign

Current row-sharded projections gather complete results to the primary; GDN
state stays primary-owned. A proper FFN TP design could keep row-sharded gate/up
results local, apply local SwiGLU, use K-column-sharded down projections and
reduce **FP32** full-hidden partials before one final FP16 cast. This changes
reduction association, needs explicit arithmetic/profile/quality qualification,
new memory admission, byte-verified packing and transfer accounting. It is not
safe to sum independently rounded FP16 partials. Head-partitioned GDN/attention
would be a subsequent ownership/state redesign, not an unmeasured kernel tweak.

Prioritize measured remaining time and end-to-end screens. Never claim a higher
theoretical occupancy, reduced API count or microbenchmark gain is an achieved
serving improvement before measuring it.
