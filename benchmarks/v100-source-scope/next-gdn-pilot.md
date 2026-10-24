# Next distinct experiment: semantic head-local GDN block pilot

**Unimplemented. No model TPS gain is claimed by this plan.**

The source-check experiment is negative: removing95.74% of measured checkpoint
syscalls does not pass the1% screen gate. Its matched CUDA profiles retain the
same kernels and host-transfer payload. Do not retune this source-check policy,
repeat denied hardware counters, or assume a higher theoretical occupancy is
the goal. The dominant target verification/projection/communication work remains.

Also do not jump directly to deeper native MTP. The already-frozen external
depth sweep at `benchmarks/v100-strided-output/v100/external-depth-v9` tested
depth2/3/4 with three repeats each. All accepted17 draft tokens on the pinned
story; depth3/4 proposed38/49 versus26 for depth2 and were slower. That does not
prove native depth3 can never help, but it requires a new acceptance/round-count
economics gate before implementing another transactional-prefix mechanism.

## Pilot before a native ownership rewrite

Evaluate a complete GDN block, excluding its following MLP, on the same two V100s
and actual trained layer weights. Match primary-only FP16 input/output, source
small-parameter precision, original residual-A8 arithmetic, independent private
states, explicit host transport, and both ownership fences. Compare current
row-TP/primary-GDN versus head-local GDN plus column-split output projection.

Use these workloads, not just the total row count:

- One sequence, one token, ordinary compact GDN.
- One sequence, three verification tokens, including prefix1/2 state/history
  traces and commit/restore tests.
- **Eight independent sequences with one token each**, matching actual native
  batched decode. One eight-token sequence is not an equivalent GDN workload.
  Do not hide the extra per-request kernel-launch/ownership cost of two ranks.

Keep block timing native/CPU-dispatch-inclusive; no Python numerical hot path or
summed concurrent timers. Three alternating trials per arm, sampled NVML and
separate CUDA timeline/copy-call diagnostics. A provisional integration gate is
>=1.10x at the actual T3 verification workload and no >2% regression at T1 or
the real eight-sequence workload. If the block fails, stop before rewriting model
cache/prefix ownership. A block gain is not a new whole-model TPS result.

## Actual Qwen geometry and numerical constraints

Current GDN uses16 key/query heads and48 value heads, all128-dimensional.
Full QKV width10240 consists of Q2048, K2048 and V6144; Z width6144.
Each rank should own8 key heads/24 corresponding value heads, giving local
QKV width5120 and Z width3072. A plain first/last5120-row QKV slice is **wrong**.

Canonical global QKV row intervals:

| Rank | Q | K | V |
|---|---|---|---|
|0|[0,1024)|[2048,3072)|[4096,7168)|
|1|[1024,2048)|[3072,4096)|[7168,10240)|

Reorder both Q8 codes and FP16 scales together, preserving every encoded value.
Select convolution coefficients/history with the same semantic channel mapping.
Split FP32 A/B rows and rate/bias by value head; retain FP32 source parameters,
the existing beta rounding and GDN normalization. Gated RMSNorm is per head.
Before the output projection, joined local QKV/Z, histories, recurrent values,
GDN states and prefix traces must match the original canonical bytes. Join
history as Q0,Q1,K0,K1,V0,V1, not by concatenating two packed local-QKV histories.

Split output-projection input columns at3072, a group32-aligned boundary; preserve
stored Q8 codes/scales and the original activation quantizer. Compute FP32
partials, transport FP32, sum before a single FP16 cast. That changes FP32
association, as the already-qualified column-MLP path does: declare it and use
the existing independent represented-weight block tolerances without loosening
them. Never claim original FP16 output-byte equivalence from the Q8 label.

Private state stays distributed during timed work. Canonical CPU joining for
diagnostic comparisons is untimed; copying every peer state back to primary on
each token would test a different design. Count and memory-admit both ranks'
private states, histories, prefix backups, constants and scratch. Replace packed
weights rather than duplicating an entire model on the peer. Exact copy-byte and
copy-call reductions must come from the pilot trace, not a layout-only forecast.

Only a positive pilot permits a new explicit model profile and native session/
MTP/batch ownership integration. That later change must independently qualify
trained teacher trajectories, target/draft correction, failure/reset/EOS/private
cohorts and fresh C1/C8 screens before any1/2/4/8 three-repeat matrix or new TPS
claim. The full >=5% external single plus HTTP1/2/4/8 goal remains active.
