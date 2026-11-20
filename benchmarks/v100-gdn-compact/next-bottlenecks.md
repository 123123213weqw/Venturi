# Remaining work, not implemented performance claims

The compact launch helps, but the complete external >=5% acceptance is not met.
Do not substitute the 1.8x primitive result or one slightly faster external cell
for single/parallel whole-engine acceptance.

## Repeated GDN normalization/decay

Source inspection shows that every value-row CTA independently normalizes the
same Q/K head and evaluates the same head decay. For Qwen there are 128 value
rows/head. A bounded next prototype could precompute original-order FP32 Q/K
normalization and exp(decay) once, then let the state kernel reuse exact stored
FP32 values. No extra FP16 cast is permissible. It must preserve the original
logical reduction/FMA order and virtual-lane behavior, explicitly account for
new scratch and locking, and beat the extra preparation launch/traffic at T1/T2
as well as larger T. This is a hypothesis, not an implemented speedup.

Keep independent state/output byte gates, zero/nonfinite/graph/chunk/alias tests,
trained logits/private caches and MTP transactions before any serving claim.
Changing reduction association requires explicit new quality qualification,
not silent inclusion in the already-qualified arithmetic profile.

## Batched prefill projection budget

The serving control still caps projection rows at 32 despite 256 context tokens
per slot. For eight equal requests, prefill splits each prompt into small
segments. A separately declared 64/128-row budget could reduce repeated layer
launches, state traffic and transport fences while keeping prompt/output/context
and arithmetic fixed. It needs complete memory admission/accounting, extended
operator/token-tail tests, byte-exact ragged batch/reset qualification and matched
C1/C8 screens. It is not an implemented gain, and a larger buffer alone is not
a throughput result. Single-request prefixes already fit 32 rows, so this would
not solve the remaining single-request gap.

## Native MTP2 and communication

The same transaction-safe longer-chain MTP requirement remains: speculative
normalized-draft-hidden recurrence, draft KV/cursor rewind, catch-up with verified
target hidden states, and exact target GDN/history rollback/replay of accepted
prefixes at rejection, EOS and output-budget boundaries. The MTP1 shortcut is
not valid at depth two. Compare successful competitor configurations rather
than treating failed configurations as zero-speed cells.

The row-sharded target still gathers results to a primary-owned GDN path. A
larger FFN column-shard/FP32-partial-reduction or head-partitioned GDN redesign
needs new ownership, admission, arithmetic and independent quality evidence.
Never sum rounded FP16 partials or infer achieved utilization from occupancy
ceilings/NVML busy percentages. Prioritize measured remaining time and actual
C1/C8 serving screens; preserve regressions and immutable controls.
