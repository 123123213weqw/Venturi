# Next measured work, not an implementation claim

1. Multi-input target verification dominates successful prefix MTP2 rounds.
   Attribute T3 residual projection work and padding before another kernel change;
   compare against unchanged arithmetic and actual end-to-end speed, not static
   occupancy alone. Two-step draft recurrence also does more head/catch-up work.
2. Independent scalar MTP requests still serialize shared target model scratch.
   Better C1 speculation does not solve C2/4/8 throughput. A future batch verifier
   must pack different request input widths and preserve per-request GDN/history,
   KV, accepted-prefix state, EOS and pending tokens. It must beat the stronger
   existing ordinary native cohort batch path, not just scalar MTP.
3. Do not infer a universal speculative speedup from the story case. Acceptance
   includes unused deeper proposals; retain original-source model/draft and
   quantify memory, EOS/short-output cases, other prompt families and long context.
4. Complete-request HTTP and post-first decode are different scopes. Retain the
   best tested pinned external MTP configuration for C1 and its non-speculative
   matrix for parallel work. No external decode or full-goal win follows merely
   from reduced prefill/replay time.
5. Hardware counters remain unavailable under the unchanged driver policy.
   Kernel resource limits and NVML busy percentages are not achieved occupancy.
   No driver/system settings are changed to improve reported measurements.

## Concrete next kernel experiment

Current SM70 T3 residual projections still dispatch a padded token4/row8
K-tiled kernel. The existing full-activation shared path only accepts T1/T2.
At K=5120, T3 with two activation passes needs 34,560 shared bytes, within the
existing 48 KiB limit; wide MLP down projections do not fit and need a fallback.
A separately gated T3 full-sharing/token3 specialization for plain and fused
projections is therefore a concrete candidate, **not a measured improvement**.
Preserve scale/FMA/reduction order and both TP ownership fences. Screen actual
checkpoint shapes/rows and exact output bytes before rebuilding the full model;
then run trained state gates and fresh C1/C8 performance. Higher theoretical
occupancy or less padding alone is not an acceptance criterion.
