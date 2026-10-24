# Next experiments after native column/row MLP integration

Current single post-first47.629 TPS needs another9.79% to reach the strongest
pinned external49.803 *1.05 =52.293 TPS floor. HTTP acceptance at1/2/4/8 is useful
progress, not a substitute for single decode. No experiment below is implemented
or proven to close that gap by this capsule.

## Measured remaining work

Separate new C1 profile: fused target projections sum440.856ms, FP32 partial down
170.734ms, remaining old row projections~134.194ms, GDN prefix41.458ms plus compact
21.673ms, FP16 single-token draft kernels36.001ms. These overlap across devices;
their sums are neither additive wall-time savings nor critical-path percentages.
Draft accounting, including warmup and catch-up, stays~80.19ms. Acceptance17/26
matches the external control, so lower acceptance is not the current explanation.
Host copies still total316.660MB in each direction for the instrumented C1 call.

C8 ordinary: fused target kernels sum2.168s, FP32 partials0.752s, GDN compact0.515s;
C8 MTP1: fused2.236s, partials0.808s, serial FP16 draft kernels0.203s, GDN prefix
0.295s plus compact0.173s. Different strategy winners remain; do not enable MTP
for all cohorts. Achieved occupancy/bandwidth remains unavailable under policy1.

## 1. Bounded integer-activation projection prerequisite

A distinct operator-only candidate is signed INT16 activation/group32 DP2A with
FP32 scales and the same stored Q8 weights. It would use one quantization pass,
16 integer-dot instructions/group instead of16 for the current two A8 passes,
and one scale/FMA instead of two. An INT16 group32 representation uses68 versus
72 bytes/group for two A8 passes. Weight traffic is unchanged; this may therefore
fail to help the dominant fused kernels. There is **no measured speedup** yet.

Installed CUDA12.9 headers were inspected and SHA-pinned in
`installed-dp2a-header-observation.json`: signed `__dp2a_lo/hi` take two INT16
values in srcA and select the lower/upper two INT8 values of srcB, accumulated in
INT32. Do not reverse the argument packing. With codes[-32767,32767], the
32-element dot bound133165088 fits INT32. Exact cancellation, all signs, tails,
padding/nonfinite rules, FP32 scaling/reduction and independent FP64 error checks
are prerequisites. Error bounds alone do not establish model quality or speed.

Start with a finite isolated comparison on actual trained fused/projection
shapes at T1/3/8; require at least5% median operator gain with no important cell
regression before native integration. Use a **new named compute profile**, not a
silent replacement of residual-A8. Any promoted model needs fresh teacher/state/
MTP gates, changed-arithmetic disclosure and C1/C8 screens before repeated matrices.
Do not launch another unbounded block/row/register/tile sweep.

## 2. Explicit faster draft arithmetic

The current MTP body retains original FP16 projections. An additive opt-in
residual-A8 draft profile might reuse qualified DP4A/fused kernels; original
FP16 remains the default. That would be a separate draft arithmetic change,
requiring independent original-source draft hidden/KV checks, corrected-state
and accept/reject/EOS tests, complete target trajectories and actual acceptance
counts. Never reuse old-profile draft hashes as new-profile quality evidence.

The~36ms C1 FP16 kernel sum/~80ms draft-accounting span is only an opportunity
bound, not proof of enough critical-path savings to close the9.79% goal gap.
Faster drafting can also lower acceptance; screen complete C1/C8 behavior before
promoting. Parallel draft execution is not supplied by packed target verification.

## 3. Semantic head-sharded attention/GDN, only after a prerequisite

Remaining GDN/attention transport and primary-only state may be worth reducing,
but MLP's result does not prove a full state-ownership rewrite will win. First
measure original48-head GDN versus two24-head halves with exact joined prefix
states and two-GPU wall. Half the heads can underfill the grid; do not predict
half the kernel latency from a head-count ratio.

A real implementation needs semantic Q/K/V and A/B ownership, local norms/conv/
gates, FP32 O partials, peer private recurrent/history/prefix state and exact
reassembled diagnostic snapshots. Existing prefix arrays index recovery depth,
not TP rank. PP's secondary session is not an interchangeable TP peer cache.
Preserve both ownership fences, source/failure/reset semantics and all MTP
transactions. Logical byte reductions are not a TPS forecast.

## Do not repeat unchanged negatives

Whole-target layer graphs already lost1–2%; adjacent CTA ordering, paired words,
T16 reuse and earlier bounded tile variants missed promotion. Higher theoretical
residency, a lower kernel-duration sum or sampled GPU busy is not new engine
evidence. Retain the faster explicit column-MLP profile while testing a distinct
candidate; keep the stronger external controls and the full goal unchanged.
