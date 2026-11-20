# Compact GDN launch geometry

## Kernel invariants

The old `gdn_recurrent` remains intact: one 256-thread CTA/value row. At DK=64
or 128, most or half of its threads have no state component. The new explicit
`venturi_gdn_compact_run` launches exactly DK threads on SM70 for these two sizes;
DK=256 and other architectures use the original operator. FP16 QKV/output, FP32
gates/state, normalization epsilon, per-component state update, FMA and output
rounding are unchanged. The kernel still processes one sequence chronologically,
not a batch with merged state. No allocation, Tensor Core or new workspace.

Simply reducing blockDim would be incorrect: the original block reduction reads
eight warp partials unconditionally. `gdn_compact_sum` retains those **eight
logical partials and their exact reduction order**. Real warps supply the same
partials as before. Missing warps receive the sum of their virtual inactive
lanes, with a barrier before consumption. Those virtual lanes track zero query,
key and state through normalization/decay/FMA rather than assuming every invalid
input behaves like a finite zero. Thus nonfinite/signed-zero propagation is not
silently changed by reading uninitialized shared memory or dropping virtual
work. Another barrier protects shared-result reads before the next reduction.
The shared array remains 32 bytes. FP32 state layout and payload are unchanged.

The legacy public `prepare_gdn` stays unchanged by default; an internal plan
entry field and explicit research `prepare_gdn_compact` select the new export
without mutating a backend or introducing a numerical fallback. SM70 BF16
execution remains rejected as before; no BF16 V100 speed claim. Packed/unpacked
operators keep their previous alias, dtype, shape and caller-owned-state contract.

## Native integration boundary

Only the existing explicit `q8-residual` target selects compact GDN in native
scalar and batch forwards. Default FP16-activation targets, PP and the original
public primitive retain legacy GDN; SM89 and DK256 fall back. MTP uses the changed
target but the unchanged FP16 draft. There is no new CLI precision switch,
Python token/layer loop, shared mutable per-request cache, smaller KV or altered
checkpoint encoding. Blocks ABI remains 16 with nine explicit build sources.

## Evidence-first sequence

Study snapshot v1 has only the explicit research operator; native model dispatch
is still old. It passes **125 operator tests**, then **1642 full-suite passes /
440 skips**, before matched primitive timing and separate profiling. Numeric tests
compare pinned old-library, candidate legacy and compact outputs/FP32 state bytes,
plus an independent FP64 matrix recurrence. They span T1/2/4/8/23/32, DK64/128/256,
normalization on/off, packed/unpacked, nonzero initial states and guarded outputs.
Chunked in-place state, graph/no-fallback, alias, raw geometry and nonfinite
classification/finite-value bytes are checked. No NaN-payload bit identity claim.

Snapshot v2 integrates residual-target dispatch, adds signed-zero/dynamic-graph
input tests, and passes **127 operator tests**, **1644 full-suite passes / 440
skips**. Local CPU-only suite: **766 passed / 1318 skipped**; skipped GPU cases
are not claimed as local GPU passes.

Fresh trained residual candidates compare complete pinned control signatures:
98 full-vocabulary vectors, four greedy trajectories/final private states,
eight heterogeneous batch slots and selected reset/continuation logits/caches.
Every byte matches. Fresh four-case MTP checks require ordinary-target output
and committed state equality with natural acceptance/rejection/replay/EOS.
Default trained FP16 signatures and unchanged draft's independent CPU reference
are reused, not rerun. The qualifier now propagates the explicit inherited
unexecuted-profile reference rather than losing its provenance in a residual-only
chain. It refuses selected profiles without complete prior signatures.
Original independent target gates and thresholds remain unchanged; no broad
original-BF16/corpus-quality claim or fresh large payload hash scan.

## Measurement scopes

The primitive benchmark uses Qwen GDN geometry Hqk=16, Hv=48, DK=128, DV=128 at
T1/2/4/8/23/32. DK64/256 T1/T8 are separately labelled non-Qwen/fallback controls.
Three repeats, 32 graph-contained primitive calls per wave, reversed alternate
arm order, immutable input state and distinct state outputs. Control and candidate
legacy use 256 threads; compact is explicit. Independent reference checks precede
timing. This warmed primitive study is not serving TPS; short windows do not
yield reliable per-wave NVML utilization. Broad telemetry is retained.

Fresh native C1/C8 screens precede conditional three-repeat non-spec C1/2/4/8
and MTP1 C1 measurements. Same official-original-derived Q8 target, declared
two-pass residual profile, FP16 KV/activation boundary, resident host TP on
physical GPUs 1,0; 32 projection rows, eight slots, 256 context capacity per
slot; identical 23 prompt IDs and 32 fixed greedy outputs. The fixed-work speed
case ignores EOS; correctness cases test EOS separately. HTTP includes the
25-ms admission window. CPU core-seconds, client latency, native TTFT/post-first,
VRAM and GPU samples are separately reported. Loading, warmup/reset and
profilers are outside timed waves; cohort TPS never sums request durations.

Fresh candidate 23-prompt/8-output C1/C8/MTP traces are separate from timing.
Identical unchanged control traces are hash-pinned and explicitly reused from
the preceding fused-projection capsule. Raw retained binaries have size/SHA256
records. Resource queries report static limits, not achieved occupancy. Actual
SM/DRAM/cache/stall counters remain unavailable under unchanged restricted driver
policy; no repeated denied campaign or privilege modification.

## Recovery

One SSH observation closed during a transient connection problem. A bounded
same-host probe with keepalive/connection deadlines succeeded. The original
worker was inspected and found terminal with exit code zero; it was not restarted.
Strict host verification, dedicated keys and no agent forwarding were retained.
No host switch, inference duplication or local Rust/CUDA compilation occurred.


## Profiling-launcher correction

The first final-profile worker (v4) tried to pin an optional previous API-audit
JSON that exists in the local frozen capsule but not in the server directory.
It failed before GPU profiling; its traceback and exit code remain preserved.
The v5 successor removes only that unused reference. Candidate source/library,
trained tests and timed campaigns are reused unchanged, not repeated or relabelled.
