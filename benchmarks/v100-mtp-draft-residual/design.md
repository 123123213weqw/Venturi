# Explicit residual-A8 MTP draft arithmetic

Previous turn is progress: a correct standalone DP2A implementation was measured
on18 trained operator shapes, failed its dominant fused gate and was frozen
without production adoption. Do not repeat that failed search. The qualified
column-MLP target remains47.629 single-decode TPS; full acceptance needs52.293.

## Native/SDK contract

Add optional ABI16 MTP-create-profiled and projection-info exports. Original
constructor/default stays original typed FP16/BF16; missing optional exports never
break it. New `draft_projection_profile="q8-residual"` explicitly selects SM70,
resident FP16, non-PP MTP body with two group32 residual-A8 passes/FP32 scales.
Stored Q8 weights/FP16 scales, FP16 activations/KV, target vocabulary/head and
the already-qualified target profile stay unchanged. This is different arithmetic,
not original FP16 parity or independent BF16/corpus quality. No DP2A, draft TP,
new token acceptance trick, paging, automatic router or whole-model graph.

The body reuses existing native FC/attention/MLP projection and fusion paths;
admit/count its additional quantization workspace. No Python numerical hot path.
Expose profile/pass/workspace/precision metadata and CLI/benchmark/qualifier/
profiler flags. Default callers retain old exports/call flow. Reject unsupported
profiles, PP, dtype, architecture or missing exports before device allocations.

## Gates

1. Full source sync/manifest before remote-only native compilation. New CPU
   preflight, independent source-equation hidden/logits/KV fixtures, exact original
   FP16 bytes vs the qualified old SO, forced first/second rejection and prefix
   recovery, EOS, private2/8 and cohort2/8, ragged/selective reset, source/failure/
   closed owners/no Torch math; full original GPU suite for old paths.
2. Fresh independent original-source trained draft block and recursive hidden/KV
   checks with unchanged L2<=.02/cosine>=.999 limits. Four trained target/MTP2
   trajectories/corrected draft MTP1 bytes, mixed-eight MTP1 cohort/reset states.
   Unchanged qualified target teacher/98-logit quality reference is pinned/reused;
   fresh ordinary target outputs/final private states must match it. Do not reuse
   old-profile draft state hashes as new-profile quality. Default FP16 gets a
   fresh regression gate before timing.
3. Same-library FP16-draft vs residual-draft unprofiled C1/C8 screens for scalar
   MTP2 and MTP1 verification cohorts, qualified target column-MLP/rows32 and all
   fixed IDs/work/policies. Ordinary256 target is unaffected; preserve its stronger
   known C8 rate. Promote a changed path only for C1 HTTP>1% or C8>2%; require three
   repeats/arm at1/2/4/8, no repeated cell regresses>2% and two improving trials.
   Reuse stronger pinned external controls unless a final near-goal candidate
   justifies a fresh complete external campaign. Full single-decode plus every
   HTTP count >=5% remains the goal, not this experiment's adoption trigger.
4. Keep all acceptance/depth/replay/unused-suffix counts. Separate draft and
   verification/prefill times, complete HTTP versus native decode/TTFT, CPU,
   request latency, VRAM/NVML and separate timeline/resources. No new top-1/corpus
   claim merely because target verification repairs a poor draft. No achieved
   occupancy/bandwidth claim under unchanged counter policy1.

Preserve all38 frozen capsules and failures; no local Rust/CUDA builds, unrelated
job/driver changes, permanent services, automatic host switch, commit or push.
