# Native head-GDN integration protocol

The previous goal turn is progress: the frozen operator pilot passes its
corrected trained-layer speed and private-state gates. This turn implements the
actual native ownership path rather than declaring the block gain to be TPS.

## Explicit alternate profile

`q8-residual-column-mlp-head-gdn`: resident SM70 FP16 activation/KV, true official
Q8 weights/scales, two-pass residual-A8 projections, host transport between two
selected V100s. Existing default/profile constructors and C structs retain their
ABI. Fail closed on unsupported residency, PP, dtype, transport or geometry.
Semantic QKV rows and output columns are repacked by bounded native upload;
source bytes/scales are not requantized. Source small-parameter flags remain.
GDN output FP32 column association is new and disclosed, as for column MLP.

Model workspaces are shared/serialized. Request GDN state/history is halved and
stored privately on both ranks, including checkpoint/prefix1/prefix2 backups.
Do not retain a duplicate old full model layout or join private state in the
hot path. Scalar and ragged batch paths use the same segment-aware native
routine. Native reset, failure, lifetime, EOS, selective request subsets and MTP
restore/correction must operate on both devices. Canonical snapshots remain
untimed diagnostic-only joins and preserve the original external shapes.

## Gates

1. Remote-only compilation from a synchronized, hashed snapshot with the new
   header included in build dependencies. Preserve failed tests/builds and every
   executed source. Test eligible fixture without changing the old DV13 fixture.
2. Focused synthetic complete-model teacher checks, T1/T2/T3/T8 chunking,
   independent8 caches, EOS/ragged/selective reset/failure/native-only generation;
   MTP2 forced acceptance/rejection/prefix restoration and corrected draft state,
   MTP1 packed cohorts1/2/4/8 and native failure/reset/close. Run applicable old
   native regression tests to detect changes in existing profiles.
3. Reuse immutable archived original-FP16 teacher tensors with explicit hashes,
   not a repeated unchanged teacher campaign. New profile must pass all98
   teacher rows (L2<=.02, centered cosine>=.999, KL<=.002, exact top1), four full
   trajectories and scalar/batch/reset private caches. This is represented-Q8
   reference qualification, not original BF16 corpus quality. Independently
   qualify trained MTP2 and packed MTP1 target/draft/prefix state handling.
4. Only after correctness, alternate accepted old/new profiles in unprofiled
   C1/C8 screens, with matched model, work, MTP depth, capacities and arithmetic
   declarations. Promote repeatable >=1% native gains without >2% regressions to
   three-repeat HTTP cohorts1/2/4/8. Preserve stronger historical native and
   external baselines, especially external single49.8033 TPS (required >=52.2935).
5. Collect CPU core-seconds, TTFT/latency/native timing scopes, NVML and a separate
   CUDA/NVTX timeline. No achieved occupancy/bandwidth claim under unchanged
   counter policy1, no repeated denied counter campaign and no privilege change.

The full >=5% external single and HTTP1/2/4/8 objective remains active until its
complete requirements are verified. No model TPS is inferred from the pilot.
