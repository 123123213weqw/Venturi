# Native MLP-only column/row TP: explicit new arithmetic

The previous turn made verified progress: isolated trained MLP block throughput
improved1.239/1.287/1.316× at T1/3/8, with encoded weights and FP32 partial transport.
That result was frozen without a model gain. This experiment integrates the
block into native scalar and packed Batch/MTP verification, not a new smaller
acceptance target. Existing row-TP attention/GDN remain unchanged.

## Implementation contract

An additive ABI16 constructor enables `q8-residual-column-mlp`: SM70 resident
FP16 target, two-pass residual-A8/FP32 scales, two GPUs and explicit host transport
only. Original API/layouts/default precision and original FP16 draft are retained.
Replace down weight storage with all-output/input-column halves, preserving
stored codes/scales without requantization; group-aligned equal cuts, zero local
padding and no alias to another weight role. Load through bounded8MiB staging,
keep source identity/pre/post checks. Do not duplicate permanent down weights.

Broadcast normalized full hidden once. Local fused gate/up and local FP16 SwiGLU
feed down kernels producing FP32 partials. Peer partials cross pinned host in
FP32; native FP32 sum precedes one FP16 cast, then the original separate FP16
residual addition. Both existing producer/consumer CUDA-event fences remain.
Explicitly admit/count local/remote/peer FP32 buffers, peer GU/activation and
pinned partial storage; private GDN/convolution/KV/prefix images do not change.
Scalar and packed Batch call the same native MLP method. Old profiles keep the
original exact call order. No other profile can accidentally read column-packed
down via the old row projection.

FP32 summation association changes versus full-K down. This is a named new
compute profile, not old-profile byte parity or lower-precision transport. Keep
the unchanged represented-Q8/FP16 teacher limits: L2<=.02, cosine>=.999,
reference KL<=.002 and top1 equality. No original BF16 or broad corpus claim.

## Gates and decisions

1. CPU preflight/missing export before allocation, ABI and memory metadata;
   complete source sync/manifest assertion before remote-only compilation.
2. Fresh independent full-MLP FP64/actual FP32 partial diagnostics, four token
   widths/two layer kinds, private scalar/batch/mixed EOS/reset, MTP2 accept/first
   and second rejection/prefix recovery, corrected draft bytes, no Torch math,
   failed/reset/closed/source owners, threaded2/8 and MTP1 cohorts2/8. Full unchanged
   native suite catches old-profile regressions; no local CUDA/Rust compilation.
3. Fresh FP16 teacher98-row hashes/private state against pinned complete prior
   signatures; new-profile98 teacher rows/four trajectories, diagnostic/generate
   exact states, mixed eight-slot scalar/batch identity and selective reset.
   Fresh trained MTP1/MTP2 prefix and independent original-source draft recurrence
   with new target hidden; no stale hidden-state or old-profile hash substitution.
4. Fresh unprofiled C1/C8 strongest-native/candidate screens for scalar MTP2,
   ordinary256 and packed MTP1. Any path with C1 HTTP>1% or C8>2% advances to three
   repeats at1/2/4/8; no repeated cell may regress>2%, qualifying cells need two
   improving trials. Then fresh pinned external comparisons, preserving stronger
   known external MTP2 for single and supported parallel configurations.
   Overall >=5% external single-decode/every count remains the completion gate.
5. Separate node/kernel/transfer timelines, CPU/latency/TTFT/VRAM/NVML data,
   acceptance/replay denominators and full precision/timing differences. Do not
   repeat denied counter runs or change policy1. Theoretical residency isn't
   achieved occupancy; summed kernel work is not a critical-path share.

Preserve all36 prior capsules and failed experiments. No commit/push, permanent
server, unrelated GPU/process/driver changes or automatic host switching.
