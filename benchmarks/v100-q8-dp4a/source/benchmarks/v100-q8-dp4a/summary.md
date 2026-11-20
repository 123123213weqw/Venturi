# Q8 integer-dot prototype and rejected controls on V100

## Status and next gate

**Operator-level progress, not a completed engine-speed goal.** The default native
Qwen runtime still uses the qualified FP16-activation Q8 arithmetic. A new,
explicitly named A8 integer-dot operator is implemented and tested; it is **not
wired into model generation or selected automatically**. No full-model quality,
end-to-end TPS or llama.cpp win is claimed for it. Earlier frozen evidence is
unchanged. The last promoted engine milestone remains C8 15.44 -> 28.84 HTTP TPS,
versus the declared practical llama.cpp non-speculative control at 63.90 TPS.

The next step is explicit native-model/TP integration of the residual-compensated
profile, followed by unchanged fixed teacher-forced CPU-reference thresholds,
trained-prompt quality/output checks, private-cache/state gates within the new
profile and matched 1/8 engine screening. Do not substitute projection gains for
this gate or silently replace the original precision path. Only a winning,
qualified engine candidate proceeds to >=3 repeats at 1/2/4/8 requests and MTP.

## Experimental progression

1. **T8 exact-order reuse failed.** Built old T8 with the six-block register bound
   and adaptive T8 with a four-block bound. All 130 fixture gates passed (125
   skipped), and real projection bytes matched. More reuse was nevertheless
   slower. Matched 23-input/32-output two-V100 cohort HTTP screen:
   C1 **15.13 -> 14.39 TPS**, C8 **28.84 -> 23.24 TPS** (one-repeat screening only;
   see exact current control values in raw reports). The T8 runtime change is
   rolled back, not promoted. New 7/8/16/32-row fixture coverage is retained.
2. **Scalar-packet DP4A was insufficient.** Explicit one-pass and residual-pass
   activation profiles passed independent tests, but the first four-coefficient
   packet kernel generally regressed single-row projections. Residual compensation
   was usually no faster even at larger row counts. Its raw measurements remain.
3. **Whole-group vector packets improve the projection.** The revised kernel
   transports each 32-code group in two `int4` packets, accumulates the group in
   integer arithmetic, then applies its scale once. This reduces repeated FP32
   scale/FMA work and short-loop overhead. It is the current staged prototype.
4. **Persistent peer submitter failed the engine screen.** Exact FP16/Q8 arithmetic
   was retained while a dedicated secondary-device CPU thread submitted work and
   fenced the correct event generation. The 130 fixture gates passed (125
   skipped), but C1 **15.22 -> 15.17 TPS**, C8 **28.87 -> 28.86 TPS**, with CPU C8
   **8.88 -> 9.16 core-seconds**. This thread/build-option implementation is removed
   from the live source, with its source and evidence preserved. No promotion.

The controls use the same two V100s, `CUDA_VISIBLE_DEVICES=1,0`, original-derived
Qwen3.5-27B Q8 artifact, FP16 KV/activations, explicit host TP transport, eight
slots with capacity 256, formatted 23-token prompt, fixed 32 greedy outputs and
25 ms batch-admission window included in client wall. Source revision, artifact
identity and profile differences remain pinned by the previous batch/comparison
capsules. No original service, clock, driver policy, model weight or license is
changed. All owned benchmark servers/profilers are stopped afterward.

## New arithmetic is explicit

The original Q8 code planes and FP16 weight scales are **unchanged**. Each FP16
activation row is quantized independently in 32-value groups, using an FP32 scale,
nearest-even signed codes in [-127,127], zero padding and a defined zero-group
rule. The optional second pass quantizes the FP32 residual of the first pass.
Integer group dots are combined with FP32 activation/weight scales and accumulated
in FP32 before the FP16 output cast. This changes activation values and reduction
order: it is **not** exact original FP16 math, even with residual compensation.

- `w8-a8-g32-f32scale`: one activation pass.
- `w8-a8-g32-f32scale-residual-a8`: high pass plus residual compensation.

`prepare_q8_dp4a(resident, tokens=..., residual_compensation=True/False)` requires
an explicit boolean choice. Caller owns an aligned device-byte workspace, and
execution performs no Torch math/allocation or implicit dense materialization.
Default `prepare()`/model execution remains untouched. Older ABI-16 libraries lack
these additive exports and give an explicit rebuild error, never a fallback.
Only the V100/FP16 path is qualified here, not SM89 or arbitrary model quality.

## Correctness

- New primitive: **72 V100 tests passed**. Independent CPU reconstruction checks
  quantized code/scale bytes, signed packing, FP64 sampled projection results,
  tails, zero/near-zero inputs, ties, two-pass compensation, workspace/alias guards,
  CUDA graph replay and execution with Torch math/allocation patched to raise.
  The final suite also checks the vector kernel's 16-byte workspace alignment.
- Final full V100 suite: **995 passed, 440 skipped**, including exact original-Q8
  byte gates and existing scalar/TP/PP/MTP/batch tests. Native BF16 on SM70 remains
  unsupported; opt-in/unavailable configurations stay explicitly skipped.
- Local full suite: **758 passed, 677 skipped**. No local CUDA or Rust compilation.
- Real checkpoint projection screen: eight unique matrix shapes, T=1/2/8/32,
  **32 cases**. Every profile passes the fixed sampled-row FP64 relative-L2 gate
  (<0.001) against its **own independently reconstructed activation profile**.
  Full outputs are also compared to the original FP16-activation projection,
  preserving errors and top-1 disagreements rather than relabeling them exact.
- The independent profile reference does not establish full-model/corpus quality.
  No original scalar state/quality gate is removed or loosened.

## Unprofiled projection results

Same immutable resident Q8 planes and seeded FP16 inputs for all three profiles;
three timed repeats, alternating profile order. Each timed wave replays three
three-node graphs (nine calls), after three warmup replays. CUDA-event projection
intervals include quantization and dot launches for A8. These graphs amortize
Python overhead **only for the operator benchmark**, not model serving.

Median speedup across eight real shapes, relative to the original projection:

| Rows | One-pass A8 | Residual A8 | Residual min–max across shapes |
|---:|---:|---:|---:|
| 1 | 1.36x | 1.40x | 1.25–1.44x |
| 2 | 1.78x | 1.53x | 1.21–1.71x |
| 8 | 2.26x | 1.60x | 1.35–1.76x |
| 32 | 2.48x | 1.68x | 1.43–1.89x |

Representative [17408,5120] projection: T1 **0.1679 -> 0.1204 ms**, T8
**0.7465 -> 0.4698 ms** with residual compensation. The long-K [5120,17408]
projection also improves: T1 **0.1762 -> 0.1252 ms**, T8 **0.7423 -> 0.5380 ms**.
CPU core-seconds, host wall and per-wave timestamps/telemetry are preserved in raw
reports; no concurrent request durations are summed into a throughput number.

Relative-L2 differences from original full FP16-activation outputs are about
**0.00495–0.00561** for one pass and **0.000068–0.000105** with compensation.
These are random-input projection errors, not a model perplexity or quality
score. One-pass top-1 disagreements occur in the screen. Residual compensation
retains all sampled full-matrix top-1 choices in these cases, but this does not
prove trained text generation parity. All error denominators are preserved.

## Diagnostics and actual competitor precision

Separate short Nsight projection calls use [17408,5120], T1/T8, 16 calls per
profile. Kernel attribution and actual launch/resource data are kept separately
from the unprofiled graph benchmark. Original calls launch 16 projection kernels;
A8 calls launch 16 quantizers + 16 integer-dot kernels. The vector kernel uses
128 threads/block, the quantizer 256. Example theoretical residency:

| Kernel | Registers/thread | Theoretical occupancy |
|---|---:|---:|
| Original T1/R4 | 56 | 50% |
| Residual integer T1 | 64 | 50% |
| Original T4/R1 (T8 input) | 40 | 75% |
| Residual integer T4 (T8 input) | 78 | 37.5% |

The faster residual T8 projection has **lower theoretical occupancy** than the
original; maximizing occupancy alone is not the objective. Achieved occupancy,
DRAM/compute/cache/stall hardware counters remain **unavailable**, with unchanged
`RmProfilingAdminOnly: 1`. CUDA driver occupancy queries are static bounds, not
measurements. No counter replay campaign or profiling privilege change occurs.
Raw 100 ms NVML busy/VRAM/power/clocks/temperature samples remain attached to the
unprofiled screens. Busy percentage is not achieved occupancy or bandwidth.

A fresh separate trace of the pinned llama.cpp **non-speculative single-request**
path now records actual `quantize_q8_1`, `quantize_mmq_q8_1`, and quantized
`mul_mat_vec_q`/`mul_mat_q` kernels. Therefore its internal activation packing is
not merely assumed from its Q8 weight label or a source file. The audit includes
startup/warmup and graph node tracing; it is **kernel-identity evidence**, not a
fresh throughput or per-request time-share comparison. Source-tree hashes are
retained as hints, not falsely presented as a recovered exact binary commit.

The first audit's observer searched only the profiler's main-thread children and
missed a child owned by another thread. A finite attempt reused the live server,
but intersected that observer's watchdog shutdown; its C8 client did not finish
and no complete trace was published. It is not scored as zero TPS or proof of an
inference/C8 failure. The corrected all-thread observer/C1 audit succeeds; failed
logs and responses are retained without repeated blind retries.

## Remaining work

1. Integrate an explicit new precision profile in native model projections and
   both TP devices, with bounded caller/model-owned quantization scratch. Preserve
   the original arithmetic path and checkpoint/state ownership contracts.
2. Run trained teacher-forced fixed CPU-reference gates and output/quality checks;
   verify scalar/batch state equivalence within the declared new profile, including
   EOS/failure/reset. No silent precision downgrade or reduced-quality win.
3. Screen actual 1/8 complete-request and single decode performance, with CPU,
   latency, VRAM and GPU telemetry. Promote only after correctness, then collect
   >=3 unprofiled repeats at 1/2/4/8 and separate full-engine profiling.
4. Qualify MTP with the new target arithmetic, state transactions and acceptance/
   replay denominators. Compare the competitor's supported controls without
   treating its failed MTP configurations as native wins.

The app goal remains **active**. This capsule does not satisfy the requested
>=5% external engine wins. No commit/push is performed.
