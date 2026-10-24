# Native Q8 Qwen-27B: llama.cpp comparison and optimization goal

App goal: **active**, created 2026-10-19. External-performance acceptance remains incomplete; no overall speed-superiority claim.
Updated: 2026-10-20 (Asia/Shanghai).

Latest T3 projection experiment: six research schedules pass byte/reference and
trained state gates, but the full-model single-repeat screen gains only 0.27% C1
and 1.14% C8 HTTP; ordinary batch rows256 is unchanged. It fails the predeclared
promotion gate. Model dispatch is restored, no new repeatable TPS is claimed,
and the qualified 42.83 single-decode /65.41 C8 ordinary-batch results remain.
See `benchmarks/v100-t3-projection/summary.md`. Next prioritize native batched MTP
verification with private-state correctness, not more token-tile sweeps.

Latest native MTP2 milestone: explicit prefix-state recovery removes recurrent
restore/replay target forwards and preserves exact target/draft committed bytes.
Three-repeat C1/2/4/8 scalar-MTP HTTP improves
**29.41/29.29/29.75/29.26 -> 33.88/33.36/33.48/33.81 TPS**. Single post-first
native decode improves **35.72 -> 42.83 TPS**, but still trails the strongest
tested pinned external MTP2 **49.80 TPS**, despite winning complete-request HTTP
33.88 vs 30.17. Ordinary native batching remains the stronger parallel path.
The first replay-based MTP2 regression is preserved, not promoted. Additional
prefix memory raises eight-slot primary NVML 22435 -> 23993 MiB (+1558); defaults
stay one proposal/replay, and the external goal remains **active**. See
`benchmarks/v100-native-mtp2/summary.md`. The historical sections below retain
their original scopes; older statements that MTP2 was absent describe those
baselines, not the current experimental extension.

Latest projection-budget tuning: an explicit 256-row native batch budget now
passes exact rows32 control logit/private-cache signatures. The kernel binary
is unchanged; this tunes an existing parameter, not a new algorithm. Three-repeat
C1/2/4/8 HTTP is **22.92/36.29/53.25/65.41 TPS**. C4 clears the pinned external
HTTP control by **5.96%**; C1/C2/C8 still trail. Decode is approximately flat,
including small regressions. Primary NVML peak grows by1722 MiB, so this remains
explicit opt-in, not a silent global default. MTP serving stays at32 rows and
its prior results are reused. Full objective remains active. See
`benchmarks/v100-prefill-budget/summary.md` for scopes and memory/latency tradeoffs.

Latest compact-GDN milestone: the residual target now uses DK-sized SM70
launches while preserving the original eight logical warp partials, output and
FP32 state bytes. Three-repeat C1/2/4/8 HTTP improves
**22.13/33.82/47.84/57.05 -> 22.96/35.81/52.05/63.18 TPS**. Ordinary single
post-first is **26.30 TPS**, MTP1 **35.69 TPS** (HTTP 29.38). C4 is 3.57% above
the pinned external HTTP control, not the required 5%; C1/C2/C8 still trail.
Goal remains active. Fresh trained state gates and 1644 V100 tests pass; unchanged
FP16/draft references are reused. See `benchmarks/v100-gdn-compact/summary.md`.

Latest measured optimization: SM70 residual same-input projection groups now
share transport fences/output copies and fuse complete row dots. Fresh candidate
logit/private-state/MTP gates remain byte-exact to the pinned qualified profile.
Three-repeat native C1/2/4/8 HTTP improves **20.23/31.55/45.35/54.69 ->
22.11/33.80/47.80/57.04 TPS**. Single ordinary post-first improves
**23.08 -> 25.52 TPS**; MTP1 post-first **31.73 -> 34.40 TPS**, HTTP
26.17 -> 28.08. All external HTTP cells still lose; best tested external MTP2
remains 30.17 HTTP / 49.80 reported post-first TPS. No completion claim.
See `benchmarks/v100-q8-projection-fusion/summary.md` for source ablations,
reference reuse, exact scopes, CPU/latency/telemetry and separate profiling.
The following sections retain historical measurements rather than replacing
stronger prior controls with convenient weaker runs.

Latest checkpoint: native greedy target execution, explicit TP/PP and experimental
one-proposal MTP now run on the original-derived Q8 checkpoint. The two-V100 MTP
campaign passes represented-weight reference and exact ordinary-target state/output
gates. Single-request TP decode improves 18.94 -> 24.55 TPS, but PP C2/4/8 regresses;
MTP remains opt-in. See `benchmarks/v100-native-mtp/summary.md`. The historical
milestones below retain their original scopes; older statements that MTP was
unfinished describe those baselines, not the current experimental wrapper.
The next native target cohort-batching milestone (ABI 16) now passes exact
scalar output/cache checks across eight heterogeneous real-model requests.
Three-repeat TP C8 complete-cohort output improves **15.44 -> 28.84 TPS**, but
single-request decode remains approximately 18.9 TPS and the new llama.cpp
non-speculative C8 control reaches 63.90 TPS. This is progress, not goal completion.
See `benchmarks/v100-native-batch/summary.md`. Production continuous batching/
paging, broad trained quality, sampling, longer MTP chains and CUDA graphs remain
incomplete. The first native batch path does not support PP or MTP wrappers.

The subsequent projection study rejects an FP16-kernel T8 experiment and a peer CPU submitter after
actual engine regressions/no improvement. A vector group32 integer-dot prototype
with residual activation compensation passes independent operator gates and
improves projection medians by about 1.40x/1.60x at one/eight rows. This is an
**explicit new arithmetic profile**; the operator study alone did not qualify
model quality or serving speed. The default FP16-activation engine remains
unchanged. A fresh competitor trace
now confirms actual Q8_1 activation-packing kernels. See
`benchmarks/v100-q8-dp4a/summary.md`.

The opt-in `q8-residual` profile is now integrated into resident FP16 native
single/TP model projections and cohort batching; MTP uses it only for the target,
keeping the draft's original FP16 arithmetic. Additive ABI-16 exports preserve
default-library compatibility. Explicit workspace ownership/accounting and
unchanged model locking protect profile scratch; no Python model/token loop is
introduced. Fixed prior CPU thresholds, 98 additional teacher-forced steps,
four original greedy trajectories, eight mixed scalar/batch cache comparisons
and natural/forced MTP transactions pass. New-profile states are exact within
that profile, not bit-identical to original FP16. Original-BF16/corpus quality
and SM89 qualification remain outstanding.

Three-repeat complete-cohort HTTP TPS at 1/2/4/8 improves
**15.19/21.56/26.59/28.81 -> 18.05/26.59/33.28/37.16**. Single-request MTP
post-first decode improves **24.55 -> 29.27 TPS**; complete-request HTTP improves
18.47 -> 22.65 TPS. Fresh pinned llama.cpp controls remain faster: C8 non-spec
66.83 TPS, C1 MTP1 whole-request 29.09 TPS and reported post-first 46.64 TPS.
These are practical engine comparisons with declared arithmetic differences,
not identical-math parity or a speed-superiority claim. See
`benchmarks/v100-native-residual/summary.md`. Final suites: V100 1020 passed /
440 skipped, local 760 / 700. Goal remains **active**.

Explicit scoped input sharing now removes duplicate QKV/z, Q/K/V and gate/up
packing and TP input transfers, without retaining a pointer-based cache across
groups. Both output-reuse event fences remain. Within each original/residual
arithmetic profile, trained old/new logits, outputs and private caches are
byte-exact. Three-repeat residual C1/2/4/8 HTTP TPS improves
**18.09/26.82/33.71/37.48 -> 18.97/27.66/34.35/38.62**; MTP1 post-first C1
29.27 -> 30.42 TPS. Original FP16-activation C1/C8 also improves, with no new
precision substitution. Fresh llama.cpp non-spec C8 reaches 62.80 TPS and MTP1
C1 reported post-first 46.77 TPS; the prior stronger C8 66.83 result remains
preserved, not superseded by a conveniently lower control. Every required
external win remains absent. See `benchmarks/v100-shared-inputs/summary.md`.

Benchmark limits are 32 projection rows versus 256 context tokens per slot.
Residual projection scratch remains 1,253,376 bytes/device. The new capsule's
erratum corrects the earlier frozen summary's implication that scratch scaled
to 256 rows; no source, timing or raw evidence in that capsule is modified.

The standalone group32-scaled FP16 Tensor Core prototype is now implemented:
unpack
integer weight codes exactly to FP16, multiply against original FP16 activations,
then apply original per-row/group scales to FP32 group accumulators. Do not
round dequantized weights to FP16 before multiplication. Changed reduction order
would require an explicit profile and fresh independent operator/trained gates.
Independent operator/tail/subnormal gates pass, but the T1/T2/T8 variants are
slower than residual DP4A; no Tensor Core model integration or quality claim is
made. Its compiled HMMA instructions and separately captured kernel dispatches
are retained, not confused with achieved Tensor Core throughput.
CUDA 12.9 documents Volta-compatible FP16 WMMA, alignment/stride requirements and
unspecified fragment-element mapping; a portable scale epilogue must not assume
fragment row indices ([NVIDIA programming guide](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html#warp-matrix-functions)).
The distinct integer-dot T8 specialization has new positive evidence. A guarded
V100 path retains T4 for small row grids/cache-resident code planes, preserves
SM89 and T1/T2 dispatch, and keeps original/residual profile logits, outputs and
private state byte-exact. Three-repeat C1/2/4/8 complete-cohort HTTP improves
**19.05/27.72/34.34/38.60 -> 19.41/28.58/35.57/43.18 TPS**. C1/C2/C4 gains
come from prefill; single-request MTP post-first remains approximately 30.4 TPS.
Fresh llama.cpp non-spec C8 is **67.29 TPS**, MTP1 C1 post-first 46.68 TPS.
All required external wins remain absent. V100 suite 1206 / 440 skips, local
761 / 885. See `benchmarks/v100-q8-wmma/summary.md`.

The TP primary-output staging hypothesis has now been tested and **rejected**.
Direct pitched stores pass exact trained scalar/batch/MTP state gates and the
prototype V100 suite (1280 passed / 440 skipped), but the C8 screen remains
43.186 -> 43.188 HTTP TPS, with no useful C1 gain. Separate traces confirm
5208 primary D2D copies disappear, yet dot time and elapsed time stay flat;
waiting moves into event/synchronize/launch APIs. The earlier 499 ms summed
copy-API duration was not removable elapsed time. Production native source is
restored exactly to the qualified T8 path; the prototype is archived, not enabled.

Three fresh repeats at llama.cpp MTP depths 2/3/4 find depth 2 best among
tested 1–4: **49.80 reported post-first TPS**, versus 46.68 at depth 1;
depths 3/4 fall to 43.13/37.37 as acceptance declines. Complete-request HTTP
at depth 2 is 30.17 TPS; Venturi's retained MTP1 HTTP/decode is 23.99/30.42.
These are separate timing scopes, not a new native MTP2 implementation or an
exhaustive best-configuration claim. See `benchmarks/v100-strided-output/summary.md`.

Next prioritize actual Q8 projection/GEMV work and correctly transactional
native multi-proposal MTP. Changed arithmetic needs separate qualification;
competitor acceptance does not predict native acceptance. Do not rerun the
unchanged rejected staging, FP16 T8, peer-thread or naive WMMA designs.

The next shared-activation study now retains **T2 only**: SM70 residual
two-token projections stage unchanged codes/scales in shared memory when
the activation plane fits 48 KiB, using 16 row warps per CTA. T1 staging is
slower and rejected; T1/T4/T8, SM89 and default FP16 dispatch stay unchanged.
Exact trained scalar/batch/MTP state gates pass. Three-repeat C2 HTTP improves
**28.55 -> 29.63 TPS (+3.77%)**; C1 MTP1 post-first **30.42 -> 31.73 TPS
(+4.31%)**, HTTP 23.99 -> 24.80. Other non-spec cells are roughly flat with
small measured regressions below 0.2%; every required external cell still loses.
Final V100 tests: 1256 passed / 440 skipped; local: 762 / 934. See
`benchmarks/v100-q8-shared-activation/summary.md` for complete scopes, unchanged
precision/reference gates, profiling and reused stronger external controls.

Next test K-tiled staging for T4/T8 and scoped multi-output projection/fence
fusion, rather than rerunning the slower full-T1 staging. An exact-reduction
coalesced-load design and native MTP2 remain hypotheses, not implemented wins;
the notes in the new capsule retain masks, barriers and transaction pitfalls.

K-tiled staging is now implemented and retained on SM70 residual T>=3, with
the original quantizer, FMA/reduction order and output/cache bytes preserved.
Three-repeat C1/2/4/8 HTTP improves **19.39/29.63/35.55/43.17 ->
20.19/31.48/45.21/54.72 TPS**. C4/C8 gains are 27.19%/26.75%; C1/C2 gains
are prefill only. Native TTFT falls about 25–26%; ordinary single decode and
MTP1 post-first do not improve (MTP 31.85 -> 31.74 TPS, a small measured
regression). MTP whole-request HTTP improves 24.97 -> 26.22 TPS from prefill.
Every required external cell remains slower than the stronger pinned controls.
Final suite: V100 1438 passed / 440 skipped, local 763 / 1115.
See `benchmarks/v100-q8-tiled-staging/summary.md` for the complete matrix,
explicit complete-signature reference reuse, failed launch recovery and separate
timelines/resources. Default FP16, T1/T2 and SM89 are unchanged by this milestone.

Next prioritize remaining synchronization/transport and single-request work:
exact scoped projection fusion, measured communication scheduling, and then
more complete FFN/head tensor partitioning. A column-sharded FFN down projection
must retain FP32 partials until reduction and disclose its changed reduction
profile with unchanged quality thresholds; it is not implemented yet. Native
MTP2 is likewise not implemented. Do not treat the new prefill gains as proof
of single-request decode acceleration or external-goal completion.

## Revised acceptance target: beat llama.cpp on two V100s

User-directed change, 2026-10-19: the next objective is a **matched llama.cpp
comparison and optimization**, not more isolated Venturi-only microbenchmarks.

1. Establish a reproducible baseline using a pinned llama.cpp binary/source and
   the **same official Qwen3.5-27B revision** as Venturi. Export its GGUF from the
   verified original checkpoint; do not compare the old uncensored Qwen3.8 file
   as if it were the same model. Record Q8 encoding/rounding and small-parameter
   differences explicitly rather than calling both representations byte-identical.
2. Run on the same two V100s, original-source Q8 weight class, FP16 KV,
   greedy policy, formatted prompt token IDs, fixed output work and context budget.
   Test **1, 2, 4 and 8 requests**, first with speculation off, then one-proposal
   MTP on both sides. Separately report the best supported llama.cpp MTP depth;
   do not handicap the competitor or quietly label unequal draft depth matched.
   Keep CUDA graphs/optimized native kernels enabled in llama.cpp when supported.
   Record native FP16 activation storage versus llama.cpp internal activation
   quantization/reduction paths: equal Q8 weight labels do not imply identical
   compute precision. Separate practical engine-default comparison from strict
   represented-weight/kernel parity; never describe the former as exact-math parity.
3. Primary metrics: single-request post-first-token TPS and concurrent complete-
   cohort output TPS (total emitted outputs / global cohort wall). Also retain
   native/server timing versus client end-to-end timing, TTFT, request completion
   distribution, CPU core-seconds, VRAM, GPU telemetry and acceptance/replays.
   Direct native-call versus HTTP timings must be labeled as different scopes.
4. Correctness before speed: identical tokenization/input positions; finite,
   aligned teacher-forced output checks across engines; preserve Q8/upstream
   numerical differences. Every Venturi optimization must retain the established
   native output/state/reference gates. Test rejection/EOS/failure boundaries for
   speculative changes. Never trade away correctness or precision for a win.
5. Optimize **the largest measured end-to-end gap**, one hypothesis at a time.
   Use a short correctness gate and 1/8-request screening first; rerun unchanged
   full suites only for promotion. Follow with 1/2/4/8 matched unprofiled repeated
   trials and separate GPU profiling for the winning change. Preserve regressions.
6. Completion requires repeatable **>=5% throughput improvement over the measured
   llama.cpp baseline** for single-request decoding and the selected parallel
   workload matrix, with at least three timed repeats per promoted cell and all
   declared correctness gates. Report per-cell wins/losses; a microbenchmark or
   internal Venturi speedup alone cannot complete this goal. No universal-speed
   promise, changed model, Q4 substitution or invented achieved-occupancy data.

Immediate priorities from the initial same-source loopback API screen:

- A valid llama.cpp layer-split/MTP1 configuration reaches about 62.4 aggregate
  complete-request TPS at C8; current native PP without MTP reaches about 24.2.
  These are one-repeat screening figures, not promoted results or a universal
  same-arithmetic comparison. See `benchmarks/v100-llama-comparison/summary.md`.
- Parallel lane: native **cohort prefill/decode** and active EOS/budget compaction
  are implemented, with private GDN/convolution/KV/cursors and exact scalar state
  parity. Shared Q8 projection/GEMM work still needs substantial acceleration;
  the existing tile only reuses each weight across four token rows. Preserve the
  unchanged Q8/FP16 arithmetic for exact-order candidates; separately qualify and
  label any new arithmetic profile. Then consider continuous admission, PP/MTP
  batching and paging. Do not mistake the bounded HTTP admission window for
  continuous native batching. The 5 ms screen fragmented C8 into 6+2; the repeated
  full-cohort benchmark explicitly uses 25 ms, included in client wall time.
- Single-request lane: profile and reduce Q8 projection/verification and launch/
  transfer cost against the actual llama.cpp baseline, including its CUDA graph
  reuse. Investigate arithmetic/activation-quantization differences explicitly;
  do not assume paging or a higher occupancy percentage fixes this short context.
- A staged exact-order T2/two-output-row kernel gives only about 6.5% median
  projection speedup and regresses the small KV projection. It is not an engine
  win and is not promoted without an end-to-end test. Avoid spending the next
  cycle on marginal occupancy tuning while the parallel gap is roughly 2.6x.

A new app goal was successfully created and returned **active** on 2026-10-19,
with the revised llama.cpp comparison and optimization objective. Earlier frozen
screening evidence retains the goal-control limitation that existed at capture
time; it is historical, not the current app goal status.

## Source and capacity

- Original source: [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B), official revision
  `fc05daec18b0a78c049392ed2e771dde82bdf654`.
- The official endpoint is unreachable from WZU_Server; an HTTPS HF mirror
  returned that same official revision. Acquisition explicitly records the
  endpoint, pins the revision, and checks sizes/LFS SHA256 plus actual file hashes.
- A dedicated CPU-only acquisition environment/job is under
  `/home/wzu/codex-build/Venturi/source-download-job/`; source target is
  `/home/wzu/codex-build/Venturi/models/qwen3.5-27b-bf16/` (the server resolves this
  under `/home/data/wangyue/repos/codex-build/Venturi/`). No GPU service was stopped.
- Selected original assets total 55,576,089,619 bytes. Partial files are not a
  completed or verified checkpoint. Read `venturi-source.json` and job status.
- Existing NInfer artifact is mixed Q4/Q8. It is not upgraded to original-quality
  Q8. Its small floating parameters may be used to qualify individual operators.
- Promoting its stored quantized text matrices to Q8, preserving small floating
  parameters, requires 28,601,606,144 bytes (~26.64 GiB), before KV/state/workspace.
  A fully resident Q8 text model does **not** fit a single 24 GiB 4090D.
- A free 32 GiB V100, multiple GPUs, or an explicitly measured offload path is
  required. V100s are currently occupied by other services; do not stop them
  without explicit authorization. SM89 test results do not qualify SM70.

## Work completed in the bootstrap milestone

- Explicit bounded-row CPU Q8 quantization from floating source, with stored
  FP16 scale rounding/nearest-even code semantics, zero-group policy, padding,
  nonfinite/overflow rejection and source immutability tests.
- Generic Q8 resident projections compose with the reusable MLP/decoder layer;
  token-shape plans and native embedding lookup share the encoded weight planes.
- Native causal depthwise convolution + SiLU with chronological raw-input history.
  Cache stores K-1 previous samples; history can update in-place safely. It is a
  single sequence, not a batch of independent requests disguised as time.
- Native GDN gate transforms with explicit beta rounding semantics, staged gated
  RMSNorm, and zero-centered decoder norm preparation. Native run() has no
  Tensor numerical fallback/implicit allocation; capture is explicitly tested.
- Strict bounded small-parameter bindings for declared GDN layers. Payload reads
  happen only after all selected binding metadata passes shape/format checks.
- Blocks C ABI 11: rebuild explicitly; earlier ABI-4/5/6/7/8/9/10 benchmark artifacts retain
  their historical source/build capsules.
- Per-operator independent FP64 checks, chunk/state/alias guards, native graph
  tests and matched unprofiled reference comparisons on the 4090D.

## Ordered remaining milestones

1. **Storage complete:** original BF16 acquisition and separate Q8 artifact,
   with independent hash/floating-tensor/sampled-row verification. Original source
   and existing models remain untouched. This is not trained quality acceptance.
2. Bind all original-source text projections/parameters with exact layout,
   quantization and norm conventions; stream tensor conversion/loading without
   retaining both a complete BF16 and a complete Q8 GPU model.
3. Compose the complete native GDN block: projections, convolution/history,
   correct Q/K/V layout, gates, recurrence and gated output projection/residual.
4. **Fixture integration implemented:** Qwen full-attention Q/K norm, per-head
   query/output gate, default text-only partial/mRoPE, and an eager hybrid
   model/session adapter. Synthetic prompt chunking/continuation passes. Native
   production runtime and trained-checkpoint execution remain separate work.
5. Qualify complete trained-checkpoint logits/state and generation/quality against
   an independent reference. Synthetic layers do not satisfy this milestone.
6. Native single-request execution/sampling and reusable whole-model CUDA graphs.
   A reusable ABI-11 native finite-logit greedy selector now covers the token
   choice operator only; the Python control/forward/staging loop and whole-model
   graphs remain unfinished. It does not supply stochastic sampling.
7. Batch-aware GDN/convolution state and attention/KV ownership; continuous
   batching and paged KV. Eight independent requests must never share one
   recurrent sequence state or attend across request boundaries.
8. SM70/FP16 Q8/GDN kernels and hybrid-fixture support are now separately
   built/tested on V100. Complete trained-model FP16 quality and performance
   qualification remain open. This is a typed port with architecture guards,
   not merely a `-gencode` change or a bypassed SM89 check.
9. Matched full-model single/concurrent comparisons and gap-driven optimization.

## Measured bottleneck to retain

The initial direct Q8 projection is a correctness baseline, not a fast prefill
GEMM. On a synthetic `[17408,5120]` Q8 projection, native graph-wrapper wall time
was ~168 us at T=1 vs ~381 us for an explicitly prepared FP32 Torch reference;
at T=128 it was **~20.53 ms vs ~0.728 ms**. These are primitive measurements, not
model speedups. The direct kernel reloads weights independently for token rows;
a tiled/reusing GEMM route is needed. Do not hide that prefill regression.

A bounded native Q8->FP32 decode workspace plus cuBLAS FP32 GEMM is a conservative
next baseline: preserves the represented weights, avoids whole-model dense
materialization and can reuse scratch sequentially. Any faster Tensor Core
approximation/decomposition needs explicit numerical and quality qualification;
do not silently BF16-round decoded Q8 weights or loosen the current error gates.

Native convolution/gate/norm helpers are faster than their explicit Torch
references in the measured cases, but this does not establish an engine win.
The benchmark report records CPU/event/wall scope and NVML diagnostics separately.
Hardware counters remain restricted (`RmProfilingAdminOnly=1`); achieved occupancy
and DRAM/SM/cache/stall metrics were not measured. Do not relabel NVML busy data.


## Packed recurrence and mixer-core integration

The native recurrence now consumes `[T,Q-flat|K-flat|V-flat]` directly with the
same q/k normalization and FP32 state contract. A prepared projected-input mixer
composes convolution/history, gate transforms, recurrence and gated RMSNorm.
It validates all input, scratch, constant and output intervals before launching
any state-changing operation. Graph replay and request-independent mutable
buffers are tested; no implicit unpack, numerical fallback or forward allocation
is introduced. This is not yet projections, residual/MLP or a whole Qwen layer.

The packed path saves 2.5 MiB of unpack scratch at T=128 for Qwen geometry. A
matched native unpack-versus-packed ABBA measurement was ~1.847 vs ~1.841 ms;
T=1 was ~22.43 vs ~22.17 us. These differences are too small to establish a
meaningful latency win. Preserve this result: eliminating copies does not remove
the sequential recurrence bottleneck. No full-model TPS claim follows.


## Original-source binding/export implementation

`QwenSourcePlan` covers all 851 text tensors in the pinned original index and
explicitly excludes 348 vision/MTP tensors. Planned payload is exactly
28,601,618,432 bytes: 402 Q8 matrices, 305 BF16 parameters and 144 FP32
constants. Main projections and vocabulary are Q8; small a/b projection matrices
remain BF16; the original GDN gated gamma remains FP32. All source roles, shapes, declared dtypes and unsupported extra text
weights are preflighted before payload conversion. Tied embeddings require one
physical candidate; conflicting duplicate heads are rejected.

`export_qwen_q8` reads source matrices through bounded safetensors row slices,
quantizes with stored FP16 scales and writes encoded planes directly to a new
v3 storage artifact. It never loads a whole BF16/Q8 model into CPU/GPU memory.
Raw zero-centered norms and per-head Q/gate row order are retained; only the
convolution transforms [channels,1,kernel] -> [kernel,channels]. A_log/dt_bias are
explicitly widened to FP32 when needed. The source config and every source file
hash are revalidated, and file changes during conversion abort publication.
A complete acquisition manifest is required by default. Explicit unverified
local-source opt-in is labeled as such and is not an original-quality claim.

Publication is atomic and no-clobber, with private temporary files cleaned up on
ordinary failure. Original source and existing mixed-Q4 artifacts are untouched.
`QwenQ8Weights` preflights the complete runtime role set before reads and exposes
one encoded projection or bounded floating parameter at a time for native upload.
The reused v3 container is storage only: this does not claim compatibility with
an external NInfer model forward or complete native Qwen generation.

Once the original acquisition reports complete, run the CPU-only converter on
the V100 host using the existing owned Torch environment (no package changes):

```bash
cd /home/wzu/codex-build/Venturi/q8-export-src
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src \
  /home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python \
  tools/export_qwen_q8.py \
  /home/wzu/codex-build/Venturi/models/qwen3.5-27b-bf16 \
  /home/wzu/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer \
  --report /home/wzu/codex-build/Venturi/q8-export-validation/export-report.json
```

Do not run conversion against an incomplete source or overwrite an existing
artifact. Synthetic fixture success and complete name coverage do not replace
actual source conversion, trained-logit/state/quality checks or full-model timing.
The owned V100 environment has Torch 2.5.1+cu121 with sm_70 in its compiled
architecture list; that is toolchain evidence only, not a qualified native SM70
backend or a reason to bypass the current SM89 guards.


## Complete GDN decoder-layer composition

`ResidentQwenGDNLayer.from_q8` uploads one encoded projection at a time and reuses
resident input norm/conv/gate/gated-norm constants. Prepared execution composes
input norm, QKV/z/a/b projection, the packed recurrent mixer, output projection,
residual and the zero-centered post-norm/SwiGLU/residual MLP. Native execution
uses owned kernels/cuBLAS; it does not invoke external model forward or numerical
Torch fallbacks. The host dispatcher is still Python; a native whole-model
execution/scheduling runtime remains a later milestone.

Shape plans share read-only weights. `QwenGDNWorkspace` owns per-lane activations,
pre-created flat/head views and one reusable linear-scratch arena. Metadata,
aliases and all handles are checked for the entire layer before projection or
state-changing launches. Exact history/history_out and state/state_out pairs
are allowed; other writable aliases are rejected. `zero_state` allocates explicit
per-request history/recurrent matrices outside capture. This raw layer API has
no hidden cursor/commit, rollback or batching scheduler; state ownership/reset,
serialization and invalidation after asynchronous failure remain explicit.

Qualification uses independently framed original-floating fixtures, independent
Q8 source quantization/CPU FP64 formulas, actual-input per-operator error gates,
and exact manual-composition parity. Native miniature cases cover T=1/9,
value/key head ratios 2/3, chronological continuation in chunks 1/2/6, independent
streams, closed-handle preflight and CUDA capture with allocations, new tensor
views and numerical fallbacks prohibited. These synthetic dimensions do not
qualify complete trained Qwen generation or all production shapes.

A retained counterexample is important: one early FP32 Q8 reduction rounds to a
neighboring BF16 value (max QKV difference 0.00048828125); downstream cancellation
produces a 0.00390625 whole-layer difference from a staged FP64 cascade. Existing
per-Op bounds pass on actual stage inputs; no operator tolerance was loosened.
The whole-chain BF16/FP64 difference remains a diagnostic/quality limitation,
not a claimed numerical fix. The allocating Torch BF16 CPU reference also uses
shape-dependent BLAS and is not guaranteed bit-identical across chunks. Native
chunk equivalence is asserted only for the tested cases. Full trained logit,
state, generation and quality acceptance is still outstanding.

A one-shot CPU dependency worker (`tools/wait_export_qwen.py`) may wait for the
existing source job and then run the converter. It never restarts a downloader
or stops a service; an observer wait expiry is not treated as source termination.
Its lock prevents duplicate exporters. Inspect live process command lines and
actual output/report before trusting a saved status file. The worker state is
under `/home/wzu/codex-build/Venturi/q8-export-job/`; pipeline completion means
storage conversion only, never completion of this inference goal.


## Full-attention decoder layer and original-source precision correction

`ResidentQwenAttentionLayer` now composes zero-centered input/Q/K norms,
per-head query/gate layout, partial split-half RoPE, causal GQA, rounded sigmoid
output gate, output/residual and post-norm MLP/residual. Dense cache ownership,
visible-length commit, submission-failure poisoning, truncate/reset and fixed-
position capture share the existing cache lifecycle, rather than an external
engine. Cache-tail and arena/head views are prepared before execution; graph
runs allocate no tensor, create no view and invoke no numerical Torch fallback.
This is dense KV only, not paged attention or a production scheduler.

The Qwen RoPE helper explicitly supports default text-only equal-axis positions.
Frequency-axis mRoPE interleaving is not even/odd channel pairing. Norm casts,
activation-typed cos/sin, separately rounded rotary products and sigmoid rounding
follow the eager Qwen convention. CPU reference checks agree exactly with the
installed upstream Qwen source where available. Native fusion additionally has
an exposed norm-only stage, letting tests qualify norm against FP64 and require
exact fused/staged rotation parity without widening operator bounds. A retained
FP16 norm/RoPE cascade example demonstrates cancellation of early rounding.

All original source files are now downloaded and hash-verified. Full real-source
preflight exposed 48 FP32 GDN gated-norm vectors, which the legacy BF16 artifact
and initial schema did not preserve. Export stopped before payload publication.
The corrected source policy retains FP32 gamma and its promoted multiplication
(no extra weighted BF16 cast). Legacy activation-typed gamma remains an explicit
separate mode. The C ABI is now **8**; historical ABI-6/7 capsules stay unchanged.
Current original-derived text payload is **28,601,618,432 bytes** (~26.64 GiB):
402 Q8 matrices, 305 BF16 tensors and 144 FP32 tensors (including widened dt_bias).
The earlier 28,601,606,144-byte figure describes the legacy BF16-gamma estimate,
not the corrected original-derived artifact. Neither fits one 24 GiB GPU.

Corrected conversion was started only after confirming the prior converter and
observer were terminal and revalidating all 851 real text shapes/dtypes. Inspect
the live export/observer PIDs and actual artifact/report, not stale saved status.
The original source is untouched. Trained hybrid execution/quality, native
whole-model runtime, paging/batching, SM70 and matched engine comparisons remain
unfinished. The eager fixture adapter below does not satisfy those conditions.


## Published storage and eager hybrid session integration

The corrected export completed and published artifact
`3d635b87db004f42aefd6730e372e530` (file SHA256
`b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8`).
Independent post-publication checks pass for full source/artifact hashes, source
provenance/config, all 449 floating tensors and 1,206 sampled Q8 rows across all
402 matrices. This is storage verification only, not all-row recomputation or
trained-model logit/quality qualification. See the final reports in
`benchmarks/qwen-hybrid-model/conversion/`; earlier acquisition/export observations
in historical capsules remain unchanged.

`ResidentQwenModel` streams encoded uploads, rejects unsupported geometry and
insufficient memory before payload reads, preserves tied embedding storage and
connects both decoder layer types to vocabulary/final norm/head. `QwenSession`
uses per-request conv/recurrent/KV state, shared sequential per-kind workspaces,
bounded retained shape lanes and whole-call synchronized position commits. A
failed append poisons the request; resetting zeroes GDN history/state and drops
KV visibility. No unsafe GDN truncation is exposed. Numerical native execution
uses CUDA/cuBLAS, but dispatch is still synchronous Python; production native
runtime/sampling/graphs and batching/paging are not implemented by this adapter.

The full 4090D suite is 853 passed / 26 skipped; local 618 / 261, V100 CPU model/
source subset 42 / 9. Independent original-source FP64 whole-fixture logits,
exact native composition/chunking, stream/state ownership and late failure/reset
pass. These miniature cases do not qualify the complete trained 27B model.

Actual native weight tensor budgeting is 28,602,955,776 bytes, before request
state, allocator/context/workspace/logits/graphs. Weights plus one dense 256K
request require at least 45,936,769,024 bytes, excluding that overhead. A fully
resident Q8 model still cannot fit a single 24 GiB 4090D. Existing GPUs remain
occupied; no service was stopped. No timing phases ran for the new helper/
hybrid path and no speed claim follows. SM70/FP16, sufficiently free GPU memory,
trained quality and matched single/concurrent comparisons are still required.


## SM70/FP16 port and real-checkpoint layer checks

Blocks ABI **9** adds typed FP16/BF16 Q8 embedding/projection and GDN recurrence
entry points, plus a compiled architecture tag. Explicit `--architecture sm70`
builds with CUDA 12.9 on the V100 host; default `sm89` builds on the 4090D. The
backend requires an exact matching hardware/library target, and SM70 rejects
BF16 before payload upload. Direct C ABI typed calls also reject BF16 on SM70.
Ada Q4 remains SM89-only. There is no automatic precision fallback or compiler
selection. CUDA compilation was remote only; no Rust compilation ran.

Selecting FP16 explicitly changes activation/cache/output precision and converts
small BF16 a/b/conv parameters to FP16. Q8 codes/scales remain unchanged; norms/
gamma and GDN gates/recurrent state retain FP32. This port is **not** a claim that
the original BF16 model has identical FP16 quality. Existing BF16/SM89 and
explicit allocating Torch reference modes remain separate.

V100 checks cover independent represented-weight FP64 operator math, packed/
unpacked/chunked GDN, FP32 state, graph/no-allocation/no-fallback execution, tiny
hybrid composition, ownership and failure/reset. `tools/qualify_qwen_q8_layers.py`
then hashes the actual published artifact and qualifies selected original-derived
GDN/full-attention layers at T=1/8 against actual-input CPU FP64 stage oracles.
Those inputs are synthetic: this is trained-weight **layer** qualification, not
a complete trained-model forward or a generation/quality result. The tool reports
small-weight cast changes and retains the existing per-operator error criteria.

See `benchmarks/sm70-fp16/summary.md` for final build/test reports and scope.
Existing GPU services stayed running; no isolated timing or achieved occupancy
measurement was performed. Full residency still requires ~26.64 GiB for weights
alone, while the V100s remain occupied. Next full-model validation needs either
sufficient free VRAM or a deliberately separate explicit layer-streaming/offload
correctness path; it must not silently alter the resident API or become a
substitute for the required production-native single/concurrent speed target.

The real trained-weight FP16 checks exposed **small-parameter precision loss**:
layer 0 casts change 62 a, 110 b and 53 convolution weights, including one
convolution underflow to zero (max cast error 2.9802322387695312e-08). This is
recorded, not assumed harmless. The current FP16 casting-profile Op checks pass,
but **lossless small-parameter execution is still required** for checkpoint
fidelity: use FP32 storage/compute for these small BF16 constants instead of
rounding them to FP16. FP16 source/quality acceptance is not complete. Preserve
this counterexample and the original checkpoint unchanged. The final V100
selected suite is 42 passed / 3 skipped; the final SM89 suite 893 / 30; local
640 / 283. Final real-weight layer 0/3 T=1/8 stage checks pass under the declared
casting profile; no complete trained-model or timing qualification follows.


## Lossless small-parameter correction (ABI 10)

The above FP16 casting-profile issue is now corrected by the model loader, not
by modifying the checkpoint or relaxing numeric gates. `upload_fp32_linear` is
an explicit mixed-parameter API: FP32 stored weights with FP16/BF16 activations,
FP32 products/accumulation and one activation-output cast. Ordinary dense upload
remains strict. A typed causal-convolution path independently tracks activation
and parameter dtype. Qwen's FP16 a/b/conv parameters widen BF16 -> FP32 exactly;
Q8 planes are unchanged. Gamma/gates/recurrent state remain FP32. Other model
adapters can reuse these operators. This is not a fast large-GEMM route.

The independent layer qualifier now computes small-projection/convolution
oracles from original parameter values, not values first rounded to FP16. It
also compares actual uploaded parameter arrays with the artifact and rejects any
changed value. V100 layer 0 preserves all 532,480 a/b/conv values exactly, with
zero changed/underflow values; the earlier 225 rounded values and one underflow
are retained as historical counterexamples. Real layer 0/3 T=1/8 again pass 70
fixed per-Op checks and exact native manual-composition/state/cache parity.
These are synthetic-input trained-weight layer checks, not full trained logits,
generation or quality approval. See `benchmarks/fp32-small-parameters/summary.md`.

Final suites: local 653 passed / 299 skipped, SM89 920 / 32, selected V100 port/
tool/build CPU+native tests 61 / 12. CUDA graph tests prohibit numerical fallback,
allocation and weight casts. Tiny source parameters that disappear in FP16 must
remain resident and produce nonzero independently checked output. Existing
BF16/FP64 cascade counterexamples and all per-Op error criteria remain unchanged.

Native FP16 weight tensor budgeting is now **28,654,073,856 bytes**: 51,118,080
bytes above the unchanged BF16-profile budget, before CUDA context/allocator,
state/KV, workspace/logits/graphs. The metadata planner supports individual layers
and is checked against actual CPU/CUDA storage. Full residency still does not
fit 24 GiB or the current free V100 memory. Existing services remain running;
no timing/occupancy improvement is claimed. Next: full-model execution and
quality, using an explicit non-resident correctness mode if needed, followed by
the required native runtime/graphs, batching/paging and matched single/concurrent
comparisons. Such a mode must not silently replace the resident API or redefine
the original performance objective.


## Complete-model streaming execution (not a production-speed substitute)

A separate `StreamingQwenModel` / `StreamingQwenSession` now executes the complete
text decoder with the same native numerical operators while loading/releasing
one weight phase at a time. The resident API remains memory-guarded and never
falls back. Neutral per-request KV buffers do not retain layer weights: each
transient resident gets its own normal cache, imports an explicitly intended
prefix through `LayerKVCache.load_prefix`, and copies the new tail back before
release. Existing owner/stream/alias checks are not bypassed. GDN state/history
stay request-owned; whole-call position commits only after complete success.
Failure poisons the request until reset; no GDN suffix rollback is invented.

Only decode/max-prefill shape lanes are retained; workspaces share across layers
of the same kind. Optional CPU traces are explicit diagnostics. Source file
identity/size/mtime/ctime and metadata/config stability are checked throughout
streaming. Prefix staging is outside capture, into an empty cache only, and
submission failure poisons it. These are Python-controlled memory transfers and
synchronizations, not a production native scheduler, graphs or paging.

The real 64-layer Q8 FP16/V100 path has emitted token IDs [57590, 248046], decoded
as **Paris** followed by EOS using the pinned original tokenizer. The case has
26 prompt tokens and an explicit greedy override of the source stochastic
sampling defaults. Peak native-run PyTorch allocation was 1,518,406,656 bytes,
excluding CUDA context/driver/other-process allocations. It is a streaming
correctness result, not evidence that all weights fit there.

`tools/qualify_qwen_streaming.py` hashes the complete artifact and tokenizer,
records actual layers/state/logits, and optionally follows the same native-
generated input sequence in CPU Torch FP16 arithmetic with the same represented
weights and FP32 small parameters. Fixed logit agreement gates are established
in the tool before the run. This is an independent numerical backend but shared
adapter, not independent upstream architecture/original-BF16/corpus quality.
Host CPU greedy sampling is explicit, not an owned native sampler. Consult
`benchmarks/qwen-streaming/summary.md` and worker/report flags for scope. The CPU
reference completed both steps: top-1 tokens match, relative logit L2 is
0.0026253 / 0.0023074, centered cosine 0.99999436 / 0.99999593, and KL(ref->native)
5.33e-7 / 2.77e-7. Both predeclared numerical gates pass. Per-layer hidden and
GDN/KV state differences are retained in the report as diagnostics, not extra
post-hoc acceptance gates. Do not infer corpus quality from this two-token case.

Final regression counts are local 666 passed / 317 skipped, 4090D 947 / 36, and
selected V100 CPU+native streaming/port/tool/build tests 81 / 23. V100 BF16 native
execution is still rejected; the full real-model case uses FP16 activations/KV
and FP32 small parameters/state. Native phase execution peaked at 1.52 GB of
PyTorch allocation because weights were offloaded, not because full residency
fits. Qualification wall/CPU/RSS include hashing, transfers, CPU reference and
diagnostics; they are not native latency or throughput measurements.

Remaining original goal conditions are unchanged: broader trained quality,
production-native runtime/sampling/graphs, batch-aware GDN and paged attention,
and matched single/concurrent engine comparisons on the target hardware. An
explicit offload correctness mode does not redefine that performance objective.


## Native greedy policy operator (ABI 11)

`prepare_greedy` now selects one token per independent logit row on the GPU,
using a lowest-index tie break and explicit nonfinite-row status. It has an
allocation-free native run path, caller-owned output/workspace, graph replay and
independent-stream checks on SM89 and SM70. This does not turn independent rows
into served requests with GDN/KV state, or make the Python model loop native.

The full V100 streaming case with this selector again emitted Paris + EOS.
Logits, all 64 hidden tensors and 128 GDN/KV tensors per step are byte-identical
to the ABI-10 native/CPU-greedy run. The fresh same-input CPU-reference worker
completed both steps with both predeclared logit gates passing (relative L2
0.0026253 / 0.0023074; centered cosine above 0.999994). This is regression and
numerical smoke evidence, not broad quality acceptance. Final suites: local
697 passed / 356 skipped, SM89 1011 / 42, selected V100 CPU+native 130 / 40.

Matched selector-only tests compare the full-logit CPU path, allocating Torch
GPU selection and the owned native selector. For one FP32 248,320-value row,
observed median host-visible latencies were 954 / 390 / 146 us on the occupied
4090D and 1282 / 195 / 66 us on V100. They include host result consumption and
are not model TPS/TTFT, parallel-serving or external-engine speed results.
Separate SM89 tracing attributes the two kernels and host/copy gaps without
substituting profiled times for unprofiled benchmarks. Other GPU services were
not stopped; hardware counters remain unavailable under the recorded policy.
See `benchmarks/native-greedy/summary.md` for all waves, CPU times, telemetry,
source/build hashes, limitations and the original remaining goal conditions.

## Whole-native runtime and V100 resident Q8 tiling (ABI 12)

The C++ runtime now owns forward execution, GDN/KV/history, positions and the
greedy generation loop. Python prepares metadata/small constants at load;
native code owns copies and mapped Q8 planes. Both explicit streaming and
resident modes passed the real 64-layer, two-step Paris + EOS qualification:
unchanged fixed CPU-reference gates and byte-identical previous-native logits,
hidden values and GDN/KV snapshots. Generation itself passes with Torch model
math forbidden. The explicit `qwen-native` CLI also passed on V100. This
completes the previously missing native control loop, not batching or graphs.

After the user authorized freeing V100 memory, the large llama-server Docker
container was stopped (not deleted, restart policy unchanged). Smaller TTS and
RWKV services remain. The real Q8 model now fits resident on the selected V100;
it still does not fit a single 24 GB GPU. Residency is explicit and never a
silent format/precision fallback.

Generic SIMT Q8 tiles preserve per-output FP32 FMA/reduction ordering and the
encoded weights. Six configurations were tested. SM70 defaults to token/row
tiles 4/4; SM89 preserves 1/1 pending separate qualification. In matched native
resident baseline/tiled/tiled/baseline phases, decode improved 9.4853 -> 14.9420
tokens/s; 26-token prompt TTFT improved 2.37990 -> 0.90326 s; one-call prompt +
32-token wall time improved 5.64253 -> 2.97144 s. CPU core-seconds dropped
5.64511 -> 2.97273 per complete request. EOS stopping was disabled for fixed
work; a separate normal story prefix confirmed matching 32-token outputs with
no EOS. Do not compare this decode TPS directly to the old two-token,
offloaded, prefill-inclusive 0.11 output/request-wall rate.

Separate Nsight attribution still places approximately 89.7% of optimized
decode kernel time in Q8 projections. Sampled GPU busy is around 99% in both
cases; that is not achieved occupancy. Hardware counters remain unavailable
under the unchanged driver policy. Local tests: 708 passed / 430 skipped;
selected V100: 157 / 66, plus original-byte/tail checks for all six tile variants.
See `benchmarks/v100-q8-optimization/summary.md` for raw timing, CPU/VRAM/GPU
accounting, source/build provenance, static resources and preserved regressions.

Original-BF16/corpus/upstream quality is still unqualified. Native batching,
paged KV, long-context performance, whole-model graphs and matched external
single/concurrent comparisons on the 4090D remain required. No universal speed
claim or completion of the larger goal follows from this V100 milestone.

## Two-V100 native row shards (ABI 13; no MTP yet)

The user confirmed the external 50 TPS observation used **two V100s and MTP**.
That service's checkpoint was Qwen3.8-27B-Uncensored-Q8_0, while the independent
native artifact is original-derived Qwen3.5-27B Q8 group-32. Consequently the
observation establishes a target, not a same-checkpoint benchmark or evidence
of a native performance win.

The native runtime now supports explicit two-device resident row sharding:
complete dot products on each device, exact output-row gathering, primary-owned
GDN/KV state, native cross-device event ordering, and no Python numerical path.
Same-model requests remain serialized. True request batching/paging and whole-
model graphs remain separate unfinished work.

Direct peer-copy capability reporting was insufficient on the tested host:
one direction returned incorrect bytes. The first prototype's failed outputs
and aborted worker are retained, not counted as valid performance data. ABI 13
requires a real bidirectional copy probe for direct transport and fails fast.
Explicit portable-pinned-host transport passed fixture tests and full trained-
model logits/hidden/GDN/KV byte comparisons against the qualified native path,
with unchanged fixed CPU gates. Host transport does not require driver changes
and does not move numerical computation to the CPU.

Experimental one-warp-per-row and packed-byte Q8 decode variants also passed
exact-byte tests but regressed most projection shapes. They are not defaulted:
`--q8-decode-mode 0` remains the promoted SIMT kernel profile. The packed variant's
isolated large-head improvement does not establish an end-to-end speed win.

Matched four-trial single/two-GPU runs measure 14.9565 -> 17.1169 decode TPS,
0.90099 -> 0.56229 s TTFT and 2.96886 -> 2.36732 s complete 32-token request
wall time. The gain is real but far short of 50 TPS. A separate profiler shows
substantial new transfer/dispatch gaps, not a twofold scaling win.

Raw timing, CPU time, both devices' telemetry, separate profiler attribution,
regressions and source/build hashes are in `benchmarks/v100-native-tp/`.
The original HF checkpoint has 15 MTP tensors, 424,699,392 parameters and
849,398,784 stored BF16 bytes. The current text artifact explicitly excludes
these tensors; neither exporting them nor finding their names implements MTP.
Next work must bind and qualify the draft head, implement native target
verification with GDN/history rollback, test acceptance and exact greedy output,
and measure complete request wall time, not draft-only TPS. See
`docs/native-mtp-plan.md` for the acceptance conditions.

The larger goal is unfinished: original-BF16/corpus/upstream quality, native
continuous batching/paged KV, long context, graphs, and matched same-checkpoint
external single/concurrent comparisons on the target hardware remain open.

## Native pipeline execution (ABI 14)

Native PP now partitions resident Q8 matrices/constants and per-request GDN,
convolution and dense KV state by contiguous layer ranges. Two stage-private
streams/locks allow distinct requests to overlap across GPUs. Transfers use
request-owned portable pinned host storage; direct peer access is neither
required nor attempted. The final stage computes its output head and greedy
selection before releasing its stage lock. Generation is one C++ call per request,
not a Python token/layer loop. This is not continuous batching, paging or MTP.

Fixture checks cover tied/untied heads, three layer splits, chunk/continuation
trace/state byte parity, native generation with Torch math forbidden, failures
in either stage, reset, lifetime, EOS, source guards and independent requests at
1/2/4/8-way concurrency. Real-model qualification compares all trained logits,
hidden/GDN/history/KV bytes with the pinned native/CPU reference before timing.
The initial scheduler released the final stage before greedy selection, creating
an avoidable dependency/lock gap. Its data are preserved separately; the final
atomic-selection scheduler repeats correctness qualification and matched timing.

The benchmark distinguishes the current whole-request-serialized TP baseline
from native overlapping PP stages. Aggregate TPS uses total emitted tokens over
cohort wall time, never summed request latencies. Separate first-token,
continuation and one-call full-request phases expose queueing and throughput.
See `benchmarks/v100-native-pp/summary.md` for the final figures and limitations.
Original-BF16/corpus/upstream quality, continuous batching/paged KV, long context,
MTP, graphs and matched external-engine comparisons remain unfinished.

The final same-library two-V100 comparison measures aggregate decode TP/PP TPS
of 17.17/14.90 (one request), 17.18/28.13 (two), 17.17/28.31 (four) and
17.18/28.24 (eight). At eight requests whole-cohort wall improves
18.314 -> 12.241 s, but mean individual completion increases 10.319 -> 12.118 s
and process CPU increases 18.380 -> 22.992 core-s. This is a throughput and
worst-observed-completion improvement, not a universal latency/CPU win. The
final full trained case is byte-identical to the pinned native reference with
unchanged fixed CPU gates; final V100 suite 789 passed / 426 skipped, local
721 / 494. Separate traces confirm stage overlap and reduced transfers. One
instrumented TP eight-request capture stalled and is retained as incomplete;
the documented device-event-tracing-disabled retry completed correctly.

## V100 CUDA occupancy and compiler launch bounds

Read-only CUDA driver occupancy queries on the exact ABI-14 binary establish
theoretical residency, not achieved hardware occupancy. Original decode uses
64 registers/thread, 256 threads/block and four resident blocks/SM (50%);
prefill uses 44 registers and five resident blocks (62.5%). The existing native
traces attribute about 90% of summed decode kernel time to Q8 projections.
The unchanged `RmProfilingAdminOnly: 1` policy still prevents achieved SM,
compute/DRAM/cache/stall counters; no repeated counter campaign or driver
permission changes were made.

Compiler launch-bound experiments expose a counterexample to maximizing
occupancy: forcing six/eight decode blocks raises theoretical residency to
75%/100% but introduces spills and slows real checkpoint projections. Those
variants are retained, not defaulted. The promoted SM70-only token-aware bound
uses six blocks for the multi-token specialization, one for decode, while
keeping the actual 256-thread launch and exact reduction/FMA order. Compiled
prefill is 40 registers, six blocks and 75% theoretical occupancy; decode is
56 registers, four blocks and still 50%. Neither promoted specialization spills.
SM89 code receives no launch-bound annotation.

Full trained-model logits, hidden values and GDN/history/KV bytes match the
pinned native reference, with unchanged fixed CPU gates and native generation
without Torch math. All 32 story outputs match in the matched benchmark.
Initial experiments each pass 72 fixture tests / 69 skips. The complete V100
suite passes 839 / 384 with original-baseline Q8 byte checks enabled; local
suite passes 729 / 494. One initial full-suite attempt failed because the
remote source sync omitted launcher scripts; the complete suite was rerun
after syncing them, before any engine throughput timing. The failed attempt
is preserved and is not a passing run.

ABBA comparisons at one/eight requests use the same official original-derived
Q8 checkpoint, FP16 activations, 23 prompt tokens and 32 fixed greedy outputs.
Single-request TP decode improves 17.14 -> 18.86 TPS; eight-request PP aggregate
decode improves 28.31 -> 33.19 TPS. Single-request TP first-token wall improves
0.49082 -> 0.43051 s, whole-request wall 2.29357 -> 2.06736 s and process CPU
2.29581 -> 2.06933 core-s. Eight-request PP whole-cohort wall improves
12.26271 -> 10.56498 s, process CPU 22.99683 -> 19.85303 core-s and mean
individual completion 12.14042 -> 10.46081 s. This is not 33 TPS per client,
MTP, continuous batching or a matched external-engine comparison.

See `benchmarks/v100-cuda-occupancy/summary.md` for compiled resource queries,
separate profiler attribution, raw telemetry, regressions and source hashes.
The larger quality, batching/paging, MTP, graph and matched-comparison goal
remains unfinished.


## Experimental one-proposal MTP (ABI 15)

A separate verified-original-source 15-tensor Q8 draft artifact now binds to the
native target's shared embedding/output head. Drafting, two-token verification,
greedy selection, target GDN/history save/restore/replay and cursor commits remain
native. No external model forward or Python token/layer loop is introduced.
Rebuild for ABI 15 and explicitly pass `--mtp-artifact`; ordinary decode remains
the default. Requests must be fresh/reset; only one-proposal greedy is supported.

Correctness: local 735 passed/532 skipped; final V100 864 passed/403 skipped.
Both TP and PP pass independent represented-MTP hidden/KV gates and exact
ordinary-target output/committed-state checks across four trained prompts.
Natural acceptance is 82.35–100%, including 87.5% on the speed case. A fixed
SM70 T2 projection dispatch removes four-token-tail waste without changing
output bytes. Missing hardware counters remain explicit; theoretical residency
is not achieved occupancy.

Matched 23-prompt/32-output ABBA trials at 1/2/4/8 requests show TP aggregate
complete-request TPS gains of 19–21%, while PP C2/4/8 regresses 23–25%. Single-
request TP post-first rate improves 18.94 -> 24.55 TPS; native TTFT slightly
worsens. TP C8 mean completion latency also worsens despite improved cohort TPS.
These tradeoffs prohibit a universal serving-speed claim or default enablement.

Full CPU/latency/memory/telemetry data, separate profiling, the preserved failed
reporting attempt and acceptance definitions are in
`benchmarks/v100-native-mtp/summary.md`; transaction/limits in
`docs/native-mtp-plan.md`. No MTP2/3, continuation, sampling, continuous batching,
paging, whole-model graph or external-engine performance acceptance is implied.


## llama.cpp comparison campaign

Current acceptance is the revised target near the top of this document. Reuse
qualified native MTP/TP/PP source and immutable evidence rather than repeating
all earlier work. New external comparison evidence belongs in a separate capsule;
old capsules, official original weights and prior service configurations stay
unchanged. Existing unrelated GPU services remain untouched. First establish the
external baseline and publish the actual gap before adding another optimization.

## Native batched MTP1 target verification

The target verifier now packs independent requests in one native cohort. Each
slot keeps its own target/draft KV, GDN/convolution state, cursors and prefix-one
rollback; MTP children borrow the target batch's permanent sessions. Drafting,
proposal heads and shifted prefill remain serial native operations. This is not
PP, paging, continuous batching, continuation or a whole-model CUDA graph.

Fresh trained output/target/draft/reset gates and 86 new selected tests pass;
final V100 suite is 1,992 passed/448 skipped and local is 785/1,655. The unchanged
independent target/teacher reference is pinned/reused. No precision change or new
original-BF16/corpus-quality claim. A setup failure from two omitted launcher
scripts and its aborted speed preflight are preserved before the repaired suite.

The initial three-repeat matrix improves C2 HTTP from ordinary 36.30 to 42.61 TPS
but regresses against stronger C1 scalar MTP2 and C8 ordinary batching. An SDK-only
same-SO comparison then removes nested repeated metadata validation from hot
public APIs, not native compute or mutation guards. Final results and the
explicitly reused external controls are in `benchmarks/v100-batched-mtp/summary.md`.
Keep all modes opt-in: no universal strategy or full-goal acceptance is implied.

## Token-tile ordering: negative engine screen

Adjacent T8 CTA ordering passes 104 new checks/full 2,096-test V100 suite and
77 actual checkpoint operator cells with exact output bytes, plus fresh trained
target/private-state and MTP1/MTP2 gates. A ~3.6% median fused event-time reduction
does not transfer sufficiently: one-repeat C8 HTTP ratios are 1.0066 ordinary256
and 1.0106 MTP1 cohorts; C1 MTP2 is flat. All promotion gates fail, so the full
three-repeat engine matrix is not launched and all nine qualified default native
sources are restored. No precision change, occupancy claim or new TPS milestone.
Evidence/reused stronger controls and the setup failure are in
`benchmarks/v100-token-tile-order/summary.md`. The goal remains unchanged and
active; next consider single-verifier shared-word/register layout and measured
host launch/source-check cost, not another identical tile/body sweep or removal
of the already-present TP ownership events.

### 2026-10-20: paired activation words, measured but not adopted

Evidence: `benchmarks/v100-activation-words/summary.md`. Four isolated layout /
compiler-bound arms preserve arithmetic; the guarded two-descriptor variant
passes the full2,085-test GPU fixture suite, trained target states and MTP1/MTP2
state gates. Median fused T3 operator rate improves3.76%, but three-repeat scalar
MTP2 single post-first decode improves only42.747→43.296 TPS and whole HTTP only
33.855→34.123 TPS (+0.79%). This misses the predeclared C1 HTTP>1% adoption gate;
all nine qualified default native dependencies are restored. C8 ordinary and
MTP1-cohort paths have no affected-kernel calls and miss screen promotion gates.
The external >=5% single/parallel objective remains active and incomplete.
Stop this small-bound sweep; the next distinct experiment is T16/row8 reuse for
packed parallel verification, not a repeat of adjacent-CTA ordering. Counter
permissions, precision, TP fences and stronger scalar/parallel controls remain
unchanged. No commit/push or permanent service was created.

### 2026-10-20: T16 reuse rejected; two-GPU graph transport prerequisite passes

Evidence: `benchmarks/v100-token16-reuse/summary.md`. Natural T16/row8 is slower
(median fused0.93143×); staging two8-row halves recovers theoretical occupancy
and gives1.03640× median fused rate, below the predeclared1.05 operator gate.
Both pass124 explicit fixtures and full2,209/2,210-test GPU suites. Neither is
promoted to trained-model or serving measurements; all nine default native
source hashes are restored. No new engine throughput or full-model quality win.
A separate unchanged-library two-V100 host-bridge CUDA Graph probe preserves
exact bytes and changing inputs with both fences. It is capability evidence
only: one graph trial is slower, and dynamic model KV/GDN/source/failure/reset
semantics are untested. Next is an explicit resident scalar-MTP2 layer-graph
prototype with position/state/source gates, not further T16 occupancy tuning.
The full >=5% external single plus1/2/4/8-request goal remains active.

## Layer graphs closed; column/row MLP prerequisite measured

The scalar MTP2 target-layer graph prototype passed full verification-logit,
private state and source/failure/reset tests, but its matched C1/C8 screen was
about1–2% slower. It is not enabled by default; executed experimental sources,
failed fixtures, exact graph-node attribution and restored-source proof are in
`benchmarks/v100-layer-graphs/summary.md`. Graph launch reduction did not remove
the underlying projection work or roughly535 MB of host copies per direction.

A separate research-only SM70 FP32 partial projection and sum now supports
bit-preserving input-column Q8 splits. A two-V100 **MLP block** benchmark, with
matched primary-only input/output and three alternating trials, improves
0.2980→0.2405ms at T1,0.3678→0.2859ms at T3 and0.5967→0.4535ms at T8
(1.239/1.287/1.316×). Actual isolated block copy payload falls84992*T→30720*T
bytes/direction. Keep partial outputs and transport FP32; there is only one
FP16 sum-output cast. The changed FP32 association produces8/19/44 different
FP16 outputs in these cells (relative L2 below1.26e-6 versus previous output),
so this is not same-profile byte equivalence or full-model quality qualification.

The operator is not wired into native model generation or the serving router.
Do not change reported best model TPS on the strength of this block result.
Next is an explicit **MLP-only** column/row TP native profile, retaining current
row-TP attention/GDN, with replaced rather than duplicate down weights, both
ownership fences, fresh teacher/state/MTP gates and C1/C8 engine screens. The
implementation contract is in `benchmarks/v100-column-tp/native-integration-contract.md`.
The >=5% external single-decode and1/2/4/8-request objective remains incomplete.

## Native MLP-only column/row TP: integrated and qualified

The preceding block-only result is now integrated into scalar and packed native
target execution through the explicit `q8-residual-column-mlp` compute profile.
Scope: resident FP16 target/KV, two SM70 GPUs and explicit host transport. Local
gate/up/SwiGLU feed input-column down halves. Encoded weights/scales are copied
without requantization, replacing rather than duplicating permanent down storage.
Partials and transport remain FP32; they are summed before one FP16 cast. The
changed FP32 association is separately qualified, not prior-profile byte parity.
Attention/GDN remain original row TP. Both ownership fences and original source,
failure, reset and private-cache semantics remain in place; defaults do not change.

Fresh represented-Q8/FP16 teacher checks cover 98 logit rows/four trajectories,
mixed-eight batch/reset state, MTP1 cohorts and MTP2 prefix/corrected draft state.
The original FP16 teacher matches the pinned prior complete signatures. New-profile
row budgets 32/256 match exactly. The full remote suite passes 2,132 tests/687 skips
and expanded integration checks pass 22; final local suite passes 803/2,028 skips.
No broad original-BF16/corpus-quality claim is implied.

Three-repeat single post-first decode improves **42.814 -> 47.629 TPS**. Best
complete-cohort HTTP throughput at 1/2/4/8 requests is **37.685/50.638/61.708/73.836
TPS**, using scalar MTP2 / MTP1 cohorts / MTP1 cohorts / ordinary batch256,
respectively. Each complete-HTTP cell exceeds the stronger retained pinned
llama.cpp result by more than 5%; no automatic strategy router is implemented.
Fresh external controls are also measured, but their slower HTTP results do not
replace the stronger historical controls. The same original model revision,
fixed IDs/work, FP16 KV, Q8 encoding differences and timing scopes are explicit.

**Full goal still incomplete:** the strongest pinned single-decode baseline is
49.803 TPS, so >=5% requires **52.293 TPS**, another **9.79%** over the new native
result. Separate C1 profiling finds about 41% less host-bridge payload and better
two-device kernel overlap, not less summed kernel work or measured occupancy.
Counter policy remains unchanged; achieved occupancy/DRAM/cache/stall counters
remain unavailable. Evidence, failures, CPU/latency/VRAM/telemetry and next
bottlenecks are in `benchmarks/v100-native-column-mlp/summary.md`.

## INT16-activation DP2A prerequisite: not promoted

A separate SM70 research library now tests one group32 INT16/FP32-scale activation
pass against the qualified two-pass residual-A8 projection on unchanged stored
Q8 weights. It is not selected by model dispatch. All59 new CPU/GPU checks pass;
the final local suite is815 passed/2,076 skipped. Independent integer/FP64,
packing, tails/strides, nonfinite and graph/changing-input gates precede timing.

Three alternating operator trials on18 actual checkpoint projection cells find
median fused throughput **0.99829x**, below the predeclared1.05 gate. MLP/GDN/
attention T3 fused ratios are0.99387/1.00854/0.99829. The FP32 column-down T3 cell
improves1.04008x, but that does not justify a full-model campaign or silently
different arithmetic. Single-token smaller output projections gain~6%; these
isolated wins do not replace the failed dominant fused gate.

Separate traces show cheaper quantization but essentially unchanged/slower
dominant projection time. T1 fused register count falls72 ->54 and theoretical
residency rises43.75% ->56.25%, yet measured operator time is flat: theoretical
occupancy is not achieved occupancy or a speed predictor. Hardware counters
remain unavailable under unchanged policy1. All owned workers/sampler are terminal
and original unrelated GPU footprint is restored. No new model/HTTP TPS is claimed;
the qualified **47.629 decode TPS** and full **52.293 TPS** goal floor remain.
Evidence/limits are in `benchmarks/v100-dp2a-pilot/summary.md`. Do not tune this
same failing operator indefinitely; next independently measure the remaining
serial FP16 draft body or the proposed semantic GDN-head split prerequisite.

## Residual-A8 MTP draft: qualified, modest engine gain

An additive opt-in native/SDK draft profile now reuses existing two-pass
residual-A8/FP32-scale projection/fusion operations for the SM70 resident FP16
single-device draft body. Original Q8 weights/scales, FP16 KV storage, shared
target vocabulary/head and the qualified column-MLP target stay unchanged;
different draft arithmetic/cache values are independently checked, not claimed
byte-equal to the FP16 draft. Default constructors, precision and old ABI16
descriptor sizes remain unchanged. The new profile rejects PP/BF16/unsupported
architecture or missing optional exports before draft-body allocation.

Native integration32 tests, full remote **2,236 passed/687 skipped**, final local
**818 passed/2,105 skipped**. Fresh trained original-source draft hidden/KV checks
for both profiles pass (new maximum relative L2 0.00077616, minimum centered
cosine0.999999699), plus four target/MTP2 trajectories, same-new-profile MTP1 vs
MTP2 corrected draft bytes, mixed-eight cohorts and selective reset. The unchanged
target teacher/98-logit quality reference is explicitly pinned/reused; fresh
ordinary target outputs/final private states match it. Five extra metadata/source
guards pass, including direct native rejection after a generated fixture's source
changes. Initial CLI-mock/local-dispatch and missing-fixture-directory setup
failures are preserved, not relabeled as numerical failures or passing runs.

Both paths pass C1/C8 screens and three-repeat1/2/4/8 matrices. Scalar MTP2 single
post-first improves **47.6568 ->48.4089 TPS** (+1.58%); complete HTTP improves
**37.6866 ->38.3460 TPS**. MTP1 cohort HTTP2/4/8 improves
**50.5622/61.7380/66.5970 ->51.8212/63.6598/68.9105 TPS**. All8 repeated cells
improve; all qualifying C1/C8 trials improve. Acceptance stays17/26 for MTP2 and
14/16 per MTP1 request; no forced proposals/target replay. Ordinary256's stronger
prior73.8363 TPS atC8 is retained, not remeasured or replaced by slower MTP.

Separate C1 tracing shows draft accounting80.52 ->61.41ms (includes warmup/catch-up),
verification essentially unchanged574.08 ->574.35ms and identical316.66MB host
payload/direction. Thus further draft-only tuning is not the main remaining
opportunity. Full objective remains incomplete: strongest pinned external single
decode49.8033 TPS, required52.2935, another **8.02%** over current native. No fresh
external engine run is claimed. CPU/latency/VRAM/NVML, source ownership, profiles,
limits and next distinct diagnostics are in `benchmarks/v100-mtp-draft-residual/summary.md`.

## Resident source-validation diagnostic (2026-10-20)

`benchmarks/v100-source-validation/summary.md` freezes an unchanged-SO CPU audit,
not another inference optimization or competitor campaign. A marked raw native
C1 MTP2 generation makes6674 descriptor/path verification pairs (6223 target,
451 draft),13,348 metadata syscalls. Actual-checkpoint warm CPU microbenchmarks
suggest approximately11.1ms of work, about1.4% of mean795.5ms raw call wall.
GPU overlap and cache/scheduling differences prevent translating that estimate
to a TPS gain. strace inflates the call to2.02s; its343ms summed metadata durations
are NOT the untraced cost. All32 IDs and17/26 acceptance remain unchanged.

The predeclared >1% and >5ms opportunity gate passes for a bounded future
resident-only lexical per-forward deduplication prototype. That implementation,
its deterministic mid-call source-mutation rejection tests, and actual C1/C8
baseline/change screen remain unfinished. Do not silently drop source checks,
alter streaming read guards, claim a win, or declare the52.29 TPS goal achieved.
