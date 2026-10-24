# Native Q8 Qwen-27B implementation goal

Status: **active**, not a completed native model or a speed-superiority claim.
Updated: 2026-10-19 (Asia/Shanghai).

## Acceptance target

Implement a fully independent native Qwen-27B text inference path, initially
W8A16: signed Q8/group-32 FP16-scaled weight matrices, BF16 activations on SM89,
and a separately qualified FP16 activation path on SM70. Small norms, GDN gate
constants and convolution parameters retain their declared floating precision;
GDN recurrent state is FP32. Weight Q8 does not imply quantized KV/state.

Forward math, cache management and production scheduling should be native;
Python may remain the acquisition/loading/testing/CLI layer. No external NInfer
forward or hidden allocating Torch fallback counts as the independent runtime.
Vendor CUDA/cuBLAS primitives are allowed and identified explicitly.

Compare a defined workload matrix, not every conceivable workload: single and
concurrent serving, matched checkpoint/quantization/activation and KV precision,
prompt/context/output lengths, warmup, concurrency and timing scopes. Require
trained-checkpoint correctness/quality checks before speed claims. Track TTFT,
decode latency, aggregate throughput, CPU time, VRAM and GPU diagnostics. Preserve
regressions. Microbenchmarks are not full-model TPS or competitive engine wins.

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
- Blocks C ABI 8: rebuild explicitly; earlier ABI-4/5/6/7 benchmark artifacts retain
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
7. Batch-aware GDN/convolution state and attention/KV ownership; continuous
   batching and paged KV. Eight independent requests must never share one
   recurrent sequence state or attend across request boundaries.
8. Separately implement/build/qualify SM70/FP16 (V100) support, including Q8/GDN;
   changing only `-gencode` or bypassing the existing SM89 guard is insufficient.
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
