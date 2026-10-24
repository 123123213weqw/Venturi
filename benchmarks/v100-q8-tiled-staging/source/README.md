# Venturi

A specialized, high-performance inference engine.

**Siphon loads the weights; Venturi runs the model.**

That is the dense safetensors path. The optional Qwen path below explicitly
uses an external NInfer/Cinference native backend, not Siphon or GPT-NeoX math.

## First milestone: a model-agnostic loader

Venturi **loads checkpoint tensors**, provides reusable Ada Q4 Linear operators,
and can generate text with its first **dense GPT-NeoX model adapter** (Pythia).
An independent, opt-in native backend supports the existing quantized Qwen
checkpoint; its upstream mathematics/kernels are explicitly attributed.
Other architectures are not executable merely because their weights can load.
The loading core has no architecture-specific configuration classes, required
weight names, layer counts, or model adapters. Architecture interpretation and
execution kernels are separate components, not part of checkpoint loading.

### What it does

- Loads a local safetensors file or single/sharded checkpoint directory.
- Uses Siphon for multi-file CUDA loading, with an explicit safetensors reference
  path for CPU tests and CUDA cross-checks.
- Inspects headers and validates shard indexes before allocating GPU weights.
- Preserves tensor names, physical shapes, storage dtypes, and packed integers.
- Retains optional `config.json`, including quantization descriptions, safetensors
  file metadata, and shard-index metadata without interpreting them.
- Returns tensors that remain valid after the loader closes.
- Reports synchronized load times, physical tensor bytes, and CUDA allocator delta.
- Verifies every loaded byte against the checkpoint, with bounded host-transfer chunks.

**Loading a checkpoint does not mean Venturi can execute its architecture.**
This layer validates storage integrity and consistency, not whether a model has
all the parameters its forward pass requires. `config.json` is optional and model
Python files are never executed.

### Quantized storage

INT8 weights and INT4 values packed into ordinary `int32`, `uint8`, or other
supported integer containers are loaded **as stored**. Scales, zero points, and
other tensors are preserved alongside them. Nothing is dequantized, transposed,
repacked, or silently cast. The original metadata is retained for future
model-specific execution components.

The loader supports byte-aligned safetensors dtypes available in the installed
PyTorch/safetensors stack. Backend/device support still applies. This safetensors
loader does not read `.ninfer`; the separate Qwen path below does. Sub-byte
floating-point formats, original `.pth` files, GGUF, checkpoint conversion and
automatic downloads are not implemented here.

The optional operator API below interprets one explicit packed format; it does
not infer an execution format from arbitrary quantization metadata.

## Independent reusable feed-forward block

`venturi.models.GatedMLPWeights` is the first standalone building block toward
a Venturi-owned model core. It composes the same explicit mathematics for
Llama-style and Qwen-style feed-forward sublayers:

```text
normalized = RMSNorm(x, norm_weight, epsilon)
gate       = Linear(normalized, gate_weight)
up         = Linear(normalized, up_weight)
activated  = SiLU(gate) * up
down       = Linear(activated, down_weight)
out        = x + down
```

The three projection weights have shapes `[I,H]`, `[I,H]`, `[H,I]`. There is no
bias, architecture detection, model graph registry, checkpoint download or
dependency on the external NInfer/Cinference model runtime. These are **not yet
complete Qwen/Llama adapters**: GDN, embeddings, final head and model-specific
binding still need independent implementations. Standalone RoPE/attention and
one complete standard decoder layer with dense KV state are now available below. The
existing `venturi qwen` generation path remains unchanged and external-backed.

Ownership and numerical boundaries:

- `DenseWeight` and `Q4RowSplitWeight` describe stored matrices; preparation
  takes owned copies. Prepared weight tensors must not be mutated.
- `CudaBackend` provides Venturi-owned CUDA RMSNorm, SwiGLU and residual-add
  kernels. Dense Linear uses a Venturi C ABI bridge to **cuBLAS**, not an
  original GEMM algorithm. No PyTorch C++ extension ABI is required.
- Q4 execution requires an **explicit** `AdaBackend`; its existing attributed
  kernels are reused, not rebranded as a new quantized GEMM. No Q8 operator is
  added by this milestone. The standalone blocks library does not include the
  external engine or its headers.
- Native activations/dense weights support FP16, BF16 and FP32 on SM89. Q4
  retains the existing FP16/BF16 activation contract. RMSNorm materializes its
  scale vector in FP32, accumulates statistics in FP32 and casts its output
  once. Inputs must have representable FP32 square-sums; epsilon must be a
  positive normal FP32 value. Each Linear, SwiGLU and residual output is also
  activation-cast; this does not promise the same bits as a fully fused MLP.
- Mutable activations and scratch are explicit `GatedMLPWorkspace` buffers.
  Allocate outside timing/graph capture and use independent buffers for
  overlapping executions. Dense cuBLAS uses a caller-owned 4 MiB scratch
  buffer, reused sequentially by the three projections. Native submission
  does not perform weight decoding/materialization into a full floating model.
- Plans have fixed token extents and may be reused for matching-shape requests
  with independent workspaces. Use `weights.upload(...).prepare(tokens=...)` to
  share one owned weight allocation across decode/prefill shapes. The older
  `weights.prepare(...)` convenience call still uploads a separate copy per call.
  There is no implicit global cache or content-based deduplication.
- Warm plans on the intended stream before capture/timing. Keep plans, graphs,
  inputs, outputs and workspaces alive until queued work completes. CUDA plans
  do not use per-operator PyTorch numerical fallbacks; `TorchBackend` is a
  separately selected, allocating CPU/CUDA reference. Its Q4 reference
  explicitly materializes exact stored values in FP32/64, not BF16-rounded
  weights.

Small CPU example (the explicit reference backend, not a CUDA speed claim):

```python
import torch
from venturi.models import GatedMLPWeights
from venturi.ops import DenseWeight, TorchBackend

h, i, tokens = 32, 80, 7
weights = GatedMLPWeights(
    norm=torch.ones(h),
    gate=DenseWeight(torch.randn(i, h) / h**0.5),
    up=DenseWeight(torch.randn(i, h) / h**0.5),
    down=DenseWeight(torch.randn(h, i) / i**0.5),
)
plan = weights.prepare(TorchBackend(), tokens=tokens,
                       dtype=torch.float32, device="cpu", epsilon=1e-6)
workspace = plan.empty_workspace()
x = torch.randn(tokens, h)
out = torch.empty_like(x)
plan.run(x, out=out, workspace=workspace)
```

Build explicitly **on the Linux CUDA server**, never implicitly at import:

```bash
PYTHONPATH=src python -m venturi.ops.build --component blocks --output-dir build/blocks
# Optional, only when composing packed Q4 projections:
PYTHONPATH=src python -m venturi.ops.build --component ada --output-dir build/ada
```

Each command prints a content-addressed library path and retains compiler output
and a build manifest. Load the first path with `CudaBackend(library)` and
prepare the same block on `device="cuda:0"` with matching stored dense-weight
dtype. For Q4, pass `q4_backend=AdaBackend(q4_library)` explicitly. Neither path
loads the external Qwen engine. The ordinary Qwen backend is not switched by
building these libraries.

### Shared-block qualification and current performance limits

On the RTX 4090D, **127 selected tests passed without skips**, including the
existing Q4 operators, new primitives, independent FP64 composition tests,
buffer/alias guards, ownership, current-stream execution, graph replay and a
real-sized near-cancellation regression. Local CPU validation passed 366 tests
with 90 CUDA/asset skips. Compilation occurred only on the remote CUDA host.

Synthetic full-size sublayers tested `[H,I]=[4096,14336]` and `[5120,17408]`,
T=1/128, with BF16 dense or packed Q4 weights. Every output of every individual
Op was compared with independent FP64 math on its actual represented inputs:
**96 stage checks passed** across both execution backends. The existing Q4
criterion was retained. All 16 full-block eager/graph output comparisons were
bit-identical within their backend. This is not a full model, pretrained
checkpoint evaluation or a claim of identical BF16 outputs across backends.

Two implementation issues were resolved rather than hidden by tolerance changes:
the Q4 reference now preserves decoded stored weights in FP32/64 instead of
rounding them to BF16, and dense GEMM requests **pedantic FP32 computation**.
The earlier default cuBLAS mode exceeded the fixed per-Op bound on two
near-cancellation outputs at T=128/N=5120/K=17408; pedantic mode passes those
same inputs and unchanged bound. Counterexamples and the regression are retained.
Separate whole-chain FP64/BF16 diagnostics can exceed a small error budget even
on the PyTorch reference because intermediate casts amplify rounding-boundary
differences. Those diagnostics remain visible; they are not relabeled as
lossless correctness or substituted for independent per-Op qualification.

Representative unprofiled **CUDA-graph wall microseconds per sublayer call**:

| Synthetic shape | Stored format | T | PyTorch reference | Native block |
|---|---|---:|---:|---:|
| Llama-style 4096/14336 | BF16 | 1 | 393.7 | 381.3 |
| Llama-style 4096/14336 | BF16 | 128 | 622.8 | 2209.9 |
| Llama-style 4096/14336 | Q4 | 128 | 1299.6 | 574.5 |
| Qwen-style 5120/17408 | BF16 | 1 | 585.0 | 573.9 |
| Qwen-style 5120/17408 | BF16 | 128 | 846.9 | 3838.9 |
| Qwen-style 5120/17408 | Q4 | 128 | 2114.9 | 858.4 |

**Strict dense prefill is substantially slower than PyTorch at present.** This
is a correctness-first foundation, not a promoted fast backend. Q4 timings
compare packed execution using the existing Ada operator with an explicitly
materialized FP32 PyTorch reference, not another quantized engine or a new Q4
algorithm. No whole-model TPS benefit follows. The phase order was fixed
(Torch eager/graph, native eager/graph), with >=2 seconds per phase after warmup;
this is descriptive microbenchmark evidence, not a multi-session ABBA claim.

Raw CUDA-event elapsed, wall time, process CPU time and 200 ms whole-device GPU
samples (busy, VRAM, power, clocks, temperature) are retained per phase. CPU
includes this Python process/native library work and benchmark waiting; 100%
means one core. GPU samples are coarse, with the first second excluded. They
are not achieved occupancy or DRAM bandwidth utilization. Counter policy stayed
`RmProfilingAdminOnly: 1`; no new counter probe, driver privileges or clocks
were changed. No kernel-attribution diagnosis is claimed by this benchmark.

Reproduce with `tools/benchmark_shared_mlp.py --library BLOCKS_LIBRARY
--q4-library ADA_LIBRARY --output-directory NEW_DIRECTORY` on the CUDA host.
Build manifests, tests, all eight cases, diagnostics and preserved failures are
in [`benchmarks/shared-mlp`](benchmarks/shared-mlp/).


## Shared weight residency and attention primitives

Weight storage is now separate from shape-dependent execution plans. This applies
to native dense, native Q4, and the explicit PyTorch reference. Uploads own copies
and establish readiness before binding plans on other streams. Shape binding does
not copy/materialize tensors; dense shapes get separate cuBLAS handles. Plans
retain storage if the resident wrapper is released. Tensors exposed through
`storage_tensors` are diagnostic, **read-only by contract**, not mutation APIs.

```python
# `weights` is GatedMLPWeights; `backend` is CudaBackend or TorchBackend.
resident = weights.upload(backend, dtype=torch.bfloat16, device="cuda:0")
decode = resident.prepare(tokens=1)
prefill = resident.prepare(tokens=128)
# These plans share weight pointers, but not activation/scratch buffers.
decode_workspace = decode.empty_workspace()
prefill_workspace = prefill.empty_workspace()
print(resident.storage_nbytes)
```

Individual operators expose `backend.upload_linear(...)` and
`backend.upload_rmsnorm(...)`, with `.prepare(tokens=...)` on each resident.
The standalone `AdaBackend.upload_linear(weight, device="cuda:0")` additionally
allows FP16/BF16 activation plans and different routes to share the same Q4 planes;
the general blocks resident fixes the activation dtype at upload. No automatic
weight conversion, repacking, global registry or runtime weight swapping occurs.
Retain plans, tensors, graphs and workspaces until all queued work completes.
Call dense plans' `.close()` only afterwards; it releases execution handles, not
other plans' shared weights. The independent blocks library is now **ABI 2**;
rebuild it explicitly rather than loading an old ABI 1 library.

### RoPE and causal grouped-query attention

Both native and reference backends expose the same explicit APIs:

```python
rope = backend.prepare_rope(tokens=128, heads=24, head_dim=256,
                            rotary_dim=256, layout="split_half",
                            dtype=torch.bfloat16, device="cuda:0")
# x/out: [128,24,256]; cos/sin: caller-selected FP32 [128,128].
rope.run(x, cos, sin, out=rotated)

attention = backend.prepare_attention(
    tokens=128, kv_tokens=640, query_heads=24, kv_heads=4, head_dim=256,
    dtype=torch.bfloat16, device="cuda:0",
)
# q/out: [128,24,256]; k/v: contiguous [640,4,256].
attention.run(q, k, v, out=attended)
```

- Layout is contiguous `[tokens, heads, channels]`. RoPE supports split-half or
  interleaved pairs, an even partial rotary extent, and exact in-place rotation.
  Frequency conventions, positions, scaling and table selection belong to the
  caller/model adapter. It does not infer a model's RoPE convention.
- Attention supports FP16/BF16/FP32, head dimensions 64/128/256, MHA/GQA/MQA,
  FP32 accumulation and a final activation cast. Adjacent groups of `Hq/Hkv`
  query heads share each KV head. The default scale is `1/sqrt(head_dim)`.
- With Q length T and KV length S >= T, query row i sees keys `0..S-T+i`.
  This bottom-right causal alignment supports prefill, chunked prefix attention,
  and one-token decode. The caller must provide already assembled, visible K/V.
  No padding mask, bias or sliding window is provided. Logits must remain finite
  in FP32; overflow/non-finite inputs have no numerical guarantee.
- Native RoPE/attention run on the caller's current stream, accept caller-owned
  output and allocate no GPU scratch. Output must not alias Q/K/V; RoPE only
  permits exact input/output aliasing. Warm execution before CUDA graph capture.
- Native attention uses **serial-key online softmax**, without a score matrix.
  It is a correctness-first baseline, **not FlashAttention, PagedAttention, a
  paged/quantized KV cache, or a promoted fast kernel**. The explicit PyTorch
  reference allocates scores/repeated KV heads and is not a native fallback.

A complete standard decoder layer with dense KV ownership/append and Q/K/V/output
projection composition is now available below. Full independent-model execution
still needs checkpoint-specific bindings, embeddings/final head and, for the
hybrid 27B checkpoint, GDN and its model-specific normalization/gating. None of
these components replaces the current external-backed `venturi qwen` path.

### Residency/attention qualification on RTX 4090D

**217 selected remote tests passed without skips**; local validation passed
416 tests with 130 CUDA/asset skips. Coverage includes FP16/BF16/FP32, independent
FP64 per-Op checks, buffer guards, both RoPE layouts/partial rotation, causal GQA,
4K-token prefix correctness, chunk/single-query equivalence, CUDA graph replay,
source isolation, shape binding without tensor copies/allocations, resident
lifetime, and overlapping streams with independent execution buffers.

For one synthetic `[H,I]=[5120,17408]` MLP prepared at T=1/17/128:

| Stored format | Separate weight MiB | Shared weight MiB | Sequence wall us, separate | Shared |
|---|---:|---:|---:|---:|
| BF16 | 1530.06 | 510.02 | 7318.2 | 7263.3 |
| Q4 g64/FP16 scale | 406.46 | 135.49 | 1916.2 | 1551.4 |

The sequence is **three sequential MLP calls in one CUDA graph**, not three
concurrent requests or a model token. Weight storage falls exactly 66.7%; all
four trials' outputs and eager/graph comparisons were bit-identical. Dense
sequence timing was effectively flat (-0.75%); Q4 sequence latency fell 19.0%
in this ABBA session only. No kernel/cache attribution, single-shape TPS benefit
or whole-model gain is inferred from that Q4 result. Whole-device NVML VRAM was
2470→1450 MiB (dense) and 1262→978 MiB (Q4), distinct from tensor-only weight bytes.

Preparation wall/CPU times are separate from execution. BF16 wall was
1.824–1.906 s separate versus 0.613–0.675 s shared; process CPU was
5.795–5.969 s versus 1.951–2.057 s (four intra-op CPU threads). Q4 preparation
was 61.0–65.5 ms separate versus **22.7–95.6 ms shared**; the slow shared trial is
retained, not excluded to claim a uniform loading speedup.

Standalone BF16 operator median CUDA-graph **wall microseconds/call**:

| Operator shape | Explicit PyTorch reference | Native |
|---|---:|---:|
| Attention T1/S129, Hq24/Hkv4/D256 | 24.50 | 56.79 |
| Attention T128/S128, Hq24/Hkv4/D256 | 62.98 | 177.49 |
| Attention T128/S640, Hq24/Hkv4/D256 | 144.88 | 1510.97 |
| Attention T128/S128, Hq32/Hkv8/D128 | 50.95 | 209.20 |
| RoPE T1/H24/D256, split-half | 11.79 | 4.40 |
| RoPE T128/H24/D256, split-half | 20.23 | 4.49 |

Every full-size operator/backend output passed the independent FP64 bound and
its eager/graph outputs were identical. **Attention is slower in every timed
case** and is not promoted. RoPE gains are against this explicit PyTorch
reference, not a third-party fused kernel. Native runs do not use Tensor math
fallbacks. Strict dense GEMM prefill remains slower as documented above.

Timing is one unprofiled ABBA session, two samples per condition, at least two
seconds per phase, three warmups, 100 graph replays per chunk. Event/wall/CPU
measurements and raw 200 ms telemetry are retained. Execution CPU is about one
busy core, including Python graph dispatch and CUDA-event waiting; this change
does not claim an idle-CPU optimization. During the Q4 sequence native GPU busy
was 98.5–100% by phase mean, power 403.5–416.3 W, SM clock 2760–2770 MHz and
temperature 54 C. These coarse whole-device samples are **not achieved occupancy
or bandwidth utilization**. Counter access remains restricted by
`RmProfilingAdminOnly: 1`; no counter retry, privilege/clock changes, profiler
attribution or timeline-gap diagnosis was performed.

An expanded test run exposed an unseeded FP16 whole-chain residual cancellation
failure. A deterministic seed-7 input and outputs are preserved under
`failures/composite-fp16-counterexample.json`: every one of its six Ops passes
the unchanged independent FP64 bound, while the composite relative-error
assertion fails. Tests now check those per-Op bounds and bit-identical native
composition instead of treating amplified whole-chain rounding as an Op defect.
No per-Op tolerance was enlarged; this is not a claim of lossless FP16 execution.

Qualification and unprofiled baseline/change measurements are retained in
[`benchmarks/residency-attention`](benchmarks/residency-attention/). Reproduce on
the Linux CUDA host with `tools/benchmark_residency_attention.py --library
BLOCKS_LIBRARY --q4-library ADA_LIBRARY --output-directory NEW_DIRECTORY`.



## Independent decoder layer with dense KV state

`DecoderLayerWeights` now composes one **bias-free, pre-RMSNorm GQA/SwiGLU**
decoder layer using the existing independent Ops:

```text
x → RMSNorm → Q / K / V projections
              Q,K → caller-selected RoPE tables
              K,V → directly into this layer/request's cache tail
              Q + visible K,V → causal attention → output projection
x + projected → existing normalized residual SwiGLU MLP → out
```

This is **not a complete Qwen/Llama checkpoint adapter**. There are no embeddings,
final norm/head, sampling, checkpoint-name bindings, Q/K normalization, attention
gates, biases, GDN, paging, prefix sharing or KV quantization in this component.
Native execution currently supports BF16 activations/dense weights and BF16 KV;
projections may independently use the existing explicit Q4 descriptor/plugin.
`TorchBackend` provides BF16 and an explicit FP64 reference mode. Other existing
model/generation backends are unchanged.

### Ownership and execution boundaries

- `weights.upload(backend, dimensions=AttentionDimensions(...))` owns one copy of
  each matrix/vector. Each T-specific `.prepare(tokens=T)` shares those pointers;
  `.at(position=P)` adds only attention-position metadata and shares the same
  Linear handles. No per-position weight upload, handle creation, repacking or
  runtime full-weight decode occurs. Upload still **copies the source**; it does
  not automatically adopt/free checkpoint tensors or deduplicate independent uploads.
- One `LayerKVCache` belongs to one resident layer and one request. Its dense
  `[capacity,Hkv,D]` K/V tensors allocate once. K/V Linear writes directly to
  `[P:P+T]`; K is rotated in place. There is **no separate append-copy kernel or
  temporary K/V projection allocation**. Attention sees only `[0:P+T]` and row i
  sees keys `0..P+i`. Uninitialized/stale tail slots are never visible.
- `DecoderWorkspace` is one execution lane's aligned arena. Norm/down-projection
  storage is reused; query/attention-output slots become MLP gate/up slots after
  attention consumes them. A single dense cuBLAS scratch buffer serves all seven
  projections. The same arena can serve smaller T and compatible sequential
  layers, including mixed dense/Q4 layers with different scratch requirements.
- CUDA cache/arena operations are bound to their creating stream. Distinct streams
  require distinct caches/workspaces. Host submission locks prevent threads from
  interleaving a shared arena's operator sequence. They do not synchronize the GPU
  or turn one arena into concurrently usable storage.
- Every call checks shapes, cache ownership/visible position, capacity, aliasing,
  alignment and plan lifetime before writes. A submission exception after execution
  starts poisons the cache. `reset()` discards visibility; `truncate(P)` discards a
  suffix but is not a fork/shared-prefix implementation. Both use stream ordering,
  not a GPU memset. A successful call commits visible length after **enqueue**,
  not after GPU completion. Keep owners/buffers alive until work completes.

Explicit composition sketch (tables and hidden states come from the caller):

```python
from venturi.models import AttentionDimensions

dims = AttentionDimensions(query_heads=24, kv_heads=4, head_dim=256,
                           rotary_dim=256, rope_layout="split_half")
resident = weights.upload(backend, dimensions=dims)  # DecoderLayerWeights
prefill = resident.prepare(tokens=128)
decode = resident.prepare(tokens=1)
cache = resident.empty_cache(capacity=2048)
workspace = prefill.empty_workspace()  # also fits decode; no new arena per step

prefill.at(position=0).run(hidden_prompt, cos_prompt, sin_prompt,
    out=prompt_out, workspace=workspace, cache=cache)
decode.at(position=cache.length).run(hidden_next, cos_next, sin_next,
    out=next_out, workspace=workspace, cache=cache)
```

### Managed fixed-position graph capture

Use `step.capture(..., out=..., workspace=..., cache=...)` outside any existing
capture. The prefix must already be valid and the caller must avoid other CUDA
work/allocations in this process during setup. Preparation uses a non-default
stream with explicit dependencies to/from the owning stream. Warmup/capture may
write uncommitted tail slots but **does not advance visible cache length**.
The returned wrapper retains its buffers; `.run()` replays and commits length.
It rejects a wrong cache position, changed tensor storage or a closed plan.
A new position needs another `.at(...)`/capture; this is not a dynamic-length
whole-model decode graph. Raw external capture of stateful `.run()` is rejected
so Python KV bookkeeping cannot silently drift from replayed GPU work.

This layer does not make native attention fast: the serial-key online-softmax
baseline and pedantic dense GEMM remain the documented performance limitations.
Qualification and matched workspace/backend measurements are under
[`benchmarks/decoder-layer`](benchmarks/decoder-layer/). Run
`tools/benchmark_decoder_layer.py --library BLOCKS_LIBRARY --q4-library ADA_LIBRARY
--output-directory NEW_DIRECTORY` on the Linux CUDA host. This benchmark runs two
synthetic standard decoder layers, **not the hybrid Qwen checkpoint or model TPS**.



### Decoder-layer qualification and measured limits

On the RTX 4090D, **246 selected tests passed without skips**. Local validation
passed 431 tests with 144 CUDA/asset skips. The new tests cover FP64 independent
full-layer references, both rotary layouts/partial rotation, prefill/chunk/decode
consistency, BF16 cache capacity/reset/truncation, failure poisoning, source
isolation, shape/position binding without weight copies, mixed dense/Q4 scratch
requirements, shared-arena host submission serialization, independent request
streams, fixed-position graph visibility, changed-pointer rejection, and native
execution without allocating Tensor/numerical fallbacks.

The full-size benchmark runs **two sequential synthetic standard layers** with
H5120/I17408/Hq24/Hkv4/D256, BF16 activation/KV, either dense BF16 or packed Q4
weights, T128/P0 or T1/P128 and capacity129. Decode prefixes contain the real
preceding synthetic prefill, not random/zero KV. All **240 per-Op FP64 checks**
passed using unchanged bounds and actual represented inputs. Separate-temporary
composition and the layer's real cache matched exactly. All **32 two-layer
workspace/graph phase outputs** were bit-identical within their backend.
Native and reference backends are **not bit-identical** to each other; these are
numerical-contract checks, not model-quality or lossless BF16 claims.

Active per-lane activation/scratch storage, not weights or persistent KV:

| Stored projections | T / prefix | Separate arenas for 2 layers | One sequentially shared arena |
|---|---:|---:|---:|
| BF16 | 128 / 0 | 30.00 MiB | 15.00 MiB |
| Q4 | 128 / 0 | 22.00 MiB | 11.00 MiB |
| BF16 | 1 / 128 | 8.172 MiB | 4.086 MiB |
| Q4 | 1 / 128 | 0.172 MiB | 0.086 MiB |

Sharing halves those workspace bytes for two compatible layers; it does not
halve the whole model's VRAM. Layer weights and request/layer KV remain distinct.
The two native layers own 1300.08 MiB of BF16 weights or 345.39 MiB of Q4 weights,
shared across T1/T128 plans; their two BF16 KV caches total 1.008 MiB. The per-layer
versus shared-arena timings differed by less than 0.3% here: **a memory/lifetime
improvement, not evidence of a meaningful kernel speedup**.

Median unprofiled **wall microseconds per two-layer sequence** with a shared arena:

| Stored projections | T / prefix | Explicit PyTorch reference | Native |
|---|---:|---:|---:|
| BF16 | 128 / 0 | 2351.6 | 11099.0 |
| BF16 | 1 / 128 | 1599.3 | 1602.7 |
| Q4 | 128 / 0 | 5437.7 | 2928.7 |
| Q4 | 1 / 128 | 3061.6 | 1667.0 |

Dense prefill is still about **4.7x slower** than the reference. Q4 compares
packed execution with explicitly FP32-materialized reference weights, not another
quantized engine. Its synthetic codes are uniform [-8,7] with a nonzero population
mean, not quantized pretrained weights. Maximum absolute cross-backend difference
4.0 is retained for the two Q4 cases (normalized RMS differences 0.0972%/0.0698%);
BF16 normalized RMS differences were about 0.716%/0.704%. Per-Op checks passed, but
these full-chain diagnostics are not relabeled as identical model behavior or
pretrained quality evidence. No full Qwen, Llama, serving or generation-TPS claim
follows from these operator/layer measurements.

Workspace comparisons use ABBA separate/shared/shared/separate; backend comparisons
use ABBA reference/native/native/reference. Each phase lasts at least two seconds
after three capture warmups, with 100 wrapper calls per event chunk. A call runs
both layer graphs and manages cache visibility; timing includes host validation,
dispatch/bookkeeping and event waiting, not just kernel durations. Every repeat
truncates back to the same prefix: this is **fixed-position replay, not a sustained
autoregressive rollout**. One ABBA session is descriptive, not a confidence interval.
Process CPU remained about one busy core; no idle-CPU optimization is claimed.

Raw 200 ms GPU busy/VRAM/power/clocks/temperature samples and event/wall/CPU time
are retained, with the first second of each phase excluded from telemetry summaries.
Both backend residents coexist during the comparison, so whole-device VRAM is not
a minimal single-backend model peak. Q4 prefill shared-arena phase means were
100% GPU busy, 421.9–427.5 W, 2628–2634 MHz SM clock and 53.6–54.6 C. These are not
achieved occupancy or bandwidth utilization. `RmProfilingAdminOnly: 1` remains;
no restricted counter retry, privilege/clock changes or new profiler attribution
was performed. The production external Qwen runner hash is unchanged; the GPU
returned to 387 MiB/0% busy after the jobs.

Initial default-stream capture and CPU-base-alignment test assumptions were fixed
and their logs preserved under `benchmarks/decoder-layer/failures`. Capture now
uses an explicit non-default preparation stream with dependencies; CUDA arenas
retain 256-byte alignment, while CPU tests check relative region alignment.


## Qwen-27B: explicit optional native backend

`venturi qwen` supports the existing NInfer v3 artifact whose text config is
`Qwen3_5ForCausalLM`: the tested 27B checkpoint has 48 linear-attention and 16
full-attention layers, mixed stored formats, and an optional paired DFlash2
component. This is **not GPT-NeoX execution, a generic HF Qwen loader, or a new
Venturi implementation of Qwen CUDA kernels**. Qwen mathematics, quantized
materialization, CUDA graphs and speculative verification run in the external,
Apache-2.0 NInfer/Cinference implementation. Venturi owns explicit artifact
inspection, verified staging, resident-process transport, CLI and measurement.

No weights are expanded to a full FP16 model. Vision and MTP are not enabled;
only text and the explicitly selected drafter are materialized. The Python
client does not create a PyTorch CUDA context or run per-token Python forwards.
One owned native process holds the GPU model across API calls. A call can run
one request or a bounded native batch; there is no HTTP server or background
service. Closing the session releases it.
Python transport time and native generation times are reported separately.

### Build against the existing Ada native implementation

Run this **on the CUDA server**, with a compatible source tree and its existing
Release SM89 Ninja build. The backend currently uses the dependency's in-tree
C++ interface, not an installed, version-independent SDK:

```bash
python tools/build_qwen_runner.py \
  --source /home/test/wangyue/engine \
  --build /home/test/wangyue/engine/build \
  --output /home/test/wangyue/venturi-work/qwen/native/venturi-qwen-runner
```

The tool compiles only Venturi's transport and links the selected existing
static libraries. It does not modify/rebuild upstream sources, install packages
or download models. The build report hashes the linked archives; LICENSE,
NOTICE and upstream provenance are copied beside the executable. The native
binary is not included in the wheel or repository. A different dependency
version must compile and pass acceptance before it can be claimed compatible.

### Inspect, generate and reuse

```bash
venturi qwen inspect /absolute/path/to/model.ninfer --json
venturi qwen generate /absolute/path/to/model.ninfer \
  --runner /absolute/path/to/venturi-qwen-runner \
  --prompt 'Explain why the sky looks blue.' --max-new-tokens 64 --json
venturi qwen benchmark /absolute/path/to/model.ninfer \
  --runner /absolute/path/to/venturi-qwen-runner \
  --prompt 'Write a Python merge-sort function.' --max-new-tokens 128 \
  --ignore-eos --warmup 1 --repeats 3 --json
```

Defaults: non-thinking user chat via the embedded template, greedy selection,
BF16 KV, context 2048, prefill chunks 128, CUDA graphs on, blocking CPU waits,
speculative decoding off. `--raw-prompt` bypasses the chat template.
`--execution eager` disables graphs; `--prefill-chunk`, `--max-context` and `--kv-dtype bf16|k8v4` are explicit
startup choices. Prompt plus requested output must fit context: no truncation.
EOS is respected in generation; only the explicit benchmark `--ignore-eos`
requests fixed output lengths. Prefix reuse is disabled for isolated-request
measurements. Startup is not hidden inside warmed TPS. Research `NINFER_*`
environment flags are removed from the child environment, preventing unnoticed
weight/state precision or head-screen changes.

```python
from venturi.backends.qwen import QwenNativeSession

with QwenNativeSession(
    '/absolute/path/to/model.ninfer',
    runner='/absolute/path/to/venturi-qwen-runner',
) as model:
    first = model.generate('Explain CUDA graphs.', max_new_tokens=64)
    second = model.generate('Now explain quantization.', max_new_tokens=64)
    print(first['text'], second['text'])
```

RPC calls are sequential; independent concurrent Python calls are rejected. Use
`generate_many` for native parallel execution within one call (below). Invalid
requests are rejected before execution where possible. Native process exit,
malformed responses and read timeout close the owned process; no other GPU job
is stopped. No fallback to a different model, precision or backend is implicit.

For the existing deployment, from the Mac checkout:

```bash
bash scripts/venturi-qwen-4090d generate \
  --prompt 'Explain CUDA graphs.' --max-new-tokens 64
```

The launcher uses the staged Venturi source and explicit native runner on
School_4090D. Model paths refer to that server. `VENTURI_QWEN_MODEL`,
`VENTURI_QWEN_RUNNER`, `VENTURI_QWEN_DIR`, `VENTURI_REMOTE_PYTHON` and
`VENTURI_SSH_HOST` override that deployment without changing credentials.

### Switch between single and parallel dispatch

`single` is the default: requests execute one at a time. `parallel` submits the
whole input list before waiting, so the existing native scheduler can form
compact decode batches on **one GPU with one copy of the weights**. This is
request batching, not tensor parallelism, multi-GPU execution, Python threads,
or a guarantee that one person's request generates faster.

```bash
# Serial list: repeat --prompt, or use --prompts-file prompts.json.
venturi qwen generate /absolute/path/to/model.ninfer \
  --runner /absolute/path/to/venturi-qwen-runner \
  --mode single --prompt 'What is 2+2?' --prompt 'Explain gravity briefly.'

# Native parallel list: reserve up to four active requests.
venturi qwen generate /absolute/path/to/model.ninfer \
  --runner /absolute/path/to/venturi-qwen-runner \
  --mode parallel --max-concurrency 4 \
  --prompt 'What is 2+2?' --prompt 'Explain gravity briefly.' --json
```

The SSH launcher accepts these same options. `--prompts-file` is a JSON array
of 1–16 strings; its path is **on the server** when using the launcher. Results
retain input order, each request has isolated recurrent/KV state, EOS is
respected, and prefix reuse remains disabled. Every prompt is prepared and
checked against the output budget before any request in that list is submitted.
One prompt alone is still B=1, even with `--mode parallel`.

For switching **without reloading the model**, reserve concurrency once:

```python
with QwenNativeSession(
    '/absolute/path/to/model.ninfer',
    runner='/absolute/path/to/venturi-qwen-runner',
    mode='single', max_concurrency=4, max_context=2048,
) as model:
    prompts = ['What is 2+2?', 'Explain gravity briefly.']
    serial = model.generate_many(prompts, max_new_tokens=64)
    model.set_mode('parallel')
    batched = model.generate_many(prompts, max_new_tokens=64)
    model.set_mode('single')
    one = model.generate('Now explain CUDA graphs.', max_new_tokens=64)
    # A per-call override does not change the session's default mode.
    batched_again = model.generate_many(prompts, mode='parallel')
```

`set_mode` is idle-only and changes dispatch, **not the startup GPU reservation**.
The native maximum concurrency is fixed at 1–8. Defaults are C=1 in single mode
and C=4 in parallel mode; parallel needs C>=2. A session initially created with
C=1 needs reopening with a larger reservation before it can run parallel. A
single request in a C=4 session does not release the other reserved lane state.
The list limit is 16, with excess active requests waiting in the native FIFO.
The session timeout covers the complete batch response, not each row separately.

`--max-context` is the per-request ceiling. `--kv-capacity` is the **shared
physical token budget**, rounded to 64-token pages; by default it reserves
`C * ceil(max_context / 64) * 64` tokens. An explicit smaller shared budget can
reduce VRAM but may reduce simultaneous admissions. It must hold one full
context and at least one page per lane; it cannot exceed the full C-lane page
capacity. No automatic KV precision change or OOM fallback is performed.
In particular, the successful **single-request 256K K8V4 test does not mean four
256K caches fit on a 24 GB GPU**. Use shorter per-request contexts for parallel
work, or explicitly budget a shared cache and accept lower concurrency.

For list benchmarking, `venturi qwen benchmark` uses the same prompt/mode
options. It reports **all output tokens / elapsed batch wall time**, including
preparation, prefill and queueing, excluding startup and warmup. It does not sum
overlapping per-request decode times. Native `runtime.mean_decode_batch` exposes
the actual compact batch size; configured C is only an upper bound. The existing
one-prompt single benchmark retains its separate decode-only metric.

**Numerical limitation:** serial and compact-batched execution are not guaranteed
to produce identical greedy token IDs. Acceptance found counterexamples with
both BF16 and K8V4 KV on the stored quantized checkpoint. Single-row dispatch
and serial resets are checked independently, alongside literal-answer/state
isolation tests. These checks are not a broad model-quality evaluation; do not
interpret successful batching as a proof of lossless numerical equivalence.

This interface requires the **protocol-3 runner** built from the current source.
Older binaries fail explicitly rather than silently ignoring concurrency. The
math, paged cache, scheduler and kernels are still the external NInfer/Cinference
implementation; this change exposes those capabilities in Venturi's public API.
`tools/validate_qwen_modes.py` is a finite production-path acceptance/comparison
with one resident process, serial/parallel mode switches, ordered exact-ID
comparisons on its stated workload, and wall-clock throughput.

Production-API acceptance on the same RTX 4090D / Qwen27B checkpoint used eight
requests per wave, **509 raw prompt tokens + 64 fixed output tokens each**,
context 2048, prefill chunk 512, graphs on, no speculation/reuse. Each comparison
kept one model resident: warm both dispatch paths, then two measured waves per
mode in single/parallel/parallel/single order. These retained runs used the
protocol-2 runner and CUDA automatic CPU waits, before the blocking-wait default.

| Startup reservation | KV profile / shared capacity | Single total tokens/s | Parallel total tokens/s | Actual parallel decode batch |
|---|---|---:|---:|---:|
| C=4 | BF16 / 8192 | 46.82 | 116.11 | 3.91 |
| C=8 | K8V4 / 4608 | 46.98 | 150.05 | 7.20 |

These are **separate within-configuration serial/parallel comparisons**, not an
isolated precision comparison. Serial state replay and one-row parallel checks
passed; four distinct literal answers were returned in the correct order under
each tested configuration. Some ordinary raw/chat greedy outputs differed
across modes; full IDs, reproduction script, the initial failed equality tests,
and revised behavioral checks are retained in
[`benchmarks/qwen/modes`](benchmarks/qwen/modes/). This is a bounded offline API,
not a new HTTP server or batch-invariant numerical implementation.

### CPU-efficient CUDA waits

The native runner defaults to `--cpu-wait blocking`. CUDA blocks the waiting
CPU thread on a synchronization primitive instead of busy-waiting for GPU
completion. This changes the **CPU waiting policy**, not the model mathematics,
KV precision, GPU stream dependencies, CUDA Graphs or request concurrency. It
only affects this owned runner process's CUDA context, not other GPU jobs or
system-wide settings. See the
[NVIDIA CUDA 13 runtime reference](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-runtime-api/group__CUDART__DEVICE.html).

```bash
# CPU-efficient default; works with single or parallel dispatch.
venturi qwen generate /absolute/path/to/model.ninfer \
  --runner /absolute/path/to/venturi-qwen-runner \
  --cpu-wait blocking --prompt 'Explain CUDA graphs.'

# Restore CUDA's previous automatic scheduling heuristic for latency comparison.
# On the tested server this heuristic actively spins while waiting.
venturi qwen benchmark /absolute/path/to/model.ninfer \
  --runner /absolute/path/to/venturi-qwen-runner \
  --cpu-wait auto --prompt 'Explain CUDA graphs.' --json
```

Python uses `QwenNativeSession(..., cpu_wait='blocking')` or `cpu_wait='auto'`.
This is a startup choice, independent of runtime `set_mode`. The protocol-3
startup response reports the applied policy and CUDA device flags; mismatched
runners fail rather than silently ignoring the option. Non-scheduling CUDA
context flags are preserved. A sleeping CPU may have extra wakeup latency;
`auto` remains available when maximum raw throughput matters more than freeing
CPU capacity.

The earlier internal `host_active_fraction` counter **is not process CPU use**:
it excludes time inside device waiting, where the driver may still spin. Use
`tools/benchmark_qwen_cpu.py` on the Linux CUDA server to measure actual native
process user+system CPU seconds from `/proc/<pid>/stat`, plus Python process CPU
separately. **100% means one logical CPU core**, not the whole machine. Measurements
include all threads in the owned native process and exclude startup/warmup.
These counters do not measure CPU power, whole-system energy or energy/token.

On the RTX 4090D / dual Xeon E5-2640 v4 server, a counterbalanced
`auto / blocking / blocking / auto` comparison used four measured requests per
process after warmup: each 509 raw prompt tokens + 64 fixed output tokens,
BF16 KV, context 2048, chunk 512, graphs enabled, no speculation or prefix reuse.
Startup and warmup were excluded. Combining both processes per policy:

| Metric | CUDA auto | Blocking waits |
|---|---:|---:|
| Native CPU utilization (100% = one logical core) | 100.03% | 2.17% |
| Native CPU seconds for 512 output tokens | 10.92 | 0.24 |
| Native + Python CPU milliseconds/output token | 21.334 | 0.478 |
| Total output tokens/s, including prefill | 46.90 | 46.27 |
| Median native first-token latency | 255.56 ms | 257.76 ms |

This freed almost one CPU core: **97.8% less native CPU time**, at the cost of
**1.35% lower output throughput** on this workload. It is not a free tokens/s or
energy-efficiency claim. All 16 measured single-request sequences matched the
unchanged protocol-2 baseline's IDs. Targeted GPU/transport/accounting validation
passed **62 tests**, including BF16/graphs, K8V4/eager, DFlash2, state replay and
parallel dispatch; local CPU validation passed **309 tests** with 56 opt-in tests
skipped. The separate pilot C4 experiment also reduced CPU use, with a small
throughput cost. Evidence and the retained pilot are in
[`benchmarks/qwen/cpu`](benchmarks/qwen/cpu/). Only the native wait policy and
transport contract changed; external model/kernel libraries were not rebuilt.

### Optional byte-exact RAM staging

The native dependency uses direct reads, which can reread a slow disk at every
startup. Explicit staging copies the complete encoded artifact into selected
storage, verifies source/destination SHA256, preserves volumes/bindings and
records the up-front cost. Nothing is dequantized or repacked. For tmpfs, the
full artifact consumes host RAM until deleted or rebooted; it is not free GPU
memory, an automatic eviction cache, or a resident decode optimization.

```bash
# On the server; inspect the JSON report before using the resulting path.
venturi qwen stage /absolute/path/to/model.ninfer \
  --directory /dev/shm/venturi --json
# --reuse verifies the staged files before returning an existing cache.
venturi qwen stage /absolute/path/to/model.ninfer \
  --directory /dev/shm/venturi --reuse --json
```

Pass the report's `path` to generate/benchmark, or set `VENTURI_QWEN_MODEL` for
the SSH launcher. The original checkpoint is never modified. A new private
staging directory is published atomically; insufficient space/source mutation
fails without replacing an existing cache. Staging and checksum time must be
included when evaluating a first cold request.

The v3 reader validates framing, file identity/sizes, object ranges, known
encoded sizes and binding/resource references. Supported inspection layouts
are contiguous BF16/FP32/INT32 and row-split Q4/Q5/Q6-G64/Q8-G32 with FP16 scales.
Unknown formats fail explicitly. Bounded encoded reads span volumes; one-row
diagnostic reconstruction is independently tested. Architecture binding and
native Op capability validation still belong to the backend.

### Speculative decoding and validation

```bash
venturi qwen benchmark /absolute/path/to/model.ninfer \
  --runner /absolute/path/to/venturi-qwen-runner \
  --speculative dflash2 --draft-tokens 8 \
  --prompt 'Write a detailed explanation of how rain forms.' \
  --max-new-tokens 128 --ignore-eos --warmup 1 --repeats 3 --json
```

`--verify-tree --draft-tokens 15 --optimized-proposal --kv-dtype k8v4` explicitly
selects the other evaluated upstream DFlash2 route. That route requires K8V4
and 24x256 attention; it is rejected before GPU startup otherwise. K8V4 changes
cache precision and is measured against a separate K8V4 baseline, not silently
substituted for BF16 KV. No new draft model is downloaded or
trained. Reports retain rounds, drafted/accepted tokens and acceptance by
position. Speculative batching/state schedules can change floating-point
rounding relative to token-by-token execution: greedy-ID agreement is measured,
not assumed. No FP32/BF16-original model-quality equivalence follows from speed.

```bash
PYTHONPATH=src python tools/run_qwen_validation.py /absolute/path/to/model.ninfer \
  --runner /absolute/path/to/venturi-qwen-runner \
  --baseline-cli /absolute/path/to/existing/ninfer \
  --output-dir /absolute/path/to/new-results --profile
```

Acceptance compares the independent product CLI with Venturi transport on the
same quantized native backend; it is not an independent mathematical oracle.
The tool measures eager/graph/graph/eager, an explicit prefill-chunk variant,
and DFlash2 chain/tree routes sequentially with prefix reuse off. Other GPU
compute jobs stop the benchmark; only the known GNOME remote desktop service
is allowed. Nsight brackets warmed generation, excluding loading; profiler
timings are not substituted for unprofiled throughput.

### Eight-request DFlash2 comparison on RTX 4090D

A 2026-10-17 production-API test compared ordinary decoding with DFlash2 chain8
at startup concurrency 8. One `generate_many` RPC submitted eight distinct
non-thinking chat prompts (mixed prose/code), not eight HTTP clients. Both modes
used the same RAM-staged quantized checkpoint, BF16 KV, context 2048, total KV
capacity 16384, prefill chunk 512, CUDA graphs, blocking CPU waits, no prefix
reuse, and 128 output tokens per request with EOS explicitly ignored. Tree
verification and the indexed proposal head were off.

Fresh resident sessions ran ordinary/DFlash2/DFlash2/ordinary. Each warmed one
eight-request wave, then measured three waves: six measured waves and 6144
output tokens per variant. Loading and warmup were excluded; aggregate
throughput divides total outputs by shared client wall time, including prefill
and JSON transport, never by summed concurrent request durations.

| Measurement | Ordinary, C8 | DFlash2 chain8, C8 |
|---|---:|---:|
| Aggregate output tokens/s | 333.76 | 346.54 |
| Median whole-batch client wall | 3.068 s | 2.955 s |
| Median internal native first-token time | 301.2 ms | 380.3 ms |
| Native CPU, 100% = one core | 4.94% | 4.46% |
| Sampled whole-device GPU busy | 97.3% | 98.7% |
| Whole-device VRAM | 18040 MiB | 19980 MiB |

The observed aggregate gain is **3.83%**, not a multi-fold C8 speedup and not
346.54 tokens/s per request. The drafter accepted 4872 of 9528 proposed tokens
(51.13%), averaging 3.98 accepted draft tokens per speculative request-round.
Mean active request rows per decode round were 7.58/5.37; startup capacity 8
does not mean every round retains eight unfinished requests. Native first-token
timestamps are not streamed client TTFT; this API returns a completed batch.

**Speculation remains experimental and off by default.** Ordinary output matched
its reference in 48/48 measured rows; DFlash2 matched 36/48. Two of eight prompts
diverged consistently in all six measured DFlash waves: CPU-cache explanation
at zero-based output index 61 and word-frequency code at index 42. Raw token IDs
and comparisons are retained. Within-session warmup replay matched for all
96 measured rows, and all four native processes exited cleanly. Repeatability
does not establish lossless equivalence or an independent mathematical oracle.

Sampled power averaged 392.7/400.2 W, SM clocks 2775.0/2771.3 MHz, memory clocks
10251 MHz in both, and temperatures 52.4/54.5 C. These whole-device samples do
not establish energy efficiency. Counter policy remained `RmProfilingAdminOnly:
1`; achieved occupancy, compute throughput, DRAM bandwidth and cache/stall
counters were unavailable. No new counter probe, privilege/clock change,
kernel edit or model download was made. This is a throughput comparison, not
a counter-verified bottleneck diagnosis. Local validation passed 332 CPU tests
with 56 hardware/environment skips; Ruff and diff checks passed.

Run `tools/benchmark_qwen_dflash.py CHECKPOINT --runner RUNNER
--output-directory NEW_DIRECTORY --repeats 3` on the CUDA host to reproduce.
The commands, complete outputs, CPU/wall counters, telemetry and summary are in
[`benchmarks/qwen/dflash-c8`](benchmarks/qwen/dflash-c8/).

### Measured Qwen result on RTX 4090D

The retained ordinary/speculative and concurrency/long-context measurements below
used the earlier CUDA automatic CPU-wait policy. The current CPU-policy comparison
is reported above; do not assume identical throughput under blocking waits.

The native integration passed **310 GPU-enabled tests with no skips** (three
expected Siphon warnings). Local CPU validation passed **259 tests**, with
51 opt-in asset/GPU tests skipped. Both ordinary and DFlash2 transports matched
the existing same-mode CLI's generated IDs on two 32-output-token chat cases;
changed-prompt/reset isolation and an actual early-EOS DFlash2 request passed.
This qualifies the integration, not the original full-precision model's quality.

Normal non-thinking chat, BF16 KV, graphs enabled, prefix reuse disabled, one
warmup and two measured requests per prompt, EOS respected (both long answers
reached the 128-token cap):

| Prompt | Ordinary decode | DFlash2 chain, K=8 | Same 128 greedy IDs |
| --- | --- | --- | --- |
| Explain the blue sky | 57.18 tokens/s | 199.10 tokens/s | yes |
| Merge two sorted lists in Python | 57.16 tokens/s | 411.30 tokens/s | yes |

**These are two prompts, not Spec-Bench or a general Qwen speed claim.** Loading,
staging and preparation are excluded from these warmed decode rates. The CUDA
kernels and DFlash2 algorithm are the external backend's, not new Venturi work.

The separate fixed-output raw-token suite used three repeats after one warmup,
128 outputs, context 2048, identical prompt IDs and no prefix reuse:

| Raw workload | BF16 KV ordinary graphs | BF16 KV chain8 | K8V4 ordinary graphs | K8V4 chain8 | K8V4 tree15 |
| --- | --- | --- | --- | --- | --- |
| Prose | 57.17 | 162.07 | 57.20 | 161.93 | 193.32 |
| Code | 57.13 | 341.97 | 57.22 | 361.67 | 561.43 |
| Repeated 512-token prefix | 56.76 | 403.04 | 57.08 | 408.61 | 825.73 |

Rates are tokens/s; BF16 ordinary graphs combine both measured graph processes.
This raw suite explicitly ignores EOS to fix output length. The repeated-prefix
workload is very predictable and is **not representative interactive throughput**.
K8V4 is a distinct precision experiment with its own baseline, not a free speedup
over the BF16 numerical contract. Tree15 also uses the indexed proposal head.

**Speculation remains experimental and off by default.** The raw prose chain
first diverged from ordinary BF16 decoding at output index 99 (99 matching
tokens). K8V4 ordinary decoding differed from BF16 at index 22. Other listed
workloads matched, and tree15 matched its same-KV ordinary sequence on these
three cases, but universal lossless greedy decoding is **not established**.
These discrepancies are retained, not hidden by a relaxed tolerance or an
assertion that plausible prose proves correctness. Their root cause has not
been isolated to a particular rounding operation or kernel.

Ordinary graph policy improved decode by about 2%, not Pythia's order-of-magnitude
gain. In the warmed 32-output Nsight trace, host kernel-launch calls fell from
19,139 to 601 plus 31 graph launches, but both paths still executed 19,139 GPU
kernels. Projection kernels accounted for approximately 95% of summed GPU kernel
duration; additional RMSNorm/RoPE fusion was therefore not the first target.
This is kernel-time attribution, not a measured DRAM utilization percentage.

For the 512-token prompt, explicitly increasing `--prefill-chunk 128` to
`--prefill-chunk 512` reduced median first-token time from about **312.9 ms to
254.3 ms** (18.7% shorter), without changing greedy IDs on the three tested
workloads. Separate prefill-only traces contained 3,797 versus 889 GPU kernel
instances and 307 versus 249 ms summed kernel duration. The larger chunk selects
different existing GEMM/fused-SwiGLU routes as well as reducing chunk overhead;
it is not just graph capture. Defaults remain conservative; this is a measured
option, not an automatically optimal choice for every context/model.

One observed cold disk load took about 125 seconds. Verified staging copied the
complete 16.94-GiB artifact in **181.1 seconds**; subsequent native starts from
that RAM copy took **6.5–7.8 seconds**, still excluded from warmed TPS. Staging
uses additional host RAM and must amortize across starts. The RAM copy remains
under the explicit path recorded in the staging report; closing a GPU session
does not remove it. No model, profiler binary trace or native executable is
committed to the repository.

Raw timing runs, numerical counterexamples, CLI acceptance, source/build hashes,
staging checksums, CPU/GPU test logs and Nsight text statistics are retained in
[`benchmarks/qwen`](benchmarks/qwen/). An initial unsupported BF16/tree request
and a test-deployment/environment failure are retained too; completed timing
runs were reused, not overwritten or selected by best throughput.

### Concurrency, paging and remaining bottlenecks

The external Engine **already implements paged KV storage**, with 64-token page
groups, startup-fixed concurrency 1–8, and compact decode batches. Venturi's
production `QwenNativeSession.generate_many` now exposes single/parallel dispatch
with prefix reuse disabled (see above). The historical measurements below used
`tools/csrc/qwen_concurrency_probe.cpp`, retained for cache/profiling experiments;
they are not measurements of a new paging implementation or HTTP serving API.

On the same 4090D/checkpoint, eight requests per wave, each with 512 raw prompt
tokens and 64 output tokens, produced these **total output rates including
prefill and queueing**, excluding model startup:

| Maximum concurrency | Total output tokens/s | Actual mean decode batch |
|---|---:|---:|
| 1 | 47.08 | 1.00 |
| 2 | 79.53 | 1.99 |
| 4 | 117.41 | 3.91 |
| 8 | 151.15 | 7.20 |

Runs used native BF16 KV (BF16 K / FP16 V), ordinary greedy decoding, no reuse,
and no speculative backend. Each process had one warmup and two measured waves;
concurrency order was 1/2/4/8 then 8/4/2/1. All four measured waves per setting
contribute to the rate. Fixed-output runs ignore EOS; separate normal chat tests
respect EOS. Higher aggregate throughput does **not** make each stream faster.
This is not a Qwen-vs-llama.cpp comparison or a universal serving benchmark.

Capacity, cache and scheduling findings:

- Eight slots with only 2,048 shared KV token-equivalents achieved about **95
  tokens/s**, versus about **151** with sufficient capacity. A workload-specific
  4,608-token budget retained the throughput while reserving **736 MiB less KV
  memory** than the 16,384-token budget. This does not establish the right budget
  for arbitrary prompt/output lengths. Paging cannot replace admission capacity.
- A 795-token chat prompt reused 778–788 tokens on warm requests. Measured
  prefill fell from approximately **475–540 ms to 34–52 ms**. Changed suffixes and
  an explicit reuse-disabled control matched their corresponding greedy outputs.
  This is one cache sequence, not a statistically qualified cache benchmark.
- Prefill chunks trade throughput for delivery gaps. Separate unprofiled
  streaming observations, excluding their warmup, recorded:

| Prefill chunk | Total output tokens/s | p95 token-commit gap | Maximum gap |
|---|---:|---:|---:|
| 128 | 121.36 | 97.38 ms | 99.31 ms |
| 256 | 138.19 | 157.33 ms | 159.04 ms |
| 512 | 150.52 | 271.40 ms | 275.67 ms |

These are native commit timestamps, not network-arrival timestamps. Long gaps
occur while other prompts prefill. A latency-budgeted chunk policy is a next
candidate, not an implemented adaptive scheduler.

NVTX-correlated Nsight node traces attributed about **96% of prefill GPU kernel
duration to projections**, versus about **86% of decode at concurrency eight**.
Recurrent updates grew from about 1.3% of single-request decode kernel time to
7.7% at concurrency eight. Prioritize quantized projection/GEMM routes, prefill
scheduling, and then recurrent-state updates over replacing the existing paging
mechanism. These short-context results do not settle long-context attention.
Nsight Compute hardware counters were denied with `ERR_NVGPUCTRPERM`; recorded
register/shared-memory limits give theoretical occupancy ceilings only, **not
measured occupancy, DRAM utilization, or tensor-core utilization**.

To reproduce on a CUDA host with the existing compatible external build:

```bash
python tools/build_qwen_runner.py --source /absolute/path/to/ninfer \
  --build /absolute/path/to/ninfer/build \
  --runner-source tools/csrc/qwen_concurrency_probe.cpp \
  --output /absolute/path/to/venturi-qwen-probe
PYTHONPATH=src python tools/benchmark_qwen_concurrency.py /absolute/path/to/model.ninfer \
  --runner /absolute/path/to/venturi-qwen-probe \
  --workloads benchmarks/qwen/workloads.json \
  --output-dir /absolute/path/to/new-evidence-directory
```

Use `--only` to select named cases; `--profile --profile-node` enables optional
Nsight graph-node traces with additional instrumentation overhead. Existing
named evidence is never overwritten. Full traces stay on the GPU host; compact
timings, launch geometry, consistency comparisons, retained configuration
failures and interpretation are in `benchmarks/qwen/concurrency/manifest.json`.
The new tools have nine passing CPU contract tests on both hosts; the full local
suite has 268 passing tests and 51 opt-in skips. Ordinary raw comparisons matched
all 384 tested sequences, and normal chat matched all 48 comparisons. This is
same-backend consistency testing, not independent FP32 mathematical qualification.
The earlier speculative numerical counterexample remains relevant.

### GPU-efficiency acceptance and the current gap

[`AGENTS.md`](AGENTS.md) requires relevant inference optimizations to include
correctness, unprofiled before/after measurements and GPU diagnostics, with
unavailable hardware counters explicitly recorded. The reusable Linux telemetry
command is:

```bash
PYTHONPATH=src python tools/benchmark_qwen_gpu.py /absolute/path/to/model.ninfer \
  --runner /absolute/path/to/venturi-qwen-runner \
  --output-directory /absolute/path/to/new-report-directory
```

The current protocol-3 / blocking-wait comparison kept one C=4-capable model
resident, with a fixed shared KV reservation of 8192 tokens. Both modes ran eight
509-token raw prompts and 64 fixed output tokens/request, context 2048, chunk
512, BF16 KV, CUDA Graphs, no speculation/reuse. Both modes were warmed before
single/parallel/parallel/single measured waves. Throughput includes preparation,
prefill and queueing, excluding startup/warmup. NVML was sampled every 200 ms;
the first second of each wave was excluded from telemetry statistics to reduce
coarse-sample boundary mixing.

| Measurement | Single dispatch | Parallel dispatch, C=4 |
|---|---:|---:|
| Total output tokens/s | 46.28 | 114.95 |
| Actual mean decode batch | 1.00 | 3.91 |
| Native CPU utilization (100%=one logical core) | 2.17% | 2.92% |
| NVML GPU busy time | 98.80% | 98.91% |
| NVML memory-controller busy time | 87.97% | 68.29% |
| Whole-device VRAM used | 16914 MiB | 16914 MiB |
| Sampled whole-device GPU power | 414.49 W | 403.63 W |

A separate `--cpu-wait auto` session repeated the same dispatch workload:
single/parallel throughput was 46.88/116.24 tokens/s, GPU busy time
99.95%/100.00%, native CPU use 100.26%/100.12% of one logical core. Restoring
busy-waiting therefore recovered only about 1.3%/1.1% throughput at the cost of
nearly a full CPU core. The earlier counterbalanced CPU comparison is the
stronger wait-policy throughput experiment. GPU telemetry sessions were ordered
blocking then auto, not counterbalanced across policies; their small power and
temperature differences do not establish causal energy savings.

These are whole-device, time-based samples, including the pre-existing desktop
process. Memory-controller busy is **not bandwidth utilization**; its mode
comparison also reflects different prefill/decode time proportions. A GPU busy
reading near 99% does not measure SM occupancy or the fraction of peak compute
achieved. See the [NVIDIA monitoring definitions](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

Separate warmed Nsight Systems node traces attributed **96.08% of prefill kernel
time to projections** in both modes, **94.29% of single decode** and **89.81% of
parallel decode**. Single decode's largest groups were fused Q4 SwiGLU GEMV
(38.48%), Q4 row-split GEMV (29.38%), dual small-T MMA projections (18.42%) and
Q8 output projection (8.01%). The next single-request kernel targets are therefore
Q4 SwiGLU and residual/down projections, rather than another Python rewrite.

Instrumented kernel-timeline coverage was 90.27%/93.21%, but node tracing increased
wall time: those figures are **not unprofiled idle fractions or guaranteed
speedup headroom**. Throughput above comes only from the unprofiled runs. The
fresh Nsight Compute probe again failed with `ERR_NVGPUCTRPERM`; achieved SM
occupancy, compute/Tensor Core utilization, DRAM GB/s, L2 hit rate and stall
counters remain unavailable. No driver/privilege settings were changed. Launch
resource ceilings in the trace analysis are theoretical, not measured occupancy.
There is currently **no defensible numerical percentage of unused compute
capacity**. See [Nsight metric definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

Both unprofiled waves within each mode produced identical IDs; cross-mode IDs
are not universally equal, consistent with the earlier batching counterexamples.
Raw telemetry, complete outputs, profiler summaries, counter-denial logs and
measurement tests are retained in
[`benchmarks/qwen/gpu-gap`](benchmarks/qwen/gpu-gap/). Binary profiler traces remain
on the server; model/kernel code was not changed in this diagnostic run.

### Projection scheduling experiment: no promoted speedup

The first GPU projection experiment targeted single-request Q4 gate/up SwiGLU,
previously about 38.48% of decode kernel time. Eleven alternative schedules
plus the original schedule were compiled in isolated directories on the 4090D:
warp count, asynchronous pipeline depth, group tile size, shared/global activation
reads and weight-copy cache policy. External sources, archives and the production
runner were not overwritten. Replacement objects link before unchanged archives.

A public-Op, rotating-weight CUDA-graph benchmark reads eight distinct matrices
(757,596,160 aggregate storage bytes, larger than device L2), with 64 calls per
graph, 30 warmup graphs and 50 timed graphs. Forward/reverse screening found the
best candidate, `g8w4s3`, at about **102.46 us versus 103.01 us** for the original:
only roughly **0.5% kernel-level gain**, not a model throughput claim. Its shared
memory fell from 23,296 to 16,768 bytes/block and registers from 40 to 39/thread,
but resource-only occupancy ceilings (33.3% to 50% in separate node traces) are
not achieved occupancy measurements. The traced target-kernel duration fell
only about 0.37%; profiler timing is not substituted for unprofiled throughput.
More warps and cached-global activation variants did not help. The earlier
write-flush benchmark is retained as a separate cache condition; its upstream
`dram_spec_pct` uses another GPU's nominal peak and **must not be used here**.

The selected candidate and baseline passed independent packed-weight decoding
and FP64 SwiGLU-oracle checks at T=1/2/17/32/512, two seeds, guarded buffers,
unchanged inputs/weights, and positive/negative graph replay at T=1/2/17. The
existing A16 criterion was not loosened. A matched, unprofiled full-model ABBA
comparison used the same resident mixed-precision Qwen checkpoint, BF16 KV,
2,048 context capacity, four reserved slots, chunk 512, CUDA graphs, blocking
CPU waits, no speculation/reuse, eight 509-token raw prompts and 64 fixed output
tokens each. Both dispatch modes warmed before measurement in each session.

| Across two sessions per variant | Original | Candidate |
|---|---:|---:|
| Single total output tokens/s, including prefill | 46.215 | 46.205 |
| Parallel total output tokens/s | 114.698 | 114.609 |
| Single native CPU, 100% = one core | 2.12% | 2.14% |
| Parallel native CPU | 3.08% | 2.97% |
| Sampled whole-device VRAM | 16,914 MiB | 16,914 MiB |

All 128 measured output rows matched their same-mode baseline token IDs, as did
the separate traced outputs. Four real-model regression tests passed, covering
session reuse, mode/state isolation and BF16/K8V4 KV. The local CPU suite passed
315 tests with 56 hardware/environment skips. Small
timing differences are not a demonstrated speedup or regression: temperatures
and clocks drifted during the session, and the unchanged parallel route also
varied. GPU busy stayed around 99%; power, clocks, temperatures and raw samples
are retained alongside the timings. Hardware-counter access remained restricted
(`RmProfilingAdminOnly: 1`); no achieved occupancy, DRAM GB/s, cache/stall or
compute-utilization conclusion follows. The candidate is **not promoted**.

Reproduce the isolated experiment only on a CUDA host with the explicitly
selected compatible external source/build (the source patch is hash-guarded):

```bash
python tools/projection/build_candidates.py --source /absolute/path/to/ninfer \
  --build /absolute/path/to/ninfer/build --output /new/experiment \
  --candidate g8w4s3
python tools/projection/evaluate_candidate.py --manifest /new/experiment/builds.json \
  --candidate g8w4s3 --output /new/qualification
python tools/build_qwen_runner.py --source /absolute/path/to/ninfer \
  --build /absolute/path/to/ninfer/build --output /new/candidate-runner \
  --ops-object /new/experiment/g8w4s3/q4_linear_swiglu_gemv.cu.o
```

Keep the ordinary runner as the baseline; use `tools/benchmark_qwen_gpu.py`
against both in baseline/candidate/candidate/baseline order with fresh output
directories. Raw results and build identities are in
[`benchmarks/qwen/projection`](benchmarks/qwen/projection/). A nominal occupancy
increase alone does not justify a dispatch change. The next distinct target is
the residual/down Q4 row-split GEMV, not further blind occupancy tuning.

### Small Q4 residual/down projection test

A follow-up tested three T=1 alternatives through the public fused `LinearAdd`
Op at `[N,K]=[5120,6144]` and `[5120,17408]`. Two schedules redistribute scalar
byte decoding over two rows/four warps or four rows/two warps. The best applies
the existing packed-word decoding strategy with **four rows/block and one
warp/row**, instead of one row/block and eight warps/row. The stored Q4 weights,
BF16 activation/output precision and residual-add formula are unchanged, but
the floating-point reduction order is different.

The read-heavy, larger-than-L2 public-Op screen measured the down projection at
about **59.9 to 52.2 us (13% lower kernel time)**; the K=6144 projection was
essentially unchanged. Both scalar multi-row alternatives made K=6144 slower.
All schedules passed the independent FP64 oracle with the existing A16 error
criterion, full-output checks at T=1/2 with two seeds, guarded buffers, input and
weight preservation, zero/restored-input graph replay, and unaffected-route
regression checks. The original archive was also timed without an override.

The same matched full-model ABBA workload described above then measured:

| Unprofiled measurement | Original | Packed-word candidate |
|---|---:|---:|
| Single total output tokens/s, including prefill | 46.221 | 46.960 |
| Single median 63-token decode time | 1.1232 s | 1.1013 s |
| Parallel total output tokens/s | 114.657 | 114.901 |
| Single native CPU, 100% = one core | 2.19% | 2.16% |
| Single sampled GPU busy | 98.8% | 99.0% |
| Whole-device VRAM | 16,914 MiB | 16,914 MiB |

This is an observed **1.60% single-request total-throughput improvement** in a
small short-context test, not a universal performance guarantee. The parallel
difference is too small to claim a benefit. Single-mode sampled power averaged
about 418/398 W; these coarse whole-device samples do not establish causal
energy savings. Achieved occupancy and DRAM/cache/stall counters remain blocked.
Separate node tracing confirmed the changed kernel and lower target duration;
its timing is not substituted for unprofiled throughput.

**The candidate is not promoted.** One of eight serial prompts changed its
greedy continuation at generated token 24, consistently across repetitions.
The wording differs slightly; this was not a measured logit-margin or broad
accuracy evaluation. Parallel outputs matched the same-mode baseline, and
each mode remained repeatable within each candidate session. Four real-model
regression tests passed, including state reuse and BF16/K8V4 mode switching;
the local CPU suite passed 317 tests with 56 environment/hardware skips.
Passing an Op error criterion does not guarantee identical autoregressive text.
Broader quality checks are needed before enabling this non-bit-identical route.

Run `tools/projection/test_rowsplit.py --source /absolute/path/to/ninfer
--build /absolute/path/to/ninfer/build --output /new/rowsplit-experiment` on the
CUDA server to reproduce the isolated source-hash-guarded build, independent
oracle checks and finite synthetic screen. The experimental runner uses the
existing explicit `--ops-object` build option and can be selected with
`--runner`; the default runner and all external archives remain unchanged.
Results, counterexamples, build commands and GPU samples are retained in
[`benchmarks/qwen/rowsplit-small`](benchmarks/qwen/rowsplit-small/).

### Controlled block-size ablation

For the experimental vectorized Q4 `LinearAdd` kernel, a separate ablation
changed **only rows/block: 2, 4, or 8**, giving 64, 128, or 256 threads/block.
Each row retained one warp, the same packed-word decode, pipeline, cache policy,
lane ownership and reduction order. This is not a comparison against the older
scalar kernel and does not select a global block size for other Ops.

| Threads/block | K=6144 kernel us | K=17408 kernel us | Single total tokens/s | Parallel total tokens/s |
|---|---:|---:|---:|---:|
| 64 | 19.568 | 52.216 | 46.932 | 114.728 |
| **128** | **19.487** | **52.176** | **46.960** | **114.817** |
| 256 | 19.504 | 52.320 | 46.869 | 114.455 |

Kernel columns are medians of unprofiled synthetic run medians. Full-model
columns use the same short-context workload above, two warmed sessions per
size, ordered **128/64/256/256/64/128**. The single-request differences versus
128 are only -0.06% and -0.20%; clocks and temperature drifted. These observations
support **retaining 128 as a reasonable experimental setting**, not a claim of
a statistically established or universal optimum.

All three passed the independent FP64 oracle/guard/replay checks. Supplemental
full-output snapshots matched bit-for-bit across both K values and two seeds;
all **192 measured generation rows** matched the same-mode 128-thread reference.
The CPU suite passed 320 tests with 56 environment/hardware skips. Native CPU
stayed around 2.1-2.2% of one core in single mode; sampled GPU busy stayed around
99% and whole-device VRAM at 16,914 MiB. Raw CPU, power, clocks and temperatures
are retained with each run; small telemetry differences are not causal energy
or CPU-efficiency claims.

Separate public-Op node traces verified 2,560/1,280/640 grid blocks, 40 registers
per thread, and 1,088/2,176/4,352 shared bytes/block. Resource-only resident-block
ceilings are 24/12/6 per SM, giving the same 100% resource-only thread ceiling.
The finite grid supplies another theoretical whole-device ceiling of about
93.6%. **Neither ceiling is achieved occupancy**, which remains unavailable
under the existing counter-access restriction. Traced timings are not model TPS.

Reproduce with `tools/projection/test_rowsplit.py --suite block-size` and the
same explicit source/build/output arguments above. The previous vectorized
candidate's greedy divergence from the original scalar implementation remains;
this ablation adds no observed output differences but does not resolve that
quality question. Production is unchanged. Evidence is in
[`benchmarks/qwen/rowsplit-blocks`](benchmarks/qwen/rowsplit-blocks/).

### Prefill GEMM tile/pipeline experiment

The next experiment targeted the Q4 fused `LinearAdd` GEMM responsible for about
34.1% of the original 509-token prefill trace. It compared the existing 64x128
tile against a different warp mapping, a 64x64 tile with two stages, and that
smaller tile with three stages. No weights, activation precision, KV precision,
decode kernels, or prefill chunk size were changed.

The result was **shape-dependent**. At T=509 the two-stage 64x64 tile reduced
synthetic Op time by about 14%, but at full T=512 it was about 13% slower. The
other warp mapping and deeper pipeline did not provide a general improvement.
Follow-up tests also found regressions at aligned T=320/384/448; nearby partial
tiles improved. A selective experimental dispatch therefore uses the smaller
tile only for **192<T<512 and (T<=256 or T%64!=0)**, preserving the existing
routes otherwise. This is an Op-local scheduling choice, not a model-specific
math or precision change.

A matched original/selective/selective/original comparison measured:

| Raw prompt tokens | Original median prefill | Selective median prefill | Observed reduction |
|---|---:|---:|---:|
| 128 | 79.6 ms | 79.6 ms | Unchanged |
| 509 | 258.4 ms | 249.6 ms | **3.4%** |
| 1,021 | 510.9 ms | 504.0 ms | **1.4%** |
| 2,048 | 1,017.9 ms | 1,018.2 ms | Unchanged |
| 8,192 | 4,158.4 ms | 4,161.0 ms | Unchanged |

Each resident session warmed all lengths, then measured three repetitions with
ascending/descending/ascending length order, 16 fixed output tokens, C=1,
context capacity 16,384, BF16 KV, chunk 512, CUDA graphs, blocking CPU waits,
and no speculation or prefix reuse. Prefill and internal native first-token
timestamps exclude loading/tokenization/warmup; the synchronous RPC does not
measure streamed client time-to-first-byte. Long prompts made of full chunks
showed no meaningful benefit, and small control differences are not regressions.

All initial variants and the selective route passed the unchanged fused A16
FP64 criterion: independent packed decode, a full-output dense oracle at T=193,
two seeds across full/sampled cases, full/tail and selector-boundary probes,
guarded buffers, input/weight preservation and zero/restored-input graph replay.
All **60 measured continuations matched the original runner**, with repeatable
state reuse. This is limited deterministic evidence, not a broad quality claim.

Separate public-Op node traces confirmed the partial route's registers fell
from 154 to 96/thread and shared memory from 45,568 to 29,184 bytes/block.
The resource-only occupancy ceiling rose from 16.7% to 25%; **achieved occupancy
is still unavailable** under the existing counter restriction. Full T=512
retained the original 147-register/45,568-byte kernel. These instrumented traces
are not used as model speed measurements. Whole-device GPU busy was about 99%,
VRAM 16,964 MiB in both cases, and native CPU about 3.4% of one core across the
mixed-length measured generation phase. Power/clocks/temperature samples are
retained; they do not establish causal energy savings.

Reproduce isolated operator tests with `tools/projection/test_rowsplit.py
--suite prefill` or `--suite prefill-selective` and explicit source/build/new
output paths. `tools/benchmark_qwen_prefill.py CHECKPOINT --runner RUNNER
--output-directory NEW_DIRECTORY` runs the end-to-end latency workload.
**Production remains unchanged**; the selective runner is a tested experiment
and is not combined with the earlier non-bit-identical decode candidate.
Evidence and commands are in
[`benchmarks/qwen/prefill-tiles`](benchmarks/qwen/prefill-tiles/).

### Full-chunk GEMM ablation: retain the original tile

A separate 4090D experiment tested a **128-output-channel x 64-token** tile
against the existing **64 x 128** tile for the same fused Q4 LinearAdd Op.
Two warp layouts used 256 or 128 threads per block, respectively. The wide-T
schedule alone changed; decode, packed weights, activation/KV precision,
residual mathematics and external archives were unchanged.

Both new layouts passed independent FP64 qualification: dense full-output
T=512 at K=6144/17408, additional sampled tile/route boundaries through T=1024,
guard/input/weight checks and zero/restored-input graph replay. Neither was
faster. The unprofiled public-Op screen at T=512 measured:

| Projection | Original 64 x 128 | 128 x 64, 256 threads | 128 x 64, 128 threads |
|---|---:|---:|---:|
| K=6144 | 326.1 us | 373.4 us | 382.7 us |
| K=17408 | 902.8 us | 1040.9 us | 1063.3 us |

The better full-chunk candidate (256 threads) was also compared against the
ordinary production runner in baseline/candidate/candidate/baseline order.
Each resident session warmed all six exact token-ID prompt lengths, then ran
three repetitions per length, with 16 generated tokens and EOS ignored.
C1, context/KV capacity 16384, BF16 KV, chunk 512, CUDA graphs, blocking CPU
wait, and no speculation or prefix reuse were fixed. Loading and warmup were
excluded. Median **native prefill**, not streamed client TTFB, was:

| Prompt tokens | Original | Candidate | Latency increase |
|---|---:|---:|---:|
| 128 (unchanged-route control) | 79.55 ms | 79.48 ms | approximately unchanged |
| 509 | 257.87 ms | 283.39 ms | 9.90% |
| 512 | 252.97 ms | 263.53 ms | 4.17% |
| 1021 | 510.80 ms | 546.26 ms | 6.94% |
| 2048 | 1016.36 ms | 1060.91 ms | 4.38% |
| 8192 | 4152.67 ms | 4333.87 ms | 4.36% |

All 72 measured continuations matched their same-length original token IDs;
state replay matched each session's warmup. This small fixture set is not broad
quality validation. Separate public-Op node traces confirmed 147 to 84
registers/thread and 45,568 to 41,984 shared bytes/block for the 256-thread
layout. Its resource-only thread occupancy ceiling increased from 16.7% to
33.3%, **yet execution became slower**. Those ceilings are not achieved
occupancy. The larger row tile also changes weight-staging/dequantization
reuse; hardware counters remain restricted, so a precise stall/bandwidth cause
is not established.

Across the mixed-length measured prefill-plus-decode phases, whole-device GPU
busy averaged 97.6/97.9%, VRAM was 16,964 MiB for both, sampled power averaged
416.4/415.8 W, SM clocks 2740.7/2740.0 MHz, and temperatures 60.9/61.4 C.
Native CPU averaged 3.43/3.57% (100% means one core).
Raw CPU time, client wall, native prefill throughput and telemetry are retained
separately. These samples do not establish energy savings. Achieved occupancy,
DRAM bandwidth and compute/cache/stall counters remain unavailable under
`RmProfilingAdminOnly: 1`; no privileges or clocks were changed.

**Both new schedules are rejected; production and the earlier selective
candidate remain unchanged.** Reproduce the operator ablation with
`tools/projection/test_rowsplit.py --suite prefill-full` and explicit
source/build/new output paths on the CUDA host. Commands, raw results,
qualification logs and the aggregate are in
[`benchmarks/qwen/prefill-full`](benchmarks/qwen/prefill-full/).

### Long-context KV retrieval screen

The same quantized Qwen checkpoint completed **near-256K prompts on the 4090D**
using the existing native K8V4 route, without weight changes, CPU KV offload,
speculative decoding, or prefix reuse. Two seeded synthetic registries per
length contain three exact tags around 5%, 50%, and 95% of the input, plus a
requested project absent from the registry. An independent string oracle grades
the returned JSON. This is a small retrieval screen, not a broad quality claim.

| Context capacity | K8V4 KV payload | Mean observed prefill | Observed decode | Checks passed |
|---|---:|---:|---:|---:|
| 2,048 | 0.049 GiB | 0.94 s | 57.1 tokens/s | 8/8 |
| 16,384 | 0.393 GiB | 8.33 s | 55.3 tokens/s | 8/8 |
| 65,536 | 1.570 GiB | 39.05 s | 51.5 tokens/s | 8/8 |
| 131,072 | 3.141 GiB | 92.16 s | 47.0 tokens/s | 8/8 |
| 262,144 | 6.281 GiB | 240.79 s | 40.1 tokens/s | 8/8 |

Each row has six exact-tag checks and two missing-key checks across two
documents, not eight independently generated documents. Actual prompts at the
largest capacity contained **261,892 and 261,885 tokens**, including the chat
template; output is part of the same 262,144-token context budget. Sampled
whole-device peak VRAM was **21.85 GiB**, including desktop/context allocations.
The 256K configuration therefore fits this workload, but initially reading the
document takes about **four minutes**. Decode speed excludes that initial wait.

BF16 KV was also tested at 2K/16K/64K. All six paired documents produced identical
greedy token sequences and passed their exact-answer checks. At 64K, BF16 KV
reserved 4 GiB versus 1.57 GiB for K8V4. There is **no BF16 reference at 128K or
256K**. These results do not prove unchanged general long-context reasoning,
coding, or factual quality, and do not invalidate the earlier prose divergence.
The K8V4 route can change attention arithmetic as well as cache storage.

One short warmup precedes each engine's two different long documents; reported
latencies can include lazy graph capture and are not a steady-state throughput
benchmark. Prefix reuse is disabled, so each document is fully recomputed.
No runtime defaults or CUDA kernels were changed for this screen.

Reproduce on the CUDA host with an existing checkpoint and native runner:

```bash
PYTHONPATH=src python tools/validate_qwen_long_context.py /absolute/path/to/model.ninfer \
  --runner /absolute/path/to/venturi-qwen-runner \
  --output-dir /absolute/path/to/new-long-context-results
```

For application use, configure `QwenNativeSession` with `max_context=262144`,
`kv_dtype="k8v4"`, `prefill_chunk=512`, and a suitable request timeout (the screen
uses 1,800 seconds). Read large document text from a file in Python; avoid passing
megabyte-sized prompts as shell arguments. Keep prompt, chat-template, and output
tokens within the context limit. Full generated documents remain on the GPU host;
the generator, expected answers, token offsets, hashes, raw responses, and compact
memory/timing evidence are retained in `benchmarks/qwen/long-context/`.

## Text generation: first complete model adapter

The first runner implements **classic dense GPT-NeoX**, validated with the pinned
FP16 EleutherAI Pythia-70M checkpoint used by the loader acceptance tests. Its
forward pass and generation loop are Venturi implementations using PyTorch
operators; they do not call a Transformers model or `model.generate`.

This is a correctness-first eager implementation, **not an optimized serving
engine or a demonstrated speedup over llama.cpp**. The existing Q4 operator
backend is independent: this runner does not silently quantize dense weights or
claim end-to-end Q4 support. Qwen-27B and its mixed `.ninfer` artifact are not
supported by this model adapter.

### Local text completion

Install the optional local tokenizer dependency:

```bash
python -m pip install -e '.[generate]'
venturi generate /absolute/path/to/pythia-70m \
  --prompt "Once upon a time" --max-new-tokens 32
```

The checkpoint directory needs its original `config.json`, safetensors weights,
and `tokenizer.json`. Generation never downloads files. The default CUDA loader
is Siphon, with the same explicit backend and buffer options as `venturi load`.
Use `--tokenizer /absolute/path/to/tokenizer.json` to select the tokenizer
explicitly, and `--json` for token IDs, stop reason, timing, and loader/runtime
information. Without `--json`, the CLI prints only the generated continuation.

If `venturi` is not found, it has not been installed in the active Python
environment. After installation, `python -m venturi` is an equivalent entry
point. Example checkpoint paths above are placeholders, not bundled models.

For the existing School_4090D deployment, the optional SSH launcher uses the
already staged source, checkpoint, and Python runtime without installing
anything on the Mac:

```bash
bash scripts/venturi-4090d generate --prompt "Once upon a time" --max-new-tokens 32
```

Run this from the repository checkout. Its defaults describe this particular
deployment, not a portable model location. `--help` lists environment overrides
for another server or checkpoint. Arguments, including quoted/multiline prompts,
are preserved across SSH. Authentication uses the existing SSH configuration;
the launcher does not change host-key checking, forward an agent, or retry
indefinitely when the network is unavailable.

For the existing pinned Pythia fixture, download only the missing tokenizer:

```bash
curl -fL --retry 3 \
  https://huggingface.co/EleutherAI/pythia-70m/resolve/a39f36b100fe8a5377810d56c3f4789b9c53ac42/tokenizer.json \
  -o "$MODEL_DIR/tokenizer.json.partial"
printf '%s  %s\n' \
  c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624 \
  "$MODEL_DIR/tokenizer.json.partial" | sha256sum -c -
mv "$MODEL_DIR/tokenizer.json.partial" "$MODEL_DIR/tokenizer.json"
```

Use a new target or check existing files before replacing them. `MODEL_DIR` is
the local fixture directory from the FP16 acceptance section. A CPU reference
execution is available explicitly:

```bash
venturi generate /absolute/path/to/pythia-70m --device cpu --loader safetensors \
  --prompt "The capital of France is" --max-new-tokens 16 --json
```

Pythia-70M is a small **base completion model**, not an instruction/chat model.
Repetition or poor answers are model behavior, not evidence of inference
correctness. No chat template, sampling, beam search, or repetition penalty is
applied. Tokenization uses no inserted special tokens, padding, or truncation;
an empty prompt uses the configured BOS token. Greedy decoding stops at EOS or
the requested output limit. Generated EOS IDs are returned but hidden by text
decoding. Nonfinite logits fail instead of being turned into an EOS token.
Tokenizer IDs outside the model vocabulary, and emitted IDs absent
from the tokenizer, are rejected rather than silently dropped.

### Python API and state ownership

```python
from venturi import load_checkpoint
from venturi.models import GPTNeoXModel
from venturi.generation import LocalTokenizer, generate_text

checkpoint = load_checkpoint("/absolute/path/to/pythia-70m", device="cuda:0")
model = GPTNeoXModel(checkpoint)
tokenizer = LocalTokenizer("/absolute/path/to/pythia-70m/tokenizer.json",
                           vocab_size=model.config.vocab_size)
result = generate_text(model, tokenizer, "Once upon a time", max_new_tokens=32)
print(result.text)

# Low-level cached forward API: one unpadded sequence, Python integer token IDs.
session = model.create_session(capacity=128)
logits = session.forward([12092, 2402, 247])  # Last-position logits [vocab].
next_logits = session.forward([int(logits.argmax())])
session.reset()  # Reuse allocated KV storage, not previous request history.
```

The model shares immutable learned weights and derived RoPE tables. Each session
owns a fixed-capacity K/V cache; requests never share mutable cache storage.
`forward(..., all_logits=True)` returns every input position's logits. Cached
chunks use absolute-position causal masking, not SDPA's upper-left rectangular
causal mask. New token positions cannot attend to stale/future cache entries.

Sessions execute/reset only on their creating CUDA stream and are not safe for
concurrent calls on the same session. Independent sessions can use different
streams; they wait for model preparation without a global fence. Caller must
retain model/session tensors until asynchronous work finishes. A failed forward
requires `reset()` before reuse. The high-level generator synchronizes its own stream for
wall-clock timings and creates one session per request by default. An explicitly
supplied reusable session is reset before each nonempty generation request.

The context is bounded by `max_position_embeddings`. Generation conservatively
reserves `prompt_tokens + max_new_tokens` cache positions; oversized requests
fail before execution, without implicit truncation or eviction. Zero requested
new tokens perform no model forward or KV allocation. The full-model eager path
allocates intermediates and is **not** CUDA-graph-captured or allocation-free.

### Model contract and validation

Supported: homogeneous FP16/BF16/FP32 learned tensors on CUDA or CPU (FP64 for
numerical tests), exact GELU, ordinary multihead causal attention, standard
partial/full RoPE with an even rotary dimension, LayerNorm, parallel or sequential
residuals, and untied input/output embeddings. No learned-weight dtype conversion is implicit.
RoPE scaling variants, tied embeddings, bias-free variants, quantization config,
and non-GPT-NeoX model types fail explicitly. Learned names/shapes/dtypes/devices
are validated independently of generic checkpoint storage checks.
Newer `partial_rotary_factor`/`rope_theta` fields must agree with the classic
`rotary_pct`/`rotary_emb_base` interpretation; conflicting descriptions fail.

Logits currently use the learned-weight dtype. An FP32-output LM head over FP16
weights is not implemented; casting already-rounded FP16 logits cannot recover
FP32 token rankings.

Known legacy attention mask, masked-bias, and serialized inverse-frequency
buffers are shape-checked but not used as learned parameters. RoPE is derived
from config in FP32, then cast to the model dtype, rather than using old
FP16-rounded inverse-frequency buffers. The FP64 test path derives it in FP64.

CPU tests compare complete logits and cached continuations with independently
written FP64 equations. Optional Transformers reference tests use a separate
model implementation and are included in CI, not the production dependency set.
For real-checkpoint validation on the CUDA host:

```bash
python -m pip install -e '.[test,generate,reference]'
VENTURI_GENERATION_MODEL=/absolute/path/to/pythia-70m python -m pytest -q
python tools/run_generation_acceptance.py /absolute/path/to/pythia-70m \
  --output /absolute/path/to/generation-acceptance.json
```

The validation tool independently checks FP32 adapter mathematics and FP16
prefill/teacher-forced decode against Transformers references loaded from the
same stored parameters. It also checks a same-precision reference with FP32 RoPE
frequencies preserved, records greedy token agreement at both precisions, and
tests a cached multi-token chunk at position 512. Error limits are dtype-aware:
large logits can have FP16 rounding intervals greater than a fixed absolute
threshold, and FP16 greedy ties need not match FP32 choices. No reference library
is invoked by `venturi generate`.

Cached execution and full-batch prefill are not numerically invariant on this
checkpoint: different matrix shapes and attention kernels can change rounding.
Validation qualifies same-schedule cached FP16/FP32 references separately, and
retains long-context cached-versus-full drift as an explicit diagnostic rather
than increasing its tolerance until it passes. Small-model FP64 oracle tests
check the underlying mathematics independently. Batch-invariant low-precision
execution is not claimed.

Generation timing fields exclude checkpoint loading, tokenizer work, and Python
process startup. `prefill_seconds` includes the first greedy selection;
`decode_seconds` covers later selections; `total_seconds` also includes session
preparation and timing fences. These are eager single-request measurements, not
cross-engine benchmark results. CLI allocator figures are sampled after loading,
before generation/session preparation, unless separately stated.

`time_to_first_token_seconds` measures entry to `generate_tokens` through the
first selected token, including input validation, KV allocation, and the initial
stream fence. It is `null` for a zero-output request. It is not socket/client
latency and excludes loading, tokenization, and text decoding.

### Standard dense generation baseline

Finish and measure the ordinary dense path before integrating Q4 operators.
`benchmark` loads one model, performs unmeasured warmup requests, then measures
independent requests with fresh KV sessions:

```bash
venturi benchmark /absolute/path/to/pythia-70m \
  --prompt "Once upon a time" --max-new-tokens 32 \
  --require-float-dtype F16 --warmup 2 --repeats 10 --json

# Existing 4090D deployment, invoked from the repository on your Mac:
bash scripts/venturi-4090d benchmark --prompt "Once upon a time" \
  --max-new-tokens 32 --require-float-dtype F16 --warmup 2 --repeats 10 --json
```

Normal EOS stopping is the default. Explicit `--ignore-eos` is available only
for benchmark workloads that need exactly the requested number of output
tokens; the report records that policy. Zero-output benchmark requests are
rejected because they cannot measure first-token latency.

Reports retain every run's token IDs, phase times, request/KV bytes, and allocator
peaks, plus median/min/max first-token and request latency. Decode throughput is
`sum(output_tokens - 1) / sum(decode_seconds)`: prefill produces the first token,
which must not count as a decode step. No decode tokens means `null`, not an
invented throughput. Reported aggregate rates are ratios of sums, not averages
of per-request rates. Identical greedy outputs across repetitions check request
repeatability, not reference-model correctness.

CUDA peaks are reset before each measured request, after synchronizing the
request stream. Warmup allocations may remain reserved. The report distinguishes
baseline, total peak, and extra peak allocated bytes. These counters cover only
this process's PyTorch allocator on the device, not total GPU VRAM or Siphon/CUDA
native allocations. Do not profile concurrent work in the same process/device:
peak counter resets are process/device-wide. CPU allocator memory is not
measured. Loading and model preparation are reported separately from warmed
generation, and no allocator cache is flushed between requests.

The measured RTX 4090D FP16 Pythia-70M baseline used two warmup requests and ten
measured requests per workload, with EOS explicitly ignored for fixed lengths:

| Prompt tokens | Output tokens | Median TTFT | Aggregate decode | Peak PyTorch allocated |
| --- | --- | --- | --- | --- |
| 4 | 32 | 5.361 ms | 212.94 tokens/s | 170.09 MiB |
| 5 | 128 | 4.218 ms | 230.22 tokens/s | 171.23 MiB |

These are separate eager workloads on a tiny base model, not a larger-model,
long-prompt, serving, or cross-engine performance claim. All ten outputs were
identical within each workload. The expanded suite passed 225 tests without
skips on the GPU host (three expected Siphon buffer-depth warnings); the local
CPU suite passed 183 tests with 42 opt-in CUDA cases skipped. Raw per-run data,
runtime versions, tested-source hashes, checkpoint identity, and the GPU test
log are in [`benchmarks/fp16-baseline`](benchmarks/fp16-baseline/).

### Comparing with llama.cpp

The paired comparison uses the **same Pythia checkpoint**, not a different
Llama-family model. llama.cpp needs a GGUF conversion of those weights. The
conversion check in `tools/compare_llamacpp.py` verifies every learned value,
including the fused QKV row permutation, and rejects quantized GGUF tensors.
The normal converter promotes bias/normalization vectors to FP32 losslessly;
matrices remain FP16. Legacy mask/serialized RoPE buffers are not learned
parameters and are omitted by the upstream converter.

Build a pinned upstream checkout on the CUDA server, not on the Mac:

```bash
LLAMA_ROOT=/absolute/path/to/pinned/llama.cpp
MODEL_DIR=/absolute/path/to/pythia-70m
cmake -S "$LLAMA_ROOT" -B "$LLAMA_ROOT/build" \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89-real \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_ROOT/build" --target llama-bench -j 8
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python "$LLAMA_ROOT/convert_hf_to_gguf.py" \
  "$MODEL_DIR" --outtype f16 --outfile "$LLAMA_ROOT/pythia-70m-f16.gguf"
g++ -O3 -std=c++17 tools/llama_compare/bench.cpp \
  -I"$LLAMA_ROOT/include" -I"$LLAMA_ROOT/ggml/include" -I"$LLAMA_ROOT/vendor" \
  -L"$LLAMA_ROOT/build/bin" -Wl,-rpath,"$LLAMA_ROOT/build/bin" \
  -lllama -o "$LLAMA_ROOT/venturi-llama-bench"
python tools/compare_llamacpp.py --checkpoint "$MODEL_DIR" \
  --llama-root "$LLAMA_ROOT" --gguf "$LLAMA_ROOT/pythia-70m-f16.gguf" \
  --llama-runner "$LLAMA_ROOT/venturi-llama-bench" \
  --llama-commit 2851b3bcbf8d993a2243ae751c0a7dfa8e944a40 \
  --output-dir /absolute/path/to/new-comparison-results \
  --prompt "Once upon a time" --max-new-tokens 32 --warmup 2 --repeats 10
```

The reported commit must match the checkout actually built; the example pins
the revision selected for this comparison. The caller supplies an environment
with Siphon and the conversion dependencies; the tool does not install packages,
download checkpoints, or build anything. Run from this repository, with Venturi
installed or `src` on `PYTHONPATH`. Use a new output directory each time to retain
failed/previous evidence instead of overwriting it.

Both measured loops reuse already allocated contexts/KV storage, reset logical
history per request, check finite logits, select greedy tokens, and ignore EOS
for explicit fixed output lengths. Input token IDs must match independently in
both tokenizers. The first selected token is prefill, not a decode step. Loading,
tokenization, context allocation, and untimed teacher-forced checks are excluded.
This is a different session-allocation contract from `venturi benchmark` above;
do not silently mix those results.

The custom C++ runner calls the public llama.cpp API. It is **not** the upstream
`llama-bench` program: the upstream tool's synthetic timings omit sampling and
must remain a separate supplementary result. The paired runner tests the default
CUDA-graph policy and a `GGML_CUDA_DISABLE_GRAPHS=1` ablation, while Venturi is
eager. An enabled graph policy alone does not prove every operation was captured.

Equal learned values and FP16 KV caches do not mean identical arithmetic:
llama.cpp retains FP32 activations/logits and uses its tanh GELU approximation
for GPT-NeoX, whereas this Venturi adapter uses exact GELU and learned-weight
precision logits. The report retains greedy-ID equality and full-vocabulary
teacher-forced error/argmax diagnostics rather than declaring numerical
equivalence from a throughput measurement. No tolerance is relaxed to force
agreement. See the pinned [GPT-NeoX graph](https://github.com/ggml-org/llama.cpp/blob/2851b3bcbf8d993a2243ae751c0a7dfa8e944a40/src/models/gptneox.cpp)
and [CUDA GELU implementation](https://github.com/ggml-org/llama.cpp/blob/2851b3bcbf8d993a2243ae751c0a7dfa8e944a40/ggml/src/ggml-cuda/unary.cuh).

Per-process NVML GPU-resident memory is sampled every 20 ms, including loading
and untimed validation. It includes CUDA context/runtime allocations and is a
sampled maximum, not an exact allocator/transient peak. Legacy GPU checkpoint
buffers retained by Venturi but omitted from GGUF also affect resident memory.
Missing NVML support is reported as unavailable, never as zero. Other active
compute processes cause the comparison to stop. An exact path supplied with
`--allow-gpu-process` can exempt a known background desktop service; all observed
background processes are recorded. No workloads are killed and no GPU
clocks/power settings are changed.

#### Measured RTX 4090D paired baseline

Pinned llama.cpp revision: `2851b3bcbf8d993a2243ae751c0a7dfa8e944a40`.
Both engines used the same learned values, FP16 KV caches, two warmups, ten
measured requests, eight CPU threads, and fixed output lengths:

| Prompt/output tokens | Venturi decode | llama.cpp default decode | llama.cpp graphs disabled | Median TTFT Venturi / llama.cpp |
| --- | --- | --- | --- | --- |
| 4 / 32 | 266.12 tokens/s | 1,699.03 tokens/s | 1,281.45 tokens/s | 3.988 / 1.097 ms |
| 5 / 128 | 239.52 tokens/s | 1,779.42 tokens/s | 1,352.03 tokens/s | 3.940 / 1.141 ms |

llama.cpp was **6.38x and 7.43x faster** in these two greedy throughput runs.
Even with its graph policy disabled it remained substantially faster, so graph
capture alone cannot be assumed to close the gap. Attribution to particular
operators or Python dispatch still requires profiling; no optimization speedup
has been demonstrated by this comparison.

All **76 learned tensors** were checked exactly after conversion (26 FP16
matrices and 50 losslessly promoted FP32 vectors). **Free-running greedy token
sequences differed in both workloads.** On 32 teacher-forced logit rows per
workload, top-1 agreement was 30/32 and 31/32; maximum absolute logit errors were
2.391 and 1.830. These remain diagnostics, not a passed numerical-equivalence
claim or an isolated explanation of every difference.

Sampled per-process GPU-resident maxima were 606/608 MiB for Venturi and 480 MiB
for either llama.cpp configuration. The unchanged GNOME remote desktop process
(354 MiB) was explicitly allowed and recorded separately. These resident samples
must not be compared directly with the PyTorch-only peaks in the earlier table.
The GPU suite passed **232 tests with no skips**, with three expected Siphon
warnings; local CPU validation passed 190 tests with 42 CUDA cases skipped.

Raw paired results, precision/cache contracts, source hashes, conversion/build
logs, and the separately labeled upstream synthetic benchmark are retained in
[`benchmarks/llamacpp`](benchmarks/llamacpp/). Upstream synthetic decode measured
approximately 2,748/2,790 tokens/s, but excludes sampling and uses synthetic
inputs, so those numbers are not the paired greedy comparison above.

### Opt-in exact-length CUDA prefill/decode graphs

`--execution cuda-graph` prepares a reusable FP16 graph for the prompt's exact
length, plus decode graphs. The default remains `eager`. No weights are quantized or cast,
no GELU approximation is introduced, and no padded/fixed-capacity attention
replaces the actual KV prefix. One graph is captured per exact decode position.
This is execution specialization, not a new attention algorithm.

```bash
# Existing deployment, from the repository checkout on your Mac:
bash scripts/venturi-4090d benchmark --execution cuda-graph \
  --prompt "Once upon a time" --max-new-tokens 32 \
  --ignore-eos --warmup 2 --repeats 10 --json
bash scripts/venturi-4090d generate --execution cuda-graph \
  --prompt "Once upon a time" --max-new-tokens 32 --json
```

Capture is an up-front cost, not a free speedup. The benchmark reports it as
`preparation_seconds`, excluded from warmed request times. Generation JSON also
reports `generation_execution.request_including_preparation_seconds` and
`time_to_first_token_including_preparation_seconds`. One-shot CLI generation can
be slower overall despite faster prepared decode. Reuse a prepared session in a
long-lived Python application to amortize capture:

```python
# model and tokenizer are the CUDA FP16 objects from the Python API example.
prompt = "Once upon a time"
ids = tokenizer.encode(prompt)
session = model.create_cuda_graph_session(
    len(ids) + 32, decode_start=len(ids), decode_steps=31,
    prefill_lengths=(len(ids),),
)
result = generate_text(model, tokenizer, prompt, max_new_tokens=32, session=session)
# Subsequent requests reset logical KV history and reuse the captured graphs.
again = generate_text(model, tokenizer, prompt, max_new_tokens=32, session=session)
```

A prepared session requires the same model and owning CUDA stream. Every decode
position must lie in its prepared range, and the request must fit its KV capacity.
Prompt contents and token counts may change within those bounds. A wider plan
(for example, capacity 256, decode_start 1, decode_steps 255) supports different
prompt lengths without recapture, at greater up-front memory/capture cost.
EOS may stop early; reset/reuse is supported. A session is not safe for
concurrent calls. Preparation must run without concurrent CUDA work in this
process. Graphs have independent private memory pools so reset/early stopping
cannot violate shared-pool replay ordering; this costs memory proportional to
the number of prepared positions. At most 256 decode graphs (257 output tokens)
are supported per plan. Unprepared decode positions fail instead of silently
capturing more graphs during a request. Only CUDA FP16 is supported for now.

The Python API's optional `prefill_lengths` defaults to `()`, preserving the
previous eager-prefill behavior. Select up to eight distinct positive lengths
that fit capacity; each gets a separate graph and private memory pool. Prefill
graphs apply only at position zero with `all_logits=False`. Different prompt
contents reuse the same length's graph; unprepared lengths, cached multi-token
chunks and all-logits prefills fall back to eager execution without recapture.
No padding, truncation, prefix reuse or approximate attention is introduced.
A blocking CPU-to-static-input copy avoids asynchronous staging-buffer reuse
hazards. Preparation and memory costs grow with the selected shapes.

`forward` returns borrowed graph logits, overwritten when the same graph is
replayed; clone them if they must outlive request reuse. Captured greedy
selection retains the nonfinite-logit check, returning a checked token with one
host read rather than separately synchronizing finite-check and argmax results.
The session is poisoned on a captured nonfinite selection and must be reset.
No CPU synchronization or data-dependent Python control flow runs inside
capture. See the [PyTorch 2.9 CUDA graph contract](https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-graphs).

For numerical validation and trace collection on the CUDA host:

```bash
python tools/run_cuda_graph_acceptance.py /absolute/path/to/pythia-70m \
  --prefill-graphs --output /absolute/path/to/graph-acceptance.json
python tools/profile_decode.py /absolute/path/to/pythia-70m \
  --execution cuda-graph --output-dir /absolute/path/to/new-profile
```

The acceptance tool requires complete same-position FP16 logit vectors and
greedy IDs to be bit-exact with the eager runner on its tested workloads. This
does not claim equality to FP32 or llama.cpp. Kineto may collect CPU operators
without CUDA kernel activity on some installations; the report explicitly
marks missing CUDA activity unavailable. `--nsys` instead brackets warmed
decode with `cudaProfilerStart/Stop` for an external Nsight Systems capture.
Instrumented timings must not be substituted for unprofiled benchmark results.

#### Earlier decode-only FP16 optimization result

Final paired RTX 4090D run: two warmups and ten measured fixed-length requests
per workload, with model/KV allocation and graph preparation outside warmed
timings in both engines:

| Output tokens | Venturi eager | Venturi prepared graphs | Paired llama.cpp | Gain over eager |
| --- | --- | --- | --- | --- |
| 32 | 253.96 tokens/s | 2,515.56 tokens/s | 1,742.18 tokens/s | 9.91x |
| 128 | 257.85 tokens/s | 2,579.64 tokens/s | 1,793.73 tokens/s | 10.00x |

This is approximately **1.44x the paired llama.cpp warmed decode rate** on these
tiny Pythia workloads, not a general engine/larger-model speed claim. Complete
same-position FP16 logit storage and greedy IDs match eager execution on 32- and
128-token acceptance cases; llama.cpp/FP32 numerical equivalence is still not
claimed. Different prompt lengths within a prepared range, EOS/reset, stream
ownership, nonfinite selection, and no hot-path allocator growth are tested.
The GPU suite passed 246 tests without skips; local CPU validation passed 200
tests with 46 CUDA cases skipped. The wheel's 24 source/attribution assets were
checked against the working tree.

Nsight traces show why this helps: eager decoding issued 153 separate kernel
launch API calls and three stream synchronizations per token. The prepared path
issues one input-fill kernel launch plus one graph launch, and one stream
synchronization per token. It still executes 156 GPU kernels per token versus
153 in eager mode: the improvement removes repeated host dispatch/input-copy
overhead, **not** the underlying model computation. GPU kernel fusion remains
a separate optimization opportunity.

Tradeoffs are explicit: preparation cost 0.616/1.365 seconds, and sampled
resident GPU memory increased from 606/608 MiB to 764/1,134 MiB for the two
workloads. Reused-plan TTFT remained approximately 4.1 ms versus llama.cpp's
1.1 ms; prefill was not optimized in that earlier measurement. A cold one-shot request can be slower than
eager generation. Using the measured request times, capture amortizes after
roughly six/four matching fixed-length requests, not necessarily after one.

Final raw comparisons, byte-level acceptance, CLI startup-inclusive metrics,
source hashes, CPU operator tables, and Nsight API/kernel statistics are in
[`benchmarks/cuda-graph`](benchmarks/cuda-graph/). An earlier full pass measured
2,482/2,601 tokens/s with the same optimization; the table uses the final pass,
not a best-of selection across runs. Binary traces remain on the GPU host.

#### Basic prefill improvements

Next-token execution now normalizes only the final hidden row before the LM
head. `all_logits=True` still normalizes/projects all rows. This eliminates
unused work without changing the per-token normalization equation; it did not
produce a separately established latency gain in this small-model test.

The prefill CPU operator trace already contains six FlashAttention calls, so
the measured change is **exact-shape graph replay and captured checked-token
selection**, not installing a missing attention algorithm or fusing kernels.
The same preparation strategy can be used by future adapters, but only the
existing GPT-NeoX adapter executes models today.

RTX 4090D, Pythia-70M FP16, synthetic prompt IDs, eight CPU threads, reused
sessions, ten warmups and fifty requests per process. Two sequential passes
ran eager/graph/graph/eager, with no concurrent engine processes. The table
combines all 100 measured requests per mode, rather than selecting a best run:

| Prompt tokens | Eager median | Prepared median | Median latency reduction |
| --- | --- | --- | --- |
| 16 | 4.036 ms | 0.473 ms | 8.53x |
| 128 | 5.207 ms | 0.546 ms | 9.54x |
| 512 | 5.380 ms | 0.816 ms | 6.60x |
| 1024 | 5.569 ms | 1.160 ms | 4.80x |

These times include prefill and finite-checked first-token selection, excluding
tokenization, loading, preparation and later decode. Host dispatch is variable:
individual eager-process medians ranged from 4.03 to 5.60 ms across these
workloads. Earlier exploratory runs are retained too. This is not a claim of
universal speedup, faster large-model GEMM, or a new llama.cpp comparison.

Graph preparation cost 0.406–0.411 seconds for the first selected shape in each
fresh process; subsequent shapes cost 0.032–0.054 seconds. Each measured graph
plan also included one decode graph for an untimed populated-KV check. Its
PyTorch live allocation ranged from 169.7 to 208.9 MiB across shapes, versus
169.5–181.3 MiB eager baselines; these are not whole-process GPU residency.
Cold CLI setup is reported separately and can dominate a single request.

All 24 full-vocabulary FP16 vectors per benchmark run matched the pre-change
code byte-for-byte: original/changed/reset prompts, prefill and one cached
continuation, at each length. Separate 32/128-output-token acceptance checked
all 160 teacher-forced rows and complete greedy sequences against eager.
Tests also cover all-logits/cached-chunk fallback, queued replays, zero-decode
plans, EOS, nonfinite poisoning, stream ownership and no hot GPU allocations.

```bash
python tools/benchmark_prefill.py /absolute/path/to/pythia-70m \
  --execution cuda-graph --warmup 10 --repeats 50 \
  --output-dir /absolute/path/to/new-prefill-results
```

Raw runs, source hashes, numerical checks and test results are retained in
[`benchmarks/prefill`](benchmarks/prefill/). Run isolated from other GPU work;
the benchmark never downloads weights, installs packages or changes GPU clocks.

### RTX 4090D generation validation

The real-checkpoint run passed **209 tests with no skips**, including previous
loader/operator checks and the new generation tests. Local CPU validation passed
168 tests, with 41 opt-in CUDA cases skipped. The wheel's 22 source/attribution
assets were verified against the repository.

All three 16-token greedy continuations matched the separate same-precision
FP16 reference, using the same FP32-frequency RoPE contract. At position 512,
cached FP16 and FP32 logits matched their respective same-schedule references
exactly in this run. **The continuations did not match FP32 greedy decoding.**

Long-context cached-versus-full-batch drift is retained as a known limitation:
FP16 raw-logit maximum difference was 4.5, and cached FP32 versus full eager
reference maximum difference was approximately 0.0745. Both exceeded their close
comparison bounds; they are explicitly diagnostic, not claimed as passing
batch-invariance checks. Same-schedule reference checks and independent FP64
mathematical tests provide separate correctness evidence.

See [`benchmarks/generation/validation.json`](benchmarks/generation/validation.json),
the [raw reference report](benchmarks/generation/rtx4090d-pythia70m.json), and the
[CLI output sample](benchmarks/generation/cli-sample.json). These results establish
the first working dense text-generation path, not serving performance or model
answer quality.

## Reusable Ada Q4 Linear backend

This is a standalone operator plugin, not a Qwen runner or a universal 4-bit
reader. It takes a weight matrix and activations without consulting layer names,
model configs, attention implementations, or speculative decoders.

### Supported contract

- Linux CUDA, compute capability **8.9**. RTX 4090D is the validation target;
  another RTX 4090 still needs a portability smoke test and its own measurements.
- Signed two's-complement Q4 codes in `[-8, 7]`, low nibble first, group size 64,
  FP16 scales, no zero points, K padded to 128: `q4_g64_fp16` /
  `row_split_k128_v1`. Scale multiplication and accumulation use FP32.
- Contiguous FP16 or BF16 activations `x[T,K]`, same-dtype output `out[T,N]`;
  computes `out = x @ W.T`, without bias or activation fusion.
- General SIMT route handles arbitrary positive N/K, including partial groups,
  row tails, and T up to 262,140. It is a functional baseline, **not a tuned
  high-throughput prefill implementation**.
- Retained Ada MMA small-token schedule: BF16, T=2..16, N divisible by 40,
  K in {5120, 6144, 25600}. Unsupported tuned requests fail explicitly; `auto`
  preserves this route where applicable.
- Dimension-independent MMA route: BF16, N divisible by 32, K divisible by 512,
  arbitrary supported T, using runtime dimensions and 16-token tiles. No model
  shape table is used. `auto` selects it for T>=2 when the retained route does
  not apply; `route="mma"` selects it explicitly, including T=1. Other operands
  use SIMT. Neither selection policy promises the fastest route on every card.

AWQ, GPTQ, GGUF, FP4, and other group sizes/packing conventions are **not**
interchangeable with this contract. They require explicit format adapters.
The current adapter can view a single encoded Q4 payload, but cannot open a
complete `.ninfer` artifact. No quantizer or model conversion is implicit.

### Build explicitly on the CUDA server

The loader and CPU tests do not compile CUDA. The optional backend uses a small
C ABI, avoiding a dependency on PyTorch's C++ ABI or Ninja. A CUDA toolkit with
Ada support and a compatible C++20 host compiler are required.

```bash
python -m venturi.ops.build --output-dir /absolute/path/to/venturi-native
# Prints the versioned path to libventuri_ada.so.
```

Compiler output and source/binary hashes are retained alongside the library in
`build.log` and `build.json`. Native `sm_89` code and `compute_89` PTX are included.
Building is never triggered by importing Venturi or by running an operator.

### Prepare once, execute without device allocations

```python
import torch
from venturi.ops import AdaBackend, Q4RowSplitWeight

# Obtain these planes from an explicit compatible checkpoint adapter.
# uint8 codes: [N, padded_K/2]; float16 scales: [N, padded_K/64].
weight = Q4RowSplitWeight(codes, scales, in_features=K)
backend = AdaBackend("/absolute/path/to/libventuri_ada.so")
plan = backend.prepare_linear(weight, tokens=8, dtype=torch.bfloat16, device="cuda:0")

x = torch.randn(8, K, dtype=torch.bfloat16, device="cuda:0")
out = plan.empty_output()  # Explicit allocation, before timing/capture.
plan.run(x, out=out)
```

`Q4RowSplitWeight.from_codes(integer_codes, fp16_scales)` packs CPU integer codes
exactly for tests/converters; it does not quantize floating-point weights.
`from_payload(uint8_payload, out_features=N, in_features=K)` views the code plane
followed by its 256-byte-aligned scale plane, validating the exact byte count.
This can wrap a compatible tensor loaded through Siphon without coupling its I/O
to native execution.

Preparation owns independent device copies, checks finite scales, initializes
the selected native route, and synchronizes preparation. It must run outside
CUDA graph capture. A plan has fixed T, dtype, and weight dimensions; its GPU
workspace requirement is zero bytes. Plans retain their backend library.

`run` writes the caller's output on PyTorch's **current stream for the plan's
device**. It performs no tensor allocation, full-weight dequantization, implicit
cast, synchronization, or shape-dependent compilation. It rejects noncontiguous
operands and output aliasing. Both MMA routes require 16-byte-aligned input.
There is no autograd implementation. Prepare and warm up before timing/capture.

The caller owns stream dependencies and lifetimes: keep the plan, inputs, and
outputs alive until asynchronous work finishes, and keep captured operands and
the plan alive across graph replays. When handing tensors between streams, use
PyTorch's `wait_stream`/events and `record_stream` or retain them explicitly;
Venturi does not silently synchronize every invocation. Do not mutate weight
contents or resize operands while they are in use. Independent outputs can be
used with the same prepared read-only weights on different streams.

Python is the orchestration layer, not a claim that per-layer Python dispatch is
free. A future model runner should capture its steady-state execution or use a
native execution plan. The benchmark reports eager and graph latencies separately
so Python launch overhead is not hidden behind kernel-only numbers.

### Native correctness and operator benchmarks

```bash
VENTURI_ADA_LIBRARY=/absolute/path/to/libventuri_ada.so python -m pytest -q
python tools/benchmark_q4_linear.py \
  --library /absolute/path/to/libventuri_ada.so \
  --output /absolute/path/to/q4-linear.json
```

Tests independently decode the signed codes and stored scales into FP64 and
compare native outputs, including non-Qwen dimensions, K/row/token tails, route
boundaries, full-size operators, nondefault streams, CUDA graph replay, owned
weight preparation, and execution allocation checks. Without the environment
variable, native tests skip and never invoke a compiler or allocate CUDA weights.

The benchmark uses **synthetic weights** and reports eager event latency and
amortized CUDA-graph operator latency, alternating `auto`/`generic` ordering.
It is not a tokens/second measurement, an inference benchmark, or evidence of
a speedup against llama.cpp. A full-model comparison also needs matched target
weights/quality, prompts, context, output length, and speculative settings.

### RTX 4090D validation

At the Q4 operator milestone, all **145 tests passed** on the RTX 4090D, including the Siphon/FP16
loader acceptance tests. Native tests cover all three operator routes. The local
CPU suite passes 107 tests and skips the 38 opt-in CUDA cases. Packaged native
sources and attribution files were checked against the source tree.

Selected BF16 graph operator latencies (synthetic Q4 weights, five repetitions):

| N | K | T | Route | Median latency |
|---:|---:|---:|---|---:|
| 4096 | 4096 | 8 | Dimension-independent MMA | 12.18 microseconds |
| 5120 | 25600 | 8 | Retained small-token MMA | 76.97 microseconds |
| 5120 | 25600 | 128 | Dimension-independent MMA | 412.70 microseconds |

Execution showed zero additional PyTorch device allocations in the measured
routes. These are **operator microbenchmarks**, not model throughput; graph
latency excludes ordinary per-operator Python launch gaps. Raw eager/graph samples,
toolchain information, correctness/test scope, and tested source hashes are in
[`benchmarks/q4/validation.json`](benchmarks/q4/validation.json) and
[`benchmarks/q4/rtx4090d-linear.json`](benchmarks/q4/rtx4090d-linear.json).
The second physical RTX 4090 portability test has not been run.

Selected native schedules/helpers originate in NInfer, Cinference, and the Ada
research code on the 4090D. Their scoped Apache-2.0 license, notices, source
hashes, and adaptation notes are retained under
`src/venturi/ops/csrc/third_party/ninfer/`; no repository-wide license is added.

## Install

Python 3.10+ is required. Install the appropriate PyTorch build for your device
first, then install Venturi:

```bash
git clone https://github.com/123123213weqw/Venturi.git
cd Venturi
python -m pip install -e '.[test]'
```

For production loading, install Siphon from this repository, not an unrelated
package with the same name. Follow its platform/build prerequisites:

```bash
git clone --recurse-submodules https://github.com/123123213weqw/siphon.git
python -m pip install ./siphon
```

Siphon is imported only when its backend is selected. It is not vendored or
installed automatically. See [Siphon's installation instructions](https://github.com/123123213weqw/siphon#install).

## Inspect without loading GPU weights

```bash
venturi inspect /absolute/path/to/checkpoint --json
venturi inspect /absolute/path/to/weights.safetensors --json
```

Directory discovery uses:

1. `model.safetensors.index.json`, if present; otherwise a unique
   `*.safetensors.index.json` file.
2. `model.safetensors`, if there is no index.
3. All `*.safetensors` files in the directory otherwise.

An index is authoritative: unrelated shards are ignored, missing entries/shards
and duplicate tensor names are rejected. Multiple noncanonical indexes are
ambiguous and rejected. HF cache symlinks are supported. An explicitly selected
file loads only that file, even if a sibling index describes other shards.
Empty shards are rejected; empty tensors are valid on the reference backend.

## Load through Siphon

```bash
venturi load /absolute/path/to/checkpoint --device cuda:0 --json
```

Siphon is the default. Its automatic I/O selection and `SIPHON_*` environment
settings remain available. Explicit options override matching defaults:

```bash
venturi load /absolute/path/to/checkpoint --device cuda:0 \
  --siphon-backend URING --buffer-size 268435456 --json
```

Backend names are translated into Siphon's public enums. Comma-separated
candidates and the `BUFFERED` policy are accepted. Siphon is opened once for the
whole shard group, on the selected CUDA device, with `copy=True`: retained weights
must not alias a temporary rotating buffer.

The buffer option controls Siphon's ring allocation, not a cap on total model
memory. Allow headroom for the complete tensors **and** loader buffers; Siphon can
round or enlarge its buffer. No automatic CPU fallback, offloading, or eviction of
other workloads occurs. Device/storage types unsupported by the selected backend
fail rather than being converted.

The CLI loads, reports, and exits. Keep weights resident with the Python API:

```python
from venturi import inspect_checkpoint, load_checkpoint

checkpoint = inspect_checkpoint("/absolute/path/to/checkpoint")
model = load_checkpoint(checkpoint.directory, device="cuda:0")

print(model.config)  # Raw optional metadata, not an architecture config class.
for name, tensor in model.weights.items():
    print(name, tensor.shape, tensor.dtype)
print(model.stats.tensor_bytes)
```

The tensor mapping is read-only; tensor contents remain mutable. No tokenizer,
recurrent state, KV cache, or inference kernel is created.
For an explicitly selected single file, pass that same file to `load_checkpoint`
rather than its parent directory.

Use the reference loader explicitly for CPU validation:

```bash
venturi load /absolute/path/to/checkpoint --loader safetensors --device cpu --json
```

## FP16 acceptance: load, verify, and measure

Ordinary FP16 safetensors are the first full-checkpoint acceptance target. Require
FP16 floating-point storage without casting or modifying integer/bool buffers:

```bash
venturi verify /absolute/path/to/checkpoint --device cuda:0 \
  --require-float-dtype F16 --buffer-size 67108864 --json
```

The dtype contract is checked **before CUDA setup**. It is a storage-dtype check,
not proof that arbitrary integer tensors are unquantized model buffers. Verification
compares every stored byte, including NaN payloads, with the CPU safetensors
reference. It copies at most `--chunk-bytes` (default 16 MiB) back from the GPU at
a time. Loading and verification times are reported separately. A mismatch exits
nonzero and identifies the tensor and failing byte offset.

Use the same verifier from Python while retaining the loaded weights:

```python
from venturi import load_checkpoint, verify_loaded_checkpoint

model = load_checkpoint("/absolute/path/to/checkpoint", require_float_dtype="F16")
verification = verify_loaded_checkpoint(model)
print(verification.verified_tensors, verification.verified_bytes)
```

For repeated acceptance measurements in fresh processes, alternating loader order:

```bash
python tools/run_fp16_acceptance.py /absolute/path/to/checkpoint \
  --device cuda:0 --runs 3 --output /absolute/path/to/acceptance.json
```

Every measured run performs a complete byte comparison. This is a model-loading
measurement, **not** an inference benchmark. No page cache is evicted, so results
must not be described as controlled cold-start performance. The JSON memory
figures cover the PyTorch allocator, not total VRAM or native Siphon allocations.

The pinned real-model fixture is documented in
[`benchmarks/fp16/pythia-70m-source.json`](benchmarks/fp16/pythia-70m-source.json).
It contains original FP16 weights plus U8 attention-mask buffers; no conversion
from a quantized checkpoint is involved. Model weights are not committed and no
model is downloaded automatically by the loader or default tests.

### Verified result: RTX 4090 D

The full Pythia-70M checkpoint passed on an RTX 4090 D with Python 3.12.3,
PyTorch 2.9.1+cu130, and safetensors 0.8.0. Its 94 tensors contain 88 FP16 tensors
and six U8 attention-mask buffers, totaling **166,019,180 storage bytes**. The
downloaded file's SHA-256 was checked against the pinned Hugging Face source.

| Loader | Runs | Median synchronized loading | Exact-byte verification |
| --- | ---: | ---: | --- |
| Siphon | 3 | 279.35 ms | Every tensor, every run |
| safetensors reference | 3 | 275.39 ms | Every tensor, every run |

The Siphon runs reported 160.08 MiB peak PyTorch allocated memory and 164 MiB peak
reserved memory. These are **not total VRAM peaks**: native buffers and the CUDA
context are excluded. The complete server suite passed **86 tests with no skips**;
clean CPU CI also passed on Python 3.10 and 3.12.

These small-checkpoint measurements do not establish a Siphon speed advantage,
controlled cold-start performance, or inference throughput. Loading and
verification are separate measurements; Python process startup is excluded.

See the [raw runs](benchmarks/fp16/rtx4090d-pythia70m.json) and
[validation record](benchmarks/fp16/validation.json).

To acquire the exact acceptance fixture explicitly (the loader never downloads it):

```bash
set -eu
export MODEL_DIR=/absolute/path/to/pythia-70m
export REV=a39f36b100fe8a5377810d56c3f4789b9c53ac42
mkdir -p "$MODEL_DIR"
curl -fL --retry 3 -C - \
  "https://huggingface.co/EleutherAI/pythia-70m/resolve/$REV/model.safetensors" \
  -o "$MODEL_DIR/model.safetensors.partial"
printf '%s  %s\n' \
  ebfa4e2f18696ebd83716a0d39fe2c025f2ff8483f72a83ca59c475692fc9d15 \
  "$MODEL_DIR/model.safetensors.partial" | sha256sum -c -
# Rename only after successful checksum verification.
mv "$MODEL_DIR/model.safetensors.partial" "$MODEL_DIR/model.safetensors"
curl -fL --retry 3 \
  "https://huggingface.co/EleutherAI/pythia-70m/resolve/$REV/config.json" \
  -o "$MODEL_DIR/config.json"
python tools/run_fp16_acceptance.py "$MODEL_DIR" --runs 3 \
  --source-manifest benchmarks/fp16/pythia-70m-source.json \
  --output /absolute/path/to/acceptance.json
```

The download commands target a new fixture directory; existing model weights do
not need to be overwritten or converted. On macOS, use `shasum -a 256 -c -` in
place of `sha256sum -c -` for the checksum step. Use a configured CUDA/Siphon
environment for GPU acceptance.

## Tests

```bash
python -m pytest -q
```

The default suite uses architecture-independent synthetic tensors, with mixed
floating-point, integer-packed quantized, boolean, scalar, and empty storage. It
requires no model download, GPU, Siphon extension, or Rust compilation. Tests
cover exact contents and metadata, shard/index validation, lifetime, and CLI.
With the `generate` and `reference` extras installed, it also exercises local
tokenization and model/oracle tests without downloading any model checkpoint.

On a GPU host with Siphon installed:

```bash
VENTURI_TEST_CUDA=1 python -m pytest -q -m cuda -k real_siphon
```

This runs mixed-dtype round trips through Siphon's automatic, `MMAP`, and
`BUFFERED` paths and verifies every tensor against the CPU originals after the
loader closes. This is a storage correctness test, not a model benchmark.

Optional acceptance for any existing checkpoint with sufficient free VRAM:

```bash
VENTURI_TEST_CUDA=1 VENTURI_MODEL_PATH=/absolute/path/to/checkpoint \
  python -m pytest -q -m cuda -k full_checkpoint
```

The default test suite never downloads a checkpoint. Real generation tests
require `VENTURI_GENERATION_MODEL` and a matching local `tokenizer.json`.

## Measurement semantics

- `tensor_bytes`: physical storage bytes, not estimated unquantized model size.
- `storage_elements`: physical elements, not logical parameter count. One packed
  integer can contain multiple quantized model values.
- `total_seconds`: inspection plus synchronized loading; excludes Python process
  startup and imports performed before the API call.
- `load_seconds`: includes CUDA setup/synchronization when applicable.
- `cuda_allocated_bytes_delta`: PyTorch allocator change, not total or peak VRAM;
  native allocations invisible to PyTorch are excluded.
- CLI `cuda_allocator`: allocated/reserved bytes and process-lifetime allocator
  peaks sampled after loading. Native allocations and CUDA context are excluded;
  the library does not reset caller-owned peak counters.

## Layout

```text
src/venturi/checkpoint.py   Header/index inspection and opaque metadata
src/venturi/dtypes.py       Storage dtype descriptions, without conversion
src/venturi/loader.py       Siphon/reference loading and owned tensor container
src/venturi/verification.py Bounded, exact byte comparison with source tensors
src/venturi/ops/            Optional reusable Ada quantized Linear backend
src/venturi/models/         Explicit model adapters and per-request KV state
src/venturi/generation.py   Local tokenizer and bounded greedy generation
src/venturi/cli.py          inspect / load / verify / generate
tools/                     Loader, operator, and generation acceptance tools
tests/                     Storage, operator, model/oracle, and opt-in CUDA tests
```

## Independent checkpoint/vocabulary bindings and GDN milestone

The independent path now has strict checkpoint bindings, dense embedding lookup,
final RMSNorm/output-head composition, and a separate Gated DeltaNet recurrence.
These are **not yet a complete independent Qwen-27B model adapter**. The existing
external Qwen runner remains unchanged by this milestone.

### Bind the actual Qwen vocabulary without dense expansion

Inspection of the current 27B artifact finds Q8/group-32 embedding and output
matrices, both `[248320,5120]`, not Q4. `NInferVocabulary` validates both named
bindings, shapes and physical formats before upload. Reads are bounded to 16 MiB;
signed codes and FP16 scale planes are preserved. Shared object IDs are uploaded
once, so tied vocabulary storage is not duplicated. Each actual untied matrix
uses 1,350,860,800 bytes (codes plus scales); lookup and projection plans share it.
There is no full-vocabulary BF16/FP32 materialization on the native path.

```python
import torch
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend, NInferVocabulary

backend = CudaBackend("/absolute/path/to/rebuilt/libventuri_blocks.so")
artifact = NInferArtifact("/absolute/path/to/qwen3_8_27b.ninfer")
vocabulary = NInferVocabulary.upload_from_ninfer(artifact, backend)
lookup = vocabulary.embedding.prepare_embedding([1, 42])  # Host-validated IDs.
embedding = torch.empty((2, 5120), dtype=torch.bfloat16, device="cuda:0")
lookup.run(out=embedding)
head = vocabulary.head.prepare(tokens=1)
logits = torch.empty((1, 248320), dtype=torch.bfloat16, device="cuda:0")
head.run(embedding[-1:], out=logits)  # Vocabulary plumbing test, NOT model output.
```

Dense `ResidentLinear.prepare_embedding(ids)` likewise shares its owned matrix
with output-head plans. IDs are uploaded during preparation; native `run()` is
allocation-free and capture-safe. Plans and buffers must outlive queued work;
private uploaded IDs and resident weights are immutable by contract. Invalid
host IDs are rejected before upload; kernels also bounds-check IDs defensively.
Q8 output projection is a straightforward FP32-accumulating dot-product baseline,
not an optimized prefill GEMM. Its explicit Torch reference dequantizes bounded
512-row tiles; that is a diagnostic baseline, not a competitive quantized engine.

### GDN is recurrence, not ordinary attention

The artifact has 48 `linear_attention`/GDN layers and 16 full-attention layers.
GDN uses 16 key/query heads, 48 value heads and 128-dimensional keys/values.
`backend.prepare_gdn(tokens=T, qk_heads=16, value_heads=48)` accepts contiguous
BF16 `q/k[T,16,128]`, `v[T,48,128]`, FP32 transformed `log_decay/beta[T,48]`,
and explicit FP32 state `[48,128,128]` in **value,key** order. Each layer's state
is 3 MiB, independent of the number of processed tokens. This is not the total
model cache size: full-attention KV and convolution history are separate.

For each head/token, with optional FP32 L2-normalized Q/K (`epsilon=1e-6`):

```text
A       = exp(log_decay) * state
state'  = A + outer(beta * (v - A @ k), k)
output  = (state' @ q) / sqrt(key_dim)
```

`plan.run(q,k,v,log_decay,beta,state=state,out=out,state_out=next_state)` supports
exact state/state_out in-place aliasing and rejects other writable overlaps.
Gates must already be transformed, finite and stable; no sigmoid, softplus,
clamping, normalization-rounding change or hidden state reset is implicit.
FP32 normalization is deliberate: HF's optional BF16-normalization boundary is
not bit-identical. A separate raw-Q/K recurrence test checks the independent
Transformers implementation with state orientation/head expansion explicit.
The caller owns state reset/commit/stream dependencies and must discard state
after failed execution. Independent streams need independent writable buffers.

### Dense checkpoint-to-generation fixture path

`LlamaWeights.bind(loaded)` validates classic HF Llama tensor names, exact
shapes/dtypes, finite values and tied-head consistency **before uploading**.
Its explicit adapter composes the independent decoder layers, embeddings, final
norm and head with one shared per-session layer arena, dense per-layer KV,
chunked prefill, and greedy generation. BF16 is supported; FP64 is an explicit
reference-fixture mode. No real Llama checkpoint was available or downloaded.
The path is qualified on generated single/sharded checkpoints, including an
independent full-sequence FP64 oracle and a Transformers tiny-model fixture.

```bash
PYTHONPATH=src python -m venturi llama inspect /absolute/path/to/classic-bf16-llama
PYTHONPATH=src python -m venturi llama generate /absolute/path/to/classic-bf16-llama \
  --blocks-library /absolute/path/to/rebuilt/libventuri_blocks.so \
  --token-ids 1,42 --max-new-tokens 8
```

This eager adapter synchronizes chunk execution, allocates returned logits,
and prepares head/ID metadata outside capture. It is not an optimized whole-model
CUDA-graph path. Fused FP32 RMSNorm/SiLU arithmetic differs from Transformers'
BF16 intermediate rounding; whole-model BF16 bit identity is not promised.
RoPE scaling, independent head width, biases, Q/K normalization, hybrid/GDN
configs and quantized HF checkpoints are explicitly rejected, never reinterpreted
as classic Llama. The generic checkpoint loader remains architecture-independent.

### Qualification and remaining work

Build the blocks library explicitly on the target Linux SM89 GPU host. That milestone used **ABI 4**; the active native-Q8
work requires **ABI 8**, so rebuild before using the new frontend operators. No CUDA/Rust compilation was
performed on the Mac. Run the opt-in native tests with `VENTURI_BLOCKS_LIBRARY`
(and `VENTURI_ADA_LIBRARY` for existing Q4 decoder regression tests).

`tools/qualify_checkpoint_gdn.py` tests actual Qwen Q8 bytes against independent
row decoding and **all 248,320 output logits against bounded CPU FP64 math**.
GDN is qualified with synthetic represented inputs at the artifact's actual head
dimensions, for T=1 and T=128, against independent FP64 outputs and final state.
Its matched unprofiled ABBA phases use identical inputs/precision, 3 warmup calls,
and >=2 seconds per phase, and retain CPU time, wall/event latency and NVML data.
These are primitive calls, not full-model generation TPS or llama.cpp comparisons.
GPU event intervals can include host submission gaps. CPU usage includes CUDA
waiting; GPU busy is not achieved occupancy. Hardware counters remain unavailable
under `RmProfilingAdminOnly=1`; no profiling permissions/settings were changed.

Next for independent **Qwen** generation: causal depthwise convolution and
per-request convolution history; gate projections/transforms; gated output and
zero-centered `(1+weight)` normalization; packed multipart Q4 projection bindings;
Q/K norm and attention output gate; partial/multimodal RoPE rules; then the hybrid
layer/session adapter and full checkpoint correctness/quality/performance tests.
Do not infer Qwen generation correctness from vocabulary lookups or synthetic GDN.


## Active native Q8 Qwen goal

The implementation goal is active; full independent Qwen inference and matched
single/parallel-serving wins are **not complete**. See
[`docs/native-q8-goal.md`](docs/native-q8-goal.md) for the acceptance matrix,
source acquisition, hardware capacity and ordered milestones.

Current additions include `Q8RowSplitWeight.from_dense` (explicit bounded-row
source quantization), generic native Q8 projection/embedding residency, native
causal convolution/history, gate transforms, staged gated RMSNorm, zero-centered
norm preparation and strict GDN constant bindings. These are reusable operators,
not yet an integrated Qwen model. Native runs are allocation-free with explicit
buffers; Python preparation/reference code is labeled separately. Rebuild the
blocks library explicitly for **ABI 8**. SM70/V100 remains unqualified.

The current all-Q8 text weight budget is about **26.6 GiB before cache/workspace**,
so it does not fit entirely on one 24 GiB 4090D. The existing mixed Q4/Q8 artifact
is not falsely relabeled as original-quality Q8. The original BF16 acquisition
job on WZU_Server uses a revision-pinned HTTPS source/mirror and checksum checks;
partial downloads are not a verified checkpoint. No running GPU service was
stopped to make room.

`tools/qualify_native_q8_bootstrap.py` preserves the initial Q8 prefill gap:
at `[17408,5120]`, T=128 the direct native baseline took ~20.53 ms versus ~0.728 ms
for a prepared, materialized FP32 reference. At T=1 native took ~168 us versus
~381 us. These are **primitive** timings, not model TPS or wins against a
competitive Q8 engine. Full-model correctness comes before performance claims.


### Packed GDN and projected-input mixer core

`PreparedGDN.run_packed` consumes chronological `[T, Q-flat | K-flat | V-flat]`
convolution output directly. It retains the existing FP32 recurrence contract,
without allocating or copying separate Q/K/V planes. Native packed/separate and
chunked execution are bit-identical in qualification; complete input storage is
checked for writable aliasing, including later tokens and K/V regions.

`ResidentGDNFrontend.prepare_mixer_core` composes native convolution/history,
gate transforms, packed recurrence and gated normalization. Caller-owned
`GDNMixerWorkspace`, convolution history and recurrent state are preflighted
before any launch; independent lanes must not share mutable buffers. The core
accepts already-projected QKV/a/b/z. It is **not** the complete GDN layer or Qwen
model: input/output projections, residual/MLP, hybrid full attention, model
session, trained-model quality and native parallel serving remain outstanding.
That packed milestone used blocks ABI 6; current work requires **ABI 8**. Earlier capsules remain historical.

At real Qwen GDN geometry with synthetic BF16 inputs, packed loads remove
**2.5 MiB of unpack scratch at T=128**. Fixed-state graph-wrapper wall medians
were 1.841 ms packed vs 1.847 ms with explicit unpack; T=1 was 22.17 vs 22.43 us.
This tiny latency difference is not a demonstrated meaningful speedup. The
recurrence remains the dominant cost. These are primitive calls, not model TPS;
see `benchmarks/gdn-packed/summary.md` for CPU/event/NVML evidence and limitations.


### Original Qwen BF16 -> separate Q8 storage

The new `QwenSourcePlan` covers every text weight in the pinned original
Qwen3.5-27B index: 851 text roles; 348 vision/MTP tensors explicitly excluded.
`tools/export_qwen_q8.py` performs CPU-only bounded-row conversion into a new
encoded Q8 artifact, preserving raw norms and head-interleaved Q/gate rows.
Sensitive small a/b projections remain BF16; recurrent gate constants are FP32.
It requires a complete, hash-checked source by default, rechecks hashes, refuses
existing outputs and never uses the mixed-Q4 artifact as a source substitute.
`QwenQ8Weights` preflights all runtime roles and reads one encoded matrix or
bounded floating parameter at a time. This is checkpoint binding/storage work,
not yet native full-model generation or an inference-speed improvement.
See `docs/native-q8-goal.md` and `benchmarks/qwen-source-bindings/summary.md`.


### Complete GDN decoder-layer operator composition

`ResidentQwenGDNLayer.from_q8` now composes the GDN layer's norm, projections,
convolution/history, gates, recurrent matrix update, gated norm, output/residual
and post-norm MLP/residual. It streams individual encoded matrices into owned
resident storage, shares weights across token-shape plans, preflights the entire
layer before state writes and uses explicit per-request state/per-lane workspace.
CUDA graph/no-fallback and independent-stream tests are included.

This is **not** the complete hybrid Qwen model, native scheduler or a trained
quality/performance win. A retained BF16/FP64 cascade counterexample is documented
alongside per-operator qualification; full trained-model quality is still open.
See `benchmarks/qwen-gdn-layer/summary.md` and `docs/native-q8-goal.md`.


### Qwen full-attention layer and original gated-norm precision

Full-attention layer composition now includes per-head query/gate packing,
zero-centered Q/K norms, default text-only partial/mRoPE, causal GQA, output gate
and the residual MLP. Dense cache ownership, visible-length commits, failure
poisoning and fixed-position CUDA capture reuse the common cache lifecycle.
These layer APIs are dense KV primitives, not paging or a serving scheduler.
The eager hybrid integration below is separate from a native production runtime.

The verified original checkpoint revealed FP32 GDN gated gamma. The converter
and native operator now retain that precision and promoted multiplication, rather
than copying the legacy BF16 artifact's rounding. The corrected payload is
28,601,618,432 bytes. Original source conversion/storage verification is now
complete; trained-model quality remains unverified. Current blocks are **ABI 10**
(the historical layer capsule used ABI 8). See
`benchmarks/qwen-attention-layer/summary.md` for checks and retained limitations.


### Hybrid Q8 text model/session correctness baseline

`ResidentQwenModel.from_q8` connects embeddings, every declared GDN/full-attention
layer, zero-centered final norm and output head. Native numerical kernels never
fall back to an external model or dense Torch forward. `QwenSession` owns request
history/state/KV, supports chunked ingestion and token continuation, reuses one
workspace per layer kind, and commits position only after the whole call succeeds.
Errors after execution starts require reset; GDN state is not truncatable.
The host dispatcher is synchronous Python, not yet native production scheduling,
sampling, whole-model graphs, continuous batching or paged KV.

The pinned original checkpoint is downloaded and the separately published Q8
artifact passed full hashes, all 449 floating tensors and 1,206 independently
quantized row samples across all 402 Q8 matrices. That is storage verification,
not trained logits/quality. The full suite reports 853 passed / 26 skipped on the
4090D. New integration checks use miniature original-format fixtures. Full model
residency needs ~26.64 GiB for weights alone, exceeding a 24 GiB 4090D; no implicit
offload or precision downgrade is allowed. Full-model quality, SM70/FP16 and
matched single/concurrent speed comparisons remain open.

See `benchmarks/qwen-hybrid-model/summary.md` for exact bytes, hashes, reproducible
source/tests, failure/reset semantics, occupied-GPU timing limitations and the
remaining acceptance work. No new throughput win is claimed.


### Explicit V100 SM70 / FP16 port

The native blocks have architecture-tagged builds (the first SM70 port used
ABI 9; the current lossless-parameter implementation is **ABI 10**). Build the SM70
variant explicitly on a Linux CUDA 12.x host (not the local Mac):

```bash
PYTHONPATH=src python -m venturi.ops.build --component blocks \
  --architecture sm70 --nvcc /usr/local/cuda/bin/nvcc \
  --output-dir /absolute/path/to/sm70-build
```

Use an explicitly selected FP16 activation profile. The default BF16 profile is
rejected on V100 before uploads; no automatic cast/offload fallback occurs.
Q8 planes stay encoded and FP32 gamma/gates/recurrent state remain FP32; small
BF16 a/b/conv weights explicitly convert to FP16. Ada Q4 is still SM89-only.
A library for one architecture cannot be silently loaded on the other target.

SM70 qualification covers Q8 lookup/projection, GDN and miniature hybrid sessions,
with operator oracles, graph/no-fallback and state/ownership/reset checks. The
selected real-checkpoint layer tool also checks actual Q8 layer weights, using
synthetic activations. None of these is a complete trained 27B generation or
quality result, and no speed win or achieved occupancy is claimed. See
`benchmarks/sm70-fp16/summary.md`. Full resident loading remains memory-guarded:
Q8 weights alone exceed the 4090D's 24 GiB and the currently free V100 memory.

The first V100 layer checks found small-parameter FP16 rounding, including one
convolution weight underflow. That historical casting profile was **not**
original-parameter fidelity or full-model quality approval. The ABI-10 fix below
preserves these parameters through FP32 storage/compute; the original artifact
bytes and the earlier counterexample capsule remain unchanged.


### Lossless small parameters for FP16 execution

Blocks **ABI 10** adds an explicit `upload_fp32_linear(DenseWeight(fp32),
dtype=torch.float16, device=...)` path and FP32-weight causal convolution.
Ordinary dense upload still rejects mismatched dtypes; no weight conversion is
silently inferred. Native execution uses FP32 parameters/products/accumulation
and one activation-output cast, without numerical Torch fallback or run-time
allocation. The simple direct projection kernel is a fidelity baseline, not an
optimized large GEMM claim. The operators are reusable outside Qwen.

The Qwen FP16 loader now widens small BF16 a/b/conv parameters losslessly to FP32,
while keeping Q8 planes encoded, activations/KV FP16 and gamma/gates/recurrent
state FP32. Original BF16 and explicit FP64 diagnostic profiles remain separate.
Memory planning accounts for the larger small parameters before uploads.

Real layer-0 checks on V100 compare **all 532,480** uploaded a/b/conv values with
the original-derived artifact: zero changed values or underflows. Real GDN and
attention layers at T=1/8 still pass 70 existing per-operator checks; tiny hybrid,
graph/no-fallback, source-tiny-value, ownership and failure/reset tests pass.
The 4090D full suite is 920 passed / 32 skipped; the selected V100 CPU/native
suite is 61 / 12. These are not complete trained-model generation/quality or
matched engine speed results. See `benchmarks/fp32-small-parameters/summary.md`.


### Explicit full-model streaming correctness mode

`StreamingQwenModel.from_q8` is a separate, deliberately slow mode. The resident
API never chooses it implicitly. It streams embedding, one declared layer at a
time, and output head in synchronized phases; a request retains only state/KV
and reusable per-kind workspaces. Native numerical layers remain owned CUDA
operators, with Q8 planes preserved and FP32 small parameters on the FP16 path.
Python handles loading, position-table setup and explicit KV copies. A fresh
owned attention cache imports each intended prefix; cache identity is never
rebound to unrelated weights. Errors poison the request until reset.

```python
import torch
from venturi.models import QwenQ8Weights, StreamingQwenModel
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend

weights = QwenQ8Weights.from_ninfer(NInferArtifact("/absolute/path/to/model.ninfer"))
model = StreamingQwenModel.from_q8(
    weights, CudaBackend("/absolute/path/to/libventuri_blocks.so"),
    dtype=torch.float16, device="cuda:0",
)
session = model.create_session(128, max_tokens=32)
logits = session.forward(prompt_token_ids)
session.close()
```

The complete 64-layer original-derived Q8 model has generated **Paris + EOS** on
V100 for the recorded 26-token chat smoke case. Peak PyTorch GPU allocation was
1,518,406,656 bytes **because weights were streamed**, not because the full model
is compressed into that memory. This is not a resident-runtime, native-sampler,
batched/paged serving, corpus-quality or throughput qualification. The diagnostic
tool uses explicit host CPU greedy sampling and optionally compares same-input
steps against an allocating CPU Torch numerical profile. That profile shares
the adapter; it is not an independent upstream architecture or original-BF16
quality reference. Both recorded steps passed the predeclared numerical gates:
identical top-1 tokens, relative logit L2 below 0.00263 and centered cosine above
0.999994. This two-token case does not qualify broader model quality. The final
4090D regression suite is 947 passed / 36 skipped; the full trained-model run
was on V100, not the 4090D. See `benchmarks/qwen-streaming/summary.md` for status,
source hashes, fixed gates and limitations. No speed win is claimed.


### Native GPU greedy selection (blocks ABI 11)

`prepare_greedy` is model-independent: contiguous FP16/BF16/FP32 logits, one or
more independent rows, explicit output/workspace, and lowest-index tie-breaking.
It returns `[token, status]` int64 pairs on the GPU. A nonfinite value invalidates
that row and writes `[-1, 1]`; valid rows have status 0. Check the status before
using a token. Native execution performs no Torch numerical fallback, hidden
allocation, CPU read or synchronization. Each concurrent invocation needs its
own buffers. SM70 still rejects native BF16. Rebuild explicitly for ABI 11;
historical ABI-10 evidence and libraries remain unchanged.

```python
from venturi.ops import prepare_greedy

selector = prepare_greedy(
    backend, rows=1, vocab_size=248320, dtype=torch.float32, device="cuda:0",
)
result = torch.empty(1, 2, dtype=torch.int64, device="cuda:0")
scratch = torch.empty(selector.workspace_bytes, dtype=torch.uint8, device="cuda:0")
selector.run(logits.view(1, -1), out=result, workspace=scratch)
# Explicit control-flow synchronization, outside CUDA graph capture:
token, status = result.cpu()[0].tolist()
if status:
    raise ValueError("Nonfinite logits; no valid token")
```

The streaming diagnostic tool defaults to `--sampler native-greedy`;
`--sampler cpu-greedy` preserves the explicit reference policy. Native selection
copies only a 16-byte result pair per row for host control flow, but the tool
still separately copies complete logits/traces for qualification. This is greedy
selection only, not temperature/top-k/top-p sampling, a device-side stop/input
loop, a native scheduler or whole-model graph qualification. See
`benchmarks/native-greedy/summary.md` for measured scope and results.

The ABI-11 full V100 streaming case again generated Paris + EOS. Both same-input
CPU-reference gates passed, and logits/all hidden/GDN/KV snapshots were byte-
identical to the previous native run. Final SM89 regression: 1011 passed /
42 skipped. This validates the greedy operator/integration, not a production
native control loop, broader quality, parallel serving or full-model speed win.

### Independent native Qwen runtime (blocks ABI 12)

`NativeQwenModel` now owns the complete C++/CUDA forward, GDN history/state,
dense KV, positions and greedy generation loop. Python only preflights metadata
and prepares small constants at load; native code owns copies and read-only
mapped Q8 source planes. A generation call has no Python layer/operator/token
callbacks. `venturi qwen-native` selects this path; the legacy `venturi qwen`
command remains the separate external backend.

Select weight residency explicitly:

- `resident`: upload and share all immutable Q8 weights once. Memory admission
  fails rather than silently offloading or lowering precision. The tested 27B
  FP16 profile owns approximately 28.66 GB of model GPU memory plus request
  KV/GDN/workspace and CUDA context; it does **not** fit a single 24 GB GPU.
- `streaming`: keep small constants resident and upload encoded large weights
  into a bounded arena through pinned staging. Only selected embedding rows
  transfer. This is an explicit low-VRAM correctness/offload mode, not the
  resident model's expected throughput.

SM70 requires explicit FP16 execution; native BF16 is rejected. Original small
BF16 parameters that require FP32 storage are preserved, not rounded to FP16.
Request state is independent, but a model currently serializes its private
stream: no continuous batching, paged KV or whole-model CUDA graphs are claimed.
Closing a Python model refuses new sessions; existing native sessions retain
their shared model. Execution failures poison a request until reset. Source
file identity/change checks remain enabled, including for resident mode.

Build on the remote CUDA host, never implicitly during inference:

```bash
LIB=$(python -m venturi.ops.build --component blocks --architecture sm70 \
  --output-dir "$HOME/codex-build/Venturi/blocks-sm70")
python -m venturi qwen-native generate /path/to/model.ninfer \
  --blocks-library "$LIB" --weight-mode resident --dtype fp16 \
  --token-ids "$TOKEN_IDS" --max-new-tokens 32
```

`TOKEN_IDS` must contain comma-separated IDs with the checkpoint's intended
prompt/chat formatting. Alternatively `--prompt` plus `--tokenizer` loads an
explicit local tokenizer; it does not automatically apply a chat template,
download dependencies or execute remote model code. Native generation is
greedy only; EOS policy is explicit. Diagnostic forward/snapshots return owned
CPU copies, not production token-stream timing.

The SM70 Q8 projection profile uses four prompt tokens per weight tile and four
output rows per single-token tile. The generic operator accepts arbitrary
supported matrix shapes; no Qwen layer numbers or model dimensions are baked
into the kernel. Per-output FP32 FMA ordering, reduction ordering, Q8 codes and
scales remain unchanged. This is SIMT reuse, **not** Tensor Core/INT8 GEMM.
Build flags `--q8-token-tile` and `--q8-row-tile` accept 1, 2, 4 or 8; 1/1
preserves original dispatch. Defaults are SM70 4/4, SM89 1/1 until a separate
SM89 full-model performance qualification. Tile choices are part of the build
cache identity, not an implicit runtime model/precision substitution.

See `benchmarks/v100-q8-optimization/summary.md` for resident/offload timing
scopes, exact-parity checks, tile regressions, CPU time, telemetry and separate
Nsight attribution. Projection microbenchmark graph replay does not establish
whole-model graph support. No external-engine or parallel-serving win follows
from these internal single-request comparisons.

### Explicit two-device projection sharding (blocks ABI 13)

Native Qwen can row-shard its resident Q8 projections across two explicitly
selected, compatible CUDA devices. Each shard evaluates complete dot products;
FP16/BF16 output rows are gathered without reducing rounded partial sums.
Embeddings and request GDN/KV state stay on the primary device. This is tensor
parallelism for **one request**, not simultaneous request serving. MTP is a
separate opt-in wrapper described below. Single-device execution remains the
default. This feature originated with ABI 13; current source requires ABI 15.

```bash
CUDA_VISIBLE_DEVICES=1,0 python -m venturi qwen-native generate /path/to/model.ninfer \
  --blocks-library "$LIB" --weight-mode resident --dtype fp16 \
  --tensor-parallel-device cuda:1 --tensor-parallel-transport host \
  --token-ids "$TOKEN_IDS" --max-new-tokens 32
```

Here logical `cuda:0` is the primary physical GPU 1, and logical `cuda:1` is
physical GPU 0. `host` explicitly stages both directions through portable pinned
host buffers using CUDA DMA; there is no CPU numerical fallback. The default
`peer` transport requires bidirectional peer capability **and actual byte-copy
probes** before large weight uploads. A failed probe stops loading and never
silently chooses another transport. This matters on the tested PCIe V100 pair:
CUDA advertises peer access, but a reverse-direction byte-copy probe fails.
No driver, IOMMU, ACS or profiling-permission settings were changed.

Session diagnostics report primary model/request bytes, secondary model/scratch
bytes, pinned host bytes and logical input/output transfer payload counters.
Host staging performs two physical DMA copies per logical inter-device payload;
these counters are not PCIe bandwidth measurements. Native private streams,
cross-device events and the model mutex protect shared projection scratch;
requests still serialize. Streaming weights with tensor parallelism are rejected.

Matched non-speculative single-request tests improve 14.9565 -> **17.1169 TPS**
on two V100s; 26-token-prompt TTFT improves 0.90099 -> **0.56229 s**. These are
internal one-/two-GPU results, not a 50 TPS or external-engine win.

The real-model two-step qualification preserves all previous-native logits,
hidden values and GDN/KV bytes and passes unchanged fixed CPU-reference gates.
See `benchmarks/v100-native-tp/summary.md` for matched unprofiled measurements,
separate multi-device profiler attribution, hardware limitations and regressions.
At that historical baseline, native MTP was not implemented. The ABI-15
one-proposal implementation and its separate qualification are described below.
The reported external 50 TPS used two GPUs plus MTP and a different checkpoint;
it is not a matched result for this non-speculative path.

### Native two-device pipeline parallelism (blocks ABI 14)

`NativeQwenPipelineModel` partitions contiguous layer ranges across two GPUs.
Each stage owns its resident Q8 projections, constants and the corresponding
per-request GDN/convolution/KV state. Only the boundary hidden tensor crosses
stages, through explicitly selected portable pinned-host DMA. The last stage
owns the output head and greedy selection. No CPU numerical fallback or Python
layer/token loop is used.

Rebuild the remote blocks library for **ABI 15**, including `qwen_pipeline.cuh`
and `qwen_mtp.cuh`. Older libraries are rejected. On the tested V100 pair:

```bash
CUDA_VISIBLE_DEVICES=1,0 python -m venturi qwen-native generate /path/to/model.ninfer \
  --blocks-library "$LIB" --weight-mode resident --dtype fp16 \
  --pipeline-device cuda:1 --pipeline-split-layer 32 \
  --pipeline-transport host --token-ids "$TOKEN_IDS" --max-new-tokens 32
```

Logical GPU 0 runs layers `[0,32)` and embeddings; logical GPU 1 runs layers
`[32,64)` and the output head. A different explicit split is supported; it must
leave at least one layer per stage. PP cannot be combined with the existing TP
option on these two devices. Single-GPU and TP APIs remain separate choices.
PP currently requires resident weights and explicit host transport; it never
tries the known-unreliable peer route or silently offloads weights.

Distinct request sessions can execute simultaneously on different stages.
Native locks serialize each individual stage, not the complete model. Greedy
selection stays within the final-stage lock so a request obtains its next-token
dependency before another request occupies that stage. Python can submit one
`session.generate(...)` call per request from separate threads; generation and
stage scheduling remain native. This is not continuous batching, paged KV,
HTTP serving or whole-model CUDA graphs. MTP requires the explicit wrapper below.

Snapshots route each layer to its owning device. A partial-stage failure poisons
the entire request until reset; resetting clears both stages. Closing the model
refuses new requests while existing sessions retain native weights. Shared model
bytes should be counted once, whereas request bytes belong to each session;
`peer_request_device_bytes` exposes the second stage's request allocation.
Tied embedding/head weights may need a copy on each stage's device.

See `benchmarks/v100-native-pp/summary.md` for correctness-first qualification,
matched TP/PP comparisons at 1, 2, 4 and 8 requests, CPU/latency/throughput/VRAM
accounting, separate CUDA timeline overlap and preserved initial-scheduler data.
A one-request PP path still traverses both stages sequentially: lower
communication alone does not guarantee lower single-request latency.

### V100 CUDA launch bounds and occupancy

The SM70 Q8 tiled projection now supplies a 256-thread compiler launch bound,
with six minimum resident blocks for the multi-token specialization and one
for decode. This is a compiler resource/scheduling hint, not a changed launch
size, arithmetic order, weight format or precision. SM89 does not receive the
hint. That experiment used ABI 14; current source requires an ABI-15 rebuild.

On the tested V100 binary, prefill uses 40 instead of 44 registers/thread and
its CUDA-API theoretical occupancy increases from 62.5% to 75%. Decode uses
56 instead of 64 registers, with the same 50% theoretical occupancy; both
specializations have no compiler-reported spills. Forcing decode to 75% or
100% theoretical occupancy instead introduced spills and substantial regressions,
so those experimental variants are not promoted.

Matched Qwen Q8/FP16 two-V100 tests, 23 prompt tokens and 32 greedy outputs,
improve single-request TP decode from 17.14 to 18.86 TPS and eight-request PP
aggregate decode from 28.31 to 33.19 TPS. TP first-token wall falls from 491 to
431 ms; complete eight-request PP cohort wall falls from 12.26 to 10.56 s.
These are internal, non-speculative measurements, not a matched llama.cpp win.

`tools/inspect_cuda_occupancy.py` hashes an explicitly supplied library, extracts
its cubins and queries the CUDA driver occupancy API. Optional native Nsight
SQLite traces add actual launch geometry and kernel attribution. The tool does
not launch kernels or obtain achieved occupancy. Driver counter access remains
restricted; NVML GPU busy is not achieved occupancy. Raw tests, compiler spill
reports, counterexamples and matched CPU/latency/throughput data are retained in
`benchmarks/v100-cuda-occupancy/summary.md`.

### Experimental native one-proposal MTP (blocks ABI 15)

`NativeQwenMTPModel` wraps a resident native target. Python handles loading and
submission only: draft forward, target verification, greedy selection and
transactional GDN rollback/replay remain native. The original checkpoint's 15
MTP tensors are exported separately; the unchanged target embedding/output head
are shared, not duplicated. This is checkpoint-specific MTP support, not an
arbitrary-model speculative decoder.

Export the separate draft payload from the same verified original BF16 source:

```python
from venturi.qwen_export import export_qwen_mtp_q8
export_qwen_mtp_q8("/path/to/verified-bf16", "/path/to/mtp-q8.ninfer")
```

After rebuilding the remote blocks library for the current ABI (16), enable it explicitly:

```bash
CUDA_VISIBLE_DEVICES=1,0 python -m venturi qwen-native generate /path/to/model.ninfer \
  --blocks-library "$LIB" --weight-mode resident --dtype fp16 \
  --tensor-parallel-device cuda:1 --tensor-parallel-transport host \
  --mtp-artifact /path/to/mtp-q8.ninfer \
  --token-ids "$TOKEN_IDS" --max-new-tokens 32
```

Use the existing pipeline options instead of the TP options to test PP. MTP is
**off by default** and currently supports one proposal, greedy decoding and
fresh/reset requests only. Longer chains, arbitrary continuation, sampling,
continuous batching, paged KV and whole-model graphs are not implemented here.
The tested two-V100 path uses original-derived Q8 weights and FP16 activations;
SM89 and other checkpoint families are not qualified by these results.

The matched short-prompt TP test improves single-request post-first-token rate
from **18.94 to 24.55 TPS**, with **87.5%** draft acceptance and exact target
output/committed-state checks. This is not a universal win: PP loses aggregate
throughput at higher request counts. Keep MTP opt-in, and compare complete-request
throughput separately from decode rate. Full 1/2/4/8-request results, CPU time,
latency, memory and separate profiling evidence are in
`benchmarks/v100-native-mtp/summary.md`. See `docs/native-mtp-plan.md` for the
implemented state transaction, acceptance gates and remaining limitations.

### Current optimization target: llama.cpp on two V100s

The next acceptance target is a **same-original-checkpoint llama.cpp comparison**
and measured end-to-end optimization, not another Venturi-only speed claim.
Screen 1/8 requests first, then promote with repeated 1/2/4/8-request tests,
correctness gates and explicit speculation/precision/timing scopes. See the
revised target in `docs/native-q8-goal.md`. Equal Q8 weight labels do not imply
identical encoded tensors or engine-internal activation math. The old uncensored
Qwen3.8 service's 50 TPS is not a same-checkpoint Qwen3.5 baseline.

`tools/serve_native_benchmark.py` is a loopback-only fixed-greedy benchmark adapter,
not a production inference API. `tools/benchmark_completion_api.py` sends the
same pretokenized fixed-work requests to it or llama-server and measures global
HTTP cohort wall separately from engine-reported internal timings. The native
adapter's scalar route submits one owned C++ generation call per request; its
batch route submits one per cohort. Neither adds a Python model/token loop.
Initial comparison evidence is kept separately in
`benchmarks/v100-llama-comparison/`, without changing earlier frozen capsules.

### Native target cohort batching (blocks ABI 16)

`NativeQwenModel.create_batch()` groups projection, normalization, MLP and output
head rows across independent requests. Each slot owns separate convolution
history, FP32 GDN state, FP16 attention KV and position. Python only submits
metadata; prefill packing, active decode compaction, greedy selection and the
generation loop remain in C++. EOS and output budgets can differ per request.

```python
# model is an already loaded resident NativeQwenModel, optionally with TP.
# max_tokens must accommodate at least the selected request count.
batch = model.create_batch([256] * 8)
try:
    outputs = batch.generate(
        [prompt_a, prompt_b],
        max_new_tokens=[32, 8],
        eos_token_ids=[[], [248046, 248044]],
        slots=[7, 2],
    )
    print(batch.info())
    batch.reset(slots=[7, 2])
finally:
    batch.close()
```

`forward()` returns owned CPU final-row logits per selected slot for diagnostics;
`snapshot(slot)` copies committed state for correctness checks. The last emitted
token is not consumed, matching scalar native generation. Invalid submissions
do not mutate state; an execution failure poisons participating slots until reset.
Shared model/workspace counters count once per cohort, not once per response.

This first version is **fixed-cohort target batching**, not continuous admission,
paged KV, PP batching or batched MTP. Short prompts currently wait for the whole
cohort's prefill. The loopback benchmark adapter enables it with `--batch-target`
(TP only, without `--mtp-artifact`); its admission window defaults to 5 ms and is
included in client wall time. Rebuild the library remotely for ABI 16; older
libraries are deliberately rejected. V100 qualification does not qualify SM89.

On the two-V100 Qwen-27B test (23 prompt IDs, 32 outputs, three repeats), target
batching improves eight-request complete-cohort HTTP throughput from **15.44 to
28.84 TPS**; single-request decode remains approximately 18.9 TPS. The explicit
25 ms benchmark admission window is included in HTTP wall time. The pinned
llama.cpp non-speculative control still reaches **63.90 TPS** at eight requests:
the external speed goal is not complete. Exact real-model scalar/cache gates,
1/2/4/8 results, CPU/latency, telemetry and separate profiles are preserved in
`benchmarks/v100-native-batch/summary.md`.

### Staged integer-dot Q8 projection research

`prepare_q8_dp4a(resident, tokens=8, residual_compensation=True)` is an explicit
experimental operator. It retains
the original Q8 weights but quantizes FP16 activation rows with FP32 group scales;
the optional second Q8 pass compensates the activation residual. This is a named
new compute profile, not exact original FP16 arithmetic. Caller-owned aligned
workspace is mandatory; execution has no Torch math/allocation fallback.

On V100, independent operator/reference tests pass and the compensated prototype
is about 1.4x faster at one row and 1.6x at eight rows across the tested projection
shapes. These are operator measurements; the following model integration has
separate qualification. The default engine arithmetic is unchanged. Rejected T8/CPU-thread experiments,
precision errors, raw timings and separate profiling are preserved in
`benchmarks/v100-q8-dp4a/summary.md`.

### Opt-in residual-activation Q8 model projections

Rebuild the native library to expose the additive ABI-16 profile exports. Select
`projection_profile="q8-residual"` in `NativeQwenModel.from_q8()`, or:

```bash
venturi qwen-native generate /path/to/text-q8.ninfer \
  --blocks-library /path/to/libventuri_blocks.so \
  --weight-mode resident --dtype fp16 --projection-profile q8-residual \
  --tensor-parallel-device cuda:1 --tensor-parallel-transport host \
  --token-ids 248045,846,198 --max-new-tokens 32
```

Supply the model's fully formatted token IDs. Omit the TP options for one GPU.
This named profile uses **two group32 activation Q8 passes with FP32 scales**;
the second compensates the first's residual. Original Q8 weights/scales, FP16
external activations/KV and FP32 GDN state remain unchanged, but projection
arithmetic/reduction and intermediate model states are **not bit-identical** to
the original FP16-activation path. The default remains `fp16` (original typed
arithmetic). Profile selection is explicit; unsupported libraries fail rather
than silently falling back. Single/TP, fixed-cohort target batching, and a
single-request MTP target are supported. MTP's draft retains original FP16
arithmetic. PP, streaming weights and BF16 targets reject this profile; SM89,
sampling and broad original-BF16/corpus quality are not qualified.

On two V100s, three repeats with 23 prompt IDs/32 outputs improve complete-cohort
HTTP TPS at 1/2/4/8 requests from **15.19/21.56/26.59/28.81** to
**18.05/26.59/33.28/37.16**. Single-request MTP post-first decode improves
**24.55 -> 29.27 TPS** (whole HTTP request: 18.47 -> 22.65 TPS). Fixed prior CPU
logit thresholds, all four original text trajectories and same-profile
scalar/batch/MTP state gates pass. This is limited trained-case qualification,
not a general quality-equivalence claim. The fresh llama.cpp control still
reaches **66.83 TPS** at eight requests; the external goal remains active.
See `benchmarks/v100-native-residual/summary.md` for exact scopes, arithmetic
differences, CPU/latency/VRAM/telemetry and separate CUDA timelines.

### Scoped shared projection inputs

New native builds share an input only within explicit adjacent QKV/z, Q/K/V
and MLP gate/up groups. They avoid duplicate TP input transfers and, for
`q8-residual`, duplicate activation packing. Cross-device fences still protect
output scratch/gather reuse. There is no persistent pointer-based activation
cache and no arithmetic change within either profile; chunk/head boundaries,
request switches, failure and reset cannot inherit a previous group's reuse flag.
This is not a fused GEMM, compressed wire format or CUDA graph.

Trained old/new comparisons preserve every logit byte on 98 steps per profile,
all four greedy trajectories and private scalar/batch cache bytes, including
eight mixed requests and selected reset/continuation. Three-repeat residual
complete-cohort HTTP TPS at 1/2/4/8 requests improves
**18.09/26.82/33.71/37.48 -> 18.97/27.66/34.35/38.62**. Original FP16-activation
C1/C8 improves **15.21/28.85 -> 15.81/29.72**; its C2/C4 speed was not remeasured.
Single-request residual MTP post-first decode improves **29.27 -> 30.42 TPS**,
while complete-request HTTP improves 22.62 -> 23.45 TPS. These are limited V100
measurements, not universal gains or SM89 qualification. Fresh llama.cpp C8
is still 62.80 TPS; the prior 66.83 TPS control also remains valid evidence.
The external goal is unfinished. See `benchmarks/v100-shared-inputs/summary.md`.

The benchmark uses **32 projection rows** and **256 context tokens per slot**:
these are distinct limits. Residual packing scratch is 1,253,376 bytes per
device, not scaled to 256 rows. The new capsule records this erratum to the
previous frozen report's prose; the original commands/data remain unchanged.

### Guarded V100 integer-dot T8 scheduling

Rebuilt SM70 libraries now reuse each Q8 weight group across eight activation
rows for large `q8-residual` projections. T1/T2 are unchanged. Small row grids
and code planes <=6 MiB retain T4; SM89 dispatch is unchanged. This is a measured
V100 heuristic, not universal autotuning. Quantization, FP32 reduction order,
outputs and private caches remain byte-identical within each profile.

Three-repeat two-V100 complete-cohort HTTP TPS at 1/2/4/8 requests improves
**19.05/27.72/34.34/38.60 -> 19.41/28.58/35.57/43.18**. C1/C2/C4 gains are
prefill gains: their post-first decode rates are essentially unchanged.
Single-request MTP decode also stays approximately **30.4 TPS**, despite lower
TTFT. Fresh llama.cpp still reaches **67.29 TPS** at C8, so the external goal
remains active. V100 suite: 1206 passed / 440 skipped; local: 761 / 885.

`prepare_q8_wmma()` is a separate **research-only** group32-scaled FP16 Tensor
Core operator. It passes independent numerical/tail/subnormal tests but is slower
than residual DP4A in the tested decode/batch regimes. It is **not selected by
native model generation** and has no trained-model quality qualification.
This milestone's serving gain comes from integer-dot scheduling, not Tensor Cores.
See `benchmarks/v100-q8-wmma/summary.md` for negative results, shape guards,
strict byte gates, raw timings and separate profiling.

### V100 TP output-copy hypothesis: rejected

A direct pitched-output prototype passes exact trained logits, scalar/batch
private caches and fresh MTP committed-state checks, plus 1280 V100 tests.
It removes 5208 primary staging copies in the C8 trace but does not improve
HTTP throughput (43.186 -> 43.188 TPS in screening). CUDA waiting shifts to
other APIs; summed copy-API duration was not removable elapsed time. The native
source is restored exactly to the qualified guarded-T8 path, and the rejected
prototype is preserved separately, not enabled in generation.

Fresh three-repeat llama.cpp MTP depth 2/3/4 tests find depth 2 best among
tested 1–4: 30.17 whole-request HTTP TPS and 49.80 reported post-first TPS.
Venturi's retained MTP1 remains 23.99 HTTP / 30.42 post-first TPS; the external
goal is not met. Native multi-proposal MTP is not implemented by this study.
See `benchmarks/v100-strided-output/summary.md` for scopes, negative results,
acceptance, profiling and the source-restore proof.

### V100 two-token shared-activation staging

The rebuilt SM70 library now stages activation codes/scales in shared memory
for eligible **T2 `q8-residual`** projections only (at most 48 KiB, 16 row warps
per CTA). Quantization, reduction order, outputs and private caches remain
byte-exact. Default FP16, T1/T4/T8, SM89 and peer transport/fences are unchanged.
No new CLI switch: use the existing explicit residual profile with the rebuilt
library. Standalone T1 shared variants were slower and are not selected.

Three-repeat two-V100 C2 HTTP improves **28.55 -> 29.63 TPS**; single-request
MTP1 post-first decode **30.42 -> 31.73 TPS**, whole-request HTTP 23.99 -> 24.80.
Other non-spec concurrency cells are roughly flat; this is not a universal
speedup or an external-engine win. Native MTP2 is still unimplemented.
V100 suite: 1256 passed / 440 skipped; local: 762 / 934. See
`benchmarks/v100-q8-shared-activation/summary.md` for byte/state gates, raw
measurements, rejected choices and separate launch-resource/timeline evidence.

### V100 K-tiled activation staging

The rebuilt SM70 residual path now stages 1,024 K-columns at a time for T>=3,
with unchanged quantization, reduction and output/private-cache bytes. The
measured tile4/row8 and tile8/row16 choices avoid a regressing small-row geometry.
T1, existing T2 staging, SM89 and default FP16 arithmetic remain unchanged.

Three-repeat C1/2/4/8 HTTP improves **19.39/29.63/35.55/43.17 ->
20.19/31.48/45.21/54.72 TPS**. C4/C8 gain about 27%; C1/C2 gain from prefill
only. Single MTP decode stays approximately 31.8 TPS, while whole-request HTTP
improves 24.97 -> 26.22. These are internal gains, not wins over llama.cpp.
V100 suite: 1438 passed / 440 skipped; local: 763 / 1115. Use the existing
explicit `q8-residual` option with the rebuilt library; no new precision switch.
See `benchmarks/v100-q8-tiled-staging/summary.md` for scopes, regressions,
reference reuse, raw data and separate profiling.
