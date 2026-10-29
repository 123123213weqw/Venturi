# Prepared operator backend contract

The public contract is `venturi.ops.OperatorBackend`, with an optional abstract
`Backend` base. `TorchBackend` and `CudaBackend` implement the same contract.
This is an incremental refactor of the existing prepared operators, **not a new
Metal/MLX backend, automatic device router, model adapter or serving runtime**.

## Boundaries

```text
model weights / architecture / request state
                  |
         OperatorBackend contract
                  |
     upload -> resident.prepare -> plan.run
       /                               \
Torch CPU/CUDA reference          native CUDA blocks
```

1. `capabilities` and `require(...)` inspect implementation metadata without
   allocating tensors, opening native libraries, probing or initializing GPUs.
2. `resolve_device(device)` checks the selected execution device;
   `validate_device(dtype, device)` additionally checks activation precision.
   Both return a canonical device. CPU indices normalize to `cpu`, matching
   actual PyTorch CPU tensor storage. Native CUDA still enforces library SM
   architecture and rejects BF16 on SM70.
3. `empty(shape, dtype=..., device=...)` allocates explicit activation/scratch
   buffers. `synchronize(device)` is explicit completion, a no-op on CPU.
   These are setup/lifecycle operations, not hidden actions in `plan.run`.
4. `upload_linear` / `upload_rmsnorm` own weight storage. Resident weights can
   bind multiple token extents without uploading again. `upload_fp32_linear`
   remains the explicitly selected mixed FP32-parameter path.
5. `prepare_linear`, `prepare_rmsnorm`, `prepare_swiglu`, `prepare_add`,
   `prepare_rope` and `prepare_attention` preserve their existing mathematical,
   dtype, layout, ownership and aliasing contracts. Plans write caller-owned
   outputs; linear scratch is explicit. Each concurrent execution needs its own
   workspace. Keep weights, plans and operands alive through queued GPU work.

`LinearWeight` and `LinearPlan` are structural protocols, not required concrete
classes. A backend supplies its own prepared implementations; it need not expose
`_native`, `_device`, a CUDA stream, or a particular display `name` to a model.
Existing `_device` and `_bind_*` aliases remain on the old adapters for compatibility.
Public binds additionally reject weights owned by a different backend instance.

## Explicit construction

```python
import torch
from venturi.ops import DenseWeight, create_backend

backend = create_backend("torch")  # numerical reference, not a native CPU kernel
backend.require("linear", dtype=torch.float32, device="cpu")
weight = DenseWeight(torch.randn(8, 4))
resident = backend.upload_linear(weight, dtype=torch.float32, device="cpu")
decode = resident.prepare(tokens=1)
prefill = resident.prepare(tokens=3)  # shares resident storage
x = torch.randn(3, 4)
out = backend.empty((3, 8), dtype=torch.float32, device="cpu")
prefill.run(x, out=out)
backend.synchronize("cpu")
```

For the existing native path, explicitly select
`create_backend("cuda", library=prebuilt_blocks_path, q4_backend=optional_ada)`.
It never compiles code or automatically finds an arbitrary library. Passing a
native library/plugin to the Torch factory is rejected, not ignored.
`"auto"`, `"metal"`, `"mlx"` and unknown backends fail clearly; none silently
fall back to CPU or claim Mac GPU support.

## Capabilities are exact, bounded claims

The currently advertised core operations are linear, embedding, RMSNorm,
SwiGLU, add, RoPE and dense causal attention. Each capability includes device
types, activation dtypes, and, where relevant, exact weight format **and layout**.

- Dense is not interchangeable with an arbitrary quantized container.
- Q4 is `q4_g64_fp16` / `row_split_k128_v1`; Q8 is `q8_g32_fp16` /
  `row_split_k128_v1`. GPTQ/AWQ/GGUF/other layouts are not inferred.
- Torch's quantized reference materializes floating weights; capability entries
  explicitly report `materializes_weights=True`. This is not a quantized kernel
  performance claim or a memory-efficient fallback.
- Native Q8 supports FP16/BF16 activations, subject to the SM70 restriction.
  The blocks Q4 route requires the explicit existing SM89 Ada plugin. The
  separate Flash-Next SM70 Q4 prototype is **not** automatically registered here.
- `cuda_graphs=True` describes the existing native prepared-operator contract,
  not qualification of every model, capture layout or the experimental routed
  Flash-Next Graph. That failed/incomplete full-model gate remains unchanged.

Metadata support does **not** mean a physical GPU exists, a tensor shape is
supported, memory fits, the checkpoint's architecture is executable, or a whole
model is qualified. Concrete preparation still validates those applicable
constraints. There is no global memory planner or automatic per-op CPU fallback.

## Integration and remaining platform work

GatedMLP and the standard decoder/Llama adapter preflight their required core
operations/formats before uploading weights. Llama checks all its layers before
the first upload. GatedMLP injects the selected backend's workspace allocator;
the original explicit six-argument plan constructor remains compatible.
Existing Qwen adapters now use public device validation and reference capability
metadata rather than comparing display names. Existing Torch/CUDA helper plans
use a private implementation-kind switch, not a display-name switch.

GDN, Qwen fused operations, native TP/PP, capture, KV allocation and scheduling
remain explicit extensions with their existing restrictions. They are **not**
made portable merely by implementing the core protocol. The Flash-Next prototype
also still contains its own Torch/CUDA execution path. External whole-model
engine adapters were removed on October 28; they are not operator plugins or
fallback branches of this interface.

The current tensor/weight boundary is still PyTorch. A future Metal or MLX adapter
must implement the contract and required extensions and deal explicitly with its
tensor/weight representation; changing a device string is insufficient. No new
Metal implementation or fully framework-independent tensor IR is claimed.

## Validation

Prefill/decode disaggregation is an explicit request-state extension above this
operator contract; see [PD runtime](pd-disaggregation.md). Core operator metadata
does not advertise checkpoint-state migration. A concrete architecture adapter
must supply its real state layout and an explicitly qualified transport. Native
PD v1 rejects TP/PP/MTP and tiered KV rather than inferring compatibility.

`tests/test_backend_contract.py` exercises metadata-only inspection, no hidden
CUDA/library load, exact quantization layout matching, explicit factory failures,
buffer allocation, backend ownership, shared resident weights, fail-before-upload
model preflight, structural plugins and display-name-independent mathematics.
Existing mathematical tests remain in place. Refactor records and before/after
fixture comparisons are stored under `operations/backend-interface-20261026-v1/`.
No new TPS or speedup claim follows from an API refactor; production settings and
frozen model/benchmark evidence are not modified.

## Container inventory is not an operator backend

`venturi.model_assets` describes immutable source identities/roles and explicit
storage budgets. The separate Torch-free `venturi.gguf_inventory` can inspect
bounded GGUF v3 headers and exact on-disk block geometry, including mixed IQ1/IQ2,
Q5, Q8 and Engram layouts. Its type table is **not** a registry of executable
Venturi quantization kernels. It never creates a prepared operator or loads a
model; arrays are summarized, and payload SHA verification is separate.

The standalone `tools/inventory_gguf.py` accepts an explicit immutable bundle,
root and exclusive output filename. `--allow-partial` is diagnostic-only;
`--verify-assets` requests expensive full-file hashes and is mutually exclusive.
Header bytes, shape/axis order, tensor spans, source manifest SHA and payload
verification status remain distinct. Declared on-disk storage is not an automatic
resident RAM/VRAM calculation.

The former external Strata and NInfer/Cinference runtime adapters and their
engine-only tools are removed, not converted into native support. Their old
protocol/pack tests survive only in dated archival evidence. Full checkpoint SHA,
container geometry, architecture binding and actual numerical generation remain
separate native admission gates. DeepSeek whole-model support is still unfinished.

`venturi.gguf_metadata` is a separate **offline full-SHA** integer-metadata
reader: unlike header inventory, its verification scans the whole containing
asset. It returns bounded named integer tensors, not projection weights or an
executable model. Engram's explicit metadata-to-hash-spec binder consumes those
values; architecture/tokenizer/layer-order authentication remains a caller gate.
It does not regenerate missing map/prime/RNG parameters, choose a backend or add
an implicit CPU inference fallback. No real target scan is allowed during the
recorded root-storage fault.

The opt-in higher-level `models.engram_bindings.load_deepseek41_engram_hash`
applies that component gate using explicit trusted config/tokenizer manifests,
verified header layer arrays and exact stored-map comparison. Its result still
reports `model_supported=False`: unloaded lookup/projection weights and runtime
state are not certified. Separately, `gguf_quantization.decode_iq4_nl_rows` is a
bounded scalar CPU format oracle, not a `PreparedEmbedding` or tuned GPU kernel.
Neither path modifies existing backend selection or production models.

`gguf_rows.VerifiedIQ4NLRows` is a separate held-descriptor CPU row-storage
component: explicit full-SHA admission once, then bounded addressed reads and
cross-request row dedup/scatter. `models.engram_lookup.EngramIQ4NLLookup` applies
Engram source/schema gates and still reports no full-model support. Neither is
an `OperatorBackend`, prepared GPU embedding, request scheduler or automatic
RAM/VRAM fallback. Storage/read/decode oracles must not be reported as inference
TPS; native execution and target GPU comparisons remain independent gates.

Q5_K and BF16 decoders in `gguf_quantization` are bounded CPU format oracles,
not new prepared linear backends. `models.engram_projection` composes explicit
already-dequantized rows/weights with F32 CPU matmul and gating for correctness
fixtures only. It does not choose storage formats, authenticate Tensor inputs,
emulate mixed GPU arithmetic or silently replace an unsupported native operator.
