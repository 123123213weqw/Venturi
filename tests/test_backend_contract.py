"""Backend metadata, public setup contract and unchanged prepared mathematics."""

from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from test_gated_mlp import make_weights
from test_llama import llama_fixture
from venturi.loader import load_checkpoint
from venturi.models import LlamaWeights
from venturi.ops import (
    Backend,
    BackendCapabilities,
    CudaBackend,
    DenseWeight,
    LinearPlan,
    LinearWeight,
    OperatorBackend,
    OperatorSupport,
    TorchBackend,
    UnsupportedOperationError,
    create_backend,
)


def forbidden(*args, **kwargs):
    raise AssertionError("Unexpected GPU probe, allocation or library load")


def test_cpu_factory_and_metadata_never_probe_or_load_cuda(monkeypatch):
    import ctypes

    monkeypatch.setattr(ctypes, "CDLL", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    b = create_backend("torch")
    assert isinstance(b, OperatorBackend)
    assert b.validate_device(torch.float32, "cpu") == torch.device("cpu")
    assert b.capabilities.supports("linear", dtype=torch.float32, device="cuda:71")
    b.synchronize("cpu")
    assert b.empty((2, 3), dtype=torch.float32, device="cpu").shape == (2, 3)


@pytest.mark.parametrize("kind", ["auto", "metal", "mlx", "mps", "unknown", None])
def test_unimplemented_backend_is_not_a_silent_fallback(kind):
    with pytest.raises(UnsupportedOperationError, match="not implemented"):
        create_backend(kind)


def test_native_factory_requires_explicit_library_and_does_not_ignore_options(
    monkeypatch,
):
    with pytest.raises(ValueError, match="prebuilt"):
        create_backend("cuda")
    for options in (dict(library="unused.so"), dict(q4_backend=object())):
        with pytest.raises(ValueError, match="takes no"):
            create_backend("torch", **options)
    import venturi.ops.blocks as blocks

    calls = []
    sentinel = object()
    monkeypatch.setattr(
        blocks, "CudaBackend", lambda *a, **k: calls.append((a, k)) or sentinel
    )
    assert (
        create_backend("cuda", library="explicit.so", q4_backend="plugin") is sentinel
    )
    assert calls == [(("explicit.so",), {"q4_backend": "plugin"})]


def test_immutable_exact_format_and_layout_capabilities():
    caps = TorchBackend().capabilities
    for fmt in ("q4_g64_fp16", "q8_g32_fp16"):
        assert caps.supports(
            "linear",
            dtype=torch.float64,
            device="cpu",
            weight_format=fmt,
            weight_layout="row_split_k128_v1",
        )
        for layout in (None, "gptq", "row_split_k256_v9"):
            assert not caps.supports(
                "linear",
                dtype=torch.float64,
                device="cpu",
                weight_format=fmt,
                weight_layout=layout,
            )
    assert not caps.supports("linear", dtype=torch.float32, device="mps")
    assert not caps.supports("moe", dtype=torch.float32, device="cpu")
    assert not caps.supports("linear", dtype=torch.int8, device="cpu")
    assert all(o.materializes_weights for o in caps.operators if o.weight_layout)
    assert caps.reference and not caps.cuda_graphs
    with pytest.raises(FrozenInstanceError):
        caps.reference = False
    with pytest.raises(UnsupportedOperationError, match="no automatic fallback"):
        caps.require("linear", dtype=torch.float32, device="mps")


@pytest.mark.parametrize("arch", [70, 89])
@pytest.mark.parametrize("plugin", [False, True])
def test_native_capability_metadata_respects_architecture_and_explicit_plugin(
    arch, plugin
):
    # Metadata test only: no CUDA/library initialization or GPU qualification.
    b = object.__new__(CudaBackend)
    b.architecture, b.q4_backend = arch, object() if plugin else None
    caps = b.capabilities
    assert caps.cuda_graphs and not caps.reference
    assert not caps.supports("linear", dtype=torch.float32, device="cpu")
    assert caps.supports("linear", dtype=torch.bfloat16, device="cuda:0") == (
        arch == 89
    )
    request = dict(weight_format="q4_g64_fp16", weight_layout="row_split_k128_v1")
    assert caps.supports("linear", dtype=torch.float16, device="cuda:0", **request) == (
        arch == 89 and plugin
    )
    assert not caps.supports("linear", dtype=torch.float32, device="cuda:0", **request)


def test_capability_definitions_are_copied_and_validated():
    devices, dtypes = {"cpu"}, {torch.float32}
    op = OperatorSupport("linear", devices, dtypes, "dense")
    devices.add("mps")
    dtypes.add(torch.float64)
    caps = BackendCapabilities("third-party", [op])
    assert caps.operators == (op,)
    assert op.device_types == frozenset({"cpu"})
    assert op.dtypes == frozenset({torch.float32})
    with pytest.raises(ValueError, match="requires an explicit format"):
        OperatorSupport("linear", {"cpu"}, {torch.float32}, weight_layout="unknown")
    with pytest.raises(TypeError, match="torch.dtype"):
        OperatorSupport("linear", {"cpu"}, {"float32"})
    with pytest.raises(TypeError):
        Backend()


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.uint8,
        torch.int64,
        torch.bool,
    ],
)
def test_public_buffer_allocation(dtype):
    b = create_backend("torch")
    x = b.empty((2, 0, 3), dtype=dtype, device="cpu")
    assert x.dtype == dtype and x.shape == (2, 0, 3)
    assert b.empty((), dtype=dtype, device="cpu").ndim == 0
    assert b.empty((1,), dtype=dtype, device="cpu:0").device == torch.device("cpu")


@pytest.mark.parametrize("shape", [(True,), (-1,), (2**31,), (1.5,), "2,3"])
def test_invalid_allocation_fails_before_torch_empty(shape, monkeypatch):
    monkeypatch.setattr(torch, "empty", forbidden)
    with pytest.raises(ValueError, match="shape"):
        TorchBackend().empty(shape, dtype=torch.float32, device="cpu")


def test_unsupported_buffer_device_and_dtype_fail_closed():
    b = TorchBackend()
    with pytest.raises(ValueError, match="CPU/CUDA"):
        b.empty((2,), dtype=torch.float32, device="mps")
    with pytest.raises(ValueError, match="complex"):
        b.empty((2,), dtype=torch.complex64, device="cpu")


def test_resident_protocol_reuse_ownership_and_legacy_aliases():
    b, other = TorchBackend(), TorchBackend()
    w = b.upload_linear(
        DenseWeight(torch.ones(3, 5)), dtype=torch.float32, device="cpu"
    )
    assert isinstance(w, LinearWeight)
    a, c = w.prepare(tokens=1), w.prepare(tokens=4)
    assert isinstance(a, LinearPlan)
    assert a.storage_tensors[0] is c.storage_tensors[0]
    assert b._device(torch.float32, "cpu") == b.validate_device(torch.float32, "cpu")
    assert b._bind_linear(w, 2).tokens == 2
    norm = b.upload_rmsnorm(
        torch.ones(5), dtype=torch.float32, device="cpu", epsilon=1e-6
    )
    assert b._bind_rmsnorm(norm, 2).tokens == 2
    for method, resident in ((other.bind_linear, w), (other.bind_rmsnorm, norm)):
        with pytest.raises(ValueError, match="different backend"):
            method(resident, tokens=1)


def test_mlp_missing_quantized_support_rejected_before_first_upload(monkeypatch):
    b = TorchBackend()
    b.capabilities = replace(
        b.capabilities,
        operators=tuple(
            o for o in b.capabilities.operators if o.weight_format != "q4_g64_fp16"
        ),
    )
    monkeypatch.setattr(b, "upload_rmsnorm", forbidden)
    with pytest.raises(UnsupportedOperationError, match="q4_g64_fp16"):
        make_weights(8, 16, q4=True).upload(b, dtype=torch.float64, device="cpu")


def test_whole_llama_missing_attention_rejected_before_embedding_upload(
    tmp_path, monkeypatch
):
    llama_fixture(tmp_path / "model")
    ckpt = load_checkpoint(tmp_path / "model", device="cpu", loader="safetensors")
    weights = LlamaWeights.bind(ckpt)
    b = TorchBackend()
    b.capabilities = replace(
        b.capabilities,
        operators=tuple(
            o for o in b.capabilities.operators if o.operation != "attention"
        ),
    )
    monkeypatch.setattr(b, "upload_linear", forbidden)
    with pytest.raises(UnsupportedOperationError, match="attention"):
        weights.upload(b, device="cpu")


def test_workspace_uses_selected_backend_allocator(monkeypatch):
    b = TorchBackend()
    allocate = b.empty
    calls = []

    def tracked(shape, **options):
        calls.append((shape, options))
        return allocate(shape, **options)

    monkeypatch.setattr(b, "empty", tracked)
    p = make_weights(8, 16).prepare(b, tokens=3, dtype=torch.float64, device="cpu")
    ws = p.empty_workspace()
    assert [c[0] for c in calls] == [(3, 8), (3, 16), (3, 16)]
    assert ws.hidden.dtype == torch.float64


def test_structural_plugin_does_not_need_inheritance_or_private_device_api():
    class Plugin:
        name = "structural-plugin"

        def __init__(self):
            inner = TorchBackend()
            self.capabilities = inner.capabilities
            for member in (
                "resolve_device",
                "validate_device",
                "require",
                "require_linear",
                "empty",
                "synchronize",
                "upload_linear",
                "upload_fp32_linear",
                "upload_rmsnorm",
                "prepare_linear",
                "prepare_rmsnorm",
                "prepare_swiglu",
                "prepare_add",
                "prepare_rope",
                "prepare_attention",
            ):
                setattr(self, member, getattr(inner, member))

    b = Plugin()
    assert isinstance(b, OperatorBackend)
    assert not hasattr(b, "_device") and not hasattr(b, "_native")
    p = make_weights(8, 16).prepare(b, tokens=2, dtype=torch.float64, device="cpu")
    x = torch.zeros(2, 8, dtype=torch.float64)
    out = torch.empty_like(x)
    assert p.run(x, out=out, workspace=p.empty_workspace()) is out
    assert torch.equal(out, x)


def test_display_name_does_not_select_operator_math_or_model_reference_mode(tmp_path):
    class RenamedBackend(TorchBackend):
        name = "my-third-party-display-name"

    weights = make_weights(16, 32)
    x = torch.arange(32, dtype=torch.float64).reshape(2, 16) / 100
    outputs = []
    for b in (TorchBackend(), RenamedBackend()):
        p = weights.prepare(b, tokens=2, dtype=torch.float64, device="cpu")
        out = torch.empty_like(x)
        p.run(x, out=out, workspace=p.empty_workspace())
        outputs.append(out)
    assert torch.equal(*outputs)
    llama_fixture(tmp_path / "model")
    ckpt = load_checkpoint(tmp_path / "model", device="cpu", loader="safetensors")
    weights = LlamaWeights.bind(ckpt)
    outputs = []
    for b in (TorchBackend(), RenamedBackend()):
        model = weights.upload(b, device="cpu")
        session = model.create_session(16, max_tokens=4)
        try:
            outputs.append(
                [session.forward([1, 3, 5]).clone(), session.forward([7]).clone()]
            )
        finally:
            session.close()
    assert all(torch.equal(a, b) for a, b in zip(*outputs))
