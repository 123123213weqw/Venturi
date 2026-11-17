"""Serialized mixed formats and pre-allocation numerical-policy admission."""

from types import SimpleNamespace
import ctypes
import pytest
import torch
from qwen_fixtures import source_config
from qwen_safetensors import write_quantized_checkpoint
from venturi.models.qwen.qwen_source import QwenSourcePlan
from venturi.models.qwen.qwen_weights import QwenWeights
from venturi.quantization.linear import validate_linear_contracts


def fixture(tmp_path, **kw):
    cfg = source_config()["text_config"]
    cfg.update(
        hidden_size=64,
        head_dim=64,
        vocab_size=32,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        intermediate_size=128,
        tie_word_embeddings=True,
    )
    path, _, _ = write_quantized_checkpoint(
        tmp_path / "mixed.safetensors", config=cfg, **kw
    )
    return QwenWeights.open(path)


def spec(**kw):
    values = dict(
        target=torch.device("cpu"),
        dtype=torch.float16,
        weight_mode="resident",
        tensor_parallel_device=None,
        stage_range=None,
        projection_profile="fp16",
        linear_implementation="auto",
        dequant_resident=False,
    )
    return SimpleNamespace(**(values | kw))


def test_mixed_serialized_roles_and_tying(tmp_path):
    formats = {
        "text/token_embedding": "q4_g64_fp16",
        "text/layers/0/mlp/down": "q6_g64_fp16",
        "text/layers/1/mlp/up": "q5_g64_fp16",
    }
    weights = fixture(tmp_path, matrix_formats=formats)
    for role, fmt in formats.items():
        assert weights.descriptor(role).format == fmt
        assert weights.read_linear(role).data.shape == weights.descriptor(role).shape
    assert weights.descriptor("text/output_head") is weights.descriptor(
        "text/token_embedding"
    )
    assert weights.descriptor("text/layers/0/mlp/up").format == "q8_g32_fp16"


@pytest.mark.parametrize(
    "overrides",
    [
        {"text/output_norm": "q4_g64_fp16"},
        {"not/a/role": "q5_g64_fp16"},
        {"text/token_embedding": "q3_bad"},
        {"text/token_embedding": None},
        {"text/output_head": "q4_g64_fp16"},
    ],
)
def test_bad_mixed_metadata_fails_closed(tmp_path, overrides):
    with pytest.raises(ValueError):
        fixture(tmp_path, matrix_formats=overrides)


def test_descriptor_populates_every_matrix_bits_and_group(tmp_path, monkeypatch):
    import venturi.models.qwen_load.descriptors as module
    from venturi.ops import TorchBackend
    from venturi.models.qwen.qwen_rope import QwenTextRoPE

    weights = fixture(
        tmp_path,
        matrix_formats={
            "text/layers/0/mlp/down": "q6_g64_fp16",
            "text/token_embedding": "q4_g64_fp16",
        },
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (8, 9))
    monkeypatch.setattr(module, "_bind", lambda *_: SimpleNamespace())
    s = spec(
        backend=TorchBackend(),
        runtime=None,
        max_tokens=3,
        profiled_create=None,
        tensor_parallel_transport="peer",
        reserve_bytes=0,
    )
    # Config requires an explicit CUDA ordinal; tensor preparation remains CPU.
    s.target = SimpleNamespace(index=0)
    backend = s.backend
    for name in (
        "upload_zero_centered_rmsnorm",
        "upload_fp32_linear",
        "upload_linear",
        "upload_qwen_qk_norm",
    ):
        original = getattr(backend, name)
        monkeypatch.setattr(
            backend,
            name,
            lambda *a, _fn=original, **kw: _fn(*a, **(kw | {"device": "cpu"})),
        )
    # GDN frontend passes the target to all TorchBackend upload/validation calls.
    original = backend.validate_device
    monkeypatch.setattr(
        backend, "validate_device", lambda dtype, device: original(dtype, "cpu")
    )
    d = module.build_descriptors(
        s,
        weights,
        weights.config,
        SimpleNamespace(rope=QwenTextRoPE.from_config(weights.config)),
    )
    assert (d.embedding.bits, d.embedding.group) == (4, 64)
    assert (d.head.bits, d.head.group) == (4, 64)
    assert (d.layers[0].matrix[6].bits, d.layers[0].matrix[6].group) == (6, 64)
    assert (d.layers[1].matrix[4].bits, d.layers[1].matrix[4].group) == (8, 32)
    assert ctypes.sizeof(d.config) == 120


def test_mixed_contracts_and_unsupported_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (8, 9))
    weights = fixture(
        tmp_path, matrix_formats={"text/layers/0/mlp/down": "q6_g64_fp16"}
    )
    validate_linear_contracts(spec(), weights)
    for kw in (
        {"dtype": torch.bfloat16},
        {"weight_mode": "streaming"},
        {"tensor_parallel_device": "cuda:1"},
        {"stage_range": (0, 1)},
        {"projection_profile": "int8"},
        {"projection_profile": "q8-residual"},
    ):
        with pytest.raises(ValueError):
            validate_linear_contracts(spec(**kw), weights)


def test_sm70_gate_is_not_opened_by_kernel_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (7, 0))
    from venturi.quantization import codecs

    monkeypatch.setitem(codecs.EXECUTION_QUALIFIED, (7, 0), {"q8_g32_fp16"})
    weights = fixture(tmp_path, matrix_format="q5_g64_fp16")
    with pytest.raises(ValueError, match="execution-qualified"):
        validate_linear_contracts(spec(), weights)


def test_one_pass_a8_cannot_inherit_a16_qualification(tmp_path, monkeypatch):
    from venturi.quantization import codecs

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (8, 9))
    monkeypatch.setitem(codecs.ACTIVATION_INT8_QUALIFIED, (8, 9), set())
    weights = fixture(tmp_path)
    validate_linear_contracts(spec(), weights)
    with pytest.raises(ValueError, match="int8 model execution qualification pending"):
        validate_linear_contracts(spec(projection_profile="int8"), weights)


def test_mixed_source_plan_can_validate_original_checkpoint(tmp_path):
    from qwen_fixtures import write_source
    from venturi.container.checkpoint import inspect_checkpoint

    raw = source_config()
    source = write_source(tmp_path / "hf")
    formats = {"text/layers/0/mlp/down": "q6_g64_fp16"}
    plan = QwenSourcePlan.from_config(raw, matrix_formats=formats)
    plan.validate_checkpoint(inspect_checkpoint(source), matrix_formats=formats)


def test_native_mtp_rejects_row_target_before_device_payload(tmp_path, monkeypatch):
    from test_qwen_mtp import mtp_fixture
    from test_row_codec_execution import _codec_artifact_from_q8
    from venturi.models import NativeQwenModel, NativeQwenMTPModel

    base, draft, _, _ = mtp_fixture(tmp_path)
    row = QwenWeights.open(
        _codec_artifact_from_q8(base.source, tmp_path / "q5.safetensors", "q5_g64_fp16")
    )
    target = object.__new__(NativeQwenModel)
    target._mode = "resident"
    target._max_tokens = 8
    monkeypatch.setattr(target, "_binding_weights", lambda: row)
    with pytest.raises(ValueError, match="all-Q8 target"):
        NativeQwenMTPModel.from_q8(target, draft)


def test_codec_qualification_does_not_imply_fused_qualification(tmp_path, monkeypatch):
    from venturi.quantization import codecs

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (8, 9))
    monkeypatch.setitem(
        codecs.FUSED_LINEAR_QUALIFIED, (8, 9), {"q4_g64_fp16", "q8_g32_fp16"}
    )
    weights = fixture(tmp_path, matrix_format="q5_g64_fp16")
    validate_linear_contracts(spec(), weights)
    with pytest.raises(ValueError, match="Fused model execution qualification pending"):
        validate_linear_contracts(spec(linear_implementation="fused"), weights)


def test_reported_native_family_must_match_numerical_contract(monkeypatch):
    from venturi.quantization.linear import validate_reported_plan
    from venturi.quantization import codecs

    monkeypatch.setitem(
        codecs.FUSED_LINEAR_QUALIFIED, (7, 0), {"q4_g64_fp16", "q8_g32_fp16"}
    )
    assert (
        validate_reported_plan("q5_g64_fp16", "fp16", (7, 0), [5, 64, 0, 1, 0, 0])
        == "chunked-dense"
    )
    with pytest.raises(ValueError, match="unqualified direct/fused"):
        validate_reported_plan("q5_g64_fp16", "fp16", (7, 0), [5, 64, 0, 0, 4, 0])
    for values in ([5, 64, 1, 2, 0, 0], [5, 32, 0, 1, 0, 0], [5, 64, 0, 2, 0, 0]):
        with pytest.raises(RuntimeError):
            validate_reported_plan("q5_g64_fp16", "fp16", (7, 0), values)
