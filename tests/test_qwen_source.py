"""HF admission, bounded safetensors export and independent source-relative checks."""

import copy, hashlib, json
from pathlib import Path
import pytest, torch
from safetensors import safe_open
from venturi.container.checkpoint import CheckpointError, inspect_checkpoint
from venturi.container.safetensors import SafetensorsContainer, SafetensorsWriter
from venturi.models.qwen.qwen_source import QwenSourcePlan
from venturi.models.qwen.qwen_weights import QwenWeights
from venturi.models.qwen.gdn_frontend import GDNFrontendWeights
from venturi.models.qwen.qwen_export import export_qwen_q8
from venturi.models.qwen.qwen_export_verify import verify_qwen_q8_export
from qwen_fixtures import source_config, fixture_tensors, write_source


@pytest.mark.parametrize(
    "bad",
    [
        {"model_type": "unknown"},
        {"quantization_config": {}},
        {"text_config": {}},
        {"text_config": dict(source_config()["text_config"], dtype="float16")},
        {"text_config": dict(source_config()["text_config"], attention_bias=True)},
        {
            "text_config": dict(
                source_config()["text_config"],
                venturi_gguf_q8_0="direct-gamma-rate-grouped-v1",
            )
        },
    ],
)
def test_source_config_rejects_unsupported_semantics(bad):
    raw = source_config()
    raw.update(bad)
    with pytest.raises(ValueError):
        QwenSourcePlan.from_config(raw)


@pytest.mark.parametrize("rows", [1, 7, 256])
def test_export_all_planes_and_float_parameters_against_independent_source(
    tmp_path, rows
):
    values = fixture_tensors()
    source = write_source(tmp_path / "hf", tensors=values)
    result = export_qwen_q8(source, tmp_path / "model.safetensors", rows_per_chunk=rows)
    weights = QwenWeights.open(result["output"])
    plan = weights.plan
    with safe_open(result["output"], framework="pt", device="cpu") as reader:
        assert set(reader.keys()) == set(weights.source.list_tensors())
        assert not any(
            x in reader.metadata()
            for x in ("objects", "bindings", "files", "uses", "venturi.roles")
        )
        for spec in plan.bindings:
            value = values[spec.source]
            if spec.format == "q8_g32_fp16":
                code = reader.get_tensor(spec.source + ".codes")
                scale = reader.get_tensor(spec.source + ".scales")
                n, k = spec.shape
                for row in range(n):
                    for begin in range(0, k, 32):
                        group = value[row, begin : begin + 32].double().tolist()
                        stored = torch.tensor(
                            max(map(abs, group)) / 127, dtype=torch.float16
                        ).item()
                        assert scale[row, begin // 32].item() == stored
                        expected = [
                            min(127, max(-127, round(x / stored))) if stored else 0
                            for x in group
                        ]
                        assert (
                            code[row, begin : begin + len(group)].tolist() == expected
                        )
            else:
                if spec.transform == "conv-kc":
                    value = value.squeeze(1).transpose(0, 1)
                expected = value.to(
                    torch.float32 if spec.format == "fp32" else torch.bfloat16
                )
                assert torch.equal(weights.read_float(spec.binding), expected)
    assert result["role_count"] == len(plan.bindings)
    assert result["tensor_count"] == len(weights.source.list_tensors())
    assert (
        hashlib.sha256(Path(result["output"]).read_bytes()).hexdigest()
        == result["sha256"]
    )
    frontend = GDNFrontendWeights.from_weights(weights, layer=0)
    assert frontend.conv.shape == (4, 308)


@pytest.mark.parametrize("tied", [False, True])
def test_tied_embedding_storage_is_not_duplicated(tmp_path, tied):
    raw = source_config()
    raw["tie_word_embeddings"] = tied
    values = fixture_tensors()
    if tied:
        values.pop("lm_head.weight")
    result = export_qwen_q8(
        write_source(tmp_path / "hf", raw=raw, tensors=values),
        tmp_path / "model.safetensors",
    )
    weights = QwenWeights.open(result["output"])
    if tied:
        assert weights.descriptor("text/output_head") is weights.descriptor(
            "text/token_embedding"
        )
        assert "lm_head.weight.codes" not in weights.source.list_tensors()
    else:
        assert weights.descriptor("text/output_head") is not weights.descriptor(
            "text/token_embedding"
        )


@pytest.mark.parametrize("bad", ["missing", "shape", "dtype"])
def test_original_checkpoint_preflight_before_output(tmp_path, bad):
    values = fixture_tensors()
    name = "model.language_model.layers.0.mlp.gate_proj.weight"
    if bad == "missing":
        values.pop(name)
    elif bad == "shape":
        values[name] = values[name][:1]
    else:
        values[name] = values[name].to(torch.int8)
    source = write_source(tmp_path / "hf", tensors=values)
    with pytest.raises(CheckpointError):
        export_qwen_q8(source, tmp_path / "model.safetensors")
    assert not list(tmp_path.glob(".venturi-safetensors-*"))


@pytest.mark.parametrize("rows", [0, True, 4097])
def test_invalid_export_chunk_budget(tmp_path, rows):
    with pytest.raises(ValueError):
        export_qwen_q8(
            tmp_path / "missing", tmp_path / "model.safetensors", rows_per_chunk=rows
        )


def test_unverified_source_requires_explicit_opt_in_and_no_clobber(tmp_path):
    source = write_source(tmp_path / "hf", verified=False)
    with pytest.raises(CheckpointError):
        export_qwen_q8(source, tmp_path / "model.safetensors")
    result = export_qwen_q8(
        source, tmp_path / "model.safetensors", allow_unverified_source=True
    )
    assert "unverified" in result["source"]["kind"]
    before = Path(result["output"]).read_bytes()
    with pytest.raises(FileExistsError):
        export_qwen_q8(source, result["output"], allow_unverified_source=True)
    assert Path(result["output"]).read_bytes() == before


def test_source_change_aborts_without_publishing(tmp_path, monkeypatch):
    source = write_source(tmp_path / "hf")
    original = SafetensorsWriter.write_slice
    changed = False

    def mutate(self, *args, **kwargs):
        nonlocal changed
        result = original(self, *args, **kwargs)
        if not changed:
            p = next(source.glob("model-*.safetensors"))
            data = bytearray(p.read_bytes())
            data[-1] ^= 1
            p.write_bytes(data)
            changed = True
        return result

    monkeypatch.setattr(SafetensorsWriter, "write_slice", mutate)
    with pytest.raises(CheckpointError, match="changed"):
        export_qwen_q8(source, tmp_path / "model.safetensors")
    assert not (tmp_path / "model.safetensors").exists()
    assert not list(tmp_path.glob(".venturi-safetensors-*"))


def test_independent_export_verifier_catches_updated_publisher_hash(tmp_path):
    source = write_source(tmp_path / "hf")
    result = export_qwen_q8(source, tmp_path / "model.safetensors")
    report = tmp_path / "report.json"
    report.write_text(json.dumps(result))
    proof = verify_qwen_q8_export(source, result["output"], report)
    assert proof["max_sampled_abs_weight_quantization_error"] >= 0
    weights = QwenWeights.open(result["output"])
    info = weights.descriptor("text/token_embedding").planes[0]
    p = Path(result["output"])
    data = bytearray(p.read_bytes())
    data[info.byte_offset] ^= 1
    p.write_bytes(data)
    result["sha256"] = hashlib.sha256(data).hexdigest()
    report.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="sample differs|code -128"):
        verify_qwen_q8_export(source, p, report)


def test_bf16_gated_norm_is_widened_exactly(tmp_path):
    values = fixture_tensors()
    name = "model.language_model.layers.0.linear_attn.norm.weight"
    values[name] = (torch.arange(13) * 0.03125 + 0.875).bfloat16()
    result = export_qwen_q8(
        write_source(tmp_path / "hf", tensors=values), tmp_path / "model.safetensors"
    )
    actual = QwenWeights.open(result["output"]).read_float("text/layers/0/gdn/norm")
    assert actual.dtype == torch.float32 and torch.equal(actual, values[name].float())
