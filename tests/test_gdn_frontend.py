"""GDN constants come from admitted Qwen tensors, never from a format directory."""

import pytest, torch
from safetensors.torch import load_file, save_file
from venturi.models.qwen.gdn_frontend import GDNFrontendWeights
from venturi.models.qwen.qwen_source import QwenSourcePlan
from venturi.models.qwen.qwen_weights import QwenWeights
from venturi.container.safetensors import SafetensorsContainer
from venturi.ops import TorchBackend
from qwen_fixtures import source_config
from qwen_safetensors import write_quantized_checkpoint


def frontend_weights(directory):
    cfg = source_config()
    plan = QwenSourcePlan.from_config(cfg)
    values = {
        "gdn/convolution": torch.full((4, 308), 0.1, dtype=torch.bfloat16),
        "gdn/a_log": torch.zeros(4),
        "gdn/dt_bias": torch.ones(4),
        "gdn/norm": torch.ones(13),
        "input_norm": torch.full((32,), 0.25, dtype=torch.bfloat16),
    }
    sources = {s.binding: s.source for s in plan.bindings}
    updates = {sources["text/layers/0/" + role]: v for role, v in values.items()}
    path, _, _ = write_quantized_checkpoint(
        directory / "model.safetensors", config=cfg, updates=updates
    )
    return QwenWeights.open(path), values


def test_bind_constants_and_prepare_reference(tmp_path):
    admitted, values = frontend_weights(tmp_path)
    weights = GDNFrontendWeights.from_weights(admitted, layer=0)
    assert weights.qk_heads == 2 and weights.value_heads == 4
    assert torch.equal(weights.conv, values["gdn/convolution"])
    resident = weights.upload(TorchBackend(), dtype=torch.float64, device="cpu")
    assert resident.storage_nbytes == sum(
        t.numel() * t.element_size() for t in resident.storage_tensors
    )
    assert torch.equal(
        resident.input_norm._weight, torch.full((32,), 1.25, dtype=torch.float64)
    )
    assert torch.equal(resident.gates._rate, torch.ones(4, dtype=torch.float64))
    for layer in (1, -1, True, 2):
        with pytest.raises(ValueError):
            GDNFrontendWeights.from_weights(admitted, layer=layer)


@pytest.mark.parametrize("bad", ["missing", "shape", "format", "config"])
def test_preflight_before_any_payload_read(tmp_path, monkeypatch, bad):
    admitted, _ = frontend_weights(tmp_path)
    path = admitted.source.path
    data = load_file(str(path))
    meta = admitted.source.metadata()
    name = admitted.descriptor("text/layers/0/gdn/norm").source
    if bad == "missing":
        data.pop(name)
    elif bad == "shape":
        data[name] = torch.ones(14)
    elif bad == "format":
        data[name] = data[name].half()
    else:
        import json

        cfg = json.loads(meta["venturi.config"])
        cfg.get("text_config", cfg)["linear_num_value_heads"] = 3
        meta["venturi.config"] = json.dumps(cfg)
    save_file(data, str(path), metadata=meta)
    monkeypatch.setattr(
        SafetensorsContainer,
        "read_bytes",
        lambda *a, **k: pytest.fail("Invalid metadata caused tensor IO"),
    )
    with pytest.raises(ValueError):
        QwenWeights.open(path)
