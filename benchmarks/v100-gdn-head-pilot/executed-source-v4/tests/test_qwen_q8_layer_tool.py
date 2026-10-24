"""Layer qualification tool tests, including deliberate numeric failure evidence."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch

from test_qwen_attention_layer import fixture_weights
from venturi.models import ResidentQwenAttentionLayer, ResidentQwenGDNLayer
from venturi.ops import TorchBackend


def load():
    spec = spec_from_file_location(
        "qwen_q8_layers",
        Path(__file__).resolve().parents[1] / "tools/qualify_qwen_q8_layers.py",
    )
    tool = module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def test_fp16_operator_check_rejects_counterexample():
    tool = load()
    ideal = torch.tensor([[0.01, 1.0, -2.0]], dtype=torch.float64)
    actual = ideal.half()
    assert tool.scalar_check("good", actual, ideal)["passed"]
    actual[0, 0] = 0.1
    with pytest.raises(AssertionError, match="bad"):
        tool.scalar_check("bad", actual, ideal)


def test_source_fixture_layer_tool_stages_and_composition(tmp_path):
    tool = load()
    weights, _, _ = fixture_weights(tmp_path)
    for i, cls in ((0, ResidentQwenGDNLayer), (1, ResidentQwenAttentionLayer)):
        resident = cls.from_q8(
            weights, TorchBackend(), layer=i, dtype=torch.float16, device="cpu"
        )
        p = resident.prepare(tokens=3)
        x = (
            torch.randn(3, 32, generator=torch.Generator().manual_seed(645)) * 0.2
        ).half()
        checked = (tool.qualify_gdn if i == 0 else tool.qualify_attention)(
            weights, resident, p, x
        )
        assert checked["manual_composition_bit_exact"]
        assert all(item["passed"] for item in checked["stages"])
        assert len(checked["stages"]) >= 15
        p.close()
