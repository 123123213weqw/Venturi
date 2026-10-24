"""Native runtime diagnostic comparison must preserve counterexamples."""

import copy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch


def load(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(directory))
    spec = spec_from_file_location(
        "native_runtime_tool", directory / "qualify_qwen_native.py"
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_saved_comparison_exact_bytes_and_rejected_states(monkeypatch):
    tool = load(monkeypatch)
    base = dict(
        trace=[dict(position=0, layer=0, hidden=torch.ones(2, 4, dtype=torch.float16))],
        state=dict(
            gdn=[(torch.zeros(1, 4, dtype=torch.float16), torch.ones(1, 2, 4)), None],
            kv=[
                None,
                (
                    torch.ones(2, 1, 4, dtype=torch.float16),
                    torch.ones(2, 1, 4, dtype=torch.float16),
                ),
            ],
        ),
    )
    assert len(tool.compare_saved(base, base)) == 5
    assert all(r["bytes_equal"] for r in tool.compare_saved(base, base))
    changed = copy.deepcopy(base)
    changed["state"]["gdn"][0][1][0, 0, 0] += 1
    assert not all(r["bytes_equal"] for r in tool.compare_saved(changed, base))
    changed["state"]["gdn"][0][1][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        tool.compare_saved(changed, base)
    changed = copy.deepcopy(base)
    changed["state"]["kv"][1] = None
    with pytest.raises(AssertionError, match="kind"):
        tool.compare_saved(changed, base)
    assert not tool.bytes_equal(torch.tensor([-0.0]), torch.tensor([0.0]))
    assert not tool.bytes_equal(torch.zeros(1, dtype=torch.float16), torch.zeros(1))
