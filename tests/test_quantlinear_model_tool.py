"""Qualification helpers must fail closed on incomplete or divergent state."""

import importlib
from pathlib import Path
import pytest
import torch


def tool(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools"))
    return importlib.import_module("qualify_quantlinear_model")


def state(value):
    return dict(
        gdn=[(torch.tensor([value]), torch.tensor([value])), None],
        kv=[None, (torch.tensor([value]), torch.tensor([value]))],
    )


def test_state_contract_identical_and_divergent(monkeypatch):
    m = tool(monkeypatch)
    good = m.compare_states(state(1.0), state(1.0))
    assert len(good) == 4 and all(c["passed"] for c in good)
    bad = m.compare_states(state(2.0), state(1.0))
    assert not any(c["passed"] for c in bad)


def test_state_contract_rejects_missing_component(monkeypatch):
    m = tool(monkeypatch)
    a, b = state(1.0), state(1.0)
    a["gdn"][0] = a["gdn"][0][:1]
    with pytest.raises(AssertionError):
        m.compare_states(a, b)


def test_paired_exact_state_rejects_missing_plane(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools"))
    m = importlib.import_module("qualify_quantlinear_paired")
    a, b = state(1.0), state(1.0)
    assert m.exact_state(a, b)
    b["gdn"][0] = b["gdn"][0][:1]
    assert not m.exact_state(a, b)
