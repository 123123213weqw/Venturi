"""Hash-pinned reuse is explicit and rejected before any model allocation."""

import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest


@pytest.fixture
def tool(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(path))
    spec = spec_from_file_location(
        "execution_policy_qualifier", path / "qualify_qwen_execution_policy.py"
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hash_pin_matches_complete_reference(tmp_path, tool):
    p = tmp_path / "report.json"
    p.write_text(json.dumps(dict(completed=True, cases=["exactly declared reuse"])))
    assert tool.read_pin(p, hashlib.sha256(p.read_bytes()).hexdigest())["completed"]


def test_hash_mismatch_never_reuses_reference(tmp_path, tool):
    p = tmp_path / "report.json"
    p.write_text('{"completed":true}')
    with pytest.raises(ValueError, match="hash mismatch"):
        tool.read_pin(p, "0" * 64)


@pytest.mark.parametrize("value", [{}, {"completed": False}, [], None])
def test_incomplete_or_nonobject_reference_rejected(tmp_path, tool, value):
    p = tmp_path / "report.json"
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="incomplete"):
        tool.read_pin(p, hashlib.sha256(p.read_bytes()).hexdigest())
