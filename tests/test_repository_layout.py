"""Retired module paths must not return as forwarding compatibility files."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_retired_paths_are_absent():
    paths = json.loads((ROOT / "tests/fixtures/retired-module-paths.json").read_text())
    for module in paths:
        assert not (ROOT / "src" / Path(*module.split(".")).with_suffix(".py")).exists()


def test_no_new_forwarding_shims():
    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text())
        assert not (ast.get_docstring(tree) or "").startswith(
            "Re-export (moved to"
        ), path


def test_unused_scaffolding_is_not_packaged():
    for name in (
        "engine/executor.py",
        "engine/sampler.py",
        "engine/scheduler.py",
        "models/qwen/qwen_weights_v2.py",
        "models/matrix_types.py",
        "quantization/base.py",
    ):
        assert not (ROOT / "src/venturi" / name).exists()


def test_layout_import_is_independent_of_torch_models_and_old_container():
    import os, subprocess, sys

    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname == 'venturi.container.safetensors' or fullname.startswith('venturi.models'):
            raise AssertionError('Unexpected layout dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from venturi.quantization.layout import row_geometry
assert row_geometry((2, 70), 'q8_g32_fp16')[-1] == 272
"""
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_removed_legacy_paths_stay_removed():
    for path in (
        "models/qwen_load/codecs.py",
        "models/qwen_load/verify_storage.py",
    ):
        assert not (ROOT / "src/venturi" / path).exists()
    for path in ("build_candidates.py", "test_rowsplit.py", "evaluate_candidate.py"):
        assert not (ROOT / "tools/projection" / path).exists()
