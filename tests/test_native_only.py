"""External model engines are removed, never silently redirected to native math."""

import ast
import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_external_command_is_not_registered_or_silently_redirected(monkeypatch, capsys):
    from venturi import cli
    from venturi.commands.registry import COMMAND_BY_NAME

    assert "qwen" not in COMMAND_BY_NAME
    assert COMMAND_BY_NAME["qwen-native"].module == "venturi.qwen_native_cli"

    def forbidden(*args, **kwargs):
        pytest.fail("Retired command must not import or start any engine")

    monkeypatch.setattr(cli, "import_module", forbidden)
    with pytest.raises(SystemExit) as caught:
        cli.main(["qwen", "generate", "old-model.ninfer", "--runner", "old-runner"])
    assert caught.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize(
    "module",
    [
        "venturi.backends",
        "venturi.backends.qwen",
        "venturi.backends.strata",
        "venturi.qwen_cli",
    ],
)
def test_removed_engine_modules_are_not_importable(module):
    code = f"""
import importlib, json, sys
try:
    importlib.import_module({module!r})
except ModuleNotFoundError:
    pass
else:
    raise AssertionError('Retired external runtime is still importable')
assert 'torch' not in sys.modules
print(json.dumps({{'removed': True}}))
"""
    p = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=20
    )
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout) == {"removed": True}


def test_native_qwen_and_ninfer_format_reader_need_no_external_engine():
    code = """
import importlib.abc,json,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in {'ninfer','cinference','strata','llama_cpp','vllm','sglang'}:
            raise AssertionError('Unexpected external engine: '+name)
sys.meta_path.insert(0,Block())
from venturi.models import NativeQwenModel,NativeQwenSession,QwenQ8Weights
from venturi.ops import CudaBackend
from venturi.ninfer import NInferArtifact
import torch
assert not torch.cuda.is_initialized()
print(json.dumps(dict(native_available=True,cuda_initialized=False)))
"""
    p = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout) == {"native_available": True, "cuda_initialized": False}


@pytest.mark.parametrize(
    "source",
    [
        "import vllm",
        "from llama_cpp import Llama",
        "def run():\n    from venturi.backends.qwen import QwenNativeSession",
        "from ninfer import Engine",
        "import strata",
    ],
)
def test_source_guard_rejects_reintroduced_external_execution(tmp_path, source):
    p = tmp_path / "venturi/new_runtime.py"
    p.parent.mkdir()
    p.write_text(source)
    report = runpy.run_path(str(ROOT / "tools/check_architecture.py"))[
        "inspect_source"
    ](tmp_path)
    assert "native-runtime-does-not-import-external-engines" in {
        v["rule"] for v in report["violations"]
    }


def test_measurement_helpers_import_without_any_engine_or_tensor_framework():
    code = f"""
import importlib.abc,json,sys
sys.path.insert(0,{str(ROOT / "tools")!r})
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in {{'venturi','torch','ninfer','cinference','strata'}}:
            raise AssertionError(name)
sys.meta_path.insert(0,Block())
import gpu_telemetry,process_metrics
assert not any(k.startswith('venturi') for k in sys.modules)
print(json.dumps(dict(independent=True)))
"""
    p = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=20
    )
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout) == {"independent": True}


def test_explicit_process_measurement_has_no_model_or_private_process_dependency(
    monkeypatch,
):
    spec = importlib.util.spec_from_file_location(
        "native_process_metrics", ROOT / "tools/process_metrics.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    samples = iter([2.0, 5.0])
    python = iter([0.1, 0.3])
    wall = iter([10.0, 12.0])
    pids = []

    def cpu(pid):
        pids.append(pid)
        return next(samples)

    monkeypatch.setattr(module, "process_cpu", cpu)
    monkeypatch.setattr(module, "python_cpu", lambda: next(python))
    monkeypatch.setattr(module, "perf_counter", lambda: next(wall))
    result = module.measure_process(321, lambda: "result")
    assert pids == [321, 321]
    assert result["native_cpu_seconds"] == 3.0
    assert result["python_cpu_seconds"] == pytest.approx(0.2)
    assert result["wall_seconds"] == 2.0
    assert result["native_cpu_percent"] == 150.0
    assert result["result"] == "result"


def test_package_configuration_no_longer_ships_an_external_runner():
    config = (ROOT / "pyproject.toml").read_text()
    assert '"venturi.backends"' not in config
    assert not (ROOT / "src/venturi/backends").exists()
    assert not (ROOT / "src/venturi/qwen_cli.py").exists()
    assert not (ROOT / "tools/build_qwen_runner.py").exists()
    assert not (ROOT / "scripts/venturi-qwen-4090d").exists()


def test_native_benchmark_tools_do_not_import_retired_engine_helpers():
    retired = {"benchmark_qwen_gpu", "benchmark_qwen_cpu", "run_qwen_validation"}
    for path in (ROOT / "tools").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in retired, (path, node.lineno)
