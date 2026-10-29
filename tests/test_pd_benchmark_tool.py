"""Benchmark input/CLI contracts without allocating a GPU or importing models."""
import importlib.util
import argparse
import json
from pathlib import Path
import subprocess
import sys

import pytest


def tool():
    path = Path(__file__).resolve().parents[1] / "tools/benchmark_pd_native.py"
    spec = importlib.util.spec_from_file_location("pd_benchmark_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value", ["", "0,0", "-1", "a", "1,"])
def test_pd_benchmark_explicit_unique_devices(value):
    module = tool()
    with pytest.raises(argparse.ArgumentTypeError):
        module.devices(value)


@pytest.mark.parametrize("body", [[], {}, [dict(prompt=[], max_new_tokens=2)],
    [dict(prompt=[True], max_new_tokens=2)], [dict(prompt=[1], max_new_tokens=True)],
    [dict(prompt=[1], max_new_tokens=2, arrival_seconds=float("nan"))],
    [dict(prompt=[1], max_new_tokens=2, arrival_seconds=-1)]])
def test_pd_benchmark_request_preflight(tmp_path, body):
    path = tmp_path / "requests.json"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        tool().specs(path)


def test_pd_benchmark_help_is_framework_free():
    path = Path(__file__).resolve().parents[1] / "tools/benchmark_pd_native.py"
    code = """import runpy,sys
class BlockFrameworks:
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in ('torch','transformers','venturi'):
   raise AssertionError('Help imported execution framework')
sys.meta_path.insert(0,BlockFrameworks())
sys.argv=[sys.argv[1],'--help']
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    p = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "--checkpoint-scope" in p.stdout and "colocated" in p.stdout
