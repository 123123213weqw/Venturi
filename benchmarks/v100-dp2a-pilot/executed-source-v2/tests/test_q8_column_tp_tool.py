"""Research operator diagnostics are distinct from engine or occupancy claims."""

from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from inspect_cuda_occupancy import compiled_launch_geometry  # noqa: E402


def test_column_tp_resource_geometry_is_static_not_achieved():
    assert compiled_launch_geometry("_ZN3vct7partialILi1ELi4ELb0EEEv") == (128, 0)
    assert compiled_launch_geometry("_ZN3vct7partialILi4ELi8ELb1EEEv") == (256, 9216)
    assert compiled_launch_geometry("_ZN3vct7partialILi8ELi16ELb1EEEv") == (512, 18432)
    assert compiled_launch_geometry("_ZN3vct3sumEPKfS1_P6__halfi") == (256, 0)


def test_column_tp_benchmark_help_does_not_initialize_cuda():
    p = Path(__file__).parents[1] / "tools/benchmark_q8_column_tp.py"
    r = subprocess.run(
        [sys.executable, str(p), "--help"], capture_output=True, text=True, check=True
    )
    assert "not model/serving TPS" in r.stdout
    assert "--profile" in r.stdout
