"""Transport capability evidence must not imply model graph qualification."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import sys


def test_probe_help_has_no_cuda_initialization():
    p = Path(__file__).resolve().parents[1] / "tools/probe_cuda_multigpu_graph.py"
    result = subprocess.run(
        [sys.executable, str(p), "--help"], capture_output=True, text=True, check=True
    )
    assert "not a model or serving benchmark" in result.stdout
    assert "--library" in result.stdout


def test_probe_cuda_error_preserves_exact_code():
    p = Path(__file__).resolve().parents[1] / "tools/probe_cuda_multigpu_graph.py"
    spec = spec_from_file_location("graph_probe", p)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    error = module.CudaError(
        "cudaStreamWaitEvent", 900, "cudaErrorStreamCaptureUnsupported"
    )
    assert error.code == 900 and error.name == "cudaErrorStreamCaptureUnsupported"
    assert str(error) == "cudaStreamWaitEvent: cudaErrorStreamCaptureUnsupported (900)"
