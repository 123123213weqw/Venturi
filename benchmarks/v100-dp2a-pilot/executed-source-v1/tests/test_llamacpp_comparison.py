import json
import sys

import numpy as np
import pytest

from tools.compare_llamacpp import check_gpu_processes, logit_diagnostics, run_process, summarize


def test_gpu_guard_allows_only_explicit_exact_desktop_paths():
    desktop = "2, /usr/libexec/gnome-remote-desktop-daemon, 354 MiB\n"
    with pytest.raises(RuntimeError, match="active compute"):
        check_gpu_processes(desktop, [])
    assert check_gpu_processes(desktop, ["/usr/libexec/gnome-remote-desktop-daemon"])[0]["pid"] == 2
    with pytest.raises(RuntimeError, match="active compute"):
        check_gpu_processes(desktop + "3, python, 100 MiB\n", ["/usr/libexec/gnome-remote-desktop-daemon"])
    assert check_gpu_processes("", []) == []


def test_weighted_decode_excludes_prefill_token():
    def run(count, decode, total):
        return {
            "generated_ids": [2] * count, "output_tokens": count,
            "decode_tokens": count - 1, "decode_seconds": decode,
            "total_seconds": total, "time_to_first_token_seconds": 0.5,
        }

    stats = summarize({"runs": [run(3, 1, 2), run(5, 3, 4)]})
    assert stats["decode_tokens_per_second"] == 6 / 4
    assert stats["request_tokens_per_second"] == 8 / 6
    assert not stats["identical_generated_ids"]
    assert summarize({"runs": [run(1, 0.1, 1)]})["decode_tokens_per_second"] is None


def test_diagnostics_never_claim_tolerance_pass(tmp_path):
    a, b = tmp_path / "actual", tmp_path / "reference"
    np.array([[0.1, 0.5], [0.3, 0.1]], dtype="<f4").tofile(a)
    np.array([[0.1, 0.4], [0.1, 0.3]], dtype="<f4").tofile(b)
    stats = logit_diagnostics(a, b, 2)
    assert stats["argmax_agreement"] == 0.5
    assert stats["max_abs_error"] > 0.19
    assert not stats["bit_exact"]
    assert "Diagnostic only" in stats["scope"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_diagnostics_fail(tmp_path, bad):
    a, b = tmp_path / "actual", tmp_path / "reference"
    np.array([bad], dtype="<f4").tofile(a)
    np.array([0], dtype="<f4").tofile(b)
    with pytest.raises(RuntimeError, match="nonfinite"):
        logit_diagnostics(a, b, 1)


def test_failed_child_is_not_reported_as_success(tmp_path):
    with pytest.raises(RuntimeError, match="Process failed"):
        run_process([sys.executable, "-c", "raise SystemExit(3)"], tmp_path / "log")


def test_child_report_and_timeout(tmp_path):
    file = tmp_path / "log"
    stats = run_process([sys.executable, "-c", "print('{}')"], file)
    assert json.loads(file.read_text()) == {}
    assert "sampled maximum" in stats["scope"]
    with pytest.raises(TimeoutError):
        run_process([sys.executable, "-c", "import time; time.sleep(10)"], file, timeout=0.05)
