"""Contaminated or unavailable GPU observations must never qualify timing."""

import importlib
from pathlib import Path
import pytest


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools"))
    return importlib.import_module("cuda_activity")


def test_clean_window_and_recorded_foreign(monkeypatch):
    m = module(monkeypatch)
    with m.ObservedWindow(lambda allowed: [], {1}) as clean:
        pass
    assert clean.clean and clean.samples == 2
    with m.ObservedWindow(lambda allowed: [dict(pid=2)], {1}) as dirty:
        pass
    assert not dirty.clean and dirty.report()["foreign_observations"]


def test_observation_failure_is_not_a_clean_window(monkeypatch):
    m = module(monkeypatch)

    def failed(_):
        raise RuntimeError("NVML unavailable")

    with m.ObservedWindow(failed, {1}) as window:
        pass
    assert not window.clean and window.errors


def test_quiet_wait_is_bounded_without_process_actions(monkeypatch):
    m = module(monkeypatch)
    with pytest.raises(TimeoutError):
        m.wait_quiet(lambda _: [dict(pid=2)], {1}, timeout=0.005, interval=0.001)
    assert m.wait_quiet(lambda _: [], {1}, stable=0)["observations"] == []
