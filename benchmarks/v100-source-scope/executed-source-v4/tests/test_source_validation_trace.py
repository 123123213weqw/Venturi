"""Synthetic strace parser tests; no checkpoint, compiler or GPU is required."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

spec = spec_from_file_location(
    "source_validation_trace",
    Path(__file__).resolve().parents[1] / "tools/summarize_native_source_validation.py",
)
module = module_from_spec(spec)
spec.loader.exec_module(module)
summarize = module.summarize


def report():
    return dict(
        checkpoint_paths=["/models/target.ninfer", "/models/draft.ninfer"],
        calls=[dict(start_unix=100, end_unix=101, output_ids_equal=True)],
    )


def test_metadata_pairs_and_interval_scope():
    text = """50 99.0 newfstatat(9</models/target.ninfer>, "", {}, AT_EMPTY_PATH) = 0 <0.001000>
50 100.1 newfstatat(9</models/target.ninfer>, "", {}, AT_EMPTY_PATH) = 0 <0.000004>
50 100.2 newfstatat(AT_FDCWD</repo>, "/models/target.ninfer", {}, 0) = 0 <0.000003>
50 100.3 fstat(10</models/draft.ninfer>, {}) = 0 <0.000002>
50 100.4 stat("/models/draft.ninfer", {}) = 0 <0.000003>
51 100.5 newfstatat(8</models/target.ninfer.backup>, "", {}, AT_EMPTY_PATH) = 0 <0.000003>
50 101.1 stat("/models/target.ninfer", {}) = 0 <0.000003>
"""
    r = summarize(text, report())
    assert r["checkpoint_calls"] == dict(descriptor=2, pathname=2)
    assert r["checkpoint_calls_by_path"]["/models/target.ninfer"] == dict(
        descriptor=1, pathname=1
    )
    assert r["all_traced_metadata_calls_in_intervals"] == 5
    assert not r["failures"]


def test_unfinished_syscall_and_tid_isolation():
    text = """[pid 50] 100.1 newfstatat(9</models/target.ninfer>, "", <unfinished ...>
[pid 51] 100.2 stat("/models/draft.ninfer", {}) = 0 <0.000003>
[pid 50] 100.3 <... newfstatat resumed>{}, AT_EMPTY_PATH) = 0 <0.000004>
"""
    r = summarize(text, report())
    assert r["checkpoint_calls"] == dict(pathname=1, descriptor=1)
    assert not r["failures"] and not r["unmatched_pending_tids"]


def test_failure_or_missing_duration_not_silently_ignored():
    text = """50 100.1 stat("/models/target.ninfer", 0) = -1 ENOENT (No such file or directory) <0.000003>
50 100.2 fstat(9</models/target.ninfer>, {}) = 0
"""
    r = summarize(text, report())
    assert len(r["failures"]) == 3
    assert r["checkpoint_calls"] == dict(pathname=1, descriptor=1)


def test_instrumentation_not_declared_throughput_or_engine_change():
    r = summarize("", report())
    assert not r["engine_change"] and r["model_outputs_exact"]
    assert r["raw_native_calls"] == 1
    assert r["instrumented_syscall_seconds"] == {}


def test_changed_source_cadence_is_not_reported_as_unchanged_engine():
    data = report()
    data["source_guards_unchanged"] = False
    r = summarize("", data)
    assert r["engine_change"] and r["source_check_cadence_changed"]
