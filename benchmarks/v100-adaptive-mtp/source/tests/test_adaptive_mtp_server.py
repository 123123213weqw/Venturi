"""Adaptive benchmark admission exposes the real selected strategy and scope."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


@pytest.mark.parametrize("depth,count", [(0, 2), (1, 2), (2, 1)])
def test_adaptive_cohort_admission_metadata(depth, count):
    path = Path(__file__).resolve().parents[1] / "tools/serve_native_benchmark.py"
    spec = spec_from_file_location("adaptive_server", path)
    tool = module_from_spec(spec)
    spec.loader.exec_module(tool)

    class Batch:
        policy = object()

        def reset(self, *args, **kw):
            raise AssertionError("Adaptive generate owns selected-bank reset")

        def generate(self, prompts, *, max_new_tokens, eos_token_ids, slots):
            self.slots = slots
            return [[i + 1] * n for i, n in enumerate(max_new_tokens)]

        def info(self):
            rows = [None] * 2
            for slot in self.slots:
                row = dict(
                    native_time_to_first_token_seconds=0.1,
                    native_generation_seconds=0.2,
                )
                if depth:
                    row["mtp"] = dict(proposed_tokens=2, accepted_tokens=1)
                rows[slot] = row
            return dict(
                selected_depth=depth,
                emitted_tokens=3 * count,
                native=dict(requests=rows),
            )

    admission = tool.CompletionBatcher(Batch(), 2, 0.05)
    try:
        futures = [admission.submit(i, [i + 1], 3) for i in range(count)]
        results = [f.result(timeout=5) for f in futures]
        assert {r["batch_cohort_id"] for r in results} == {1}
        for r in results:
            assert r["adaptive_mtp"]["selected_depth"] == depth
            assert r["adaptive_mtp"]["emitted_tokens"] == 3 * count
            assert "count once" in r["adaptive_mtp_scope"]
            assert all(v is None for v in r["batch_stats"].values())
            assert ("mtp_stats" in r) == bool(depth)
    finally:
        admission.close()


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--batch-target"],
        ["--batch-mtp"],
        ["--mtp-depth", "2"],
        ["--mtp-probe-interval", "-1"],
    ],
)
def test_conflicting_adaptive_flags_fail_before_loading(monkeypatch, extra):
    import sys

    path = Path(__file__).resolve().parents[1] / "tools/serve_native_benchmark.py"
    spec = spec_from_file_location("adaptive_flags", path)
    tool = module_from_spec(spec)
    spec.loader.exec_module(tool)
    argv = [
        str(path),
        "nonexistent-target",
        "--library",
        "nonexistent-library",
        "--mode",
        "tp",
        "--port",
        "18954",
        "--ready-file",
        "unused",
        "--adaptive-mtp",
    ]
    if extra:
        argv += ["--mtp-artifact", "nonexistent-draft"] + extra
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as e:
        tool.main()
    assert e.value.code == 2
