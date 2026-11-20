"""A one-shot observer must never restart downloads or kill a live process."""

from importlib.util import module_from_spec, spec_from_file_location
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest


def tool():
    spec = spec_from_file_location(
        "wait_export_tool", Path(__file__).parents[1] / "tools/wait_export_qwen.py"
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(tmp_path, status):
    source = tmp_path / "source"
    source.mkdir()
    (source / "venturi-source.json").write_text(
        json.dumps(dict(status=status, revision="test-pin"))
    )
    return SimpleNamespace(
        source=str(source),
        output=str(tmp_path / "q8.ninfer"),
        report=str(tmp_path / "report.json"),
        job_directory=str(tmp_path / "job"),
        source_pid=1234,
        source_wait_seconds=0,
        poll_seconds=1,
    )


@pytest.mark.parametrize(
    "alive,expected",
    [
        (True, "source-wait-expired-observer-only"),
        (False, "source-stopped-before-complete"),
    ],
)
def test_wait_expiry_is_not_source_termination_and_never_restarts(
    tmp_path, monkeypatch, alive, expected
):
    m = tool()
    a = args(tmp_path, "downloading")
    monkeypatch.setattr(m, "source_alive", lambda *args: alive)

    def forbidden(*args, **kwargs):
        raise AssertionError("An incomplete source launched a subprocess")

    monkeypatch.setattr(m.subprocess, "Popen", forbidden)
    assert m.run(a) in (5, 6)
    assert (
        json.loads((Path(a.job_directory) / "state.json").read_text())["status"]
        == expected
    )
    assert (
        json.loads((Path(a.source) / "venturi-source.json").read_text())["status"]
        == "downloading"
    )


def test_complete_dependency_starts_exactly_one_converter(tmp_path, monkeypatch):
    m = tool()
    a = args(tmp_path, "complete")
    calls = []

    class Child:
        pid = 4242

        def wait(self):
            # A second observer while output does not yet exist must fail its
            # lock before launching another converter or replacing live state.
            with pytest.raises(BlockingIOError):
                m.run(a)
            assert (
                json.loads((Path(a.job_directory) / "state.json").read_text())["status"]
                == "export-running"
            )
            Path(a.output).write_bytes(b"artifact")
            Path(a.report).write_text(
                json.dumps(
                    dict(
                        output=a.output,
                        file_bytes=8,
                        artifact_id="0" * 32,
                        sha256=hashlib.sha256(b"artifact").hexdigest(),
                    )
                )
            )
            return 0

    def start(command, **kwargs):
        calls.append(command)
        return Child()

    monkeypatch.setattr(m.subprocess, "Popen", start)
    assert m.run(a) == 0 and len(calls) == 1
    assert "--report" in calls[0]
    result = json.loads((Path(a.job_directory) / "state.json").read_text())
    assert result["status"] == "complete" and result["export_pid"] == 4242
    with pytest.raises(FileExistsError):
        m.run(a)
    assert len(calls) == 1


def test_source_failure_is_reported_without_restart(tmp_path, monkeypatch):
    m = tool()
    a = args(tmp_path, "failed")

    def forbidden(*args, **kwargs):
        raise AssertionError("Source failure restarted work")

    monkeypatch.setattr(m.subprocess, "Popen", forbidden)
    assert m.run(a) == 4
    assert (
        json.loads((Path(a.job_directory) / "state.json").read_text())["status"]
        == "source-failed"
    )
