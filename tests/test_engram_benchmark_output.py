"""The standalone Engram benchmark must never overwrite an earlier run."""

from pathlib import Path
import runpy

import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[1] / "tools/benchmark_engram_lookup_storage.py"
)


@pytest.mark.parametrize("existing", ["asset/lookup.gguf", "measurements/results.json"])
def test_engram_benchmark_rejects_nonempty_output_directory(tmp_path, existing, capsys):
    output = tmp_path / "benchmark"
    sentinel = output / existing
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"keep this existing output")
    main = runpy.run_path(str(_SCRIPT))["main"]

    with pytest.raises(SystemExit) as err:
        main(["--output-directory", str(output), "--rows", "128"])

    assert err.value.code == 2
    assert "must be new or empty" in capsys.readouterr().err
    assert sentinel.read_bytes() == b"keep this existing output"
    assert list(p for p in output.rglob("*") if p.is_file()) == [sentinel]
