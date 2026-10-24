"""Exercise finite evidence recovery without a network or GPU dependency."""

import base64
import hashlib
import json
from pathlib import Path
import runpy
import zlib

import pytest

OUT = Path(__file__).resolve().parent
BASE = runpy.run_path(str(OUT / "fetch_verified.py"))
BATCH = runpy.run_path(str(OUT / "fetch_verified_batched.py"))


def record(data=b"verified evidence\n", path="nested/report.json"):
    return dict(path=path, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), payload=base64.b64encode(zlib.compress(data)).decode())


def test_full_record_and_compressed_manifest(tmp_path):
    row = record()
    manifest = dict(done=True, manifest={row["path"]: {k: row[k] for k in ["sha256", "bytes"]}}, exit_code=1)
    last = dict(done_zlib=base64.b64encode(zlib.compress(json.dumps(manifest).encode())).decode())
    result, transferred, more = BATCH["consume"](json.dumps(row) + "\n" + json.dumps(last), tmp_path)
    assert result == manifest and transferred == 1 and not more
    assert (tmp_path / row["path"]).read_bytes() == b"verified evidence\n"


def test_truncated_final_record_is_not_a_file(tmp_path):
    row = record()
    result, transferred = BASE["consume"](json.dumps(row) + '\n{"path":"partial', tmp_path)
    assert result is None and transferred == 1
    assert not (tmp_path / "partial").exists()


@pytest.mark.parametrize("path", ["../escape", "/absolute"])
def test_unsafe_destination_rejected(tmp_path, path):
    with pytest.raises(AssertionError):
        BASE["consume"](json.dumps(record(path=path)), tmp_path)


def test_wrong_hash_does_not_overwrite_verified_file(tmp_path):
    row = record(path="report.json")
    BASE["consume"](json.dumps(row), tmp_path)
    bad = dict(record(b"corrupt", "report.json"), sha256="0" * 64)
    with pytest.raises(AssertionError):
        BASE["consume"](json.dumps(bad), tmp_path)
    assert (tmp_path / "report.json").read_bytes() == b"verified evidence\n"


def test_batch_marker_does_not_fake_completion(tmp_path):
    result, transferred, more = BATCH["consume"]('{"batch_more":true}', tmp_path)
    assert result is None and transferred == 0 and more


def test_terminal_manifest_requires_every_file(tmp_path):
    row = record()
    manifest = dict(done=True, manifest={row["path"]: {k: row[k] for k in ["sha256", "bytes"]}})
    with pytest.raises(AssertionError):
        BASE["consume"](json.dumps(manifest), tmp_path)
