"""VerifiedQuantRows: type-aware admission, merged coalesced reads, budgets,
poisoning. Decoder math itself is covered by the pinned-source qualification;
these tests exercise the row-store lease discipline."""

import hashlib
import os
import struct

import pytest
import torch

from venturi.container.gguf.gguf_quant_rows import VerifiedQuantRows, quant_row_bytes
from venturi.container.gguf import gguf_quantization as quant
from venturi.container.model_assets import ModelAsset


def quant_fixture(root, *, type_id=10, rows=4, width=256):
    """Hand-built GGUF with one quantized table and its decoded expectation."""
    if type_id == 10:
        row_bytes = 84
        def row_payload(row):
            out = bytearray()
            out += bytes((row + 1) | ((row + 3) << 4) for _ in range(16))  # scales
            out += bytes(range(64))
            out += struct.pack("<e", 1.0 + row) + struct.pack("<e", 0.25)
            return bytes(out)
        decoder = quant.decode_q2_k_rows
    elif type_id == 18:
        row_bytes = 98
        def row_payload(row):
            out = bytearray()
            out += struct.pack("<e", 1.0)
            out += bytes(range(64))
            out += struct.pack("<8I", *(0x10000000 * ((row + k) % 16) for k in range(8)))
            return bytes(out)
        decoder = quant.decode_iq3_xxs_rows
    else:
        raise ValueError(type_id)
    payload = bytearray()
    for row in range(rows):
        payload += row_payload(row)
    expected = decoder(bytes(payload), width)
    payload += bytes((-len(payload)) % 32)  # GGUF table-end alignment
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
    name = b"blk.0.experts.weight"
    header += struct.pack("<Q", len(name)) + name + struct.pack("<IQQIQ", 2, width, rows, type_id, 0)
    raw = header + bytes((-len(header)) % 32) + bytes(payload)
    path = root / "quant.gguf"
    path.write_bytes(raw)
    revision = "b" * 40
    asset = ModelAsset(
        "quant.gguf", len(raw), hashlib.sha256(raw).hexdigest(),
        f"https://huggingface.co/venturi-fixtures/quant/resolve/{revision}/quant.gguf",
        "venturi-fixtures/quant", revision, "quant.gguf", "sparse_lookup",
    )
    return asset, expected


def open_reader(root, **kw):
    shape_kw = {k: kw.pop(k) for k in ("type_id", "rows", "width") if k in kw}
    asset, expected = quant_fixture(root, **shape_kw)
    reader = VerifiedQuantRows(
        asset, root, "blk.0.experts.weight", (expected.shape[0], expected.shape[1]),
        type_id=shape_kw.get("type_id", 10), **kw,
    )
    return reader, expected


@pytest.mark.parametrize("type_id,width", [(10, 256), (18, 256)])
def test_round_trip_merged_and_deduplicated(tmp_path, type_id, width):
    reader, expected = open_reader(tmp_path, type_id=type_id, width=width)
    with reader:
        first = torch.tensor([[3, 1], [3, 0]])
        second = torch.tensor([2])
        out_a, out_b = reader.lookup_many([first, second])
        torch.testing.assert_close(out_a, expected[[3, 1, 3, 0]].reshape(2, 2, width), rtol=0, atol=0)
        torch.testing.assert_close(out_b, expected[[2]], rtol=0, atol=0)
        unique, payload, calls = reader.last_read
        assert unique == 4 and payload == 4 * reader.row_bytes
        assert calls <= 4  # coalescing merges the span into few preads


def test_row_bounds_and_budget_fail_before_io(tmp_path):
    reader, expected = open_reader(tmp_path)
    with reader:
        with pytest.raises(ValueError, match="bounds"):
            reader.lookup_many([torch.tensor([len(expected)])])
        with pytest.raises(ValueError, match="budget"):
            reader.lookup_many([torch.zeros(8192, dtype=torch.int64)])
        assert reader.last_read is None  # nothing read on rejected requests


def test_closed_and_poisoned_states(tmp_path):
    reader, expected = open_reader(tmp_path)
    reader.close()
    with pytest.raises(RuntimeError):
        reader.lookup_many([torch.tensor([0])])
    assert reader.closed


def test_wrong_type_or_shape_rejected(tmp_path):
    asset, expected = quant_fixture(tmp_path, type_id=10)
    with pytest.raises(ValueError, match="differs"):
        VerifiedQuantRows(asset, tmp_path, "blk.0.experts.weight", expected.shape, type_id=18)
    with pytest.raises(ValueError, match="differs"):
        VerifiedQuantRows(asset, tmp_path, "blk.0.experts.weight", (expected.shape[0], 512), type_id=10)
    with pytest.raises(ValueError, match="Unsupported"):
        VerifiedQuantRows(asset, tmp_path, "blk.0.experts.weight", expected.shape, type_id=99)


def test_source_mutation_poisons_lease(tmp_path):
    reader, expected = open_reader(tmp_path)
    path = tmp_path / "quant.gguf"
    with reader:
        reader.lookup_many([torch.tensor([0])])
        before = path.stat()
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"7")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        with pytest.raises(OSError, match="changed"):
            reader.lookup_many([torch.tensor([1])])


def test_row_bytes_geometry():
    assert quant_row_bytes(10, 256) == 84
    assert quant_row_bytes(18, 512) == 196
    assert quant_row_bytes(20, 32) == 18
    assert quant_row_bytes(30, 64) == 128
    with pytest.raises(ValueError):
        quant_row_bytes(10, 100)  # not a multiple of the 256-value block
