"""Storage integrity and official-format interoperability, not legacy framing."""

import json
import os

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from venturi.container.safetensors import SafetensorsContainer, SafetensorsWriter


def frame(path, header, payload=b"\x01\x02\x03\x04"):
    encoded = (
        header.encode() if isinstance(header, str) else json.dumps(header).encode()
    )
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)
    return path


def descriptor(**updates):
    return dict(dtype="U8", shape=[4], data_offsets=[0, 4], **updates)


@pytest.mark.parametrize(
    "spec",
    [
        {"dtype": "UNKNOWN", "shape": [4], "data_offsets": [0, 4]},
        {"dtype": [], "shape": [4], "data_offsets": [0, 4]},
        {"dtype": "U8", "shape": [True], "data_offsets": [0, 4]},
        {"dtype": "U8", "shape": [-4], "data_offsets": [0, 4]},
        {"dtype": "U8", "shape": [2**63], "data_offsets": [0, 4]},
        {"dtype": "U8", "shape": [4] * 17, "data_offsets": [0, 4]},
        {"dtype": "U8", "shape": [3], "data_offsets": [0, 4]},
        {"dtype": "F16", "shape": [4], "data_offsets": [0, 4]},
        {"dtype": "U8", "shape": [4], "data_offsets": [False, 4]},
        {"dtype": "U8", "shape": [4], "data_offsets": [-1, 3]},
        {"dtype": "U8", "shape": [4], "data_offsets": [4, 0]},
        {"dtype": "U8", "shape": [4], "data_offsets": [0, 5]},
        {
            "dtype": "U8",
            "shape": [4],
            "data_offsets": [0, 4],
            "object": "not-supported",
        },
    ],
)
def test_malformed_tensor_schema_rejected(tmp_path, spec):
    with pytest.raises(ValueError):
        SafetensorsContainer(frame(tmp_path / "bad.safetensors", {"x": spec}))


@pytest.mark.parametrize(
    "header",
    [
        '{"x":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"x":{}}',
        '{"x":{"dtype":"U8","dtype":"F32","shape":[4],"data_offsets":[0,4]}}',
        '{"__metadata__":{"source":"a","source":"b"}}',
        '{"__metadata__":{"x":NaN}}',
        {"__metadata__": {"source": {"objects": []}}},
        [],
        {"": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}},
    ],
)
def test_json_and_metadata_are_strict(tmp_path, header):
    with pytest.raises(ValueError):
        SafetensorsContainer(frame(tmp_path / "bad.safetensors", header))


@pytest.mark.parametrize("ranges", [[(0, 3), (2, 4)], [(0, 1), (2, 4)], [(0, 3)]])
def test_no_overlap_holes_or_trailing_bytes(tmp_path, ranges):
    header = {
        str(i): dict(dtype="U8", shape=[b - a], data_offsets=[a, b])
        for i, (a, b) in enumerate(ranges)
    }
    with pytest.raises(ValueError):
        SafetensorsContainer(frame(tmp_path / "bad.safetensors", header))


@pytest.mark.parametrize(
    "data",
    [b"", b"\x03", (1000).to_bytes(8, "little") + b"{}", (0).to_bytes(8, "little")],
)
def test_truncated_or_impossible_header(tmp_path, data):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        SafetensorsContainer(path)


def test_header_read_budget_and_tensor_read_bounds(tmp_path):
    path = frame(tmp_path / "model.safetensors", {"x": descriptor()})
    with pytest.raises(ValueError):
        SafetensorsContainer(path, max_header_bytes=2)
    source = SafetensorsContainer(path)
    for args in [dict(offset=-1), dict(size=5), dict(size=True), dict(max_bytes=3)]:
        with pytest.raises(ValueError):
            source.read_bytes("x", **args)
    assert source.read_bytes("x", 4, 0) == b""
    assert source.read_bytes("x", 1, 2) == b"\x02\x03"


def test_file_identity_replacement_and_symlinks(tmp_path):
    path = frame(tmp_path / "model.safetensors", {"x": descriptor()})
    alias = tmp_path / "alias"
    alias.symlink_to(path)
    with pytest.raises(ValueError):
        SafetensorsContainer(alias)
    source = SafetensorsContainer(path)
    replacement = frame(tmp_path / "replacement", {"x": descriptor()})
    os.replace(replacement, path)
    with pytest.raises(RuntimeError):
        source.read_bytes("x")


def test_official_reader_writer_scalars_empty_and_physical_order(tmp_path):
    path = tmp_path / "official.safetensors"
    tensors = {
        "matrix.codes": torch.tensor([[1, -2, 3]], dtype=torch.int8),
        "matrix.scales": torch.tensor([[0.125]], dtype=torch.float16),
        "empty": torch.empty((0, 3)),
        "scalar": torch.tensor(7.0),
    }
    save_file(tensors, str(path), metadata={"source": "synthetic"})
    source = SafetensorsContainer(path)
    assert source.logical_offset("matrix.scales") < source.logical_offset(
        "matrix.codes"
    )
    assert source.tensor_info("scalar").shape == ()
    assert source.read_bytes("empty") == b""
    assert len(source.byte_spans()) == 1
    with safe_open(path, framework="pt") as official:
        for name in tensors:
            expected = (
                official.get_tensor(name)
                .reshape(-1)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
            assert source.read_bytes(name) == expected


def test_writer_interleaved_tensor_chunks_and_no_clobber(tmp_path):
    path = tmp_path / "model.safetensors"
    with SafetensorsWriter(
        path, {"scales": ("F16", (1,)), "codes": ("I8", (4,))}, {"source": "test"}
    ) as writer:
        writer.write_slice("codes", b"\x01\x02")
        writer.write_slice("scales", b"\x00\x3c")
        with pytest.raises(ValueError):
            writer.write_slice("codes", b"\x03", 0)
        writer.write_slice("codes", b"\x03\x04", 2)
        writer.publish()
        with pytest.raises(ValueError):
            writer.write_slice("codes", b"", 4)
    assert not list(tmp_path.glob("*.partial"))
    with safe_open(path, framework="pt") as official:
        assert torch.equal(
            official.get_tensor("codes"), torch.tensor([1, 2, 3, 4], dtype=torch.int8)
        )
        assert official.get_tensor("scales").item() == 1
    with pytest.raises(FileExistsError):
        SafetensorsWriter(path, {}, {})


def test_writer_partial_abort_and_racing_destination_preserved(tmp_path):
    path = tmp_path / "model.safetensors"
    with pytest.raises(ValueError):
        with SafetensorsWriter(path, {"x": ("U8", (4,))}, {}) as writer:
            writer.write_slice("x", b"\x01")
            writer.publish()
    assert not path.exists() and not list(tmp_path.glob("*.partial"))
    with pytest.raises(FileExistsError):
        with SafetensorsWriter(path, {"x": ("U8", (1,))}, {}) as writer:
            writer.write_slice("x", b"\x01")
            path.write_bytes(b"owned by another publisher")
            writer.publish()
    assert path.read_bytes() == b"owned by another publisher"
    assert not list(tmp_path.glob("*.partial"))


def test_writer_initialization_failure_cleans_owned_temporary(tmp_path, monkeypatch):
    import tempfile

    create = tempfile.NamedTemporaryFile

    def short_header(*args, **kwargs):
        stream = create(*args, **kwargs)
        stream.write = lambda data: 0
        return stream

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", short_header)
    with pytest.raises(OSError, match="Short safetensors header"):
        SafetensorsWriter(tmp_path / "model.safetensors", {"x": ("U8", (1,))}, {})
    assert list(tmp_path.iterdir()) == []
