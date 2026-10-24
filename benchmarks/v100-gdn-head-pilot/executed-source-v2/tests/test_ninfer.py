"""Independent v3 framing/packing fixtures; no native backend or model download."""

import copy
import json
import struct

import pytest

from venturi.ninfer import MAGIC, NInferArtifact, PART_MAGIC


def write_artifact(tmp_path, *, manifest=None, payload=b"\x00\x00\x80?", split=None):
    if manifest is None:
        manifest = {
            "components": {
                "text": {
                    "config": {
                        "architectures": ["Qwen3_5ForCausalLM"],
                        "max_position_embeddings": 64,
                        "vocab_size": 100,
                    }
                }
            },
            "objects": [
                {
                    "id": "w",
                    "kind": "tensor",
                    "shape": [1],
                    "format": "fp32",
                    "layout": "contiguous_le_v1",
                    "offset": 0,
                    "bytes": len(payload),
                }
            ],
            "bindings": {"text/weight": {"object": "w"}},
            "uses": [],
        }
    manifest = copy.deepcopy(manifest)
    path = tmp_path / "model.ninfer"
    artifact_id = bytes(range(16))
    chunks = [payload] if split is None else [payload[:split], payload[split:]]
    manifest["files"] = [{"path": None, "payload_bytes": len(chunks[0])}]
    if split is not None:
        manifest["files"].append(
            {"path": "continuation.bin", "payload_bytes": len(chunks[1])}
        )
    encoded = json.dumps(manifest).encode()
    header = MAGIC + struct.pack("<Q", len(encoded)) + artifact_id + encoded
    path.write_bytes(header + bytes((-len(header)) % 4096) + chunks[0])
    if split is not None:
        part_header = PART_MAGIC + struct.pack("<Q", 1) + artifact_id
        (tmp_path / "continuation.bin").write_bytes(
            part_header + bytes(4096 - 32) + chunks[1]
        )
    return path


def test_framing_binding_parts_and_cross_file_reads(tmp_path):
    path = write_artifact(tmp_path, split=1)
    artifact = NInferArtifact(path)
    assert artifact.read_object("w") == b"\x00\x00\x80?"
    assert artifact.read_object("w", offset=1, size=2) == b"\x00\x80"
    assert artifact.read_object("w", offset=4, size=0) == b""
    assert artifact.summary()["formats"] == {"fp32": 1}
    for kwargs in ({"offset": -1}, {"size": 5}, {"size": True}, {"max_bytes": 1}):
        with pytest.raises(ValueError):
            artifact.read_object("w", **kwargs)


@pytest.mark.parametrize(
    "corruption", ["magic", "length", "truncated", "part-id", "part-index", "padding"]
)
def test_invalid_file_framing(tmp_path, corruption):
    path = write_artifact(tmp_path, split=2)
    file = tmp_path / "continuation.bin" if corruption.startswith("part") else path
    data = bytearray(file.read_bytes())
    if corruption == "magic":
        data[0] ^= 1
    if corruption == "length":
        data[8:16] = struct.pack("<Q", 2**40)
    if corruption == "truncated":
        data.pop()
    if corruption == "part-id":
        data[16] ^= 1
    if corruption == "part-index":
        data[8] = 2
    if corruption == "padding":
        data[4095] = 1
    file.write_bytes(data)
    with pytest.raises(ValueError):
        NInferArtifact(path)


@pytest.mark.parametrize(
    "change",
    [
        "overlap",
        "shape",
        "size",
        "binding",
        "range",
        "resource",
        "use",
        "format",
        "root",
    ],
)
def test_invalid_directory(tmp_path, change):
    base = NInferArtifact(write_artifact(tmp_path)).manifest
    base.pop("files")
    if change == "overlap":
        base["objects"].append(copy.deepcopy(base["objects"][0]))
    if change == "shape":
        base["objects"][0]["shape"] = [True]
    if change == "size":
        base["objects"][0]["shape"] = [2]
    if change == "binding":
        base["bindings"]["text/weight"] = {"object": "missing"}
    if change == "range":
        base["bindings"]["text/weight"] = {"parts": [{"object": "w", "range": [0, 2]}]}
    if change == "resource":
        base["components"]["text"]["resources"] = {"tokenizer.json": "w"}
    if change == "use":
        base["uses"] = [{"parameter": "missing", "input": "x"}]
    if change == "format":
        base["objects"][0]["format"] = "unknown"
    if change == "root":
        base["unexpected"] = 1
    with pytest.raises(ValueError):
        NInferArtifact(write_artifact(tmp_path, manifest=base))


@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_quantized_rows_against_independent_bitstream_encoder(tmp_path, bits):
    # Deliberately cross group boundaries and physical K padding. Construct
    # base/high planes lane-by-lane without production geometry/decoder helpers.
    rows, columns, padded = 2, 70, 128
    group = 32 if bits == 8 else 64
    expected, base, high, scales = [], bytearray(), bytearray(), bytearray()
    for row in range(rows):
        values = []
        for begin in range(0, padded, group):
            scale = 0.125 * (row + 1) if begin < columns else 0.0
            words = []
            for lane in range(group):
                col = begin + lane
                low = -(2 ** (bits - 1)) + (1 if bits == 8 else 0)
                signed = (
                    low + ((col + row * 3) % (2**bits - (1 if bits == 8 else 0)))
                    if col < columns
                    else 0
                )
                words.append(signed % 2**bits)
                if col < columns:
                    values.append(signed * scale)
            if bits == 8:
                base.extend(words)
            else:
                base.extend(
                    words[i] % 16 | (words[i + 1] % 16) << 4 for i in range(0, group, 2)
                )
                if bits > 4:
                    stream = sum(
                        (word >> 4) << (i * (bits - 4)) for i, word in enumerate(words)
                    )
                    high.extend(stream.to_bytes(group * (bits - 4) // 8, "little"))
            scales.extend(struct.pack("<e", scale))
        expected.append(values)
    payload = (
        bytes(base)
        + bytes((-len(base)) % 256)
        + bytes(high)
        + bytes((-len(high)) % 256)
        + bytes(scales)
    )
    format = f"q{bits}_g{group}_fp16"
    manifest = {
        "components": {"text": {"config": {}}},
        "objects": [
            {
                "id": "w",
                "kind": "tensor",
                "shape": [rows, columns],
                "format": format,
                "layout": "row_split_k128_v1",
                "offset": 0,
                "bytes": len(payload),
            }
        ],
        "bindings": {
            "text/weight": {
                "parts": [
                    {"object": "w", "range": [0, columns]},
                    {"object": "w", "range": [columns, 2 * columns]},
                ]
            }
        },
        "uses": [],
    }
    artifact = NInferArtifact(
        write_artifact(tmp_path, manifest=manifest, payload=payload, split=127)
    )
    for row in range(rows):
        assert artifact.decode_row("w", row) == expected[row]
    with pytest.raises(ValueError):
        artifact.decode_row("w", rows)
