"""Row-codec execution qualification harness: parameterized over codecs.

The machinery is format-agnostic: build a synthetic hybrid artifact in any
row codec (generic symmetric requantizer), run native vs the torch reference
on the SAME weights, and require aligned logits plus identical greedy tokens.
A codec joins EXECUTABLE_CODECS below when its native path exists; the
production gate (venturi.codecs.EXECUTION_QUALIFIED) flips only in the change
that lands the passing evidence. Adding q5/q6 later = kernel + one line here.
"""

import ctypes

import pytest
import torch

from test_codec_loading import _full_artifact
from test_qwen_native import native_engine
from venturi.codecs import codec_for, dequant_q4_reference
from venturi.models import NativeQwenModel, ResidentQwenModel
from venturi.models.qwen_weights import QwenQ8Weights
from venturi.ninfer import NInferArtifact, row_geometry
from venturi.ops import CudaBackend

CAPABILITY = torch.cuda.get_device_capability() if torch.cuda.is_available() else None

# Codecs with a native execution path today. q5/q6 additionally need the
# third (high-bit) plane in the matrix ABI before they can join.
EXECUTABLE_CODECS = ["q4_g64_fp16"]

pytestmark = pytest.mark.skipif(
    CAPABILITY != (8, 9), reason="row-codec dense qualification targets SM89"
)


def _library():
    import os

    lib = ctypes.CDLL(os.environ["VENTURI_BLOCKS_LIBRARY"])
    lib.venturi_q4_dequant_rows.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.venturi_q4_dequant_rows.restype = ctypes.c_int
    return lib


def _random_codes_scales(device, n, k, fmt, seed):
    """Random but contract-valid codes/scales for one row codec matrix."""
    codec = codec_for(fmt)
    padded = (k + 127) // 128 * 128
    groups = padded // codec.group
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 256, (n, groups * codec.bits * codec.group // 8), generator=g, dtype=torch.uint8)
    scales = (
        torch.rand(n, groups, generator=g) * 0.25 + 0.03125
    ).to(torch.float16)
    return codes.to(device), scales.to(device)


@pytest.mark.parametrize("fmt,k", [("q4_g64_fp16", 70), ("q4_g64_fp16", 128), ("q4_g64_fp16", 200)])
def test_row_codec_dequant_kernel_matches_reference(fmt, k):
    lib = _library()
    n, seed = 19, 1207
    padded = (k + 127) // 128 * 128
    groups = padded // 64
    codes, scales = _random_codes_scales(torch.device("cuda"), n, k, fmt, seed)
    out = torch.empty(n, k, dtype=torch.float16, device="cuda")
    rc = lib.venturi_q4_dequant_rows(
        codes.data_ptr(), scales.data_ptr(), out.data_ptr(),
        0, n, k, padded, torch.cuda.current_stream().cuda_stream,
    )
    assert rc == 0
    # Rebuild the stored-plane byte image (codes | 256-align gap | scales) and
    # dequantize it through the reference path.
    base = n * groups * 32
    raw = (
        codes.cpu().numpy().tobytes()
        + bytes((-base) % 256)
        + scales.cpu().numpy().astype("<f2").tobytes()
    )
    assert len(raw) == row_geometry((n, k), "q4_g64_fp16")[-1]

    class _FakeArtifact:
        def read_object(self, _id, offset=0, size=None, max_bytes=None):
            stop = offset + (size if size is not None else len(raw) - offset)
            return raw[offset:stop]

    expected = dequant_q4_reference(
        _FakeArtifact(), {"shape": [n, k], "bytes": len(raw), "id": "x"}
    )
    assert torch.equal(out.cpu(), expected), float(
        (out.cpu().float() - expected.float()).abs().max()
    )


def _requant(values_f32, fmt):
    """Generic symmetric requantizer for any row codec (with high plane)."""
    import numpy as np

    codec = codec_for(fmt)
    rows, padded = values_f32.shape
    grouped = values_f32.reshape(rows, padded // codec.group, codec.group)
    amax = np.maximum(np.abs(grouped).max(axis=2), 1e-12).astype(np.float32)
    scale = np.maximum(
        (amax / float(2 ** (codec.bits - 1) - 1)).astype("<f2").astype(np.float32),
        np.float32(6e-8),
    )
    q = np.clip(
        np.rint(grouped / scale[:, :, None]),
        -(2 ** (codec.bits - 1)),
        2 ** (codec.bits - 1) - 1,
    ).astype(np.int32)
    base = q & 0x0F
    high = None
    if codec.bits == 8:
        packed = q.astype(np.uint8).reshape(rows, -1)
    else:
        packed = (
            base[:, :, 0::2].astype(np.uint8)
            | (base[:, :, 1::2].astype(np.uint8) << 4)
        ).reshape(rows, padded // 2)
        if codec.high_plane:
            # High bits: (bits-4) per lane, LSB first, little-endian stream.
            bits_per = codec.bits - 4
            lanes = (q >> 4).reshape(rows, -1)
            bits = np.zeros((rows, lanes.shape[1] * bits_per), dtype=np.uint8)
            for bit in range(bits_per):
                bits[:, bit::bits_per] = (lanes >> bit) & 1
            high = np.packbits(bits, axis=1, bitorder="little")
    return packed, high, scale.astype("<f2")


def _codec_artifact_from_q8(q8_artifact, out_path, fmt):
    """Generic miniature converter: q8 -> dequant -> requant(fmt) per object."""
    import json
    import struct

    import numpy as np

    from venturi.ninfer import MAGIC

    codec = codec_for(fmt)
    manifest = json.loads(json.dumps(q8_artifact.manifest))
    payload = bytearray()
    for obj in manifest["objects"]:
        payload += bytes((-len(payload)) % 256)
        if obj.get("format") == "q8_g32_fp16":
            n, k = obj["shape"]
            _, _, _, _, scales_off, _ = row_geometry((n, k), "q8_g32_fp16")
            groups8 = ((k + 127) // 128 * 128) // 32
            codes = np.frombuffer(
                q8_artifact.read_object(
                    obj["id"], size=n * groups8 * 32, max_bytes=n * groups8 * 32
                ),
                np.int8,
            ).reshape(n, groups8 * 32)
            scales = np.frombuffer(
                q8_artifact.read_object(
                    obj["id"], offset=scales_off, size=n * groups8 * 2,
                    max_bytes=n * groups8 * 2,
                ),
                "<f2",
            ).reshape(n, groups8)
            values = codes.astype(np.float32) * np.repeat(
                scales.astype(np.float32), 32, axis=1
            )
            packed, high, scale = _requant(values, fmt)
            obj["format"] = fmt
            obj["offset"] = len(payload)
            codes_b = packed.tobytes()
            payload += codes_b + bytes((-len(codes_b)) % 256)
            if high is not None:
                high_b = high.tobytes()
                payload += high_b + bytes((-len(high_b)) % 256)
            payload += scale.tobytes()
            obj["bytes"] = len(payload) - obj["offset"]
        else:
            obj["offset"] = len(payload)
            payload += q8_artifact.read_object(obj["id"], max_bytes=obj["bytes"])
    manifest["files"] = [{"path": None, "payload_bytes": len(payload)}]
    encoded = json.dumps(manifest).encode()
    header = MAGIC + struct.pack("<Q", len(encoded)) + bytes(range(16)) + encoded
    out_path.write_bytes(
        header + bytes((-len(header)) % 4096) + bytes(payload)
    )
    return out_path


class _codec_enabled:
    """Qualification-campaign override: the production table lives in
    venturi.codecs and is read dynamically by the spec and descriptor gates."""

    def __init__(self, fmt):
        self._fmt = fmt

    def __enter__(self):
        import venturi.codecs as codecs
        cap = CAPABILITY
        self._old = dict(codecs.EXECUTION_QUALIFIED)
        codecs.EXECUTION_QUALIFIED[cap] = set(codecs.EXECUTION_QUALIFIED.get(cap, set())) | {self._fmt}
        return self

    def __exit__(self, *exc):
        import venturi.codecs as codecs
        codecs.EXECUTION_QUALIFIED.clear()
        codecs.EXECUTION_QUALIFIED.update(self._old)


@pytest.mark.parametrize("fmt", EXECUTABLE_CODECS)
def test_row_codec_native_logits_and_greedy_match_reference(tmp_path, native_engine, fmt):
    b, dtype, dev = native_engine
    if dtype != torch.float16:
        pytest.skip("row-codec dense path targets FP16 activations")
    from test_qwen_model import hybrid_fixture

    q8_weights, _ = hybrid_fixture(tmp_path)
    path = _codec_artifact_from_q8(
        q8_weights.artifact, tmp_path / f"hybrid-{fmt}.ninfer", fmt
    )
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(path), matrix_format=fmt)
    with _codec_enabled(fmt):
        model = NativeQwenModel.from_q8(
            weights, b, weight_mode="resident", dtype=dtype,
            device=dev, max_tokens=3, reserve_bytes=0,
        )
    reference = ResidentQwenModel.from_q8(
        weights, b, dtype=dtype, device=dev, reserve_bytes=0
    ).create_session(24, max_tokens=3, reserve_bytes=0)
    try:
        s = model.create_session(24)
        for ids in ([1, 2, 3, 4, 5], [6], [7, 8, 9]):
            actual = s.forward(ids, all_logits=True)
            expected = reference.forward(ids, all_logits=True).cpu()
            assert actual.shape == expected.shape
            delta = (actual.float() - expected.float()).abs().max().item()
            scale = expected.float().abs().max().item() + 1e-6
            assert delta <= 3e-2 * scale + 2e-2, (delta, scale)
        prompt = [1, 4, 3, 2, 1]
        expected_tokens = []
        for step in range(6):
            logits = reference.forward(prompt if step == 0 else [expected_tokens[-1]])
            expected_tokens.append(int(logits.cpu().argmax()))
        assert s.generate(prompt, max_new_tokens=6) == expected_tokens
        s.close()
    finally:
        reference.close()
        model.close()


def test_never_qualified_codec_still_refused(tmp_path, native_engine):
    """q4 is production-qualified on SM89 since 2026-11-11; q6 (registered for
    loading, never qualified) must still be refused for native creation."""
    b, dtype, dev = native_engine
    path = _full_artifact(tmp_path, "q6_g64_fp16")
    weights = QwenQ8Weights.from_ninfer(
        NInferArtifact(path), matrix_format="q6_g64_fp16"
    )
    with pytest.raises(ValueError, match="qualification"):
        NativeQwenModel.from_q8(
            weights, b, weight_mode="resident", dtype=dtype,
            device=dev, max_tokens=4, reserve_bytes=0,
        )
