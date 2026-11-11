"""q4 execution qualification on the dense path: kernel bits, native vs reference.

Tier 1: the dequant kernel must equal the vectorized decode_row reference bit
for bit (including K padding-group edges). Tier 2: a complete synthetic hybrid
model in q4_g64_fp16 must produce reference-aligned logits and identical
greedy tokens through the native runtime. The qualification override of the
execution table is local to this module; production enablement happens only
in the change that lands the passing evidence.
"""

import ctypes

import pytest
import torch

from test_codec_loading import _full_artifact, _quant_payload, _write_artifact
from test_qwen_native import native_engine
from venturi.models import NativeQwenModel, ResidentQwenModel
from venturi.models import qwen_load
from venturi.codecs import dequant_q4_reference
from venturi.models.qwen_weights import QwenQ8Weights
from venturi.ninfer import NInferArtifact, row_geometry
from venturi.ops import CudaBackend

CAPABILITY = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
pytestmark = pytest.mark.skipif(
    CAPABILITY != (8, 9), reason="q4 dense qualification targets SM89"
)


def _library():
    import os

    path = os.environ["VENTURI_BLOCKS_LIBRARY"]
    lib = ctypes.CDLL(path)
    lib.venturi_q4_dequant_rows.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.venturi_q4_dequant_rows.restype = ctypes.c_int
    return lib


def _random_q4_object(device, n, k, seed):
    import numpy as np

    g = torch.Generator().manual_seed(seed)
    padded = (k + 127) // 128 * 128
    groups = padded // 64
    codes = torch.randint(0, 256, (n, groups * 32), generator=g, dtype=torch.uint8)
    scales = (
        torch.rand(n, groups, generator=g) * 0.25 + 0.03125
    ).to(torch.float16)
    codes = codes.to(device)
    scales = scales.to(device)
    return codes, scales


@pytest.mark.parametrize("k", [70, 128, 200])
def test_q4_dequant_kernel_matches_reference(k):
    lib = _library()
    n, seed = 19, 1207
    padded = (k + 127) // 128 * 128
    groups = padded // 64
    codes, scales = _random_q4_object(torch.device("cuda"), n, k, seed)
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


def _requant_q4(values_f32):
    import numpy as np

    rows, padded = values_f32.shape
    grouped = values_f32.reshape(rows, padded // 64, 64)
    amax = np.maximum(np.abs(grouped).max(axis=2), 1e-12).astype(np.float32)
    scale = np.maximum((amax / 7.0).astype("<f2").astype(np.float32), np.float32(6e-8))
    q = np.clip(np.rint(grouped / scale[:, :, None]), -8, 7).astype(np.int8) & 0x0F
    packed = (
        q[:, :, 0::2].astype(np.uint8) | (q[:, :, 1::2].astype(np.uint8) << 4)
    ).reshape(rows, padded // 2)
    return packed, scale.astype("<f2")


def _q4_artifact_from_q8(q8_artifact, out_path):
    """Miniature of tools/make_q4_artifact: q8 -> dequant -> q4 per object."""
    import json
    import struct

    import numpy as np

    from venturi.ninfer import MAGIC, row_geometry

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
            packed, scale4 = _requant_q4(values)
            obj["format"] = "q4_g64_fp16"
            obj["offset"] = len(payload)
            payload += packed.tobytes()
            payload += bytes((-len(packed.tobytes())) % 256)
            payload += scale4.tobytes()
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


@pytest.fixture
def q4_model(tmp_path, native_engine):
    b, dtype, dev = native_engine
    if dtype != torch.float16:
        pytest.skip("q4 dense path targets FP16 activations")
    from test_qwen_model import hybrid_fixture

    q8_weights, _ = hybrid_fixture(tmp_path)
    q4_path = _q4_artifact_from_q8(q8_weights.artifact, tmp_path / "hybrid-q4.ninfer")
    weights = QwenQ8Weights.from_ninfer(
        NInferArtifact(q4_path), matrix_format="q4_g64_fp16"
    )
    with _q4_enabled():
        model = NativeQwenModel.from_q8(
            weights, b, weight_mode="resident", dtype=dtype,
            device=dev, max_tokens=3, reserve_bytes=0,
        )
    reference = ResidentQwenModel.from_q8(
        weights, b, dtype=dtype, device=dev, reserve_bytes=0
    ).create_session(24, max_tokens=3, reserve_bytes=0)
    yield weights, model, reference, dev
    reference.close()
    model.close()


class _q4_enabled:
    """Qualification-campaign override: the production table lives in
    venturi.codecs and is read dynamically by the spec and descriptor gates."""

    def __enter__(self):
        import venturi.codecs as codecs
        self._old = dict(codecs.EXECUTION_QUALIFIED)
        codecs.EXECUTION_QUALIFIED[(8, 9)] = {"q8_g32_fp16", "q4_g64_fp16"}
        return self

    def __exit__(self, *exc):
        import venturi.codecs as codecs
        codecs.EXECUTION_QUALIFIED.clear()
        codecs.EXECUTION_QUALIFIED.update(self._old)


def test_q4_native_logits_and_greedy_match_reference(q4_model):
    weights, model, reference, dev = q4_model
    s = model.create_session(24)
    for ids in ([1, 2, 3, 4, 5], [6], [7, 8, 9]):
        actual = s.forward(ids, all_logits=True)
        expected = reference.forward(ids, all_logits=True).cpu()
        assert actual.shape == expected.shape
        delta = (actual.float() - expected.float()).abs().max().item()
        scale = expected.float().abs().max().item() + 1e-6
        assert delta <= 3e-2 * scale + 2e-2, (delta, scale)
    # Greedy continuation must be identical end to end.
    prompt = [1, 4, 3, 2, 1]
    expected_tokens = []
    for step in range(6):
        logits = reference.forward(prompt if step == 0 else [expected_tokens[-1]])
        expected_tokens.append(int(logits.cpu().argmax()))
    assert s.generate(prompt, max_new_tokens=6) == expected_tokens
    s.close()


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
