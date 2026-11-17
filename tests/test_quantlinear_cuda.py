"""Integrated QuantLinear numerical and failure contracts (not just NativeQ5)."""

import ctypes as C
from contextlib import closing
import pytest
import torch
from test_row_codec_execution import _random_planes
from venturi.quantization.codecs import dequantize_planes
from test_qwen_native import native_engine
from qwen_native_runtime import native_runtime

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Explicit CUDA libraries required"
)


@pytest.mark.parametrize("bits", [4, 5])
@pytest.mark.parametrize("k", [255, 257, 511, 512])
@pytest.mark.parametrize("t", [1, 4])
def test_fused_impulse_keeps_last_valid_column(native_engine, bits, k, t):
    b, dtype, dev = native_engine
    if dtype != torch.float16:
        pytest.skip("row codecs use explicit FP16 activations")
    n = 19
    padded = (k + 127) // 128 * 128
    codes = torch.zeros(n, padded // 2, dtype=torch.uint8, device=dev)
    codes[:, (k - 1) // 2] = 1 << (4 * ((k - 1) % 2))
    high = (
        torch.zeros(n, padded // 8, dtype=torch.uint8, device=dev)
        if bits == 5
        else None
    )
    scales = torch.ones(n, padded // 64, dtype=torch.float16, device=dev)
    x = torch.zeros(t, k, dtype=torch.float16, device=dev)
    x[:, -1] = 1
    y = torch.full((t, n), float("nan"), dtype=torch.float16, device=dev)
    fn = getattr(b._native, f"venturi_q{bits}_fused_run")
    pointers = (
        [x.data_ptr(), codes.data_ptr()]
        + ([high.data_ptr()] if high is not None else [])
        + [scales.data_ptr(), y.data_ptr()]
    )
    fn.argtypes = [C.c_void_p] * len(pointers) + [C.c_int] * 5 + [C.c_void_p]
    fn.restype = C.c_int
    rc = fn(*pointers, t, n, k, padded, 0, torch.cuda.current_stream(dev).cuda_stream)
    b._check(rc) if hasattr(b, "_check") else None
    assert rc == 0
    assert torch.equal(y.cpu(), torch.ones(t, n, dtype=torch.float16))


@pytest.mark.parametrize("bits", [4, 5])
@pytest.mark.parametrize("k", [255, 512])
def test_fused_uses_declared_weight_contract(native_engine, bits, k):
    b, dtype, dev = native_engine
    if dtype != torch.float16:
        pytest.skip("FP16 row codec")
    n, t = 19, 4
    padded = (k + 127) // 128 * 128
    c, h, s = _random_planes(dev, n, k, f"q{bits}_g64_fp16", 613)
    x = (torch.randn(t, k, device=dev) * 0.2).half()
    y = torch.empty(t, n, device=dev, dtype=dtype)
    fn = getattr(b._native, f"venturi_q{bits}_fused_run")
    ptr = (
        [x.data_ptr(), c.data_ptr()]
        + ([h.data_ptr()] if h is not None else [])
        + [s.data_ptr(), y.data_ptr()]
    )
    fn.argtypes = [C.c_void_p] * len(ptr) + [C.c_int] * 5 + [C.c_void_p]
    fn.restype = C.c_int
    assert fn(*ptr, t, n, k, padded, 0, torch.cuda.current_stream(dev).cuda_stream) == 0
    w = dequantize_planes(
        c.cpu(),
        s.cpu(),
        h.cpu() if h is not None else None,
        (n, k),
        f"q{bits}_g64_fp16",
    )
    if bits == 4:
        integers = dequantize_planes(
            c.cpu(), torch.ones_like(s.cpu()), None, (n, k), "q4_g64_fp16"
        ).double()
        w = integers * s.cpu().double().repeat_interleave(64, -1)[:, :k]
    reference = (x.cpu().double() @ w.double().T).half()
    torch.testing.assert_close(y.cpu(), reference, rtol=2e-3, atol=2e-3)


def test_hmma_geometry_error_is_negative_and_owned(native_engine):
    b, dtype, dev = native_engine
    fn = b._native.venturi_q5_hmma_run
    fn.argtypes = [C.c_void_p] * 5 + [C.c_int] * 4 + [C.c_void_p]
    fn.restype = C.c_int
    assert fn(None, None, None, None, None, 1, 16, 64, 128, None) < 0
    b._native.venturi_blocks_error.restype = C.c_char_p
    assert b"HMMA" in b._native.venturi_blocks_error()


@pytest.mark.parametrize("t", [1, 4, 8, 17])
@pytest.mark.parametrize("n", [19, 129])
@pytest.mark.parametrize("unaligned", [False, True])
def test_hmma_matches_independent_decoded_weight(native_engine, t, n, unaligned):
    b, dtype, dev = native_engine
    if dtype != torch.float16:
        pytest.skip("FP16 row codec")
    k = 512
    c, h, s = _random_planes(dev, n, k, "q5_g64_fp16", 971)
    x = (torch.randn(t, k, device=dev) * 0.2).half()
    if unaligned:
        x = (
            torch.empty(x.numel() + 1, device=dev, dtype=x.dtype)[1:]
            .copy_(x.flatten())
            .view_as(x)
        )
        c = (
            torch.empty(c.numel() + 1, device=dev, dtype=c.dtype)[1:]
            .copy_(c.flatten())
            .view_as(c)
        )
        assert x.is_contiguous() and c.is_contiguous()
        assert x.data_ptr() % 16 == 2 and c.data_ptr() % 4 == 1
    y = torch.empty(t, n, device=dev, dtype=dtype)
    fn = b._native.venturi_q5_hmma_run
    fn.argtypes = [C.c_void_p] * 5 + [C.c_int] * 4 + [C.c_void_p]
    fn.restype = C.c_int
    assert (
        fn(
            x.data_ptr(),
            c.data_ptr(),
            h.data_ptr(),
            s.data_ptr(),
            y.data_ptr(),
            t,
            n,
            k,
            k,
            torch.cuda.current_stream(dev).cuda_stream,
        )
        == 0
    )
    w = dequantize_planes(c.cpu(), s.cpu(), h.cpu(), (n, k), "q5_g64_fp16")
    torch.testing.assert_close(
        y.cpu(), (x.cpu().double() @ w.double().T).half(), rtol=2e-3, atol=2e-3
    )


@pytest.mark.parametrize("fmt", ["q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16", "mixed"])
def test_integrated_plan_chunks_states_and_greedy(tmp_path, native_engine, fmt):
    from contextlib import ExitStack
    from test_qwen_model import hybrid_fixture
    from test_row_codec_execution import _codec_artifact_from_q8, _codec_enabled
    from venturi.models import NativeQwenModel
    from venturi.models.qwen.qwen_weights import QwenWeights

    b, dtype, dev = native_engine
    if dtype != torch.float16:
        pytest.skip("FP16 row codec")
    base, _ = hybrid_fixture(tmp_path)
    overrides = (
        {
            "text/token_embedding": "q5_g64_fp16",
            "text/layers/0/mlp/down": "q6_g64_fp16",
            "text/layers/1/mlp/up": "q4_g64_fp16",
        }
        if fmt == "mixed"
        else None
    )
    path = _codec_artifact_from_q8(
        base.source,
        tmp_path / "quantlinear.safetensors",
        "q8_g32_fp16" if fmt == "mixed" else fmt,
        overrides,
    )
    weights = QwenWeights.open(path)
    runtime = native_runtime(b)
    models = []
    try:
        with ExitStack() as gates:
            for codec in ("q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16"):
                gates.enter_context(_codec_enabled(codec))
            for implementation in ("auto", "dequant"):
                models.append(
                    NativeQwenModel.from_q8(
                        weights,
                        runtime,
                        weight_mode="resident",
                        dtype=dtype,
                        device=dev,
                        max_tokens=32,
                        reserve_bytes=0,
                        linear_implementation=implementation,
                    )
                )
        for model in models:
            for codec, contract in model.projection_info["linear_contracts"].items():
                if codec != "q8_g32_fp16":
                    assert contract["selected_family"] == "chunked-dense"
                    assert contract["weight_decode"] == "fp16-rounded-q-times-scale"
        sessions = [m.create_session(128) for m in models]
        try:
            for ids in ([1], [2, 3, 4, 5], list(range(8)), list(range(16)), [7, 6, 5]):
                values = [s.forward(ids, all_logits=True) for s in sessions]
                torch.testing.assert_close(*values, rtol=1e-2, atol=5e-3)
                snapshots = [s.snapshot() for s in sessions]
                for kind in ("gdn", "kv"):
                    for left, right in zip(snapshots[0][kind], snapshots[1][kind]):
                        if left is None:
                            assert right is None
                            continue
                        for a, z in zip(left, right):
                            torch.testing.assert_close(a, z, rtol=1e-2, atol=5e-3)
            outputs = [s.generate([1, 4, 3, 2, 1], max_new_tokens=12) for s in sessions]
            assert outputs[0] == outputs[1]
        finally:
            for s in sessions:
                s.close()
    finally:
        for m in models:
            m.close()


@pytest.mark.parametrize("fmt", ["q8_g32_fp16", "q5_g64_fp16"])
def test_explicit_a8_model_workspace_and_batch(
    tmp_path, native_engine, fmt, monkeypatch
):
    from test_qwen_model import hybrid_fixture
    from test_row_codec_execution import _codec_artifact_from_q8, _codec_enabled
    from venturi.models import NativeQwenModel
    from venturi.models.qwen.qwen_weights import QwenWeights

    b, dtype, dev = native_engine
    if dtype != torch.float16:
        pytest.skip("A8 requires FP16 storage")
    if fmt.startswith("q5") and b.architecture != 70:
        pytest.skip("Q5 A8 kernel is SM70-only")
    base, _ = hybrid_fixture(tmp_path)
    weights = (
        base
        if fmt.startswith("q8")
        else QwenWeights.open(
            _codec_artifact_from_q8(base.source, tmp_path / "a8.safetensors", fmt)
        )
    )
    from venturi.quantization import codecs

    # Local qualification campaign only; the production A8 gate stays closed.
    monkeypatch.setitem(
        codecs.ACTIVATION_INT8_QUALIFIED,
        tuple(torch.cuda.get_device_capability()),
        {fmt},
    )
    with _codec_enabled(fmt):
        model = NativeQwenModel.from_q8(
            weights,
            native_runtime(b),
            weight_mode="resident",
            dtype=dtype,
            device=dev,
            max_tokens=16,
            reserve_bytes=0,
            projection_profile="int8",
        )
    try:
        info = model.projection_info
        assert info["activation_quantization_passes"] == 1
        maxk = max(
            s.shape[1] for s in weights.plan.bindings if s.format.startswith("q")
        )
        assert (
            info["primary_projection_workspace_bytes"] == 16 * ((maxk + 31) // 32) * 36
        )
        prompts = [[1, 2, 3], [4, 5], [6], [7, 8, 9, 10]]
        with closing(model.create_session(32)) as session:
            expected = []
            for prompt in prompts:
                session.reset()
                expected.append(session.generate(prompt, max_new_tokens=6))
        with closing(model.create_batch([32] * 4)) as batch:
            actual = batch.generate(
                prompts, max_new_tokens=[6] * 4, slots=[0, 1, 2, 3], eos_token_ids=[]
            )
        assert actual == expected
    finally:
        model.close()


@pytest.mark.parametrize("t", [1, 4, 17])
def test_q5_a8_matches_independent_integer_projection(native_engine, t):
    from q8_dp4a_reference import quantize_reference

    b, dtype, dev = native_engine
    if dtype != torch.float16 or b.architecture != 70:
        pytest.skip("Explicit SM70 W5A8 kernel")
    n, k = 19, 512
    c, h, s = _random_planes(dev, n, k, "q5_g64_fp16", 194)
    x = (torch.randn(t, k, device=dev) * 0.2).half()
    y = torch.empty(t, n, device=dev, dtype=dtype)
    needed = t * ((k + 31) // 32) * 36
    workspace = torch.empty(needed, device=dev, dtype=torch.uint8)
    quant = b._native.venturi_q8_dp4a_quantize
    quant.argtypes = [C.c_void_p, C.c_void_p, C.c_uint64] + [C.c_int] * 3 + [C.c_void_p]
    quant.restype = C.c_int
    stream = torch.cuda.current_stream(dev).cuda_stream
    assert quant(x.data_ptr(), workspace.data_ptr(), needed, t, k, 1, stream) == 0
    fn = b._native.venturi_q5_dp4a_fused_run
    fn.argtypes = (
        [C.c_void_p, C.c_uint64] + [C.c_void_p] * 4 + [C.c_int] * 5 + [C.c_void_p]
    )
    fn.restype = C.c_int
    assert (
        fn(
            workspace.data_ptr(),
            needed,
            c.data_ptr(),
            h.data_ptr(),
            s.data_ptr(),
            y.data_ptr(),
            n,
            n,
            t,
            k,
            k,
            stream,
        )
        == 0
    )
    _, _, activation = quantize_reference(x.cpu(), 1)
    q = dequantize_planes(
        c.cpu(), torch.ones_like(s.cpu()), h.cpu(), (n, k), "q5_g64_fp16"
    ).double()
    weights = q * s.cpu().double().repeat_interleave(64, -1)
    torch.testing.assert_close(
        y.cpu(), (activation @ weights.T).half(), rtol=2e-3, atol=2e-3
    )


@pytest.mark.parametrize("fmt", ["q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16"])
@pytest.mark.parametrize(
    "k", [1, 7, 8, 9, 63, 64, 65, 127, 128, 129, 255, 511, 512, 513]
)
@pytest.mark.parametrize("offset", [0, 1])
def test_row_decode_octets_exact_tails_offsets_and_extreme_scales(
    native_engine, fmt, k, offset
):
    b, dtype, dev = native_engine
    if dtype != torch.float16:
        pytest.skip("Canonical decode output is FP16")
    n, first, rows = 9, 2, 5
    c, h, s = _random_planes(dev, n, k, fmt, 617)
    s[:, 0] = 2**-24
    s[1, 0] = 0
    s[3, 0] = 65504
    if offset:
        c = (
            torch.empty(c.numel() + 1, dtype=c.dtype, device=dev)[1:]
            .copy_(c.flatten())
            .view_as(c)
        )
        if h is not None:
            h = (
                torch.empty(h.numel() + 1, dtype=h.dtype, device=dev)[1:]
                .copy_(h.flatten())
                .view_as(h)
            )
    expected = dequantize_planes(
        c.cpu(), s.cpu(), h.cpu() if h is not None else None, (n, k), fmt
    )
    storage = torch.full((rows * k + 2,), float("nan"), dtype=torch.float16, device=dev)
    out = storage[offset : offset + rows * k].view(rows, k)
    fn = b._native.venturi_row_dequant_rows
    fn.argtypes = [C.c_void_p] * 4 + [C.c_int] * 5 + [C.c_void_p]
    fn.restype = C.c_int
    args = [
        c.data_ptr(),
        h.data_ptr() if h is not None else None,
        s.data_ptr(),
        out.data_ptr(),
    ]
    assert (
        fn(
            *args,
            first,
            rows,
            k,
            (k + 127) // 128 * 128,
            int(fmt[1]),
            torch.cuda.current_stream(dev).cuda_stream,
        )
        == 0
    )
    assert torch.equal(
        out.cpu().view(torch.uint8),
        expected[first : first + rows].contiguous().view(torch.uint8),
    )
    assert torch.isnan(storage[offset + rows * k :]).all()
    ids = torch.tensor([8, 0, 3], dtype=torch.int64, device=dev)
    indexed = torch.empty((3, k), dtype=torch.float16, device=dev)
    fn = b._native.venturi_row_dequant_indexed
    fn.argtypes = [C.c_void_p] * 5 + [C.c_int] * 4 + [C.c_void_p]
    fn.restype = C.c_int
    assert (
        fn(
            c.data_ptr(),
            h.data_ptr() if h is not None else None,
            s.data_ptr(),
            ids.data_ptr(),
            indexed.data_ptr(),
            3,
            k,
            (k + 127) // 128 * 128,
            int(fmt[1]),
            torch.cuda.current_stream(dev).cuda_stream,
        )
        == 0
    )
    assert torch.equal(
        indexed.cpu().view(torch.uint8),
        expected[[8, 0, 3]].contiguous().view(torch.uint8),
    )


@pytest.mark.parametrize("bits", [5, 6])
def test_row_decode_requires_high_plane_before_launch(native_engine, bits):
    b, _, _ = native_engine
    fn = b._native.venturi_row_dequant_rows
    fn.argtypes = [C.c_void_p] * 4 + [C.c_int] * 5 + [C.c_void_p]
    fn.restype = C.c_int
    assert fn(1, None, 1, 1, 0, 1, 64, 128, bits, None) < 0
