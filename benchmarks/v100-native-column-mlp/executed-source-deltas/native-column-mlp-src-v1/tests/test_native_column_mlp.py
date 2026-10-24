"""New FP32 association requires model/state gates, not old-profile byte claims."""

import ctypes as C
import os
from types import SimpleNamespace

import pytest
import torch

from q8_dp4a_reference import projection_reference
from test_qwen_model import hybrid_fixture
from test_qwen_mtp import mtp_fixture, load_models
from test_qwen_mtp2 import (
    test_depth2_output_target_and_corrected_draft_caches as depth2_checks,
)
from test_qwen_tensor_parallel import same_state
from test_mtp_prefix import prefix as prefix_fixture
from venturi.models import NativeQwenModel
from venturi.ops import CudaBackend
from venturi.ops.q8_column_tp import slice_q8_columns

PROFILE = "q8-residual-column-mlp"


@pytest.fixture
def engine():
    path = os.environ.get("VENTURI_NATIVE_COLUMN_MLP_LIBRARY")
    if not path:
        pytest.skip("explicit SM70 native column-MLP library required")
    return CudaBackend(path), torch.float16, "cuda:0"


def load(weights, engine, rows=8):
    b, dtype, device = engine
    return NativeQwenModel.from_q8(
        weights,
        b,
        weight_mode="resident",
        dtype=dtype,
        device=device,
        max_tokens=rows,
        reserve_bytes=0,
        tensor_parallel_device="cuda:1",
        tensor_parallel_transport="host",
        projection_profile=PROFILE,
    )


def test_column_profile_rejects_missing_tp_and_old_export_before_device_allocation(
    monkeypatch,
):
    from venturi.models.qwen_weights import QwenQ8Weights

    weights = object.__new__(QwenQ8Weights)
    backend = object.__new__(CudaBackend)
    backend._native = SimpleNamespace()
    kwargs = dict(
        weight_mode="resident", dtype=torch.float16, projection_profile=PROFILE
    )
    with pytest.raises(ValueError, match="two-device host TP"):
        NativeQwenModel.from_q8(weights, backend, **kwargs)
    with pytest.raises(RuntimeError, match="exports absent"):
        NativeQwenModel.from_q8(
            weights,
            backend,
            tensor_parallel_device="cuda:1",
            tensor_parallel_transport="host",
            **kwargs,
        )
    with pytest.raises(ValueError, match="complete resident FP16"):
        NativeQwenModel.from_q8(
            weights,
            backend,
            tensor_parallel_device="cuda:1",
            tensor_parallel_transport="host",
            **{**kwargs, "weight_mode": "streaming"},
        )


@pytest.mark.parametrize("tokens", [1, 2, 3, 8])
def test_native_full_mlp_against_independent_fp64_and_actual_fp32_partials(
    tmp_path, engine, tokens
):
    weights, _ = hybrid_fixture(tmp_path)
    model = load(weights, engine)
    r = model.create_session(32)
    try:
        x = torch.randn(
            tokens,
            weights.config["hidden_size"],
            generator=torch.Generator().manual_seed(983),
        ).half()
        h, i = weights.config["hidden_size"], weights.config["intermediate_size"]
        out = torch.empty(tokens, h, dtype=torch.float16)
        activation = torch.empty(tokens, i, dtype=torch.float16)
        partials = torch.empty(2, tokens, h, dtype=torch.float32)
        fn = model._native.venturi_qwen_column_mlp_debug
        fn.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_int] + [C.c_void_p] * 3
        fn.restype = C.c_int
        for layer in [0, 1]:
            model._backend._check(
                fn(
                    r._handle,
                    layer,
                    x.data_ptr(),
                    tokens,
                    out.data_ptr(),
                    activation.data_ptr(),
                    partials.data_ptr(),
                )
            )
            prefix = f"text/layers/{layer}/mlp/"
            gate, up, down = [
                weights.read_linear(prefix + name) for name in ["gate", "up", "down"]
            ]
            a = projection_reference(x, gate.codes, gate.scales, 2).half().double()
            b = projection_reference(x, up.codes, up.scales, 2).half().double()
            ideal = (torch.nn.functional.silu(a) * b).half().double()
            assert (activation.double() - ideal).norm() / (ideal.norm() + 1e-12) < 0.001
            for rank in [0, 1]:
                piece = slice_q8_columns(down, rank * (i // 2), (rank + 1) * (i // 2))
                ref = projection_reference(
                    activation[:, rank * (i // 2) : (rank + 1) * (i // 2)].contiguous(),
                    piece.codes,
                    piece.scales,
                    2,
                )
                assert (partials[rank].double() - ref).norm() / (
                    ref.norm() + 1e-12
                ) < 2e-6
            assert torch.equal(out, (partials[0] + partials[1]).half())
            ref = projection_reference(activation, down.codes, down.scales, 2)
            assert (out.double() - ref).norm() / (ref.norm() + 1e-12) < 0.001
        assert r.position == 0
        info = model.projection_info
        assert (
            info["column_mlp_enabled"]
            and info["partial_and_host_transport_dtype"] == "float32"
        )
        assert info["column_mlp_buffers"]["column_down_matrices"] == 4
        assert r.info()["peer_output_copy_bytes"] > 0
    finally:
        r.close()
        model.close()


@pytest.mark.parametrize("count", [2, 8])
def test_column_model_scalar_batch_mixed_eos_and_selective_reset(
    tmp_path, engine, count
):
    weights, _ = hybrid_fixture(tmp_path)
    model = load(weights, engine)
    batch = model.create_batch([40] * count)
    refs = [model.create_session(40) for _ in range(count)]
    try:
        prompts = [[1, 2, 3], [4], [5, 6], [7, 8, 9, 10]]
        ps = [prompts[j % 4] for j in range(count)]
        budgets = [1 + (j * 3) % 9 for j in range(count)]
        expected = [
            r.generate(p, max_new_tokens=n) for r, p, n in zip(refs, ps, budgets)
        ]
        eos = [[v[-1]] if j % 2 else [] for j, v in enumerate(expected)]
        for r in refs:
            r.reset()
        expected = [
            r.generate(p, max_new_tokens=n, eos_token_ids=e)
            for r, p, n, e in zip(refs, ps, budgets, eos)
        ]
        assert batch.generate(ps, max_new_tokens=budgets, eos_token_ids=eos) == expected
        for j, r in enumerate(refs):
            same_state(batch.snapshot(j), r.snapshot())
        before = [batch.snapshot(j) for j in range(count)]
        batch.reset(slots=[0, count - 1])
        batch.forward([[3], [7, 2]], slots=[0, count - 1])
        for j in range(1, count - 1):
            same_state(before[j], batch.snapshot(j))
    finally:
        batch.close()
        for r in refs:
            r.close()
        model.close()


@pytest.mark.parametrize(
    "forced", ["native", "accept", "reject-first", "reject-second"]
)
def test_column_mtp_prefix_target_and_corrected_draft_bytes(
    tmp_path, engine, monkeypatch, forced
):
    prefix_fixture.__wrapped__(monkeypatch)
    depth2_checks(tmp_path, engine, "tp", PROFILE, forced)


def test_column_failure_reset_closed_owner_and_source_mutation(
    tmp_path, engine, monkeypatch
):
    weights, draft, _, _ = mtp_fixture(tmp_path)
    model, mtp = load_models(weights, draft, engine, "tp", profile=PROFILE)
    r = mtp.create_session(40)
    try:
        r._inject_failure(layer=2, position=0)
        with pytest.raises(RuntimeError, match="injected"):
            r.generate([1, 2, 3], max_new_tokens=9, draft_tokens=2, recovery="prefix")
        assert r.failed
        r.reset()
        expected = r.generate(
            [1, 2, 3], max_new_tokens=8, draft_tokens=2, recovery="prefix"
        )
        r.reset()
        mtp.close()
        model.close()

        def forbidden(*a, **k):
            raise AssertionError("Torch math/allocation in native column MLP")

        with monkeypatch.context() as p:
            for name in ["empty", "zeros", "mm", "matmul", "softmax", "argmax", "cat"]:
                p.setattr(torch, name, forbidden)
            assert (
                r.generate(
                    [1, 2, 3], max_new_tokens=8, draft_tokens=2, recovery="prefix"
                )
                == expected
            )
        r.reset()
        source = weights.artifact.files[0].path
        stamp = source.stat()
        os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1000000))
        with pytest.raises(RuntimeError, match="source changed"):
            r.generate([1, 2, 3], max_new_tokens=8, draft_tokens=2, recovery="prefix")
    finally:
        r.close()
        mtp.close()
        model.close()
