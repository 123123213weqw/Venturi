"""Prefix-state snapshots eliminate replay, not target/draft state correctness."""

import ctypes as C
import os
import pytest
import torch

import test_qwen_mtp2 as old
from test_gdn import gdn_inputs, gdn_oracle
from venturi.models import NativeQwenMTPSession
from venturi.ops import CudaBackend


@pytest.fixture
def engine():
    p = os.environ.get("VENTURI_MTP_PREFIX_LIBRARY")
    if not p:
        pytest.skip("explicit prefix-trace native library required")
    return CudaBackend(p), torch.float16, "cuda:0"


@pytest.fixture
def prefix(monkeypatch):
    original = NativeQwenMTPSession.generate

    def generate(self, *args, **kw):
        two = kw.get("draft_tokens", 1) == 2
        if two:
            kw["recovery"] = "prefix"
        result = original(self, *args, **kw)
        if two:
            stats = self.stats()
            assert (
                stats["replayed_input_tokens"] == 0
                and stats["target_recovery"] == "prefix-trace"
            )
            assert (
                stats["prefix_trace_extra_primary_bytes"]
                == stats["primary_target_backup_bytes"]
            )
            assert (
                stats["prefix_trace_extra_secondary_bytes"]
                == stats["secondary_target_backup_bytes"]
            )
        return result

    monkeypatch.setattr(NativeQwenMTPSession, "generate", generate)


@pytest.mark.parametrize("mode", ["single", "tp", "pp"])
@pytest.mark.parametrize("profile", ["fp16", "q8-residual"])
@pytest.mark.parametrize(
    "forced", ["native", "accept", "reject-first", "reject-second"]
)
def test_prefix_transaction_output_target_and_draft_cache(
    tmp_path, engine, prefix, mode, profile, forced
):
    old.test_depth2_output_target_and_corrected_draft_caches(
        tmp_path, engine, mode, profile, forced
    )


@pytest.mark.parametrize("mode", ["single", "tp", "pp"])
def test_prefix_failure_reset_lifetime(tmp_path, engine, prefix, mode, monkeypatch):
    old.test_depth2_failure_reset_lifetime_and_native_only(
        tmp_path, engine, mode, monkeypatch
    )


@pytest.mark.parametrize("mode", ["tp", "pp"])
@pytest.mark.parametrize("count", [2, 8])
def test_prefix_concurrent_private_requests(tmp_path, engine, prefix, mode, count):
    old.test_depth2_concurrent_private_requests(tmp_path, engine, mode, count)


@pytest.mark.parametrize("tokens", [2, 3])
@pytest.mark.parametrize("dk", [64, 128, 256])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_gdn_intermediate_state_bytes_and_independent_oracle(
    engine, tokens, dk, packed, normalize
):
    backend, _, _ = engine
    values = gdn_inputs(tokens=tokens, dk=dk, dtype=torch.float16, device="cuda:0")
    q, k, v, g, beta, state = values
    if not normalize:
        k.mul_(0.1)
    qkv = torch.cat([q.flatten(1), k.flatten(1), v.flatten(1)], 1)
    out, final = torch.empty_like(v), torch.empty_like(state)
    guards = [
        torch.full((state.numel() + 2,), 47, dtype=torch.float32, device="cuda:0")
        for _ in range(2)
    ]
    trace = [p[1:-1].view_as(state) for p in guards]
    fn = backend._native.venturi_gdn_prefix_run
    fn.argtypes = backend._native.venturi_gdn_run.argtypes[:-1] + [
        C.c_void_p,
        C.c_void_p,
        C.c_int,
        C.c_void_p,
    ]
    fn.restype = C.c_int
    ptrs = [q.data_ptr(), k.data_ptr(), v.data_ptr()]
    if packed:
        ptrs = [qkv.data_ptr() + j * 2 * q.shape[1] * dk for j in range(3)]
    backend._check(
        fn(
            *ptrs,
            *[x.data_ptr() for x in [g, beta, state, out, final]],
            tokens,
            2,
            4,
            dk,
            13,
            int(normalize),
            int(packed),
            0,
            dk**-0.5,
            trace[0].data_ptr(),
            trace[1].data_ptr(),
            tokens - 1,
            torch.cuda.current_stream().cuda_stream,
        )
    )
    assert all(p[0] == p[-1] == 47 for p in guards)
    for t in range(1, tokens + 1):
        p = backend.prepare_gdn(
            tokens=t,
            qk_heads=2,
            value_heads=4,
            key_dim=dk,
            value_dim=13,
            normalize_qk=normalize,
            dtype=torch.float16,
        )
        expected = torch.empty_like(v[:t])
        sf = torch.empty_like(state)
        p.run(
            *(x[:t] for x in [q, k, v, g, beta]),
            state=state,
            out=expected,
            state_out=sf,
        )
        actual = final if t == tokens else trace[t - 1]
        assert torch.equal(actual.view(torch.uint8), sf.view(torch.uint8))
        if t == tokens:
            assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
        _, ref = gdn_oracle(
            *(x[:t] for x in [q, k, v, g, beta]), state, normalize=normalize
        )
        torch.testing.assert_close(actual.cpu().double(), ref, atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize("kernel", [1, 2, 4])
@pytest.mark.parametrize("prefixes", [1, 2])
def test_history_prefix_bits_and_guards(engine, kernel, prefixes):
    backend, _, _ = engine
    channels = 17
    x = torch.arange(3 * channels, device="cuda:0", dtype=torch.float16).view(
        3, channels
    )
    h = -torch.arange(
        (kernel - 1) * channels, device="cuda:0", dtype=torch.float16
    ).view(kernel - 1, channels)
    guards = [
        torch.full((h.numel() + 2,), 47, device="cuda:0", dtype=torch.float16)
        for _ in range(2)
    ]
    outs = [p[1:-1].view_as(h) for p in guards]
    fn = backend._native.venturi_history_prefix_run
    fn.argtypes = [C.c_void_p] * 4 + [C.c_int] * 4 + [C.c_void_p]
    fn.restype = C.c_int
    backend._check(
        fn(
            x.data_ptr(),
            h.data_ptr(),
            outs[0].data_ptr(),
            outs[1].data_ptr(),
            channels,
            kernel,
            prefixes,
            0,
            torch.cuda.current_stream().cuda_stream,
        )
    )
    for i in range(prefixes):
        expected = torch.cat([h, x[: i + 1]])[-(kernel - 1) :] if kernel > 1 else h
        assert torch.equal(outs[i].view(torch.uint8), expected.view(torch.uint8))
    assert all(p[0] == p[-1] == 47 for p in guards)


def test_prefix_prepare_idempotent_and_reset_memory(engine, tmp_path):
    a, b, _, _ = old.mtp_fixture(tmp_path)
    target, model = old.load_models(a, b, engine, "tp")
    s = model.create_session(24)
    try:
        before = s.stats()
        s.prepare_prefix()
        first = s.stats()
        s.prepare_prefix()
        assert s.stats() == first
        assert (
            first["prefix_trace_extra_primary_bytes"]
            == first["primary_target_backup_bytes"]
            > 0
        )
        assert first["mtp_request_device_bytes"] == before["mtp_request_device_bytes"]
        s.generate([1, 2, 3], max_new_tokens=5, draft_tokens=2, recovery="prefix")
        s.reset()
        assert (
            s.stats()["prefix_trace_extra_primary_bytes"]
            == first["prefix_trace_extra_primary_bytes"]
        )
        assert s.stats()["draft_position"] == 0
    finally:
        s.close()
        model.close()
        target.close()
