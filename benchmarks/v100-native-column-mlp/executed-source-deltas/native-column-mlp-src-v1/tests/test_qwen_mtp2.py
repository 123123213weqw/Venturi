"""Two-proposal native transactions: target and corrected draft cache equality."""

from concurrent.futures import ThreadPoolExecutor
import os
from types import SimpleNamespace

import pytest
import torch

from test_qwen_mtp import mtp_fixture, load_models, mtp_oracle
from test_qwen_tensor_parallel import same_state
from venturi.models import NativeQwenMTPSession
from venturi.ops import CudaBackend


@pytest.fixture
def engine():
    path = os.environ.get("VENTURI_MTP2_LIBRARY")
    if not path:
        pytest.skip("explicit native MTP2 library required")
    return CudaBackend(path), torch.float16, "cuda:0"


def same_draft(a, b):
    assert a["position"] == b["position"]
    for k in ["keys", "values"]:
        assert torch.equal(a[k].view(torch.uint8), b[k].view(torch.uint8)), k


@pytest.mark.parametrize("mode", ["single", "tp", "pp"])
@pytest.mark.parametrize("profile", ["fp16", "q8-residual"])
@pytest.mark.parametrize(
    "forced", ["native", "accept", "reject-first", "reject-second"]
)
def test_depth2_output_target_and_corrected_draft_caches(
    tmp_path, engine, mode, profile, forced
):
    if profile != "fp16" and mode == "pp":
        pytest.skip("residual PP is not implemented")
    a, b, _, _ = mtp_fixture(tmp_path)
    target, model = load_models(a, b, engine, mode, profile=profile)
    ref, old, s = (
        target.create_session(40),
        model.create_session(40),
        model.create_session(40),
    )
    prompt = [1, 2, 3, 4, 5]
    try:
        for maximum, stop in [
            (1, None),
            (2, None),
            (3, None),
            (4, None),
            (11, None),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (11, 5),
        ]:
            for request in [ref, old, s]:
                request.reset()
            full = ref.generate(prompt, max_new_tokens=maximum)
            eos = [] if stop is None or stop >= len(full) else [full[stop]]
            ref.reset()
            expected = ref.generate(prompt, max_new_tokens=maximum, eos_token_ids=eos)
            baseline = old.generate(prompt, max_new_tokens=maximum, eos_token_ids=eos)
            assert baseline == expected
            wrong = [(v + 1) % a.config["vocab_size"] for v in full]
            if forced == "accept":
                s._force_proposals(full)
            elif forced == "reject-first":
                s._force_proposals(wrong)
            elif forced == "reject-second":
                s._force_proposals(
                    [v if i % 2 else wrong[i] for i, v in enumerate(full)]
                )
            actual = s.generate(
                prompt, max_new_tokens=maximum, eos_token_ids=eos, draft_tokens=2
            )
            assert actual == expected
            same_state(s.snapshot(), ref.snapshot())
            same_draft(s.draft_snapshot(), old.draft_snapshot())
            stats = s.stats()
            assert stats["draft_tokens_per_round"] == 2
            assert (
                stats["draft_position"] == s.position == len(prompt) + len(expected) - 1
            )
            assert sum(stats["proposed_by_depth"]) == stats["proposed_tokens"]
            assert sum(stats["accepted_by_depth"]) == stats["accepted_tokens"]
            assert stats["unused_suffix_proposals"] == stats["proposed_tokens"] - sum(
                stats["verified_by_depth"]
            )
            assert (
                stats["draft_rewind_calls"]
                == stats["verification_calls_by_input_width"]["3"]
            )
            if forced == "reject-first":
                assert (
                    stats["accepted_tokens"] == 0 and stats["verified_by_depth"][1] == 0
                )
            if maximum == 11 and stop is None and forced == "reject-second":
                assert (
                    stats["accepted_by_depth"][0] > 0
                    and stats["verified_by_depth"][1] > 0
                    and stats["accepted_by_depth"][1] == 0
                )
            if maximum == 4 and stop is None and forced == "accept":
                assert (
                    stats["proposed_by_depth"] == stats["accepted_by_depth"] == [1, 1]
                )
                assert stats["verified_draft_catchup_inputs"] == 2
            with pytest.raises(RuntimeError, match="fresh/reset"):
                s.generate([actual[-1]], max_new_tokens=1, draft_tokens=2)
    finally:
        for request in [ref, old, s]:
            request.close()
        model.close()
        target.close()


@pytest.mark.parametrize("mode", ["single", "tp", "pp"])
def test_depth2_failure_reset_lifetime_and_native_only(
    tmp_path, engine, mode, monkeypatch
):
    a, b, _, _ = mtp_fixture(tmp_path, tied=True)
    target, model = load_models(a, b, engine, mode)
    ref, s = target.create_session(40), model.create_session(40)
    prompt = [1, 2, 3, 4, 5]
    expected = ref.generate(prompt, max_new_tokens=9)
    try:
        s._inject_failure(layer=3, position=5)
        with pytest.raises(RuntimeError, match="injected"):
            s.generate(prompt, max_new_tokens=9, draft_tokens=2)
        assert s.failed
        s.reset()
        assert s.stats()["draft_tokens_per_round"] == 1
        target.close()
        model.close()

        def forbidden(*a, **kw):
            raise AssertionError("MTP2 used Python/Torch numerical fallback")

        with monkeypatch.context() as p:
            for name in [
                "empty",
                "zeros",
                "matmul",
                "mm",
                "softmax",
                "sigmoid",
                "cat",
                "tensor",
            ]:
                p.setattr(torch, name, forbidden)
            actual = s.generate(prompt, max_new_tokens=9, draft_tokens=2)
        assert actual == expected
        assert not s.failed
        same_state(s.snapshot(), ref.snapshot())
    finally:
        ref.close()
        s.close()
        model.close()
        target.close()


@pytest.mark.parametrize("mode", ["tp", "pp"])
@pytest.mark.parametrize("count", [2, 8])
def test_depth2_concurrent_private_requests(tmp_path, engine, mode, count):
    a, b, _, _ = mtp_fixture(tmp_path)
    target, model = load_models(a, b, engine, mode)
    sessions = [model.create_session(32) for _ in range(count)]
    refs = [target.create_session(32) for _ in range(count)]
    prompts = [[i + 1, 2, 3] for i in range(count)]
    try:
        expected = [s.generate(p, max_new_tokens=8) for s, p in zip(refs, prompts)]
        with ThreadPoolExecutor(max_workers=count) as pool:
            jobs = [
                pool.submit(s.generate, p, max_new_tokens=8, draft_tokens=2)
                for s, p in zip(sessions, prompts)
            ]
            assert [j.result(timeout=120) for j in jobs] == expected
        for s, ref in zip(sessions, refs):
            same_state(s.snapshot(), ref.snapshot())
    finally:
        for s in sessions + refs:
            s.close()
        model.close()
        target.close()


def test_repeated_draft_normalized_hidden_cpu_oracle(tmp_path, engine):
    a, b, raw, _ = mtp_fixture(tmp_path)
    target, model = load_models(a, b, engine)
    s = model.create_session(16)
    cache = None
    hidden = (
        torch.randn(
            1, a.config["hidden_size"], generator=torch.Generator().manual_seed(718)
        ).half()
        * 0.31
    )
    token = 3
    try:
        for _ in range(3):
            actual = s.draft_forward([token], hidden)
            expected, logits, cache = mtp_oracle(raw, a.config, [token], hidden, cache)
            torch.testing.assert_close(
                actual["hidden"].double(), expected, atol=0.008, rtol=0.02
            )
            torch.testing.assert_close(
                actual["logits"].double(), logits, atol=0.008, rtol=0.02
            )
            snap = s.draft_snapshot()
            torch.testing.assert_close(
                snap["keys"].double(), cache[0], atol=0.008, rtol=0.02
            )
            torch.testing.assert_close(
                snap["values"].double(), cache[1], atol=0.008, rtol=0.02
            )
            # The next-step input is the previous normalized draft output.
            hidden = actual["hidden"]
            token = int(actual["logits"].argmax())
    finally:
        s.close()
        model.close()
        target.close()


def test_depth2_explicit_preflight_and_old_library_missing_export():
    s = object.__new__(NativeQwenMTPSession)
    for n in [None, True, 0, 3, "2"]:
        with pytest.raises(ValueError, match="explicit 1 or 2"):
            s.generate([1], max_new_tokens=1, draft_tokens=n)
    for recovery in [None, "invalid", "prefix"]:
        with pytest.raises(ValueError, match="requires explicit MTP2"):
            s.generate([1], max_new_tokens=1, recovery=recovery)
    s.model = SimpleNamespace(_max_tokens=2)
    with pytest.raises(ValueError, match="three projection"):
        s.generate([1], max_new_tokens=1, draft_tokens=2)
    s.model = SimpleNamespace(_max_tokens=3, _native=SimpleNamespace())
    with pytest.raises(RuntimeError, match="export absent"):
        s.generate([1], max_new_tokens=1, draft_tokens=2)
    with pytest.raises(RuntimeError, match="export absent"):
        s.generate([1], max_new_tokens=1, draft_tokens=2, recovery="prefix")
    with pytest.raises(RuntimeError, match="prefix export absent"):
        s.prepare_prefix()


@pytest.mark.parametrize(
    "recovery,fail_prepare", [("replay", False), ("prefix", False), ("prefix", True)]
)
def test_cli_depth2_route_and_required_artifact(
    tmp_path, monkeypatch, capsys, recovery, fail_prepare
):
    import json
    from venturi import qwen_native_cli as module
    from test_qwen_model import hybrid_fixture

    weights, _ = hybrid_fixture(tmp_path)

    def forbidden(*a, **kw):
        raise AssertionError("Invalid CLI touched CUDA")

    monkeypatch.setattr(module, "CudaBackend", forbidden)
    base = [
        "generate",
        str(weights.artifact.path),
        "--token-ids",
        "1,2,3",
        "--blocks-library",
        "explicit.so",
        "--weight-mode",
        "resident",
        "--dtype",
        "fp16",
        "--mtp-draft-tokens",
        "2",
        "--mtp-recovery",
        recovery,
    ]
    with pytest.raises(SystemExit):
        module.main(base)
    assert (
        "--mtp-artifact" if recovery == "replay" else "Prefix recovery"
    ) in capsys.readouterr().err
    calls, events = [], []

    class Session:
        def prepare_prefix(self):
            calls.append("prepare")
            if fail_prepare:
                raise RuntimeError("injected prefix allocation failure")

        def generate(self, ids, **kw):
            assert calls == (["prepare"] if recovery == "prefix" else [])
            calls.append(kw)
            return [3, 2]

        def info(self):
            return dict(position=4, failed=0)

        def close(self):
            events.append("session")

    class Model:
        @classmethod
        def from_q8(cls, *a, **kw):
            return cls()

        def create_session(self, capacity):
            return Session()

        def close(self):
            events.append("model")

    monkeypatch.setattr(module, "NativeQwenModel", Model)
    monkeypatch.setattr(
        module,
        "NativeQwenMTPModel",
        SimpleNamespace(from_q8=lambda target, weights: target),
    )
    monkeypatch.setattr(
        module, "QwenMTPWeights", SimpleNamespace(from_ninfer=lambda a: object())
    )
    monkeypatch.setattr(module, "CudaBackend", lambda a: object())
    argv = base + [
        "--mtp-artifact",
        str(weights.artifact.path),
        "--max-tokens",
        "3",
        "--max-new-tokens",
        "2",
    ]
    if fail_prepare:
        with pytest.raises(SystemExit):
            module.main(argv)
        assert "injected prefix allocation failure" in capsys.readouterr().err
        assert calls == ["prepare"] and events[-2:] == ["session", "model"]
        return
    assert module.main(argv) == 0
    assert events[-2:] == ["session", "model"]
    assert calls[-1]["draft_tokens"] == 2
    assert calls[-1]["recovery"] == recovery
    report = json.loads(capsys.readouterr().out)
    assert (
        report["speculative_draft_tokens"] == 2 and report["mtp_recovery"] == recovery
    )


def test_server_rejects_unpaired_depth_before_runtime(monkeypatch):
    import sys
    from test_completion_benchmark import load

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "serve",
            "not-opened.ninfer",
            "--library",
            "absent.so",
            "--mode",
            "tp",
            "--port",
            "18953",
            "--ready-file",
            "unused",
            "--mtp-depth",
            "2",
        ],
    )
    with pytest.raises(SystemExit) as e:
        load("serve_native_benchmark").main()
    assert e.value.code == 2


@pytest.mark.parametrize("artifact", [False, True])
def test_server_prefix_rejects_missing_depth_or_artifact(artifact, monkeypatch):
    import sys
    from test_completion_benchmark import load

    args = [
        "serve",
        "not-opened.ninfer",
        "--library",
        "absent.so",
        "--mode",
        "tp",
        "--port",
        "18953",
        "--ready-file",
        "unused",
        "--mtp-recovery",
        "prefix",
    ]
    if artifact:
        args += ["--mtp-artifact", "not-opened-draft.ninfer"]
    else:
        args += ["--mtp-depth", "2"]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as e:
        load("serve_native_benchmark").main()
    assert e.value.code == 2
