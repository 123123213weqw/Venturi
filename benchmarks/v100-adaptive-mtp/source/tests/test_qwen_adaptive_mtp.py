"""Fresh strategy switching must preserve native state and failure boundaries."""

import os

import pytest
import torch

from venturi.adaptive_mtp import AdaptiveMTPPolicy
from venturi.models import AdaptiveQwenMTPBatch
from venturi.ops import CudaBackend
from test_qwen_mtp_batch import models
from test_qwen_tensor_parallel import same_state
from test_native_head_gdn import eligible_mtp, load
from venturi.models import NativeQwenMTPModel


@pytest.fixture
def engine():
    path = os.environ.get("VENTURI_ADAPTIVE_MTP_LIBRARY")
    if not path:
        pytest.skip("Explicit native adaptive MTP library required")
    return CudaBackend(path)


@pytest.mark.parametrize("count", [1, 2, 4, 8])
@pytest.mark.parametrize("profile", ["fp16", "q8-residual-column-mlp-head-gdn"])
def test_switching_ragged_eos_reset_state(tmp_path, engine, count, profile):
    if profile.endswith("head-gdn"):
        weights, draft, _, _ = eligible_mtp(tmp_path)
        target = load(weights, (engine, torch.float16, "cuda:0"), rows=32)
        model = NativeQwenMTPModel.from_q8(
            target, draft, reserve_bytes=0, draft_projection_profile="q8-residual"
        )
    else:
        target, model = models(tmp_path, engine, True, profile=profile)
    policy = AdaptiveMTPPolicy(probe_interval=0)
    adaptive = AdaptiveQwenMTPBatch(model, [48] * 8, policy=policy)
    refs = [target.create_session(48) for _ in range(count)]
    slots = list(reversed(range(8)))[:count]
    prompts = [[i + 1, 2, 3] for i in range(count)]
    budgets = [7 + i % 3 for i in range(count)]
    try:
        adaptive.prepare()
        full = [
            r.generate(p, max_new_tokens=n) for r, p, n in zip(refs, prompts, budgets)
        ]
        for depth in [0, 1, 2, 0, 2, 1] if count == 1 else [0, 1, 0, 1]:
            for stop in [False, True]:
                for r in refs:
                    r.reset()
                eos = [
                    [full[i][i % len(full[i])]] if stop else [] for i in range(count)
                ]
                expected = [
                    r.generate(p, max_new_tokens=n, eos_token_ids=e)
                    for r, p, n, e in zip(refs, prompts, budgets, eos)
                ]
                assert (
                    adaptive.generate(
                        prompts,
                        max_new_tokens=budgets,
                        eos_token_ids=eos,
                        slots=slots,
                        strategy=depth,
                    )
                    == expected
                )
                info = adaptive.info()
                assert info["selected_depth"] == depth
                assert info["emitted_tokens"] == sum(map(len, expected))
                for slot, ref in zip(slots, refs):
                    same_state(adaptive.snapshot(slot), ref.snapshot())
        if count > 1:
            with pytest.raises(ValueError, match="unsupported"):
                adaptive.generate(prompts, max_new_tokens=budgets, strategy=2)
        with pytest.raises(ValueError, match="capacity"):
            adaptive.generate(prompts, max_new_tokens=48)
        adaptive.reset()
        with pytest.raises(RuntimeError, match="No completed"):
            adaptive.info()
    finally:
        adaptive.close()
        for ref in refs:
            ref.close()
        model.close()
        target.close()


def test_failure_is_not_fallback_or_training_sample(tmp_path, engine, monkeypatch):
    target, model = models(tmp_path, engine, True)
    adaptive = model.create_adaptive_batch([32] * 4)
    try:
        adaptive.prepare()
        bank = adaptive._banks[1]
        original = bank.generate

        def fail(*args, **kwargs):
            bank._inject_failure(slot=3, layer=3, position=0)
            return original(*args, **kwargs)

        monkeypatch.setattr(bank, "generate", fail)
        with pytest.raises(RuntimeError, match="injected"):
            adaptive.generate([[1, 2]] * 4, max_new_tokens=7, strategy=1)
        assert not adaptive.policy.info()["buckets"][0]["samples"]
        with pytest.raises(RuntimeError, match="reset required"):
            adaptive.generate([[1]], max_new_tokens=7, strategy=0)
        monkeypatch.setattr(bank, "generate", original)
        adaptive.reset()
        output = adaptive.generate([[1]], max_new_tokens=7, strategy=0)
        assert len(output[0]) == 7
        adaptive.close()
        with pytest.raises(RuntimeError, match="closed"):
            adaptive.generate([[1]], max_new_tokens=7)
    finally:
        adaptive.close()
        model.close()
        target.close()


def test_adaptive_generation_has_no_python_tensor_math(tmp_path, engine, monkeypatch):
    target, model = models(tmp_path, engine, True)
    adaptive = model.create_adaptive_batch([32] * 2)
    ref = target.create_session(32)
    try:
        adaptive.prepare()
        expected = ref.generate([1, 2], max_new_tokens=7)

        def forbidden(*args, **kw):
            raise AssertionError("Adaptive generation used Python tensor math")

        with monkeypatch.context() as patch:
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
                patch.setattr(torch, name, forbidden)
            for depth in [0, 1, 2, 0]:
                assert adaptive.generate(
                    [[1, 2]], max_new_tokens=7, strategy=depth
                ) == [expected]
    finally:
        adaptive.close()
        ref.close()
        model.close()
        target.close()


def test_adaptive_benchmark_http_with_native_fixture(tmp_path, engine):
    """Exercise the actual opt-in HTTP entry, not merely a mocked batcher."""
    from concurrent.futures import ThreadPoolExecutor
    import json
    from pathlib import Path
    import socket
    import subprocess
    import sys
    import time
    import urllib.request

    weights, draft, _, _ = eligible_mtp(tmp_path / "source")
    ready = tmp_path / "ready.json"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    tool = Path(__file__).resolve().parents[1] / "tools/serve_native_benchmark.py"
    command = [
        sys.executable,
        str(tool),
        str(weights.artifact.path),
        "--library",
        os.environ["VENTURI_ADAPTIVE_MTP_LIBRARY"],
        "--mtp-artifact",
        str(draft.artifact.path),
        "--mode",
        "tp",
        "--port",
        str(port),
        "--slots",
        "8",
        "--capacity",
        "128",
        "--projection-rows",
        "32",
        "--projection-profile",
        "q8-residual-column-mlp-head-gdn",
        "--mtp-draft-projection-profile",
        "q8-residual",
        "--adaptive-mtp",
        "--batch-window-ms",
        "50",
        "--ready-file",
        str(ready),
    ]
    with (tmp_path / "server.log").open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while (
                not ready.exists()
                and proc.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.1)
            assert ready.exists(), (tmp_path / "server.log").read_text()
            assert json.loads(ready.read_text())["adaptive_mtp"]

            def request(slot):
                body = dict(
                    prompt=[1, 2, 3],
                    n_predict=7,
                    id_slot=slot,
                    ignore_eos=True,
                    stream=False,
                    cache_prompt=False,
                    temperature=0,
                )
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/completion",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=60) as response:
                    return json.load(response)

            singles = [request(0) for _ in range(3)]
            assert [r["adaptive_mtp"]["selected_depth"] for r in singles] == [0, 1, 2]
            assert singles[0]["tokens"] == singles[1]["tokens"] == singles[2]["tokens"]
            with ThreadPoolExecutor(max_workers=8) as pool:
                for depth in [0, 1]:
                    rows = list(pool.map(request, range(8)))
                    assert len({r["batch_cohort_id"] for r in rows}) == 1
                    assert all(
                        r["adaptive_mtp"]["selected_depth"] == depth for r in rows
                    )
                    assert all(r["tokens"] == singles[0]["tokens"] for r in rows)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
