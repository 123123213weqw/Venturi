"""Batched MTP must preserve every target/draft byte, not only token output."""

import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest
import torch
from test_qwen_mtp import mtp_fixture
from test_qwen_mtp2 import same_draft
from test_qwen_tensor_parallel import same_state
from venturi.models import (
    NativeQwenModel,
    NativeQwenMTPModel,
    NativeQwenMTPBatchSession,
)
from venturi.ops import CudaBackend


@pytest.fixture
def engine():
    p = os.environ.get("VENTURI_MTP_BATCH_LIBRARY")
    if not p:
        pytest.skip("explicit native MTP-batch library required")
    return CudaBackend(p)


def models(
    tmp_path, backend, tp, profile="q8-residual", tied=False, draft_profile="fp16"
):
    a, b, _, _ = mtp_fixture(tmp_path, tied=tied)
    target = NativeQwenModel.from_q8(
        a,
        backend,
        weight_mode="resident",
        dtype=torch.float16,
        device="cuda:0",
        max_tokens=32,
        reserve_bytes=0,
        projection_profile=profile,
        **(
            dict(tensor_parallel_device="cuda:1", tensor_parallel_transport="host")
            if tp
            else {}
        ),
    )
    return target, NativeQwenMTPModel.from_q8(
        target, b, reserve_bytes=0, draft_projection_profile=draft_profile
    )


@pytest.mark.parametrize("tp", [False, True])
@pytest.mark.parametrize("profile", ["fp16", "q8-residual"])
@pytest.mark.parametrize("count", [1, 2, 4, 8])
@pytest.mark.parametrize("forced", ["native", "accept", "reject", "mixed"])
def test_target_and_corrected_draft_bytes(tmp_path, engine, tp, profile, count, forced):
    target, model = models(tmp_path, engine, tp, profile, tied=count == 4)
    batch = model.create_batch([40] * count)
    refs = [target.create_session(40) for _ in range(count)]
    old = [model.create_session(40) for _ in range(count)]
    prompts = [
        [(i + j + 1) % target.config["vocab_size"] for j in range(1 + i % 5)]
        for i in range(count)
    ]
    slots = list(reversed(range(count)))
    try:
        full = [r.generate(p, max_new_tokens=11) for r, p in zip(refs, prompts)]
        for stop in [False, True]:
            batch.reset()
            for r in refs + old:
                r.reset()
            eos = [
                [full[i][min(i % 4, len(full[i]) - 1)]] if stop else []
                for i in range(count)
            ]
            expected = [
                r.generate(p, max_new_tokens=11, eos_token_ids=e)
                for r, p, e in zip(refs, prompts, eos)
            ]
            for i in range(count):
                assert (
                    old[i].generate(prompts[i], max_new_tokens=11, eos_token_ids=eos[i])
                    == expected[i]
                )
                if forced != "native":
                    good = forced == "accept" or (forced == "mixed" and i % 2 == 0)
                    batch._force_proposals(
                        slots[i],
                        full[i]
                        if good
                        else [(v + 1) % target.config["vocab_size"] for v in full[i]],
                    )
            result = batch.generate(
                prompts, max_new_tokens=11, eos_token_ids=eos, slots=slots
            )
            assert result == expected
            info = batch.info()
            assert (
                info["shared_mtp_selection_device_bytes"] > 0
                and info["all_target_backup_device_bytes"] > 0
            )
            for i, slot in enumerate(slots):
                same_state(batch.snapshot(slot), refs[i].snapshot())
                same_draft(batch.draft_snapshot(slot), old[i].draft_snapshot())
                s = batch.stats(slot)
                assert (
                    s["replayed_input_tokens"] == 0
                    and s["checkpoint_save_wall_seconds"] == 0
                    and s["prefix_trace_extra_primary_bytes"] == 0
                )
                assert (
                    s["draft_position"]
                    == info["requests"][slot]["position"]
                    == len(prompts[i]) + len(result[i]) - 1
                )
                if forced == "reject":
                    assert s["accepted_tokens"] == 0
            if not stop and count >= 2:
                assert info["max_group_rows"] >= count
            with pytest.raises(RuntimeError, match="fresh/reset"):
                batch.generate(prompts, max_new_tokens=1, slots=slots)
    finally:
        batch.close()
        for r in refs + old:
            r.close()
        model.close()
        target.close()


@pytest.mark.parametrize("tp", [False, True])
@pytest.mark.parametrize("maximum", [1, 2, 3, 4, 11])
def test_ragged_budgets_eos_tails_and_selective_reset(tmp_path, engine, tp, maximum):
    target, model = models(tmp_path, engine, tp)
    b = model.create_batch([40] * 8)
    refs = [model.create_session(40) for _ in range(8)]
    prompts = [[i + 1, 2, 3] for i in range(8)]
    budgets = [min(maximum, i % 5 + 1) for i in range(8)]
    try:
        expected = [
            r.generate(p, max_new_tokens=n) for r, p, n in zip(refs, prompts, budgets)
        ]
        assert b.generate(prompts, max_new_tokens=budgets) == expected
        before = [b.snapshot(i) for i in range(8)]
        draft = [b.draft_snapshot(i) for i in range(8)]
        for i in range(8):
            same_draft(draft[i], refs[i].draft_snapshot())
        b.reset(slots=[7, 0])
        assert (
            b.info()["requests"][0]["position"]
            == b.info()["requests"][7]["position"]
            == 0
        )
        for i in range(1, 7):
            same_state(b.snapshot(i), before[i])
            same_draft(b.draft_snapshot(i), draft[i])
        for i in [0, 7]:
            refs[i].reset()
        want = [refs[i].generate(prompts[i], max_new_tokens=maximum) for i in [7, 0]]
        assert (
            b.generate(
                [prompts[i] for i in [7, 0]], slots=[7, 0], max_new_tokens=maximum
            )
            == want
        )
        for i in [7, 0]:
            same_draft(b.draft_snapshot(i), refs[i].draft_snapshot())
    finally:
        b.close()
        for r in refs:
            r.close()
        model.close()
        target.close()


@pytest.mark.parametrize("position", [0, 3])
def test_failure_poison_reset_lifetime_no_python_math(
    tmp_path, engine, position, monkeypatch
):
    target, model = models(tmp_path, engine, True)
    b = model.create_batch([32] * 4)
    refs = [model.create_session(32) for _ in range(4)]
    prompts = [[i + 1, 2, 3] for i in range(4)]
    try:
        expected = [r.generate(p, max_new_tokens=7) for r, p in zip(refs, prompts)]
        b._inject_failure(slot=3, layer=3, position=position)
        with pytest.raises(RuntimeError, match="injected"):
            b.generate(prompts, max_new_tokens=7)
        assert all(r["failed"] for r in b.info()["requests"])
        b.reset(slots=[3])
        assert (
            b.info()["requests"][0]["failed"] and not b.info()["requests"][3]["failed"]
        )
        assert b.generate([prompts[3]], slots=[3], max_new_tokens=7) == [expected[3]]
        b.reset()
        target.close()
        model.close()

        def forbidden(*a, **kw):
            raise AssertionError("MTP batch used Python/Torch numerical fallback")

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
            assert b.generate(prompts, max_new_tokens=7) == expected
        for i in range(4):
            same_draft(b.draft_snapshot(i), refs[i].draft_snapshot())
    finally:
        b.close()
        for r in refs:
            r.close()
        model.close()
        target.close()


def test_overlapping_batches_scalar_and_original_target(tmp_path, engine):
    target, model = models(tmp_path, engine, True)
    batches = [model.create_batch([32] * 4) for _ in range(2)]
    scalar = model.create_session(32)
    ref = target.create_session(32)
    prompts = [[i + 1, 2, 3] for i in range(4)]
    try:
        expected = []
        for p in prompts:
            ref.reset()
            expected.append(ref.generate(p, max_new_tokens=9))
        for _ in range(2):
            for b in batches:
                b.reset()
            scalar.reset()
            ref.reset()
            with ThreadPoolExecutor(max_workers=4) as pool:
                jobs = [
                    pool.submit(b.generate, prompts, max_new_tokens=9) for b in batches
                ]
                a = pool.submit(
                    scalar.generate,
                    prompts[0],
                    max_new_tokens=9,
                    draft_tokens=2,
                    recovery="prefix",
                )
                c = pool.submit(ref.generate, prompts[0], max_new_tokens=9)
                assert all(j.result(timeout=60) == expected for j in jobs)
                assert a.result(timeout=60) == c.result(timeout=60) == expected[0]
            for i in range(4):
                same_state(batches[0].snapshot(i), batches[1].snapshot(i))
                same_draft(batches[0].draft_snapshot(i), batches[1].draft_snapshot(i))
    finally:
        for b in batches:
            b.close()
        scalar.close()
        ref.close()
        model.close()
        target.close()


def test_cpu_model_capacity_export_and_continuation_preflight():
    with pytest.raises(ValueError, match="MTP model"):
        NativeQwenMTPBatchSession.create(object(), [12])
    model = object.__new__(NativeQwenMTPModel)
    model.target = object()
    model._dtype = torch.float16
    model._max_tokens = 4
    model._cfg = dict(max_position_embeddings=20)
    for capacities in [[], [12] * 3, [0], [21], [True]]:
        with pytest.raises(ValueError):
            NativeQwenMTPBatchSession.create(model, capacities)
    model._backend = SimpleNamespace(_native=SimpleNamespace())
    with pytest.raises(RuntimeError, match="exports absent"):
        NativeQwenMTPBatchSession.create(model, [12])
    with pytest.raises(ValueError, match="continuation"):
        NativeQwenMTPBatchSession().forward([[1]])


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--mtp-artifact", "draft", "--mtp-depth", "2"],
        ["--mtp-artifact", "draft", "--mtp-recovery", "prefix"],
        ["--mtp-artifact", "draft", "--batch-target"],
        ["--mtp-artifact", "draft", "--mode", "pp"],
    ],
)
def test_server_batch_mtp_preflight_before_runtime(extra, monkeypatch):
    import sys
    from test_completion_benchmark import load

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "serve",
            "not-opened",
            "--library",
            "absent",
            "--mode",
            "tp",
            "--port",
            "18953",
            "--ready-file",
            "unused",
            "--batch-mtp",
        ]
        + extra,
    )
    with pytest.raises(SystemExit) as e:
        load("serve_native_benchmark").main()
    assert e.value.code == 2


def test_mtp_cohort_response_counters_not_summed():
    from test_completion_benchmark import FakeBatch, load

    class FakeMTP(FakeBatch):
        def info(self):
            r = super().info()
            for i, x in enumerate(r["requests"]):
                x["mtp"] = {
                    "proposed_tokens": i + 1,
                    "accepted_tokens": i,
                    "replayed_input_tokens": 0,
                }
            r.update(
                {
                    k: 1
                    for k in [
                        "shared_verification_wall_seconds",
                        "shared_prefill_complete_seconds",
                        "shared_native_generation_seconds",
                        "mtp_model_device_bytes",
                        "all_mtp_request_device_bytes",
                        "all_target_backup_device_bytes",
                        "shared_mtp_selection_device_bytes",
                        "prefill_strategy",
                        "draft_strategy",
                        "target_verification",
                    ]
                }
            )
            return r

    b = FakeMTP()
    admission = load("serve_native_benchmark").CompletionBatcher(b, 8, 0.02)
    try:
        jobs = [admission.submit(i, [i + 1], 3) for i in range(4)]
        rows = [j.result(timeout=2) for j in jobs]
        assert len(b.calls) == 1 and {r["batch_cohort_id"] for r in rows} == {1}
        assert [r["mtp_stats"]["proposed_tokens"] for r in rows] == [1, 2, 3, 4]
        assert {
            r["mtp_batch_stats"]["shared_native_generation_seconds"] for r in rows
        } == {1}
        assert all("count once" in r["mtp_batch_stats_scope"] for r in rows)
    finally:
        admission.close()


def metadata_fake_batch():
    import threading

    state = dict(metadata=0, changed=False, generated=0, positions=[0, 0])

    def metadata():
        state["metadata"] += 1
        if state["changed"]:
            raise RuntimeError("metadata/source changed")

    def info(h, shared, rows, times):
        shared[0] = 2
        for i in range(2):
            rows[4 * i] = state["positions"][i]
            rows[4 * i + 2] = 20
        return 0

    def generate(
        h, slots, count, tokens, offsets, maximum, eos, eo, out, stride, counts
    ):
        state["generated"] += 1
        for j in range(count):
            counts[j] = maximum[j]
            for k in range(maximum[j]):
                out[j * stride + k] = 3
            state["positions"][slots[j]] = offsets[j + 1] - offsets[j] + counts[j] - 1
        return 0

    def stats(h, slot, values, times):
        values[11] = state["positions"][slot]
        return 0

    native = SimpleNamespace(
        info=info,
        generate=generate,
        stats=stats,
        extra_info=lambda *a: 0,
        cohort_timings=lambda *a: 0,
    )
    batch = NativeQwenMTPBatchSession()
    batch.capacities = (20, 20)
    batch._handle = 1
    batch._native = native
    batch._lock = threading.RLock()
    batch._finalizer = SimpleNamespace(alive=True)

    def check(rc):
        assert rc == 0

    batch.model = SimpleNamespace(
        _cfg=dict(vocab_size=8),
        projection_info={},
        _metadata=metadata,
        _backend=SimpleNamespace(_check=check),
    )
    return batch, state


def test_metadata_checked_once_per_public_info_stats_and_generate():
    b, state = metadata_fake_batch()
    assert len(b.info()["requests"]) == 2 and state["metadata"] == 1
    b.stats(0)
    assert state["metadata"] == 2
    assert b.generate([[1], [2]], max_new_tokens=2) == [[3, 3], [3, 3]]
    assert state["metadata"] == 3 and state["generated"] == 1
    with pytest.raises(RuntimeError, match="fresh/reset"):
        b.generate([[1]], max_new_tokens=1)
    assert state["generated"] == 1


def test_metadata_mutation_guard_preserved_without_recursive_checks():
    b, state = metadata_fake_batch()
    state["changed"] = True
    for call in [
        b.info,
        lambda: b.stats(0),
        lambda: b.generate([[1]], max_new_tokens=2),
    ]:
        with pytest.raises(RuntimeError, match="metadata/source changed"):
            call()
    assert (
        state["metadata"] == 3
        and state["generated"] == 0
        and state["positions"] == [0, 0]
    )
