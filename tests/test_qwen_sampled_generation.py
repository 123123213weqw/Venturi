"""Seeded sampling on the native Qwen runtime: the Python-driven sampled loop
collapses to greedy at cold temperature, is seed-deterministic, and leaves the
default (no sampling) native greedy path untouched."""

import os

import pytest
import torch

from venturi.engine.sampling import SamplingParams
from venturi.models import NativeQwenModel
from venturi.models.qwen.native_runtime import QwenRuntime
from venturi.ops import CudaBackend
from test_native_head_gdn import eligible_mtp

pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    ops = os.environ.get("VENTURI_OPS_LIBRARY")
    qwen = os.environ.get("VENTURI_QWEN_LIBRARY")
    if not (ops and qwen and torch.cuda.is_available()):
        pytest.skip("Explicit split ops/Qwen CUDA libraries and CUDA required")
    weights, _, _, _ = eligible_mtp(
        tmp_path_factory.mktemp("sampling-source") / "source"
    )
    model = NativeQwenModel.from_q8(
        weights,
        QwenRuntime(qwen, CudaBackend(ops)),
        weight_mode="resident",
        dtype=torch.float16,
        device="cuda:0",
        max_tokens=32,
        reserve_bytes=0,
    )
    yield model
    model.close()


def test_cold_sampling_collapses_to_greedy(engine):
    session = engine.create_session(64)
    try:
        prompt = [5, 9, 2]
        greedy = session.generate(prompt, max_new_tokens=12)
        session.reset()
        cold = session.generate(
            prompt, max_new_tokens=12, sampling=SamplingParams(temperature=1e-9, seed=3)
        )
        assert cold == greedy
    finally:
        session.close()


def test_same_seed_replays_identically(engine):
    session = engine.create_session(64)
    try:
        prompt = [7, 1, 4]
        params = SamplingParams(temperature=0.9, top_k=8, top_p=0.95, seed=42)
        first = session.generate(prompt, max_new_tokens=10, sampling=params)
        session.reset()
        second = session.generate(prompt, max_new_tokens=10, sampling=params)
        assert first == second and len(first) == 10
    finally:
        session.close()


def test_default_generate_unchanged_native_path(engine):
    session = engine.create_session(64)
    try:
        tokens = session.generate([1, 2, 3], max_new_tokens=4)
        assert len(tokens) == 4 and all(
            0 <= t < engine._cfg["vocab_size"] for t in tokens
        )
        session.reset()
        again = session.generate([1, 2, 3], max_new_tokens=4)
        assert again == tokens  # native greedy stays deterministic
    finally:
        session.close()


def test_sampling_eos_counts_and_leaves_the_same_cursor_as_greedy(engine):
    session = engine.create_session(64)
    try:
        prompt = [5, 9, 2]
        token = session.generate(prompt, max_new_tokens=1)[0]
        greedy_position = session.position
        session.reset()
        result = session.generate(
            prompt,
            max_new_tokens=12,
            eos_token_ids=(token,),
            sampling=SamplingParams(top_k=1, seed=2**64 - 1),
        )
        assert result == [token]
        assert session.position == greedy_position == len(prompt)
    finally:
        session.close()
