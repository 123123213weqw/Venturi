"""Explicit native row-scarcity trajectories/state, not just underfilled plans."""
import pytest

from test_qwen_batch import batch_engine, load
from test_qwen_tensor_parallel import same_state
from venturi.runtime import ContinuousScheduler


@pytest.mark.cuda
@pytest.mark.parametrize("prefill_policy", ["balanced", "balanced-short"])
@pytest.mark.parametrize("tp,profile", [
    (False, "fp16"), (True, "q8-residual"),
    (True, "q8-residual-column-mlp-head-gdn"),
])
def test_native_balanced_row_scarcity_has_no_early_sample_and_matches_scalar(tmp_path, batch_engine, tp, profile, prefill_policy):
    if profile.endswith("head-gdn"):
        from test_native_head_gdn import eligible_mtp, load as head_load
        weights, _, _, _ = eligible_mtp(tmp_path / "head")
        model = head_load(weights, batch_engine, metadata_mode="sealed")
    else:
        model = load(tmp_path, batch_engine, tp, profile=profile, metadata_mode="sealed")
    batch = model.create_batch([64] * 4)
    refs = [model.create_session(64) for _ in range(4)]
    scheduler = ContinuousScheduler(batch, token_budget=8, prefill_chunk_size=8,
                                    prefill_policy=prefill_policy)
    prompts = [[1 + (i + j) % 7 for j in range(8)] for i in range(4)]
    try:
        expected = [ref.generate(prompt, max_new_tokens=5, eos_token_ids=[])
                    for ref, prompt in zip(refs, prompts)]
        tickets = [scheduler.submit(prompt, max_new_tokens=5) for prompt in prompts]
        for _ in range(3):
            assert scheduler.tick() == ()
            assert not any(t.future.done() for t in tickets)
        first = scheduler.tick()
        assert len(first) == 4
        while scheduler.has_work:
            scheduler.tick()
        assert [t.future.result().output_ids for t in tickets] == [tuple(x) for x in expected]
        for slot, ref in enumerate(refs):
            same_state(batch.snapshot(slot), ref.snapshot())
    finally:
        scheduler.close()
        batch.close()
        for ref in refs:
            ref.close()
        model.close()
