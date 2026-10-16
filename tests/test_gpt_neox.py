from dataclasses import replace

import pytest
import torch

from venturi import load_checkpoint
from venturi.models import GPTNeoXConfig, GPTNeoXModel
from neox_reference import full_logits, tiny_checkpoint


@pytest.fixture
def make_model(make_checkpoint):
    def create(*, parallel=True, rotary_pct=0.5, dtype=torch.float64):
        config, weights = tiny_checkpoint(parallel, rotary_pct)
        stored = {k: v.to(dtype) for k, v in weights.items()}
        path = make_checkpoint(config=config, tensors=stored)
        checkpoint = load_checkpoint(path, device="cpu", loader="safetensors")
        return GPTNeoXModel(checkpoint), config, stored, checkpoint

    return create


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("rotary_pct", [0.0, 0.5, 1.0])
def test_full_forward_matches_independent_fp64(make_model, parallel, rotary_pct):
    model, config, weights, _ = make_model(parallel=parallel, rotary_pct=rotary_pct)
    ids = [2, 5, 11, 17, 8]
    actual = model.create_session(8).forward(ids, all_logits=True)
    torch.testing.assert_close(
        actual, full_logits(ids, config, weights), rtol=2e-11, atol=2e-11
    )


@pytest.mark.parametrize("chunks", [[1, 1, 1, 1, 1], [2, 3], [3, 1, 1]])
def test_kv_cache_absolute_causality(make_model, chunks):
    model, config, weights, _ = make_model()
    ids = [2, 5, 11, 17, 8]
    expected = full_logits(ids, config, weights)
    session = model.create_session(8)
    begin = 0
    for count in chunks:
        actual = session.forward(ids[begin : begin + count], all_logits=True)
        torch.testing.assert_close(
            actual, expected[begin : begin + count], rtol=2e-11, atol=2e-11
        )
        begin += count
        assert session.position == begin


def test_future_tokens_do_not_change_earlier_logits(make_model):
    model, *_ = make_model()
    first = model.create_session(8).forward([2, 3, 4, 5], all_logits=True)
    second = model.create_session(8).forward([2, 3, 8, 9], all_logits=True)
    torch.testing.assert_close(first[:2], second[:2], rtol=0, atol=0)


def test_outer_autocast_does_not_change_declared_dtype(make_model):
    model, *_ = make_model(dtype=torch.float32)
    expected = model.create_session(8).forward([2, 3, 4])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = model.create_session(8).forward([2, 3, 4])
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_preserves_storage_and_matches_oracle(make_model, dtype):
    model, config, weights, _ = make_model(dtype=dtype)
    actual = model.create_session(8).forward([2, 5, 11, 17], all_logits=True)
    expected = full_logits(
        [2, 5, 11, 17],
        config,
        {name: tensor.double() for name, tensor in weights.items()},
    )
    assert actual.dtype == dtype and model.dtype == dtype
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=4 * torch.finfo(dtype).eps,
        atol=max(0.002, torch.finfo(dtype).eps),
    )


def test_sessions_reset_and_capacity(make_model):
    model, config, weights, _ = make_model()
    a, b = model.create_session(6), model.create_session(6)
    assert a._keys.data_ptr() != b._keys.data_ptr()
    pointer = a._keys.data_ptr()
    a.forward([2, 3])
    b.forward([8])
    a.forward([4])
    torch.testing.assert_close(
        b.forward([9]), full_logits([8, 9], config, weights)[-1], rtol=2e-11, atol=2e-11
    )
    a.reset()
    assert a.position == 0 and a._keys.data_ptr() == pointer
    torch.testing.assert_close(
        a.forward([7, 6]),
        full_logits([7, 6], config, weights)[-1],
        rtol=2e-11,
        atol=2e-11,
    )
    assert a.cache_bytes == 2 * 2 * 2 * 6 * 8 * 8
    with pytest.raises(ValueError, match="capacity"):
        a.forward([2] * 5)
    assert a.position == 2
    for capacity in [0, -1, True, 33]:
        with pytest.raises(ValueError):
            model.create_session(capacity)


@pytest.mark.parametrize("ids", [[], [-1], [31], [True], torch.tensor([2]), "hello"])
def test_input_validation_before_state_mutation(make_model, ids):
    model, *_ = make_model()
    session = model.create_session(8)
    with pytest.raises(ValueError):
        session.forward(ids)
    assert session.position == 0
    session.forward([2])


def test_execution_failure_requires_reset(make_model, monkeypatch):
    model, config, weights, _ = make_model()
    session = model.create_session(8)
    with monkeypatch.context() as patch:
        patch.setattr(
            model,
            "_linear",
            lambda *args: (_ for _ in ()).throw(RuntimeError("injected")),
        )
        with pytest.raises(RuntimeError, match="injected"):
            session.forward([2])
    with pytest.raises(RuntimeError, match="reset"):
        session.forward([2])
    session.reset()
    torch.testing.assert_close(
        session.forward([2]),
        full_logits([2], config, weights)[-1],
        rtol=2e-11,
        atol=2e-11,
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("model_type", "qwen3_5"),
        ("hidden_size", True),
        ("num_attention_heads", 3),
        ("hidden_act", "gelu_new"),
        ("partial_rotary_factor", 0.75),
        ("rope_theta", 1000000),
        ("rotary_pct", 0.375),
        ("rotary_pct", 1.1),
        ("rope_scaling", {}),
        ("quantization_config", {}),
        ("use_parallel_residual", "yes"),
        ("tie_word_embeddings", True),
        ("layer_norm_eps", float("nan")),
        ("eos_token_id", [-1]),
        ("bos_token_id", True),
    ],
)
def test_unsupported_config_rejected(key, value):
    config, _ = tiny_checkpoint()
    config[key] = value
    with pytest.raises(ValueError):
        GPTNeoXConfig.from_dict(config)


@pytest.mark.parametrize(
    "case", ["missing", "shape", "dtype", "unknown", "legacy-shape"]
)
def test_binding_validation(make_model, case):
    _, _, _, checkpoint = make_model()
    weights = dict(checkpoint.weights)
    key = "gpt_neox.layers.0.attention.query_key_value.weight"
    if case == "missing":
        weights.pop(key)
    if case == "shape":
        weights[key] = weights[key].T.contiguous()
    if case == "dtype":
        weights[key] = weights[key].float()
    if case == "unknown":
        weights["some.other.weight"] = torch.ones(1)
    if case == "legacy-shape":
        weights["gpt_neox.layers.0.attention.bias"] = torch.ones(2)
    with pytest.raises(ValueError):
        GPTNeoXModel(replace(checkpoint, weights=weights))


def test_legacy_buffers_do_not_override_rope(make_model):
    model, config, weights, checkpoint = make_model()
    extended = dict(checkpoint.weights)
    for i in range(config["num_hidden_layers"]):
        root = f"gpt_neox.layers.{i}.attention."
        extended[root + "bias"] = torch.zeros(1, 1, 32, 32, dtype=torch.uint8)
        extended[root + "masked_bias"] = torch.tensor(-123.0)
        extended[root + "rotary_emb.inv_freq"] = torch.zeros(2, dtype=torch.float16)
    reloaded = GPTNeoXModel(replace(checkpoint, weights=extended))
    torch.testing.assert_close(
        model.create_session(8).forward([2, 3]),
        reloaded.create_session(8).forward([2, 3]),
        rtol=0,
        atol=0,
    )


def test_transformers_reference(make_model):
    transformers = pytest.importorskip("transformers")
    model, config, weights, _ = make_model(dtype=torch.float32)
    hf_config = transformers.GPTNeoXConfig(**config)
    hf_config._attn_implementation = "eager"
    reference = transformers.GPTNeoXForCausalLM(hf_config).eval()
    reference_weights = dict(weights)
    # Transformers 5 renamed the head; keep the reference-only compatibility
    # translation separate from Venturi's explicit classic checkpoint contract.
    if "lm_head.weight" in reference.state_dict():
        reference_weights["lm_head.weight"] = reference_weights.pop("embed_out.weight")
    reference.load_state_dict(reference_weights, strict=True)
    ids = [2, 5, 11, 17, 8]
    with torch.inference_mode():
        expected = reference(torch.tensor([ids]), use_cache=False).logits[0]
    actual = model.create_session(8).forward(ids, all_logits=True)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
