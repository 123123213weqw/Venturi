"""Real safetensors bindings and full tiny-model fixtures, without downloads."""

import json

import pytest
import torch
from safetensors.torch import save_file

from venturi.generation import generate_tokens
from venturi.loader import load_checkpoint
from venturi.models import LlamaConfig, LlamaWeights
from venturi.ops import TorchBackend


def llama_fixture(directory, *, dtype=torch.float64, tied=False, sharded=False):
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(
        model_type="llama",
        vocab_size=37,
        hidden_size=128,
        intermediate_size=80,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=tied,
        bos_token_id=1,
        eos_token_id=None,
    )
    g = torch.Generator().manual_seed(934)

    def weight(shape):
        return (torch.randn(shape, generator=g, dtype=torch.float64) * 0.06).to(dtype)

    tensors = {
        "model.embed_tokens.weight": weight((37, 128)),
        "model.norm.weight": (1 + weight((128,))),
    }
    if not tied:
        tensors["lm_head.weight"] = weight((37, 128))
    for n in range(2):
        prefix = f"model.layers.{n}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            tensors[prefix + name + ".weight"] = 1 + weight((128,))
        for name, shape in (
            ("self_attn.q_proj", (128, 128)),
            ("self_attn.k_proj", (64, 128)),
            ("self_attn.v_proj", (64, 128)),
            ("self_attn.o_proj", (128, 128)),
            ("mlp.gate_proj", (80, 128)),
            ("mlp.up_proj", (80, 128)),
            ("mlp.down_proj", (128, 80)),
        ):
            tensors[prefix + name + ".weight"] = weight(shape)
    (directory / "config.json").write_text(json.dumps(config))
    if sharded:
        mapping = {}
        for index in range(2):
            name = f"model-{index + 1:05d}-of-00002.safetensors"
            part = dict(list(tensors.items())[index::2])
            save_file(part, directory / name)
            mapping.update({key: name for key in part})
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": mapping})
        )
    else:
        save_file(tensors, directory / "model.safetensors")
    return config, tensors


def llama_oracle(config, weights, ids):
    """Independent full-sequence FP64 formula; no production layer/op imports."""
    c = config
    h, heads, kvheads = (
        c["hidden_size"],
        c["num_attention_heads"],
        c["num_key_value_heads"],
    )
    d, t = h // heads, len(ids)
    w = {k: v.double() for k, v in weights.items()}
    x = w["model.embed_tokens.weight"][ids]
    # Both engines deliberately use FP32 position frequencies/tables.
    inv = c["rope_theta"] ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)
    phase = torch.arange(t, dtype=torch.float32)[:, None] * inv
    cos, sin = phase.cos().double()[:, None], phase.sin().double()[:, None]

    def norm(x, gamma):
        return (
            x
            * torch.rsqrt(x.square().mean(-1, keepdim=True) + c["rms_norm_eps"])
            * gamma
        )

    def rope(x):
        a, b = x.chunk(2, -1)
        return torch.cat((a * cos - b * sin, a * sin + b * cos), -1)

    for n in range(c["num_hidden_layers"]):
        p = f"model.layers.{n}."
        y = norm(x, w[p + "input_layernorm.weight"])
        q = rope((y @ w[p + "self_attn.q_proj.weight"].T).reshape(t, heads, d))
        k = rope((y @ w[p + "self_attn.k_proj.weight"].T).reshape(t, kvheads, d))
        v = (y @ w[p + "self_attn.v_proj.weight"].T).reshape(t, kvheads, d)
        k, v = (a.repeat_interleave(heads // kvheads, dim=1) for a in (k, v))
        scores = torch.einsum("thd,shd->hts", q, k) / d**0.5
        scores.masked_fill_(torch.ones(t, t, dtype=torch.bool).triu(1), -torch.inf)
        attended = torch.einsum("hts,shd->thd", scores.softmax(-1), v).reshape(t, h)
        x = x + attended @ w[p + "self_attn.o_proj.weight"].T
        y = norm(x, w[p + "post_attention_layernorm.weight"])
        gate, up = (
            y @ w[p + "mlp.gate_proj.weight"].T,
            y @ w[p + "mlp.up_proj.weight"].T,
        )
        x = x + (torch.nn.functional.silu(gate) * up) @ w[p + "mlp.down_proj.weight"].T
    head = (
        w["model.embed_tokens.weight"]
        if c["tie_word_embeddings"]
        else w["lm_head.weight"]
    )
    return norm(x, w["model.norm.weight"]) @ head.T


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("sharded", [False, True])
def test_bind_upload_full_prefill_chunk_decode_and_generation(tmp_path, tied, sharded):
    c, weights = llama_fixture(tmp_path, tied=tied, sharded=sharded)
    bound = LlamaWeights.bind(
        load_checkpoint(tmp_path, device="cpu", loader="safetensors")
    )
    assert bound.config == LlamaConfig.from_dict(c)
    model = bound.upload(TorchBackend(), device="cpu")
    assert (model.head is model.embedding) is tied
    assert model.storage_nbytes == sum(
        t.numel() * t.element_size() for t in weights.values()
    )
    session = model.create_session(32, max_tokens=3)
    ids = [1, 5, 2, 3, 7, 8, 4]
    expected = llama_oracle(c, weights, ids)
    actual = session.forward(ids, all_logits=True)
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    session.reset()
    parts = [session.forward(ids[:2], all_logits=True)]
    for token in ids[2:]:
        parts.append(session.forward([token], all_logits=True))
    torch.testing.assert_close(torch.cat(parts), expected, atol=3e-12, rtol=3e-12)
    result = generate_tokens(model, ids[:3], max_new_tokens=4, session=session)
    history, predicted = ids[:3], []
    for _ in range(4):
        token = int(llama_oracle(c, weights, history)[-1].argmax())
        predicted.append(token)
        history = history + [token]
    assert result.generated_ids == tuple(predicted)
    assert result.kv_cache_bytes == session.cache_bytes
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.forward([1])


@pytest.mark.parametrize(
    "key,value",
    [
        ("model_type", "qwen3_5"),
        ("rope_scaling", {"rope_type": "llama3"}),
        ("quantization_config", {"bits": 4}),
        ("attention_bias", True),
        ("qk_norm", True),
        ("partial_rotary_factor", 0.5),
        ("hidden_act", "gelu"),
        ("head_dim", 128),
        ("sliding_window", 32),
        ("tie_word_embeddings", 1),
        ("rope_theta", float("nan")),
        ("num_key_value_heads", 3),
        ("eos_token_id", [99]),
    ],
)
def test_unsupported_config_rejected_before_upload(tmp_path, key, value):
    c, _ = llama_fixture(tmp_path)
    c[key] = value
    with pytest.raises(ValueError):
        LlamaConfig.from_dict(c)


@pytest.mark.parametrize(
    "mutation", ["missing", "unexpected", "shape", "dtype", "nan", "tied_conflict"]
)
def test_binding_failures_no_device_upload(tmp_path, mutation):
    from dataclasses import replace

    llama_fixture(tmp_path, tied=mutation == "tied_conflict")
    loaded = load_checkpoint(tmp_path, device="cpu", loader="safetensors")
    weights = dict(loaded.weights)
    name = "model.layers.0.self_attn.k_proj.weight"
    if mutation == "missing":
        weights.pop(name)
    elif mutation == "unexpected":
        weights[name + ".bias"] = torch.zeros(1)
    elif mutation == "shape":
        weights[name] = weights[name][:1]
    elif mutation == "dtype":
        weights[name] = weights[name].float()
    elif mutation == "nan":
        weights[name][0, 0] = torch.nan
    else:
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"].clone() + 1
    with pytest.raises(ValueError):
        LlamaWeights.bind(replace(loaded, weights=weights))


def test_session_preflight_failure_and_runtime_poison(tmp_path, monkeypatch):
    llama_fixture(tmp_path)
    model = LlamaWeights.bind(
        load_checkpoint(tmp_path, device="cpu", loader="safetensors")
    ).upload(TorchBackend(), device="cpu")
    session = model.create_session(4, max_tokens=2)
    for ids in ([], [-1], [37], [True], [1] * 5):
        with pytest.raises(ValueError):
            session.forward(ids)
        assert session.position == 0 and not session.failed

    def fail(*args, **kwargs):
        raise RuntimeError("injected submission failure")

    with monkeypatch.context() as m:
        m.setattr(type(session.caches[1]._owner.norm), "prepare", fail)
        with pytest.raises(RuntimeError, match="injected"):
            session.forward([1])
    assert session.failed
    with pytest.raises(RuntimeError, match="reset"):
        session.forward([1])
    session.reset()
    assert session.position == 0 and not session.failed
    session.forward([1])
    session.close()


def test_cli_explicit_reference_and_inspect(tmp_path, capsys):
    from venturi.cli import main

    llama_fixture(tmp_path)
    assert main(["llama", "inspect", str(tmp_path)]) == 0
    assert len(json.loads(capsys.readouterr().out)["bindings"]) == 21
    assert (
        main(
            [
                "llama",
                "generate",
                str(tmp_path),
                "--backend",
                "torch-reference",
                "--device",
                "cpu",
                "--token-ids",
                "1,2,3",
                "--max-new-tokens",
                "3",
            ]
        )
        == 0
    )
    assert len(json.loads(capsys.readouterr().out)["generated_ids"]) == 3
