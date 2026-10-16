import json

import pytest
import torch

from neox_reference import full_logits, tiny_checkpoint
from venturi import load_checkpoint
from venturi.cli import main
from venturi.generation import LocalTokenizer, generate_text, generate_tokens
from venturi.models import GPTNeoXModel


@pytest.fixture
def model_fixture(make_checkpoint):
    config, weights = tiny_checkpoint()
    path = make_checkpoint(config=config, tensors=weights)
    model = GPTNeoXModel(load_checkpoint(path, device="cpu", loader="safetensors"))
    return model, config, weights, path


@pytest.fixture
def tokenizer_path(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {f"t{i}": i for i in range(31)}
    vocab.pop("t2")
    vocab["hello"] = 2
    vocab.pop("t3")
    vocab["world"] = 3
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="t4"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(["t0", "t1"])
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="t0 $A", special_tokens=[("t0", 0)]
    )
    tokenizer.enable_truncation(1)
    tokenizer.enable_padding(length=4, pad_id=0, pad_token="t0")
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    return path


def test_greedy_tokens_against_uncached_oracle(model_fixture):
    model, config, weights, _ = model_fixture
    prompt = [2, 5, 9]
    expected = []
    for _ in range(6):
        expected.append(
            int(full_logits(prompt + expected, config, weights)[-1].argmax())
        )
    result = generate_tokens(model, prompt, max_new_tokens=6, eos_token_ids=())
    assert result.generated_ids == tuple(expected)
    assert result.prompt_ids == tuple(prompt)
    assert result.finish_reason == "length"
    assert result.prefill_seconds >= 0 and result.decode_seconds >= 0
    assert result.total_seconds >= result.prefill_seconds + result.decode_seconds
    assert result.kv_cache_bytes == 2 * 2 * 2 * 9 * 8 * 8


def test_eos_and_zero_token_budget(model_fixture, monkeypatch):
    model, *_ = model_fixture
    first = generate_tokens(
        model, [2, 3], max_new_tokens=1, eos_token_ids=()
    ).generated_ids[0]
    result = generate_tokens(model, [2, 3], max_new_tokens=8, eos_token_ids=(first,))
    assert result.generated_ids == (first,) and result.finish_reason == "eos"
    with monkeypatch.context() as patch:
        patch.setattr(
            model,
            "create_session",
            lambda *args: pytest.fail("zero-token generation allocated cache"),
        )
        result = generate_tokens(model, [2, 3], max_new_tokens=0)
        assert result.generated_ids == () and result.kv_cache_bytes == 0


def test_nonfinite_logits_do_not_become_eos(model_fixture, monkeypatch):
    from types import SimpleNamespace

    model, *_ = model_fixture
    broken = SimpleNamespace(
        forward=lambda ids: torch.full((31,), float("nan")), cache_bytes=0
    )
    monkeypatch.setattr(model, "create_session", lambda *args: broken)
    with pytest.raises(RuntimeError, match="nonfinite logits"):
        generate_tokens(model, [2, 3], max_new_tokens=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": -1},
        {"max_new_tokens": True},
        {"max_new_tokens": 30},
        {"eos_token_ids": (-1,)},
        {"eos_token_ids": (31,)},
    ],
)
def test_invalid_generation_request(model_fixture, kwargs):
    model, *_ = model_fixture
    with pytest.raises(ValueError):
        generate_tokens(model, [2, 3, 4], **kwargs)


def test_tokenizer_is_local_explicit_and_untruncated(tokenizer_path):
    tokenizer = LocalTokenizer(tokenizer_path, vocab_size=31)
    assert tokenizer.encode("hello world") == (2, 3)
    assert tokenizer.decode([0, 2, 3, 1]) == "hello world"
    assert len(tokenizer.sha256) == 64
    with pytest.raises(ValueError, match="absent"):
        tokenizer.decode([31])
    with pytest.raises(ValueError, match="vocabulary"):
        LocalTokenizer(tokenizer_path, vocab_size=3)


def test_text_completion_and_empty_bos(model_fixture, tokenizer_path):
    model, *_ = model_fixture
    tokenizer = LocalTokenizer(tokenizer_path, vocab_size=31)
    result = generate_text(model, tokenizer, "hello world", max_new_tokens=3)
    assert result.prompt_ids == (2, 3)
    assert result.text == tokenizer.decode(result.generated_ids)
    assert generate_text(model, tokenizer, "", max_new_tokens=1).prompt_ids == (0,)


def test_cli_generation_json_and_plain(model_fixture, tokenizer_path, capsys):
    _, _, _, path = model_fixture
    args = [
        "generate",
        str(path),
        "--device",
        "cpu",
        "--loader",
        "safetensors",
        "--tokenizer",
        str(tokenizer_path),
        "--prompt",
        "hello world",
        "--max-new-tokens",
        "3",
    ]
    assert main(args + ["--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["generation"]["prompt_ids"] == [2, 3]
    assert summary["generation"]["generated_ids"]
    assert "Venturi GPT-NeoX" in summary["execution"]
    assert main(args) == 0
    assert capsys.readouterr().out.rstrip("\n") == summary["generation"]["text"]
    assert main(args + ["--max-new-tokens", "40"]) == 1
    assert "max_position_embeddings" in capsys.readouterr().err


def test_cli_rejects_unknown_arch_before_loading(make_checkpoint, capsys, monkeypatch):
    path = make_checkpoint()
    import venturi.cli

    monkeypatch.setattr(
        venturi.cli,
        "load_checkpoint",
        lambda *args, **kwargs: pytest.fail("unsupported architecture loaded weights"),
    )
    assert main(["generate", str(path), "--prompt", "hello"]) == 1
    assert "model_type='gpt_neox'" in capsys.readouterr().err


def test_malformed_tokenizer(tmp_path):
    pytest.importorskip("tokenizers")
    path = tmp_path / "tokenizer.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="Invalid tokenizer"):
        LocalTokenizer(path, vocab_size=31)
