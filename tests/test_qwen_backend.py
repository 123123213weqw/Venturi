"""Resident-process contracts tested with a CPU-only fake native runner."""

import os
import sys

import pytest

from test_ninfer import write_artifact
from venturi.backends.qwen import QwenNativeSession
from venturi.cli import main
from venturi.ninfer import NInferArtifact


@pytest.fixture
def fixture(tmp_path):
    artifact = write_artifact(tmp_path)
    runner = tmp_path / "runner with spaces"
    runner.write_text(
        f"#!{sys.executable}\n"
        + """
import json,os,sys,time
config=json.loads(sys.stdin.readline())
print(json.dumps(dict(ok=True,event="ready",protocol=1,pid=os.getpid(),config=config,
                     inherited_flags=[k for k in os.environ if k.startswith("NINFER_")])) ,flush=True)
for line in sys.stdin:
 r=json.loads(line)
 if r.get("op")=="tokenize":
  print(json.dumps(dict(ok=True,token_ids=[1,2,3])),flush=True)
  continue
 prompt=r.get("prompt", "")
 if prompt=="ERROR":
  print(json.dumps(dict(ok=False,error="bad request",available=True)),flush=True)
  continue
 if prompt=="MALFORMED":
  print("not JSON",flush=True)
  continue
 if prompt=="PARTIAL":
  sys.stdout.write('{"ok":'); sys.stdout.flush(); time.sleep(10)
  continue
 if prompt=="STDERR":
  sys.stderr.write("diagnostic "*20000); sys.stderr.flush()
 ids=[-1] if prompt=="BAD_IDS" else list(range(1,r["max_new_tokens"]+1))
 print(json.dumps(dict(ok=True,generated_ids=ids,text="answer",reused_prompt_tokens=0,
  timings=dict(first_token_seconds=.01,prefill_seconds=.01,decode_seconds=.03,
               generation_wall_seconds=.04,total_seconds=.05))),flush=True)
"""
    )
    runner.chmod(0o755)
    return artifact, runner


def session(fixture, **kwargs):
    artifact, runner = fixture
    return QwenNativeSession(
        artifact, runner=runner, max_context=32, prefill_chunk=16, **kwargs
    )


def test_resident_requests_tokenizer_benchmark_and_clean_exit(fixture, monkeypatch):
    monkeypatch.setenv("NINFER_GDN_STATE_F16", "1")
    with session(fixture) as model:
        assert model.ready["inherited_flags"] == []
        pid = model.ready["pid"]
        assert model.tokenize("hello") == [1, 2, 3]
        assert model.generate("hello", max_new_tokens=4)["generated_ids"] == [
            1,
            2,
            3,
            4,
        ]
        assert model.generate(token_ids=[2, 7], max_new_tokens=4)["text"] == "answer"
        stats = model.benchmark("hello", max_new_tokens=4, warmup=1, repeats=2)
        assert stats["decode_tokens_per_second"] == 100
        assert stats["generation_wall_tokens_per_second"] == 75
        assert stats["identical_generated_ids"]
        assert model.ready["pid"] == pid
    assert model._process.poll() == 0
    with pytest.raises(RuntimeError, match="closed"):
        model.generate("hello")
    model.close()


def test_request_error_does_not_poison_valid_session(fixture):
    with session(fixture) as model:
        with pytest.raises(ValueError, match="bad request"):
            model.generate("ERROR")
        assert model.generate("valid")["text"] == "answer"
        assert model.generate("STDERR")["text"] == "answer"


@pytest.mark.parametrize(
    "prompt,match",
    [
        ("MALFORMED", "Native backend failed"),
        ("PARTIAL", "timed out"),
        ("BAD_IDS", "invalid generated"),
    ],
)
def test_broken_protocol_timeout_or_ids_closes_owned_process(fixture, prompt, match):
    model = session(fixture, timeout=5)
    model.timeout = 0.2
    with pytest.raises(RuntimeError, match=match):
        model.generate(prompt)
    assert model._process.poll() is not None
    assert model._closed


@pytest.mark.parametrize(
    "arguments",
    [
        {"max_new_tokens": 0},
        {"max_new_tokens": True},
        {"token_ids": []},
        {"token_ids": [True]},
        {"token_ids": [100]},
        {"token_ids": [1] * 32},
        {"prompt": "hello", "token_ids": [1]},
        {"raw_prompt": 1},
    ],
)
def test_invalid_requests_before_transport(fixture, arguments):
    with session(fixture) as model:
        with pytest.raises(ValueError):
            model.generate(**arguments)
        assert model.generate("valid")["text"] == "answer"


@pytest.mark.parametrize(
    "options",
    [
        {"timeout": 0},
        {"timeout": float("nan")},
        {"draft_tokens": 16},
        {"speculative": "dflash2"},
        {"verify_tree": True},
        {"cuda_graph": 1},
    ],
)
def test_bad_startup_options_do_not_spawn(fixture, options):
    with pytest.raises(ValueError):
        session(fixture, **options)


@pytest.mark.parametrize(
    "kv,draft,heads", [("bf16", 15, 24), ("k8v4", 8, 24), ("k8v4", 15, 8)]
)
def test_tree_requirements_are_explicit_before_native_startup(
    fixture, kv, draft, heads
):
    path, runner = fixture
    manifest = NInferArtifact(path).manifest
    manifest["components"]["dflash2"] = {"config": {}, "target": "text"}
    manifest["components"]["text"]["config"].update(
        num_attention_heads=heads, head_dim=256
    )
    write_artifact(path.parent, manifest=manifest)
    with pytest.raises(ValueError, match="verify-tree route requires"):
        QwenNativeSession(
            path,
            runner=runner,
            max_context=32,
            prefill_chunk=16,
            speculative="dflash2",
            verify_tree=True,
            kv_dtype=kv,
            draft_tokens=draft,
        )


def test_cli_explicit_backend_and_json(fixture, capsys, monkeypatch):
    monkeypatch.delenv("VENTURI_QWEN_RUNNER", raising=False)
    artifact, runner = fixture
    assert main(["qwen", "inspect", str(artifact), "--json"]) == 0
    assert '"format": "ninfer-v3"' in capsys.readouterr().out
    assert main(["qwen", "generate", str(artifact), "--prompt", "hello"]) == 1
    assert "explicit --runner" in capsys.readouterr().err
    assert (
        main(
            [
                "qwen",
                "generate",
                str(artifact),
                "--runner",
                str(runner),
                "--max-context",
                "32",
                "--prefill-chunk",
                "16",
                "--prompt",
                "hello",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "answer"
    assert os.access(runner, os.X_OK)
