"""Resident-process contracts tested with a CPU-only fake native runner."""

import json
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
print(json.dumps(dict(ok=True,event="ready",protocol=3,cpu_wait=config["cpu_wait"],
                     limits=dict(max_concurrency=config["max_concurrency"],
                                 kv_capacity=((config["kv_capacity"]+63)//64)*64),pid=os.getpid(),config=config,
                     inherited_flags=[k for k in os.environ if k.startswith("NINFER_")])) ,flush=True)
def generation(r):
 prompt=r.get("prompt", "")
 ids=[-1] if prompt=="BAD_IDS" else list(range(1,r["max_new_tokens"]+1))
 return dict(ok=True,generated_ids=ids,text="answer",input=prompt,reused_prompt_tokens=0,
  timings=dict(first_token_seconds=.01,prefill_seconds=.01,decode_seconds=.03,
               generation_wall_seconds=.04,total_seconds=.05))
for line in sys.stdin:
 r=json.loads(line)
 if r.get("op")=="tokenize":
  print(json.dumps(dict(ok=True,token_ids=[1,2,3])),flush=True)
  continue
 if r.get("op")=="generate_many":
  results=[dict(generation(item),index=i) for i,item in enumerate(r["requests"])]
  if r["requests"][0]["prompt"]=="BAD_BATCH": results.reverse()
  print(json.dumps(dict(ok=True,mode=r["mode"],results=results,wall_seconds=.1)),flush=True)
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
 print(json.dumps(generation(r)),flush=True)
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
        {"cpu_wait": "spin"},
        {"cpu_wait": True},
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


def test_switch_modes_on_same_resident_model(fixture):
    with session(fixture, max_concurrency=4) as model:
        pid = model.ready["pid"]
        assert model.mode == "single"
        assert model.config["kv_capacity"] == 256  # Four independent 64-token pages.
        prompts = ["first", "second", "first"]
        serial = model.generate_many(prompts, max_new_tokens=4)
        assert serial["mode"] == "single"
        assert [r["input"] for r in serial["results"]] == prompts
        model.set_mode("parallel")
        parallel = model.generate_many(prompts, max_new_tokens=4)
        assert parallel["mode"] == "parallel"
        assert [r["generated_ids"] for r in parallel["results"]] == [
            r["generated_ids"] for r in serial["results"]
        ]
        assert parallel["output_tokens_per_second"] == 120
        assert (
            model.generate_many(prompts, mode="single", max_new_tokens=4)["mode"]
            == "single"
        )
        assert (
            model.mode == "parallel"
        )  # Per-call override does not mutate the session.
        assert model.generate("one", max_new_tokens=4)["text"] == "answer"
        model.set_mode("single")
        assert model.ready["pid"] == pid
        stats = model.benchmark_many(
            prompts, mode="parallel", warmup=1, repeats=2, max_new_tokens=4
        )
        assert stats["output_tokens_per_second"] == 120
        assert stats["identical_generated_ids"]
        assert (
            "decode_tokens_per_second" not in stats
        )  # Overlapping request times cannot be summed.


def test_parallel_default_and_shared_page_budget(fixture):
    with session(fixture, mode="parallel") as model:
        assert model.config["max_concurrency"] == 4
    with session(fixture, mode="parallel", max_concurrency=2, kv_capacity=65) as model:
        assert model.ready["limits"]["kv_capacity"] == 128
    with session(fixture) as model:
        with pytest.raises(ValueError, match="reserved at startup"):
            model.set_mode("parallel")
        with pytest.raises(ValueError, match="reserved at startup"):
            model.generate_many(["hello"], mode="parallel")


@pytest.mark.parametrize(
    "options",
    [
        {"mode": "automatic"},
        {"mode": "parallel", "max_concurrency": 1},
        {"max_concurrency": 0},
        {"max_concurrency": 9},
        {"max_concurrency": True},
        {"kv_capacity": 0},
        {"kv_capacity": True},
        {"kv_capacity": 65},
        {"max_concurrency": 4, "kv_capacity": 128},  # Only two pages for four lanes.
    ],
)
def test_invalid_capacity_or_mode_does_not_spawn(fixture, options, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Invalid configuration must be rejected before spawning")

    monkeypatch.setattr("venturi.backends.qwen.subprocess.Popen", fail)
    with pytest.raises(ValueError):
        session(fixture, **options)


@pytest.mark.parametrize("prompts", [[], "one", ["ok", 1], ["x"] * 17])
def test_invalid_batch_is_rejected_before_transport(fixture, prompts, monkeypatch):
    with session(fixture, mode="parallel") as model:

        def fail(request):
            pytest.fail("Invalid batch must not be sent")

        monkeypatch.setattr(model, "_exchange", fail)
        with pytest.raises(ValueError):
            model.generate_many(prompts)


def test_bad_batch_order_closes_process(fixture):
    with session(fixture, mode="parallel") as model:
        with pytest.raises(RuntimeError, match="batch result/order"):
            model.generate_many(["BAD_BATCH", "other"])
        assert model._process.poll() is not None


def test_reject_old_runner_instead_of_silently_serializing(fixture):
    _, runner = fixture
    runner.write_text(runner.read_text().replace("protocol=3", "protocol=2"))
    with pytest.raises(RuntimeError, match="protocol-3"):
        session(fixture)


def test_reject_runner_that_ignores_capacity(fixture):
    _, runner = fixture
    runner.write_text(
        runner.read_text().replace(
            'max_concurrency=config["max_concurrency"]', "max_concurrency=1"
        )
    )
    with pytest.raises(RuntimeError, match="did not reserve"):
        session(fixture, mode="parallel")


def test_mode_change_is_idle_only(fixture):
    with session(fixture, max_concurrency=4) as model:
        model._lock.acquire()
        try:
            with pytest.raises(RuntimeError, match="active request"):
                model.set_mode("parallel")
            with pytest.raises(RuntimeError, match="sequential request"):
                model.generate_many(["hello"])
        finally:
            model._lock.release()
        assert model.mode == "single"
    with pytest.raises(RuntimeError, match="closed"):
        model.set_mode("single")


def test_cli_multiple_prompts_and_json_file(fixture, capsys, tmp_path):
    artifact, runner = fixture
    args = [
        "qwen",
        "generate",
        str(artifact),
        "--runner",
        str(runner),
        "--max-context",
        "32",
        "--prefill-chunk",
        "16",
        "--max-new-tokens",
        "4",
    ]
    assert (
        main(
            args
            + ["--mode", "parallel", "--prompt", "one", "--prompt", "two", "--json"]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["batch"]["mode"] == "parallel"
    assert [r["input"] for r in report["batch"]["results"]] == ["one", "two"]
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(["three", "four"]))
    assert main(args + ["--prompts-file", str(path)]) == 0
    assert capsys.readouterr().out.strip() == "[0] answer\n[1] answer"
    path.write_text("{broken")
    assert main(args + ["--prompts-file", str(path)]) == 1
    assert "venturi qwen:" in capsys.readouterr().err
    args[1] = "benchmark"
    assert (
        main(
            args
            + [
                "--mode",
                "parallel",
                "--prompt",
                "one",
                "--prompt",
                "two",
                "--warmup",
                "0",
                "--repeats",
                "1",
            ]
        )
        == 0
    )
    assert "80.000 tokens/s (batch wall time)" in capsys.readouterr().out


@pytest.mark.parametrize("bad", ["BAD_IDS", "BAD_BATCH"])
def test_bad_batch_payload_closes_session(fixture, bad):
    with session(fixture, mode="parallel") as model:
        with pytest.raises(RuntimeError, match="invalid generated|batch result/order"):
            model.generate_many([bad, "valid"], max_new_tokens=4)
        assert model._closed


def test_batch_validates_generation_options_before_transport(fixture, monkeypatch):
    with session(fixture, mode="parallel") as model:

        def fail(request):
            pytest.fail("Invalid generation options must not reach transport")

        monkeypatch.setattr(model, "_exchange", fail)
        for options in ({"max_new_tokens": 0}, {"raw_prompt": 1}, {"token_ids": [1]}):
            with pytest.raises(ValueError):
                model.generate_many(["valid", "other"], **options)


@pytest.mark.parametrize("policy", ["auto", "blocking"])
def test_explicit_cpu_wait_policy(fixture, policy):
    with session(fixture, cpu_wait=policy) as model:
        assert model.config["cpu_wait"] == model.ready["cpu_wait"] == policy
        assert model.generate("hello", max_new_tokens=4)["text"] == "answer"


def test_reject_runner_ignoring_cpu_policy(fixture):
    _, runner = fixture
    runner.write_text(
        runner.read_text().replace('cpu_wait=config["cpu_wait"]', 'cpu_wait="auto"')
    )
    with pytest.raises(RuntimeError, match="did not apply"):
        session(fixture, cpu_wait="blocking")


@pytest.mark.parametrize("policy", ["auto", "blocking"])
def test_cli_cpu_wait_is_forwarded(fixture, capsys, policy):
    artifact, runner = fixture
    assert (
        main(
            [
                "qwen",
                "generate",
                str(artifact),
                "--runner",
                str(runner),
                "--prompt",
                "hello",
                "--max-context",
                "32",
                "--prefill-chunk",
                "16",
                "--max-new-tokens",
                "4",
                "--cpu-wait",
                policy,
                "--json",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["backend"]["cpu_wait"] == policy
