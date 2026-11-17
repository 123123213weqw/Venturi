"""Same-artifact row QuantLinear model/state qualification, never a TPS run.

The bounded chunked decode+cuBLAS reference avoids a full FP16 resident shadow.
It is independent of fused projection kernels, not an upstream BF16/corpus oracle.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import time
import torch
from qualify_qwen_streaming import logits_comparison, metrics
from venturi.models import NativeQwenModel, QwenWeights
from venturi.models.qwen.native_runtime import QwenRuntime
from venturi.ops import CudaBackend
from venturi.quantization import codecs


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compare_states(actual, reference):
    out = []
    for kind in ("gdn", "kv"):
        assert len(actual[kind]) == len(reference[kind])
        for layer, (a, b) in enumerate(zip(actual[kind], reference[kind])):
            assert (a is None) == (b is None)
            if a is None:
                continue
            assert len(a) == len(b)
            for index, (x, y) in enumerate(zip(a, b)):
                out.append(
                    dict(
                        kind=kind,
                        layer=layer,
                        component=index,
                        passed=torch.allclose(x, y, rtol=1e-2, atol=5e-3),
                        **metrics(x, y),
                    )
                )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--library", required=True)
    p.add_argument("--qwen-library", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--qualification-campaign", action="store_true")
    p.add_argument("--expect-layers", type=int, default=64)
    p.add_argument("--expect-hidden", type=int, default=5120)
    a = p.parse_args()
    out = Path(a.output)
    assert not out.exists(), "Never overwrite qualification evidence"
    torch.set_num_threads(4)
    weights = QwenWeights.open(a.artifact)
    stamp = weights.source.stamp
    assert (
        weights.config["num_hidden_layers"] == a.expect_layers
        and weights.config["hidden_size"] == a.expect_hidden
    )
    actual_sha = digest(a.artifact)
    assert actual_sha == a.expected_sha256, "Artifact hash mismatch"
    data = json.loads(Path(a.cases).read_text())
    cases = data["cases"]
    backend = CudaBackend(a.library)
    arch = tuple(torch.cuda.get_device_capability())
    report = dict(
        completed=False,
        passed=False,
        scope=__doc__,
        artifact_sha256=actual_sha,
        artifact_bytes=Path(a.artifact).stat().st_size,
        ops_sha256=digest(a.library),
        qwen_sha256=digest(a.qwen_library),
        architecture=arch,
        torch_version=torch.__version__,
        model_dimensions={
            k: weights.config[k]
            for k in ("num_hidden_layers", "hidden_size", "vocab_size")
        },
        tokenizer_sha256=data["tokenizer_sha256"],
        qualification_override=a.qualification_campaign,
        original_bf16_or_corpus_quality_qualified=False,
        logit_gates=dict(
            top1_equal=True,
            relative_l2_max=0.02,
            centered_cosine_min=0.999,
            kl_max=0.002,
        ),
        state_gates=dict(rtol=0.01, atol=0.005),
        implementations={},
    )

    def save():
        tmp = out.with_suffix(".partial")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(out)

    save()
    reference = []
    old = set(codecs.EXECUTION_QUALIFIED.get(arch, set()))
    if a.qualification_campaign:
        assert arch == (7, 0) and all(
            b.format in ("q5_g64_fp16", "bf16", "fp32") for b in weights.plan.bindings
        )
        codecs.EXECUTION_QUALIFIED[arch] = old | {"q5_g64_fp16"}
    began = time.perf_counter()
    cpu = time.process_time()
    try:
        for implementation in ("dequant", "auto"):
            load = time.perf_counter()
            model = NativeQwenModel.from_q8(
                weights,
                QwenRuntime(a.qwen_library, backend),
                weight_mode="resident",
                dtype=torch.float16,
                max_tokens=32,
                reserve_bytes=512 << 20,
                metadata_mode="sealed",
                linear_implementation=implementation,
            )
            record = dict(
                load_wall_seconds=time.perf_counter() - load,
                projection_info=model.projection_info,
                cases=[],
            )
            report["implementations"][implementation] = record
            save()
            try:
                for index, case in enumerate(cases):
                    expected = reference[index] if implementation == "auto" else None
                    session = model.create_session(256)
                    saved = dict(logits=[], states={}, ids=[])
                    item = dict(
                        name=case["name"],
                        prompt_token_ids=case["prompt_token_ids"],
                        generated_ids=[],
                        logits=[],
                        states=[],
                    )
                    record["cases"].append(item)
                    save()
                    try:
                        for step in range(case["max_new_tokens"]):
                            ids = (
                                case["prompt_token_ids"]
                                if step == 0
                                else [(expected or saved)["ids"][step - 1]]
                            )
                            logits = session.forward(ids)
                            token = int(logits.argmax())
                            saved["ids"].append(token)
                            item["generated_ids"].append(token)
                            if expected:
                                check = logits_comparison(
                                    logits, expected["logits"][step]
                                )
                                item["logits"].append(check)
                                save()
                                assert check[
                                    "passed"
                                ], "Unchanged same-artifact logit/top1 gate failed"
                            else:
                                saved["logits"].append(logits)
                            if step in (0, case["max_new_tokens"] - 1):
                                state = session.snapshot()
                                saved["states"][step] = state
                                if expected:
                                    checks = compare_states(
                                        state, expected["states"][step]
                                    )
                                    item["states"].append(
                                        dict(step=step, components=checks)
                                    )
                                    save()
                                    assert all(
                                        c["passed"] for c in checks
                                    ), "Same-artifact private state gate failed"
                                else:
                                    saved["states"][step] = state
                        session.reset()
                        generated = session.generate(
                            case["prompt_token_ids"],
                            max_new_tokens=case["max_new_tokens"],
                            eos_token_ids=[],
                        )
                        assert (
                            generated == saved["ids"]
                        ), "Native generation differs from diagnostic trajectory"
                        if expected:
                            assert (
                                generated == expected["ids"]
                            ), "Autoregressive token trajectory differs"
                        generated_state = session.snapshot()
                        teacher_state = saved["states"][case["max_new_tokens"] - 1]
                        item["native_generate_state_bytes_equal"] = all(
                            torch.equal(x, y)
                            for kind in ("gdn", "kv")
                            for left, right in zip(
                                generated_state[kind], teacher_state[kind]
                            )
                            if left is not None
                            for x, y in zip(left, right)
                        )
                        assert item[
                            "native_generate_state_bytes_equal"
                        ], "Native generate changed same-implementation state bytes"
                        item["native_generate_equal"] = True
                        if not expected:
                            reference.append(saved)
                        item["session_info"] = session.info()
                        save()
                    finally:
                        session.close()
                    print(
                        json.dumps(
                            dict(stage=implementation, case=case["name"], passed=True)
                        ),
                        flush=True,
                    )
                if implementation == "auto":
                    batch = model.create_batch([256] * 32)
                    record["batches"] = []
                    try:
                        for count in (4, 8, 16, 32):
                            slots = list(range(count))
                            batch.reset(slots=slots)
                            selected = [i % len(cases) for i in slots]
                            outputs = batch.generate(
                                [cases[i]["prompt_token_ids"] for i in selected],
                                max_new_tokens=[
                                    cases[i]["max_new_tokens"] for i in selected
                                ],
                                slots=slots,
                                eos_token_ids=[],
                            )
                            expected_outputs = [reference[i]["ids"] for i in selected]
                            entry = dict(
                                concurrency=count,
                                tokens_equal=outputs == expected_outputs,
                                output_ids=outputs,
                                state_components=0,
                                state_failures=[],
                            )
                            record["batches"].append(entry)
                            save()
                            assert entry[
                                "tokens_equal"
                            ], "Grouped generation changed reference greedy trajectory"
                            for slot, index in zip(slots, selected):
                                checks = compare_states(
                                    batch.snapshot(slot),
                                    reference[index]["states"][
                                        cases[index]["max_new_tokens"] - 1
                                    ],
                                )
                                entry["state_components"] += len(checks)
                                entry["state_failures"].extend(
                                    dict(slot=slot, **c)
                                    for c in checks
                                    if not c["passed"]
                                )
                            save()
                            assert not entry[
                                "state_failures"
                            ], "Grouped private-state gate failed"
                            print(
                                json.dumps(
                                    dict(stage="batch", count=count, passed=True)
                                ),
                                flush=True,
                            )
                    finally:
                        batch.close()
            finally:
                model.close()
                del model
                gc.collect()
                torch.cuda.empty_cache()
        weights.source.check_source()
        assert weights.source.stamp == stamp
        report.update(completed=True, passed=True)
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        codecs.EXECUTION_QUALIFIED[arch] = old
        report.update(
            wall_seconds=time.perf_counter() - began,
            cpu_seconds=time.process_time() - cpu,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        save()


if __name__ == "__main__":
    main()
