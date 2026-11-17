"""Paired same-shape native model qualification; stream CPU snapshots, no TPS.

Two GPU instances run the same pinned artifact and schedule. The reference is
chunked FP16 dequant+cuBLAS, not an original-BF16/corpus oracle. No inference
weights/KV are offloaded to host; CPU copies exist only for validation.
"""

import argparse
import gc
import json
from pathlib import Path
import resource
import time
import torch
from qualify_quantlinear_model import digest, compare_states
from qualify_qwen_streaming import logits_comparison
from venturi.models import QwenWeights, NativeQwenModel
from venturi.models.qwen.native_runtime import QwenRuntime
from venturi.ops import CudaBackend
from venturi.quantization import codecs


def exact_state(a, b):
    for kind in ("gdn", "kv"):
        if kind not in a or kind not in b or len(a[kind]) != len(b[kind]):
            return False
        for left, right in zip(a[kind], b[kind]):
            if (left is None) != (right is None):
                return False
            if left is not None and (
                len(left) != len(right)
                or not all(torch.equal(x, y) for x, y in zip(left, right))
            ):
                return False
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--library", required=True)
    p.add_argument("--qwen-library", required=True)
    p.add_argument(
        "--reference-qwen-library",
        help="Frozen ABI-compatible serial runtime; defaults to candidate runtime",
    )
    p.add_argument("--cases", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--qualification-campaign", action="store_true")
    a = p.parse_args()
    output = Path(a.output)
    assert not output.exists(), "Never overwrite raw qualification"
    assert (
        torch.cuda.device_count() == 2
    ), "Explicit two-GPU correctness worker required"
    torch.set_num_threads(4)
    weights = QwenWeights.open(a.artifact)
    assert (weights.config["num_hidden_layers"], weights.config["hidden_size"]) == (
        64,
        5120,
    ), "Real 27B only, no small substitute"
    sha = digest(a.artifact)
    assert sha == a.expected_sha256
    data = json.loads(Path(a.cases).read_text())
    cases = data["cases"]
    backend = CudaBackend(a.library)
    arch = tuple(torch.cuda.get_device_capability())
    old = set(codecs.EXECUTION_QUALIFIED.get(arch, set()))
    if a.qualification_campaign:
        assert arch == (7, 0) and all(
            b.format in ("q5_g64_fp16", "bf16", "fp32") for b in weights.plan.bindings
        )
        codecs.EXECUTION_QUALIFIED[arch] = old | {"q5_g64_fp16"}
    report = dict(
        scope=__doc__,
        artifact_sha256=sha,
        artifact_bytes=Path(a.artifact).stat().st_size,
        ops_sha256=digest(a.library),
        qwen_sha256=digest(a.qwen_library),
        reference_qwen_sha256=digest(a.reference_qwen_library or a.qwen_library),
        architecture=arch,
        torch_version=torch.__version__,
        model_dimensions={
            k: weights.config[k]
            for k in ("hidden_size", "num_hidden_layers", "vocab_size")
        },
        original_bf16_or_corpus_quality_qualified=False,
        tokenizer_sha256=data["tokenizer_sha256"],
        logit_gates=dict(
            top1_equal=True,
            relative_l2_max=0.02,
            centered_cosine_min=0.999,
            kl_max=0.002,
        ),
        state_gates=dict(rtol=0.01, atol=0.005),
        implementations={},
        batches=[],
        completed=False,
        passed=False,
    )

    def save():
        temp = output.with_suffix(".partial")
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(output)

    save()
    models = []
    began = time.perf_counter()
    cpu = time.process_time()
    try:
        for ordinal, implementation in enumerate(("dequant", "auto")):
            t = time.perf_counter()
            models.append(
                NativeQwenModel.from_q8(
                    weights,
                    QwenRuntime(
                        (
                            (a.reference_qwen_library or a.qwen_library)
                            if ordinal == 0
                            else a.qwen_library
                        ),
                        backend,
                    ),
                    weight_mode="resident",
                    dtype=torch.float16,
                    device=f"cuda:{ordinal}",
                    max_tokens=32,
                    reserve_bytes=512 << 20,
                    metadata_mode="sealed",
                    linear_implementation=implementation,
                )
            )
            info = models[-1].projection_info
            assert (
                info["linear_contracts"]["q5_g64_fp16"]["selected_family"]
                == "chunked-dense"
            ), "Unqualified fusion must not enter the safe campaign"
            report["implementations"][implementation] = dict(
                device=ordinal,
                projection_info=info,
                load_wall_seconds=time.perf_counter() - t,
                cases=[],
            )
            save()
        for case in cases:
            sessions = [m.create_session(256) for m in models]
            records = []
            for implementation in ("dequant", "auto"):
                entry = dict(name=case["name"], generated_ids=[], logits=[], states=[])
                report["implementations"][implementation]["cases"].append(entry)
                records.append(entry)
            try:
                final_states = None
                for step in range(case["max_new_tokens"]):
                    ids = (
                        case["prompt_token_ids"]
                        if step == 0
                        else [records[0]["generated_ids"][-1]]
                    )
                    values = [s.forward(ids) for s in sessions]
                    for entry, logits in zip(records, values):
                        entry["generated_ids"].append(int(logits.argmax()))
                    check = logits_comparison(values[1], values[0])
                    records[1]["logits"].append(check)
                    save()
                    assert check["passed"], "Matched logit/top1 gate failed"
                    if step in (0, case["max_new_tokens"] - 1):
                        states = [s.snapshot() for s in sessions]
                        checks = compare_states(states[1], states[0])
                        records[1]["states"].append(
                            dict(
                                step=step,
                                components=checks,
                                bytes_equal=exact_state(*states),
                            )
                        )
                        save()
                        assert all(
                            c["passed"] for c in checks
                        ), "Matched private-state gate failed"
                        if step == case["max_new_tokens"] - 1:
                            final_states = states
                assert records[0]["generated_ids"] == records[1]["generated_ids"]
                for i, s in enumerate(sessions):
                    s.reset()
                    tokens = s.generate(
                        case["prompt_token_ids"],
                        max_new_tokens=case["max_new_tokens"],
                        eos_token_ids=[],
                    )
                    assert (
                        tokens == records[i]["generated_ids"]
                    ), "Native generation changed qualified greedy trajectory"
                    records[i]["native_generate_state_bytes_equal"] = exact_state(
                        s.snapshot(), final_states[i]
                    )
                    assert records[i]["native_generate_state_bytes_equal"]
                save()
                print(
                    json.dumps(
                        dict(stage="paired_case", name=case["name"], passed=True)
                    ),
                    flush=True,
                )
            finally:
                for s in sessions:
                    s.close()
        batches = [m.create_batch([256] * 32) for m in models]
        try:
            for count in (4, 8, 16, 32):
                slots = list(range(count))
                indices = [i % len(cases) for i in slots]
                for b in batches:
                    b.reset(slots=slots)
                values = [
                    b.generate(
                        [cases[i]["prompt_token_ids"] for i in indices],
                        max_new_tokens=[cases[i]["max_new_tokens"] for i in indices],
                        slots=slots,
                        eos_token_ids=[],
                    )
                    for b in batches
                ]
                expected = [
                    report["implementations"]["dequant"]["cases"][i]["generated_ids"]
                    for i in indices
                ]
                entry = dict(
                    concurrency=count,
                    tokens_equal=values[0] == values[1] == expected,
                    output_ids=values[1],
                    state_components=0,
                    state_failures=[],
                    state_bytes_equal=True,
                )
                report["batches"].append(entry)
                save()
                assert entry["tokens_equal"], "Grouped greedy output changed"
                for slot in slots:
                    states = [b.snapshot(slot) for b in batches]
                    checks = compare_states(states[1], states[0])
                    entry["state_components"] += len(checks)
                    entry["state_failures"].extend(
                        dict(slot=slot, **c) for c in checks if not c["passed"]
                    )
                    entry["state_bytes_equal"] &= exact_state(*states)
                save()
                assert not entry["state_failures"], "Matched grouped state gate failed"
                print(
                    json.dumps(dict(stage="paired_batch", count=count, passed=True)),
                    flush=True,
                )
        finally:
            for b in batches:
                b.close()
        weights.source.check_source()
        report.update(passed=True, completed=True)
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        for m in models:
            m.close()
        codecs.EXECUTION_QUALIFIED[arch] = old
        gc.collect()
        report.update(
            wall_seconds=time.perf_counter() - began,
            cpu_seconds=time.process_time() - cpu,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        save()


if __name__ == "__main__":
    main()
