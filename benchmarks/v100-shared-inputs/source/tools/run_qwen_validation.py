"""Sequential Qwen backend acceptance, graph ablation and speculative evaluation.

Existing local checkpoint/native builds only. The external backend owns Qwen
math and kernels. This tool does not claim those as new Venturi implementations.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

from venturi.backends.qwen import QwenNativeSession
from venturi.ninfer import NInferArtifact, ROW_FORMATS


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def idle_gpu():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    for line in output.splitlines():
        if (
            line.strip()
            and line.split(",")[1].strip() != "/usr/libexec/gnome-remote-desktop-daemon"
        ):
            raise RuntimeError(f"GPU is occupied; no workloads were killed: {line}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--runner", required=True)
    parser.add_argument("--baseline-cli", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retain and reuse completed runs in this output directory",
    )
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=args.resume)
    write(out / "background.json", {"compute_processes": idle_gpu()})
    artifact = NInferArtifact(args.checkpoint)
    if args.resume:
        previous = json.loads((out / "artifact.json").read_text())
        assert (
            previous["artifact_id"] == artifact.artifact_id
            and previous["payload_bytes"] == artifact.payload_bytes
        )
    else:
        write(out / "artifact.json", artifact.summary())
    samples = []
    for format in ROW_FORMATS:
        obj = next(o for o in artifact.objects.values() if o.get("format") == format)
        values = artifact.decode_row(obj["id"], 0)
        assert all(math.isfinite(v) for v in values)
        samples.append(
            {
                "object": obj["id"],
                "format": format,
                "row": 0,
                "columns": len(values),
                "min": min(values),
                "max": max(values),
                "prefix": values[:16],
            }
        )
    write(out / "quantized-row-samples.json", samples)
    options = dict(
        runner=args.runner,
        max_context=2048,
        prefill_chunk=128,
        kv_dtype="bf16",
        timeout=600,
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("NINFER_")}
    if not args.resume:
        chat_prompts = [
            "Explain why the sky looks blue. Give a detailed answer.",
            "Write a Python function to merge two sorted lists and explain its complexity.",
        ]
        # Independent product CLI uses the same external backend, not a separate
        # mathematical oracle. This checks prompt binding/transport/session isolation.
        expected = []
        for index, prompt in enumerate(chat_prompts):
            idle_gpu()
            command = [
                args.baseline_cli,
                args.checkpoint,
                "--prompt",
                prompt,
                "--max-new",
                "32",
                "--max-context",
                "2048",
                "--kv-capacity",
                "2048",
                "--prefill-chunk",
                "128",
                "--kv-dtype",
                "bf16",
                "--no-cuda-graph",
                "--greedy",
                "--no-thinking",
                "--raw-output",
                "--print-token-ids",
                "--presence-penalty",
                "0",
                "--frequency-penalty",
                "0",
                "--top-k",
                "0",
                "--top-p",
                "1",
                "--min-p",
                "0",
            ]
            env = {k: v for k, v in os.environ.items() if not k.startswith("NINFER_")}
            run = subprocess.run(
                command, capture_output=True, text=True, timeout=600, env=env
            )
            (out / f"baseline-{index}.stdout").write_text(run.stdout)
            (out / f"baseline-{index}.stderr").write_text(run.stderr)
            if run.returncode:
                raise RuntimeError(f"External baseline failed: {run.stderr[-4000:]}")
            found = re.search(
                r"^tokens\s+generated ids\s+([^\n]*)", run.stderr, re.MULTILINE
            )
            if not found:
                raise RuntimeError("Baseline CLI did not publish generated IDs")
            expected.append([int(t) for t in found.group(1).split()])
        write(out / "baseline-ids.json", expected)
        idle_gpu()
        with QwenNativeSession(args.checkpoint, cuda_graph=False, **options) as session:
            checks = []
            for prompt, ids in zip(chat_prompts, expected):
                actual = session.generate(prompt, max_new_tokens=32)
                assert actual["generated_ids"] == ids, "Wrapper/CLI greedy IDs differ"
                checks.append(actual)
            # Repeat the first after the second: recurrent/KV histories must reset.
            assert (
                session.generate(chat_prompts[0], max_new_tokens=32)["generated_ids"]
                == expected[0]
            )
            paragraph = "Inference engines read model weights, update recurrent state and attend to earlier tokens. "
            long_ids = session.tokenize(paragraph * 200)[:512]
            workloads = [
                {
                    "name": "prose",
                    "ids": session.tokenize(
                        "Write a detailed explanation of how rain forms:"
                    ),
                },
                {
                    "name": "code",
                    "ids": session.tokenize(
                        'def merge_sorted(a, b):\n    """Merge two sorted lists."""\n'
                    ),
                },
                {"name": "long-512", "ids": long_ids},
            ]
            assert len(long_ids) == 512
            write(
                out / "acceptance.json",
                {
                    "passed": True,
                    "cli_ids_equal": True,
                    "changed_prompt_reset": True,
                    "ready": session.ready,
                    "runs": checks,
                    "scope": "Same external implementation through independent product CLI vs Venturi transport. Not an FP32 numerical or model-quality qualification.",
                },
            )
        write(out / "workloads.json", workloads)
    else:
        assert json.loads((out / "acceptance.json").read_text())["passed"]
        workloads = json.loads((out / "workloads.json").read_text())
    modes = [
        ("eager-a", dict(cuda_graph=False)),
        ("graph-a", dict(cuda_graph=True)),
        ("graph-b", dict(cuda_graph=True)),
        ("eager-b", dict(cuda_graph=False)),
        ("graph-prefill512", dict(cuda_graph=True, prefill_chunk=512)),
        ("dflash-chain8", dict(cuda_graph=True, speculative="dflash2", draft_tokens=8)),
        ("k8v4-graph", dict(cuda_graph=True, kv_dtype="k8v4")),
        (
            "k8v4-chain8",
            dict(
                cuda_graph=True, kv_dtype="k8v4", speculative="dflash2", draft_tokens=8
            ),
        ),
        (
            "dflash-tree15",
            dict(
                cuda_graph=True,
                speculative="dflash2",
                draft_tokens=15,
                verify_tree=True,
                optimized_proposal=True,
                kv_dtype="k8v4",
            ),
        ),
    ]
    reports = {}
    reused_modes = []
    for name, extra in modes:
        previous = out / f"{name}.json"
        if args.resume and previous.exists():
            rows = json.loads(previous.read_text())
            assert [row["name"] for row in rows] == [w["name"] for w in workloads]
            assert all(len(row["runs"]) == args.repeats for row in rows)
            assert all(
                len(run["generated_ids"]) == args.max_new_tokens
                for row in rows
                for run in row["runs"]
            )
            for row in rows:
                for key, value in {**options, **extra}.items():
                    if key in row["ready"]["config"]:
                        assert row["ready"]["config"][key] == value
            reports[name] = rows
            reused_modes.append(name)
            print(f"Reusing preserved {name}", flush=True)
            continue
        idle_gpu()
        print(f"Running {name}", flush=True)
        with QwenNativeSession(args.checkpoint, **{**options, **extra}) as session:
            rows = []
            for workload in workloads:
                result = session.benchmark(
                    None,
                    token_ids=workload["ids"],
                    max_new_tokens=args.max_new_tokens,
                    warmup=1,
                    repeats=args.repeats,
                    ignore_eos=True,
                )
                assert all(
                    len(r["generated_ids"]) == args.max_new_tokens
                    for r in result["runs"]
                )
                rows.append({"name": workload["name"], **result})
                print(
                    name,
                    workload["name"],
                    result["decode_tokens_per_second"],
                    flush=True,
                )
            reports[name] = rows
            write(out / f"{name}.json", rows)
    comparisons = []
    for index, workload in enumerate(workloads):
        base_ids = reports["eager-a"][index]["runs"][0]["generated_ids"]
        for name, rows in reports.items():
            ids = rows[index]["runs"][0]["generated_ids"]
            common = next(
                (i for i, (a, b) in enumerate(zip(base_ids, ids)) if a != b),
                min(len(base_ids), len(ids)),
            )
            kv = rows[index]["ready"]["config"]["kv_dtype"]
            matched = reports["k8v4-graph" if kv == "k8v4" else "eager-a"][index][
                "runs"
            ][0]["generated_ids"]
            comparisons.append(
                {
                    "workload": workload["name"],
                    "mode": name,
                    "kv_dtype": kv,
                    "same_kv_greedy_ids": ids == matched,
                    "same_greedy_ids": base_ids == ids,
                    "common_prefix_tokens": common,
                    "decode_tokens_per_second": rows[index]["decode_tokens_per_second"],
                }
            )
    write(
        out / "comparison.json",
        {
            "comparisons": comparisons,
            "scope": "All modes use the same artifact and disabled research flags/prefix reuse. BF16 and K8V4 KV are separate precision comparisons. Speculative batch/state schedules can change floating-point rounding; differing IDs are not declared lossless.",
        },
    )
    if args.profile:
        for name, extra, workload, output_tokens in (
            ("eager", dict(cuda_graph=False), workloads[0], 32),
            ("graph", dict(cuda_graph=True), workloads[0], 32),
            ("prefill128", dict(cuda_graph=True), workloads[2], 1),
            ("prefill512", dict(cuda_graph=True, prefill_chunk=512), workloads[2], 1),
        ):
            prefix = out / f"nsys-{name}"
            if (
                args.resume
                and prefix.with_suffix(".nsys-rep").exists()
                and (out / f"nsys-{name}-stats.csv").exists()
            ):
                continue
            idle_gpu()
            config = dict(
                max_context=2048,
                prefill_chunk=128,
                kv_dtype="bf16",
                device=0,
                speculative="none",
            )
            config.update(extra)
            request = dict(
                op="generate",
                token_ids=workload["ids"],
                max_new_tokens=output_tokens,
                ignore_eos=True,
            )
            lines = [
                config,
                request,
                request,
                {**request, "profile": True},
                {"op": "close"},
            ]
            command = [
                "nsys",
                "profile",
                "--trace=cuda,nvtx",
                "--sample=none",
                "--cpuctxsw=none",
                "--cuda-graph-trace=node",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "-o",
                str(prefix),
                args.runner,
                args.checkpoint,
            ]
            with (
                (out / f"nsys-{name}.stdout").open("w") as stdout,
                (out / f"nsys-{name}.log").open("w") as stderr,
            ):
                subprocess.run(
                    command,
                    input="".join(json.dumps(r) + "\n" for r in lines),
                    text=True,
                    stdout=stdout,
                    stderr=stderr,
                    check=True,
                    timeout=600,
                    env=env,
                )
            stats = subprocess.run(
                [
                    "nsys",
                    "stats",
                    "--report",
                    "cuda_api_sum,cuda_gpu_kern_sum",
                    "--format",
                    "csv",
                    str(prefix) + ".nsys-rep",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
            (out / f"nsys-{name}-stats.csv").write_text(stats.stdout)
    sources = Path(__file__).resolve().parents[1] / "src"
    write(
        out / "completed.json",
        {
            "passed": True,
            "resumed_preserved_modes": reused_modes,
            "source_snapshot_scope": "Snapshot at completion; preserved runs retain their original first-attempt source snapshot and native build hashes.",
            "source_sha256": {
                str(p.relative_to(sources)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(sources.rglob("*.py"))
            },
            "benchmark_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "limitations": [
                "Native backend integration, not independent Qwen CUDA kernels.",
                "Same quantized backend CLI is the numerical reference; no original BF16 checkpoint qualification.",
                "Measured workloads are small, fixed-output and single-user; no universal TPS claim.",
            ],
        },
    )
    print("Qwen validation completed", flush=True)


if __name__ == "__main__":
    main()
