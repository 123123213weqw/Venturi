"""Validate DFlash2 transport against its native CLI and time EOS-respecting chat."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from run_qwen_validation import idle_gpu
from venturi.backends.qwen import QwenNativeSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--runner", required=True)
    parser.add_argument("--baseline-cli", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    prompts = [
        "Explain why the sky looks blue. Give a detailed answer.",
        "Write a Python function to merge two sorted lists and explain its complexity.",
    ]
    environment = {k: v for k, v in os.environ.items() if not k.startswith("NINFER_")}
    references = []
    for index, prompt in enumerate(prompts):
        idle_gpu()
        command = [
            args.baseline_cli,
            args.checkpoint,
            "--prompt",
            prompt,
            "--spec",
            "dflash2",
            "--draft-tokens",
            "8",
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
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=600, env=environment
        )
        (out / f"cli-{index}.stdout").write_text(result.stdout)
        (out / f"cli-{index}.stderr").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(result.stderr[-4000:])
        found = re.search(
            r"^tokens\s+generated ids\s+([^\n]*)", result.stderr, re.MULTILINE
        )
        if not found:
            raise RuntimeError("CLI generated IDs missing")
        references.append([int(t) for t in found.group(1).split()])
    records = {}
    for mode in ("none", "dflash2"):
        idle_gpu()
        with QwenNativeSession(
            args.checkpoint,
            runner=args.runner,
            max_context=2048,
            prefill_chunk=128,
            cuda_graph=True,
            kv_dtype="bf16",
            speculative=mode,
            draft_tokens=8,
            timeout=600,
        ) as session:
            if mode == "dflash2":
                for prompt, reference in zip(prompts, references):
                    actual = session.generate(prompt, max_new_tokens=32)
                    if actual["generated_ids"] != reference:
                        raise RuntimeError(
                            "DFlash2 wrapper differs from same-mode external CLI"
                        )
            records[mode] = [
                session.benchmark(
                    prompt, max_new_tokens=128, warmup=1, repeats=2, ignore_eos=False
                )
                for prompt in prompts
            ]
        (out / f"{mode}.json").write_text(json.dumps(records[mode], indent=2) + "\n")
    comparisons = []
    for index, prompt in enumerate(prompts):
        base = records["none"][index]
        spec = records["dflash2"][index]
        base_ids = base["runs"][0]["generated_ids"]
        spec_ids = spec["runs"][0]["generated_ids"]
        comparisons.append(
            {
                "prompt": prompt,
                "same_greedy_ids": base_ids == spec_ids,
                "plain_tokens_per_second": base["decode_tokens_per_second"],
                "dflash_tokens_per_second": spec["decode_tokens_per_second"],
                "plain_output_tokens": len(base_ids),
                "dflash_output_tokens": len(spec_ids),
                "scope": "Normal non-thinking chat, EOS respected, maximum 128 outputs. Native CLI parity is same speculative schedule, not an original-BF16 mathematical reference.",
            }
        )
    report = {
        "passed": True,
        "dflash_cli_ids_equal": True,
        "kv_dtype": "bf16",
        "comparisons": comparisons,
        "reference_ids": references,
    }
    (out / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
