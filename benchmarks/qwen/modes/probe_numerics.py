"""Reproduce serial/batched output differences; neither route is a quality oracle."""

import argparse
import json
from pathlib import Path

from venturi.backends.qwen import QwenNativeSession


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--runner", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for kv in ("bf16", "k8v4"):
        with QwenNativeSession(
            args.checkpoint,
            runner=args.runner,
            max_context=512,
            max_concurrency=4,
            prefill_chunk=128,
            kv_dtype=kv,
            timeout=600,
        ) as model:
            prompts = [
                "The capital of France is",
                "def add(a, b):\n    return",
                "The sequence 2, 4, 6, 8 continues with",
                "Once upon a time in a quiet village",
            ] * 2
            chat = [
                "Reply only with YES.",
                "What is 2+2?",
                "Explain gravity in one sentence.",
            ]
            comparisons = []
            for name, inputs, opts in (
                (
                    "raw",
                    prompts,
                    dict(max_new_tokens=64, raw_prompt=True, ignore_eos=True),
                ),
                ("chat", chat, dict(max_new_tokens=32)),
            ):
                serial = model.generate_many(inputs, mode="single", **opts)
                parallel = model.generate_many(inputs, mode="parallel", **opts)
                serial_again = model.generate_many(inputs, mode="single", **opts)
                if [r["generated_ids"] for r in serial["results"]] != [
                    r["generated_ids"] for r in serial_again["results"]
                ]:
                    raise AssertionError("Serial state/reset regression")
                diffs = []
                for i, (a, b) in enumerate(zip(serial["results"], parallel["results"])):
                    x, y = a["generated_ids"], b["generated_ids"]
                    diffs.append(
                        dict(
                            index=i,
                            equal=x == y,
                            first_difference=next(
                                (
                                    j
                                    for j, pair in enumerate(zip(x, y))
                                    if pair[0] != pair[1]
                                ),
                                None if x == y else min(len(x), len(y)),
                            ),
                        )
                    )
                comparisons.append(
                    dict(
                        name=name,
                        prompts=inputs,
                        options=opts,
                        serial=serial,
                        parallel=parallel,
                        differences=diffs,
                        serial_reset_equal=True,
                    )
                )
            rows.append(dict(kv_dtype=kv, ready=model.ready, comparisons=comparisons))
    output.write_text(
        json.dumps(
            dict(
                cases=rows,
                scope="Output comparison only, not a logit-error or broad semantic-quality oracle. Complete IDs retain discovered counterexamples; serial replay tests state reset. No backend math was changed.",
            ),
            indent=2,
        )
        + "\n"
    )
    for row in rows:
        for case in row["comparisons"]:
            print(row["kv_dtype"], case["name"], case["differences"], flush=True)


if __name__ == "__main__":
    main()
