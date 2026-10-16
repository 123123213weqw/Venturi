"""Strict real-checkpoint eager/graph parity; no tolerance relaxation or downloads."""

import argparse
import json
from pathlib import Path

import torch

from venturi import load_checkpoint
from venturi.generation import LocalTokenizer, generate_tokens
from venturi.models import GPTNeoXModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    model = GPTNeoXModel(load_checkpoint(args.checkpoint, require_float_dtype="F16"))
    tokenizer = LocalTokenizer(
        Path(args.checkpoint) / "tokenizer.json", vocab_size=model.config.vocab_size
    )
    records = []
    for prompt, budget in (("Once upon a time", 32), ("The capital of France is", 128)):
        ids = tokenizer.encode(prompt)
        graph = model.create_cuda_graph_session(
            len(ids) + budget, decode_start=len(ids), decode_steps=budget - 1
        )
        eager = model.create_session(len(ids) + budget)
        expected = eager.forward(ids)
        actual = graph.forward(ids)
        if not torch.equal(expected.view(torch.int16), actual.view(torch.int16)):
            raise RuntimeError("Eager prefill parity failed")
        teacher = []
        for _ in range(budget - 1):
            token = int(expected.argmax())
            teacher.append(token)
            expected = eager.forward([token])
            actual = graph.forward([token])
            if not torch.equal(expected.view(torch.int16), actual.view(torch.int16)):
                raise RuntimeError("Captured logits are not bit-exact with eager logits")
            if graph.select_token(actual) != int(expected.argmax()):
                raise RuntimeError("Captured greedy selection differs")
        expected_result = generate_tokens(model, ids, max_new_tokens=budget, eos_token_ids=())
        actual_result = generate_tokens(
            model, ids, max_new_tokens=budget, eos_token_ids=(), session=graph
        )
        if actual_result.generated_ids != expected_result.generated_ids:
            raise RuntimeError("Captured greedy sequence differs")
        # Different prompt contents with the same shape must not see old KV data.
        probe = tuple((token + 17) % model.config.vocab_size for token in ids)
        a = generate_tokens(model, probe, max_new_tokens=4, eos_token_ids=())
        b = generate_tokens(model, probe, max_new_tokens=4, eos_token_ids=(), session=graph)
        if a.generated_ids != b.generated_ids:
            raise RuntimeError("Prepared request history leaked")
        records.append({
            "prompt": prompt, "prompt_tokens": len(ids), "max_new_tokens": budget,
            "bit_exact_teacher_rows": budget, "bit_exact_greedy_ids": True,
            "changed_prompt_isolated": True, "preparation_seconds": graph.preparation_seconds,
            "graph_count": graph.decode_steps, "generated_ids": actual_result.generated_ids,
        })
        del graph, eager
    report = {
        "passed": True, "runs": records,
        "scope": "Same stored FP16 weights, same-position/same-shape eager and graph paths, complete logit vectors bit-exact on tested real checkpoint. Not a claim of agreement with FP32/llama.cpp, nor arbitrary models/drivers/contexts.",
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
