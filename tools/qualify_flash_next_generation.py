"""New generation entrypoint/direct target loading vs pinned independent text oracle.

Tiny SYNTHETIC complete checkpoints only, all four experts present. No actual
original full-model quality, TPS or performance comparison is inferred.
"""

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from qualify_flash_next_layer import references
from qualify_flash_next_ple import SuppliedRows
from qualify_flash_next_text import (
    text_references, make_text_oracle, make_cache, oracle_step, assert_state,
    UPSTREAM_SHA256, CACHE_SHA256, SDPA_SHA256,
)
from venturi.flash_next_generation import generate_flash_next_tokens
from venturi.models.flash_next_expert_policy import bind_fp8_expert_policy
from venturi.models.flash_next_loader import load_floating_text, load_weight_decoded_fp8_text
from venturi.models.flash_next_moe import TensorExpertStore
from venturi.models.flash_next_ngram import PreparedNGramTable


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream-modeling', required=True)
    p.add_argument('--upstream-cache', required=True)
    p.add_argument('--upstream-sdpa', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--library')
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    device = torch.device(a.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(1024*2**20/torch.cuda.get_device_properties(device).total_memory, device)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tests'))
    from test_flash_next_loader import fixture as floating_fixture, OPTIONS
    from test_flash_next_fp8_loader import fixture as fp8_fixture
    ref, qsa, flags, moe = references(a.upstream_modeling, a.upstream_cache, a.upstream_sdpa)
    text_references(a.upstream_modeling, ref)
    ref['OriginalNGram'] = ref['Qwen4ExpTextNGramEmbedding']
    cases = []
    dtypes = (torch.float32, torch.bfloat16) if device.type == 'cpu' else (torch.float32, torch.float16)
    for encoding in ('floating', 'fp8-weight-only'):
        for dtype in dtypes:
            root = out/(encoding+'-'+str(dtype).split('.')[-1])
            if encoding == 'floating':
                root.mkdir()
                checkpoint, independent, values = floating_fixture(root, dtype=torch.bfloat16)
                independent.to(device=device, dtype=dtype)
                # Fixture experts are NOT registered module parameters. Bind
                # their serialized BF16 precision before the execution cast,
                # just as an independently expanded checkpoint would do.
                for layer in independent.layers:
                    bank = layer.mlp.expert_store
                    layer.mlp.expert_store = TensorExpertStore(bank.geometry, {
                        e: tuple(v.bfloat16().to(device=device, dtype=dtype) for v in pair)
                        for e, pair in bank.experts.items()})
                loader = load_floating_text
            else:
                checkpoint, independent, values = fp8_fixture(root, execution_dtype=dtype)
                independent.to(device=device)
                loader = load_weight_decoded_fp8_text
            loaded = loader(checkpoint, **dict(OPTIONS, execution_device=str(device), execution_dtype=dtype))
            candidate, store = loaded.core, loaded.tables[1]
            for key, parameter in candidate.named_parameters():
                torch.testing.assert_close(parameter, dict(independent.named_parameters())[key], atol=0, rtol=0)
                assert parameter.device == device and parameter.dtype == dtype
            policy = None
            if encoding == 'fp8-weight-only':
                policy = bind_fp8_expert_policy(candidate, checkpoint,
                    backend='cuda-byte-decode' if device.type == 'cuda' else 'cpu-vectorized',
                    retained_bytes_per_layer=(2048, 2048), library=a.library)
            ref['Qwen4ExpTextNGramEmbedding'] = SuppliedRows
            oracle, head = make_text_oracle(ref, qsa, moe, independent, store.config, values,
                store.multipliers, device, dtype)
            for chunk in (1, 2, 16):
                prompt, expected, cache, visible, expected_steps = (2, 31, 5, 6, 7), [], make_cache(ref), None, []
                def step(ids):
                    nonlocal visible
                    tokens = torch.tensor([ids], device=device)
                    current = torch.ones_like(tokens, dtype=torch.bool)
                    visible = current if visible is None else torch.cat((visible, current), -1)
                    logits = oracle_step(oracle, head, cache, tokens, None, visible)
                    expected_steps.append(logits.clone())
                    return logits
                with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                    for offset in range(0, len(prompt), chunk):
                        logits = step(prompt[offset:offset+chunk])
                    expected.append(int(logits[0, -1].argmax()))
                    for _ in range(5):
                        logits = step([expected[-1]])
                        expected.append(int(logits[0, -1].argmax()))
                recorded = []
                def record(module, args, output):
                    recorded.append((output[0].clone(), output[1]))
                handle = candidate.register_forward_hook(record)
                try:
                    with PreparedNGramTable(store) as reader:
                        actual = generate_flash_next_tokens(candidate, reader, prompt,
                            max_new_tokens=6, eos_token_ids=(), prefill_chunk_size=chunk)
                finally:
                    handle.remove()
                assert actual.generated_ids == tuple(expected) and actual.committed_tokens == cache.seen
                assert len(recorded) == len(expected_steps)
                for (logits, _), expected_logits in zip(recorded, expected_steps):
                    torch.testing.assert_close(logits, expected_logits, atol=0, rtol=0)
                assert_state(recorded[-1][1], cache)
                cases.append(dict(encoding=encoding, dtype=str(dtype), device=str(device), chunk_size=chunk,
                    prompt_ids=prompt, generated_ids=actual.generated_ids, steps=len(recorded),
                    all_logits_exact=True, final_state_exact=True, direct_target_weights_exact=True,
                    expert_policy=policy, loading_plan=dict(loaded.report)))
    report = dict(scope=__doc__, torch=str(torch.__version__), cases=cases, reference_runtime_flags=flags,
        upstream_sha256=UPSTREAM_SHA256, upstream_cache_sha256=CACHE_SHA256, sdpa_sha256=SDPA_SHA256,
        actual_official_checkpoint_loaded=False, native_full_model=False, throughput_measured=False,
        quality_measured=False, activation_quantization=False,
        chunk_contract='Compare EACH explicit schedule with the same independent upstream schedule; no cross-schedule bit identity')
    (out/'qualification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
