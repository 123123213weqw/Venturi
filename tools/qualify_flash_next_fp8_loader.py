"""Explicit FP8-weight bridge -> floating original text oracle, tiny synthetic checkpoint.

Synthetic weights only. No original full model, native kernel, TPS or quality.
"""

import argparse
import json
from pathlib import Path
import sys

import torch

from qualify_flash_next_layer import references
from qualify_flash_next_ple import SuppliedRows
from qualify_flash_next_text import (
    text_references, make_text_oracle, run_case, request_trajectory, partial_trajectory,
    UPSTREAM_SHA256, CACHE_SHA256, SDPA_SHA256,
)
from venturi.models.flash_next_loader import load_weight_decoded_fp8_text
from venturi.models.flash_next_ngram import PreparedNGramTable


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream-modeling', required=True)
    p.add_argument('--upstream-cache', required=True)
    p.add_argument('--upstream-sdpa', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    device = torch.device(a.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(1024*2**20/torch.cuda.get_device_properties(device).total_memory, device)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tests'))
    from test_flash_next_fp8_loader import fixture
    from test_flash_next_loader import OPTIONS
    ref, qsa, flags, moe = references(a.upstream_modeling, a.upstream_cache, a.upstream_sdpa)
    text_references(a.upstream_modeling, ref)
    original_ngram = ref['Qwen4ExpTextNGramEmbedding']
    cases, trajectories, partial, bindings = [], [], [], []
    dtypes = (torch.float32, torch.bfloat16) if device.type == 'cpu' else (torch.float32, torch.float16)
    for dtype in dtypes:
        checkpoint, independent, table = fixture(out/('tiny-synthetic-'+str(dtype).split('.')[-1]), execution_dtype=dtype)
        loaded = load_weight_decoded_fp8_text(checkpoint, **OPTIONS)
        candidate = loaded.core.to(device=device, dtype=dtype)
        independent.to(device=device)
        for key, parameter in candidate.named_parameters():
            torch.testing.assert_close(parameter, dict(independent.named_parameters())[key], atol=0, rtol=0)
        store = loaded.tables[1]
        ref['Qwen4ExpTextNGramEmbedding'] = SuppliedRows
        ref['OriginalNGram'] = original_ngram
        oracle, head = make_text_oracle(ref, qsa, moe, independent, store.config, table, store.multipliers, device, dtype)
        counts = [dict() for _ in candidate.layers]
        for i, layer in enumerate(candidate.layers):
            original = layer.mlp.expert_store.read
            def counted(expert, *, device, dtype, original=original, counter=counts[i]):
                counter[expert] = counter.get(expert, 0)+1
                return original(expert, device=device, dtype=dtype)
            layer.mlp.expert_store.read = counted
        with PreparedNGramTable(store).prepare() as reader:
            for b, n, cached in ((1, 1, False), (2, 5, False), (8, 1, False), (1, 1, True), (2, 3, True), (8, 1, True)):
                for masking in (False, True):
                    cases.extend(run_case(candidate, oracle, head, ref, reader, b, n, cached, masking, device, dtype))
            trajectories.extend(request_trajectory(candidate, oracle, head, ref, store, device, dtype))
            partial.extend(partial_trajectory(candidate, oracle, head, ref, store, device, dtype))
        bindings.append(dict(report=dict(loaded.report), expert_read_calls=counts, static_weights_exact=True,
                             source_dtype=str(dtype), execution_device=str(device)))
    report = dict(scope='Tiny synthetic block-FP8 expert/global-FP8 PLE checkpoint, CPU weight decode to explicit floating text reference',
        cases=cases, greedy_trajectory_steps=trajectories, partial_acceptance_steps=partial, bindings=bindings,
        torch=str(torch.__version__), reference_runtime_flags=flags,
        upstream_sha256=UPSTREAM_SHA256, upstream_cache_sha256=CACHE_SHA256, sdpa_sha256=SDPA_SHA256,
        actual_official_checkpoint_loaded=False, whole_official_model_tested=False, native_kernel_implemented=False,
        throughput_measured=False, quality_measured=False,
        state_contract='Request-private hash+PLE/GDN/QSA state commit; partial prefix replay from base; explicit full reset',
        expert_scope='All4 synthetic experts present, no pruning; selected CPU FP32 dequant products -> chosen floating dtype',
        activation_quantization=False, fp8_gemm=False, w8a8_engine_equivalence=False,
        ple_scope='Official separate global-scale CPU BF16 decode contract; full tiny table only')
    (out/'qualification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
