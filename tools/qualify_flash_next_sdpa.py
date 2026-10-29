"""Explicit QSA SDPA-math oracle and empty-row gate, not a speed/default switch.

Runs pinned upstream SDPA integration, main attention and QSA indexer. Actual
block weights reuse the verified original BF16 sample, not a complete model.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from safetensors.torch import load_file
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from qualify_flash_next_attention import attention_reference, cases
from qualify_flash_next_qsa import INDEX_ABI_SHIM, REVISION, UPSTREAM_SHA256, positions
from venturi.models.flash_next_attention import FlashNextQSAAttention, QSAAttentionGeometry, QSAState
from venturi.models.flash_next_qsa import QSAIndexerGeometry
from qualify_flash_next_attention import Cache


SDPA_SHA256 = "ba7f7c4e1ac7a0ec457e5c3035e54d0e0183e5bd20f938784814774c7724a616"


def reference(modeling, sdpa_path):
    attention = attention_reference(modeling)
    scope = attention.__init__.__globals__
    data = Path(sdpa_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != SDPA_SHA256:
        raise ValueError("Pinned SDPA integration source hash differs")
    names = {"repeat_kv", "use_gqa_in_sdpa", "create_position_bias_mask", "sdpa_attention_forward"}
    selected = [n for n in ast.parse(data).body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != names:
        raise ValueError("Pinned SDPA helper definitions missing")
    version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    flags = dict(_is_torch_greater_or_equal_than_2_8=version >= (2, 8),
                 _is_torch_greater_or_equal_than_2_13=version >= (2, 13),
                 _is_torch_mps_available=torch.backends.mps.is_available(),
                 _is_torch_xpu_available=hasattr(torch, "xpu") and torch.xpu.is_available(),
                 _is_torch_npu_available=hasattr(torch, "npu") and torch.npu.is_available())
    class NoPagedCache:
        pass
    def unsupported_warning(*args):
        raise AssertionError("Unsupported SDPA attention-weight request reached oracle")
    scope.update(flags, PagedAttentionCache=NoPagedCache, logger=SimpleNamespace(warning_once=unsupported_warning))
    module = ast.fix_missing_locations(ast.Module(
        body=ast.parse("from __future__ import annotations").body + selected, type_ignores=[]))
    exec(compile(module, str(sdpa_path), "exec"), scope)
    class ExplicitSDPA:
        @staticmethod
        def get_interface(name, default):
            if name != "sdpa":
                raise ValueError("SDPA qualification must explicitly request sdpa")
            return scope["sdpa_attention_forward"]
    scope["ALL_ATTENTION_FUNCTIONS"] = ExplicitSDPA
    return attention, flags


def compare(oracle_class, geometry, weights, device, dtypes, real):
    g, report = geometry, []
    config = SimpleNamespace(hidden_size=g.indexer.hidden_size, indexer_n_heads=g.indexer.query_heads,
        indexer_kv_heads=1, indexer_head_dim=g.indexer.head_dim, indexer_budget=g.indexer.token_budget,
        indexer_compress_ratio=g.indexer.compress_ratio, rms_norm_eps=g.indexer.eps,
        num_attention_heads=g.query_heads, num_key_value_heads=g.kv_heads, head_dim=g.head_dim,
        attention_dropout=0.0, attention_bias=False, _attn_implementation="sdpa")
    for dtype in dtypes:
        candidate = FlashNextQSAAttention(g, attention_backend="sdpa").to(device=device, dtype=dtype).eval()
        candidate.load_state_dict(weights, strict=True)
        oracle = oracle_class(config, 3).to(device=device, dtype=dtype).eval()
        oracle.load_state_dict(candidate.state_dict(), strict=True)
        for name, x, past, visible in cases(g, real):
            x = x.to(device=device, dtype=dtype)
            if past is not None:
                past = QSAState(*(t.to(device=device, dtype=dtype) for t in (past.raw_indexer_keys, past.keys, past.values)))
            preserved = None if past is None else past.truncate(past.raw_indexer_keys.shape[1])
            pos = tuple(v.to(device=device, dtype=dtype) for v in positions(x.shape[0], visible.shape[-1], 64 if real else 2))
            for additive in (False, True):
                visibility = visible.to(device=device)
                mask = torch.where(visibility, torch.zeros((), dtype=dtype, device=device),
                                   torch.finfo(dtype).min) if additive else visibility
                cache = Cache(past)
                with sdpa_kernel(SDPBackend.MATH), torch.inference_mode():
                    actual, state, probabilities = candidate(x, pos, mask, past=past)
                    expected, expected_probabilities = oracle(x, pos, mask, cache)
                if not torch.isfinite(actual).all() or probabilities is not None or expected_probabilities is not None:
                    raise AssertionError("SDPA nonfinite output or incorrect probability-return contract")
                for a, b in ((actual, expected), (state.raw_indexer_keys, cache.raw),
                             (state.keys, cache.keys), (state.values, cache.values)):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                if past is not None:
                    for a, b in zip((past.raw_indexer_keys, past.keys, past.values),
                                    (preserved.raw_indexer_keys, preserved.keys, preserved.values)):
                        assert torch.equal(a, b)
                empty = ~visibility.any(-1).squeeze(1)
                if empty.any() and actual[empty].count_nonzero():
                    raise AssertionError("Bias-free fully masked SDPA output is not zero")
                report.append(dict(case=name, dtype=str(dtype), device=str(device), actual_official_weights=real,
                    additive_mask=additive, output_exact=True, all_output_finite=True, all_three_states_exact=True,
                    empty_query_rows=int(empty.sum().item()), empty_query_output_exact_zero=True))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream-modeling", required=True)
    p.add_argument("--upstream-sdpa", required=True)
    p.add_argument("--samples", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    device = torch.device(a.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(1536 * 2**20 / torch.cuda.get_device_properties(device).total_memory, device)
    dtypes = (torch.float32, torch.bfloat16) if device.type == "cpu" else (torch.float32, torch.float16)
    oracle, flags = reference(a.upstream_modeling, a.upstream_sdpa)
    torch.manual_seed(116)
    g = QSAAttentionGeometry(QSAIndexerGeometry(8, 2, 4, 4, 2), 4, 2, 4)
    model = FlashNextQSAAttention(g)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter)*0.15)
    report = compare(oracle, g, model.state_dict(), device, dtypes, False)
    root = Path(a.samples)
    provenance = json.loads((root/"ranges.json").read_text())
    if provenance["revision"] != REVISION or hashlib.sha256((root/"model.safetensors").read_bytes()).hexdigest() != provenance["artifact_sha256"]:
        raise ValueError("Original attention sample differs")
    g = QSAAttentionGeometry.from_dict(json.loads((root/"config.json").read_text()))
    report.extend(compare(oracle, g, load_file(root/"model.safetensors"), device, dtypes, True))
    result = dict(scope="Explicit floating QSA SDPA MATH and empty-row correctness, not an optimization/default switch",
        cases=report, upstream_sha256=UPSTREAM_SHA256, sdpa_source_sha256=SDPA_SHA256,
        reference_index_abi_shim=INDEX_ABI_SHIM, reference_runtime_flags=flags,
        sdpa_backend="MATH only, both oracle/candidate in caller-owned scoped context", torch=str(torch.__version__),
        original_revision=REVISION, official_source_bytes=provenance["total_bytes"],
        batch_history_scope="B8/P16 and separate B1/P4096,B2/P4100, not B8 long-history",
        paged_cache_tested=False, fused_sdpa_backend_tested=False, native_kernel_implemented=False,
        whole_model_tested=False, throughput_measured=False)
    (output/"qualification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
