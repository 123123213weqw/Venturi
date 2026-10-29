"""Pinned complete decoder-layer forward on small synthetic weights/states.

All four small experts exist, no routing pruning. PLE embedding producer is
supplied rows. Not an original full layer/checkpoint/model/quality/TPS test.
"""

import argparse
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from qualify_flash_next_moe import reference_class as moe_reference
from qualify_flash_next_ple import reference as ple_reference
from qualify_flash_next_qsa import UPSTREAM_SHA256, positions
from qualify_flash_next_sdpa import reference as sdpa_reference, SDPA_SHA256
from qualify_flash_next_gdn import CACHE_SHA256
from venturi.models.flash_next_gdn import GDNState


def references(modeling, cache, sdpa):
    scope = ple_reference(modeling, cache)
    names = {"Qwen4ExpTextGatedResidual", "Qwen4ExpTextDecoderLayer"}
    selected = [n for n in ast.parse(Path(modeling).read_bytes()).body if isinstance(n, ast.ClassDef) and n.name in names]
    if {n.name for n in selected} != names:
        raise ValueError("Pinned decoder/HC definitions missing")
    scope["GradientCheckpointingLayer"] = nn.Module  # inference-only, no checkpoint wrapper
    module = ast.fix_missing_locations(ast.Module(
        body=ast.parse("from __future__ import annotations").body + selected, type_ignores=[]))
    exec(compile(module, str(modeling), "exec"), scope)
    qsa, flags = sdpa_reference(modeling, sdpa)
    return scope, qsa, flags, moe_reference(modeling)


def make_oracle(ref, qsa, moe, candidate, kind, device, dtype):
    config = SimpleNamespace(hidden_size=8, hc_count=2, hc_lowrank=3, rms_norm_eps=1e-6,
        linear_num_key_heads=2, linear_num_value_heads=6, linear_key_head_dim=4, linear_value_head_dim=4,
        linear_conv_kernel_dim=4, hidden_act="silu", output_gate_type="sigmoid", layer_types=["linear_attention"],
        num_experts=4, num_experts_per_tok=2, norm_topk_prob=True, moe_intermediate_size=6,
        shared_expert_intermediate_size=6, ple_embed_dim=16, ple_conv_kernel_size=4, ngram_size=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4, attention_dropout=0.0, attention_bias=False,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=4, indexer_budget=4, indexer_compress_ratio=2,
        _attn_implementation="sdpa")
    # Construct independent original component modules and run the original
    # decoder forward, without its full checkpoint/model constructor dispatch.
    cls = ref["Qwen4ExpTextDecoderLayer"]
    oracle = cls.__new__(cls)
    nn.Module.__init__(oracle)
    oracle.layer_type = "linear_attention" if kind == "gdn" else "indexed_attention"
    for name in ("attn_hyper_connection", "mlp_hyper_connection"):
        block = ref["Qwen4ExpTextGatedResidual"](config).to(device=device, dtype=dtype)
        block.load_state_dict(getattr(candidate, name).state_dict(), strict=True)
        setattr(oracle, name, block)
    block = ref["Qwen4ExpTextGatedDeltaNet"](config, 0) if kind == "gdn" else qsa(config, 0)
    block = block.to(device=device, dtype=dtype)
    block.load_state_dict(candidate.attention.state_dict(), strict=True)
    setattr(oracle, "linear_attn" if kind == "gdn" else "self_attn", block)
    oracle.mlp = moe(config).to(device=device, dtype=dtype)
    weights = dict(candidate.mlp.state_dict())
    for name, index in (("gate_up_proj", 0), ("down_proj", 1)):
        weights["experts."+name] = torch.stack([candidate.mlp.expert_store.experts[i][index] for i in range(4)]).to(
            device=device, dtype=dtype)
    oracle.mlp.load_state_dict(weights, strict=True)
    oracle.ple = None
    if candidate.ple is not None:
        oracle.ple = ref["Qwen4ExpTextPLELayer"](config, 0, 0).to(device=device, dtype=dtype)
        oracle.ple.load_state_dict(candidate.ple.state_dict(), strict=True)
    return oracle.eval()


def make_cache(ref):
    class CombinedCache(ref["ReferenceCache"]):
        def __init__(self):
            self.layers = [ref["LinearAttentionLayer"](number_of_states=3)]
            self.raw, self.keys, self.values = None, None, None

        def update_indexer(self, keys, layer):
            self.raw = keys.clone() if self.raw is None else torch.cat((self.raw, keys), 1)
            return self.raw

        def update(self, keys, values, layer):
            self.keys = keys.clone() if self.keys is None else torch.cat((self.keys, keys), 2)
            self.values = values.clone() if self.values is None else torch.cat((self.values, values), 2)
            return self.keys, self.values
    return CombinedCache()


def assert_state(actual, cache, kind, has_ple, seen):
    if kind == "gdn":
        torch.testing.assert_close(actual.attention.conv, cache.get_conv_state(0), atol=0, rtol=0)
        torch.testing.assert_close(actual.attention.recurrent, cache.get_recurrent_state(0), atol=0, rtol=0)
        assert actual.attention.tokens_seen == seen
    else:
        for a, b in ((actual.attention.raw_indexer_keys, cache.raw), (actual.attention.keys, cache.keys),
                     (actual.attention.values, cache.values)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    if has_ple:
        torch.testing.assert_close(actual.ple.conv, cache.get_conv_state(0, state_idx=1), atol=0, rtol=0)
        assert actual.ple.tokens_seen == seen


def state_tensors(state):
    if state is None:
        return ()
    a = state.attention
    tensors = (a.conv, a.recurrent) if isinstance(a, GDNState) else (a.raw_indexer_keys, a.keys, a.values)
    return tensors + (() if state.ple is None else (state.ple.conv,))


def run_case(candidate, oracle, ref, kind, has_ple, b, length, cached, masked, dtype, device):
    rng = torch.Generator().manual_seed(719)
    cache = make_cache(ref)
    past, records = None, []
    steps = (("prefix", 3), ("proposal", length)) if cached else (("prefill", length),)
    seen = 0
    for phase, n in steps:
        x = (torch.randn(b, n, 16, generator=rng)*0.1).to(device=device, dtype=dtype)
        rows = (torch.randn(b, n, 16, generator=rng)*0.1).to(device=device, dtype=dtype)
        pos = tuple(v.to(device=device, dtype=dtype) for v in positions(b, seen+n, 2))
        visible = torch.arange(seen+n, device=device).view(1, 1, 1, -1) <= torch.arange(
            seen, seen+n, device=device).view(1, 1, -1, 1)
        visible = visible.expand(b, 1, -1, -1).clone()
        conv_mask = None
        if masked:
            conv_mask = torch.ones(b, n, dtype=torch.bool, device=device)
            conv_mask[0, ::2] = False
            visible[0, :, :, 1::3] = False
            if b > 1:
                conv_mask[-1] = False
                visible[-1, :, -1] = False
        preserved = None if past is None else past.snapshot()
        if has_ple:
            oracle.ple.ple_embedding.rows = rows
        with sdpa_kernel(SDPBackend.MATH), torch.inference_mode():
            actual, proposed = candidate(x, position_embeddings=pos, attention_mask=visible, conv_mask=conv_mask,
                                        ple_rows=rows if has_ple else None, past=past)
            expected = oracle(x, pos, attention_mask=visible, conv_mask=conv_mask, past_key_values=cache,
                              ple_input_ids=torch.zeros(b, n, dtype=torch.long, device=device))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert_state(proposed, cache, kind, has_ple, seen+n)
        if not torch.isfinite(actual).all() or any(not torch.equal(a, b) for a, b in zip(state_tensors(past), state_tensors(preserved))):
            raise AssertionError("Layer nonfinite output or mutated input cache")
        records.append(dict(kind=kind, ple=has_ple, batch=b, length=n, past_length=seen, phase=phase, masked=masked,
            device=str(device), dtype=str(dtype), output_exact=True, all_output_finite=True, all_states_exact=True,
            past_unchanged=True))
        past, seen = proposed, seen+n
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream-modeling", required=True)
    p.add_argument("--upstream-cache", required=True)
    p.add_argument("--upstream-sdpa", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    device = torch.device(a.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(1024*2**20/torch.cuda.get_device_properties(device).total_memory, device)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tests"))
    from test_flash_next_layer import module as make_candidate
    ref, qsa, flags, moe = references(a.upstream_modeling, a.upstream_cache, a.upstream_sdpa)
    dtypes = (torch.float32, torch.bfloat16) if device.type == "cpu" else (torch.float32, torch.float16)
    report = []
    for dtype in dtypes:
        for kind in ("gdn", "qsa"):
            for has_ple in (False, True):
                candidate = make_candidate(kind, has_ple, dtype).to(device=device).eval()
                oracle = make_oracle(ref, qsa, moe, candidate, kind, device, dtype)
                for b, n, cached in ((1, 1, False), (2, 5, False), (8, 1, False), (2, 1, True), (2, 3, True)):
                    for masked in (False, True):
                        report.extend(run_case(candidate, oracle, ref, kind, has_ple, b, n, cached, masked, dtype, device))
    result = dict(scope="Complete decoder-layer original forward, small synthetic weights/geometry only", cases=report,
        upstream_sha256=UPSTREAM_SHA256, upstream_cache_sha256=CACHE_SHA256, sdpa_sha256=SDPA_SHA256,
        reference_runtime_flags=flags, torch=str(torch.__version__),
        fixture="H8/HC2, all4 experts/top2, GDN K2/V6/dim4 or small QSA; optional supplied-row PLE",
        oracle_harness="Original decoder forward and original components; inference-only nn.Module base, explicit module binding, supplied embedding rows, KV concatenation adapter",
        state_contract="Independent prefix/proposal evolution, actual linear cache slots0/1, explicit QSA raw/main KV; no ngram-history slot2 assertion",
        actual_checkpoint_tested=False, whole_model_tested=False, native_kernel_implemented=False, throughput_measured=False)
    (output/"qualification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
