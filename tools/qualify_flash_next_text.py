"""Pinned text forward + real ngram hashing on tiny SYNTHETIC two-layer weights.

No original full checkpoint, quality, native generation or performance claim.
Original text/layer/component bodies execute independently. Tiny resident oracle
embedding vs bounded CPU safetensors row reader, with original cache slots0/1/2.
"""

import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from qualify_flash_next_layer import references, make_oracle
from qualify_flash_next_qsa import UPSTREAM_SHA256
from qualify_flash_next_gdn import CACHE_SHA256
from qualify_flash_next_sdpa import SDPA_SHA256
from venturi.models.flash_next_ngram import NGramRequest, PreparedNGramTable
from venturi.models.flash_next_request import FlashNextRequest


def text_references(path, scope):
    data = Path(path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == UPSTREAM_SHA256
    tree = ast.parse(data)
    functions = {'_splitmix64', '_build_layer_multipliers', '_is_prime', '_find_nth_prime_after'}
    classes = {'Qwen4ExpTextRotaryEmbedding', 'Qwen4ExpTextNGramEmbedding'}
    constants = {'_MASK64', '_SPLITMIX_GAMMA', '_SPLITMIX_M1', '_SPLITMIX_M2', '_PRIME_1'}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
            selected.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in functions | classes:
            if isinstance(node, ast.ClassDef) and node.name == 'Qwen4ExpTextRotaryEmbedding':
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == 'forward':
                        method.decorator_list = []  # only static/default positions; no dynamic update
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == 'Qwen4ExpTextModel':
            forward = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
            forward.decorator_list = []  # inference output/decorator infrastructure excluded
            selected.append(ast.ClassDef(name='OriginalTextModel', bases=[ast.Attribute(
                value=ast.Name(id='nn', ctx=ast.Load()), attr='Module', ctx=ast.Load())],
                keywords=[], body=[forward], decorator_list=[]))
    scope.update(math=math, Qwen4ExpModelOutputWithPast=SimpleNamespace)
    module = ast.fix_missing_locations(ast.Module(body=ast.parse('from __future__ import annotations').body+selected,
                                                  type_ignores=[]))
    exec(compile(module, str(path), 'exec'), scope)
    return scope


def make_text_oracle(ref, qsa, moe, candidate, c, table, multipliers, device, dtype):
    # Keep prior supplied-row constructor only while binding independent layer
    # weights; replace it with original fully hashed embedding before execution.
    oracle = ref['OriginalTextModel']()
    oracle.config = SimpleNamespace(ple_layer_ids=[2], eos_token_id=31, hc_count=2, num_hidden_layers=2,
                                   layer_types=['linear_attention', 'indexed_attention'])
    oracle.embed_tokens = nn.Embedding(32, 8).to(device=device, dtype=dtype)
    oracle.embed_tokens.load_state_dict(candidate.embed_tokens.state_dict())
    layers = []
    for i, (layer, kind) in enumerate(zip(candidate.layers, ('gdn', 'qsa'))):
        block = make_oracle(ref, qsa, moe, layer, kind, device, dtype)
        attention = block.linear_attn if kind == 'gdn' else block.self_attn
        attention.layer_idx = i
        if kind == 'qsa':
            attention.indexer.layer_idx = i
        if block.ple is not None:
            block.ple.layer_idx = i
            config = SimpleNamespace(ngram_size=c.ngram_size, heads_per_ngram=c.heads_per_ngram,
                vocab_size=c.vocab_size, ngram_vocab_size_base=c.ngram_vocab_size_base, seed=0,
                eos_token_id=c.eos_token_id, make_ngram_vocab_size_divisible_by=c.ngram_padding)
            embedding = ref['OriginalNGram'](config, 16, i).to(device=device, dtype=dtype)
            with torch.no_grad():
                embedding.layer_multipliers.copy_(torch.tensor(multipliers, device=device))
                embedding.ngram_embedding.weight.copy_(table.to(device=device, dtype=dtype))
            block.ple.ple_embedding = embedding
        layers.append(block)
    oracle.layers = nn.ModuleList(layers)
    config = SimpleNamespace(hidden_size=8, hc_count=2, hc_lowrank=3, rms_norm_eps=1e-6,
        max_position_embeddings=4096, head_dim=4,
        rope_parameters=dict(rope_type='default', rope_theta=10000000., partial_rotary_factor=.5))
    # Source default inverse-frequency buffer stays FP32, not a weight cast.
    oracle.rotary_emb = ref['Qwen4ExpTextRotaryEmbedding'](config).to(device=device)
    oracle.hyper_connection_mixer = ref['Qwen4ExpTextGatedResidual'](config, use_combine=False).to(device=device, dtype=dtype)
    oracle.hyper_connection_mixer.load_state_dict(candidate.hyper_connection_mixer.state_dict())
    head = nn.Linear(8, 32, bias=False).to(device=device, dtype=dtype)
    head.load_state_dict(candidate.lm_head.state_dict())
    return oracle.eval(), head.eval()


def make_cache(ref):
    class TextCache(ref['ReferenceCache']):
        def __init__(self):
            self.layers = [ref['LinearAttentionLayer'](number_of_states=3) for _ in range(2)]
            for layer in self.layers:
                layer.indexer_position_ids = None
            self.raw, self.keys, self.values, self.seen = {}, {}, {}, 0
        def get_seq_length(self):
            return self.seen
        def update_indexer(self, keys, layer):
            self.raw[layer] = keys.clone() if layer not in self.raw else torch.cat((self.raw[layer], keys), 1)
            return self.raw[layer]
        def update(self, keys, values, layer):
            self.keys[layer] = keys.clone() if layer not in self.keys else torch.cat((self.keys[layer], keys), 2)
            self.values[layer] = values.clone() if layer not in self.values else torch.cat((self.values[layer], values), 2)
            return self.keys[layer], self.values[layer]
    return TextCache()


def assert_state(actual, cache):
    for a, b in ((actual.layers[0].attention.conv, cache.get_conv_state(0)),
                 (actual.layers[0].attention.recurrent, cache.get_recurrent_state(0)),
                 (actual.layers[1].attention.raw_indexer_keys, cache.raw[1]),
                 (actual.layers[1].attention.keys, cache.keys[1]),
                 (actual.layers[1].attention.values, cache.values[1]),
                 (actual.layers[1].ple.conv, cache.get_conv_state(1, state_idx=1))):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert actual.tokens_seen == cache.seen
    assert actual.layers[0].attention.tokens_seen == cache.seen and actual.layers[1].ple.tokens_seen == cache.seen


def oracle_step(oracle, head, cache, ids, mask, visible):
    b, n = ids.shape
    total = cache.seen+n
    causal = torch.arange(total, device=ids.device)[None, None, None, :] <= torch.arange(
        cache.seen, total, device=ids.device)[None, None, :, None]
    output = oracle(ids, attention_mask=dict(indexed_attention=causal & visible[:, None, None, :],
        linear_attention=mask), past_key_values=cache, use_cache=True)
    cache.seen = total
    return head(output.last_hidden_state)


def run_case(candidate, oracle, head, ref, store, b, length, cached, masking, device, dtype):
    cache = make_cache(ref)
    base = store.store if isinstance(store, PreparedNGramTable) else store
    reqs = [NGramRequest(base.config, base.multipliers) for _ in range(b)]
    past, visible, records = None, None, []
    for phase, n in ((('prefix', 3), ('proposal', length)) if cached else (('prefill', length),)):
        ids = (torch.arange(b*n).reshape(b, n)+2) % 30
        if n > 1:
            ids[:, 1] = 31
        ids = ids.to(device=device)
        mask = None
        if masking:
            mask = torch.ones(b, n, dtype=torch.bool, device=device)
            mask[0, ::2] = False
            if b > 1:
                mask[-1] = False
        current = torch.ones_like(ids, dtype=torch.bool) if mask is None else mask
        visible = current if visible is None else torch.cat((visible, current), -1)
        ple_ids = ids if mask is None else torch.where(mask, ids, 31)
        plans = [r.preview(t) for r, t in zip(reqs, ple_ids.cpu().tolist())]
        rows = torch.stack(store.lookup_many(plans)).to(device=device, dtype=dtype)
        produced = []
        handle = oracle.layers[1].ple.ple_embedding.register_forward_hook(lambda m, args, output: produced.append(output.clone()))
        try:
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                actual, state = candidate(ids, conv_mask=mask, ple_rows={1: rows}, past=past)
                expected = oracle_step(oracle, head, cache, ids, mask, visible)
        finally:
            handle.remove()
        torch.testing.assert_close(rows, produced[0], atol=0, rtol=0)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert torch.isfinite(actual).all() and torch.equal(actual.argmax(-1), expected.argmax(-1))
        assert_state(state, cache)
        for request, plan in zip(reqs, plans):
            request.commit(plan)
        tails = cache.get_conv_state(1, state_idx=2).cpu().tolist()
        for request, tail in zip(reqs, tails):
            eos = max((i for i, token in enumerate(tail) if token == 31), default=-1)
            assert request.history == tuple(tail[eos+1:])
        seen = state.tokens_seen
        expected_pos = torch.arange(seen, device=device).view(1, 1, seen).expand(3, b, -1)
        assert torch.equal(cache.layers[1].indexer_position_ids, expected_pos)
        records.append(dict(batch=b, length=n, past_length=seen-n, phase=phase, masking=masking,
            device=str(device), dtype=str(dtype), logits_exact=True, greedy_exact=True, hash_rows_exact=True,
            all_states_exact=True, canonical_ngram_tail_exact=True, full_positions_exact=True, all_finite=True))
        past = state
    return records


def request_trajectory(candidate, oracle, head, ref, store, device, dtype):
    req, cache = FlashNextRequest(candidate, store), make_cache(ref)
    records, tokens, visible = [], [2, 31, 5], None
    for step in range(5):
        ids = torch.tensor([tokens], device=device)
        current = torch.ones_like(ids, dtype=torch.bool)
        visible = current if visible is None else torch.cat((visible, current), -1)
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            proposal = req.preview(tokens)
            expected = oracle_step(oracle, head, cache, ids, None, visible)
            torch.testing.assert_close(proposal.logits, expected, atol=0, rtol=0)
            next_id = int(expected[0, -1].argmax())
            assert int(proposal.logits[0, -1].argmax()) == next_id
            req.commit(proposal)
            torch.testing.assert_close(req.last_logits, expected[:, -1], atol=0, rtol=0)
            assert_state(req.state, cache)
        records.append(dict(step=step, input_tokens=tokens, next_token=next_id, logits_exact=True,
                            all_states_exact=True, device=str(device), dtype=str(dtype)))
        tokens = [next_id]
    return records



def partial_trajectory(candidate, oracle, head, ref, store, device, dtype):
    req, cache, visible, records = FlashNextRequest(candidate, store), make_cache(ref), None, []
    steps = (([2, 3], None, 2), ([4, 31, 7], [True, False, True], 1),
             ([5, 6, 7], [False, True, True], 2), ([8], None, 0), ([9], None, 1))
    for tokens, bits, count in steps:
        ids = torch.tensor([tokens], device=device)
        mask = None if bits is None else torch.tensor([bits], device=device, dtype=torch.bool)
        current = torch.ones_like(ids, dtype=torch.bool) if mask is None else mask
        proposal_visible = current if visible is None else torch.cat((visible, current), -1)
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            cloned = copy.deepcopy(cache)
            proposal = req.preview(tokens, mask=bits)
            expected = oracle_step(oracle, head, cloned, ids, mask, proposal_visible)
            torch.testing.assert_close(proposal.logits, expected, atol=0, rtol=0)
            req.commit(proposal, count)
            if count:
                accepted = current[:, :count]
                visible = accepted if visible is None else torch.cat((visible, accepted), -1)
                accepted_logits = oracle_step(oracle, head, cache, ids[:, :count], None if mask is None else mask[:, :count], visible)
                torch.testing.assert_close(req.last_logits, accepted_logits[:, -1], atol=0, rtol=0)
            assert_state(req.state, cache)
            tail = cache.get_conv_state(1, state_idx=2)[0].cpu().tolist()
            eos = max((i for i, token in enumerate(tail) if token == 31), default=-1)
            assert req.history == tuple(tail[eos+1:])
        records.append(dict(input_tokens=tokens, mask=bits, accepted=count, logits_exact=True,
            committed_states_exact=True, hash_history_exact=True, device=str(device), dtype=str(dtype)))
    return records

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
    from test_flash_next_text import module as candidate_module, table_fixture
    ref, qsa, flags, moe = references(a.upstream_modeling, a.upstream_cache, a.upstream_sdpa)
    text_references(a.upstream_modeling, ref)
    # Avoid changing original PLE constructor dispatch used by make_oracle:
    # full original ngram module is attached only after its layer weights bind.
    from qualify_flash_next_ple import SuppliedRows
    original_ngram = ref['Qwen4ExpTextNGramEmbedding']
    store, table = table_fixture(out/'tiny-synthetic-table')
    reader = PreparedNGramTable(store).prepare()
    cases, trajectories, partial = [], [], []
    dtypes = (torch.float32, torch.bfloat16) if device.type == 'cpu' else (torch.float32, torch.float16)
    try:
        for dtype in dtypes:
            candidate = candidate_module(dtype, device)
            ref['Qwen4ExpTextNGramEmbedding'] = SuppliedRows
            # Original class remains available under a constructor-independent
            # name; make_text_oracle restores it at the actual binding point.
            ref['OriginalNGram'] = original_ngram
            oracle, head = make_text_oracle(ref, qsa, moe, candidate, store.config, table, store.multipliers, device, dtype)
            for b, n, cached in ((1, 1, False), (2, 5, False), (8, 1, False), (1, 1, True), (2, 3, True), (8, 1, True)):
                for masking in (False, True):
                    cases.extend(run_case(candidate, oracle, head, ref, reader, b, n, cached, masking, device, dtype))
            trajectories.extend(request_trajectory(candidate, oracle, head, ref, store, device, dtype))
            partial.extend(partial_trajectory(candidate, oracle, head, ref, store, device, dtype))
    finally:
        reader.close()
    report = dict(scope='Tiny synthetic H8/HC2/V32 two-layer GDN+QSA text core, all4 experts/top2, actual hash/table bridge',
        cases=cases, greedy_trajectory_steps=trajectories, partial_acceptance_steps=partial, torch=str(torch.__version__), reference_runtime_flags=flags,
        upstream_sha256=UPSTREAM_SHA256, upstream_cache_sha256=CACHE_SHA256, sdpa_sha256=SDPA_SHA256,
        oracle='Original text forward/default rotary, decoder/components, full NGramEmbedding forward+slots0/1/2; explicit KV concatenation adapter and mask dict, all logits then head',
        table_scope='Tiny random BF16 CPU safetensors reader vs same tiny resident original embedding; not actual official table/weights',
        actual_checkpoint_tested=False, whole_official_model_tested=False, native_kernel_implemented=False,
        throughput_measured=False, quality_measured=False)
    (out/'qualification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
