"""Bounded pinned original text arithmetic with lazy experts/PLE, no full banks.

Static parameters alias the candidate's already bound immutable weights: this
checks independent ORIGINAL forward/cache/hash arithmetic, NOT an independent
checkpoint binding. Reference experts decode on CPU; native decode is separate.
Only the weight indexing/I/O bridge replaces huge original resident arrays.
"""

from types import SimpleNamespace

import torch
from torch import nn

from qualify_flash_next_layer import references
from qualify_flash_next_text import text_references, oracle_step
from venturi.flash_next import FlashNextConfig
from venturi.models.flash_next_fp8 import BlockFP8ExpertStore
from venturi.models.flash_next_gdn import GDNState
from venturi.models.flash_next_moe import MoEGeometry
from venturi.models.flash_next_ngram import NGramProposal


class SelectedPair:
    """One expert pair through ORIGINAL gate-up/down indexing order, no pruning."""
    def __init__(self, store, device, dtype):
        self.store, self.device, self.dtype, self.pending = store, device, dtype, None

    def take(self, expert, position):
        expert = int(expert)
        if position == 0:
            if self.pending is not None:
                raise RuntimeError('Reference expert gate/down indexing order changed')
            values = self.store.read(expert, device=self.device, dtype=self.dtype)
            self.pending = (expert, values[1])
            return values[0]
        if self.pending is None or self.pending[0] != expert:
            raise RuntimeError('Reference expert down indexing lacks its gate')
        value = self.pending[1]
        self.pending = None
        return value


class ProjectionIndex:
    def __init__(self, pair, position):
        self.pair, self.position = pair, position

    def __getitem__(self, expert):
        return self.pair.take(expert, self.position)


class AddressedEmbedding(nn.Module):
    """Original hash indices -> validated reader; no producer/hash substitution."""
    def __init__(self, reader, device, dtype):
        super().__init__()
        self.reader = reader
        self.weight = nn.Parameter(torch.empty(0, device=device, dtype=dtype), requires_grad=False)

    def forward(self, indices):
        c = self.reader.table_key[0]
        batch, length, heads = indices.shape
        if heads != c.ngram_heads:
            raise ValueError('Original ngram head geometry changed')
        rows = tuple(tuple(row) for row in indices.reshape(-1, heads).cpu().tolist())
        proposal = NGramProposal(None, 0, self.reader.table_key, (0,)*len(rows), rows)
        output = self.reader.lookup(proposal)
        return output.to(device=self.weight.device, dtype=self.weight.dtype).reshape(batch, length, heads, -1)


def make_full_oracle(checkpoint, candidate, reader, *, modeling, cache_source, sdpa_source):
    ref, qsa, flags, moe = references(modeling, cache_source, sdpa_source)
    text_references(modeling, ref)
    c = FlashNextConfig.from_dict(dict(checkpoint.config))
    raw = dict(checkpoint.config['text_config'])
    raw.update(layer_types=['linear_attention' if kind == 'linear_attention' else 'indexed_attention' for kind in c.layer_types],
               _attn_implementation='sdpa')
    for name, value in dict(seed=0, norm_topk_prob=True, attention_bias=False, attention_dropout=0.).items():
        raw.setdefault(name, value)
    config = SimpleNamespace(**raw)
    device, dtype = candidate.embed_tokens.weight.device, candidate.embed_tokens.weight.dtype
    original_ngram = ref['Qwen4ExpTextNGramEmbedding']
    layers = []
    geometry = MoEGeometry.from_dict(dict(checkpoint.config))
    for i, current in enumerate(candidate.layers):
        block = ref['Qwen4ExpTextDecoderLayer'].__new__(ref['Qwen4ExpTextDecoderLayer'])
        nn.Module.__init__(block)
        block.layer_type = config.layer_types[i]
        with torch.device('meta'):
            for name in ('attn_hyper_connection', 'mlp_hyper_connection'):
                setattr(block, name, ref['Qwen4ExpTextGatedResidual'](config))
            attention = ref['Qwen4ExpTextGatedDeltaNet'](config, i) if block.layer_type == 'linear_attention' else qsa(config, i)
            block.mlp = moe(config)
            block.ple = ref['Qwen4ExpTextPLELayer'](config, i, c.ple_layers.index(i+1)) if current.ple is not None else None
        for name in ('attn_hyper_connection', 'mlp_hyper_connection'):
            getattr(block, name).load_state_dict(getattr(current, name).state_dict(), strict=True, assign=True)
        attention.load_state_dict(current.attention.state_dict(), strict=True, assign=True)
        setattr(block, 'linear_attn' if block.layer_type == 'linear_attention' else 'self_attn', attention)
        missing = block.mlp.load_state_dict(current.mlp.state_dict(), strict=False, assign=True)
        if set(missing.missing_keys) != {'experts.gate_up_proj', 'experts.down_proj'} or missing.unexpected_keys:
            raise ValueError('Full reference static MoE schema changed')
        source = BlockFP8ExpertStore(checkpoint, geometry,
            f'model.language_model.layers.{i}.mlp.experts', decode_strategy='vectorized')
        pair = SelectedPair(source, device, dtype)
        for key, index in (('gate_up_proj', 0), ('down_proj', 1)):
            del block.mlp.experts._parameters[key]
            setattr(block.mlp.experts, key, ProjectionIndex(pair, index))
        if block.ple is not None:
            weights = current.ple.state_dict()
            missing = block.ple.load_state_dict(weights, strict=False, assign=True)
            if missing.unexpected_keys or any(not key.startswith('ple_embedding.') for key in missing.missing_keys):
                raise ValueError('Full reference PLE conditioning schema changed')
            embedding = block.ple.ple_embedding
            if not isinstance(embedding, original_ngram):
                raise ValueError('Original ngram producer was substituted')
            embedding.ngram_embedding = AddressedEmbedding(reader, device, dtype)
            embedding.layer_multipliers = torch.tensor(reader.table_key[2], dtype=torch.long, device=device)
            embedding.ngram_heads_vocab_sizes = torch.tensor(embedding.head_vocab_sizes, dtype=torch.long, device=device)
            embedding.ngram_heads_offsets = torch.tensor(embedding.head_offsets, dtype=torch.long, device=device)
        layers.append(block)
    oracle = ref['OriginalTextModel']()
    oracle.config, oracle.embed_tokens = config, candidate.embed_tokens
    oracle.layers = nn.ModuleList(layers)
    oracle.hyper_connection_mixer = candidate.hyper_connection_mixer
    # Keep original final mixer arithmetic too, not the candidate implementation.
    with torch.device('meta'):
        mixer = ref['Qwen4ExpTextGatedResidual'](config, use_combine=False)
    mixer.load_state_dict(candidate.hyper_connection_mixer.state_dict(), strict=True, assign=True)
    oracle.hyper_connection_mixer = mixer
    oracle.rotary_emb = ref['Qwen4ExpTextRotaryEmbedding'](config).to(device=device)
    oracle.eval()
    if any(p.device.type == 'meta' for p in oracle.parameters()):
        raise ValueError('Unbound reference parameter remains on meta')

    class FullCache(ref['ReferenceCache']):
        def __init__(self):
            self.layers = [ref['LinearAttentionLayer'](number_of_states=3) for _ in layers]
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
    return oracle, candidate.lm_head, FullCache, flags


def assert_full_state(state, cache):
    for i, layer in enumerate(state.layers):
        attention = layer.attention
        if isinstance(attention, GDNState):
            pairs = ((attention.conv, cache.get_conv_state(i)), (attention.recurrent, cache.get_recurrent_state(i)))
        else:
            pairs = ((attention.raw_indexer_keys, cache.raw[i]), (attention.keys, cache.keys[i]), (attention.values, cache.values[i]))
        if layer.ple is not None:
            pairs += ((layer.ple.conv, cache.get_conv_state(i, state_idx=1)),)
        for actual, expected in pairs:
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert state.tokens_seen == cache.seen


def run_oracle_step(oracle, head, cache, tokens):
    ids = torch.tensor([tokens], device=next(head.parameters()).device, dtype=torch.long)
    visible = torch.ones(1, cache.seen+len(tokens), device=ids.device, dtype=torch.bool)
    return oracle_step(oracle, head, cache, ids, None, visible)
