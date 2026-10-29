"""Tiny synthetic text/hash/request integration
not original model quality."""

from concurrent.futures import ThreadPoolExecutor
import json

import pytest
from safetensors.torch import save_file
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from test_flash_next import raw_config
from test_flash_next_layer import module as layer_module
from venturi.checkpoint import inspect_checkpoint
from venturi.flash_next import FlashNextConfig
from venturi.models.flash_next_ngram import PreparedNGramTable, ShardedNGramTable
from venturi.models.flash_next_request import FlashNextRequest
from venturi.models.flash_next_residual import FlashNextGatedResidual
from venturi.models.flash_next_text import FlashNextTextCore


def module(dtype=torch.float32, device='cpu'):
    layers = [layer_module('gdn', False, dtype), layer_module('qsa', True, dtype)]
    torch.manual_seed(810)
    m = FlashNextTextCore(nn.Embedding(32, 8), layers, FlashNextGatedResidual(8, 2, 3, use_combine=False),
                         nn.Linear(8, 32, bias=False), rotary_dim=2).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if not name.startswith('layers.'):
                p.copy_(torch.randn_like(p)*0.1)
    return m


def table_fixture(path, *, dtype=torch.bfloat16):
    path.mkdir(parents=True, exist_ok=True)
    raw = raw_config()
    c = FlashNextConfig.from_dict(raw)
    _, _, padded = c.ngram_geometry()
    table = (torch.randn(padded, 4, generator=torch.Generator().manual_seed(817))*.1).to(dtype)
    prefix = 'model.language_model.layers.1.ple.ple_embedding'
    weights = {prefix+'.layer_multipliers': torch.tensor([7, 13, 19], dtype=torch.int64)}
    rows = padded // c.split_ngram_parts
    weights.update({f'{prefix}.ngram_embedding.shard_{i}.weight': table[i*rows:(i+1)*rows].clone()
                    for i in range(c.split_ngram_parts)})
    save_file(weights, path/'model.safetensors')
    (path/'config.json').write_text(json.dumps(raw))
    store = ShardedNGramTable(inspect_checkpoint(path), c)
    return store, table


def assert_states_equal(a, b):
    assert a.tokens_seen == b.tokens_seen and torch.equal(a.visible_keys, b.visible_keys)
    for x, y in zip(a.layers, b.layers):
        from venturi.models.flash_next_gdn import GDNState
        names = ('conv', 'recurrent') if isinstance(x.attention, GDNState) else ('raw_indexer_keys', 'keys', 'values')
        for name in names:
            torch.testing.assert_close(getattr(x.attention, name), getattr(y.attention, name), atol=0, rtol=0)
        if x.ple is not None:
            assert torch.equal(x.ple.conv, y.ple.conv) and x.ple.tokens_seen == y.ple.tokens_seen


def test_text_manual_vector_repeat_and_final_head():
    m = module()
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    rows = torch.randn(2, 3, 16)
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        logits, state = m(ids, ple_rows={1: rows})
        hidden = m.embed_tokens(ids).repeat(1, 1, 2)
        assert not torch.equal(hidden, m.embed_tokens(ids).repeat_interleave(2, -1))
        pos = m.position_embeddings(2, 3, device='cpu', dtype=hidden.dtype)
        mask = torch.ones(2, 1, 3, 3, dtype=torch.bool).tril()
        for i, layer in enumerate(m.layers):
            hidden, _ = layer(hidden, ple_rows=rows if i == 1 else None, position_embeddings=pos, attention_mask=mask)
        expected = m.lm_head(m.hyper_connection_mixer(hidden))
    assert torch.equal(logits, expected) and state.tokens_seen == 3


def test_request_partial_acceptance_replays_all_layers_hash_and_mask(tmp_path):
    table, _ = table_fixture(tmp_path)
    m = module()
    req, direct = FlashNextRequest(m, table), FlashNextRequest(m, table)
    with sdpa_kernel(SDPBackend.MATH):
        for r in (req, direct):
            p = r.preview([2, 3])
            r.commit(p)
        proposal = req.preview([4, 5, 6, 7], mask=[True, False, True, True])
        assert req.history == (2, 3) and req.state.tokens_seen == 2
        req.commit(proposal, 2)
        p = direct.preview([4, 5], mask=[True, False])
        direct.commit(p)
    assert req.history == ()  # masked PLE id became EOS before hashing
    assert_states_equal(req.state, direct.state)
    assert torch.equal(req.last_logits, direct.last_logits)
    assert req.state.tokens_seen == 4  # EOS is not a recurrent-cache reset
    snapshot = req.state
    snapshot.layers[0].attention.recurrent.zero_()
    assert not torch.equal(snapshot.layers[0].attention.recurrent, req.state.layers[0].attention.recurrent)
    with pytest.raises(ValueError, match='Stale'):
        req.commit(proposal)


def test_request_zero_acceptance_eos_reset_foreign_and_owned_logits(tmp_path):
    table, _ = table_fixture(tmp_path)
    m = module()
    a, b = FlashNextRequest(m, table), FlashNextRequest(m, table)
    with sdpa_kernel(SDPBackend.MATH):
        p = a.preview([2, 31, 5])
        expected_logits = p.logits[:, -1].clone()
        p.logits.zero_()  # public output is not a live internal cache
        with pytest.raises(ValueError, match='foreign'):
            b.commit(p)
        a.commit(p)
        assert a.history == (5,) and a.state.tokens_seen == 3
        assert torch.equal(a.last_logits, expected_logits)
        owned = a.last_logits
        owned.zero_()
        assert torch.equal(a.last_logits, expected_logits)
        before = a.state
        p = a.preview([7, 8])
        a.commit(p, 0)
        assert_states_equal(a.state, before)
        assert a.history == (5,)
        p = a.preview([7])
        a.reset()
        with pytest.raises(ValueError, match='Stale'):
            a.commit(p)
        assert a.history == () and a.state is None and a.last_logits is None


def test_failed_preview_or_prefix_replay_keeps_state_and_pending(tmp_path, monkeypatch):
    table, _ = table_fixture(tmp_path)
    m, req = module(), None
    req = FlashNextRequest(m, table)
    with sdpa_kernel(SDPBackend.MATH):
        p = req.preview([1, 2])
        req.commit(p)
        before = req.state
        old = req.preview([3, 4, 5])
        original = m.layers[-1].mlp.forward
        def fail(*args):
            raise RuntimeError('selected expert failed')
        monkeypatch.setattr(m.layers[-1].mlp, 'forward', fail)
        with pytest.raises(RuntimeError, match='expert failed'):
            req.preview([6])
        with pytest.raises(RuntimeError, match='expert failed'):
            req.commit(old, 1)
        assert req.history == (1, 2)
        assert_states_equal(req.state, before)
        monkeypatch.setattr(m.layers[-1].mlp, 'forward', original)
        req.commit(old, 1)
    assert req.history == (2, 3) and req.state.tokens_seen == 3


def test_request_reader_failure_and_superseded_ticket(tmp_path, monkeypatch):
    store, _ = table_fixture(tmp_path)
    reader = PreparedNGramTable(store).prepare()
    req = FlashNextRequest(module(), reader)
    with sdpa_kernel(SDPBackend.MATH):
        old = req.preview([2])
        new = req.preview([3])
        with pytest.raises(ValueError, match='Stale'):
            req.commit(old)
        reader.close()
        with pytest.raises(RuntimeError, match='closed'):
            req.preview([4])
        req.commit(new)
    assert req.history == (3,)


def test_concurrent_requests_share_reader_not_history(tmp_path):
    store, _ = table_fixture(tmp_path)
    reader = PreparedNGramTable(store).prepare()
    m = module()
    requests = [FlashNextRequest(m, reader) for _ in range(2)]
    def work(index):
        with sdpa_kernel(SDPBackend.MATH):
            p = requests[index].preview([index+2, 31, index+5])
            requests[index].commit(p)
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(work, range(2)))
    reader.close()
    assert [r.history for r in requests] == [(5,), (6,)]
    assert all(r.state.tokens_seen == 3 for r in requests)


def test_bounds_and_rejections_precede_embedding_work(tmp_path, monkeypatch):
    m = module()
    m.max_sequence = 1
    def forbidden(*args):
        raise AssertionError('must admit before embedding')
    monkeypatch.setattr(m.embed_tokens, 'forward', forbidden)
    with pytest.raises(ValueError, match='budget'):
        m(torch.tensor([[1, 2]]), ple_rows={1: torch.zeros(1, 2, 16)})
    store, _ = table_fixture(tmp_path)
    req = FlashNextRequest(module(), store)
    for ids in ([], [True], [-1], [32]):
        with pytest.raises(ValueError):
            req.preview(ids)
    with pytest.raises(ValueError, match='mask'):
        req.preview([1], mask=[1])
    with sdpa_kernel(SDPBackend.MATH):
        p = req.preview([2])
        for count in (True, -1, 2):
            with pytest.raises(ValueError, match='prefix'):
                req.commit(p, count)
        req.commit(p)


def test_plain_text_rotary_preserves_default_fp32_inverse_frequency_after_weight_cast():
    base = module()
    m = FlashNextTextCore(base.embed_tokens, base.layers, base.hyper_connection_mixer, base.lm_head,
                         rotary_dim=4).to(dtype=torch.bfloat16).eval()
    cos, sin = m.position_embeddings(2, 4100, device='cpu', dtype=torch.bfloat16)
    inv = 1.0/(10000000.**(torch.arange(0, 4, 2, dtype=torch.float32)/4))
    freq = torch.arange(4100).view(1, -1, 1).expand(2, -1, -1).float()*inv
    assert torch.equal(cos, torch.cat((freq.cos(), freq.cos()), -1).to(torch.bfloat16))
    assert torch.equal(sin, torch.cat((freq.sin(), freq.sin()), -1).to(torch.bfloat16))
    rounded = torch.arange(4100).view(1, -1, 1).float()*inv.to(torch.bfloat16).float()
    assert not torch.equal(cos[:1], torch.cat((rounded.cos(), rounded.cos()), -1).to(torch.bfloat16))
