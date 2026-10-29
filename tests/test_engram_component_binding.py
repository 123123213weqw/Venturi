"""Generated tiny metadata fixtures; NO official weights or model generation."""
from dataclasses import replace
import hashlib
import json
import struct

import pytest
import torch

from venturi.model_assets import ModelAsset, ModelBundle
from venturi.models.engram_bindings import load_deepseek41_engram_hash
from venturi.models.engram_hash import EngramHashState
from test_gguf_metadata import write_gguf
from test_tokenizer_assets import MAP, tokenizer_json


MODEL = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'
REV = 'a'*40  # synthetic source identity; NEVER fetched or called official data


def config():
    return dict(architectures=['DeepseekV41ForCausalLM'], model_type='deepseek_v41',
        text_config=dict(model_type='deepseek_v41_text', vocab_size=len(MAP),
            engram_compressed_vocab_size=10, engram_pad_token_id=2,
            num_hidden_layers=4, hidden_size=8, hc_mult=4,
            engram_max_ngram_size=3, engram_n_heads=2, engram_head_dim=4,
            engram_layer_ids=[1, 3], engram_num_embeddings=[36, 88], rms_norm_eps=1e-20))


def header():
    return [('general.architecture', 8, 'deepseek41'),
        ('deepseek41.block_count', 4, 4), ('deepseek41.embedding_length', 4, 8),
        ('deepseek41.hyper_connection.count', 4, 4),
        ('deepseek41.engram_max_ngram_size', 4, 3), ('deepseek41.engram_n_heads', 4, 2),
        ('deepseek41.engram_head_dim', 4, 4), ('deepseek41.engram_pad_token_id', 4, 2),
        ('deepseek41.engram.present', 7, True),
        ('deepseek41.engram_compressed_vocab_size', 4, 10),
        ('deepseek41.engram.source', 8, REV),
        ('deepseek41.attention.layer_norm_rms_epsilon', 6, 1e-20),
        ('deepseek41.engram.layer_ids', 9, (4, [1, 3]))]


def tensors():
    return [('engram_token_map.weight', 26, (len(MAP),), MAP),
        ('engram_primes.weight', 27, (2, 4), (5, 7, 11, 13, 17, 19, 23, 29)),
        ('engram_offsets.weight', 27, (2, 4), (0, 5, 12, 23, 0, 17, 36, 59)),
        ('engram_multipliers.weight', 27, (2, 3), (3, 5, 7, 9, 11, 13))]


def json_asset(root, name, value):
    raw=json.dumps(value, ensure_ascii=False).encode();(root/name).write_bytes(raw)
    return ModelAsset(name, len(raw), hashlib.sha256(raw).hexdigest(),
        f'https://huggingface.co/{MODEL}/resolve/{REV}/{name}', MODEL, REV, name, 'metadata')


def inputs(root, *, cfg=None, hdr=None, data=None, tokenizer=None):
    pytest.importorskip('tokenizers')
    metadata = write_gguf(root, tensors() if data is None else data, header() if hdr is None else hdr)
    bundle = ModelBundle(MODEL, REV, 'GENERATED-NOT-WEIGHTS', (metadata,))
    return bundle, dict(config_asset=json_asset(root, 'config.json', config() if cfg is None else cfg),
        config_root=root, tokenizer_asset=json_asset(root, 'tokenizer.json', tokenizer_json() if tokenizer is None else tokenizer),
        tokenizer_root=root, metadata_asset=metadata, metadata_root=root)


def test_full_sha_config_layer_order_and_tokenizer_bind_private_hash_component(tmp_path):
    bundle, kwargs = inputs(tmp_path)
    bound = load_deepseek41_engram_hash(bundle, **kwargs)
    assert bound.token_identity.compared_stored_map and not bound.model_supported
    assert bound.hash_spec.layer_ids == (1, 3) and bound.hash_spec.table_rows == (36, 88)
    assert bound.hash_spec.token_map == MAP and bound.config_norm_epsilon == 1e-20
    assert bound.gguf_norm_epsilon == struct.unpack('<f', struct.pack('<f', 1e-20))[0]
    assert bound.config_norm_epsilon != bound.gguf_norm_epsilon  # do not hide precision conversion
    state = EngramHashState(bound.hash_spec, batch_size=1, max_sequence_tokens=4)
    ids = torch.tensor([[1, 2, 3]])
    full = state(ids, 0);state.reset()
    assert torch.equal(full, torch.cat([state(ids[:, i:i+1], i) for i in range(3)], dim=1))
    # Returned component is private; it is NOT a lease on later file contents.
    (tmp_path/'config.json').write_bytes(b'corrupted')
    assert bound.hash_spec.token_map == MAP
    with pytest.raises(ValueError):load_deepseek41_engram_hash(bundle, **kwargs)


@pytest.mark.parametrize('key,value',[
    ('num_hidden_layers', True), ('num_hidden_layers', 5), ('hidden_size', 9), ('hc_mult', 1),
    ('engram_n_heads', 3), ('engram_head_dim', 5), ('engram_max_ngram_size', 4),
    ('engram_pad_token_id', 3), ('engram_compressed_vocab_size', 11),
    ('engram_num_embeddings', [36, 89]), ('engram_num_embeddings', [36]),
    ('engram_layer_ids', [3, 1]), ('engram_layer_ids', [1, 1]), ('engram_layer_ids', [1, 4]),
    ('engram_layer_ids', [1, True]), ('model_type', 'qwen4_exp'),
    ('rms_norm_eps', 1e-6), ('rms_norm_eps', True), ('rms_norm_eps', 0.0),
    ('rms_norm_eps', -1e-20), ('rms_norm_eps', '1e-20'), ('rms_norm_eps', 10**400),
])
def test_authenticated_but_incompatible_config_is_not_accepted(tmp_path, key, value):
    cfg = config();cfg['text_config'][key] = value
    bundle, kwargs = inputs(tmp_path, cfg=cfg)
    with pytest.raises(ValueError):load_deepseek41_engram_hash(bundle, **kwargs)


@pytest.mark.parametrize('key,typ,value',[
    ('deepseek41.engram.layer_ids', 9, (4, [3, 1])),
    ('deepseek41.engram.layer_ids', 9, (6, [1., 3.])),
    ('deepseek41.engram_head_dim', 7, True),
    ('deepseek41.engram.present', 7, False),
    ('deepseek41.engram.source', 8, 'b'*40),
    ('general.architecture', 8, 'deepseek4'),
    ('deepseek41.attention.layer_norm_rms_epsilon', 6, 1e-6),
    ('deepseek41.attention.layer_norm_rms_epsilon', 6, 0.),
])
def test_valid_sha_header_is_not_enough_for_profile_admission(tmp_path, key, typ, value):
    hdr = [(k, t, v) if k != key else (key, typ, value) for k, t, v in header()]
    bundle, kwargs = inputs(tmp_path, hdr=hdr)
    with pytest.raises(ValueError):load_deepseek41_engram_hash(bundle, **kwargs)


def test_missing_header_layer_order_and_scalar_fail_closed(tmp_path):
    for key in ('deepseek41.engram.layer_ids', 'deepseek41.engram.source'):
        bundle, kwargs = inputs(tmp_path, hdr=[item for item in header() if item[0] != key])
        with pytest.raises(ValueError):load_deepseek41_engram_hash(bundle, **kwargs)


def test_valid_sha_same_range_token_map_is_compared_not_silently_repaired(tmp_path):
    data=tensors();wrong=list(MAP);wrong[2]=0
    data[0]=(data[0][0], 26, (len(MAP),), tuple(wrong))
    bundle, kwargs = inputs(tmp_path, data=data)
    with pytest.raises(ValueError, match='token 2.*no repair'):
        load_deepseek41_engram_hash(bundle, **kwargs)


@pytest.mark.parametrize('name', ['config_asset', 'tokenizer_asset'])
def test_other_source_revision_is_rejected_before_filesystem_access(tmp_path, name):
    bundle, kwargs = inputs(tmp_path)
    asset=kwargs[name]
    kwargs[name]=replace(asset, revision='b'*40, url=asset.url.replace(REV, 'b'*40))
    (tmp_path/asset.filename).unlink()
    with pytest.raises(ValueError, match='source identity'):load_deepseek41_engram_hash(bundle, **kwargs)


def test_nonmember_metadata_source_and_other_architecture_are_rejected(tmp_path):
    bundle, kwargs=inputs(tmp_path)
    with pytest.raises(ValueError, match='bundle member'):
        load_deepseek41_engram_hash(bundle, **(kwargs|{'metadata_asset':replace(kwargs['metadata_asset'], sha256='0'*64)}))
    with pytest.raises(ValueError, match='bundle required'):
        load_deepseek41_engram_hash(replace(bundle, model='some/model'), **kwargs)
    cfg=config();cfg['architectures']=['Qwen4ForCausalLM']
    bundle, kwargs=inputs(tmp_path, cfg=cfg)
    with pytest.raises(ValueError, match='architecture/config'):load_deepseek41_engram_hash(bundle, **kwargs)
