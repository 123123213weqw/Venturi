"""Offline DeepSeek41 Engram hash component binding, NOT a model loader.

Authenticate caller-trusted immutable config/tokenizer assets, fully hash the
containing GGUF, then require stored integer parameters and layer order to agree.
No table reads, quantized projections, GPU allocation, download or map repair.
The manifest is a trust input: a matching self-supplied SHA is not proof of
upstream authorship. Only the component below is qualified, not the whole model.
"""
from dataclasses import dataclass
import math
import struct

from venturi.gguf_metadata import read_verified_integer_metadata
from venturi.model_assets import ModelAsset, ModelBundle
from venturi.model_config import read_verified_json_asset
from venturi.tokenizer_assets import CompressedTokenIdentity, verify_deepseek41_token_map
from venturi.models.engram_hash import EngramHashSpec, bind_engram_hash_metadata


_MODEL = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'
_NAMES = ('engram_token_map.weight', 'engram_primes.weight',
          'engram_offsets.weight', 'engram_multipliers.weight')
_LAYER_KEY = 'deepseek41.engram.layer_ids'


@dataclass(frozen=True)
class EngramComponentBinding:
    hash_spec: EngramHashSpec
    token_identity: CompressedTokenIdentity
    config_asset: ModelAsset
    tokenizer_asset: ModelAsset
    metadata_asset: ModelAsset
    metadata_header_sha256: str
    head_dim: int
    hidden_size: int
    residual_copies: int
    config_norm_epsilon: float
    gguf_norm_epsilon: float

    @property
    def model_supported(self):
        return False  # integer hash component only, not table weights or runtime


def _integer(config, name, low, high):
    value = config.get(name)
    if type(value) is not int or not low <= value <= high:
        raise ValueError('Invalid bounded config integer: ' + name)
    return value


def load_deepseek41_engram_hash(bundle, *, config_asset, config_root,
                               tokenizer_asset, tokenizer_root,
                               metadata_asset, metadata_root):
    """Explicit offline full-SHA admission; never accepts a partial/header cache.

    Hashing metadata scans the ENTIRE containing asset. Call only against healthy
    storage; do not run per decode or against the quarantined failed root SSD.
    Returned private parameters survive later file changes; they do not lease or
    certify the still-unloaded sparse lookup/projection weights.
    """
    if not isinstance(bundle, ModelBundle) or bundle.model != _MODEL:
        raise ValueError('Explicit DeepSeek41-Flash-1017 bundle required')
    for asset, name in ((config_asset, 'config.json'), (tokenizer_asset, 'tokenizer.json')):
        if (not isinstance(asset, ModelAsset) or asset.repository != bundle.model
                or asset.revision != bundle.official_revision or asset.filename != name
                or asset.source_filename != name or asset.role != 'metadata'):
            raise ValueError('Config/tokenizer source identity must match official bundle revision')
    if not isinstance(metadata_asset, ModelAsset) or metadata_asset not in bundle.files:
        raise ValueError('Integer metadata source must be an explicit bundle member')
    config = read_verified_json_asset(config_asset, config_root)
    if (config.get('architectures') != ['DeepseekV41ForCausalLM']
            or config.get('model_type') != 'deepseek_v41'
            or not isinstance(config.get('text_config'), dict)):
        raise ValueError('DeepSeek41 architecture/config profile differs')
    text = config['text_config']
    if text.get('model_type') != 'deepseek_v41_text':
        raise ValueError('DeepSeek41 text configuration required')
    vocab = _integer(text, 'vocab_size', 1, 2**20)
    compressed = _integer(text, 'engram_compressed_vocab_size', 1, vocab)
    pad = _integer(text, 'engram_pad_token_id', 0, vocab-1)
    layers = _integer(text, 'num_hidden_layers', 1, 2**16)
    width = _integer(text, 'hidden_size', 1, 2**20)
    copies = _integer(text, 'hc_mult', 1, 64)
    ngram = _integer(text, 'engram_max_ngram_size', 2, 8)
    heads = _integer(text, 'engram_n_heads', 1, 64)
    head_dim = _integer(text, 'engram_head_dim', 1, 2**16)
    layer_ids = text.get('engram_layer_ids')
    rows = text.get('engram_num_embeddings')
    if (not isinstance(layer_ids, list) or not 1 <= len(layer_ids) <= 64
            or any(type(x) is not int or not 0 <= x < layers for x in layer_ids)
            or len(set(layer_ids)) != len(layer_ids)
            or not isinstance(rows, list) or len(rows) != len(layer_ids)
            or any(type(x) is not int or not 1 <= x < 2**63 for x in rows)):
        raise ValueError('Explicit ordered Engram layers/table rows required')
    eps = text.get('rms_norm_eps')
    if type(eps) not in (int, float) or not 0 < eps < 1 or not math.isfinite(eps):
        raise ValueError('Explicit finite positive RMS epsilon required')
    eps = float(eps)
    metadata = read_verified_integer_metadata(metadata_asset, metadata_root, _NAMES,
                                               header_array_names=(_LAYER_KEY,))
    if metadata.header_array(_LAYER_KEY).values != tuple(layer_ids):
        raise ValueError('Verified header/config Engram layer order differs')
    scalars = dict(metadata.scalars)
    expected = {
        'general.architecture': 'deepseek41',
        'deepseek41.block_count': layers,
        'deepseek41.embedding_length': width,
        'deepseek41.hyper_connection.count': copies,
        'deepseek41.engram_max_ngram_size': ngram,
        'deepseek41.engram_n_heads': heads,
        'deepseek41.engram_head_dim': head_dim,
        'deepseek41.engram.present': True,
        'deepseek41.engram_pad_token_id': pad,
        'deepseek41.engram_compressed_vocab_size': compressed,
        'deepseek41.engram.source': bundle.official_revision,
    }
    for key, value in expected.items():
        if type(scalars.get(key)) is not type(value) or scalars[key] != value:
            raise ValueError('Verified header/config scalar differs: ' + key)
    stored_eps = scalars.get('deepseek41.attention.layer_norm_rms_epsilon')
    # GGUF stores this scalar as F32. Do not use a loose absolute tolerance that
    # would accept 1e-6 in place of 1e-20; preserve both representations explicitly.
    rounded = struct.unpack('<f', struct.pack('<f', eps))[0]
    if type(stored_eps) is not float or stored_eps not in (eps, rounded) or stored_eps <= 0:
        raise ValueError('Verified header/config RMS epsilon differs')
    spec = bind_engram_hash_metadata(metadata, layer_ids=layer_ids, max_ngram_size=ngram,
        n_heads=heads, compressed_vocab_size=compressed, pad_token_id=pad,
        vocab_size=vocab, table_rows=rows)
    tokenizer = read_verified_json_asset(tokenizer_asset, tokenizer_root, max_bytes=32*2**20)
    identity = verify_deepseek41_token_map(tokenizer, spec.token_map, compressed)
    return EngramComponentBinding(spec, identity, config_asset, tokenizer_asset,
        metadata_asset, metadata.header_sha256, head_dim, width, copies, eps, stored_eps)
