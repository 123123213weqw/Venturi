"""Pinned-source CPU tokenizer normalization qualification, NOT stored-map/model validation.

Read only the explicit small official config/tokenizer assets. Extract one
SHA-pinned pure function, never import upstream's module or regenerate checkpoint
primes/multipliers. No network, GPU, weights or inference speed measurement.
"""
import argparse
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct

from venturi.model_assets import ModelAsset
from venturi.model_config import read_verified_json_asset
from venturi.tokenizer_assets import _compressed_lookup


REPOSITORY = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'
REVISION = '1bae0e1088993b9b82de08f59de5be5a34e42170'
SOURCE_SHA256 = '11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897'
ASSETS = (
    ('config.json', 3311, '8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879'),
    ('tokenizer.json', 6367257, 'c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'),
)


def _source_function(raw):
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('Pinned official Engram source SHA256 differs')
    tree = ast.parse(raw)
    selected = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'build_compressed_token_map')
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[])),
                 '<SHA-pinned official normalization function>', 'exec'), scope)
    return scope[selected.name]


def _compare(reference, data, label, expected_counts=None, expected_map=None):
    from tokenizers import Tokenizer
    backend = Tokenizer.from_str(json.dumps(data, ensure_ascii=False, allow_nan=False))
    class Wrapper:
        backend_tokenizer = backend
        def __len__(self):return backend.get_vocab_size(with_added_tokens=True)
    expected, count = reference(Wrapper())
    actual, actual_count, version = _compressed_lookup(data)
    exact = actual == tuple(expected) and count == actual_count
    counts = (len(actual), actual_count)
    if expected_counts is not None:exact = exact and counts == expected_counts
    if expected_map is not None:exact = exact and actual == expected_map
    digest = hashlib.sha256(b''.join(struct.pack('<i', v) for v in actual)).hexdigest()
    return dict(label=label, vocabulary_size=counts[0], compressed_vocabulary_size=counts[1],
        byte_exact=exact, computed_map_i32_sha256=digest, tokenizers_version=version,
        compared_checkpoint_stored_map=False)


def qualify(source, tokenizer, config):
    from tokenizers import Tokenizer, decoders, models
    reference = _source_function(source)
    tokens = ['<unk>', ' The', 'the', 'THE', ' ', '', 'é', 'Ｅ', 'e', 'E',
              '\ufffdA', '\ufffda', '<special>', 'ß', 'ss', '\t', '\n', ' \t \n', '\ue000']
    word = Tokenizer(models.WordLevel({t:i for i,t in enumerate(tokens)}, unk_token='<unk>'))
    expected = (0,1,1,1,2,3,4,4,4,4,5,6,7,8,9,2,2,2,2)
    byte_tokens = ['<unk>', 'Ã', '©', 'a', 'A', 'Ġ', 'é', 'ß']
    byte = Tokenizer(models.BPE({t:i for i,t in enumerate(byte_tokens)}, [], unk_token='<unk>'))
    byte.decoder = decoders.ByteLevel()
    text = config['text_config']
    cases = [_compare(reference, json.loads(word.to_str()), 'synthetic-unicode-whitespace',
                      (len(tokens), 10), expected),
             _compare(reference, json.loads(byte.to_str()), 'synthetic-byte-decoder-partials'),
             _compare(reference, tokenizer, 'actual-pinned-public-tokenizer',
                      (text['vocab_size'], text['engram_compressed_vocab_size']))]
    return cases


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-source', required=True)
    p.add_argument('--assets-root', required=True, help='Only pinned config.json and tokenizer.json, NOT model weights')
    p.add_argument('--output', required=True)
    a = p.parse_args(argv)
    source_path = Path(a.reference_source)
    if source_path.stat().st_size != 8138:raise ValueError('Pinned source size differs')
    source = source_path.read_bytes()
    assets = [ModelAsset(name, size, sha,
        f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}',
        REPOSITORY, REVISION, name, 'metadata') for name,size,sha in ASSETS]
    config, tokenizer = [read_verified_json_asset(asset, a.assets_root, max_bytes=32*2**20)
                         for asset in assets]
    cases = qualify(source, tokenizer, config)
    result = dict(scope='CPU normalization and raw tokenizer identity only; NOT stored checkpoint map, lookup weights, full model correctness, GPU or TPS',
        official_repository=REPOSITORY, official_revision=REVISION, official_source_sha256=SOURCE_SHA256,
        assets=[asdict(asset) for asset in assets], all_byte_exact=all(c['byte_exact'] for c in cases),
        real_checkpoint_map_compared=False, cases=cases)
    with Path(a.output).open('x') as out:out.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'assets'}))
    if not result['all_byte_exact']:raise SystemExit(1)


if __name__ == '__main__':main()
