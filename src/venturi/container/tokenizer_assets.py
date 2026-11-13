"""DeepSeek41 tokenizer/map preparation; no model code or implicit hash repair.

Use only an explicitly authenticated tokenizer JSON. Recompute normalization for
comparison, NEVER return/install a replacement map on mismatch. Inspection alone
is not validation of the checkpoint's stored map. No downloads/autoinstall.
"""
from dataclasses import dataclass
import hashlib
import json
import struct


@dataclass(frozen=True)
class CompressedTokenIdentity:
    vocabulary_size: int
    compressed_vocabulary_size: int
    map_i32_sha256: str
    tokenizers_version: str
    compared_stored_map: bool
    normalization_profile: str = 'deepseek41_nfkc_nfd_accent_lower_ascii_whitespace_v1'


def _compressed_lookup(data):
    if not isinstance(data,dict):raise ValueError('Explicit tokenizer JSON object required')
    try:
        import tokenizers
        from tokenizers import Regex,Tokenizer,normalizers
    except ImportError as e:
        raise RuntimeError('Optional prebuilt tokenizers package required; no automatic installation') from e
    serialized=json.dumps(data,ensure_ascii=False,allow_nan=False)
    if len(serialized.encode())>32*2**20:raise ValueError('Tokenizer JSON exceeds preparation budget')
    backend=Tokenizer.from_str(serialized)
    count=backend.get_vocab_size(with_added_tokens=True)
    if not 1<=count<=2**20:raise ValueError('Bounded nonempty tokenizer vocabulary required')
    sentinel='\ue000'
    normalizer=normalizers.Sequence([
        normalizers.NFKC(),normalizers.NFD(),normalizers.StripAccents(),normalizers.Lowercase(),
        normalizers.Replace(Regex(r'[ \t\r\n]+'),' '),normalizers.Replace(Regex(r'^ $'),sentinel),
        normalizers.Strip(),normalizers.Replace(sentinel,' '),
    ])
    keys={};lookup=[]
    for token in range(count):
        raw=backend.id_to_token(token)
        if raw is None:raise ValueError('Tokenizer IDs are not contiguous')
        text=backend.decode([token],skip_special_tokens=False)
        if '\ufffd' in text:key=raw  # preserve partial UTF-8 byte-token raw form
        else:
            normalized=normalizer.normalize_str(text)
            key=normalized if normalized else text
        if key not in keys:keys[key]=len(keys)
        lookup.append(keys[key])
    return tuple(lookup),len(keys),tokenizers.__version__


def _identity(lookup,count,version,compared):
    h=hashlib.sha256()
    for value in lookup:h.update(struct.pack('<i',value))
    return CompressedTokenIdentity(len(lookup),count,h.hexdigest(),version,compared)


def inspect_deepseek41_tokenizer(data):
    """Comparison-only digest; does NOT qualify or replace a stored token map."""
    lookup,count,version=_compressed_lookup(data)
    return _identity(lookup,count,version,False)


def verify_deepseek41_token_map(data,stored_map,compressed_vocab_size):
    """Require exact per-ID normalization agreement, with no repair/fallback."""
    if (type(compressed_vocab_size) is not int or not 1<=compressed_vocab_size<=2**20
            or not isinstance(stored_map,(tuple,list)) or not 1<=len(stored_map)<=2**20
            or any(type(v) is not int or not 0<=v<compressed_vocab_size for v in stored_map)):
        raise ValueError('Explicit bounded stored compressed map/vocabulary required')
    computed,count,version=_compressed_lookup(data)
    if len(stored_map)!=len(computed) or count!=compressed_vocab_size:
        raise ValueError('Tokenizer/stored map vocabulary identity differs')
    for token,(expected,got) in enumerate(zip(stored_map,computed)):
        if expected!=got:
            raise ValueError(f'Stored compressed map mismatch at token {token}: stored={expected},computed={got}; no repair')
    return _identity(computed,count,version,True)
