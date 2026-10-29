"""Local tokenizer.json handling, independent of models and tensor frameworks."""

import hashlib
from pathlib import Path


class LocalTokenizer:
    """Load only tokenizer.json, with no network access or model Python execution."""

    def __init__(self, path: str | Path, *, vocab_size: int):
        try:
            from tokenizers import Tokenizer
        except ImportError as error:
            raise RuntimeError(
                "Install Venturi's generation extra: pip install -e '.[generate]'"
            ) from error
        self.path = Path(path).expanduser().resolve(strict=True)
        content = self.path.read_bytes()
        self.sha256 = hashlib.sha256(content).hexdigest()
        try:
            self._tokenizer = Tokenizer.from_str(content.decode("utf-8"))
        except Exception as error:
            raise ValueError(f"Invalid tokenizer.json: {error}") from error
        # Disable serialized serving policy: generation supplies one raw prompt,
        # without truncation, padding, or postprocessor-inserted special tokens.
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        self._ids = frozenset(
            self._tokenizer.get_vocab(with_added_tokens=True).values()
        )
        if not self._ids or min(self._ids) < 0 or max(self._ids) >= vocab_size:
            raise ValueError(
                "Tokenizer vocabulary does not fit the model embedding vocabulary"
            )
        self.vocab_size = vocab_size

    def encode(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise ValueError("prompt must be a string")
        try:
            return tuple(self._tokenizer.encode(text, add_special_tokens=False).ids)
        except Exception as error:
            raise ValueError(f"Tokenizer encoding failed: {error}") from error

    def decode(self, ids) -> str:
        if any(token not in self._ids for token in ids):
            raise ValueError("Model emitted a token ID absent from tokenizer.json")
        try:
            return self._tokenizer.decode(list(ids), skip_special_tokens=True)
        except Exception as error:
            raise ValueError(f"Tokenizer decoding failed: {error}") from error
