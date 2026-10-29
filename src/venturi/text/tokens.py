"""Shared token-ID validation; preserves the existing greedy input policy."""

from typing import Sequence


def validate_token_ids(ids: Sequence[int], vocab_size: int, *, allow_empty=False):
    if not isinstance(ids, (list, tuple)):
        raise ValueError("token IDs must be a list or tuple of integers")
    if not ids and not allow_empty:
        raise ValueError("At least one input token is required")
    if any(type(token) is not int or not 0 <= token < vocab_size for token in ids):
        raise ValueError(f"token IDs must be integers in [0,{vocab_size})")
