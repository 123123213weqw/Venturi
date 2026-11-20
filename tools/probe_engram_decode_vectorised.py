"""Vectorised IQ4_NL row decode versus the scalar oracle, with an equality gate.

The scalar decode in gguf_quantization.decode_iq4_nl_rows is documented as an
oracle rather than a tuned kernel; this probe measures what a vectorised decode
would cost and refuses to report a speedup unless the output is bitwise
identical to the oracle on the same bytes.

Scope: decoding only. Reads, admission, wkv projection, gating and generation
are not exercised. No GPU path and no model weights are involved.
"""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from venturi.container.gguf.gguf_inventory import inspect_gguf
from venturi.container.gguf.gguf_quantization import _IQ4_NL_VALUES, decode_iq4_nl_rows

BLOCK_BYTES = 18
WEIGHTS_PER_BLOCK = 32
LUT = np.array(_IQ4_NL_VALUES, dtype=np.int16)


def decode_vectorised(raw, width):
    """Same arithmetic as the oracle, expressed as array operations.

    The oracle unpacks each F16 scale to a Python float (F64 arithmetic) and
    rounds the product into F32 at tensor construction. Doing the multiply in
    F64 and casting once reproduces that bit for bit.
    """
    row_bytes = (width // WEIGHTS_PER_BLOCK) * BLOCK_BYTES
    if len(raw) % row_bytes or not raw:
        raise ValueError('Complete nonempty IQ4_NL rows required')
    rows = len(raw) // row_bytes
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(rows, width // WEIGHTS_PER_BLOCK, BLOCK_BYTES)
    scales = np.ascontiguousarray(blocks[:, :, 0:2]).view(np.float16)[:, :, 0].astype(np.float64)
    packed = blocks[:, :, 2:]
    low = LUT[packed & 15].astype(np.float64)
    high = LUT[packed >> 4].astype(np.float64)
    values = np.concatenate([low, high], axis=-1) * scales[:, :, None]
    return torch.from_numpy(values.reshape(rows, width).astype(np.float32))


def best_of(fn, repeats):
    best = float('inf')
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - started) * 1e3)
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(description='IQ4_NL vectorised decode probe')
    parser.add_argument('--asset', required=True, help='GGUF holding the lookup table')
    parser.add_argument('--table', default='blk.1.engram_embd.weight')
    parser.add_argument('--rows', type=int, default=120_000_000)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--request-rows', type=int, default=8184,
                        help='rows per call; 341 tokens x 24 columns by default')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2)
    arguments = parser.parse_args(argv)

    path = Path(arguments.asset)
    row_bytes = (arguments.width // WEIGHTS_PER_BLOCK) * BLOCK_BYTES
    offset = {item.name: item.offset for item in
              inspect_gguf(path, expected_bytes=path.stat().st_size).tensors}[arguments.table]
    ids = np.unique(np.random.default_rng(arguments.seed).integers(
        0, arguments.rows, size=arguments.request_rows))
    handle = os.open(path, os.O_RDONLY)
    try:
        raw = b''.join(os.pread(handle, row_bytes, offset + int(value) * row_bytes)
                       for value in ids)
    finally:
        os.close(handle)
    print(f'{len(ids):,} rows, {len(raw):,} packed bytes, data offset {offset}')

    oracle = decode_iq4_nl_rows(raw, arguments.width, max_values=2**20)
    vectorised = decode_vectorised(raw, arguments.width)
    identical = torch.equal(oracle, vectorised)
    print(f'bitwise identical to the scalar oracle: {identical}')
    if not identical:
        print('refusing to time a non-identical decode; '
              f'max absolute difference {float((oracle - vectorised).abs().max())}')
        return 1

    scalar_ms = best_of(lambda: decode_iq4_nl_rows(raw, arguments.width, max_values=2**20),
                        arguments.repeats)
    vector_ms = best_of(lambda: decode_vectorised(raw, arguments.width), arguments.repeats)
    rows = len(ids)
    print(f'  scalar oracle  {scalar_ms:>8.2f} ms  {scalar_ms / rows * 1000:>6.2f} us/row')
    print(f'  vectorised     {vector_ms:>8.2f} ms  {vector_ms / rows * 1000:>6.2f} us/row')
    print(f'  speedup        {scalar_ms / vector_ms:>8.1f}x')
    print('\nDecoding only. Reads, admission, wkv projection, gating and generation '
          'are not measured.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
