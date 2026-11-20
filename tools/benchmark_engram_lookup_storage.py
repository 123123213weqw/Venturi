"""Engram sparse-lookup storage and decode benchmark, NOT model inference.

Measures the admitted VerifiedIQ4NLRows component: full-SHA admission, addressed
IQ4_NL row reads, IQ4_NL decode and scatter. Hash/n-gram construction, wkv
projection, gating, attention and generation are NOT measured. This is a
storage/IO characterization, not a model-TPS or llama.cpp comparison.

The synthetic asset is generated locally with pseudo-random nibbles. Zero-filled
files are deliberately avoided: filesystem compression or deduplication of zero
pages makes residency and read-cost measurements meaningless.

Cold-phase numbers are refused when the platform cannot evict the page cache
(no os.posix_fadvise, e.g. macOS). Labelling resident measurements as "cold" is
the specific failure this guard exists to prevent.

The component serialises every lookup under an internal RLock, so the thread
sweep measures whether callers can overlap I/O at all, not peak device IOPS.
"""
from dataclasses import dataclass
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np

from venturi.container.gguf.gguf_rows import VerifiedIQ4NLRows
from venturi.container.model_assets import ModelAsset

BLOCK_BYTES = 18
WEIGHTS_PER_BLOCK = 32
GGUF_ALIGNMENT = 32
TYP_IQ4_NL = 20
TABLE_NAME = 'blk.1.engram_embd.weight'
REVISION = 'a' * 40
METADATA = (('general.architecture', 'deepseek41-engram'),
            ('deepseek41.engram.source', REVISION))
OUTPUT_VALUE_BUDGET = 2**20


def row_bytes_for(width):
    if width % WEIGHTS_PER_BLOCK or not WEIGHTS_PER_BLOCK <= width <= 2**16:
        raise ValueError('IQ4_NL width must be a positive multiple of 32')
    return width // WEIGHTS_PER_BLOCK * BLOCK_BYTES


def write_gguf(path, *, rows, width, seed=0, chunk_rows=1 << 15):
    """Write one IQ4_NL tensor into a GGUF the warehouse component admits."""
    row_bytes = row_bytes_for(width)
    blocks = width // WEIGHTS_PER_BLOCK
    header = b'GGUF' + struct.pack('<IQQ', 3, 1, len(METADATA))
    for key, value in METADATA:
        raw = value.encode()
        header += struct.pack('<Q', len(key)) + key.encode() + struct.pack('<IQ', 8, len(raw)) + raw
    name = TABLE_NAME.encode()
    header += struct.pack('<Q', len(name)) + name
    header += struct.pack('<IQQIQ', 2, width, rows, TYP_IQ4_NL, 0)
    header += bytes((-len(header)) % GGUF_ALIGNMENT)
    rng = np.random.default_rng(seed)
    chunk = np.zeros((chunk_rows, blocks, BLOCK_BYTES), dtype=np.uint8)
    chunk[..., 0:2] = np.array([0.5], dtype=np.float16).view(np.uint8)
    digest = hashlib.sha256()
    total = 0
    with open(path, 'wb') as stream:
        stream.write(header)
        digest.update(header)
        total += len(header)
        written = 0
        while written < rows:
            take = min(chunk_rows, rows - written)
            body = chunk[:take]
            body[..., 2:] = rng.integers(0, 256, size=(take, blocks, 16), dtype=np.uint8)
            raw = body.tobytes()
            stream.write(raw)
            digest.update(raw)
            total += len(raw)
            written += take
        stream.flush()
        os.fsync(stream.fileno())
    expected = len(header) + rows * row_bytes
    if total != expected:
        raise RuntimeError(f'payload size {total} != expected {expected}')
    return ModelAsset(path.name, total, digest.hexdigest(),
                      'https://huggingface.co/venturi-fixtures/rows/resolve/'
                      f'{REVISION}/{path.name}',
                      'venturi-fixtures/rows', REVISION, path.name, 'sparse_lookup')


def eviction_support():
    if hasattr(os, 'posix_fadvise') and hasattr(os, 'POSIX_FADV_DONTNEED'):
        return True, 'posix_fadvise(POSIX_FADV_DONTNEED)'
    return False, f'unsupported on {sys.platform}'


def evict(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


@dataclass
class Phase:
    seconds: float
    operations: int
    payload_bytes: int
    read_calls: int

    @property
    def ms_per_call(self):
        return self.seconds * 1e3 / self.operations if self.operations else float('nan')

    @property
    def read_calls_per_call(self):
        return self.read_calls / self.operations if self.operations else float('nan')


def timed(operation, store, *, min_seconds, max_operations=200000):
    started = time.perf_counter()
    operations = 0
    payload = 0
    calls = 0
    while True:
        operation()
        operations += 1
        stats = store.last_read
        if stats is not None:
            payload += stats.payload_bytes
            calls += stats.read_calls
        elapsed = time.perf_counter() - started
        if elapsed >= min_seconds or operations >= max_operations:
            return Phase(elapsed, operations, payload, calls)


def abba(store, cold, warm, *, min_seconds, capable):
    """Alternate A/B/B/A so monotone drift cannot be read as a treatment effect."""
    order = [('cold', cold), ('warm', warm), ('warm', warm), ('cold', cold)] if capable \
        else [('warm', warm)] * 4
    return [{'condition': kind,
             'seconds': (phase := timed(runner, store, min_seconds=min_seconds)).seconds,
             'operations': phase.operations,
             'payload_bytes': phase.payload_bytes,
             'read_calls': phase.read_calls,
             'ms_per_call': phase.ms_per_call,
             'read_calls_per_call': phase.read_calls_per_call}
            for kind, runner in order]


def main(argv=None):
    parser = argparse.ArgumentParser(description='Engram lookup storage benchmark')
    parser.add_argument('--output-directory', required=True)
    parser.add_argument('--rows', type=int, default=120_000_000,
                        help='table rows; 120M x 72 B = 8.0 GiB at width 128')
    parser.add_argument('--width', type=int, default=128, help='head_dim, multiple of 32')
    parser.add_argument('--columns', type=int, default=24,
                        help='(max_ngram_size-1) * n_heads, from the real config')
    parser.add_argument('--phase-seconds', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--keep', action='store_true')
    arguments = parser.parse_args(argv)

    import torch

    output = Path(arguments.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    workspace = output / 'asset'
    workspace.mkdir(exist_ok=True)
    asset_path = workspace / 'lookup.gguf'
    result = {'rows': arguments.rows, 'width': arguments.width,
              'columns': arguments.columns, 'phases': [], 'not_measured': [
                  'n-gram hash construction (EngramHashState)',
                  'wkv projection, gating and residual fuse',
                  'attention, MoE/MLP and generation; no GPU path is exercised',
                  'achieved DRAM bandwidth or page-cache hit counters']}

    capable, mechanism = eviction_support()
    result['eviction'] = {'supported': capable, 'mechanism': mechanism}
    if not capable:
        result['not_measured'].append(
            'cold (evicted) numbers: page-cache eviction unavailable on this host; '
            'every reported figure is a resident figure and none is labelled cold')

    max_rows = OUTPUT_VALUE_BUDGET // arguments.width
    result['output_budget'] = {'max_values': OUTPUT_VALUE_BUDGET,
                               'max_rows_per_call': max_rows,
                               'max_tokens_per_call': max_rows // arguments.columns}
    print(f'geometry: width={arguments.width} row={row_bytes_for(arguments.width)} B '
          f'columns={arguments.columns}', flush=True)
    print(f'component output budget: {max_rows:,} rows/call = '
          f'{max_rows // arguments.columns} tokens/call', flush=True)

    print(f'generating {arguments.rows:,} rows ...', flush=True)
    started = time.perf_counter()
    asset = write_gguf(asset_path, rows=arguments.rows, width=arguments.width,
                       seed=arguments.seed)
    result['generation_seconds'] = time.perf_counter() - started
    result['asset'] = {'bytes': asset.bytes, 'sha256': asset.sha256}
    print(f'  {asset.bytes / 2**30:.2f} GiB in {result["generation_seconds"]:.1f}s', flush=True)

    shapes = {TABLE_NAME: (arguments.rows, arguments.width)}
    required = dict(METADATA)
    if capable:
        evict(asset_path)
    started = time.perf_counter()
    store = VerifiedIQ4NLRows(asset, workspace, expected_shapes=shapes,
                              required_metadata=required, max_values=OUTPUT_VALUE_BUDGET)
    result['admission_seconds'] = time.perf_counter() - started
    result['admission_mib_per_s'] = asset.bytes / 2**20 / max(result['admission_seconds'], 1e-9)
    print(f'admission: {result["admission_seconds"]:.2f}s = '
          f'{result["admission_mib_per_s"]:.0f} MiB/s', flush=True)

    # Prove the budget bound instead of asserting it from source reading.
    over = max_rows // arguments.columns + 1
    try:
        store.lookup(TABLE_NAME, torch.zeros((1, over, arguments.columns), dtype=torch.int64))
        result['output_budget']['exceeding_call_raises'] = False
        print(f'  WARNING: a {over}-token call did not raise; the budget contract changed',
              flush=True)
    except ValueError:
        result['output_budget']['exceeding_call_raises'] = True
        print(f'  budget probe: {over}-token call rejected as expected', flush=True)

    limit = max_rows // arguments.columns
    tokens = sorted({t for t in (1, 64, 128, 256, limit) if 1 <= t <= limit})
    rng = np.random.default_rng(arguments.seed + 1)
    ids = {t: torch.from_numpy(rng.integers(0, arguments.rows,
                                            size=(1, t, arguments.columns), dtype=np.int64))
           for t in tokens}

    for t in tokens:
        def warm(t=t):
            store.lookup(TABLE_NAME, ids[t])
        def cold(t=t):
            if capable:
                evict(asset_path)
            store.lookup(TABLE_NAME, ids[t])
        phases = abba(store, cold, warm, min_seconds=arguments.phase_seconds, capable=capable)
        result['phases'].append({'tokens': t, 'requested_rows': t * arguments.columns,
                                 'phases': phases})
        for phase in phases:
            print(f'  tokens={t:<5} {phase["condition"]:<5} '
                  f'{phase["ms_per_call"]:>9.3f} ms/call  '
                  f'calls={phase["operations"]:>7}  '
                  f'preads/call={phase["read_calls_per_call"]:>8.0f}', flush=True)

    coalescing = []
    for t in tokens:
        flat = np.unique(ids[t].reshape(-1).numpy())
        adjacent = int((np.diff(flat) == 1).sum()) if len(flat) > 1 else 0
        runs = int(len(flat)) - adjacent
        coalescing.append({'tokens': t, 'requested': t * arguments.columns,
                           'unique': int(len(flat)), 'predicted_read_calls': runs,
                           'merge_rate': float((len(flat) - runs) / len(flat))})
    result['coalescing'] = coalescing
    measured = {p['tokens']: p['phases'][0]['read_calls_per_call']
                for p in result['phases'] if p['phases']}
    for entry in coalescing:
        entry['measured_read_calls'] = measured.get(entry['tokens'])
        entry['prediction_matches'] = (
            measured.get(entry['tokens']) is not None
            and abs(measured[entry['tokens']] - entry['predicted_read_calls']) <= 1)
    for entry in coalescing:
        print(f'  coalescing tokens={entry["tokens"]:<5} unique={entry["unique"]:>7,} '
              f'runs={entry["predicted_read_calls"]:>7,} '
              f'merge={entry["merge_rate"] * 100:>5.2f}%', flush=True)

    single = torch.from_numpy(np.unique(rng.integers(0, arguments.rows, size=256)))
    concurrency = []
    for threads in (1, 4, 16, 32):
        started = time.perf_counter()
        if threads == 1:
            for value in single.tolist():
                store.lookup(TABLE_NAME, torch.tensor([[value]]))
        else:
            def one(value):
                store.lookup(TABLE_NAME, torch.tensor([[value]]))
            with ThreadPoolExecutor(threads) as pool:
                list(pool.map(one, single.tolist()))
        elapsed = time.perf_counter() - started
        concurrency.append({'threads': threads, 'calls': len(single), 'seconds': elapsed,
                            'calls_per_s': len(single) / elapsed})
        print(f'  threads={threads:<3} {len(single):,} single-row calls  '
              f'{len(single) / elapsed:>8.0f} calls/s', flush=True)
    result['concurrency'] = concurrency

    store.close()
    result['store_closed'] = store.closed
    (output / 'measurements').mkdir(exist_ok=True)
    (output / 'measurements' / 'results.json').write_text(json.dumps(result, indent=1) + '\n')
    if not arguments.keep:
        for item in workspace.iterdir():
            item.unlink()
        workspace.rmdir()
    print(f'\nwrote {output / "measurements" / "results.json"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
