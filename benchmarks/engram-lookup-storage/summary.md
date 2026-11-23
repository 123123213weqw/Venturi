# Engram sparse-lookup storage and decode

## Scope and qualification

The admitted `VerifiedIQ4NLRows` component: full-SHA admission, addressed
`os.pread` of IQ4_NL rows, IQ4_NL decode and scatter into per-request outputs.

Not measured, and not implied by any number here: n-gram construction
(`EngramHashState`) including tokenizer compression, `wkv` projection, gate
arithmetic, residual fusion, attention, MoE/MLP and generation. No GPU path is
exercised. This is not a model-TPS or llama.cpp comparison, and it uses a
generated fixture rather than DeepSeek-V4.1 weights.

The component is the shipped `sparse_lookup` storage prototype, not a tuned
production path: its decode oracle documents itself as "not a tuned production
CPU/GPU kernel". The measurements below characterise what it costs as written,
and separate that from what the storage underneath costs.

## Host

| Property | Value |
|---|---|
| Kernel | Linux 6.8.0-136-generic |
| CPU cores | 64 |
| RAM | 251 GiB |
| Storage | `/dev/sda`, `MR9361-8i`, `queue/rotational=1` (spinning RAID), ext4 `relatime` |

The storage is a rotational array. Every cold figure below is therefore a
spinning-disk figure and must not be carried to an NVMe host. Page-cache
eviction was available (`posix_fadvise(POSIX_FADV_DONTNEED)`), so cold and
resident conditions are genuinely different states rather than labels on the
same state.

The 251 GiB of RAM matters for interpretation: a 110 GB table would be resident
on this host, so this machine cannot exercise the regime that motivates the
design (a table much larger than RAM). Cold numbers here are produced by forced
eviction, not by capacity pressure.

## Fixture

Pass a new or empty `--output-directory`; the script refuses to overwrite
existing assets or measurements. Use `--keep` to retain the newly generated
GGUF for the vectorised probe.

`tools/benchmark_engram_lookup_storage.py` writes a single-tensor GGUF
(`blk.1.engram_embd.weight`, IQ4_NL) that passes the same `inspect_gguf` and
admission checks as a real asset: 120,000,000 rows x 128 = 8.05 GiB, 72 B rows,
tensor data offset 256. Row payloads are pseudo-random; zero-filled fixtures are
avoided because filesystem compression of zero pages makes residency
measurements meaningless.

## Matched unprofiled measurements

One warmup is implicit in each phase; ABBA order (cold, warm, warm, cold) with
at least two seconds per condition. Two phase observations per condition are not
sufficient to establish a sub-percent difference; the effects reported here are
multiples, not sub-percent.

| Tokens | Rows | Condition | Wall ms/call | ms/token | preads/call | Cold/warm |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 24 | resident | 1.00 | 0.955 | 24 | |
| 1 | 24 | evicted | 3.29 | 3.286 | 24 | 3.4x |
| 64 | 1,536 | resident | 64.3 | 1.004 | 1,536 | |
| 64 | 1,536 | evicted | 255.2 | 3.987 | 1,536 | 4.0x |
| 128 | 3,072 | resident | 125.9 | 0.984 | 3,071 | |
| 128 | 3,072 | evicted | 528.2 | 4.127 | 3,071 | 4.2x |
| 256 | 6,144 | resident | 249.6 | 0.975 | 6,144 | |
| 256 | 6,144 | evicted | 1,016.3 | 3.970 | 6,144 | 4.1x |
| 341 | 8,184 | resident | 341.6 | 1.002 | 8,184 | |
| 341 | 8,184 | evicted | 1,371.4 | 4.022 | 8,184 | 4.0x |

**Result: per-token cost is flat in sequence length.** 0.955 ms/token at T=1 and
1.002 ms/token at T=341 resident; 3.29 to 4.02 ms/token evicted. Prefill does not
amortise the per-row work, so the cost is set by rows fetched, not calls made.
Batching cannot reduce it either: see the concurrency table.

Eviction costs about 126 us per pread here (1,371.4 - 341.6 ms over 8,184
preads), close to the 132 us median raw cold read measured independently. The
cold/warm ratio of ~4 is a property of this rotational array; the same ratio on
NVMe would be far smaller and would make decode dominate even more strongly.

## Where the time actually goes

Attribution at T=341, table fully resident, each stage isolated and timed
best-of-3 against the real packed bytes:

| Stage | ms | Share of stages |
|---|---:|---:|
| `decode_iq4_nl_rows` (scalar oracle) | 308.70 | 95.4% |
| pread loop (8,182 reads) | 10.80 | 3.3% |
| `sorted(set(...))` | 3.05 | 0.9% |
| positions list | 0.77 | 0.2% |
| `b''.join` | 0.20 | 0.1% |
| `index_select` scatter | 0.17 | 0.1% |
| **stage sum** | **323.69** | |
| measured component total | 332.0 | |

**Result: when the table is resident, decode is 28.6x the read cost and 95% of
the work.** Raw preads on the same rows take 1.5 us each; the component spends
40 us per row, and 36.9 us of that is the scalar decode. Optimising prefetch,
coalescing or I/O on a resident table would move at most 3% of the time.

A vectorised decode over the same bytes reproduces the oracle **bitwise**
(`torch.equal` true, max absolute difference 0.0). Two observations of the same
work, both refusing to report timing unless the outputs match:

| Observation | Scalar ms | Vectorised ms | Scalar us/row | Vectorised us/row | Speedup |
|---|---:|---:|---:|---:|---:|
| ad hoc stage probe | 301.58 | 9.37 | 36.86 | 1.14 | 32.2x |
| `tools/probe_engram_decode_vectorised.py` | 303.95 | 9.97 | 37.15 | 1.22 | 30.5x |

**Result: about 30x, at bitwise-identical output.** The remaining spread is
run-to-run, not a difference between implementations.

Projection from the measured components, T=341:

| Configuration | ms/token | Lookup-only ceiling |
|---|---:|---:|
| shipped decode, resident | 1.00 | ~1,000 tok/s |
| shipped decode, evicted | 4.02 | ~250 tok/s |
| vectorised decode, resident | 0.098 | ~10,200 tok/s |
| vectorised decode, evicted | 3.12 | ~320 tok/s |

These ceilings cover the lookup alone and exclude `wkv`, gating, attention and
generation, so they are upper bounds on throughput, not throughput predictions.
They are extrapolations of measured stage times, not measured end-to-end rates.

## Output budget bounds the call shape

`lookup_many` rejects a call when `sum(requested rows) * width > max_values`, and
`__init__` caps `max_values` at `2**20`. The probe is executed every run: a
342-token call was rejected at `columns=24, width=128`, and 341 tokens is the
largest accepted call. At `columns=16` the cap allows exactly 512 tokens; above
that the caller must chunk. Any prefetch or batching design has to respect this
rather than assume one lookup per prefill.

| width | columns | max rows/call | max tokens/call |
|---:|---:|---:|---:|
| 128 | 16 | 8,192 | 512 |
| 128 | 24 | 8,192 | 341 |
| 128 | 32 | 8,192 | 256 |
| 256 | 24 | 4,096 | 170 |

The real `columns` is `(engram_max_ngram_size - 1) * engram_n_heads` from the
checkpoint; the shipped values were not available, so 24 is an assumption.

## Concurrency: the component serialises, and threads make it worse

| Threads | Single-row calls/s |
|---:|---:|
| 1 | 3,455 |
| 4 | 2,391 |
| 16 | 2,368 |
| 32 | 2,870 |

`lookup_many` holds `threading.RLock` across read, decode and scatter, and issues
one `pread` per coalesced run inside that lock. Concurrency therefore buys
nothing and contention costs about 30% at 4 and 16 threads. This is a property
of the class, not of the device: raw preads on this storage reached 87k IOPS at
16 threads on the local dry run. Any parallel prefetch must issue reads outside
this lock, which the current API does not permit.

## Coalescing does not pay

Measured from the component's own `last_read` and from the exact run
decomposition it uses (`sorted(set(...))`, then merge adjacent): merge rate is
0.00% at T=1, 64, 256 and 341, and 0.03% at T=128. At 120M rows the probability
that two of ~8,000 uniformly scattered ids are adjacent is negligible. Do not
budget for adjacent-row merging at this table size.

## Admission is CPU-bound, not disk-bound

Admission read 8.05 GiB at 259 MiB/s (31.84 s), while pure `hashlib.sha256` on
resident data reaches 316 MiB/s on this host. Admission is therefore about 82%
of raw SHA throughput and limited by hashing, not by the array. At this rate a
110 GB asset costs about **405 s (6.8 min) of single-threaded SHA** before the
first lookup, which is a real start-up cost that parallel or sharded hashing
would have to address.

## Provenance and reproduction

Host path: `/home/wzu/codex-build/venturi-engram-storage-20261120`.
Raw ABBA phases: `measurements/results.json`. Stage attribution and vectorisation
check: `measurements/decomposition.json`. Budget arithmetic:
`measurements/output-budget.json`.

```bash
PYTHONPATH=src python tools/benchmark_engram_lookup_storage.py \
  --output-directory /absolute/path/engram-lookup-storage \
  --rows 120000000 --width 128 --columns 24 --phase-seconds 2

PYTHONPATH=src python tools/probe_engram_decode_vectorised.py \
  --asset /absolute/path/asset/lookup.gguf --rows 120000000 --width 128
```

No CUDA or Rust compilation ran; this is CPU-only Python against a generated
fixture. Per this repository's policy, no compilation was run on the Mac, and
the local host contributed only the fixture-generator and admission correctness
checks (documented in the tool's own guard rails), never a timing figure.
