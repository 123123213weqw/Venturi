# Explicit native prefill/decode disaggregation

This is an opt-in Venturi-owned execution path, not an external whole-model
engine, a production route switch, TP4/TP8, or cross-host serving. The generic
runtime uses only the standard library. The first adapter is resident native
Qwen Q8 with TP1, static KV, FP16/BF16 activations and FP32 GDN state.

## Boundaries

```text
request queue -> bounded destination reservation -> prefill target pool
                                                  |
                         RequestState + explicit StateTransport
                                                  |
                                          decode target pool -> token events
```

- `runtime.state`: immutable checkpoint/layout/precision/arithmetic identity,
  component descriptors, opaque leased `RequestState`, and `TransferReceipt`.
- `runtime.pd.PDScheduler`: model-independent routing, slot ownership, chunked
  prefill, per-step grouped decode, cancellation/deadlines and completion futures.
- `PDWorkerPool`: one inference thread per explicitly supplied P/D target. Idle
  workers wait on a condition variable; no import starts threads or a device.
- `models.qwen_pd`: the architecture adapter. It exports valid KV prefixes,
  convolution histories and **FP32 recurrent matrices**, not just attention KV.
- `qwen_handoff.cuh`: additive native C API for validated cross-device state
  copy/commit. No model math or existing kernel dispatch is replaced.

Other architectures must declare their actual state through an adapter. They do
not gain PD support simply because they implement a linear operator. The shared
runtime never imports Qwen, Torch, a CUDA library, model loaders or an HTTP stack.

## Request and transfer ownership

The scheduler reserves a compatible D slot before admitting a request to P. This
caps outstanding handoffs and applies backpressure instead of accumulating an
unbounded queue of GPU state. The request must fit `prompt + output budget` on
both sides. The pending queue is bounded separately. Admission prefers a lightly
loaded compatible decoder and a fitting P slot; GPU topology does not override
measured state-transfer costs automatically.

Each P tick spends at most that target's prepared row budget on rotating prompt
chunks. Only the final prompt chunk is sampled. Each D tick groups one input
token from each active request on that decoder. Independent P/D lanes can run
concurrently; requests do not wait for an unrelated cohort to finish.

P samples the **first output token**, but that token has not yet been consumed
into KV/GDN state. State is copied at `position == prompt length`; D's first
input is exactly that pending token. The token event is published after a valid
transfer receipt. A one-token output or EOS on the first token finishes on P
without an unnecessary handoff. There is no sampling RNG/MTP-state migration:
this version is ordinary greedy decoding only.

Export is an opaque view/lease, not CPU materialization or a serialized network
message. Reset or append invalidates its epoch. Producer, weights and buffers
must stay alive and unmodified until transfer returns. Destination must be fresh.
Checkpoint/precision/arithmetic, component geometry, exact pinned source files,
capacity and cursor are checked before mutation. Only valid attention prefixes
are copied, never the entire reserved context capacity. A partial native import
poisons the destination; cursor commit happens only after copies synchronize and
source guards pass. Reuse requires an explicit reset.

## Explicit transports and current limitations

- `NativeQwenStateTransport(mode="host")` uses at most 8 MiB of portable pinned
  staging per handoff. This moves **state**, not weights or model computation.
- `mode="peer"` uses `cudaMemcpyPeerAsync` and enables process-local CUDA peer
  access when supported. Unsupported access fails; it never falls back to host.
- Both modes currently require different devices, the same process/native
  library and identical pinned checkpoint files. This is **not CUDA IPC**, NIXL,
  RDMA, a Tailscale KV transport, or a cross-machine worker protocol.
- Transfer is synchronous and locks participating batch/model owners. In
  particular, destination decoding can wait for a state copy. Copy/compute
  overlap, reusable transfer arenas and per-slot asynchronous ACKs remain
  follow-on optimizations, not implemented performance claims.
- No TP/PP/MTP, tiered KV, prefix reuse, precision conversion, state compression,
  automatic backend switch, HTTP/SSE serving, production credentials or model
  download is introduced. Existing generation/scheduling/default routes remain.

Cancellation and request deadlines take effect at safe boundaries; they do not
interrupt an in-flight CUDA operation. A cancelled transfer retains both leases
until copying finishes. A lane failure fails closed, with other in-flight lanes
retaining state until their own boundaries. `PDWorkerPool.close(timeout=...)`
joins all owners before resetting targets. A timeout is **not** a stopped owner:
retain batches/models and close/join again; never free them while workers live.
`pool.info()` exposes alive-worker and failure state.

## Explicit construction

```python
from venturi.models.qwen_pd import NativeQwenPDWorker, NativeQwenStateTransport
from venturi.runtime import PDScheduler, PDWorkerPool

# p_model and d_model are explicitly loaded from the SAME verified checkpoint
# using the SAME native library, on different devices. No implicit model load.
p_batch = p_model.create_batch([4096] * 4)
d_batch = d_model.create_batch([4096] * 4)
scheduler = PDScheduler(
    [NativeQwenPDWorker(p_batch)], [NativeQwenPDWorker(d_batch)],
    NativeQwenStateTransport(mode="host"), prefill_chunk_size=32,
)
pool = PDWorkerPool(scheduler).start()
ticket = pool.submit(prompt_ids, max_new_tokens=128, eos_token_ids=eos_ids)
completion = ticket.future.result()
pool.close()  # Must join successfully before closing caller-owned batches/models.
```

Each model must have enough prepared rows for its slot count. The example does
not promise a model/context fits. Four P targets plus four D targets creates a
4P+4D pool, **not** two TP4 engines. Optional `event_sink` receives token events
on inference threads and must be nonblocking; it is not an SSE implementation.
Applications may instead own individual `tick_prefill(i)` / `tick_decode(i)`
calls and dispatch returned events outside inference work.

## Qualification and benchmarking

`tests/test_pd_scheduler.py` uses independent integer recurrence/prefix fixtures
for routing, state isolation, first-token ownership, bounded admission,
cancellation, errors, thread overlap, timeouts and lifetime guards. These CPU
tests are not GPU correctness or throughput measurements.

`tests/test_pd_native.py` opts in with `VENTURI_PD_TEST=1` and an explicit
`VENTURI_BLOCKS_LIBRARY` (SM89) or `VENTURI_SM70_LIBRARY`. Generated small hybrid
checkpoints compare cross-device continuation logits, tokens, FP32 GDN state,
convolution history and KV against no-handoff execution. Four P plus four D
cases require eight GPUs. Skips never count as passed GPU tests.

`tools/benchmark_pd_native.py` compares explicit `--mode pd` and `--mode colocated`
on the union of the same supplied devices. It uses the same request JSON and
ordinary greedy outputs, requires at least three repeats, records a shared wall
interval, CPU time, token/latency metrics, transfer counters and sampled GPU data.
It never sums request durations as elapsed time. Source scope must be explicitly
`--checkpoint-scope trained` or `generated-fixture`; small fixture measurements
are not trained-model TPS. Loading/preparation/warmup and profiler runs remain
separate. Example request JSON is a list of objects containing `prompt` token
IDs, `max_new_tokens`, and optional `arrival_seconds`.

The benchmark is a CLI utility, not a production API. It does not establish that
4+4 is optimal or that PD improves single-request speed. Compare colocated,
2P+6D, 4P+4D and workload-specific splits under the same latency goals. Initial
implementation/validation evidence lives in `operations/pd-native-20261029-v1/`;
transport-only L40 evidence is separately frozen in `pd-l40-intake-20261029-v1`.

Public CPU/GPU qualification and outstanding limits are recorded in
[the PD v1 qualification report](pd-qualification.md).
