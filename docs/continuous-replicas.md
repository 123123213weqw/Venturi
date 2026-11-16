# Explicit continuous replica pool

`ContinuousReplicaPool` routes **whole requests** to independently prepared,
homogeneous `ContinuousScheduler` instances. Each replica has one inference
worker, its own batch slots and private model state. The routing layer uses only
the standard library and does not know Qwen, GLM, Torch, CUDA or HTTP.

```text
tokenized requests -> bounded replica router
                     |-- worker 0 -> continuous scheduler -> prepared batch 0
                     `-- worker 1 -> continuous scheduler -> prepared batch 1
```

This is request replication, not tensor/pipeline parallelism or PD. It does not
load models, select cards, download weights, migrate requests, share prefixes,
enable MTP, or change production defaults. A batch adapter must independently
qualify its ragged mixed-step/reset semantics; routing does not make an unsupported
model executable. A PP/TP model is not automatically a supported batch adapter.

## Usage and ownership

```python
from venturi.runtime import (
    ContinuousReplica, ContinuousReplicaPool, ContinuousScheduler,
)

# model_a/model_b are already loaded and qualified on explicit devices.
# Authenticate the SAME checkpoint, tokenizer, precision and arithmetic profile.
# The identity string is an application assertion, not authentication by the pool.
batches = [model_a.create_batch([1024]*8), model_b.create_batch([1024]*8)]
replicas = tuple(
    ContinuousReplica(str(i), authenticated_identity,
        ContinuousScheduler(batch, token_budget=32, prefill_chunk_size=16))
    for i, batch in enumerate(batches)
)
pool = ContinuousReplicaPool(replicas, max_requests=128, routing="round-robin")
try:
    pool.start()  # explicit; construction/import starts no worker/device
    ticket = pool.submit(prompt_ids, max_new_tokens=64, eos_token_ids=eos_ids)
    completion = ticket.future.result(timeout=120)
    assert completion.request_id == ticket.request_id  # globally unique in pool
finally:
    pool.close(timeout=180)  # shared join deadline across ALL owners
    # Only after successful joins may the caller free borrowed batches/models.
    for batch in batches:
        batch.close()
```

Do not separately tick, submit to, reset or close leased schedulers/batches.
After a close timeout, retain **all** borrowed buffers and join again; an
observation timeout does not stop native work. Workers are non-daemon and need
explicit close. No destructor silently frees live GPU state.

Public futures resolve at request completion, not at the end of a cohort.
Future callbacks execute on the owner and must remain short/nonblocking; they
must not join that owner or close the pool. Streaming callers should instead
own scheduler token events; this pool does not add token streaming/HTTP.

## Routing and overload

- Opt-in `least-work` minimizes outstanding `(prompt length + output budget) / slot
  count`, rotating ties. It is a deterministic metadata heuristic, **not** a
  measured cost model, remaining-work estimator or a universal speed win.
- Default `round-robin` supplies an explicit matched control, still respecting healthy
  replica, context capacity and pending queue constraints.
- `max_requests` counts queued + active + cancelled-but-not-retired requests.
  Overflow is rejected synchronously, not buffered without bound.
- A request stays on its chosen replica for its complete lifetime. Model state
  is never merged between requests/replicas. There is no cross-replica prefix
  cache, automatic retry or backend/model substitution.
- Unhealthy owners stop receiving work; already affected requests fail. Other
  healthy replicas can still serve. `close()` signals every owner before joining
  any, and surfaces execution/cleanup errors.

`pool.cancel(ticket)` and `ticket.future.cancel()` cancel at a safe step boundary.
The client future can become cancelled immediately, but the request continues
to consume admission capacity until native work retires. `scheduler.contains`
reports this distinction without probing the GPU. The worker's optional
nonblocking `on_boundary` observer reclaims deferred bookkeeping after tick/close.
No cancelled state can be reused without the scheduler's ordinary reset.

## Qualification and performance scope

The CPU tests cover concurrent 4/8/16/32-request submission, exact synthetic
trajectories, independent owner overlap, mixed capacity admission, overload,
cancellation races, callback reentrancy, failure isolation and close timeouts.
Opt-in `tests/test_continuous_replicas_native.py` requires **two already admitted
idle GPUs** and explicit paired native libraries:

```sh
VENTURI_REPLICA_TEST=1 VENTURI_OPS_LIBRARY=/absolute/libventuri_ops.so \
VENTURI_QWEN_LIBRARY=/absolute/libventuri_qwen.so \
python -m pytest -q tests/test_continuous_replicas_native.py
```

`tools/benchmark_continuous_replicas.py` uses actual local Qwen Q8/tokenizer
files, independent native B1 token references and heterogeneous real prompts.
It compares both policies with at least three repetitions, alternating order,
fresh scheduler warmups and fixed output budgets. A token counterexample stops
qualification and is saved rather than hidden. Hardware admission and monitoring
are the launcher's responsibility; the tool never stops another process.

Aggregate throughput is actual output tokens / **one common fenced wall
interval**, including prefill/admission; it is **not decode-only TPS**, summed
request TPS or HTTP latency. Load/startup/warmup/teardown/correctness checks are
outside that interval. Per-request TTFT/latency and process CPU/RSS are separate.
GPU busy samples are not achieved occupancy or bandwidth. Profile separately;
missing counters and unsupported models must remain explicitly unqualified.
