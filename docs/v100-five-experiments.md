# V100 native Qwen research switches

These are explicit **experimental, default-off** paths for the resident FP16
`q8-residual-column-mlp-head-gdn` target on two SM70 devices with host-staged TP.
They do not change stored Q8 weights, KV precision, request state ownership or
the default loader. No automatic architecture/quality fallback is introduced.

```python
model = NativeQwenModel.from_q8(
    weights, backend, weight_mode="resident", dtype=torch.float16,
    tensor_parallel_device="cuda:1", tensor_parallel_transport="host",
    projection_profile="q8-residual-column-mlp-head-gdn",
    max_tokens=32, experimental_flags=1 | 2 | 4 | 8,
)
```

The immutable creation mask selects independent experiments:

| Bit | Experiment | Boundaries |
|---|---|---|
| 1 | Segment-batched convolution and compact GDN | Multiple one-token requests, at most32, DK64/128, no verification prefixes; otherwise original segment loop. Grid.y selects a request-owned cache. |
| 2 | FP32-weight A/B dots plus GDN gates | Same256-thread reduction association; A/B explicitly round to FP16 before gate math and beta rounds to FP16 again. |
| 4 | Rank-local GPU-only head-GDN CUDA Graphs | One-token segments only; transfers, cross-device event waits, source checks and prefix verification stay outside capture. |
| 8 | Earlier primary submission | Same host-staged transfers/F32 partials and stream/event fences; enqueue primary compute before submitting peer work, for GDN and MLP. |

Graph keys include rank, layer, token count, input address, ordered segment rows
and private state/history addresses. A synchronized384-entry bound limits the
cache. A dedicated cache mutex serializes cache lookup/launch against cohort retirement.
Session destruction synchronizes and clears executables before freeing
request state. Reset preserves pointers and explicitly zeroes private state;
replay never restores stale state contents. Capture itself does not execute a
state update; the first graph launch executes exactly once. Unsupported shapes
use the original rank body, not a silent arithmetic change. Graph allocation
cost/memory is additional to the static tensor workspace accounting and must be
measured; graph capture failures surface as errors. As with existing native
request destruction, callers must not destroy a request concurrently with its
native work.

The fifth experiment is separate: Qwen3.8 MTP must be exported from the **same
verified original source** as the target. The independent export verifier now
accepts explicit `component="mtp"` and an external source manifest. This is
storage verification, not model-quality qualification. Adaptive MTP chooses
between fresh cohorts, not halfway through a decode transaction. Batched MTP2
remains unsupported. Acceptance alone is not speed: charge draft, verification,
rollback/reset and policy costs, and compare against the fastest fixed mode.

## Evidence

The case plan, executed harness versions, failures, raw correctness/timing and
profiler results belong to `benchmarks/v100-five-experiments/`. The target is the
same WZU_Server two-V100 host. No result from Qwen3.5, production Uncensored GGUF,
or a different precision should be substituted for this Qwen3.8 native Q8 case.

Status: all five actual Qwen3.8 experiments completed, including three-repeat
C1/C8 controls, exact private-state/logit gates and separate V100 counter/node-level
profiles. All-four ordinary decode is32.842→33.562 TPS at C1 and133.327→142.166
aggregate TPS at C8. Adaptive MTP selects depth2 at C1 and ordinary at C8; its
whole-call scope is separate. Graph-only and earlier-primary changes do not merit
default-on promotion. See the evidence README for raw scope, outliers and limits.
Production restored unchanged; defaults remain off.
