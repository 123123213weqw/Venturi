# Native Qwen Q5 service routing on V100

This extends the operator work in `q5-adaptive-sm70.md` into the owned C++ Qwen
session and grouped-service loops. It does **not** replace the production
llama.cpp container, add MTP, change activation precision, or implement a
production admission/security layer. Legacy routing remains the default.

## Execution and ownership

```
HTTP admission -> native cohort generate -> grouped Qwen projections
                                           -> trunk: cuBLAS
                                           -> head: (N,K,M) lookup
                                              cuBLAS / scalar / WMMA
```

`NativeQwenModel` owns one immutable-budget scratch allocation and one local
calibration table. Both explicit `cublas` and `adaptive` modes use the same
weights, cuBLAS handle, stream, output/cache allocations and 256 MiB scratch
budget. Reconstruction is included on every cuBLAS call. There is no persistent
FP16 weight shadow. Existing load-time legacy dequantization scratch remains
allocated; reported model/NVML memory includes it, not just the new scratch.

Preparation/calibration must precede request creation. Calibration uses the
actual resident vocabulary-head matrix, deterministic probe activations,
full-output cuBLAS checks (`atol=rtol=.002`), two warmups, balanced CUDA-event
samples, medians and a 3% minimum gain. Scalar is considered only at M<=8. This
is a head-only shape policy, not independent tuning of every layer's weight matrix. Tables
are model-local and never imported across checkpoints/DSOs/GPUs.

**Whole-model qualification overrides operator wins.** Two rejected policies
are retained with their raw numerical counterexamples:

1. All projections / all M: isolated checks passed, but held-out 96/128-token
   repeated-input logits/state failed.
2. Trunk adaptive only at M<=8: the 128-token / 32-step continuation still failed.

The current service policy therefore adapts **only the vocabulary head, M<=8**.
All projections that feed hidden activations, GDN state or KV stay on cuBLAS.
This keeps teacher-forced state identical rather than assuming small local
rounding errors stay small through the whole recurrence. It is not proof that
the rejected kernels have a bit-decoding bug; accumulation-order amplification
is a possible explanation, not an isolated root-cause finding. Full-trunk
adaptive deployment and adaptive M>8 are deferred. Unknown head shape/M falls
back to cuBLAS. Do not advertise these results as full-trunk adaptive speedups.

The routing hot path performs no Python callback, allocation or online timing.
Counters distinguish actual route calls/token rows, safety-guarded cuBLAS calls
and unknown-head-table fallback. The optional routing ABI is version 2; stale
version-1 controls are rejected by the Python binding.
A mode switch synchronizes and locks the model; budget changes and calibration
after session creation fail. Switching does not reset cache state: callers must
reset requests for matched A/B tests. Graphs, TP/PP, mixed codecs, BF16, activation
quantization and resident-FP16 shadows are rejected by this optional policy.

## API

```python
model = NativeQwenModel.from_q8(
    q5_weights, explicit_qwen_runtime, weight_mode="resident",
    dtype=torch.float16, device="cuda:0", max_tokens=32,
    metadata_mode="sealed",
)
model.configure_q5_linear("cublas", workspace_limit_bytes=256 * 1024**2)
model.calibrate_q5_linear([1,2,3,4,5,6,7,8,16,32])  # explicit startup cost
model.configure_q5_linear("adaptive")
batch = model.create_batch([192] * 8)
# batch.generate(...) owns the complete prefill/greedy decode loop in C++.
print(model.q5_linear_info())
```

The benchmark HTTP adapter accepts `--q5-route cublas|adaptive`,
`--q5-workspace-mib`, `--q5-calibration-rows` and `--mode single --batch-target`.
It is loopback-only and is not the public production server.

## Reproduction and measurement contract

Build on the remote Linux V100 host, never locally:

```sh
PYTHONPATH=src python -m venturi.models.qwen.build \
  --ops-library /absolute/ops/libventuri_ops.so --output-dir qwen-build \
  --nvcc /usr/local/cuda/bin/nvcc
PYTHONPATH=src python tools/benchmark_qwen_q5_service.py \
  --artifact /absolute/qwen38-27b-text-q5.safetensors \
  --library /absolute/ops/libventuri_ops.so \
  --qwen-library /absolute/qwen/libventuri_qwen.so \
  --prompts operations/q5-qwen-service-20261117/prompts.json \
  --output /absolute/service-results.json --repeats 3 --output-tokens 32
```

The tool hashes the complete encoded checkpoint and both DSOs, records provenance
and calibration, then checks four 16/32/96/128-token inputs with 32-step forced
continuations, all-layer GDN/KV state and independent native greedy generation.
Grouped 1/2/4/8-request checks also force 32 steps, checking the first/last slot's
state. Predetermined numerical gates: logits max absolute <=.25 and relative L2
<=1%, all state tensors elementwise equal, all outputs finite. Report greedy/token equality separately:
FP32 reduction orders differ and these gates are not bit-exact equivalence.

The throughput driver runs in a separate CPU process. Requests start at a common
barrier; aggregate output TPS = total emitted tokens / global cohort wall.
Unprofiled A/B order is AB, BA, AB, with a discarded cohort per route/case first.
Record full HTTP request latency, native timing and server/client process CPU
separately. Benchmark admission waits for the exact cohort, dispatches as soon
as it is full, and rejects partial cohorts after a 5-second collection deadline.
The ordinary service adapter keeps its existing time-window policy. Timings include prompt work,
32 fixed output tokens, HTTP/JSON and admission; exclude load/calibration and
source digest. Input is fixed pretokenized IDs (no tokenizer in timed work).
Longer cases repeat 16 IDs, not a natural long-document benchmark. Context
capacity is 192, not the checkpoint's advertised maximum context.

NVML samples accompany each run. Separate `--profile-only` runs use CUDA profiler
capture plus NVTX route ranges; never use their TPS as unprofiled performance.
Hardware counters remain missing when profiling policy disallows them; GPU busy
and kernel attribution are not substitutes for achieved occupancy/DRAM bandwidth.

This campaign checks execution parity of two routes on the **same Q5 artifact**.
It does not establish Q5-vs-BF16 model quality, perplexity, natural-language task
accuracy, long-context stability, multi-GPU speedup, or superiority over every
external quantized inference engine. Checkpoint metadata has source-shard hashes
but no pinned upstream repository/revision; do not invent one.

The HTTP listener uses backlog 128. The driver rejects a wave unless all replies
belong to one cohort with exactly the requested count: a 7+1 split is not an
8-row comparison. Such an admission/backlog error was observed and its timing
report was rejected. Raising backlog alone did not eliminate the timeout-driven
split, so final timing uses fixed complete cohorts with a separate deadline.
`--correctness-report` explicitly reuses completed native gates only after
checking exact checkpoint/DSO SHA256, execution conditions, policy and calibrated
routes, plus identical prompt token IDs; its report records the source evidence digest. `--verify-long-batch`
adds a 128-input / 8-slot / 32-step state check before separate profiler capture.

## Measured result

Single Tesla V100-PCIE-32GB, the same resident 64-layer Qwen3.8-27B text Q5
artifact (SHA256 `dd57246737055d02c7cd19aa584c766981c2ea11d475b6608597b2057fa93bae`),
32 generated tokens/request, fixed complete cohorts. End-to-end output TPS:

| Prompt tokens | Concurrency | cuBLAS | Adaptive head | Gain |
|---:|---:|---:|---:|---:|
| 16 | 1 | 4.029 | 4.170 | 3.50% |
| 16 | 2 | 7.744 | 7.928 | 2.38% |
| 16 | 4 | 14.398 | 14.870 | 3.28% |
| 16 | 8 | 26.011 | 26.799 | 3.03% |
| 128 | 1 | 3.522 | 3.629 | 3.05% |
| 128 | 8 | 12.621 | 12.804 | 1.45% |

The measured routing selected scalar at M=1–2 and WMMA at M=3–8. Startup
calibration took 10.08 s, outside timing. All 48 HTTP cohorts (including warmups)
matched the corresponding baseline output IDs. The 32-step forced comparisons,
including the additional 128-token / 8-slot case, matched every sampled GDN/KV
tensor elementwise, with max logits error 0.0078125. This validates the head-only
experiment against per-call-decode cuBLAS under the measured conditions, not
against the existing legacy route or another inference engine.

See [the timing table with request latency and process CPU time](../operations/q5-qwen-service-20261117/results.csv).
The measurements in this document were taken from the experimental branch
based on `cf7c80a7` (operator ABI 17, Qwen ABI 1). They do not qualify a port
to the operator ABI 18 / Qwen ABI 3 QuantLinear branch.
