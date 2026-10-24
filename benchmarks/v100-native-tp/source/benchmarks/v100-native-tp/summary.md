# Native Q8 Qwen: two-V100 row sharding

## Result and scope

Native resident Qwen3.5-27B, FP16 activations/KV, unchanged original-derived
Q8 group-32 weights and FP32 small parameters/state. One request uses either
one V100 or two V100s. **No MTP, continuous batching, paged KV or whole-model
CUDA graph is implemented or implied by this result.**

The user confirmed their approximately 50 TPS external observation used two
V100s plus MTP. The stopped service also used a different checkpoint:
`Qwen3.8-27B-Uncensored-Q8_0.gguf`, tensor split 1,1, `draft-mtp`, maximum draft
length 3. Native uses official original-derived Qwen3.5-27B. The observation is
not an independently reproduced, same-checkpoint precision/workload comparison.
The larger performance goal remains unfinished.

Four matched unprofiled trials per mode, in single/parallel/parallel/single
phases, two trials per phase. Same ABI-13 library, checkpoint, FP16 execution
profile, 26-token prompt, 32 emitted tokens, concurrency one, explicit greedy
selection and EOS-disabled fixed work. Timings exclude model load, session
preparation, reset and a full-length warmup. Decode is a single native call
emitting the next 31 tokens after a separately timed first token.

| Metric | One V100 | Two V100s, explicit host transport |
|---|---:|---:|
| Decode output tokens/s | 14.9565 | **17.1169** |
| Decode wall, 31 tokens | 2.072682 s | 1.811073 s |
| Decode process CPU | 2.073569 core-s | 1.812840 core-s |
| TTFT, 26-token prompt | 0.900992 s | **0.562288 s** |
| TTFT process CPU | 0.901429 core-s | 0.562868 core-s |
| One-call prompt + 32 output wall | 2.968862 s | **2.367324 s** |
| One-call process CPU | 2.970095 core-s | 2.369841 core-s |
| Output tokens / whole-request wall | 10.7785 | 13.5174 |

Decode speedup is **1.14445x**, TTFT speedup 1.60237x, whole-request speedup
1.25410x. All warmup/split/unsplit and cross-mode token IDs match. Fixed work
continues after EOS and is not a normal answer-length quality evaluation.
A separate 23-token story prompt has no EOS in its 32-token output, again exact
cross-mode IDs: 14.9713 -> 17.1664 decode TPS, 0.77938 -> 0.49007 s TTFT, and
2.84474 -> 2.29276 s whole-request wall. These story trials are not pooled with
the four main trials.

## Native implementation and transport counterexample

Blocks ABI 13 adds explicit resident two-device row shards. Each shard computes
complete dot products, preserving FP32 FMA/reduction ordering and Q8 codes and
scales. Gathered rows are the same FP16 values, not rounded partial sums in an
all-reduce. Embedding lookup and GDN/KV state remain primary-owned. Tied heads,
odd row tails, multi-token output pitches and source identity checks remain
supported. Streams/events and the model mutex protect shared scratch; four
Python threads are a serialization/state-isolation test, not parallel serving.

Both cards are Tesla V100-PCIE-32GB, SM70. The topology is `NODE`, not NVLink.
`CUDA_VISIBLE_DEVICES=1,0` selects physical GPU 1 as logical primary 0 and
physical GPU 0 as logical secondary 1. Smaller TTS/RWKV services stayed running
and idle. The previously user-authorized stopped llama-server container remains
stopped, not deleted; restart policy is unchanged. No driver, clocks, IOMMU,
ACS, authorization, TLS/SSH verification or profiling permissions were changed.

Crucially, both peer directions advertise access but **direct copies are not
reliable**. An earlier probe appeared exact in one direction; the final preserved
fresh-process probe fails in **both** directions, leaving all 1,024 destination
values at the sentinel in each case. No direction is treated as a healthy route.
Explicit host copies are exact in both directions. Initial native direct-peer
outputs failed parity; its threaded worker stopped progressing and was
terminated after identifying the owned PID. Failed logs/source and the abort
record are retained in `v100/validation-v1-failed/`; none are speed evidence.

Direct transport now requires actual bidirectional byte probes at 64 B, 4 KiB
and 128 KiB before large uploads, and fails fast rather than silently falling
back. All accepted trained/timing tests explicitly select portable-pinned-host
transport. Transfers are native CUDA D2H/H2D DMA, not CPU model computation or
Python layer callbacks. Admission checks both devices and rejects streaming
weights with tensor parallelism. Single-GPU execution remains the default.

Native memory for the main benchmark, excluding other services/CUDA contexts:

- Single primary model: 28,658,268,288 B; request: 177,783,344 B.
- Two-GPU primary model/scratch: 15,065,595,008 B; secondary weights/scratch:
  13,606,491,136 B; request remains primary-owned: 177,783,344 B.
- Pinned host buffers: 8,388,608 B single, 15,750,144 B two-GPU.
- Logical inter-device payload counters are not physical PCIe byte counters:
  host staging performs two DMA operations per logical payload. There are no
  additional Q8 weight uploads during resident decode.

## Correctness and regression evidence

The full trained 64-layer, two-step Paris + EOS case passes the unchanged fixed
CPU-reference logit gates. Every native logit, hidden, GDN/history and KV tensor
is byte-identical to the pinned previously qualified single-GPU native run.
A fresh one-call native generation with Torch numerical/model calls forbidden
emits `[57590, 248046]`, final cursor 27. The CLI independently emits those IDs;
load is 39.3742 s, generation including prefill 0.63868 s. That CLI rate is not
steady-state decode TPS, and its explicit-ID call does not verify tokenizer or
artifact hashes again; the full qualifier separately verifies their pinned hashes.

Reference scope is the represented Q8/FP16 profile with shared metadata/loading
adapters, not original-BF16 corpus quality or an independent upstream architecture
qualification. A two-step diagnostic and two short benchmark prompts do not
establish arbitrary-context or broad quality equivalence.

- Final local suite: **714 passed / 468 skipped**; Ruff and diff whitespace checks pass.
- Final full V100 suite: **769 passed / 413 skipped**.
- Earlier selected V100 TP/native tests: 126 passed / 43 skipped.
- Fixture tests cover tied/untied heads, row tails, chunked/continuation outputs,
  complete state bytes, threaded scratch isolation, failure/reset/lifetime and
  forbidden Torch math. SM70 BF16 execution is explicitly skipped/rejected.
- The first full-suite collection lacked NumPy; a second deployment omitted the
  launcher scripts. Both failed attempts are retained. NumPy 1.26.4 was installed
  only into a job-local overlay, not the shared venv; scripts were synchronized,
  and the final complete suite passed. These are not ignored test failures.
- SM89 was not exercised in this milestone. Historical SM89 results are not
  relabelled as ABI-13/two-GPU qualification.

## GPU diagnostics and remaining room

NVML samples during the timed TTFT/decode/unsplit windows, including all device
users, not achieved occupancy or bandwidth:

| Sample mean | Single primary | Two-GPU primary | Two-GPU secondary |
|---|---:|---:|---:|
| GPU busy | 99.43% | 76.15% | 65.39% |
| Memory-controller busy | 54.18% | 34.40% | 33.52% |
| Device used memory | 31,629 MiB | 18,545 MiB | 17,225 MiB |
| Power | 232.77 W | 168.85 W | 149.75 W |
| Temperature | 45.83 C | 41.16 C | 38.82 C |
| SM clock | 1,375.67 MHz | 1,380 MHz | 1,380 MHz |
| Memory clock | 877 MHz | 877 MHz | 877 MHz |

Two GPUs draw more combined sampled power; these means do not establish energy
per request. Process CPU remains roughly one core-second per wall-second;
that includes native/CUDA waiting/dispatch, not Python model math.

Separate Nsight CUDA/NVTX runs use eight output tokens and seven decode steps.
Their instrumented timings are not substituted into unprofiled TPS:

| Decode trace | Single | Two-GPU |
|---|---:|---:|
| Host range | 477.835 ms | 435.525 ms |
| Summed kernel durations | 464.685 ms | 486.712 ms |
| Q8 summed kernel durations | 417.093 ms | 439.678 ms |
| Cross-device activity union | 464.701 ms | 334.133 ms |
| Gaps between this process's activities | 7.021 ms | **95.672 ms** |

The two-GPU sum includes overlapping devices: trace-device kernel totals are
267.481 and 219.231 ms; never add them to calculate request latency or infer
occupancy. Gaps are this process's traced timeline, not device-wide idle time.
The dual trace adds 5,621 H2D and 5,621 D2H operations, about 69.705 MB in each
direction, versus seven tiny copies per direction single-GPU. D2H/H2D summed
GPU copy durations are 12.490/24.998 ms. Host API durations include 35.124 ms in
`cudaMemcpyAsync`, 12.599 ms in `cudaMemcpy2DAsync`, 16.673 ms in stream-event
wait submission and 12.532 ms in event recording. These API durations are not
CPU compute time; API nesting and GPU overlap prevent simple addition into a
complete causal breakdown. Raw per-range host API attribution is preserved and
its thread/range filtering passed a synthetic check.

This is measured room to reduce fine-grained transfers/dispatch and amortize
projection work, not evidence that merely using a second GPU doubles speed.
Q8 projections remain dominant in summed kernel time. Hardware-counter policy
is unchanged (`RmProfilingAdminOnly: 1`); achieved occupancy, SM throughput,
DRAM bandwidth and cache/stall counters remain unavailable. No repeated counter
campaign or privilege change was attempted under that unchanged policy.

## Preserved kernel experiment, not promoted

One-warp-per-output-row variants keep eight virtual accumulators and original
FMA/reduction order. Mode 2 packs weight-byte transport; it does not reduce
weight precision. Both variants pass exact-byte/tail tests: each 53 passed /
51 skipped. Real projection-only graph microbenchmarks show most regress:

| Matrix N x K | Original 4/4 | Warp mode 1 | Packed mode 2 |
|---|---:|---:|---:|
| 248320 x 5120 | 2.78145 ms | 4.03354 ms | 2.39067 ms |
| 17408 x 5120 | 0.20410 ms | 0.30738 ms | 0.19784 ms |
| 5120 x 17408 | 0.20750 ms | 0.32778 ms | **0.27566 ms** |
| 1024 x 5120 | 0.01520 ms | 0.01993 ms | **0.02531 ms** |

All eight shapes, full build logs, original source and negative results are
retained in `v100/warp/`. The isolated large-head improvement does not justify
promoting mode 2 to the full model. Default `--q8-decode-mode 0` remains unchanged;
experimental flags participate in build-cache identity.

## Next and reproducibility

The original HF checkpoint's 15 MTP tensors contain 424,699,392 parameters /
849,398,784 BF16 bytes. They are inventoried, **not exported or implemented**.
Native MTP needs draft-head qualification, target hidden/position alignment,
verification, GDN/history rollback and acceptance/EOS tests before performance
claims. Reducing transfer/dispatch overhead remains relevant alongside MTP.
See `docs/native-mtp-plan.md`. Broader original-BF16/upstream quality, native
batching/paging, long context, graphs and matched external comparisons remain open.

`metrics.json` is recomputed by `python3 benchmarks/v100-native-tp/summarize.py`.
Raw JSON/CSV/logs and failed attempts are under `v100/`; full snapshot/PT and
Nsight binary/SQLite files stay on the server in their recorded job paths.
`source/`, source/evidence hashes and launchers preserve executed critical
sources, build identities and diagnostic scripts; prior capsules are audited
without mutation. An incomplete freeze attempt that incorrectly expected the
earlier one-direction probe result is retained separately in `freeze-attempt-1/`;
its assertion failed, and it was not accepted as final evidence. The final raw
probe and summary correctly record failures in both directions. This milestone
is not committed/pushed or goal-completing.
