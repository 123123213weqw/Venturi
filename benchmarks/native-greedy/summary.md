# Native greedy selector: correctness, full-model regression and measured scope

Date: 2026-10-19 (Asia/Shanghai). Original Q8 inference goal: **active**.

## What changed

Blocks ABI **11** adds an owned, model-independent, two-pass GPU greedy selector.
FP16/BF16/FP32 logits remain on-device. Output is one int64 `[token, status]` pair
per row: finite logits select the lowest index of an equal maximum; any NaN or
infinity invalidates only that row and emits `[-1, 1]`. Callers must check status
before using the token. Signed zero, subnormals, all-equal negative extremes,
one-ULP differences and cross-tile ties are covered by exact independent checks.
No stochastic policy or quality parity with source sampling defaults is claimed.

Preparation validates explicit shape/dtype/device/architecture. Native execution
has no numerical Torch fallback, hidden allocation, host scalar read or sync.
It supports CUDA graph capture/replay and reentrant execution on independent
streams with caller-owned separate buffers. Workspace is exactly
`rows * min(ceil(vocab/4096),128) * 16` bytes, aligned to 16. This is a stateless
operator, not a batching scheduler. Independent rows are not eight served API
requests with complete independent model histories.

At the real vocabulary size 248,320, each row uses 61 first-stage CTAs with 256
threads and one final CTA with 256 threads. This schedule is not a block-size
ablation/tuning result. SM70 supports FP16/FP32 and rejects native BF16. ABI-10
libraries are rejected explicitly by current Python bindings; historical capsules
retain their original source/library versions for reproduction.

The streaming qualifier now defaults to `--sampler native-greedy`, selecting
before its separate diagnostic logit copy. Host control flow transfers 16 bytes
per row rather than a full FP32 vocabulary vector (993,280 bytes per row).
However, qualification still separately copies full logits/hidden/state for
diagnostics. Python still owns the generation loop, stop/input policy, phase
loading, position tables and KV transfers. This is **not** production-native
runtime/scheduling, stochastic sampling or whole-model graph qualification.

## Source/build provenance

CUDA source SHA256:
`4ecc8db6ecb60fb20e35e10fa7e4c0cd5a0ad9d83b4610cc1b8eb0261a79b545`.

| Target | Library SHA256 | Compiler |
| --- | --- | --- |
| SM70/V100 | `1f531257bea9c60905c87b2ac368eb294f416e01c4525810c3cc3b0ff79d7825` | nvcc 12.9.86 |
| SM89/4090D | `c4ba88cddeab7ae9a7b83f6cbe7dd5a22d3191a9ba3a6073b4156bc0129ca1c0` | nvcc 13.0.88 |

Remote builds and cache verification are explicit. `build.json` may name the
v2 source directory where the identical current CUDA source was compiled; v3
corrected tests reuse that verified cache. No local CUDA/Rust compilation or
environment dependency changes occurred. Both targets' worker source hashes
match the frozen core source. The new selector is not external NInfer code.

Source acquisition/true-Q8 export are unchanged from the prior milestones. Full
artifact SHA256 remains
`b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8`.
There is no Q4-to-Q8 quality promotion or altered original checkpoint.

## Trained-model regression

The complete 64-layer V100 FP16/Q8 streaming path using native greedy again emits
`[57590, 248046]`, Paris then EOS, on the same pinned 26-token chat case from
`../qwen-streaming/case.json`. All logits, 64 hidden tensors and 128 GDN/KV tensors
per step are **byte-identical** to the previous ABI-10/CPU-greedy native run.
See `v100/native-abi10-regression.json`; its prior report SHA is pinned and its
checker validates matching artifact, case, dtype, positions and tensor structure.
This is strong regression parity, not an independent original-BF16 quality check.

The current full report is `v100/full-model-run1/report.json`. The fresh same-input
CPU profile completed both steps and **both fixed gates passed**. The original
worker/child (3131708 / 3132547) completed with exit 0; neither was restarted.
The CPU profile shares the adapter and represented Q8/FP16/FP32 parameters. It is
not independent upstream architecture or corpus/original-BF16 quality. The
predeclared logit gates are unchanged: top-1 match, relative L2 <= 0.02, centered
cosine >= 0.999, and KL(ref->native) <= 0.002. No gate was relaxed after observation.

| Step | Native / CPU top-1 | Relative logit L2 | Centered cosine | KL(ref->native) |
| --- | --- | --- | --- | --- |
| 0 | 57590 / 57590 | 0.002625315 | 0.9999943574 | 5.33031e-7 |
| 1 | 248046 / 248046 | 0.002307423 | 0.9999959268 | 2.76891e-7 |

Full qualification wall time 947.001844 s, process CPU 2886.337392 s, peak process
RSS 33,015,734,272 bytes. These include complete hashing, uploads, streaming,
diagnostic traces and allocating CPU reference. They are not native latency/TPS
or a baseline/change performance comparison. Per-layer/state differences remain
in the report as diagnostics, not extra post-hoc acceptance gates.

Streaming weights are not fully resident. Current native-pass peak PyTorch
allocation is 1,519,401,472 bytes, excluding context/driver/other processes. The
diagnostic also retains a previous device-logit vector until reassignment, so
this peak is not an optimized buffer-lifetime result. Full native FP16 weight
tensors still require 28,654,073,856 bytes before KV/state/workspaces; that does
not fit 24 GiB. No full trained-model 4090D or engine-speed result follows.

## Matched unprofiled selector measurements

`tools/benchmark_greedy.py`, seed 8107, FP32, vocab 248,320, rows 1 and 8. Identical
cross-tile ties, finite-row validation, lowest-index policy and host-consumed
`[token,status]` results. Four alternating/reversed-order waves, 100 calls each,
after 10 warmup calls per mode. All results match exactly. CPU thread count 4.

- **CPU**: full logits D2H, allocating CPU finite scan/argmax, host result.
- **Torch GPU**: allocating GPU finite scan/argmax, small result D2H, host result.
- **Native GPU**: caller-buffer native selection, small result D2H, host result.

Median wave host-visible latency, microseconds per call:

| Target | Rows | CPU full-logit path | Torch GPU path | Native GPU path |
| --- | ---: | ---: | ---: | ---: |
| 4090D | 1 | 954.367 | 390.213 | 145.560 |
| 4090D | 8 | 11,340.576 | 456.828 | 146.063 |
| V100 | 1 | 1,282.262 | 195.454 | 66.286 |
| V100 | 8 | 11,105.415 | 201.779 | 78.619 |

Median process CPU time (all process threads), microseconds per call:

| Target | Rows | CPU full-logit path | Torch GPU path | Native GPU path |
| --- | ---: | ---: | ---: | ---: |
| 4090D | 1 | 3,776.160 | 441.195 | 145.575 |
| 4090D | 8 | 34,819.625 | 457.230 | 146.085 |
| V100 | 1 | 5,098.615 | 276.455 | 66.305 |
| V100 | 8 | 41,035.385 | 261.660 | 78.640 |

Raw waves, CPU/wall times, row-selection throughput and VRAM peaks are preserved
under each target's `microbench/report.json`. Process CPU can exceed wall time
because the CPU baseline uses multiple threads. Separate native CUDA-event
loops include both kernels **and launch gaps**, excluding host result copies;
they are not isolated kernel duration or hardware utilization.

These are observed selector-only improvements, **not** model TPS, TTFT, whole-
model CPU reduction, eight-request serving or matched external-engine wins.
Other services remained running. The 4090D has substantial interference: native
waves include approximately 441/401 us outliers and Torch GPU waves vary widely.
Do not infer universal or isolated-hardware speed superiority from medians.

Whole-device samples at 100 ms (all processes, not kernel-specific):

| Target | Samples | GPU busy range | Device VRAM range MiB | Power W | SM clock MHz | Temperature C |
| --- | ---: | --- | --- | --- | --- | --- |
| 4090D | 59 | 37-45% | 15,091-15,528 | 170.84-179.06 | 2,775 | 45-47 |
| V100 physical GPU 1 | 54 | 0-24% | 29,269-29,667 | 36.08-57.55 | 1,230 | 31-32 |

Raw clock/memory-controller samples are retained. Neither busy percentage nor
memory-controller busy is achieved SM occupancy or DRAM bandwidth. Existing
`RmProfilingAdminOnly=1` is recorded; no privileges/settings were changed and no
new hardware-counter campaign was attempted. Achieved SM/DRAM/cache/stall counters
remain unavailable. Existing GPU services were not stopped.

## Separate profiler attribution and remaining overhead

SM89 Nsight Systems 2025.3.2 trace (`sm89/profiler/`, raw binary/SQLite stay remote)
is separate from unprofiled measurements. It has 10 native and 10 CPU NVTX ranges
for each row count, plus warmup outside ranges. The tested parser joins activity
by the **launch API correlation ID and host thread**, not accidental GPU/CPU
timestamp overlap. It checks a single process, exactly two owned kernels for
native selection, no selection kernels for CPU policy and the correct D2H bytes.

Median profiler-only microseconds (not unprofiled latency):

| Rows | Partials kernel | Finish kernel | Between-kernel gap | Final kernel -> small-copy gap | Small D2H copy | Full-logit D2H copy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.744 | 1.280 | 5.216 | 33.505 | 0.960 | 89.921 |
| 8 | 4.480 | 1.280 | 7.985 | 50.337 | 0.896 | 844.474 |

Host gaps between the two launch API calls are 1.698 / 2.486 us. The short
owned kernels plus remaining host/copy gaps motivate reducing orchestration and
synchronization next; they do not identify all gaps as Python, prove GPU idle
time (other processes run), measure occupancy, or justify a whole-model speed
claim. Python loop, host stop checks and per-phase staging remain. Native runtime
work must address those and the far larger full-model weight-transfer volume.

## Tests and retained failures

- Local full suite: 697 passed / 356 skipped (16.26 s).
- V100 selected selector/streaming/port/tool/build CPU+native: 130 passed /
  40 skipped (7.36 s); missing NumPy warning retained. This is not 130 GPU cases.
- SM89 final full suite: 1011 passed / 42 skipped (43.47 s), including the two
  trace-parser tests. The earlier corrected suite was 1009 / 42 (46.55 s);
  both logs remain. `sm89/final-tests/tests.log` is authoritative.
- Trace-parser synthetic tests cover queued GPU work outside its submitting
  host range, wrong copy sizes/direction and multiple-process rejection.

`initial-attempts/` preserves failures, not just successful results. First remote
compiles failed because the infinity constant needed an explicit
`math_constants.h` include. After adding it, the graph test initially prohibited
PyTorch's own pre-capture setup synchronization, not an operator sync; the guard
now wraps `run()` inside capture and replay, and still forbids allocation,
numerical fallback and scalar reads. The new clean source directory initially
omitted two launcher scripts needed by unrelated quoting tests; these were
included without changing them. All affected suites were rerun. The malformed
first shell exit marker is preserved as raw text, not parsed as valid JSON;
subsequent workers use plain exit-code files. Terminal workers were confirmed
before new attempts; no active job was duplicated or killed.

## Reproduce and remaining goal

Core API example is in README. Remote roots are `greedy-src-v3` and
`greedy-validation-v3` beneath the existing V100/4090D project directories.
`worker.sh`, `worker-process.json`, build provenance, logs, fixed case references
and each benchmark command are retained. Use new output directories; never
overwrite an existing full-model/benchmark/profiler run. `source/` freezes source,
tests, launcher scripts and docs; `source-sha256.json` and `evidence-check.json`
record verified provenance and qualification scope. Older capsules stay intact.

Still required: independent upstream/original-BF16/corpus quality; an owned native
forward/control/cache runtime rather than Python orchestration; reusable whole-
model graphs; request-safe batch-aware GDN and paged KV/continuous batching; and
matched same-checkpoint/precision/lengths single and concurrent external-engine
benchmarks on 4090D. Greedy policy does not substitute for stochastic sampling.
The original objective has not been narrowed or completed.
