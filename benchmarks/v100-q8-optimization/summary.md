# V100 resident Q8 projection optimization

## Result

Full Qwen3.5-27B, original-derived signed Q8/group-32 weights, FP16
activations/KV and preserved FP32 small parameters/state, one native request:

| Unprofiled measurement | Native resident baseline | Native resident tiled |
|---|---:|---:|
| Time to first token, 26-token prompt | 2.37990 s | 0.90326 s |
| Decode, subsequent 31 tokens | 3.26822 s | 2.07469 s |
| Decode throughput | 9.4853 tokens/s | **14.9420 tokens/s** |
| One-call request, prompt + 32 output tokens | 5.64253 s | 2.97144 s |
| Output/request-wall rate, including prefill | 5.6712 tokens/s | 10.7692 tokens/s |
| Process CPU time for one-call request | 5.64511 core-s | 2.97273 core-s |

The kernel change is 1.5753x faster for decode, 2.6348x for first-token
latency, and 1.8989x for this complete fixed-length request. Four timed trials
per implementation were measured in baseline/tiled/tiled/baseline order, two
per phase, with one full warmup request per phase. Each trial includes a split
request and a separate one-call request. Tiled decode spans
2.07408–2.07548 s; baseline spans 3.26501–3.27004 s. Checkpoint, precision,
prompt IDs, capacities, output length and policy match. Load, preparation,
warmup and state reset are excluded. TTFT generates one token; a second C++ call
continues the same native request for 31 tokens. An independent one-call
32-token run checks the exact same token IDs. There is no Python token loop.

EOS stopping is disabled to fix the work. The capital-of-France case naturally
emits Paris + EOS after two tokens, so its 32-token stress sequence deliberately
includes post-EOS tokens. It is not an ordinary 32-token answer. A separate
23-token story prompt produced the same 32-token prefix with both libraries,
without any EOS in those 32 tokens: decode 9.4894 -> 14.9672 tokens/s and
first-token latency 2.10752 -> 0.77974 s. See `story-case.json` and
`story-output.txt`. This is a smoke test, not a corpus-quality evaluation.

The old approximately 0.11 output tokens/s figure used a two-output-token
**offloaded**, prefill-inclusive request. It is not comparable to steady-state
resident decode TPS. The real CLI now also completed that ordinary Paris + EOS
request in 0.97333 s generation time; its separate model load was 32.31579 s.
That single CLI smoke timing is not substituted for the repeated benchmark.

Recompute the machine-readable measurements with `python summarize.py`.
`metrics.json` includes CPU time, latency ranges and telemetry. Raw repeated
records live in `v100/matched-phase-*/report.json` and `v100/story-phase-*`.

## VRAM and service change

The user explicitly authorized stopping the workloads holding the V100 memory.
The `/qwen38-27b` llama-server container was stopped normally through Docker;
its restart policy was not modified. It had allocations on both V100s. The
smaller RWKV service on GPU 0 and TTS service on GPU 1 were left running because
stopping the large container alone provided sufficient memory. No unrelated
containers, driver settings, clocks or profiling privileges were changed.

The selected physical GPU 1 went from approximately 3.2 GiB free to 29,306 MiB
free. The Q8 model is now explicitly **resident**, not silently offloaded.
Native ownership during the 32-token workload is 28,658,268,288 model bytes
plus 177,783,344 request bytes, excluding CUDA context and Torch loader/cache
allocations. The 8 MiB pinned staging buffer is host memory. The read-only
28,601,624,576-byte logical mapping is not an expanded dense CPU model.
Physical NVML device usage during the timed runs was 31,629 MiB, including TTS,
contexts and other process allocations, not just Venturi. Raw CUDA allocations
are not captured by Torch allocator peaks.

`v100/stopped-service.txt`, before/after GPU records, `service-final.txt` and
`restore-service.txt` preserve the service identity and restart command. The
container remains stopped, not deleted. All optimization workers exited; the
model is not left loaded as a persistent service.

## Implementation and ablation

The generic Q8 projection entry point now supports SIMT token and row tiling:

- Prefill: one weight value is reused across four prompt tokens.
- Single-token decode: one activation is reused across four output rows.
- The original per-output 256-lane column traversal, FP32 FMA order and
  two-stage reduction order are retained, followed by the same output cast.
- Q8 codes, FP16 scales and checkpoint math are unchanged. There is no new
  quantization, FP16 weight reconstruction or Tensor Core/INT8 GEMM claim.
- No model dimensions, layer indices or Qwen-specific branching are embedded
  in this projection kernel. Supported arbitrary matrix/tail shapes remain
  accepted. Performance outside this workload still needs its own validation.

Six build variants were checked: token/row tiles 4/1, 4/4, 8/4, 2/4, 4/2 and
4/8. Projection-only tests use actual checkpoint matrices, shared immutable
GPU operands, exact baseline-byte checks and alternating timing order. Graph
replay amortizes microbenchmark Python overhead; **the full native model is
not graph-captured**. CUDA-event intervals are not full-engine latency or
hardware-counter DRAM bandwidth.

Representative projection-only medians in milliseconds:

| Shape and tokens | Original | 4/4 | Other observations |
|---|---:|---:|---|
| 17,408 x 5,120, T=26 | 8.6125 | 3.0130 | token tile 2: 3.7774; tile 8: 3.7171 |
| 5,120 x 17,408, T=26 | 8.4373 | 3.4277 | token tile 2: 3.6477; tile 8: 3.8318 |
| 248,320 x 5,120, T=1 | 4.7156 | 2.8018 | row tile 2: 2.8418; row tile 8: 3.1805 |
| 17,408 x 5,120, T=1 | 0.3371 | 0.2036 | row tile 2: 0.2046; row tile 8: 0.2329 |

Larger tiles are not automatically better. Row tile 2 slightly wins on a few
small shapes; 4 was selected for the dominant large projections and complete
model validation, not as a claim of universal optimality. All waves and
regressions remain in `v100/projection-ablation-v2` and `v100/row-ablation`.

Explicit build flags accept 1, 2, 4 or 8 and participate in cache identity.
SM70 defaults to 4/4; SM89 retains 1/1 pending its own full-model performance
qualification. An explicit 1/1 build retains the original dispatch. The
default SM70 build was verified to resolve to the tested cached binary.

## Correctness

Both resident baseline and tiled runtime completed the full trained-model
qualification against the pinned previous native and CPU evidence:

- Paris + EOS token IDs `[57590, 248046]`, position 27.
- Both predeclared CPU-reference logit gates passed unchanged.
- Logits, every layer's hidden values, GDN histories/states and KV prefixes
  matched the previous native run byte-for-byte across both teacher-forced steps.
- Actual generation ran through one C++ call with Torch numerical operations
  and Python model math forbidden.
- All repeated 32-token baseline/tiled outputs match exactly, including the
  independent story prompt. No resident weight uploads occurred during decode.
- Native CLI integration passed using explicit IDs, resident mode and FP16.

Local complete suite: **708 passed / 430 skipped**. Selected V100 native/CPU,
port, precision, selector, build, timing, trace, CLI and qualification suites:
**157 passed / 66 skipped**. Each of the six tile variants separately passed
28 original-byte/tail/guard checks, with 28 native-BF16 cases skipped on SM70.
The pre-existing V100 NumPy warning remains; no packages were installed or
existing virtual environments modified.

CPU reference shares checkpoint binding/profile helpers and represents this
Q8/FP16 execution profile. Neither these two teacher-forced steps nor matching
old outputs qualify original-BF16/corpus quality or independently establish the
upstream architecture. Previous numerical counterexamples are retained.

## GPU attribution and diagnostics

Separate Nsight CUDA/NVTX traces use an eight-output-token workload (seven
decode steps). Runtime **and driver** API correlation joins attribute kernels
to their submitting CPU ranges even when GPU execution occurs after the range.
They are not used as unprofiled throughput results.

| Profiler-only quantity | Baseline | Tiled |
|---|---:|---:|
| TTFT total kernel time | 2,396.1 ms | 902.4 ms |
| Main prefill Q8 family | 2,345.7 ms | 848.7 ms plus the single-token head |
| Seven decode steps, all kernels | 737.7 ms | 465.2 ms |
| Seven decode steps, Q8 projection family | 690.1 ms | 417.5 ms |
| Gaps between this process's decode activities | 7.08 ms | 6.95 ms |

Projections account for approximately 93.5% of baseline decode kernel time and
89.7% after tiling. They remain the main short-context target, not attention or
paged allocation. Resident hot-loop H2D copies were only token IDs: 208 bytes
for prefill and 56 bytes for seven decode steps; D2H was 16 bytes per selected
token. Large weight uploads occur at load, outside these ranges.

During the unprofiled timed windows, sampled device GPU busy averaged 99.69%
baseline / 99.43% tiled; sampled memory-controller busy averaged 43.26% /
54.39%. Power averaged 211.20 / 233.58 W, temperature 49.05 / 49.39 C, SM
clock 1,380 / 1,374.75 MHz, memory clock 877 MHz for both. These are device-wide
sampled proxies, not per-process achieved occupancy or bandwidth. GPU busy
being high before and after does not mean both kernels perform equally well.

The driver still reports `RmProfilingAdminOnly: 1`; achieved occupancy,
SM/DRAM/cache/stall hardware counters are unavailable. No privilege/policy
change or repeated counter campaign was attempted. Static binary resources
show 256-thread blocks, 44 registers/thread for the FP16 four-token tile and
64 for the FP16 four-row tile, 128 shared bytes and no stack/local allocation.
Those resource counts are **not achieved occupancy**. CPU sampling/context
switch tracing was disabled; gaps are not a complete CPU-cause analysis or
device-wide idle measurement.

Raw profiler reports/SQLite and large numerical snapshots stay on V100 under
`/home/wzu/codex-build/Venturi/v100-resident-optimization-v1/`.
Their generated paths and summaries are retained locally. Source/build
manifests, integrity checks, raw logs and bounded corrected orchestration
failures are preserved in this capsule; failed helper attempts are not timing
samples.

## Remaining scope

This is a single-request, short-context, internal native-baseline comparison.
It does not compare against llama.cpp or the stopped server, which used a
different checkpoint. It does not qualify 4090D speed, all workloads/models,
parallel serving, batch-aware GDN, paged attention, whole-model graphs or
speculative decoding. A 28.66 GB resident model does not fit a 24 GB 4090D.
The larger native-engine goal remains unfinished. Projection efficiency is
still the largest measured short-context opportunity; batching/scheduling and
broader quality are separate required milestones.
