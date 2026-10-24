# Native residual-activation Q8 projections: model integration

2026-10-19; **opt-in experimental profile, external speed goal still active**.
This capsule follows the operator-only `v100-q8-dp4a` study without modifying it.
`python3 benchmarks/v100-native-residual/summarize.py` recomputes `metrics.json`
from raw reports and timed-window telemetry. Source, executed-snapshot deltas,
build provenance and evidence manifests accompany this report.

## What changed

`NativeQwenModel.from_q8(..., projection_profile="q8-residual")` now routes Q8
model projections, including the output head and both TP devices, through the
qualified vector group32 integer-dot kernel. Original Q8 weight codes and F16
scales are unchanged. Each FP16 activation group is represented by a high A8
pass plus an A8-quantized residual, each with FP32 scales. Integer dot products
feed a changed FP32 reduction, followed by FP16 output storage. **This is not
original FP16 arithmetic, nor bit-identical original-profile model state.**
Embedding lookup, small FP32 parameters, normalization, GDN, convolution history
and FP16 KV storage retain their original implementations. Pinned-host TP still
transfers FP16 activations/outputs; this is not compressed inter-GPU transport.
Model-owned primary/peer scratch is allocated once, memory-admitted and counted;
existing model serialization protects it. At the qualification capacity of 32
rows, additional scratch is 1,253,376 bytes per device; benchmark capacity 256
scales accordingly. Each request still owns independent recurrent/KV state.

Default arithmetic is unchanged. Additive ABI-16 creation/info exports leave old
ABI-16 default libraries compatible; explicit profile selection fails with a
rebuild instruction if the exports are absent. Descriptor sizes remain
104/40/24/264. Cached metadata preserves model-close/session-lifetime behavior.
Only complete resident FP16 targets accept the profile: single GPU, TP and
fixed-cohort target batching. Single-request MTP uses the new profile for its
**target only**, keeping original FP16 draft arithmetic. No PP/staged, streaming
weight, BF16, SM89, sampling, continuous-admission, paging or graph qualification
is implied. The one-pass A8 prototype is **not** exposed as a model profile.

Exact library SHA256:
`79a31157faea8ce166e755fea3619969e110cab1c9c9f820a5c89de05ab226a7`.
All six CUDA source hashes match every executed snapshot. `source/` contains
current tested sources; `executed-source-deltas/` preserves earlier Python files
where metadata, CLI plumbing or diagnostic flags differed. The final snapshot
matches all 221 executed current source/test/tool hashes. Compilation occurred
on WZU_Server; no local Rust/CUDA compilation, commit or push was performed.

## Correctness before timing

- Initial profile fixture gate: 16 passed. Default regression gate: 130 passed /
  125 skipped. Expanded profile gate: **24 passed**, including scalar/TP/batch,
  ragged positions, EOS/budgets, selected reset, failure poisoning, source guards,
  raw-C invalid calls, shared ownership, lifetime and Torch-math rejection.
- Original FP16 control still matches the pinned full-model native logits,
  hidden values and GDN/history/KV bytes at the two fixed CPU-reference steps.
  Thus the default path was not silently replaced by alternate arithmetic.
- Both profiles pass the **unchanged** fixed CPU-logit gates: identical top1,
  relative L2 <=0.02, centered cosine >=0.999, KL <=0.002. Residual profile L2
  is 0.002934/0.002603 at those two steps; the independent CPU reference and
  original native snapshots are hash-pinned in `qualification/report.json`.
- Across **98 additional teacher-forced steps** on the original four trained
  trajectories, maximum L2 versus original-profile native logits is 0.0029351,
  minimum centered cosine 0.99998759 and maximum KL 0.000093657. All top1 gates
  pass. Actual native generation preserves all original story/capital/Python/
  arithmetic outputs (32/2/32/32 tokens), without Torch model math.
- Eight heterogeneous requests with different prompts/budgets/EOS behavior match
  their respective scalar new-profile outputs and committed cache bytes exactly.
  Diagnostic versus native generation state is also exact **within the profile**.
  These are not claims of bit-equal states across arithmetic profiles.
- Natural one-proposal MTP passes all four trained cases with exact same-profile
  ordinary-target outputs and committed state. Fixture tests additionally force
  accept/reject, reset/failure and EOS boundaries. Story acceptance remains
  **14/16 = 87.5%**, with two rollbacks/replays; no forced proposals in timing.
  The unchanged draft's independent original-source FP64 block gate is reused
  from immutable `v100-native-mtp`, hash-pinned in `qualification-reuse.json`;
  `qwen_mtp.cuh` is byte-identical to that qualified draft source. A missing
  duplicate report under the remote source checkout is not a missing local
  reference, and no fresh draft-reference run is claimed.
- Final full suites: **V100 1020 passed / 440 skipped; local 760 / 700**. A real
  CLI invocation selects the profile, reports two passes and emits the pinned
  eight-token smoke output. Skipped optional/hardware tests are not passes.

This is limited trained-case qualification against represented/native/CPU
references, **not original-BF16 or corpus-level quality equivalence**. Changed
arithmetic must remain explicit. No quality thresholds were weakened.

## Matched workload and precision disclosures

Same two Tesla V100-PCIE-32GB GPUs, logical order `CUDA_VISIBLE_DEVICES=1,0`, no
NVLink, explicit native pinned-host TP. Same official Qwen3.5-27B source revision
`fc05daec18b0a78c049392ed2e771dde82bdf654`, original-derived target Q8 artifact
`3d635b87db004f42aefd6730e372e530`, FP16 KV, 23 formatted prompt IDs, 32 fixed
greedy outputs, EOS ignored for this speed case, no input-cache reuse, eight
preallocated slots at capacity 256. Original/residual use the **same library**.
Native batching uses a 25 ms benchmark admission window included in HTTP wall;
every promoted wave is verified unsplit. Default adapter window stays 5 ms.

Pinned llama.cpp production-fork binary SHA256:
`e5b1b5782cd5b4acec5316ac5bd40123e0b0c9826643ae4148b0ec6e359f5fa2`.
Its original-source Q8_0 GGUF SHA256:
`eb07277161915232e227a38608fc3ea80c0c17dc1c1583a9f3c5671edb6b5513`.
Commands, loaded maps and responses are retained. Same tensor split, FP16 KV,
Flash Attention, backend sampling and supported graphs/optimized kernels remain
enabled; speculation-off and MTP1 are distinct controls. The binary is pinned;
its build-info source hint is not a verified exact source commit.

**Q8 labels do not establish arithmetic parity.** Native and GGUF have different
encodings/rounding and selected small-parameter precision. The previous actual
competitor trace confirms Q8_1 activation-packing kernels, unlike original
native FP16 math or this explicit compensated A8 profile. This is a practical
same-original-source engine comparison, not a represented-weight exact-math
comparison. Previous failed C8 MTP configurations remain in older evidence;
they were not retried unchanged, counted as zero TPS, or treated as native wins.
Best-supported depth/configuration optimization remains part of the active goal.

## Unprofiled speed: three repeats per promoted cell

Complete-cohort HTTP TPS = **total outputs / global cohort wall**. Never sum
concurrent request latencies. Warmups are separate; telemetry is sampled during
these unprofiled runs, and CUDA tracing is performed only afterward.

| Requests | Native original | Native residual | Internal gain | llama.cpp non-spec | Residual / llama |
|---:|---:|---:|---:|---:|---:|
| 1 | 15.19 | 18.05 | +18.9% | 24.50 | 0.737x |
| 2 | 21.56 | 26.59 | +23.3% | 37.98 | 0.700x |
| 4 | 26.59 | 33.28 | +25.2% | 50.67 | 0.657x |
| 8 | 28.81 | 37.16 | +29.0% | 66.83 | 0.556x |

| Single request | Original | Residual | llama.cpp |
|---|---:|---:|---:|
| Non-spec post-first TPS | 18.90 | 21.76 | 34.29 |
| MTP1 post-first TPS | 24.55 | 29.27 | 46.64 |
| MTP1 complete-request HTTP TPS | 18.47 | 22.65 | 29.09 |
| Non-spec native TTFT, seconds | 0.4241 | 0.3073 | not same timing scope |
| MTP1 native TTFT, seconds | 0.4341 | 0.3184 | not same timing scope |

Post-first native C1 rate is 31/(generation minus native TTFT); llama's rate is
31/engine-reported predicted_seconds. Neither is a streaming client TTFT or
complete-request rate. Nonstream HTTP includes reset/admission/adapter overhead.
Native internal TTFT excludes prior HTTP/admission/reset time; scalar C-entry can
include model-mutex wait. Cohort post-first C8 rate improves 44.82 -> 56.49 TPS
using **248/(last native finish minus shared first-token time)**, not summed
per-client times; it is not 56 TPS per client or an HTTP/competitor rate.

## CPU and latency, independently reported

| Requests | Global wall s, original -> residual | Process CPU core-s | Mean individual HTTP completion s |
|---:|---:|---:|---:|
| 1 | 2.1070 -> 1.7727 | 2.083 -> 1.747 | 2.1063 -> 1.7720 |
| 2 | 2.9681 -> 2.4070 | 2.947 -> 2.383 | 2.9671 -> 2.4061 |
| 4 | 4.8143 -> 3.8460 | 4.793 -> 3.823 | 4.8122 -> 3.8439 |
| 8 | 8.8849 -> 6.8897 | 8.893 -> 6.890 | 8.8809 -> 6.8862 |

Single MTP1 wall improves 1.7326 -> 1.4129 s, CPU 1.733 -> 1.413 core-s.
Native polling remains CPU-active; lower core-seconds here largely reflect less
elapsed work, not elimination of CPU usage. Fresh llama non-spec C8 wall is
3.8306 s, CPU 3.843 core-s, mean individual completion 3.7779 s. MTP1 C1 wall is
1.1000 s, CPU 1.097 core-s. Trial ranges and per-request completions are raw;
three repeats are not a universal or confidence-bound performance guarantee.

## NVML diagnostics and memory

Timed-window telemetry at approximately 100 ms is retained for both physical
GPUs. At C8, primary/peer mean GPU busy changes **87.9/71.3% -> 83.6/63.1%**,
while whole-cohort throughput rises. Memory-controller busy changes
27.9/25.8% -> 35.2/32.8%; this is **not DRAM bandwidth in GB/s**. Mean power is
227.4/196.9 -> 132.8/105.3 W, temperatures 58.7/53.6 -> 46.5/43.1 C; these are
sampled observations, not an energy-efficiency integral or controlled thermal
experiment. SM clocks stay approximately 1380 MHz, memory 877 MHz; no clocks,
power limits or profiling permissions were changed.

C8 device-wide VRAM peaks are 20,233/17,225 -> 20,235/17,227 MiB. These include
original unrelated allocations and CUDA/runtime/loader memory, not just native
model bytes. The exact model scratch query and memory admission are separate.
MTP1 primary peak 22,433 -> 22,435 MiB, peer 17,225 -> 17,227 MiB. Fresh llama
non-spec C8 peaks 17,515/17,977 MiB; MTP1 C1 18,373/18,835 MiB. No claim that
all these allocation/lifetime policies are identical. Complete telemetry for
all matrix cells is in `metrics.json`.

## Separate CUDA timelines and compiled resources

Instrumented runs deliberately use **23 prompt / 8 output** tokens, not the
32-output unprofiled campaign. Both C1 and C8 traces have one native-call NVTX
range; a legacy attribution field `requests: 1` means one C-call, **not one
logical request in C8**. Metadata identifies eight logical requests explicitly.
Summed kernel durations across two devices exceed wall time and are attribution,
not TPS. Gaps refer to this process's captured activities, not device-wide idle.

| Diagnostic | C1 original -> residual | C8 original -> residual |
|---|---:|---:|
| Kernel launches | 11,696 -> 18,112 | 30,634 -> 41,050 |
| Summed projection time, including quantization | 977.88 -> 640.44 ms | 6254.20 -> 3949.16 ms |
| Projection share of summed kernels | 91.45 -> 87.49% | 90.96 -> 86.41% |
| GDN recurrent summed kernel time | 56.34 -> 56.36 ms | 491.51 -> 490.81 ms |
| Gaps between captured activities | 113.73 -> 114.26 ms | 240.13 -> 241.52 ms |
| H2D bytes | unchanged 293,048,560 | unchanged 2,344,388,480 |
| D2H bytes | unchanged 293,048,448 | unchanged 2,344,387,584 |

The kernel replacement reduces time despite adding activation-quantizer launches;
this is not a graph/launch-count improvement. Projections still dominate. TP
transport did not shrink, and two-device overlap drops as kernels shorten.
C8 simultaneous two-device kernel time changes 2854.34 -> 1703.80 ms; this is
not achieved SM occupancy. Residual quantization itself is only 46.60 ms summed
at C8, so packing reuse alone cannot plausibly close the entire external gap.

Read-only resource queries on this exact binary report 128-thread residual
T1/T2/T4 kernels at 64/62/78 registers and 50/50/37.5% **theoretical** occupancy.
The 256-thread quantizer uses 38 registers, 75% theoretical occupancy. No local
spills in these functions. Actual achieved occupancy, compute throughput, DRAM
GB/s, cache/stall counters remain **unavailable** under unchanged
`RmProfilingAdminOnly: 1`; no repeated failing counter campaign or driver-policy
changes were made. Lower theoretical occupancy can accompany faster kernels.

## Remaining gap and next hypothesis

Every promoted cell improves versus native original, but **every external cell
still loses**. At C8, native residual needs about 1.80x its current throughput
merely to match the fresh llama non-spec control, more to clear the >=5% goal.
MTP1 native post-first decode remains about 0.628x the competitor's reported
post-first rate. No goal-completion or universal speed claim is justified.

Next investigate explicit reuse of identical inputs across Q/K/V and gate/up
projections and the corresponding duplicate TP transport. Such reuse needs
forward/chunk/head-boundary invalidation, failure/reset handling and peer events;
pointer/shape equality alone is unsafe when scratch is overwritten next token.
Further projection/reduction and batching choices need new correctness and 1/8
unprofiled screens before another promoted matrix. Do not reintroduce the
previously regressing T8 or CPU-peer-worker paths without new evidence.

All owned workers/profilers/servers are terminal; loopback port 18947 is closed.
Unrelated services remain untouched; original physical-GPU footprints restored
at **3651/3189 MiB**, both idle. Large `.pt`, `.sqlite`, `.nsys-rep` and library
files remain remote with hashes in retained-file/build records; collected local
JSON/logs/commands are not substitutes for missing traces. No files in the 19
previous frozen capsules were changed. The active goal is **not complete**.
