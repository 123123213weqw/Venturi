# V100 CUDA occupancy and launch bounds

The CUDA tuning improves native Qwen Q8/FP16 speed on the two V100s while
preserving the qualified numerical results. Single-request TP decode improves
17.14 to **18.86 TPS**; eight-request PP improves 28.31 to **33.19 aggregate
decode TPS**. This is not 33 TPS per request or a matched llama.cpp comparison.

**Achieved occupancy is not measured.** The current non-admin runner has no
effective capabilities and the driver still reports `RmProfilingAdminOnly: 1`.
Existing counter-denial evidence remains applicable; no repeated NCU campaign,
driver privilege changes, clock settings or service stops were performed.
The new measurements below are CUDA-API **theoretical residency limits**, not
hardware-counter occupancy, SM throughput or DRAM bandwidth.

## Compiled kernel residency

Both GPUs report 80 SMs, 2048 resident threads/SM, 64 warps/SM, 65536 registers/SM
and 96 KiB shared memory/SM. The actual Q8 launch remains 256 threads/block.

| Specialization | Before registers per thread | After registers per thread | Before resident blocks per SM | After resident blocks per SM | Before theoretical occupancy | After theoretical occupancy |
|---|---:|---:|---:|---:|---:|---:|
| Single-token decode, four output rows | 64 | 56 | 4 | 4 | 50% | 50% |
| Four-token prefill tile, one output row | 44 | 40 | 5 | 6 | 62.5% | 75% |

The promoted binary has zero compiler-reported spill stores/loads for both
FP16 specializations. Decode gets faster **without increasing its occupancy
ceiling**. Its changed code generation matters; these data do not identify a
unique instruction, cache or stall cause without the missing hardware counters.

`tools/inspect_cuda_occupancy.py` checks the library SHA256, extracts its compiled
cubins, loads them using the CUDA driver API, queries function resources and
calls `cuOccupancyMaxActiveBlocksPerMultiprocessor`. Trace joins use each
request's submitting thread and API correlation. Recorded register/shared usage
must match the compiled symbol. Device-wide small-grid residency bounds are
reported separately from the per-SM resource ceiling. The tool launches no
kernels and does not change an engine's block size.

The previous native traces attribute 90.42% of TP and 89.84% of PP summed
decode kernel time to Q8 projections. TP's smallest Q8 grid has 128 blocks:
its grid-limited device-average residency ceiling is only 20%, despite a 50%
resource ceiling. PP's corresponding 256-block grid has a 40% grid ceiling.
Neither number is achieved occupancy. Tiny single-block normalization kernels
also cannot fill 80 SMs, but their kernel-time contribution is much smaller.

The [CUDA driver occupancy API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__OCCUPANCY.html)
defines the maximum active-block calculation. The compiler's register tradeoffs
are described in NVIDIA's [launch bounds documentation](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#launch-bounds).

## Counterexamples to maximizing occupancy

Two experimental variants apply a minimum of six/eight blocks to **all** Q8
tiles. Both pass byte-parity fixtures and all 22 real-matrix/token microbenchmark
cases, but they are not promoted.

For a real 10240 by 5120 projection, one FP16 input token:

| Variant | Registers per thread | Theoretical occupancy | Compiler spill stores and loads | Median event time per projection |
|---|---:|---:|---|---:|
| Original | 64 | 50% | No local memory in queried function | 0.12681 ms |
| Forced six blocks | 40 | 75% | 8 B stores, 16 B loads | 0.24441 ms |
| Forced eight blocks | 32 | 100% | 148 B stores, 172 B loads | 1.01280 ms |

Spill byte counts are compiler reports, not measured runtime traffic. Higher
theoretical occupancy accompanies approximately 1.93x and 7.99x slower decode
projections. Four-token and 26-token prefill projections often improve with the
six-block hint, without spilling. The accepted version therefore uses:

```cpp
#if defined(VENTURI_BLOCKS_ARCH) && VENTURI_BLOCKS_ARCH == 70
__launch_bounds__(256, (Tokens > 1 ? 6 : 1))
#endif
```

This also tells the compiler that decode launches at most 256 threads, producing
56-register code rather than the original 64-register version. The internal
`prefill-only` experiment label predates that observation; the accepted change
is **token-aware launch bounds**, not literally prefill-only. SM89 does not
receive the annotation. There are no Qwen dimension/layer special cases, new
weight layouts, precision substitutions, Tensor Core arithmetic or changed
FMA/reduction order. C ABI 14 is unchanged.

Microbenchmarks share the same resident Q8 planes and activation bytes among
libraries. Each case checks exact output bytes before timing, uses three warmup
graph replays, then four alternating-order waves of 25 calls per library.
Projection-only graph replay does not establish whole-model CUDA graph support.
Event times are not complete engine TPS or hardware-counter bandwidth.

## Correctness before engine timing

- Each experimental library: **72 passed / 69 skipped**, including Q8 tails,
  guard regions and native single/TP/PP state and generation checks.
- Full trained model: all two-step logits, hidden values, GDN/history and KV
  bytes match the pinned previous-native reference; unchanged fixed CPU gates
  pass. Native generation with Torch math forbidden returns `[57590, 248046]`
  and finishes at cursor 27.
- V100 complete suite: **839 passed / 384 skipped**. The explicit original-library
  byte-comparison tests are enabled, explaining the difference from earlier suite
  skip counts. Local final suite: **729 passed / 494 skipped**.
- Final source containing the promoted annotation is hash-identical to the
  executed candidate's three native source files. Its final V100 postcheck:
  **80 passed / 69 skipped**.
- All 32 pinned story IDs match for every warmed/timed request at concurrency
  one/eight in both TP and PP.

An initial complete-suite attempt failed two launcher tests because `scripts/`
was omitted from the remote source sync. It stopped before engine throughput
timing. After syncing the missing scripts, the entire suite passed before the
ABBA measurements. The failed log, exit code 1 and recovery record remain under
`v100/token-aware-initial/` and `v100/token-aware/prerequisites.json`. No numerical
gate was weakened and no native change was needed for that recovery.

These are regression checks against the represented Q8/FP16 profile, not
original-BF16 corpus quality or an independent upstream architecture audit.

## Matched unprofiled engine measurements

Two Tesla V100 PCIE 32 GB cards, `CUDA_VISIBLE_DEVICES=1,0`; logical GPU 0 is
physical GPU 1. Explicit pinned-host DMA transport remains selected because
previous actual peer-copy probes were unreliable. PP uses a 32/32 layer split;
TP retains whole-request serialization, not batched TP.

Official source `Qwen/Qwen3.5-27B`, revision
`fc05daec18b0a78c049392ed2e771dde82bdf654`, original-derived Q8 artifact
`b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8`.
Weights, FP16 activations/KV, formatting, 23 prompt IDs, context/capacity and
32 greedy output IDs remain fixed. EOS stopping is disabled for fixed work;
none of these 32 IDs is EOS. No MTP, continuous batching or paged KV.

Each mode uses baseline/candidate/candidate/baseline phases, one trial per
concurrency per phase: **two trials per version/mode/concurrency**. Model load,
session setup/reset and a full-length warmup for each count are outside timing.
A common barrier measures total cohort wall time. A one-token first-token wave,
31-token continuation wave and separate unsplit 32-token call expose distinct
scopes. Aggregate rates divide total outputs by cohort wall, never summed
request latencies. Two trials are not production tail statistics.

| Mode and requests | Decode TPS before | Decode TPS after | Complete cohort wall before | Complete cohort wall after | Process CPU before | Process CPU after |
|---|---:|---:|---:|---:|---:|---:|
| TP 1 | 17.14 | 18.86 | 2.29357 s | 2.06736 s | 2.29581 core-s | 2.06933 core-s |
| TP 8 | 17.14 | 18.91 | 18.33206 s | 16.54439 s | 18.41820 core-s | 16.59154 core-s |
| PP 1 | 14.91 | 17.68 | 2.84802 s | 2.42845 s | 2.85092 core-s | 2.43147 core-s |
| PP 8 | 28.31 | 33.19 | 12.26271 s | 10.56498 s | 22.99683 core-s | 19.85303 core-s |

TPS in the table is **aggregate continuation/decode TPS**. Complete-cohort wall
and CPU columns are from the independent unsplit 32-output call. Single-request
TP first-token wall improves 0.49082 to 0.43051 s; PP 0.77923 to 0.68403 s.
Eight-request PP first-token cohort wall improves 3.65572 to 3.22350 s and mean
individual complete-request latency improves 12.14042 to 10.46081 s. The eight-
request TP first-token candidate range is 3.40881 to 3.77110 s: its first-token
gain is noisier than decode. Raw min/max trial values are in `metrics.json`.

## GPU telemetry and separate timelines

The samples below are device-wide means in the unprofiled timed windows,
including existing idle service allocations. They are not achieved occupancy
or DRAM bandwidth. No clocks were forced.

| Eight-request PP metric | Physical GPU 1 before | Physical GPU 1 after | Physical GPU 0 before | Physical GPU 0 after |
|---|---:|---:|---:|---:|
| GPU busy | 89.11% | 89.57% | 94.01% | 92.94% |
| Memory controller busy | 48.88% | 56.26% | 52.66% | 60.14% |
| Used VRAM | 18481 MiB | 18481 MiB | 18943 MiB | 18943 MiB |
| Power | 215.39 W | 229.24 W | 215.91 W | 232.56 W |
| Temperature | 53.71 C | 54.40 C | 52.46 C | 53.38 C |
| SM clock | 1374.56 MHz | 1326.75 MHz | 1380 MHz | 1349.24 MHz |
| Memory clock | 877 MHz | 877 MHz | 877 MHz | 877 MHz |

Candidate PP clocks range down to 1267/1305 MHz respectively under the unchanged
device policy. Speed improves despite that clock behavior, but power increases;
these sampled means do not establish energy per token. Single-request TP GPU
busy actually decreases while TPS increases. Full telemetry for every mode and
count is retained in `metrics.json` and raw CSVs.

Separate Nsight captures use eight outputs, seven continuation steps, one
request for TP and eight for PP, `--sample=none --cpuctxsw=none --trace=cuda,nvtx
--cuda-event-trace=false`. Both reports complete with the expected outputs.
Instrumented durations are not substituted for the engine timings above.

New TP continuation attribution: 410.07 ms **summed across devices** in kernels,
94.08 ms gaps between this process's CUDA activities, 5621 H2D and 5621 D2H
copies. New PP eight-request continuation: 3155.46 ms summed kernel time,
1464.42 ms simultaneous two-device kernel coverage and 4.26 ms activity gaps;
112 H2D and 112 D2H copies. TP transfers/host gaps remain a meaningful separate
bottleneck; the launch-bound change does not remove them. Kernel sums can exceed
wall because devices overlap. Activity gaps are not device-wide idle time and
cross-device coverage is not SM occupancy.

## Source and reproduction

Native change: `src/venturi/ops/csrc/blocks.cu`, the `q8_tiled` declaration only.
Rebuild explicitly on the CUDA server with the existing SM70 build command;
the source hash changes the build cache key. TP/PP CLI options are unchanged.
The exact tested library remains at
`/home/wzu/codex-build/Venturi/v100-occupancy-prefill-v1/candidate/libventuri_blocks.so`,
SHA256 `f64d8b6561751407f364cd510676e726d0085688d44ef990a9bc60740190898b`.
Baseline SHA256 is
`29c2a8cc614387a9029b1cfd16dc78ebd3d9524f4083e1293eea6ea21737e6d4`.
Compiler commands, logs, source hashes, SASS, API resource queries, CPU/wall/raw
timings, NVML CSVs and correctness reports are retained here. Large `.so`, model,
state snapshots and Nsight binary/SQLite files stay on the V100 server.

Run `python3 benchmarks/v100-cuda-occupancy/summarize.py` to recompute metrics.
Prior evidence capsules remain immutable. No commit or push was made. Native
batching/paging, MTP, full-model CUDA graphs, long context, upstream/corpus
quality and matched external-engine wins remain unfinished.
