# Native Qwen target cohort batching on two V100s

## Status

**Implemented and qualified; external speed goal is not complete.** This first
native batching path improves parallel throughput, not single-request decoding.
It is opt-in target cohort batching, not continuous admission, PP/MTP batching,
paged attention or whole-model CUDA graphs. Earlier evidence capsules are unchanged.

## Matched workload and precision

Two Tesla V100-PCIE-32GB GPUs, logical order `CUDA_VISIBLE_DEVICES=1,0`, TP with
explicit pinned-host DMA transport. No peer-access fallback, service restart,
clock/power change or profiling-policy change. CPU threads: 4. Eight preallocated
slots, capacity 256 each. Same formatted 23 prompt IDs, 32 greedy outputs, EOS
ignored, no prompt reuse, one full warmup at each concurrency, three measured
repeats at 1/2/4/8 requests. Python HTTP is loopback only. Rate is **total outputs /
global complete-cohort HTTP wall**, including reset, admission and scheduling.
Native single post-first-token rate is a different scope.

Target: official Qwen3.5-27B revision
`fc05daec18b0a78c049392ed2e771dde82bdf654`, NInfer artifact
`3d635b87db004f42aefd6730e372e530`, target SHA256
`b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8`.
Native Q8/group32 with FP16 scales and activations/KV, FP32 GDN state; original
small-parameter representation retained. The new grouping changes no Q8 codes,
rounding, FMA or reduction order.

The competitor is the existing **pinned production llama.cpp fork**, not latest
stock upstream. The same-source original Q8_0 GGUF is reused, SHA256
`eb07277161915232e227a38608fc3ea80c0c17dc1c1583a9f3c5671edb6b5513`.
Its Q8 encoding/scales, 96 alpha/beta Q8 matrices versus native widened original
BF16 parameters, and internal activation arithmetic are **not identical** to
native. This is a declared practical engine comparison, not exact-math parity.
The detailed conversion audit is in the unchanged `v100-llama-comparison` capsule.

## Implementation

Blocks ABI **16**, explicit remote rebuild required. The library hash is
`d0961ff33c9c8635915359f4af23fd2451899387cc282e38e404f4a7807ffeee`.

- `NativeQwenModel.create_batch(capacities)` / `NativeQwenBatchSession`: bounded
  selected slots, independently owned history/GDN/KV/cursors, one native generate
  call, heterogeneous prompts, budgets and EOS, active-row compaction.
- `qwen_batch.cuh`: grouped normalization/Q8 projections/MLP/head, sequence-local
  convolution/GDN and attention updates; ragged prefill chunks share workspace.
  The vocabulary head is evaluated only for each request's final prefill row.
- Shared scratch uses `Session(..., caches=false)`; immutable model weights are
  shared. No extra full-model or per-request GDN/KV allocation for the scratch.
- Invalid preflight does not mutate state. Execution errors drain streams and
  poison selected participants until reset; other slots remain separate. Owned
  CPU diagnostic logits/snapshots and model-close lifetime are tested.
- The loopback admission adapter groups queued submissions, never runs Python
  token/layer math. Its **default 5 ms window fragmented the initial C8 screen
  into 6+2**, which failed the full-cohort gate. The repeated matrix explicitly
  uses **25 ms**, included in HTTP wall; every timed cohort is verified unsplit.
  This tuning does not implement admission while a native generation call runs.

## Correctness

- Local full suite: **756 passed, 558 skipped**.
- V100 batch fixtures: **14 passed, 13 skipped**; scalar/TP/PP/MTP regressions:
  **50 passed, 46 skipped**; admission/HTTP fixtures: **19 passed**.
- V100 full suite: **856 passed, 458 skipped**. Native BF16 is unavailable on
  SM70. Unchanged historical projection-baseline comparisons are not rerun by
  pretending the new ABI library is its own baseline.
- Real 27B: eight requests comprising reordered copies of four trained prompts,
  lengths 22/23/26/28, EOS termination and output budgets 8/13/32. Every generated
  ID and committed convolution/GDN/KV byte matches its scalar request. Story IDs
  also match the frozen reference. Reordered subset continuation at unequal
  positions gives exact final-row logit bytes and leaves all slot states exact.
- Production generation passes with Torch math/allocation entry points patched
  to raise. Fixture checks additionally cover failures/reset, selected subsets,
  tied vocabulary, model-close lifetime and concurrent independent batch owners.
- This reuses previously qualified represented-weight target operators; it is
  not a new original-BF16 or broad corpus-quality campaign. No MTP batch claim.

## Three-repeat unprofiled results

Complete-cohort HTTP output TPS:

| Requests | Native scalar TP | Native batched TP | Batch / scalar | llama.cpp tensor, MTP off |
|---:|---:|---:|---:|---:|
| 1 | 15.37 | 15.23 | 0.99x | 24.42 |
| 2 | 15.40 | 21.63 | 1.40x | 39.74 |
| 4 | 15.39 | 26.62 | 1.73x | 50.29 |
| 8 | 15.44 | 28.84 | 1.87x | 63.90 |

Single native post-first-token decode: **18.94 -> 18.92 TPS**, effectively
unchanged; HTTP C1 regresses about 0.9% with the explicit admission window.
All valid timed responses retain the same 23 fresh input IDs and pinned 32 output
IDs. CPU, latency and per-trial variation are retained in `metrics.json` and raw
responses. Trials are sequential configurations, not an interleaved randomized
study; clocks/temperatures are recorded, and no 5%-external-win claim is made.

C8 native scalar -> batch: global wall **16.580 -> 8.877 s**, server CPU
**16.72 -> 8.88 core-seconds**; mean request completion **9.368 -> 8.873 s**,
maximum observed completion **16.592 -> 8.876 s**. Early scalar requests can
finish much sooner than a cohort; throughput/tail improvement is not a universal
per-request latency win. Native C8 batch first-token time is **3.313 s** from the
common native call, including all cohort prefill. This nonstreaming HTTP test
does **not** measure streaming client TTFT. Scalar native C-entry timings can
include model-mutex waits but exclude preceding reset/HTTP queue time; do not
compare them as complete client TTFT. Native counters show 6 prefill groups +
31 decode groups and 256 head rows, not eight independent 32-step native loops.
Batch native C8 post-first-token rate is **44.88 TPS**, using 248 subsequent
outputs / (last native cohort finish - shared first-token time), not summed
request durations. This excludes prefill/admission and is not comparable to the
63.90 complete-request HTTP rate. Prefill therefore remains a substantial cost.

The new non-speculative llama.cpp control remains **2.22x faster at C8**. Its
C8 CPU is **4.02 core-seconds** and mean completion **3.945 s**. The native goal
of winning by >=5% at single and parallel counts is not reached.

## GPU diagnostics

Measured-window NVML samples are at 100 ms, filtered to the unprofiled timed
waves; values include existing unrelated allocations. Native C8 scalar -> batch:

| Physical GPU | Busy % | Power W | Device MiB | Mean SM MHz |
|---|---:|---:|---:|---:|
| 1, primary | 73.1 -> 88.1 | 187.1 -> 228.0 | 20,209 -> 20,233 | 1378.8 -> 1375.4 |
| 0, secondary | 60.8 -> 71.5 | 169.8 -> 198.0 | 17,225 -> 17,225 | 1380 -> 1380 |

Memory clock is 877 MHz. Temperatures and clock ranges are in `metrics.json`;
no fixed-clock control is claimed. The faster llama.cpp C8 control has only
55.8/56.1% NVML busy and about 122.9/118.6 W: higher busy percentage is not proof
of better inference efficiency. Memory-controller busy is not GB/s.

Separate CUDA/NVTX attribution and static launch/residency inspection are stored
under `v100/diagnostics-v6/`. They use 23 prompt IDs and **8** outputs per request,
not the timed 32-output matrix. Scalar has eight native-call NVTX ranges; batch
has **one range for eight logical requests**. The legacy trace field `requests`
counts ranges, not batch logical requests. Profiler timings are not throughput.
`--cuda-event-trace=false` avoids the previously diagnosed event-tracing issue.

| Separate 8-output diagnostic | Scalar | Batch |
|---|---:|---:|
| Kernel launches | 93,568 | 30,634 |
| Q8 projection share of summed kernel duration | 91.45% | 90.95% |
| Gaps between this process's CUDA activities | 907.1 ms | 234.6 ms |
| H2D / D2H copy counts | 51,392 / 51,392 | 10,429 / 10,424 |
| H2D / D2H bytes | 2.3444 / 2.3444 GB | 2.3444 / 2.3444 GB |

These sums include overlapping GPU activities; they are not wall-time fractions
or achieved bandwidth. Grouping cuts launch/copy counts and timeline gaps, but
Q8 projections still dominate. The batch Q8 T4/R1 kernel has 40 registers/thread,
256 threads/block and **75% theoretical** residency; scalar T1/R4 has 56 registers
and **50% theoretical** residency. These are existing arithmetic kernels, not
a new Tensor Core GEMM. Raw launch shapes, resources and grid bounds are retained.

**Achieved occupancy, compute/DRAM/cache/stall counters are unavailable** under
unchanged `RmProfilingAdminOnly: 1`. CUDA occupancy API results are theoretical
residency bounds only, never substituted for achieved occupancy. No new NCU
replay campaign or driver privilege change is attempted.

## Failed experiments and remaining work

1. Initial GPU fixture failed because the expected `[N,V]` result was mistakenly
   concatenated into `[N*V]`; values were identical. The test was corrected to
   stack rows without loosening numerical gates; the failed source/log remains.
2. Initial 5 ms admission split C8 into 6+2. Its 23.95 HTTP TPS is only a fragmented
   screen, not the full C8 batch result. The 25 ms recovery and all repeats retain
   cohort IDs, actual request counts and shared-counter scope.
3. The fresh full llama.cpp layer/MTP1 matrix completed three C1 repeats
   (26.50 complete-request TPS) but disconnected during C2 after its warmup. The
   failed arm is retained, not scored as zero or a native win. A bounded fresh-C8
   recovery also disconnected before any valid measured C8 wave. Its server log
   and responses remain separately preserved; no endless retry campaign or
   blanket claim that MTP never works (the prior screen had a valid C8 result).

Next priorities: accelerate the measured Q8 projection/GEMM path (including
prefill) rather than equating batching with an optimized GEMM; screen exact-order
larger token tiles with proper register bounds before promotion. Single-request
projection/TP transfer/launch costs still require work. Any activation-quantized
or Tensor Core arithmetic alternative needs a separately named precision profile
and new numerical/quality gates. Continuous admission, PP/MTP batching, paging,
sampling and broad checkpoint/hardware coverage remain unfinished.

Raw failures, per-trial responses, telemetry, launchers, compiled-source hashes,
retained-trace hashes and exact source snapshots are preserved in this capsule.
No commit/push or local Rust compilation is performed.
