# V100 SwiGLU + two-pass activation quantization fusion

## Decision

Retain the standalone research producer; **do not enable it by default or deploy
it to production yet**. Both 128- and 256-thread variants passed the predeclared
operator gate. The 256-thread variant is a useful integration candidate, not a
proven whole-model winner. Down projection remains a separate kernel.

The fast path replaces:

```text
gate/up F16 -> SwiGLU -> global F16 temporary -> two-pass A8 workspace
```

with:

```text
gate/up F16 -> fused SwiGLU + F16 rounding + two-pass A8 workspace
```

It preserves the old F16 rounding boundary in registers, both residual-A8 code
planes, FP32 activation scales, Q8 weight bytes and FP32 Down partials. It does
not use the earlier single-A8 experiment. Fusion also specializes the fixed
two-pass loop; the measured improvement is not attributable solely to launch
removal.

## Correctness

- **122 checks passed on the V100** before speed testing.
- Packed workspace bytes and all Down output bytes match the pinned separate
  implementation in every benchmark case.
- Tests cover group tails, guards, all 65,536 F16 gate bit patterns with six up
  values, zero/subnormal/extreme/nonfinite inputs, allowed exact debug aliases,
  forbidden overlaps, and CUDA-graph replay with changing inputs.
- Finite/Inf intermediate F16 bits and NaN masks match; debug NaN payloads are
  not a public contract. Invalid-group packed scales remain canonical and exact.
- Independent FP64 sampled Down checks also pass. This is not a full-model
  activation-distribution, quality or serving correctness test.

## Canonical unprofiled result

Use **attempt-v2**, not the larger preliminary downstream gains from v1. V2
uses the **same intermediate/workspace/Down-output addresses for every timed
arm**, and checks captured-graph outputs. V1 used per-arm allocations and is
preserved as an allocator/cache-placement-confounded preliminary experiment.

Confirmation, layer 0, pinned separate versus fused-256, GPU-event medians:

| Tokens | Scope | Separate, us | Fused, us | Latency reduction |
| ---: | --- | ---: | ---: | ---: |
| 1 | SwiGLU + quantization | 4.529 | 2.928 | 35.36% |
| 8 | SwiGLU + quantization | 5.432 | 3.511 | 35.36% |
| 1 | Producer + Down partial | 66.880 | 65.092 | 2.67% |
| 8 | Producer + Down partial | 113.893 | 111.697 | 1.93% |
| 1 | One-rank MLP projection chain | 180.706 | 179.017 | 0.93% |
| 8 | One-rank MLP projection chain | 277.144 | 273.622 | 1.27% |

The one-rank chain includes input quantization, gate/up projection, producer
and Down partial. It **excludes** normalization, residuals, TP communication/sum,
attention, GDN, other layers and serving. Tokens are operator rows, **not 8
concurrent API requests**. There is no whole-model TPS measurement here.

The screen covers layers 0/31/63 and tokens 1/2/4/8/23/32, 18 cases. Fused-256
median screen latency reductions are 33.66% for the producer, 2.27% with Down,
and 0.92% for the one-rank projection chain. There is a separate T1/T8
confirmation. The gate was >=5% median producer improvement and >=1% median
producer+Down improvement in both screen and confirmation, with no >2%
confirmed producer+Down regression. These are **operator**, not model gates.

### Workload and timing controls

- WZU_Server, physical GPU0 V100 SM70; GPU UUID recorded in raw reports.
- Actual production **Qwen3.8-27B-Uncensored Q8_0 GGUF** gate/up/down weight
  codes/scales, without weight requantization. GGUF architecture is `qwen35`;
  this does not relabel old Qwen3.5 whole-model results as production 3.8.
- Hidden dimension 5120; gate/up output slice 8704; Down slice 5120 x 8704,
  representing one of two column-TP MLP shards. Inputs are **synthetic F16**
  hidden states; gate/up activations come from the actual Q8 projections.
- Pinned projection SO and all weight/input addresses are shared across arms.
  Same-source cloned separate producer is an additional compilation control.
- Twenty logical calls per graph, five replays per wave, seven alternating
  forward/reverse arm-order repeats; five warmup replays per graph. Medians and
  all raw waves are retained. Kernel counts are 2->1, 3->2 and 5->4 respectively.
- Separate SwiGLU baseline is out-of-place to avoid mutating gate between
  replays; native in-place producer performance has not been tested here.
- CPU, wall and GPU event time are separate. For T8 producer+Down, median wall
  time is 114.239->112.151 us and process CPU time 114.411->112.637 us per call.
  The benchmark consumes about one CPU core, including synchronization/spin
  waits; this is not an isolated host-launch-overhead measurement. These are
  graph-node reductions, not a Python launch per removed node.
- Peak Torch reserve is below 1 GiB; initial free VRAM must exceed 2 GiB. Idle
  checks precede cases/repeats. No production stop/restart is performed.

## Separate hardware-counter experiment

NCU profiling is **not** used for the unprofiled timing table. Counters below
are T8/K8704 producer-only, separate eager launches, one captured invocation
per kernel with metric replay. Cache and clock control are both `none`.

| Counter | Separate SwiGLU | Separate quantization | Fused-256 |
| --- | ---: | ---: | ---: |
| Achieved SM occupancy | 35.37% | 39.70% | 40.10% |
| Issue-active | 23.01% | 34.82% | 33.87% |
| DRAM throughput / sustained peak | 12.42% | 4.83% | 8.92% |
| Global load request bytes | 278528 | 139264 | 278528 |
| Global store request bytes | 139264 | 278528 | 278528 |

Combined global load/store requests fall by 278528 bytes (272 KiB), consistent
with eliminating a F16 write and read. These request counters are **not off-chip
HBM byte savings**. Measured DRAM writes are zero in these short cached profiles;
DRAM reads and L2 sectors are retained in the CSVs. This is not sustained
full-model HBM bandwidth. Achieved occupancy is not theoretical occupancy, GPU
busy, or an overall speed metric.

ALU/XU pipe activity and long/short-scoreboard/MIO stalls are retained in the
raw counter files. These partial instruction-pipe diagnostics are not total
SM compute/FLOP throughput; a total compute-throughput metric was not collected
in this campaign. Static compilation uses 23 registers/thread and no spills
for the fast fused variants; those are not achieved-occupancy measurements.

Screen telemetry spans all arms, warmups and gaps: GPU busy 0-100% (median97%),
power 37.54-251.57 W, SM clocks 1230-1380 MHz, memory clock 877 MHz, temperature
34-54 C and total device memory including production 27155-27399 MiB. These
samples are not per-arm bandwidth measurements; clocks were not locked/changed.

## Production and reproducibility

The production container remains running/healthy, with unchanged container ID,
start time, restart count, canonical config fingerprint and profiling policy
in before/after audits. A subsequent local port8001 health request returned
HTTP200. No default/native source, production configuration or old evidence
capsule was changed by this experiment. No commit/push was made.

- `plan.json`, `plan-v2.json`: pre-measurement scopes/gates and address-control
  refinement.
- `build-command.json`, `attempt-v1/build.log`: remote NVCC build and resources.
- `attempt-v1/tests.log`: correctness results.
- `attempt-v{1,2}/`: raw timing reports, telemetry, source hashes, counter CSVs,
  commands, exit codes and production audits. V2 reuses the identical CUDA SO.
- `source-v2/`: relevant prototype/test/control source snapshot.
- `recomputed.json`: offline recomputation of medians, CPU/wall, counters/gates.
- `completion.json`, `evidence-sha256.json`: stage and evidence integrity.

From the repository root, the **offline** auditor is:

```sh
python3 benchmarks/v100-swiglu-quant/summarize.py
```

CUDA builds and GPU tests must run on the authorized remote V100, never on the
Mac. The build command and worker scripts contain the exact remote invocation;
use a new output directory for a rerun rather than replacing frozen evidence.
The pinned control SO is an existing row-pair-opt-in native build, not the
production llama.cpp binary. Next work is opt-in native integration and full
model correctness/performance testing; neither is claimed by this capsule.
