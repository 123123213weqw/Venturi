# Native Qwen-27B batch-8 decode counters

## Result

Both GPU counter arms completed successfully in attempt 2. Twenty full-model
split cohorts matched all pinned greedy IDs and cursor positions. Production
was restored healthy with the exact original configuration; the existing Mac
API tunnel returned `RESTORE_OK` in the post-restore chat probe.

The six unprofiled decode controls averaged **118.622 aggregate tokens/s**
(14.828 tokens/s per request), 2.09067 seconds per 248-token cohort, and 2.09281
process CPU seconds (approximately one CPU core).

| Physical GPU | HBM read / write per 31 steps, decimal GB | Profiled read+write GB/s | Mean DRAM % sustained peak | Mean achieved active-warp occupancy |
| --- | ---: | ---: | ---: | ---: |
| 0 | 444.615 / 22.336 | 101.623 | 11.16% | 47.693% |
| 1 | 445.664 / 23.789 | 90.948 | 11.48% | 45.830% |

The GPU range duration under instrumentation averaged 4.59493 seconds for GPU 0
and 5.16176 seconds for GPU 1, **2.20x and 2.47x** the normal control wall time.
The bandwidth/percentage columns describe instrumented execution. They are not
direct measurements of normal, uninstrumented bandwidth utilization and do
**not** establish that normal execution has roughly ninefold bandwidth headroom.
Traffic totals cover repeated decoding, not simultaneous VRAM capacity.

Normal sampled GPU busy averaged 73.67% / 85.08%, power 147.13 / 158.47 W,
and peak observed used VRAM 14,509 / 20,917 MiB (physical GPUs 0 / 1).
SM/memory clocks were 1380/877 MHz in those samples; temperature was 41–44 C.
These are coarse NVML samples, not occupancy or achieved compute-throughput
counters. Full-model SM-throughput, cache-hit and stall counters were not
collected in the final five-metric campaign.

Raw reports, failures, timing controls, recovery details and restoration checks
are retained below; `recomputed.json` is the machine-readable result.

## Workload and scope

- Host: `WZU_Server`; two Tesla V100-PCIE-32GB GPUs.
- Official Qwen3.5-27B export: original Q8/group32 weights, FP16 activations/KV,
  qualified `q8-residual-column-mlp-head-gdn` native host-transport TP backend.
  No MTP, weight conversion, new kernels, or backend rebuild in this experiment.
- Eight independent fixed-cohort requests, identical pinned 23-token prompt,
  greedy 32-token output, EOS stopping disabled. Projection rows and slot
  capacities are 256. This is short-context decoding, not a 256K-context result.
- Prompt prefill and the first generated token are **outside** the counter
  range. The pending first token is passed to the existing continuation API,
  which performs the remaining **31 decode steps / 248 emitted tokens**.
- The range includes one continuation API re-entry and normal native host-TP
  transfers/CPU submission. It is not isolated kernel time or HTTP throughput.
  The continuation's first group is counted as `prefill` by the existing API,
  but all eight inputs are single pending tokens. Inspection of the qualified
  `qwen_batch.cuh` confirms `decode` only changes the diagnostic counters, not
  the group computation. No prompt tokens are reprocessed in the range.

## Correctness and measurements

`profile_decode.py` validates uninterrupted versus split generation and pinned
output IDs in standalone unprofiled controls. Every profiled cohort also checks
all greedy IDs, cursor positions and group/head/row counters. An aligned tiny
hybrid fixture additionally checks bitwise canonical GDN/KV caches for all eight
slots, including under counter replay. Full-model caches are not claimed to have
been independently bitwise-snapshotted in this experiment.

The successful attempt uses three unprofiled repeats before and after counters.
Each GPU has three separately profiled decode ranges. Nsight Compute application
range replay re-executes the application twice per GPU invocation; it does not
profile or serialize every individual kernel. The selected CUDA context is
current at `cudaProfilerStart`, and both devices synchronize before range start
and before `cudaProfilerStop`.

Metrics are DRAM read/write bytes, GPU range duration, DRAM percent of sustained
peak, and achieved active-warp occupancy. Requested SM-throughput/L2-hit metrics
in the tiny exploratory probe required seven passes; the full-model campaign
uses only the five core metrics (two passes). Tiny-probe L2 hit percentages
slightly above 100 are retained raw but not interpreted as valid hit rates.

**Profiler measurements are not unprofiled performance.** `summarize.py` reports
counter-range bandwidth as total read+write bytes divided by GPU range duration,
and separately reports normal cohort decode TPS/CPU time. It retains the ratio
between profiled GPU duration and the unprofiled wall-time control. Neither
NVML busy nor active-warp occupancy is substituted for bandwidth utilization.
Per-GPU results come from separate executions, not simultaneous measurements;
their bandwidths, times or percentages must not be summed as an observed
two-GPU aggregate. Occupancy/percentage summaries are means of three ranges.
See NVIDIA's [application range replay and overhead documentation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
for the profiling mode and its limitations.

GPU busy, VRAM, power, clocks and temperature are sampled every 0.5 seconds.
Unprofiled telemetry summaries filter samples to the measured decode intervals;
model loading, prompt processing and counter replay are excluded from them.

## Bounded attempts and production restoration

The user explicitly authorized a temporary pause and exact-parameter restore.
Only `qwen38-27b`, container ID
`fc9e37261f26c3127d5b65080b392ddeda97ba293dfd5573519354222328d70d`,
is paused, after three idle slot observations. This existing llama.cpp production
container is distinct from the native Qwen3.5 benchmark checkpoint.

- `attempt-v1/`: retained failed first attempt. Its first counter pass completed,
  but whole-model replay with redundant non-measured warmups exceeded the
  180-second per-arm limit. No complete full-model counter result is claimed
  from that attempt. Exact original production was restored healthy.
- `attempt-v2/`: same full 31-step measured workload. Only the redundant
  non-measured warmups inside application replay are omitted; correctness is
  still checked against pinned IDs and the separate before/after controls.
  Per-arm timeout is 300 seconds and campaign budget is 900 seconds.

A root transient systemd unit owns all profiling descendants, with process-group
cleanup, a Python `finally` restore, and an independent `ExecStopPost` restore
of the exact container ID. Root is used only for one-off counter access.
`RmProfilingAdminOnly` stays 1. GPU clocks/cache policy, driver settings,
production arguments/environment/mounts and XHarness settings are not changed.
The production fingerprint sorts Docker mounts, whose returned list order was
observed to vary even without configuration changes.

`attempt-notes.json` records recovery and script hashes. `recomputed.json` is
produced from final raw reports by:

```sh
python3 summarize.py attempt-v2
```

Use the recorded raw Nsight CSV units (bytes and nanoseconds); bandwidth is
decimal GB/s. Re-running the service-management script is not a passive analysis
operation: it requires a newly agreed maintenance window and a new artifact path
and transient-unit name. The exclusive start marker prevents accidental reuse.
