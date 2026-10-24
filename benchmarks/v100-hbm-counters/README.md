# V100 administrator counter-access validation

Date: 2026-10-21. Host: `WZU_Server` (`wzu-SYS-4029GP-TRT`).

This is a **small operator compatibility probe, not a Qwen-27B bandwidth
benchmark**. No inference implementation, production settings, driver policy,
or GPU clock settings were changed. Production was not stopped for these probes;
the original `qwen38-27b` container remained running and healthy afterward.

## Confirmed

- `sudo -n true` succeeds. Nsight Compute 2025.2.1 supports `gv100` and can
  collect counters through one-off `sudo`, with `RmProfilingAdminOnly: 1`
  unchanged.
- `admin-probe-v3` collected one real tensor-fill kernel. Earlier failed probes
  are retained in `raw/`; exit zero alone is not evidence of counter collection.
- `api-range-probe-v2` and `api-range-probe-selected1` collected **application
  range** counters, each for its selected CUDA context. They bracket three
  small tensor additions on each GPU with `cudaProfilerStart/Stop`.
- With `CUDA_VISIBLE_DEVICES=1,0`, logical device 0 is physical GPU 1 (UUID
  `GPU-f65db2e8-87bb-c88d-363f-23c79f793548`); logical device 1 is physical GPU 0
  (UUID `GPU-18fbf081-3c6d-d65b-11b8-288a4ce1ab28`). Separate runs select each
  context; a single selected-context range is not a two-GPU aggregate.

Collected metrics:

```text
dram__bytes_read.sum
dram__bytes_write.sum
gpu__time_duration.sum
dram__throughput.avg.pct_of_peak_sustained_elapsed
sm__warps_active.avg.pct_of_peak_sustained_active
```

`summary.json` extracts the requested columns from the original wide CSV files.
The tiny operations, Python launch gaps, warm caches, unrelated production
allocation, and profiler replay mean **these values must not be presented as
model bandwidth, GPU peak bandwidth, or unprofiled TPS**. The two range runs are
not simultaneous and their durations or percentages must not be summed.

## Reproduction notes

On this host, root needs explicit Python dependency paths, including
`/home/wzu/.local/lib/python3.10/site-packages`; avoid forwarding the entire user
environment with `sudo -E`. The successful invocation uses `/usr/bin/env` before
`ncu`, `--target-processes all`, and a bounded 120-second timeout.

For the range probes use `--replay-mode app-range`, **without**
`--profile-from-start` or NVTX filters, and run `probe_ranges.py --selected 0
--profiler-api` (then a separate run with `--selected 1`). The selected context
is current before `cudaProfilerStart`. `--profile-from-start` is explicitly
unsupported in this replay mode; that failed attempt is retained.

Both successful modes use `--clock-control none --cache-control none --csv
--page raw`, and range mode additionally uses `--disable-extra-suffixes`.
Uncontrolled clock/cache warnings are expected and are retained rather than
silenced. Application replay executes the probe multiple times; this is not
an unprofiled timing experiment.

## Next measurement

Full-model C1/C8 counter sampling requires a separately approved short
production maintenance window. Use the qualified native Q8 checkpoint,
explicit per-GPU context selection, identical fixed input/output workloads,
correctness checks, and per-range DRAM bytes/duration. Keep prefilling and
decoding scope explicit. Run unprofiled throughput controls separately; never
use profiler wall time as TPS or host-transfer bytes as HBM traffic.

Use a bounded campaign and restore the exact original production container
and parameters on success, failure, or timeout. No full-model HBM result is
claimed by this directory yet.
