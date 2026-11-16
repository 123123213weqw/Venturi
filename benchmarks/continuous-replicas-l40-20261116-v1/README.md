# Two idle L40s: continuous replica qualification

Source: `ed5400a898f185c4e7329050179b7bd907116b08`, 2026-11-16/17.
This is a **functional runtime qualification and matched routing experiment**,
not a production promotion or a claim of a faster model kernel.

## What actually ran

- Two independent Venturi-native **Qwen3.5-4B text-only Q8/G32** models, FP16
  activations, ordinary greedy generation; no external inference engine.
- Physical L40 indices 2/3 were idle before admission. Task-private UUID locks,
  foreign-compute checks and independent timeouts bounded the jobs. No service,
  other user's task, production model/route, driver or monitoring policy changed.
- 16 private slots/GPU, 32 input rows/step, prefill chunks <=16, capacity 71
  tokens/slot. 32 concurrent requests is **up to 16 active requests per GPU**,
  not a per-GPU B32 claim. Actual prompts have 17–46 tokens, output budget 24.
- Local floating source -> separate 4,473,294,824-byte artifact, SHA256
  `494e58042d7d428e8396da1fd6dfbb1228269028683c8bd915143078d0b09a08`.
  Source/config/index hashes are recorded; the local source is **not independently
  authenticated as an official quality baseline**. No BF16/Q8 quality claim.

The intended Qwen3.8-27B Q8 export completed on another server, but slow verified
SSH transport and unavailable direct routes prevented its L40 qualification in
this task. Its partial transfer was preserved. The fallback to local 4B was
explicitly reported **before** execution. GLM-5.3-Flash's existing experimental
PP8 path could not be tested with only three idle GPUs. Neither model receives
a throughput/support claim from these results.

## Correctness

CPU: **1916 passed / 2910 skipped**, architecture check 133 modules / 0 violations.
GPU: **17 targeted checks passed** on final source, including two-device
FP16/BF16 fixtures at 4/8/16/32 requests, split native ABI and worker/state checks.
GitHub CPU CI passed on Python 3.10 and 3.12 for the source commit.

For **all 32 distinct real prompts**, independent native B1 sessions on both
devices agreed exactly. All pooled outputs in both routing policies, warmups and
three timing repeats also agreed token-for-token with those references. This
qualifies scheduling/state consistency, not an independent model-math oracle.
The fixed-work replay emits short thinking prefixes, not complete answers.
Separate no-thinking smoke prompts followed the source chat template, respected
EOS, and returned `42`, `早上好`, and `Paris`. This is smoke, not a quality suite.

## Three-repeat matched measurements

Aggregate output TPS = actual output tokens / **one common fenced wall interval**,
including prefill/admission. This is NOT decode-only TPS or HTTP/streaming TTFT.
Both policies use the same resident models, input/output budgets, capacities and
precision, alternating AB/BA/AB order with fresh scheduler warmups. No profiler
in timing runs. Native model load/batch preparation took 7.65 s after framework
imports, checkpoint hashing and source admission, with warm filesystem cache.
This is neither whole-process startup nor disk-cold startup. Exact value and
scope are in `results.json`.

| Concurrency | Default round-robin output TPS | Opt-in least-work output TPS | RR TTFT p50 (s) | RR process CPU (s) |
|---:|---:|---:|---:|---:|
| 4 | 114.87 | 114.69 | 0.324 | 1.535 |
| 8 | 117.49 | 118.62 | 0.634 | 3.029 |
| 16 | 116.73 | 116.47 | 1.348 | 6.148 |
| 32 | 116.38 | 117.60 | 2.847 | 12.336 |

Entries are medians across three repetitions. Least-work does **not** establish
a general speed win; round-robin remains the simpler default. The earlier full
campaign is retained, not replaced or pooled to select favorable measurements.
Shared-host CPU/IO load was not isolated; an existing CPU-only JuiceFS download
continued. CPU seconds sum process execution across threads, not elapsed time.

Selected-GPU samples during timed RR intervals had median 99–100% GPU busy and
6,125 MiB VRAM **per card**. Sampling is coarse (~1 s), with only 6/6/14/32
card-sample points at C4/8/16/32; it is not a claim of sustained SM occupancy.
Power, clocks, temperature, per-run latency and process RSS are in `results.json`;
peak process RSS was ~9.94 GiB. No host-wide memory total is attributed to us.

## Independent profiling: reusable bottleneck, not idle-GPU guessing

Nsight Systems 2024.2.3 captured **56,424 kernels on both devices** in a separate
C16 round-robin replay. Profiled outputs matched the references.

- **90.4% of summed observed kernel time** was `q8_head`. Despite the historical
  name, the current native `project()` uses it for **general Q8 linear projections**
  in the default profile, not just the output head. It directly dequantizes and
  accumulates FP32 FMAs in SIMT blocks; this is not a Tensor Core GEMM claim.
- GDN recurrence was 5.5%; causal attention 2.4%. CPU CUDA API time was dominated
  by stream synchronization waiting for GPU work, not proof of CPU starvation.
- Kernel interval coverage within each device's own first-to-last-kernel span
  was 98.62% / 98.86%. On the common span, GPU0 had a ~0.399 s tail after it
  finished first. These timeline intervals are **not achieved SM occupancy**.
- Achieved occupancy, compute throughput, DRAM bandwidth and cache/stall counters
  remain **unmeasured** (`RmProfilingAdminOnly=1`); no privilege, DCGM or driver
  change was made.

The next independent performance PR should therefore target the **reusable Q8
projection** and batching/weight reuse, with arithmetic/quality gates, rather
than interpreting 100% busy as efficient matrix utilization or adding router
complexity. This PR intentionally makes no CUDA kernel change.

## Evidence and reproduction

- `results.json`: environment, identities, source/library hashes, three-repeat
  summaries, GPU samples, separate profiler attribution, smoke and retained failures.
- `qualified-replay.json.gz`: complete final unprofiled run, IDs and completions.
- `initial-replay.json.gz`: earlier full unprofiled campaign, preserved separately.
- `selected-gpu-samples.jsonl.gz`: only selected cards; excludes shared-node jobs.
- `profile-stats_*.csv`: separate native kernel/API/NVTX attribution.

Run `tools/benchmark_continuous_replicas.py --help` for explicit local artifacts
and library paths. The tool does not reserve GPUs or stop services; admission
and monitoring belong to the launcher. Full raw logs, the private job guard and
the 3.3 MiB Nsight report are retained in the local task artifact directory, not
packaged as runtime code. All task GPU owners exited; cards were released.
