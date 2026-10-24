# Adaptive MTP implemented and tested; original production service restored

2026-10-21. This is an opt-in **fresh-cohort measured-cost selector**, not DSpark,
a confidence head, per-token depth adaptation, a new CUDA kernel or a production
OpenAI server. No static concurrency-to-depth table is built into the policy.

## Results

Official Qwen3.5-27B source revision
`fc05daec18b0a78c049392ed2e771dde82bdf654`, original-source Q8/group32 target and
draft artifacts, existing residual-A8 FP16 arithmetic, FP16 KV, two V100-PCIE-32GB
GPUs, logical order physical 1/0, explicit host TP. The target projection profile
is `q8-residual-column-mlp-head-gdn`; the draft uses `q8-residual`.

All variants share the same target/draft weight owner, **row256** configuration,
eight 256-context slots and the same 23 formatted prompt / 32 fixed greedy
output tokens with EOS stopping disabled. Loading and prepared-bank allocation
are excluded. Rates below include cohort policy/reset/native execution/metadata
and return, not merely post-first decode. Each cell has three timed repeats.

| Requests | Automatic choice, all three trials | Auto aggregate TPS | Ordinary fixed TPS | Best fixed TPS | Auto vs best fixed |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Scalar MTP2 / prefix | 42.5962 | 28.9958 | 42.7214 | -0.293% |
| 2 | Packed MTP1 | 59.5222 | 47.4182 | 59.5305 | -0.014% |
| 4 | Packed MTP1 | 72.4296 | 68.8011 | 72.3983 | +0.043% |
| 8 | Ordinary packed, no MTP | 84.1281 | 84.1412 | 84.1412 | -0.016% |

The policy tracks the best measured candidate within ~0.3% here. The +0.043%
cell is noise-scale, **not** a demonstrated speedup over the best fixed strategy.
Against ordinary decoding, improvements are +46.905%, +25.526%, +5.274% and
-0.016%. At eight requests it avoids the slower fixed MTP1 path (77.6727 TPS).
It does not raise the kernel-speed ceiling.

Fixed controls use the same adaptive owner with explicit depth override and
learning disabled, so this isolates strategy selection rather than measuring
total new-wrapper overhead against the bare original API. Automatic trials
learn from their successful costs. Order alternates between fixed-to-auto and
auto-to-fixed. These are native cohort rates, **not** HTTP rates or the historical
52.95 post-first decode figure with a different row configuration/timing scope.
No new llama.cpp comparison or closure of the older full-comparison goal is claimed.

## Exploration is not free

One correctness/warmup pass runs for every candidate before comparison. Actual
cold-policy exploration and two additional labelled calibration observations per
candidate are retained separately from steady timing; they are not hidden in the
timed averages:

| Requests | Cold-policy probe cohorts | Cold probe wall seconds | Extra calibration wall seconds |
| ---: | ---: | ---: | ---: |
| 1 | 3 | 2.6222 | 5.2510 |
| 2 | 2 | 2.4252 | 4.8515 |
| 4 | 2 | 3.6293 | 7.2559 |
| 8 | 2 | 6.3288 | 12.6751 |

Runtime defaults use alpha 0.3, a 3% cost-improvement margin over ordinary
decoding, bounded workload buckets and periodic stale-strategy rechecks. Probes
can worsen latency. Unit tests demonstrate measured degradation disabling MTP
and a later successful recheck enabling it again; this workload is not evidence
for adaptation to every real traffic distribution. Acceptance counters are
diagnostic observations, not trained confidence estimates.

## Correctness and failure behavior

- **47 remote tests pass**, including generated hybrid GPU fixtures at 1/2/4/8
  requests, ragged budgets, reordered slots, EOS, repeated strategy switching,
  exact target GDN/KV state bytes, and explicit failure/reset behavior.
- Both original FP16 and the existing semantic-head-GDN residual-Q8 target
  profiles are covered; the aligned fixture uses the residual-Q8 draft too.
- A test forbids Python Torch numerical operations during generation. Native
  code still performs all drafting, target verification, recovery and sampling.
- The actual loopback HTTP adapter runs on the native fixture: cold single
  requests select 0/1/2, eight-request cohorts select their 0/1 probes, IDs match,
  and cohort metadata is exposed correctly. This is not an OpenAI API test.
- **75 full-model cohorts / 8,384 emitted IDs** exactly match the pinned greedy
  target trajectory. Every fixed candidate passes before timed comparisons.
  This is one short trajectory, not a new original-BF16 corpus-quality campaign.
- Failed native calls are not training samples and do not retry under another
  strategy. The owner is poisoned until explicit reset. No in-flight GDN/KV
  state is migrated between banks.

Local final checks: 36 pass, 11 GPU tests explicitly skip. An unrelated globally
autoloaded NoneBot pytest plugin initially broke setup; final runs disable plugin
autoload. Initial fixture tests also exposed test-construction mistakes: odd
GDN geometry was ineligible for the head profile, a debug failure injected before
automatic reset was correctly cleared, and a periodic-recheck test expected the
wrong age. Those failed logs are retained. The final gates were not weakened;
aligned geometry, injection after reset and correct probe-age expectations pass.
The original local plugin-setup log was overwritten by the final local run; that
failure is recorded in the conversation rather than misrepresented as retained raw evidence.

## CPU and GPU data

| Requests | Auto cohort wall seconds | Process CPU core-seconds | Physical 0 / 1 mean sampled GPU busy | Samples per GPU |
| ---: | ---: | ---: | --- | ---: |
| 1 | 0.75124 | 0.75427 | 67.25% / 78.50% | 4 |
| 2 | 1.07523 | 1.08007 | 66.17% / 75.50% | 6 |
| 4 | 1.76723 | 1.77562 | 74.22% / 82.56% | 9 |
| 8 | 3.04298 | 3.05616 | 74.28% / 85.00% | 18 |

These short sampled windows have low sample counts, not precision occupancy
measurements. All banks are preallocated: observed device-used VRAM is
16,333/26,949 MiB (physical 0/1), including unrelated allocation. Separate bank
request/backup buffers add memory; shared weight bytes must not be counted again
for every bank record. Detailed bank memory and raw NVML power/clock/temperature
samples are captured. Auto C8 mean sampled power is 163.4/173.2 W, maximum
temperature 53/56 C; sampled SM/memory clocks are 1380/877 MHz throughout timed
auto windows. No clocks or driver policy were changed.

`RmProfilingAdminOnly: 1` remains in effect. Achieved occupancy, DRAM bandwidth,
cache/stall counters were not obtained or substituted with NVML busy. No kernel
or transport optimization is claimed, and no new kernel/timeline profile is
presented as routing throughput evidence. CPU utilization remains about one core;
the change does not implement CPU/GPU asynchronous scheduling.

## Production restoration and scope

The user explicitly authorized the temporary production pause. A finite transient
user unit waits for an idle production slot, runs only the owned benchmark, and
restores the original container in `finally`, backed by an independent
`ExecStopPost` restore action and bounded unit/runtime timeouts.

Container `fc9e37261f26c3127d5b65080b392ddeda97ba293dfd5573519354222328d70d`
is restored and healthy. The before/after command/environment/mount fingerprint
is identical. A real chat call through the original Mac port 8001 returns the
expected `RESTORE_OK`. **XHarness still uses the original llama.cpp production
route; adaptive native MTP was not silently deployed there.** No other service,
global route, host-key policy or profiling privilege was changed.

No Rust/CUDA compilation or Git push occurred. The qualified native backend SHA
remains `33a4becb5f9c8440492e9cc6352ea87fccab2692a01adc48e6cc756fa93bc7b8`.
All 1,459 files in the preceding frozen evidence manifest still verify unchanged.

## Reproduction and records

- `docs/adaptive-mtp.md`: API, manual controls and opt-in benchmark HTTP flag.
- `remote/full-model/rows.json`: every warmup/probe/calibration/timed cohort.
- `remote/full-model/summary.json`, `configuration.json`, `gpu-samples.json`.
- `recompute.py` → `recomputed.json`: recomputed controls and telemetry windows.
- `remote/fixture-v3-tests.log`: final 47-test remote result.
- `remote/production-before.json`, `production-restored.json`,
  `production-api-restored.json`: restoration proof.
- `source-pins.json`: matching tested local/remote Python source hashes.
- `final-remote-audit.log`: artifact/library hashes, counter policy and process audit.

This implementation is reusable at the policy level, Qwen-specific at the native
adapter level. It does not establish long-context, heterogeneous production
traffic, stochastic-sampling or universally optimal routing behavior.
