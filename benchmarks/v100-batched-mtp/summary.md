# Native batched MTP1 verification on two V100s

2026-10-20. **Experimental and opt-in; full external speed goal remains active.**
This implements a native target-verification cohort with request-private target
and draft state. It is not batched draft execution, continuous batching, paging,
PP, or a whole-model CUDA graph. Existing ordinary batching and scalar MTP2
remain stronger at some request counts and are retained.

## Fixed work and correctness

Same official source revision `fc05daec18b0a78c049392ed2e771dde82bdf654`, original-
derived native Q8 target/draft, resident FP16 activations/KV, host TP on physical
GPUs 1,0. Target projections keep two A8 residual passes/F32 FMA; the draft keeps
its prior FP16 projection profile. No precision or projection dispatch change.
The pinned external GGUF's Q8_0/Q8_1/scales/small parameters are **not identical**
to native group32 Q8; prior encoding/quality audits remain applicable.

Speed case: 23 formatted prompt IDs, 32 fixed greedy outputs per request, ignore
EOS, eight prepared slots of capacity 256, no cache reuse, one warmup per cell.
MTP uses projection row budget 32; the stronger ordinary batch uses 256, unchanged
from its qualified configuration. Loading/allocation/warmup/profiling are outside
unprofiled waves. Report HTTP cohort wall including handler reset/admission,
queueing/replies and client archival/validation/future collection. Native call
TTFT/generation exclude those. Do not sum concurrent request durations.

The new API packs each active request's pending+proposal target inputs and
projects **all** verifier rows. Per-request convolution/GDN prefix-one traces
reuse the existing checkpoint backup; rejected/early-EOS proposals restore the
right prefix with no target replay. Accepted draft catch-up uses verified target
hidden. Cursors remain prompt+emitted-1; the final output remains unconsumed.
One permanent target cache exists per slot, borrowed by its MTP child.

Four trained cases (story/capital/Python/arithmetic), mixed-eight ragged/EOS cases,
reversed slot ordering, exact target/draft state and selective reset pass. The
existing independent target/98-teacher-logit and draft-architecture references
are pinned/reused; no new original-BF16/corpus-quality claim is made. No Python
model computation is used in the native generation loop. Single/TP and original
FP16/residual profiles, forced accept/reject/mixed paths, failures/reset/lifetime
and overlapping shared-model requests are covered by the fixtures.

Final V100 suite: **1,992 passed /448 skipped**; new selected tests: **86 passed**
(77 GPU +9 CPU). Local: **785 passed /1,655 skipped**. The first SDK test snapshot
omitted two launcher scripts: two setup failures/1,990 passes/448 skips and the
aborted dependent preflight are retained. The repaired complete snapshot reruns
all tests before timing; failed logs are not relabeled as passing.

## Initial native-cohort experiment

The predeclared trigger was C1 > best scalar MTP2 by 1%, or C2/C8 > ordinary batch
by 2%. C2 passes, so both the ordinary control and candidate receive three
unprofiled repeats at 1/2/4/8; the stronger scalar MTP2 gets a fresh C1 control.
All outputs match the pinned expected IDs; trained acceptance is not inferred
from output text alone.

| Requests | Ordinary batch256 HTTP TPS | Batched MTP1, initial SDK | Stronger scalar MTP2 |
|---:|---:|---:|---:|
| 1 | 22.924 | 28.920 | 33.814 |
| 2 | 36.295 | 42.614 | — |
| 4 | 53.260 | 52.982 | — |
| 8 | 65.430 | 58.556 | — |

This is a useful C2 gain, **not a universal speedup**. Single native post-first
MTP1 is 39.62 TPS; the stronger scalar MTP2 remains 42.77 TPS. In the identical
story case MTP1 accepts 14/16 proposals (87.5%) and replays zero target inputs.
A shared verifier round time is repeated per participating slot; never sum it.

## SDK-only optimization and final comparisons

The first SDK built verbose public info during capacity checks, then validated
metadata again for every per-slot stats entry. The fix uses bounded native rows
for generation preflight and a private stats parser under the validated owner
lock. **Hot public generate/info/stats still validate metadata once; native
source guards are unchanged.** Changed-metadata and nonfresh requests must fail
before generation. The same native SO and all nine native dependency hashes are
used for old/new arms; `sdk-only-diff-proof.json` records executable differences.

Fresh C1/C8 screening passes the separate SDK trigger (C1 HTTP>1% or C8>2%). The
following matrix has **three unprofiled repeats per cell**, each after warmup:

| Requests | Old SDK HTTP TPS | New SDK HTTP TPS | HTTP gain | Server CPU core-s old → new |
|---:|---:|---:|---:|---:|
| 1 | 28.946 | 31.453 | +8.66% | 1.083 → 0.993 |
| 2 | 42.648 | 45.402 | +6.46% | 1.477 → 1.383 |
| 4 | 53.025 | 55.215 | +4.13% | 2.387 → 2.293 |
| 8 | 58.667 | 59.873 | +2.06% | 4.363 → 4.270 |

This improves API/CPU efficiency, **not native decoding math**. C1 post-first
native TPS is effectively unchanged, **39.580 → 39.566**. It does not replace the
stronger scalar MTP2 control. All individual repeats, output IDs and counter
values are retained in `metrics.json` and the raw per-wave reports.

| New SDK cohort | HTTP wall s | Mean client completion s | Mean native TTFT s | Native TTFT min–max s |
|---:|---:|---:|---:|---:|
| 1 | 1.0174 | 1.0165 | 0.1853 | 0.1852–0.1853 |
| 2 | 1.4096 | 1.4081 | 0.2772 | 0.1846–0.3697 |
| 4 | 2.3182 | 2.3154 | 0.4620 | 0.1846–0.7397 |
| 8 | 4.2758 | 4.2705 | 0.8331 | 0.1850–1.4812 |

Native TTFT is an internal same-call timestamp, **not streaming client TTFT**.
Serial warmup staggers first outputs; averaging those times does not describe
when all requests have their first output. For fixed 32-output cohorts, the
remaining `31 × requests` outputs / `(native end − all-prefills-complete)` rate
is **39.57/62.70/81.39/90.21 TPS**. It is a separately named native steady-cohort
scope, not HTTP throughput, per-request speed, or summed overlapping timers.

### Stronger native and pinned external controls

The original-source external baseline and best tested external MTP2 C1 control
are explicitly reused by report/file hash. Prior failed parallel external MTP
configurations remain failures, not zero-TPS controls. This is not a claim of an
exhaustive search over all possible competitor configurations.

| Requests | Best measured native strategy | Native HTTP TPS | Reused external HTTP TPS | Native / external |
|---:|---|---:|---:|---:|
| 1 | Scalar MTP2 prefix | 33.814 | 30.173, MTP2 | 1.121× |
| 2 | Batched MTP1, new SDK | 45.402 | 37.907, non-spec | 1.198× |
| 4 | Batched MTP1, new SDK | 55.215 | 50.254, non-spec | 1.099× |
| 8 | Ordinary batch256 | 65.430 | 67.287, non-spec | 0.972× |

Against fresh ordinary batching, new-SDK MTP1 is +25.09% at C2 and +3.67% at C4,
but **−8.49% at C8**. At C1 it still loses to scalar MTP2. This table describes
individually tested strategies; an automatic shared-cache workload router has
**not** been implemented or qualified. Do not auto-enable MTP1 for every cohort.

The separate single-request native post-first milestone is freshly **42.77 TPS**
(previous qualified 42.83), against external reported **49.80 TPS**, or 0.859×.
The >=5% target requires roughly 52.29 TPS single decode and 70.65 TPS C8 HTTP.
Despite C2/C4 and C1 HTTP gains, **the full speed objective is not achieved**.
External/native encoding and timing differences remain explicitly disclosed.

## GPU diagnostics, memory and retained limits

All unprofiled arms sample NVML every 100 ms: device busy, memory-controller busy,
VRAM, power, temperature and clocks, filtered to timed waves. Full per-cell
min/mean/max is in `metrics.json`; busy is not achieved SM occupancy/bandwidth.
For the initial C8 matched ordinary/new-native arms:

| Path | Physical GPU1 / GPU0 used MiB | Mean GPU busy % | Mean power W | Mean temperature °C |
|---|---:|---:|---:|---:|
| Ordinary256 | 21,957 /17,297 | 80.0 /55.9 | 155.4 /125.8 | 46.6 /43.3 |
| Batched MTP1 | 22,459 /17,227 | 78.7 /53.4 | 165.8 /122.2 | 47.7 /43.1 |

Both arms hold 1380 MHz SM/877 MHz memory clocks in those timed samples. Used
VRAM includes the preexisting unrelated allocations; different row budgets also
change workspace size, so their net difference is **not isolated MTP overhead**.
Native accounting separately exposes shared draft model 455,514,240 bytes,
additional eight-slot MTP buffers 218,859,264 bytes, existing per-slot target
backups 1,231,552,512 bytes total and shared greedy scratch 31,744 bytes. Backups
are 146.8125 MiB/slot, reused for the prefix-one trace. They are not absent, extra
prefix-two copies, or duplicate permanent target sessions. Old/new SDK uses the
same allocations. Memory observations do not imply KV quantization.

Separate full-32-output Nsight runs use one native NVTX range per cohort and
`--cuda-event-trace=false`. The attribution's `requests=1` counts that range;
metadata records one or eight logical requests. Profiles are not timed TPS runs:

| Full call profile | Kernel launches | Summed kernel time ms | Process activity gaps ms | Two-device kernel overlap ms |
|---|---:|---:|---:|---:|
| Initial SDK MTP1 C1 | 31,904 | 1,053.21 | 194.40 | 336.02 |
| Initial SDK MTP1 C8 | 69,592 | 4,693.88 | 368.32 | 1,436.15 |
| Stronger ordinary256 C8 | 82,688 | 4,197.58 | 423.20 | 1,275.01 |

New-SDK C1/C8 retain the same launch counts and essentially the same summed kernel
time (1,053.06/4,693.76 ms). C1's profiled API host span drops 1.0800→1.0286 s
while its native call is ~1.022 s in both profiles; this agrees with host validation
removal, not a faster projection kernel. Instrumented profile variation is not
substituted for the three-repeat unprofiled matrix.

Projection/quantization-family kernels comprise ~86% of C1 and 83% of C8 summed
kernel time. C8 MTP1 launches fewer kernels but spends *more* summed kernel time
than ordinary batching. Serial prefill is only part of its loss: about 1.48 s vs
ordinary 1.25 s; its remaining 248-output interval is also slightly longer. Draft
batching/shared heads and multi-token weight-tile reuse remain hypotheses to test,
not already implemented improvements or proven cache/bandwidth diagnoses.

Kernel-time sums/overlap/process timeline gaps are not cohort latency, device-wide
idle, achieved occupancy or DRAM throughput. Static compiled-resource queries,
raw `cuobjdump` resources and exact SO/trace hashes are retained separately.
Achieved occupancy/compute/bandwidth/cache/stall counters remain unavailable under
unchanged `RmProfilingAdminOnly: 1`; no driver privilege or system policy changes.

## Reproduction and disposition

- Native library SHA256: `f07bf75a975b963911aff5fefd5614dbf0b8b5b74e499f5525b6ad5d5317369a`.
- ABI16, original public structure sizes unchanged; new exports are optional.
- `v100/model-v2/qualification` owns trained gates; SDK reuses those exact native
  sources/library and reruns the complete fixture suite.
- Original snapshots and differing executed files, commands, complete worker logs,
  terminal audit, counter policy and retained remote trace hashes are preserved.
- `reporting-corrections.json` retains a local aggregation mistake: zero-replay
  applies to cohort/prefix modes, not the intentionally retained scalar MTP1
  replay control. Raw measurements/model code did not change; no GPU rerun.
- `summarize.py` recomputes metrics; source/evidence manifests freeze this capsule
  and verify all 30 prior capsules without changing their contents.
- All nine owned workers are terminal; failed setup/preflight remain classified.
  Port 18953 is closed; prior GPU footprint 3,651/3,189 MiB is restored. No unrelated
  service was stopped. No commit/push or local Rust/CUDA compilation.

**Decision:** retain the explicit target-cohort API and tested SDK optimization;
do not switch the universal default. Continue with measured projection tile/cache
reuse and draft/host scheduling studies while preserving stronger C1/C8 controls.
