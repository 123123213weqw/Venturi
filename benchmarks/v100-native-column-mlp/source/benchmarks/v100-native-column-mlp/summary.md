# Native MLP column/row TP: full-model gain, single-decode goal still open

2026-10-20. **Qualified opt-in profile, not a default precision change.** The
MLP block prerequisite is now integrated into actual native scalar and packed
target generation. On the same two V100-PCIE-32GB GPUs, single post-first decode
improves **42.814 -> 47.629 TPS**. Best measured complete-cohort HTTP rates exceed
the stronger retained pinned external controls at all four request counts.
However, single decode still trails external **49.803 TPS**; the full >=5%
objective needs **52.293 TPS**, another **9.79%** over current native performance.
**The goal remains active and incomplete.**

## Implementation and precision

`q8-residual-column-mlp` is an additive ABI16 constructor/profile for resident
FP16 target activations/KV, two SM70 GPUs and explicit pinned-host TP transport.
Logical 0/1 maps to physical 1/0. The previous byte probe rejects actual peer
transport despite advertised capability; there is no silent fallback. Attention,
GDN, convolution and primary-owned private caches retain the original row-TP path.

Each MLP broadcasts normalized hidden once, then runs fused local gate/up and
local FP16 SwiGLU. Down weights are replaced with all-output/input-column halves;
stored group32 Q8 codes and FP16 scales are copied without requantization through
bounded 8 MiB staging. Full down weights are not also retained. Both ranks use
the original two-pass residual-A8 quantizer/FP32 scales and produce FP32 down
partials. The peer partial crosses host in **FP32**, is summed in FP32, and is
cast once to FP16 before the original separate residual add. Both producer and
consumer ownership events remain. The packed verifier uses the same native MLP
method, not a Python implementation or a reduced set of verifier rows.

Splitting K changes FP32 association: **new-profile quality checks, not byte
equivalence to old full-K arithmetic**. The original FP16 draft profile, ABI
descriptor sizes and old constructors/defaults remain unchanged. Explicit SDK
metadata reports local layouts and buffers; older libraries/missing exports and
unsupported hardware/transport fail preflight. No PP, streaming fallback, paged
KV, continuous batching, whole-model graph or automatic strategy router is added.

Official parent revision: `fc05daec18b0a78c049392ed2e771dde82bdf654`.
Target artifact: `3d635b87db004f42aefd6730e372e530`; draft:
`1a4307f5e7954a929d7ad70539875915`. Native group32 Q8 encoding, small parameters,
residual-A8 arithmetic and external GGUF Q8_0/Q8_1 **are not identical precision**.
Prior original-source encoding audits remain applicable. FP16 KV is unchanged;
there is no broad original-BF16 or corpus-quality claim.

## Correctness before timing

- Remote full original suite: **2,132 passed /687 skipped**. Initial integration
  subset: 12 passed; expanded integration subset: **22 passed**. Later CLI tests:
  11 passed. Final local suite: **803 passed /2,028 skipped**. CUDA compilation
  and GPU tests run only on WZU_Server; no local Rust/CUDA build.
- Independent represented-FP64 full MLP checks inspect actual FP32 partials and
  FP16 outputs at T1/2/3/8 and both layer kinds. Tests cover tails, large/canceling
  partials, mixed EOS, selective reset, concurrent private sessions, failed/closed
  owners and changed sources. Native generation passes with Torch math disabled.
- A fresh original FP16 teacher matches all **98 full-logit hashes and final
  private-state signatures** across four story/capital/Python/arithmetic
  trajectories. The new
  profile passes the same 98 teacher rows: maximum relative L2 **0.00323312**,
  minimum centered cosine **0.999993056**, maximum reference KL **0.0000936612**,
  and all top-1 IDs equal. Fixed limits remain L2 <=0.02, cosine >=0.999, KL <=0.002
  and exact top-1. Maximum absolute logit difference is 0.06640625.
- Diagnostic/generate states, mixed-eight scalar/batch outputs/private caches,
  selective reset, MTP1 cohorts and MTP2 prefix/first-or-second rejection and
  corrected draft states pass. Four trained MTP2 trajectories match ordinary
  target outputs/committed states and corrected MTP1 draft states. Three fresh
  independent original-source draft recurrence checks pass; these use the
  declared seed/recurrent-hidden sequence, not a full BF16 model trajectory.
- New-profile projection budgets **32 and 256** produce exact corresponding
  98-logit/state signatures plus mixed-eight/reset results. That gate reuses
  hashed teacher tensors from the fresh qualification rather than rerunning the
  unchanged teacher. Large teacher `.pt` files remain remote by SHA256.

## Matched unprofiled engine measurements

Fixed case: 23 formatted prompt IDs, 32 greedy outputs/request, ignore EOS,
capacity 256/slot, eight prepared slots, no prefix reuse. One warmup precedes each
cell; loading/allocation/profiling are excluded. Scalar MTP2 and MTP1 cohorts use
projection budget 32; stronger ordinary batching keeps its qualified budget 256.

HTTP throughput is `32 * requests / mean complete cohort wall`, including
handler reset/admission/queueing, reply, client archival/validation and future
collection. **Never sum concurrent request durations.** Native TTFT/generation
are internal call scopes, not streaming client TTFT. Single post-first is
`31 / mean(native generation - native TTFT)`; external is
`31 / mean(predicted_ms/1000)`. The native and external timer boundaries are
explicitly different; HTTP is the matched complete-request scope.

All three path pairs pass C1/C8 screening and get **three unprofiled repeats per
1/2/4/8-request cell**, plus fresh external measurements. All 12 repeated native
cells improve; every C1/C8 pair improves in all three paired trials. No repeated
cell regresses >2%. Individual trials and all counter values remain in metrics
and raw wave reports, including the slower first external C1 trials.

| Requests | Scalar MTP2 HTTP old -> new | Ordinary256 old -> new | MTP1 cohort old -> new |
|---:|---:|---:|---:|
|1|33.872 -> 37.685 (+11.26%)|22.990 -> 25.715 (+11.86%)|31.505 -> 35.761 (+13.51%)|
|2|33.379 -> 37.037 (+10.96%)|36.285 -> 41.553 (+14.52%)|45.480 -> 50.638 (+11.34%)|
|4|33.924 -> 37.595 (+10.82%)|53.281 -> 60.011 (+12.63%)|55.226 -> 61.708 (+11.74%)|
|8|34.083 -> 37.475 (+9.95%)|65.378 -> 73.836 (+12.94%)|59.900 -> 65.492 (+9.34%)|

MTP2 acceptance remains **17/26**, conditional depths 11/13 and 6/11, with 13
three-input verification calls, 63 total target inputs and zero target replay or
forced proposals. MTP1 remains 14/16 per request. Different acceptance is not the
explanation for the measured gain. Scalar MTP serializes parallel requests;
it is not the recommended parallel path merely because its C1 result is best.

### Stronger external controls are retained

Fresh pinned external MTP2 C1 HTTP is 28.898 TPS and non-speculative C2/C4/C8 is
35.045/50.024/65.716 TPS. Do **not** use these weaker values alone to inflate the
win. The table uses the stronger historical three-repeat controls instead.
Declared full server commands match except loopback port; case, benchmark code
and explicit CUDA/thread/NCCL/library-path environment match. Inherited environment
is not exhaustively snapshotted; fresh vs historical HTTP differences are not
attributed to an invented cause. Compatibility proof and all raw trials are kept.

| Requests | Explicit native strategy | Native HTTP TPS | Stronger pinned llama HTTP TPS | Ratio |
|---:|---|---:|---:|---:|
|1|Scalar MTP2 prefix|37.685|30.173, MTP2|1.249x|
|2|MTP1 verification cohort|50.638|37.907, non-speculative|1.336x|
|4|MTP1 verification cohort|61.708|50.254, non-speculative|1.228x|
|8|Ordinary batch256|73.836|67.287, non-speculative|1.097x|

These are individually selected tested modes, **not an implemented automatic
router or exhaustive competitor configuration search**. Prior external parallel
MTP failures remain failures, not zero-TPS controls. Prior weaker external
depth1/3/4 controls remain preserved. Single native decode **47.629** versus fresh
external **49.743**, or stronger prior **49.803**, still misses the full goal.

## CPU, latency, memory and GPU telemetry

Best-native cells below; times are seconds and CPU is server process core-seconds
per complete cohort. Native TTFT is averaged over requests, including staggered
prefills; it does not mean all clients received a first token at that time.

| Requests / strategy | HTTP wall | CPU core-s | Client completion mean / max | Native TTFT mean |
|---|---:|---:|---:|---:|
|1 / MTP2|0.8491|0.8467|0.8483 /0.8501|0.1624|
|2 / MTP1 cohort|1.2639|1.2367|1.2623 /1.2634|0.2441|
|4 / MTP1 cohort|2.0743|2.0467|2.0713 /2.0736|0.4066|
|8 / ordinary256|3.4671|3.4667|3.4632 /3.4662|1.0833|

C1 matched MTP2 CPU falls 0.9433 ->0.8467 core-s; C8 ordinary falls 3.9167 ->3.4667.
These are real CPU-time reductions, not proof that CPU-only launch overhead was
the limiting factor. All path/count CPU/latency and fresh external measurements
are available in `metrics.json`; no CPU or internal timer is substituted for HTTP.

Timed-wave NVML samples every 100ms; values below are physical **GPU1 /GPU0**.
Used memory includes the unrelated preexisting 3189/3651 MiB allocations. Sparse,
lagging busy samples are not achieved occupancy, DRAM bandwidth or cache/stall data.

| Requests / strategy | Device-used MiB | Mean GPU busy % | Mean power W | Maximum temperature C |
|---|---:|---:|---:|---:|
|1 / MTP2|23993 /17229|77.0 /52.9|159.9 /124.5|42 /40|
|2 / MTP1 cohort|22461 /17229|78.6 /54.4|158.4 /125.2|45 /41|
|4 / MTP1 cohort|22461 /17229|82.6 /54.0|176.4 /127.5|48 /43|
|8 / ordinary256|21969 /17319|84.2 /57.7|171.2 /134.0|50 /46|

Clocks, memory-controller busy and all per-cell min/mean/max samples are retained.
Best-cell SM clocks are 1380 MHz except a sampled C4 minimum of1372; memory clock
877 MHz. Versus matched controls, C1 uses the same primary MiB and +2 peer MiB;
ordinary256 uses +12/+22 MiB. Logical weight upload remains **28,549,120,000 bytes**.
At row budget32, explicit local/remote/peer/pinned FP32 partial buffers each use
655,360 bytes; peer gate/up1,114,112 and activation557,056. There are64 replaced
down matrices. Metadata separately accounts permanent models, per-request caches,
MTP backups and prefix traces; a single CLI call is not a throughput benchmark.

## Separate GPU attribution: reduced transport, not fewer kernels

One instrumented native call per cohort, never mixed into unprofiled TPS. The
parser's `requests:1` for C8 counts one NVTX native-call range; metadata records
eight logical requests. Earlier completed C1 captures are SHA-pinned and reused
in the repaired profile worker; they are not silently repeated or double counted.

| Instrumented path | Host span ms | Kernels | Sum kernel ms | H2D bytes | D2H bytes | Two-GPU kernel overlap ms |
|---|---:|---:|---:|---:|---:|---:|
|Same-SO old row MTP2 C1|961.833|27026|968.559|535484944|535484976|295.150|
|Column MTP2 C1|862.902|28946|979.764|316660240|316660272|330.530|
|Column ordinary256 C8|3502.659|86784|4257.170|2118782336|2118782976|1469.563|
|Column MTP1 cohort C8|3894.782|72792|4777.131|2232040320|2232039680|1633.325|

C1 payload falls **40.86% each direction**, transfer calls17785 ->12985, while
kernel count increases1920 and summed kernel work rises. Process-only activity
gaps shrink160.095 ->126.645ms; those gaps are not device-wide idle or necessarily
CPU-only work. Device kernel unions increase578.837/389.721 ->584.407/395.357ms;
better overlap and less host serialization matter. Summed durations are not
critical-path percentages. Draft wall is effectively unchanged0.08020 ->0.08019s
(including warmup/catch-up); instrumented verifier wall0.64594 ->0.57300s.

Compiled FP32-partial kernels at T1/4/8 use48/44/50 registers,128/256/512 threads,
0/9216/18432 shared bytes and zero reported local-memory allocation. Driver
residency ceilings62.5/62.5/50% are **theoretical**, not achieved occupancy.
`RmProfilingAdminOnly:1` remains unchanged. Achieved SM occupancy, measured DRAM
throughput, cache and stall counters remain unavailable; no repeated denied
counter campaign or privilege change was attempted.

## Provenance, failures and use

Candidate SO SHA256:
`19ee200feef03d2ddd10e313b680c393fbf8ee170f367d08d0bbd6b5edaebc13`.
All10 compiled dependencies match current native sources. Control hashes and
exact commands are in `v100/campaign-v6/campaign-report.json`. Full target/draft/
GGUF hashes are freshly checked after all inference jobs. All80 pinned release
files match the original external bundle; no exact upstream source-commit claim
is inferred from its build-info text. Large SO/PT/SQLite/nsys files remain remote
by recorded SHA rather than being downloaded wholesale.

Preserved failed attempts: campaign-v3 was canceled while waiting, **before any
timing**; budget-v4 raises a tool NameError before model construction, fixed in
budget-v5; profile-v7 completes two C1 traces then fails C8 argument parsing
(missing `--mode batch`), repaired in profile-v9 by reusing the completed traces;
cli-v8 stops on that failed prerequisite without attempting CLI, repaired in
cli-v10. Failures are not relabeled as native correctness or performance results.

The actual public CLI command and returned fixed IDs/precision metadata are in
`v100/cli-v10/actual-cli.json.command.json` and `actual-cli.json`. Opt in using
`--projection-profile q8-residual-column-mlp --dtype fp16 --weight-mode resident
--tensor-parallel-device cuda:1 --tensor-parallel-transport host` with the matching
library. Scalar MTP2 additionally uses `--mtp-draft-tokens 2 --mtp-recovery prefix`.
HTTP adapter modes remain explicit `--batch-mtp` versus `--batch-target
--projection-rows 256`; no automatic mode switch is claimed.

All36 prior capsules are reverified without mutation; all five executed source
snapshots are preserved. All10 owned workers are terminal; no owned inference
process, periodic sampler or port18953 remains. PhysicalGPU0/1 returns to
3651/3189 MiB. No unrelated jobs, driver settings, permanent service, commit or
push is changed. Recompute metrics/compatibility from the preserved scripts;
the evidence/source manifests freeze this capsule. The measured next choices,
and why none is yet a promised speedup, are in `next-bottlenecks.md`.
