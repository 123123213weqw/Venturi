# PR #14: QuantLinear correctness, reusable dispatch and measured safe paths

This campaign repairs PR #14 rather than qualifying its historical speed claims.
Frozen earlier raw benchmarks remain untouched. Work was done in an isolated
worktree; the original 538-entry dirty working tree was not modified.

## What is shared / what is model-specific

- `quantization/linear.py`: codec/architecture/numerical-policy admission,
  implementation qualification and generic diagnostics.
- `ops/csrc/row_codec_dispatch.cuh`: prepared plans by bits/group/architecture/
  activation arithmetic. No per-projection model-name or matrix-shape lookup.
- `ops/csrc/weight_dequant.cuh`: canonical Q4/Q5/Q6 contiguous and indexed row
  decode, reusable for projections, embeddings and MoE tables in other adapters.
- The Qwen adapter binds checkpoint roles to the common matrix descriptors.
  Mixed-format role metadata is validated before upload; nothing is re-encoded.
  Its native chunked projection consumer uses one strided 2D copy for a multi-row
  chunk instead of one D2D copy per token. Other consumers can reuse the decoder;
  this PR does not claim they have already integrated it.

Operator ABI **18** and Qwen runtime ABI **3** are a required explicit pair.
Old ABI-2 defaults are rejected even when struct byte sizes happen to match.
Formats are unchanged: no checkpoint conversion is required by the loader.

## Deliberately closed alternatives

Canonical low-bit `auto` uses FP16 decode plus cuBLAS, bounded by one chunk;
there is no implicit full-model FP16 shadow, activation quantization, CPU fallback
or OOM precision change. Q8 retains its existing kernels.

The SM70 fused-Q5 real-artifact campaign passed logits but failed **50 private
state components** at the unchanged state gate. An L40 fused-Q4 fixture failed
2/320 elements (max abs 0.0074463 > 0.005). Both remain research-only, not `auto`.
Explicit activation INT8 also remains closed: the independent model fixture's
relative L2 0.0553 exceeds 0.02. Preserving these failures matters more than a
faster unqualified trajectory. Low-bit/mixed MTP, TP and PP are rejected; two
replicas in the benchmark are **not** tensor parallelism.

## Arithmetic-preserving optimization

Original row decode assigned one thread to serially unpack a group of 64 weights.
The revised decoder assigns adjacent octets to adjacent threads: aligned packed
32-bit reads and 16-byte FP16 writes, scalar fallbacks for offsets/odd K/tails.
The signed integer times FP16 scale still rounds exactly once to FP16. It neither
changes bit rate nor quantizes activations.

Tests cover Q4/Q5/Q6, K=1..513 edge geometries, offset input/high/output views,
partial/indexed rows, zero/subnormal/extreme finite scales and missing high-plane
errors. Multi-token output copies include a vocabulary-size-8193 chunk tail,
compared byte-for-byte with an explicitly frozen independent DSO pair.

On an idle L40, three repetitions per format/shape gave **5.43–6.52x** row-decode
microkernel speedups with bit-identical output. Shapes were 8192x5120 and
5120x17408. This is CUDA-event timing of repeated/hot row decoding, **not full-model
TPS, cold-start performance, achieved occupancy or DRAM-counter utilization**.
Raw samples, CPU/wall time and sampled GPU diagnostics are under `sm89-coalesced/`.

## Real-model procedure

Target: `V100_222`, host `root1`, two V100-PCIE-32GB cards. User explicitly
allowed a temporary production pause. No production parameters/routes, system
permissions, monitoring, drivers or other-user processes were changed. Every
window captured exact original argv/env/cwd/executable/log/affinity/umask in a
private file and launched an independent restoration watchdog **before** pause.
Private environments and model/DSO binaries are not committed here.

Artifact: actual 64-layer / hidden-5120 Qwen-27B text Q5 checkpoint,
17,686,008,520 bytes, SHA256
`dd57246737055d02c7cd19aa584c766981c2ea11d475b6608597b2057fa93bae`.
The whole destination file was SHA-verified before model work. This is not a 4B
substitute. Original upstream BF16 checkpoint/revision is not established here;
quantization accuracy against BF16 or corpus quality remains unqualified.

`cases.json` contains three tokenizer-backed sanity cases and the **exact**
16-input-token / 32-output-token timing case. Forced output ignores EOS. Native
reference/candidate logits, greedy trajectories, convolution/GDN/KV private
states and C4/C8/C16/C32 grouped states/output must pass unchanged gates before
TPS. Reference is the same Q5 artifact with canonical FP16 decode, not BF16.
No tolerance widening or scalar-versus-batch shape substitution is allowed.

Timing uses two independent native instances, one per card, and loopback HTTP:
32 slots/card, capacity256, projection rows32, sealed metadata, FP16 activations,
continuous decode, round-robin chunked prefill16, no MTP/graph/prefix reuse.
Three clean repetitions per C1/4/8/16/32; throughput is total completed output
tokens divided by the **common** cohort wall time, including admission, prefill,
decode and HTTP. It is not decode-only TPS or streaming HTTP TTFT. Scheduler
first-token availability is reported separately. CPU, response latency, RSS,
GPU busy/VRAM/power/clocks/temperature are recorded, not replaced by one TPS.

The second node22 campaign completed all 15 reference cells but hit its bounded
campaign deadline before any candidate timing. Its overall incomplete status is
preserved under `node22-reference-v2/`. The next window reuses only that completed
arm after checking exact artifact, reference DSO hashes, case hash, output IDs,
server configuration and harness identity. This saves another long production
pause; the different wall-clock windows are explicit. It is not an interleaved
A/B campaign and does not eliminate clock/system drift.

## Preserved failures and scope limits

- Earlier WZU runs were interrupted by independent CUDA jobs/service restarts;
  they are not qualified timing. No other-user job was stopped.
- First node22 qualification used three semantic cases; its timing helper found
  no matching 16/32 fixture and refused to report TPS. Exact workload matching
  is now self-described and unit-tested; never fall back to the last case.
- Partial/invalid output yields null aggregate TPS and a failing CLI status.
  Contaminated cohorts are retained separately and excluded, with bounded timing
  attempts. Failed generation is never silently retried.
- NVML process sampling is every50ms; absence of an observed foreign context is
  not proof of exclusivity between samples. GPU busy is not achieved SM occupancy.
- Profiler runs are separate. Hardware counters unavailable under current policy
  remain unmeasured; no privilege/DCGM changes are made to obtain them.
- Q4/Q6 have operator/synthetic coverage, not an original-source real27B quality
  qualification on SM70. No arbitrary GPU, low-bit TP/PP/MTP or production-server
  compatibility claim follows from this campaign.

See `status.json` and the final qualified summary for completed outcomes, timings
and restoration receipts; no number in this document supersedes raw failures.

## Completed outcomes (2026-11-17)

- CPU final: **1957 passed / 3174 skipped** (CUDA absent locally).
- CI Python3.10/3.12: green at source/gate head `2b4bbe8a`, run44100554606.
- L40: **388 passed / 215 skipped**, including independent row decode and
  frozen-runtime multi-token chunk tails, plus existing TP/PP/PD/MTP regressions.
  This is not a low-bit TP/PP/MTP support claim.
- V100: **191 passed / 153 skipped**; native BF16 unsupported and explicit
  off-path research tests are skipped rather than silently emulated.
- Actual27B: all4 scalar cases/native-generation replays passed; C4/8/16/32
  output trajectories and all **7680 private-state components** matched, with
  grouped private states byte-identical. Original BF16/corpus quality remains
  unmeasured. Only canonical A16 Q5 execution qualification on SM70 is opened.

| Concurrent requests | Safe frozen control TPS | Optimized safe TPS | Ratio |
|---:|---:|---:|---:|
| 1 | 0.750 | 5.385 | 7.18x |
| 4 | 2.777 | 19.155 | 6.90x |
| 8 | 5.307 | 34.606 | 6.52x |
| 16 | 5.208 | 28.973 | 5.56x |
| 32 | 6.717 | 30.433 | 4.53x |

These are **three-repeat medians**, not best runs. C16 candidate values were
42.511 / 28.973 / 28.878 TPS; C32 were30.433 / 29.447 / 31.081. The table preserves
that non-monotonic scaling. C16/C32 dynamic/cohort behavior remains a follow-up
bottleneck; this campaign does not establish one exclusive cause. GPU busy was
already ~98–100% on many old slow cells, demonstrating that busy is not efficiency.

At C8 the median response latency fell46.823→7.284s and scheduler first-token
availability3.166→0.739s. Summed two-server CPU time per cohort fell95.08→14.76s;
CPU/wall remains ~2cores, consistent with active host waiting/launch work;
this is an inference, not CPU counter attribution. This is reduced work
per completed cohort, not elimination of CPU consumption. VRAM peak remains
~25410–25412MiB per card; maximum serverRSS ~18GiB per process. Full cells retain
clock, temperature, power, busy and memory samples.

Separate C16 one-step Nsight attribution (not TPS): total row-decode kernel time
**1280.687→91.544ms**, and GDN **41.354→16.283ms**. Main GEMM time remains
~193.5ms and becomes the dominant individual kernel family. This confirms the
row-decode bottleneck moved; it does not attribute the whole end-to-end speedup
to one change. Traces additionally cover promoted batched GDN/conv and strided
copies. `profile-summary.json` records recorded-activity gaps, CUDA API counts
and raw-trace SHA256s. Counter occupancy/DRAM bandwidth were not measured
(`RmProfilingAdminOnly=1`), no permission/monitor changes. Raw Nsight reports and
SQLite stay remotely at the paths in the receipt rather than bloating this PR.

Final restoration: original PID1374268→PID1386561, HTTP200, exact command/env/
cwd/binary/log/affinity/umask identity preserved. No production deployment or
default/routing change was performed. See `node22-coalesced/production-restore-primary.json`.

The timed native source was `6eb0777c`; `2b4bbe8a` only records the now-earned
Python qualification gate plus its regression, without changing timed kernels.
Historical47–55TPS fused claims are **not current qualified results**.

The separately recorded GPU activity interval (first to last kernel/copy) was
1593.202→347.380ms, with recorded gaps14.355→3.399ms. This interval is not the
whole HTTP response and its coverage is not SM occupancy. CUDA copy API calls
changed12307 `cudaMemcpyAsync` →291 async +751 `cudaMemcpy2DAsync`; kernel launches
4290→2850. These explain the reusable launch/copy reduction without asserting
that a single profiled step predicts every continuous-scheduler cohort.
