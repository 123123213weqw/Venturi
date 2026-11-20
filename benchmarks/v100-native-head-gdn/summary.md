# Native semantic-head GDN integration — qualified engine gain

Status: accepted opt-in native profile; full external-comparison goal remains
active pending the final matched controls/configuration audit. No commit/push.
All 42 earlier capsules remain immutable. `metrics.json` is recomputed from raw
reports by `summarize.py`; profiled measurements are not throughput results.

## Implementation and precision

`q8-residual-column-mlp-head-gdn` extends the qualified residual-A8/column-MLP
profile with semantic GDN-head TP on the two V100s. Each rank owns half of the
Q/K/V heads, convolution history, FP32 recurrent state, rollback/prefix buffers
and the corresponding output columns. Encoded original Q8 codes/scales and small
parameters are preserved. Gates are batched once per rank; FP32 output partials
cross the explicit host bridge, sum in FP32 and cast once to FP16. That changed
association is separately qualified, not described as byte-identical to the old
row output. Source checks remain intact. There are no new CUDA kernels or Python
operations in the native generation loop. Defaults and ABI16 descriptors remain
unchanged; unsupported configurations fail closed.

The remote-built/actually loaded CUDA library is
`33a4becb5f9c8440492e9cc6352ea87fccab2692a01adc48e6cc756fa93bc7b8`.
All 11 C++/CUDA dependency hashes match executed source-v2/v3. Compilation was
remote only. Per-rank model-shared workspace at row budget32 is1,254,400 bytes;
private recurrent caches and MTP rollback/prefix storage remain per request.

## Correctness before timing

* 26 focused two-GPU tests pass: independent source equations, T1/2/3/8,
  private/ragged cohorts, selective reset, EOS, injected failure and MTP
  acceptance/rejection/rollback/prefix handling without Python math.
* Remote full regression:2,242 passed/719 skipped, with one CPU assertion still
  expecting10 build dependencies. Source-v3 updates it to11 and asserts the new
  dependency;38 closure tests pass. This is not mislabeled an all-green first run.
  Final local suite:831 passed/2,131 skipped.
* Actual27B:98 hash-pinned original-FP16 teacher rows, four full trajectories,
  scalar/native canonical caches, mixed-eight/private-cache and selective-reset
  gates pass. Maximum relative L2=.002581494, minimum centered cosine=.999996313,
  maximum KL=8.292874e-5; all98 top1s match. Unchanged thresholds .02/.999/.002.
  Archived teacher files are hash checked/reused, not an invented fresh teacher
  campaign or broad original-BF16 corpus-quality claim.
* Trained MTP2 independently checks two block chunks and three recurrent steps;
  four cases preserve committed target and corrected draft state. Packed MTP1
  checks four references/two eight-request cohorts and untouched selective reset.
* Initial missing fixture directory, stale dependency-count assertion and an
  incompatible qualification CLI option are preserved. The latter stopped at
  preflight; rerun without that option passes all trained MTP checks.

## Unprofiled speed: same work, two V100s

23 fixed formatted IDs,32 greedy output tokens ignoring EOS, FP16 KV,256 context,
8 prepared slots, row budget32, same original-source Q8 target and trained draft.
Logical primary/peer0/1 map to physical1/0. Both profiles use explicit host TP.
C1/C8 screens use two alternating pairs per mode; both pass and promote to
three repeats in every1/2/4/8 cell. No forced proposals or target-input replay.
MTP2 acceptance17/26; MTP1 acceptance14/16 per request, unchanged.

Single native post-first decode is31/(generation minus first-ready):
**48.48697 →52.95941 TPS** (+9.22%). Candidate trials:
52.92176 /53.01174 /52.94480. All exceed the retained pinned external best
49.80330 by at least5% (floor52.29347); combined ratio1.06337.

Complete HTTP TPS is32*N/global cohort wall, not a sum of overlapping timers:

| Requests | Scalar MTP2 old → new | Packed MTP1 old → new | Selected new |
|---|---:|---:|---:|
|1|38.5193 →42.2671|36.2806 →39.6892|42.2671|
|2|37.6888 →41.3790|51.8425 →57.2975|57.2975|
|4|38.5907 →41.3051|63.5383 →70.2869|70.2869|
|8|38.5660 →42.1254|68.8521 →76.4590|76.4590|

Every selected trial clears5% over the strongest retained external HTTP cells
30.1731/37.9074/50.2537/67.2872. The stronger prior ordinary256 native C8 result
73.8363 is retained; new packed C8 exceeds it3.55%. Scalar MTP2 C8 still loses to
the external engine: **no single mode wins every cell**, and no automatic
strategy router is implemented. Ordinary256 with this new profile is not yet
qualified/timed. Fresh matched no-speculative and competitor controls remain a
separate final phase; no fresh external timing is claimed in this capsule.

## CPU, latency and VRAM

C1 scalar CPU .8300→.7533 core-seconds; native first-ready .15488→.13821s;
native generation .79422→.72356s; complete cohort .83075→.75709s.
C8 packed CPU3.7167→3.3433 core-seconds and wall3.71811→3.34820s.
Full mean/max client latency, both-device NVML samples, power, memory/SM clocks,
temperatures and all per-trial denominators are retained in metrics/raw reports.

C8 device-used VRAM includes eight prepared slots and unrelated baseline users:
scalar primary/peer23,995/17,229→21,365/19,583 MiB;
packed22,461/17,229→20,619/18,815 MiB. More peer state is intentional.
Packed C8 sampled busy means85.33/58.43→85.95/74.10%; these are **not occupancy**.
SM clock1380MHz in selected windows; maximum temperature49C.

## Separate timeline/resource diagnosis

Separate C1 scalar instrumented spans843.80→770.91ms; C8 packed3767.64→3387.36ms.
C1 two-device kernel overlap331.25→400.93ms; C8 overlap1642.01→2011.71ms.
Host payload per direction decreases316.66→276.40MB (C1) and2232.04→1945.78MB
(C8). Summed kernel time actually rises3.23%/1.93%, and launch counts rise:
benefit is distribution/overlap/less host bridging, not less total GPU work.
Process-activity gaps125.18→101.90ms and302.57→235.40ms are not device-idle
measurements. Exact-SO static resources/actual launch geometry are recorded;
the GDN prefix kernel remains128 threads/47 registers, grid6144→3072 per rank.

`RmProfilingAdminOnly: 1` is unchanged. Achieved occupancy, bandwidth, cache/stall
and compute-throughput counters remain unavailable; no theoretical/resource
estimate is substituted. Eight SQLite/nsys binaries remain remote with hashes;
local attribution/resource reports suffice to reproduce the summary.

Six workers are terminal; no owned model process, periodic sampler or port18953
remains. Original physical0/1 VRAM3651/3189MiB is restored. Strict same-host SSH
recovery used bounded retries/manifest checks and never blindly relaunched an
unknown dispatch. See `network-recovery.md` and preserved failed attempts.
