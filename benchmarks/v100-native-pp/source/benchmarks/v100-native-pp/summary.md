# Native two-V100 PP versus TP

## Scope and result

Native Qwen3.5-27B, original-derived Q8 group-32 weights, FP16 activations/KV and
preserved FP32 small parameters/GDN state. **Both modes use two V100-PCIE-32GB
GPUs**, explicit host DMA, the same ABI-14 library and the same checkpoint.
No MTP, continuous batching, paged KV, HTTP serving or whole-model graph is
involved. The TP baseline is the existing whole-request-serialized row-sharded
runtime. PP is a new native stage-overlapping runtime; this is not a comparison
against optimized batched TP in another engine.

Correctness gates ran before timing. Unprofiled final phases are TP/PP/PP/TP,
one trial per concurrency per phase: **two trials per mode at each of 1/2/4/8
requests**. Every request uses the pinned 23-token non-thinking story prompt,
32 fixed greedy output tokens and EOS stopping disabled; none of the 32 output
IDs is EOS. Warmup/split/whole outputs all exactly match the previously pinned
single-native reference. Load, preparation, reset and full-work warmup are
outside timing. Each request makes one native generate call per timed phase.

### Aggregate throughput

Decode emits 31 continuation tokens per request after the separately timed
first token. Aggregate TPS is **all cohort tokens / cohort wall time**, never a
sum of request latencies or an average/sum of per-request TPS.

| Concurrent requests | TP decode TPS | PP decode TPS | TP whole-request TPS | PP whole-request TPS |
|---:|---:|---:|---:|---:|
| 1 | **17.17** | 14.90 | **13.97** | 11.24 |
| 2 | 17.18 | **28.13** | 13.99 | **19.24** |
| 4 | 17.17 | **28.31** | 13.98 | **20.27** |
| 8 | 17.18 | **28.24** | 13.98 | **20.91** |

Whole-request TPS includes prefill and all 32 emitted tokens per request. At
eight requests, PP improves aggregate decode throughput approximately 64.4%
and complete-request throughput approximately 49.6%. These are **aggregate**
rates, not 28 TPS for every client. A single request is faster with TP here.

### Latency and CPU trade-offs

Each complete-request cohort starts at a common barrier. Individual completion
latency includes its scheduling/queue wait from that barrier. PP improves cohort
makespan and the worst observed completion at higher concurrency, **not every
individual latency statistic**. Serialized TP finishes early queued requests
sooner; overlapping PP spreads GPU service across requests.

| Requests | TP cohort wall | PP cohort wall | TP mean request completion | PP mean request completion | TP process CPU | PP process CPU |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.290 s | 2.848 s | 2.290 s | 2.848 s | 2.293 core-s | 2.851 core-s |
| 2 | 4.575 s | 3.327 s | 3.433 s | 3.310 s | 4.583 core-s | 5.701 core-s |
| 4 | 9.154 s | 6.316 s | 5.729 s | 6.263 s | 9.173 core-s | 11.465 core-s |
| 8 | 18.314 s | 12.241 s | 10.319 s | 12.118 s | 18.380 core-s | 22.992 core-s |

At eight requests the worst observed completion drops 18.319 -> 12.266 s, but
mean completion rises 10.319 -> 12.118 s. Do not advertise a universal latency
win. These two-trial observations are not production p99/SLO estimates.

The raw `native_call_wall_seconds` field is host-visible request-wrapper time,
including metadata/preflight and the native call, not GPU-only execution time.

Mean request first-token latencies from the separate first-token wave, TP/PP:
0.491/0.778 s at one request, 0.736/0.985 s at two, 1.231/1.412 s at four and
2.255/2.269 s at eight. The entire first-token wave finishes before the separate
continuation wave begins. This controlled split is not a measurement of token
streaming or HTTP TTFT inside the complete-request call.

CPU core-seconds are process-global per cohort, not summed overlapping request
process measurements. Per-thread CPU is recorded separately. At eight requests
PP uses about 1.88 core equivalents versus TP's 1.00, and **more total CPU time**
although cohort wall time is shorter. CPU time includes native/CUDA waiting and
dispatch, not a Python model loop. Faster GPU throughput is not an all-resource
or energy-efficiency guarantee.

## Implementation

Blocks **ABI 14** adds stage construction, pipeline/session ownership and native
forward/generation/snapshot/reset APIs. The existing config descriptor remains
104 bytes; ABI-13 libraries are rejected before new bindings are used. Build
cache identity includes `qwen_pipeline.cuh` as well as the runtime/kernel source.

The primary logical GPU runs embeddings and layers `[0,32)`; the second GPU runs
`[32,64)`, final normalization, the output head and greedy selection.
Only stage-owned Q8 matrices/constants and per-request GDN/history/KV are stored
on each GPU. A request-owned portable pinned buffer moves the hidden state at
the boundary. Values remain the same FP16 bits; there is no CPU model math,
rounded-partial-sum reduction, weight offloading or hidden precision change.
Tied embedding/head parameters may be copied to both devices when necessary.

Each stage has its own native stream/model lock. A request releases the first
stage before acquiring the second; other requests can occupy the other GPU.
Independent Python threads submit one whole native call per request, but layer,
token, cache and stage execution remain C++. Stage locks are not continuous
batching or a native central admission queue. The pipeline split is explicit;
32/32 was tested, not asserted globally optimal. No automatic TP/PP switching or
simultaneous TP x PP configuration is implemented on these two GPUs.

The final implementation keeps output-head selection and the selected-token D2H
inside the final-stage lock. The initial version released that lock before
selection and could strand a previous request behind another complete stage.
Initial data are preserved, not substituted for final results. Two-request PP
initially measured about 20 aggregate decode TPS; atomic final-stage selection
raises it to about 28.13 without changing target math or output IDs. Correctness
and the complete comparison were rerun after this scheduling change.

The host's direct peer-copy byte checks previously failed despite advertised
capability. PP explicitly uses host DMA and never tries peer access; TP timing
also explicitly selects host transport. The cards have `NODE` topology, not
NVLink. `CUDA_VISIBLE_DEVICES=1,0` maps the first stage to physical GPU 1 and the
second to physical GPU 0. Smaller TTS/RWKV services were left running; no other
GPU service was stopped, restarted or reconfigured in this milestone. Driver,
IOMMU/ACS, clocks and profiling policy were not changed.

## Correctness

Fixture checks cover three splits (1/2/3 in a four-layer hybrid), tied/untied
heads, odd shapes, chunk and continuation logits, every hidden/GDN/history/KV
byte, source guards, EOS/capacity checks, failure in either stage, reset and
closed-model session lifetime. Concurrent fixtures use **different prompts**
and compare each request's generated IDs and complete state at 1/2/4/8 requests.
Generation also passes with Torch numerical/model calls forbidden.

The real trained 64-layer Paris + EOS qualification is repeated for the initial
and final scheduler. Both preserve every previous-native logit/hidden/GDN/
history/KV byte and pass the unchanged fixed CPU-reference logit gates. Fresh
native generation emits `[57590,248046]`, cursor 27, without Torch math. The
final CLI separately exercises PP with these EOS/position expectations.

- Final local suite: **721 passed / 494 skipped**; Ruff and diff whitespace checks pass.
- Final V100 full suite: **789 passed / 426 skipped**.
- Final selected PP/TP/native fixture suite: **30 passed / 27 skipped**.
- Initial implementation suites and the slower initial scheduler are retained.
  One queued profiler plan was superseded before capture; only its owned waiter
  was stopped, with PID/command/child checks and a recorded reason.
- SM70 BF16 execution remains explicitly rejected/skipped. SM89 was not exercised
  in this milestone, nor was a four-GPU TP x PP configuration.

Quality scope remains the represented Q8/FP16 profile and pinned prior-native
reference with shared metadata/loading adapters. This is not original-BF16
corpus quality or an independent upstream architecture qualification. Short
contexts and 32-token outputs do not qualify long-context production behavior.

## GPU diagnostics

Device-wide NVML means during timed windows at eight requests:

| Metric | TP physical GPU 1 / 0 | PP physical GPU 1 / 0 |
|---|---:|---:|
| GPU busy | 75.76% / 64.77% | 89.29% / 94.04% |
| Memory-controller busy | 34.50% / 33.59% | 48.94% / 52.47% |
| Used VRAM | 20,057 / 17,223 MiB | 18,481 / 18,943 MiB |
| Power | 170.91 / 154.09 W | 218.78 / 219.96 W |
| Temperature | 52.79 / 50.01 C | 58.67 / 57.68 C |
| SM clock | 1,380 / 1,380 MHz | 1,372.96 / 1,379.97 MHz |
| Memory clock | 877 / 877 MHz | 877 / 877 MHz |

These include other users/contexts, not only native allocations. Samples are
proxies, **not achieved occupancy or DRAM bandwidth**. PP's primary clock ranged
1,342..1,380 MHz under unchanged automatic clock/power policy. Both devices'
complete samples/minima/maxima and native model/request/pinned accounting are
preserved in raw reports and `metrics.json`. Shared model memory is counted once;
request memory is per session, including `peer_request_device_bytes` for PP.
TP transfer counters are model-wide; PP bridge/selection counters are per request.
Do not sum duplicated TP model counters across sessions.

Separate Nsight CUDA/NVTX captures use eight output tokens and seven decode
steps, at one and eight requests. Device-event completion tracing is disabled
consistently with `--cuda-event-trace=false`. These are **profiled diagnostics**,
not replacement TPS measurements:

| Decode trace | TP, 1 request | PP, 1 request | TP, 8 requests | PP, 8 requests |
|---|---:|---:|---:|---:|
| Host cohort span | 440.917 ms | 482.716 ms | 3,546.669 ms | 2,078.915 ms |
| Summed kernel durations | 486.180 ms | 463.966 ms | 3,886.346 ms | 3,705.866 ms |
| Simultaneous two-device kernel intervals | 171.748 ms | 0 ms | 1,374.978 ms | **1,713.211 ms** |
| Gaps between this process's activities | 94.314 ms | 7.587 ms | 764.852 ms | **4.417 ms** |
| H2D copies | 5,621 | 14 | 44,968 | **112** |
| D2H copies | 5,621 | 14 | 44,968 | **112** |

At eight requests the TP decode trace transfers about 557.642 MB in each
host/device direction; PP transfers about 0.574 MB in each. PP moves boundary
hidden states and token IDs/selection results, not every projection's shards.
No resident decode weight uploads occur. The traces demonstrate actual stage
overlap at eight requests and sequential stages at one; **overlap is not achieved
SM occupancy**. Kernel sums can exceed wall time because GPUs overlap. Activity
gaps are process-local trace gaps, not device-wide idle time or a complete causal
account of every CPU wait. Per-device interval unions, copies, kernel attribution
and the synthetic cross-thread/correlation overlap test are retained.

An earlier eight-request TP capture stalled during warmup while GPUs were idle.
Nsight warned that device-side CUDA-event completion tracing can introduce false
dependencies. Only the owned target/worker and its orphaned samplers were stopped;
PID/command checks and canonical log-path checks are recorded. The termination
trap wrote an exit marker of zero, but the raw report is **incomplete**, so that
marker is explicitly not treated as success. The failed capture is preserved in
`v100/failed-profiler-event-trace/` and contributes no performance data. Retrying
with device-event completion tracing disabled completed all four captures with
expected IDs. This is an unresolved instrumentation interaction, not evidence
that unprofiled eight-request performance failed, nor a proven driver root cause.
No engine math or driver configuration was changed for the retry.

Achieved occupancy, SM throughput, actual DRAM bandwidth and cache/stall counters
remain unavailable under the unchanged `RmProfilingAdminOnly: 1` policy. No
repeated hardware-counter campaign or permission change was attempted.


## Reproduce and remaining work

`python3 benchmarks/v100-native-pp/summarize.py` recomputes the comparison from
preserved JSON/CSV. Frozen source, build manifests, launchers, source/evidence
hashes and critical initial-source variants preserve provenance. Large snapshot
PT and Nsight binary/SQLite files remain on the server at recorded paths; their
small reports/logs are in `v100/`. Prior evidence capsules are audited unchanged.

The useful choice depends on the metric: TP wins the tested single-request
latency; PP wins aggregate throughput at 2/4/8 requests but trades CPU cost and
some individual latencies. Improving stage balancing, request scheduling,
continuous batching and native projection/launch efficiency remains possible.
MTP, paging, long context, original-BF16/upstream quality and matched external
comparisons remain unfinished. This does not establish the external 50 TPS
claim, universal superiority, a completed larger goal or a GitHub push.
