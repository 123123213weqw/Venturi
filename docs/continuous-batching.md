# Opt-in native continuous scheduling

This is an ordinary-greedy request scheduler over an explicitly prepared native
Qwen Q8 batch, not a new model loader/backend or a replacement production route.
Existing scalar/cohort generation, MTP and CLI defaults are unchanged. It does
not make the current mixed Q4 artifact, Flash-Next or DeepSeek fully executable.

Separate prefill/decode device pools use the explicit
[PD runtime](pd-disaggregation.md), not an automatic mode of this scheduler.
Neither path silently changes the production route or imports an external engine.

## Scheduling and ownership

`venturi.runtime.ContinuousScheduler` uses the standard library. No import starts
a GPU, model, worker or HTTP server. The caller supplies a prepared ordinary
`NativeQwenBatchSession`, keeps its model alive, and exclusively leases that batch.
Submission and cancellation can come from other threads; exactly one owner calls
`tick()`. Simultaneous/reentrant ticks and reset/close during execution fail.

Each tick:

1. Retire cancellations at a safe iteration boundary and admit pending requests
   into compatible free slots, resetting each newly admitted slot before use.
2. Schedule one input token for each active decode. Spend remaining token rows
   on bounded round-robin chunks of incomplete prompts.
3. Make one native mixed-group call. Projection/norm/MLP rows are grouped;
   attention KV, convolution history, recurrent state and positions stay private.
4. Sample only decode rows and the final chunk of a prompt. Partial prefills do
   not emit tokens. Complete each request's future at its own EOS/output limit;
   the freed slot can accept a new request at the next tick.

The last emitted token is not consumed, matching existing native generation.
Slot reuse does not migrate or share cached user state. Static capacities reserve
memory; retiring a request does not imply a `cudaFree` per token/request. No
attention paging, prefix caching, state preemption or RAM fallback is introduced.

`token_budget` limits input rows per iteration, **not context capacity**. It must
cover the prepared slot count and cannot exceed `batch.max_token_rows`.
`prefill_chunk_size` caps each selected prompt chunk; a finite `max_pending`
rejects excessive queued work instead of growing an unbounded request queue.
Prompts, budgets and EOS IDs are validated before native execution. A request
must fit at least one configured slot; best-fit admission preserves larger slots
when possible. No context, dtype or output budget is silently reduced.

### Experimental balanced prefill sharing

`prefill_policy="round-robin"` remains the default. Explicit
`prefill_policy="balanced"` instead max-min shares the remaining input-row budget
between incomplete prompts, capped by each prompt's remaining length and
`prefill_chunk_size`. Unused short-prompt shares are redistributed; final
one-row remainders rotate between requests. All live decodes still go first.
There is no added admission delay, change to model arithmetic, or extra device
allocation. The selected policy appears in `scheduler.info()`.

For example, with eight 16-token prompts, 32 prepared rows and a per-request
chunk cap of 16, balanced sharing feeds four tokens from **each** request per
iteration rather than finishing two prompts first. This can avoid a partially
filled mixed group and a small decode tail. It can also delay the first token
of the earliest requests while the other prompts catch up: it is a scheduling
tradeoff, not an unconditional latency improvement. Do not infer HTTP or
streaming speed from a native replay.

The loopback benchmark adapter exposes the same explicit selection as
`--continuous-target --prefill-policy balanced`; requesting balanced sharing
without a continuous target is rejected before model/device access. Neither
this option nor any measurement changes the production route or model default.

`prefill_policy="balanced-short"` is the more conservative experimental option:
it uses sharing **only** when there are no live decodes, all incomplete prompts
have the same remaining length, and that length fits the configured chunk cap.
Otherwise it uses the original round-robin plan, including for long or
heterogeneous prompts and new prefills alongside running decoders. This is a
deterministic metadata gate, not a GPU-utilization guess or an MTP/backend switch.
The same earliest-token tradeoff still applies to the uniform short cohort.

The unrestricted policy's regression is retained, not hidden: the first matched
two-V100, native Q8 replay campaign improved fixed count 4/8 throughput by
2.31%/2.29%, but reduced the mixed-arrival replay's throughput by 2.45%. For fixed
count 8, the earliest native token event moved from a median 0.156 s to 0.622 s.
This is why neither policy replaces the latency-oriented default automatically.
Raw inputs, exact greedy/private-state checks, separate profiler reports and
three matched timing repeats are in
`operations/prefill-balance-20261029-v1/`; the short-cohort gate is tested in a
separate campaign, never scored using the unrestricted policy's timing results.

```python
from venturi.runtime import ContinuousScheduler

# model is already explicitly loaded, qualified and resident; this creates no
# backend, chooses no quantization and never downloads/compiles anything.
batch = model.create_batch([1024] * 8)
scheduler = ContinuousScheduler(
    batch, token_budget=32, prefill_chunk_size=16, max_pending=128,
)
try:
    a = scheduler.submit(prompt_ids_a, max_new_tokens=32, eos_token_ids=[eos_id])
    b = scheduler.submit(prompt_ids_b, max_new_tokens=64, eos_token_ids=[eos_id])
    while scheduler.info()["active_requests"] or scheduler.info()["pending_requests"]:
        for event in scheduler.tick():
            # Hand off lightweight metadata to a client/streaming queue. Do not
            # tokenize, format large JSON or perform blocking IO in this loop.
            consume(event.request_id, event.token_id, event.finish_reason)
    result = a.future.result()
    print(result.output_ids, result.time_to_first_token_seconds)
finally:
    scheduler.close()   # Cancel/reset the lease, NOT the caller's batch/model.
    batch.close()
```

The model's prepared row limit must be at least 32 for that example. The example
budgets are illustrative, not universal tuned values.

## Explicit worker and loopback adapter

`ContinuousWorker(scheduler).start()` gives the prepared scheduler exactly one
inference thread. Idle waiting uses a condition variable, not a polling loop.
Concurrent callers use `worker.submit(...)` and await each ticket's future.
Neither construction nor import starts the worker, loads weights or probes CUDA.
`worker.close(timeout=...)` waits for a safe boundary, cancels outstanding work
and resets leased slots. If it times out, the owner is **still alive**: retain its
batch/model and join it successfully before freeing those buffers. Execution or
cleanup failure is surfaced, never silently restarted on another engine.

The existing loopback-only, pretokenized benchmark adapter now has an explicit
`--continuous-target` mode, for example:

```sh
python tools/serve_native_benchmark.py /absolute/model.ninfer \
  --library /absolute/libventuri_blocks.so --mode tp --port 18033 \
  --ready-file /absolute/ready.json --slots 8 --capacity 512 \
  --projection-rows 32 --projection-profile q8-residual-column-mlp-head-gdn \
  --continuous-target --prefill-chunk-size 16 --max-pending 128 --metadata-mode sealed
```

This is an opt-in benchmark surface, **not a production API deployment** or a
claim that every model/precision/backend supports continuous scheduling. It
rejects PP, MTP, adaptive/fixed-cohort overrides and target graphs before loading
the model. Existing modes/defaults remain unchanged. Continuous mode imposes no
artificial cohort window and accepts prompts beyond the legacy 32-token limit,
subject to explicit static capacity/output limits and chunked execution.

Its `id_slot` identifies a serialized HTTP client lane, not the native cache slot;
the scheduler assigns/reset private native slots dynamically. The response's
`native_timings` is intentionally null. `scheduler_timings` measures submission
to this request's native token availability/completion, **including** admission
and scheduling; it is neither native-only timing nor HTTP streaming TTFT. HTTP
JSON encoding runs on request threads. There is no HTTP token streaming yet.

## Failure and cancellation

`scheduler.cancel(ticket)` (or cancelling its future) is observed at a safe
boundary. In-flight native work is not interrupted midway through GPU kernels;
cancelled requests do not keep a reusable live request state. No cancelled slot
is reassigned before reset. Foreign tickets are rejected.

A native/admission-reset/result-contract failure fails outstanding futures and
poisons the scheduler owner. There is no retry on a different strategy or
external engine. `scheduler.reset()` is explicit recovery; it resets all leased
slots and cancels outstanding requests. `close()` is idempotent after a
successful close. Reset failures propagate rather than pretending cleanup passed.
Future callbacks must be lightweight and must not call `tick()` recursively.

## Native step boundary

`batch.step(appends, slots=..., sample_slots=..., decode_slots=...)` commits exactly
one group bounded by the prepared row workspace. Each selected slot occurs once;
sample/decode slots must be subsets; decode slots must each append one token.
An empty sample list commits a partial prefill without computing its vocabulary
head or copying full logits to the CPU. Otherwise it returns greedy token IDs in
the requested sample-slot order. Python does no tensor execution.

The additive `venturi_qwen_batch_step` C export preserves the legacy blocks ABI
and fixed-cohort API. Only this method explicitly rejects old libraries lacking
the optional symbol. MTP batch handles are rejected before native access; they
are a different C++ object and cannot be reinterpreted as ordinary batches.
Execution failures poison participating native slots; invalid metadata does not
change committed state. Existing group math/kernels and weight bytes are reused.

## Qualification status (2026-10-29)

The initial WZU two-V100 remote build passed. The 22 selected native FP16 cases
passed without skips, including old cohort regressions, mixed-step private-state
checks and continuous completion/refill against scalar output/cache snapshots.
Continuous fixtures cover single/TP FP16, residual-A8, column-MLP and head-GDN
profiles, and no Python tensor math/allocation fallback. CPU contracts include
random arrivals, lengths, cancellations, queue/capacity admission and failures.

These initial tests are tiny correctness fixtures, **not real 27B TPS, quality or
performance qualification**. The latest local CPU regression run has 1780 passed
and 2635 opt-in tests skipped (full-cpu-v5); skipped GPU tests are not claimed as
passing. Worker/HTTP CPU contracts pass, but their newly added GPU cases require
separate qualification of the frozen candidate-v2 source. Matched full-model
throughput, CPU, latency and separate GPU profiling remain night-goal work under
`operations/continuous-batch-night-20261028-v1/`. No production default was changed.

### Initial real-model scheduling result

Frozen candidate-v1 passed real Qwen3.8-27B Q8 token and private GDN/KV state
comparisons, including the 24-request staggered trace. Three alternating native
replay repeats on the same two V100s (16 prompt / 32 output tokens for fixed
counts, FP16 KV, ordinary decoding without MTP) produced these median aggregate
complete-work rates:

| Requests | Old cohort TPS | Continuous v1 TPS |
| --- | ---: | ---: |
| 1 | 31.08 | 26.89 |
| 2 | 52.37 | 46.35 |
| 4 | 77.47 | 68.56 |
| 8 | 103.12 | 94.06 |
| Mixed 24-request trace | 20.99 | 22.45 |

The mixed trace gained 6.93% throughput and reduced P95 response latency 8.63%,
but fixed-count throughput regressed and the predeclared promotion screen
**failed**. This is not an HTTP, streaming-TTFT, MTP or external-engine speed
comparison. No production default changed; the original model container and
watchdog were restored with identical configuration. Full metrics, workload
lengths, repetitions, CPU and sampled GPU telemetry are in
`operations/continuous-batch-night-20261028-v1/evidence/r2-three-repeat-analysis.json`.
The separate NCU child failed its Python dependency import before collecting
counters; achieved occupancy/bandwidth are not available from that failed run.

A follow-on **opt-in** `metadata_mode="sealed"` prototype separates an immutable
loaded descriptor from the mutable checkpoint reader, avoiding per-step whole-
directory JSON serialization while keeping file/native source guards. Default
`checked-json` behavior remains available and unchanged. The sealed path and
worker/HTTP adapter still require their own native/real-model qualification; CPU
contracts or a source upload do not establish a speedup.

The frozen candidate-v4 follow-on subsequently passed **42 actual native tiny
fixture tests, zero skips**, covering checked/sealed metadata, worker-thread
device ownership, private state and MTP binding/source guards. This does not
replace the pending trained 27B sealed-mode/HTTP qualification or establish its
TPS. The original production model stayed running throughout those tiny tests.

## Qualified optional path: 2026-10-29 night result

The corrected **candidate-v4** runtime passed trained-model token and private
GDN/KV byte comparisons, the same 24-request trace, and three alternating
unprofiled comparisons against the original native cohort. Only the candidate
uses `metadata_mode="sealed"`; CUDA inputs, kernels, weight bytes and projection
precision are unchanged. `checked-json` remains the constructor default.

| Requests | Old native cohort TPS | Continuous + sealed TPS | Change |
| --- | ---: | ---: | ---: |
| 1 | 30.94 | 31.02 | +0.25% |
| 2 | 52.17 | 52.27 | +0.19% |
| 4 | 77.39 | 75.78 | -2.09% |
| 8 | 102.79 | 100.49 | -2.24% |
| Mixed 24-request trace | 20.93 | 23.35 | +11.59% |

Mixed P95 complete-response latency fell **16.54 → 14.48 seconds (-12.46%)**.
The predeclared screen passed: C1/C8 within 3% of control; mixed P95 at least 10%
lower and no more than 5% throughput loss. These are three-repeat descriptive
medians, not statistical confidence or a claim that every workload is faster.
The retained C4/C8 regression is real. Nothing was changed in production defaults.

Conditions: same WZU two V100-PCIE-32GB GPUs, original-derived Qwen3.8-27B Q8
(group32/FP16 scales), residual-A8 + column-MLP + head-GDN profile, FP16 KV,
32 prepared token rows, chunk cap 16, eight private static-capacity-512 slots.
Fixed cases use eight distinct pinned 16-token prompts, selecting the first
1/2/4/8, with 32 outputs each; the mixed trace has 24 arrivals at recorded
0..0.84 seconds, prompt lengths 16/128/16/256 and output budgets 8/32/4/16.
Both replay arms have zero artificial admission sleep. There is no MTP, paging,
prefix sharing or precision downgrade in this comparison. Load/preparation and
profiler runs are outside scored generation. All rates divide emitted tokens by
a shared first-arrival-to-last-response interval, never summed request times.

The actual continuous-worker loopback server also passed three fixed-work HTTP
trials per count with pinned distinct outputs: about **30.85/51.01/76.09/99.76
aggregate TPS** for 1/2/4/8. A one-token A returned while 32-token B remained
in flight, then a 256-token prompt C reused A's logical lane with correct output.
That is separate **unpaired HTTP qualification**, not an old/new HTTP speedup
claim or HTTP token streaming. Its temporary server was stopped after testing.

### What changed and what is still expensive

The original model-directory guard reserialized/hash-checked 347,236 bytes of
metadata at every public step. The real-directory CPU-only attribution measured
about **5.25 ms/check**, versus **0.0051 ms/check** for the immutable capsule.
This microbenchmark created no model and used no GPU: it is not a 1000× model
speedup. File identity/size/mtime/ctime and native mapping guards remain checked.
MTP cross-artifact binding explicitly reopens a digest-matched detached reader.

Separate C8/4-output Nsight traces show the continuous iteration's median
NVTX-prefix-before-first-device-activity fall **5.36 → 0.278 ms**. This prefix
includes planning/validation/launch work, so CPU attribution is needed to infer
its cause. Per-device kernel/memcpy/memset intervals were unioned rather than
summing concurrent GPU time. Those timeline fractions are not achieved occupancy.

Actual hardware counters were collected with existing sudo access and explicit
process-only dependency paths; no driver/clock/cache-control policy, permissions,
firmware or registers were changed. The first profile sampled two **prefill**
projection kernels (32 rows, device 1): achieved occupancy about 36%, read+write
DRAM 515–610 decimal GB/s. A separate **decode batch-8** step at context 16
sampled one row-pair kernel per V100: achieved occupancy 34.82–35.01%, DRAM
445–455 GB/s, 79 registers/thread, long-scoreboard stalls about 37% per active
warp. These are selected-kernel samples, not whole-model occupancy/bandwidth;
clocks/cache were intentionally uncontrolled and profiler time never enters TPS.
The C1 dispatch uses a different global-dot kernel, not row-pair; an empty C1
row-pair filter was preserved as missing data rather than reported as counters.

Evidence: `operations/continuous-batch-night-20261028-v1/`, especially
`evidence/r3-three-repeat-analysis-with-nvml.json`, `evidence/r3-http-summary.json`,
`evidence/final-cpu-diagnostics-v1/`, and `evidence/r4-decode8-counters.json`.
The initial failed R2 screen and the failed profiler-import/empty-filter attempts
remain archived. Current local regression: **1780 passed, 2635 skipped**; selected
actual SM70/FP16 native checks: **42 passed, zero skips**. Skips are not passes.

### Limits

This qualifies the explicit ordinary native Qwen Q8 path, not mixed-Q4 full-model
loading, Flash-Next, DeepSeek, Metal/CPU native Qwen, PP/MTP continuous scheduling,
CUDA graphs, OpenAI-compatible production serving or 256K context performance.
It is not a comparison against vLLM/SGLang/llama.cpp, nor against the separate
Uncensored GGUF production model's stronger MTP configuration. Baseline TTFT is
call-start plus native relative timing (an estimate), while candidate token
availability comes from actual step events; neither is HTTP streaming TTFT.
Longer contexts, natural-length arrivals and deployment QoS need new matched
measurements. Keep the path opt-in rather than turning these results into a
universal dispatch/default rule.

The C1 follow-up selected the actual `vqf::global_dot` decode kernel, not the
nonexistent C1 row-pair launch. Two hardware-counter samples were obtained and
the profiled token IDs plus all canonical private GDN/KV bytes matched ordinary
generation. See `evidence/r5-decode1-counters.json`. This later run followed a
verified host reboot; its individual kernel samples are not a paired C1/C8
speedup experiment. All profiler clocks/cache controls remained `none`.

To use the qualified Python path, explicitly load the target with
`NativeQwenModel.from_q8(..., metadata_mode="sealed")` before preparing its
ordinary batch and starting `ContinuousWorker`. Omitting that argument keeps the
legacy `checked-json` default and does not claim the corrected per-step cost.

Final C1 decode samples (one per V100, global-dot rather than row-pair) show
41.31–41.34% achieved occupancy, 718–732 GB/s read+write DRAM, and 88–90%
reported DRAM throughput. Long-scoreboard stall percentages are high in these
samples. This suggests memory traffic/dependencies remain important for single
request projection; it does not prove whole-model bandwidth saturation or imply
that higher theoretical occupancy alone will raise TPS. No new kernel arithmetic
was introduced in this night task.

Final restoration audit verified the same original production container/config,
health200, the exact original watchdog crontab, no armed owned restore timers or
active owned services, and no owned GPU compute processes. Only the original
production llama-server remains on both GPUs. The source changes remain local
and opt-in; no GitHub push or production model/routing replacement was made.

## 2026-10-29: short-cohort gate qualification (not a new default)

The separate `r2` campaign compares the original round-robin scheduler with
explicit `balanced-short`, not with the unrestricted balancing experiment.
Both use the same original native Q8 artifact (SHA-256
`41f99992b679aef7df2b6c671b76b09820730846c2c95a52406e7004c72f58f0`),
two V100-PCIE-32GB GPUs, resident host-transport TP,
`q8-residual-column-mlp-head-gdn`, sealed metadata with source-file guards,
F16 KV/values, residual-A8 projections and FP32 GDN. There are eight prepared
512-token slots, a 32-row budget, chunk cap 16, no graphs/MTP/admission sleep.
All 29 compiled C++/CUDA inputs are unchanged; the same pinned SM70 library is
reused, not rebuilt with different arithmetic or flags.

Three alternating matched, **unprofiled** fresh-process repeats per arm, each
with its own identical warmup, give these medians. Fixed-count cases use distinct
16-token prompts and 32 outputs/request. The mixed case retains the same 24
planned arrivals, 16/128/256-token prompts and 360 total output tokens.

| Case | Original aggregate TPS | Gated aggregate TPS | Change |
| --- | ---: | ---: | ---: |
| 1 request | 31.086 | 31.115 | +0.09% (essentially unchanged) |
| 2 requests | 52.381 | 52.418 | +0.07% (essentially unchanged) |
| 4 requests | 75.806 | 77.652 | +2.44% |
| 8 requests | 100.667 | 102.937 | +2.25% |
| Mixed arrivals | 23.392 | 23.399 | +0.03% (essentially unchanged) |

Mixed response P95 is 14.4510/14.4471 s; its earlier 2.45% unrestricted-balancing
regression is not reproduced by this gate. These are only three repeats of this
trace, not confidence intervals or a guarantee for every workload. This is native
replay throughput, **not** paired HTTP/streaming speed, a strongest-MTP comparison,
or a production Uncensored-GGUF benchmark. Load/preparation is reported separately;
filesystem caches are warm, with no cache drops or cold-storage claim.

For fixed count 8, both arms consume 376 input rows and emit the same 256 tokens.
Native group iterations fall from 36 to 35, with token-emitting step groups
36 to 32. Short-prompt progress stays aligned instead of creating early decodes
and a small late decode tail. Not every kernel's launch count decreases: the
changed group/segment shapes also change projection dispatch and GDN launches.

There is a real **earliest-TTFT tradeoff**: the first count-8 token event moves
from a median 0.1557 s to 0.6234 s (count 4: 0.1533 to 0.3063 s), although the last
prefill finishes sooner. Thus use `--continuous-target --prefill-policy
balanced-short` only as an explicit bulk-cohort option; default round-robin and
production routing remain unchanged. The gate adds no wait to collect requests.

### CPU, GPU and correctness evidence

- Fixed count 8 process CPU: 2.5452/2.4899 core-seconds per 256 outputs, both about
  one active CPU core. This is reduced CPU work for the same completion, not an
  idle-CPU claim. Peak process RSS is recorded separately, including mapped weights.
- Sampled count-8 VRAM is unchanged at 16,304/14,400 MiB on GPU0/GPU1. GPU busy is
  about 83%/73%, SM/memory clocks 1380/877 MHz, power about 151/159 W and maximum
  temperature 44 C. Those are 200 ms NVML samples, not achieved occupancy/bandwidth.
- Separate Nsight timelines use **four**, not 32, outputs/request: eight/seven
  iteration ranges, median host prefix 0.282/0.267 ms. GPU timeline activity unions
  are about 80%/76%; these are not occupancy measurements or scored TPS.
- Two separately replayed first-prefill projection kernels on **GPU1**, with
  32 input rows, report achieved SM occupancy 36.37/35.89% before versus
  36.24/35.92% after, and read+write DRAM bandwidth 605.7/513.8 versus
  609.6/523.3 decimal GB/s. Registers remain 79/thread. These are individual
  prefill kernels, not decode8 or whole-model counters; caches/clocks are
  uncontrolled and the warnings are retained. There is no evidence here of a
  faster projection kernel: the measured improvement is scheduling/dispatch.
- Local regression: **1,806 passed, 2,743 skipped** (GPU tests skipped locally).
  Explicit V100 native qualification: **96 passed, zero skipped**, including
  row-scarcity/no-early-sample trajectories, ordinary/Q8 single/TP profiles,
  immutable metadata and worker ownership. SM70 F16 cases only; other dtypes and
  devices are not qualified by this count.
- Real 27B fixed/mixed greedy IDs and complete canonical private KV/GDN state
  signatures match the pinned native reference exactly before timing. This is
  trajectory parity at unchanged Q8 precision, not a new BF16/perplexity evaluation.

Complete timing/CPU/latency/RAM records, aligned NVML samples, kernel attribution,
timeline gaps, hardware counters, source/library hashes, failed first experiment
and restoration proofs are retained in `operations/prefill-balance-20261029-v1/`.
The original production container/configuration/watchdog were restored after both
leases; no model default, driver, permission or clock policy changed. Nothing was
pushed to GitHub automatically.
