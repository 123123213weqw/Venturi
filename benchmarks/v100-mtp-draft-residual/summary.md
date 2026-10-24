# Residual-A8 MTP draft: qualified opt-in, modest complete-engine gain

2026-10-20. **Single post-first decode47.657 ->48.409 TPS (+1.58%).** This
implements and measures a real native MTP draft profile, not only an operator.
The full external goal is still incomplete: stronger pinned llama.cpp49.803 TPS,
required52.293 TPS, another **8.02%** over the new native result. Existing stronger
ordinary batching and external controls remain in acceptance. No automatic router.

## Implementation and precision

New optional ABI16 `venturi_qwen_mtp_create_profiled` and projection-info exports
select two group32 residual-A8 activation passes with FP32 scales for the existing
native FC/attention/MLP draft body. Shared Q8 vocabulary/head retain the target
profile. Existing original typed FP16/BF16 constructor/default, old-library default
binding compatibility and descriptor sizes are unchanged. Missing optional exports
fail explicit-profile preflight, never silently fall back.

Scope: resident FP16, SM70, non-PP target; the body remains on one target device,
not draft TP or packed drafting. Target is the previously qualified column/row
MLP TP profile; target attention/GDN remain row TP. Explicit CLI:
`--mtp-draft-projection-profile q8-residual`; API:
`NativeQwenMTPModel.from_q8(target, weights, draft_projection_profile="q8-residual")`.
The benchmark, qualifiers, profilers and actual public CLI carry the same option.

Stored Q8 codes/FP16 scales and FP16 activation/KV **storage dtype** are unchanged.
Arithmetic and numerical draft-cache values can differ from FP16; no old-profile
byte-parity or original-BF16/corpus-quality claim. Corrected-draft byte equality
is checked **within the new profile**. Target F32 GDN state and F32 MLP partial/
host transport remain unchanged. Native Q8 encodings/small parameters/arithmetic
are not identical to external GGUF Q8_0/Q8_1; prior original-source audits apply.

Official parent revision `fc05daec18b0a78c049392ed2e771dde82bdf654`; target artifact
`3d635b87db004f42aefd6730e372e530`, draft `1a4307f5e7954a929d7ad70539875915`.
Candidate library SHA256:
`6d43160ce4e993b4be5ed600d9be21aad070634729d8e5f7b3f74b734da85de2`.
Ten native build dependencies, compiled only on WZU_Server. The rejected standalone
DP2A pilot is not included in production model dispatch or this arithmetic.

## Correctness before speed

- New native integration subset **32 passed**; full remote **2,236 passed /
  687 skipped**; final local **818 passed /2,105 skipped**. Ruff passes. Original
  TP/PP/MTP paths remain covered. No local Rust/CUDA compilation.
- Fixtures independently check source-equation draft hidden/logits/incremental
  KV, original default FP16 bytes vs the qualified old SO, prefix forced accept/
  first-or-second rejection, EOS/budgets, corrected draft bytes, failure/reset/
  closed owners, private2/8 and cohort2/8, ragged/selective reset, overlapping
  target/scalar/cohort owners and native generation with Torch math disabled.
- Fresh trained original-source FP64 represented-Q8/FP16 reference: two block
  chunks plus three recursive-hidden steps **for each draft profile**. New
  maximum relative L2 **0.000776159**, minimum centered cosine **0.999999699**;
  control0.000759693/0.999999711. Fixed limits remain L2<=0.02/cosine>=0.999.
  This separately checks the draft; target repair alone is not quality evidence.
  Shared target head retains its prior qualification, not a new independent full
  CPU-vocabulary or corpus evaluation.
- Four trained story/capital/Python/arithmetic MTP2-prefix trajectories match
  ordinary target output and final private-state pins. Same-new-profile MTP1 and
  MTP2 corrected draft KV match exactly. Mixed-eight ragged/EOS and full-eight
  MTP1 cohorts plus two-slot selective reset preserve target/draft identities.
- Unchanged target teacher/98-logit quality evidence is SHA-pinned and reused,
  not an invented fresh teacher campaign. Fresh ordinary target trajectories and
  committed private states match it. Default trained FP16 drafting is freshly
  regressed before candidate timing.
- Five additional generated-fixture guard checks reject changed draft metadata/
  source before scalar/cohort native generation, and a direct native call rejects
  changed source without relying on SDK validation. No trained payload is modified.

## Fixed unprofiled work and denominators

Same two V100-PCIE-32GB GPUs; logical0/1=physical1/0, explicit pinned-host TP.
Same source artifacts,23 formatted prompt IDs,32 greedy outputs/request, ignore
EOS, capacity256/slot, eight prepared slots, projection row budget32 for both MTP
modes. Same SO/SDK for control and candidate; only explicit draft profile differs.
One warmup/cell precedes timing; allocation/loading/profiling are excluded.

Complete HTTP TPS is32*requests/mean global cohort wall, including reset/admission/
queueing/reply/client archival/validation/future collection. Native TTFT/generation
exclude those. Single post-first is31/mean(native generation-native TTFT);
external is31/mean(predicted_ms/1000). These internal timer boundaries are not
identical, nor streaming client TTFT. Never sum concurrent request durations.

Both changed paths pass predeclared C1>1% or C8>2% screens. Each then gets three
unprofiled repeats/arm at1/2/4/8. All8 repeated native cells improve, none loses>2%,
and every paired C1/C8 trial improves. Scalar C8's~1.74% gain misses the separate
2% C8 trigger, but its qualifying C1 gain permits adoption; do not relabel it a
passed C8 trigger. Original precision/defaults stay unchanged and the new path
remains explicit opt-in.

| Requests | Scalar MTP2 HTTP FP16 -> residual draft | MTP1 cohort HTTP FP16 -> residual draft |
|---:|---:|---:|
|1|37.687 ->38.346 (+1.75%)|35.721 ->36.366 (+1.81%)|
|2|36.804 ->37.635 (+2.26%)|50.562 ->51.821 (+2.49%)|
|4|37.352 ->38.104 (+2.01%)|61.738 ->63.660 (+3.11%)|
|8|37.624 ->38.279 (+1.74%)|66.597 ->68.911 (+3.47%)|

Single post-first FP16 trials47.677/47.677/47.616 versus new
48.427/48.352/48.447 TPS. Mean-duration-derived rates47.6568 ->48.4089, not means
of per-trial TPS. Story MTP2 still accepts17/26, conditional depths11/13 and6/11,
13 three-input verifications,63 target inputs, zero replay/forced proposals.
Four-case acceptance is unchanged:17/26,1/2,20/21,19/22. MTP1 still14/16/request.
The gain is not from forcing proposals, omitting verifier rows or better acceptance.

### Stronger strategies and external controls

No external engine run is repeated this turn. Stronger original-source pinned
three-repeat controls are explicitly reused, including best-tested external MTP2
C1 and supported non-speculative parallel configurations. Earlier weaker
depth1/3/4 and failed parallel MTP results remain preserved; failures are not
zero-TPS baselines. This is not an exhaustive competitor configuration search.

| Requests | Best measured native strategy | Native HTTP TPS | Stronger retained external HTTP TPS | Ratio |
|---:|---|---:|---:|---:|
|1|New residual-draft scalar MTP2|38.346|30.173, MTP2|1.271x|
|2|New residual-draft MTP1 cohort|51.821|37.907, non-speculative|1.367x|
|4|New residual-draft MTP1 cohort|63.660|50.254, non-speculative|1.267x|
|8|**Prior retained** ordinary256|73.836|67.287, non-speculative|1.097x|

The C8 ordinary rate is **not freshly remeasured with this new SO**. Its unaffected
target algorithm and stronger qualified result are retained; do not replace it
with the slower fresh68.911 MTP1 rate (only1.024x external). These are separately
selected strategies, not an automatic shared-cache workload router. All known
HTTP floors are exceeded, but single decode still trails49.803 and misses52.293;
the full goal is incomplete and needs a final current-state audit before completion.

## CPU, latency, VRAM and telemetry

Times are seconds; CPU is process core-seconds per complete cohort. Rows below
show newly selected scalar C1 / MTP1 C2/C4/C8 (fresh MTP C8, not retained ordinary).
Complete per-arm/count/trial data and min/mean/max NVML are in `metrics.json`.

| Requests / new strategy | HTTP wall | CPU core-s | Client completion mean / max | Native TTFT mean |
|---|---:|---:|---:|---:|
|1 / scalar MTP2|0.8345|0.8333|0.8336 /0.8410|0.15528|
|2 / MTP1 cohort|1.2350|1.2100|1.2335 /1.2364|0.23307|
|4 / MTP1 cohort|2.0107|1.9833|2.0078 /2.0111|0.38803|
|8 / MTP1 cohort|3.7150|3.7133|3.7098 /3.7187|0.69958|

C1 matched CPU0.8500 ->0.8333; C8 matched MTP1 CPU3.8400 ->3.7133 core-s.
Scalar internal TTFT averages regress atC4 (0.55499 ->0.57816) andC8
(0.95445 ->0.97890), despite faster cohort completion. Interleaved request
scheduling/staggered warmups affect these averages; no fairness or streaming-TTFT
improvement is claimed. MTP1 cohort TTFT improves at every measured count.

NVML100ms samples, physicalGPU1/GPU0: new scalar C1 uses23995/17229MiB, mean
busy78.0/55.4%, power144.5/115.6W, maximum temperatures42/40C. New MTP1 C8 uses
22461/17229MiB, mean busy85.0/58.7%, power175.3/134.7W, maxima50/45C. Used memory
includes unrelated3189/3651MiB allocations; busy and memory-controller busy are
not achieved occupancy/DRAM bandwidth. Clocks/temperature/power ranges and sparse
sample counts are retained rather than condensed into an occupancy claim.

Native accounting explicitly adds **1,253,376 bytes** of primary draft projection
workspace at row budget32 (5,013,504 at the128-row qualification budget). Body
model bytes455,514,240 ->456,767,616 at32. No secondary draft workspace, weight
requantization or duplicated vocabulary/head. Scalar NVML grows2MiB on primary;
MTP1 peak is unchanged at MiB resolution despite nonzero logical cost. Do not
interpret allocator/residency granularity as zero memory overhead.

## Separate profiles: remaining target verification dominates

Same-library C1 profiles have the same17/26 acceptance and fixed work. Profiler
runs are separate from unprofiled TPS. Host span863.658 ->844.345ms; kernels
28946 ->29036 (+90), summed kernel time980.006 ->961.242ms. Both retain
**7102 H2D/5883 D2H calls,316,660,240/316,660,272 bytes**. Cross-device kernel
overlap330.641 ->330.507ms is essentially unchanged. Process-only activity gaps
125.810 ->125.711ms are not device-wide idle or pure CPU overhead.

Draft accounting80.519 ->61.411ms includes warmup/catch-up; prefill165.576 ->
157.723ms. Verification574.075 ->574.348ms is unchanged. Do not attribute all
19ms draft-accounting reduction to post-first decoding. Added quantization/fused
draft kernels replace eight typed projections with five quantizers plus five
projections per body call, explaining90 extra launches across45 calls.

Remaining C1 summed target work: fused T4-tile projections327.118ms, FP32 partials
T4-tile122.359ms, remaining row T4-tile72.399ms, GDN-prefix~41ms. These are summed
kernel durations across devices, not critical-path percentages or additive savings.
Fresh MTP1 C8 profile host span3.7596s,73304 kernels, summed4.6520s; full attribution
is retained. Its one NVTX native-call range counts eight logical requests.

Exact compiled-resource inspection is separate, never a hardware-counter replay.
`RmProfilingAdminOnly:1` is unchanged. Achieved occupancy, measured DRAM bandwidth,
cache/stall counters remain unavailable; no privilege changes or repeated denied
campaign. Existing kernels' theoretical residency is not achieved occupancy.
Nsight disables unsupported CPU IP/backtrace/context-switch sampling; CUDA/NVTX
capture/export and the actual public CLI smoke test succeed.

## Failures, preservation and next work

Initial local CLI mock tests reject the new default keyword; the default CLI
call flow is restored while explicit opt-in still forwards it. A local generated
guard dispatcher has a syntax error and launches no remote job. Guard-v5 then
fails because its generated fixture parent directory is missing, **before any
GPU model construction**. Guard-v6 creates that directory and passes all five
checks using the unchanged qualified library. Only the failed guard setup is
repaired; the full suite, trained qualification and timed matrices are not rerun.
All failed logs/sources and the initial terminal audit are preserved.

All38 frozen capsules are reverified without edits. Executed source snapshot,
build hashes, exact commands, payload identity/full-hash reuse and large remote
SO/SQLite/nsys pins are retained. The trained draft is freshly full-hashed by
qualifiers; the previously full-hashed target has unchanged dev/inode/size/mtime/
ctime and is explicitly reused, not rescanned. Six owned workers are terminal
(one preserved setup failure); no sampler, owned inference job or port18953
remains. Original GPU0/1 footprint3651/3189MiB is restored. No unrelated jobs,
driver settings, permanent service, local Rust/CUDA build, commit or push.

Next prioritize **target verification/dataflow and measured host work**, not
another draft-arithmetic or tile sweep. Source inspection confirms repeated
fd+path identity checks in resident `weight()`; measure their counts/untraced
cost before proposing bounded check deduplication, retaining all load/public/
pre-post identity guarantees and adversarial source-change tests. Separately
measure exact FP32 GDN A/B-plus-gate fusion or the previously proposed semantic
head-shard prerequisite. No check is removed, new fusion/head split implemented,
or future TPS gain promised by this capsule.
