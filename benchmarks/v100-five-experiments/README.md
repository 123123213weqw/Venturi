# Five native V100 experiments — Qwen3.8-27B

Completed 2026-10-22/23 Asia/Shanghai. **Production restored, unchanged.**
All results are native-to-native controls, not a new llama.cpp superiority claim.

## Fixed case

- Same WZU_Server pair of Tesla V100-PCIE-32GB (SM70), original device UUIDs.
- Official Qwen3.8-27B revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Native Q8 group32/FP16-scale target `db1c60fea76c415abc892cf6c0eed6c9`;
  SHA256 `41f99992b679aef7df2b6c671b76b09820730846c2c95a52406e7004c72f58f0`.
- FP16 activations/KV, residual two-plane A8 projections, FP32 GDN and TP
  partials. Host-staged two-device TP; no paging, HTTP, sampling or MTP in the
  first table. Row-pair, SwiGLU-quant and RMSNorm-quant enabled in **both** arms.
- Eight pinned heterogeneous16-token prompts;32 fixed output tokens, EOS disabled,
  capacity256/request, projection rows32. C1 uses the first prompt; C8 all eight.
- Three balanced-order repeats. Warmup/reset/loading outside scored ordinary
  calls. Graph instantiation is warmed, **not** a cold-start result.
- Post-first decode counts31*C output tokens/global wall. Complete cohort counts
  32*C/global wall. Never sum concurrent request durations. CPU time is separate.

## Four independent switches and their combination

`experimental_flags` is immutable at creation, default0. Same binary/source for
all masks; no per-trial rebuild or precision change.

| Mask | Change | C1 decode TPS | Gain | C8 aggregate decode TPS | Gain |
|---:|---|---:|---:|---:|---:|
|0|RMS-on control|32.84239|—|133.32690|—|
|1|Cross-request convolution/GDN kernels|32.81194|-0.09%|141.04359|+5.79%|
|2|FP32 A/B dots + gates, FP16 boundaries retained|33.50936|+2.03%|137.44978|+3.09%*|
|4|Cached rank-local head-GDN CUDA Graphs|32.92393|+0.25%|132.47784|-0.64%|
|8|Earlier primary GDN/MLP submission|32.75136|-0.28%|133.09472|-0.17%|
|15|All four|33.56177|+2.19%|142.16623|+6.63%|

*Mask2 C8 is **not a stable3% gain**: the three paired changes are
+0.88%, +1.06%, +7.61%; raw TPS134.622/134.676/143.416. Keep the fast trial,
do not silently discard it or attribute its cause. Sampled clocks remain
1380/877MHz and temperature39–40C; these observations do not explain the outlier.
C1 mask2 is consistent (+1.89% to+2.17%).

Mask1 C8 gains are +5.65/+5.86/+5.85%; mask15 gains +6.58/+6.73/+6.58%.
Complete-cohort control→mask15 is C1 **31.10836→31.82863 TPS (+2.32%)** and
C8 **103.43888→108.62908 (+5.02%)**. TTFT, process CPU seconds/cores and every
individual trial are in `summary.json`; all raw reports are under `evidence/window`.
Process CPU utilization remains approximately one core: fewer launches do not
establish that the process was CPU-bound or that CPU utilization disappeared.

**Decision:** GDN batching and A/B fusion are useful research paths. Graphs and
earlier-primary scheduling do not merit default-on promotion from these results.
The all-four result is measured, but is not proof that every constituent helps.
No default was changed, and no unmeasured combination is advertised as faster.

## Fifth experiment: Qwen3.8 fixed/adaptive MTP

A fresh MTP artifact was exported from the **same verified original source**,
not the old Qwen3.5 draft. SHA256
`f8069a9db66373bc9b8ccdb16c6fbea4f3f93e8521bd8a2c3b0c3bc59f430859`.
Independent CPU verification checks the source/artifact hashes, all floating
payloads and independently quantized sampled rows. Verifier now supports
explicit `component="mtp"` and an external source-manifest path.

These TPS include the **whole fresh adaptive-adapter call**, including reset,
policy, draft, verification and metadata. They are not post-first TPS and not
HTTP serving TPS. Target mask0 is fixed throughout this separate experiment.

| Mode | C1 whole-call TPS | C8 aggregate whole-call TPS |
|---|---:|---:|
|Ordinary target|30.73144|105.51494|
|Fixed MTP1|44.56588|96.02147|
|Fixed MTP2|46.46509|Unsupported, not serialized|
|Adaptive|46.55114|105.95389|

After labelled calibration, adaptive chooses MTP2 at C1 and ordinary at C8 in
all three measured repeats. C1 improvement versus its own ordinary control is
about51.5%; the0.4% C8 auto/control difference is not promoted as a speedup.
MTP1 acceptance is82.35% at C1 and83.46% at C8, yet C8 MTP1 is slower. MTP2
proposal acceptance is65.38%. **Acceptance is not net throughput.** Cold online
exploration costs7.605s across the two tested buckets; calibration/probing costs
are preserved separately, not hidden inside a claimed free adaptive decision.
Peak sampled MTP VRAM with all banks prepared is18,709/16,229MiB, versus ordinary
C8 approximately16,167/14,501MiB. Banks share weights but own extra private state.

## Correctness and reproducibility

- Local CPU/source/API/evidence suite:95 passed,208 CUDA-dependent cases skipped.
  This includes three parser regressions. No CUDA/Rust compilation locally.
- Latest remote build/fixtures:10/10 pass, including reset/reorder/EOS/fault
  recovery, exact bytes at1/2/8 slots, graph replay/lifetime, and closing another
  cohort while decoding. Extra frozen replay/EOS/fault fixture run:8/8 pass.
- Actual27B all five candidate masks pass **byte-exact** teacher logits,
  per-layer hidden traces, private GDN/KV states and32-token trajectories versus
  control. All checked floating outputs/states finite; first two scalar/cohort
  trajectories agree. Control payload also exactly matches prior qualified
  RMS-on Qwen3.8 baseline (`baseline-previous-equality.json`).
- All MTP strategies and adaptive calls match ordinary greedy IDs and committed
  private target-state hashes, including each measured call.
- Executed16 CUDA inputs and319 executed src/tests/tools files match local
  sources. SO SHA256 `72a32604f9ee19781773a40880ca3eb969f4d77333cfbc1c6a4b3f25cc141eb5`.
- All2,102 pinned files across five older capsules remain unchanged.
-262 raw evidence members locally downloaded and individually SHA256 verified;
  complete archive SHA256
  `4bbc8f13650ea38ca0f54c5bbe18fbef346a35df53fc872591b8ef1ef0fb528d`.
  Shared libraries/original `.nsys-rep` remain remotely hash-pinned; raw SQLite,
  timing reports, NCU CSVs, failures and restoration records are local.

## Separate diagnostics, not scored timings

Real C8 operator counters on V100, one replay-profiled launch each:

|Kernel|Work/launch|Duration us|Achieved occupancy|SM throughput|DRAM throughput|Measured DRAM R+W GB/s|
|---|---:|---:|---:|---:|---:|---:|
|old conv|1 request|6.912|12.48%|4.26%|0.31%|1.32|
|batched conv|8 requests|6.496|24.60%|16.57%|0.76%|3.61|
|old compact GDN|1 request|19.712|53.56%|62.48%|4.37%|34.88|
|batched GDN|8 requests|98.016|92.80%|80.35%|25.73%|225.81|
|old FP32 single A/B projection|8 tokens, one matrix|12.480|29.45%|16.27%|7.35%|42.94|
|fused A+B+gates|8 tokens, both matrices|14.976|28.59%|15.39%|11.68%|69.49|

Compare equal work: eight old GDN launches total157.696us versus98.016us;
two old A/B projections total24.960us, before their separate gate launch, versus
14.976us fused. Replay/cache/clock warnings are preserved. Clocks/caches were not
forced, profiling policy unchanged. These are **kernel** measurements, not whole
model occupancy or whole-model HBM bandwidth. Cache and scoreboard metrics are
in `counters.json`; NCU's wide CSV unit row is parsed explicitly.

Nsight default graph-level tracing omitted graph nodes. Original traces are
preserved, **not** misreported as eliminated kernels. A separate drained/restored
window explicitly captures `--cuda-graph-trace=node` for both arms. Four post-first
C8 steps: **14,112→7,968 kernels**, **19,550→10,718 CUDA API calls**; H2D/D2H
transfer work is unchanged. Kernel-duration sum across both GPUs310.584→291.767ms
is overlapping device time, not wall TPS. Row-pair Q8 projection families still
account for~63.8% of candidate kernel-duration sum. Per-device traced activity gaps
actually increase52.46→55.22ms and72.46→76.22ms; they are this process's trace
gaps, **not** measured device-wide idle or proof of CPU causality. Instrumentation
perturbs scheduling; profiler wall times are never scored.

NVML sampled busy/VRAM/power/clocks/temperature inside scored calls are in
`telemetry.json`. NVML busy is not achieved occupancy. The all-four graph path
adds approximately6–12MiB/GPU in these sampled runs; static tensor scratch is
unchanged. Neither high occupancy nor a large launch-count reduction implies a
proportional model speedup.

## Restoration, limitations and next work

Both maintenance windows drain requests, capture exact container ID/config,
arm independent restoration timers and restore the same container. Final fresh
HTTP smoke returns200 and `OK`; original Qwen3.8-Uncensored llama.cpp production
model/config fingerprint unchanged. All own workers and timers inactive. The old
own abandoned RMS runtime mask is removed without starting its unit.

Initial network outages and two CPU preflight verifier failures are retained.
Recovery preserves host keys and hardware identity; no alternate GPU, local CUDA
build, local Rust build, source/precision substitution or production deployment.

This is an eight-prompt, short-context greedy qualification, not256K, broad corpus
quality, stochastic decoding or arbitrary-model qualification. The older external
llama.cpp goal is not completed by native-only wins. Next research idea, **not
implemented or measured here**: compact GDN repeats the same Q/K normalization
for3 value heads ×128 value rows per Q/K head. Reusing byte-equivalent FP32 norm
scalars could remove redundant reductions; it needs its own correctness/speed
experiment. Projection kernels remain the larger measured GPU-time component.
