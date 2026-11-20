# V100 shared-activation projection: retain T2 only

## Result

Retain shared activation staging **only for eligible SM70 two-token residual
projections**. Three-repeat matched engine tests improve C2 complete-request
HTTP **28.552 -> 29.627 TPS (+3.77%)** and C1 MTP1 post-first decoding
**30.423 -> 31.734 TPS (+4.31%)**. One-, four- and eight-request non-speculative
throughput is roughly flat, with small measured regressions below 0.2%.

This is an internal improvement, **not** an external win. Reused pinned
llama.cpp controls are still stronger: C2 non-spec HTTP 37.907, C8 67.287;
best tested C1 MTP depth 2 has HTTP 30.173 and reported post-first 49.803 TPS.
The >=5% external goal is incomplete. No commit/push, local Rust/CUDA build,
clock change or driver privilege change.

## Implementation and arithmetic

`q8_shared.cuh` is an SM70 research operator. Each CTA stages the current
quantized activation's two int4 halves into separate shared-memory planes,
alongside unchanged FP32 group scales. It then evaluates the original complete
row dot with the **same** integer products, compensated activation passes,
per-group FP32 FMA order and warp reduction. Weights/scales and output storage
are unchanged. This is not a new arithmetic profile or Tensor Core GEMM.

The operator supports T1/T2, row blocks 4/8/16 and at most 48 KiB activation
staging; no function attribute mutation or allocation occurs during run/graph
capture. Unsupported shapes fail explicitly in the standalone operator.
`qwen_runtime.cuh::Model::project` selects row-block 16 only when compiled for
SM70, activation passes=2, T=2 and the actual activation plane fits 48 KiB.
Large-K T2 falls back to the qualified kernel. T1/T4/T8, SM89, default FP16,
PP, peer gather and event fences are unchanged. Shared-linears' scoped packing
reuse and private request state are preserved. No persistent GPU buffer is
added; existing quantization scratch remains 1,253,376 bytes/device at 32 rows.

Native model selection stays in C++, not a Python token/model loop. The optional
Python research plan is not silently dispatched by ordinary `ResidentLinear`.
Blocks ABI remains 16; native build provenance now includes eight source files.

Control library SHA256:
`ccf03d8caa89ced089b80cd5b8eb49cb494e53baf855c78c99e9c8f5ed9627c5`.
Candidate:
`/home/data/wangyue/artifacts/venturi-shared-activation-model-v2/build/43feb26c8cde610e/libventuri_blocks.so`,
SHA256 `1aba1423dc511b7cb17f03c0a4b71948f5fdd4f836e0dcfe9e0190b15bddbee7`.

## Operator screen: T1 rejected, T2 selectively retained

Eight unique real TP-first shard shapes at T1/T2: 15 supported combinations,
plus one explicitly unsupported T2/K17408 activation plane. Six arms compare
original FP16 arithmetic, candidate/control residual DP4A, and shared row blocks
4/8/16. Three timed repeats per arm use nine graph-contained calls per wave,
after warmup; order reverses on alternate repeats. These are standalone GPU-event
microseconds, **not inference TPS**. All shared outputs/workspaces are exact to
the original residual-profile control, with independent sampled FP64 gates.

| Shared row block | T1 median speed vs qualified residual control | Supported T2 median speed |
|---:|---:|---:|
| 4 | .936x | 1.051x |
| 8 | .976x | 1.100x |
| 16 | .979x | 1.112x |

For the `[124160,5120]` output-head shard, T2 improves 1.0003 -> .8033 ms
(1.245x). The `[8704,5120]` MLP shard improves .0846 -> .0716 ms (1.182x).
These selected results do not establish a universal row-block rule or speedup
for unsupported K. T1 staging is slower and is **not enabled in the model**.
Raw per-shape trials, wall/CPU times and broad NVML samples are retained.
Sub-millisecond event windows do not provide reliable per-wave NVML utilization.

## Correctness before engine speed

- Operator selection: **142 passed** (50 new shared/preflight cases, 72 existing
  DP4A cases, 20 pinned variant cases). Covers both activation pass counts,
  tiny/odd/tail and wide-K shapes, output guards, byte equality, independent
  FP64 reconstruction, graph capture, alias and invalid-pointer/geometry errors.
- Operator-only/model initial full suites: **1,240 passed / 456 skipped**.
  Sixteen old scoped-input-sharing tests lacked their opt-in environment then;
  they are enabled and passed in the final suite, not counted as early passes.
- Both trained original `fp16` and residual profiles: 98 full-vocabulary vectors
  each, four original greedy trajectories, final scalar private states, eight
  heterogeneous batched requests and selected reset/continuation are byte-exact
  against the pinned qualified T8 control. No Torch math during generation.
- Fresh trained MTP qualification passes four cases with matching ordinary-target
  IDs and committed target state, natural acceptance/replay/EOS, no forced draft
  proposals. Only the unchanged draft block's independent CPU represented-Q8
  hidden/KV reference is reused: its original full reference and intermediate
  reference-chain hashes are explicitly pinned. The draft block keeps FP16
  arithmetic; its shared vocabulary head retains the target's unchanged T1
  profile. No new independent CPU shared-head/corpus-quality claim.
- Final two-V100 suite: **1,256 passed / 440 skipped**. Local non-GPU suite:
  **762 passed / 934 skipped**. All four owned remote workers finish with exit 0.

Independent prior CPU thresholds remain top-1, relative L2 <= .02, centered
cosine >= .999 and KL <= .002. Exact new/old bytes preserve those gates;
this is not a fresh original-BF16 or broad corpus-quality campaign.

## Matched engine matrix, three unprofiled repeats

Two V100-PCIE-32GB devices, logical `1,0`; explicit pinned-host TP; original-source
Q8 artifact; explicit residual profile, FP16 KV; 23 formatted prompt IDs and
32 fixed greedy outputs, ignore EOS for this speed case; no prompt-cache reuse;
eight slots at capacity 256, maximum projection rows 32; 25 ms batch admission
included in HTTP wall. Warmup, loading and diagnostics excluded from timed waves.
All emitted IDs match the pinned case; fixed cohorts remain unsplit. One-repeat
1/2/8 and MTP1 screens precede promotion. The stated promotion rule requires no
material C1/C8 regression and at least one targeted C2/MTP gain above 3%; that
internal rule does **not** replace the original external >=5% acceptance target.

| Requests | Control HTTP TPS | Shared-T2 HTTP TPS | Change | CPU core-seconds old -> new | Mean client completion seconds old -> new |
|---:|---:|---:|---:|---:|---:|
| 1 | 19.412 | 19.380 | -.16% | 1.623 -> 1.623 | 1.64786 -> 1.65047 |
| 2 | 28.552 | 29.627 | +3.77% | 2.213 -> 2.137 | 2.24047 -> 2.15926 |
| 4 | 35.521 | 35.518 | -.008% | 3.583 -> 3.580 | 3.60145 -> 3.60174 |
| 8 | 43.189 | 43.174 | -.034% | 5.927 -> 5.927 | 5.92326 -> 5.92516 |

HTTP aggregates are all outputs / **global** cohort wall, not summed request
latencies. C2 wall falls 2.24156 -> 2.16016 s; C1 is 1.64850 -> 1.65118 s,
C4 3.60354 -> 3.60382, C8 5.92744 -> 5.92947. C2 post-first native cohort
TPS is 37.210 -> 39.101, using 62 outputs / (last finish minus common first-token
timestamp); it is not HTTP or per-client TPS. C1 ordinary native post-first is
23.112 -> 23.065, not improved. Native C2 TTFT .53262 -> .53235 s is essentially
flat; native TTFT excludes prior HTTP/queue/reset and is not streaming client TTFT.

Single-request MTP1 HTTP improves **23.993 -> 24.801 TPS (+3.37%)**; post-first
native decode **30.423 -> 31.734 TPS (+4.31%)**; whole wall 1.33374 -> 1.29027 s;
CPU 1.337 -> 1.287 core-seconds; client completion 1.33291 -> 1.28951 s;
native TTFT .27766 -> .27732 s. Acceptance stays 14/16 (87.5%), two rollback/
replays and zero forced proposals. Post-first is 31 / same-call generation-minus-
first-token time. This is **not native MTP2**, batched MTP or a single T1 gain.

Stronger previous three-repeat llama.cpp non-spec controls (C1/2/4/8 HTTP
24.54/37.91/50.25/67.29), MTP1 and fresh-tested depths 2/3/4 are reused by exact
report hashes. Best tested depth 2 remains HTTP 30.17 / reported post-first
49.80 TPS; reported decode is 31 / mean `predicted_ms`. No new competitor run,
exhaustive optimal-configuration claim or lower-baseline selection. Earlier
failed parallel speculative runs stay preserved, not rerun unchanged or counted
as zero/native wins. Complete best-supported parallel MTP remains outstanding.

## GPU diagnostics, separate from throughput

Unprofiled C2 NVML primary/peer: GPU busy 75.03/55.02 -> 75.82/54.14%; controller
busy 36.17/34.32 -> 37.05/34.92% (**not GB/s**); power 116.51/96.93 ->
118.14/96.96 W; temperature 37.68/36.72 -> 39.65/38.10 C. SM/memory clocks
1,380/877 MHz unchanged; whole-device used memory 20,235/17,227 MiB unchanged,
including unrelated allocations. C8 busy is 83.87/59.72 -> 83.70/59.72%, with
unchanged memory. Full timed-window samples and MTP telemetry are in `metrics.json`.

Separate **23-prompt/8-output** timelines: fresh C2/MTP controls and candidates;
C1/C8 control traces reused from the preceding pinned T8 capsule, candidates
fresh. Not serving TPS. C2 kernel count remains 18,930; 4,720 dots use shared
staging. Combined original+shared dot sum falls **935.96 -> 902.68 ms**; GDN
114.49 -> 114.48; activity gaps 121.55 -> 122.11; instrumented native host span
942.93 -> 926.36 ms. MTP1 has 2,022 staged dots, unchanged 10,061 kernels;
dot sum 489.38 -> 474.83 ms; GDN 53.50 unchanged; host span 519.27 -> 511.24 ms.
Kernel sums overlap across GPUs and are not elapsed time or utilization.

C1/C8 traces execute **zero shared kernels**, as intended, with unchanged kernel
counts 15,808/37,306. C1 dot is 564.92 -> 564.82 ms; C8 3,239.06 -> 3,237.65.
No Tensor Core kernels run. H2D/D2H traffic is unchanged: C2 497,623,520 /
497,623,296 bytes; C8 about 1.990 GB each direction. No copy or fence removal.

Captured T2/row16 shared dots use 512 threads, 54 registers and dynamic shared
23,040 bytes (K5120) or 27,648 (K6144), no spill, **50% theoretical** occupancy;
old T2 uses 62 registers and also 50%. In standalone T1, fewer registers/higher
theoretical occupancy can still run slower. These are compiled/launch residency
ceilings, never achieved occupancy or causal proof of cache/bandwidth behavior.
Achieved occupancy, compute/DRAM throughput and cache/stall counters remain
unavailable under `RmProfilingAdminOnly: 1`; no repeated denied counter campaign.

## Provenance and remaining work

Official revision `fc05daec18b0a78c049392ed2e771dde82bdf654`, target ID
`3d635b87db004f42aefd6730e372e530`, previously verified target SHA256
`b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8`;
MTP SHA256 `a1a18b5f828d9b1cc1028fdc8463564aa4ffdbfe527311e1c0baa48cccfc1641`;
GGUF SHA256 `eb07277161915232e227a38608fc3ea80c0c17dc1c1583a9f3c5671edb6b5513`.
Reused verified payload hashes, not fresh scans. Native Q8/group32 F16 scales,
two A8 high/residual passes with FP32 scales/reduction and small-parameter
handling differ from GGUF Q8_0/Q8_1 arithmetic. Q8 labels are not identical math.
Default FP16 arithmetic and source/ABI compatibility remain intact.

The next unmeasured hypotheses are K-tiled staging for T4/T8, scoped multi-output
projection/fence fusion, and coalesced half-group loads with the exact original
reduction tree; see `next-kernel-notes.md`. Native multi-proposal MTP still needs
correct draft recurrence, rewind/catchup and GDN transaction qualification.
Do not assume these designs or competitor acceptance guarantee native speed.

Raw commands, failed T1 choices, unsupported geometry, code snapshots, resource
reports and large-artifact retention hashes are preserved. Original unrelated
allocations restore to idle 3,651/3,189 MiB, temporary port 18952 closed, no owned
inference processes left. Prior 23 capsules verify unchanged. The full external
performance goal remains unachieved.
