# V100 native producer integration and RMSNorm fusion screen

## Outcome

1. **SwiGLU/two-pass-A8 fusion is integrated into an explicit native build
   option, default OFF.** Eligible native column-MLP execution uses it on both
   V100s. Debug diagnostics explicitly materialize the old F16 intermediate;
   normal execution does not. Other profiles/geometries retain the old path.
2. **RMSNorm/two-pass-A8 fusion is a separate research operator**, not native
   model integration. Chunk8 and chunk32 pass the predeclared operator gate;
   retain chunk8 as the strongest integration candidate. Chunk128 is rejected:
   its producer is approximately 23% slower in confirmation. Preserve all arms.
3. Production was not stopped, restarted, switched or reconfigured. There is
   **no whole-model TPS, 8-API-concurrency or llama.cpp victory claim** here.

## Correctness and precision

257 checks passed: 8 mocked build-orchestration checks, 8 eligible-size native
checks, 122 producer byte/edge/graph checks, 48 existing native column/GDN/MTP
regressions, and 71 RMS fusion checks. The mocked tests do not compile code.
Actual CUDA compilation and GPU checks ran on WZU_Server's two V100s.

Both operators preserve F16 intermediate rounding, original group32 two-pass
residual-A8 codes and FP32 scales, Q8 weight bytes and existing projections.
Packed workspace/projection outputs match the separate producer byte-for-byte
in the tested cases. Fused RMS uses exactly the original 256-thread FP32
reduction association. Finite/Inf debug bits and NaN masks match; debug NaN
payloads are not a public contract. Independent FP64 sample checks are retained.

The eligible-size native fixture has hidden32/intermediate17408: it deliberately
exercises K8704 rather than silently testing only the fallback. OFF/ON checks
cover full MLP activations and both FP32 partials, T1/2/8/23/32 and T33 fallback,
complete hybrid trajectories/logits/traces/private GDN/KV states, generation,
EOS and selective reset of an eight-slot cohort. Existing MTP regressions use
small fallback geometries; a full-size eligible MTP corpus is not qualified.
Fixtures use the existing Qwen3.5 structural schema; they are not production
Qwen3.8 full-model quality results. The ordinary native path contains no new
Torch math or inference-time compilation.

## Unprofiled confirmation

Physical GPU0 V100, same current-source OFF/ON libraries, with row-pair projection
dispatch enabled **in both**. Actual Qwen3.8 Q8_0 gate/up/down weight slices,
synthetic hidden inputs. RMS additionally uses synthetic effective FP32 norm
weights (1 + small noise), not checkpoint-derived norm distributions. All timed
arms share exact intermediate/workspace/output addresses and projection kernels.

GPU-event median latency, layer0, seven alternating repeats:

| Operator chain | Tokens | Separate, us | Candidate, us | Latency reduction |
| --- | ---: | ---: | ---: | ---: |
| Native SwiGLU + two-pass quantization | 1 | 4.372 | 2.877 | 34.19% |
| Native SwiGLU + two-pass quantization | 8 | 5.417 | 3.553 | 34.41% |
| Native producer + Down partial | 1 | 65.700 | 64.061 | 2.49% |
| Native producer + Down partial | 8 | 114.012 | 111.012 | 2.63% |
| Native one-rank MLP projection chain | 1 | 179.651 | 178.104 | 0.86% |
| Native one-rank MLP projection chain | 8 | 277.309 | 274.749 | 0.92% |
| Research RMS + two-pass quantization, chunk8 | 1 | 12.872 | 4.116 | 68.02% |
| Research RMS + two-pass quantization, chunk8 | 8 | 12.513 | 4.495 | 64.08% |
| Research RMS producer + Gate/Up projections | 1 | 130.826 | 116.367 | 11.05% |
| Research RMS producer + Gate/Up projections | 8 | 179.958 | 165.560 | 8.00% |

The one-rank MLP chain includes input quantization, Gate/Up, SwiGLU quantization
and Down partial. It excludes norms, residuals, TP communication/sum, attention,
GDN, other layers and serving. The RMS consumer chain excludes SwiGLU and Down.
These are operator rows, **not eight concurrent requests**. Combined-chain
gains are measured directly; do not obtain them by adding isolated kernel times
or assign downstream cache effects entirely to launch removal.

Each screen covers layers0/31/63 and T1/2/4/8/23/32 (18 cases), plus independent
layer0 T1/T8 confirmation. Twenty logical calls per graph, five replays/100 calls
per wave, five warmup replays, seven alternating arm-order repeats. All raw waves
are retained. The shared-address control prevents per-arm allocation placement
from being credited as a producer optimization. Separate baselines are
out-of-place; native in-place producer performance is not established by the
isolated timing table. Eligible native in-place correctness is explicitly tested.

Screen median reductions: native producer33.53%, native producer+Down2.25%,
native one-rank MLP0.86%; RMS chunk8 producer65.45%, RMS producer+Gate/Up8.71%.
The operator gate is >=5% median producer and >=1% median consumer-chain reduction
in both screen and confirmation, with no >2% confirmed consumer regression. It
is not a model performance gate.

### CPU, wall and GPU scopes

T8 native producer+Down: wall114.408->111.433 us, process CPU114.599->111.961 us.
T8 RMS producer+Gate/Up: wall180.624->165.899 us, CPU180.799->166.200 us. Process
CPU consumption remains approximately one core because synchronization/spin
waits count. These are not isolated CPU launch-overhead savings, nor do graph
node reductions imply one fewer Python launch per logical call. GPU-event,
wall and process CPU medians are reported separately in `recomputed.json`.

## Why RMS improved, and its trade-off

The old norm has **one CTA per token**: batch8 launches only8 blocks on80 SMs.
Chunk8 independently recomputes each token's original reduction in20 CTAs,
then normalizes/rounds/quantizes eight groups per CTA: **160 blocks** for batch8.
This is not less arithmetic. It trades redundant statistics/cache reads for
parallelism, drops the F16 intermediate and one kernel, and specializes two
passes. It cannot be described as fusion alone.

Separate NCU producer-only measurements, T8/K5120 (not normal graph timings):

| Metric | Original RMS | Original quantization | Fused RMS chunk8 |
| --- | ---: | ---: | ---: |
| Grid blocks | 8 | 160 | 160 |
| Achieved SM occupancy | 12.44% | 24.32% | 24.55% |
| SM throughput / sustained peak | 1.07% | 20.82% | 19.62% |
| DRAM throughput / sustained peak | 1.05% | 2.95% | 2.43% |
| Long-scoreboard stalls | 66.32% | 10.72% | 25.42% |
| Global load request bytes | 327680 | 81920 | 1884160 |
| Global store request bytes | 81920 | 163840 | 163840 |

Global load requests grow **4.6x** versus the two separate kernels combined.
The measured profile's DRAM read bytes are104960+86016 versus109312, with DRAM
writes0 in these short cached profiles. Thus repeated statistics are largely
cache-served here; this is **not** a blanket bandwidth/arithmetic reduction or
a sustained full-model HBM result. Chunk128 issues too few independent blocks
and performs more group work per block, consistent with its observed slowdown.

Native SwiGLU T8 counters: achieved occupancy36.09%/39.88% for the two separate
kernels versus39.88% fused; SM throughput17.41%/31.54% versus32.50%; DRAM
throughput10.99%/4.33% versus7.97%. Combined global requests drop272 KiB. The
profiled fused kernel uses23 registers/thread. Occupancy, SM throughput, GPU
busy and DRAM throughput are different metrics and are not interchangeable.
SM throughput is an NCU hardware utilization aggregate, not measured TFLOP/s.

Counters, cache sectors and stalls were collected separately, with no cache
flush/clock control and no profiling-policy change. Do not use instrumented
durations as unprofiled throughput. Telemetry spans all arms/warmups/gaps:

- Native screen: busy0-100% (median81%), power40.76-261.60W,
  SM1297-1380MHz, memory877MHz, temperature32-48C.
- RMS screen: busy0-100% (median96%), power37.65-253.29W,
  SM1230-1380MHz, memory877MHz, temperature37-50C.
- Total device memory, including production, remains below27705MiB in screens;
  each benchmark's Torch reserve is below1GiB. Idle/free-memory guards are active.

## Enable and limitations

**Run only on the authorized remote V100**, from the synchronized repository:

```sh
python -m venturi.ops.build --component blocks --architecture sm70 \
  --q8-row-pair --q8-swiglu-quant --output-dir <new-remote-build-directory>
```

Load the returned SO with `CudaBackend` and an explicit native column-MLP
profile (`q8-residual-column-mlp` or its head-GDN variant). Routing is restricted
to SM70/F16 column-MLP, intermediate-per-rank8704 and T1..32. Defaults, unsupported
shapes and T33+ keep the old path. The flag is boolean, SM70-blocks-only and part
of the cache identity. `model.projection_info['swiglu_quant_fusion']` reports the
compile opt-in, shape eligibility and token range. Older libraries remain usable.

Built OFF: `2abf99fc86d5c0c8`; ON: `0765d1f981a26bfa`, under the recorded remote
artifact directory. Their source hashes match; the only differing `-D` flag is
`VENTURI_Q8_SWIGLU_QUANT=0/1`. Weight slice hashes match the earlier frozen
SwiGLU experiment. RMS remains only in `tools/projection/rms_quant_probe.cu`;
there is no native RMS switch or production deployment yet.

Before/after audits show the same running/healthy container, start time, restart
count, configuration fingerprint and counter policy throughout this campaign.
The local port8001 health probe timed out during intermittent SSH/tailnet
connectivity; server-side health checks succeeded. No tunnel/global SSH setting
was changed. Recovery reused the configured rescue port on **the same host**,
with host-key checks intact; no benchmark moved to other hardware.

Raw data is in `attempt-v1/`, relevant snapshots/diffs in `source/` and
`pre-edit/`, and deterministic recomputation in `summarize.py`. Failed source
syncs were not compiled or benchmarked; the executed source is the separately
recorded v4 snapshot. Previous frozen evidence capsules are unchanged. No
commit or push was performed. Offline audit from the repository root:

```sh
python3 benchmarks/v100-producer-fusion-native/summarize.py
```

Next work is to integrate RMS into the two-device normalized-input dataflow,
including any F16 output still needed by the peer, then qualify full-model
correctness and actual serving throughput. Neither step is claimed complete.
