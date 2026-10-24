# V100 native RMSNorm + two-pass activation packing

## Result

Implemented and qualified an **opt-in** native producer, including Qwen3.8 BF16
GDN-norm source compatibility. OFF/ON full-model comparisons show modest,
repeatable gains, **not** a several-fold improvement or a llama.cpp victory.

Official `Qwen/Qwen3.8-27B`, revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, original BF16 source → native
Q8 group32 / FP16 scales. Artifact ID `db1c60fea76c415abc892cf6c0eed6c9`;
SHA256 `41f99992b679aef7df2b6c671b76b09820730846c2c95a52406e7004c72f58f0`.
**Not** the production Uncensored GGUF weights.

Same two V100-PCIE-32GB GPUs, resident host-staged native TP with
`q8-residual-column-mlp-head-gdn`. Both arms enable row-pair and SwiGLU packing;
only `VENTURI_Q8_RMS_QUANT` differs. 16 input tokens per prompt, 32 fixed output
tokens, capacity 256 per slot, row budget 32. FP16 values/KV; two residual-A8
activation planes, FP32 scales/accumulation, GDN state and TP partials. Greedy,
EOS stopping disabled, **no MTP, HTTP serving, or Python token/model loop**.

Three paired repeats, OFF/ON → ON/OFF → OFF/ON, warmup/reset outside timing.
C1 and actual eight-slot native cohort, not eight scalar calls. Median TPS:

| Active requests | Post-first decode OFF → ON | Change | Complete generation OFF → ON | Change |
|---|---:|---:|---:|---:|
| 1 | 32.1333 → 32.7947 | +2.058% | 30.5454 → 31.1376 | +1.939% |
| 8 | 131.9353 → 133.4055 | +1.114% | 102.3992 → 103.4198 | +0.997% |

Decode numerator: 31 tokens for C1, 248 for C8. Complete-generation numerator:
32 / 256 tokens, measured in a separate unsplit call including prefill/first
token. C8 rates are **aggregate**: ON decode is 16.676 TPS per request, not
133 TPS for each request. Loading, warmup, reset, correctness copies and profilers
are excluded from these scores. Per-trial raw ranges and paired deltas are in
`recomputed.json`; three repeats do not establish universal speedups.

## Correctness and compatibility

- Remote suites: 39 build/source + 8 eligible native fusion + 71 producer-byte
  + 48 fallback/native/MTP regression cases passed; 1 existing CPU test skipped.
  Five additional local CPU-only evidence-harness tests passed.
- Native eligible fixture hidden5120 covers T1/2/8/23/32 and T33 fallback,
  exact normalized FP16 bytes, full MLP output/activation, both FP32 TP partials,
  scalar traces/state, mixed EOS, reset and injected-failure recovery.
- Full official Qwen3.8 OFF/ON: identical teacher logits, per-layer hidden traces,
  recurrent/KV state and 32-token trajectories, including the eight-slot cohort.
  All checked floats finite; the first two scalar/cohort trajectories also agree.
  Raw qualification report SHA256 values match the published gate marker.
- Complete source hashes match immutable official Hub LFS SHA256 / Git blob IDs.
  449 floating payloads match source exactly; 1,206 sampled Q8 rows match an
  independent scalar nearest-even quantizer. All 48 BF16 GDN norms widen exactly
  to FP32. This is not a BF16-versus-Q8 generation-quality evaluation.
- Preserve sigmoid attention output gating and SiLU GDN gated normalization.
  The inspected official `qwen3_5` implementation does not consume the extra
  `output_gate_type: "swish"` metadata; no invented swish attention kernel.

## Implementation boundaries

`src/venturi/ops/csrc/q8_rms_producer.cuh` reproduces the original 256-thread FP32
RMS reduction, rounds to FP16, packs both A8 planes, and **retains materialized
FP16 normalized values for the peer**. The peer's input packing and existing
transport/event fences are unchanged. The primary packing is consumed immediately
under the same stream/model lock; no stale pointer cache.

Selection: explicit SM70 build flag `--q8-rms-quant`, resident FP16 residual
column MLP, hidden5120, rows1..32. Other paths fall back. Input/final norms and
attention/GDN input packing are not fused. Global build default stays OFF;
select the ON library explicitly. See `docs/native-rms-quant.md`.

## Hardware diagnostics (separate from TPS)

Selected post-attention producer in the real C8 decode path, one NCU replay
sample per arm. Baseline is RMS alone; ON is RMS **plus packing plus FP16
materialization**, so these are not identical-work isolated kernels.

| Metric | RMS baseline | Fused ON |
|---|---:|---:|
| Kernel time | 18.976 µs | 8.672 µs |
| Grid, block | 8 CTAs, 256 threads | 160 CTAs, 256 threads |
| Achieved active SM occupancy | 12.42% | 24.70% |
| SM throughput (% sustained peak) | 1.06% | 20.28% |
| DRAM throughput (% sustained peak) | 1.11% | 8.34% |
| Read+write DRAM rate | 5.585 GB/s | 40.786 GB/s |
| L1 global load bytes | 327,680 | 1,884,160 |
| L2 read sectors | 8,295 | 55,240 |
| Long-scoreboard stall / active warp | 66.69% | 26.09% |

The eight-group chunk repeats statistics to expose CTAs. It has **more** repeated
reads, not universally less arithmetic or bandwidth. NCU used unmodified clocks,
uncontrolled caches and 11 kernel-replay passes; cache writeback/eviction can
cross kernel boundaries. These numbers are not whole-model bandwidth or TPS.
No global profiler privilege, clock or driver-policy settings were changed.

The first NVTX-triggered Nsight attempt produced no report; that failure is
retained. A separate full-capture repair succeeded, with offline raw-row filtering
to four post-first C8 tokens. Raw trace pins show kernel launches 14,368 → 14,112
and CUDA runtime calls 19,805 → 19,549: **64 fewer launches per decode step**.
Copies remain 3,100 in both. Full originals remain on the V100 host, with SHA256
pins and an exact filtered transfer-window copy. Trace timings are instrumented;
CUDA event-completion tracing warns of possible false dependencies. Do not use
these traces as unprofiled throughput or device-wide-idle/CPU-causality proof.

## CPU, latency and telemetry

| Requests | Decode CPU seconds OFF → ON (median) | CPU cores OFF → ON | TTFT seconds OFF → ON (median) |
|---|---:|---:|---:|
| 1 | 0.972384 → 0.952529 | 1.007932 → 1.007849 | 0.088527 → 0.087991 |
| 8 | 1.892883 → 1.873559 | 1.007307 → 1.007339 | 0.624158 → 0.623019 |

Lower elapsed/CPU time is not reduced average CPU-core utilization. GPU telemetry
sampled memory, GPU busy, power, clocks and temperature every 0.2 seconds. Decode-
window per-trial samples are summarized in `recomputed.json`; `GPU busy` is not
occupancy or memory bandwidth. SM/memory clocks in the sampled first C1 baseline
window were 1380/877 MHz, without forcing clocks. VRAM samples include the fixed
benchmark allocation of both a scalar session and eight-slot cohort even during
C1, not the minimum memory required for a standalone C1 request.

## Production and evidence

User authorized drained temporary maintenance and original-parameter restoration.
Both GPU windows restored the same container ID/configuration fingerprint and
verified health/API model listing; the candidate was **not deployed** to production.
Production remains the original Qwen3.8-Uncensored llama.cpp/GGUF/MTP path.
Independent restore watchdogs were installed for the finite windows and cancelled
after health passed. An aborted v3 test unit was runtime-masked to prevent a delayed duplicate launch; removal of that Venturi-owned test-unit mask could not be rechecked after the last network interruption. It is not the production container/service. Last retained repair restoration: 2026-10-22 15:07:30 +08:00.

The network became unreliable during **artifact retrieval after experiments had
finished**. `partial-transfer/` contains only complete tar members received over
authenticated SSH, with per-file SHA256 in `partial-transfer-manifest.json`.
All six raw timing reports, full-model qualification reports, NCU CSVs, build and
storage gates, and both production-restoration records are already present and
independently recomputed. Full remote archive SHA256 is pinned in
`remote-archive-pin.json`; the entire archive has **not yet been downloaded and
verified**. Remaining local trace transfer is an archival follow-up, not a new
benchmark result or a claim that production is down.

Recompute currently available complete members:

```sh
python benchmarks/v100-rms-native/summarize.py --allow-partial-transfer
```

Once the complete archive is fetched and its pinned SHA256 verified into `remote/`,
run without that explicit opt-in. Failed network, dependency and profiler attempts
are retained; do not overwrite them. No commits/pushes, new license, local CUDA or
Rust compilation, GPU-hardware substitution, or subagents were used. Previous
frozen capsules are unchanged.
