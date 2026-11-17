# Night TPS campaign 2026-11-16/17

Scope: Qwen3.8-27B Q8 (native safetensors artifact), two V100-PCIE-32GB,
resident TP, projection q8-residual-column-mlp-head-gdn, sealed metadata,
continuous scheduling. Goal: 4/8/16/32 concurrency aggregate TPS + GPU busy.

Environment notes:
- torch 2.14.1+cu130 and 2.11.0+cu128 host builds lack SM70 compute kernels;
  venturi-sm70 venv uses torch 2.7.1 (PyPI cu126, sm70 included).
- Production llama.cpp container stopped at 23:33 (no auto-restart watchdog
  verified); restore via /home/wzu/restore-production.sh.
- Q8 artifact exported from unverified local HF source (allow-unverified);
  verified re-export running to qwen38-27b-text-q8-verified.safetensors.

Decisions log:
- Q5 SM70 W5A16 reference kernel landed (commit d8be7ec, 33 tests green).
- Baseline uses serve_native_benchmark.py --continuous-target, slots 8,
  capacity 512, projection-rows 32, chunk 16, round-robin (historical
  qualified config); wider slots/rows for 16/32 cohorts measured separately.

## Final results (03:00)

### Throughput (Qwen3.8-27B Q8, two V100 TP host-transport, greedy 16-in/32-out, 3-repeat medians)
| Concurrency | HTTP serve (8 slots, qualified cfg) | HTTP (32 slots) | native direct (32 slots) |
|---|---|---|---|
| 1 | 29.9 | - | - |
| 2 | 49.0 | - | - |
| 4 | 72.5 | - | - |
| 8 | 89.0 | 91.5 | 94.2 |
| 16 | - | 91.8 | 100.7 |
| 32 | - | 91.8 | 105.1 |

TPS is flat beyond 8 concurrent decodes on this TP topology.

### Root-cause evidence (nsys, /tmp/night-prof.nsys-rep)
- vqf::tiled_dot 47.6% GPU time, vct::partial 21.4%; gdn+conv+attention only ~19%.
- 81,048 host-pinned PCIe round trips per 33-step 32-concurrent window (~2,456/step,
  4 hops per un-fused matrix x 39 layers): H2D 0.99s + D2H 0.66s over the 11.6s blast.
- GPU busy 37-45% during decode (dmon): transport dependency chain, not kernel saturation.
- P2P transport probe fails on this host (byte probe); host transport is the only
  qualified TP path here -> architectural ceiling ~105 TPS native at 32 concurrent.

### Experiments with PASSING token equality but NO throughput gain (kept as evidence)
- Parallel per-segment streams for conv/GDN/attention (commit 96f63ac): 92->86.
- Wide q8 fused tiles 16/32 with SM70 64KiB-per-block shared fix: flat 85.
  (First attempt at tile32+passes2 hit the 64KiB limit: cudaErrorInvalidValue;
  fixed by Passes==1-only widest tile.)
- GPU util 37-45% throughout: confirms serialization in transport, not kernels.

### Landed assets
- Q5_g64_fp16 SM70 W5A16 kernel family, 33/33 tests, ABI 1 (commit d8be7ec):
  build/q5/libflash_next_q5_sm70.so. Prerequisite for one-instance-per-GPU 27B
  (19.5GiB weights fit a single 32GiB V100 with KV; Q8 does not).
- Verified Q8 artifact: /home/data/wangyue/models/venturi-native-q8-20261116/
  qwen38-27b-text-q8-verified.safetensors + manifest + export report.
- SM70-capable venv: /home/wzu/venvs/venturi-sm70 (torch 2.7.1+cu126; host torch
  cu128/cu130 builds carry no SM70 kernels).
- serve slots widened 1..8 -> 1..64 (experimental, opt-in, needs qualification).

### Recommended next steps (not done tonight)
1. Q5 exporter for the 27B main matrices (mirror tools/export_qwen_q8.py onto
   q5_g64_fp16 planes), then EXECUTION_QUALIFIED[(7,0)] += q5_g64_fp16 after the
   qualification campaign (bit-tests exist; needs token/state equality + resident smoke).
2. Two single-GPU Q5 instances + least-loaded concurrent routing: eliminates the
   2,456 hops/step host transport entirely. Expected scaling beyond the 105 TPS
   native ceiling; both GPUs fully busy instead of 37-45%.
3. Adaptive MTP for the 1-4 concurrency band (AdaptiveMTPPolicy exists; continuous
   mode currently rejects MTP).

## Kernel pass3 experiments (2026-11-17 evening, production stopped per owner)

CORRECTION: the earlier "587 GB/s" figure was a transcription error. The pass2
tool printed **59 GB/s** effective; kernel reality is ~115 GB/s (58.5 MB plane
in 0.51 ms). D2D copy reference: 751 GB/s aggregate. The kernel sits ~6.5x
from the hardware limit; projection is the sole end-to-end bottleneck (90.1%
GPU time, 98.8% busy, fresh nsys on the Q5 single-GPU path).

| Experiment | Result | Decision |
|---|---|---|
| exp1 A-tile uint4 loads | 0.996 -> 0.933 ms (+6.7%), A/B PASS | LANDED |
| exp2b 4-warp blocks (N64) | 1.137 ms (-22%): A-reuse loss > occupancy gain | rejected |
| exp3 register prefetch | design flaw in first cut, superseded by exp4 plan | deferred |
| exp4 PTX m8n8k4 direct-feed | cutlass signature confirmed (A{2},B{2},D{8}); probe
  run: c0/c1 = row g confirmed, c2/c3 = 4g+2 anomaly unresolved; layout needs
  one-hot scan. No shared/sync design ready once locked. | next session |

exp4 probe evidence: /tmp/ptx_mma_selftest.cu pattern on the server.
