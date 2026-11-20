# Two-V100 llama.cpp comparison: initial screen and optimization direction

## Status / revised goal

The user changed the objective to **compare against llama.cpp and optimize the
actual gap**. Project acceptance is in `docs/native-q8-goal.md`: qualify correctness,
then require repeatable >=5% wins in single decoding and the chosen 1/2/4/8-request
matrix. This capsule is an **initial one-repeat 1/8-request screen**, not promotion,
a completed goal, or an external-engine win. The app's older blocked goal record
cannot be rewritten/resumed by the available goal-control API; the revised
project objective is saved without falsely completing the older goal.

**Finding:** Venturi is still substantially behind. A valid llama.cpp layer/MTP1
arm reaches **62.40 aggregate complete-request TPS at C8**, versus **24.20** for
native PP without MTP (about **2.58x**). This is a more useful optimization target
than maximizing occupancy or reporting another isolated internal speedup.

## Matched workload, deliberately different representations

- Same official **Qwen/Qwen3.5-27B**, revision
  `fc05daec18b0a78c049392ed2e771dde82bdf654`; no old uncensored Qwen3.8 substitution.
  Original source file hashes are verified before conversion and file identity/
  timestamps checked unchanged afterward. No new checkpoint download was needed.
- Standard original-source GGUF Q8_0 export includes 851 target + 15 MTP tensors:
  **29,047,083,200 bytes**, SHA256
  `eb07277161915232e227a38608fc3ea80c0c17dc1c1583a9f3c5671edb6b5513`.
  Remote reusable model is
  `/home/data/wangyue/models/venturi-qwen35-comparison-v1/qwen3.5-27b-original-q8_0.gguf`.
  Export and source/converter hashes are in `v100/export-report.json`.
- Native target/MTP artifacts and ABI-15 library are the previously qualified
  originals in `../v100-native-mtp/summary.md`; native executed library SHA256
  `dd8476377133a506aa49ff7b25f839e6a1b9931a9774a0c5c3230025cd404af0`.
- The competitor is the user's **pinned production llama.cpp fork**, not a claim
  about latest stock upstream. Its `--version` reports commit unknown. Supplied
  BUILD_INFO/source hints and hashes of the binary/libraries are retained;
  source provenance is not inferred solely from that version string.
- Two V100-PCIE-32GB, `CUDA_VISIBLE_DEVICES=1,0`, four OMP/MKL/engine CPU threads,
  8 preallocated slots, capacity 256 per slot. Native TP/PP use explicit host
  CUDA DMA; owned llama processes use `NCCL_P2P_DISABLE=1` rather than retrying
  the previously failing peer route. No driver/global environment changes.
- Same pretokenized 23-ID formatted story prompt, exactly 32 greedy output IDs,
  EOS ignored, no prefix reuse, no sampling penalties. One full warmup wave per
  count, then one measured wave. Model loading and slot allocation are excluded;
  per-request reset and all HTTP submission/completion work are included.
- Both sides use the same loopback HTTP client/barrier and fixed output work.
  Native has a thin benchmark adapter over one C++ generate call per request,
  **not** a Python token/layer loop or a production streaming API. Aggregate rate
  is `32 * requests / global cohort wall`, never summed request durations.
- **This is not identical-arithmetic/precision parity.** Native uses its existing
  Q8/group-32 FP16-scale policy, FP16 activation/KV storage and preserved BF16 GDN
  a/b projections. GGUF Q8_0 encoding/rounding differs; the standard export has
  506 Q8_0 and 360 FP32 tensors, including quantized small projection differences.
  Engine-internal activation packing/reductions may differ too. The source/header
  audit is not proof of executed kernel selection. See `v100/precision-audit.json`.
  Equal Q8 labels and FP16 KV do not establish equal compute arithmetic or corpus
  quality. Do not market this practical engine screen as exact-math parity.

## Results: complete-request loopback API throughput

All successful measured responses, including valid C1 cells of partial arms,
match the **same 32 pinned native IDs** on this one prompt. That is a narrow
cross-engine smoke, not broad trained quality or teacher-forced logit qualification.

| Engine/configuration | C1 output/request-wall TPS | C8 aggregate TPS | Arm outcome |
|---|---:|---:|---|
| Venturi TP, no MTP | 15.37 | 15.42 | Complete |
| Venturi TP, MTP1 | 18.45 | 18.37 | Complete |
| Venturi PP, no MTP | 13.05 | 24.20 | Complete |
| llama.cpp tensor, no MTP | 21.17 | 61.27 | Complete |
| llama.cpp layer, no MTP | 17.24 | 58.77 | Complete |
| llama.cpp layer, MTP1 | 23.38 | 62.40 | Complete |
| llama.cpp tensor, MTP1 | 24.82 | — | Valid C1; C8 incomplete |
| llama.cpp tensor, MTP3 | 23.74 | — | Valid C1; C8 incomplete |
| llama.cpp layer, MTP3 | 23.52 | — | Valid C1; C8 incomplete |
| llama.cpp tensor, MTP1, unified KV retry | 24.90 | — | Valid C1; C8 incomplete |

Failed/incomplete C8 arms do **not** have a throughput result. Do not treat them
as zero TPS, a native win, or evidence that llama.cpp generally cannot serve MTP
concurrently: the layer/MTP1 arm completed all eight requests correctly.

Single-request internal timing is separate from the API table. In the ordinary
recovered tensor/MTP1 cell, llama reports **46.28 post-first TPS**, versus native
TP/MTP1's **24.53** from same-call native total-minus-first-ready timing. These are
engine-accounted scopes, not streaming client TTFT. The API includes costs outside
those timers: e.g. llama tensor/MTP1 prompt+prediction is about 1.015 s internally
versus 1.289 s HTTP cohort wall. No concurrent decode TPS is synthesized.

MTP1 acceptance is **14/16 = 87.5%** on both sides. The llama MTP3 C1 cells report
17/38 = 44.7% and do not beat MTP1 here. Longer drafting is not automatically faster.
CUDA graph reuse is recorded in llama's logs; native whole-model graphs are not
implemented. Fresh equivalent external kernel attribution is still required before
claiming exactly how much of the gap comes from graphs, packing or scheduling.

## CPU, latency and GPU observations

Selected C8 rows (one measured wave, same 256 emitted outputs):

| Configuration | Global wall (s) | Server CPU core-s | Mean completion (s) | Max completion (s) |
|---|---:|---:|---:|---:|
| Venturi TP/MTP1 | 13.932 | 14.38 | 13.658 | 13.932 |
| Venturi PP/no MTP | 10.576 | 20.02 | 10.446 | 10.576 |
| llama tensor/no MTP | 4.179 | 4.19 | 4.126 | 4.177 |
| llama layer/MTP1 | 4.103 | 4.10 | 4.062 | 4.101 |

CPU is Linux process utime+stime (all server threads), not submitting-client CPU
or summed per-request CPU. Latency includes queue/HTTP/reset work. One wave is
insufficient to make p95 claims. `metrics.json` contains all measured cells and
100 ms NVML samples filtered by their exact global-wave windows.

At C8, native PP GPU busy averages **90.1%/94.0%**, while llama layer/MTP1 averages
**36.3%/38.3%** (physical GPUs 1/0), yet llama has higher throughput. Corresponding
mean powers are 230/234 W versus 99/101 W; peak device used VRAM 18,581/19,043 MiB
versus 17,941/19,541 MiB. Native SM clocks vary 1275–1380/1320–1380 MHz, while this
llama arm samples 1380 MHz. Temperature/controller-busy/clocks/ranges are retained,
not silently normalized. Busy percentage is **not useful-work efficiency**.

Driver policy remains `RmProfilingAdminOnly: 1`. Achieved SM occupancy, compute/
DRAM/cache/stall counters remain unavailable, not zero. No privilege/clock changes
or repeated counter campaign. This screen has no new whole-engine Nsight capture;
older native attribution cannot quantify the fresh external gap by itself.

## One bounded kernel experiment, not promoted

A separately built SM70 candidate changes only T2 projection row tiling from one
to two output rows per block. Per-output FP32 FMA/reduction order is unchanged.
It is retained under `v100/pair-source` and build provenance under `v100/pair-build`.

- **62 passed / 61 skipped** relevant Q8/MTP tests.
- All **30** real projection/token cases retain exact baseline output bytes.
- T2 median projection speed ratio **1.0648x**; representative query projection
  improves about 7.1%, but the small [1024,5120] KV projection **regresses 5.2%**.
- Candidate uses 40 registers/thread, no local bytes, 75% CUDA-API theoretical
  occupancy versus the baseline T2's 32 registers/100%. Faster micro-results do
  not imply higher achieved occupancy or an end-to-end gain.
- No trained end-to-end candidate TPS/quality promotion campaign was run. The
  runtime/default source stays unchanged. Do not spend a long promotion campaign
  on this marginal tile while the observed C8 gap is roughly 2.6x.

## Recovery and integrity

1. Initial export completed and original source stayed unchanged. Initial llama
   validation incorrectly treated `tokens_cached` as reused input. In this API it
   can be 54 retained KV tokens after a fully evaluated 23-token prompt. The first
   worker's failed arms/logs are preserved and overall exit **1** is intentional.
2. A regression-tested gate now checks actual `timings.prompt_n == 23` and
   `tokens_evaluated == 23`; recovered responses also show `cache_n == 0`.
   Raw warmup and failing responses are saved. This fixes interpretation, not
   numerical correctness or the required amount of work. Adapter tests:
   **17 local passes and 17 V100 passes**. The initial native successful arms
   remain valid: their zero cached-count field already passed the old gate.
3. Recovered tensor MTP1/3 and layer MTP3 close connections during C8 warmup.
   Do not call this unstable SSH: model files/startup/C1 and other C8 arms work.
   Peak sampled VRAM is well below capacity. A bounded tensor/MTP1 retry with
   explicit unified KV still fails; the owned server's final return code is
   **−11 (SIGSEGV)**. Its logs/state are preserved. No driver reset, unrelated
   service stop, endless identical retry or false successful C8 result.
4. The candidate build's first local dispatcher had a Python quoting syntax error
   before any remote execution. The failed launcher is retained as `.py.txt`;
   the repaired CPU-only build and GPU micro-screen complete successfully.
5. Export and pair build/screen exit 0. Initial/recovery/unified campaign exit 1
   reflects the preserved incomplete arms, not a network-blocked overall project.
   All owned servers/workers are terminal, GPU usage returns to **3651/3189 MiB**
   (physical 0/1), and unrelated services/configurations remain unchanged.

No commit/push, no local Rust compilation, no old evidence capsule mutation.
The new capsule freezes this finite **incomplete screen**, not a completed goal.
Source/evidence hashes and explicit per-worker terminal checks are separate.

## Next implementation, in priority order

1. Native **batched decode with independent per-request GDN/convolution/KV/cursor**
   and grouped projections, then an active-slot scheduler. Prove heterogeneous
   prompt/state/EOS/rejection/reset correctness against scalar native requests.
   Do not concatenate independent requests as time tokens in one GDN cache.
2. Single-request projection/verification and launch/transfer work, measured
   against llama, including graph reuse and explicit arithmetic/packing audit.
3. Repeat controls >=3 times and expand to 1/2/4/8 requests and additional prompts/
   lengths; collect separate external/native timelines. Pin stock-upstream as an
   additional comparison before making general llama.cpp claims. Requalify any
   changed arithmetic rather than quietly weakening precision or state gates.

The current best native short-prompt choices remain TP/MTP1 for one request and
ordinary PP for eight, but **neither beats the valid external baseline** here.
