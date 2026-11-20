# V100 grouped/fused projections: retained, external goal incomplete

## Result

Three-repeat, unprofiled, fixed-work native engine comparison:

| Requests | Previous HTTP TPS | Grouped + fused HTTP TPS | Gain | Reused llama.cpp HTTP TPS |
|---:|---:|---:|---:|---:|
| 1 | 20.226 | 22.106 | 9.30% | 24.536 |
| 2 | 31.548 | 33.803 | 7.15% | 37.907 |
| 4 | 45.346 | 47.802 | 5.41% | 50.254 |
| 8 | 54.694 | 57.040 | 4.29% | 67.287 |

Unlike the preceding prefill-only C1 gain, this change also improves ordinary
single-request native post-first decode **23.079 -> 25.521 TPS (+10.58%)**.
C2/C4/C8 native post-first cohort rates improve **39.078/62.046/80.791 ->
42.338/66.353/85.222 TPS**. These internal rates exclude first outputs and HTTP
admission; they are not directly interchangeable with competitor HTTP rates.

Single MTP1 native post-first improves **31.729 -> 34.395 TPS (+8.40%)**;
complete-request HTTP **26.169 -> 28.081 TPS (+7.31%)**. All three repetitions
retain acceptance **14/16 proposals (87.5%)**, 16 verification calls, two replayed
inputs and 32 emitted outputs. No forced proposals or acceptance cherry-picking.
The best successfully tested external depth remains MTP2: 30.173 HTTP TPS and
49.803 reported post-first TPS. Native MTP2 is still not implemented.

**Every required external HTTP cell still loses.** Native/GGUF Q8 encodings,
small parameters and activation arithmetic differ; no identical-precision or
>=5% external-superiority claim. The goal stays active. No commit/push or local
Rust/CUDA compilation was performed.

## What changed and why the ablation matters

Same-input gate/up, qkv/z and Q/K/V already shared input packing/copies, but
previously fenced and gathered each matrix independently. The new path retains
both cross-device ownership fences at **group** scope, gives peer results
distinct admitted scratch spans, copies the combined peer output once, then
gathers each pitched slice. Payload bytes do not shrink.

A new descriptor-driven GPU kernel computes complete row dots for two/three
matrices using the same quantized input. Quantization, integer products, FP32
FMA/reduction order, FP16 rounding and private cache bytes are unchanged.
Primary pitched output removes the grouped scratch-to-output copy. No partial
FP16 sums, Tensor Core dispatch, external forward engine or Python token loop.
Default FP16, SM89, PP, streaming and non-host TP retain old paths. Public ABI
16; nine explicit native build dependencies. No new precision/CLI switch.

Source ablation 0 is the pinned prior library; 1 changes only grouped transport
while retaining original projection kernels; 2 adds fusion and direct pitched
primary stores. Single-repeat screens (not promoted speed claims):

| Arm | C1 HTTP | C2 HTTP | C8 HTTP | MTP1 C1 HTTP |
|---|---:|---:|---:|---:|
| Prior fences/kernels | 20.241 | 31.613 | 54.723 | 26.291 |
| Grouped transport, original kernels | 21.620 | 33.199 | 56.172 | 27.656 |
| Grouped + fused/pitched | 22.088 | 33.827 | 56.913 | 28.031 |

Most single-request improvement is already present in grouped transport.
Mode 2 is a **fusion-plus-pitched-store bundle**, not a pure kernel-only serving
ablation. The separate operator benchmark packs input only once in every arm:
18 real TP-first group/token combinations, three repeats and nine graph-contained
calls/wave. All output/workspace bytes match, with sampled independent FP64
relative L2 below .001. Fused median speedups span 1.018–1.239x; these short,
warmed operator events are not serving TPS or reliable per-wave NVML windows.
See `design.md` for ownership, fallback and test details.

Control library SHA256:
`a6599e636cc0c018261310de7d83e1e8dd41d6edcc635f9161402a89f7775a3f`.
Mode 1:
`6698f47bae3405279102890818782c992e8179aec966af8b5a08941160bb28b1`.
Retained mode 2:
`/home/data/wangyue/artifacts/venturi-projection-fusion-model-v3/build/671f471635529cef/libventuri_blocks.so`,
SHA256 `698452e2b2a39f3a5bb4d08d58e37a3330857f6444cc462f96147f53a7ab5e3d`.

## Correctness and explicit reuse

Both candidates execute fresh residual-profile trained checks against complete
hash-pinned control signatures: 98 full-vocabulary vectors, four original greedy
trajectories and final private state, eight heterogeneous batch slots, selected
reset/continuation logits/positions/caches. Every byte matches, not just token
IDs. Both execute fresh four-case natural MTP output/committed-target-state
checks, including rejection/replay and EOS. CPU/Torch model math is forbidden
in native generation checks.

Unchanged control records, trained FP16 signatures and independent represented-
weight FP16 draft reference are explicitly reused, not freshly executed. The
new `--profiles` helper records which candidate profiles actually ran. Prior
independent target gates remain top-1, relative L2<=.02, centered cosine>=.999
and KL<=.002. No broad original-BF16/corpus-quality claim or weakened tolerance.
Original artifact validation is reused; these are not fresh 27-GB hash scans.

79 new fused operator cases cover geometry/tails/strides/sentinels, both pass
counts, exact original bytes, independent FP64, graph replay, aliases, nonfinite/
zero inputs, preflight and invalid raw descriptors. Initial selected GPU suite
**221 passed**. Final affected build full V100 suite **1517 passed / 440 skipped**;
local CPU suite **765 passed / 1192 skipped**. Skipped tests are not GPU passes.
Previous immutable capsules remain unchanged.

## Workload, latency, CPU and memory

Same original-derived target, explicit two-pass residual profile and FP16 KV;
two V100s in physical order 1,0, resident host TP, 32 projection rows, eight slots,
capacity 256. Identical 23 formatted prompt IDs and 32 fixed greedy outputs;
fixed-work speed case ignores EOS, whereas separate correctness cases exercise
EOS. HTTP wall includes the 25-ms cohort admission window. Loading, warmup/reset,
diagnostics and profilers are excluded. TPS is outputs/global cohort wall,
not an average/sum of concurrent request rates or latencies.

| Requests | Native TTFT seconds, old -> new | Server CPU core-s/cohort | Mean client completion seconds |
|---:|---:|---:|---:|
| 1 | .19770 -> .19260 | 1.553 -> 1.420 | 1.58145 -> 1.44688 |
| 2 | .39708 -> .38690 | 2.003 -> 1.870 | 2.02755 -> 1.89240 |
| 4 | .77572 -> .76066 | 2.797 -> 2.653 | 2.82065 -> 2.67565 |
| 8 | 1.57370 -> 1.54292 | 4.677 -> 4.487 | 4.67635 -> 4.48439 |

Native batch TTFT starts at the common C-call and includes cohort prefill; it
is not streamed client TTFT. MTP1 native TTFT .20864 -> .20323 s, CPU 1.220 ->
1.140 core-s, client completion 1.22199 -> 1.13878 s. Trial/min/max latency,
counters and all timed windows are in `metrics.json` and raw HTTP reports.

C8 timed-window primary/peer NVML GPU busy is 80.0/48.8 -> 81.2/49.3%; this is
not achieved occupancy. Device-used peaks remain **20235/17227 MiB**, including
unrelated idle allocations and CUDA context; MTP1 primary peak remains 22435 MiB.
No Qwen VRAM saving is claimed. Quantization scratch stays 1,253,376 bytes/device.
Raw power, temperature, clocks and memory-controller samples and window
statistics are retained separately for each physical GPU.

## Separate profiling: counts, gaps and remaining limits

Fresh instrumented 23-prompt/**8-output** C1/C8 traces, not the 32-output HTTP
speed cells above:

| Trace | Host span ms | Total kernel launches | Summed dot ms across GPUs | Inter-activity gap ms |
|---|---:|---:|---:|---:|
| C1 old | 536.4 | 15808 | 426.1 | 103.0 |
| C1 grouped only | 506.1 | 15808 | 425.9 | 80.7 |
| C1 fused/pitched | 496.4 | 13504 | 406.9 | 81.4 |
| C8 old | 2327.5 | 37306 | 1794.6 | 214.8 |
| C8 grouped only | 2283.3 | 37306 | 1796.8 | 170.2 |
| C8 fused/pitched | 2262.1 | 33562 | 1749.1 | 173.5 |

Dot sums overlap across devices and are not elapsed time. Gaps are intervals
between this process's recorded activities, not proof the whole GPU is idle.
The grouped-only mode keeps dot duration approximately unchanged but reduces
fences: runtime event records **6416 -> 4112** at C1 and **10416 -> 6672** at C8;
wait-event counts fall identically. D2H calls **5272 -> 4120 / 8552 -> 6680**,
while H2D/D2H payloads remain approximately **.249 / 1.990 GB each direction**
at C1/C8. This is scheduling/API overhead reduction, not reduced PCIe payload.

Mode 2 additionally reduces primary D2D copies **400 -> 128** at C1 and
**7040 -> 3504** at C8, with C8 D2D bytes **961331200 -> 182763520**.
C8 summed runtime memcpy2D API time falls 348 -> 46 ms, but launch API time
rises 862 -> 975 ms; waits can move between APIs. Nested API sums are neither
CPU core-seconds nor additive elapsed savings. The actual serving gains are
only those measured in unprofiled repeats above.

GDN remains approximately **56.4 ms at C1 / 492 ms at C8**, on the primary,
unchanged by this optimization. Transfer payload and primary/peer imbalance
also remain. `next-bottlenecks.md` records bounded GDN and longer-chain MTP
hypotheses, not already-implemented speedups.

Exact compiled/trace resource joins show observed fused T1 global **72 registers,
128 threads, 0 dynamic shared bytes, theoretical 43.75% residency**; T2 full
staging **56 registers, 512 threads, 23040 shared bytes, theoretical 50%**; T8
K-tiled **56 registers, 512 threads, 18432 shared bytes, theoretical 50%**.
Reported local bytes are zero. These are static ceilings, **not achieved
occupancy**. Increasing occupancy is not itself the optimization objective.
Actual SM/DRAM/cache/stall counters remain unavailable under unchanged
`RmProfilingAdminOnly: 1`; no privilege changes or repeated denied campaign.

Raw SQLite/Nsight binaries remain remotely retained with SHA256/size records;
local evidence includes attribution, resource queries and read-only copy/API
correlation audits. All five owned workers are terminal, the benchmark port is
closed, and the original unrelated **3651/3189 MiB** physical-GPU footprints
are restored. No unrelated services/processes were stopped.

## Evidence layout and unfinished work

- `v100/study-v1`: mode-1 build, selected/full fixture tests, source identities.
- `v100/model-v2`: matched operator benchmark, mode-1 trained/MTP gates.
- `v100/model-v3`: mode-2 build/full fixture and trained/MTP gates.
- `v100/campaign-v4`: matched screens and three-repeat HTTP matrix/MTP1.
- `v100/final-v5`: separate model/operator traces and compiled resource queries.
- `summarize.py`: recomputes rates, CPU/latency and timed-window telemetry.
- `executed-source-deltas`, source/evidence manifests and terminal audit retain
  exact executed variants, prior references and owned-process cleanup.

The original external controls are explicitly reused from pinned WMMA/depth
capsules, not remeasured or selected weaker. Best-tested is not exhaustive best-
configuration; failed prior parallel MTP configurations are not zero-speed cells.
Native MTP2, batched MTP, paging/continuous batching, sampling, whole-model graphs,
long-context/broad quality and the required external wins remain unfinished.
