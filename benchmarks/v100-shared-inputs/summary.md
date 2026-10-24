# Scoped projection-input sharing on two V100s

2026-10-19; **arithmetic-preserving internal gain, external goal still active**.
Recompute with `python3 benchmarks/v100-shared-inputs/summarize.py`.
Raw reports, commands, loaded library maps, telemetry, source and hash manifests
are retained. Previous frozen evidence is unchanged.

## Implementation and lifetime

The model now groups only explicit, adjacent projections with identical inputs:
GDN QKV/z, attention Q/K/V, and MLP gate/up. `Model.shared_linears` validates
common K and distinct input/output pointers, then shares the input transport
and (for `q8-residual`) the packed activation buffer. Matrices still execute
separate projection kernels with their original codes/scales and output layouts.
This is **not fused GEMM**, compressed inter-GPU transport or a CUDA graph.

Reuse flags exist only on the helper's host stack. No pointer/shape cache survives
a group, chunk, head, layer, request switch, failed call or reset. Primary and
peer readiness are tracked separately, including a projection with no peer rows.
The primary producer event, peer wait, peer completion event and primary gather
wait remain for **every** projection; skipped input copies do not remove the
fences protecting output scratch. Existing model mutex ownership is unchanged.
Single/streaming/PP original arithmetic retains its serial projection order;
TP saves transfers and the residual profile also saves quantizer launches.

No new GPU allocation, model descriptor, C ABI or Python execution loop is
introduced. ABI remains 16. Original `fp16` and opt-in `q8-residual` arithmetic
are individually unchanged; the latter still differs from original FP16 math.
All model Python code is unchanged from the preceding qualified capsule.

Pinned unshared library SHA256:
`79a31157faea8ce166e755fea3619969e110cab1c9c9f820a5c89de05ab226a7`.
Shared library SHA256:
`861b528b0d7464c53ce51a4f29140984bda1ebfccf55b539aa82d5492f0f1ed4`.
Six native source hashes match both executed snapshots; current source/test/tool
hashes match the final full-suite snapshot. CUDA compilation occurred remotely
on WZU_Server. No local Rust/CUDA compilation, commit or push.

## Correctness gates

- **40 targeted GPU tests passed**: 16 new unshared/shared controls plus the 24
  existing explicit-profile tests. Both profiles, single/TP, tied/untied weights,
  1/8 private batch owners, interleaved sessions, chunk/head scratch reuse,
  selected reset/continuation, hidden traces and cache bytes are exercised.
  Existing tests also cover 1/2/4/8 ragged batches, EOS/budgets, source guards,
  failure poisoning, forced MTP accept/reject, ownership and Torch-math rejection.
- Original regression gate: **130 passed / 125 skipped**. Final V100 full suite:
  **1036 passed / 440 skipped**; local: **760 / 716**. Optional/hardware skips
  are not passing GPU tests.
- Trained old/new controls compare **98 complete-vocabulary logit vectors per
  profile** byte-for-byte using SHA256 of contiguous CPU diagnostic bytes, with
  dtype and shape included. All original story/capital/Python/arithmetic outputs
  remain unchanged. Diagnostic versus actual native generation committed state
  is exact, with Torch math patched out during generation.
- Eight heterogeneous requests, including EOS and an independently truncated
  budget, retain old/new private GDN/history/KV bytes and positions. Selected
  reset followed by unequal new prompts also preserves exact logits and every
  participating/nonparticipating private cache. No cross-profile state-equality
  claim is made.
- `qualify_qwen_shared_inputs.py` checks the hash of the preceding qualified
  report and pins the exact old binary before comparison. Its prior independent
  CPU thresholds/quality scope are reused **transitively through exact new/old
  bytes**, not rerun or weakened. The original artifact is loaded by both arms;
  prior verified original-source payload identity is reused, not a fresh large
  payload SHA scan. `prior-reference-pins.json` pins the immutable references.
- Newly executed natural MTP passes all four trained output/committed-target
  state gates. Story remains **14/16 proposals accepted**, two rollbacks/replays,
  no forced proposals in timing. The unchanged independent draft block gate is
  reused from the prior capsule; no fresh draft-reference or changed draft math
  is claimed.

Quality remains limited to these represented/native/CPU references and trained
cases, not original-BF16/corpus equivalence or SM89 qualification. MTP is still
fresh/reset-only, greedy, one proposal, and not batched MTP. No paging, continuous
admission, 256K context, sampling or whole-model graph implementation is implied.

## Matched work and arithmetic disclosures

Same two Tesla V100-PCIE-32GB GPUs in logical order `CUDA_VISIBLE_DEVICES=1,0`,
explicit pinned-host TP, no NVLink. Same official Qwen3.5-27B revision
`fc05daec18b0a78c049392ed2e771dde82bdf654`, target Q8 artifact
`3d635b87db004f42aefd6730e372e530`, original Q8 codes/F16 scales, FP16 KV and
FP32 GDN. Timed work: 23 formatted prompt IDs, 32 fixed greedy outputs, ignored
EOS, no input reuse, eight slots, **256 context tokens per slot and 32 projection
rows**. The 25 ms batch admission window is included in HTTP wall; every wave
is verified unsplit. Adapter default stays 5 ms. Warmups are separate.

Precision is not silently equalized: native original activations use FP16;
`q8-residual` uses two group32 A8 passes with FP32 scales and a different FP32
reduction. MTP's draft retains original FP16 arithmetic. The pinned production-
fork llama.cpp uses different GGUF Q8_0 encoding/rounding and selected small-
parameter precision; its actual Q8_1 activation packing was traced previously.
This is practical same-source comparison, **not identical-math parity**.

Competitor binary SHA256:
`e5b1b5782cd5b4acec5316ac5bd40123e0b0c9826643ae4148b0ec6e359f5fa2`;
original-source GGUF:
`eb07277161915232e227a38608fc3ea80c0c17dc1c1583a9f3c5671edb6b5513`.
Tensor split 1,1, FP16 KV, Flash Attention, backend sampling and supported graphs
remain enabled. Source-build hints are not a verified exact binary source commit.
Unchanged failed C8 MTP configurations were not retried, counted as zero or
interpreted as native wins. Competitor best-config/depth work remains required.

## Unprofiled throughput

One-repeat C1/C8 screens precede promotion. Every cell below has **three timed
repeats**. Complete-cohort HTTP TPS is total outputs/global cohort wall; never
sum concurrent request durations. Profiler runs are separate.

| Requests | Residual unshared | Residual shared | Internal gain | Fresh llama non-spec |
|---:|---:|---:|---:|---:|
| 1 | 18.09 | 18.97 | +4.88% | 24.50 |
| 2 | 26.82 | 27.66 | +3.10% | 36.29 |
| 4 | 33.71 | 34.35 | +1.87% | 52.35 |
| 8 | 37.48 | 38.62 | +3.04% | 62.80 |

Original FP16-activation C1 improves **15.21 -> 15.81** HTTP TPS; C8
**28.85 -> 29.72**. Its C2/C4 performance was not remeasured in this milestone.
Residual C8 native post-first cohort rate improves **56.84 -> 58.87 TPS**, using
248/(last native finish minus shared first-token time), not per-client TPS.

| C1 metric | Residual unshared | Residual shared | Fresh llama |
|---|---:|---:|---:|
| Non-spec post-first TPS | 21.78 | 22.98 | 34.35 |
| MTP1 post-first TPS | 29.27 | 30.42 | 46.77 |
| MTP1 whole HTTP TPS | 22.62 | 23.45 | 29.25 |
| Non-spec native TTFT, s | 0.3032 | 0.2972 | different timing scope |
| MTP1 native TTFT, s | 0.3186 | 0.3080 | different timing scope |

C1 post-first native rate is 31/(native generation minus native TTFT); llama's
is 31/engine-reported predicted_seconds. Nonstream HTTP does not measure client
streaming TTFT. Native internal TTFT excludes HTTP/admission/prior reset time;
scalar C-entry timing can include model-mutex waiting. Raw engine/HTTP timings
are not interchangeable.

The prior valid llama C8 control reached **66.83 TPS**. The new 62.80 is normal
campaign evidence with scheduling/run variability, **not grounds to discard the
stronger prior control or redefine the goal downward**. New native still loses
to both. Three repeats do not establish universal or confidence-bound speed.

## CPU, completion latency and sampled GPU diagnostics

| Requests | Residual global wall s, unshared -> shared | Process CPU core-s | Mean individual HTTP completion s |
|---:|---:|---:|---:|
| 1 | 1.7688 -> 1.6865 | 1.747 -> 1.660 | 1.7680 -> 1.6858 |
| 2 | 2.3859 -> 2.3142 | 2.363 -> 2.290 | 2.3846 -> 2.3131 |
| 4 | 3.7966 -> 3.7267 | 3.773 -> 3.703 | 3.7944 -> 3.7244 |
| 8 | 6.8296 -> 6.6283 | 6.837 -> 6.633 | 6.8255 -> 6.6242 |

Residual MTP1 C1 wall 1.4146 -> 1.3647 s; CPU 1.413 -> 1.363 core-s. CPU
polling remains active: core-seconds mostly fall with elapsed time, not because
CPU consumption disappeared. Original FP16-activation C8 wall 8.8749 -> 8.6138 s,
CPU 8.877 -> 8.617 core-s. Fresh llama non-spec C8 wall 4.0763 s, CPU 4.083
core-s, mean individual completion 4.0081 s. All trial/request values are retained.

At residual C8, approximately 100 ms NVML samples show physical primary/peer mean
GPU busy **83.7/63.2 -> 85.7/64.0%**; memory-controller busy
35.6/33.3 -> 36.6/34.1%, **not GB/s**. Mean power 132.9/105.2 -> 135.3/106.5 W;
temperatures 44.7/41.7 -> 45.5/42.1 C. SM clocks stay 1380 MHz, memory 877 MHz.
Original FP16-primary C8 mean clocks vary 1378.1 -> 1372.1 MHz; no clocks, limits
or profiling permissions were changed. These are sampled observations, not an
energy integral or controlled thermal experiment. Full C1/2/4/8 telemetry for
both devices, including extrema, is in `metrics.json`.

VRAM is unchanged: residual native C8 peaks **20,235/17,227 MiB**, MTP1 C1
22,435/17,227 MiB; original FP16 C8 20,233/17,225 MiB. These are device-wide
values including unrelated services and CUDA allocations, not just model bytes.
New shared groups allocate no additional GPU memory. Projection scratch remains
**1,253,376 bytes/device at max_tokens=32**, independently of context capacity256.
`prior-report-corrections.json` corrects the preceding frozen summary's prose
that conflated those limits. Its commands/data/manifests were not rewritten.

## Separate CUDA timeline proof

New instrumented captures use **23 prompt / 8 output** tokens. The exact old
unshared captures from `v100-native-residual` are reused unchanged, hash-pinned,
with the same workload and exact old binary. Their timings are not fresh
unprofiled controls. C8 has eight logical requests but one native-call NVTX
range; legacy `requests: 1` counts the range. Sum-of-kernel durations across
both devices is attribution, not elapsed time, TPS or achieved occupancy.

| Diagnostic | Residual C1, unshared -> shared | Residual C8, unshared -> shared |
|---|---:|---:|
| Total kernel launches | 18,112 -> 15,808 | 41,050 -> 37,306 |
| Quantizer launches | 6,416 -> 4,112 | 10,416 -> 6,672 |
| Quantizer summed time | 17.76 -> 11.57 ms | 46.60 -> 32.14 ms |
| Dot summed time | 622.68 -> 625.29 ms | 3902.55 -> 3926.86 ms |
| GDN recurrent summed time | 56.36 -> 56.34 ms | 490.81 -> 490.65 ms |
| Gaps between captured activities | 114.26 -> 103.71 ms | 241.52 -> 214.49 ms |
| H2D bytes | 293,048,560 -> 248,811,760 | 2,344,388,480 -> 1,990,494,080 |
| D2H bytes | 293,048,448 -> 248,811,648 | 2,344,387,584 -> 1,990,493,184 |

Copies fall by **44,236,800 bytes per direction at C1** and **353,894,400 at
C8** (about 15.1%). At C8 H2D/D2H copy counts fall 10,429/10,424 -> 8557/8552;
summed times 350.62/258.18 -> 293.03/218.80 ms. Original FP16 traces show the
same byte/count reductions but unchanged kernel-launch counts, as expected
without activation packing. Cross-device output fences and state results remain
correct. Captured gaps are not device-wide idle time; simultaneous-device kernel
time is not achieved SM occupancy.

Dot kernels themselves did **not** accelerate; their small timing variation is
not a kernel win. Residual C8 projections including packing still account for
**86.44% of summed kernel time**; GDN about 10.71%. Compiler resources remain
64/62/78 registers for 128-thread residual T1/T2/T4, with 50/50/37.5%
**theoretical** occupancy; quantizer 38 registers/256 threads, 75% theoretical.
Exact-binary read-only queries are retained. Achieved occupancy, compute/DRAM
throughput and cache/stall counters are still **unavailable** under unchanged
`RmProfilingAdminOnly: 1`. No repeated counter campaign or privilege changes.

## Remaining work

Every native repeated arm improves internally, but **every required external
comparison still loses**. Residual C8 is 0.615x the fresh competitor, 0.578x the
prior stronger control. MTP1 post-first C1 is about 0.650x the fresh competitor.
The >=5% external speed objective is not complete, nor are batched MTP and the
competitor best-depth/configuration requirements.

The next priority is the dominant projection kernel. `docs/native-q8-goal.md`
records an **unimplemented** FP16 Tensor Core/group32-scale research hypothesis
and its operator/reference gates. Merely eliminating more submission work will
not plausibly close the entire gap. New arithmetic/reduction must remain named
and independently qualified; no silent one-pass A8 or lower-quality substitution.

All four owned workers are terminal; all servers/profilers stopped and loopback
18948 closed. Original physical-GPU footprints restored at **3651/3189 MiB**,
both idle; unrelated services untouched. Large traces/libraries remain remote
with retained-file/build hashes. All 20 previous frozen capsules are unchanged.
