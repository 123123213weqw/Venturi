# Native MTP2: correct transactions, then remove target replay

2026-10-20, two V100-PCIE-32GB GPUs. **Progress; the external goal is not complete.**
Depth1 and replay remain defaults; depth2/prefix recovery are explicit opt-ins.
No commit/push, local CUDA/Rust build, driver-policy change or other-service stop.

## Result

Three unprofiled repeats per cell, 23 fixed formatted prompt IDs, 32 fixed greedy
outputs/request, ignore-EOS speed case, eight prepared slots/capacity 256,
projection rows 32, resident host TP on physical GPUs 1,0, FP16 KV. Each rate is
all emitted outputs / global HTTP cohort wall, **not per-client TPS or summed
request time**. Loading, allocation, preparation, warmup and profiling are
outside timed waves. HTTP handler reset, admission/queueing, generation and reply
are inside client time. Global harness wall also includes client response-file
archival/validation and future collection; individual client completion timestamps
precede that postprocessing. Scalar MTP is not target cohort batching.

| Requests | Fresh pinned MTP1 TPS | MTP2 prefix TPS | Gain | CPU core-s, old -> new | Mean client completion s, old -> new |
|---:|---:|---:|---:|---:|---:|
| 1 | 29.411 | 33.879 | 15.19% | 1.083 -> 0.943 | 1.087 -> 0.944 |
| 2 | 29.294 | 33.360 | 13.88% | 2.200 -> 1.940 | 2.182 -> 1.916 |
| 4 | 29.746 | 33.483 | 12.56% | 4.473 -> 3.953 | 3.792 -> 3.746 |
| 8 | 29.260 | 33.808 | 15.54% | 9.170 -> 8.030 | 8.566 -> 7.314 |

Single native post-first TPS (31 outputs / mean same-call total-minus-first-token
wall) improves **35.718 -> 42.830**, 19.91%. Internal native TTFT is about 0.185 s
in both arms; it is not streaming client TTFT. C8 global wall falls 8.749 -> 7.572 s,
maximum observed client completion 8.876 -> 7.676 s. No universal CPU/latency claim
outside these fixed cohorts. Trial values and all timing denominators are in
`metrics.json` and raw timed reports.

### Retain the negative ablation and stronger controls

Fresh C1/C8 screening:

- Pinned MTP1: 29.474/29.307 HTTP TPS.
- Candidate default MTP1: 29.458/29.487; no meaningful default-path gain.
- Candidate MTP2 **checkpoint replay**: 25.741/25.947; slower, not promoted.
- Candidate MTP2 **prefix recovery**: 33.898/34.402; promoted only to the full
  three-repeat experiment, not enabled by default.

The original replay-binary screen is independently preserved at 25.725/25.888.
This prevents relabelling the first attempt as a successful optimization.

Explicitly reused, immutable three-repeat stronger controls:

| Path | C1 | C2 | C4 | C8 | Scope |
|---|---:|---:|---:|---:|---|
| Native ordinary batch, rows256 | 22.925 | 36.291 | 53.248 | 65.407 | HTTP complete-cohort TPS |
| Pinned llama.cpp tensor, non-spec | 24.536 | 37.907 | 50.254 | 67.287 | HTTP complete-cohort TPS |
| Best tested pinned llama.cpp MTP2 | 30.173 | — | — | — | HTTP complete-request TPS |

Native prefix MTP2 C1 HTTP is **12.28% above** that best-tested external MTP2
control, but native post-first 42.830 is **14.00% below** external reported 49.803.
These are different timing scopes, not conflicting results. Scalar MTP remains
below ordinary native batching at C2/4/8. The external goal requires single decode
and every parallel cell >=5% faster; it is **not met**. The failed prior parallel
external MTP configurations stay failures, not zero-rate controls. This is not an
exhaustive best-configuration claim for the competitor.

External binary SHA256:
`e5b1b5782cd5b4acec5316ac5bd40123e0b0c9826643ae4148b0ec6e359f5fa2`.
Native and GGUF Q8 scales/rounding/small parameters differ; external Q8_0/Q8_1
and native group32 Q8 plus two A8 residual passes are **not identical precision**.
No reduced arithmetic, KV quantization, forced acceptance or weaker quality gate
was substituted to obtain these gains.

### Timing-scope correction to earlier HTTP prose

Some earlier frozen summaries described reset as excluded from HTTP waves. The
unchanged adapter resets each slot inside its request handler, so HTTP includes
that reset; only same-call native generation/TTFT and separate profiler ranges
exclude it. Global benchmark wall also includes the client archival/validation
noted above. Raw timings, source, comparisons and older capsule hashes are
unchanged; do not reinterpret those HTTP rates as pure GPU generation.

## Why prefix recovery helps

Two-proposal drafting recurs on the normalized draft output. Verify pending input
plus two proposals, accept only a matching prefix, and keep the final emitted
correction/bonus/EOS unconsumed. Attention cursor rewind alone cannot restore a
GDN recurrent target. The first implementation saved pre-state, restored it and
replayed pending plus accepted inputs. The story case incurred 12 replayed inputs
and 0.298 s restore/replay in trained qualification.

The new request-owned trace captures GDN state and convolution history after
input 1/input 2. A rejection/EOS restores the correct prefix and cursor **without a
new target forward**. Prefix1 reuses the old backup; prefix2 adds storage. Draft
KV is still rewound and repaired with verified target hidden rows. Full acceptance
already has the correct final target state. Failure clears temporary descriptors
and poisons both states until reset. Native locks/transport fences are preserved.

Fresh trained story qualification: 12 -> 0 replayed inputs, restore/recovery wall
about 0.298 -> 0.0044 s. Verification wall stays about 0.61 s. These are nested native
wall counters, not an additive GPU timing breakdown or speed-test substitutes.
Natural story acceptance is 17/26=65.38% of **all drafted proposals**; depths accept
[11,6] of proposed [13,13], verified [13,11]. The unused second-depth suffix counts
in the overall denominator. Other trained cases accept 1/2, 20/21 and 19/22; all have
zero target replay in prefix mode. No forced proposals occur in trained/timed runs.

## Correctness and scope

- Final V100 suite: **1772 passed /448 skipped**; selected prefix/MTP suite:
  **113 /27**. Local CPU suite: **776 /1444**; CUDA skips are not passes.
- Forced acceptance/rejection at both depths, EOS and budgets 1/2/3/4/11, original
  FP16 and supported residual single/TP/PP modes, target state vs ordinary greedy,
  repaired draft KV vs MTP1, failure/reset/lifetime and 2/8-request ownership tests.
- GDN trace output/final/every prefix state is byte-equal to legacy GDN and passes
  an independent FP64 state oracle for DK64/128/256, packed/unpacked, normalized/
  unnormalized. Convolution-history bit copies have independent references/guards.
- Four trained original-source trajectories pass exact target IDs and complete
  committed GDN/history/KV signatures, fresh ordinary controls and repaired draft
  KV equality to fresh MTP1. Three recursive draft steps pass independent
  represented-weight FP64 gates (relative-L2<=.02, centered-cosine>=.999); observed
  hidden relative errors are approximately0.00074/0.00070/0.00058.
- Existing target teacher-vector/CPU and old draft-block references are pinned
  and reused; no unchanged 98-vector qualification campaign is rerun or called new.
  This is not original-BF16/corpus quality, arbitrary sampling/continuation,
  SM89 qualification, production continuous batching/paging or long-context proof.
- Optional exports preserve ABI16 structures and old depth-one compatibility.
  CLI preparation failures close sessions explicitly; old libraries reject missing
  depth2/prefix exports clearly instead of invoking a Python numerical fallback.

The trained qualification command completed successfully, but its wrapper then
looked up `recovery` instead of `target_recovery`. That failure and downstream
stopped wrappers are preserved. A reporting-only recovery validates and reuses
the exact completed report by SHA256; it does not rerun or waive a numerical gate.
One uncaught predecessor assertion lacked an exit-code file; its traceback and
observed terminal process state are recorded rather than inventing an OS status.

## Memory, CPU and GPU diagnostics

New logical prefix storage: 153,944,064 bytes/primary slot (146.8125 MiB), or
1174.5 MiB across eight slots. These are **additional** to old MTP request/backup
fields. Actual primary NVML device usage is **22435 -> 23993 MiB (+1558)**;
secondary stays 17227 MiB. The extra 383.5 MiB above logical payload has not been
separately attributed; allocator/residency overhead is plausible, not proven.
Do not use logical counters as physical admission bounds. Reset retains buffers;
preparation is outside timing, and completed workers release their allocations.

Timed-window NVML samples (all include pre-existing unrelated allocations):

| Arm / requests | Primary / secondary GPU busy mean | Power W mean, primary / secondary |
|---|---:|---:|
| MTP1 C1 | 73.6% /52.9% | 127.8 /105.9 |
| Prefix MTP2 C1 | 73.1% /51.5% | 150.8 /108.4 |
| MTP1 C8 | 72.5% /52.3% | 129.5 /105.3 |
| Prefix MTP2 C8 | 73.9% /51.4% | 146.1 /114.9 |

Temperature ranges 36–49 C. SM clocks mostly 1380 MHz; candidate C4 primary includes
1335 MHz samples. Memory clocks and controller busy plus all per-cell samples are
retained in metrics/raw CSV. Faster execution does not imply lower instantaneous
power or higher achieved occupancy. CPU core-seconds and request latency are
reported separately above.

Separate short 23-prompt/8-output C1/C8 profiles are retained for MTP1, replay MTP2
and prefix MTP2. **That short story prefix has no rejections**, so it cannot prove
the full 32-output rollback savings. Additional same-binary 23/32 profiles explicitly
compare replay vs prefix at C1/C8 and exercise the affected later rejections.
See profiling details/metrics below for actual attribution and static resources.
Nsight 2025.1.3 uses CUDA event tracing disabled; instrumentation is excluded from
all unprofiled TPS. RmProfilingAdminOnly remains 1: achieved SM occupancy,
compute/DRAM/cache/stall counters are unavailable. Theoretical residency, NVML
busy and summed kernel times are not replacements.

## Reproduction and evidence

`design.md` records transactions, counters and memory; `source-notes.md` records
primary architecture references; `next-bottlenecks.md` records unimplemented next
work. `summarize.py` recomputes output/global-wall rates and timed-window telemetry
from raw evidence; `metrics.json` includes trial rates, CPU/latency, acceptance,
stronger reused controls and profile attribution. Remote traces/libraries stay
retained by exact path/SHA256; local build/resource metadata and launch commands
are included. All executed source versions and differences are preserved.

Qualified baseline library:
`e8090c0531b3b794f34ca267baf1b61fcb84ba6d58c1e4550795ad330ae6965a`.
Replay implementation:
`17e64d758d64cf682b784a14bf80b4844398877628e9239c3b38420bb13afab1`.
Prefix-capable library:
`9a4d8655c9f781746ab2240c86e51c79d02a3653dc09c7262a51130d661c0894`.

Official source revision:
`fc05daec18b0a78c049392ed2e771dde82bdf654`.
Target artifact SHA256:
`b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8`;
draft SHA256:
`a1a18b5f828d9b1cc1028fdc8463564aa4ffdbfe527311e1c0baa48cccfc1641`.
Original-derived GGUF SHA256:
`eb07277161915232e227a38608fc3ea80c0c17dc1c1583a9f3c5671edb6b5513`.

All 28 prior capsules remain unchanged (the legacy decoder manifest's previously
recorded unavailable paths remain explicit). Final process/port/GPU-footprint audit
and source/evidence manifests make this a frozen milestone, not goal completion.
