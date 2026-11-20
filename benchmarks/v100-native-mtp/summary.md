# Native one-proposal MTP on two V100s

## Decision

Implemented **experimental, opt-in, greedy one-proposal MTP** for the native
Qwen3.5-27B target, including TP and PP. It improves single-request throughput,
but is **not promoted as a universal or default acceleration**: PP regresses
at 2/4/8 requests, and TP's mean concurrent completion latency also regresses
although its cohort throughput improves. No matched external-engine win or
50 TPS claim follows. Longer chains and the larger production goal are unfinished.

## Exact workload / provenance

- WZU_Server: two Tesla V100-PCIE-32GB, SM70, PCIe/NODE topology, no NVLink.
  `CUDA_VISIBLE_DEVICES=1,0`; logical primary 0 is physical GPU 1. Explicit host
  staging uses CUDA pinned DMA; the previously failing peer route is not retried.
- Official Qwen3.5-27B revision `fc05daec18b0a78c049392ed2e771dde82bdf654`.
  Target artifact ID `3d635b87db004f42aefd6730e372e530`, SHA256
  `b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8`.
  Original-derived Q8/group-32 FP16 scales; FP16 activations/KV, FP32 GDN state.
  No original BF16-versus-Q8 corpus-quality equivalence is claimed.
- Separate original-source MTP artifact: 15 tensors, 451,341,312 file bytes,
  451,267,584 payload bytes; ID `1a4307f5e7954a929d7ad70539875915`, SHA256
  `a1a18b5f828d9b1cc1028fdc8463564aa4ffdbfe527311e1c0baa48cccfc1641`.
  Eight matrices are Q8; seven norms retain BF16 storage. Shared vocabulary is
  absent from this artifact and borrowed from the unchanged target at runtime.
- Final ABI-15 SM70 library SHA256
  `dd8476377133a506aa49ff7b25f839e6a1b9931a9774a0c5c3230025cd404af0`.
  Native CUDA source hashes and compiler arguments are in `v100/build-final/build.json`.
  Exact executed Python/native source hashes are in `v100/diagnostics-v1/executed-source.json`.
- Speed case: 23 already-formatted prompt IDs, 32 pinned greedy output IDs,
  EOS disabled, capacity 55, max forward chunk 23. PP split is 32/32 layers.
  Prompt and pinned outputs are retained in each raw report. All timed outputs
  equal those pinned IDs, and all production proposal overrides are disabled.
- For each mode and concurrency: baseline/MTP/MTP/baseline, two measured waves
  per variant, each preceded by a full-output warmup and reset. Model/draft weights
  remain resident in **both** variants. Slot allocation, load, warmup/reset,
  correctness snapshots and profiling are outside timed waves. OMP/MKL threads 4.
  This is a short deterministic benchmark, not a broad prompt/length campaign.

## Correctness first

The draft consumes token `i` paired with target **post-final-norm** hidden `i-1`;
position zero uses explicit zeros. Embedding RMS output precedes hidden RMS output
in the FC concatenation. The draft uses its own full-attention KV. Norms use the
checkpoint's zero-centered convention (`1+w`). Architecture references:
[vLLM MTP](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py),
[vLLM target](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_next.py),
[SGLang MTP](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_5_mtp.py).
These are consulted references, not external forward/runtime dependencies.

Verification consumes `[pending, proposal]`. Target GDN state/convolution history
are saved on-device first. On mismatch or EOS at the first verified selection,
restore them and the target cursor, then replay pending only. Stale attention
KV tails are hidden by committed position. On acceptance, catch up draft KV for
the accepted input and emit proposal plus bonus. Both cursors finish at prompt
length + emitted - 1; the last emitted token is unconsumed. Execution failures
poison the complete request until reset. See `docs/native-mtp-plan.md`.

Completed gates:

- Local full suite: **735 passed, 532 skipped**; final V100 full suite:
  **864 passed, 403 skipped**. Final selected Q8/MTP/TP/PP fixtures:
  **92 passed, 88 skipped**. Skips are explicit device/feature exclusions, not passes.
- Independent FP64 reconstruction from original MTP tensors, including independent
  represented-Q8 reconstruction, passes fixed relative L2 <= 0.02 / centered cosine
  >= 0.999 for normalized hidden and committed KV in **both** TP and PP. Worst
  hidden relative L2 is 0.00075970, minimum centered cosine 0.9999997114.
  Shared target head has previous independent qualification; no fresh independent
  CPU draft-head top1/corpus gate is claimed.
- Existing target two-step pinned qualification preserves previous-native logits,
  hidden/GDN/history/KV bytes and unchanged fixed CPU gates (`qualification-v2/target-qualification`).
- Final trained MTP: four prompts in each mode, exact ordinary-target output IDs
  and committed target state bytes, plus old pinned story output. Fixture coverage
  includes forced acceptance/rejection, first/interior EOS, length boundaries,
  failure/reset, lifetime after model close, tied vocabulary and C8 distinct PP
  requests. Native production generation also passes with Torch math forbidden.
- All 30 real projection/token cases preserve original output bytes after the
  two-token tile change. Real trained verification/committed-state gates are rerun
  after that change, not inferred from plausible text.
- Actual public CLI smoke generates the pinned first eight story tokens through
  the MTP wrapper; `v100/diagnostics-v1/cli-smoke.json` records depth one.

Natural proposal acceptance (same for TP and PP; diagnostic prompts):

| Prompt | Outputs | Accepted / proposed | Acceptance | Replayed inputs |
|---|---:|---:|---:|---:|
| Story | 32 | 14 / 16 | 87.50% | 2 |
| Capital, EOS enabled | 2 | 1 / 1 | 100% | 1 |
| Python | 32 | 15 / 15 | 100% | 0 |
| Arithmetic | 32 | 14 / 17 | 82.35% | 3 |

Acceptance means matching draft tokens / proposed draft tokens, not emitted
outputs or probability mass. EOS requires rollback here even on a matching draft.
At story depth one, target forward calls fall from 32 to 20, but consumed target
inputs increase from 54 to 58 because verification/replay does extra work.

## Matched unprofiled speed

**Complete-request aggregate TPS = all `32 * request_count` emitted outputs /
global cohort wall**, including prompt work. It is not per-client decode TPS.

| Mode | Requests | Baseline TPS | MTP TPS | Change | Cohort wall baseline → MTP (s) | CPU core-seconds baseline → MTP |
|---|---:|---:|---:|---:|---:|---:|
| TP | 1 | 15.49 | 18.73 | +20.9% | 2.066 → 1.709 | 2.068 → 1.711 |
| TP | 2 | 15.48 | 18.67 | +20.6% | 4.134 → 3.428 | 4.141 → 3.445 |
| TP | 4 | 15.49 | 18.51 | +19.5% | 8.263 → 6.917 | 8.293 → 6.962 |
| TP | 8 | 15.45 | 18.45 | +19.4% | 16.568 → 13.877 | 16.608 → 14.041 |
| PP | 1 | 13.18 | 14.66 | +11.2% | 2.428 → 2.182 | 2.430 → 2.185 |
| PP | 2 | 22.25 | 16.75 | **−24.7%** | 2.876 → 3.822 | 4.912 → 4.515 |
| PP | 4 | 23.35 | 17.88 | **−23.4%** | 5.483 → 7.159 | 9.926 → 9.321 |
| PP | 8 | 24.01 | 18.34 | **−23.6%** | 10.663 → 13.960 | 19.932 → 18.789 |

Single-request post-first-token output rate uses **31 / mean(native total −
native first-token-ready time)** from the same call, not unrelated prefill tests:

| Mode | Baseline decode TPS | MTP decode TPS | Change | Native TTFT baseline → MTP |
|---|---:|---:|---:|---:|
| TP | 18.94 | **24.55** | +29.6% | 423.9 → 434.8 ms |
| PP | 17.80 | **21.07** | +18.4% | 675.4 → 689.1 ms |

MTP warms its own prompt cache, so it does not improve TTFT in this case. The
native timestamp measures internal readiness, excluding Python pre-call work or
waits; the API returns the whole sequence, not a stream. Concurrent decode TPS
is deliberately absent. Request completion/CPU/global-wall timestamps are raw
separate fields, and native TTFT is not an end-user HTTP time-to-first-byte.

TP is still model-serialized; requests in the MTP path interleave at verification
boundaries. At C8, mean completion latency **worsens 9.349 → 13.588 s** despite the
improved cohort TPS. PP C8 mean completion worsens **10.558 → 13.565 s**. Higher
acceptance or lower CPU consumption therefore does not establish a serving win.

## Verification-kernel correction

Initially two-token verification used the four-token prefill tile. In a separate
matched TP1 control, MTP only improved complete-request TPS 15.48 → 15.83 and
post-first rate 18.92 → 19.68. Preserve this result, not just the final speedup.

The final SM70 default-profile dispatch uses an exact two-token specialization
only when `tokens == 2` and the configured token tile is four. Other prompt sizes
and SM89 dispatch are unchanged. Actual geometry remains 256 threads; arithmetic,
FMA/reduction order, Q8 codes and scales are unchanged. Representative Q projection
[12288,5120], T2: **0.278548 → 0.175084 ms** (37.1% lower projection latency).
This microbenchmark uses CUDA-event graph replay to amortize launch overhead;
it is not whole-model CUDA-graph support or a 37% whole-model speed claim.

The final FP16 T2 kernel uses 32 registers/thread, zero local bytes, 64 shared
bytes and CUDA-API **theoretical** eight blocks/SM (100%). T1 remains 56 registers,
50% theoretical; T4 remains 40 registers, 75%. These are resource/residency limits,
**not achieved occupancy**. This does not contradict the earlier spilled T1
forced-residency regressions: it is a different, smaller T2 specialization.

## GPU diagnostics / remaining bottleneck

100 ms NVML telemetry is filtered to measured global-wave windows using the
recorded server UTC offset, excluding warmup/load. Selected means, GPU physical
order 1 / 0:

| Scope | GPU busy baseline → MTP | Power baseline → MTP (W) | Peak used VRAM baseline → MTP (MiB) |
|---|---:|---:|---:|
| TP C1 GPU1 | 73.0 → 78.0% | 182.0 → 203.1 | 18,991 → 19,205 |
| TP C1 GPU0 | 60.9 → 62.1% | 161.2 → 177.5 | 17,223 → 17,223 |
| PP C8 GPU1 | 89.6 → 58.1% | 230.5 → 172.6 | 18,479 → 19,261 |
| PP C8 GPU0 | 94.3 → 64.9% | 233.4 → 181.8 | 19,391 → 20,327 |

All 1/2/4/8 variants retain busy/controller-busy, power, VRAM, temperature and
SM/memory clock mean/min/max in `metrics.json`; raw CSVs are retained. Clocks
vary naturally (e.g. PP C8 baseline SM 1260–1372 MHz, MTP 1260–1380 MHz on GPU1).
No clocks, driver settings, privileges or unrelated services were changed.
GPU busy/controller-busy percentages are not SM occupancy/DRAM-bandwidth counters.

Separate Nsight captures use 23 prompt / **8 output** tokens, warmup outside the
`native_unsplit` range, `--cuda-event-trace=false`, no CPU sampling. They are
not speed baselines and are never combined with the 32-output throughput table.

- TP C1: Q8 projections are 92.08% of **summed kernel duration**; process-activity
  timeline gaps 74.33 ms within a 722.02 ms activity span. Concurrent two-device
  kernel union overlap is 386.83 ms; kernel sums may exceed wall time.
- PP C8: Q8 projections are 91.82% of summed kernel duration; gaps 46.08 ms within
  5.435 s captured activity span; simultaneous two-device kernel union 2.590 s.
  These gaps describe captured activities of this process, not whole-device idle.
- PP's shared draft/vocabulary locks and extra dependency stages are a plausible
  source of lost pipeline overlap, consistent with lower sampled GPU busy. The
  traces do not isolate the causal cost of each lock or establish a bandwidth/
  occupancy limit. A batch-aware draft/verification scheduler is the next target.
- Counter policy remains `RmProfilingAdminOnly: 1`; achieved occupancy, compute
  utilization, achieved DRAM bandwidth and cache/stall counters are **unavailable**.
  No repeated NCU campaign under the unchanged denial policy. CUDA residency API
  and Nsight timeline overlap are not substitutes for these missing counters.

## Memory and production scope

Additional resident shared MTP model allocation: **455,514,240 bytes** (~434 MiB),
counted once, on TP primary or PP last stage. Embedding/head are shared, not
reuploaded. At benchmark max_tokens 23, each MTP request adds 19,104,888 draft
bytes plus 153,944,064 total target backup bytes. PP splits backups equally,
adds 235,704 first-stage embedding/ID bytes and 235,520 pinned bridge bytes.
Base `info()` fields describe target allocations only; named MTP/backup fields
are additional. Target model bytes and TP model-global counters must not be
summed once per slot. VRAM telemetry includes unrelated existing allocations,
allocator overhead and intentionally resident draft weights in both variants.

Fresh/reset generation only, one proposal, greedy, resident weights. No MTP2/3,
probabilistic acceptance/sampling, arbitrary continuation, continuous batching,
paged KV, HTTP streaming, whole-model graphs, long-context campaign or SM89/
other-family qualification. Short EOS can cost extra even at 100% acceptance.
Keep MTP optional; for this workload prefer MTP TP at C1, ordinary PP at C2–8.
That is a measured local choice, not an automatic universal policy.

## Reproduce and evidence integrity

Use the source in this capsule and the exact remote library/model artifacts:

```bash
CUDA_VISIBLE_DEVICES=1,0 PYTHONPATH=src:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python tools/benchmark_qwen_mtp.py /path/to/target.ninfer \
  --mtp-artifact /path/to/mtp-q8.ninfer --library /path/to/libventuri_blocks.so \
  --case /path/to/pinned-case.json --mode tp --concurrency 1,2,4,8 \
  --output-directory /path/to/new-output
```

Choose `--mode pp` for the matched pipeline campaign. See archived worker/launcher
files for exact server paths, environment and commands. Public CLI usage is in
the root README; actual tested CLI output is in the diagnostics directory.
Recompute evidence-only summaries with `python benchmarks/v100-native-mtp/summarize.py`.

Preserved outcomes:

- Initial fixture/export workers completed, original full target qualification
  passed. The v2 overall diagnostic worker **failed** with `KeyError: centered_cosine`
  because its reporting helper omitted that field. Its incomplete report, failure,
  logs and historical source remain in `qualification-v2`; it is not a successful
  complete MTP qualification. Explicit cosine calculation was repaired with a
  CPU regression test; v3 completed, and final v4 reran both block references.
- Control/final native source variants and the initial four-token-tile engine
  result remain retained. The old immutable target and previous evidence capsules
  are not rewritten. Large `.pt`, SQLite and `.nsys-rep` files remain remotely;
  exact locations/sizes/hashes are listed in `remote-retained-files.json`.
- Final main and separate profiler/CLI workers both have terminal exit 0 and
  completed reports. GPU memory returns to the original service footprint:
  physical GPU0 3651 MiB, GPU1 3189 MiB. Owned test/profiler GPU processes have ended.
- No commit/push, no local Rust compilation and no larger-goal completion claim.
  Source/evidence hashes, terminal checks and prior-capsule verification are frozen
  separately. Failed attempts remain visible rather than overwritten as success.
