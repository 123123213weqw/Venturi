# V100 K-tiled activation staging: retained, external goal incomplete

## Result

Three-repeat matched native engine measurements retain the K-tiled shader:

| Concurrent requests | Control HTTP TPS | Tiled HTTP TPS | Gain |
|---:|---:|---:|---:|
| 1 | 19.393 | 20.192 | 4.12% |
| 2 | 29.634 | 31.478 | 6.22% |
| 4 | 35.548 | 45.214 | 27.19% |
| 8 | 43.171 | 54.718 | 26.75% |

**C1/C2 gains are prefill gains**, not decode improvements. C1 ordinary native
post-first is 23.081 -> 23.039 TPS (-.18%); C2 native post-first cohort is
39.116 -> 38.952 (-.42%). C4/C8 post-first cohort improves
49.603 -> 61.854 / 66.006 -> 80.741 TPS. These cohort rates exclude first outputs
and use the final native finish minus the common first-token timestamp, not
summed client durations or competitor HTTP throughput.

MTP1 C1 whole-request HTTP improves **24.966 -> 26.224 TPS (+5.04%)**, again from
prefill: post-first native decode is **31.847 -> 31.736 TPS (-.35%)**. Do not
attribute a new single-token or speculative decode speedup to this change.
The stronger control measurement and the small regressions remain in evidence.

All external cells still lose. Reused pinned llama.cpp non-spec C1/2/4/8 HTTP
controls are 24.54/37.91/50.25/67.29 TPS; best tested MTP depth 2 is HTTP 30.17
and reported post-first 49.80 TPS. No claim of >=5% external superiority,
completion, commit/push, local Rust/CUDA build or universally faster models.

## What changed

The previous shader repeatedly addressed activation codes/scales in global
memory. `q8_shared.cuh` now also offers a K-tiled operator: each CTA cooperatively
stages 32 scale groups (1,024 columns) in separate int4-half planes plus unchanged
FP32 scales, computes the original lane's integer/group-FMA work, then advances
K. Both barriers are taken by **every warp**, including invalid rows and token
and K tails. Integer products, residual activation passes, FP32 FMA order, warp
reduction and FP16 output rounding remain unchanged. No Tensor Core dispatch.

Tile 4 / tile 8 use at most 9,216 / 18,432 dynamic shared bytes at two passes,
independent of K. No function-attribute mutation or new global workspace.
Original quantization scratch stays 1,253,376 bytes/device at 32 projection rows.

Native `Model::project` selects this only for SM70 residual T>=3. The measured
policy uses token tile 8 / 16 row warps when T>=8 and shard rows>=1,024;
otherwise tile 4 / 8 row warps. Small grids avoid the regressing tile-8 choice.
T1, existing T2 staging, SM89, default FP16 arithmetic, PP, input-sharing scopes,
peer host transport and both event fences are unchanged. Native model/token
execution remains C++, not Python. Blocks ABI 16 and eight native build sources.

Control SHA256:
`1aba1423dc511b7cb17f03c0a4b71948f5fdd4f836e0dcfe9e0190b15bddbee7`.
Candidate:
`/home/data/wangyue/artifacts/venturi-tiled-staging-model-v3/build/398e4f4a1dd037ea/libventuri_blocks.so`,
SHA256 `a6599e636cc0c018261310de7d83e1e8dd41d6edcc635f9161402a89f7775a3f`.

## Operator evidence before integration

Eight real TP-first shapes at T4/8/23/32: 32 combinations, nine arms, three
repeats per arm; each wave contains nine graph-contained calls after warmup,
with alternate repeat order reversed. Original FP16 and residual arithmetic
are separately labelled. All six tiled variants match the residual control's
output/workspace bytes, with independent sampled FP64 gates.

The retained fixed geometry's median micro speedups are about 1.56x at T4,
1.48x at T8, 1.73x at T23 and 1.81x at T32. The small `[512,5120]` shard at T8
regresses about 10% with tile8/row16; tile4/row8 instead improves about 12%.
That counterexample is retained and determines the conservative small-row policy.
These are projection microbenchmarks, not serving TPS or general autotuning.
Raw event/wall/CPU measurements and broad NVML samples are retained; short event
windows do not yield reliable per-wave NVML utilization.

Initial/native-model operator selections: **324 passed**, including 182 new
K-tiled numeric/tail/guard/graph/alias/preflight/invalid-geometry cases, both
activation pass counts, nonfinite and zero graph replays, plus prior operators.
K tails, token tails, partial row blocks and wide-K cases pass byte and FP64 gates.

## Trained correctness and reference reuse

Both original `fp16` and residual candidates match the pinned previous complete
signatures: 98 full-vocabulary vectors per profile, four original greedy
trajectories, scalar final private states, eight heterogeneous batched requests,
and selected-reset/continuation logits, positions and all eight private caches.
Native generation forbids Python/Torch math. New trained MTP checks pass four
cases against ordinary-target output and committed state, including natural
acceptance/replay/EOS and zero forced proposals. Draft-block FP16 arithmetic is
unchanged; its shared target vocabulary head retains the unchanged T1 profile.

The qualifier now explicitly reuses **complete** prior control records rather
than re-executing the unchanged control model. It verifies the frozen prior
report/library/artifact identity and the pinned original four-case file SHA256
`fd5f82c3a4b4d22dfa05d577ba50c67019b47646440a54616e565d7bdf008d0b`.
It still executes both candidate profiles and compares every logit/cache/reset
signature, not only greedy tokens. Fixed geometry is resident FP16, 32 projection
rows, capacity 256, host TP on logical devices 0/1. The new records equal the
previous qualified candidate records exactly.

Erratum: this run's inherited `artifact_hash_scope` prose says both controls
are loaded. In reuse mode that sentence is stale: **only candidates are
re-executed**; `control_mode` correctly records reused signatures. The current
helper fixes that prose for future runs. Executed source and raw report remain
preserved rather than rewritten.

The original independent CPU gates remain top-1, relative L2<=.02, centered
cosine>=.999, KL<=.002, preserved through exact bytes. The unchanged MTP draft's
represented-Q8 hidden/KV CPU reference is explicitly hash-pinned and reused,
not rerun; original shared-head and target reference qualifications remain
separate. No fresh BF16/corpus-quality campaign or weakened threshold claim.

Final V100 suite: **1,438 passed / 440 skipped**. Earlier full suites had
1,422 / 456 because 16 scoped-input-sharing tests lacked their opt-in environment;
those tests are enabled and pass in the final suite. Local non-GPU:
**763 passed / 1,115 skipped**. Local skips are not GPU passes.

## Engine work, timing and CPU/latency

Two V100-PCIE-32GB devices, logical `1,0`, pinned-host TP; same original-derived
Q8 artifact, explicit two-pass residual activation profile, FP16 KV; 23 formatted
prompt IDs, 32 fixed greedy outputs, ignore EOS for the speed case, no prompt
cache reuse; eight slots at capacity 256; projection row limit 32; 25 ms batch
admission included in HTTP wall. Loading, warmup and diagnostics are excluded.
All IDs match the pinned case and fixed cohorts remain unsplit. One-repeat
1/2/8 and MTP1 screens precede conditional three-repeat 1/2/4/8 promotion.

HTTP TPS is all emitted outputs / **global cohort elapsed wall**, never summed
concurrent latencies. Native TTFT excludes prior queue/HTTP/reset and includes
whole-cohort prefill; it is not client streaming TTFT.

| Requests | Whole wall seconds old -> new | CPU core-seconds old -> new | Mean client completion seconds old -> new | Native TTFT seconds old -> new |
|---:|---:|---:|---:|---:|
| 1 | 1.65009 -> 1.58482 | 1.623 -> 1.560 | 1.64940 -> 1.58396 | .26688 -> .19798 |
| 2 | 2.15965 -> 2.03319 | 2.137 -> 2.007 | 2.15856 -> 2.03225 | .53249 -> .39781 |
| 4 | 3.60073 -> 2.83097 | 3.577 -> 2.797 | 3.59867 -> 2.82884 | 1.05252 -> .77664 |
| 8 | 5.92995 -> 4.67857 | 5.930 -> 4.683 | 5.92562 -> 4.67573 | 2.13725 -> 1.57413 |

MTP1 whole wall 1.28177 -> 1.22026 s; CPU 1.283 -> 1.220 core-seconds;
client completion 1.28113 -> 1.21944 s; native TTFT .27316 -> .20807 s.
Acceptance stays 14/16 (87.5%), two rollback/replays, no forced proposals.
Post-first is 31 / same-call native generation-minus-first-token duration;
competitor reported post-first is 31 / mean `predicted_ms`, not HTTP.
This is not MTP2, continuous/paged batching or a client-streaming benchmark.

The preceding stronger external non-spec and depth 1–4 controls are explicitly
reused by report hashes. No new competitor run, conveniently lower control,
exhaustive best-configuration claim or successful parallel MTP comparison.
Earlier failed speculative configurations remain failures, not zeros or wins.

## GPU evidence, separate profiler runs

Unprofiled C8 NVML primary/peer GPU busy 83.91/59.81 -> 79.82/48.80%; controller
busy 22.64/19.91 -> 27.99/24.41% (**not GB/s**); power 132.32/101.16 ->
148.13/111.20 W; temperature 43.70/40.42 -> 44.79/40.96 C. Device-used memory
20,235/17,227 MiB unchanged, including unrelated allocations; SM/memory clocks
1,380/877 MHz unchanged. C4 busy 80.88/59.46 -> 76.01/47.90%; raw windows,
MTP telemetry, sample ranges and trial latency variation are in `metrics.json`.
Lower sampled busy does not mean worse throughput or measure achieved occupancy.

Separate **23-prompt/8-output** traces, not serving TPS: fresh C4 control and
all candidates; C1/C2/C8/MTP controls reused from the exact prior T2 library.
Kernel counts are unchanged. C4 summed dot time **1,763.31 -> 978.33 ms**,
GDN 233.83 -> 233.91, gaps 145.44 -> 148.52; host span 1,661.04 -> 1,269.17.
C8 dot **3,237.65 -> 1,794.75 ms**, GDN 490.70 -> 490.77, gaps 211.91 ->
214.28; host span 3,046.46 -> 2,325.73. C4/C8 use 8,016/10,416 tiled dots,
with 23,588/37,306 total kernels. Kernel sums overlap across GPUs and are not
elapsed time, compute throughput or utilization.

C1 has 800 tiled prefill dots, unchanged T1 decode; host span 605.65 -> 537.70 ms.
C2 has 1,600 tiled prefill dots, unchanged T2 staging; span 926.36 -> 789.45.
MTP1 has 800 tiled prefill dots, unchanged verification; span 511.24 -> 441.77.
No Tensor Core kernels or copy/fence removal. H2D/D2H traffic remains unchanged
(about 1.990 GB each direction in the C8 trace). Remaining GDN/transport/launch
work is not accelerated by this shader change.

Actual selected launch resources: tile4/row8 uses 256 threads, 48 registers,
9,216 shared bytes, no spill, **62.5% theoretical** occupancy; tile8/row16 uses
512 threads, 49 registers, 18,432 shared bytes, no spill, **50% theoretical**.
Prior global T4/T8 register limits were 78/124 and 37.5%/25% theoretical.
These residency ceilings are not achieved occupancy or causal cache/bandwidth
proof. Achieved occupancy, compute/DRAM throughput and cache/stall counters
remain unavailable under `RmProfilingAdminOnly: 1`; no repeated denied campaign
or profiling-policy change.

## Provenance, failure and next work

Official revision `fc05daec18b0a78c049392ed2e771dde82bdf654`, target ID
`3d635b87db004f42aefd6730e372e530`, previously verified target SHA256
`b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8`;
MTP SHA256 `a1a18b5f828d9b1cc1028fdc8463564aa4ffdbfe527311e1c0baa48cccfc1641`;
GGUF SHA256 `eb07277161915232e227a38608fc3ea80c0c17dc1c1583a9f3c5671edb6b5513`.
Reused verified payload hashes, not fresh scans. Native group32 Q8/F16 scales,
compensated two-pass A8/F32 scales/reduction and small-parameter handling differ
from GGUF Q8_0/Q8_1 arithmetic. Q8 labels are not identical precision/arithmetic.

One initial launcher failed because generated benchmark scripts were missing,
after successful native build/operator/full-suite tests. The corrected source
snapshot reused the same binary only after all eight native hashes matched;
raw failure, commands and source omissions are retained. No failed timing is
counted as a speedup. Subsequent composite commands use fail-fast execution.

Next target the unchanged projection-group synchronization/transport and single-
request path, rather than claim more T1 speed from this prefill shader. Exact
same-input projection fusion and more complete intermediate/head tensor
partitioning need measured experiments; FP32 partial-reduction designs require
explicit arithmetic labels and fresh fixed quality gates. Native multi-proposal
MTP remains unimplemented and needs correct draft/cache/rollback semantics.

All owned workers are terminal (the preserved initial failure is exit 1; recovery
and subsequent jobs exit 0); port 18953 closed; unrelated allocations restored
to idle 3,651/3,189 MiB. Large binaries/traces stay remote with retention hashes.
Current and executed source snapshots, cached-reference pins and prior 24 frozen
capsules are verified without rewriting old evidence. The external goal remains
active and unachieved.
