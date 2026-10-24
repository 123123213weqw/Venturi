# INT16 activation DP2A: correctness passes, dominant speed gate fails

2026-10-20. **Research-only, not promoted into model dispatch.** The previous
native column-MLP integration was real progress: single post-first decode47.629
TPS versus the full52.293 TPS target floor. This distinct prerequisite tests
whether different integer activation packing can close that remaining9.79% gap.
It does not. No new full-model quality, serving TPS or goal-completion claim.

## Implementation and scope

Separate remotely compiled SM70 SO, never included by production `blocks.cu` or
selected by a model constructor. The qualified production library and its10
native dependencies, target/draft source and all model defaults remain unchanged.
The explicit research Python module loads that separate library; no automatic
build, fallback, checkpoint mutation or external model forward is used.

Candidate: group32 FP16 activations -> one INT16 pass, nearest-even codes clamped
to[-32767,32767], FP32 scale maxabs/32767. Stored original Q8 codes and FP16 scales
are unchanged. Signed DP2A packs two INT16 values in srcA and selects lower/upper
INT8 pairs from srcB; four independent integer chains evaluate each32-product
group, then one FP32 scale/FMA and the usual warp reduction. Full outputs are
FP16; column-down partials remain FP32. No FP16 partial transport substitution.
Invalid groups get zero codes/NaN scales; logical tails get zero codes. This is
**different arithmetic**, not prior-profile byte equivalence or model approval.

Compared with two residual-A8 passes, activation workspace falls72 ->68 bytes/
group and quantization runs one pass. Both still need16 integer-dot instructions/
32 weights; original weight traffic is unchanged. Static SASS confirms actual
`IDP.2A.LO/HI.S16.S8`, not a silent FP16/DP4A fallback. The560 static DP2A entries
across compiled variants are not executed-instruction counts.

## Correctness and provenance

- Standalone library SHA256:
  `96b41c84b209a97e0b7aa196e1b2cedb2e6d883ce55e4b0c79532eebffd2c876`.
  Control SO SHA256:
  `19ee200feef03d2ddd10e313b680c393fbf8ee170f367d08d0bbd6b5edaebc13`.
- **59 CPU/GPU tests pass** before timing: signed packing/lane mapping and all
  INT16/INT8 extrema, exact CPU-reference quantizer codes/scales, tails/padding,
  strides/guards, both FP16 and FP32 projection outputs, fused2/3 matrices,
  graph replay/changing input/no Torch math during native execution, zero,
  smallest FP16 subnormal, maximum finite FP16, NaN/Inf and invalid geometry.
- Actual checkpoint cells check up to32 sampled output rows/matrix against
  independent FP64 reconstructions of **each** activation profile, and against
  unquantized FP16-activation/encoded-weight FP64 output. Each profile passes
  the declared FP16<0.001 or FP32<2e-6 reconstruction limit. Maximum sampled
  relative L2 versus unquantized activation is0.000249815 control and0.000248982
  candidate. Random normalized-state-shaped inputs are **not trained hidden
  activation samples**, original BF16 evaluation, full model logits or corpus
  quality evidence.
- Updated resource parser:12 tests pass. Final local suite: **815 passed /
  2,076 skipped**; Ruff passes. The unchanged production full GPU suite and trained
  teacher/private-state/MTP evidence are explicitly reused from the preceding
  frozen capsule; they do not qualify this unintegrated arithmetic as a model.
- Same target artifact `3d635b87db004f42aefd6730e372e530`, official parent revision
  `fc05daec18b0a78c049392ed2e771dde82bdf654`. The preceding capsule's fresh full
  target hash/original-source audit is reused; this pilot does not repeat that
  28.6GB full-file scan. Actual encoded source planes are read without requantizing.

## Finite unprofiled comparison

One active V100 (logical0 =physical1) measures one rank's actual native TP
projection geometry. Both arms share **the same resident weight tensors**, FP16
inputs, output dtype, preallocated workspace and launch geometry family. No
cross-device transport, norm/residual, caches, tokenizer, scheduler, HTTP, TTFT or
complete model is timed. T means token rows, **not concurrent requests**.

Each call includes quantization plus complete fused/row/FP32-partial projection.
Graphs hold8 calls;128 replays/trial ->1024 calls. Three alternating trials per
arm/cell, warmup first, no profiler. CUDA events report joined single-device
operator latency; CPU core-seconds and host wall are separate. Graphs reduce
Python noise, not evidence that whole-model CUDA graphs are enabled.

| Actual row/column geometry | T1 old -> new us (ratio) | T3 old -> new us (ratio) | T8 old -> new us (ratio) |
|---|---:|---:|---:|
|Fused MLP gate/up,2x[8704,5120]|114.552 ->115.078 (0.99543x)|132.723 ->133.542 (0.99387x)|219.026 ->215.608 (1.01585x)|
|Fused GDN QKV/Z,[5120/3072,5120]|57.438 ->57.794 (0.99383x)|71.326 ->70.722 (1.00854x)|114.899 ->112.513 (1.02121x)|
|Fused attention Q/K/V,[6144/512/512,5120]|51.932 ->52.427 (0.99057x)|60.152 ->60.255 (0.99829x)|99.661 ->97.687 (1.02021x)|
|Column-down FP32,[5120,8704]|64.371 ->63.871 (1.00781x)|76.535 ->73.586 (1.04008x)|116.587 ->115.232 (1.01176x)|
|Row GDN output,[2560,6144]|27.327 ->25.731 (1.06205x)|30.347 ->30.473 (0.99586x)|46.057 ->44.319 (1.03921x)|
|Row attention output,[2560,6144]|27.329 ->25.756 (1.06109x)|30.356 ->30.438 (0.99732x)|45.995 ->44.322 (1.03774x)|

Median of the nine fused ratios: **0.998288x**, below **1.05x**. Minimum of all18
cells:0.990566x, so the separate >=0.98 no-regression gate passes, but promotion
still **fails**. Do not cherry-pick the~6% smaller-output gains or4% partial gain
and bypass the dominant fused gate. No model integration, teacher/MTP rerun,
C1/C8 serving screen or expensive1/2/4/8 engine matrix is launched for this pilot.

CPU/wall examples (core-us/call and wall-us/call, respectively):

| Cell | CPU control -> candidate | Wall control -> candidate |
|---|---:|---:|
|Fused MLP T1|114.84 ->114.87|114.84 ->115.31|
|Fused MLP T3|132.91 ->133.68|132.83 ->133.61|
|Fused MLP T8|219.57 ->215.98|219.43 ->215.89|
|Column-down T3|76.55 ->73.67|76.58 ->73.62|

All18 cells, individual event-time trials, CPU/wall and quality/telemetry values
are retained in `metrics.json` and the raw `v100/campaign-v2/timed/report.json`.
Operator throughput ratios must never be substituted for complete-request TPS.

## Separate profiling and occupancy limits

T1/3/8 MLP-fused and FP32-down traces contain16 **direct** calls/arm/range,
separate from graph timing. At MLP T3, quantization sum falls0.054912 ->0.047200ms,
but projection rises2.177828 ->2.219076ms. One-pass quantization is cheaper, yet
the dominant projection does not become faster. T8 projection improves slightly;
that is consistent with the small unprofiled T8 gain, not a new model milestone.
Instrumented FP32-down T3 differs somewhat from the unprofiled4% result; only
unprofiled measurements enter the promotion gate.

Theoretical residency/resource examples from exact compiled symbols:

| Kernel | Registers control -> candidate | Threads | Dynamic shared bytes old -> new | Theoretical occupancy old -> new |
|---|---:|---:|---:|---:|
|Fused T1|72 ->54|128|0 ->0|43.75% ->56.25%|
|Fused T3 (T4 tile)|53 ->51|256|9216 ->8704|50% ->50%|
|FP32 down T3 (T4 tile)|44 ->51|256|9216 ->8704|62.5% ->50%|
|Fused T8|56 ->54|512|18432 ->17408|50% ->50%|

Reported local-memory bytes are zero. Better theoretical occupancy at T1 still
gives flat/slower measured time; worse theoretical occupancy in partial T3 still
gives a measured gain. Neither proves achieved occupancy or bottleneck cause.
Each library's profile join matches its own exact symbols/register/shared usage;
the other library's kernels are explicitly marked absent, never assigned false
resource data. Policy `RmProfilingAdminOnly:1` is unchanged; achieved occupancy,
DRAM bandwidth, cache/stall counters remain unavailable. No privilege edits or
unchanged denied counter campaign. Nsight reports CPU IP/backtrace/context-switch
sampling unsupported and disables it; CUDA/NVTX tracing succeeds.

NVML100ms samples are sparse/lagging (e.g.3/4/6 samples per MLP arm at T1/3/8).
PhysicalGPU1 uses3687MiB including its3189MiB unrelated baseline; GPU0 remains
at3651MiB. MLP T8 mean busy86.3/100%, mean power247.0/239.8W, temperature maxima
43/43C for control/candidate. Busy percentages are **not** achieved SM occupancy
or DRAM bandwidth. T1 samples include1230 ->1380MHz clock ramp; T3 holds1380MHz,
T8 control samples minimum1372MHz/candidate1380MHz; memory877MHz. All per-cell
clock/power/temperature/used-memory and memory-controller-busy ranges are retained,
not hidden in a single utilization percentage.

## Preservation and next action

All37 frozen capsules are reverified. Both source snapshots are preserved and
hash-checked; no old capsule is edited. Both owned workers are terminal, sampler
is stopped, and original GPU0/1 footprint3651/3189MiB is restored. Compile/test/
profile output and remote binary/SQLite/nsys/SASS SHA pins remain available.
No unrelated jobs, driver settings, permanent services, local Rust/CUDA builds,
commit or push. No failed GPU experiment or timed result is relabeled as passing.

**Keep the qualified column-MLP model path and its47.629 TPS milestone.** Do not
spend another turn on an unbounded DP2A tile/register sweep. Next independently
measure the remaining serial FP16 draft body against an explicit qualified
alternative, or the semantic GDN-head-split prerequisite, before another large
native state-ownership rewrite. Neither is promised to close the remaining gap;
the complete >=5% external single-decode and1/2/4/8 objective is unchanged.
