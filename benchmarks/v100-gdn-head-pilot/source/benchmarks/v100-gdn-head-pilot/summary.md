# Two-V100 semantic head-local GDN pilot

**Positive corrected operator gate; not integrated into model generation yet.**
No new model TPS or external-baseline win is claimed. The full >=5% single plus
HTTP cohorts1/2/4/8 goal remains incomplete. Production inference sources and
the accepted CUDA blocks library are unchanged.

## Corrected unprofiled result

Native host dispatch, explicit host transport, primary-only FP16 block I/O,
actual Qwen3.5-27B Q8 layers0/62. Each arm includes input norm, QKV/Z/A/B,
convolution, independent GDN, gated norm, output projection and residual add.
Three alternating trials per arm/cell, 2000 measured blocks per trial after
8 warmups. Medians below are wall time **per block**, including native CPU
submission and final synchronization, not end-to-end request latency or TPS.

| Layer | Independent sequences × tokens | Row TP (us) | Head TP (us) | Speedup |
|---|---:|---:|---:|---:|
|0|1 ×1|261.560|205.642|1.272x|
|62|1 ×1|260.435|205.683|1.266x|
|0|1 ×3, prefix1/2|333.784|256.849|1.300x|
|62|1 ×3, prefix1/2|335.020|258.369|1.297x|
|0|8 ×1|641.775|478.171|1.342x|
|62|8 ×1|644.566|480.447|1.342x|

Both three-token cells exceed the predeclared1.10x gate, with no >2% regression
in single-token or real independent-eight cells. See recomputed `metrics.json`
for individual paired trials, CPU core-seconds/block, NVML samples, transport
counts, reference errors and separate trace attribution.

## Correctness and arithmetic

- 9 focused two-GPU tests pass: exact joined pre-output arrays/state/history,
  prefix1/2 restore followed by one/two-token shadow continuation, repeated
  native scratch reuse, private selective reset, and no Python numerical hot
  path. Existing odd-value-width fixtures stay intact; this pilot has a
  separate eligible group-aligned fixture rather than weakening its guards.
- For trained layers0/62, all20 independent sequence/block FP64 original-source
  references pass unchanged L2<=0.02/cosine>=0.999 gates. Canonical row/head
  pre-output arrays, packed FP32 state and history/prefixes are byte-identical;
  projected output L2 versus the row arm is below5e-4. Timed-final row/head
  state/history are separately checked after equal recurrence counts.
- This is **not** a new98-row whole-model teacher, greedy trajectory, corpus
  quality or complete MTP qualification. Those remain mandatory for integration.
- Both arms preserve stored Q8 codes/FP16 scales, two residual-A8 passes, source
  small-parameter flags, FP16 activations and packed FP32 GDN state. Head-local
  output uses FP32 column partial/transport/sum before one FP16 cast: explicitly
  changed association, not original output-byte equivalence.
- The corrected host DSO contains only orchestration, no new CUDA kernels. Its
  SHA is `202a38580846334ac9d08893b4f85064f6f5eae8f65f451c2200cfe0c78fa8e1`;
  the reused CUDA library remains
  `6d43160ce4e993b4be5ed600d9be21aad070634729d8e5f7b3f74b734da85de2`.

## Preserved failures and dispatch correction

1. Fixtures-v1 failed remote host compilation on an invalid pointer cast; no
   GPU test ran. Fixed explicitly in the next source snapshot.
2. Fixtures-v2 correctly rejected the original DV13 fixture: local output
   columns are not Q8-group aligned. A separate HV6/DV64 fixture preserves the
   old fixture and alignment check; fixtures-v3 passes all9 checks.
3. Trained-v4 had an incorrect original-source directory and failed before any
   trained block timing. The checkpoint's existing repository-local directory
   was verified, then trained-v5 ran successfully. No download or host switch.
4. Initial trained-v5/profile-v6 scheduled gates separately per independent
   sequence. Production batch code actually batches the gate calculation once.
   Those positive results are preserved but **superseded**. Source-v4 corrects
   both arms, fixtures-v7 requalifies them, and trained-v8/profile-v9 produce the
   final numbers. The final trace explicitly checks32 gate launches in each
   row range and64 in each head range (32 blocks, two head ranks).

Whole local suite:828 passed/2108 skipped. Remote CPU-plus-pilot suite:817
passed/2119 skipped; this is not a full native-engine GPU qualification rerun.
Two synthetic trace parser tests pass locally and on the V100 node. All builds
ran remotely; no local Rust/CUDA build, commit, push or driver-policy change.

Raw report caveat: `final_state_signature` is the initial one-block correctness
snapshot, not the2000-block timed-final signature. The final timed canonical
state/history equality is asserted separately. No signature is relabeled or
raw report overwritten.

## Why the block improves: separate CUDA diagnostics

For32 profiled layer0 three-token blocks, H2D and D2H payload **each** falls
from4,227,072 to2,949,120 bytes (-30.23%). H2D copy calls fall160→64;
D2H128→64. These bytes/calls exactly match native counters, not a layout
forecast. Per single-row block this is44,032→30,720 bytes in each direction.

The GDN prefix grid changes6144 CTAs on the original device to3072 on each
rank, with the same128-thread block,47 registers/thread,32 static shared bytes
and zero traced local bytes/thread. It is a redistribution of work, not a new
kernel or demonstrated achieved-occupancy improvement. The column partial has
640 CTAs/rank,256 threads,44 registers and9216 dynamic shared bytes.

Two-device kernel overlap rises2.209→5.062ms; per-device kernel unions become
more balanced (trace-local IDs:7.759/3.487→6.597/6.066ms). Gaps between this
process's captured activities fall2.471→1.322ms; these are not device-wide idle
measurements. **Summed** kernel time actually rises11.246→12.663ms, as launches
rise544→800. More overlap and less communication, not a smaller summed GPU
work total, explain the observed shorter instrumented span. Profiled timings
are never used as unprofiled speed results.

Unprofiled median CPU core-time per verification block falls344.3→265.6us
(layer0) and344.9→268.0us (layer62). Independent-eight CPU time falls650.3→
487.1us and653.6→489.4us. This is process core-time, not a claim of an equal
application-wide CPU percentage reduction.

NVML windows are retained per trial, including sampled busy/memory-controller
busy, VRAM, power, clocks and temperature. T3/eight-request clocks remain
1380MHz during the selected windows; initial T1 includes a clock ramp, while
layer62 T1 also wins with1380MHz throughout. Both arms are resident together;
peak physical-GPU usage4161/3721MiB is **not** a model-serving VRAM footprint.
Original3651/3189MiB allocations are restored, with no owned worker/sampler left.
Achieved occupancy/bandwidth/stall counters remain unavailable under unchanged
`RmProfilingAdminOnly: 1`. Exact-library static resources are reused and pinned;
no repeated denied counter campaign. Nsight CPU IP/backtrace and context-switch
sampling are unsupported; CUDA/NVTX attribution succeeds separately.

Worst trained projected-output relative L2 versus the row arm is1.328e-5;
independent source-reference maximum L2 is5.739e-4 and minimum cosine0.999999835.
Fresh streamed hashes of the three original-source shards used by layers0/62
match the official download manifest/LFS hashes at revision
`fc05daec18b0a78c049392ed2e771dde82bdf654`; file identities remain unchanged
through both corrected timing and profile campaigns. Target/draft full hashes
are reused only with verified unchanged identities. No new external run occurs.

## Next

The positive pilot permits native ownership integration, not an automatic
default change. Follow `next-native-integration.md`: resident semantic weight
uploads, distributed private Session/Batch/MTP checkpoint/prefix/reset/failure
ownership, canonical diagnostic joining, independent trained qualification,
then fresh C1/C8 engine screens before the full cohort/external comparison.
