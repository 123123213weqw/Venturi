# V100 lexical source-check scope: rejected performance prototype

## Decision

**Do not ship this optimization.** The resident-only prototype eliminates
**95.7447% of checkpoint metadata syscalls**, preserves numerical/state/source
integrity gates, but fails every predeclared paired performance screen. Removing
checks is not itself a serving speedup or a CPU-efficiency win.

The prototype and all results are preserved in this capsule and remote artifacts.
After measurement,11 modified production files were restored and three
prototype-only files removed. Current `src/tests/tools` exactly matches the289-file
pre-experiment capsule40 manifest. The qualified residual-draft engine remains
the production implementation. No source-scope build/API option is shipped.

Previous turn is progress: the unchanged-engine syscall audit identified and
bounded this opportunity. This turn completes its implementation, correctness
qualification,18-run paired screen, separate profiles and evidence-driven
rejection. The full >=5% external single plus HTTP1/2/4/8 goal is **not complete**.

## What was tested

An explicit SM70-blocks build option selected an isolated cache key and native
mode query, leaving original per-weight checks as the experimental default0.
Mode1 used a thread-local lexical Model owner only inside resident Session.forward,
Batch.group and MTP draft_step. Native entry and post-GPU-sync exit checks stayed
forced, before cursor commit. Every streaming/load/upload/unscoped/foreign-model
guard remained. Nesting, exceptions and streaming scopes restored/cleared owners;
there was no cross-call/time/activation-pointer validation cache or stamp refresh.

The guard predicate was unchanged, but resident detection cadence intentionally
changed: a source modification could be rejected at the forced exit instead of
the next internal projection. This was disclosed and tested, not described as
identical guard cadence. Failed work returned no valid SDK generation; a changed
source still required model recreation, not reset accepting a new stamp.

No arithmetic, Q8 codes/scales, small parameters, state layout, output IDs,
acceptance, proposal count or precision changed relative to the matched native
control. Existing target residual-column-MLP and residual-A8 draft profiles,
FP16 KV and explicit host TP stayed fixed. This is not a new claim of identical
native/llama arithmetic merely because both weights are Q8.

### Source and binary provenance

- Remote-only CUDA compilation after a complete source manifest in snapshot1.
  Default0 SO SHA `061290e9ccd089017161ca9265348526bf7f71d1db1c9abc5c8fdec91c2d677b`,
  path `/home/data/wangyue/artifacts/venturi-source-scope-fixtures-v1/build/cf7db603d5edda81/libventuri_blocks.so`.
- Experimental1 SO SHA `1b82d19104f4987db96672e74430d3cf0a6da51997c951a4e54ecd57fcef18b9`,
  path `/home/data/wangyue/artifacts/venturi-source-scope-fixtures-v1/build/fb37c600c0b94279/libventuri_blocks.so`.
- Both10-file CUDA dependency manifests match every later executed snapshot.
  Snapshots1/2/3/4 are preserved under `executed-source-v1` through
  `executed-source-v4`; the remote folders are
  `/home/data/wangyue/repos/codex-build/Venturi/source-scope-src-v1` through `v4`.
  Later differences are test/qualification/reporting harness repairs, not native
  arithmetic changes. Reproduce the prototype from snapshot4, **not** the restored
  production tree. `implementation.diff` and all launchers/commands are retained.
- Official source revision remains `fc05daec18b0a78c049392ed2e771dde82bdf654`.
  Target SHA `b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8`;
  draft SHA `a1a18b5f828d9b1cc1028fdc8463564aa4ffdbfe527311e1c0baa48cccfc1641`.
  Previous full hashes are explicitly reused with unchanged dev/inode/size/
  mtime/ctime; the new trained qualifier also freshly hashes the full draft.
  No fresh full target scan, external throughput run or80-file external-bundle
  rehash is claimed. All40 previous frozen capsules are verified.

## Correctness gates

- **29 focused GPU/CPU tests passed**, including fake-compiler boolean/architecture/
  cache-key checks, exact new-default bytes against the qualified old SO,
  deterministic source mutation after GPU completion/before native exit,
  mtime/path replacement, streaming/resident/column TP, target/draft MTP and
  native cohorts. Failure/cursor, rejected reset, recreation, nested/foreign/
  streaming owner scopes, exception unwinding, private threads and closed owners
  were checked. Test pauses were never armed in inference or timing commands.
- **Full remote suite:2269 passed /687 skipped**, snapshot2. This includes the
  existing independent fixtures and forced-prefix/EOS/ragged/private/reset checks.
  Latest parser snapshot4 separately passes5 tests. The prototype local suite
  passes835 tests /2122 GPU-dependent skips.
- Fresh trained four-case ordinary/MTP2 outputs, canonical GDN/history/KV and
  corrected draft bytes match pinned prior states. MTP counters remain exact;
  story accepts17/26 proposals, with13 verifications,15 target and45 draft
  forwards. Fresh mixed-eight/full-eight MTP1 cohorts and selective reset match
  previous full references and private-state signatures. Actual public CLI emits
  the pinned32 IDs and reports the experimental execution policy.
- Independent98-row represented-weight teacher and trained draft block/recurrence
  quality gates are SHA-pinned and explicitly **reused**, not freshly repeated
  for unchanged arithmetic. No broader original-BF16/corpus quality claim.
- The first fixture worker failed after19 passing checks because the test tried
  to inspect MTP `.failed` through SDK `.info()` after deliberately changing its
  source. That accessor correctly revalidated metadata and rejected it. The
  repaired test reads the native failure bit without refreshing stamps and still
  asserts that the SDK refuses the changed source. Both logs/sources are retained;
  no production guard or numerical threshold was weakened.

## Unprofiled paired screens

Same two V100-PCIE32GB GPUs; logical0=physical1 primary; host TP, fixed formatted
23 prompt IDs/32 output IDs, greedy/ignore EOS, eight256-capacity slots. MTP
projection budget32; ordinary batching budget256. Complete HTTP throughput is
32*N divided by global cohort wall, including reset/admission/queue/reply/client
archival/future collection. Native C1 decode is31/(generation minus TTFT), not
complete HTTP throughput or a sum of concurrent timers.

Three paired C1/C8 trials per path: scope0→1, then1→0, then0→1. Each server is
separate, with its own warmup, exact loaded-SO map and100ms NVML samples. No
strace/nsys run overlaps these measurements. **18 server experiments**, all
outputs exact. Mode1's decode gain must exceed1% in C1 decode or HTTP, with two
improving paired trials and no specified HTTP regressions, to permit a full
three-repeat1/2/4/8 matrix. Every path fails; no such matrix is promoted.

| Path | C1 native decode0→1 | C1 HTTP0→1 | C8 HTTP0→1 |
|---|---:|---:|---:|
|Scalar MTP2/prefix|48.4438→48.4688 (+0.052%)|38.6056→38.5241 (-0.211%)|38.4965→38.7744 (+0.722%)|
|Native MTP1 batch|45.8149→45.8817 (+0.146%)|36.4036→36.4284 (+0.068%)|68.9235→68.8667 (-0.082%)|
|Ordinary256 batch|29.3123→29.4287 (+0.397%)|25.5993→25.6556 (+0.220%)|73.8828→73.9186 (+0.049%)|

Scalar MTP2 C1 decode improves in only1/3 paired trials; MTP1 and ordinary improve
in3/3 but remain below the threshold. These small deltas are not promoted as
repeatable product speedups. This turn has **no fresh C2/C4 results**; the stronger
previous accepted matrices remain the evidence for those counts.

### CPU and latency, not just TPS

Mean server CPU core-seconds0→1:

| Path | C1 | C8 |
|---|---:|---:|
|Scalar MTP2|0.8267→0.8300|7.0933→7.1200|
|Native MTP1 batch|0.8533→0.8567|3.7067→3.7133|
|Ordinary256 batch|1.2233→1.2200|3.4600→3.4667|

No overall CPU-efficiency gain is established; these process counters also have
tick granularity. C1 native TTFT is essentially unchanged for MTP2 at155.075ms;
C8 scalar mean native TTFT **worsens833.69→897.23ms** despite the slight complete
cohort increase. C8 batch TTFT699.93→699.30ms; ordinary1082.59→1082.71ms.
Mean/max client completion, trial timings and every native counter are retained
in `metrics.json`; no streaming-TTFT or scheduler-fairness claim is implied.

### NVML

Both arms retain the same measured VRAM. Physical1/0 respectively:

- Scalar C1:23995/17229MiB; mean busy78.33/55.67%→77.63/56.04%, mean
  power148.32/120.69W→146.24/119.55W.
- MTP1 C8:22461/17229MiB; busy85.66/58.82%→85.19/58.90%,
  power173.55/134.40W→174.49/134.82W.
- Ordinary C8:21969/17319MiB; busy84.41/57.86%→84.76/58.15%.

These are averages of trial-window sample means, not load/startup averages.
SM/memory clocks remain1380/877MHz in these windows; full min/mean/max temperature,
power, clocks, busy and VRAM rows are preserved. NVML busy is not achieved
occupancy; memory-controller busy is not DRAM bandwidth.

## Separate syscall and CUDA diagnostics

The actual experimental SO makes **284 verification pairs /568 syscalls** per
marked raw native C1 generation, down from6674/13348 in the pinned unchanged-SO
audit. Target pairs6223→193; draft451→91. The omitted6390 pairs equal
15 target forwards×402 internal checks plus45 draft forwards×8 internal checks.
Output IDs, forward/proposal counts and post-call source checks remain exact;
the reduction is not less requested inference work. strace time is instrumented,
not untraced CPU cost or a serving-TPS score.

Separate matched CUDA profiles:

| Path | Host span0→1 | Kernel count0/1 | Summed kernel time0→1 |
|---|---:|---:|---:|
|MTP2 C1|833.917→846.025ms|29036/29036|961.003→960.912ms|
|MTP1 C8|3773.318→3756.491ms|73304/73304|4649.945→4650.264ms|

Summed kernel time spans two concurrent GPUs and is **not** wall time. C1 device
kernel unions remain565.672/395.331→565.560/395.352ms; simultaneous two-device
kernel time338.095→330.960ms. C1 transfers remain7102 H2D/5883 D2H calls,
316660240/316660272 bytes. C8 transfers remain12187/10059 calls,
2232040320/2232039680 bytes. Removing stat calls did not remove any of that work.

C1 summed T4 fused projection remains327.098→327.084ms and column partial
122.292→122.314ms; C8 fused projection2249.649→2249.416ms and partial
806.555→807.011ms. Process-activity gaps124.369→127.102ms atC1 and
302.488→295.026ms atC8 are **not device-wide GPU idle or CPU-only time**.
The instrumentation deltas are not accepted as speedups, especially where the
unprofiled screen disagrees. Results are consistent with metadata work not being
the dominant device critical path, but do not prove that a particular percentage
of all stat time overlaps GPU work.

Static resource reports are collected for both exact SOs. Achieved occupancy,
compute/DRAM/cache/stall counters remain unavailable; policy1 is unchanged and
no repeated denied counter campaign or privilege change occurs. CUDA/NVTX
capture succeeds; unsupported CPU sampling/context-switch warnings are preserved.
Large SO/SQLite/nsys files stay remote with hashes, not silently missing evidence.

## Rejection and remaining goal

`production-restoration.json` records restoration of exactly the experimental
diff, after checking the entire working source tree against the executed manifest
to avoid overwriting unrelated work. All289 production files match capsule40;
their10 native dependencies match the accepted6d43160 SO. Restored local suite:
**822 passed /2105 skipped**. No unnecessary recompilation or unchanged trained
campaign is rerun just to recreate the prior proven implementation.

Keep the previously accepted full-matrix single result **48.4089 TPS**, not this
rejected screen's tiny delta. Strongest pinned external single remains49.8033;
the >=5% floor is **52.2935 TPS**. Previous accepted best complete HTTP1/2/4/8
38.346/51.821/63.660/73.836 TPS is explicitly retained, not remeasured as a new
matrix here. No fresh external run or full-goal completion claim.

Next is the measured semantic head-local GDN/column-output **block pilot** in
`next-gdn-pilot.md`, before any deep private-cache ownership rewrite. Do not repeat
this failed source-check search or assume deeper MTP will help without checking
the already-existing depth/acceptance evidence. All five workers are terminal,
including the preserved first test issue; owned samplers/servers are gone,18953
is closed, original3651/3189MiB unrelated GPU allocations are restored. No local
Rust/CUDA compilation, unrelated-job changes, commit or push.
