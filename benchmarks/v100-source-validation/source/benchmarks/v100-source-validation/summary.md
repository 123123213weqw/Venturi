# V100 resident source-validation audit

## Result: a measured opportunity, not an implemented speedup

The unchanged qualified native engine makes **6674 checkpoint-verification pairs,
13,348 metadata syscalls** in one fixed23-prompt/32-output-token MTP2 generation.
Target:6223 pairs; draft:451 pairs. Each pair checks both the mapped descriptor
and pathname dev/inode/size/mtime/ctime against the initial load stamp.

Four100,000-pair untraced CPU loops on each actual checkpoint give median pair
times **1.66655us target /1.61785us draft**. Count times those independent costs
suggests **11.1006ms**, about **1.3954%** of mean795.5083ms raw native call wall.
That is an opportunity estimate, not a strict upper bound, measured inference
critical-path cost, or predicted TPS gain: GPU overlap, cache and scheduling
behavior differ. No source guard was removed, and no engine optimization is
implemented or adopted by this capsule.

The predeclared >1% and >5ms eligibility gate passes for a bounded follow-up
prototype. Correctness qualification and an actual baseline/change speed screen
remain unfinished. The estimate cannot explain or close the remaining8.02%
single-decode goal gap by itself.

## Workload, ownership and provenance

- Exactly the SO qualified in capsule39:
  `/home/data/wangyue/artifacts/venturi-mtp-draft-residual-fixtures-v1/build/089e914d53d145f0/libventuri_blocks.so`,
  SHA256 `6d43160ce4e993b4be5ed600d9be21aad070634729d8e5f7b3f74b734da85de2`.
  All native core source files are unchanged. Full source snapshot synchronized
  to `/home/data/wangyue/repos/codex-build/Venturi/source-validation-src-v1`;
 289 src/tests/tools files asserted before the remote CPU-only g++ compilation.
- Two V100-PCIE32GB GPUs, logical0=physical1 primary; explicit host TP because
  the previously tested peer path is not byte-correct. Q8 residual-column-MLP
  target, residual-A8 draft, FP16 KV,32 projection rows, eight prepared private
 256-capacity prefix sessions, greedy fixed IDs/ignore EOS.
- Original source revision `fc05daec18b0a78c049392ed2e771dde82bdf654`;
  target SHA `b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8`,
  draft SHA `a1a18b5f828d9b1cc1028fdc8463564aa4ffdbfe527311e1c0baa48cccfc1641`.
  These are reused previous full payload hashes with unchanged dev/inode/size/
  mtime/ctime identities, **not a fresh full payload scan here**. Existing trained
  quality/private-cache gates are reused by exact SO/profile identity, not rerun.
- One owned detached worker, one owned100ms NVML sampler, no inference server or
  HTTP clients. strace starts the owned child normally, never attaches to other
  jobs or changes policy. No local Rust/CUDA compilation, new privileges,
  host switch, permanent services, commit or push. All39 prior frozen capsules
  remain verified; legacy decoder-layer's recorded source exception is retained.

## Measurements and instrumentation limits

| Scope | Raw native wall | Process CPU core-seconds | Calling-thread CPU |
|---|---:|---:|---:|
| Untraced trial1 |795.9284ms|0.796759|0.795825|
| Untraced trial2 |795.1025ms|0.796143|0.795068|
| Untraced trial3 |795.4940ms|0.796192|0.795379|
| Untraced mean |795.5083ms|0.796365|0.795424|
| Separate strace trial |2021.6468ms|instrumented, not a control|instrumented|

Loading, prefix preparation, warm generation, reset, SDK checks, Python result
inspection and archival are outside the marked ctypes-native interval. This
scope is **not complete HTTP throughput** and is not comparable as a new TPS win
against the previous HTTP campaign. No per-request CPU sum is used as throughput.

The trace contains exactly13,348 metadata syscalls inside the marked interval,
all identified as target/draft checks, with matching descriptor/path totals,
no failed checks or unpaired unfinished/resumed records. Raw trace and unit-tested
TID/interval/path-aware parser are included. strace reports summed metadata time
**343.413ms**; its instrumented call is **2.54x** the untraced call. Do **not**
mistake that343ms for normal source-validation cost or assume it is all removable.

All three untraced calls plus the separate traced call produce the pinned32 IDs,
accept17/26 proposals, make13 verification calls,15 target forwards and45 draft
forwards. No forced proposal, target replay, numerical tolerance change or cache
sharing was introduced. Four parser tests pass locally/remotely; the full local
suite passes **822 tests /2105 GPU-dependent skips**. No new full remote GPU suite
or independent trained-quality campaign is claimed for unchanged inference code.
The initial local parser-test import failure is documented, not concealed.

### NVML samples during the three untraced marked calls

| Physical GPU | Samples | Used VRAM | Mean busy | Mean power | SM/memory clocks | Max temperature |
|---|---:|---:|---:|---:|---:|---:|
|1 primary|23|23,995MiB|79.17%|149.40W|1380/877MHz|38C|
|0 secondary|23|17,229MiB|55.83%|124.41W|1380/877MHz|38C|

Complete min/mean/max rows and separate instrumented samples are in `metrics.json`.
Sampler timestamps are UTC+8 and filtered to the marked intervals, not whole
loading/runtime averages. Memory-controller busy percentages are not bandwidth.
No new CUDA/hardware-counter capture: the exact-SO preceding C1 timeline/resource
reports are explicitly SHA-pinned/reused, including their unavailable achieved
occupancy/counter limits. RmProfilingAdminOnly remains1. The source-check trace
does not locate GPU-idle intervals or establish that all metadata calls are on
the GPU critical path.

## Required next work; full goal remains active

1. If testing this opportunity, implement **resident-only lexical per-forward
   deduplication**, never a cross-request/time cache. Keep forced public/start/end
   integrity checks and every original streaming/read guard. A foreign/nested
   model on the same thread must not inherit another owner's validation scope.
2. Deterministically mutate a disposable source after entry and before return;
   require native post-check rejection with no returned generation and correct
   failure/reset behavior. Test exception unwinding, reset/closed owners,
   independent threads/cohorts, metadata/path replacement and streaming guards.
   Existing actual checkpoint files must remain read-only and unchanged.
3. Only after correctness, screen unprofiled baseline/change C1/C8 with the same
   complete HTTP scopes and independent GPU diagnostics. Count reduction alone
   is not a speed win. Preserve a zero/negative result instead of dropping guards
   or forcing a full matrix. The implementation and those screens are **not done**.

The last qualified optimization remains residual-A8 draft: **47.66 ->48.41 single
decode TPS**. Strongest pinned llama.cpp single decode is49.8033; the >=5% goal is
52.2935 TPS. Retained best complete HTTP cohort rates are38.346/51.821/63.660/
73.836 TPS at1/2/4/8 (C8 from the stronger prior ordinary256 campaign, not this
diagnostic). No new external run or acceptance claim. All owned processes/sampler
terminated, no18953 listener remains, and original3651/3189MiB unrelated GPU
allocations are restored.
