# T3 projection study

Previous goal turn: verified progress (native MTP2 prefix recovery, frozen capsule).
Current experiment preserves arithmetic and targets measured verification cost:
prior full 32-output traces attribute about 85.5% of summed CUDA kernel time to Q8
projections. Native MTP2 is still below the strongest tested external decode rate.

## Research first, dispatch second

- Add explicit SM70-only research entries for exactly three tokens, full activation
  staging (at most 48KiB) or K-tiled token3 staging, row groups 4/8/16. Existing
  ordinary APIs and model dispatch are unchanged in study source-v1.
- Kernel templates retain each lane's group ordering, two independent integer dot
  accumulators, F32 scale/FMA sequence, full warp reduction and FP16 output. No
  split-K partial reduction, quantization/profile change, tensor-core substitution,
  host numeric fallback, new cache or transport-fence modification.
- Checkpoint microbenchmarks use all eight distinct Q8 matrix shapes, first TP row
  shards and three actual fused groups. Control is the pinned prefix-capable
  library 9a4d8655...; the new library's unchanged T4 path is an additional control.
  Every output byte must match; sampled rows have an independent represented-
  weight FP64 gate (relative-L2<.001). Three alternating repeats use three-node
  graphs and three graph replays (nine calls), reporting event, CPU and wall times.
  These operator graphs do not imply whole-model CUDA graphs or serving TPS.
- Full sharing loses on the large fused groups. Do not advertise the standalone
  512-row V projection's 20% best-arm result as an engine gain: deployed Q/K/V
  execute together. The generic token3 K-tiled row8 path is the measured candidate,
  not per-shape best-of-six selection.
- Source-v2 changes only residual T3 model dispatch and its fused equivalent to
  token3/row8 K tiles. T1/T2, T>=4, original-FP16 profile, PP limitations and SM89
  dispatch are unchanged. Dynamic shared payload falls 9216 -> 6912 bytes for that
  specialization; persistent workspaces and ownership remain unchanged. This is
  not an achieved-occupancy or bandwidth claim.

## Qualification and timing

Run both complete fixture suites before timing. Fresh candidate target logits,
four trajectories, eight mixed private caches and reset/continuation signatures
must equal the pinned qualified residual control. The existing 98-vector CPU/
quality reference is reused through exact signatures, not re-executed/relaxed.
Fresh trained MTP2-prefix output/target/draft-state gates and independent recursive
draft checks follow; one-proposal draft arithmetic itself is unchanged.

Fresh C1/C8 MTP2-prefix and ordinary rows256 batch screens precede conditional
three-repeat C1/2/4/8 matrices. Keep the best existing ordinary batch and external
controls even when a weaker scalar MTP control makes a larger ratio. MTP uses
rows32, ordinary batch rows256; both keep capacity256 and eight slots. Fixed speed
case is 23 formatted prompt IDs/32 greedy outputs, ignore EOS, FP16 KV, resident
host TP on physical GPUs 1,0. No artifact or format conversion.

HTTP global cohort wall includes handler reset/queueing/replies plus client
archival/validation/future collection; same-call native generation/TTFT excludes
reset and HTTP. Loading, slot preparation, warmup and profiling are outside timed
waves. CPU core-seconds, latency and all-output/global-wall TPS stay separate.

Fresh candidate full 23/32 C1/C8 profiles are separate from unprofiled trials.
The exact pinned prior full 23/32 baseline profiles are reused by SHA256, with
unchanged profiler tool/case/library and explicit provenance. Do not substitute
short 8-output captures that contain no draft rejections. Static resources and
sampled NVML telemetry are retained; RmProfilingAdminOnly stays 1 and achieved
SM/compute/DRAM/cache/stall counters remain unavailable.

No broad original-BF16/corpus quality, other architecture, sampling, continuation,
production continuous batching/paging, external best-configuration or goal-
completion claim follows from this study. Preserve all 29 earlier capsules.

## Final decision

The trained-gated source-v2 candidate fails the internal screen's C1>1% /C8>2%
complete-request promotion rule. Actual ratios are 1.002729 /1.011391. Ordinary
batch ratios are 1.000333 /1.000185. The small single post-first screen moves
42.8406 -> 43.1897 TPS, not a repeated decode result or an external win.
Do not launch a full matrix or publish that point as a new qualified speed.

Restore the model dispatch to source-v1, verified against all nine native source
hashes of the already full-suite-tested research build. Keep the explicit research
entries for reproduction, but no ordinary/model API automatically uses them.
The immutable source-v2 snapshot, its complete tests/trained qualification and
full 32-output candidate profiles remain evidence of this rejected experiment.
Current model performance therefore retains the earlier qualified three-repeat
MTP2/batch controls rather than selecting the largest microbenchmark ratio.
