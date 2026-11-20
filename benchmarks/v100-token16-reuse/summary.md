# T16 weight reuse: bounded negative result; graph prerequisite verified

**Disposition:** neither T16 implementation clears the predeclared operator gate.
All nine qualified default native dependency hashes are restored. No new trained
model qualification or serving TPS campaign was run for either rejected arm.
The full >=5% external single-decode and1/2/4/8-request objective remains active.
Nine owned workers are terminal; original unrelated GPU allocations are intact,
no benchmark port or sampler remains, and33 earlier capsules verify unchanged.
No local Rust/CUDA compilation, commit/push, permanent service or driver change.

## What was tested

SM70 residual projections with >=16 packed rows, >=1024 output rows and an encoded
row-plane larger than6 MiB use T16/row8 instead of T8/row16. Both plain and
2/3-descriptor fused groups are included; tiny/cache-resident weights, T1/T2/T3/T4,
T8 controls, single-pass model dispatch and the original FP16 profile stay intact.
A separate research export preserves the old tile4/8 API and ABI16 structures.
Python research preparation explicitly rejects a library lacking that export.

A full tile still computes128 outputs/CTA, but shares weights across16 rather
than8 token rows. The first arm doubles shared storage to36,864 bytes. The second
retains weights while staging two sets of8 activation rows, using18,432 shared
bytes and twice as many barriers per K tile. Neither forces a compiler residency
bound. Integer chains, per-output K/pass/FP32 scale/FMA order, reductions, FP16
casts, global quantization/workspace, Q8 scales, KV and TP fences remain intact.
This is logical weight reuse, not a measurement of halved DRAM traffic.

Each arm runs77 actual checkpoint first-TP-row-shard cells: eight unique plain
shapes plus three fused groups, across8/16/23/32/46/92/184 token rows. Each cell
has three alternating unprofiled repeats, three graph nodes × three replays
(nine calls/wave), exact complete output bytes, sampled represented-weight FP64
relative L2<.001, GPU event, CPU and wall timing. The staged study additionally
retains the natural arm as a pinned ablation. No best-arm selection is performed.
These projection graphs are not model graphs or serving measurements.

The declared gate requires median eligible fused rate>=1.05, median eligible
plain rate>=.98 and minimum unchanged fused T8 rate>=.98:

| Arm | Median eligible fused rate ratio | Median eligible plain ratio | Minimum fused T8 ratio | Advances? |
|---|---:|---:|---:|---|
| Natural T16 | 0.93143 | 0.93055 | 0.99694 | No |
| Two-stage T16 | 1.03640 | 1.01071 | 0.99495 | No |

Fresh staged-study T16 fused timings, milliseconds per projected call:

| Group | Original T8 | Natural T16 ablation | Staged T16 |
|---|---:|---:|---:|
| MLP gate/up | 0.441927 | 0.470268 | 0.430357 |
| GDN QKV/z | 0.222759 | 0.246286 | 0.224533 |
| Attention Q/K/V | 0.197536 | 0.208452 | 0.193696 |

The staged design recovers the initial regression, but its GDN group at the
actual16-row verifier width remains slower. Its3.64% median gain across larger
eligible cases is below5%; it is not a demonstrated end-to-end improvement.
Do not replace the declared gate with the most favorable group or token count.

## Correctness and generated resources

Each implementation passes124 explicit fixtures. The natural full GPU suite is
**2,209 passed,552 skipped**; staged is **2,210 passed,552 skipped** (one extra
resource-parser CPU check). Coverage includes both activation-pass counts,
T8 and16/17/23/31/32/33/46/92/184 tails, row/K tails, eligible and tiny fused groups,
pitches/sentinels, packed workspace identity, invalid native arguments, graph
replay with nonfinite/zero inputs and no Python math/allocation in dispatch.
Historical tile-order-specific checks stay unbound rather than use the wrong SO.

Local suites pass788/789 tests before natural/staged sync, respectively, and
**791 passed,1,973 skipped** after restoration and two graph-probe CPU checks.
Complete pre-build manifests gate each remote build. Executed-source differences
are preserved for both immutable snapshots; all nine selected default native
source hashes match the qualified control. Research wrappers/tools remain
explicit and require the archived experimental library; model dispatch is old.

| Specialization | Threads | Dynamic shared bytes | Registers/thread | Local bytes/thread | Theoretical resident CTAs/SM | Theoretical occupancy |
|---|---:|---:|---:|---:|---:|---:|
| Original plain T8/row16 | 512 | 18,432 | 49 | 0 | 2 | 50% |
| Original fused T8/row16 | 512 | 18,432 | 56 | 0 | 2 | 50% |
| Natural plain T16/row8 | 256 | 36,864 | 64 | 0 | 2 | 25% |
| Natural fused T16/row8 | 256 | 36,864 | 71 | 0 | 2 | 25% |
| Staged plain T16/row8 | 256 | 18,432 | 60 | 0 | 4 | 50% |
| Staged fused T16/row8 | 256 | 18,432 | 64 | 0 | 4 | 50% |

Separate eager Nsight/SASS diagnostics retain nine actual fused-MLP T16 calls
on one V100 per run. Selected natural/staged disassembly has no LDL/STL opcodes;
static dot/FMA counts stay256/32, while BAR.SYNC sites rise2→4. Static instruction
counts and loop structure are not dynamic executed counts, bandwidth or bank
conflict measurements. Recovering theoretical residency is not a universal win.

## CPU/GPU diagnostics and limits

Per-case/wave CPU core-seconds and wall/event times are in the operator reports.
No concurrent durations are summed into a rate. `operator_telemetry.py` joins
100 ms NVML samples to declared microbenchmark intervals; very short waves may
have no sample and NVML has sampling lag. These mixed-shape means are not model
utilization or per-kernel power measurements. In the staged study, selected
control/staged samples number26/31, with mean GPU busy72.0%/80.3%, device-used
VRAM4,982/5,076 MiB, power241.5/244.8 W, SM clock1,335/1,321 MHz, memory clock877 MHz,
and temperature43.6/43.5 C. Sizes differ across sampled cases and the extra
ablation owns resident weights; these figures are not a model-memory regression.
Clocks were not fixed or normalized; three alternating repeats and every trial
remain retained. No power/clock settings were changed.

`RmProfilingAdminOnly:1` is unchanged. Achieved occupancy, DRAM throughput,
cache/bank-conflict and stall counters are unavailable. No static occupancy,
NVML memory-controller busy or kernel-time share substitutes for them. Profiler
runs are separate from unprofiled measurements. Full-model quality/trajectory,
KV/GDN state and EOS/reset gates are intentionally not claimed for rejected arms;
operator byte checks are not broad BF16/corpus quality evidence.

## Separate next-action prerequisite: two-V100 CUDA Graph transport

Using the **unchanged qualified native library**, a synthetic T3/K128/17-output-row
probe captures and replays both V100s through pinned D2H/H2D bridges with both
producer/consumer fences. Full output bytes match eager execution over repeated
replay, zero input and restored finite input. No P2P bypass or fence removal.
All owned CUDA streams/events/graph handles and telemetry samplers are cleaned up.

Three alternating20-call eager trials take4.112/4.499/4.068 ms, with CPU
4.109/4.502/4.065 ms. Graph trials take0.985/**5.726**/0.988 ms, with CPU
0.984/**5.685**/0.988 ms. The slower graph trial is retained, not discarded.
This tiny transport/projection probe establishes **capture capability only**,
not a serving speedup. It does not test model dynamic KV/RoPE/GDN positions,
source mutation, model fault/reset handling or full MTP graph execution.

Next prototype explicit resident scalar-MTP2 target-layer graphs with real
position patching, source/failure/ownership invariants and exact committed
state checks before C1/C8 screens. The bounded T16 geometry study is closed;
do not continue an unbounded occupancy sweep. Current qualified default single
and C8 controls remain about42.75 post-first and65.43 HTTP TPS, against pinned
external49.80 post-first and67.29 C8 HTTP TPS. The goal floors remain52.29 and70.65,
with all1/2/4/8 matched comparisons still required. No new engine TPS is claimed.
