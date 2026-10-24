# Paired activation word planes for the T3 verifier

Previous goal turn: verified progress. Adjacent CTA ordering was numerically
correct but failed all engine promotion gates; its default code is restored.
Do not repeat its77 cells or claim a new qualified TPS result from that screen.
The full goal remains single decode >=5% above pinned49.80 TPS and matched
parallel1/2/4/8 wins, not merely improved operator timing or occupancy.

## One bounded memory-layout change

Target only the SM70 K-tiled specialization with token_tile4, two activation
passes and row_tile8. This is the actual T3 target verifier body. T1/T2, tile8,
other row/pass specializations, original FP16 projection profile, CTA grid,
quantizer, FP16 KV, model/draft protocol and both TP ownership fences stay intact.

Current shared data uses two int4 activation planes. The candidate uses four
int2 planes pairing corresponding low/high words: `(lo.x, hi.x)`, then y/z/w.
Each dot consumes one pair before the next, keeping the original two four-step
integer accumulators, their final addition and every FP32 scale/FMA/reduction
in its original order. Global packed inputs and shared scale offsets stay the
same; total shared storage stays36 bytes/group/pass (9,216 bytes per target CTA).
There are no new persistent allocations, weaker scales or changed output casts.

The hypothesis is shorter activation live ranges without a naive strided scalar
load from the old AoS layout. Four paired planes trade more shared instructions
for potentially fewer live registers; they avoid the eight scalar-plane loads
of the initial conceptual SoA design. Register allocation, issue cost, lost ILP
or staging temporaries may erase any benefit. No forced high-residency bound or
unbounded row/tile sweep. NVIDIA documents consecutive32-bit words mapping to
consecutive shared banks; this motivates layout analysis, not a claim of measured
bank conflicts or speedup. [CUDA12.9.1 guide](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html).

## Gates

- New byte/reference tests cover3/4/5/7/8/9/17/23/32 tokens, both passes,
  tiny/large row and K tails, fused2/3 descriptors and pitched sentinels,
  unchanged packed inputs, graph replay/nonfinite/zero and no Torch math.
- The public ResidentLinear wrapper is not assumed to expose payload-only shape
  fields; fixture outputs use source row counts. Ruff/local gates precede sync.
- Ninety-two new GPU checks and the complete fixture suite run on the V100s.
  Leave unchanged historical tile-order checks skipped rather than bind their
  research variable to this build, which lacks its geometry export.
- Static resource inspection now derives K-tile **dynamic shared bytes** alongside
  threads. One CPU test checks that geometry. Queries do not launch kernels or
  measure achieved occupancy. Driver privileges/policy stay unchanged.
- Operator study: eight actual checkpoint shapes through first TP row shards and
  three fused groups, T3/T4 affected cells plus T8 controls,33 cells total. Same
  original Q8/F16 scales, two A8 residual passes/F32 scales and FP16 I/O. Exact
  full outputs and sampled represented-weight FP64 relative-L2<.001. Three
  alternating unprofiled repeats,3 graph nodes×3 replays=9 calls/wave; event,
  CPU/wall and NVML telemetry retained. These are not serving TPS or model graphs.
- Promote to trained/model measurement only if median fused T3 control/candidate
  event ratio>=1.02, unchanged fused T8 controls>=.98, and target native functions
  retain all local-memory/stack/spill indicators explicitly. A register drop
  alone is not a win; spills are measured costs, not an automatic speed claim.
  Preserve all negative cases and generated resources.

If promoted, reuse unchanged independent teacher/control work by exact hashes
only after fresh target logits/trajectory/private states and MTP1/MTP2 gates.
Then screen C1/C8 scalar MTP2, ordinary256 and MTP1 cohorts (plus known C2 cohort).
Only per-path C1 HTTP>1% or C8>2% (cohort C2>2% also permits) admits a three-repeat
1/2/4/8 matrix. Keep stronger native/external controls and all timing/precision
scope disclosures. If engine gates fail, restore qualified native sources.

No local Rust/CUDA compilation, unrelated GPU process changes, commit/push or
old-capsule edits. Counter restriction remains explicit; static residency,
NVML busy, kernel shares and process gaps are not achieved occupancy/bandwidth.

## Phase B: one evidence-driven compiler bound (separate from initial gate)

Initial unbounded planes pass all92 checks/full2,085-test suite and all33 operator
cells, but median fused T3 rate is1.00365×, below1.02. Plain registers fall48→46
without changing five-block residency; fused stays53/four blocks. Disassembly
keeps64 integer-dot and8 FFMA instructions but replaces16 vector shared loads
with32 pair loads and doubles shared-store instructions. The initial hypothesis
does not produce the intended dominant fused live-range reduction. Its failed
gate, restored-source proof and raw measurements remain separate and unchanged.

One finite follow-up uses the **same paired layout** with `minBlocks=5` only for
SM70 fused tile4/pass2/row8. This asks the compiler to constrain that specialization
near the relevant allocation boundary; it does not repeat prior six/eight-block
sweeps or change tile/row sizes. Measure actual registers, stack/local accesses,
byte equality and timing. Spills are costs to expose, not automatic evidence for
or against speed. The original AoS control, natural unbounded-pairs ablation and
new bounded candidate get three alternating repeats in33 cells. Only bounded
candidate versus original control can pass the unchanged T3>=1.02/T8>=.98 gate;
no best-arm cherry-picking or relabeling the initial failure. Default/model
promotion still requires the separate trained and C1/C8 engine gates.

## Attribution control: AoS with the same five-block bound

Before model adoption, a final bounded2×2 ablation separates **layout** from
**compiler constraint**: original AoS/min1, pairs/min1, pairs/min5 and AoS/min5.
There are no further block/row/tile sweeps. The AoS/min5 source starts from the
qualified original native files and changes only the same fused specialization's
bound. It is built/tested in its own immutable snapshot; the paired candidates
are not overwritten. Its33-cell fresh four-arm study explicitly names each arm.
Only each new candidate vs original control can meet the same T3/T8 gate; do not
rename an ablation or a static occupancy increase as a model win. If either arm
qualifies, confirm the selected arm through the separate trained/C1/C8 gates.

## Guarded integration (not retroactive promotion)

Both five-block arms pass the operator median gate, but their **three-descriptor
attention group regresses**. The complete cases/ablations remain preserved.
Integration selects the measured paired-five kernel only for two-descriptor
SM70 T4/P2/R8 groups with a non-tiny row grid (>=1024), retaining the original
plain operators and original three-descriptor/T8 paths. This is a reusable
geometry/descriptor guard, not a model-name special case or a universal speed
claim. A separate named kernel leaves those original bodies untouched.

The actual guarded source gets its own build,92 invariant checks/full suite and
33-cell original-control comparison before trained/model timing. Fixtures include
eligible two-descriptor groups and their nonfinite/graph cases, plus guarded
three-descriptor controls; no threshold is relaxed. Both prior bound variants
remain optional, explicitly named ablations. No further block/row/tile sweep.

## Default adoption is stricter than a one-wave screen

The guard is not adopted merely because the operator median or a noisy engine
screen passes. A promoted path must retain its declared gain in its three-repeat
1/2/4/8 matrix: C1 HTTP >1%, or C8 HTTP >2%, or MTP1-cohort C2 HTTP >2%, with no
matrix cell slower by more than2%. The qualifying cell must improve in at least
two of three corresponding control/candidate trials. Preserve all cell/trial
values, including failures. This is incremental default adoption, not the full
external >=5% single-decode and parallel goal. Unpromoted paths remain screens,
not qualified repeatable improvements. If no path qualifies, restore all nine
qualified native source hashes; retain the research/test/diagnostic evidence.
