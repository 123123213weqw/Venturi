# Canonical TP prerequisite: FP32 partial projections

The correctness-qualified scalar layer-graph experiment failed its screen and
is frozen in the preceding capsule. All nine qualified default native sources
were restored before this experiment. No subagents, local CUDA/Rust compilation,
commit or push. Existing eager TP/MTP/PP and their defaults remain unchanged.

## Deliberate scope

Add an explicit research operator, not a full model or new default TP mode:
bit-preserving stored Q8 group32 column slices; the existing two-pass residual-A8
quantizer; FP32 partial projection output; FP32 two-input sum then one FP16 cast.
No FP16 rounding/transport of partial outputs. FP32 summation association changes
relative to the old full-K reduction: declare this new arithmetic, use independent
FP64 represented references and retain numerical counterexamples. It is not
same-profile byte equivalence with the old target. No one-pass A8 option.

Use Qwen O/down shapes N5120/K6144 and N5120/K17408, split K evenly. Explicit
slices must begin/end at stored scale-group boundaries (except final logical K).
Preparation copies stored codes/scales, not floating requantization. Source
padding is not a logical weight; new padding is zero and never read as input.
Same input distribution/seed for controls. Caller owns all device scratch.

## Gates

CPU invalid-geometry/encoded-byte/padding/ownership checks, missing-export and
build dependency tests first. Remote compilation only after complete source sync
and manifest assertion. Native SM70 operator tests cover tails, zero/nonfinite,
alias rejection, quantization-byte parity, FP32 output error, cancellation/large
partials, graph replay and no Torch math/allocation in execution.

An unprofiled two-V100 complete MLP-block benchmark compares existing output-row
TP to paired column/row TP. Same full trained Q8 gate/up/down matrices and primary-
only FP16 normalized input/output, two-pass A8, FP32 accumulation, FP16 final
output, explicit changed association. Both use
pinned host transport with producer/consumer CUDA-event fences. Three alternating
trials; include logical copy bytes, CPU core-time, GPU events, host wall and NVML.
Start/end with synchronized owned streams; do not infer model TPS from operators.
Use bounded token widths1/3/8 and actual Qwen MLP dimensions, not a tile sweep.
The focused projection fixtures separately cover both actual O/down shapes.
An initial experiment used replicated column inputs/outputs; retain it separately
and repeat with matched primary-only ownership before drawing the block result.
Profiles, static resources and achieved-counter unavailability are separate.

Operator correctness is necessary, not sufficient. Full head-aware QKV/GDN/MLP
sharding, private cache/prefix/MTP ownership and fresh model/reference gates remain
required. No promotion to a model path from operator timing alone.
