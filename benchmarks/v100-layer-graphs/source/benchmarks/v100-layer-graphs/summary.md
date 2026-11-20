# Target-layer CUDA Graphs: correct, not promoted

## Matched two-V100 screen

Official Qwen3.5-27B source revision `fc05daec18b0a78c049392ed2e771dde82bdf654`;
native Q8 group32/FP16 weight scales, two-pass residual-A8/FP32 arithmetic,
FP16 activation/KV, original FP16 draft. Same 23 prompt IDs, 32 greedy outputs,
ignore EOS, capacity256, eight allocated slots; host-bridge TP, no cache reuse.
One unprofiled timed repeat per cell after slot warmup. HTTP TPS uses complete
cohort wall, including reset/admission/queue/reply and client archival/validation.
Post-first uses31 outputs/(native generation−native first-output time).

| Arm | C1 HTTP TPS | C1 post-first TPS | C8 HTTP TPS | C1 / C8 server CPU core-s |
|---|---:|---:|---:|---:|
| Strongest scalar MTP2 control |33.870|42.803|33.956|0.94 /8.03|
| New library, graphs off |33.879|42.852|33.886|0.95 /8.05|
| New library, graphs on |33.560|42.338|33.331|0.95 /8.11|

Graph/control HTTP ratios0.99084/0.98161 fail the declared screen. No full
three-repeat matrix or fresh external rerun was triggered. These C8 results are
**serialized scalar MTP2 requests**, not the stronger ordinary256 or packed MTP1
cohorts. Previously measured stronger parallel paths remain unchanged. Current
external single-decode reference49.803 TPS and >=5% floor52.293 remain unmet.
This is not evidence that all graphs, models or workload lengths are slower.

## Correctness and actual execution

Native library SHA256 `4c997ecd1e45728b83e8e23f1573c9b79a398a3afb6b9e4acfcd97a58cb3eda5`.
Fourteen focused tests pass; complete remote suite2105 passed/674 skipped.
The initial undersized synthetic KV guard failure and a subsequent fixture-only
reset API failure are retained, not hidden. Fresh trained gates pass98 teacher
rows, four trajectories/eight mixed-reset slots, four full-verification-byte and
committed-state comparisons, healthy-reset executable reuse, and independent
three-step original-source draft recurrence. No broad BF16 corpus claim.

Each timed request executes13 graph replays, zero captures and zero warm calls;
3040 graph nodes. Acceptance stays17/26, zero target replay/forced proposals.
Full verification FP16 logits, private GDN/convolution/committed KV and corrected
draft state match the graph-off control. Dynamic KV/RoPE positions, rejection,
EOS, concurrent request ownership, failed/reset/closed owners and source mutation
after launch are covered. Target layers only: head/draft and other widths eager.

## Separate diagnostic traces

Node tracing (`--cuda-graph-trace=node`, CUDA event tracing disabled) retains all
measured graph nodes under actual submitting API correlations:22672 kernels and
16848 copies at C1, none missing. No speculative time-window parser fix needed.
Eager/graph C1 instrumented spans969.34/990.20ms; kernel sums968.45/997.48ms,
not critical-path percentages. Eager launches27026 kernels; graph has4562 eager
launches plus13 graph launches but still27234 executed kernels, including208
integer KV-store kernels. The data movement is not eliminated:535,484,944 versus
535,484,996 H2D bytes and535,484,976 D2H bytes each. Extra52 H2D bytes are13 cursors.
Process-only activity gaps165.22/161.04ms are not device-wide idle or pure CPU
submission time. API-duration sums include blocking and profiling perturbation.

Sampled C1 primary/secondary GPU busy74.6/53.0% control and74.4/51.8% graph;
SM/memory clocks approximately1380/877MHz. Control total device memory23993/17227
MiB, graph24003/17231MiB; graph C8 warms all slots and reaches24075/17259MiB.
Graph driver storage is **not** just the4-byte device/pinned cursors. Metrics
include all raw per-wave latency, CPU, power, clock, temperature and memory data.
Busy is not achieved occupancy; counter policy remains `RmProfilingAdminOnly:1`.
Achieved occupancy, DRAM throughput, cache and stall counters remain unavailable.

## Disposition

All nine default native source hashes restored to the qualified `f07bf75…` build.
Graph SDK/diagnostic methods remain explicitly opt-in and require the archived
experimental library; executed ten-dependency sources are preserved for rebuilding.
No permanent service, GPU job, sampler, commit or push remains. Earlier34 frozen
capsules remain unchanged. The overall speed goal is active and incomplete.
