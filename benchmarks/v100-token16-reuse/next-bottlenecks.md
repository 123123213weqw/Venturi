# Remaining routes after the bounded T16 study

Do not use theoretical residency alone to declare a kernel win or loss. Preserve
the natural T16 failure and the separate two-stage weight-retention experiment;
only their original-control operator/state/engine gates decide adoption. This
is not a reason for an unbounded row/tile/block-count search.

## Single request: measure the resident source-check/launch path

T16 does not affect steady-state MTP2 target width3. The retained single control
is about42.75 post-first TPS, below the pinned external49.80 and52.29 goal floor.
Use a separate syscall/timeline diagnostic before editing guards. `Model::weight`
calls `source()` for every projection; the complete resident step already holds
model/session ownership. A possible scoped resident-only pre/post validation
must catch source mutation before returning success, preserve fail/poison/reset
semantics and keep nonresident streaming checks unchanged. Do not simply remove
source checks, declare CUDA events absent (they already exist), or drop either
host-TP scratch/consumer ownership fence. Native graphs need explicit dynamic
KV/GDN positions and error/source handling, not a graph-only microbench claim.

Deeper speculative decoding is a separate architecture change, not an integer
flag: current target/draft prefix traces, state/history snapshots and exported
depth counters explicitly support only two proposed tokens. Any extension needs
fresh mixed rejection/EOS, exact committed recurrent/history/KV/cursor tests and
independent corrected-draft recurrence gates. More proposals can be slower.

## Parallel: distinguish target, prefill and proposal-head work

Pinned optimized-SDK C8 MTP1 evidence (not a new timing run): three HTTP cohort
walls4.2747/4.2767/4.2758 s; first-wave shared native time4.23049 s, complete
prefill1.48100 s and verification wall2.37287 s. Acceptance remains14/16 per
request. Stronger ordinary256 first-wave native TTFT is1.24515 s, not an assumed
sub-millisecond prefill. Batching MTP prefills therefore cannot be credited with
eliminating the full1.48 s: ordinary target prefill itself is already substantial.
These scopes and the source paths are pinned in the earlier batched-MTP capsule.

`MTPBatch` currently serializes per-slot prefill/shifted draft warmup, draft steps,
proposal heads and corrected catch-up. Its target verifier is genuinely packed,
and it borrows the target Batch sessions rather than duplicating permanent KV.
Shared proposal heads could remove repeated target-head reads, but their gain
must be attributed separately from the full draft block. In the pinned C8 trace,
T1 DP4A kernels sum212.85 ms and FP16/Q8 draft-body kernels sum202.71 ms versus
4.318 s instrumented host span; these sums are not critical-path shares. Do not
promise a large total speedup from head batching without a matched run.

A batched MTP prefill must export every segment's normalized target hidden row
for shifted draft warmup, not merely `Batch.last_hidden`. Preserve the zero
initial hidden, per-token right shift, each request's private target/draft KV,
GDN/convolution/cursor, and EOS/failure isolation. Start with the existing borrowed
Batch and no duplicate cache. Respect `max_tokens/active` packing; changing a
projection-row workspace limit is not itself native prefill batching.

The full >=5% external single plus1/2/4/8 HTTP objective remains unchanged.

## New verified prerequisite: two-GPU host-bridge CUDA Graphs

A separate synthetic T3/K128/17-row projection probe now captures/replays both
V100s with pinned D2H/H2D transport and both producer/consumer fences. Exact
outputs survive repeated replay, zero input and restored finite input. This
uses the unchanged qualified native library, not the slower T16 kernels.
All six alternating eager/graph trials remain retained; one graph trial is
slower than eager. The tiny probe is capability evidence, not a model speedup.
Dynamic KV/GDN positions and model source/failure/reset semantics are untested.

Next implement an explicit resident-only **scalar MTP2 target-layer graph**
prototype, starting with native CUDA capture diagnostics rather than pretending
model graph support already exists. Keep embedding/head/selection boundaries
outside the first prototype if needed. Validate/patch all attention KV write
positions, RoPE table pointers and past lengths; GDN/history prefix buffers and
both TP bridge fences must remain correct. Preserve native source checks with
explicit before/after replay validation, fail/poison/reset and fault injection;
nonresident streaming and variable-width fallbacks must stay unchanged. Compare
exact per-step/logit/target/draft states before any C1/C8 speed screen. Current
packed parallel verification needs per-slot graph positions, a separate scope.
