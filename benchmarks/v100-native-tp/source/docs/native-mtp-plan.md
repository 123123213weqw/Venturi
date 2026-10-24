# Native MTP implementation acceptance plan

Status: planned, not implemented. The native two-device path is non-speculative.
The original Qwen3.5-27B BF16 checkpoint contains a single MTP block; the current
text-only Q8 artifact excludes it explicitly. Original checkpoint inventory is
preserved with the two-V100 evidence. Do not reinterpret missing tensors as a
working speculative path or reuse the external engine's implementation.

## Bindings and math

1. Export a separately described Q8 MTP payload from the pinned original BF16
   source. Preserve tensor names, shape, revision, grouping, rounding and small-
   parameter precision. Validate encoded values, scales and all 15 bindings.
   Do not synthesize weights from the already quantized target or change its
   immutable artifact identity.
2. Add native-owned MTP model/session objects. Share the target embedding and
   output head where checkpoint metadata requires sharing; do not duplicate
   multi-gigabyte weights. Keep the MTP attention cache separate from target KV.
3. Qualify the two input RMS norms, embedding/hidden concatenation order, input
   projection, gated full-attention block, MLP, final norm and shared head against
   an independent reference. Confirm whether the target hidden input is before
   or after final normalization and whether norm weights use a unit offset.
   Verify positions and shifted prompt/next-token alignment explicitly.
4. Expose target hidden values needed by the draft without copying whole-model
   intermediate tensors to Python. Production drafting and verification must
   stay inside the native runtime; diagnostic snapshots remain separate.

## Transactional target verification

- First prove draft-length one before increasing to three. The target verifies
  the pending token plus proposed draft tokens using its multi-token forward.
  Every proposal is checked against the target's greedy selection; draft quality
  changes acceptance, never which target tokens are emitted.
- Target GDN state and convolution history are mutable and not recoverable by
  reducing the cursor alone. Back up their pre-verification state. On rejection,
  restore it and replay only the accepted input prefix, or qualify an equivalent
  per-token commit mechanism. Hide stale KV tails by cursor bounds. Keep the
  corresponding MTP cache/hidden state aligned as well.
- Maintain the established cursor convention: the last emitted token has not
  necessarily been consumed. Handle bonus tokens, zero acceptance, full acceptance,
  EOS inside a verified chunk, output-length truncation and capacity boundaries.
  A partially consumed draft batch must not commit unwanted GDN/history updates.
- Preserve failure poisoning/reset, source-file identity checks, session lifetime,
  primary/secondary memory admission and shared-model scratch serialization.

## Required evidence before promotion

- Generated IDs exactly match non-speculative native target decoding across
  rejection/full-acceptance/EOS/boundary fixtures and multiple trained prompts.
  Compare hidden/GDN/history/KV snapshots at committed positions, not just text.
- Reference correctness for the MTP block is separate from target output parity;
  a broken draft can still produce exact target output with near-zero acceptance.
- Record proposed/accepted draft tokens, acceptance by depth, target verification
  calls, replayed input tokens, emitted tokens and draft/verification/rollback
  wall time. Define every denominator. Never equate acceptance with probability
  mass or draft-only throughput with served output throughput.
- Matched unprofiled one-/two-GPU tests: checkpoint and precision, prompt/context,
  output count, EOS policy, concurrency, warmup and timing scope held constant.
  Include total CPU time, TTFT, decode TPS, complete request wall time and both
  GPUs' VRAM/power/clocks/busy samples. Keep profiler experiments separate.
- Request concurrency requires independently stateful draft/target slots and a
  batch-aware scheduler. A shared MTP head does not imply eight requests are
  served concurrently. Measure aggregate emitted tokens divided by global wall
  time, never a sum of per-request durations.

The user's external 50 TPS observation used two V100s, MTP and a different
checkpoint. Reproduce a same-checkpoint, same-precision workload before claiming
an external-engine win. No universal speed guarantee follows from this plan.
