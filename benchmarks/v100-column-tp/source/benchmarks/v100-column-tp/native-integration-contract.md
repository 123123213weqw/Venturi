# Next native integration: MLP-only column/row TP

Do not wait for an entire head-sharded attention/GDN rewrite to test this gain.
An opt-in **MLP-only** path can use the qualified current row-TP attention/GDN and
replace the MLP communication pattern. This is planned work, not implemented
model support. The existing best decode and parallel TPS remain unchanged.

1. New explicit resident SM70 two-GPU Q8-residual-column-MLP profile; FP16 KV,
   FP32 recurrent state, unchanged original FP16 draft. Preserve old profile IDs,
   ABI16 layouts and defaults. Reject unsupported nonresident, PP/single-GPU
   combinations before allocating; never fall back to old row-down with
   column-packed weights. Report the changed FP32 association explicitly.
2. Gate/up retain the existing output-row shards. Replace (do not duplicate)
   each down matrix with bit-preserving input-column halves: all output rows,
   half input columns, original FP16 scale groups. No requantization or new
   checkpoint download. Group-aligned cuts and independently padded local rows;
   source identity/pre/post validation, spans and byte admission remain enforced.
   Reuse original file mappings; ensure another projection cannot alias the down
   offset with a different interpretation. Do not run CUDA packing on this Mac.
3. Broadcast primary normalized hidden once, with the existing producer fence.
   Each rank performs local fused gate/up then local FP16 SwiGLU; down produces
   full-hidden FP32 partials on both ranks. Peer D2H/H2D transports **4 bytes per
   output**, then a native FP32 sum and a single FP16 cast on the primary. The
   existing FP16 residual add stays separate. No final broadcast is required
   while attention/GDN remain primary-output row TP. Preserve both producer and
   consumer scratch-ownership fences on every call and across calls/graphs.
4. Prefer explicitly admitted local/remote FP32 scratch over unsafe aliases.
   Max projection rows32 needs655360 bytes per FP32 hidden buffer for Qwen;
   root two/peer one plus pinned bridge are modest, but include them in model
   memory counters. Private request GDN/convolution/KV/prefix storage is unchanged.
   Batch/MTP borrowers must execute the same new MLP path, never old down linear.
   Session diagnostic traces, failure/reset/close and source mutation keep their
   existing ownership/poisoning guarantees.
5. Correctness: small full-MLP independent represented FP64 references, group
   tails and local-activation equivalence; unchanged old-profile fixtures;
   fresh new-profile98-row/four-trajectory/eight mixed-reset quality gates against
   the pinned teacher with unchanged bounds. Do not require or claim old-profile
   bytes after changed association. Then compare ordinary/new-profile MTP1/MTP2
   committed private states, mixed rejection/EOS/reset/cursors, prefix recovery,
   selective batch reset and original-source corrected-draft recurrence. Draft
   quality must use the new target hidden states, not stale old-profile hidden.
6. Unprofiled strongest-native versus candidate C1/C8 screens for scalar MTP2,
   ordinary packed decoding and MTP1 target cohorts. Promoted paths receive
   three1/2/4/8-request repeats and fresh pinned external comparisons; preserve
   all weaker cells. Separate diagnostics and all current precision/timing gates
   remain in force. No engine promotion from the microbenchmark alone.

Matched primary-only operator copies are84992*T versus30720*T bytes per
host-bridge direction per MLP. For the previous63-target-input/64-layer call,
the MLP-only payload estimate is342,687,744 versus123,863,040 bytes/direction.
This is not a measured model reduction or TPS forecast: real integration may
have different launch ordering, source checks, pipeline overlap and allocation.
Single-stage isolated times cannot simply be added to predict whole-model time.
