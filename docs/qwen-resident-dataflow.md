# Qwen resident dataflow before graphs

Optimize resident serving in two independent steps: remove byte movement and
ownership transitions that are not needed, then reuse GPU submission for
stable subgraphs. The CUDA Graph experiment must use the same kernels and
arithmetic as its eager control. It is not a codec change, a quantization
policy, or an implicit replacement for memory-efficient attention.

## Existing change: direct indexed embedding

The existing Q4/Q5/Q6 indexed row operator produces dense FP16 activation
rows in input token order, including repeated IDs. The native resident
session now points that operator at the final activation. It no longer
builds and uploads an identity ID array, reserves separate embedding scratch,
or runs a second full-width gather. The stored weight layout and the
operator's elementwise arithmetic do not change; architecture and codec
admission are unchanged.

For M tokens and hidden width H this removes one GPU kernel, M*8 H2D
bytes, and the intermediate activation write/read (4*M*H logical bytes).
The actual HBM reduction depends on caching. Per-session device workspace
also decreases by max_tokens*(2*H+8); this is not a model throughput claim.

Run `tests/test_qwen_resident_embedding_io.py` with an independently built
baseline/candidate Qwen pair and the same operator binary specified through
VENTURI_IO_BASE_LIBRARY, VENTURI_IO_CANDIDATE_LIBRARY and
VENTURI_IO_OPS_LIBRARY. Operator equivalence does not grant new codec
admission.

## Next candidates: copies and commit boundary

- Batch attention projects V to a temporary group tensor, then copies each
  request slice into its private cache. Write straight to the destination
  when it is contiguous and disjoint; retain the scatter path for noncontiguous
  slots or paged layouts until a safe strided output contract exists. The
  validity frontier must not advance on failed writes.
- The trunk writes every segment's last hidden row to slot storage; the
  sampling head gathers selected rows again. In a sampled step with all needed
  rows still resident, make the head consume them directly or gather them
  once. No premature storage reuse when a different API needs the old row.
- Separate enqueue from commit. A terminal sampled step currently fences
  between the trunk and head even though GPU consumers are ordered; a later
  change should prove that one completion fence can cover both. Preserve
  source validation, result/error publication and poisoning/reset semantics.
  Do not apply this automatically to peer transport, spill or debug traces.

## Graph experiment, not implemented here

Start with the resident single-GPU pure-decode attention subgraph. Keep the
shape and backing allocations stable; place slot/position/valid length and
cache descriptors in a fixed-address device buffer. Enqueue descriptor
updates outside capture. Do not capture one graph per sequence position or
scan a full maximum-capacity cache just to pretend positions are static.

Support a few reusable participation buckets first, not every ragged shape.
A graph miss uses the same eager numerical path. Keep cache allocation,
scheduler decisions, source checks, error handling and completion sync out
of the captured region. Scope executable and storage lifetimes to the owning
replica. Reset/cancel/reuse must invalidate ownership before old graph inputs
could be observed. The repository's existing native head-GDN graph is a
separate experimental TP path, not proof ordinary Qwen batch already does this.

Use a matched 2x2 test (old/new dataflow by eager/replay). Measure CUDA copy
counts and bytes, kernel time, timeline/CPU gaps and graph hit fraction; report
capture/instantiate time separately from warm TPS. Check logits, KV/GDN
and cancellation/reset before scoring short and long rolling SSE at matched
arrival patterns. Do not promise a large graph speedup if GPU compute or KV
traffic already dominates the critical path.
