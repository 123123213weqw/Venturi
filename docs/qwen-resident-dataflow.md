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

## Control and comparison

Use a matched 2x2 test (old/new dataflow by descriptor-eager/descriptor-replay)
and retain the original pointer-based eager path as a reference. Measure
CUDA copy counts and bytes, kernel time, timeline/CPU gaps and graph hit
fraction; report capture/instantiate time separately from warm TPS. Check
logits, KV/GDN and cancellation/reset before scoring rolling SSE at matched
arrival patterns. The repository's experimental head-GDN graph is a separate
TP path, not evidence for this single-device batch.

## Resident batch prototype

The guarded prototype implements the two paths independently on a prepared
single-device batch with private contiguous KV:

```python
batch = model.create_batch(
    [1024] * 4,
    resident_dataflow=True,
    attention_execution="graph",
)
print(batch.resident_execution_info())
```

- With resident dataflow, a one-segment V projection targets its cache
  allocation directly. Disjoint slots retain the existing staging/copy.
  An immediate greedy head reads consecutive last rows in place or gathers
  its required rows once. The eager path still stores last-hidden for APIs
  that separate prefill from head. In a combined step, only the head fences;
  request positions advance after completion and finite-result validation.
- `legacy` (default) uses the original per-segment attention. `descriptor`
  batches the same per-query reduction and reads actual KV lengths from one
  upload per pure-decode iteration. `graph` replays those exact kernels and
  the output gate for native M=1/2/4/8; other widths run descriptor-eager.
  Mixed iterations keep the legacy attention path. No allocation or state
  frontier update is captured. No query/KV padding to a maximum length.
- The capture cache is bounded by four widths per full-attention layer and
  owned by the batch. It is keyed by layer and width, not slot or position.
  Buffers remain live until graph destruction. First capture is on use, not
  preflight; prepare expected widths before timing latency-sensitive work.
  Invalid CUDA execution poisons the normal participating slots and requires
  reset rather than silently pretending a failed graph completed.
- `resident_execution_info()` reports cumulative submissions, descriptor
  transfer bytes, copy bytes, explicit completion fences, graph hit/fallback
  and host capture/instantiate time. Bytes are copy payloads, not measured
  HBM bandwidth. Default legacy batching and older native pairs do not
  require the additive resident creation entry. Both experiments stay off. Do not switch a live batch's execution path.

`tests/test_qwen_resident_batch_flow.py` compares raw state across all six
combinations with rotating slot order, changing positions, ragged head
selection, an injected failure and reset. Qwen/ops must be a newly paired build.
