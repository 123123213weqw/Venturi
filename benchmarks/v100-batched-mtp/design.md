# Native batched MTP1 target verification

## Scope and ownership

This opt-in path batches target verification, not the draft block or prefill.
`NativeQwenMTPModel.create_batch(capacities)` creates private target/draft caches
for fresh/reset request slots. It supports resident FP16 execution with either
single GPU or host TP, including the explicit Q8 residual target projection
profile. It does not implement PP, continuation, sampling, paging, continuous
admission or whole-model CUDA graphs. Ordinary batching and scalar MTP1/MTP2
remain separate, unchanged paths; the rejected T3 dispatch is not re-enabled.

A single native `MTPBatch.generate` owns the token loop. Its `Batch` owns one
permanent target session per slot. MTP children borrow those sessions rather than
allocating duplicate target KV/GDN/history caches. Children are destroyed before
the owner. Model/workspace/selection bytes count once; private target caches,
additional MTP request buffers and target backups have disjoint accounting.

## Protocol

1. Perform each request's existing native shifted prompt/draft warmup. Emit its
   first greedy output, which remains pending/unconsumed. This prefill is serial.
2. Draft one proposal per active slot using its previous verified target hidden.
   Draft blocks, proposal heads and catch-up remain native but serial.
3. Pack `[pending, proposal]` target rows, or a one-row tail when one output
   remains. Positions and segment boundaries are independent per request.
4. Run one target cohort forward. Save each request's convolution/GDN state after
   its first row in that request's existing checkpoint buffers. Project **all**
   packed normalized target rows through the output head, not only segment tails.
5. On mismatch/first-selection EOS, restore the prefix-one recurrent/history state
   and cursor without replay. On acceptance, commit both target rows and catch up
   draft KV using the verified target hidden, not the old draft hidden. Emit the
   correction or accepted proposal plus bonus subject to budget/EOS.
6. Keep `target_position == draft_position == prompt_length + emitted - 1`.
   The last output, including EOS, remains pending/unconsumed. Errors poison all
   selected cohort slots; selected reset restores them without changing others.

The ownership lock covers the full call. Never hold the target model lock while
entering draft work (which acquires draft then shared vocabulary ownership).
Clear borrowed trace pointers and settle streams on success and exceptions.
Both existing host-TP ownership fences remain. No arithmetic/scale/FMA/KV change.

## SDK and interface

The native ABI remains 16 with unchanged fixed public structures. New optional
`venturi_qwen_mtp_batch_*` exports are bound only for this API; older libraries
continue to serve existing scalar/ordinary APIs and reject this new mode clearly.
The one-double `extra_info` export from prototype v1 stays one-double; the later
common-cohort prefill/end timestamps use a separate additive export.

The SDK revision removes nested repeated metadata validation when `generate`
checks capacity or `info` collects eight per-slot stats. Each public entry retains
its source/metadata guard, native guards remain unchanged, and changed-source and
nonfresh-reentry tests precede performance measurement. Private unchecked parsing
is used only under the owner lock after the validated public entry.

`tools/serve_native_benchmark.py --batch-mtp` selects this experimental mode,
requiring TP, a draft artifact and depth one. Its fixed-cohort HTTP admission
window is 25 ms in this study. It is a loopback benchmark adapter, not a production
OpenAI server. No automatic workload switch or universal default is introduced.

## Timing and evidence

HTTP throughput is all emitted outputs divided by global cohort wall, including
reset, admission, queueing, reply and client archival/validation/future collection.
Native call timings exclude those steps. Per-slot TTFT is staggered because
prefill is serial. Use the once-per-cohort prefill-complete/end timestamps for
cohort steady-state throughput; never sum overlapping per-slot durations or
shared verification times. Group counters cover target verification, not the
serial prefill; per-request logical counters include prefill. Verification wall
is repeated across participating slots and is explicitly nonadditive.

Correctness includes 1/2/4/8 slots, single/TP, original FP16 and residual projection
profiles, natural/forced accept/reject/mixed proposals, arbitrary slot order,
ragged budgets/EOS, exact target/draft state, selected reset, failure cleanup and
shared-model lifetime/overlap. Four real trained prompts plus mixed-eight and
homogeneous-eight cohorts reuse the pinned independent target/teacher quality
reference only by hash. They do not establish new original-BF16/corpus quality.

Speed promotion is conditional on C1 > scalar MTP2 by 1%, or C2/C8 > stronger
ordinary batching by 2%. Promoted arms get three unprofiled repeats at 1/2/4/8.
SDK-only old/new arms share the **same exact native SO** and nine source hashes;
their separate trigger is C1 HTTP >1% or C8 >2%. Profiles remain separate from
speed runs. Restricted hardware counters remain unavailable; static residency,
NVML busy and process timeline gaps are not achieved occupancy/bandwidth/idle.

## Retained setup failure

The first SDK-only snapshot omitted the two launcher shell scripts. Its complete
suite therefore reports two launcher failures, 1,990 passes and 448 skips; its
86 new selected tests passed. The dependent speed worker stopped at the failed
preflight assertion and never created a campaign or server. Both original logs
are retained. A fresh snapshot includes the missing scripts and reruns the full
suite before the SDK old/new speed study. This is a setup failure, not a numerical
regression or a passing qualification. It is not erased by the repaired run.
