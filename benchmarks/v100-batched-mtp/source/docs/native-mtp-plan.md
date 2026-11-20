# Native MTP implementation and acceptance

Status: **experimental greedy one- or two-proposal implementation**, additive
blocks ABI16 exports. One proposal remains the default. The current extension
is recorded in `benchmarks/v100-native-mtp2/summary.md`; the original depth-one
implementation/evidence below is retained as historical protocol and scope.
MTP stays off unless explicitly selected. Neither ordinary target batching nor
this wrapper is a production continuous-batching or paged-cache scheduler.

## Current explicit depth-two extension

- Native API: `session.generate(ids, max_new_tokens=32, draft_tokens=2)` retains
  the qualified checkpoint/replay ablation. Add `recovery="prefix"` for saved
  intermediate GDN/history recovery, with zero replayed target inputs.
- Call `session.prepare_prefix()` on a fresh/reset request to allocate outside
  generation timing; reset retains these allocations. The CLI does this during
  its preparation stage. CLI flags: `--mtp-artifact ... --mtp-draft-tokens 2
  --mtp-recovery prefix`. The benchmark uses `--mtp-depth 2` instead of the CLI's
  `--mtp-draft-tokens`. Existing libraries need the optional new exports for
  these opt-ins; default depth-one compatibility remains.
- The second proposal consumes normalized first-step draft hidden. Target
  verification captures all normalized hidden rows and accepts only the matching
  prefix. On rejection/EOS, restore real GDN/history prefix bytes and the KV cursor;
  draft speculative KV is rewound and corrected using verified target hidden.
  Last emitted correction/bonus/EOS remains pending, not consumed.
- Prefix2 adds one recurrent-state snapshot per slot beyond the old backup;
  prefix1 reuses that backup. Current Qwen checkpoint: 153,944,064 primary bytes
  per slot, reported separately from old request/backup accounting. Request-local
  descriptors and failure cleanup prevent shared-state leakage. CLI preparation
  failures explicitly close the newly created session before its model owner.
- Original FP16 draft and the target's explicitly selected existing projection
  profile are unchanged. Exact forced transaction/state tests and four trained
  target/draft trajectories pass; three recursive draft steps have independent
  represented-weight FP64 gates. Existing target teacher-vector/quality references
  are pinned and reused, not relabelled as new corpus qualification.
- HTTP cohort wall includes handler reset, queueing and client response archival/
  validation; native same-call timers and separate profile ranges exclude reset.
- The first replay-based MTP2 version is a retained performance regression, not
  a default optimization. Prefix recovery is separately measured and remains
  opt-in; scalar MTP still does not batch target verification across requests.

No MTP3+, stochastic sampling, arbitrary continuation, long-context campaign,
whole-model graphs or universal external speed-superiority claim. Detailed
counters, memory, precision differences, timing scopes and regressions are in the
new capsule's design/summary. The external-performance goal remains active.

## Bindings and math (original depth-one implementation)

- `export_qwen_mtp_q8` exports exactly 15 original MTP tensors: eight Q8/group-32
  FP16-scaled matrices and seven BF16 norm tensors. All bindings/configuration
  and verified-source hashes must match the target. The existing target artifact
  remains unchanged; no Q4-to-Q8 conversion or synthetic draft weights are used.
- `NativeQwenMTPModel.from_q8(target, draft_weights)` shares the native target
  embedding and output head. Its separate one-layer draft owns only its MTP
  weights and cache. PP places the draft on the last stage and transfers selected
  embedding rows from the first stage through request-owned pinned DMA buffers.
  TP uses the existing row-sharded target head. There is no CPU numerical fallback.
- Draft input is the RMS-normalized embedding of token `i` concatenated **before**
  the RMS-normalized target hidden state from position `i-1`. Target hidden means
  **after target final normalization**, not the raw residual. Token zero pairs
  with an explicit zero hidden state. Zero-centered norm parameters use `1+w`.
- Input projection, one gated full-attention/MLP block, MTP final norm and the
  shared output head then produce the proposal. Separate draft KV uses the
  token's actual position. Production hidden capture stays GPU-to-GPU.
- Primary reference implementations consulted for architecture/alignment:
  [vLLM Qwen MTP](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py),
  [vLLM Qwen target](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_next.py)
  and [SGLang Qwen MTP](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_5_mtp.py).
  These are architecture references, not runtime dependencies. The trained source
  is [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B), pinned to revision
  `fc05daec18b0a78c049392ed2e771dde82bdf654`.

## Transactional target verification

1. Prefill the target; warm the shifted draft cache; emit the first target token.
2. Feed the actual pending token and previous target hidden to the draft.
3. Save target GDN recurrent state and convolution history on-device. Verify
   `[pending, proposal]` through the target in one two-token forward.
4. If the proposal matches the first target selection and is not EOS, emit it
   and the second target selection (bonus). Catch up draft KV for the accepted
   input, and retain the corresponding normalized target hidden state.
5. On mismatch, or EOS at the first verified output, restore target GDN/history
   and cursor, replay **pending only**, and emit the correct target token.
   Assert replay selection agrees. Stale attention KV tails remain hidden by
   the committed cursor; reducing the cursor alone would not restore GDN.
6. If only one output remains in the budget, take an ordinary target step and
   advance the draft cache without its head. The last emitted token remains
   unconsumed: both cursors equal `prompt_length + emitted_tokens - 1`.

At chain length one the draft has consumed only the actual pending input before
verification, so rejected proposals never enter its KV. This is **not** a valid
shortcut for longer draft chains; those require additional rollback logic.
Execution failures poison both request states; reset clears both. Native shared
owners preserve model lifetime after Python handles close. Source-file identity
checks, memory admission and per-model/stage scratch locking remain active.

## Evidence and measurement

- Fixtures cover independent MTP equations, shifted/chunked caches, forced
  acceptance/rejection, EOS, max-token boundaries, failures/reset, tied vocabulary,
  model lifetime and eight independently stateful concurrent PP requests.
- The trained MTP normalized hidden and KV values pass an independent original-
  source FP64 reconstruction of represented Q8 weights: relative L2 <= 0.02 and
  centered cosine >= 0.999. Shared-head target math has its previous independent
  qualification; there is no new independent CPU shared-head/corpus-quality gate.
- Trained native generations match ordinary native target IDs and committed
  GDN/history/KV bytes across four prompts. Story output also matches the older
  pinned native IDs. These checks are not a broad BF16/corpus-quality claim.
- Acceptance is matching draft tokens divided by proposed draft tokens, with
  warmup excluded; depth is one. Track verification calls, replayed inputs,
  emitted outputs and draft/verification/checkpoint/rollback wall times.
  Nested/overlapping stage times are not an additive GPU execution breakdown.
- Matched unprofiled ABBA trials hold checkpoint, precision, prompt/context,
  output count, EOS policy and warmup constant at 1/2/4/8 requests. Both variants
  retain the draft model weights; only MTP slots allocate draft state/backups.
  Loading, slot allocation, warmup/reset, diagnostics and profiling are excluded.
- Aggregate TPS is all emitted outputs divided by **global cohort wall**.
  Only single-request post-first-token TPS uses same-call native total-minus-
  first-token timestamps. Concurrent decode TPS is deliberately not synthesized
  from averaged/summed request durations. Native TTFT is internal readiness,
  not client streaming TTFT; Python submission/pre-call waits are excluded.
- TP target execution remains model-serialized; PP can overlap different stages.
  Eight sessions are not continuous batching. Shared draft/vocabulary locks and
  verification/rollback can reduce PP throughput despite high acceptance.
- NVML busy/controller percentages are not achieved occupancy or DRAM bandwidth.
  Separate Nsight traces and CUDA-API theoretical residency are retained. Hardware
  counters remain unavailable under the unchanged restricted driver policy.

## Historical depth-one limits / next work

One proposal, greedy, resident weights, fresh/reset generation, qualified only
on the tested two-V100 Qwen checkpoint/precision. No MTP2/3, arbitrary continuation,
probabilistic acceptance/sampling, continuous batching, paged KV, HTTP streaming,
whole-model graphs, long-context campaign or matched external-engine comparison.
Very short EOS responses can pay rollback cost despite matching the proposal.

Next prioritize the measured PP contention and verification/draft head cost;
then qualify a transaction-safe longer chain before enabling it. Do not make
MTP the default merely because a single-request benchmark improves. The user's
external 50 TPS used two V100s, MTP and a different checkpoint: it is not a
same-checkpoint baseline and no external-engine win is established here.

## Native target-verification cohorts (experimental)

`NativeQwenMTPModel.create_batch([capacity, ...])` returns an owning native
MTP1 target-verification cohort. A target `Batch` owns the private caches and
MTP children borrow them; no second permanent target session is allocated per
slot. Draft execution, proposal vocabulary heads and shifted prompt warmup remain
serial native operations. Only target verification/projection is packed across
requests. Neither PP nor continuation/paging/continuous admission is supported.

For an already constructed complete resident FP16 single/host-TP target and
verified draft weights:

```python
mtp = NativeQwenMTPModel.from_q8(target, draft_weights)
batch = mtp.create_batch([256] * 8)
try:
    outputs = batch.generate(
        [prompt_a, prompt_b], max_new_tokens=[32, 16],
        eos_token_ids=[eos_a, eos_b], slots=[7, 2],
    )
    info = batch.info()
    # Reset only selected requests; other private caches remain untouched.
    batch.reset([7, 2])
finally:
    batch.close()
    mtp.close()
```

Each selected request must be fresh/reset. The last output remains unconsumed;
positions equal prompt length plus emitted outputs minus one. A one-row prefix
trace reuses each request's existing target backup. Rejection/early EOS restores
that prefix without target replay; accepted catch-up uses verified target hidden.
Errors poison the whole selected cohort, which requires reset before reuse.

The benchmark adapter exposes explicit `--batch-mtp`, requiring TP, depth one and
`--mtp-artifact`. It is opt-in rather than an automatic strategy. Ordinary batch,
scalar MTP2-prefix and this mode must be compared at the same workload before
selection: faster two-request verification is not an eight-request/single win.
Per-request verification wall overlaps; cohort counters/timestamps count once.
See `benchmarks/v100-batched-mtp/design.md` for ownership, validation and scopes,
and its `summary.md` for matched results, regressions and limitations.
