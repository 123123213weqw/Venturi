# Next: native batched verification, not another open-ended tile sweep

The T3 search is bounded: six strategies, all distinct checkpoint shapes and
actual fused groups. Modest projection gains cannot alone close the external
single-decode gap or the parallel gap. Scalar MTP still serializes shared model
scratch; ordinary native batching is the stronger parallel control.

## Proposed first implementation (not yet implemented)

Start native **one-proposal batched verification**, while retaining the faster
qualified two-proposal scalar option. Do not introduce a Python token loop.

1. Reuse `vr::Batch::group` for packed target projections and independently owned
   request caches. Add a borrowed-target option to MTP sessions so one target state
   belongs to exactly one owner, with draft/checkpoint/session lifetime explicit.
   Do not allocate a second permanent target cache per request.
2. Add request-local intermediate GDN/history capture in the batch GDN segment
   loop. One proposal needs only prefix1, reusing the existing backup; no prefix2
   allocation or target replay is necessary. Cursor rewind alone remains invalid.
3. Add bounded all-row normalized-hidden capture and all-row target greedy selection
   for verification. The existing batch head selects only the last row per slot,
   so reusing it unchanged would verify the wrong proposal. Allocate/count the
   additional selection scratch explicitly and preserve TP transport fences.
4. Prefill can initially call the existing native per-request warm protocol;
   batch the expensive steady-state target verification. Serial native draft steps
   are an explicit first-stage limitation, not batched-draft claims. Preserve
   shifted target-hidden alignment and the unconsumed last output in both caches.
5. Native cohort loop packs pending+proposal per active request, handles ordinary
   tails, per-slot acceptance/EOS/budgets, true-hidden draft catch-up, state restore
   and compaction. Failed calls poison every participating state; selected reset
   must preserve untouched slots. Enforce fresh/reset-only generation until a
   separately correct continuation protocol exists.
6. Keep lock order outer MTP-batch -> target-batch/model; never hold target model
   mutex while entering `draft_step`, which acquires draft then target vocabulary
   mutexes. Concurrent ordinary/scalar/batched calls need ownership tests; shared
   scratch cannot be reused while either GPU still consumes it.
7. Bind additive native create/generate/info/stats/snapshot/reset/failure hooks;
   retain old libraries and default behavior. Only then permit an explicit MTP
   cohort adapter, not silent use of the ordinary batch wrapper or a fake list of
   serial sessions. Fixed cohorts are not continuous batching or paged attention.
8. Gate against ordinary generation and qualified MTP1 IDs, complete target and
   draft bytes with forced accept/reject/EOS/tails, heterogeneous 2/4/8 requests,
   failures/reset/lifetime and shared-model overlap. Fresh trained checks precede
   C1/C8 screens; compare parallel speed against the stronger ordinary batch,
   then three-repeat C1/2/4/8 and the pinned external controls.

After the batched one-proposal path is independently qualified, consider an
explicit policy selecting the existing scalar MTP2 path for a one-request cohort
and batched MTP1 for larger cohorts. Reuse the same owned target state rather than
duplicating model weights/caches. This is a future integration step, not support
claimed by this experiment. Remaining single-request launch/transport gaps may
also justify a separately gated fixed-shape graph study; process timeline gaps
are not device-wide idle or a prediction of graph speedup.
