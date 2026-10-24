# Native semantic-head GDN ownership

The frozen operator pilot is now wired into the actual native model, single
requests, packed target batches, MTP2 checkpoint/prefix recovery and MTP1 batches.
The explicit opt-in profile is `q8-residual-column-mlp-head-gdn`. The original
constructors/default arithmetic remain available. Public C descriptor sizes and
existing export signatures are unchanged; new exports are additive.

## Layout and hot path

- Each selected SM70 rank owns corresponding Q/K heads and value-head groups.
  QKV codes and scale planes are permuted together, in bounded native load-time
  staging. GDN output is column-split at a group32-aligned boundary. Large old
  row-packed copies are replaced, not kept as a second full model layout.
- Small A/B, convolution, rates, bias and gamma are copied into native ownership
  with original bytes/flags. Native FP16 A/B projections retain FP32 weights;
  source gamma and convolution precision flags remain explicit.
- Both ranks share one normalized input broadcast, run QKV/Z/A/B/gates, then
  independent per-request convolution/GDN and gated norm. Their output partials
  stay FP32 through host transport/sum, then cast once to FP16. This changes
  output FP32 association and is qualified as new arithmetic, not old-byte parity.
- Head workspaces live once per serialized model, not per request. The new
  profile does not allocate unused legacy full-GDN workspaces in each Session.
  Scalar segments use stack-backed spans; packed segments reuse one reserved
  vector across layers. No Python numerical operation or per-layer vector
  allocation is added to generation.
- Both reusable cross-device event fences protect pinned input/partial buffer
  reuse across head GDN, column MLP and following projections. Private cache
  content is never joined on the host during generation.

## Distributed requests and transactions

Every request owns primary/peer history and packed FP32 state. Checkpoint slots
record layer, kind, rank and device; copies and prefix1/prefix2 restores use the
owning stream. Reset zeros both ranks, and existing failure paths drain both
streams before the request can be reused. Canonical snapshot diagnostics scatter
Q0/Q1/K0/K1/V0/V1 history channels and concatenate value-head states, untimed.
Attention KV remains primary-owned FP16 and uses the existing committed cursor.

Memory admission checks primary and peer workspaces/cache/backup allocations.
New metadata separates peer request caches and the peer components of MTP
backups. Compatibility note: legacy `primary_target_backup_bytes` and
`prefix_trace_extra_primary_bytes` mean first-target-stage totals across TP
ranks, not just physical rank0. The new physical-primary and peer component
fields split those totals; adding them again would double-count backups.

## Verification performed

- Native source-v2 compiled remotely to SHA
  `33a4becb5f9c8440492e9cc6352ea87fccab2692a01adc48e6cc756fa93bc7b8`.
  All 11 CUDA/native dependency hashes match source-v3, whose changes only close
  a build-dependency count assertion and expose the new profile to test tools.
- 26 focused two-V100 tests pass, including T1/T2/T3/T8 complete-model chunks,
  private ragged eight-request state, selective reset, EOS, injected failures,
  native-only generation, MTP2 forced acceptance/rejection and prefix recovery,
  and MTP1 packed cohorts1/2/4/8 with corrected draft caches.
- The broad source-v2 regression run has 2242 passing/719 skipped tests and one
  expected dependency-count assertion (10→11 headers), not a numerical failure.
  Source-v3 closes that assertion and runs the applicable draft arithmetic
  controls: 38 tests pass. Whole local suite: 831 pass/2131 skip.
- The actual trained target passes all 98 archived original-FP16 teacher rows,
  four complete trajectories, native generation versus diagnostic cache bytes,
  eight independent mixed-budget sequences and selective reset/continuation.
  Archived teacher tensor hashes are checked; the unchanged teacher campaign is
  not rerun. Report SHA:
  `6e4c1623bc10488bbcb92f92d26d9f0346c3f727db65673acb9808e3970dba50`.
- Fresh trained MTP2 block/recurrence/target/corrected-draft gates and packed
  MTP1 private-cohort/reset gates pass. The initial invocation with skipped block
  references is preserved as a CLI preflight failure; it was rerun with fresh
  reference checks, not bypassed or weakened.

These quality checks retain the original L2/cosine/KL/top1 thresholds. They do
not establish original BF16 corpus quality. All builds are remote; no local
Rust/CUDA compilation, commit, push, driver-policy change or unrelated kill.

## Performance status

The first unprofiled single-request screen measures 52.805 native post-first TPS
versus 48.440 for the previous profile, with the same new library, residual
MTP draft, two V100s and fixed work. This is a screen, not a final repeated win.
Alternating C1/C8 screens and promoted three-repeat1/2/4/8 matrices run before
separate timeline/resource diagnostics. Stronger historical external/native
controls remain retained. The full goal remains active pending final evidence.
