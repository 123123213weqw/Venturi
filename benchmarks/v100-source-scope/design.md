# Experimental lexical resident source-validation scope

Previous turn is progress: capsule40 froze unchanged-engine evidence of6674
descriptor/path pairs per C1 MTP2 generation. Independent CPU loops suggest
11.10ms/~1.40% work, not critical-path time; its >1%/>5ms opportunity gate passed.
This turn must test a real implementation rather than restate the diagnosis.
The full single+HTTP1/2/4/8 >=5% objective remains unchanged and incomplete.

## Execution-only opt-in contract

New boolean `build_blocks_backend(..., resident_source_scope=True)` and explicit
SM70-blocks build flag select a separate provenance/cache key. The original
per-weight checks stay the default. No arithmetic, Q8 weights/scales, small
parameters, precision, GDN/KV state layout, proposal count, acceptance or timing
policy changes. SDK metadata and native mode query disclose execution policy.

Within native resident Session.forward, Batch.group and MTP draft_step only,
after a forced source entry check, a thread-local lexical owner scope suppresses
redundant `weight()` and resident-column-MLP internal checks. Mandatory exit checks
after GPU synchronization remain forced, before cursor commit; public/native
metadata guards, unscoped/foreign model checks, every load/upload/read and every
streaming per-weight guard remain. Nested scopes restore owners on return and
exception; streaming scopes clear rather than inherit a resident owner. There
is no pointer/activation cache, elapsed-time cache, global validation epoch,
cross-thread owner state or source-stamp refresh. Modified sources still require
model recreation; reset must not bless a changed stamp.

The integrity contract intentionally changes detection cadence for resident
compute: a modification can be reported at the forced exit rather than at the
next internal projection. This is acceptable only if failed work returns no valid
generation and existing failure/reset/private-state contracts remain correct.
No early-abort timing or resistance to privileged timestamp forgery is claimed.

## Correctness and measured adoption gates

1. Preserve all40 frozen capsules. Full named-snapshot source sync/manifest
   before remote-only default/scoped CUDA compilation. Fake-compiler local tests
   verify boolean/architecture rejection, default behavior and cache separation;
   no local Rust/CUDA compilation.
2. Deterministic disposable-fixture pause after GPU synchronization/before native
   exit check, then mtime/path replacement: require rejection for resident,
   streaming and column TP scalar, target/draft MTP, and native cohorts. Check
   failure/no cursor commit, refusal to reset a changed source and successful
   recreation. Check nested/foreign/streaming owner scopes, exception unwinding,
   private concurrent requests, failure/reset and closed owners. Preserve exact
   default/scoped fixture bytes versus qualified6d43160 control; run full native
   GPU regression including existing forced-prefix/EOS/ragged/selective-reset
   checks. Test hooks remain unarmed in all inference and benchmark commands.
3. Because arithmetic and source payloads are unchanged, reuse pinned independent
   trained-quality gates by exact profile/payload identity, but run fresh trained
   four-case target/MTP trajectories/corrected draft states and mixed8/selective
   reset for the new SO before timing. Do not repeat unchanged represented-weight
   source-equation/corpus gates as if they independently test this host change.
4. Three-repeat unprofiled C1/C8 screen, alternating per-weight/scoped arms, fixed
   complete HTTP cohort/native timing scopes and qualified residual draft. Test
   scalar MTP2 and batch MTP1 separately; preserve stronger ordinary256 C8.
   Promotion requires >1% C1 post-first decode or C1 HTTP, at least two improving
   paired C1 trials, and no >1% C1 HTTP or >2% C8 HTTP regression. A promoted path
   gets three unprofiled1/2/4/8 repeats per arm. A failed screen is retained as a
   negative result, not promoted via syscall-count reduction alone.
5. Separate source-syscall audit must confirm reduction on the actual new SO;
   separate CUDA timeline/resources and NVML/CPU/latency/VRAM must disclose the
   actual critical-path effect. No strace/nsys timing becomes serving TPS, no
   counter privilege changes or achieved-occupancy guesses. Stronger external
   controls remain pinned; no fresh expensive external campaign unless a final
   near-goal candidate justifies it.

Only owned workers/samplers/server groups may be stopped. Preserve original
3651/3189MiB unrelated VRAM, verified direct WZU_Server route, driver policy1,
source identities and failed artifacts. No automatic host switch, commit or push.
