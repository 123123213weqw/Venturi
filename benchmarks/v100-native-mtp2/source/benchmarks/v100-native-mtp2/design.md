# Native two-proposal MTP transactions

## Scope

New additive ABI16 exports implement an explicit greedy two-proposal path.
Default one-proposal generation remains unchanged. Native Python API selection
is `session.generate(..., draft_tokens=2)`; native CLI uses
`--mtp-artifact ... --mtp-draft-tokens 2`; the loopback benchmark uses
`--mtp-depth 2`. Old ABI16 libraries retain depth-one compatibility and reject
depth two with a rebuild-required error, not a numerical fallback. At least three
projection rows are required. Only fresh/reset generation is supported.

Drafting, verification, checkpoints, replay, greedy selection and cursor commits
are C++; Python owns loading, buffer/slot setup and the one native entry call.
No external forward, probabilistic acceptance, continuation, new weight profile,
KV quantization or automatic default enablement. The draft keeps its original
FP16 arithmetic; the target retains its explicit existing residual profile.

## Protocol

1. Prefill target and warm shifted draft cache exactly as before; emit the first
   target selection. The last emitted token remains pending/unconsumed.
2. Draft the next token from pending input plus the last verified normalized
   target hidden. The second speculative step consumes that proposal and the
   **normalized first draft output**, not a stale target hidden or raw residual.
3. Save target GDN/history and verify `[pending, proposal1, proposal2]` through
   the native target, capturing normalized target hidden rows and all selections.
   A near-budget tail uses two-input verification or one ordinary target step.
4. Accept only the matching proposal prefix, stopping at mismatch or EOS. Emit
   the correct target selection at rejection, or a final bonus on full acceptance.
   EOS itself and the last emitted bonus/correction remain unconsumed.
5. On partial commit, restore actual GDN/history and cursor, replay exactly the
   pending plus accepted input prefix, and assert the replay selection matches.
   Attention KV tails remain hidden by the committed cursor. Cursor rewind alone
   is not a valid rollback for the recurrent target.
6. The pending-input draft KV entry was based on a verified hidden and is kept.
   Rewind the attention-only draft cursor to that prefix, then overwrite later
   speculative entries using the **verified target** hidden rows. Single-row
   catch-up preserves the existing draft arithmetic. Corrected draft KV must
   equal depth one's committed draft KV, not merely have the same cursor.
7. Retain the last committed target hidden and assert both cursors equal the
   target input-prefix length. Exceptions poison both states; reset clears all
   state and statistics. Shared models retain their previous scratch mutexes;
   private GDN/history/KV/checkpoints/hidden buffers never cross requests.

The implementation reuses existing request-owned GPU buffers; no extra persistent
GPU allocation is introduced by this replay-based method. CPU depth counters and
small per-round token vectors are additional. Default depth-one target/draft math
and the original generation body are preserved, with reset bookkeeping extended.

## Counters and acceptance

Overall acceptance is accepted-prefix proposals / **all generated proposals**.
`proposed_by_depth` includes unused second proposals after first-depth rejection.
`verified_by_depth` counts only proposals whose comparison is reached;
`accepted_by_depth` counts matching prefix comparisons. Conditional per-depth
acceptance divides accepted / verified, not accepted / generated at other depths.
`unused_suffix_proposals` preserves that denominator distinction. Width2/3 calls,
draft rewinds, true-hidden catch-up inputs, forced-by-depth, checkpoint/verification/
rollback wall times and ordinary-target replayed input tokens are separately named.
Nested timings are not an additive GPU breakdown. Forced hooks stay test-only.

## Qualification

Fixture tests cover single/host-TP/PP, original FP16 and supported residual modes,
accept/reject at both depths, EOS positions, output limits, ties/shared vocabulary,
corrected draft KV versus MTP1, target state versus ordinary generation, failed
execution/reset/lifetime, and concurrent TP/PP request ownership. A repeated draft
oracle checks recursive normalized hidden/KV equations independently. CLI tests
check depth routing and preflight; old libraries stay usable at depth one.

Fresh trained qualification checks four original trajectories against ordinary
native target output/committed GDN/history/KV, complete pinned prior final-state
signatures, and freshly executed depth-one corrected draft KV. A separate three-
step recursive draft reference reconstructs represented Q8 weights from official
source tensors in FP64 with the existing relative-L2<=.02, centered-cosine>=.999
gates. Embedding reads are bounded token-row slices, not a full vocabulary copy.
No CPU shared-head or original-BF16/corpus-quality claim. Unchanged independent
target/teacher-vector and draft-block references are pinned and reused, not
relaxed or relabelled as fresh passes.

## Performance protocol

Pinned prior MTP1, candidate default MTP1 and candidate MTP2 get fresh C1/C8 screens
only after qualification. An internal improvement triggers three-repeat C1/2/4/8
MTP comparisons; a regression remains explicit and cannot become a default win.
All use the same 23 formatted prompt IDs/32 fixed greedy outputs, original-derived
Q8 target/draft, FP16 KV, resident host TP on physical GPUs1,0, eight preallocated
slots/capacity256 and projection rows32. No cache reuse; EOS ignored only for the
fixed-work speed case. Loading, slot/prefix allocation, warmup and profiling are
outside timed HTTP waves; per-request reset in the HTTP handler is inside them.
MTP uses independent scalar requests with model locks, not native cohort batching.
The previous stronger ordinary rows256 batch control is explicitly retained.

Separate fresh depth1/2 C1/C8 23-prompt/8-output traces retain timeline and compiled
resource attribution. Hardware counters stay unavailable under unchanged restricted
driver policy; theoretical residency/NVML busy are not achieved occupancy or
bandwidth. Exact library/source/controls and remote raw trace hashes are frozen.

## Prefix-state recovery follow-up

The first replay implementation is retained as an explicit negative ablation.
The trained story trace measured 12 replayed target inputs and about 0.298 s of
restore/replay, motivating a second, separately qualified opt-in recovery path.

`session.generate(..., draft_tokens=2, recovery="prefix")` and CLI
`--mtp-draft-tokens 2 --mtp-recovery prefix` select this additive ABI16 path.
The benchmark uses `--mtp-depth 2 --mtp-recovery prefix`. Defaults remain depth1
and checkpoint replay. Old libraries need the new optional exports for prefix
recovery; neither old export behavior nor existing ABI structure sizes change.
`session.prepare_prefix()` allocates before timing and is valid only on fresh/
reset sessions; generation can allocate lazily if explicitly selected without it.

During verification, the GDN kernel retains the recurrent state after the first
and second inputs. A separate bit-copy kernel saves corresponding convolution
histories from the original history and raw projected QKV. On partial commit,
restore that prefix and advance the attention cursor; **do not rerun any target
input**. This does not mean cursor-only rollback: actual GDN/history bytes are
restored. All-accepted verification already has the correct final state. Draft
KV still needs rewind and catch-up with true verified target hidden rows.

Prefix1 reuses the existing checkpoint buffer; prefix2 needs one extra recurrent
snapshot per session. For this checkpoint it is 153,944,064 bytes (146.8125 MiB)
on the primary device, or 1,231,552,512 bytes for eight prepared slots. These are
additional allocations, not part of the previously reported MTP request/backups.
Explicit extra-primary/secondary counters prevent omission or double-counting.
Preparation and allocation are excluded from timed serving waves; HTTP handler
reset is included, while native-generation/profile timers exclude reset;
retained buffers remain allocated across reset and are overwritten before use.

Trace descriptors belong to individual sessions and are attached only for that
native verification call, cleared on success and failure. Shared model scratch
locking and both GPU transport ownership fences are unchanged. PP has a private
trace per stage; ordinary generation and native batch paths do not attach one.
The GDN trace preserves the original eight logical warp reductions and adds
state stores, not reduced-precision arithmetic. Output/final/prefix state bytes
are checked against legacy GDN plus an independent FP64 state oracle for
DK64/128/256, packed/unpacked and normalized/unnormalized cases. History bit
copies have independent concatenation references and guard checks. Full forced
MTP transactions, state/EOS/failure/reset and concurrent ownership checks are
rerun with prefix recovery; the trained four-case output/target/draft gates and
three-step draft recurrence are also fresh for this binary.

One qualification wrapper subsequently looked up `recovery` instead of the
report's actual `target_recovery` key. The numerical command had already completed
successfully. Preserve that wrapper failure and its downstream stopped workers;
a reporting-only recovery validates and reuses the exact completed report by
SHA256 without repeating unchanged trained GPU/CPU work. This is not a failed
numerical gate reclassified as a pass.

## Memory telemetry is not identical to logical payload accounting

Across eight prepared slots, logical new recurrent storage is 1174.5 MiB.
Measured primary NVML device usage increases 22435 -> 23993 MiB (+1558), while
secondary stays 17227 MiB. This includes existing unrelated device allocations.
The extra 383.5 MiB above logical payload is not silently omitted or labelled
another model tensor. Allocation/residency overhead is plausible, but its exact
cause has not been isolated by this campaign. Ordinary candidate MTP1 and replay
MTP2 both retain 22435 MiB, so the difference is specific to prepared prefix mode.
Keep the measured peak for admission planning; no physical-memory equality or
universal low-overhead claim follows from logical byte counters.
