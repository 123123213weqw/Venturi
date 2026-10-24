# Native integration after the corrected pilot gate

This is not implemented by the operator pilot. Keep the accepted engine and its
Q8 profiles unchanged until the following explicit alternate profile qualifies.
Use the corrected source-v4 pilot, not the superseded per-sequence gate schedule.

## Production seams to change together

1. Add an explicit resident SM70, FP16, two-GPU host-TP profile layered on the
   qualified residual-A8/column-MLP profile. Reject unsupported PP, streaming,
   device count, geometry or arithmetic; do not silently fall back. Expose the
   new output-association change and private distributed-state memory in native
   and Python metadata. Keep existing ABI entry points/profiles working.
2. In `Model` weight ownership/upload, mark GDN QKV semantic rows and GDN output
   columns by role, reject aliasing across incompatible roles. Upload only each
   rank's encoded rows/columns and scales using bounded staging; do not retain
   both old and new full-model packed layouts. Split convolution and FP32 A/B,
   rate/bias; preserve source gamma dtype and convolution flags. Admission must
   include peer constants, scratch, private recurrent/history caches, MTP
   checkpoints and prefix1/2 backups before allocation.
3. Refactor the single-session and `Batch::group` GDN paths onto one native
   segment-aware dispatcher: batched QKV/Z/A/B/gates, per-request convolution
   and recurrence, per-head gated norm, column-FP32 output sum. Reuse persistent
   streams/buffers and both host-buffer ownership fences. Private states stay
   distributed through prefill, decode, mixed/ragged cohorts and verification;
   no per-token canonical reassembly or Python numerical fallback.
4. Extend `Session`, `VerificationTrace`, `MTPCheckpoint` and `MTPPrefixes`
   ownership consistently. Every backup slot needs its device/stream; prefix
   commit, restore/replay, selective reset, failure poisoning and teardown must
   drain/copy both ranks and retain positions. The MTP draft can retain its
   existing explicitly qualified residual profile; no requirement to convert
   the full-attention-only draft into head-local GDN. Update canonical snapshot
   diagnostics to scatter peer Q/K/V histories and join head states untimed,
   preserving the existing external shapes rather than flat concatenation.

## Mandatory acceptance sequence

- Remote-only compilation from a freshly synchronized, fully hashed snapshot.
- Eligible synthetic two-rank operator and native-session tests, alongside
  unchanged odd-width fallback/rejection fixtures. Check pre-output canonical
  arrays and independent state/history bytes; output association uses unchanged
  reference tolerances, never a claim of original-output bytes.
- Exercise one/two/three verification rows, prefix1/2 restore and continuation,
  ragged prefill and noncontiguous request subsets, EOS, injected failures,
  reset/reuse, batch and MTP draft correction. Python numerical operations must
  be prohibited during native generation, and memory ownership reported.
- Independent trained teacher/reference qualification on all 98 rows of the
  pinned four cases, plus complete greedy trajectories/acceptance/private
  cache correctness. Do not infer full-model quality from the current layer0/62
  block reference. Preserve numerical counterexamples rather than loosening
  gates. Reuse unchanged original-source payload hashes with verified identity.
- Fresh alternating unprofiled C1/C8 engine screens against the accepted native
  profile at matched MTP depth, capacity, rows, checkpoint and input work.
  Only promote meaningful wins to three-repeat HTTP cohorts1/2/4/8 and separate
  timelines. Preserve the stronger external single49.8033 TPS reference and all
  native cohort baselines; no model TPS extrapolation from block speedups.

Achieved hardware counters remain unavailable under the unchanged driver
policy. Record actual launch geometry, static resource use, kernel/copy timing,
process activity gaps, CPU time and sampled NVML without renaming any of those
as achieved occupancy or measured DRAM bandwidth. The full external >=5% goal
remains incomplete until its complete acceptance matrix passes.
