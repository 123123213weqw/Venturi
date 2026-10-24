# Complete Qwen GDN decoder-layer composition

## Implemented path and limits

`ResidentQwenGDNLayer.from_q8` streams individual encoded weights into owned
resident operators and composes input norm -> QKV/z/a/b -> convolution/history,
gates, packed GDN recurrence, gated norm -> output projection/residual ->
zero-centered post-attention norm/SwiGLU MLP/residual. No external model forward
or hidden numerical Torch fallback runs in the native path. The host dispatcher
is still Python; a complete hybrid model/session and native execution scheduler
are not implemented by this layer milestone.

Each token-shape plan shares resident read-only weights and uses independent
caller-owned activation/scratch/state buffers. Flat/head views and scratch-prefix
views are created before capture. One arena is reused sequentially by projection
handles; the entire layer's metadata, writable intervals and closed handles are
preflighted before any projection or state write. Exact paired history/state
updates are allowed. Zero/reset, serialization, request ownership and invalidation
after asynchronous failure remain caller responsibilities; there is no hidden
cursor, rollback or batching scheduler.

## Tests

| Environment | Scope | Result |
|---|---|---|
| Mac | Full suite, native opt-in cases skipped | 573 passed, 222 skipped |
| 4090D, Torch 2.9.1+cu130, blocks ABI 6 | Full suite including native GDN layer | 773 passed, 22 skipped |
| V100 host, owned Torch 2.5.1+cu121 | CPU source/layer/dependency-worker tests | 43 passed, 8 skipped |

The final dependency-worker mutex assertion was additionally checked on the Mac
and V100 host (4 passed). It cannot launch a second converter before publication.
No package/environment changes were made; the existing V100 NumPy import warning
is retained, not suppressed. No V100 GPU backend qualification is implied.

Miniature independently framed source fixtures have H=32, I=64, Hqk=2, Hv=4/6,
Dk=64, Dv=13 and Kconv=4. They use nontrivial random floating values,
nonzero chronological history and nonzero state; they are **not trained Qwen
weights or quality evidence**. Checks cover T=1/9, head-repeat ratios 2/3,
continuation chunks 1/2/6, per-operator independent FP64 math on actual inputs,
exact manual-composition parity, native chunk equivalence in the tested cases,
input/state/scratch/constant alias rejection before writes, plan closure without
invalidating other shapes, two independent CUDA streams/request states and graph
capture with numerical Torch functions, tensor allocation and runtime view
creation prohibited. Wrong layer/precision is rejected before payload reads.

## Numerical counterexamples retained, not hidden

The initial whole-chain BF16-vs-FP64 bound failed for Hv=6/T=9. Tracing shows the
first native discrepancy at QKV [2,320]: a valid FP32 dot-reduction rounds to the
neighboring BF16 value (maximum QKV absolute difference 0.00048828125). It changes
later convolved/normalized values and residual cancellation, resulting in a
0.00390625 maximum whole-layer output difference from the staged FP64 cascade.
The Torch BF16 CPU reference also uses shape-dependent BLAS: on the 4090D host
its chunked QKV differs from its whole-sequence QKV by up to 0.00048828125 and
its final output by 0.00390625. This CPU reference is not claimed chunk-bit-exact.
Native chunked and whole outputs/state are bit-identical for the tested cases.

No existing operator error floor was widened. Per-Op qualification follows the
existing project policy: independent FP64 formulas on each Op's actual inputs,
Q8 linear floor 2e-4, other BF16 Op floor 2e-5, and separate strict FP32 gate/state
bounds. All these checks pass, along with exact parity against manually composing
the same qualified native operators. FP64 reference-mode complete cascades also
pass. Whole-chain BF16/FP64 differences remain a diagnostic and trained-quality
limitation; this is **not a precision fix or a claim of full-model correctness**.
Trained logits/state, prompt chunking, generation and quality acceptance remain
required before inference-speed claims.

A separate verifier-harness failure was repaired: CPU FP64 `.cpu().double()` can
alias the original tensor. An in-place SwiGLU overwrote a supposedly saved oracle
input. The verifier now clones that saved value; runtime math and tolerances did
not change. Original failures, detailed stage/chunk diagnostics, reproducer and
fixture/source hashes are retained under failures/ and source/.

## Performance / provenance

No speed optimization or competitive speed claim is made for this composition
milestone. Existing unprofiled primitive measurements and the large direct-Q8
prefill regression remain in their prior capsules; no profiler or repeated
counter campaign ran here. Achieved occupancy/SM/DRAM/cache/stall counters remain
unmeasured under RmProfilingAdminOnly=1; prior NVML busy is not reinterpreted.

Blocks ABI 6 uses the unchanged native library
`/home/test/wangyue/venturi-work/qwen/own-blocks-build/afd14a02433a8f9c/libventuri_blocks.so`.
Source/build/compiler hashes, raw full-suite logs and the unchanged external
runner hash are retained. No CUDA or Rust compilation ran on the Mac. An SSH
banner failure occurred at the V100 jump connection before execution; routing
that hop through the existing Tailscale transport recovered the full native test
run. Host-key/TLS verification and no-agent-forwarding were preserved. No unrelated
service was stopped or modified.

## Source/conversion pipeline

At 2026-10-18T17:12:45.150022+00:00: original BF16 source status is **downloading**, **9/11 shard files present**. Partial allocated bytes (0) are not verified download progress. The source PID and one-shot dependency-worker PID were checked live with matching command lines, not inferred from stale state files.

One-shot worker PID 2741760 waits for existing download PID 2573351 to publish
complete source verification. It then invokes the CPU exporter in the existing
owned Torch environment. It never restarts downloads/stops services; observer
expiry does not mean the download terminated. A process lock prevents duplicate
exporters; existing artifact/report paths require inspection rather than overwrite.
Read `/home/wzu/codex-build/Venturi/q8-export-job/state.json`, verify live process
command lines, and inspect the actual output/report before relying on saved status.
The planned output is `/home/wzu/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer`.
At the recorded snapshot conversion had **not started**; the source was incomplete.

## Next required work

- Finish verified original acquisition and full Q8 conversion; validate actual
  source bindings/payload and trained layer math, not just fixture/index coverage.
- Implement Qwen full attention: zero-centered Q/K norm, per-head query/gate
  layout, gated output and partial/interleaved-mRoPE text positions. Interleaved
  mRoPE axis selection is not the same as even/odd rotary pairing; text-only
  equal-axis position IDs must be handled explicitly.
- Compose hybrid model/session, then trained-logit/state/generation/quality gates.
- Native execution/sampling/graphs, batched independent recurrent state and paged
  KV; SM70/FP16 qualification; matched full-model single/parallel benchmarks.
- Fix Q8 prefill GEMM and recurrent prefill bottlenecks based on real-model data.

The full goal remains active. Q8 text weights need ~26.64 GiB before state/KV;
there is still no full residency on a 24 GiB 4090D. A free 32+ GiB GPU was requested
for later full-model tests; other V100 services remain untouched.
