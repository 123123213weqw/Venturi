# Q8 hybrid model/session integration and completed storage verification

## Scope and status

The goal remains **active**. A complete mixed GDN/full-attention **operator-level
model/session adapter** now works on independently generated miniature source
checkpoints. The native numerical path uses owned CUDA kernels/cuBLAS, not an
external model engine. The host orchestrator is conservative synchronous Python.
This does **not** yet qualify trained Qwen-27B forward/generation/quality,
production native scheduling/sampling/graphs, paging/batching, SM70 or competitive
single/concurrent throughput.

## Original source and Q8 artifact

- Source: `Qwen/Qwen3.5-27B`, revision
  `fc05daec18b0a78c049392ed2e771dde82bdf654`; 11 shards / 19 selected assets.
- Official endpoint was unreachable; the recorded HTTPS mirror returned the
  pinned official revision. All selected source hashes were independently
  rechecked. Existing source and legacy mixed-Q4 artifact remain untouched.
- Published artifact on WZU_Server:
  `/home/wzu/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer`.
- Artifact ID: `3d635b87db004f42aefd6730e372e530`.
- File SHA256:
  `b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8`.
- File bytes: **28,601,956,352**; text tensor payload: **28,601,618,432**.
  Format counts: 402 Q8 matrices, 305 BF16 tensors, 144 FP32 tensors. The 48
  original GDN gated gamma vectors retain FP32 precision and its cast semantics.
- `qwen_export_verify` independently checks full source/artifact hashes,
  normalized config and source provenance, all **449 floating tensors**, and
  source rows 0/middle/last from **all 402 Q8 matrices (1,206 rows)** using scalar
  nearest-even quantization with the stored FP16 divisor. It never invokes the
  production quantizer. Maximum sampled absolute weight error: 0.00140380859375.
- This is storage verification, **not all-row Q8 recomputation or activation/
  trained-logit/quality verification**. Original BF16 remains the reference.
- Reports, pipeline completion and pinned manifest are in `conversion/`.

## Model/session behavior

`ResidentQwenModel.from_q8` validates supported geometry/RoPE and the memory
budget before any weight payload read. It streams one encoded matrix at a time;
no native dense model materialization, implicit offload or precision downgrade.
Embedding/head share one resident object when explicitly tied. Final norm is
zero-centered. Explicit Torch BF16/FP64 reference mode is not a native fallback.

`QwenSession.forward` supports chunked prompt ingestion and autoregressive token
continuation. Each request owns chronological conv history, FP32 recurrent state
and dense attention KV. Shape plans reuse weights. One workspace per layer kind
is reused sequentially across layers; only decode/max-prefill shape lanes are
retained, while other tail-shape lanes close after completion. Returned logits
own their storage. Independent request/stream ownership is tested, not a batched
serving scheduler or measured overlapping throughput.

All layer metadata is preflighted before numerical execution of a chunk. Public
position commits only after the entire call succeeds and CUDA completion is
synchronized. A later failure poisons the whole request, even if private state
partially advanced; reset to an empty sequence is required. There is no unsafe
GDN truncate/rollback. Reset zeroes both conv history and recurrent state. Plans
close only after work completes. IDs/tables remain alive through synchronization.
These synchronizations and Python validation overhead are deliberate baseline
limitations, not performance improvements.

## Correctness results

| Environment | Result | Qualification |
|---|---|---|
| Local Mac | **618 passed, 261 skipped** | Full suite; native CUDA opt-in cases skipped |
| 4090D, Torch 2.9.1+cu130, blocks ABI 8 | **853 passed, 26 skipped** | Full suite including native hybrid fixture integration |
| V100 host, unchanged Torch 2.5.1+cu121 environment | **42 passed, 9 skipped** | Model/source tests on CPU only; not SM70 qualification |

New integration cases cover four alternating GDN/attention layers, independent
original-source FP64 end-to-end logits, exact native manual-composition parity,
chunk continuation, tied embeddings, workspace sharing, state independence,
stream rejection, bounded retained lanes, owned logits, invalid IDs/capacity,
closed later-layer preflight, and late-layer/second-chunk failure poisoning and
reset. Native fixture chunk parity is bit-exact for tested cases. CPU BF16 BLAS
is shape-dependent, so CPU BF16 chunk bit-identity is not asserted. Existing
per-operator bounds and retained BF16/FP64 cascade counterexamples are unchanged.

A first CUDA test exposed a **256-byte residency estimate overcount** for tiny
unaligned vocabulary planes: storage-file inter-plane padding is not uploaded.
The estimate now counts encoded plane tensors, including K padding but excluding
file-only gaps. Exact actual-vs-planned resident bytes are tested on CPU and CUDA;
`failures/initial-native-residency-budget.log` preserves the failed run.

## Capacity and timing limitations

Real-artifact metadata gives **28,602,955,776 native resident tensor bytes**, before
CUDA context, allocator/workspaces, request state, logits and graphs. The extra
bytes versus file tensor payload reflect effective FP32 norm storage. A fully
resident model cannot fit one 24 GiB 4090D. `native-residency-plan.json` records:

| One request capacity | Dense BF16 KV | GDN state/history | Weights + request tensors only |
|---|---:|---:|---:|
| 4,096 | 268,435,456 | 153,944,064 | 29,025,335,296 bytes |
| 32,768 | 2,147,483,648 | 153,944,064 | 30,904,383,488 bytes |
| 262,144 | 17,179,869,184 | 153,944,064 | 45,936,769,024 bytes |

The V100s remain occupied by existing services; SM70/FP16 implementation and
qualification are also still required. No process was stopped. The 4090D had
~14–15 GiB occupied by another process during small correctness tests. Those
checks are not latency benchmarks. The earlier helper benchmark was guarded out
before timing; there is **no new CPU/event/wall/TPS speed measurement**. No matched
full-model comparison was possible. No inference speed win or occupancy gain is
claimed; workspace sharing here is structural correctness/footprint evidence.

`RmProfilingAdminOnly=1` is recorded; achieved occupancy, SM/DRAM/cache/stall
counters remain unavailable. No new profiler/counter campaign ran. Keep the
known direct-Q8 prefill regression rather than masking it with proxy results.

## Reproduction and next acceptance work

`source/` and `source-sha256.json` freeze the current code/tests/tools/docs.
`build.json` hashes the matching unchanged CUDA source and library. Native
compilation/tests ran on remote Linux only; no Rust compilation or local CUDA
builds ran. Existing external runner hash is retained unchanged.

Next: implement/qualify the SM70/FP16 path and obtain a sufficiently free GPU, or
explicitly design/measure offload with the same policy in both engines. Then
qualify trained logits/state/generation/quality, implement native whole-model
runtime/sampling and graphs, add batch-aware recurrent states/paged KV, and run
matched single/concurrent engine benchmarks. Synthetic integration is not a
substitute for these remaining goal conditions.
