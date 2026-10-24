# Original Qwen source bindings and streaming Q8 export

## Scope

This milestone implements the complete checkpoint role schema, a bounded CPU-only
source converter and encoded runtime weight access. It does **not** implement a
whole Qwen model/session or qualify trained logits, generation, quality, native
single/concurrent serving, V100 kernels or a competitive speed win.

## What changed

- `QwenSourcePlan` preflights all original text roles without reading payloads.
  Against the pinned Qwen3.5-27B index: **851 text tensors**, **348 vision/MTP
  tensors explicitly excluded**; no unknown text weights silently ignored.
- Main vocabulary/projection/MLP matrices use signed Q8/group-32 FP16 scales.
  Small a/b projections, raw norm parameters and convolution remain BF16;
  A_log/dt_bias are FP32 (exact widening when original storage is BF16).
- Planned text payload: **28,601,606,144 bytes = 26.6373 GiB**: 402 Q8 matrices,
  353 BF16 parameters and 96 FP32 vectors. This still cannot fully reside on one
  24 GiB 4090D, before any state/KV/workspace.
- `export_qwen_q8` reads bounded safetensors row slices, quantizes using the
  stored FP16 divisor, and streams separate encoded planes directly to disk.
  No complete BF16 or Q8 model is materialized as application-owned tensors.
- Full source metadata/hashes and source-file stability are checked. Incomplete
  acquisitions are rejected. Explicit unverified local-source opt-in is labeled
  as unverified, never as recovered original precision. Existing outputs and
  original/mixed-Q4 checkpoints are never replaced. Publication is atomic and
  private temporary files are removed on ordinary failure.
- Per-head `[query D | gate D]` row order and raw zero-centered norms are
  preserved. Convolution alone changes `[channels,1,kernel] -> [kernel,channels]`.
- `QwenQ8Weights` preflights the entire unfused role set before a payload read or
  GPU allocation and reads one encoded linear/one bounded floating parameter
  for explicit native upload. The v3 container is a storage reuse, not a claim
  of compatibility with external NInfer model-forward bindings.

## Validation

| Environment | Scope | Result |
|---|---|---|
| Local Mac | Full suite, CUDA opt-in cases skipped | 558 passed, 215 skipped |
| 4090D, Torch 2.9.1+cu130, ABI 6 | Full suite including new exported-artifact -> native Q8/dense/norm integration | 751 passed, 22 skipped |
| V100 host, existing owned Torch 2.5.1+cu121 environment | Source schema/export/binding tests on CPU, GPU integration skipped | 28 passed, 1 skipped |

New checks cover independent safetensors framing, nearest-even codes and stored
scales, row chunk sizes 1/7/256, partial groups and K padding, explicit head-gate
boundary rows, asymmetric convolution taps, raw gamma, tied-storage ownership,
unknown/missing/quantized metadata, all-role preflight before reads, bounded
source row accesses, no CUDA in CPU export, source checksum/mutation rejection,
nonfinite data, low disk, no-clobber publication races and report preflight.
Native integration exercises loaded Q8 QKV/query projections, BF16 gate
projections and zero-centered norm against independent FP64 math. Existing
GDN/vocabulary binders consume the generated artifact in tests.

The first V100 CPU test attempt failed in the *fixture writer*: the installed
safetensors save helper required absent NumPy. This was not hidden or fixed by
changing a shared environment. Fixtures now independently write the safetensors
framing with bounded small CPU payloads; all source tests pass without NumPy.
The import-time Torch NumPy warning remains recorded. The failure log is retained.
Production artifact writes use direct bounded tensor-buffer views, not NumPy.

Ruff and whitespace checks pass. Source/build/test logs, source capsule, upstream
metadata hashes and the unchanged external runner hash are retained. No local
CUDA/Rust compilation occurred. No inference-performance optimization/timing
claim was introduced in this milestone, so no new inference profiler campaign
was run; the earlier Q8 prefill regression remains in the bootstrap report.
Hardware-counter policy remains restricted; prior NVML data is not relabeled as
achieved occupancy/SM throughput/DRAM bandwidth. Existing GPU services were not
stopped or changed.

## Actual original-source sample check

One *complete source shard* was independently SHA256-checked against its pinned
LFS digest before row sampling. Six actual BF16 rows (0, 42, 248319 from the
embedding and output head) were quantized on the V100 host with Torch 2.5.1.
All codes and FP16 scales exactly match an independent Python nearest-even
oracle. This is row-level quantization evidence, not complete source validation,
trained-model logits or a quality score.

| Source | Row | Max absolute weight quantization error | Relative L2 weight error |
|---|---:|---:|---:|
| lm_head.weight | 0 | 0.000268936 | 0.00544348 |
| lm_head.weight | 42 | 0.000206470 | 0.00535077 |
| lm_head.weight | 248319 | 0.000437260 | 0.00851616 |
| model.language_model.embed_tokens.weight | 0 | 0.000488281 | 0.00771489 |
| model.language_model.embed_tokens.weight | 42 | 0.000626564 | 0.00828271 |
| model.language_model.embed_tokens.weight | 248319 | 0.000039995 | 0.00521264 |

Shard: `model.safetensors-00001-of-00011.safetensors`. SHA256: `9019228d172c87d5603266c2d56672d119e838facffa164de803a1ebf0d716d2`.

## Live acquisition snapshot

At 2026-10-18T16:47:33.123538+00:00: status **downloading**, **4/11 shard files present**,
PID 2573351 present and its command matches the original download helper.
Partial allocated bytes (6,039,805,952) are only a disk estimate, not
verified download progress. Acquisition is not complete until the helper checks
all declared file hashes and atomically publishes a complete manifest. The
revision remains `fc05daec18b0a78c049392ed2e771dde82bdf654`. Network metadata timeouts are handled
with bounded backoff/retries; no duplicate download was launched or healthy job
restarted. Verified SSH/TLS and no-agent-forwarding were retained.

## Next required work

1. Finish the verified acquisition; execute the full CPU converter, then inspect
   every real source/output binding and verify the separate artifact/report.
2. Compose the complete native GDN layer: input norm, QKV/z/a/b projections,
   mixer/state/history, output projection, residual and MLP. The projected-input
   core is implemented; complete layer forward is not.
3. Implement full Qwen gated attention/QK norms and partial/interleaved RoPE,
   then the hybrid model/session and trained checkpoint correctness/quality.
4. Native serving/sampling/graphs, independent batched GDN state and paged KV;
   qualify SM70/FP16 and matched whole-model single/parallel comparisons.
5. Fix the measured direct-Q8 prefill GEMM gap without silently rounding decoded
   represented weights or loosening correctness gates.

A free 32+ GiB GPU was requested for later full-Q8 validation. Other V100 services
remain untouched. The goal is active; none of the remaining whole-model
requirements is inferred from these storage and operator tests.
