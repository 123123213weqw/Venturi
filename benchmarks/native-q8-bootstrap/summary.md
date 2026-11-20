# Native Q8 bootstrap: qualified primitives, not a completed model

2026-10-19 (Asia/Shanghai). Goal remains **active**. No full independent Qwen generation, parallel-native runtime or competitive-engine speed win is claimed.

## Implementation and correctness

- Generic signed Q8/group-32 FP16-scaled projection descriptors, bounded-row explicit source quantization and shared native projection/embedding residency. MLP and standard decoder composition accept Q8 weights.
- Native causal convolution/history + staged SiLU; GDN gates with explicit beta rounding; staged gated RMSNorm; zero-centered norm preparation; strict small-parameter bindings for declared GDN layers.
- GDN frontend timing uses real layer-0 constants from the existing artifact and synthetic BF16 activations. Generic Q8 projection timing uses synthetic codes/scales at [17408,5120], NOT trained source weights.
- Every timed output passes independent CPU FP64 gates; convolution history is exact. Gate criteria and staged rounding are recorded per case. Native capture/no-numerical-fallback, metadata/alias/dtype, source isolation, shared-weight and chunk tests pass.
- Full 4090D suite: **699 passed, 22 skipped**. Local CPU suite: **519 passed, 202 skipped**. Ruff and diff whitespace checks pass. No Rust/CUDA compilation on the Mac.
- Retained FP64 Torch convolution reference chunk counterexample: one 2.78e-17 difference from shape-dependent elementwise dispatch. Only that reference comparison uses an epsilon-level chunk check; native chunk output remains exact and per-op native gates were not loosened.

## Matched, unprofiled primitive timing

Fixed input/state, 3 warmup calls and >=2 s per ABBA phase. Both sides use captured graph replay wrappers. Projection reference materializes exact represented Q8 weights as FP32 in preparation, not each call. Reference is diagnostic, not another competitive Q8 engine. Preparation/loading, checkpoint conversion, cache-management and sampling are excluded. Event intervals may include gaps; process CPU includes event waiting. Primitive calls/s is NOT model tokens/s.

| Case | Backend | Wall us/call | Event us/call | Primitive calls/s | Process CPU % |
|---|---|---:|---:|---:|---:|
| conv-t1 | torch-reference | 30.76 | 28.37 | 32509.00 | 99.96 |
| conv-t1 | native | 7.37 | 5.23 | 135743.57 | 99.98 |
| gates-t1 | torch-reference | 13.43 | 10.98 | 74467.92 | 99.95 |
| gates-t1 | native | 8.68 | 6.30 | 116567.52 | 99.96 |
| gated-norm-t1 | torch-reference | 22.21 | 19.66 | 45022.41 | 99.92 |
| gated-norm-t1 | native | 8.50 | 6.15 | 119760.02 | 99.96 |
| zero-norm-t1 | torch-reference | 13.16 | 10.51 | 75986.26 | 99.96 |
| zero-norm-t1 | native | 9.64 | 7.18 | 104964.47 | 99.96 |
| q8-projection-t1 | torch-reference | 381.40 | 378.14 | 2621.92 | 99.94 |
| q8-projection-t1 | native | 167.80 | 164.67 | 5959.41 | 99.92 |
| conv-t128 | torch-reference | 81.92 | 79.00 | 12206.93 | 99.94 |
| conv-t128 | native | 50.35 | 47.82 | 19861.77 | 99.95 |
| gates-t128 | torch-reference | 16.34 | 13.82 | 61187.33 | 99.95 |
| gates-t128 | native | 8.60 | 6.23 | 116313.77 | 99.95 |
| gated-norm-t128 | torch-reference | 42.04 | 39.09 | 23789.33 | 99.95 |
| gated-norm-t128 | native | 11.35 | 8.88 | 88118.72 | 99.96 |
| zero-norm-t128 | torch-reference | 24.99 | 22.47 | 40022.09 | 99.96 |
| zero-norm-t128 | native | 10.54 | 7.86 | 94894.38 | 99.97 |
| q8-projection-t128 | torch-reference | 728.50 | 725.71 | 1372.95 | 99.92 |
| q8-projection-t128 | native | 20533.88 | 20529.08 | 48.70 | 99.98 |

**Retained regression:** T=128 direct native Q8 projection is ~20.53 ms versus ~0.728 ms for the prepared FP32 reference (about 28x slower). T=1 is ~168 us versus ~381 us. The direct row-dot kernel does not reuse weights across token rows; a bounded dequantization/tiled GEMM route is needed. That is a code-structure inference, not a hardware-counter attribution. Do not promote the single-token primitive gain into an engine-speed claim.

## GPU diagnostics

NVML per-case/per-backend means after excluding the first second of phases; raw samples retained. Power/clocks were not forced.

| Case | Backend | Samples | GPU busy % | Memory-controller busy % | VRAM MiB | Power W | SM MHz | Memory MHz | Temperature C |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| conv-t1 | native | 10 | 29.80 | 0.00 | 822.00 | 96.50 | 2628.00 | 10251.00 | 41.00 |
| conv-t1 | torch-reference | 10 | 91.80 | 0.00 | 822.00 | 102.72 | 2790.00 | 10251.00 | 41.00 |
| gates-t1 | native | 10 | 24.00 | 0.00 | 822.00 | 92.92 | 2655.00 | 10251.00 | 41.30 |
| gates-t1 | torch-reference | 10 | 79.00 | 0.00 | 822.00 | 102.23 | 2790.00 | 10251.00 | 41.50 |
| gated-norm-t1 | native | 10 | 24.70 | 0.00 | 822.00 | 94.93 | 2655.00 | 10251.00 | 42.00 |
| gated-norm-t1 | torch-reference | 10 | 87.00 | 0.00 | 822.00 | 103.47 | 2790.00 | 10251.00 | 42.00 |
| zero-norm-t1 | native | 10 | 64.80 | 0.00 | 822.00 | 103.88 | 2790.00 | 10251.00 | 42.10 |
| zero-norm-t1 | torch-reference | 10 | 74.70 | 0.00 | 822.00 | 104.77 | 2790.00 | 10251.00 | 42.50 |
| q8-projection-t1 | native | 10 | 98.00 | 73.20 | 1298.00 | 400.35 | 2775.00 | 10251.00 | 53.80 |
| q8-projection-t1 | torch-reference | 10 | 99.00 | 100.00 | 1298.00 | 282.27 | 2782.50 | 10251.00 | 47.10 |
| conv-t128 | native | 10 | 94.60 | 0.00 | 912.00 | 155.28 | 2775.00 | 10251.00 | 46.80 |
| conv-t128 | torch-reference | 10 | 95.90 | 0.00 | 912.00 | 246.77 | 2775.00 | 10251.00 | 49.70 |
| gates-t128 | native | 10 | 24.40 | 0.00 | 872.00 | 95.50 | 2647.50 | 10251.00 | 43.50 |
| gates-t128 | torch-reference | 10 | 82.30 | 0.00 | 872.00 | 110.20 | 2775.00 | 10251.00 | 44.50 |
| gated-norm-t128 | native | 10 | 74.70 | 0.00 | 892.00 | 238.78 | 2775.00 | 10251.00 | 50.00 |
| gated-norm-t128 | torch-reference | 10 | 92.00 | 0.00 | 892.00 | 218.38 | 2775.00 | 10251.00 | 48.50 |
| zero-norm-t128 | native | 10 | 62.40 | 0.00 | 892.00 | 143.51 | 2775.00 | 10251.00 | 46.00 |
| zero-norm-t128 | torch-reference | 10 | 87.90 | 0.00 | 892.00 | 180.87 | 2775.00 | 10251.00 | 47.00 |
| q8-projection-t128 | native | 10 | 100.00 | 77.00 | 1356.00 | 422.90 | 2746.50 | 10251.00 | 55.50 |
| q8-projection-t128 | torch-reference | 10 | 99.90 | 66.60 | 1356.00 | 422.54 | 1879.50 | 10251.00 | 55.70 |

GPU busy is not achieved occupancy; memory-controller busy is not DRAM bandwidth utilization. Hardware counters remain restricted (RmProfilingAdminOnly=1; prior ERR_NVGPUCTRPERM). Achieved occupancy, SM/DRAM throughput, cache and stall counters were not measured. No profiling-permission change or repeated counter campaign; no profiler mixed into timings.

## Source acquisition and capacity

The user authorized original-source download. A dedicated CPU-only job on WZU_Server acquires Qwen/Qwen3.5-27B, pinned to official revision fc05daec18b0a78c049392ed2e771dde82bdf654. Direct official access failed with network-unreachable evidence; the HTTPS mirror returned the same pinned revision and acquisition resumed there. Selected source assets total 55,576,089,619 bytes; LFS/file hashes are checked before a complete status is published. Partial or sparse-file sizes do not mean verified completion.

The source job path is /home/wzu/codex-build/Venturi/source-download-job/ and the source target is /home/wzu/codex-build/Venturi/models/qwen3.5-27b-bf16/ (server resolves it under /home/data/wangyue/repos/codex-build/Venturi/). Existing services were not stopped. Download/source job runs on the V100 host, not the 4090D timing host.

All-Q8 text weights require about 26.6 GiB before KV/state/workspace; full residency on one 24 GiB 4090D is impossible. V100 needs a separate SM70/FP16 port and free VRAM; current services occupy its GPUs. Do not relabel the existing mixed-Q4/Q8 artifact as original-quality Q8.

## Reproduce

```bash
cd /home/test/wangyue/venturi-work/qwen/src
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src \
  /home/test/wangyue/rwkv-distill-2.9b/.venv/bin/python tools/qualify_native_q8_bootstrap.py \
  "$(cat /home/test/wangyue/venturi-work/qwen/staged-path.txt)" \
  --blocks-library /home/test/wangyue/venturi-work/qwen/own-blocks-build/8600ad41ddd46902/libventuri_blocks.so \
  --output-directory /absolute/path/to/a-new-measurement-directory
```

ABI 5. Build JSON, successful compiler log (empty), raw phase/qualification/telemetry data, full test logs and source hashes/capsule are retained. External Qwen runner remains unchanged; see external-runner.sha256.

Network recovery preserved host-key verification and no agent forwarding. Direct SSH and one proxy transport were intermittent; the existing authorized WZU_Server jump route recovered 4090D access. Current source transfer used successful rsync and, after a later protocol timeout, a verified-path compressed Python/CUDA source bundle; no Rust commands ran. No temporary failure was classified as a hard block.

## Next

Complete verified original source acquisition and separate Q8 artifact; integrate full GDN and Qwen gated-attention/partial-RoPE hybrid forward with request state; qualify full trained-model math/quality; then native scheduling/graphs, batched GDN and paged KV. Optimize Q8 prefill GEMM without silent precision loss. See docs/native-q8-goal.md for the acceptance contract.

## Acquisition snapshot

Recorded source state: **downloading**, process present=True, completed shard files=0; completed shard bytes=0. Partial allocated bytes=8,178,900,992 are a disk-usage estimate, **not verified downloaded bytes**. Read source-acquisition-state.json for the timestamp and pinned source metadata.
