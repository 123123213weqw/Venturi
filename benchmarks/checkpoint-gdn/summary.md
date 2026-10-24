# Checkpoint vocabulary and GDN qualification

2026-10-18, School_4090D, RTX 4090 D (SM89), CUDA 13.0, Torch 2.9.1+cu130. Compilation and CUDA tests ran only on the remote server. No model downloads.

## Scope

- New native Q8 vocabulary lookup/head, strict NInfer binding/upload API, dense shared embedding/head support, and GDN recurrence.
- Actual Qwen artifact: 64 layers (48 GDN, 16 full attention); Q/K heads 16, V heads 48, key/value dimensions 128. Untied vocabulary matrices are Q8/group-32, each [248320,5120].
- Complete classic dense fixture adapter also connects checkpoint -> embedding -> independent decoder layers/KV -> final norm/head -> greedy generation. No real Llama checkpoint was available; generated fixtures only.
- NOT a complete independent Qwen forward, trained-model quality test, model TPS result, or llama.cpp comparison.

## Correctness

- Full suite on 4090D: **629 passed, 22 skipped**. Local CPU suite: **484 passed, 167 skipped**; optional CUDA/real-checkpoint integrations skipped. See full logs.
- Actual Q8 rows are bit-exact to independent artifact decoding after BF16 rounding. All **248,320 head logits** pass the independent CPU FP64 gate on the fixed represented input.
- Q8 head bound: abs(actual - rounded FP64) <= 2*BF16_eps*abs(FP64) + 2e-4. Largest rounded error 0.00048828125; largest unrounded error 0.0010843651034520008. Zero failures.
- GDN T=1/T=128 outputs pass the existing 2*BF16_eps*abs(FP64)+2e-5 gate; final FP32 state passes abs(error)<=2e-6+2e-4*abs(FP64). Native final-state maximum absolute errors: 7.27e-8 / 1.29e-7.
- Native GDN rounded output maximum errors: 0 / 6.103515625e-5 (unrounded maxima 5.86e-5 / 2.11e-4). No bounds were loosened after observing outputs.
- Small fixtures cover chunked/in-place recurrence, raw/normalized QK, head expansion, state orientation, zero updates, stream independence, graph capture/no numerical fallback, alias/shape/dtype guards and embedding scalar dispatch.
- Public NInferVocabulary API additionally loads both real matrices and composes real token embeddings with the head: exact embedding rows, all finite logits, selected rows including top-10 checked with FP64 (rounded error zero). This is vocabulary plumbing, not a decoder hidden state or meaningful generated text.
- Physical resident vocabulary weight bytes: **2,701,721,600 total**; no dense full-vocabulary expansion. Each GDN layer state is 3 MiB; this excludes convolution and ordinary-attention KV.

## Unprofiled timing

ABBA, >=2 s/phase, 3 warmup calls. Identical input, BF16 Q/K/V or hidden vector, FP32 GDN gates/state. GDN compares captured graph replay wrappers with fixed initial state and separate state_out; the head compares eager prepared calls. No checkpoint loading, ID upload, sampling, or full-model decoding is timed. Event intervals include possible host gaps. Calls/s below are primitive calls, NOT model tokens/s.

| Primitive | Backend | Wall us/call | Event us/call | Primitive calls/s | Process CPU % |
|---|---|---:|---:|---:|---:|
| gdn-t1 | torch-reference | 49.89 | 47.16 | 20049.14 | 99.96 |
| gdn-t1 | native | 22.03 | 19.45 | 45403.29 | 99.95 |
| gdn-t128 | torch-reference | 3584.76 | 3581.13 | 278.96 | 99.96 |
| gdn-t128 | native | 1852.74 | 1849.51 | 539.74 | 99.96 |
| text-output_head | torch-reference | 106265.04 | 106260.39 | 9.41 | 99.93 |
| text-output_head | native | 2294.94 | 2291.07 | 435.74 | 99.95 |

The Torch Q8 reference repeatedly dequantizes 512-row tiles and submits many allocating FP32 matmuls. It is a diagnostic reference, **not another optimized Q8 kernel**; its large gap must not be advertised as an engine/model speedup. Native Q8 projection is a direct FP32-reduction baseline, not tiled prefill GEMM. Approximately 100% process CPU means roughly one busy core here, including CUDA event waits; no CPU-efficiency improvement is claimed.

## GPU diagnostics

NVML samples exclude the first second of each phase; raw telemetry retained. Per-case/per-backend means:

| Primitive | Backend | Samples | GPU busy % | Memory-controller busy % | VRAM MiB | Power W | SM MHz | Temperature C |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gdn-t1 | native | 10 | 86.20 | 0.00 | 862.00 | 301.52 | 2775.00 | 52.50 |
| gdn-t1 | torch-reference | 10 | 93.80 | 0.00 | 862.00 | 188.37 | 2782.50 | 46.00 |
| gdn-t128 | native | 10 | 100.00 | 0.00 | 878.00 | 294.57 | 2775.00 | 53.00 |
| gdn-t128 | torch-reference | 10 | 100.00 | 0.00 | 878.00 | 231.38 | 2782.50 | 48.60 |
| text-output_head | native | 10 | 100.00 | 77.00 | 2218.00 | 406.39 | 2775.00 | 54.50 |
| text-output_head | torch-reference | 12 | 19.08 | 7.17 | 2218.00 | 126.45 | 2626.25 | 44.00 |

NVML GPU busy is **not achieved occupancy**; memory-controller busy is **not DRAM bandwidth utilization**. Hardware counters remain blocked by RmProfilingAdminOnly=1 (prior ERR_NVGPUCTRPERM); no privileges or profiling settings changed and no repeated counter campaign was run. No new profiler attribution/timeline campaign was needed for this functional milestone; achieved occupancy/SM throughput/DRAM bandwidth/cache/stall metrics were not measured. All throughput runs were unprofiled; clocks/power limits were not forced.

## Reproduce on the remote server

```bash
cd /home/test/wangyue/venturi-work/qwen/src
PYTHONPATH=src /home/test/wangyue/rwkv-distill-2.9b/.venv/bin/python -m venturi.ops.build \
  --component blocks --output-dir /home/test/wangyue/venturi-work/qwen/own-blocks-build
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src \
  /home/test/wangyue/rwkv-distill-2.9b/.venv/bin/python tools/qualify_checkpoint_gdn.py \
  "$(cat /home/test/wangyue/venturi-work/qwen/staged-path.txt)" \
  --blocks-library /home/test/wangyue/venturi-work/qwen/own-blocks-build/7eccd063b8097aec/libventuri_blocks.so \
  --output-directory /absolute/path/to/a-new-qualification-directory
```

Blocks ABI **4**, library SHA256 e13ad1780739921375889422bd4484041f3b6fbba6012a67aac0a02780a633ac. Build source/flags/compiler recorded in build.json; successful compiler output was empty. Earlier intermediate ABI-3 tests are retained as tests-initial.log; final/full suites use the ABI-4 library above. Final source capsule and hashes are retained. Post-run report enrichment is explicit in results.json; results-raw.json and phase JSONs retain original numerical/timing measurements.

The external Qwen runner checksum remains 3c7da6aeb337ee374ce912934c9335252e39326cd9cba3e2163696c06d15dfb4. No external engine sources or production runner were modified by this milestone.

## Next integration work

GDN causal convolution/history; gate projections/transforms and gated output norm; Qwen zero-centered (1+weight) normalization; packed multipart Q4 projection bindings; Q/K norm and attention output gate; partial/multimodal RoPE conventions; then the hybrid model/session adapter and full trained-checkpoint quality and speed comparisons. Ordinary attention is never substituted for GDN.
