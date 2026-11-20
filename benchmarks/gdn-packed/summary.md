# Packed GDN and projected-input mixer core

## Scope and qualification

Native 4090D/SM89, BF16 activations and FP32 gates/state. The benchmark uses synthetic represented inputs at the actual Qwen artifact geometry: 16 Q/K heads, 48 value heads, key/value width 128. It does not execute a trained Qwen model and is not a model-TPS or llama.cpp comparison.

The baseline copies strided Q/K/V slices into reusable contiguous buffers at T=128 and runs the existing native recurrence; the changed path reads the packed convolution output directly. At T=1 slices are already contiguous and the baseline has no copies. Packed/separate output and final state are bit-identical. Both pass independent CPU FP64 output and final-state checks; no tolerance was relaxed.

The projected-input mixer composition additionally covers convolution/history, gate transforms, packed recurrence and staged gated normalization. Its unit tests cover an independent staged oracle, whole-versus-chunked state, exact in-place history/state pairs, preflight-before-mutation alias rejection, CUDA graph capture with numerical/allocation fallbacks prohibited, and independent streams/request states. Input/output projections, residual/MLP and hybrid attention are not yet part of this core.

Local full suite: **530 passed, 214 skipped**. 4090D full suite after mixer integration: **722 passed, 22 skipped**. The earlier packed-only native suite is retained separately (715 passed, 22 skipped). Ruff and whitespace checks pass. No CUDA or Rust compilation ran on the Mac.

## Matched unprofiled measurements

Three warmups, fixed-state captured graph replay wrappers, ABBA phases of at least two seconds per condition. Read-only initial state and distinct state_out avoid benchmarking accumulated repeated history. CPU time includes CUDA-event waiting; CUDA event intervals can include host submission gaps. Two phase observations per condition are insufficient to establish a sub-percent latency improvement.

| Tokens | Condition | Wall us/call | CUDA event us/call | CPU % of one core | Calls/s | Unpack scratch bytes |
|---:|---|---:|---:|---:|---:|---:|
| 1 | unpack-native | 22.43 | 19.66 | 99.96 | 44573.54 | 0 |
| 1 | packed-native | 22.17 | 19.62 | 99.94 | 45117.99 | 0 |
| 128 | unpack-native | 1846.62 | 1842.88 | 99.99 | 541.54 | 2,621,440 |
| 128 | packed-native | 1841.17 | 1836.91 | 99.97 | 543.13 | 0 |

**Result:** T=128 removes three unpack copies and 2.5 MiB of scratch, but ~1.847 ms versus ~1.841 ms is effectively unchanged for this experiment. T=1 is also effectively unchanged (~22.43 vs ~22.17 us). Do not present either difference as a meaningful speed win. The recurrent kernel still loops sequentially over tokens and repeats reductions per value row; copy elimination does not address that dominant code path. This is code-structure reasoning, not hardware-counter attribution.

## GPU diagnostics

Whole-device NVML samples with the first second of each phase excluded. These are not achieved occupancy, FLOPS utilization or measured DRAM bandwidth.

| Tokens | Condition | Samples | GPU busy mean % | Mem-controller busy mean % | VRAM mean MiB | Power mean W | SM clock mean MHz | Memory clock mean MHz | Temperature mean C |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | unpack-native | 10 | 85.00 | 0.00 | 840.00 | 289.37 | 2775.00 | 10251.00 | 52.40 |
| 1 | packed-native | 10 | 86.50 | 0.00 | 840.00 | 311.80 | 2775.00 | 10251.00 | 53.00 |
| 128 | unpack-native | 10 | 100.00 | 0.00 | 862.00 | 279.41 | 2775.00 | 10251.00 | 53.50 |
| 128 | packed-native | 10 | 99.50 | 0.00 | 862.00 | 300.18 | 2775.00 | 10251.00 | 53.50 |

Achieved occupancy, SM/DRAM throughput, cache and stall counters were not measured: the profiling-admin policy remains restricted (`RmProfilingAdminOnly=1`, prior ERR_NVGPUCTRPERM). No driver privilege/system setting was changed and no repeated counter campaign was run. Profilers were not mixed into these timings.

## Provenance and reproduction

Blocks ABI **6**; rebuild explicitly. The measured library is `/home/test/wangyue/venturi-work/qwen/own-blocks-build/afd14a02433a8f9c/libventuri_blocks.so`. Build JSON/log, compiler/source/library hashes, source capsule, qualification JSON, raw ABBA phases, telemetry and full test logs are retained. The prior ABI-5 bootstrap capsule remains separate. The external Qwen runner was not edited; its unchanged SHA256 is retained in external-runner.sha256.

```bash
cd /home/test/wangyue/venturi-work/qwen/src
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src \
  /home/test/wangyue/rwkv-distill-2.9b/.venv/bin/python tools/qualify_gdn_packed.py \
  "$(cat /home/test/wangyue/venturi-work/qwen/staged-path.txt)" \
  --blocks-library /home/test/wangyue/venturi-work/qwen/own-blocks-build/afd14a02433a8f9c/libventuri_blocks.so \
  --output-directory /absolute/path/to/a-new-measurement-directory
```

Source transfer/build/test used the existing WZU_Server SSH jump route after direct-path failures. Host-key/TLS verification and no-agent-forwarding were retained. No unrelated services were stopped.

## Original checkpoint acquisition and remaining work

Snapshot at 2026-10-18T16:24:36.990878+00:00: revision-pinned original BF16 source acquisition on WZU_Server is **downloading**, process present=True, shard files present=2/11. These present files are not a fully verified checkpoint; whole acquisition publishes complete only after hashes pass. Partial allocated bytes (4,173,340,672) are a disk estimate, not verified progress.

Next: finish verified source acquisition and separately identified Q8 conversion/bindings; complete projections/residual/MLP and Qwen full-attention/partial-RoPE hybrid forward; qualify trained logits/state and generation; native serving with independent batched recurrent state and paged KV; separately qualify SM70/V100. Generic Q8 prefill still needs a weight-reusing GEMM path: the earlier direct-Q8 regression is preserved in the bootstrap report. Q8 text weights alone need ~26.6 GiB, so a 24 GiB 4090D cannot hold the entire model; current V100 services remain untouched. The implementation goal is **active, not complete**.
