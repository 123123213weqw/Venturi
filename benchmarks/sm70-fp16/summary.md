# Explicit SM70 FP16 port: operators, hybrid fixtures and real-weight layers

## Status and scope

The full goal remains **active**. Blocks now have an explicit typed SM70/FP16
path, built and exercised on V100, and a separately tested SM89 FP16/BF16 path.
Real original-derived Q8 GDN/full-attention layer checks pass for **synthetic
activations**, not complete trained-model hidden states, logits or generation.
No full-model quality, native production scheduler/sampling, continuous batching,
paged KV, matched engine throughput or universal speed claim follows.

**Checkpoint-fidelity issue discovered:** the current explicit FP16 profile also
casts the small BF16 a/b/conv weights to FP16. The real layer-0 check found 62 a
weights, 110 b weights and 53 convolution weights changed; **one convolution
weight underflowed to zero**. Maximum cast error is 2.9802322387695312e-08; no
conversion overflow/nonfinite value occurred. This is not assumed harmless.
The artifact itself is unchanged. A lossless FP32 storage/compute option for
these small parameters is the next required refinement, before declaring source
parameter fidelity or trained FP16 quality accepted. The current oracles qualify
the explicitly cast profile, not preservation of every original BF16 parameter.

## Implementation

- Blocks **ABI 9** adds a compiled architecture tag and FP16/BF16 typed Q8
  embedding/projection and GDN recurrence entry points. Old ABIs reject before
  binding new symbols. Existing ABI-8 capsules/libraries remain historical.
- `build_blocks_backend(..., architecture="sm70")` / CLI `--architecture sm70`
  builds explicitly on the V100 CUDA 12.9 host; the default remains SM89 on the
  4090D. Architecture is part of build flags/cache/provenance. No automatic
  compiler selection/fallback or local CUDA/Rust compilation occurred.
- Library/hardware must match exactly. SM70 rejects BF16 before weight upload,
  including direct C ABI type guards. Ada Q4 remains SM89-only.
- Q8 codes and FP16 scales are unchanged, not decoded into native dense model
  weights. FP32 gamma, gate transforms and GDN recurrent state stay FP32.
  Selecting FP16 explicitly changes activations/cache/output and currently the
  small a/b/conv parameters, with the limitation above. Default BF16 is not
  silently reinterpreted. Reference Torch execution is explicit, not a fallback.
- Generic vocabulary binding accepts an explicit FP16 profile and preflights
  unsupported target/dtype before payload reads. Geometry, ownership, alias,
  graph/capture and synchronous session failure/reset contracts remain intact.

## Final builds and regression evidence

| Environment | Evidence | Result |
|---|---|---|
| Local Mac | Full suite; opt-in CUDA cases skipped, fake compiler orchestration tests only | **640 passed, 283 skipped** |
| 4090D, Torch 2.9.1+cu130 / nvcc 13.0.88 | Full suite, native ABI 9 SM89 FP16/BF16 and existing regressions | **893 passed, 30 skipped** |
| V100, unchanged Torch 2.5.1+cu121 / nvcc 12.9.86 | Selected FP16 port + oracle/tool/build tests, CPU and native cases | **42 passed, 3 skipped** |

V100 is physical GPU 1 remapped by `CUDA_VISIBLE_DEVICES=1`. The selected suite is
not a full V100 run of legacy BF16 tests, which intentionally remain unsupported
on SM70. It covers Q8 65/129/5120-wide projections and embedding, independent
FP64 errors, GDN Dk=64/128/256 with/without normalization, packed/unpacked/chunk
parity, FP32 state, full miniature GDN Op checks, mixed hybrid integration,
stream/state independence and late failure/reset. Native Q8/GDN graph capture
prohibits allocations, views and numerical Torch fallback. CPU-only/capture and
SM70-only guard skips are explicit in logs. No existing per-Op bound was widened.

Both final architectures compile this CUDA source SHA256:
`01531ccdf40e75f926f4af5ccca1ac879552dd877463b8511878b9bf5e34f478`.

- SM70 library: `7a29f137caa9052a/libventuri_blocks.so`, SHA256
  `0e4ffccd34ea21a3c360b7a87927b480c890640d192eb7e809d9fb203ddbe28e`.
- SM89 library: `2c05202bbca99b24/libventuri_blocks.so`, SHA256
  `158aeff115a3272182c2c2b14721ee87f57e1e082025b244c7c527f06606b61d`.

`v100/sm70-qualification-final/` and `sm89/` retain compiler commands/output,
source/library hashes, test results and GPU state. An earlier build/report is
retained separately; `initial-kernel/blocks.cu` matches that initial build. The
final build additionally rejects BF16 at the direct elementwise C ABI boundary.

## Actual original-derived checkpoint layer qualification

`tools/qualify_qwen_q8_layers.py` recomputes the full artifact SHA256, verifies the
complete role schema, uploads **one layer at a time**, and compares each stage
with CPU FP64 represented-weight math on that stage's **actual native input**.
Q8 CPU oracles tile decoded rows and do not BF16/FP16-round the represented Q8
weights. Small a/b/conv values intentionally follow the selected FP16 casting
profile; source cast statistics expose that limitation rather than hide it.

Artifact: `3d635b87db004f42aefd6730e372e530`, SHA256
`b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8`.
The earlier independent original-source storage verification remains separate.

| Original layer | T | Checked stages | Resident layer tensors | Largest rounded FP64 stage error |
|---|---:|---:|---:|---:|
| 0, GDN | 1 | 19 | 407,757,696 bytes | 0.0009765625 |
| 0, GDN | 8 | 19 | 407,757,696 bytes | 0.001953125 |
| 3, full attention | 1 | 16 | 395,552,768 bytes | 0.0009765625 |
| 3, full attention | 8 | 16 | 395,552,768 bytes | 0.001953125 |

Each element passes its existing magnitude-dependent criterion; these maximum
errors are not a whole-model tolerance. GDN FP32 state max errors are
1.385598871750382e-08 (T=1) and 2.4638430895507213e-07 (T=8). Full native layer
versus manual native composition/state/cache is bit-exact for these cases.
Norm is separately qualified and actual-input fused/staged rotary parity is
bit-exact; the retained earlier norm/RoPE cancellation counterexample is not
hidden by widening a cascade bound.

The final job completed normally, and the artifact inode/size/mtime remained
unchanged through hash/qualification. Its ~110-second wall duration includes
hashing, uploads, CPU decoding/oracles and synchronization: **not layer latency,
decode TPS or an engine benchmark**. Only two layers and short synthetic inputs
were checked. Full trained logits/generation/quality remain unqualified.

## GPU capacity and performance limitations

Existing llama/Python services remained running. V100 free memory was about
3.0 GiB on the selected GPU before these small checks; background busy reached
92–94%. The 4090D retained ~14–15 GiB of other process allocations. These are
correctness runs, not isolated timing. No service, driver setting, SSH trust
policy or environment dependency was changed. No Rust command ran.

`RmProfilingAdminOnly=1` is recorded. Achieved occupancy, SM/DRAM/cache/stall
counters were not collected; no repeated profiler campaign ran. Snapshot busy
percentage is not achieved occupancy or bandwidth. There is no new matched
unprofiled baseline/change timing, CPU/event/wall throughput measurement, or
speed win. The earlier direct-Q8 prefill regression remains an open bottleneck.

Full Q8 residency still needs ~26.64 GiB for weights alone, above the 4090D's
capacity and current V100 free memory. A free sufficiently large GPU or a
separate **explicit**, measured layer-streaming/offload correctness path is
needed for full-model validation; offload must not silently replace the resident
API or redefine the production-native single/concurrent performance target.

## Reproduction

From the source capsule on a Linux V100 host with the existing dependencies:

```bash
PYTHONPATH=src python -m venturi.ops.build --component blocks \
  --architecture sm70 --nvcc /usr/local/cuda/bin/nvcc \
  --output-dir /absolute/path/to/sm70-build
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src VENTURI_SM70_LIBRARY=/absolute/path/to/libventuri_blocks.so \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q tests/test_fp16_port.py tests/test_qwen_q8_layer_tool.py tests/test_ada_build.py
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python tools/qualify_qwen_q8_layers.py /absolute/path/to/qwen3.5-27b-text-q8.ninfer \
  --blocks-library /absolute/path/to/libventuri_blocks.so --dtype fp16 \
  --expected-sha256 b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8 \
  --layers 0 3 --tokens 1 8 --output-directory /absolute/path/to/fresh-output
```

Use the correct visible GPU/free-memory check for the host. Output folders are
no-clobber; inspect the exact job PID/report before any restart. `source/` and
`source-sha256.json` preserve the current implementation. Next: lossless small
parameter execution, full trained-model validation, native runtime/sampling/
graphs, batch-aware state/paged KV and matched single/concurrent comparisons.
