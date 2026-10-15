# Venturi

A specialized, high-performance inference engine.

**Siphon loads the weights; Venturi runs the model.**

## First milestone: a model-agnostic loader

Venturi **inspects and loads checkpoint tensors**, and has an optional reusable
Ada Q4 Linear backend. It does **not yet execute complete models**.
The loading core has no architecture-specific configuration classes, required
weight names, layer counts, or model adapters. Architecture interpretation and
execution kernels are separate components, not part of checkpoint loading.

### What it does

- Loads a local safetensors file or single/sharded checkpoint directory.
- Uses Siphon for multi-file CUDA loading, with an explicit safetensors reference
  path for CPU tests and CUDA cross-checks.
- Inspects headers and validates shard indexes before allocating GPU weights.
- Preserves tensor names, physical shapes, storage dtypes, and packed integers.
- Retains optional `config.json`, including quantization descriptions, safetensors
  file metadata, and shard-index metadata without interpreting them.
- Returns tensors that remain valid after the loader closes.
- Reports synchronized load times, physical tensor bytes, and CUDA allocator delta.
- Verifies every loaded byte against the checkpoint, with bounded host-transfer chunks.

**Loading a checkpoint does not mean Venturi can execute its architecture.**
This layer validates storage integrity and consistency, not whether a model has
all the parameters its forward pass requires. `config.json` is optional and model
Python files are never executed.

### Quantized storage

INT8 weights and INT4 values packed into ordinary `int32`, `uint8`, or other
supported integer containers are loaded **as stored**. Scales, zero points, and
other tensors are preserved alongside them. Nothing is dequantized, transposed,
repacked, or silently cast. The original metadata is retained for future
model-specific execution components.

The loader supports byte-aligned safetensors dtypes available in the installed
PyTorch/safetensors stack. Backend/device support still applies. Native `.ninfer`
containers, sub-byte floating-point formats, original `.pth` files, GGUF, checkpoint conversion,
automatic downloads, and full-model inference are not implemented yet.

The optional operator API below interprets one explicit packed format; it does
not infer an execution format from arbitrary quantization metadata.

## Reusable Ada Q4 Linear backend

This is a standalone operator plugin, not a Qwen runner or a universal 4-bit
reader. It takes a weight matrix and activations without consulting layer names,
model configs, attention implementations, or speculative decoders.

### Supported contract

- Linux CUDA, compute capability **8.9**. RTX 4090D is the validation target;
  another RTX 4090 still needs a portability smoke test and its own measurements.
- Signed two's-complement Q4 codes in `[-8, 7]`, low nibble first, group size 64,
  FP16 scales, no zero points, K padded to 128: `q4_g64_fp16` /
  `row_split_k128_v1`. Scale multiplication and accumulation use FP32.
- Contiguous FP16 or BF16 activations `x[T,K]`, same-dtype output `out[T,N]`;
  computes `out = x @ W.T`, without bias or activation fusion.
- General SIMT route handles arbitrary positive N/K, including partial groups,
  row tails, and T up to 262,140. It is a functional baseline, **not a tuned
  high-throughput prefill implementation**.
- Retained Ada MMA small-token schedule: BF16, T=2..16, N divisible by 40,
  K in {5120, 6144, 25600}. Unsupported tuned requests fail explicitly; `auto`
  preserves this route where applicable.
- Dimension-independent MMA route: BF16, N divisible by 32, K divisible by 512,
  arbitrary supported T, using runtime dimensions and 16-token tiles. No model
  shape table is used. `auto` selects it for T>=2 when the retained route does
  not apply; `route="mma"` selects it explicitly, including T=1. Other operands
  use SIMT. Neither selection policy promises the fastest route on every card.

AWQ, GPTQ, GGUF, FP4, and other group sizes/packing conventions are **not**
interchangeable with this contract. They require explicit format adapters.
The current adapter can view a single encoded Q4 payload, but cannot open a
complete `.ninfer` artifact. No quantizer or model conversion is implicit.

### Build explicitly on the CUDA server

The loader and CPU tests do not compile CUDA. The optional backend uses a small
C ABI, avoiding a dependency on PyTorch's C++ ABI or Ninja. A CUDA toolkit with
Ada support and a compatible C++20 host compiler are required.

```bash
python -m venturi.ops.build --output-dir /absolute/path/to/venturi-native
# Prints the versioned path to libventuri_ada.so.
```

Compiler output and source/binary hashes are retained alongside the library in
`build.log` and `build.json`. Native `sm_89` code and `compute_89` PTX are included.
Building is never triggered by importing Venturi or by running an operator.

### Prepare once, execute without device allocations

```python
import torch
from venturi.ops import AdaBackend, Q4RowSplitWeight

# Obtain these planes from an explicit compatible checkpoint adapter.
# uint8 codes: [N, padded_K/2]; float16 scales: [N, padded_K/64].
weight = Q4RowSplitWeight(codes, scales, in_features=K)
backend = AdaBackend("/absolute/path/to/libventuri_ada.so")
plan = backend.prepare_linear(weight, tokens=8, dtype=torch.bfloat16, device="cuda:0")

x = torch.randn(8, K, dtype=torch.bfloat16, device="cuda:0")
out = plan.empty_output()  # Explicit allocation, before timing/capture.
plan.run(x, out=out)
```

`Q4RowSplitWeight.from_codes(integer_codes, fp16_scales)` packs CPU integer codes
exactly for tests/converters; it does not quantize floating-point weights.
`from_payload(uint8_payload, out_features=N, in_features=K)` views the code plane
followed by its 256-byte-aligned scale plane, validating the exact byte count.
This can wrap a compatible tensor loaded through Siphon without coupling its I/O
to native execution.

Preparation owns independent device copies, checks finite scales, initializes
the selected native route, and synchronizes preparation. It must run outside
CUDA graph capture. A plan has fixed T, dtype, and weight dimensions; its GPU
workspace requirement is zero bytes. Plans retain their backend library.

`run` writes the caller's output on PyTorch's **current stream for the plan's
device**. It performs no tensor allocation, full-weight dequantization, implicit
cast, synchronization, or shape-dependent compilation. It rejects noncontiguous
operands and output aliasing. Both MMA routes require 16-byte-aligned input.
There is no autograd implementation. Prepare and warm up before timing/capture.

The caller owns stream dependencies and lifetimes: keep the plan, inputs, and
outputs alive until asynchronous work finishes, and keep captured operands and
the plan alive across graph replays. When handing tensors between streams, use
PyTorch's `wait_stream`/events and `record_stream` or retain them explicitly;
Venturi does not silently synchronize every invocation. Do not mutate weight
contents or resize operands while they are in use. Independent outputs can be
used with the same prepared read-only weights on different streams.

Python is the orchestration layer, not a claim that per-layer Python dispatch is
free. A future model runner should capture its steady-state execution or use a
native execution plan. The benchmark reports eager and graph latencies separately
so Python launch overhead is not hidden behind kernel-only numbers.

### Native correctness and operator benchmarks

```bash
VENTURI_ADA_LIBRARY=/absolute/path/to/libventuri_ada.so python -m pytest -q
python tools/benchmark_q4_linear.py \
  --library /absolute/path/to/libventuri_ada.so \
  --output /absolute/path/to/q4-linear.json
```

Tests independently decode the signed codes and stored scales into FP64 and
compare native outputs, including non-Qwen dimensions, K/row/token tails, route
boundaries, full-size operators, nondefault streams, CUDA graph replay, owned
weight preparation, and execution allocation checks. Without the environment
variable, native tests skip and never invoke a compiler or allocate CUDA weights.

The benchmark uses **synthetic weights** and reports eager event latency and
amortized CUDA-graph operator latency, alternating `auto`/`generic` ordering.
It is not a tokens/second measurement, an inference benchmark, or evidence of
a speedup against llama.cpp. A full-model comparison also needs matched target
weights/quality, prompts, context, output length, and speculative settings.

### RTX 4090D validation

On the RTX 4090D, all **145 tests passed**, including the existing Siphon/FP16
loader acceptance tests. Native tests cover all three operator routes. The local
CPU suite passes 107 tests and skips the 38 opt-in CUDA cases. Packaged native
sources and attribution files were checked against the source tree.

Selected BF16 graph operator latencies (synthetic Q4 weights, five repetitions):

| N | K | T | Route | Median latency |
|---:|---:|---:|---|---:|
| 4096 | 4096 | 8 | Dimension-independent MMA | 12.18 microseconds |
| 5120 | 25600 | 8 | Retained small-token MMA | 76.97 microseconds |
| 5120 | 25600 | 128 | Dimension-independent MMA | 412.70 microseconds |

Execution showed zero additional PyTorch device allocations in the measured
routes. These are **operator microbenchmarks**, not model throughput; graph
latency excludes ordinary per-operator Python launch gaps. Raw eager/graph samples,
toolchain information, correctness/test scope, and tested source hashes are in
[`benchmarks/q4/validation.json`](benchmarks/q4/validation.json) and
[`benchmarks/q4/rtx4090d-linear.json`](benchmarks/q4/rtx4090d-linear.json).
The second physical RTX 4090 portability test has not been run.

Selected native schedules/helpers originate in NInfer, Cinference, and the Ada
research code on the 4090D. Their scoped Apache-2.0 license, notices, source
hashes, and adaptation notes are retained under
`src/venturi/ops/csrc/third_party/ninfer/`; no repository-wide license is added.

## Install

Python 3.10+ is required. Install the appropriate PyTorch build for your device
first, then install Venturi:

```bash
git clone https://github.com/123123213weqw/Venturi.git
cd Venturi
python -m pip install -e '.[test]'
```

For production loading, install Siphon from this repository, not an unrelated
package with the same name. Follow its platform/build prerequisites:

```bash
git clone --recurse-submodules https://github.com/123123213weqw/siphon.git
python -m pip install ./siphon
```

Siphon is imported only when its backend is selected. It is not vendored or
installed automatically. See [Siphon's installation instructions](https://github.com/123123213weqw/siphon#install).

## Inspect without loading GPU weights

```bash
venturi inspect /absolute/path/to/checkpoint --json
venturi inspect /absolute/path/to/weights.safetensors --json
```

Directory discovery uses:

1. `model.safetensors.index.json`, if present; otherwise a unique
   `*.safetensors.index.json` file.
2. `model.safetensors`, if there is no index.
3. All `*.safetensors` files in the directory otherwise.

An index is authoritative: unrelated shards are ignored, missing entries/shards
and duplicate tensor names are rejected. Multiple noncanonical indexes are
ambiguous and rejected. HF cache symlinks are supported. An explicitly selected
file loads only that file, even if a sibling index describes other shards.
Empty shards are rejected; empty tensors are valid on the reference backend.

## Load through Siphon

```bash
venturi load /absolute/path/to/checkpoint --device cuda:0 --json
```

Siphon is the default. Its automatic I/O selection and `SIPHON_*` environment
settings remain available. Explicit options override matching defaults:

```bash
venturi load /absolute/path/to/checkpoint --device cuda:0 \
  --siphon-backend URING --buffer-size 268435456 --json
```

Backend names are translated into Siphon's public enums. Comma-separated
candidates and the `BUFFERED` policy are accepted. Siphon is opened once for the
whole shard group, on the selected CUDA device, with `copy=True`: retained weights
must not alias a temporary rotating buffer.

The buffer option controls Siphon's ring allocation, not a cap on total model
memory. Allow headroom for the complete tensors **and** loader buffers; Siphon can
round or enlarge its buffer. No automatic CPU fallback, offloading, or eviction of
other workloads occurs. Device/storage types unsupported by the selected backend
fail rather than being converted.

The CLI loads, reports, and exits. Keep weights resident with the Python API:

```python
from venturi import inspect_checkpoint, load_checkpoint

checkpoint = inspect_checkpoint("/absolute/path/to/checkpoint")
model = load_checkpoint(checkpoint.directory, device="cuda:0")

print(model.config)  # Raw optional metadata, not an architecture config class.
for name, tensor in model.weights.items():
    print(name, tensor.shape, tensor.dtype)
print(model.stats.tensor_bytes)
```

The tensor mapping is read-only; tensor contents remain mutable. No tokenizer,
recurrent state, KV cache, or inference kernel is created.
For an explicitly selected single file, pass that same file to `load_checkpoint`
rather than its parent directory.

Use the reference loader explicitly for CPU validation:

```bash
venturi load /absolute/path/to/checkpoint --loader safetensors --device cpu --json
```

## FP16 acceptance: load, verify, and measure

Ordinary FP16 safetensors are the first full-checkpoint acceptance target. Require
FP16 floating-point storage without casting or modifying integer/bool buffers:

```bash
venturi verify /absolute/path/to/checkpoint --device cuda:0 \
  --require-float-dtype F16 --buffer-size 67108864 --json
```

The dtype contract is checked **before CUDA setup**. It is a storage-dtype check,
not proof that arbitrary integer tensors are unquantized model buffers. Verification
compares every stored byte, including NaN payloads, with the CPU safetensors
reference. It copies at most `--chunk-bytes` (default 16 MiB) back from the GPU at
a time. Loading and verification times are reported separately. A mismatch exits
nonzero and identifies the tensor and failing byte offset.

Use the same verifier from Python while retaining the loaded weights:

```python
from venturi import load_checkpoint, verify_loaded_checkpoint

model = load_checkpoint("/absolute/path/to/checkpoint", require_float_dtype="F16")
verification = verify_loaded_checkpoint(model)
print(verification.verified_tensors, verification.verified_bytes)
```

For repeated acceptance measurements in fresh processes, alternating loader order:

```bash
python tools/run_fp16_acceptance.py /absolute/path/to/checkpoint \
  --device cuda:0 --runs 3 --output /absolute/path/to/acceptance.json
```

Every measured run performs a complete byte comparison. This is a model-loading
measurement, **not** an inference benchmark. No page cache is evicted, so results
must not be described as controlled cold-start performance. The JSON memory
figures cover the PyTorch allocator, not total VRAM or native Siphon allocations.

The pinned real-model fixture is documented in
[`benchmarks/fp16/pythia-70m-source.json`](benchmarks/fp16/pythia-70m-source.json).
It contains original FP16 weights plus U8 attention-mask buffers; no conversion
from a quantized checkpoint is involved. Model weights are not committed and no
model is downloaded automatically by the loader or default tests.

### Verified result: RTX 4090 D

The full Pythia-70M checkpoint passed on an RTX 4090 D with Python 3.12.3,
PyTorch 2.9.1+cu130, and safetensors 0.8.0. Its 94 tensors contain 88 FP16 tensors
and six U8 attention-mask buffers, totaling **166,019,180 storage bytes**. The
downloaded file's SHA-256 was checked against the pinned Hugging Face source.

| Loader | Runs | Median synchronized loading | Exact-byte verification |
| --- | ---: | ---: | --- |
| Siphon | 3 | 279.35 ms | Every tensor, every run |
| safetensors reference | 3 | 275.39 ms | Every tensor, every run |

The Siphon runs reported 160.08 MiB peak PyTorch allocated memory and 164 MiB peak
reserved memory. These are **not total VRAM peaks**: native buffers and the CUDA
context are excluded. The complete server suite passed **86 tests with no skips**;
clean CPU CI also passed on Python 3.10 and 3.12.

These small-checkpoint measurements do not establish a Siphon speed advantage,
controlled cold-start performance, or inference throughput. Loading and
verification are separate measurements; Python process startup is excluded.

See the [raw runs](benchmarks/fp16/rtx4090d-pythia70m.json) and
[validation record](benchmarks/fp16/validation.json).

To acquire the exact acceptance fixture explicitly (the loader never downloads it):

```bash
set -eu
export MODEL_DIR=/absolute/path/to/pythia-70m
export REV=a39f36b100fe8a5377810d56c3f4789b9c53ac42
mkdir -p "$MODEL_DIR"
curl -fL --retry 3 -C - \
  "https://huggingface.co/EleutherAI/pythia-70m/resolve/$REV/model.safetensors" \
  -o "$MODEL_DIR/model.safetensors.partial"
printf '%s  %s\n' \
  ebfa4e2f18696ebd83716a0d39fe2c025f2ff8483f72a83ca59c475692fc9d15 \
  "$MODEL_DIR/model.safetensors.partial" | sha256sum -c -
# Rename only after successful checksum verification.
mv "$MODEL_DIR/model.safetensors.partial" "$MODEL_DIR/model.safetensors"
curl -fL --retry 3 \
  "https://huggingface.co/EleutherAI/pythia-70m/resolve/$REV/config.json" \
  -o "$MODEL_DIR/config.json"
python tools/run_fp16_acceptance.py "$MODEL_DIR" --runs 3 \
  --source-manifest benchmarks/fp16/pythia-70m-source.json \
  --output /absolute/path/to/acceptance.json
```

The download commands target a new fixture directory; existing model weights do
not need to be overwritten or converted. On macOS, use `shasum -a 256 -c -` in
place of `sha256sum -c -` for the checksum step. Use a configured CUDA/Siphon
environment for GPU acceptance.

## Tests

```bash
python -m pytest -q
```

The default suite uses architecture-independent synthetic tensors, with mixed
floating-point, integer-packed quantized, boolean, scalar, and empty storage. It
requires no model download, GPU, Siphon extension, or Rust compilation. Tests
cover exact contents and metadata, shard/index validation, lifetime, and CLI.

On a GPU host with Siphon installed:

```bash
VENTURI_TEST_CUDA=1 python -m pytest -q -m cuda -k real_siphon
```

This runs mixed-dtype round trips through Siphon's automatic, `MMAP`, and
`BUFFERED` paths and verifies every tensor against the CPU originals after the
loader closes. This is a storage correctness test, not a model benchmark.

Optional acceptance for any existing checkpoint with sufficient free VRAM:

```bash
VENTURI_TEST_CUDA=1 VENTURI_MODEL_PATH=/absolute/path/to/checkpoint \
  python -m pytest -q -m cuda -k full_checkpoint
```

No full-size checkpoint is required for this milestone.

## Measurement semantics

- `tensor_bytes`: physical storage bytes, not estimated unquantized model size.
- `storage_elements`: physical elements, not logical parameter count. One packed
  integer can contain multiple quantized model values.
- `total_seconds`: inspection plus synchronized loading; excludes Python process
  startup and imports performed before the API call.
- `load_seconds`: includes CUDA setup/synchronization when applicable.
- `cuda_allocated_bytes_delta`: PyTorch allocator change, not total or peak VRAM;
  native allocations invisible to PyTorch are excluded.
- CLI `cuda_allocator`: allocated/reserved bytes and process-lifetime allocator
  peaks sampled after loading. Native allocations and CUDA context are excluded;
  the library does not reset caller-owned peak counters.

## Layout

```text
src/venturi/checkpoint.py   Header/index inspection and opaque metadata
src/venturi/dtypes.py       Storage dtype descriptions, without conversion
src/venturi/loader.py       Siphon/reference loading and owned tensor container
src/venturi/verification.py Bounded, exact byte comparison with source tensors
src/venturi/cli.py          venturi inspect / venturi load / venturi verify
tools/                     Reproducible FP16 acceptance measurement
tests/                     Generic CPU tests and opt-in CUDA acceptance
```
