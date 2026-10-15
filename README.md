# Venturi

A specialized, high-performance inference engine.

**Siphon loads the weights; Venturi runs the model.**

## First milestone: a model-agnostic loader

Venturi currently **inspects and loads checkpoint tensors**, not model execution.
The loading core has no architecture-specific configuration classes, required
weight names, layer counts, or model adapters. Architecture interpretation and
execution kernels belong in separate, future components.

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
PyTorch/safetensors stack. Backend/device support still applies. Native sub-byte
floating-point formats, original `.pth` files, GGUF, checkpoint conversion,
automatic downloads, and inference are not implemented yet.

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
