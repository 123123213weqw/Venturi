# Venturi

**A specialized inference engine that owns its weight storage, its CUDA kernels and
its generation loop.**

[![Tests](https://github.com/123123213weqw/Venturi/actions/workflows/tests.yml/badge.svg)](https://github.com/123123213weqw/Venturi/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776ab.svg)](#requirements)
[![PyTorch](https://img.shields.io/badge/pytorch-2.5%2B-ee4c2c.svg)](#requirements)
[![CUDA](https://img.shields.io/badge/cuda-12.9-76b900.svg)](#requirements)
[![Status](https://img.shields.io/badge/status-research%20preview-orange.svg)](#project-status)

> **Research preview.** Venturi is not production software and does not yet beat a
> mature engine on any end-to-end workload. It is deliberately narrow: two complete
> text-generation adapters (dense GPT-NeoX and classic Llama), native kernels
> qualified operator by operator, and one much larger target — a fully owned Qwen
> Q8 runtime — that is still unfinished. Every gap is listed in
> [Project status](#project-status) instead of being left for you to discover.

Venturi does not select, embed or fall back to an external whole-model engine.
Unsupported architectures, checkpoint layouts, dtypes and hardware profiles fail
closed. What it does share with other projects is stated precisely in
[Acknowledgements](#acknowledgements).

## Contents

- [Project status](#project-status)
- [What Venturi is / is not](#what-venturi-is--is-not)
- [Supported models](#supported-models)
- [Requirements](#requirements)
- [Install](#install)
- [Quickstart](#quickstart)
- [Command line](#command-line)
- [Performance status](#performance-status)
- [Architecture](#architecture)
- [Testing](#testing)
- [Measurement semantics](#measurement-semantics)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Historical records](#historical-records)

## Project status

| Area | State | Notes |
| --- | --- | --- |
| Safetensors loading, sharding, verification | **Works** | Bounded reads, pinned staging, per-tensor byte verification, no model Python executed |
| Dense GPT-NeoX adapter (Pythia) | **Works** | Complete generation path on PyTorch operators; greedy only; validated against FP16 references |
| Classic Llama adapter | **Works** | Complete native path with per-session KV and chunked prefill/decode; eager, single stream, no scheduler |
| Owned native operators (RMSNorm, SwiGLU, residual, RoPE, causal GQA, Q4/Q5/Q6/Q8 row codecs) | **Works** | Qualified per operator against independent FP64 oracles |
| Native Qwen Q8 runtime | **Experimental** | Runs greedy generation on qualified exports; still missing parts of a complete independent Qwen implementation |
| Full hybrid Qwen model (GDN + full attention) | **Incomplete** | Layer compositions exist; model session, trained-model quality and matched serving wins do not |
| Flash-Next (Qwen4-Exp) | **Experimental** | Intake and floating reference generation only; not native serving |
| DeepSeek V4.1 | **Admission only** | Tensor admission and CPU oracles; no Venturi-native executor |
| Continuous batching, PD pools, TP/PP, MTP | **Experimental** | Explicit, opt-in, hardware-gated; not a general serving story |
| Dense prefill performance | **Slower than PyTorch** | Correctness-first sublayers; see [Performance status](#performance-status) |

Start with [Quickstart](#quickstart) if you want to run something now, and with
[`docs/architecture.md`](docs/architecture.md) if you want to know where code belongs.

## What Venturi is / is not

**What it is**

- A loader for local safetensors checkpoints (single file, or sharded directory)
  that preserves physical names, shapes, storage dtypes and packed integers, and
  verifies every loaded byte.
- A set of explicitly prepared native operators with caller-owned workspaces,
  fixed-shape plans, graph capture and no hidden global caches.
- Two complete, readable adapters — dense GPT-NeoX and classic Llama — whose
  forward passes and generation loops are Venturi code, not `transformers` calls.
- An engineering log where measurements, counterexamples and rejections are
  retained next to the code that produced them.

**What it is not**

- Not an OpenAI-compatible API server, not a chat product, not a drop-in
  replacement for vLLM, llama.cpp or TensorRT-LLM.
- Not "supports N architectures": loading a checkpoint is not executing it.
- Not a general quantization toolkit. The registry holds the four row formats this
  repository actually qualifies: `q4_g64_fp16`, `q5_g64_fp16`, `q6_g64_fp16`,
  `q8_g32_fp16`.
- Not a performance claim. Where Venturi is slower, the numbers below say so.

## Supported models

| Family | Reference checkpoint | Path | State |
| --- | --- | --- | --- |
| Dense GPT-NeoX | EleutherAI **Pythia-70M**, FP16 | `venturi generate`, `venturi benchmark` | Complete adapter, greedy, no sampling or chat template |
| Qwen3.5 hybrid (GDN + full attention) | **Qwen3.5-27B** re-exported as explicit Q8 safetensors | `venturi qwen-native` + two native libraries | Greedy only; ~26.6 GiB of weights before cache/workspace, so it does not fit one 24 GiB card |
| Classic Llama | Any local Llama-format checkpoint with `tokenizer.json` | `venturi llama` | Native operator path, eager single stream; not a serving path |
| Qwen4-Exp (Flash-Next) | — | `venturi flash-next` | Experimental inspection and floating reference generation |
| DeepSeek V4.1 | — | asset admission + CPU oracles | No native executor; no TPS is claimed |

Anything else is rejected rather than routed somewhere else. Model downloads are
never performed by the library; you supply checkpoint paths.

## Requirements

- **Python** 3.10 or newer.
- **PyTorch** 2.5+ (CUDA build) and **safetensors** 0.5+, installed by the package.
- **Linux + NVIDIA GPU** for any runtime use. There is no Metal, ROCm or CPU
  serving path; CPU is an explicit reference backend for tests and small checks.
- **CUDA 12.9** toolchain (`nvcc`) to build the native operator and Qwen libraries.
  Prebuilt binaries are not published.
- Primary validation target: **RTX 4090D (SM89)**. **V100 (SM70)** support is an
  explicit, opt-in profile that remains unqualified in general.

## Install

```bash
git clone https://github.com/123123213weqw/Venturi.git
cd Venturi
python -m pip install -e '.[test]'
```

Optional extras:

| Extra | Adds | Needed for |
| --- | --- | --- |
| `.[test]` | pytest, numpy | Running the test suite |
| `.[generate]` | tokenizers | Text generation and prompt tokenization |
| `.[reference]` | transformers | Model/oracle comparison tests |

Checkpoint loading and inspection need no extra component: the staged CUDA loader
and the safetensors reference path ship with the package.

## Quickstart

Every path below is a placeholder. Venturi never downloads checkpoints, and no
model is bundled.

### 1. Inspect a checkpoint without touching GPU weights

```bash
venturi inspect /absolute/path/to/checkpoint --json
venturi inspect /absolute/path/to/weights.safetensors --json
```

Directory discovery order is `model.safetensors.index.json` (or a unique
`*.safetensors.index.json`), then `model.safetensors`, then all `*.safetensors`
files. An index is authoritative: missing entries, duplicate tensor names and
ambiguous multiple indexes are rejected.

### 2. Load and verify

```bash
venturi load /absolute/path/to/checkpoint --device cuda:0 --json
venturi verify /absolute/path/to/checkpoint --device cuda:0 --json
```

The staged loader prefetches shard chunks into a ring of pinned host slots with
reader threads while non-blocking device copies run on a dedicated stream. Every
returned tensor owns its storage; nothing aliases a rotating buffer.

### 3. Generate text (dense GPT-NeoX)

```bash
python -m pip install -e '.[generate]'
venturi generate /absolute/path/to/pythia-70m \
  --prompt "Once upon a time" --max-new-tokens 32 --json
```

The checkpoint directory needs its original `config.json`, safetensors weights and
`tokenizer.json`. Generation is greedy, applies no chat template, and stops at EOS
or the requested token limit. Pythia-70M is a base completion model: repetition or
poor text is model behavior, not an inference bug.

The same path from Python:

```python
from venturi import load_checkpoint
from venturi.models import GPTNeoXModel
from venturi.text import LocalTokenizer
from venturi.engine.generation import generate_text

checkpoint = load_checkpoint("/absolute/path/to/pythia-70m", device="cuda:0")
model = GPTNeoXModel(checkpoint)
tokenizer = LocalTokenizer(
    "/absolute/path/to/pythia-70m/tokenizer.json", vocab_size=model.config.vocab_size
)
print(generate_text(model, tokenizer, "Once upon a time", max_new_tokens=32).text)
```

The independent Llama adapter runs the same generation contract over Venturi's
native operators instead of PyTorch operators, and needs an explicit operator
library for the CUDA backend:

```bash
venturi llama generate /absolute/path/to/llama-checkpoint \
  --ops-library /absolute/path/to/libventuri_ops.so \
  --prompt "Once upon a time" --max-new-tokens 32
venturi llama inspect /absolute/path/to/llama-checkpoint   # bindings only, no upload
```

### 4. Run the native Qwen Q8 runtime

```bash
venturi qwen-native inspect /absolute/path/to/qwen-q8.safetensors

venturi qwen-native generate /absolute/path/to/qwen-q8.safetensors \
  --ops-library /absolute/path/to/libventuri_ops.so \
  --qwen-library /absolute/path/to/libventuri_qwen.so \
  --weight-mode resident --device cuda:0 --dtype fp16 \
  --token-ids 1,2,3 --max-tokens 256 --max-new-tokens 32
```

The libraries must be built and qualified for the target architecture first (next
section). The checkpoint is standard safetensors with explicit Q8 planes and must
come from a supported export; renaming another artifact is not conversion.

### 5. Build the native libraries

Two concrete libraries, one dependency direction: `libventuri_qwen.so` links the
operator ABI exported by `libventuri_ops.so`. There is no plugin registry, no
`RTLD_GLOBAL` and no sibling-file guessing.

```bash
export PYTHONPATH=src
OPS=$(python -m venturi.ops.build --component ops --architecture sm89 \
  --nvcc /path/to/cuda-12.9/bin/nvcc --output-dir build/ops)
QWEN=$(python -m venturi.models.qwen.build --ops-library "$OPS" \
  --nvcc /path/to/cuda-12.9/bin/nvcc --output-dir build/qwen)
```

The Qwen output directory contains the operator binary it was linked against.
Python verifies the ABI, the architecture and the actually linked operator
function address, not just matching filenames. Full contract:
[`docs/native-library-split.md`](docs/native-library-split.md).

## Command line

`venturi` dispatches explicitly; nothing is imported until a command is selected.

| Command | Purpose |
| --- | --- |
| `venturi inspect` | Inspect safetensors storage without loading weights |
| `venturi load` | Load tensors using an explicitly selected loader |
| `venturi verify` | Verify loaded tensors against stored bytes |
| `venturi generate` | Raw continuation with the dense GPT-NeoX adapter |
| `venturi benchmark` | Benchmark the explicit dense GPT-NeoX path |
| `venturi qwen-native` | Owned Q8 C++/CUDA runtime: inspect or explicit greedy generation |
| `venturi llama` | Independent classic Llama checkpoint bindings and generation |
| `venturi flash-next` | Experimental Qwen4-Exp inspection or floating reference generation |

`python -m venturi` is an equivalent entry point. Family commands take a
subcommand, e.g. `venturi qwen-native inspect <path>`.

## Performance status

Venturi is **not** faster than existing engines today, and this section exists to
say so with numbers rather than adjectives.

Dense GPT-NeoX on RTX 4090D, FP16 Pythia-70M, two warmup and ten measured
requests, EOS ignored for fixed output length:

| Prompt tokens | Output tokens | Median TTFT | Aggregate decode |
| ---: | ---: | ---: | ---: |
| 4 | 32 | 5.361 ms | 212.94 tokens/s |
| 5 | 128 | 4.218 ms | 230.22 tokens/s |

Native sublayer microbenchmarks on the same device (CUDA-graph wall time per
sublayer call) show where the work still is:

| Synthetic shape | Stored | T | PyTorch reference | Venturi native |
| --- | --- | ---: | ---: | ---: |
| Qwen-style 5120/17408 | BF16 | 128 | 846.9 µs | 3838.9 µs |
| Qwen-style 5120/17408 | Q4 | 128 | 2114.9 µs | 858.4 µs |

Strict dense prefill is currently ~4.5x slower than PyTorch; packed Q4 is faster
than an explicitly materialized FP32 reference, which is not a comparison against
another quantized engine. The accepted next milestone is a same-checkpoint
comparison against `llama.cpp` on two V100s — it is a target, not a result.

Rules these numbers follow:

- Profiler runs never share a process with unprofiled throughput measurements.
- Theoretical occupancy is never reported as achieved occupancy, and a
  memory-controller busy percentage is never called bandwidth utilization.
- If hardware counters are unavailable (`RmProfilingAdminOnly=1`), the measurement
  is recorded as unavailable instead of being estimated.
- Rejected optimizations stay in the record: a WMMA Q4 experiment failed its
  performance gate and is not in the active source.

Evidence, per-phase raw data and preserved failures live under
[`benchmarks/`](benchmarks/); the narrative history is in
[`docs/milestones.md`](docs/milestones.md).

## Architecture

Venturi is built so that a format reader, a checkpoint load, a prepared operator
and a whole-model runtime are never interchangeable. The dependency direction is:

```text
container/      physical tensor storage, bounded reads, checkpoint metadata   (no Torch, no model)
quantization/   encoded geometry and the single codec/qualification registry   (pure)
models/<family>/ architecture binding and mathematical semantics
    qwen/csrc/    model structure, sessions, KV/GDN ownership, TP/PP/MTP
ops/            numerical kernels and opaque execution contexts
    csrc/         operator source + ops_api.h (concrete C ABI)
runtime/        active request state and scheduling
commands/       CLI parsing; not an injection surface for old module globals
```

Two native libraries sit on this boundary: `ops/` builds `libventuri_ops.so`
(ABI 17) and `models/qwen/csrc/` builds `libventuri_qwen.so` (runtime ABI 1) on
top of it. The operator source never includes a model runtime, and the Qwen source
includes only the operator ABI and stateless helpers.

### Repository layout

| Path | Contents |
| --- | --- |
| `src/venturi/container/` | Safetensors/GGUF physical records, staged transfer, metadata |
| `src/venturi/quantization/` | Row layouts, codecs, verification |
| `src/venturi/models/` | Per-family adapters: `qwen/`, `neox/`, `llama.py`, `flash_next/`, `deepseek/` |
| `src/venturi/ops/` | Prepared operator backends and `csrc/` kernels |
| `src/venturi/runtime/` | Session/KV/scheduling state |
| `src/venturi/engine/` | Shared generation implementation |
| `src/venturi/text/` | Tokenizer, token IDs, text contracts |
| `tests/` | Storage, operator, model/oracle and opt-in CUDA tests |
| `tools/` | Qualification and benchmark entry points (not a public API) |
| `docs/` | Contracts and design notes |
| `benchmarks/` | Frozen evidence snapshots: raw logs, JSON, source capsules |
| `scripts/` | Deployment-specific launchers (defaults are not portable) |

`benchmarks/` is the largest directory by far. Its snapshots are immutable
evidence for the revisions they were taken on; they are not runtime dependencies
and never describe current support.

## Testing

```bash
python -m pytest -q
python tools/check_architecture.py   # static layering guard over src/
```

The default suite needs no GPU, no checkpoint download and no compilation. It
covers storage contents and metadata, shard/index validation, tensor lifetime and
CLI behaviour with synthetic tensors.

CUDA tests are opt-in and run on a GPU host:

```bash
VENTURI_TEST_CUDA=1 python -m pytest -q -m cuda -k real_staged

VENTURI_TEST_CUDA=1 VENTURI_MODEL_PATH=/absolute/path/to/checkpoint \
  python -m pytest -q -m cuda -k full_checkpoint
```

Real generation tests additionally require `VENTURI_GENERATION_MODEL` and a
matching local `tokenizer.json`. Native kernel tests require the operator library
to be built first; they are skipped, never silently replaced, when it is absent.

## Measurement semantics

These definitions are part of the contract, not footnotes:

- `tensor_bytes` is physical storage bytes, not an estimated unquantized size.
- `storage_elements` counts physical elements; one packed integer can hold several
  quantized values.
- `load_seconds` includes CUDA setup and synchronization when applicable.
- `cuda_allocated_bytes_delta` is a change in this process's PyTorch allocator, not
  total or peak VRAM, and excludes native allocations.
- Decode throughput is `sum(output_tokens - 1) / sum(decode_seconds)`; prefill
  produces the first token and must not be counted as a decode step. With no decode
  tokens the value is `null`, never an invented number.
- Aggregate rates are ratios of sums, not averages of per-request rates.
- Identical greedy outputs across repetitions show request repeatability, not
  reference-model correctness.

## Documentation

| Document | Contents |
| --- | --- |
| [`architecture.md`](docs/architecture.md) | Module boundaries, current cleanup direction, remaining work |
| [`backend-interface.md`](docs/backend-interface.md) | Prepared operator contract: capabilities, ownership, native/reference boundaries |
| [`native-library-split.md`](docs/native-library-split.md) | The two-library build and load contract |
| [`qwen-safetensors.md`](docs/qwen-safetensors.md) | Direct Qwen safetensors loading, plane schemas, intentional API breaks |
| [`continuous-batching.md`](docs/continuous-batching.md) | Opt-in native continuous scheduling |
| [`pd-disaggregation.md`](docs/pd-disaggregation.md) | Explicit prefill/decode pools and request-state handoff |
| [`vram-first-kv.md`](docs/vram-first-kv.md) | Experimental VRAM-first native KV |
| [`adaptive-mtp.md`](docs/adaptive-mtp.md) | Measured-cost fresh-cohort routing for MTP |
| [`native-mtp-plan.md`](docs/native-mtp-plan.md) | Native MTP implementation and acceptance |
| [`native-rms-quant.md`](docs/native-rms-quant.md) | Post-attention RMSNorm + residual-A8 packing (SM70) |
| [`milestones.md`](docs/milestones.md) | Historical milestone and qualification log |
| [`NAMING.md`](NAMING.md) | Naming rules: architecture names live in adapters, mechanisms stay neutral |

## Contributing

Contributions are welcome where they fit the existing contracts. Before opening a
pull request:

1. **Keep the layering.** Run `python tools/check_architecture.py`. Mechanism code
   (`ops/`, `container/`, `quantization/`) must not grow model names; architecture
   binding belongs in `models/<family>/`. See [`NAMING.md`](NAMING.md).
2. **Qualify correctness before performance.** An optimization needs the relevant
   correctness checks *and* an unprofiled baseline/change comparison on the target
   GPU, with checkpoint, precision, lengths, context, concurrency, warmup and
   timing scope stated explicitly.
3. **Report CPU time, latency and throughput separately.** Do not sum concurrent
   request durations to compute aggregate throughput.
4. **Record rejections.** If an idea fails its gate, say so and keep the evidence;
   a negative result is a contribution here.
5. **Do not claim untested paths.** If a measurement was unsupported or
   permission-denied, write that down instead of substituting an estimate.

[`AGENTS.md`](AGENTS.md) is the full working agreement used for this repository,
including the GPU diagnostics expected for performance work.

## License

**No license file is present in this repository yet.** Until one is added, the
contents are "all rights reserved" by default and are *not* open source, no matter
how open the code looks. Adding an OSI-approved license is a maintainer decision
that has not been made. If you need to use, vendor or redistribute this code,
contact the maintainer first.

## Acknowledgements

Venturi exists because other people's work is good enough to depend on and to
argue with:

- **PyTorch**, **safetensors** and **tokenizers** for storage, tensor and text
  primitives that Venturi does not reimplement.
- **NVIDIA CUDA** and **cuBLAS**. Dense projections call cuBLAS through a Venturi
  C ABI bridge; they are not an original GEMM algorithm.
- The **Ada Q4** operator lineage: retained quantized kernels carry explicit
  upstream attribution. The current owned SIMT Q4 baseline is deliberately slower
  than the historical vendor MMA path and is not a performance promotion.
- **llama.cpp**, used as a pinned comparison harness for the same-checkpoint
  experiment. Its numbers are never mixed into Venturi's own reports.

Earlier prototypes were removed rather than preserved as compatibility shims: the
`.ninfer` reader, the external whole-model engine adapters and their build tools.
They are not silently redirected to a different implementation.

## Historical records

This README used to be a 144 KB chronological log. That log is preserved verbatim
at [`docs/milestones.md`](docs/milestones.md): every ABI increment, benchmark
table, GPU diagnostic and failed hypothesis, in its original order, with the
revision it was measured on. Read it when you need to know *why* a design looks
the way it does — not as a description of what runs today.
