# Two concrete native libraries

## Responsibility and dependency direction

```text
Python CudaBackend -> libventuri_ops.so
                         ^
                         | concrete operator C ABI
Python QwenRuntime -> libventuri_qwen.so
        |
NativeQwenModel / sessions / batch / MTP / pipeline
```

- `ops/csrc`: numerical operator kernels and their opaque execution contexts.
  No Qwen model/session, checkpoint mapping, request cache or model scheduler.
- `models/qwen/csrc`: Qwen structure, sessions, KV/GDN ownership, mappings,
  staged uploads, batch/TP/PP/MTP orchestration and model-specific state kernels.
- `ops_api.h`: concrete function signatures and buffer layouts, not a model
  protocol, callback table or plugin registry.
- `cuda_common.cuh`: stateless CUDA scalar conversion, error checking and block
  reductions shared as source. No mutable state, kernels or model definitions.

The operator source must never include a model runtime. Qwen includes only the
operator ABI and stateless helpers, never an operator implementation header.
Its projection/quantization calls go through concrete C functions. cuBLAS
contexts are opaque: each dense call receives T/N/K explicitly rather than
mutating an internal `Linear` layout. Floating arithmetic is unchanged.

The existing explicit Ada Q4 operator library remains separate and unchanged;
it is not part of the Qwen two-library dependency chain. No plugin discovery,
auto-registration, generic model interface or implicit fallback was introduced.

## Build explicitly on Star

Use the remote Linux CUDA compiler; no local CUDA/Rust compilation is required.
From the synchronized checkout, using an appropriate Python environment:

```bash
export PYTHONPATH=src
OPS=$(python -m venturi.ops.build --component ops --architecture sm89 \
  --nvcc /data/wangyue/cuda-12.9/bin/nvcc --output-dir build/ops)
QWEN=$(python -m venturi.models.qwen.build --ops-library "$OPS" \
  --nvcc /data/wangyue/cuda-12.9/bin/nvcc --output-dir build/qwen)
```

Operator builds fingerprint only their own transitive source includes. Editing
Qwen does not invalidate the operator cache. The Qwen build fingerprints its own
includes and pins the concrete operator binary. Operator upgrades require an
explicit newly linked pair and validation, not rewriting the model implementation.
This deliberately favors deterministic dependencies over automatic hot-swapping.

The ops binary has a build-specific SONAME. The Qwen output directory contains
that dependency beside the runtime, with `$ORIGIN` lookup. Copy both when moving
an artifact. Distinct operator builds can coexist without ambiguous symbol
resolution. Python verifies ABI, architecture and the actual linked operator
function address, not just matching filenames. No `RTLD_GLOBAL`, global search
path editing or silent sibling-file guessing is used.

## Explicit Python and CLI loading

```python
import torch
from venturi.ops import CudaBackend
from venturi.models import QwenRuntime, QwenWeights, NativeQwenModel

operators = CudaBackend('/absolute/libventuri_ops.so')
runtime = QwenRuntime('/absolute/libventuri_qwen.so', operators)
weights = QwenWeights.open('/absolute/text-q8.safetensors')
model = NativeQwenModel.from_q8(
    weights, runtime, weight_mode='resident', dtype=torch.float16,
    device='cuda:0', max_tokens=128,
)
```

```bash
python -m venturi qwen-native generate /absolute/text-q8.safetensors \
  --ops-library "$OPS" --qwen-library "$QWEN" \
  --weight-mode resident --dtype fp16 --token-ids 1,2 --max-new-tokens 8
```

Metadata-only `inspect` needs neither DSO. Numerical reference adapters continue
to receive an operator backend, not a Qwen runtime. Tests use explicit
`VENTURI_OPS_LIBRARY` and `VENTURI_QWEN_LIBRARY`; actual execution does not build
or discover libraries. Two-arm operator benchmarks must provide matching runtime
paths for both arms rather than binding a candidate runtime to a control library.

This is an intentional interface break: operator ABI **17**, Qwen runtime ABI
**1**. The old monolithic ABI-16 library, `--blocks-library`,
`VENTURI_BLOCKS_LIBRARY` and `build_blocks_backend` are not compatibility aliases.
Runtime errors and operator errors have independent thread-local channels;
model failures must be checked through QwenRuntime, not CudaBackend.

## Verification boundary

Tests cover source dependency direction, Qwen-only source changes, ELF exports
and DT_NEEDED, wrong-pair rejection, relocation, independent error channels and
fresh-process operator execution without loading/importing Qwen. GPU regression
covers the unchanged scalar/model oracles, physical safetensors plane order,
low-bit on-demand/shadow modes, state/cursor failures and single-device MTP.

Validation targets Star physical GPU2 only. TP/PP and dedicated SM70 profiles
remain conditional tests, not qualified by this single-card run. This split is
not a throughput optimization or a trained-27B quality/production promotion.

### Shared-node framework startup budget

During this validation the installed PyTorch 2.13 environment took about 41 s
for one measured import; another bounded startup probe exceeded 90 s. These
are startup/environment observations, not inference measurements or a numerical
regression. Framework-import isolation tests retain their default 30 s watchdog;
Star uses explicit `VENTURI_FRAMEWORK_IMPORT_TIMEOUT=120` (bounded to 30..300 s).
Their dependency/no-initialization assertions remain unchanged. The benchmark
server's argument preflight remains framework-free: runtime imports occur only
after arguments are validated. No global Torch installation/settings were changed.

## Validation record (Star GPU2, 2026-11-13)

- Operator build: `39c2f359c871c106`, ABI 17; Qwen build:
  `46b66a7365f8ccd7`, ABI 1. Both compiled remotely with CUDA 12.9 for SM89.
- Full GPU2 suite: 2212 passed, 9 failed, 2505 skipped. Five failures were
  uninitialized preflight test mocks accidentally entering the real DSO loader;
  one new independent-operator fixture had FP32 weights with an explicit FP16
  activation contract. Test fixtures were corrected, not production validation
  relaxed. Three failures were subprocess startup watchdogs; all dependency,
  dtype, value and ownership assertions stayed intact.
- Targeted rerun: **all 9 passed**. Combined unique-test outcome: **2221 passed,
  2505 skipped, no unresolved failures**. This was not a single all-green run.
- Local non-CUDA suite: 1851 passed, 2564 skipped, 311 CUDA-marker deselections.
- Baseline/split comparison used the identical synthetic Q8 safetensors file on
  the same physical GPU2. FP16/BF16 x resident/streaming produced byte-identical
  logits, layer traces, KV/GDN snapshots and greedy tokens. Retained output JSON
  SHA256: `5ac8f7502c44b6d710afb56fc8d880ee12e514923ad6b22428c152cf6ecc6c25`.
- A real Qwen-only source-edit build probe hit the unchanged operator cache;
  both operator binary and compiler-log bytes/mtime stayed unchanged, while a
  distinct Qwen library was compiled. The probe used the first split pair
  (`a25814f316f0983f` ops); final artifacts above incorporate comment cleanup.
- Static dependency scan: zero violations. A clean wheel has 168 entries and
  exactly matches current packaged source. Stale build staging was removed to
  prevent the previously moved runtime headers reappearing under ops.

The isolated Star workdir is `/data/wangyue/codex-build/Venturi-split-20261113/`.
It retains `split-libraries.env` (library paths only), the GPU2 runners, raw JUnit
reports, startup diagnostics, same-weight parity JSON and the build-boundary
probe. Original checkouts, production services/routes/models/keys, other users'
processes and GitHub main were not modified. No trained 27B quality/TPS, two-GPU
TP/PP or dedicated SM70 profile qualification is claimed by this run.
