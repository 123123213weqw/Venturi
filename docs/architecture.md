# Repository architecture

## Scope

Venturi has explicit owned model execution and numerical-reference paths. A format reader,
checkpoint load, prepared operator and whole-model runtime are not interchangeable
interfaces. Knowing a format or having a CPU oracle does not mean that a model
is executable. External whole-model engines are no longer shipped or selected;
there is no implicit native/CPU/external fallback.

This document describes the **incremental October 28 refactor**, not a completed
rewrite, new GPU kernels, Metal support or a throughput improvement. The existing
uncommitted model/kernel work and frozen measurements remain in place.

## Native-only policy (October 28)

The external NInfer/Cinference and Strata runtime adapters, `qwen` command,
runner build/launch tools and engine-only tests were deliberately removed at the
user's request. Native `qwen-native` execution is unchanged and is not an alias
for the old command. The `backends` directory no longer exists in supported
source or the wheel. A new dependency guard rejects external engine imports.

Format support and infrastructure are independent: `venturi.ninfer` remains an
owned format reader, not an import of the external NInfer engine. CUDA/cuBLAS,
PyTorch, safetensors, tokenizers and the separate Siphon loading path remain.
The SM89 Ada Q4 plugin still contains attributed upstream kernel code and its
LICENSE/NOTICE/provenance. Removing an engine adapter does not make that code
original or replace it with an unqualified kernel.

Engine-independent GPU telemetry/idle admission and process metrics now live in
`tools/gpu_telemetry.py` and `tools/process_metrics.py`. Native qualification tools
no longer import helper functions from an external-engine benchmark. Their
measurement equations and non-destructive admission policy are preserved.

The removed sources and historical tests are retained as dated **non-package**
evidence in `operations/native-only-20261028-v1/source-before/`; benchmark and
operation snapshots were not rewritten. Offline reference comparisons remain
assessment tools, not a runtime dependency or inference fallback. No production
process, route, library, service or remote checkout was changed. Native DeepSeek
whole-model execution remains unfinished and must fail explicitly, not use Strata.

## Directory responsibilities

| Location | Responsibility | Must not become |
| --- | --- | --- |
| `src/venturi/` | Reusable implementation and compatibility entry points | A reader of experiment directories or production configuration |
| `src/venturi/commands/` | Explicit built-in command metadata and command execution | Automatic model/plugin/device discovery |
| `src/venturi/text/` | Local tokenizer, token IDs and greedy-loop structural contracts | A model family, serving scheduler or Torch-dependent package |
| Root storage modules | Safetensors/NInfer/GGUF metadata and explicit asset admission | Native model support or automatic weight conversion |
| `src/venturi/models/` | Architecture bindings, resident weights, sessions and composition | A CLI, model download service or import-time model registry |
| `src/venturi/ops/` | Prepared operators, ownership, native bindings and kernels | Model-family policy or request routing |
| External whole-model adapters | **Removed**; no supported runtime package or CLI | A hidden fallback for unsupported native models |
| `tests/` | Numerical, ownership, failure, API and dependency contracts | A benchmark or proof that skipped GPU tests passed |
| `tools/` | Explicit build/export/qualify/benchmark utilities | A dependency imported by the implementation |
| `scripts/` | Existing launchers | Hidden infrastructure mutation |
| `benchmarks/` | Frozen source/input/config/result evidence | Live source or an overwriteable scratch directory |
| `operations/` | Dated operational state, logs and resumable-job evidence | A replacement for reusable code or a default production route |
| `docs/` | Contracts, capabilities, limitations and architecture | Undated synthetic numbers presented as current model TPS |

Root storage/model-family modules are **not all physically relocated yet**. Keep
old import paths until each separate migration has compatibility tests. Moving
all files at once would invalidate tools, source fingerprints and users' imports.

## Dependencies and execution

```text
cli.py -- static command metadata --> selected command handler
                                         |
                    inspect / load / bind architecture explicitly
                                         |
                  model weights + sessions / request-owned state
                                         |
                   OperatorBackend -> prepared operator plans
                                         |
                     Torch reference OR explicit native CUDA

text/tokenizer + token validation     (no model/framework dependency)
generation.py -> structural model/session contract + Torch greedy loop

External whole-model runtime adapters: removed, not an execution branch
```

Storage metadata must not depend on execution. Operators must not import model
families or command policy. Implementations must not import the CLI. Shared text
must not import models, operators, loaders or tensor frameworks. CPU format oracles
such as `gguf_quantization` deliberately use Torch when executing; they are not
part of the framework-free metadata boundary.

The existing [operator contract](backend-interface.md) still owns capabilities,
explicit device/precision/layout checks, residency, plans and buffers. This
refactor does not replace that contract. The former external RPC engines have
been removed, not folded into it.

## Implemented in this pass

### 1. Explicit, lazy command dispatch

`cli.py` now dispatches using the immutable `commands.registry` table. Only the
selected handler is imported. The five existing checkpoint commands live in
`commands.checkpoint`; their options, JSON fields, raw-prompt semantics, loader
defaults and error/preflight ordering remain intact. The Llama, qwen-native and
Flash-Next commands keep their existing modules and command lines. The former
external `qwen` command is removed rather than aliased. No entry-point plugin
scan is performed.

The existing `venturi.cli.load_checkpoint`, `inspect_checkpoint` and
`verify_loaded_checkpoint` hook identities remain available lazily for callers
and tests. Injected hooks are passed only to the checkpoint handler.

Importing the dispatcher and requesting root/checkpoint-command help no longer
imports Torch. This is **not** a claim that invoking every family command is
framework-free. Existing Llama/native-Qwen handlers still import their selected
execution dependencies. Safetensors inspection still uses the existing `pt`
reader, which may itself import Torch when a file is opened; format validation
was not replaced with a permissive custom parser.

### 2. Lazy public model/operator namespaces

`venturi.models` and `venturi.ops` keep all 52 and 40 original exports in their
original order. Each name resolves to the same underlying implementation object,
then caches it normally. `__all__`, `dir`, unknown-attribute errors and the
old concrete-module imports remain available. Importing either namespace alone
does not import Torch or all other implementations. Requesting an actual
operator/model may still load its required implementation dependencies.

### 3. Shared text no longer belongs to GPT-NeoX

`text.tokenizer.LocalTokenizer` and `text.tokens.validate_token_ids` contain the
unchanged implementations. Existing imports through `generation` and
`models.gpt_neox` re-export those same objects. Llama and native-Qwen/Flash-Next
text users now depend on the shared layer rather than a GPT-NeoX tokenizer path.

The greedy loop is typed against `text.contracts.GenerationModel` and its session
requirements, instead of importing a concrete GPT-NeoX model for an annotation.
Its Torch arithmetic, optional session hooks, EOS rules, cache ownership, timing,
reset and token-selection behavior are unchanged. The protocol is not a generic
serving/MTP API or a promise that any arbitrary model can use that loop.

The safetensors dtype/byte-size table is available without importing Torch;
explicit `torch_dtype(...)` resolution still imports and checks installed Torch.

### 4. Enforced dependency boundaries

Run from any working directory:

```bash
python /absolute/path/to/Venturi/tools/check_architecture.py
python /absolute/path/to/Venturi/tools/check_architecture.py --json
```

The stdlib-only AST guard scans `src/` without importing the engine or probing a
device. CI runs it before the full tests. It records eager/deferred imports and
enforces the boundaries above. Type-only imports are excluded. It does **not**
prove absence of every dynamic import or legacy cycle; fresh-process tests also
block Torch/Transformers/Siphon imports to catch real transitive regressions.

Regression tests cover export identities, selection isolation, command forwarding,
compatibility hooks and numerical/CLI behavior. Evidence for this pass is under
`operations/repository-architecture-20261028-v1/`. Import dependency counts are
not startup benchmarks, GPU measurements or model TPS.

## Follow-on migrations (not completed by this pass)

The separate October 29 [native PD extension](pd-disaggregation.md) adds
framework-free request-state/transport contracts and explicitly owned prefill
and decode pools under `runtime/`. Architecture-specific state descriptors and
native copies stay in `models/qwen_pd.py` and the CUDA host boundary. This does
not merge PD into the existing single-target scheduler or make every backend,
model, TP/PP layout or cross-host transport supported. See its dated qualification
evidence and limits before treating it as a serving capability.

1. **Storage namespaces:** move safetensors/GGUF/NInfer readers into explicit
   format modules behind old-path re-exports; retain exact SHA, fingerprint,
   shape/layout, bounded-I/O and checkpoint admission contracts.
2. **Model families:** group GPT-NeoX, Llama, dense Qwen, qwen4_exp/Flash-Next and
   DeepSeek components separately. Extract common blocks only where math, dtype,
   state layout and buffer ownership really agree; do not merge by tensor shape.
3. **Request/runtime state:** separate resident weights, per-request KV/state,
   batch scheduling, TP/PP and speculative controllers. Preserve explicit
   concurrency ownership; avoid a giant universal mutable engine object.
4. **Native bindings:** split platform ABI/loading, reusable operators and
   SM/dtype/layout-specific kernels. Keep existing kernel paths until paired
   target correctness and unprofiled speed tests qualify a replacement.
5. **Developer surface:** group tools and numerical/reference/integration/GPU
   tests with compatibility launchers. Split the long historical README only
   with a link/anchor migration, not by deleting the evidence.

Each migration needs its own pre-change source snapshot and unchanged baseline,
API/CLI compatibility checks and applicable numerical/failure tests. GPU-affecting
changes additionally require target correctness, same-condition repeated speed
comparisons and the diagnostics required by `AGENTS.md`, with profiler runs kept
separate. Architecture cleanup alone is not permission to change production,
resume an expired job/automation, repair storage or auto-push GitHub.
