> Native execution now uses [two explicit libraries](native-library-split.md).
> The acceptance record below is the storage-cleanup milestone before that split.

# Direct Qwen safetensors loading (2026-11-13)

## Decision and scope

The active Qwen path no longer reads `.ninfer`, wraps its directory in JSON, or
imports third-party NInfer CUDA sources. This is an intentional API/file-format
break, not a compatibility layer. Re-export from the original HF checkpoint;
renaming an old file is not conversion. Frozen benchmark/operation records refer
to their original revisions and do not qualify this implementation.

```text
verified original HF files
  -> bounded CPU Q8 export -> one standard .safetensors file
  -> strict physical tensor/header admission
  -> QwenSourcePlan + typed QwenTensor roles
  -> explicit code/high/scale offsets + small mathematical parameters
  -> owned native runtime / explicitly selected reference adapter
```

Responsibilities:
- `container/safetensors.py`: physical tensor schema, identity, bounded reads,
  streaming writes and no-clobber publication. No model or Torch imports.
- `quantization/layout.py`: pure geometry. `row_payload_bytes` counts actual
  tensor bytes; `row_geometry` describes aligned GPU-local scratch, not files.
- `quantization/codecs.py`: codec properties, reference decode and execution gates.
- `models/qwen/qwen_source.py`: original HF names, Qwen geometry and semantics.
- `models/qwen/qwen_weights.py`: exact tensor-set admission and typed roles.
- `models/qwen_load/`: native validation, budgets and descriptor construction.
- `models/qwen/csrc/weight_source.cuh`: read-only physical spans; no Qwen tensor names.
- `models/qwen/csrc/staged_uploader.cuh`: one 32 MiB pinned buffer, synchronized chunk
  uploads. Simpler ownership and accurate memory accounting take priority over
  parallel cold-load throughput; no new performance result is claimed.

## Physical representation

Small floating tensors retain original HF names and mathematical parameters.
Convolution is explicitly transposed to chronological `[kernel, channels]`.
Norms remain raw zero-centered HF weights; `A_log` remains logarithmic until the
model frontend performs its explicit transform. Historical GGUF bridge flags
that represented different semantics are rejected, not guessed.

For a matrix with original HF name `P`, shape `[N,K]`:

| Tensor | Dtype | Shape |
| --- | --- | --- |
| `P.codes` | I8 for Q8; U8 otherwise | `[N, groups*32]` |
| `P.scales` | F16 | `[N, groups]` |
| `P.high` (Q5/Q6 only) | U8 | `[N, groups*8*(bits-4)]` |

`groups = align(K,128)/group_size`; group size is 32 for Q8 and 64 for Q4/5/6.
Q8 forbids code -128. Scales must be finite and nonnegative. Dense admitted
matrices use their original name and explicit floating dtype. File plane order
and adjacency are irrelevant: each native plane has its own validated offset.
GPU-local alignment is a separate allocation detail.

Standard string metadata contains only:
- `venturi.config`: original HF JSON config, including the wrapper model type;
- `venturi.component`: `text` or `mtp`;
- `venturi.matrix_format`: the declared main-matrix codec;
- `venturi.source`: acquisition/hash evidence.

There are no serialized `objects`, `bindings`, `uses`, multipart directories or
UUID containers. Roles are derived from Qwen architecture, not stored as another
universal schema. Official `safetensors.safe_open` can inspect every physical
tensor. Header validation is not a payload hash or trained-model quality check.

## Export, verify, inspect

Run on a host with sufficient RAM/disk; these commands do not allocate a GPU.
The verified source manifest must cover the original config/index/weight files.
Existing output/report files are never overwritten.

```bash
PYTHONPATH=src python tools/export_qwen_q8.py /absolute/hf-source /absolute/text-q8.safetensors \
  --rows-per-chunk 256 --report /absolute/export-report.json
PYTHONPATH=src python tools/verify_qwen_q8_export.py /absolute/hf-source /absolute/text-q8.safetensors \
  --export-report /absolute/export-report.json --output /absolute/verification.json
PYTHONPATH=src python -m venturi qwen-native inspect /absolute/text-q8.safetensors
```

```python
from venturi.models.qwen.qwen_weights import QwenWeights
weights = QwenWeights.open('/absolute/text-q8.safetensors')
matrix = weights.descriptor('text/token_embedding')
# Explicit CPU inspection/reference decode, not an implicit native fallback:
q8 = weights.read_linear('text/token_embedding')
```

Native execution still requires an explicit rebuilt blocks library, supported
architecture/precision/profile, sufficient memory, and an explicit resident or
streaming mode. Storage support never grants execution support. Dense FP16
admission does not silently enable a native dense model. Q4/5/6 native execution
retains its restricted single-device resident FP16 profile. The exporter in this
change produces Q8, not a new low-bit conversion pipeline.

MTP is a separate safetensors checkpoint opened by `QwenMTPWeights.open(path)`.
Target/draft config and nonempty source evidence must match. The CLI option is
`--mtp-checkpoint`; `--mtp-artifact` and `from_ninfer` are not aliases.

## Deliberately not retained

- NInfer objects-in-safetensors metadata: changes the envelope, not the coupling.
- Automatic format/model/engine fallback: hides unsupported configurations.
- Old QwenQ8Weights/from_ninfer APIs and external-engine tuning harnesses.
- Vendor Q4 kernels and the rejected owned WMMA prototype. The active independent
  SIMT plugin is an ABI-2 correctness baseline; previously measured multi-token
  latency regressed 6.2–16.3x against the old vendor MMA. It is not a production
  speed promotion.

## Validation boundary

Tests cover official-reader interoperability, malformed headers, bounded reads,
source replacement, atomic publication, all signed Q4/Q5/Q6 codes, original-HF
Q8 export equality, independent export verification, tied heads and MTP lineage.
GPU regression tests include official-writer files with scales physically before
codes; they must run on Star with a compatible driver/library pair. A successful
compile or a skipped GPU test is not GPU execution evidence. This migration does
not establish trained 27B quality, full-model TPS or production readiness.

### Regression findings fixed during GPU2 validation

The first GPU2 pass exposed an unwired explicit `dequant_resident` option and a
stale MTP oracle. The option now reaches load validation/descriptors without
changing its default. Its Q4/5/6 restriction remains explicit; Q8 cannot silently
expand into dense residency. The MTP FP64 equations now apply the already
specified activation-rounding boundaries for Q/K normalization, separate rotary
products and sigmoid gating, consistent with the primitive oracle/kernel
contract. The test tolerances were not increased and production arithmetic was
not changed to match the test. TP/PP tests explicitly require two visible devices;
they are not qualified by a single physical GPU2 run.

Source identity guards compare file identity/size/timestamps; they are not a
continuous cryptographic integrity monitor. Inputs must remain immutable while
loaded. A write inside a coarse filesystem timestamp tick can leave stat values
unchanged. Trusted full-file hashes are checked explicitly at export/verification;
those are distinct guarantees. Tests of metadata-change detection set a changed
timestamp explicitly rather than assuming every instantaneous write is observable.

## Acceptance record (2026-11-13)

- Pinned upstream source: `b19d9e29abf5de0121ec4b7411390d14812f31bb`.
- Star, physical GPU **2**, RTX 4090 D; only its UUID was exposed to the test
  process (logical CUDA device 0). Other GPUs were not used for these runs.
- Updated system NVIDIA driver: `580.178.04`; CUDA compiler: 12.9; PyTorch:
  `2.13.0+cu129`. No system driver/library was installed or downgraded by this work.
- Full Star suite: **2214 passed, 2505 skipped, zero failures**. Skips include
  two-GPU TP/PP, dedicated SM70 candidate libraries, opt-in experiments and
  absent real checkpoints. These are not passing qualifications.
- Local non-CUDA suite: **1848 passed, 2560 skipped, 311 deselected**.
- Static architecture guard: zero violations. Wheel payload: 164 files,
  no retired reader or vendored kernel source. Whitespace/undefined-name checks
  passed for implementation source.
- Native blocks build identity: `e824afff5ad4628d`; Ada SIMT build identity:
  `1803dcbc5469fa33` (ABI 2, unchanged by the storage migration).

The active Star checkout and builds are isolated under
`/data/wangyue/codex-build/Venturi-cleanup-20261113/`. The reproducible GPU2 runner
and JUnit report are stored there as `run-star-safetensors-gpu2-full.sh` and
`safetensors-gpu2-suite.xml`. A bounded set of frozen Python test references was
copied unchanged, not installed as runtime dependencies or edited as evidence.
Original dirty checkouts, production models/routes/services/keys and GitHub main
were not changed. No trained 27B checkpoint quality or model TPS is certified.
