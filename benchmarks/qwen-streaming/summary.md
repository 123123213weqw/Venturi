# Complete-model Q8 streaming qualification

Date: 2026-10-19 (Asia/Shanghai). Original implementation goal: **active**.

## Result and scope

The complete 64-layer Qwen3.5-27B text decoder executes using Venturi-owned native
numerical CUDA operators on SM70/V100 with Q8 weights, FP16 activations/KV, and
FP32 small parameters/gates/recurrent state. The recorded 26-token text-only chat
case generated `[57590, 248046]`: **Paris**, then EOS. The complete artifact and
original tokenizer hashes were checked before execution. This is a complete
trained-checkpoint **streaming forward smoke test**, not a production-native
runtime, independent upstream architecture check or broad quality qualification.

Both same-input CPU Torch reference steps completed and passed the numerical
gates declared in the tool before the run. The reference follows native-generated
inputs with the same represented Q8 weights and FP16/FP32 execution profile. It
shares the model adapter and therefore cannot independently detect every
architecture, template or source-binding error. The independent tiny-fixture
FP64 and real-weight per-operator checks remain separate evidence.

The full-model run was on V100. The 4090D suite below checks fixtures/operators
and regression behavior; it is **not** a full trained-model 4090D result.

## Implementation

- `StreamingQwenModel.from_q8` is explicit. The resident API remains guarded and
  never silently switches to offload. Embedding, one decoder layer and output
  head load in separate synchronized weight phases; only final norm persists.
- Python performs weight reads/uploads, position tables, explicit KV copies and
  synchronizations. The qualifier performs host CPU Torch greedy argmax. These
  are not a native scheduler/sampler or whole-model CUDA graphs.
- Request-owned neutral KV buffers do not retain residents. Each transient
  attention resident gets its normal owned cache, imports the intended prefix
  through `LayerKVCache.load_prefix`, runs, then copies the new tail back. Cache
  identity is not rebound; shape, alias, capacity, stream and failure checks
  remain. Prefix staging is outside capture into an empty cache only.
- GDN history and FP32 recurrent state persist per request. Position commits
  only after the entire forward call succeeds. A failure after execution starts
  poisons the request until reset; there is no fictitious GDN suffix rollback.
- Max-prefill and decode lanes reuse per-kind workspaces. Optional traces own
  CPU snapshots. Scalar observers cannot retain layer plans/weights. Weakref
  tests check that normal forward does not retain transient residents.
- Source file identity/size/mtime/ctime and metadata/config stability are checked
  throughout. The model owns separately reopened metadata, not the caller's
  mutable manifest. These checks supplement the qualifier's full SHA256.
- This single-request diagnostic does not qualify concurrent memory admission,
  continuous batching, paged attention, graph capture or allocator efficiency.
  CPU reference mode explicitly expands one phase; its conversion scratch is
  additional to resident tensor budgeting and has a conservative RAM preflight.

## Source and precision

Official source: `Qwen/Qwen3.5-27B`, pinned revision
`fc05daec18b0a78c049392ed2e771dde82bdf654`. The original BF16 acquisition and
separate export/independent verification are recorded in
`../qwen-source-bindings/` and the preceding goal capsules. The mixed-Q4 artifact
was **not** promoted and called original-quality Q8.

- Artifact: `/home/wzu/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer`.
- Artifact ID: `3d635b87db004f42aefd6730e372e530`.
- SHA256: `b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8`.
- File bytes: 28,601,956,352; 851 text tensors: 402 Q8, 305 BF16, 144 FP32.
- Tokenizer SHA256:
  `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`.
- Exact prompt, IDs, template hash, EOS IDs and explicit greedy override are in
  `case.json`; local decoding with existing tokenizers 0.22.2 is recorded in
  `native-decoding.json`. Source stochastic defaults were not used.

ABI 10 and CUDA source are unchanged from the lossless-small-parameter milestone.
Small BF16 a/b/conv parameters widen losslessly to FP32 on the FP16 path; Q8
planes stay encoded. Previous FP16 underflow and BF16/FP64 cascade counterexamples
remain immutable in their earlier capsules. Passing this case does not erase
those observations or qualify original-BF16 quality.

CUDA source SHA256:
`75b87c140fa07feebabcc3ba353ce8b9798889d4ae6fda0faf7b8d3d01990aa4`.

| Target | Library SHA256 | CUDA compiler |
| --- | --- | --- |
| SM70 | `94fa82adac3f2ea6d5c06f1405e15f342862f7e7fbca08cdc4ae83ed9c9de783` | 12.9.86 |
| SM89 | `4c14b35fc2ab5cf99125975163ade8a1c746fc8bf11f788fc1b391538eb0b2c8` | 13.0.88 |

Verified cached remote libraries were reused; no CUDA or Rust compilation was
performed locally. Builds/commands/hashes are in each target's `build.json`.
The external runner binary SHA remains
`3c7da6aeb337ee374ce912934c9335252e39326cd9cba3e2163696c06d15dfb4`.

## Full-model numerical results

Authoritative report: `v100/full-model-run1/report.json`.

Predeclared per-step logit gates: equal top-1; relative L2 <= 0.02; centered
cosine >= 0.999; KL(reference -> native) <= 0.002. Logits/traces/states must be
finite and shape-matched. These are initial numerical smoke gates, not corpus
quality tolerances or a claim of bitwise equality.

| Step | Consumed position | Native / reference top-1 | Max abs logit error | Relative L2 | Centered cosine | KL(ref->native) | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 26 | 57590 / 57590 | 0.04589844 | 0.002625315 | 0.9999943574 | 5.33031e-7 | pass |
| 1 | 27 | 248046 / 248046 | 0.02825928 | 0.002307423 | 0.9999959268 | 2.76891e-7 | pass |

All 64 hidden layer snapshots and 128 GDN/KV tensor comparisons per step are
retained. These are diagnostics, not additional after-the-fact acceptance gates:

- Prefill: maximum hidden relative L2 0.003529864 at layer 58 (max abs 4.5);
  maximum state relative L2 0.003600135 at GDN layer 54 recurrent state.
- Continuation: maximum hidden relative L2 0.002726761 at layer 63 (max abs
  0.21875); maximum state relative L2 0.003374101 at GDN layer 60 recurrent state.
- Maximum state absolute difference in both steps is 0.3125 in layer-63 V.
  This is not zero-error state parity; the report preserves every comparison.

The last EOS is emitted, not fed back: positions 26 then 27 are intentional.
Large `.pt` snapshots stay in the remote run directory, outside the repository.
The original worker PID 3040553 completed normally and was never restarted.

## Memory and timing scope

Native pass peak PyTorch allocated bytes: **1,518,406,656**; reserved:
**1,553,989,632**. This is low because weights stream, not because all weights fit
in 1.52 GB. It excludes CUDA context/driver allocations and other processes.
The full native FP16 weight-tensor budget remains 28,654,073,856 bytes before
state/KV/workspaces, so full residency cannot fit a 24 GiB 4090D.

Request state/KV uses 155,779,072 bytes at capacity 28. Retained workspace is
9,652,864 bytes after prefill and 10,024,128 after decode. CPU reference preflight
includes conversion scratch and an 8 GiB reserve; the simple resident budget
alone does not cover expanded-reference scratch.

Entire qualification: wall 908.66454 s; process CPU 2745.108742 s; peak process
RSS 33,002,692,608 bytes. These include hashing, uploads, staging, all diagnostic
snapshots and allocating CPU reference. **They are not native TTFT, token latency,
TPS, CPU-efficiency measurements or a matched engine comparison.**

Existing services were left running. Before testing, V100 physical GPU 1 had
about 3,238 MiB free and 29,257 MiB occupied; the 4090D had about 9,007 MiB free
and 15,074 MiB occupied. This is not isolated benchmark hardware. Sampled idle
busy percentage/free memory are only context, not occupancy or efficiency.
No achieved occupancy/SM/DRAM/cache/stall counters were collected. Existing
`RmProfilingAdminOnly=1` policy is recorded; privileges/settings were not changed
and no new counter campaign was run. No performance win is claimed.

## Tests and recovery

| Environment | Suite | Passed | Skipped | Time |
| --- | --- | --- | --- | --- |
| Local CPU | Full regression | 666 | 317 | 16.24 s |
| 4090D SM89 | Full CPU + native regression | 947 | 36 | 42.82 s |
| V100 SM70 | Selected streaming/port/tool/build CPU + native | 81 | 23 | 6.70 s |

Skips include unavailable local CUDA, explicit SM70 BF16 rejection and tests
outside selected device scope. The V100 environment's missing NumPy warning is
retained; no environment dependencies were changed. Tests cover complete tiny
FP64-source-oracle sessions, chunking/continuation, KV/GDN ownership, resident
parity, poisoned-request reset, source/config mutation, native numerical-fallback
prohibition and phase release. Test counts are not all GPU-only cases.

Logs include an earlier terminal 945/34 SM89 run and one dispatch failure before
target execution. The final 947/36 run is authoritative. Transient network
failures were diagnosed instead of restarting unknown work: partial tokenizer
SSH transfer was recovered through normal-TLS official pinned HTTPS and verified
against the original hash. A failed nested School route recovered over strict
direct SSH after confirming no active pytest. Later Tailscale CLI `nc` returned
HTTP 502 while normal OS TCP/SSH routes were healthy; report fetches used those
existing authorized routes with host-key checks intact. See
`network-recovery.json`. No keys were copied/forwarded, verification disabled,
services stopped, router changed or worker duplicated.

## Reproduction and provenance

Remote source: `/home/wzu/codex-build/Venturi/streaming-src`.
Use a fresh output directory for a new run; the recorded one must not be reused.

```sh
cd /home/wzu/codex-build/Venturi/streaming-src
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python -u \
  tools/qualify_qwen_streaming.py \
  /home/wzu/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer \
  --blocks-library /home/data/wangyue/repos/codex-build/Venturi/sm70-build/308dc54fb9e2a32a/libventuri_blocks.so \
  --expected-sha256 b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8 \
  --case /home/wzu/codex-build/Venturi/streaming-validation/case.json \
  --tokenizer-file /home/wzu/codex-build/Venturi/models/qwen3.5-27b-bf16/tokenizer.json \
  --output-directory /home/wzu/codex-build/Venturi/streaming-validation/full-model-run2 \
  --dtype fp16 --run-cpu-reference
```

`source/` and `source-sha256.json` freeze the current source, tests and docs,
including pre-existing goal work. The four critical worker-source hashes match
`v100/full-model-process.json`; the unchanged CUDA source matches both builds.
`evidence-check.json` records verified status flags. Previous capsules are not
rewritten.

## Still required

Broader trained-checkpoint corpus and independent upstream/original-BF16 quality;
owned native runtime/sampling and reusable whole-model CUDA graphs; batch-aware
GDN and request-safe paged KV/continuous batching; and matched same-checkpoint,
precision, lengths and concurrency single/parallel engine benchmarks on 4090D.
No full-model artifact transfer to 4090D or engine comparison was performed in
this milestone. Offload correctness does not redefine the original speed goal.
