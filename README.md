# Venturi

A specialized, high-performance inference engine.

**Siphon loads the weights; Venturi runs the model.**

## First milestone: a model-agnostic loader

Venturi **loads checkpoint tensors**, provides reusable Ada Q4 Linear operators,
and can generate text with its first **dense GPT-NeoX model adapter** (Pythia).
Other architectures are not executable merely because their weights can load.
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
automatic downloads, and quantized full-model execution are not implemented yet.

The optional operator API below interprets one explicit packed format; it does
not infer an execution format from arbitrary quantization metadata.

## Text generation: first complete model adapter

The first runner implements **classic dense GPT-NeoX**, validated with the pinned
FP16 EleutherAI Pythia-70M checkpoint used by the loader acceptance tests. Its
forward pass and generation loop are Venturi implementations using PyTorch
operators; they do not call a Transformers model or `model.generate`.

This is a correctness-first eager implementation, **not an optimized serving
engine or a demonstrated speedup over llama.cpp**. The existing Q4 operator
backend is independent: this runner does not silently quantize dense weights or
claim end-to-end Q4 support. Qwen-27B and its mixed `.ninfer` artifact are not
supported by this model adapter.

### Local text completion

Install the optional local tokenizer dependency:

```bash
python -m pip install -e '.[generate]'
venturi generate /absolute/path/to/pythia-70m \
  --prompt "Once upon a time" --max-new-tokens 32
```

The checkpoint directory needs its original `config.json`, safetensors weights,
and `tokenizer.json`. Generation never downloads files. The default CUDA loader
is Siphon, with the same explicit backend and buffer options as `venturi load`.
Use `--tokenizer /absolute/path/to/tokenizer.json` to select the tokenizer
explicitly, and `--json` for token IDs, stop reason, timing, and loader/runtime
information. Without `--json`, the CLI prints only the generated continuation.

If `venturi` is not found, it has not been installed in the active Python
environment. After installation, `python -m venturi` is an equivalent entry
point. Example checkpoint paths above are placeholders, not bundled models.

For the existing School_4090D deployment, the optional SSH launcher uses the
already staged source, checkpoint, and Python runtime without installing
anything on the Mac:

```bash
bash scripts/venturi-4090d generate --prompt "Once upon a time" --max-new-tokens 32
```

Run this from the repository checkout. Its defaults describe this particular
deployment, not a portable model location. `--help` lists environment overrides
for another server or checkpoint. Arguments, including quoted/multiline prompts,
are preserved across SSH. Authentication uses the existing SSH configuration;
the launcher does not change host-key checking, forward an agent, or retry
indefinitely when the network is unavailable.

For the existing pinned Pythia fixture, download only the missing tokenizer:

```bash
curl -fL --retry 3 \
  https://huggingface.co/EleutherAI/pythia-70m/resolve/a39f36b100fe8a5377810d56c3f4789b9c53ac42/tokenizer.json \
  -o "$MODEL_DIR/tokenizer.json.partial"
printf '%s  %s\n' \
  c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624 \
  "$MODEL_DIR/tokenizer.json.partial" | sha256sum -c -
mv "$MODEL_DIR/tokenizer.json.partial" "$MODEL_DIR/tokenizer.json"
```

Use a new target or check existing files before replacing them. `MODEL_DIR` is
the local fixture directory from the FP16 acceptance section. A CPU reference
execution is available explicitly:

```bash
venturi generate /absolute/path/to/pythia-70m --device cpu --loader safetensors \
  --prompt "The capital of France is" --max-new-tokens 16 --json
```

Pythia-70M is a small **base completion model**, not an instruction/chat model.
Repetition or poor answers are model behavior, not evidence of inference
correctness. No chat template, sampling, beam search, or repetition penalty is
applied. Tokenization uses no inserted special tokens, padding, or truncation;
an empty prompt uses the configured BOS token. Greedy decoding stops at EOS or
the requested output limit. Generated EOS IDs are returned but hidden by text
decoding. Nonfinite logits fail instead of being turned into an EOS token.
Tokenizer IDs outside the model vocabulary, and emitted IDs absent
from the tokenizer, are rejected rather than silently dropped.

### Python API and state ownership

```python
from venturi import load_checkpoint
from venturi.models import GPTNeoXModel
from venturi.generation import LocalTokenizer, generate_text

checkpoint = load_checkpoint("/absolute/path/to/pythia-70m", device="cuda:0")
model = GPTNeoXModel(checkpoint)
tokenizer = LocalTokenizer("/absolute/path/to/pythia-70m/tokenizer.json",
                           vocab_size=model.config.vocab_size)
result = generate_text(model, tokenizer, "Once upon a time", max_new_tokens=32)
print(result.text)

# Low-level cached forward API: one unpadded sequence, Python integer token IDs.
session = model.create_session(capacity=128)
logits = session.forward([12092, 2402, 247])  # Last-position logits [vocab].
next_logits = session.forward([int(logits.argmax())])
session.reset()  # Reuse allocated KV storage, not previous request history.
```

The model shares immutable learned weights and derived RoPE tables. Each session
owns a fixed-capacity K/V cache; requests never share mutable cache storage.
`forward(..., all_logits=True)` returns every input position's logits. Cached
chunks use absolute-position causal masking, not SDPA's upper-left rectangular
causal mask. New token positions cannot attend to stale/future cache entries.

Sessions execute/reset only on their creating CUDA stream and are not safe for
concurrent calls on the same session. Independent sessions can use different
streams; they wait for model preparation without a global fence. Caller must
retain model/session tensors until asynchronous work finishes. A failed forward
requires `reset()` before reuse. The high-level generator synchronizes its own stream for
wall-clock timings and creates one session per request by default. An explicitly
supplied reusable session is reset before each nonempty generation request.

The context is bounded by `max_position_embeddings`. Generation conservatively
reserves `prompt_tokens + max_new_tokens` cache positions; oversized requests
fail before execution, without implicit truncation or eviction. Zero requested
new tokens perform no model forward or KV allocation. The full-model eager path
allocates intermediates and is **not** CUDA-graph-captured or allocation-free.

### Model contract and validation

Supported: homogeneous FP16/BF16/FP32 learned tensors on CUDA or CPU (FP64 for
numerical tests), exact GELU, ordinary multihead causal attention, standard
partial/full RoPE with an even rotary dimension, LayerNorm, parallel or sequential
residuals, and untied input/output embeddings. No learned-weight dtype conversion is implicit.
RoPE scaling variants, tied embeddings, bias-free variants, quantization config,
and non-GPT-NeoX model types fail explicitly. Learned names/shapes/dtypes/devices
are validated independently of generic checkpoint storage checks.
Newer `partial_rotary_factor`/`rope_theta` fields must agree with the classic
`rotary_pct`/`rotary_emb_base` interpretation; conflicting descriptions fail.

Logits currently use the learned-weight dtype. An FP32-output LM head over FP16
weights is not implemented; casting already-rounded FP16 logits cannot recover
FP32 token rankings.

Known legacy attention mask, masked-bias, and serialized inverse-frequency
buffers are shape-checked but not used as learned parameters. RoPE is derived
from config in FP32, then cast to the model dtype, rather than using old
FP16-rounded inverse-frequency buffers. The FP64 test path derives it in FP64.

CPU tests compare complete logits and cached continuations with independently
written FP64 equations. Optional Transformers reference tests use a separate
model implementation and are included in CI, not the production dependency set.
For real-checkpoint validation on the CUDA host:

```bash
python -m pip install -e '.[test,generate,reference]'
VENTURI_GENERATION_MODEL=/absolute/path/to/pythia-70m python -m pytest -q
python tools/run_generation_acceptance.py /absolute/path/to/pythia-70m \
  --output /absolute/path/to/generation-acceptance.json
```

The validation tool independently checks FP32 adapter mathematics and FP16
prefill/teacher-forced decode against Transformers references loaded from the
same stored parameters. It also checks a same-precision reference with FP32 RoPE
frequencies preserved, records greedy token agreement at both precisions, and
tests a cached multi-token chunk at position 512. Error limits are dtype-aware:
large logits can have FP16 rounding intervals greater than a fixed absolute
threshold, and FP16 greedy ties need not match FP32 choices. No reference library
is invoked by `venturi generate`.

Cached execution and full-batch prefill are not numerically invariant on this
checkpoint: different matrix shapes and attention kernels can change rounding.
Validation qualifies same-schedule cached FP16/FP32 references separately, and
retains long-context cached-versus-full drift as an explicit diagnostic rather
than increasing its tolerance until it passes. Small-model FP64 oracle tests
check the underlying mathematics independently. Batch-invariant low-precision
execution is not claimed.

Generation timing fields exclude checkpoint loading, tokenizer work, and Python
process startup. `prefill_seconds` includes the first greedy selection;
`decode_seconds` covers later selections; `total_seconds` also includes session
preparation and timing fences. These are eager single-request measurements, not
cross-engine benchmark results. CLI allocator figures are sampled after loading,
before generation/session preparation, unless separately stated.

`time_to_first_token_seconds` measures entry to `generate_tokens` through the
first selected token, including input validation, KV allocation, and the initial
stream fence. It is `null` for a zero-output request. It is not socket/client
latency and excludes loading, tokenization, and text decoding.

### Standard dense generation baseline

Finish and measure the ordinary dense path before integrating Q4 operators.
`benchmark` loads one model, performs unmeasured warmup requests, then measures
independent requests with fresh KV sessions:

```bash
venturi benchmark /absolute/path/to/pythia-70m \
  --prompt "Once upon a time" --max-new-tokens 32 \
  --require-float-dtype F16 --warmup 2 --repeats 10 --json

# Existing 4090D deployment, invoked from the repository on your Mac:
bash scripts/venturi-4090d benchmark --prompt "Once upon a time" \
  --max-new-tokens 32 --require-float-dtype F16 --warmup 2 --repeats 10 --json
```

Normal EOS stopping is the default. Explicit `--ignore-eos` is available only
for benchmark workloads that need exactly the requested number of output
tokens; the report records that policy. Zero-output benchmark requests are
rejected because they cannot measure first-token latency.

Reports retain every run's token IDs, phase times, request/KV bytes, and allocator
peaks, plus median/min/max first-token and request latency. Decode throughput is
`sum(output_tokens - 1) / sum(decode_seconds)`: prefill produces the first token,
which must not count as a decode step. No decode tokens means `null`, not an
invented throughput. Reported aggregate rates are ratios of sums, not averages
of per-request rates. Identical greedy outputs across repetitions check request
repeatability, not reference-model correctness.

CUDA peaks are reset before each measured request, after synchronizing the
request stream. Warmup allocations may remain reserved. The report distinguishes
baseline, total peak, and extra peak allocated bytes. These counters cover only
this process's PyTorch allocator on the device, not total GPU VRAM or Siphon/CUDA
native allocations. Do not profile concurrent work in the same process/device:
peak counter resets are process/device-wide. CPU allocator memory is not
measured. Loading and model preparation are reported separately from warmed
generation, and no allocator cache is flushed between requests.

The measured RTX 4090D FP16 Pythia-70M baseline used two warmup requests and ten
measured requests per workload, with EOS explicitly ignored for fixed lengths:

| Prompt tokens | Output tokens | Median TTFT | Aggregate decode | Peak PyTorch allocated |
| --- | --- | --- | --- | --- |
| 4 | 32 | 5.361 ms | 212.94 tokens/s | 170.09 MiB |
| 5 | 128 | 4.218 ms | 230.22 tokens/s | 171.23 MiB |

These are separate eager workloads on a tiny base model, not a larger-model,
long-prompt, serving, or cross-engine performance claim. All ten outputs were
identical within each workload. The expanded suite passed 225 tests without
skips on the GPU host (three expected Siphon buffer-depth warnings); the local
CPU suite passed 183 tests with 42 opt-in CUDA cases skipped. Raw per-run data,
runtime versions, tested-source hashes, checkpoint identity, and the GPU test
log are in [`benchmarks/fp16-baseline`](benchmarks/fp16-baseline/).

### Comparing with llama.cpp

The paired comparison uses the **same Pythia checkpoint**, not a different
Llama-family model. llama.cpp needs a GGUF conversion of those weights. The
conversion check in `tools/compare_llamacpp.py` verifies every learned value,
including the fused QKV row permutation, and rejects quantized GGUF tensors.
The normal converter promotes bias/normalization vectors to FP32 losslessly;
matrices remain FP16. Legacy mask/serialized RoPE buffers are not learned
parameters and are omitted by the upstream converter.

Build a pinned upstream checkout on the CUDA server, not on the Mac:

```bash
LLAMA_ROOT=/absolute/path/to/pinned/llama.cpp
MODEL_DIR=/absolute/path/to/pythia-70m
cmake -S "$LLAMA_ROOT" -B "$LLAMA_ROOT/build" \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89-real \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_ROOT/build" --target llama-bench -j 8
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python "$LLAMA_ROOT/convert_hf_to_gguf.py" \
  "$MODEL_DIR" --outtype f16 --outfile "$LLAMA_ROOT/pythia-70m-f16.gguf"
g++ -O3 -std=c++17 tools/llama_compare/bench.cpp \
  -I"$LLAMA_ROOT/include" -I"$LLAMA_ROOT/ggml/include" -I"$LLAMA_ROOT/vendor" \
  -L"$LLAMA_ROOT/build/bin" -Wl,-rpath,"$LLAMA_ROOT/build/bin" \
  -lllama -o "$LLAMA_ROOT/venturi-llama-bench"
python tools/compare_llamacpp.py --checkpoint "$MODEL_DIR" \
  --llama-root "$LLAMA_ROOT" --gguf "$LLAMA_ROOT/pythia-70m-f16.gguf" \
  --llama-runner "$LLAMA_ROOT/venturi-llama-bench" \
  --llama-commit 2851b3bcbf8d993a2243ae751c0a7dfa8e944a40 \
  --output-dir /absolute/path/to/new-comparison-results \
  --prompt "Once upon a time" --max-new-tokens 32 --warmup 2 --repeats 10
```

The reported commit must match the checkout actually built; the example pins
the revision selected for this comparison. The caller supplies an environment
with Siphon and the conversion dependencies; the tool does not install packages,
download checkpoints, or build anything. Run from this repository, with Venturi
installed or `src` on `PYTHONPATH`. Use a new output directory each time to retain
failed/previous evidence instead of overwriting it.

Both measured loops reuse already allocated contexts/KV storage, reset logical
history per request, check finite logits, select greedy tokens, and ignore EOS
for explicit fixed output lengths. Input token IDs must match independently in
both tokenizers. The first selected token is prefill, not a decode step. Loading,
tokenization, context allocation, and untimed teacher-forced checks are excluded.
This is a different session-allocation contract from `venturi benchmark` above;
do not silently mix those results.

The custom C++ runner calls the public llama.cpp API. It is **not** the upstream
`llama-bench` program: the upstream tool's synthetic timings omit sampling and
must remain a separate supplementary result. The paired runner tests the default
CUDA-graph policy and a `GGML_CUDA_DISABLE_GRAPHS=1` ablation, while Venturi is
eager. An enabled graph policy alone does not prove every operation was captured.

Equal learned values and FP16 KV caches do not mean identical arithmetic:
llama.cpp retains FP32 activations/logits and uses its tanh GELU approximation
for GPT-NeoX, whereas this Venturi adapter uses exact GELU and learned-weight
precision logits. The report retains greedy-ID equality and full-vocabulary
teacher-forced error/argmax diagnostics rather than declaring numerical
equivalence from a throughput measurement. No tolerance is relaxed to force
agreement. See the pinned [GPT-NeoX graph](https://github.com/ggml-org/llama.cpp/blob/2851b3bcbf8d993a2243ae751c0a7dfa8e944a40/src/models/gptneox.cpp)
and [CUDA GELU implementation](https://github.com/ggml-org/llama.cpp/blob/2851b3bcbf8d993a2243ae751c0a7dfa8e944a40/ggml/src/ggml-cuda/unary.cuh).

Per-process NVML GPU-resident memory is sampled every 20 ms, including loading
and untimed validation. It includes CUDA context/runtime allocations and is a
sampled maximum, not an exact allocator/transient peak. Legacy GPU checkpoint
buffers retained by Venturi but omitted from GGUF also affect resident memory.
Missing NVML support is reported as unavailable, never as zero. Other active
compute processes cause the comparison to stop. An exact path supplied with
`--allow-gpu-process` can exempt a known background desktop service; all observed
background processes are recorded. No workloads are killed and no GPU
clocks/power settings are changed.

#### Measured RTX 4090D paired baseline

Pinned llama.cpp revision: `2851b3bcbf8d993a2243ae751c0a7dfa8e944a40`.
Both engines used the same learned values, FP16 KV caches, two warmups, ten
measured requests, eight CPU threads, and fixed output lengths:

| Prompt/output tokens | Venturi decode | llama.cpp default decode | llama.cpp graphs disabled | Median TTFT Venturi / llama.cpp |
| --- | --- | --- | --- | --- |
| 4 / 32 | 266.12 tokens/s | 1,699.03 tokens/s | 1,281.45 tokens/s | 3.988 / 1.097 ms |
| 5 / 128 | 239.52 tokens/s | 1,779.42 tokens/s | 1,352.03 tokens/s | 3.940 / 1.141 ms |

llama.cpp was **6.38x and 7.43x faster** in these two greedy throughput runs.
Even with its graph policy disabled it remained substantially faster, so graph
capture alone cannot be assumed to close the gap. Attribution to particular
operators or Python dispatch still requires profiling; no optimization speedup
has been demonstrated by this comparison.

All **76 learned tensors** were checked exactly after conversion (26 FP16
matrices and 50 losslessly promoted FP32 vectors). **Free-running greedy token
sequences differed in both workloads.** On 32 teacher-forced logit rows per
workload, top-1 agreement was 30/32 and 31/32; maximum absolute logit errors were
2.391 and 1.830. These remain diagnostics, not a passed numerical-equivalence
claim or an isolated explanation of every difference.

Sampled per-process GPU-resident maxima were 606/608 MiB for Venturi and 480 MiB
for either llama.cpp configuration. The unchanged GNOME remote desktop process
(354 MiB) was explicitly allowed and recorded separately. These resident samples
must not be compared directly with the PyTorch-only peaks in the earlier table.
The GPU suite passed **232 tests with no skips**, with three expected Siphon
warnings; local CPU validation passed 190 tests with 42 CUDA cases skipped.

Raw paired results, precision/cache contracts, source hashes, conversion/build
logs, and the separately labeled upstream synthetic benchmark are retained in
[`benchmarks/llamacpp`](benchmarks/llamacpp/). Upstream synthetic decode measured
approximately 2,748/2,790 tokens/s, but excludes sampling and uses synthetic
inputs, so those numbers are not the paired greedy comparison above.

### Opt-in exact-length CUDA decode graphs

`--execution cuda-graph` prepares reusable FP16 decode graphs while keeping
prefill eager. The default remains `eager`. No weights are quantized or cast,
no GELU approximation is introduced, and no padded/fixed-capacity attention
replaces the actual KV prefix. One graph is captured per exact decode position.
This is execution specialization, not a new attention algorithm.

```bash
# Existing deployment, from the repository checkout on your Mac:
bash scripts/venturi-4090d benchmark --execution cuda-graph \
  --prompt "Once upon a time" --max-new-tokens 32 \
  --ignore-eos --warmup 2 --repeats 10 --json
bash scripts/venturi-4090d generate --execution cuda-graph \
  --prompt "Once upon a time" --max-new-tokens 32 --json
```

Capture is an up-front cost, not a free speedup. The benchmark reports it as
`preparation_seconds`, excluded from warmed request times. Generation JSON also
reports `generation_execution.request_including_preparation_seconds` and
`time_to_first_token_including_preparation_seconds`. One-shot CLI generation can
be slower overall despite faster prepared decode. Reuse a prepared session in a
long-lived Python application to amortize capture:

```python
# model and tokenizer are the CUDA FP16 objects from the Python API example.
prompt = "Once upon a time"
ids = tokenizer.encode(prompt)
session = model.create_cuda_graph_session(
    len(ids) + 32, decode_start=len(ids), decode_steps=31,
)
result = generate_text(model, tokenizer, prompt, max_new_tokens=32, session=session)
# Subsequent requests reset logical KV history and reuse the captured graphs.
again = generate_text(model, tokenizer, prompt, max_new_tokens=32, session=session)
```

A prepared session requires the same model and owning CUDA stream. Every decode
position must lie in its prepared range, and the request must fit its KV capacity.
Prompt contents and token counts may change within those bounds. A wider plan
(for example, capacity 256, decode_start 1, decode_steps 255) supports different
prompt lengths without recapture, at greater up-front memory/capture cost.
EOS may stop early; reset/reuse is supported. A session is not safe for
concurrent calls. Preparation must run without concurrent CUDA work in this
process. Graphs have independent private memory pools so reset/early stopping
cannot violate shared-pool replay ordering; this costs memory proportional to
the number of prepared positions. At most 256 decode graphs (257 output tokens)
are supported per plan. Unprepared decode positions fail instead of silently
capturing more graphs during a request. Only CUDA FP16 is supported for now.

`forward` returns borrowed graph logits, overwritten when the same position is
replayed; clone them if they must outlive request reuse. Captured greedy
selection retains the nonfinite-logit check, returning a checked token with one
host read rather than separately synchronizing finite-check and argmax results.
The session is poisoned on a captured nonfinite selection and must be reset.
No CPU synchronization or data-dependent Python control flow runs inside
capture. See the [PyTorch 2.9 CUDA graph contract](https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-graphs).

For numerical validation and trace collection on the CUDA host:

```bash
python tools/run_cuda_graph_acceptance.py /absolute/path/to/pythia-70m \
  --output /absolute/path/to/graph-acceptance.json
python tools/profile_decode.py /absolute/path/to/pythia-70m \
  --execution cuda-graph --output-dir /absolute/path/to/new-profile
```

The acceptance tool requires complete same-position FP16 logit vectors and
greedy IDs to be bit-exact with the eager runner on its tested workloads. This
does not claim equality to FP32 or llama.cpp. Kineto may collect CPU operators
without CUDA kernel activity on some installations; the report explicitly
marks missing CUDA activity unavailable. `--nsys` instead brackets warmed
decode with `cudaProfilerStart/Stop` for an external Nsight Systems capture.
Instrumented timings must not be substituted for unprofiled benchmark results.

#### Measured FP16 optimization result

Final paired RTX 4090D run: two warmups and ten measured fixed-length requests
per workload, with model/KV allocation and graph preparation outside warmed
timings in both engines:

| Output tokens | Venturi eager | Venturi prepared graphs | Paired llama.cpp | Gain over eager |
| --- | --- | --- | --- | --- |
| 32 | 253.96 tokens/s | 2,515.56 tokens/s | 1,742.18 tokens/s | 9.91x |
| 128 | 257.85 tokens/s | 2,579.64 tokens/s | 1,793.73 tokens/s | 10.00x |

This is approximately **1.44x the paired llama.cpp warmed decode rate** on these
tiny Pythia workloads, not a general engine/larger-model speed claim. Complete
same-position FP16 logit storage and greedy IDs match eager execution on 32- and
128-token acceptance cases; llama.cpp/FP32 numerical equivalence is still not
claimed. Different prompt lengths within a prepared range, EOS/reset, stream
ownership, nonfinite selection, and no hot-path allocator growth are tested.
The GPU suite passed 246 tests without skips; local CPU validation passed 200
tests with 46 CUDA cases skipped. The wheel's 24 source/attribution assets were
checked against the working tree.

Nsight traces show why this helps: eager decoding issued 153 separate kernel
launch API calls and three stream synchronizations per token. The prepared path
issues one input-fill kernel launch plus one graph launch, and one stream
synchronization per token. It still executes 156 GPU kernels per token versus
153 in eager mode: the improvement removes repeated host dispatch/input-copy
overhead, **not** the underlying model computation. GPU kernel fusion remains
a separate optimization opportunity.

Tradeoffs are explicit: preparation cost 0.616/1.365 seconds, and sampled
resident GPU memory increased from 606/608 MiB to 764/1,134 MiB for the two
workloads. Reused-plan TTFT remained approximately 4.1 ms versus llama.cpp's
1.1 ms; prefill was not optimized. A cold one-shot request can be slower than
eager generation. Using the measured request times, capture amortizes after
roughly six/four matching fixed-length requests, not necessarily after one.

Final raw comparisons, byte-level acceptance, CLI startup-inclusive metrics,
source hashes, CPU operator tables, and Nsight API/kernel statistics are in
[`benchmarks/cuda-graph`](benchmarks/cuda-graph/). An earlier full pass measured
2,482/2,601 tokens/s with the same optimization; the table uses the final pass,
not a best-of selection across runs. Binary traces remain on the GPU host.

### RTX 4090D generation validation

The real-checkpoint run passed **209 tests with no skips**, including previous
loader/operator checks and the new generation tests. Local CPU validation passed
168 tests, with 41 opt-in CUDA cases skipped. The wheel's 22 source/attribution
assets were verified against the repository.

All three 16-token greedy continuations matched the separate same-precision
FP16 reference, using the same FP32-frequency RoPE contract. At position 512,
cached FP16 and FP32 logits matched their respective same-schedule references
exactly in this run. **The continuations did not match FP32 greedy decoding.**

Long-context cached-versus-full-batch drift is retained as a known limitation:
FP16 raw-logit maximum difference was 4.5, and cached FP32 versus full eager
reference maximum difference was approximately 0.0745. Both exceeded their close
comparison bounds; they are explicitly diagnostic, not claimed as passing
batch-invariance checks. Same-schedule reference checks and independent FP64
mathematical tests provide separate correctness evidence.

See [`benchmarks/generation/validation.json`](benchmarks/generation/validation.json),
the [raw reference report](benchmarks/generation/rtx4090d-pythia70m.json), and the
[CLI output sample](benchmarks/generation/cli-sample.json). These results establish
the first working dense text-generation path, not serving performance or model
answer quality.

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

At the Q4 operator milestone, all **145 tests passed** on the RTX 4090D, including the Siphon/FP16
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
With the `generate` and `reference` extras installed, it also exercises local
tokenization and model/oracle tests without downloading any model checkpoint.

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

The default test suite never downloads a checkpoint. Real generation tests
require `VENTURI_GENERATION_MODEL` and a matching local `tokenizer.json`.

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
src/venturi/ops/            Optional reusable Ada quantized Linear backend
src/venturi/models/         Explicit model adapters and per-request KV state
src/venturi/generation.py   Local tokenizer and bounded greedy generation
src/venturi/cli.py          inspect / load / verify / generate
tools/                     Loader, operator, and generation acceptance tools
tests/                     Storage, operator, model/oracle, and opt-in CUDA tests
```
