# VRAM-first KV prototype — two V100s, 2026-10-23

## Decision

Implemented, **opt-in/default OFF**. KV commits GPU memory as prefixes grow. No
CPU spill backing is allocated while the admitted prefixes fit in VRAM. Under
pressure, inactive request/layer entries are evicted before active-entry LRU;
exact bytes are reloaded before GPU attention. GDN, weights and activations stay
on GPU. This is an OOM-avoidance mechanism, **not a speed optimization or a new
paged-attention kernel**. The original production service was restored, not
replaced with this prototype.

## Exact scope / provenance

- Host `WZU_Server`, `wzu-SYS-4029GP-TRT`, the same two V100-PCIE-32GB GPUs:
  `GPU-18fbf081-3c6d-d65b-11b8-288a4ce1ab28`,
  `GPU-f65db2e8-87bb-c88d-363f-23c79f793548`.
- Official-source **Qwen3.8-27B native Q8** payload
  `qwen3.8-27b-text-q8-rms-v1.ninfer`, SHA256
  `41f99992b679aef7df2b6c671b76b09820730846c2c95a52406e7004c72f58f0`.
  This is NOT the production Uncensored GGUF and is not a llama.cpp comparison.
- Q8 group32 weights/FP16 scales; FP16 values/KV, residual-A8 projection planes,
  FP32 accumulation/GDN state. Existing row-pair, SwiGLU-quant and RMS-quant build
  flags ON, runtime experiment mask 0 in full-model runs. No KV quantization,
  speculation/MTP, HTTP overhead or Python token loop in the benchmark.
- Native two-device **host TP** head-GDN profile, max projection rows 32. C1 means
  one request on the SAME two GPUs, not one GPU. C8 is a fixed eight-slot cohort.
- Main controls: pinned 16-token prompts, 32 greedy output tokens, EOS disabled,
  capacity 256. One full warm generation/reset per count; TTFT, subsequent 31
  tokens and whole 32-token call have separate timers. C8 post-first denominator
  is 248 output tokens / cohort wall time, never summed request times.
- First implementation: three alternating-order paired repeats, stored in
  `evidence/window-v2/`. Fatal VMM-driver-error ownership was then hardened;
  final native library requalified and paired once in `evidence/window-final-v3/`.
- Final .so SHA256
  `a8102f1aee9cbd3b70153d3e1cc462dd0477ee4ef2042706cc84f6881c5677e6`,
  remote path `/home/data/wangyue/artifacts/venturi-kv-tier-v3/build/4c0ee19b5eb6d607/libventuri_blocks.so`.
  All **17** native source inputs match the final 27B benchmark.
- A later Python-only diagnostic-lifetime/close-race fix was tested separately:
  requests/batches continue after public model-owner close, without querying its
  deleted handle. **323** current src/tests/tools files match that executed test
  snapshot; the native binary and arithmetic are unchanged. The 27B timings
  precede this diagnostic-only wrapper fix. See `executed-source-verification.json`
  and the exact measured wrapper snapshots under `measured-v3-source/`.

## Correctness / failure behavior

Full-model static, demand-GPU and forced-spill paths matched **logits, layer
traces, greedy IDs and committed KV/GDN state hashes exactly**, with finite
values checked. Final hardened forced-spill qualification also matched the
original full-model reference. A separate C8 512-input/8-continuation-token test
matched last logits and all committed states exactly.

Remote small-model regression: **22 passed** on the final wrapper/native binary,
including forced spill/reload, reset, selected-slot reordering, EOS compaction,
failed-request reset, cold-cohort concurrent close, graph/other experiment-mask
compatibility, owner-close lifetime and query/close locking. Local regression:
**38 passed, 205 skipped** (GPU tests explicitly skipped on the Mac).

The native VMM unit crosses physical page boundaries, verifies spill/reload
bytes and teardown, and fault-injects physical allocation OOM without exhausting
shared GPUs. Insufficient pool/RAM quotas fail before recurrent forward mutation.
Non-OOM driver/access failures fail closed and require model recreation; partial
mapping ownership is retained for cleanup, never recycled to another request.
These are **injected allocation/access failures**, not a real device-exhaustion
stress test. Full-model forced pressure is a configured 128 MiB GPU cap.

The first fixture failure is preserved in `evidence/build-v1/`: its test assumed
one full-attention layer and incorrectly expected a zero-RAM one-entry budget to
fit the multi-layer fixture. The test budget was corrected to the actual layer
count; this was not a numerical counterexample hidden from the report.

## Unprofiled speed

Post-first decode TPS; C8 columns are **aggregate**, not per-request speed.

| Mode | C1, first 3 mean | C8, first 3 mean | C1 final native recheck | C8 final native recheck |
|---|---:|---:|---:|---:|
| Legacy static KV | 32.974 | 134.685 | 32.771 | 133.366 |
| VRAM-first, fits GPU | 32.947 | 134.492 | 32.734 | 132.492 |
| VRAM-first, 128 MiB cap | 32.961 | 79.353 | 32.614 | 77.346 |

C1 fits the 128 MiB pool and does **not** spill. No GPU-resident speed gain is
claimed: first C8 mean is -0.14%; the single final pair is -0.66%, not a statistical
confidence bound. Forced C8 spill is about 41–42% slower here.

Whole 16-input/32-output C8 TPS: first means **104.659 / 104.640 / 65.778**;
final pair **103.475 / 103.299 / 64.114** (static / fits / forced spill).
CPU is approximately **1.00–1.01 process cores**, separately reported in every
trial; this does not establish CPU-boundness.

Context sensitivity matters. At **512 input tokens/request**, one unprofiled C8
correctness run measured static vs forced-spill prefill **23.977 vs 37.820 s**
(4096 total input tokens), and eight-token continuation **60.155 vs 20.950 aggregate
TPS**. This is one sample at a different context/timing scope, not a replacement
for the short-context three-repeat comparison. Longer RAM-backed contexts must
not be advertised using the 79 TPS short-context result.

## Memory: dynamic capacity is not free

V100 VMM granularity was measured at **2 MiB per K or V plane**. A short
request/layer therefore needs a 4 MiB GPU mapping. At C8/capacity256:

| Mode | Tier physical GPU pool | CPU backing capacity | Spill/reload during timed 31-token C8 decode |
|---|---:|---:|---|
| Fits GPU | 512 MiB | 0 | 0 / 0 |
| Forced GPU cap | 128 MiB | 512 MiB | 3968 / 3968 |

Forced phase copy deltas: **516,030,464 D2H + 503,840,768 H2D bytes**. No-pressure
phase copy/eviction/OOM deltas were all zero. CPU backing capacity is not RSS or
pinned memory. Existing pinned model staging is reused; whole prefixes are copied
synchronously in this first implementation. RAM quota conservatively pre-admits
backing for every possible victim, not only the currently evicted subset.

The tiny legacy capacity requests only 144 MiB KV for eight active slots plus
the extra empty scalar control. VMM's 512 MiB minimum-page pool can therefore be
**larger**, not smaller. NVML C8 primary samples were 16167 MiB static, 16679 MiB
fits, 16295 MiB forced; peer 14501 MiB throughout. Treat NVML/context/allocator
usage separately from requested buffer counts; they are not interchangeable.

A separate **capacity 65536** C8 check created and ran short prompts with a
512 MiB physical KV pool and zero CPU backing, matching the short reference.
At that declared capacity, the old formula requires **32 GiB KV for eight slots**
before weights/GDN/workspaces (calculation, not an attempted real OOM).
**It did not run a full 64K prompt:** only 16 input/32 output tokens (47 committed
positions). There is no full-64K/256K speed, quality or memory qualification.
RoPE tables, GDN, weights and workspaces are still eager. A whole request/layer
prefix must fit the GPU pool. GPU pages are retained for reuse until model close;
non-KV allocation failures do not trigger weight offload.

## Separate GPU diagnostics (never used as throughput scores)

NVML C8 short decode samples: clocks **1380/877 MHz** unchanged, **39–43 C**, GPU busy
approximately **69–85%** without spill versus **38–69%** with spill; power ranges
and per-device samples are in `summary.json`. GPU busy is not achieved occupancy
or DRAM bandwidth. No clocks or profiling privilege policy were changed.

Nsys, four C8 post-first steps, static -> forced spill:

- **14,112 -> 14,112 kernels**, **3100 -> 5148 copies**;
- copy bytes **292,012,800 -> 366,985,984**;
- runtime API records **19,550 -> 27,746**;
- observed common GPU activity span **243.101 -> 417.687 ms**;
- primary no-recorded-activity fraction **24.34 -> 54.14%**, peer **32.78 -> 60.90%**;
- kernel union times remained close; projection row-pair kernels were about
  **58% of summed kernel time**, not 58% SM occupancy.

Copy/synchronization gaps increased without adding model arithmetic. This trace
has **no driver API table**: VMM map/unmap latency is not separately measured, so
not all the additional gap can be assigned to mapping calls. Idle fractions are
based on recorded kernels/copies/memsets over a common span, not an NVML metric.

NCU recorded one primary-device `causal_attention` kernel (24 CTAs x 256 threads):
static/spill **12.49/12.49% achieved active-cycle SM occupancy**, **3.05/3.05% SM
throughput**, **24.352/23.648 us**, **87,680/87,424 DRAM read bytes** and zero
measured writes; derived read+write DRAM rate **3.60/3.70 GB/s**. Long-scoreboard
stall ~33.2%, short ~10.6%. These are **one short-context attention kernel**, not
whole-model or projection occupancy/bandwidth. Caches/clocks were uncontrolled
as explicitly warned in the raw reports; no speed gain is inferred from the
sub-microsecond duration difference. Both 12-counter runs succeeded with existing
one-off sudo permissions; `RmProfilingAdminOnly: 1` stayed unchanged.

## Use / implementation

See [parameter and limitation guide](../../docs/vram-first-kv.md). Example additions
to an otherwise fully specified native resident model:

```python
kv_cache="vram-first",
kv_gpu_budget_bytes=0,           # use free GPU capacity minus reserve
kv_ram_budget_bytes=4 * 2**30,   # explicit backing quota, not a context promise
```

CLI: `--kv-cache vram-first --kv-gpu-budget-mib 0 --kv-ram-budget-mib 4096`.
A zero RAM quota forbids spill. MTP/PP are explicitly rejected. Only SM70 host-TP
C1/C8 is qualified here; other model families/devices are not qualified by this
experiment. The byte-tier manager can be reused behind other model bindings,
but those bindings/tests have not been added in this task.

Code: `src/venturi/ops/csrc/qwen_kv_tier.cuh`, runtime/batch integration in
`qwen_runtime.cuh` / `qwen_batch.cuh`, Python configuration and diagnostics in
`src/venturi/models/qwen_native.py`, CLI in `src/venturi/qwen_native_cli.py`.

## Restoration / evidence

Both maintenance windows drained active requests and had independent restoration
timers. Same production container ID/config fingerprint restored, health/models
checked, and a final four-token production chat request returned HTTP 200 with a
nonempty response. Original production remains **llama.cpp Qwen3.8-27B-Uncensored
GGUF Q8_0 + its original parameters**, not this native checkpoint. Test workers
and restoration timers are inactive. The temporary timeout drop-in for our own
experiment unit was removed. No production arguments, driver policy, clocks or
host keys were changed.

SSH temporarily stalled during final report retrieval. Bounded fresh IPv4/IPv6,
9023 and existing authorized jump routes were tried, preserving verification and
never changing benchmark GPUs. After a later 1200-byte Tailscale probe established
a direct path, a fresh owned SSH session recovered the exact same V100 host and
confirmed completed tests/restored production. `network-recovery.json` preserves
failed attempts; a disco pong alone was not treated as proof that SSH worked.

`evidence.tgz`: SHA256
`ad337d1edfa94357dec389d7d247c5be815111b1b502117f9c3bbc229a9fd0d7`.
All **189** raw members were size/hash verified locally. `summary.json` is derived
by `summarize.py`, not a substitute for raw reports. Previous six capsules
(**2474 pinned files**) remain byte-identical. No Rust was compiled locally and no
production deployment or GitHub push was performed.

Next opportunities, not implemented here: dirty-tail D2H copies, overlap/prefetch
with bounded pinned staging, less wasteful small-prefix packing, and separately
qualified batched/split-context attention kernels for long-context execution.
