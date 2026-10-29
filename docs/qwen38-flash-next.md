# Qwen3.8-Flash-Next: explicit architecture intake

Updated 2026-10-25, Asia/Shanghai. **Native full-model generation is not
implemented.** These components are foundations, not a renamed dense-27B adapter.

## Source identity and architecture

- Original model: `Qwen/Qwen3.8-Flash-Next`, pinned HF revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
- Official quantized variant: `Qwen/Qwen3.8-Flash-Next-FP8`, pinned revision
  `236dfdf285828023ca3bcd3f37366c58a3469b13`.
- Container/text types: `qwen4_exp` / `qwen4_exp_text`, **not** `qwen3_5`.
- 48 layers:36 GDN and12 QSA/global-attention layers; hidden2560;24 query
  heads,2 KV heads,head dimension256;4 residual streams with lowrank320.
- 512 routed experts,10 selected per token,expert width640 plus shared expert.
- PLE at one-based decoder layer2:order2/3,8 heads each,16 total; each addressed
  row has160 values.128 contiguous checkpoint table shards are storage pieces,
  not128 independent models or requests. Nominal context262144 is not a tested
  Venturi context limit for this model.

Official architecture/model information:
[model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
[architecture report](https://arxiv.org/abs/2608.30320).
Full131-shard original headers and index were checked without downloading360GB
of payloads.1658 tensors account for359,999,963,128 source payload bytes:

| Component | Bytes | GiB |
|---|---:|---:|
|Text core, including experts|251,487,307,800|234.216|
|BF16 ngram tables|102,400,491,520|95.368|
|MTP|5,214,301,696|4.856|
|Vision|897,862,112|0.836|

Marketing active-parameter counts do not imply only active weights must be
stored. Header/source-row sampling does not authenticate all weight payloads.

## Added components

- `venturi.flash_next`: strict configuration/partial binding-schema checks,
  index/header consistency, exact source component accounting, explicit
  capacity-only expert-Q8/Q4 estimates. Always reports
  `native_generation_supported=false`. No implicit remote code or GPU init.
- `NGramRequest`: request-private token history with non-mutating previews,
  accepted-prefix commit, EOS segment boundaries, reset and stale-ticket checks.
  Serialized I64 multipliers are required, never guessed from a missing config.
- `ShardedNGramTable`: original floating CPU safetensors mmap row reads. Full
  tables are never uploaded or eagerly materialized by this reader. Coalesced
  `lookup_many` deduplicates addressed rows across requests but never merges KV,
  GDN or history. Caller-owned executor prefetch does not commit tokens.
- `FP8ShardedNGramTable`: separately selected official PLE encoding, E4M3FN
  rows plus **one BF16 per-tensor `weight_scale`**, decoded only for requested
  rows on CPU. This is neither INT8 nor the main model's block128 FP8 GEMM
  format. Output matches the documented/reference order FP8→BF16→BF16 scale
  multiplication. No FP8 Tensor Core is needed for this CPU path.

The existing dense loader, runtime defaults, quantization arithmetic, TP/PP,
CUDA library and production routing are unchanged. No full Flash-Next forward,
continuous serving scheduler, MTP integration or QSA executor is exposed.

## Inspect without loading weights

```sh
PYTHONPATH=src python -m venturi flash-next inspect \
  benchmarks/qwen38-flash-next-intake/evidence \
  --headers-dir benchmarks/qwen38-flash-next-intake/evidence/headers \
  --profile expert-q8-g32 --gpu-budget-gib 30 30 --ram-budget-gib 200 \
  --batch-size 8 --context 8192
```

Without `--headers-dir`, inspect requires an actual complete local safetensors
checkpoint. The current strict fused-expert manifest schema targets the
original BF16 source; official FP8 splits experts into per-expert tensors and
has a different binding/export contract. The standalone FP8 PLE reader does
not mean the entire FP8 model is bound. There is no native `generate` command. A later dated section documents the
explicit experimental `generate-reference` entrypoint and its qualification limits.

## Placement/format decision

First preserve original BF16 PLE rows as the correctness oracle. Keep the table
in CPU RAM or file-backed mappings, and gather only the required rows. A token
requires16×160×2=5120 output bytes; batch8 requires40960 bytes, **not**102.4GB
of transfer per step. Pin/transfer bounded gathered buffers, not the entire
table. mmap does not ensure warm/resident RAM; disk faults, OS page cache and
prefetch costs must be measured separately.

Official FP8 PLE cuts code storage to51,200,245,760 bytes plus its2-byte scale
(47.684GiB), still producing BF16 rows. This is a changed numerical checkpoint,
not an exact BF16-quality claim. Main-model FP8 is block-scaled128×128 and is
not equivalent to the PLE per-tensor scale. Do not feed FP8 bytes to INT8 DP4A.

Capacity lower bounds below use **only routed experts quantized**, retaining
other text tensors in original precision, CPU BF16 PLE, no vision/MTP, batch8
and8192 tokens,60GiB total GPU budget. They include a1.5GiB dense FP16 K/V
backing lower bound, but exclude GDN, QSA-indexer, PLE state, scratch, CUDA
context/graphs, OS reserve, per-rank layout and topology constraints.

| Expert storage proposal | GPU-preferred text weights GiB | Minimum CPU GiB including ideal GPU overflow |
|---|---:|---:|
|Original BF16|234.216|271.084|
|Q8 group32/FP16 scales|128.747|165.615|
|Q4 group64/FP16 scales|68.981|105.849|

These are not executable configurations or an exporter. A two-V100 full
resident Q8 path is impossible at this weight size; an explicit expert
cache/offload path is necessary. Q4 still does not make the entire source core
fit in60GiB. Conversely an8×L40 capacity budget could hold the original main
core with host PLE, but this does not establish supported TP/EP geometry.

For4090D24GB/128GB host RAM, original BF16 PLE plus the Q4-expert/core overflow
already exceeds a conservative host budget. Official FP8 PLE or another
explicitly qualified table codec/cache changes that calculation. Never rely on
OOM followed by uncontrolled CPU allocations; admit against explicit budgets.

## Ngram embedding is not ngram speculation

PLE hashes real token context into trained embedding tables and is mandatory
model computation. Ngram speculative decoding proposes repeated token spans
without a trained draft. They may coexist, but are independent. Speculation
must checkpoint/restore the new model's GDN, PLE convolution/history, QSA/KV and
four-stream state as appropriate, accepting only a verified prefix. Reusing the
27B MTP wrapper without these contracts is incorrect.

## Next implementation gates

1. Gated residual read/inject and final mixer: independent FP32/BF16 fixtures,
   four-stream state and grouped RMS conventions.
2. MoE router, selected expert GEMMs and shared-expert gate: establish original
   floating semantics before Q8/Q4 codecs/cache dispatch; do not prune experts.
3. New GDN geometry plus QSA indexer/compression/masks/position handling;
   dense attention cannot be substituted as an unlabelled optimization.
4. Complete text-only floating reference, then owned native binding/forward,
   tokenizer/chat/EOS, checkpointed per-request state and batched correctness.
5. Only then evaluate Q8/Q4 placement, bounded RAM/expert caching, MTP and
   ngram speculation at1/2/4/8 requests. Compare actual emitted tokens over
   cohort wall time; profile separately and preserve losing configurations.

Evidence and current overnight handoff are in
`benchmarks/qwen38-flash-next-intake/README.md`. User authorized overnight
unattended work and temporary GPU service stops. Restore services that were
running before a lease; do not resurrect old stopped Flash-Next jobs. The new
thread heartbeat is `flash-next`, limited to six hourly continuations and no
new experiments after2026-10-25 08:00 Asia/Shanghai. Do not resume the separate
blocked27B llama.cpp goal or old L40 queues/heartbeats as part of this work.

## 2026-10-25 02:03 continuation

- Added opt-in `PreparedNGramTable`: persistent bounded file/slice handles,
  explicit preparation/close, serialized executor/close safety, owned outputs,
  original/official-FP8 output-bit tests. No whole-table eager materialization.
  The new same-condition BF16/FP8 V100 ablation completed and restored production;
  raw evidence retrieval/trace aggregation is pending intermittent SSH recovery.
  See `benchmarks/qwen38-flash-next-prepared-v1/README.md`. No default switch.
- Added floating `FlashNextGatedResidual` plus bounded checkpoint bindings.
  Real official layer0-attention-GR/final-mixer weights (26.34MB) passed12
  exact-bit pinned-reference CPU comparisons in FP32/BF16.16 independent fixture
  cases cover grouped normalization, zero-centering, read/inject/final-mix and
  ownership. This is a component gate, **not native full-model support**.
  See `benchmarks/qwen38-flash-next-residual-v1/README.md` for raw source samples
  and the separately prepared correctness-only V100 check. Full local tests:
  955 passed/2583 skipped; no local Rust build or GitHub push.
- Next MoE gate must apply pinned configuration defaults: official JSON omits
  `norm_topk_prob`, but the pinned upstream default is **True**. Preserve FP32
  router softmax, selected-score renormalization and cast boundaries. Then
  qualify expert gather/weighted accumulation/shared gate before quantization.

The earlier intake evidence/source snapshot remains unchanged. New capsules use
new paths/units; interrupted log transfer is not a reason to repeat completed
GPU experiments. No production service changes are needed for bounded small
GR correctness-only checks, and they must not be scored as throughput under
concurrent serving load.

### 02:13 completed follow-on gates

Prepared-reader raw scores/CSV and separate trace aggregates are now verified:
BF16 C1 serial2.526→0.857ms/step; C8 coalesced15.116→6.087ms/step,
three paired unprofiled repeats all win. FP8 corresponding2.402→0.858 and
14.483→5.892ms. These are **warm synthetic PLE gather/H2D component timings**,
not model TPS. Preparation1.6ms is excluded and reported separately. C1 still
benefits from avoiding coalescing overhead; no automatic/default selector changed.

The GR-only V100 check also completed:12 exact-bit pinned-oracle cases in
explicit FP32/FP16, plus12 CPU FP32/BF16 cases using seven actual official BF16
weights. Original serving stayed running with identical PID/config and HTTP200;
this was correctness-only, not a throughput test. Both new units are terminal.
Next independent work is MoE router/expert/shared-gate correctness, followed by
QSA/GDN/PLE integration. Native GR kernels/fusion and full generation remain
unimplemented. Do not repeat completed experiments to recover/copy raw files.

## 2026-10-25 03:17 MoE and QSA-indexer gates

`flash_next_moe` now provides floating512-way/top10 routing, selected expert
weighted accumulation, sigmoid-gated shared expert and bounded original fused3D
expert/core checkpoint reads. Missing sampled experts are errors, never pruned.
60 CPU/V100 oracle cases pass (48 synthetic,12 official-weight cases). Real
router/shared plus11 experts account for120.59MB of sampled original BF16 data,
**not a complete checkpoint/layer**. Positive-scaled seed-direction cases are
limited coverage. BF16 routing adds expert26 for some rows where FP32 does not;
actual top10 must not be replaced with a fixed first-token expert set/cache.
Details/failure recovery: `benchmarks/qwen38-flash-next-moe-v1/README.md`.

`flash_next_qsa` implements a bounded floating **indexer**, not main attention:
explicit raw unrotated key state, visible-index block pooling, partial RoPE,
ReLU/head-summed scores, top blocks plus tail and bool/additive selection masks.
112 CPU/V100 mask/raw-state cases pass (80 synthetic,32 actual-indexer-weight/
synthetic-state). The three real weights total3.28MB.4096/4100 synthetic past
keys test pruning, not model context/generation. Torch2.5's int64 scatter ABI
needed an explicit cast in candidate and an index-only pinned-oracle shim;
v1 failures/source are preserved separately. See
`benchmarks/qwen38-flash-next-qsa-v2/README.md`.

QSA contracts for native assembly:
- Pool consecutive **visible indices**, not fixed absolute four-token blocks;
  padding/holes can change grouping. FP32 mean casts before grouped RMS/RoPE.
- Indexer128 uses the main model's supplied64 rotary dimensions (main head256
  times0.25), not its own128*0.25. Full past/current positions are required.
- Budget2048 counts complete blocks; tail adds up to3 tokens (2049/2050 in
  fixtures). Keep raw indexer state distinct from main normalized/rotated K/V.
- Main q-projection interleaves query/gate per head before the split; do not
  split the whole projected feature vector into two halves.
- Sparse selection alone does not allow deleting all unselected historical K/V:
  later queries may select them. This is not a magic KV-capacity reduction.

Both successful new units are terminal; original productionPID/config/health
remained unchanged and no service was stopped this round. Local full suite995
pass/2583skip. Native full-model generation, native MoE/QSA kernels, expert
cache/EP, full main attention and GDN/PLE/HC integration are still absent.
Next independent gates: main QSA attention block and GDN (K16/V48, dims128,
conv4; SiLU convolution but **sigmoid output gate**, not the dense adapter's
assumed gate), then complete layer/reference/state assembly before speed work.

## 2026-10-25 03:58 GDN core continuation

`flash_next_delta` adds independently qualified floating recurrent/chunk64 delta
core and direct-weight sigmoid-gated RMS.128 same-mode CPU/V100 core cases plus
four gated-RMS cases match the pinned source exactly. All inputs/states here
are **synthetic**, including official K16→V48/D128 and batch8 geometry. This is
not the projection/convolution/GDN frontend, a native kernel or model TPS.

Chunk/recurrent arithmetic order differs: raw rounding counterexamples are in
`benchmarks/qwen38-flash-next-gdn-v1/`. No cross-mode bit-identity/quality claim.
FP32 state layout is `[B,HV,DK,DV]`, distinct from existing native layout; a
future transpose/integration must be separately checked. Rejected MTP suffixes
need a base snapshot plus accepted-prefix replay/per-token snapshots, not slicing
the final recurrent matrix. Production remained running unchanged/healthy.
Full eager QSA attention qualification is proceeding in a separate new capsule;
no completed old PLE/MoE/indexer experiment is being replayed.

### 04:05 main QSA attention gate completed

`flash_next_attention` now supplies a bounded floating **main attention block**:
indexer selection, Q/K/V, zero-centered Q/K RMS/shared partial RoPE, GQA,
per-head interleaved query/gate, sigmoid output gate and output projection.
Explicit proposed state separates raw indexer K from normalized/rotated main
K/V, with owned accepted-prefix clones and no automatic commit. Nine actual
original BF16 weights102.89MB, but activations/past are synthetic. New v2 CPU26
plus V10026 checks match pinned eager outputs/probabilities/state; eight remote
fixtures pass. Full local suite1011pass/2583skip. **Not native/full generation.**

V1 CUDA B8/P4100 FP32 hit its1.5GiB process allocator cap while materializing
GQA-expanded K. Failure/source/log are preserved. V2 adds pre-allocation
expanded-KV byte admission and separately checks B1/B2 long history and B8/P16;
**B8 long-history GPU support is not established**. Default state budget and
arithmetic stay unchanged; this is a reference guard, not a GQA optimization.

Pinned eager finite-min mask addition produces NaNs for fully masked rows;
20 of the52 new CPU/V100 cases preserve those rows with `equal_nan`. Do not
claim all-finite outputs or safe padded serving. An explicitly qualified SDPA/
native empty-row contract is still required. Both v1 failed/v2 successful units
are terminal, no production stop and serving PID/config/health unchanged.
Separate evidence in `benchmarks/qwen38-flash-next-attention-v1/` and `-v2/`.
Next: SDPA/empty rows, GDN projection/SiLU-conv/history/A-dt binding and actual-
weight block qualification, PLE conv/complete-layer/state assembly, then native
and format/cache performance. Old evidence and dense runtime remain untouched.

## 2026-10-25 05:04 full GDN block and explicit SDPA gates

`flash_next_gdn` now includes the complete **floating GDN block**: projections,
chronological raw SiLU conv4 state, K16->V48 expansion, FP32 A/dt log-decay,
beta, chunk64/recurrent core, direct-weight sigmoid RMS and output projection.
Actual9 original BF16 tensors115.92MB exercise bounded checkpoint binding.
CPU48+V10048 strict output/conv/FP32-state comparisons pass;48 actual-weight/
synthetic-history and48 fully synthetic cases.13 remote fixture tests pass.
This is not an owned native kernel, full decoder/model or quality/TPS run.

A retained v1 CPU counterexample found that cached-single must slice convolution
output **before** SiLU, matching source: applying activation to the extra output
then slicing changed SIMD rounding, amplified through RMS. V2 fixes ordering,
adds a regression, keeps exact tolerances and actual weights unchanged. See
`benchmarks/qwen38-flash-next-gdn-block-v1/` (local failure, no GPU launch) and
`-v2/` (completed gate). Current GDN state is explicit/private/no auto-commit;
padding zeros input but still advances/decays history, not an inactive-slot skip.

Main QSA now has an **explicit** `attention_backend="sdpa"` option, default
still eager.104 CPU/V100 pinned SDPA-MATH output/state cases pass; all finite,
40 fully masked query rows exactly zero. Raw bool/additive masks and cached
chunks use their explicit cache-aligned causality, not an additional upper-left
SDPA triangle. No probabilities returned for SDPA. Original upstream SDPA source
and runtime flags are pinned/reported; no eager-NaN oracle repair. See
`benchmarks/qwen38-flash-next-sdpa-v1/`. Fused/flash/efficient SDPA, paged cache,
MPS execution, engine hot-switching and performance are NOT tested/claimed.

Both new successful units terminal, production never stopped, PID/config and
health200 unchanged. Local full suite1028pass/2583skip. No native defaults,
compiler/driver policy, GitHub or old goal/queue changes. Next: mandatory PLE
projection/gates/dilated short-conv plus existing CPU row reader, then HC/full
layer and unified transactional per-request state. PLE belongs before attn-HC:
its conv kernel4/dilation3 needs9 past normalized-gated activations (cache slot1),
ngram history needs2 token IDs (slot2), independent of GDN conv/RNN slot0.
Signed-sqrt PLE gate and grouped zero-centered norms must match the source;
do not reuse GDN's conv4/direct-weight RMS or confuse PLE with speculation.

## 2026-10-25 06:02 PLE and full floating decoder-layer assembly

`flash_next_ple` implements key/query grouped zero-centered norms, signed-sqrt
similarity gate, shared value, normalized dilated SiLU conv4/dilation3/history9.
Bounded binding loads6 actual BF16 conditioning weights65.68MB, not the giant
table.88 exact CPU/V100 output/conv-state comparisons pass (48 actual-weight/
synthetic-input/history,40 fully synthetic). Original oracle embedding producer
is supplied rows, not a whole-table/token-history-slot2 validation. A separate
CPU bridge fixture joins existing coalesced prepared table reads/private EOS
proposals to this block, explicit device/dtype, without committing histories.
Official FP8 row decoding remains separately qualified; no full FP8 PLE binding
or quality claim. See `benchmarks/qwen38-flash-next-ple-v1/`.

`flash_next_layer` now assembles PLE BEFORE attn-HC, attention read/inject and
MoE read/inject, with explicit private `LayerState` snapshots and no auto-commit.
224 CPU/V100 evaluated steps match original decoder forward/components exactly,
including repeated independent base-prefill steps. **All layer weights/geometry
are small synthetic fixtures** (all4 toy experts/top2), not an original layer or
full model. Missing experts remain errors and failed proposals leave base state
untouched. Explicit SDPA-MATH QSA, not an eager/default switch. See
`benchmarks/qwen38-flash-next-layer-v1/` for harness/scope/raw evidence.

Both new units successful/terminal, production never stopped, PID/config/health
unchanged. Local full suite1041pass/2583skip. No new native/export/serving or
TPS/quality claim. Next embeddings/final HC mixer/head/text-core and unified
transactional request state, real token-hash/table bridge across complete layers.
Ngram EOS resets hash segment history, **not automatically PLE/GDN/KV conv/state**;
request EOS termination/reset must be an explicit coordinated lifecycle. Partial
speculative acceptance needs base-state replay, never recurrent-matrix slicing.

## 2026-10-25 07:50 continuous integration and owned V100 byte decoder

User requested continuous work rather than heartbeat polling at07:12; the
`flash-next` automation is PAUSED, this current session continued directly.
No old dense goals/queues/automations were resumed. Production stayed running
through every new check, unchanged PID/config/health200; no default model/route,
clock/power/driver policy, local Rust build, sub-agent or GitHub push change.

New text-v1/loader-v1/request-v2 gates connect embeddings -> full HC-vector
repeat -> all floating layers -> final mixer/head and private per-request
hash/conv/RNN/raw-indexer/main-KV/padding/prediction state. Masked PLE IDs become
EOS before hashing; EOS does NOT reset model histories. Partial acceptance
replays accepted prefixes and retains replay prediction logits, never recurrent
matrix cropping or mutable proposal-logit reuse. Fixtures include stale/foreign/
superseded/reset/failure isolation. These are **tiny synthetic** complete-checkpoint
and greedy-ID tests, not real-model text quality/generation qualification.

Strict original floating binding and a SEPARATE opt-in FP8-weight->floating
bridge now preflight all48 official layers. BF16131 headers/1658 tensors and
FP8(all newly collected)131 headers/152089 tensors pass exact-index/shape/dtype
metadata gates. Payloads absent, **no full original model loaded**.1067 static
text tensors9,895,397,120 bytes; original fused experts241,591,910,400 bytes vs
FP8 codes/scales120,810,700,800; PLE102,400,491,520 vs51,200,245,762. These are
file/parameter partitions, NOT peak/resident RAM/VRAM or model-speed estimates.
Main FP8 bridge decodes selected expert blocks, separate global-scale PLE rows;
no activation FP8/W8A8 numerical-equivalence claim. Vision/MTP remain ignored.

Actual official layer0/expert0 FP8 code/scales4,915,800 bytes were range-sampled,
all code values independently bit-decoded and compared with pinned SGLang
block dequantization. A generic retained-weight LRU supports explicit CPU/CUDA
keys/dtypes, miss coalescing, immutable owned storage, eviction/live leases and
CUDA fill events/record_stream. Cache budget counts ONLY retained weight storage,
not live callers/reservations/miss scratch or total-process OOM admission.
No automatic RAM spill, prefetch/placement scheduler, packed Q4/Q8 cache or
backend hot switch. `bind_fp8_expert_policy` is explicit setup-only, validates
ALL stores before rebinding, leaves core placement/dtype/table/serving unchanged.
CPU/native GPU toy full-text oracle gates pass; not whole-original-model native.

Owned `flash_next_fp8_decode.cu` now decodes E4M3FN bytes + BF16 block128
multipliers on SM70 using integer bit layout/FP32 product, with exact FP32/FP16/
BF16 storage outputs. **No FP8 hardware or BF16 GEMM required.** Explicit prebuilt
library only, remotely compiled with CUDA12.9; no import-time/local build.
Grouped error flag rejects NaN codes, invalid scales and overflow before return;
error synchronization means graph capture is NOT qualified. Generic matrix tails,
signed zero, BF16-subnormal scales and actual expert weights match reference bits.
Row-tiled v2 removes per-element64-bit division,16 registers vs30 in linear build;
30 decoder cases+4 standalone MLP cases pass (v1 had27+4). These are reusable
native OPERATORS inside a Python floating reference, **not a native full engine**.

Same-campaign standalone B1/B8 expert placement v3 (3 repeats,32 iterations,
4 warmups, identical FP16 weights/input/GEMM shapes) mean wall ms:

|policy|B1|B8|
|---|---:|---:|
|GPU resident reference|0.158|0.183|
|CPU predecoded file + H2D|5.836|5.277|
|CPU tiled FP8 decode + H2D|139.390|137.318|
|CPU vectorized FP8 decode + H2D|108.284|110.628|
|CPU decoded hot RAM cache + H2D|1.894|1.923|
|GPU decoded hot cache|0.210|0.223|
|Native CUDA byte decode (linear v1), uncached|2.365|2.288|
|Native-source GPU decoded hot cache|0.210|0.227|

One expert repeatedly reused, tiny warm files, no actual routing/cache hit-rate,
cold185GB streaming, quality or model TPS. Each cache holds9,830,400 bytes;
retained budget is not total VRAM. Direct-GPU fetch includes I/O/H2D/decode/
flag-sync in fetch wall time, not later no-op copy-event phase. Profiler runs
are separate, CUDA event spans include host gaps and are NOT achieved bandwidth.
V2 placement campaign hit180s timeout AFTER correctness; no final score survived,
failure retained. V3 uses bounded32 iterations/420s cap/incremental checkpoints,
not a rerun/overwrite of frozen v1/v2. See dedicated placement capsules.

Resident-code kernel-only warm stream intervals (3 repeats,200 launches):
linear->row-tiled gate/up21.835->14.292us; down21.826->15.889us. Host launch gaps
still included; no end-to-end model speedup inferred. Ptxas register counts are
NOT achieved occupancy. Counter follow-up is separate; see counter capsule for
actual availability/runtime-path recovery, never substitute theoretical figures.

Latest full CPU suite1096pass/2583skip. Old snapshots/batch-sweep unchanged.
Remaining: whole original checkpoint/payload acquisition and quality/generation,
owned attention/GDN/routing/GEMM integration, real multi-expert working-set cache
policy/offload/prefetch/VRAM admission, TP/PP/MTP and real serving performance.
No claim that this Flash-Next path is ready for XHarness or faster than llama.cpp.

07:56 counter recovery completed in a new capsule: existing absolute NCU path
and explicit evidenced runtime package paths allow authorized sudo probes,
without driver-policy/package changes. For actual640x2560 FP16 weight decode:
linear->row-tiled kernel23.008->16.000us; measured active-warps occupancy82.66%
->74.77%, SM-throughput71.67%->61.09%, DRAM-throughput17.64%->25.69% of peak
sustained elapsed. Fewer instructions/less division improves speed even though
occupancy decreases. This is separate single-launch hardware profiling, NOT
unprofiled throughput. Raw metrics/units/caveats in cuda-counters-v2. GPU0 free
5328MiB vs5452MiB at intake (124MiB delta); only original production compute PID
remains, no test/profiler processes. Cause unproven, no GPU/driver reset attempted.
All experiment units terminal; production same PID1888496/config/health200.

Safety boundary for follow-on work: the decoded expert LRU is qualified for eager
execution only. Do NOT capture hot-cache pointers in CUDA graphs: eviction has
no captured-graph lease/pinning contract. Native decoder rejects capture, but a
cache hit does not invoke it; event/record_stream eager lifetime safety is not
a graph-replay guarantee. Add explicit capture admission/pinned leases and test
before enabling graphs on this new path. Existing dense graph backend unchanged.

Final closure:28 frozen evidence capsules rehashed,0 mismatches; full CPU suite
1096pass/2583skip, lint/diff-check pass. All owned GPU tasks/profile descendants
terminal and unique SSH mux closed; production PID/config/health200 unchanged.
Heartbeat remains PAUSED per user's no-polling request. Own AC-only awake helper
expires08:00 automatically; no permanent power settings or unrelated assertions
changed. No full-model native readiness/completion claim is made.

## 2026-10-25 daytime continuation: explicit reference generation

The user explicitly requested further adaptation after the overnight closure.
This is a new bounded daytime correctness campaign, not a restart of the paused
heartbeat or old dense-model goals/queues. `flash-next` remains PAUSED.

`venturi flash-next generate-reference` now exposes a **raw-text, single-request
floating reference**. It composes the existing strict full-checkpoint loader,
local tokenizer.json (no network/model Python/chat-template execution), prepared
CPU PLE row reader, private request transactions, explicitly chunked prefill,
cached single-token decode, greedy finite-logit checks and emitted-EOS stopping.
Every invocation has fresh private state. EOS in the prompt still affects ngram
hash segments rather than terminating prefill/resetting decoder caches. The last
emitted token is not forwarded merely to return it. Zero output performs no
forward/row lookup. No sampling, streaming server, batching, MTP or graphs.

Static core loading now supports explicit `cpu` / indexed `cuda:N` and floating
execution dtype. It casts meta parameters BEFORE allocating directly on the
chosen device, copies individual original static weights and rejects nonfinite/
cast-overflow values. Source and converted static parameter byte budgets are
separate; a metadata-only plan never initializes CUDA or loads tensor payloads.
On pre-Ampere CUDA, BF16 GEMM is rejected; choose FP16/FP32 explicitly. Runtime
also rejects a static core larger than free CUDA memory, **not** a proof of total
state/scratch/expert-cache OOM admission. No automatic placement/spill/fallback,
TP/PP or production routing changes. Experts/PLE remain CPU file-backed; the
setup-only FP8 expert policy/cache/native byte decoder can be selected explicitly.

The new command is deliberately named `generate-reference`, NOT native
`generate` or XHarness support. It requires a COMPLETE local checkpoint, explicit
encoding/dtype/core budget, and makes no original-checkpoint quality/TPS claim.
For example, only after full original payloads and tokenizer are available:

```sh
PYTHONPATH=src python -m venturi flash-next generate-reference /complete/local/checkpoint \
  --weight-encoding fp8-weight-only --dtype float16 --device cuda:0 \
  --core-budget-mib 12000 --execution-core-budget-mib 12000 \
  --expert-backend cuda-byte-decode --decoder-library /explicit/prebuilt/sm70/library.so \
  --expert-cache-mib-per-layer 32 --max-sequence 256 --prefill-chunk-size 16 \
  --prompt "Hello" --max-new-tokens 16
```

Those example budgets are parameter/cache bounds, NOT a tested full-model memory
configuration; current production-free VRAM cannot fit the static core. Add
`--plan-only` for metadata-only admission (no tokenizer required). No header-only
artifact or uncensored checkpoint is silently accepted as the official payload.
Request timing separates prefill/decode CPU and wall time, TTFT and decode-token
count (first token belongs to prefill); excludes loading/tokenization/preparation.
Each prefill schedule is independently checked against the SAME pinned upstream
schedule. GDN chunk/recurrent rounding can differ; no cross-chunk bit-identity
claim. SDPA-MATH is selected explicitly, not an unqualified fused backend.

Local tests:1131 passed/2583 skipped. New focused campaign:91 pass. An initial
oracle harness mismatch is preserved: synthetic TensorExpertStore weights are
not module parameters, so module.to() did not round them to serialized BF16
before promotion to FP32. The independent oracle now explicitly binds serialized
source precision first; no production arithmetic or tolerances were changed.
CPU twelve trajectories (floating/FP8 × FP32/BF16 × chunk1/2/16), six emitted tokens
each, match upstream every-step logits and final caches exactly. These are tiny
synthetic complete fixtures with all four experts, not model quality/TPS.

Daytime gate completion: V100 FP32/FP16 twelve new generation trajectories,
96 forward steps, all logits and final caches EXACT vs pinned independent
upstream, including direct target loading and native-byte-decoded FP8 experts.
Production never stopped, same PID/config/health200. Initial remote helper lacked
`tokenizers`;41 tests passed/3 skipped. A pinned0.22.2 Linux binary wheel was
unpacked ONLY into new artifact-local dependencies (no global/production package
changes), closing the three CPU CLI tests. Subsequent CPU-only CLI-v2 adds safe
`generation_config.json` EOS-list handling and passes51 tests; no GPU campaign
was replayed. The generation numeric loop is unchanged by this metadata helper.

Official generation EOS includes248046 and248044, while ngram segment EOS is
248044. CLI default now reads only bounded local EOS metadata, preserves the
ngram setting, supports explicit/ignore overrides, and does NOT silently apply
the official sampling parameters or execute chat templates. Official pinned
config/index/tokenizer/generation metadata/LICENSE are locally staged and
verified by LFS SHA256 or pinned Git blob ID; a Chinese tokenizer roundtrip
passes. Tokenizer roundtrip is NOT generated model text or a quality check.

Payload acquisition remains incomplete: a large HTTPS shard response was
truncated at32,398,530 bytes despite declaring the full1,040,155,912-byte range.
The partial was not published; an interrupted retry and new bounded4MiB ranges
continued safely to72,473,794 bytes. Six exact-length ranges succeeded; observed
local HTTPS reads~3–4s per4MiB vs SSH writes~14–45s here. These are transfer
observations, not inference benchmarks. Full official SHA256 remains pending,
**zero complete verified weight shards**, no whole original checkpoint load.
V100 direct outbound HF IPv4/IPv6 failed; two authorized same-host jump routes
are reachable but did not establish a fast bulk transfer. No TLS/host-key/DNS
system policy changes. This is unfinished acquisition, not a claimed hard block
or an excuse to use the unrelated uncensored model as official baseline.

Daytime closure: final local suite1138pass/2583skip; remote CPU CLI/loader51pass,
all V100 generation qualification terminal.71 repository frozen capsules checked,
zero hash mismatches (new generation capsule464 frozen files). No owned GPU/
staging processes remain, own SSH mux closed; production same PID/config/health200
and same free VRAM5328/6588MiB as daytime intake. No unmanaged background download
or model switch; first shard partial remains resumable, not authenticated/published.

## 2026-10-25 full official payload / real-generation campaign (in progress)

New user request: finish complete original weights, run real text generation,
measure TPS and establish actual capacity. Same two-V100 host retained; do not
substitute L40 uncensored weights or claim existing dense27B results apply.
Live state is under `operations/flash-next-full-model-20261025-v1/`.

Official ModelScope repository is reachable directly from V100. All131 LFS
size/SHA256 entries match the original fixed HF FP8 revision exactly; source file
revisions pinned individually (weight revision47b13b2ed4491728d4509761a557d6e75bc96c9c).
This changes TRANSPORT only, not the checkpoint/authority/precision. Original HF
LICENSE is retained separately because the mirror's license bytes differ. The
five config/index/tokenizer assets match original HF SHA256 too. Full campaign:
137 files185,553,630,797 bytes. Exact HTTPS ranges, bounded memory/CPU/disk/time,
resumable private .parts, exclusive writer lock, complete SHA256 before atomic
no-overwrite publication. Local17 downloader protocol/admission tests pass.
No production stop is needed for downloads. Six-hour upper bound after correcting
a too-short initial wall budget from measured~11–12MB/s, not a hard-failure claim.
Authorized same-host jump/LAN paths were checked;32MiB LAN transfer3.474s did not
establish a faster bulk route. No SSH/login-key/TLS/network/driver policy changed.

Capacity plan before actual runtime: full172.8GiB files fit~2.16TB free disk, NOT
two32GiB GPUs. Static core9.216GiB, proposed256MiB decoded expert cache/layer=12GiB
retained budget; file-backed112.514GiB expert codes/scales and47.684GiB PLE. Host
available~90GiB cannot keep all encoded banks/tables resident simultaneously.
B1/context256 state lower bounds in capacity-plan.json; scratch/live leases/
allocator/driver overhead and base/proposed overlap excluded. This is a placement
proposal, NOT measured OOM-free capacity or usable262k-context proof.

New generic independently executed original48-layer-capable text/cache/hash
oracle is prepared. Static parameters alias already validated loader weights;
this is independent ARITHMETIC/state/hash checking, not an independent binding.
Huge resident expert/PLE arrays replaced ONLY at weight-indexing/I/O boundaries;
original routing/GEMMs/hash/decoder bodies execute. CPU FP8 expert dequant vs owned
SM70 native dequant; no W8A8 activation equivalence. Tiny CPU cases pass; remote
preflight61 tests +6 V100 FP32/FP16 same-mode state/logit cases pass. Full local
suite1157pass/2583skip. No real checkpoint generation/TPS completed at this note.

One inotify-driven continuation for THIS ongoing request (not an app heartbeat)
waits for authenticated complete payloads, then runs a guarded real-model lease:
save original identity/config, drain requests, independent restore timer plus
finally/ExecStopPost. The runtime stores raw Docker inspection only remotely
mode0600 and must NOT transfer it into public evidence. llama.cpp /slots retains
id_task after completion: explicit is_processing is the authoritative active flag,
verified by four local sanitized classifier tests; never confuse stale task IDs
with a running request. If originally stopped, do not resurrect that service.

Planned actual benchmark: one V100/FP16 floating GEMMs with official FP8 weight
decode; all48 text layers/all512 experts available, no vision/MTP/native full
runtime/TP/PP claim. Exact original-source logits/all-state gate first, then
actual ChatML-literal text smoke (no Jinja code), then3 alternating uncached vs
256MiB/layer cache pairs, same prompt/chunk16/output32/concurrency1/explicit
ignore-EOS. Loading/reader preparation/warmup excluded and recorded. Report CPU,
TTFT, prefill/decode/request wall, emitted-token counts, peak process RSS/Torch
VRAM and sampled GPU busy/power/clocks/temperature. Nsight timeline/counters are
separate unscored runs, with bounded own process groups and restore protection.
Do not report these planned measurements as achieved TPS or full native support.

Full-download recovery note: an empty private `.part` from a prior bounded stop
exposed an exclusive-create/existence bug, NOT a network/data-hash failure. Fixed
and regression-tested in a separate download-overlay-v4 (18 local/Linux tests).
All verified files/private partials preserved; latest downloader v4 and inotify
consumer v3, same frozen numeric engine/preflight source. No production pause.

Download v4 preserved37 fully authenticated files57,728,547,538 bytes, but a
single read timeout exhausted per-range attempts and canceled the remaining
independent queue. V5 keeps other files progressing and retries failed files
in at most2 later rounds with30/60s backoff under the SAME6h wall deadline.
8workers/16MiB ranges/60s read timeout/6 range attempts. Integrity mismatches
remain fail-closed, private, not erased or automatically re-downloaded.21 local
and Linux protocol/queue tests pass, including recovered timeout, bounded
persistent failure and no retry/publication of SHA mismatch. Inotify consumer
v4 rearmed; original numeric preflight/source unchanged. Production never
paused so far, same PID/config/health200; no real full-model TPS measured yet.

## 2026-10-25 full official checkpoint / real-model closure

Full original FP8 payloads COMPLETE:131 weight shards+6 assets,137 files,
185,553,630,797 bytes; final full SHA256 rehash passed. Exact original small
non-Range metadata imported under exclusive lock from pinned HF bytes, no
source/hash weakening. See `benchmarks/qwen38-flash-next-full-v1/README.md`.

Real48-layer text model loads on ONE32GiB V100 with static core+decoded expert
LRU and CPU-file-backed experts/PLE. Two original-weight source-arithmetic steps
have exact logits/all states; actual32-token Chinese generation passes. Static
weights alias original-source oracle: no independent binding/BF16/W8A8 quality
claim.32 tokens end during thinking, not a completed final answer/quality eval.

Three paired unprofiled repetitions, B1/13 input/32 output/chunk16: no-cache
0.243798 decode TPS vs256MiB/layer cache0.505454,2.073x. All six IDs exact. Cached
request TPS0.381307, prefill22.591s, whole-request83.922s, CPU83.957s. Physical
GPU23045MiB (~22.505GiB), Torch peak allocated21.337GiB/reserved22.123GiB. Host
peak RSS3.176GiB excludes OS page cache; zero scored major-fault/read-byte deltas
means WARM page-cache, not cold-SSD TPS. Original172.810GiB cannot be fully VRAM-
resident on two V100s. ShortB1 test does not establish batch8/long-context or even
an executed256-token context. Native full runtime/TP/PP/MTP/XHarness still absent.

Unscored timeline33.829s:115978 launches, kernel+copy+memset interval union10.934%
(~89% no recorded GPU activity, NOT occupancy).39710 memcpy APIs/31648 stream
sync APIs. Selected native FP8 decoder NCU occupancy77.01%, DRAM25.60%,16regs,
16.128us. These are component counters, NOT whole-model occupancy/bandwidth.
Next targets: host/expert preparation, transfer/sync batching, buffer/mapping
reuse and grouped/native MoE. Not a claim the native decoder kernel is the main
end-to-end limiter; CPU-stack attribution remains unmeasured.

Two interrupted score campaigns retained:10-minute production watchdog and
30-minute wash producer independently auto-started old serving. Final lease
respects existing wash flock without killing active work and only pauses exact
model-watchdog cron entry with private original+merge-safe recovery. Independent
restore guard/finally/ExecStopPost; final original container config unchanged,
running/health200, original crontab byte-exact, own jobs/profiles terminal. No
raw Docker/crontab private state transferred into public evidence, no production
parameters/driver/routes changed. Heartbeat remains paused. No local Rust,
sub-agents, old-goal/queue resumption or automatic GitHub push.

## 2026-10-25 GPU-hot / parallel CPU-cold experiment (not qualified)

User requested CPU/GPU hybrid experts, correctness first, then matched TPS.
Implemented opt-in fixed-hot banks, one8-worker CPU pool, packed cold output
transfer, original-order reduction, bounded CPU decoded LRU (48GiB total max;
40.054GiB retained in final direct-half diagnostic), setup/close and fail-closed
worker/cache lifecycle. Defaults unchanged; no native quantized CPU kernel claim.

CPU fixtures and54 actual-weight component cases pass; same20 distinct expert
pairs' decoded weights exact CPU/native GPU in each trial. Component activations
synthetic; timers coexist with production and are diagnostic, NOT model speedups.
Full48-layer matched fixed-hot/all-GPU control exact at3 cached steps. Both CPU
variants FAIL strict predeclared full-model gates: stagedFP32 step1 max-logit
0.1953125 vs0.05 limit; directCPUHalf0.19140625. Direct-half next step also exceeds
0.005 state-relative-error gate at layer13. Trace shows selected expert SET
changes at zero-based layer47/step1 and layer12/step2, consistent with rounding
plus sensitive router boundaries; complete operator-level cause still unproven.

Direct-half32-token diagnostic greedy IDs exactly match GPU, but that does NOT
replace failed full logit/state gates. No three-repeat TPS campaign or new
profiler/counter score was launched; original0.505 TPS is still ONLY the previous
GPU-only reference. Do not silently relax thresholds or call hybrid a qualified
production replacement. See new `qwen38-flash-next-hybrid-v1` capsule for source,
all failures/route data, resource cost and three counterexample hashes.

Original serving running/health200/config unchanged, original crontab byte-exact,
wash lock released on supervisor exit; all own experiments/restore timers terminal.
Private Docker/crontab snapshots stay remote. No default backend, local Rust,
subagent, automatic GitHub push or old automation/goal/queue resumption.

## 2026-10-25 expert Q4 trial: qualified canonical fallback, rejected packed GEMV

Opt-in `flash_next_q4` now provides explicit K64/FP16-scale symmetric RTN Q4,
SHA-verified immutable lazy expert archives and a bounded GPU packed LRU. Source
is **official FP8 -> FP16 -> Q4**, not a complete BF16-derived/GPTQ/AWQ export.
Router/shared/core/PLE/state precision and defaults stay unchanged. No retained
40GiB decoded CPU expert bank; host file cache remains real RAM consumption.

20 actual expert pairs pass40 CPU/GPU decoded-weight-bit checks,120 native
packed-linear/FP32-dot checks and60 synthetic-input expert checks. Nevertheless
native packed GEMV FAILS all32 full-model logit/state checks (max logit2.2734,
1745 failed state-tensor checks), despite32 generated IDs matching FP8. That arm
remains rejected and received NO model TPS score. Counterexample/hash retained.

A **different canonical path**, packed Q4 cache+transient GPU decode+Torch FP16
GEMMs, passes independent CPU-vs-CUDA nibble-decode three48-layer cached steps
with ALL logits/states BIT-EXACT. Same-Q4 32-step short teacher quality screen:
mean KL0.008666,max0.106389,top1 agreement32/32. Actual32-token generation matches
FP8 sample (ends during thinking); NOT held-out quality/perplexity/BF16/W8A8 proof.

Three alternating paired unprofiled repeats, ONE V10032GB,B1/13 input/32 output,
31 decode transitions/chunk16/reserve256, same256MiB/layer retained GPU budget,
32-token same-prompt warmups: FP8 decoded LRU0.444607 decodeTPS -> Q4 canonical
1.462998 (**3.2905x**); requestTPS0.333885->1.136604,prefill26.116->6.964s.
Repeat ranges0.4208–0.4714 vs1.3846–1.6294 retained. Warm scored read/major-fault
bytes zero. Different precision/archive/cache coverage/host preparation: NOT a
pure INT4 GEMM speedup. No scored conversion; loading/warmup outside and reported.
Old frozen0.505-TPS experiment used4-token warmups, not this new baseline.

Actual Torch peaks21.339/21.210GiB, reserved26.811GiB; physical high-water27845MiB
(~27.19GiB), so do not claim24GB fit or21.2GiB total VRAM. RSS cumulative peak
5.317GiB excludes unmapped OS page cache, not a complete host-RAM requirement.
Cache entries max102 vs27/layer. GPU busy11.36->19.10%;clocks1230/877MHz unchanged.
Separate unscored timeline13+3 steps:105888 kernels,5974 copies,~80.03% no GPU
activity (NOT occupancy); single Q4 decoder NCU84.25% achieved occupancy and
22.4% DRAM (~195.85GB/s read+write) are NOT whole-model counters. Host dispatch,
sync/transfer batching and grouped/fused MoE remain opportunities.

Full-expert capacity estimate59.765625GiB vs FP8 112.513733GiB; add9.2158GiB core
and47.6839GiB CPU PLE. Still not fully resident on two32GB GPUs. Actual lazy
archive5974/24576 expert pairs/~14.528GiB encoded, NOT a full Q4 export; retain
original source for unseen experts and do not advertise warm repeated-prompt
TPS as cold first-use latency. Need offline conversion and broader quality,
then native/grouped work; no CPU-cold packed arithmetic/batch8/long-context/
TP/PP/MTP/XHarness/native-full-runtime support or default format switch claimed.

See `benchmarks/qwen38-flash-next-q4-v1/README.md` for raw paired scores, rejected
kernel, canonical exact checks, source composition, separate profiles and actual
memory scope. Final local1214pass/2583skip, remote61focusedpass. Production
restored running/health200/same config and byte-exact crontab; own units/profiles
terminal, restore timers inactive. No local Rust/CUDA compile, subagents, old
queue/goal/heartbeat resumption or automatic GitHub push; prior73 frozen capsules
unchanged. Private Docker/crontab backups stay remote.

## 2026-10-26 complete Q4 text bundle and held-out evaluation

Full text-only export is now complete:24,576 routed experts,125,295,494,140B
(116.6905GiB) mixed storage including unchanged9.2158GiB core/47.6839GiB PLE.
Official FP8->FP16->RTN Q4, not BF16-derived/HF/GGUF/all-Q4. Mandatory banks,
source-independent fixture reload and real3cached full-model logits/states exact;
no inference conversion fallback. CPU export34.4min, all payloads twicehashed,
original production unchanged during export. Actual runtime still canonical
GPU-decode+Torch FP16 linear, NOT prior rejected packed GEMV or native serving.

Three paired freshprocess+verifiedOSdata-cold cohorts,4distinctprompts/rotation,
B1/32outputs31decode/chunk16/context512,256MiB/layer cache/no promptwarmup:
FP8weight-decoded.420592 -> Q4.818034 weighteddecodeTPS (~1.945x), Q4case range
.50018–1.31793. MedianrequestTTFT48.288->28.679s. Different protocol fromold
samepromptwarm1.463TPS; don't conceal coldIO andwithincohortcachewarming.
ScoredphysicalGPUpeak23051/23335MiB, processcumulativeRSS3.10/7.17GiB excludes
OSfilecache, not minimumRAM or24GBqualification. Fullcold/processstartup and
noevictfreshprocess times remain separate in frozen newexportcapsule.

Heldout256realnexttargets PPL11.297724->11.420776 (+1.089%),meanKL.027043,
top1agreement90.234%;8closed7/8both (Yes-vs-lowercaseyes strictfailure retained),
2openrawanswers/EOScomplete; bothEnglishanswers violate2-sentenceformat.
Smallfixedscreengates pass, not publicbenchmark/BF16/W8A8/losslessqualityproof.
Separateprofile28.006srange/113094kernels/14406copies,94.344%no recordedGPUactivity
(NOToccupancy). SelectedQ4decode84.66%achievedoccupancy/22.49%DRAM/~197.07GB/s,
NOTwholemodelcounters. Host dispatch/copies/groupedMoE remain relevant.
See frozen `benchmarks/qwen38-flash-next-q4-export-v1/README.md` (979hashedfiles),
all74oldercapsules unchanged. All6exportevaluationleases originallyrestored
production/health/config/cron before subsequent nightoptimization campaign.

## 2026-10-26 graph dispatch experiment — not qualified, driver recovery failure

Opt-in coalesced expert-pairH2D, B1one-route-download andshared routedMoEgraph
use existing Q4decode+canonicalFP16Torchlinear (notpackedGEMV); no defaults/API
changed. V1graph rejected capturedependency onexternalprepared event, preserved;
v2ownedworkspaceskipsexternalwait withconstruction/replay/close ordering.
CPU48focusedpass; actual-weightSM70componentgraph/coalesced18exactchecks each.
Fullgraph3cached+1prefill+2teacherchecks exact, then abortedCUDAunknownerror.
It has NOT passed the complete preregistered gate and received NO modelTPSscore.

At07:40:27 hostkernelreportsGPU1Xid31 then BOTHGPUXid154 `Node Reboot Required`.
FreshhostCUDAinitialization alsofails; NVMLavailable/ECC0 isnotCUDAhealth.
Cause notestablished; logpid2051/modprobe isnot enoughforattribution. Original
production container/config unchanged butrestartloop/healthunavailable despite
boundedfinally/ExecStopPost restoration. Ownfollow-onCPUcontrollersstopped;
no furtherperformance tests admitted. No reboot/reset/driverpermissionchange
performed onsharedhost. Recoveryfailure communicated; exactfinalrestore/cron/
unitstatewillbe audited separately. Goal5TPSanddualGPUtestsremainUNMET.

## 2026-10-26 post-repair closure — coalesced full correctness passes, no new TPS

User repaired the host-wide CUDA fault. Same two V100 UUIDs/driver580.178.04:
both RecoveryActionNone and fresh CUDA health pass; original production health200.
A fresh **coalesced NO-GRAPH** lease reused immutable v2 source and canonical
Q4decode/FP16Torchlinear. All **12** full48-layer checkpoints have bit-exact
logits and every state tensor:3cached+1prefill+8teacher. No gate relaxation.
This qualifies the selected histories, not arbitrary-model/quality/serving proof.
Graph v2 remains incompletely qualified (6exact checkpoints then CUDAerror); do
not enable it or infer a Graph speedup. Fault root cause remains unestablished.

Full child admitted07:58:24 before the08:00 nightly cutoff; finished08:01:33.
Restoration08:01:52, final08:02:37 audit: original running container/configSHA,
health200, original crontab byte-exact, bothGPU RecoveryActionNone and no active
owned units/timers. No new experiment launched after cutoff; closure only.
Paired performance plan cancelled before admission. **No new TPS measured**;
previous frozen canonical Q4 cold-datafile score **.818034 weighted decodeTPS**
remains the latest comparable result, NOT the optimized path's score.5TPS goal
and dualGPU performance remain unmet. App goal paused, no automatic resumption.
New evidence/source/failures and final restoration proof are frozen separately
in `benchmarks/qwen38-flash-next-q4-dispatch-v1/README.md`; all75 prior capsules
re-audited unchanged. Final local CPU suite1234pass/2584skip predates cutoff.
