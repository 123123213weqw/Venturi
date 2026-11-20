# V100 direct-primary-output experiment: correct, no speed gain, rolled back

## Decision and retained engine

**Do not promote direct full-N-pitched primary TP output.** The byte-correct
prototype removed the intended staging copies but did not improve serving
throughput. Production native source is restored exactly to the preceding,
qualified guarded-T8 implementation. The rejected source/tests and library are
retained for future fused-projection work; they are not enabled in generation.

The retained three-repeat non-spec HTTP TPS is **19.41 / 28.58 / 35.57 / 43.18**
for 1/2/4/8 requests. Retained single-request MTP1 post-first decoding is
**30.42 TPS**. These are explicitly reused from `v100-q8-wmma`, not new speed
measurements of the restored source. That capsule's earlier T8 improvement is
preserved, not attributed to this unsuccessful experiment.

The fresh competitor depth study finds **49.80 reported post-first TPS** at
MTP depth 2, versus Venturi's retained 30.42 MTP1 TPS. External superiority is
still absent. No goal-completion claim, commit, push or local Rust/CUDA build.

## Hypothesis and correctness

The primary integer projection previously produced compact `[T,N_first]`
scratch, then copied it into its portion of `[T,N_full]`. The prototype adds
an internal pitched-output C export and compile-time store specialization,
keeping the old contiguous C ABI, arithmetic body, group-32 quantizer, reduction
order, peer compact output, host transport and both producer/completion fences.
It validates pitch >= shard width and `(T-1)*pitch+N_first < INT32_MAX` before
launch. Only explicit residual-profile TP with T>1 uses it in the experimental
model; original FP16 and unsharded paths remain unchanged. Scratch allocation
is retained, so no memory-admission/VRAM reduction is claimed.

Before speed measurements:

- **166 operator tests passed**: 74 new pitched-store numeric/tail/sentinel,
  minimum-pitch, invalid-geometry/pointer, graph/workspace-reuse/native-only
  tests, plus 20 pinned old/new tile cases and 72 existing DP4A tests.
  Both activation pass counts and T1/2/3/8/23/32 cover tiny, cached and guarded
  large shards, including odd widths/K tails. FP64 gates and exact old/new
  output/workspace bytes pass; peer columns remain untouched.
- **130 passed / 125 skipped** in default native regression selection.
- Both trained `fp16` and `q8-residual` profiles pass 98 complete-vocabulary
  vectors each, four pinned original greedy trajectories, scalar final private
  caches, eight heterogeneous batched requests and selected-reset continuation
  checks byte-exact against the qualified T8 binary. Native generation forbids
  Python/Torch math.
- **Fresh four-case trained MTP target committed-state/output checks passed**,
  including natural acceptance/replay and EOS, with no forced proposals.
  Only the unchanged draft's independent represented-weight CPU block reference
  is reused, with a pinned prior-report hash. This is not a fresh independent
  draft reference or original-BF16 broad corpus-quality qualification.
- Prototype full V100 suite: **1,280 passed / 440 skipped**. Prototype local:
  **761 passed / 959 skipped**. After restoring production source, local:
  **761 passed / 885 skipped**. The retained native library/source exactly
  matches the prior **1,206 passed / 440 skipped** V100 qualification; that
  production GPU suite is reused, not rerun or confused with prototype tests.

Fixed independent CPU thresholds remain top-1, relative L2 <= .02, centered
cosine >= .999 and KL <= .002, reused through exact old/new bytes. No threshold
relaxation or precision substitution was used.

Production library SHA256:
`ccf03d8caa89ced089b80cd5b8eb49cb494e53baf855c78c99e9c8f5ed9627c5`.
Rejected prototype SHA256:
`ccbe237f07f7fe8867be61ed37915900988544e91f98f59129205fa53d4f3eb7`,
retained at `/home/data/wangyue/artifacts/venturi-strided-fixtures-v1/build/000467e9f559906b/libventuri_blocks.so`.
The initial fixture
failure was test code, not compilation; the same compiled native library was
reused only after all seven native-source hashes matched.

## Matched speed screen (one repeat, no promotion)

Same two V100-PCIE-32GB devices, logical `1,0`, pinned-host TP, original-source
Q8 artifact, explicit residual arithmetic, FP16 KV, 23 formatted prompt IDs,
32 fixed greedy outputs, ignore EOS for this case, no prompt cache reuse, eight
slots, capacity 256 per slot, projection rows 32, and 25 ms HTTP cohort admission.
Warmups precede measured waves. Outputs and global cohort wall define HTTP TPS;
concurrent request durations are not summed. Both cohorts remain unsplit.

| Requests | Control HTTP TPS | Prototype HTTP TPS | Change | Wall seconds old -> new | CPU core-seconds old -> new | Mean client completion seconds old -> new |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19.403 | 19.238 | -0.85% | 1.64919 -> 1.66334 | 1.62 -> 1.64 | 1.64840 -> 1.66258 |
| 8 | 43.186 | 43.188 | +0.005% | 5.92785 -> 5.92757 | 5.93 -> 5.93 | 5.92409 -> 5.92369 |

Native C1 post-first TPS is 23.102 -> 23.045; native TTFT .26668 -> .26761 s.
C8 native TTFT is 2.13777 -> 2.14007 s. Native TTFT excludes HTTP/prior
queue/reset work and includes the whole batch prefill; it is not client
streaming TTFT. One screen cannot establish a small regression's significance,
but provides no useful improvement. Promotion was rejected; **no 1/2/4/8
three-repeat candidate campaign or fresh unprofiled candidate MTP speed claim**
is made. The three-repeat retained matrix is separately labelled as reused.

## Why deleting copies did not make the engine faster

Separate instrumented **23-prompt/8-output** traces, not throughput results:
non-spec control traces are reused hash-pinned from the prior capsule; candidate
traces and both MTP1 control/candidate traces are fresh.

C8 primary D2D 2D-copy API calls fall **5,208 -> 0**. Total D2D device copies
fall 7,040 -> 1,832 and bytes 961,331,200 -> 9,584,640; summed device D2D time
falls 16.89 -> 2.52 ms. C1 removes 400 copies; MTP1 removes 1,603 such calls.
The intended store path really executes; no Tensor Core model dispatch occurs.

However C8 dot-kernel sum stays **3,239.06 -> 3,237.71 ms**; kernel count remains
37,306; GDN 492.04 -> 492.18 ms; activity gaps 214.40 -> 215.71 ms; instrumented
native host span 3,053.37 -> 3,056.43 ms. H2D/D2H counts/bytes are unchanged
(about 1.990 GB each direction). C1 span is 605.78 -> 605.72 ms and MTP1
517.83 -> 519.01 ms, also essentially unchanged. Device kernel sums can overlap
across GPUs and are not elapsed time or utilization.

CPU API sums illustrate the trap: C8 `cudaMemcpy2DAsync` falls 523.52 -> 26.42 ms,
but stream-wait-event rises 164.88 -> 470.34 ms, stream-synchronize
978.62 -> 1,125.08 ms and launch API 1,168.30 -> 1,249.28 ms. This is consistent
with waiting moving to other calls rather than eliminating critical GPU work;
it is not proof that each API time difference is independently causal.
CPU API durations can include GPU waits/nesting, and must not be added as an
elapsed-time savings forecast. See `api-wait-audit.json` and copy attribution.

Unprofiled C8 primary/peer GPU busy 83.83/59.68 -> 83.26/59.68%; memory-controller
busy 22.61/19.93 -> 22.60/19.93% (**not GB/s**); power 132.08/101.10 ->
131.29/100.66 W; temperature 39.93/37.96 -> 40.44/38.00 C. SM/memory clocks
remain 1,380/877 MHz; device-used memory remains 20,235/17,227 MiB, including
unrelated resident allocations. No clock or policy changes.

Resource queries find identical register/theoretical limits for contiguous and
pitched residual kernels: T1 64 registers/50%, T2 62/50%, T4 78/37.5%, T8
124/25%; block 128 threads, no shared memory or spill. These are **theoretical**,
not achieved occupancy. Achieved occupancy, compute/DRAM throughput and cache/
stall counters remain unavailable under `RmProfilingAdminOnly: 1`; no repeated
counter campaign or privilege change.

## Stronger external MTP control: depths 1–4

Pinned llama binary SHA256:
`e5b1b5782cd5b4acec5316ac5bd40123e0b0c9826643ae4148b0ec6e359f5fa2`.
Same original-source GGUF, Q8_0, FP16 KV, tensor split 1,1, eight slots/total
context 2,048, batch/ubatch 256, four CPU threads, backend sampling and greedy
fixed 23/32 IDs. Depth 1/non-spec controls are reused from the immediately
preceding three-repeat campaign; depths 2/3/4 each have **three fresh unprofiled
repeats**, with matching IDs and no failures. No native-MTP2 implementation is
implied by these competitor measurements.

| MTP max proposals | Complete-request HTTP TPS | Reported post-first TPS | Accepted / proposed tokens | CPU core-seconds | Mean client completion seconds | Peak device-used MiB primary/peer |
|---:|---:|---:|---:|---:|---:|---:|
| 1 (reused) | 29.117 | 46.681 | 14/16 (87.5%) | 1.100 | 1.09788 | 18,373/18,835 |
| 2 | **30.173** | **49.803** | 17/26 (65.4%) | 1.060 | 1.05949 | 18,997/19,459 |
| 3 | 27.623 | 43.132 | 17/38 (44.7%) | 1.157 | 1.15744 | 19,619/20,081 |
| 4 | 25.227 | 37.367 | 17/49 (34.7%) | 1.270 | 1.26756 | 20,243/20,705 |

HTTP is 32 / mean whole-request wall; reported decode is **31 / mean
`predicted_ms`**, not HTTP or TTFT. Accepted/proposed values are per repeated
case, not totals multiplied by three, not acceptance-per-round probabilities.
Depth 2 is best **among the tested settings**, not globally optimal across all
contexts/configurations. Larger depth is not automatically faster. Earlier
failed parallel speculative configurations remain preserved and were not rerun
unchanged, nor counted as zeros/native wins. A complete best-supported parallel
speculative comparison is still outstanding.

## Source, scope and next work

Checkpoint provenance and representation differences are unchanged and pinned
through `prior-reference-pins.json`: official revision
`fc05daec18b0a78c049392ed2e771dde82bdf654`, target ID
`3d635b87db004f42aefd6730e372e530` with previously verified SHA256
`b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8`;
MTP SHA256 `a1a18b5f828d9b1cc1028fdc8463564aa4ffdbfe527311e1c0baa48cccfc1641`;
GGUF SHA256 `eb07277161915232e227a38608fc3ea80c0c17dc1c1583a9f3c5671edb6b5513`.
No fresh payload scan is claimed. Native group-32 Q8/F16 scales, two A8
high/residual passes with FP32 scales/reduction, small-parameter handling and
GGUF/Q8_1 arithmetic are not identical. Native original `fp16` remains default;
MTP draft stays original FP16 arithmetic. No broad BF16/corpus-quality claim.

Next priority is actual projection/GEMV kernel work, which still dominates
summed device time, together with a correctly transactional native two-proposal
MTP experiment. Memory-layout/latency/bandwidth hypotheses need measurements;
unavailable bandwidth counters do not prove a bandwidth diagnosis. Any changed
reduction/arithmetic needs separate quality qualification. Multi-proposal MTP
needs independently correct draft recurrence and target prefix commit/rollback;
competitor acceptance does not predict native acceptance. Neither is already
implemented or promised to close the speed gap. Do not spend further promotion
runs on this unchanged rejected-copy hypothesis.

Raw failed test/launcher/prerequisite attempts are preserved with honest missing
exit-code fields (`preserved-failures.json`). All owned workers are terminal,
both temporary ports closed, unrelated GPU allocations restored to idle
3,651/3,189 MiB. Large traces/library remain remote with retention hashes.
`metrics.json`, commands, source resolver, restore proof and manifests distinguish
executed prototype code, restored production code, reused controls and fresh
measurements. Prior 22 frozen capsules are verified unchanged.
