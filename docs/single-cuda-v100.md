# Single-request SM70 CUDA work (2026-10-29)

## Scope and bottleneck selection

This is Venturi-native execution on the same two WZU V100-PCIE-32GB GPUs,
not an external llama.cpp server benchmark. The source is the official-source
Qwen3.8-27B Q8 artifact (SHA256
`41f99992b679aef7df2b6c671b76b09820730846c2c95a52406e7004c72f58f0`),
not the production Uncensored GGUF. Precision stays group32 Q8/F16 weight scales,
two residual A8 activation passes, F16 activation/KV boundaries and FP32 GDN
state/accumulation. No MTP, graph, shorter context or different checkpoint is
used to manufacture a gain.

The first separately captured native C1 trace has four NVTX iteration ranges:
one 16-token prefill, followed by three single-token decode steps. Decode-only
attribution explicitly excludes the first range. Summed kernel time across
both devices is not elapsed time or occupancy:

- `vqf::global_dot<1,2,4>`: about 51.2% of decode kernel time.
- `vct::partial<1,4,false,2>`: about 25.1%.
- Compact GDN: about 3.7%, approximately 16 microseconds per rank-local call.
- A/B FP32 projections and RMSNorm remain additional small launch/latency costs.

Thus Q8 projections remain the main arithmetic path to investigate. Two
selected actual decode projections have separate NCU counters; these are not
whole-model bandwidth/occupancy. The existing path is already near the measured
memory bandwidth limit for those shapes, so increasing block size or occupancy
alone is not a demonstrated large speedup.

## Opt-in warp GDN decode

`gdn_warp_decode.cuh` gives each physical warp one value row. Each lane holds four
register planes covering the original 128 key coordinates. Each plane retains
its original 32-lane shuffle reduction. The four real logical warp partials and
four virtual inactive partials are then reduced in the **original eight-partial
order**. This removes inter-warp shared-memory handshakes without changing
normalization, exponentiation, FMA association, FP16 output rounding or FP32 state
storage. Virtual signed-zero/nonfinite propagation is retained rather than
assuming all absent coordinates contribute a harmless positive zero.

Only explicit SM70 builds with `--gdn-warp-decode` enable this path, and only
`venturi_gdn_compact_run` at T=1, DK=128, dtype=FP16 selects it. Multi-token
prefill/verification, DK64/256, SM89, BF16 rejection and the ordinary public GDN
operator retain their original behavior. Tail warps return uniformly; state and
input address calculations are widened. There is no new scratch allocation,
shared request state, tensor precision or ABI-version change. The default is off.

Example (run only on the authorized remote Linux CUDA build machine):

```sh
PYTHONPATH=src python -m venturi.ops.build \
  --component blocks --architecture sm70 \
  --q8-token-tile 4 --q8-row-tile 4 --q8-row-pair \
  --q8-swiglu-quant --q8-rms-quant --gdn-warp-decode \
  --output-dir /an/owned/isolated/build-directory
```

Pass the resulting explicitly pinned library to the native backend. This flag
is not a silent runtime backend/precision switch or a production deployment.

## Evidence and limits

The operation directory is `operations/single-cuda-v100-20261029-v1/`.
`source-before`, `source-warp-v1` and `source-warp-v2` plus their SHA256 manifests
are separate immutable inputs. V2 widens extreme-geometry indexing and adds
partial-CTA and genuine T1 nonfinite coverage; it does not overwrite V1 results.

Qualification includes exact old/new output and FP32 state bytes, independent
FP64 recurrence, normalized/unnormalized, packed/unpacked, nonzero states,
guarded output/state buffers, graph/in-place chunks, virtual nonfinite lanes,
and odd value-head/value-dimension tail warps. BF16 is not silently converted on
SM70. Small native batch, scheduler/reset/lease correctness precedes trained
full-model fixed-eight and mixed-arrival token/private-state comparisons.
This is equivalence to the qualified Q8 path, not new original-BF16 corpus or
perplexity validation.

The trained timed workload is C1, 16 input tokens, 32 fixed greedy outputs,
capacity512, eight prepared slots, row budget32/chunk16, host-staged TP,
`metadata_mode="sealed"`, round-robin scheduling and experimental mask0.
Three alternating baseline/change process runs exclude load/prepare/warmup.
End-to-end native TPS counts all32 outputs over the common request wall;
post-first decode TPS counts31 over observed inter-token event intervals.
Neither includes HTTP/tokenization. Filesystem caches/resident runtime are warm;
no cold-storage startup claim is made. GPU profilers run separately.

Production configuration and route are never changed. Every full-model campaign
saves the exact original container/watchdog, drains requests, verifies an
independent restore timer plus ExecStopPost **before** stopping the model,
honors the existing producer lock, and restores the original running state and
health. Private environment/full Docker/crontab snapshots stay on the server.
No driver/firmware/register/unlock, privilege-policy, local Rust compilation or
automatic GitHub push is performed.

### Trace-scope trap: resets are not decode-step work

Do not treat the complete capture's `cudaMemsetAsync`/API summary as per-decode
cost. The replay includes one slot's admission reset and, after the last timed
response/NVTX range, the scheduler's close/reset of all eight prepared slots.
For this four-output trace that explains1728 memset calls (nine slot resets),
not432 redundant clears in each decode. Those cleanup calls are outside scored
request wall time and outside decode-only NVTX attribution. Investigating a
launch/synchronization reduction next requires phase-correlated API rows, not
the unfiltered all-capture API report.

## Final measured results (qualified V2)

Three **unprofiled** alternating C1 repeats; medians, same native workload:

| Metric | Baseline | Warp GDN V2 | Change |
|---|---:|---:|---:|
| Post-first decode TPS | 32.806810 | 33.207838 | +1.222% |
| All-output native TPS (includes prefill) | 31.074001 | 31.427151 | +1.136% |
| Request wall seconds | 1.029800 | 1.018228 | -1.124% |
| Process CPU core-seconds/request | 1.030790 | 1.018951 | -1.149% |
| Native observed TTFT seconds | 0.084803 | 0.084613 | -0.224% |

Raw decode TPS arrays are `[32.76322392632501, 32.818823567036695, 32.80680999610648]` and `[33.27122118664759, 33.1926756227734, 33.20783826997247]`. The earlier V1 campaign retained a smaller 0.678% decode / 0.224% all-output gain, and a +3.574% TTFT regression. Neither campaign proves a broad latency/throughput win. V2 improves this workload by about1%; the opt-in is **not promoted to default**.

The rank-local Hqk8/Hv24/DK128/DV128 synthetic immutable-state operator screen improves 15.24768 -> 8.12847 microseconds (1.876x). This is GPU-event time per call, with32 graph-contained calls/replay and6400 calls/wave, three alternating repeats—not eager Python launch latency or whole-model TPS. The actual-model separately instrumented decode trace changes compact GDN from about16.10–16.28 to8.79–8.85 microseconds/call. NCU replay durations are different and are not substituted for unprofiled timings.

Two actual decode GDN NCU samples/device (context16, prefill outside capture):

| Counter | Baseline | V2 |
|---|---:|---:|
| Threads/block; blocks |128;3072|128;768|
| Registers/thread |44|32|
| Achieved occupancy |52.90–53.15%|48.79–49.58%|
| SM throughput, reported |about62%|about47%|
| Barrier stall |23.82–23.83%|0%|

The faster kernel has **lower** occupancy/SM-throughput percentages because it
executes less cross-warp synchronization/work; those percentages alone are not
the objective. DRAM read/write and cache/scoreboard counters remain in the raw
JSON. Some replay-derived L2 ratios exceed100%; preserve those raw values and
warnings rather than calling them literal hit probabilities. Clocks/cache
control are disabled; no profiling permission policy is changed.

Fresh selected Q8 decode projections still measure **736.86/744.27 GB/s**
(read+write decimal bytes/time), NCU-reported DRAM throughput91.37/92.00%,
achieved occupancy41.35/41.38%. This samples two real projection calls, not every
shape or a whole-model hardware roofline. Removing GDN barriers does not remove
the dominant Q8 streaming or host-TP/launch dependencies.

Timed-window200ms NVML samples (not occupancy): baseline busy70.1/72.3%,
V2 busy76.7/69.4% on GPUs0/1, with15 versus14 samples/device. Both arms peak at
16304/14400 MiB device-used memory, median SM/memory clocks1380/877 MHz.
Median power119.12/122.88 ->117.88/121.84 W, maximum temperatures37/35 ->36/34 C.
Short-window NVML lag/sample counts prohibit reading these as an independent
utilization or energy-efficiency improvement. Explicit native buffer byte
accounting is identical; no VRAM/RAM offload/saving is claimed.

### Qualification and reproducibility

- CPU suite:1807 passed,2824 skipped; skips are not GPU passes.
- Actual V100 suite:307 passed,108 deselected,0 skipped; additional T1
  dynamic-input/signed-zero/no-fallback graph test:1 passed,0 skipped.
- Trained 27B fixed-eight and mixed-arrival greedy token IDs plus private KV/GDN
  state hashes match the pinned original path exactly. Nonfinite operator tests
  compare classifications and finite-value bytes, not unspecified NaN payloads.
- Fresh whole-file28,601,956,352-byte artifact SHA256 matches the pinned original.
- Final retained library SHA256:
  `eeb1bb8af7bff79ea77d5465bb0607c2f00541cd7e3c879abd6495660f9903d9`;
  remote path:
  `/home/data/wangyue/repos/codex-build/Venturi/single-cuda-v100-20261029-v1/build/warp-v2/80325883f32ab6c5/libventuri_blocks.so`.
- All14 owned worker/build/analysis/restore handles are inactive. Original
  production container/configuration/watchdog are restored, health200, with only
  its original-model process on both GPUs. This is not a production model switch.
- All147 public evidence files and compiled input manifests are SHA-verified.
  The initial exporter omitted root JSON summaries because of a suffix-filter
  bug; the first archive and counterexample are retained and a new corrected
  archive was verified. No GPU measurements were rerun to repair packaging.
- Local terminal audit:23 checks passed, including old frozen capsule preservation,
  exact current/frozen sources, source/library provenance, correctness, three
  repeats/arm, restoration and raw GPU counter availability.

### Next experiments (not implemented gains)

1. Obtain phase-correlated launch/TP-transfer dependency costs and compare
   overlap/capture changes against this same precision/context/native baseline.
2. Screen T1 RMSNorm (65 calls/step) and A/B gate projection launches; the existing
   explicit fused A/B research flag needs matched qualification with this path.
3. Investigate other important Q8 projection shapes, not just the two sampled
   kernels; preserve the current complete-row reduction and residual-A8 precision
   before accepting a packing/cache/tiling change.

A paged KV allocator, higher occupancy target or a smaller context by itself is
not evidence of faster single-request decode. Larger jumps via weight compression
or MTP require separate precision/quality/acceptance and timing scopes.
