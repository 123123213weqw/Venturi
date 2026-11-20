# Two-V100 final controls: numeric target passes, full goal remains open

SSH recovery succeeded on the same verified V100 host. All 16 remote jobs
are terminal; 1,161 raw metadata/report/log/response files are captured and
hash-verified, and the original 3651/3189 MiB background GPU footprint is
restored. No owned model server, sampler or port 18953 remains. All 43 prior
frozen capsules and 295 qualified production/test/tool source files are unchanged.

## Final fresh results

Same official Qwen3.5-27B revision `fc05daec18b0a78c049392ed2e771dde82bdf654`,
declared Q8 profiles, FP16 KV, two V100-PCIE-32GB GPUs (physical 1/0), 23
formatted prompt IDs and 32 greedy outputs ignoring EOS. Eight prepared
256-context slots. Loading is excluded; one warmup precedes three timed
repeats. Complete HTTP rates include admission/prefill/client overhead and
use total emitted outputs/global cohort wall, never summed request latencies.

| Scope | Native TPS | Strongest repeated pinned llama.cpp TPS | Gain |
|---|---:|---:|---:|
| Single post-first decode | 52.95121 | 50.05183 | 5.79% |
| Complete HTTP, 1 request(s) | 42.10749 | 30.56779 | 37.75% |
| Complete HTTP, 2 request(s) | 57.23141 | 37.97583 | 50.70% |
| Complete HTTP, 4 request(s) | 70.29590 | 58.77979 | 19.59% |
| Complete HTTP, 8 request(s) | 82.97479 | 67.28722 | 23.31% |

Strategies are explicit: scalar MTP2/prefix recovery at 1 request; packed
MTP1 at 2/4; ordinary packed row256 without MTP at 8. There is no automatic
strategy router, paged attention, continuous batching or long-context claim.
Single post-first native trials are 52.94818/52.96643/52.93901 TPS; all clear
the stronger fresh external 50.05183 × 1.05 = 52.55442 floor. Earlier
qualified/confirmation trials also clear that floor. Every final HTTP
trial clears its strongest historical-or-fresh repeated external floor,
and also clears the highest valid one-repeat external screen by >=5%.
The faster historical native 83.85 C8 result is retained, not substituted
for the lower fresh 82.97 result. See `selected-comparison.json` and
`recomputed-controls.json` for all trials and provenance.

## Why the full goal is not marked complete

The pinned production llama.cpp fork still lacks a repeatably completed
one-proposal MTP matrix at 2/8 requests. Single MTP1 and fixed-C4 standard-
sampler MTP1 have valid repeated controls; remaining parallel attempts
crash or close the loopback API. Requested backend sampling, standard
sampling, layer/tensor split, graph-disabled compatibility and fresh
fixed-concurrency probes are preserved. Graph-disabled fallbacks never
replace stronger graph-enabled references. A failed arm is not zero TPS.
A successful screen is not silently promoted; a failed C2 campaign is not
proof that C8 is unsupported, which is why isolated C8 repeats were tested.
Crash root cause is unproven. The measured >=5% primary numeric target
passes, but a complete paired-MTP acceptance claim remains unavailable.
Next required work: establish a repeatably working pinned comparator MTP
configuration for those counts and finish the matched control matrix.

The executable and all seven actually loaded libllama/libggml engine
library hashes match the retained release. This is not a comparison with
latest stock upstream, and its exact source commit is unverified.

## Correctness and precision

Ordinary row256 qualifies against 98 hash-verified original-FP16 teacher
rows: 98/98 top1 agreement, max relative L2 0.00258149423, minimum centered
cosine 0.9999963130, max reference KL 8.29287368e-5. Four trajectories,
mixed-eight private state, EOS and reset/continuation pass. Reuse the
unchanged phase 43 trained MTP/committed-state/failure qualifications;
do not repeat unchanged full engine suites. Q8 codes/scales and original
small values stay unchanged; residual-A8 arithmetic and FP32 partial
reduction association are separately qualified, not FP16 byte parity.

The finite cross-engine diagnostic covers 11 aligned next-token positions
across four cases, with 64 finite numeric-ID alternatives each. Top1 agrees
10/11; Python step 7 differs (native 3165, external 6707). Top64 overlap is
61–64; maximum absolute logprob difference over external top64 is 0.471455.
These are declared different Q8 encodings/internal arithmetic, not exact
cross-engine or original-BF16 corpus-quality parity. Truncated top64 does
not provide full-vocabulary cross-engine KL. Native full-logit hashes
exactly match their qualified report; cached native tensors stay remote.

Two diagnostic assumptions were repaired, not native accuracy gates:
ignore-EOS masks generated EOS while pre-sampling probabilities retain
it; diagnostic-only `ignore_eos=false` resolves that mismatch. A Float32
peak logprob 0 with 63 positive alternatives and tail 2.804e-8 is valid
rounding, not a point mass. The repaired guard requires a Float64-resolved
tail within one Float32 ULP and retains finite/ID/normalization/greedy
checks and the original manufactured-point-mass rejection. No values
are clamped; native reference thresholds and benchmark policy do not change.

## CPU, latency and GPU evidence

| Requests | Native cohort seconds | Server CPU core-seconds | Client completion min–max seconds |
|---:|---:|---:|---:|
| 1 | 0.759960 | 0.756667 | 0.756980–0.761104 |
| 2 | 1.118267 | 1.090000 | 1.114184–1.118098 |
| 4 | 1.820874 | 1.793333 | 1.816920–1.819694 |
| 8 | 3.085274 | 3.083333 | 3.076399–3.086597 |

Native single first-ready is ~0.13819 s. External prompt evaluation (~0.337 s
for the strongest single MTP2 arm) is not streaming-client TTFT. The
non-streamed loopback benchmark cannot measure client TTFT; it is null,
not inferred from completion time. All raw per-request timings remain.

Final native C8 sampled GPU busy is 85.24%/75.93% (physical1/0), peak
device-used VRAM 20791/18151 MiB including unrelated background allocation.
Raw NVML provides sample counts, power, temperature and clocks. Timed
SM/memory clocks are 1380/877 MHz; no clock policy changed. NVML GPU busy
or memory-controller busy is not achieved SM occupancy or DRAM bandwidth.

Separate same-library ordinary C8 profiling: instrumented host span
3499.20→3134.80ms; summed kernel time 4256.08→4386.03ms (+3.1%, not a
critical-path regression); simultaneous two-GPU kernel span 1468.63→
1862.85ms. Host payload each direction 2.1188→1.8427GB (-13.0%). This
supports reduced transfer/serialization and more overlap, not a claim
that every GEMM became faster. Process activity gaps are not global
device-idle time. Profiler rates are not used as unprofiled TPS.

Exact-SO resource/actual-launch joins are retained. Achieved occupancy,
compute/DRAM/cache/stall counters remain unavailable under unchanged
`RmProfilingAdminOnly: 1`; static/theoretical residency is not substituted.

## Regressions and evidence retention

Ordinary single decode 32.31653 remains below the retained non-speculative
comparator 34.44882 TPS. Scalar MTP2 at 8 requests remains ~42.13 TPS and
loses to ordinary batching. There is no universal-mode/model/hardware win.
The setup loader failure 127, all actual SIGSEGV/API failures, dependent
preflight stops, partial valid cells, unpromoted screens and diagnostic
failures remain. The initial 97-file network checkpoint is unchanged.

Network recovery used bounded verified same-host connections, resumable
file hashes and shared-compressed 8 KiB archive chunks. Source deduplication
used authenticated byte-identical worker pins; remote manifests confirmed
or replaced provisional seeds, and 17 unlisted seed files were removed
rather than misrepresented as emitted preflight evidence. Large native
tensor/profiler binaries remain remote with size/SHA pins. No host-key,
router, SSH-server, agent-forwarding, profiling privilege or clock policy
was weakened. No local Rust/CUDA compilation, new production change,
commit or push occurred in this phase.

Authoritative audit: `completion-audit-final.json`. Recompute scripts do
not authorize goal completion. This evidence capsule closes the recovered
experiment, not the outstanding complete-paired-MTP requirement.
