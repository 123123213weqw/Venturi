# Adaptive MTP: measured-cost fresh-cohort routing

This opt-in layer selects among existing native paths without changing Q8 codes,
scales, activation arithmetic, KV precision, sampling, or CUDA kernels:

- **0:** ordinary packed target decoding;
- **1:** packed-target MTP1 with private draft/target state;
- **2:** scalar MTP2 with prefix recovery, available only for one request.

Selection happens once per fresh `generate` cohort. It does **not** change depth
inside an in-flight generation or migrate committed GDN/KV state between banks.
There is no continuation, continuous batching, paged KV, stochastic sampling,
trained confidence head, or automatic production/XHarness deployment here.

## Reusable policy

`venturi.adaptive_mtp.AdaptiveMTPPolicy` uses only the Python standard library;
the Venturi package retains its normal installation dependencies. Other model
adapters can supply supported depths and successful-cohort observations.

The policy keeps a bounded LRU of 128 workload buckets by default. A key contains
exact request count, power-of-two buckets of maximum/total prompt lengths and
maximum/total output budgets, and whether EOS stopping is enabled. These are
coarse workload features, not a predictor of prompt difficulty or token confidence.

Each strategy learns an EWMA of **whole-cohort wall seconds / actual emitted
tokens**, with alpha 0.3. Accepted/proposed counters are retained for diagnostics;
they do not masquerade as calibrated confidence predictions. Request durations
are never summed to compute throughput. Failures are not performance samples.

For an unseen workload, the default policy first probes ordinary decoding, then
supported/useful MTP depths. After exploration it selects measured MTP only when
its estimated cost is at least 3% below ordinary decoding. Stale alternatives
are periodically rechecked after 16 successful bucket observations. More than
one stale alternative can therefore be probed on successive calls. **Probes can
increase latency; this is not a guarantee of beating the best fixed strategy.**

Set `probe_interval=0` to disable automatic exploration/rechecking. Unmeasured
workloads then use ordinary decoding; explicit successful manual observations
can still teach the policy. `strategy=0`, `1`, or `2` manually overrides selection;
`learn=False` excludes that execution from policy history. A one-token budget
does not probe useless drafts. Batched MTP2 is rejected rather than serialized
and misleadingly labelled a batching optimization.

Policies are in-memory and execution-instance local. Use a separate policy for
each model, precision, device placement, library and projection configuration.
There is no automatic reuse of historical V100 throughput thresholds, no profile
download, and no persistent cross-model calibration cache in this version.

## Native Qwen API

Keep the existing resident FP16 native target and its MTP model open. PP is not
supported by this adapter. The target and draft profiles remain explicit when
loading the model; adaptive selection cannot change them.

```python
from venturi.adaptive_mtp import AdaptiveMTPPolicy

# mtp_model is the existing NativeQwenMTPModel; target remains open too.
policy = AdaptiveMTPPolicy(probe_interval=16)
with mtp_model.create_adaptive_batch([256] * 8, policy=policy) as adaptive:
    adaptive.prepare()  # Allocate supported banks outside generation timing.
    outputs = adaptive.generate(
        [formatted_prompt_ids], max_new_tokens=32, eos_token_ids=[]
    )
    print(adaptive.info()["decision"])
    # Repeat on the same instance to learn from completed cohorts.
    outputs = adaptive.generate(
        [formatted_prompt_ids] * 4, max_new_tokens=32
    )
```

Separate ordinary-batch, MTP1-batch and scalar-MTP2 banks share model weights but
have **additional, disjoint** request/backup buffers. Preparing all banks costs
more VRAM than preparing only one fixed strategy. `prepare([0, 1])` can stage only
those banks, but a later automatic MTP2 choice will allocate its bank on demand.
Preparation time is reported separately and excluded from learned execution
cost; callers must include it when assessing actual cold-request latency.

Generation resets only the selected bank's participating slots. Snapshots are
diagnostic and accessible only for active slots from the last completed cohort.
Native errors propagate and poison the adaptive owner; there is **no retry using
another strategy**. Call `reset()` explicitly to recover. Reset clears bank state
but preserves policy history. Closing the adaptive owner closes its banks, not
the caller-owned target or MTP model. Calls on one owner are serialized.

## Benchmark HTTP adapter

`tools/serve_native_benchmark.py --adaptive-mtp` enables persistent policy learning
inside the existing loopback-only, token-ID, fixed-greedy benchmark adapter. It
requires `--mtp-artifact` and TP, and rejects conflicting fixed-mode switches.
`--mtp-probe-interval 0` disables automatic probes. Responses include the actual
`adaptive_mtp.decision`, selected depth, preparation/execution costs and acceptance
counters. Those totals are **shared per cohort**, not additive across responses.
Unavailable scalar group counters are null, not invented zero measurements.

This remains a benchmark adapter, **not an OpenAI production server**. The original
XHarness llama.cpp route and configuration are not replaced by this feature.

## Validation and performance protocol

- Unit tests cover cold exploration, cost feedback, disabling/re-enabling MTP,
  periodic probes, manual controls, workload isolation, eviction, invalid values,
  and benchmark-admission metadata.
- Two-V100 generated hybrid fixtures cover 1/2/4/8 requests, ragged budgets,
  reordered slots, EOS, strategy switching and exact target-state bytes for both
  FP16 and the existing semantic-head-GDN residual-Q8 profile. Injected failures
  must poison the owner, not trigger fallback or become learning observations.
- `tools/benchmark_qwen_adaptive_mtp.py` checks every candidate against the pinned
  original greedy trajectory before comparison. It separates warmup, actual
  cold-policy exploration, additional calibration and three timed repeats.
  Fixed and automatic paths use one shared target/draft owner and one projection
  row configuration. Timed results include policy, reset, native execution,
  metadata and return overhead, but exclude loading and prepared-bank allocation.
- Compare automatic execution against ordinary decoding **and the strongest fixed
  candidate**, not merely the weakest fixed MTP depth. Record CPU seconds and
  per-window NVML samples separately. NVML busy is not achieved occupancy;
  denied hardware-counter access is not bypassed.

The experiment and production restoration records are under
`benchmarks/v100-adaptive-mtp/`. Results from a short homogeneous prompt are not
evidence for arbitrary corpus quality, 256K contexts, or universal speedup.

## Design reference

The motivation is DeepInfer's cost-aware variable-length verification and CPU/GPU
overlap, not a code or kernel transplant. Reference reviewed at commit
`774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`:
[DeepInfer deployment Q&A](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/inference_deployment_qa.md).
Its confidence-head DSpark implementation and architecture-specific bounded
replay are not implemented by this measured-cost policy.
