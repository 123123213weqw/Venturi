# Experimental SM89 Q5 shape calibration

The SM89 build exposes the same grouped GEMV and shared-memory WMMA
row-split weights pipeline alongside canonical unpack + FP32-pedantic
cuBLAS. The SM70 export continues to require an SM70 build. Admission is
not expanded to new weight formats. The SM89 model experiment is explicit,
single-device, resident FP16, Q5-G64 and static private KV with no cache
shadow. This does not change the default or permit PD handoff of the
experimental numeric mode.

```python
batch = model.create_batch([4096] * 8, resident_dataflow=True,
                           attention_execution="descriptor")
# Each rule is [M, N, K, route]: 0=dense, 1=grouped, 2=WMMA.
batch.prepare_ada_tiles([[4, 17408, 5120, tested_route]],
                        direct_outputs=True)
```

Routes are exact keys at execution, not matrix names. A new batch owns
its own immutable table and request state; inference can reuse the
model's read-only resident weights. The initializer rejects duplicates,
invalid dimensions, unsupported routes and grouped M>8. Unlisted or
incompatible shapes use the safe dense path. The
calibration can generate a sparse-derived routing table separately from
the C++ hot path. There is no tuning synchronization or table mutation
during inference.

The one-device study identifies narrow, largest-output and high-cost
representative matrix shapes by inspecting the checkpoint geometry.
Ten anchors benchmark two existing candidate routes each with three
samples, using the whole projection (including any dense unpack and
output copy). Nearest legal anchors predict other shape classes, then
the resulting plan is hashed before the independent full atlas is read.
The atlas tests all supported candidates for M=1 through 32 over every
distinct actual projection geometry. Every tested path checks an
independent FP64 dot over sampled output channels, and the inferred
rules must pass the same held-out numeric check before service testing.
Changing the floating-point reduction order can still change logits,
stationary request state and generated tokens. No bitwise or semantic
quality guarantee follows from a dot tolerance.

The 95% target is an observed service-throughput ratio to the full
atlas under a specified workload and latency budget, not utilization
and not a cross-device guarantee. Report single-lane decode, rolling
prefill/decode, P95 and tuning time separately. A fastest-cell atlas is
a reference policy and is not a proof of the fastest service policy.
