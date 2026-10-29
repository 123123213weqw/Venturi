# Native PD v1 qualification — 2026-10-29

This is an **opt-in, same-process** implementation. It does not replace a
production API or establish cross-host, TP4/PP/MTP, trained-27B performance or
quality support. See [the PD contract](pd-disaggregation.md) and
[public source/result pins](pd-qualification.json).

## Verified

- A clean export of the public Git index: **1840 CPU tests passed, 2844 skipped**
  in 28.01 s. The skipped GPU tests are not passes. Private operational tooling
  and logs are deliberately outside this export. Dependency guard: 110 modules,
  zero boundary violations.
- Remote native compilation succeeded for **SM89 and SM70**. SM70 was compiled
  only, not executed on a V100 in this campaign. The CUDA 12.5 build needed
  explicit typed `std::array` host-side pair ranges in three headers; no kernel
  numerical math was changed for that compatibility repair.
- On idle devices of an eight-L40 shared host, **16 CUDA tests passed and four
  skipped**. Two unrelated GPU jobs were left running and were not used by the
  tests; only six devices were visible to the qualification process. The four
  4P4D cases require eight available devices and remain unqualified.
- FP16/BF16, explicit host/peer transports, 1P1D/2P2D: generated four-layer hybrid
  fixture checks exact continuation logits/tokens, valid KV, FP32 GDN recurrent
  state and convolution history against ordinary no-handoff execution. Tests
  cover source/inactive-slot isolation, stale leases and invalid destinations.

## Not a speed or trained-model claim

The fixture is synthetic, not a trained Qwen-27B checkpoint. Passing these tests
is not a real-model generation-quality check or a model TPS result. The normal
uncontended benchmark was deferred because the host has other tenants. Any
separate fixture CLI smoke on idle devices of that same shared host must retain
that contention caveat and must not be promoted to an isolated speed improvement.
GPU busy/memory-controller samples are not achieved occupancy or DRAM bandwidth;
this qualification did not measure those hardware counters.

Next milestones are an eight-available-device 4P4D correctness run, a same-weight
trained-model PD versus colocated comparison with at least three matched repeats,
and then an independently tested serving/cancellation/streaming integration.
State handoff currently synchronizes participating owners; asynchronous overlap
and cross-host transport are not implemented. No service or production route
was changed by this work.

## Separate shared-host fixture CLI smoke

The [15-arm smoke run](../benchmarks/pd-native-v1/README.md) completed with three
repeats per arm and identical baseline/PD token hashes. Selected devices were
released afterward; unrelated jobs remained running. On this six-prompt-token,
16-output-token generated workload PD was **slower**, so it remains opt-in.
This does not establish trained-model TPS or long-prefill behavior. The report
retains CPU/latency/RSS/transfer and coarse GPU samples, with explicit sampling
and shared-host limitations. No achieved occupancy/bandwidth was measured.
