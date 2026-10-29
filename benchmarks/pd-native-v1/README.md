# Native PD v1 generated-fixture CLI smoke

**Not Qwen-27B TPS. Not an isolated-host performance qualification.**
The four-layer generated Q8 fixture only checks that the benchmark and cleanup
paths work on real native GPUs. Two unrelated GPU jobs shared the L40 host;
selected devices were idle and no existing job was stopped.

[Structured sanitized results](fixture-smoke.json) retain all three unprofiled
repeats, shared wall/CPU time, queue/TTFT/max token gap, output hashes, setup/RSS,
scheduler/transfer counters and available GPU snapshots. One warmup round per
arm. Ordinary greedy; no MTP/prefix sharing. Six prompt tokens, 16 output tokens;
C1/C2/C4/C8 compare 1P1D to colocated on the same two-device union; C8 also
compares 2P2D to colocated on the same four-device union. Host and peer transport
are explicit. Every baseline/PD group has identical output-ID hashes, all 15
arms have three repeats, cleanup passed, and selected devices returned to zero
GPU memory use after test processes exited.

In this **very short, synthetic workload**, PD was slower than colocated. This
is useful negative evidence against enabling PD by default; it is not a
trained-model or long-prefill conclusion, and peer copying was not consistently
faster. Handoff synchronization and a dedicated P/D split add costs; this smoke
run did not separately attribute those costs. Host tenancy, fixed arm order and
millisecond sample durations prevent a robust optimization claim. GPU snapshots
are too coarse to establish achieved occupancy, DRAM bandwidth or kernel-time
attribution. No profiler was mixed into these timings.

[Implementation and GPU correctness scope](../../docs/pd-qualification.md).
Next: eight-device 4P4D correctness, then same-weight trained-model, realistic
long/short mixed arrivals, randomized/interleaved matched timing on a controlled
host. Do not use these fixture rates to advertise model-serving speed.
