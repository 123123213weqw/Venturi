# Venturi inference performance work

For each inference-performance optimization, run the relevant correctness checks
and an unprofiled baseline/change comparison on the target GPU. Keep checkpoint,
precision, input/output lengths, context, concurrency, warmup and timing scope
explicit. Report CPU time, latency and throughput separately; do not sum
concurrent request durations to calculate aggregate throughput.

Include GPU diagnostics for the affected path: sampled GPU busy percentage,
VRAM usage, power, clocks and temperature; kernel-time attribution and timeline
gaps when diagnosing a bottleneck; and achieved SM occupancy, compute throughput,
DRAM bandwidth, cache/stall counters when hardware-counter access permits.
Record unsupported or permission-denied measurements explicitly. Never substitute
theoretical occupancy, memory-controller busy percentage, or kernel-time share
for achieved occupancy or bandwidth utilization. Do not change driver privileges
or system-wide profiling settings without an explicit user request. If access
is denied, check the current policy and report the missing counters; do not
repeat long counter campaigns while the blocking condition is unchanged.

Keep profiler runs separate from unprofiled throughput measurements because
instrumentation and counter replay perturb execution. Preserve numerical
counterexamples and state/quality limitations, not just plausible output text.
