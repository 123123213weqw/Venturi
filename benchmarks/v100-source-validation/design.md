# Resident checkpoint validation: measure before changing it

The residual-A8 MTP draft is qualified and frozen in capsule39. Single decode
48.4089 TPS still misses the stronger pinned external49.8033 and goal52.2935.
Its timeline has unchanged target verification and substantial primary projection
work. Investigate one distinct CPU cost, not another failed DP2A/tile sweep.

## Bounded diagnostic, no engine change

The native resident `Model::weight` currently calls `source()` for every weight;
column projection/MLP also have explicit checks. Each `Mapping::verify` compares
both open-descriptor and pathname dev/inode/size/mtime/ctime with the initial
stamp. Never remove those integrity checks merely to obtain a benchmark win.

1. Sync all current sources to a new named V100 snapshot. Reuse the qualified
   residual-draft SO6d43160 and unchanged target/draft payload identities. No new
   CUDA/Rust compilation, trained-quality campaign or competitor run is needed
   for unchanged inference code. Compile only a CPU metadata microbenchmark
   remotely after source-manifest verification.
2. Run three untraced raw native MTP2 calls after one warm generation, with the
   qualified story23/32 IDs, FP16 KV, column-MLP target/residual draft,32 projection
   rows, eight prepared256-capacity slots, two V100s and explicit host transport.
   Loading, reset, SDK checks and result inspection are outside marked intervals.
   This is NOT complete HTTP throughput or a new engine-vs-engine comparison.
3. Separately start one child under ordinary unprivileged strace, tracing metadata
   syscalls only. Do not attach to unrelated processes or change ptrace/driver
   policy. Mark the raw native interval with wall-clock timestamps, pair split
   records by TID, distinguish descriptor/path checks and filter exact artifact
   paths. All generated IDs/counters must remain correct. Instrumented syscall
   durations must not be presented as untraced cost or achieved device occupancy.
4. Measure four100,000-pair warm CPU `fstat+stat+same` loops per actual checkpoint.
   Check unchanged metadata before/after. Count times median CPU pair latency is
   only an opportunity estimate: calls can overlap GPU work, and independent
   loops differ from inference cache/scheduling behavior. No predicted TPS claim.

## Decision gate recorded before results

Only consider a follow-up deduplication implementation if counted checks times
untraced median pair latency exceeds1% of untraced whole-call wall AND exceeds
5ms per generation. Even then require lexical resident-only validation scope,
forced public/start/end checks, unchanged streaming read guards, deterministic
mid-call mutation rejection and failure/reset/concurrent ownership qualification.
That scope and its tests are NOT implemented by this diagnostic. If this bound
fails, preserve the negative result and move to measured target projection/GDN
work rather than spend a full performance campaign on source checks.

Keep all39 previous frozen capsules and diagnostic failures. Preserve original
3651/3189MiB unrelated GPU allocations, explicit host selection, verified SSH,
counter policy1, and no local builds/commit/push. Collect NVML samples separately
from syscall counters; reuse the qualified preceding GPU timeline with its exact
SO identity instead of pretending this strace run measures CUDA utilization.
