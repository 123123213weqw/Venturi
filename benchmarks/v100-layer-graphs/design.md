# Native scalar MTP2 target-layer CUDA Graph prototype

Previous turn: verified progress. Both T16 implementations passed projection
correctness but failed the declared operator gate and were restored. The separate
two-V100 host-bridge probe proved graph capture/replay with changing inputs and
both ownership fences; it did not prove dynamic model state or serving speed.
The full >=5% external single-decode and1/2/4/8-request objective is unchanged.

## Explicit scope and execution contract

An additive opt-in native/SDK API enables request-owned target-layer graphs for
resident SM70 FP16/Q8-residual scalar MTP2 **prefix2/T3 verification** only. PP,
borrowed MTP1 cohorts, streaming and unsupported GDN gate profiles are rejected explicitly. Existing APIs,
ABI16 descriptors, quantization/scales/KV precision and graph-off execution stay
unchanged. Other widths, prefill, diagnostic layer traces and injected failures
use the original eager path. Embedding, output norm/head/selection and draft
recurrence remain eager. This is not whole-engine or parallel graph support.

Every graph owns a4-byte GPU position and portable pinned4-byte host position.
The cursor is copied before launch on the primary stream. A positioned attention
kernel traverses exactly the original key order/reduction/FMA math; positioned
QK uses the same norm/cast/rotary equations and block geometry. Only integer
cache/table addressing changes. V projects into the existing spare workspace,
then an integer16-bit copy kernel writes the actual dynamic KV location. This
avoids stale captured descriptor destinations and does not allocate another KV
cache. If KV width exceeds the spare-buffer width, an explicitly admitted T3
value scratch is allocated instead; its bytes are separately reported. Original kernels remain unmodified. GDN/history prefix images and both
cross-device producer/scratch and consumer/gather fences remain in the graph.

The first eligible call executes the positioned kernels eagerly to warm module
entry points. The next captures target layers, instantiates and launches once;
subsequent calls replay. Prefix-buffer identities are checked every call, and a
changed identity invalidates the graph. Healthy resets preserve the executable;
a failed reset clears it for safe rewarming. Graph lifetime counters, node count,
position storage and explicit fallback scope are reported separately. Capture
updates transport counters once; replay adds the captured byte deltas once.
Internal driver graph storage is not implied by the4-byte cursor figure.

Model/session ownership remains serialized. Resident source is validated before
replay, at the eager head boundary and after synchronized forward completion;
source mutation must not yield a successful stale response. Per-projection checks
still run in eager/streaming paths and during capture; replay does not pretend to
execute host checks at each node. Capture failure ends/cleans the original capture
and poisons the request; injected layer failure selects the existing CPU-checked
eager path. Reset/close must settle owned work and destroy executable/cursor
resources without freeing model weights or another request's cache.

## Gates

1. CPU preflight/missing-export/build-cache checks and remote SM70 compilation.
   Full source manifest asserted before each build; the new header is part of
   the block build identity and excluded from the unrelated Ada component.
2. Hybrid fixture exact output/target GDN, convolution, committed KV and corrected
   draft state across native/forced accept, first/second rejection, EOS/length,
   repeated healthy resets, failure after graph capture, and closed-model owners.
   Check actual replay/node counters so eager fallbacks cannot pass as graphs.
   Test source mutation after launch before return, not only Python metadata.
3. Fresh trained target teacher/control bytes with the unchanged graph-off path,
   then graph-on MTP2-prefix target/draft state, per-verification logit bytes and
   source-reference gates. Reuse independent unchanged teachers only by pinned
   hashes. Never infer graph logit correctness from plausible IDs alone.
4. Fresh C1/C8 screens against the strongest qualified native scalar MTP2 control,
   keeping ordinary256 and MTP1-cohort controls. Any promoted path receives three
   unprofiled1/2/4/8 repeats. C1 HTTP>1% or C8>2% is the screen/repeat adoption gate;
   no repeated cell may regress >2% and qualifying cells need two improving trials.
   The external completion gate remains >=5% on single decode and every required
   matched parallel count, not merely faster graph launch or HTTP prefill.
5. Separate full-call traces and sampled CPU/latency/TTFT/VRAM/GPU telemetry;
   preserve acceptance, target replay and actual graph/fallback work denominators.
   No achieved occupancy/DRAM/cache/stall claim while the counter policy is1.

All C++/CUDA/Rust compilation stays remote. No unrelated process/driver changes,
commit/push, permanent service or edits to the34 earlier evidence capsules.

## Initial fixture discovery and repair

The first build safely rejects the synthetic fixture: it has hidden32 but KV
width64, unlike the trained model. The guard prevents an undersized spare-buffer
write; its failed run remains immutable. The repair admits dedicated T3 value
scratch only for that geometry and reports/account its bytes. It does not weaken
the guard or alter the fixture to dodge a valid model shape. The trained Qwen
case still reuses its adequate spare buffer without this allocation.

## Diagnostic bytes and source-guard coverage

Untimed verification tracing retains complete FP16 head bytes and frame widths
for graph-on/eager MTP2 comparison, not merely greedy IDs or a hash-only sample.
Tracing is disabled by default and never enabled in speed/profiler runs. A
separate native diagnostic pause occurs after graph launch; a test thread mutates
only its temporary fixture source during that pause and verifies native failure
before a stale response can be returned. The nonblocking pause poll is test-only
and requires retaining the live session until its generation thread joins.

Capture uses thread-local mode under the target model mutex, allowing independent
resident draft work while preserving exclusive target streams/scratch. Fixtures
exercise concurrent2/8-session capture/replay. Failure injection after capture
uses the original checked layer loop and requires reset; capture cleanup and
request-local byte accounting remain explicit.
