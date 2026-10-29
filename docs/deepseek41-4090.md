# DeepSeek V4.1 / single 4090D overnight qualification

## Native-only policy update (2026-10-28)

The former `venturi.backends.strata` adapter and `tools/benchmark_strata_dsv41.py`
were removed from supported source, tools and packaging, along with the external
NInfer/Cinference model-engine route. Their historical source and tests are kept
only in `operations/native-only-20261028-v1/source-before/`. Statements below about
the old comparator/harness describe the prior experiment, **not an available
runtime or a fallback**. Community sources remain architecture/format references.
Full native DS41 execution is still unsupported; no real generation/TPS result
has been added. No old download, build, goal, automation or production route was
resumed by removing these adapters.


## Read-only recovery observation (2026-10-28 09:41 Asia/Shanghai)

Fresh strict SSH to School_4090D now succeeds. The same 4090D UUID is present
(24,564 MiB total, 387 MiB used, 0% GPU busy at the snapshot). The root and
/home mounts report ordinary rw with no emergency_ro; ext4 sysfs error counters
are zero. Uptime is about 38 minutes. This is a **read-only snapshot**, not a
write/fsync/SMART or sustained-IO health certificate, and does not establish the
original physical fault or repair method. HALT_GPU_ROOT_STORAGE_FAULT is retained.

The original RWKV sidecar is **not currently running at its previous endpoint**:
no matching process was found and localhost:8417/health refused the connection.
No service was started/restarted/stopped by this check. The retained HDD supervisor
result now confirms the prior download child exited with code 1 at **October 27
04:00:42 Shanghai**, before the deadline; its log records DNS resolution failures
and an incomplete-checkpoint exception. Stored progress remains **0/7 full-SHA
verified**, with no complete-manifest marker. Current download-network health and
actual weight payload integrity were not tested. There are no matching old
owned staging processes in the fresh snapshot.

Evidence is in operations/deepseek41-4090-recovery-20261028-v1/. The previous
night operation and all frozen closeout evidence remain unchanged. The automation
is still PAUSED; no old goal/queue, download, source build or GPU experiment was
resumed. Native whole-model support and real TPS remain unestablished.


## Overnight closeout (2026-10-27 08:00 Asia/Shanghai)

The run is closed; no new experiments are started after the deadline. The final
08:00:34 read to the configured School_4090D endpoint reset before SSH returned
any target state. HDD staging termination, later download/SHA progress and current
original sidecar health therefore **remain unconfirmed**. Its absolute supervisor
cutoff is configured, not observed proof of termination. The protected original
service was never paused/signalled; no restart/repair was attempted. Its last
actual HTTP200/worker-healthy observation is **02:36 Shanghai**, not a current
health check. Dual-V100 production was untouched. See `cutoff-closeout-v2.json`
and `cutoff-0800-read-v1.json` in the operation directory. Native whole-model
generation and real TPS remain unimplemented/unmeasured; **20 TPS was not achieved**.
Only `deepseek-v4-1-4090` was explicitly paused through the automation tool;
the persisted PAUSED state and unchanged prompt/schedule/target were verified
(`automation-pause-v1.json`). Other automations and old goals were left alone.


> **October 27 storage incident:** the target root SSD (`sda4`) entered ext4
> `emergency_ro`; reads from `sda` fail with `DID_BAD_TARGET` / `EIO`, including
> system executables. GPU experiments and source builds are halted by
> `HALT_GPU_ROOT_STORAGE_FAULT`. Do not pause/restart the original GPU sidecar:
> it was last confirmed healthy at 02:36, but later SSH reads failed and its
> restoration cannot be guaranteed on this system.
> No remount, fsck, reboot, driver/firmware/register/permission changes were made.
> Root-cause hardware diagnosis is not established. Continue only independent
> local CPU work and explicitly isolated HDD asset staging, if healthy; never
> silently replace the requested benchmark hardware or report a TPS result.

## Scope and status (2026-10-26)

Target: **School_4090D**, one unmodified RTX 4090D 24 GB, genuine
DeepSeek-V4.1-Flash-1017, with a **20 decode tokens/s goal, not an achieved result**.
No driver/firmware/register/boot/privilege changes. The dual-V100 Qwen3.8
uncensored production route must remain untouched. No automatic GitHub push.
New experiments stop at **2026-10-27 08:00 Asia/Shanghai**, then restore and close.

Venturi does **not** currently implement this complete architecture natively.
The community comparator is **external Strata `strata-dsv41`**; its tokens/s must
never be attributed to Venturi CUDA kernels. That former explicit adapter was separate from `OperatorBackend` and has now
been removed; see the native-only policy update above.

Evidence and live continuation state: `operations/deepseek41-4090-night-20261026-v1/`.
Check remote `stage.pid`, its process command, `stage.lock`, child processes and
`stage-result.json` before resuming: an interrupted SSH session is not proof that
the download or build stopped. Staging never pauses a service or starts inference.

### Readiness — do not confuse CPU components with a usable model

| Path | Available evidence | Still missing / not established |
| --- | --- | --- |
| Exact model/quant identities | Immutable seven-file manifest, official small config/tokenizer authenticated | Complete target weights: last confirmed 0/7 full-file SHA at 02:36; later progress unknown |
| GGUF inventory/storage contracts | CPU schema, full-SHA descriptor, bounded row/decode and failure tests | Actual whole-checkpoint runtime admission on healthy storage |
| Engram | Generated IQ4_NL + Q5_K/BF16 -> F32 synthetic sublayer/reference comparisons | Real 5120-hidden weights, native operators and mixed-GPU arithmetic |
| Router / mHC | Selected CPU semantic oracles and counterexamples | Native whole-layer GPU execution and integration |
| Attention / state | Pinned source/architecture analysis | Full MLA/window/compressors/indexer/candidate-cache implementation and validation |
| Venturi native generation / 20 TPS | **Not supported / not measured** | Full native architecture, real quality checks and paired target benchmarks |
| External Strata comparator | Retired October 28; source/protocol tests archived, not packaged | No supported external runtime or fallback; historical external data is not native Venturi |

The SSD incident prevents safe target builds/tests, but is **not the only missing
prerequisite**: full native architecture integration is also unfinished. CPU test
totals, format decoders, source archives and small reference files do not make
`venturi generate` work for this model. Unmeasured TPS is **null**, not zero and
not an extrapolation from small operators. Existing Qwen production paths were
not redirected to these experimental components.

## Pinned inputs

- Official model: [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017),
  revision `1bae0e1088993b9b82de08f59de5be5a34e42170`.
- [ISTA-DAS GSQ-RCO GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF),
  revision `653a9874aa38c356233c5f3cc8c96b19a43b57eb`.
- [Unsloth GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF),
  revision `60ad3651641b5c97679c807e064ab1d7a3ca1455`:
  only the Q8 DSpark drafter and a 5.8 MB metadata-only shard.
- [Strata source](https://github.com/Niko1221/Strata/tree/4b9e05a2a806a3fe24f7b2560d98626c37dade35),
  must be built explicitly for **sm_89**, CUDA 13, CPU AVX2 on the target, not downloaded
  as an sm_86/sm_120-only prebuilt package. Its ggml dependency is separately
  pinned to `3cf03257f219afbe7334045ff7c6a06ac68c627d`.

The local manifests contain exact per-file sizes, SHA256 and immutable URLs.
The first IST shard changed metadata upstream: current HF hashes differ from
Strata setup's older pins. Do not rewrite the checkpoint, silently accept its old
hash, or bypass the hash check to make an installer work. Use the explicit new
manifest and plain pack recipe, then check pack geometry and generated outputs.

### First candidate: 1.75-bit backbone

| Asset role | Bytes | Placement intention |
| --- | ---: | --- |
| Four IQ1_M backbone shards | 120,721,131,328 | dense GPU, hot experts GPU, cold experts RAM |
| IQ4_NL Engram | 110,595,290,816 | SSD-backed sparse lookup; bounded RAM caching |
| Q8_0 DSpark | 7,976,914,240 | explicit speculation experiment, not always beneficial |
| Unsloth metadata shard | 5,780,589 | hash/token-map metadata only |
| **Total download/storage** | **239,299,116,973** | approximately **222.86 GiB**, not VRAM demand |

This is a **mixed** quantization bundle, not uniform IQ1_M everywhere. Engram is
an embedding lookup, not n-gram speculative decoding. DSpark is a draft model.
1.75-bit quality is lossy: the quantizer reports 98.7% of the base model's mean
on three selected task benchmarks, which is not 98.7% quality on every task or
lossless numerical equivalence. Strata's separate token-distribution tests report
larger prose degradation. Keep those author claims separate from our own tests.

## Actual machine / capacity

Read-only SSH inventory:

- UUID `GPU-7dea0248-3093-8d77-7f6f-05f532efac36`, 24,564 MiB VRAM.
- Dual Xeon E5-2640 v4: 20 physical / 40 logical cores, AVX2, two NUMA nodes.
- About 125 GiB installed RAM, 115 GiB available at intake; swap already used
  about 13 GiB. Never equate swap capacity with fast resident model capacity.
- `/dev/sda4` root on **Intel SATA SSD**, about 332 GB free at intake.
- `/home` on **ST4000NM0035 mechanical HDD**, not NVMe, about 2.6 TiB free.
- The initial placement was `/var/tmp/venturi-deepseek41-20261026-iq1m` on the SSD
  with a **32 GiB free-space reserve**. **That root SSD has since failed; do not
  resume this path.** The fresh HDD path below is asset staging only, not an
  approved replacement for latency-sensitive Engram inference. The external pack references the
  original GGUF bytes; no extra full-weight copy is needed.
- Intake GPU process: user `test`, PID `1540787`, `rwkv_agent.sidecar` on
  localhost:8417, about 21.1 GiB VRAM. Also a desktop process ~354 MiB.
  **Neither has been stopped during intake/staging.**

Admission estimates are not benchmark results. Derive actual dense, expert,
draft and workspace sizes from the built pack before invoking the generic
`model_assets.plan_moe_storage`; do not derive routed-expert bytes from bpw alone.
Use available RAM / free VRAM, not their installed capacities, and count a cold
expert once (file cache OR an anonymous RAM store). A sparse lookup file does not
have to be entirely resident, but its page-fault latency still matters.

## Community findings worth testing, not copying as measured facts

[Strata DS41 documentation](https://github.com/Niko1221/Strata/blob/main/docs/DEEPSEEK41.md)
describes GPU-resident dense weights/hot experts, a parallel CPU expert lane,
sparse SSD Engram and DSpark. Its tuned RAM store is named `--ssd-cache` despite
occupying host RAM. Its single-request DS41 engine does not yet implement
concurrent request batching; don't claim eight-request native support from this.

[RTX4090 issue #3935](https://github.com/Niko1221/Strata/issues/3935) contains a
successful sm_89 source build and higher decode rates, but on a **48 GB modified
4090 with DDR5 and NVMe**, not our 24 GB / older DDR4 / SATA machine. Older
DeepSeek-V4, R1 distilled models, synthetic kernel throughput and repetitions
served from retained conversation state are not interchangeable V4.1 results.

## Correctness and measurement gate

1. Complete verified weights and sm_89 build; validate pack tensor spans, all
   files, Engram hash/token map, original six routed experts, and tokenizer.
   Plain pack recipe only: no `--skip-tensors` / `--dense-override` workaround.
2. Save the 4090 sidecar identity/configuration/state, establish `finally` and an
   independent timed restoration guard **before** any temporary pause. No GPU
   use while it owns the card. Do not stop gnome or V100 production for a test.
3. Smoke tests with known answers, arithmetic, Chinese, instruction following
   and code; preserve token IDs/text/failures. Plausible text alone is not a
   numerical oracle or comprehensive generation-quality validation.
4. Reference: external same-quant greedy decode, six experts, DSpark **none**
   (not `off`, which can retain drafter weights). Candidate: DSpark on/auto,
   block/confidence sweeps only after equal-greedy-token verification on the
   fixed prompts. Mismatches are retained, not deleted or reclassified as speed.
5. Start with context 8192, batch/concurrency one, four distinct prompts,
   128 output tokens, at least three paired unprofiled repetitions. Disclose
   actual EOS counts, reused prompt tokens, warmup and timing scope. Cold model
   load and warm resident decode are separate. Never drop original top-k six
   experts or use a shorter output silently to reach 20 TPS.
6. Record engine prefill/decode time and separately host-observed TTFT, request
   wall time, CPU time, RSS/anonymous/file cache, storage reads/faults, VRAM,
   GPU busy/power/clocks/temperature. Use achieved counters only if allowed;
   profiling is separate, no driver privilege edits.
7. Restore only services that were originally running and verify their original
   health/configuration. Do not leave an experimental API as a production route.

## Architecture work

`model_assets.py` adds immutable asset identities/roles and an explicit byte-level
MoE storage-admission plan, without importing Torch or loading weights. It is not
a live eviction policy. `OperatorBackend` remains the prepared-operator contract;
`backends/strata.py` is a separate explicit, serialized **external token runtime**.
No implicit model/backend fallback, discovery, executable search or native label.
The existing native generation paths are unchanged. Synthetic contract tests do
not establish DS41 model correctness or performance.

`models/moe_routing.py` now provides CPU semantic oracles for softmax, sigmoid
and sqrt-softplus selection plus the asymmetric clamped expert activation. An
explicit qualification extracted only `Gate.forward` / `Expert.forward` from
the hashed official source, without importing its CUDA-dependent model: **28
synthetic cases were byte-exact**, including top-k one/six, normalization on/off,
image-token selection bias and weighted/unweighted expert activation. This does
not validate real checkpoint bindings, projections, attention, Engram or a full
forward pass. No existing native model uses these CPU oracles as a fallback.

The archived former `tools/benchmark_strata_dsv41.py` was an external-only harness: pinned executable
and model identities, file/span binding, GPU exclusivity check, original top-k
six, retained-prompt reuse disabled, distinct fixed prompts, three repeats,
CPU/IO/RSS and GPU samples, raw STAT/DONE/draft acceptance and token outputs.
It requires an already qualified pack and does not stop a service itself.

### Additional community lead (October 26)

[NInfer-all's DS41 implementation](https://github.com/iamwavecut/ninfer-all/blob/master/docs/deepseek-v41.md)
also uses the selected IQ1_M + IQ4_NL bundle and has explicit host/file residency
and a CPU share. Its reported 4090 results differ sharply between pinned host
experts and file reads. Its host tests used substantially more RAM than this
machine; its authors specifically leave 128 GB hosts untested. The CPU share
and DSpark can trade expert-cache VRAM against CPU/PCIe work, so they require
our own paired trials. This is another external comparator / architecture
reference, not a measured Venturi improvement. Do not download or convert a
second full copy into the nearly filled root SSD; retain the same hash-pinned
source assets and use roomy storage for any additional converted artifact.

The verified downloader retains its old 8 GiB per-file admission default. Large
GGUF shards/Engram require the explicit `--max-file-gib 128` option, full SHA256,
private resumable partials, disk admission and bounded network recovery.
Direct HF HTTPS resets were observed. A public-mirror exact-range probe passed
with the mirror hostname resolved to `160.16.86.14`. The downloader can now use
an explicit mirror transport and per-connection DNS route, while retaining the
original HF manifest, full SHA256 and TLS hostname verification. It changes no
system DNS or TLS settings. The first remote attempt exposed a Python 3.12+
removed `check_hostname` keyword; this was fixed by carrying verification in an
SSLContext. The failed attempt is retained and the v3 helper resumed the same
private partials. Do not report a launched job as a completed checkpoint.

### Lightweight metadata/runtime startup

The package root now lazily resolves its existing public checkpoint API. Importing
The prior import qualification of `venturi.model_assets` and the now-removed
`venturi.backends.strata` did not import Torch or
safetensors as an incidental package-initialization side effect. Explicitly
requested checkpoint symbols retain their original object identities.

Local Mac M5 experiment: fresh Python processes, warm OS file cache, one warmup
and three alternating before/after pairs, identical metadata module and Python:

| Metadata import only, median | Eager root | Lazy root |
| --- | ---: | ---: |
| Wall time | 468.0 ms | 6.7 ms |
| Process CPU time | 2.812 s | 0.0067 s |
| Peak RSS | 199.3 MiB | 19.8 MiB |

Evidence: `import-startup/paired-results.json` in the operation capsule, with
before/after root snapshots. These numbers are **not** GPU/model startup, model
weight RAM, inference latency or TPS improvements. Target-server import timing
has not been measured. Latest full local suite: **1359 passed, 2584 skipped**;
CUDA skips are not GPU correctness passes.

### Hyper-connection residual correctness foundation (October 27)

`models/hyperconnections.py` adds explicit CPU semantic references for flattened
stream RMS projection, pre/post coefficients, finite-iteration Sinkhorn and
residual collapse/expansion. Two implementation traps are covered: the residual
matrix is **source -> destination** (not conventional left multiplication), and
each sublayer consumes the **previous** sublayer's pre coefficients, not its own.
The initial collapse selects stream zero, not a mean of streams.

Seventeen CPU tests include independent scalar float64 Sinkhorn equations and
FP32/FP16/BF16 residual dtype checks. Hash-bound qualification executes selected
official pure-Torch methods and block ordering on fake sublayers: 57 checks pass,
including a retained counterexample for incorrect non-lagged ordering. Projection
qualification injects our CPU coefficient split; it does **not** execute or
independently verify the official TileLang Sinkhorn GPU kernel. No real weights,
attention/FFN sublayers, native GPU path or full-model correctness/TPS are claimed.

The external protocol adapter now accepts and retains the pinned engine's `PP`
prefill progress lines (including duplicate final progress). Previously it would
reject even a valid short prompt's final progress line. Malformed progress,
prompt-count mismatches and progress after output tokens fail closed. Tokenizer
Python source must now match an explicit SHA256 before execution. These are
CPU protocol fixtures, not measured GPU improvements. Latest full local suite:
**1387 passed, 2584 skipped** (`full-cpu-regression-v6.log`).
After source synchronization, **141 focused tests also passed on School_4090D**
with CUDA hidden from that test process. This does not exercise the GPU or model.

### Download continuation detail

The v3 mirror campaign made real partial progress but hit HTTP 429 limits.
The v4 helper was launched and its live process/partial progress confirmed at
00:42 on October 27. It uses two workers, 64 MiB ranges and largest-file-first
scheduling. Its optional in-memory HTTPS redirect reuse avoids a new mirror
metadata lookup for every range; expired/range-bound redirects return through
the same pinned source. Signed URLs are never logged or persisted; exact ranges
and complete-file SHA remain mandatory. No full backbone shard had been verified
at that observation. The progress states include resumed offsets, so summing
them and dividing by the new process's elapsed time is **not network throughput**.
Use two timestamped full-directory byte snapshots for progress rate.

At 01:15 a controlled v5 continuation was confirmed alive with four independent
file workers, preserving all sequential partials and full-file hash requirements.
Engram remained incomplete: one connection per 111 GB lookup file is itself a
critical path. The tested v6 downloader therefore adds **opt-in bounded parallel
range prefetch within each file**, with a single ordered writer/hash consumer.
Out-of-order completions live only in bounded RAM buffers; the `.part` file
remains a contiguous prefix, never a sparse file with a misleading full length.
Tests cover forced out-of-order arrival, failed ranges, safe resume and full
hash publication. Numeric HTTP Retry-After is honored with a bounded delay.
Defaults remain one connection per file. A v6 launch is not evidence of a complete
download; consult live process/progress and the canonical complete marker.

The original source downloader had a separate defect: its exported bash function
did not inherit `errexit`, so failed 900-second curl/tar commands could still
create source-complete markers. On this link an archive needs longer than 900 s.
The independent source bootstrap now downloads both pinned archives with a
deadline-bound longer transfer and exact archive SHA256, then builds from **fresh
`-verified` source/build directories**. Never trust the old source markers or
merge their incomplete trees into a successful source build.

The first long source transfers subsequently both failed with curl code 92
(HTTP/2 stream errors). The v2 source bootstrap explicitly uses HTTP/1.1,
retaining the failed partials/logs and the same complete archive hashes. Its
launch is not a successful build. A separate 8.5 MB transport fallback contains
2,333 byte-identical build/tool/license files selected from the two **locally
SHA-verified pinned archives**, with its own whole-bundle and per-file hashes.
It is not a full Git checkout. Its extractor was locally checked for new files,
exact reuse and refusal to overwrite a modified source. That source bundle's
SSH transfer is also incomplete; do not build from its partial bytes.

The two recovery builders now have a shared build-admission lock and accept only
source-qualified completion markers plus an executable hash, not the original
empty legacy hash file. Synchronize their latest local versions before starting
an alternate build; do not duplicate a live download/build. At 01:47 a v7 download
configuration was dispatched with eight prefetched ranges per file (same tested
v6 helper, four files, 2 GiB nominal range buffers). SSH broke before its final
process check: inspect its PID, the root lock and live progress before treating
it as active or attempting any restart. No network-rate improvement or inference
TPS is inferred from this dispatch.

### Root storage failure and containment (02:00 onward)

The v7 download had already terminated when inspected at 02:00. Its log reports
local `EIO` followed by `EROFS`; no assets were published as a complete model.
The SSD's existing partials were left untouched, not bulk-read/copied/deleted.
The filesystem reports `rw,emergency_ro` even though `statvfs`'s `ST_RDONLY` flag
is clear, with about 295 GB still free: checking free space or that flag alone
would miss this failure. Kernel records show `sda` `DID_BAD_TARGET` read errors.
SMART/journal utilities themselves could not be executed because of root I/O
errors, so no SMART health or exact physical cause is claimed.

Read-only diagnosis used the existing HDD Python and kernel READ_ALL operation,
without clearing logs. `/home` (`sdb1`) remains mounted writable, its ext4 error
count is zero and a **4 KiB write/fsync/read probe passed**. This is not a SMART
test or proof of hardware health. All owned staging controllers were already
gone by containment; no production process was signalled. The original sidecar
remained HTTP 200, same model/context, worker alive, PID/start identity unchanged.
The two-V100 Qwen production route was not touched.

The downloader now separates **local storage faults** from **source transport
faults**. It detects `emergency_ro` before attempting writes, probes only its
selected output mount, cancels queued work on local I/O failure, avoids network
retries for that failure and prints the failure report off the damaged output
instead of repeatedly trying to save it there. A source/network `EIO` remains a
bounded transport retry, not a falsely labelled disk fault. Out-of-order range
fetching, contiguous partials and full-file SHA requirements are unchanged.
New CPU cases reproduce the mount-flag trap, write failure, cancellation and
error classification. **52 downloader tests pass**; full local suite:
**1395 passed / 2584 skipped**, not GPU or full-model validation.

A separate fresh HDD staging controller was confirmed live at 02:29 with the same immutable
manifest, existing conda CA bundle/TLS verification, no reads of old SSD partials,
CUDA hidden from its process and an independent absolute **08:00 cutoff**. Its
output is under `/home/test/codex-models/`, not the failed root volume. This is
asset preparation for later qualification, **not** a running inference model or
a repaired machine. Confirm its live PID/progress from `status.json` before
starting anything else. Do not remove the GPU halt marker automatically.

The original direct build and later mirror download have independent results.
A failed original `stage-result.json` does not imply a later resumed download
failed: require the canonical `.venturi-complete.json` and verified external
binary independently. Local source-archive transfer recovery also exhausted
bounded routes; do not infer the original remote build stopped when SSH failed.

## Continuation

Read `status.json` and live remote state first. `start_stage.py` records finite
same-host SSH recovery and source rsync attempts. Transport can fail transiently:
IPv4, the observed same-host Tailscale IPv6 address, and Tailscale `nc` have been
tried with host-key checks intact. Do not repeatedly launch jobs because a log
fetch times out. All stages and results must keep `native_deepseek41_supported`
and `full_model_tps` honest until real implementation/measurements exist.

### Actual container layout, before any GPU admission

A bounded, Torch-free GGUF v3 inspector now checks counts/string/array/header
budgets, block-aligned rows, canonical non-overlapping offsets and declared-file
bounds without reading tensor payloads. Header-only and full-length files are
never labelled SHA-verified weights. The standalone `tools/inventory_gguf.py`
can record the same explicit bundle inventory; full SHA scanning is opt-in.

On healthy HDD data, all four backbone directories (1,008 source tensors), both
Engram directories and the 78-tensor DSpark directory matched the pinned Strata
parser's tensor names/types/dimensions/offsets. Missing directories were obtained
as **separate diagnostic prefixes**, not inserted into the active download or
published as loadable weights. Their payloads remain unverified. Only the small
5,780,589-byte metadata-only source was fully fetched and SHA-verified separately.

The first comparison of that zero-tensor metadata file retained an offset
counterexample: Strata rounds its unused data pointer even when no tensor blob
exists; the pinned gguf.cpp uses the unpadded header end. Our inspector follows
the latter. This is not a corrupt checkpoint or a model-quality failure; the
metadata-only file has no tensor payload. Further target cross-comparison was
interrupted by the SSH session failure below, not silently marked passed.

**Advisory declared raw tensor bytes from unverified-payload headers:**

| Backbone storage category | Bytes | GiB |
| --- | ---: | ---: |
| 120 routed expert tensors | 114,555,617,280 | 106.688 |
| Other non-integer tensors | 6,159,667,648 | 5.737 |
| Engram integer hash metadata | 517,952 | 0.00048 |

The nominal IQ1_M backbone actually mixes **IQ1_S, IQ1_M, IQ2_XXS and IQ2_S**
experts; other matrices include Q5_K/Q8_0/BF16/F32. Do not assume a uniform bit
width or infer runtime VRAM from this table: the engine may convert individual
norm/router/embedding storage and needs caches, KV and workspaces. The two lookup
tables have 256-value IQ4_NL rows, **144 stored bytes per row**; their combined
110,595,290,400 tensor bytes are disk lookup storage, not resident VRAM. HDD
random-lookup performance is not an approved substitute for the failed SSD.

Pure CPU header geometry qualification constructed **48,488 logical expert and
output-group slices** across the six weight-bearing files. Exact byte coverage
passes; missing, duplicate and wrongly typed slices are retained rejections.
This creates no executable pack and validates no tensor values or model outputs.
Evidence: `hdd-header-inventory-v1/v2.json`, `header-slice-qualification-v1.json`.

The external harness previously checked only that pack spans fit a source file;
a truncated/skipped model could pass that shallow check. It now binds each slice
to its actual GGUF tensor's dtype, dimensions and row stride, rejects overlaps,
requires complete source coverage and all backbone/lookup files, and requires
the draft asset when speculation is selected. Full SHA and independent real
model correctness are still separate gates. The token adapter also completes
partial pipe writes with a deadline and reaps an engine that stops reading
commands; neither change is an inference-speed claim.

Both upstream source archive `.direct-v2.part` files on HDD were found complete
and **exact-SHA-verified using HDD Python** at 02:36. Their shell bootstrap's code
126 came from the unreadable root `sha256sum`, not wrong archive hashes. No build
was started and no qualified external binary exists.

### SSH session-layer failure, 02:51 onward

After earlier successful HDD-only work, both same-host IPv4 and IPv6 read-only
continuations timed out. One finite fresh SSH debug probe reached TCP, verified
the unchanged host key and **authenticated with the existing key**, but received
no session-channel confirmation before its bounded keepalive timeout. This is
not evidence of a DNS/key/host-identity problem. Its exact cause is unconfirmed;
root SSD failure is already independently established. Do not switch off host
verification, repeatedly retry the same failing session, or relabel the host as
repaired. Target jobs may survive SSH loss: the HDD supervisor has its independent
08:00 cutoff. Last confirmed download processes and sidecar health are timestamped
in `status.json`; do not claim a later live/healthy state without a new read.

### Engram CPU semantics while GPU admission is halted

`models/engram_residual.py` adds a CPU oracle for **already projected** keys and
values: normalization per residual copy, the product of query/key weights,
signed floored sqrt before sigmoid, shared-value expansion and masked identity.
It does not implement table lookup, dequantization or the large wkv projection.
Sixteen independent/admission tests pass, and **54 synthetic prepared-key/value
cases are byte-exact against the pinned official `Engram.forward`** with those
upstream stages replaced by fixtures. Zero dot is deliberately not an exactly
half gate because its magnitude is floored before sqrt. This is not a full
Engram or model-quality check.

`models/engram_hash.py` adds a separate explicit-map CPU hash oracle. Immutable
parameters are shared; each fixed-batch state owns only N-1 compressed tokens per
sequence, supports private forks/reset, rejects discontinuous positions and
prevents masked/image spans from leaking into earlier n-grams. It consumes
supplied map/moduli/multipliers rather than silently regenerating normalization,
primes or RNG parameters. Twenty-four tests cover hand hashes, chunking, aliases,
compressed pad IDs, masks, state isolation, limits and overflow admission.
**72 synthetic cases match selected official `NgramHashState.forward` byte for
byte**, including chunk boundaries, batch two, 24 hash columns and int64-limit
products. This checks hashing/carry only, not actual tokenizer/map identity,
prime/RNG generation, row retrieval, GPU or speculative decoding.

The new pinned `inference/engram.py` was recovered via the already authorized
public mirror after direct HF TCP timeouts, with TLS intact. Its 8,138 bytes
match the original source API's Git blob `a36bc6c6b8cbf2d3dbac850863c23b55a7ecb5c7`
and SHA256 `11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897`.
No target SSH or production process was required for that independent work.

The mHC qualifier now includes the actual checkpoint RMS epsilon **1e-20** as
well as the prior generic 1e-6 fixture: **111 selected pure-Torch/projection/order
checks pass**. A tiny-magnitude counterexample demonstrates that an adapter must
bind the checkpoint epsilon rather than assume the generic oracle default.
The Sinkhorn split is still the injected CPU reference, **not** qualification of
the official TileLang GPU kernel. At this stage the full local suite had **1517 passed,
2584 skipped**; these are not target GPU passes or TPS measurements. New engine
source changes remain local pending a working target session and later protected
qualification; earlier 141 remote CPU tests do not cover these later changes.

### Verified integer metadata -> hash spec binding (03:30)

`venturi.gguf_metadata.read_verified_integer_metadata` now handles the small
integer tensors separately from quantized projection weights. It requires an
explicit `ModelAsset`, exact full length and **full source-file SHA256**, then
reads only named, bounded I8/I16/I32/I64 values. The scan and reads share a file
descriptor, the parsed header hash must match that descriptor, and before/after
fingerprints detect concurrent source changes. Header-only or partial evidence
is rejected. Verification necessarily scans the entire containing asset (about
39 GB for the real first shard); this is an explicit offline operation, **not**
a decode path or an instruction to read the failed SSD. No real target weight
scan was attempted during the storage incident.

`bind_engram_hash_metadata` checks the actual IST directory geometry:
I32 token map `[vocab]`, I64 primes/offsets `[layers,(N-1)*heads]`, and I64
multipliers `[layers,N]`. It validates declared vocabulary/row counts, cumulative
offsets, compressed IDs and int64 product bounds without regenerating parameters.
The caller must still independently authenticate architecture/tokenizer identity
and layer order; the binder does not certify those supplied configuration fields.
There is no table lookup/dequantization or full-model binding yet.

A second qualification passes generated integer-only GGUF fixtures through the
**full-SHA reader -> binder -> hash/carry -> selected official forward** chain:
72 comparisons are byte-exact. These tiny fixtures contain no real embedding or
projection weights. Valid-file-SHA but wrong dtype/shape/offset/configuration,
same-length corruption, concurrent change and unverified-header admission are
retained rejection cases. Latest full local suite: **1551 passed / 2584 skipped**
(`full-cpu-regression-v12.log`); no target GPU or TPS result follows.

The bounded 03:18 HDD/proc status read again timed out at SSH session level.
No fresh download or service health result was obtained; the last confirmed
02:36 identities and the independent 08:00 cutoff remain the authoritative state.
No job was duplicated, service signalled, root scan performed or source built.

### Config/tokenizer identity and IQ4_NL CPU preparation (04:25)

The higher-level `models/engram_bindings.py` closes the **caller-supplied
configuration** gap for the hash component: explicit trusted manifest assets,
full-SHA JSON, exact architecture/revision, verified integer header layer order,
width/copy/head/pad/vocabulary fields, table-row counts, and exact per-token
stored-map comparison. It never repairs a mismatching stored map. It retains
the config epsilon `1e-20` and its GGUF F32 representation
`9.999999682655225e-21` separately; a loose absolute tolerance would incorrectly
accept `1e-6`. The manifest itself remains a trust input, not proof of authorship.
This component returns `model_supported=False`: it does not authenticate/load
the still-unloaded sparse table/projection weights or enable native generation.

`gguf_metadata.py` now also decodes only explicitly named bounded integer header
arrays after full containing-asset SHA verification, rather than treating an
array summary as layer-order evidence. `model_config.py` rejects duplicate JSON
keys, nonfinite numbers, symlinks, nonregular files and source changes. Generated
tiny GGUF/config/tokenizer fixtures cover the complete component binding path,
including valid-SHA but semantically wrong metadata and same-range wrong maps.
No real 39 GB shard was scanned on the failed SSD or from a partial download.

The actual pinned **public tokenizer JSON** (6,367,257 bytes) was fetched locally
over the existing verified-TLS mirror route, checked against the retained
official API's Git blob `6a15814dd25c934028034531da744c689f87ff21`, and assigned
SHA256 `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`.
It is a small CPU reference asset, **not** another model-weight download.
`qualify_deepseek41_tokenizer.py` compares only the SHA-pinned official
`build_compressed_token_map` function: synthetic Unicode/whitespace cases,
partial UTF-8 byte-token decoding, and **all 129,280 actual raw token IDs** agree
exactly, producing **99,092 compressed IDs** as declared in the authenticated
official config. Computed I32 map digest:
`c60a86322ec17b4142bfef3c57a8d81fb428550cdf487f88a4320cb59fe46481`.
This has **not** been compared to the real GGUF's stored token-map tensor.
It qualifies raw normalization, not chat templates, image processing or quality.

`gguf_quantization.py` adds an explicitly scalar **CPU IQ4_NL row decoder**.
The nonlinear codebook is not signed/uniform int4 or the Flash-Next Q4 format:
32 values occupy 18 bytes, with the low nibbles forming columns 0–15 and high
nibbles columns 16–31. At width 256 that is 144 bytes/row, matching the retained
Engram header geometry. Private float32 results preserve finite negative,
signed-zero and subnormal F16 scales; nonfinite scales and incomplete/unbounded
rows are rejected. No table file is opened or materialized by this function.
Against only the selected dequantizer in the pinned llama.cpp Python source,
**all 63,488 finite F16 scale bit patterns** and all 256 nibble pairs are
byte-exact, including width-256 rows. This is a synthetic format oracle, **not**
real embedding values, storage row retrieval, a tuned kernel or a TPS result.

Evidence: `tokenizer-reference-fetch-v1.json`,
`engram-tokenizer-qualification-v1.json`, `cpu-components-v2.log` (117 passed),
`iq4nl-reference-source-v1.json`, `iq4nl-cpu-qualification-v1.json`, and
`full-cpu-regression-v14.log` (**1650 passed / 2584 skipped**). Skipped GPU tests
are not GPU passes. All these implementation changes remain local;
the earlier remote CPU tests do not qualify them on the target.

The one bounded 04:00 HDD/proc read was closed by the same verified target SSH
endpoint before any state returned. There is **no** updated HDD completion or
sidecar health observation. No replacement task, GPU experiment, build, repair,
production signal, failed-SSD scan or GitHub push was performed.

### Bounded CPU row store and hash-to-table component (05:12)

`gguf_rows.VerifiedIQ4NLRows` now provides an opt-in **CPU storage component**:
explicit table schemas, exact full source SHA/header/fingerprint admission, one
held read-only descriptor, and bounded addressed `pread` calls thereafter. The
entire asset is scanned once on opening; this is expensive offline admission,
not a cheap per-token check or a measured cold-start time. It must never be
pointed at the failed SSD/partial target files during this incident. No real
lookup source was opened this turn.

Within one explicit `lookup_many` call, request IDs are snapshotted, duplicate
rows are removed across requests, adjacent addressed rows are coalesced, and
results are scattered back in the original shapes/order with independent
storage. Token/history states are **not** merged. Empty requests perform no
payload reads. Budgets include all requested output values, not just unique
rows. There is no persistent row cache, mmap, full-table materialization,
background prefetch or serving scheduler. Stats count successful payload bytes
and `pread` calls, **not** disk pages, physical I/O, GPU bandwidth or TPS.
Changed/truncated/EIO/nonfinite payloads fail closed and poison the lease;
successful short reads are completed with bounded forward progress. Source
descriptors close on constructor/context failures and explicit `close()`.

`models.engram_lookup.EngramIQ4NLLookup` binds that store to an already bound
Engram hash component and the **same explicit model bundle**. The table source
must declare `deepseek41-engram` with the same official revision; names
`blk.<layer>.engram_embd.weight`, row counts, IQ4_NL dtype and head width must
agree. It maps `[B,T,layers,hash_columns]` to per-layer
`[B,T,hash_columns,head_dim]` rows, supports cross-request row deduplication and
does not advance caller history. A caller may hash a private fork and commit it
only after all dependent I/O/execution succeeds. This is **not** complete Engram:
the real candidate's `engram_wkv` is Q5_K `[25600,6144]`, while q/k are BF16
`[4,5120]`; those weights/projections and native layer execution remain unbound.
`model_supported` remains false, and no production backend was redirected.

The generated-table qualifier compares the complete **CPU hash -> full-SHA
table admission -> addressed row decode/scatter** chain against two independent
selected primary functions: official `NgramHashState.forward` and pinned
llama.cpp IQ4_NL decoding. All **72 cases are byte-exact**, including batch 1/2,
chunk boundaries, masked spans, int64-limit hash products, two lookup layers,
duplicate requests, and the actual **24 hash columns / width 256** geometry.
The fixtures are only 65 KB / 650 KB, with synthetic parameters and table bytes;
they are not samples from a real model. No real checkpoint, quality, GPU or
end-to-end performance qualification follows from this result.

Evidence: `cpu-row-binding-v1.log`, `cpu-row-components-v2.log`,
`engram-row-qualification-v1.json` and `full-cpu-regression-v16.log`
(**1705 passed / 2584 skipped**, local CPU only). Negative fixtures retain
wrong-dtype uniform Q4 with identical byte geometry, valid-SHA but wrong source
revision/shape/layer mapping, admission/lookup races, EIO/short reads, output
aliasing, concurrent callers, nonfinite scales despite valid SHA, and no caller
history advance on failed lookup.

The single 05:00 same-host IPv4 status read, after hourly backoff, was reset
during SSH key exchange. No remote script state, download completion, service
health or repair evidence returned. There was no retry campaign, new remote job,
service signal, root payload access or GPU/source-build work. The last confirmed
02:36 observations and the independent configured 08:00 supervisor cutoff remain
unchanged; current target job/service liveness is still unknown.

### CPU Q5_K/BF16 formats and synthetic projected Engram continuation

`gguf_quantization.py` now also has bounded immutable-bytes CPU row decoders for
**Q5_K** (256 values/176 bytes, independent scale/min and high-bit planes) and
**BF16** (exact little-endian bit expansion into private F32 tensors). These are
not uniform Q5/int4, quantizers, checkpoint readers, matmul kernels or GPU ops.
Nonfinite scale/weight patterns are rejected, not silently replaced. Signed zero
and subnormal bit patterns are retained without changing process-wide denormal
settings. Default output budgets prevent accidental whole-weight expansion.

Against the selected SHA-pinned llama.cpp Python methods, the Q5_K check tests
**63,488 finite F16 scale bit patterns in each of d and dmin independently**
(126,976 blocks total, not every possible pair of scales); bounded width 6144
rows also pass. BF16 tests all **65,280 finite bit patterns** exactly. All 4096
nonfinite Q5_K slot patterns and 256 nonfinite BF16 patterns are rejected.
This is binary format correctness on synthetic bytes, not the accuracy/quality
of the actual lossy checkpoint.

`models.engram_projection.reference_engram_from_rows` provides a separate
explicit **dequantized-F32 CPU** projection/gate oracle: flatten only the hash
column/head-embedding axes, WKV `[ (copies+1)*hidden, columns*head_dim ]`, first
`copies*hidden` outputs as keys and last `hidden` as the shared value. Query/key
weights, norm epsilon, masks and budgets are explicit; outputs preserve stream
dtype and inputs/history are not mutated. It builds no autograd graph. This
does **not** emulate BF16/FP8/fused GPU accumulation or choose runtime precision.

The extended generated-table qualifier now compares **hash -> full-SHA IQ4_NL
rows -> decoded Q5_K WKV/BF16 q+k -> F32 projection -> Engram gate/residual**
against independent selected official hash/Engram methods and pinned primary
dequantizers. All **72 cases are byte-exact**. The larger fixture uses the actual
24-column/256-wide row geometry but a deliberately tiny **hidden size 8**, not
the real 5120: its WKV is `[40,6144]`, not `[25600,6144]`. Real lookup/body weights
remain unopened/unbound. No native layer, real model quality or TPS is claimed.
The updated evidence includes exact implementation source hashes; earlier
qualification capsules are preserved rather than rewritten.

Evidence: `projection-formats-qualification-v2.json`,
`engram-row-projection-qualification-v3.json`, `cpu-projection-components-v3.log`
(70 passed), and `full-cpu-regression-v17.log` (**1751 passed / 2584 skipped**).
These are local CPU checks; the target GPU remains unqualified and the new code
has not been synced/built there. Native mixed IQ experts, actual weight binding,
MLA/compressors/indexer/candidate state and full runtime integration are still
required before honest whole-model generation/benchmarking.

The one bounded 06:00 same-host IPv6 minimal status read was again reset at SSH
key exchange. It returned no service/job/download state. No new remote job or
signal, root scan, repair, GPU test, source build or GitHub push followed.
`CLOSEOUT.md` in the operation directory records a bounded 08:00 containment
decision tree, including unknown remote liveness and no new full-file scan at
the cutoff. The HALT marker and original services remain untouched.

### Pre-cutoff operational preparation (07:00 hour)

The 07:03 bounded IPv4 read was closed before remote state returned; current
job/service liveness remains unknown, not failed or restored by inference.
No new target job, signal or GPU/build/repair work followed. An optional
**one-shot cutoff cleanup** tool was prepared locally with exact last-observed
supervisor/child argv and start ticks, fixed host/time/owner guards, Linux pidfds,
fresh identity rechecks and no numeric/name-wide kill fallback. Twenty-two local
**fake-process** tests pass; it has not been executed on the target. Invalid
poll events, re-exec/PID reuse, missing `/proc`, EIO and uninterruptible processes
must remain unconfirmed rather than being declared stopped. Original service
PID 1540787 is explicitly excluded. See `CLOSEOUT.md` and
`cutoff-helper-preparation-v1.json` for the bounded deadline procedure.

Latest full suite: **1773 passed / 2584 skipped** (`full-cpu-regression-v18.log`).
The extra 22 checks are operational safeguards, **not** more native inference
support or a throughput improvement. Real checkpoint completion, full native
architecture integration, actual generation/quality and target TPS are still
unestablished. All previous source/qualification evidence is retained; no new
large experiment or source refactor is needed immediately before the cutoff.
