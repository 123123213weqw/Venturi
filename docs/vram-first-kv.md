# Experimental VRAM-first native KV

Opt-in for complete resident native Qwen targets (single device or TP). The
legacy `static` path stays the default. MTP rollback and PP are rejected rather
than silently selecting a different execution path. This is KV storage, not
weight/activation offload and not a new paged-attention kernel.

```python
model = NativeQwenModel.from_q8(
    weights, backend, weight_mode="resident", dtype=torch.float16,
    # Existing device/TP/projection options still apply.
    kv_cache="vram-first",
    kv_gpu_budget_bytes=0,          # free VRAM minus reserve; optional hard cap
    kv_ram_budget_bytes=8 * 2**30,   # explicit spill backing quota; 0 forbids spill
)
batch = model.create_batch([65536] * 8)
print(model.kv_cache_info())
```

CLI additions to `venturi qwen-native generate`:
`--kv-cache vram-first --kv-gpu-budget-mib 0 --kv-ram-budget-mib 8192`.
All other required checkpoint/library/precision/device options remain explicit.
A large declared capacity is **not** evidence of a tested full-length prompt.

## Residency and admission

- CUDA VMM reserves contiguous virtual K/V ranges for each request/full-attention
  layer. Only used prefixes get physical GPU pages. Existing attention kernels
  and exact KV dtype/bytes do not change.
- Free GPU pages are pooled/reused. If the admitted prefixes fit in VRAM, no CPU
  spill backing is allocated. Optional GPU caps can force the pressure path for
  reproducible tests without deliberately causing a real device OOM.
- Under pressure, prepare bounded pageable CPU backing, then evict whole
  request/layer prefixes: inactive entries first, followed by active-entry LRU.
  Reload the current prefix before its GPU attention operation. Recurrent GDN
  state is never moved or replayed by this mechanism.
- Allocate GPU physical pages and CPU backing before each chunk/group's embedding/GDN mutation.
  A physical-page allocation OOM can select RAM fallback if a whole working
  prefix still fits; other CUDA errors are not treated as OOM. Never restart a
  partially executed recurrent forward. Non-OOM VMM driver errors fail closed:
  close/recreate the model, rather than reusing a partial mapping. Other execution failures poison the request
  until reset. A scalar `forward()` admission failure leaves its cursor/GDN/KV
  unchanged; generation/cohort wrappers may conservatively require reset even
  on admission failure (earlier chunks may already be committed).
- At least one whole `(request, full-attention layer)` K+V prefix must fit the GPU
  pool. Budget/quota admission rejects impossible contexts before computation.
  CPU backing quota includes temporary old+new allocation during growth.
- Reloads issue DMA through a private pinned two-slot ring and copy stream;
  compute attaches to each entry's completion event. Eviction completes the
  dirty prefix before unmapping. Full dirty-prefix
  copies, VMM mapping costs and PCIe bandwidth can make spill substantially
  slower. This is an OOM-avoidance path, not a throughput optimization.
- Reset releases CPU backing, detaches shared/read-only mappings and advances
  the entry's incarnation. Ordinary private mappings remain hot. Physical pages
  remain cached in the pool for reuse until model
  close. Existing borrowers retain their page references; resetting/reusing the
  donor cannot overwrite their KV or make an old key valid again.

## Accounting and limitations

`model.kv_cache_info()` reports model-wide physical GPU pool, mapped GPU bytes,
reserved virtual addresses, CPU backing capacity, logical valid K/V bytes,
copy bytes, evictions/reloads and OOM counts. Session/batch `info()` includes
these separately as `shared_model_kv_cache`; **do not add shared totals once per
request**. Existing request bytes exclude the separately reported tier pool.
CPU backing capacity is not process RSS, nor pinned memory. Virtual address
reservation is not physical GPU allocation or NVML usage.

Query the device's actual VMM granularity; on the tested V100 it is 2 MiB per
plane, so even a short request/layer prefix needs 4 MiB K+V. This can use MORE
VRAM than a tiny eagerly allocated static cache, while saving substantial VRAM
when declared context capacity is large. RoPE tables, GDN caches, workspaces and
weights are still eagerly resident. Non-KV allocation failures do not trigger
weight offload. Pool memory is retained (no background global VRAM reclamation).

Qualification and measured results belong to the independent evidence capsule
`benchmarks/v100-vram-first-kv/`. No claim of full 64K/256K-context speed or quality
is implied by this feature or the capacity-allocation checks.

Existing requests/batches retain native model/KV ownership after the public model
handle closes. New requests are rejected. Request `info()` then marks model-wide
KV diagnostics unavailable instead of querying the freed model handle or claiming
zero cache bytes. Explicit model diagnostics and model close share a lock.

## Prefix sharing (unified KV layer orchestration, model-side scheduling)

The unified KV layer (`ops/csrc/kv/kv_prefix.cuh`, `PrefixManager`) owns the
entire sharing orchestration: prefix records, freezing of the donor's
auxiliary recurrent state into registry-owned device snapshots, read-only
page mapping into borrowers, LRU capacity with active borrowers pinned, and
lifetime after the donor closes (reference counts keep shared pages alive).
Models only schedule: which sessions publish/borrow, which KV entries and
recurrent buffers participate, and where decoding resumes.

```python
donor = model.create_session(capacity)
donor.forward(shared_prefix_tokens, all_logits=True)
donor.publish_prefix(key=7, tokens=len(shared_prefix_tokens))

later = model.create_session(capacity, prefix_key=7, prefix_tokens=len(shared_prefix_tokens))
# decoding resumes at the prefix boundary; full-attention layers reuse the
# donor's physical KV pages read-only, GDN recurrent state is restored from
# the frozen snapshot, and results match a full prefill bit-for-bit.
```

Keys are explicit caller-chosen integers; automatic token-prefix detection is
future work. Sharing requires the VRAM-first tier enabled before requests and
is rejected (fail-closed) on missing keys, lengths different from the donor's
exact committed position, failed donors, mismatched borrowed lengths/layouts,
non-fresh borrower entries, or the head-GDN two-device profile.
Borrowers that later spill create their own host copies; shared pages
themselves are never written (hardware read-only mappings).

### Lifetime and boundary contracts

- Publish only at the donor's **current healthy committed cursor**. Earlier
  GDN states cannot be reconstructed from its current buffers; passing a shorter
  prefix is rejected, not silently paired with a newer recurrent state.
- Borrow validates the recorded byte length, layout, fresh destination capacity
  and every auxiliary destination **before any mapping or state restoration**.
- Records bind donor incarnation as well as address: reset, destruction and
  allocator address reuse cannot revive an old prefix. Donor append preserves
  old prefix bytes; both donors and borrowers COW based on physical references.
- Live entries hold type-erased reference-counted publication leases. Their
  reset/destruction releases LRU protection without pool-to-manager callbacks;
  the lock order remains manager then pool. Republishing uses a new lease.
- Snapshot ownership is RAII. A rejected full registry allocates no snapshots;
  allocation/copy/replacement failures retain the old record and free new state.
- Admission counts shared physical pages once and reserves only append write-set
  COW copies. A last remaining alias can become writable without an extra page.
  Complete untouched prefix pages stay shared after the first decode append.
  The ordinary non-sharing admission path remains O(entries).

Boundary-page COW still uses a bounded synchronous host bounce (one affected
page per plane for a small append), **not a new D2D/async COW implementation**.
These bytes are included in the D2H/H2D counters. Native CUDA tests and independent
same-condition measurements are required before claiming throughput benefits;
host-only POSIX page-alias regressions are not CUDA numerical validation.

Auxiliary prefix snapshots consume additional device memory outside the K/V
physical-page pool counters and `kv_gpu_budget_bytes`. Record count is bounded
(default 64), but there is not yet a separate snapshot-byte quota. A snapshot
allocation failure preserves the old record; it does not select another engine
or infer that the entire model/context fits within the KV-page budget.

### Prefix-fix qualification (2026-11-16, RTX 4090 D)

`benchmarks/kv-prefix-sharing-fix-4090d/validation.json` records the fix against
PR #11 head `490bd31459fcb9b014ba0bc26c83797abbcff8b4`, source/build hashes,
raw repetitions, separate profiler traces and sampled GPU diagnostics.
The local CPU suite passed **1888** tests with **2902 skipped** (GPU/asset-gated
coverage is not implied). The final Star_4090D run passed **24**, with no skips:
7 CUDA KV-mechanism tests, 9 generated hybrid-Qwen end-to-end tests, 7
ABI/split checks and 1 CPU numeric-checker regression. Qwen used Q8 fixture
weights, FP16 native/KV execution and F32 recurrent state. Shared-prefix and
full-prefill paths were checked for finite, byte-exact logits/state, not merely
plausible text. These are **not original 27B checkpoint quality tests**.

The append microbenchmark uses the device's 2 MiB VMM granule, a prefix of
`8 * granule + 32` bytes **per plane**, and an 8-byte append per plane. Setup,
publication, borrowing and full-byte validation are outside its C++ critical
timer. Each arm has a fresh warmup and three unprofiled matched repetitions in
AB/BA/AB order. Profiling is a separate invocation, including setup/validation.

| Final GPU 2 mechanism measurement | PR head baseline | Fix |
| --- | ---: | ---: |
| Changed physical pages, K+V | 18 | 2 |
| Physical KV pool | 72 MiB | 40 MiB |
| COW bounce bytes, both directions | 72 MiB (derived) | 8 MiB (recorded) |
| Critical-region wall median | 865.870 ms | 96.572 ms |
| Critical-region process-CPU median | 868.573 ms | 96.912 ms |

The final run used idle **GPU 2** because GPU 0 had become occupied. Other
tasks were left running. Earlier-source runs on GPU 0 measured much lower
absolute latencies (baseline 20.960--22.206 ms, fix 1.866--2.081 ms); retain
these separately, **do not pool them or claim a stable cross-run speedup**.
The physical-page/copy-volume reduction is verified, but device/shared-host
effects behind the latency difference have not been isolated. The unchanged
private-reset fast path was rebuilt and retested in the final run.

CUDA was built remotely with the existing CUDA 12.9 compiler for SM89; the
default CUDA 11.5 compiler's C++20 failure is retained in the external evidence.
No service, driver, firmware, register or profiling-privilege change was made.
NVML samples cover whole campaigns, not individual short COW intervals.
Achieved SM occupancy, measured DRAM bandwidth, original-model TTFT and
whole-model TPS are **unmeasured**. D2D/async COW and a separate auxiliary
snapshot-byte quota remain follow-up work, not implicit parts of this fix.
