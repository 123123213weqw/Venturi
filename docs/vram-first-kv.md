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
- Initial copies are synchronous, exact-byte, through the model's existing
  pinned staging buffer; there is no claimed async overlap. Full dirty-prefix
  copies, VMM mapping costs and PCIe bandwidth can make spill substantially
  slower. This is an OOM-avoidance path, not a throughput optimization.
- Reset releases CPU backing and invalidates logical KV. Physical pages remain
  cached for reuse until model close; pages above a reset prefix are unmapped on
  the next admission. There is no allocation/release oscillation per token.

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
