# Bounded dense projection staging in resident Qwen

The canonical row-codec path still uses the same decoded FP16 values, tile
order, cuBLAS M/N/K and FP32-pedantic compute. No new Q4/Q5/Q6 arithmetic is
introduced. The three changes are independent:

1. Allocate decode scratch for max(min(N,8192)*K) over the actual matrices
   using chunked dense, not 8192*max(K) across unrelated shapes. Output
   scratch is similarly sized to the largest reached tile, and unused
   embedding staging is not allocated. This reduces the default reservation
   without changing any execution tile, logits, or KV capacity.
2. Optional direct output permits a GEMM to write into the final activation
   if its existing ldc works: single output tile, or one activation row.
   The output address must have the same 256-byte alignment guarantee as
   the compact scratch. Other cases keep the staged copy, since changing
   ldc or rounding would require separate qualification.
3. Optional decoded weight tiles save repeated compressed reads and FP16
   expansion writes. Reserve the actual request slots and static KV before
   allocating these immutable tiles. Fill in deterministic trunk traversal
   order, then head if budget permits. Store the same independently decoded
   chunks with equal address alignment in one bounded buffer. No eviction,
   per-step allocations, on-the-fly repacking, kernel retuning, or heuristic
   claim that these are the hottest shapes. Uncached tiles use old staging.

```python
batch = model.create_batch([8192] * 4)
batch.prepare_dense_io(direct_outputs=True, shadow_budget_bytes=96 * 2**20)
print(batch.dense_io_info())
```

The opt-ins are single-device, resident FP16, static KV and chunked dense
only. Prepare at most once per model before its first row-codec projection,
after creating every required slot/batch. A regular inference path with no
preparation adds no shadow and keeps the original output route. Admission
preserves the model reserve guard; fewer bytes than requested may be used
if the next eligible tile would not fit. The existing scratch and encoded
planes remain resident: shadows do not reduce the remaining FP16 GEMM read
or replace compact weights. Never assume a large FP16 cache will improve
total memory. Keep any larger cache outside a production KV budget unless
the real workload qualifies it.

Counters cover the model's physical workspace, plan bytes, setup time,
explicit copy payload, unpack payload and cached tile hits. They are not
hardware traffic or achieved occupancy. A 96-MiB candidate is described as
memory neutral only when the measured plan uses fewer bytes than the actual
legacy-to-exact-workspace saving. An Nsight capture is separate from
unprofiled service scores. No cross-GPU/codec promotion is implied from SM70
Q5 results. The fixture includes a multi-tile vocabulary head, for which
multi-row ldc-changing writes must remain on the old branch.
