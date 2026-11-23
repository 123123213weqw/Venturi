# Experimental packed row tiles on SM70

The shared projection pipeline implements `X[M,K] × W[N,K]ᵀ` for
Venturi's signed, row-split Q4/Q5/Q6 G64 with FP16 scales. The
compile-time reader changes the high-plane width and sign extension;
the register/SMEM staging, accumulation and output store are shared.
It does not implement other Q4/Q5/Q6 storage layouts such as GGML K quants.

- **Grouped GEMV:** a warp decodes one weight pair once per loop and
  uses it in multiple activation-row accumulators in registers.
- **WMMA:** blocks stage 64 K values at a time. Each warp decodes its
  16 output channels into shared memory; a loaded weight fragment can
  feed multiple 16-row activation fragments. One warp-wide MMA can
  combine rows from different requests in the same scheduler step.
  Separate blocks can still reread the same compressed weight.
- Neither route writes an intermediate FP16 weights matrix to global
  memory. The original workspace remains allocated for dense fallback.

Build the paired SM70 ops and Qwen libraries. For an ordinary local
resident FP16 batch with static KV, opt in before the model's first
projection:

```python
batch = model.create_batch([8192] * 8, resident_dataflow=True,
                           attention_execution="descriptor")
batch.prepare_shared_tiles(direct_outputs=True)
print(batch.dense_io_info())
```

The pilot uses grouped GEMV at `M≤4` and WMMA above it, only for
compatible chunked-dense matrices with `N≥4096` and unpadded logical K.
`M` counts activation rows, not requests: mixed prefill and decode use
the same selector. Other shapes keep the existing FP32-pedantic cuBLAS
path. This is an initial shape heuristic, not a guarantee about every
matrix or workload. The operator also has masked edges for arbitrary
logical K / partial M and N tiles; runtime admission is narrower.
No additional model format or device is admitted by this feature.

The default and the opt-in for exact Q5 prefill reuse are unchanged.
Use one numeric mode per model; do not combine this with persistent
FP16 weight shadows, exact Q5 prefill reuse, or PD handoff. Both tile
routes round weights to FP16 and accumulate in FP32, but they reorder
floating-point operations. Results are not bitwise interchangeable
with the canonical dense path, and greedy agreement on one fixture
does not qualify serving quality.

The telemetry exposes counts of the two tile routes and the logical
size of the FP16 weight writes avoided at the projection boundary.
Use CUDA profiling to inspect actual launches. Logical byte counts
do not measure hardware DRAM traffic or Tensor Core utilization.

Run the explicit operator oracle and native mixed fixture on SM70:

```sh
VENTURI_TILE_OPS=/path/to/libventuri_ops.so \
VENTURI_TILE_QWEN=/path/to/libventuri_qwen.so \
PYTHONPATH=src python -m pytest -q tests/test_sm70_shared_row_tiles.py
```
