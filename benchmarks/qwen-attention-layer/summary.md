# Qwen full-attention layer and original-precision correction

## Scope

Native SM89 operator composition for a Qwen full-attention decoder layer, plus
preservation of original FP32 GDN gated gamma. These are layer/helper results,
not trained 27B forward, engine TPS, paging, serving or SM70 qualification. The
subsequent hybrid integration and current source capsule are recorded in
`../qwen-hybrid-model/summary.md`.

## Implementation

- Per-head packed `[Q D | output gate D]`, zero-centered Q/K RMSNorm, partial
  split-half text RoPE, causal GQA, rounded sigmoid output gate, output/residual
  and the post-norm MLP/residual.
- Default text-only equal-axis mRoPE. Frequency-axis interleaving does not change
  channel pairing into even/odd rotation. Cast boundaries and rotary products
  are explicit; norm-only execution permits independent stage qualification.
- Common dense cache owner/stream/length/poison/reset lifecycle, pre-bound tail
  views, one activation arena and fixed-position graph capture. Not paged KV.
- Real source preflight found 48 FP32 gated-norm vectors, unlike the legacy
  BF16 artifact. The initial export failed before publication. The corrected
  converter/binder retains FP32 gamma; multiplication promotes to FP32 without
  an extra weighted BF16 cast. Activation-typed legacy gamma remains separate.
- Blocks **ABI 8**, library/source hashes in `build.json`. Older ABIs rejected.

## Evidence

| Environment | Scope | Result |
|---|---|---|
| Local Mac | Full suite before the later storage-verifier/hybrid additions | 604 passed, 253 skipped |
| 4090D | Full suite, ABI 8; BF16/FP16/F32 helper and BF16 decoder CUDA cases | 833 passed, 24 skipped |
| V100 host | Selected CPU tests, existing Torch environment unchanged | 69 passed, 37 skipped |
| V100 host | Independent storage-verifier unit tests | 29 passed, 1 skipped |

Tests include FP64 actual-input operator oracles, exact manual composition,
chunked cache continuation, stream ownership, alias/address guards, closed
plans, enqueue-failure poison/reset and fixed-position graph replay with tensor
allocation/view creation and numerical fallback prohibited. Optional installed
upstream Qwen CPU comparisons pass locally; they are skipped on the 4090D where
that upstream module is unavailable. No native-vs-upstream bit-identity claim.

Retained failures:

- `failures/initial-fp16-cascade.log`: an early norm rounding changes later
  rotary cancellation. Norm qualification plus exact actual-input fused/staged
  rotary parity were added; no numeric floor was widened and no precision fix
  is claimed.
- `failures/source-fp32-gamma-preflight.json` and `original-source-audit.json`:
  complete original dtype audit and the stopped initial converter.
- The earlier GDN BF16/FP64-cascade counterexample remains in the GDN capsule.

## Timing and GPU diagnostics

The real-geometry helper timing attempt stopped **before timed phases** because
another Python process occupied 14,714 MiB. `qualification.log` retains that
failure. No workload was stopped, no guard bypassed, no speed result produced.
Rerun `tools/qualify_qwen_attention.py` on an idle GPU with a fresh output folder.
Use the independently fetched original small norms and synthetic activations;
that proxy still cannot establish trained-model performance.

`RmProfilingAdminOnly=1` remains unchanged. Achieved occupancy, SM/DRAM/cache/stall
counters were not measured; NVML busy percentage is not a substitute. No new
profiler campaign ran. Retain the earlier direct-Q8 prefill regression (~20.53 ms
vs ~0.728 ms for one synthetic projection); no engine win follows from fusion.

## Conversion status

The corrected export subsequently published and independently passed storage
verification. Final artifact/report/hash evidence is under
`../qwen-hybrid-model/conversion/`. The old `conversion-state.json` here is a
historical in-progress observation, not current status. Full trained quality,
SM70, batching/paging, native runtime and matched engine comparisons remain open.
