# SM89 Q5 M-bucket experiment: RTX 4090 D

**Experiment, not a default.** This checks whether ten traffic-weighted
calibration points can replace most shape sweeps without hurting
steady-state service throughput or P95. It does not qualify a new numeric
mode for production use.

## Setup

- Single RTX 4090 D (24 GiB), SM89, Qwen 27B text with the same resident
  Q5-G64 weight file used by the prior SM70/SM89 experiments, FP16
  projections with FP32 accumulation.
- Old ten-point policy, M-bucket ten-point policy, and a previously
  measured 256-shape atlas. The old ten-point traffic histogram selected
  the new anchors. The M-bucket plan was pinned before checking the atlas.
- One immutable loaded model, a fresh KV/GDN batch for each arm, static
  KV 8 × 4096, token budget 32, prefill chunks of 16, descriptor-eager.
  Six arms in order: old / bucket / atlas / atlas / bucket / old.
- Held-out rolling C4/C8 prompts: 73, 101, 137 and 169 tokens, with 16 or
  24 generated tokens. Aggregate TPS is token count divided by wall time
  in the common steady window. C1 is a separate 32-step pure decode test.

## Results

| Workload | Old 10 points | M-bucket 10 points | Full atlas | P95 inter-token latency (old / bucket / atlas) |
| --- | ---: | ---: | ---: | ---: |
| C1 pure decode | 25.23 TPS | 25.20 TPS | 17.72 TPS | 39.7 / 39.7 / 56.5 ms |
| C4 mixed | 25.48 TPS | 25.46 TPS | 26.46 TPS | 132.4 / 123.3 / 123.2 ms |
| C8 mixed | 40.40 TPS | 41.69 TPS | 39.75 TPS | 147.8 / 136.0 / 135.8 ms |

C4 retained 96.2% of the atlas throughput; the C8 result exceeded the
atlas for this workload. The atlas is the fastest measured route per
projection, not an upper bound on whole-service throughput. Compared
with the old ten points, P95 improved 6.9% and 8.0% for C4 and C8.
The new calibration took 3.37 s (60 timed samples), excluding file
transfer and model load. The previous atlas took 12.42 s (5,184 timed
samples); these are separate runs, not a cold-start comparison.

In separate fixed mixed replays, median step time changed from
129.73 to 120.38 ms at M=19 and from 139.29 to 127.70 ms at M=23.
These are like-for-like comparisons. CUDA activity capture showed that
switching some medium-width masked projections back to dense cuBLAS
reduced total step time despite extra unpacking and GEMM launches. The
wide MLP case at M=2 picked WMMA instead. Neither a single M rule nor
choosing only by device type fits both cases. NVML median power did
increase (about 247→285 W at C4 and 284→298 W at C8); energy and achieved
Tensor Core occupancy were not measured.

All held-out C4/C8 requests produced the same token sequences as the old
ten-point plan in these runs. Fixed replay's logits differed (0.113%
relative L2); KV/GDN are not bit-identical. No long-session quality,
other codec, M>32, or cross-device performance claim is made.

## Regression checks

Build the paired SM89 libraries, then run the CPU bucket invariants and
SM89 numeric/native-batch fixtures:

```sh
python -m venturi.ops.build --component ops --architecture sm89 --output-dir /path/to/ops
python -m venturi.models.qwen.build --output-dir /path/to/qwen --ops-library /path/to/libventuri_ops.so
VENTURI_ADA_TILE_OPS=/path/to/libventuri_ops.so \
VENTURI_ADA_TILE_QWEN=/path/to/libventuri_qwen.so \
python -m pytest -q tests/test_m_buckets.py tests/test_sm89_budgeted_row_tune.py
```

The performance results come from the opt-in experiment at `4ca2772e`.
The review branch keeps the same SM89 ops sources and dispatch rules but
drops the independent SM70 online-tuning experiment; the whole-model
performance campaign was not rerun on the review branch.
