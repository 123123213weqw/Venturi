# Night TPS campaign 2026-11-16/17

Scope: Qwen3.8-27B Q8 (native safetensors artifact), two V100-PCIE-32GB,
resident TP, projection q8-residual-column-mlp-head-gdn, sealed metadata,
continuous scheduling. Goal: 4/8/16/32 concurrency aggregate TPS + GPU busy.

Environment notes:
- torch 2.14.1+cu130 and 2.11.0+cu128 host builds lack SM70 compute kernels;
  venturi-sm70 venv uses torch 2.7.1 (PyPI cu126, sm70 included).
- Production llama.cpp container stopped at 23:33 (no auto-restart watchdog
  verified); restore via /home/wzu/restore-production.sh.
- Q8 artifact exported from unverified local HF source (allow-unverified);
  verified re-export running to qwen38-27b-text-q8-verified.safetensors.

Decisions log:
- Q5 SM70 W5A16 reference kernel landed (commit d8be7ec, 33 tests green).
- Baseline uses serve_native_benchmark.py --continuous-target, slots 8,
  capacity 512, projection-rows 32, chunk 16, round-robin (historical
  qualified config); wider slots/rows for 16/32 cohorts measured separately.
