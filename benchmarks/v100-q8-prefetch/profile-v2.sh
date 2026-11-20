#!/bin/sh
# Explicit, bounded operator counters only; never stops or changes production.
set -eu
s=/home/data/wangyue/repos/codex-build/Venturi/q8-prefetch-src-v2
d=/home/data/wangyue/artifacts/venturi-q8-prefetch-v2
py=/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python
deps=/home/data/wangyue/repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps
qualified=/home/data/wangyue/artifacts/venturi-native-head-gdn-fixtures-v2/build/68769d5f68c715ef/libventuri_blocks.so
test "$(cat "$d/tests.exit")" = 0
test "$(systemctl --user show venturi-q8-prefetch-confirm-v2 -p ActiveState --value)" = inactive
metrics=dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active,lts__t_sector_hit_rate.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
for case in mlp-fused mlp-down; do
  for arm in qualified row-pair; do
    # Keep full explicit command/output provenance. Root is used only for these
    # counters; no sudo -E, clock changes, or profiling-policy changes.
    name=$case-$arm
    test ! -e "$d/counter-$name.csv"
    set +e
    sudo -n /usr/bin/timeout 100 /usr/bin/env CUDA_VISIBLE_DEVICES=0 \
      PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      CUDA_CACHE_PATH="$d/cuda-cache" PYTHONPATH="$deps:$s/src:$s/tests:/home/wzu/.local/lib/python3.10/site-packages" \
      /usr/local/cuda/bin/ncu --clock-control none --cache-control none \
      --target-processes all --kernel-name-base demangled \
      --kernel-name 'regex:.*(vqf::tiled_dot|vct::partial|vpf::dot|vpf::row_pair).*' \
      --launch-skip 8 --launch-count 1 --metrics "$metrics" \
      --csv --page raw --disable-extra-suffixes --log-file "$d/counter-$name.csv" \
      "$py" "$s/tools/projection/benchmark_q8_prefetch.py" \
      --gguf /home/wzu/models/Qwen3.8-27B-Uncensored-GGUF/Qwen3.8-27B-Uncensored-Q8_0.gguf \
      --library "$d/libprefetch_probe.so" --qualified-library "$qualified" \
      --output-directory "$d/profile-$name" --tokens 8 --profile-case "$case" --profile-arm "$arm" \
      >"$d/counter-$name.log" 2>&1
    rc=$?
    set -e
    echo "$rc" >"$d/counter-$name.exit"
    test "$rc" = 0 || exit "$rc"
  done
done
docker inspect --format '{{.State.Status}} {{.State.Health.Status}}' qwen38-27b
grep RmProfilingAdminOnly /proc/driver/nvidia/params
