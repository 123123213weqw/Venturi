set -eu
OUT=/home/wzu/codex-build/Venturi/v100-resident-optimization-v1
trap 'echo "$?" > "$OUT/ablation-v2-exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/v100-tiling-src-v1
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python -u tools/benchmark_q8_tiles.py /home/wzu/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer --baseline /home/data/wangyue/repos/codex-build/Venturi/sm70-build/9b826970163c1fb4/libventuri_blocks.so --candidates "$OUT/tiling-libraries.json" --output-directory "$OUT/projection-ablation-v2" --gpu-index 1 > "$OUT/projection-ablation-v2.log" 2>&1
