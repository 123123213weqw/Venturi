set -eu
OUT=/home/wzu/codex-build/Venturi/native-runtime-validation
trap 'echo "$?" > "$OUT/exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/native-runtime-src-v1
export PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1
PY=/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python
LIB=$(cat /home/wzu/codex-build/Venturi/native-runtime-build-v1/library-path.txt)
$PY -u tools/qualify_qwen_native.py /home/wzu/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer --expected-sha256 b2428c6e524914853d636b268c21f2ed298ba3d19c682fdef4973e8d634edab8 --blocks-library "$LIB" --previous-directory /home/wzu/codex-build/Venturi/greedy-validation-v3/full-model-run1 --previous-report-sha256 cb443b28c13f074372da52918fb9a89064523552216af9b0c555ed9ae12b1732 --tokenizer-file /home/wzu/codex-build/Venturi/models/qwen3.5-27b-bf16/tokenizer.json --dtype fp16 --weight-mode streaming --output-directory "$OUT/full-model-v1"
