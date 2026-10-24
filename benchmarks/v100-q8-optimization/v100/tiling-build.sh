set -eu
OUT=/home/wzu/codex-build/Venturi/v100-resident-optimization-v1
trap 'echo "$?" > "$OUT/tiling-build-exit-code.txt"' EXIT
cd /home/wzu/codex-build/Venturi/v100-tiling-src-v1
export PYTHONPATH=src
/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python "$OUT/tiling-build.py"
