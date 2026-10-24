# Using the explicit two-V100 native profile

Use the remote Linux CUDA environment, not macOS. Neither the CLI nor constructors
silently compile or download a backend/model. On the qualified V100 node:

```sh
cd /home/data/wangyue/repos/codex-build/Venturi/native-head-gdn-src-v3
export CUDA_VISIBLE_DEVICES=1,0
export PYTHONPATH=/home/data/wangyue/repos/codex-build/Venturi/v100-tp-diagnostics-v2/python-deps:$PWD/src
PY=/home/data/wangyue/repos/codex-build/.venvs/b2-v100/bin/python
LIB=/home/data/wangyue/artifacts/venturi-native-head-gdn-fixtures-v2/build/68769d5f68c715ef/libventuri_blocks.so
MODEL=/home/data/wangyue/repos/codex-build/Venturi/models/qwen3.5-27b-text-q8.ninfer
DRAFT=/home/data/wangyue/repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer

# Scalar MTP2: already formatted pinned prompt,32 fixed outputs, FP16 KV.
"$PY" -m venturi qwen-native generate "$MODEL" \
  --blocks-library "$LIB" --weight-mode resident --dtype fp16 \
  --device cuda:0 --tensor-parallel-device cuda:1 \
  --tensor-parallel-transport host \
  --projection-profile q8-residual-column-mlp-head-gdn \
  --mtp-artifact "$DRAFT" --mtp-draft-projection-profile q8-residual \
  --mtp-draft-tokens 2 --mtp-recovery prefix \
  --token-ids 248045,846,198,7734,264,2716,3255,883,264,11952,6618,310,5984,13,248046,198,248045,74455,198,248068,271,248069,271 \
  --max-tokens 32 --capacity 256 --max-new-tokens 32 --eos-token-ids ''
```

The printed native timings exclude model loading and CLI initialization. This is
not an HTTP or streaming-client TTFT benchmark. Raw text input needs an explicit
local tokenizer and formatting; the CLI never invents a chat template.

For the controlled parallel benchmark, use the same loopback adapter and fixed
common-barrier client as the recorded commands. Keep eight prepared256-context
slots on both engines. Packed MTP1 uses `--batch-mtp --mtp-depth 1` and row32;
ordinary packed target uses `--batch-target` without a draft and row256. The
benchmark admission window is explicitly25ms (included in complete HTTP wall),
not continuous batching/paged attention or a production OpenAI server. Strategies
are selected explicitly; no automatic single/parallel strategy router is claimed.

Weights remain Q8, FP16 KV and original small-parameter values are retained;
residual-A8 activation arithmetic and FP32 partial reduction association are
explicitly different compute profiles. This is not identical arithmetic to
llama.cpp Q8_0/Q8_1. Unsupported geometry, architecture, dtype, transport, PP or
missing optional exports fail closed rather than silently switching engines.
