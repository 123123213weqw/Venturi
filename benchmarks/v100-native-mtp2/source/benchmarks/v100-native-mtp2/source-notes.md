# Primary architecture references consulted

Reviewed 2026-10-20 (Asia/Shanghai). These are architecture references, not runtime
or copied implementation dependencies; Venturi's cache transaction code is native.

- [vLLM Qwen3.5 MTP predictor](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py)
  normalizes embedding and prior hidden separately, concatenates/projects them,
  applies its draft block, and returns final-normalized hidden states.
- [vLLM base proposer](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/llm_base_proposer.py)
  feeds the returned hidden state into subsequent speculative steps. Together
  these support the normalized-draft recurrence used here.
- [SGLang Qwen3.5 MTP](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_5_mtp.py)
  independently shows the separate pre-FC embedding/hidden normalizations and
  concatenation into its MTP branch.

These moving upstream pages are not a claim about the pinned llama.cpp binary's
exact source commit or an exhaustive best external configuration. Numerical proof
comes from the independently reconstructed represented-weight draft recurrence
and target/draft committed-state tests, with all scope limitations retained.
