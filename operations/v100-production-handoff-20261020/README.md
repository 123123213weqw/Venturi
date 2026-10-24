# V100 production route restored — 2026-10-20

## Use in XHarness

- Provider: **qwen**
- Model: **qwen3.8-27b-uncensored**
- Base URL: `http://127.0.0.1:8001/v1`
- Context: **262,144 tokens**, as reported by the running server.
- Maximum output budget: **32,768 tokens**; minimum output reserve and safety
  margin: **4,096 tokens each**.
- Thinking: **on** by default, with explicit **Thinking off / Thinking on** choices.

These settings were applied live to the existing `qwen` provider only, with a
revision check (21 → 22). Other providers, the default selection and existing
user chats were not changed. No app restart is needed. The owned test session
is titled **V100 production readiness test**.

## Restored server, not a new native deployment

The original Docker container `qwen38-27b` was restarted, not recreated:
`fc9e37261f26c3127d5b65080b392ddeda97ba293dfd5573519354222328d70d`.
Its existing command, environment, mounts and `unless-stopped` restart policy
were preserved. This is the existing **llama.cpp** production route and its
uncensored Q8_0 GGUF, **not** the official Qwen3.5 checkpoint / native Venturi
benchmark runtime from the frozen comparison campaign.

Existing server parameters:

| Parameter | Value |
| --- | --- |
| Hardware | Two V100-PCIE-32GB GPUs, existing device order 0,1 |
| Weights / KV | Q8_0 / FP16 |
| Distribution | Tensor parallel, split 1,1, all layers on GPU |
| Context / active slots | 262,144 / 1 |
| Batch / microbatch | 4,096 / 2,048 |
| Speculation | Native model MTP, maximum 3 draft tokens |
| Attention / sampling | Flash attention and existing backend sampling enabled |
| Chat / vision | Jinja template, deepseek reasoning format, existing vision projector |
| Cache | Existing prompt/idle-slot cache and 131,072-token RAM cache retained |

The one-slot server queues concurrent requests; it is not the eight-slot native
benchmark configuration. No new parallel-throughput or numerical-equivalence
claim is made. Existing production experimental environment flags remain as
recorded in `production-parameters.json`.

The existing SSH LaunchAgent forwards Mac loopback 8001 to V100 loopback 8000.
It was already running and was not replaced. Normal host-key checking and no
agent forwarding are retained. Original server binding/authentication was not
changed. Mac connectivity is needed for this desktop route. The unrelated
idle-wash gateway remains unchanged.

## Verified

- Model health and model identity through the actual Mac tunnel.
- Nonstreaming and SSE streaming chat, including the completion sentinel.
- Thinking off and thinking on, with a separate reasoning response on the latter.
- Structured function call and continuation using a fresh synthetic tool result.
- A real XHarness desktop session selected this route, streamed a response,
  invoked its `read` tool on a generated fixture, consumed the tool result and
  returned the exact unseen fixture value. The turn completed successfully.
- Its request headers show context 262,144, reserved output 32,768, minimum
  output 4,096 and safety margin 4,096 on both steps.
- Final container identity is unchanged, running with healthy HTTP responses.

The readiness turn used roughly 8.3–8.6K input tokens, not a full-256K context
stress test. Vision is loaded and advertised; image correctness was not tested
here. This is a functional restoration/smoke test, not a throughput benchmark.
No inference code, frozen evidence, driver settings or clocks were changed;
no Rust/CUDA compilation or Git push was performed.

## Recovery record

The first settings mutation observation timed out after 35 seconds. A read
confirmed revision 21 and unchanged settings before one bounded retry with a
longer observation timeout. That retry completed in approximately 3.3 seconds;
revision 22 and preservation of all unrelated settings were verified. Neither
the model nor XHarness was restarted in response to that observation timeout.

Private pre-change XHarness settings are retained outside the repository at
`/Users/wangyue/Library/Application Support/com.xlang.xharness/venturi-production-handoff-20261020/settings-before.json`.
Remote startup intent/result records are at
`/home/data/wangyue/artifacts/venturi-production-handoff-20261020/`.
API secrets and desktop authentication tokens are not included in these records.
