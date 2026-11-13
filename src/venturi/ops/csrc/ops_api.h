// Concrete operator C ABI; opaque handles, explicit geometry, caller-owned buffers.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#define VENTURI_OPS_ABI 17
#define VENTURI_EXPORT __attribute__((visibility("default")))
struct VQFOutput {const std::int8_t* codes;const __half* scales;__half* out;int n,stride;};
static_assert(sizeof(VQFOutput)==32);
extern "C" {
VENTURI_EXPORT int venturi_add_run(const void* a, const void* b, void* out,
                              std::int64_t count, int dtype, void* stream);
VENTURI_EXPORT int venturi_attention_run(const void* q, const void* k, const void* v,
    void* out, int tokens, int kv_tokens, int hq, int hkv, int d, int dtype,
    float scale, void* raw_stream);
VENTURI_EXPORT int venturi_blocks_abi();
VENTURI_EXPORT int venturi_blocks_arch();
VENTURI_EXPORT const char* venturi_blocks_error();
VENTURI_EXPORT int venturi_conv_run(const void* x, const void* w, const void* history,
    void* out, void* history_out, int tokens, int channels, int kernel, int dtype, int fp32_weight, void* raw_stream);
VENTURI_EXPORT void* venturi_dense_create(int dtype);
VENTURI_EXPORT void venturi_dense_destroy(void* opaque);
VENTURI_EXPORT int venturi_dense_run(void* opaque, const void* x, const void* w, void* out, int tokens, int n, int k,
                                 void* workspace, std::size_t workspace_bytes, void* raw_stream);
VENTURI_EXPORT std::uint64_t venturi_dp2a_bytes(int tokens,int k);
VENTURI_EXPORT int venturi_dp2a_integer_dot(const void* a,const void* w,void* out,int groups,void* raw);
VENTURI_EXPORT int venturi_dp2a_quantize(const void* x,void* workspace,std::uint64_t available,int t,int k,void* raw);
VENTURI_EXPORT int venturi_embedding_run(const void* weight, const std::int64_t* ids,
    void* out, int tokens, int vocab, int hidden, int dtype, void* raw_stream);
VENTURI_EXPORT int venturi_fp32_linear_run(const void* x,const float* w,void* out,
    int tokens,int n,int k,int dtype,void* raw_stream);
VENTURI_EXPORT int venturi_gated_rms_run(const void* x, const void* z, const void* w, void* out,
    int tokens, int heads, int features, int dtype, int fp32_weight, float epsilon, void* raw_stream);
VENTURI_EXPORT int venturi_gates_run(const void* a, const void* b, const float* rate, const float* bias,
    float* g, float* beta, int tokens, int heads, int dtype, int round_beta, void* raw_stream);
VENTURI_EXPORT int venturi_gdn_compact_run(const void* q, const void* k, const void* v,
    const float* g, const float* beta, const float* state, void* out, float* state_out,
    int tokens, int hqk, int hv, int dk, int dv, int normalize, int packed, int dtype, float scale, void* raw_stream);
VENTURI_EXPORT int venturi_gdn_prefix_run(const void* q, const void* k, const void* v,
    const float* g, const float* beta, const float* state, void* out, float* state_out,
    int tokens, int hqk, int hv, int dk, int dv, int normalize, int packed, int dtype, float scale,void* prefix0,void* prefix1,int prefixes,void* raw_stream);
VENTURI_EXPORT int venturi_gdn_run(const void* q, const void* k, const void* v,
    const float* g, const float* beta, const float* state, void* out, float* state_out,
    int tokens, int hqk, int hv, int dk, int dv, int normalize, int packed, int dtype, float scale, void* raw_stream);
VENTURI_EXPORT int venturi_gdn_warp_decode_enabled();
VENTURI_EXPORT int venturi_greedy_run(const void* logits, std::int64_t* out, void* workspace,
    std::size_t workspace_bytes, int rows, int vocab, int dtype, void* raw_stream);
VENTURI_EXPORT int venturi_history_prefix_run(const void* x,const void* old,void* first,void* second,int channels,int kernel,int prefixes,int dtype,void* stream);
VENTURI_EXPORT int venturi_q4_fused_run(const void* x,const void* codes,const void* scales,void* out,int t,int rows,int k,int padded,int dtype,void* stream);
VENTURI_EXPORT int venturi_q8_column_partial_run(const void* workspace,std::uint64_t available,
    const std::int8_t* codes,const __half* scales,float* out,int tokens,int n,int k,int padded,void* raw_stream);
VENTURI_EXPORT int venturi_q8_column_sum(const float* a,const float* b,void* out,int count,void* raw_stream);
VENTURI_EXPORT std::uint64_t venturi_q8_dp4a_bytes(int tokens,int k,int passes);
VENTURI_EXPORT int venturi_q8_dp4a_quantize(const void* x,void* workspace,std::uint64_t available,int tokens,int k,int passes,void* stream_ptr);
VENTURI_EXPORT int venturi_q8_dp4a_run(const void* workspace,std::uint64_t available,const std::int8_t* codes,const __half* scales,void* out,int tokens,int n,int k,int padded,int passes,void* stream_ptr);
VENTURI_EXPORT int venturi_q8_embedding_run(const std::int8_t* codes, const __half* scales,
    const std::int64_t* ids, void* out, int tokens, int vocab, int k, int padded, int dtype, void* raw_stream);
VENTURI_EXPORT int venturi_q8_fused_run(const void* workspace,std::uint64_t available,const VQFOutput* outputs,int count,int tokens,int k,int padded,int passes,void* stream_ptr);
VENTURI_EXPORT int venturi_q8_head_run(const void* x, const std::int8_t* codes, const __half* scales,
    void* out, int tokens, int vocab, int k, int padded, int dtype, void* raw_stream);
VENTURI_EXPORT int venturi_q8_prefetch_fused_run(const void* workspace,std::uint64_t available,
    const VQFOutput* outputs,int count,int tokens,int k,int padded,int mode,void* raw_stream);
VENTURI_EXPORT int venturi_q8_prefetch_partial_run(const void* workspace,std::uint64_t available,
    const std::int8_t* codes,const __half* scales,float* out,int tokens,int n,int k,int padded,int mode,void* raw_stream);
VENTURI_EXPORT int venturi_q8_projection_fused_run(const void* workspace,std::uint64_t available,
    const VQFOutput* outputs,int count,int tokens,int k,int padded,int passes,void* stream);
VENTURI_EXPORT int venturi_q8_projection_partial_run(const void* workspace,std::uint64_t available,
    const std::int8_t* codes,const __half* scales,float* out,int tokens,int n,int k,int padded,void* stream);
VENTURI_EXPORT int venturi_q8_rms_quant_eligible(int t,int k);
VENTURI_EXPORT int venturi_q8_rms_quant_enabled();
VENTURI_EXPORT int venturi_q8_row_pair_enabled();
VENTURI_EXPORT int venturi_q8_shared_run(const void* workspace,std::uint64_t available,const std::int8_t* codes,const __half* scales,void* out,int tokens,int n,int k,int padded,int passes,int rows,void* stream_ptr);
VENTURI_EXPORT int venturi_q8_swiglu_quant_eligible(int tokens,int k);
VENTURI_EXPORT int venturi_q8_swiglu_quant_enabled();
VENTURI_EXPORT int venturi_q8_swiglu_quant_run(const void* gate,const void* up,void* intermediate,
    void* workspace,std::uint64_t available,int tokens,int k,int materialize,void* raw_stream);
VENTURI_EXPORT int venturi_q8_t3_fused_run(const void* workspace,std::uint64_t available,const VQFOutput* outputs,int count,int tokens,int k,int padded,int passes,int rows,int staging,void* stream_ptr);
VENTURI_EXPORT int venturi_q8_t3_run(const void* workspace,std::uint64_t available,const std::int8_t* codes,const __half* scales,void* out,int tokens,int n,int k,int padded,int passes,int rows,int staging,void* stream_ptr);
VENTURI_EXPORT int venturi_q8_tiled_shared_run(const void* workspace,std::uint64_t available,const std::int8_t* codes,const __half* scales,void* out,int tokens,int n,int k,int padded,int passes,int rows,int tile,void* stream_ptr);
VENTURI_EXPORT int venturi_q8_wmma_run(const void* x,const std::int8_t* codes,const __half* scales,
    void* out,int tokens,int n,int k,int padded,int warps,void* stream_ptr);
VENTURI_EXPORT int venturi_qwen_gate_run(const void* x, const void* qg, void* out,
    int tokens, int heads, int d, int dtype, void* raw_stream);
VENTURI_EXPORT int venturi_qwen_qk_run(const void* qg, const void* key,
    const float* qgamma, const float* kgamma, const float* cos, const float* sin,
    void* qout, void* kout, int tokens, int hq, int hkv, int d, int r, int dtype,
    float epsilon, void* raw_stream);
VENTURI_EXPORT int venturi_rms_q8_quantize(const void* x,const float* w,void* workspace,std::uint64_t available,
    int t,int k,float epsilon,int chunk,void* intermediate,void* raw_stream);
VENTURI_EXPORT int venturi_rms_run(const void* x, const float* w, void* out, int tokens,
                               int features, int dtype, float epsilon, void* raw_stream);
VENTURI_EXPORT int venturi_rope_run(const void* x, const float* cos, const float* sin,
    void* out, int tokens, int heads, int d, int r, int interleaved, int dtype, void* raw_stream);
VENTURI_EXPORT int venturi_row_dequant_indexed(const void* codes,const void* high,const void* scales,const std::int64_t* ids,void* out,int count,int k,int padded,int bits,void* stream);
VENTURI_EXPORT int venturi_row_dequant_rows(const void* codes,const void* high,const void* scales,void* out,int row,int rows,int k,int padded,int bits,void* stream);
VENTURI_EXPORT int venturi_swiglu_q8_quantize(const void* gate,const void* up,void* workspace,
    std::uint64_t available,int tokens,int k,int threads,void* intermediate_debug,void* raw_stream);
VENTURI_EXPORT int venturi_swiglu_run(const void* a, const void* b, void* out,
                                  std::int64_t count, int dtype, void* stream);
}
