// Native-only, compile-time opt-in for the SM70 row-pair experiment.
// Operator qualification is NOT full-model performance/quality qualification.
#pragma once
#ifndef VENTURI_Q8_ROW_PAIR
#define VENTURI_Q8_ROW_PAIR 0
#endif
static_assert(VENTURI_Q8_ROW_PAIR==0 || VENTURI_Q8_ROW_PAIR==1);
#if VENTURI_Q8_ROW_PAIR
#include "q8_prefetch.cuh"
#endif

namespace vprojection {
// Keep unmeasured single-request/MTP/tiny/prefill geometries on the old path.
// These predicates deliberately describe the measured two-device 27B shards;
// other models/geometries retain the ordinary implementation, even if enabled.
inline bool fused_eligible(const VQFOutput* outputs,int count,int tokens,int k,int passes) {
    if(VENTURI_BLOCKS_ARCH!=70 || passes!=2 || tokens<8 || tokens>32 || k!=5120 || !outputs)return false;
    return (count==2 && ((outputs[0].n==8704 && outputs[1].n==8704) ||
                        (outputs[0].n==5120 && outputs[1].n==3072))) ||
           (count==3 && outputs[0].n==6144 && outputs[1].n==512 && outputs[2].n==512);
}
inline bool partial_eligible(int tokens,int n,int k) {
    // K=3072,T=8 was neutral; leave that case on the qualified old kernel.
    return VENTURI_BLOCKS_ARCH==70 && tokens>=8 && tokens<=32 && n==5120 &&
           (k==8704 || (k==3072 && tokens>8));
}
inline int fused(const void* workspace,std::uint64_t available,const VQFOutput* outputs,
    int count,int tokens,int k,int padded,int passes,void* stream) {
#if VENTURI_Q8_ROW_PAIR
    if(fused_eligible(outputs,count,tokens,k,passes))
        return venturi_q8_prefetch_fused_run(workspace,available,outputs,count,tokens,k,padded,3,stream);
#endif
    return venturi_q8_fused_run(workspace,available,outputs,count,tokens,k,padded,passes,stream);
}
inline int partial(const void* workspace,std::uint64_t available,const std::int8_t* codes,
    const __half* scales,float* out,int tokens,int n,int k,int padded,void* stream) {
#if VENTURI_Q8_ROW_PAIR
    if(partial_eligible(tokens,n,k))
        return venturi_q8_prefetch_partial_run(workspace,available,codes,scales,out,tokens,n,k,padded,3,stream);
#endif
    return venturi_q8_column_partial_run(workspace,available,codes,scales,out,tokens,n,k,padded,stream);
}
}

// Explicit test/provenance exports, not an environment-variable hot switch.
extern "C" int venturi_q8_row_pair_enabled(){return VENTURI_Q8_ROW_PAIR;}
extern "C" int venturi_q8_projection_fused_run(const void* workspace,std::uint64_t available,
    const VQFOutput* outputs,int count,int tokens,int k,int padded,int passes,void* stream) {
    return vprojection::fused(workspace,available,outputs,count,tokens,k,padded,passes,stream);
}
extern "C" int venturi_q8_projection_partial_run(const void* workspace,std::uint64_t available,
    const std::int8_t* codes,const __half* scales,float* out,int tokens,int n,int k,int padded,void* stream) {
    return vprojection::partial(workspace,available,codes,scales,out,tokens,n,k,padded,stream);
}
