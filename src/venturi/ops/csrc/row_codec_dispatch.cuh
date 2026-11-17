// Model-independent QuantLinear capability plans. Admission/quality qualification
// remains separate: a registered kernel is not evidence of a qualified model.
#pragma once
#include "ops_api.h"

namespace rowcodec {
enum class Precision : int { WxA16=0, WxA8=1, WxA8Residual=2 };
enum class Implementation : int { Auto=0, Dequant=1 };
enum class Path : int { Direct, ChunkedDense, Quantized };
using ScalarRun=int (*)(const void*,const void*,const void*,const void*,void*,int,int,int,int,int,void*);
using HmmaRun=int (*)(const void*,const void*,const void*,const void*,void*,int,int,int,int,void*);
using QuantRun=int (*)(const void*,std::uint64_t,const void*,const void*,const void*,void*,int,int,int,int,int,void*);
struct LinearPlan {
    int bits,group;
    Precision precision;
    Path path;
    ScalarRun scalar_run;
    HmmaRun hmma_run;
    int min_hmma_tokens,hmma_k_align;
    QuantRun quant_run;
};
inline int q4_adapter(const void* x,const void* c,const void*,const void* s,void* y,int t,int n,int k,int p,int d,void* stream) {
    return venturi_q4_fused_run(x,c,s,y,t,n,k,p,d,stream);
}
inline int q8_adapter(const void* x,const void* c,const void*,const void* s,void* y,int t,int n,int k,int p,int d,void* stream) {
    return venturi_q8_head_run(x,static_cast<const std::int8_t*>(c),static_cast<const __half*>(s),y,t,n,k,p,d,stream);
}
// Fully validate format/layout (canonical row-split K128), group, architecture,
// activation dtype and numerical policy at load. No shape-dependent policy switch.
inline const LinearPlan* prepare(int bits,int group,Precision precision,int arch,int dtype,Implementation implementation) {
    if ((arch!=70 && arch!=89) || (dtype!=0 && dtype!=1)) return nullptr;
    if (bits==8 && group==32 && implementation==Implementation::Auto) {
        static const LinearPlan a16{8,32,Precision::WxA16,Path::Direct,q8_adapter,nullptr,0,0,nullptr};
        static const LinearPlan a8{8,32,Precision::WxA8,Path::Quantized,nullptr,nullptr,0,0,nullptr};
        static const LinearPlan residual{8,32,Precision::WxA8Residual,Path::Quantized,nullptr,nullptr,0,0,nullptr};
        if (precision==Precision::WxA16) return &a16;
        if (dtype==0 && precision==Precision::WxA8) return &a8;
        if (dtype==0 && precision==Precision::WxA8Residual) return &residual;
        return nullptr;
    }
    if (group!=64 || dtype!=0 || bits<4 || bits>6) return nullptr;
    if (precision==Precision::WxA16) {
        static const LinearPlan dense[]{
            {4,64,Precision::WxA16,Path::ChunkedDense,nullptr,nullptr,0,0,nullptr},
            {5,64,Precision::WxA16,Path::ChunkedDense,nullptr,nullptr,0,0,nullptr},
            {6,64,Precision::WxA16,Path::ChunkedDense,nullptr,nullptr,0,0,nullptr}};
        if (implementation==Implementation::Dequant || bits==6) return &dense[bits-4];
        if (implementation!=Implementation::Auto) return nullptr;
        static const LinearPlan q4{4,64,Precision::WxA16,Path::Direct,q4_adapter,nullptr,0,0,nullptr};
        static const LinearPlan q5{5,64,Precision::WxA16,Path::Direct,venturi_q5_fused_run,nullptr,0,0,nullptr};
        static const LinearPlan q5_sm70{5,64,Precision::WxA16,Path::Direct,venturi_q5_fused_run,venturi_q5_hmma_run,4,64,nullptr};
        if (bits==4) return &q4;
        if (bits==5) return arch==70?&q5_sm70:&q5;
    }
    if (bits==5 && arch==70 && precision==Precision::WxA8 && implementation==Implementation::Auto) {
        static const LinearPlan q5{5,64,Precision::WxA8,Path::Quantized,nullptr,nullptr,0,0,venturi_q5_dp4a_fused_run};
        return &q5;
    }
    return nullptr;
}
} // namespace rowcodec
