// SM70 single-token experiment: a warp owns one value row. Four register
// planes replace four physical warps, not the original reduction association.
#pragma once
#ifndef VENTURI_GDN_WARP_DECODE
#define VENTURI_GDN_WARP_DECODE 0
#endif
static_assert(VENTURI_GDN_WARP_DECODE==0 || VENTURI_GDN_WARP_DECODE==1);

__device__ __forceinline__ float gdn_warp_sum(float a,float b,float c,float d,float inactive) {
    #pragma unroll
    for(int delta=16;delta;delta>>=1) {
        a+=__shfl_down_sync(0xffffffff,a,delta);
        b+=__shfl_down_sync(0xffffffff,b,delta);
        c+=__shfl_down_sync(0xffffffff,c,delta);
        d+=__shfl_down_sync(0xffffffff,d,delta);
        inactive+=inactive;
    }
    // Every lane participates in the broadcasts. The original eight logical
    // warp partials (including virtual inactive warps) occupy lanes 0..7.
    const float p0=__shfl_sync(0xffffffff,a,0),p1=__shfl_sync(0xffffffff,b,0);
    const float p2=__shfl_sync(0xffffffff,c,0),p3=__shfl_sync(0xffffffff,d,0);
    const int lane=threadIdx.x&31;
    float value=lane==0?p0:lane==1?p1:lane==2?p2:lane==3?p3:lane<8?inactive:0.0f;
    #pragma unroll
    for(int delta=16;delta;delta>>=1)value+=__shfl_down_sync(0xffffffff,value,delta);
    return __shfl_sync(0xffffffff,value,0);
}

__global__ __launch_bounds__(128) void gdn_warp_decode(const __half* q,const __half* k,
    const __half* v,const float* g,const float* beta,const float* state,
    __half* out,float* state_out,int hqk,int hv,int dv,bool normalize,float scale) {
    constexpr int dk=128;
    const unsigned value_row=blockIdx.x*4+threadIdx.x/32;
    const int lane=threadIdx.x&31;
    // Uniform within a warp; no block barriers or inter-warp dependencies.
    if(value_row>=static_cast<unsigned>(hv*dv))return;
    const int head=value_row/dv,qhead=head/(hv/hqk);
    const auto qbase=static_cast<std::int64_t>(qhead)*dk;
    float s[4],qt[4],kt[4];
    #pragma unroll
    for(int j=0;j<4;++j) {
        const int col=j*32+lane;
        s[j]=state[static_cast<std::int64_t>(value_row)*dk+col];
        qt[j]=read(q[qbase+col]);kt[j]=read(k[qbase+col]);
    }
    float vq=0.0f,vk=0.0f,vs=0.0f;
    if(normalize) {
        const float qs=gdn_warp_sum(qt[0]*qt[0],qt[1]*qt[1],qt[2]*qt[2],qt[3]*qt[3],0.0f);
        const float ks=gdn_warp_sum(kt[0]*kt[0],kt[1]*kt[1],kt[2]*kt[2],kt[3]*kt[3],0.0f);
        const float qr=rsqrtf(qs+1e-6f),kr=rsqrtf(ks+1e-6f);
        #pragma unroll
        for(int j=0;j<4;++j){qt[j]*=qr;kt[j]*=kr;}
        vq*=qr;vk*=kr;
    }
    const float decay=expf(g[head]);
    #pragma unroll
    for(int j=0;j<4;++j)s[j]*=decay;
    vs*=decay;
    const float memory=gdn_warp_sum(s[0]*kt[0],s[1]*kt[1],s[2]*kt[2],s[3]*kt[3],vs*vk);
    const float delta=beta[head]*(read(v[value_row])-memory);
    #pragma unroll
    for(int j=0;j<4;++j)s[j]=fmaf(delta,kt[j],s[j]);
    vs=fmaf(delta,vk,vs);
    const float result=gdn_warp_sum(s[0]*qt[0],s[1]*qt[1],s[2]*qt[2],s[3]*qt[3],vs*vq)*scale;
    if(lane==0)out[value_row]=cast<__half>(result);
    #pragma unroll
    for(int j=0;j<4;++j)state_out[static_cast<std::int64_t>(value_row)*dk+j*32+lane]=s[j];
}
