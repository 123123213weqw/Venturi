// Explicit SM70 research paths. Included inside vr after HeadGDNSegment.
#pragma once
struct DecodeSegments {
    int count=0;
    struct Slot {int row;void *history,*state;} slots[32];
};
static_assert(sizeof(DecodeSegments)<3072,"V100 kernel parameter budget");
template<class W> __global__ void conv_segments(const __half* x,const W* weights,__half* out,
    DecodeSegments batch,int channels,int kernel) {
    const auto slot=batch.slots[blockIdx.y];auto* history=static_cast<__half*>(slot.history);
    for(int col=blockIdx.x*blockDim.x+threadIdx.x;col<channels;col+=gridDim.x*blockDim.x){
        float accum=0.0f;
        for(int tap=0;tap<kernel;++tap){const float value=tap==kernel-1?read(x[slot.row*channels+col]):read(history[tap*channels+col]);
            accum=fmaf(value,read(weights[tap*channels+col]),accum);}
        const float conv=read(cast<__half>(accum));out[slot.row*channels+col]=cast<__half>(conv*sigmoid(conv));
        for(int row=0;row<kernel-2;++row)history[row*channels+col]=history[(row+1)*channels+col];
        history[(kernel-2)*channels+col]=x[slot.row*channels+col];
    }
}
__global__ void gdn_segments(const __half* qkv,const float* decay,const float* beta,__half* out,
    DecodeSegments batch,int hqk,int hv,int dk,int dv,float scale){
    __shared__ float partial[8];
    const auto slot=batch.slots[blockIdx.y];auto* state=static_cast<float*>(slot.state);
    const int head=blockIdx.x/dv,row=blockIdx.x%dv,col=threadIdx.x,qhead=head/(hv/hqk);
    const auto state_idx=static_cast<std::int64_t>(blockIdx.x)*dk+col;
    const auto base=static_cast<std::int64_t>(slot.row)*(2*hqk*dk+hv*dv);
    float qt=read(qkv[base+qhead*dk+col]),kt=read(qkv[base+hqk*dk+qhead*dk+col]);
    float vq=0.0f,vk=0.0f,vs=0.0f;
    const float qs=gdn_compact_sum(qt*qt,0.0f,partial),ks=gdn_compact_sum(kt*kt,0.0f,partial);
    qt*=rsqrtf(qs+1e-6f);kt*=rsqrtf(ks+1e-6f);vq*=rsqrtf(qs+1e-6f);vk*=rsqrtf(ks+1e-6f);
    const auto gate_idx=slot.row*hv+head;
    float s=state[state_idx];s*=expf(decay[gate_idx]);vs*=expf(decay[gate_idx]);
    const float memory=gdn_compact_sum(s*kt,vs*vk,partial);
    const float delta=beta[gate_idx]*(read(qkv[base+2*hqk*dk+head*dv+row])-memory);
    s=fmaf(delta,kt,s);vs=fmaf(delta,vk,vs);
    const float value=gdn_compact_sum(s*qt,vs*vq,partial)*scale;
    if(!col)out[static_cast<std::int64_t>(gate_idx)*dv+row]=cast<__half>(value);
    state[state_idx]=s;
}
// Match both 256-thread dot reductions and the two FP16 boundary roundings.
__global__ void ab_gates(const __half* input,const float* wa,const float* wb,const float* rate,const float* bias,
    __half* a,__half* b,float* decay,float* beta,int heads,int hidden){
    __shared__ float pa[8],pb[8];
    const int token=blockIdx.x/heads,head=blockIdx.x%heads;
    float sa=0.0f,sb=0.0f;
    for(int col=threadIdx.x;col<hidden;col+=blockDim.x){const float x=read(input[token*hidden+col]);
        sa=fmaf(x,wa[head*hidden+col],sa);sb=fmaf(x,wb[head*hidden+col],sb);}
    const float av=block_sum(sa,pa),bv=block_sum(sb,pb);
    if(!threadIdx.x){const auto aa=cast<__half>(av),bb=cast<__half>(bv);a[blockIdx.x]=aa;b[blockIdx.x]=bb;
        const float z=read(aa)+bias[head];const float sp=z>20.0f?z:log1pf(expf(z));
        decay[blockIdx.x]=-rate[head]*sp;beta[blockIdx.x]=read(cast<__half>(sigmoid(read(bb))));}
}
struct RankGraph {
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;int device;
    explicit RankGraph(int d):device(d){}
    ~RankGraph(){int old=0;(void)cudaGetDevice(&old);(void)cudaSetDevice(device);
        if(exec)(void)cudaGraphExecDestroy(exec);if(graph)(void)cudaGraphDestroy(graph);(void)cudaSetDevice(old);}
};
