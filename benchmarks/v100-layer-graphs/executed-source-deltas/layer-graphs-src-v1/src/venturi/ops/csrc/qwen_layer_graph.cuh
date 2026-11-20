// Included within namespace vr after Model/PrefixTrace. Original kernels and
// arithmetic stay unchanged; only cache/table addressing uses a device cursor.
#pragma once
namespace graph_ops {
template<class T>
__global__ void positioned_qk(const T* qg, const T* key, const float* qgamma,
    const float* kgamma, const float* cos, const float* sin, T* qout, T* kout,
    int hq, int hkv, int d, int r, float epsilon, const int* position) {
    const int pos=*position;cos+=static_cast<std::int64_t>(pos)*(r/2);sin+=static_cast<std::int64_t>(pos)*(r/2);kout+=static_cast<std::int64_t>(pos)*hkv*d;
    __shared__ float partial[8];
    __shared__ float normalized[256];
    const int token = blockIdx.x / (hq+hkv), h = blockIdx.x % (hq+hkv);
    const bool query = h < hq;
    const int head = query ? h : h-hq;
    const std::int64_t row = static_cast<std::int64_t>(token) * (query ? hq : hkv) + head;
    const T* input = query ? qg + row * 2*d : key + row*d;
    T* output = query ? qout + row*d : kout + row*d;
    const float* gamma = query ? qgamma : kgamma;
    const int col = threadIdx.x;
    const float x = col < d ? read(input[col]) : 0.0f;
    const float inv = rsqrtf(block_sum(x*x,partial)/d + epsilon);
    if (col < d) normalized[col] = read(cast<T>((x*inv)*gamma[col]));
    __syncthreads();
    if (col >= d) return;
    if (col >= r) { output[col]=cast<T>(normalized[col]); return; }
    const int half = r/2, other = col < half ? col+half : col-half;
    const std::int64_t offset = static_cast<std::int64_t>(token)*half + col%half;
    const float c=read(cast<T>(cos[offset])), t=read(cast<T>(sin[offset]));
    const float first=read(cast<T>(__fmul_rn(normalized[col],c)));
    const float second=read(cast<T>(__fmul_rn(normalized[other],t)));
    output[col]=cast<T>(col < half ? first-second : first+second);
}
template<class T>
__global__ void positioned_attention(const T* q, const T* k, const T* v, T* out,
                                 int tokens, const int* position, int hq, int hkv,
                                 int d, float scale) {
    const int kv_tokens=*position+tokens;
    __shared__ float partial[8];
    __shared__ float factors[3];
    const int row = blockIdx.x;
    const int token = row / hq;
    const int head = row % hq;
    const int kv_head = head / (hq / hkv);
    const int col = threadIdx.x;
    const std::int64_t qbase = static_cast<std::int64_t>(row) * d;
    const float query = col < d ? read(q[qbase + col]) : 0.0f;
    float acc = 0.0f, maximum = -INFINITY, denom = 0.0f;
    const int last = kv_tokens - tokens + token;
    for (int key = 0; key <= last; ++key) {
        const std::int64_t base = (static_cast<std::int64_t>(key) * hkv + kv_head) * d;
        const float product = col < d ? query * read(k[base + col]) : 0.0f;
        const float score = block_sum(product, partial) * scale;
        if (col == 0) {
            const float next = fmaxf(maximum, score);
            const float alpha = expf(maximum - next), beta = expf(score - next);
            denom = denom * alpha + beta;
            maximum = next;
            factors[0] = alpha; factors[1] = beta; factors[2] = denom;
        }
        __syncthreads();
        if (col < d) acc = acc * factors[0] + read(v[base + col]) * factors[1];
        // All consumers must finish before thread 0 updates shared factors.
        __syncthreads();
    }
    if (col < d) out[qbase + col] = cast<T>(acc / factors[2]);
}
__global__ void store_values(const std::uint16_t* input,std::uint16_t* cache,const int* position,int tokens,int width) {
    const auto count=static_cast<std::int64_t>(tokens)*width;
    for(auto i=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=static_cast<std::int64_t>(gridDim.x)*blockDim.x)
        cache[static_cast<std::int64_t>(*position)*width+i]=input[i];
}
}
struct LayerGraph {
    Buffer position;Pinned host_position;
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    int device=0;bool warm=false;void* output=nullptr;
    std::vector<void*> trace_pointers;
    std::uint64_t captures=0,replays=0,fallbacks=0,warm_calls=0,nodes=0,input_delta=0,output_delta=0;
    explicit LayerGraph(Model& m):device(m.c.device) {m.guard(4);position.allocate(4);host_position.allocate(4);*static_cast<int*>(host_position.ptr)=0;}
    void clear() noexcept {
        int old=0;(void)cudaGetDevice(&old);(void)cudaSetDevice(device);
        if(executable)(void)cudaGraphExecDestroy(executable);if(graph)(void)cudaGraphDestroy(graph);
        executable=nullptr;graph=nullptr;warm=false;nodes=0;output=nullptr;trace_pointers.clear();
        (void)cudaSetDevice(old);
    }
    ~LayerGraph(){clear();}
};
