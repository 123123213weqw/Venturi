// Model-owned Q5 routing. Calibration is explicit and never runs in a request.
#pragma once
namespace vr {
struct Q5Record {
    int n=0,k=0,t=0,route=0,valid=0;
    double ms[3]={0,0,0};float error[3]={0,0,0};
};
__global__ void q5_probe_input(__half* x,std::uint64_t count) {
    for(auto i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=std::uint64_t(gridDim.x)*blockDim.x)
        x[i]=__float2half_rn(float(int((i*48271+17)%257)-128)/128.f);
}
__global__ void q5_probe_error(const __half* a,const __half* b,std::uint64_t count,unsigned* stats) {
    for(auto i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=std::uint64_t(gridDim.x)*blockDim.x) {
        const float av=__half2float(a[i]),bv=__half2float(b[i]),e=fabsf(av-bv);
        if(!isfinite(av)||!isfinite(bv)||e>.002f+.002f*fabsf(bv))atomicExch(stats,1U);
        if(isfinite(e))atomicMax(stats+1,__float_as_uint(e));
    }
}
struct Q5Router {
    int mode=0; // 0 legacy, 1 bounded decode+TC cuBLAS, 2 calibrated adaptive
    Buffer scratch;
    std::map<std::array<int,3>,Q5Record> records;
    std::uint64_t calls[3]={0,0,0},rows[3]={0,0,0},misses=0,guarded=0;
    static constexpr std::uint64_t blas=4ULL*1024*1024;
    int chunk(int n,int k) const {
        require(scratch.bytes>=blas+std::uint64_t(k)*2+255,"Q5 scratch cannot hold a row");
        int r=std::min<std::uint64_t>({std::uint64_t(n),65535,(scratch.bytes-blas)/(k*2ULL)});
        while((std::uint64_t(r)*k*2+255)/256*256+blas>scratch.bytes)--r;
        require(r>0,"Q5 scratch row overflow");return r;
    }
    void run(int route,void* dense,const VRMatrix& m,const void* x,const void* codes,
             const void* high,const void* scales,void* out,int t,int n,cudaStream_t stream) {
        if(route==0)op(venturi_q5_cublas_run(dense,x,codes,high,scales,out,t,n,m.k,padded(m),scratch.ptr,scratch.bytes,chunk(n,m.k),stream));
        else if(route==1)op(venturi_q5_fused_run(x,codes,high,scales,out,t,n,m.k,padded(m),0,stream));
        else {require(route==2 && m.k%64==0,"invalid Q5 WMMA route");op(venturi_q5_hmma_run(x,codes,high,scales,out,t,n,m.k,padded(m),stream));}
    }
    void project(void* dense,const VRMatrix& m,const void* x,const void* codes,
                 const void* high,const void* scales,void* out,int t,int n,cudaStream_t stream,bool qualified) {
        int route=0;
        if(mode==2 && qualified && t<=8){auto it=records.find({n,m.k,t});if(it==records.end())++misses;else route=it->second.route;}
        else if(mode==2)++guarded;
        ++calls[route];rows[route]+=t;run(route,dense,m,x,codes,high,scales,out,t,n,stream);
    }
    void reset(){std::fill(calls,calls+3,0);std::fill(rows,rows+3,0);misses=0;guarded=0;}
    // One actual resident representative per (N,K). All outputs are checked;
    // the policy is shape-based, not a claim of per-weight optimality.
    void calibrate(void* dense,const VRMatrix& m,const void* codes,const void* high,
                   const void* scales,int t,cudaStream_t stream,int repeats,int iterations,double gain) {
        Buffer x,out,ref,stats;
        const auto count=std::uint64_t(t)*m.n;
        x.allocate(std::uint64_t(t)*m.k*2);out.allocate(count*2);ref.allocate(count*2);stats.allocate(8);
        q5_probe_input<<<256,256,0,stream>>>(static_cast<__half*>(x.ptr),std::uint64_t(t)*m.k);check(cudaGetLastError());
        run(0,dense,m,x.ptr,codes,high,scales,ref.ptr,t,m.n,stream);
        Q5Record rec;rec.n=m.n;rec.k=m.k;rec.t=t;
        // Scalar is a decode candidate only; do not spend minutes timing an
        // already uncompetitive high-M scalar path on the huge vocabulary head.
        for(int route=0;route<3;++route){
            if((route==1 && t>8)||(route==2 && m.k%64))continue;
            run(route,dense,m,x.ptr,codes,high,scales,out.ptr,t,m.n,stream);
            check(cudaMemsetAsync(stats.ptr,0,8,stream));
            q5_probe_error<<<256,256,0,stream>>>(static_cast<const __half*>(out.ptr),static_cast<const __half*>(ref.ptr),count,static_cast<unsigned*>(stats.ptr));
            check(cudaGetLastError());unsigned value[2];check(cudaMemcpyAsync(value,stats.ptr,8,cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
            std::memcpy(&rec.error[route],value+1,4);
            if(!value[0]){rec.valid|=1<<route;for(int w=0;w<2;++w)run(route,dense,m,x.ptr,codes,high,scales,out.ptr,t,m.n,stream);}
        }
        require(rec.valid&1,"Q5 cuBLAS probe produced nonfinite results");
        cudaEvent_t a=nullptr,b=nullptr;check(cudaEventCreate(&a));
        try{check(cudaEventCreate(&b));std::vector<double> samples[3];
            for(int rep=0;rep<repeats;++rep)for(int j=0;j<3;++j){
                int route=rep%2?(rep+2-j)%3:(rep+j)%3;if(!(rec.valid&(1<<route)))continue;
                check(cudaEventRecord(a,stream));
                for(int it=0;it<iterations;++it)run(route,dense,m,x.ptr,codes,high,scales,out.ptr,t,m.n,stream);
                check(cudaEventRecord(b,stream));check(cudaEventSynchronize(b));float elapsed=0;check(cudaEventElapsedTime(&elapsed,a,b));samples[route].push_back(elapsed/iterations);
            }
            for(int r=0;r<3;++r)if(!samples[r].empty()){auto& v=samples[r];std::sort(v.begin(),v.end());rec.ms[r]=v[v.size()/2];}
            for(int r=1;r<3;++r)if((rec.valid&(1<<r))&&rec.ms[r]<rec.ms[rec.route]&&rec.ms[r]<rec.ms[0]*(1-gain))rec.route=r;
            // Whole-Qwen qualification permits only small-M vocabulary head
            // routing. The caller never calibrates state-affecting matrices.
            // Operator eligibility alone does not qualify a model trajectory.
            if(t>8)rec.route=0;
            records[{m.n,m.k,t}]=rec;
        }catch(...){if(b)cudaEventDestroy(b);cudaEventDestroy(a);throw;}
        check(cudaEventDestroy(b));check(cudaEventDestroy(a));
    }
};
} // namespace vr
