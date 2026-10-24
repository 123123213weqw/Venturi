// Included inside namespace vr, after Buffer/VRMatrix and before Model.
// Stored weights/scales are preserved. The output column sum is an explicit
// alternate FP32 association; it never claims original FP16-output byte parity.
#pragma once
struct HeadGDNPart {void *history=nullptr,*state=nullptr;void* prefix_history[2]{};void* prefix_state[2]{};};
struct HeadGDNSegment {int row=0,count=0,prefixes=0;HeadGDNPart rank[2];};
struct HeadGDNWork {
    std::map<std::string,std::unique_ptr<Buffer>> buffers;
    std::uint64_t bytes=0;
    void allocate(const char* name,std::uint64_t n) {auto b=std::make_unique<Buffer>();b->allocate(n);bytes=add(bytes,n);buffers.emplace(name,std::move(b));}
    char* p(const char* name) const {return buffers.at(name)->at();}
    static std::uint64_t estimate(const VRConfig& c) {
        const std::uint64_t channels=add(mul(c.gqk,c.gkd),mul(c.gvh/2,c.gvd)),values=mul(c.gvh/2,c.gvd),heads=c.gvh/2;
        return mul(c.max_tokens,add(mul(add(mul(channels,2),add(mul(values,3),mul(heads,2))),2),mul(heads,8)));
    }
    void allocate(const VRConfig& c) {
        const std::uint64_t channels=add(mul(c.gqk,c.gkd),mul(c.gvh/2,c.gvd)),values=mul(c.gvh/2,c.gvd),heads=c.gvh/2,t=c.max_tokens;
        for(auto name:{"qkv","convolved"})allocate(name,mul(mul(t,channels),2));
        for(auto name:{"z","recurrent","mixed"})allocate(name,mul(mul(t,values),2));
        for(auto name:{"a","b"})allocate(name,mul(mul(t,heads),2));
        for(auto name:{"decay","beta"})allocate(name,mul(mul(t,heads),4));
        require(bytes==estimate(c),"head GDN workspace estimate mismatch");
    }
};
inline std::vector<int> head_gdn_rows(const VRConfig& c,int rank) {
    require(rank==0 || rank==1,"invalid GDN head rank");
    const int q=c.gqk*c.gkd,v=c.gvh*c.gvd;std::vector<int> rows;rows.reserve(q+v/2);
    for(auto part:{std::pair<int,int>{rank*q/2,(rank+1)*q/2},{q+rank*q/2,q+(rank+1)*q/2},{2*q+rank*v/2,2*q+(rank+1)*v/2}})
        for(int i=part.first;i<part.second;++i)rows.push_back(i);
    return rows;
}
template<class M> void head_gdn_upload_rows(M& m,Buffer& target,const VRMatrix& matrix,int rank) {
    const auto indices=head_gdn_rows(m.c,rank);const auto local=static_cast<std::uint64_t>(indices.size());
    for(bool scale:{false,true}) {
        const auto stride=scale?mul(padded(matrix)/32,2):static_cast<std::uint64_t>(padded(matrix));
        require(stride<=m.stage_bytes,"head GDN encoded row exceeds staging");
        for(std::uint64_t start=0;start<local;) {
            const auto count=std::min(local-start,m.stage_bytes/stride);
            for(std::uint64_t j=0;j<count;++j)m.read_bytes(static_cast<char*>(m.staging.ptr)+mul(j,stride),add(scale?matrix.scales:matrix.offset,mul(indices[start+j],stride)),stride);
            const auto offset=add(scale?mul(local,padded(matrix)):0,mul(start,stride));
            {Device device(rank?m.c.second_device:m.c.device);const auto stream=rank?m.second_stream.ptr:m.stream.ptr;
                check(cudaMemcpyAsync(target.at(offset),m.staging.ptr,mul(count,stride),cudaMemcpyHostToDevice,stream));check(cudaStreamSynchronize(stream));}
            m.uploaded=add(m.uploaded,mul(count,stride));start+=count;
        }
    }
}
// Load-time only: gather source constants on CPU without changing their bytes.
// No full tensor expansion or new GPU gathering kernel is used during forward.
template<class M> const void* head_gdn_constant(M& m,const void* source,int rows,int cols,int scalar_bytes,const std::vector<int>& indices,int rank) {
    require(source && rows>0 && cols>0 && (scalar_bytes==2 || scalar_bytes==4),"invalid head GDN constant source");
    cudaPointerAttributes attr{};check(cudaPointerGetAttributes(&attr,source));require(attr.type==cudaMemoryTypeDevice && attr.device==m.c.device,"head GDN constant source device mismatch");
    std::vector<unsigned char> input(mul(mul(rows,cols),scalar_bytes)),selected(mul(mul(rows,indices.size()),scalar_bytes));
    check(cudaMemcpy(input.data(),source,input.size(),cudaMemcpyDeviceToHost));
    for(int row=0;row<rows;++row)for(std::size_t j=0;j<indices.size();++j){require(indices[j]>=0 && indices[j]<cols,"head GDN constant column out of range");std::memcpy(selected.data()+mul(add(mul(row,indices.size()),j),scalar_bytes),input.data()+mul(add(mul(row,cols),indices[j]),scalar_bytes),scalar_bytes);}
    auto b=std::make_unique<Buffer>();{Device device(rank?m.c.second_device:m.c.device);b->allocate(selected.size());check(cudaMemcpy(b->ptr,selected.data(),selected.size(),cudaMemcpyHostToDevice));}
    const void* result=b->ptr;if(rank)m.second_device_bytes=add(m.second_device_bytes,b->bytes);else m.device_bytes=add(m.device_bytes,b->bytes);m.constants.push_back(std::move(b));return result;
}
template<class M> void head_gdn_constants(M& m,const VRLayer& source,VRLayer& target,int rank) {
    const auto& c=m.c;const int packed=2*c.gqk*c.gkd+c.gvh*c.gvd;
    auto rows=head_gdn_rows(c,rank);std::vector<int> heads,ab,gamma;
    for(int i=rank*c.gvh/2;i<(rank+1)*c.gvh/2;++i){heads.push_back(i);for(int k=0;k<c.hidden;++k)ab.push_back(i*c.hidden+k);}
    for(int i=0;i<c.gvd;++i)gamma.push_back(i);
    target.conv=head_gdn_constant(m,source.conv,c.convolution,packed,source.conv_fp32?4:2,rows,rank);
    target.rate=head_gdn_constant(m,source.rate,1,c.gvh,4,heads,rank);target.bias=head_gdn_constant(m,source.bias,1,c.gvh,4,heads,rank);
    target.a=head_gdn_constant(m,source.a,1,c.gvh*c.hidden,4,ab,rank);target.b=head_gdn_constant(m,source.b,1,c.gvh*c.hidden,4,ab,rank);
    target.gamma=head_gdn_constant(m,source.gamma,1,c.gvd,source.gamma_fp32?4:2,gamma,rank);
}
template<class M> void head_gdn_rank(M& m,int rank,const VRLayer& layer,const void* input,int t,std::span<const HeadGDNSegment> segments) {
    const auto& c=m.c;const auto stream=rank?m.second_stream.ptr:m.stream.ptr;auto& work=m.head_work[rank];auto& quant=rank?m.second_quant:m.first_quant;
    auto& weights=rank?m.second_weights:m.resident_weights;const int h=c.gvh/2,qk=c.gqk/2,v=h*c.gvd,width=2*qk*c.gkd+v;
    VQFOutput out[2]{};
    for(int i=0;i<2;++i){const auto& matrix=layer.matrix[i];const auto& b=*weights.at(matrix.offset);const int n=i?v:width;out[i]={static_cast<const std::int8_t*>(b.ptr),reinterpret_cast<const __half*>(b.at(mul(n,padded(matrix)))),reinterpret_cast<__half*>(work.p(i?"z":"qkv")),n,n};}
    op(venturi_q8_dp4a_quantize(input,quant.ptr,quant.bytes,t,c.hidden,2,stream));op(venturi_q8_fused_run(quant.ptr,quant.bytes,out,2,t,c.hidden,padded(layer.matrix[0]),2,stream));
    for(int j=0;j<2;++j)op(venturi_fp32_linear_run(input,static_cast<const float*>(j?layer.b:layer.a),work.p(j?"b":"a"),t,h,c.hidden,0,stream));
    op(venturi_gates_run(work.p("a"),work.p("b"),static_cast<const float*>(layer.rate),static_cast<const float*>(layer.bias),reinterpret_cast<float*>(work.p("decay")),reinterpret_cast<float*>(work.p("beta")),t,h,0,1,stream));
    for(const auto& segment:segments) {
        const auto& cache=segment.rank[rank];auto at=[&](const char* name,int stride,int bytes=2){return work.p(name)+mul(mul(segment.row,stride),bytes);};
        if(segment.prefixes)op(venturi_history_prefix_run(at("qkv",width),cache.history,cache.prefix_history[0],cache.prefix_history[1],width,c.convolution,segment.prefixes,0,stream));
        op(venturi_conv_run(at("qkv",width),layer.conv,cache.history,at("convolved",width),cache.history,segment.count,width,c.convolution,0,layer.conv_fp32,stream));
        auto* q=at("convolved",width);const auto stride=mul(qk*c.gkd,2);auto* decay=reinterpret_cast<float*>(at("decay",h,4));auto* beta=reinterpret_cast<float*>(at("beta",h,4));
        if(segment.prefixes)op(venturi_gdn_prefix_run(q,q+stride,q+2*stride,decay,beta,static_cast<const float*>(cache.state),at("recurrent",v),static_cast<float*>(cache.state),segment.count,qk,h,c.gkd,c.gvd,1,1,0,c.gdn_scale,cache.prefix_state[0],cache.prefix_state[1],segment.prefixes,stream));
        else op(venturi_gdn_compact_run(q,q+stride,q+2*stride,decay,beta,static_cast<const float*>(cache.state),at("recurrent",v),static_cast<float*>(cache.state),segment.count,qk,h,c.gkd,c.gvd,1,1,0,c.gdn_scale,stream));
    }
    op(venturi_gated_rms_run(work.p("recurrent"),work.p("z"),layer.gamma,work.p("mixed"),t,h,c.gvd,0,layer.gamma_fp32,c.epsilon,stream));
    const auto local=m.column_descriptor(layer.matrix[2]);const auto& b=*weights.at(local.offset);auto& partial=rank?m.column_peer:m.column_local;
    op(venturi_q8_dp4a_quantize(work.p("mixed"),quant.ptr,quant.bytes,t,v,2,stream));
    op(venturi_q8_column_partial_run(quant.ptr,quant.bytes,static_cast<const std::int8_t*>(b.ptr),reinterpret_cast<const __half*>(b.at(mul(local.n,padded(local)))),static_cast<float*>(partial.ptr),t,c.hidden,v,padded(local),stream));
}
template<class M> void head_gdn_forward(M& m,int index,const void* normalized,void* output,int t,std::span<const HeadGDNSegment> segments) {
    require(m.head_gdn_enabled && t>0 && t<=m.c.max_tokens && !segments.empty(),"invalid head GDN invocation");
    int end=0;for(const auto& segment:segments){require(segment.row==end && segment.count>0 && segment.count<=t-end && (segment.prefixes==0 || ((segment.prefixes==1 || segment.prefixes==2) && segment.count==segment.prefixes+1)),"invalid private head GDN segment partition");for(const auto& rank:segment.rank){require(rank.history && rank.state,"missing private head GDN cache");for(int j=0;j<segment.prefixes;++j)require(rank.prefix_history[j] && rank.prefix_state[j],"missing private peer prefix cache");}end+=segment.count;}require(end==t,"head GDN segment rows incomplete");
    m.source();const auto bytes=mul(mul(t,m.c.hidden),2),partial_bytes=mul(bytes,2);
    check(cudaMemcpyAsync(m.host_input.ptr,normalized,bytes,cudaMemcpyDeviceToHost,m.stream.ptr));check(cudaEventRecord(m.first_ready.ptr,m.stream.ptr));
    {Device device(m.c.second_device);check(cudaStreamWaitEvent(m.second_stream.ptr,m.first_ready.ptr,0));check(cudaMemcpyAsync(m.second_input.ptr,m.host_input.ptr,bytes,cudaMemcpyHostToDevice,m.second_stream.ptr));
        head_gdn_rank(m,1,m.head_peer_layers[index],m.second_input.ptr,t,segments);check(cudaMemcpyAsync(m.column_host.ptr,m.column_peer.ptr,partial_bytes,cudaMemcpyDeviceToHost,m.second_stream.ptr));check(cudaEventRecord(m.second_ready.ptr,m.second_stream.ptr));}
    head_gdn_rank(m,0,m.layers[index],normalized,t,segments);check(cudaStreamWaitEvent(m.stream.ptr,m.second_ready.ptr,0));check(cudaMemcpyAsync(m.column_remote.ptr,m.column_host.ptr,partial_bytes,cudaMemcpyHostToDevice,m.stream.ptr));op(venturi_q8_column_sum(static_cast<const float*>(m.column_local.ptr),static_cast<const float*>(m.column_remote.ptr),output,static_cast<int>(mul(t,m.c.hidden)),m.stream.ptr));
    m.peer_input_bytes=add(m.peer_input_bytes,bytes);m.peer_output_bytes=add(m.peer_output_bytes,partial_bytes);
}
