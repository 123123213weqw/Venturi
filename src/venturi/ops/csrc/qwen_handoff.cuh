// Explicit same-process PD handoff. No tensor arithmetic or hidden transport
// fallback. Only static, unsharded request state is accepted by this version.
#pragma once

namespace vr {
struct StateCopy { const Buffer* source; Buffer* target; std::uint64_t bytes; };

inline std::vector<StateCopy> handoff_plan(Batch& producer,int from,Batch& consumer,int to,int position) {
    auto& a=*producer.model;auto& b=*consumer.model;
    require(&a!=&b && a.c.device!=b.c.device,"PD requires distinct model instances on different devices");
    auto adjusted=b.c;adjusted.max_tokens=a.c.max_tokens;
    require(compatible(a.c,adjusted) && !a.parallel() && !b.parallel() &&
            !a.stage && !b.stage && !a.kv_tier && !b.kv_tier &&
            a.activation_passes==b.activation_passes && a.experiments==b.experiments,
            "PD v1 requires compatible TP1 static-state models and arithmetic");
    require(a.files.size()==b.files.size(),"PD checkpoint source spans differ");
    for(std::size_t i=0;i<a.files.size();++i) {
        const auto& x=*a.files[i];const auto& y=*b.files[i];
        require(x.path==y.path && x.logical==y.logical && x.begin==y.begin &&
                x.bytes==y.bytes && x.same(y.stamp),"PD requires identical pinned checkpoint files");
    }
    require(from>=0 && from<static_cast<int>(producer.requests.size()) &&
            to>=0 && to<static_cast<int>(consumer.requests.size()),"invalid PD slot");
    auto& s=*producer.requests[from];auto& d=*consumer.requests[to];
    require(!s.failed && !d.failed && !s.verification_trace && !d.verification_trace &&
            s.position==position && position>0 && d.position==0 && position<=d.capacity,
            "PD requires a stable live source prefix and fresh compatible destination");
    std::vector<StateCopy> plan;
    auto append=[&](const std::unique_ptr<Buffer>& src,std::unique_ptr<Buffer>& dst,std::uint64_t n) {
        require(src && dst && n<=src->bytes && n<=dst->bytes,"PD state plane geometry mismatch");
        plan.push_back({src.get(),dst.get(),n});
    };
    for(int i=0;i<a.c.layers;++i) {
        require(a.layers[i].kind==b.layers[i].kind,"PD layer kinds differ");
        if(a.layers[i].kind==0) {
            require(s.history[i]->bytes==d.history[i]->bytes && s.state[i]->bytes==d.state[i]->bytes,
                    "PD recurrent state layout mismatch");
            append(s.history[i],d.history[i],s.history[i]->bytes);
            append(s.state[i],d.state[i],s.state[i]->bytes);
        } else {
            const auto n=mul(mul(position,a.c.hkv*a.c.dim),2);
            append(s.keys[i],d.keys[i],n);append(s.values[i],d.values[i],n);
        }
    }
    return plan;
}
} // namespace vr

// Additive ABI: output = [payload bytes, position, component count, pinned bytes].
// Host mode is bounded pinned staging, NOT CPU inference. Peer mode fails closed
// if unsupported; it never silently substitutes host transport.
extern "C" int venturi_qwen_batch_handoff(void* source,int source_slot,void* target,int target_slot,
    int position,int transport,std::uint64_t* output) {
    try {
        vr::require(source && target && source!=target && output && (transport==0 || transport==1),
                    "invalid PD handoff arguments");
        auto& a=*static_cast<vr::Batch*>(source);auto& b=*static_cast<vr::Batch*>(target);
        vr::require(a.model.get()!=b.model.get(),"PD model owners must differ");
        std::scoped_lock batches(a.mutex,b.mutex);
        std::scoped_lock models(a.model->mutex,b.model->mutex);
        {vr::Device device(a.model->c.device);a.model->source();a.model->sync();}
        {vr::Device device(b.model->c.device);b.model->source();b.model->sync();}
        const auto plan=vr::handoff_plan(a,source_slot,b,target_slot,position);
        std::uint64_t total=0,largest=0;
        for(const auto& part:plan){total=vr::add(total,part.bytes);largest=std::max(largest,part.bytes);}
        vr::Pinned staging;
        const auto chunk=std::min<std::uint64_t>(largest,8*1024*1024);
        if(transport==0 && chunk)staging.allocate(chunk);
        if(transport==1) {
            const std::array<std::pair<int,int>,2> links{{{a.model->c.device,b.model->c.device},
                                                        {b.model->c.device,a.model->c.device}}};
            for(auto pair:links) {
                vr::Device device(pair.first);int allowed=0;
                check(cudaDeviceCanAccessPeer(&allowed,pair.first,pair.second));
                vr::require(allowed,"PD peer transport unsupported; select host explicitly");
                const auto status=cudaDeviceEnablePeerAccess(pair.second,0);
                if(status==cudaErrorPeerAccessAlreadyEnabled)(void)cudaGetLastError();else check(status);
            }
        }
        auto& destination=*b.requests[target_slot];
        // The cursor is committed only after ALL planes and source guards pass.
        // A failed partial import cannot be used until an explicit slot reset.
        destination.failed=true;
        try {
            for(const auto& part:plan) {
                if(transport==1) {
                    vr::Device device(b.model->c.device);
                    if(part.bytes)check(cudaMemcpyPeerAsync(part.target->ptr,b.model->c.device,
                        part.source->ptr,a.model->c.device,part.bytes,b.model->stream.ptr));
                } else {
                    for(std::uint64_t offset=0;offset<part.bytes;offset+=chunk) {
                        const auto n=std::min(chunk,part.bytes-offset);
                        {vr::Device device(a.model->c.device);
                            check(cudaMemcpyAsync(staging.ptr,part.source->at(offset),n,
                                cudaMemcpyDeviceToHost,a.model->stream.ptr));a.model->sync();}
                        {vr::Device device(b.model->c.device);
                            check(cudaMemcpyAsync(part.target->at(offset),staging.ptr,n,
                                cudaMemcpyHostToDevice,b.model->stream.ptr));b.model->sync();}
                    }
                }
            }
            {vr::Device device(b.model->c.device);b.model->sync();b.model->source();}
            a.model->source();
            destination.position=position;destination.failed=false;
            destination.first_token_seconds=destination.generation_seconds=0;
        } catch(...) {
            // Finish outstanding copies before releasing source leases/staging.
            try {vr::Device device(b.model->c.device);b.model->sync();}catch(...){}
            try {vr::Device device(a.model->c.device);a.model->sync();}catch(...){}
            throw;
        }
        output[0]=total;output[1]=position;output[2]=plan.size();output[3]=staging.bytes;
        return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
