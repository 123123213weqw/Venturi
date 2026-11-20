// Native two-stage pipeline: resident layer partitions and request-owned state.
// Each stage releases its model lock before the next stage; other requests can
// execute on the other GPU. No Python token/layer callbacks or peer-copy fallback.
#pragma once

namespace vr {
inline bool compatible(const VRConfig& a,const VRConfig& b) {
    return a.dtype==b.dtype && a.layers==b.layers && a.hidden==b.hidden &&
        a.intermediate==b.intermediate && a.vocab==b.vocab && a.hq==b.hq &&
        a.hkv==b.hkv && a.dim==b.dim && a.rotary==b.rotary && a.gqk==b.gqk &&
        a.gvh==b.gvh && a.gkd==b.gkd && a.gvd==b.gvd && a.convolution==b.convolution &&
        a.max_tokens==b.max_tokens && a.max_position==b.max_position &&
        a.epsilon==b.epsilon && a.gdn_scale==b.gdn_scale && a.attention_scale==b.attention_scale;
}
struct Pipeline {
    std::shared_ptr<Model> first,second;
    Pipeline(std::shared_ptr<Model> a,std::shared_ptr<Model> b):first(std::move(a)),second(std::move(b)) {
        require(first->stage && second->stage && first->begin==0 && first->end==second->begin &&
            second->end==second->c.layers && first->c.device!=second->c.device &&
            compatible(first->c,second->c),"incompatible native pipeline stages");
        require(first->files.size()==second->files.size(),"pipeline source spans differ");
        for (std::size_t i=0;i<first->files.size();++i) {
            const auto& a=*first->files[i];const auto& b=*second->files[i];
            require(a.path==b.path && a.logical==b.logical && a.begin==b.begin && a.bytes==b.bytes &&
                a.same(b.stamp),"pipeline stages must use identical pinned source files");
        }
    }
};
using PipelineHandle=std::shared_ptr<Pipeline>;
struct PipelineSession {
    std::shared_ptr<Pipeline> model;
    std::unique_ptr<Session> first,second;
    Pinned activation;
    std::uint64_t activation_bytes=0,selection_bytes=0,handoffs=0;
    std::mutex mutex;
    explicit PipelineSession(std::shared_ptr<Pipeline> p,int capacity):model(std::move(p)) {
        std::scoped_lock locks(model->first->mutex,model->second->mutex);
        { Device d(model->first->c.device);first=std::make_unique<Session>(model->first,capacity); }
        { Device d(model->second->c.device);second=std::make_unique<Session>(model->second,capacity); }
        activation.allocate(mul(mul(model->first->c.max_tokens,model->first->c.hidden),2));
    }
    bool failed() const { return first->failed || second->failed; }
    void poison() { first->failed=true;second->failed=true; }
    void input(const std::int64_t* tokens,int n) {
        require(!failed(),"native Qwen pipeline request failed; reset required");
        require(first->position==second->position,"native pipeline cursor mismatch");
        first->input(tokens,n);second->input(tokens,n);
    }
    void forward(const std::int64_t* tokens,int n,float* logits,bool all,void* trace) {
        input(tokens,n);const auto& c=model->first->c;const int initial=first->position;bool started=false;
        try {
            for (int start=0;start<n;start+=c.max_tokens) {
                const int t=std::min(c.max_tokens,n-start),pos=initial+start;
                const auto bytes=mul(mul(t,c.hidden),2);
                {
                    std::lock_guard<std::mutex> lock(model->first->mutex);Device d(model->first->c.device);
                    started=true;model->first->source();first->embedding(tokens+start,t,first->p("x"));
                    auto* x=first->run_layers(pos,t,first->p("x"),trace,n,start);
                    check(cudaMemcpyAsync(activation.ptr,x,bytes,cudaMemcpyDeviceToHost,model->first->stream.ptr));
                    model->first->sync();
                }
                // The request owns pinned storage. Its D2H is complete before
                // stage one can overwrite stage scratch or stage two reads it.
                {
                    std::lock_guard<std::mutex> lock(model->second->mutex);Device d(model->second->c.device);
                    model->second->source();
                    check(cudaMemcpyAsync(second->p("x"),activation.ptr,bytes,cudaMemcpyHostToDevice,model->second->stream.ptr));
                    auto* x=second->run_layers(pos,t,second->p("x"),trace,n,start);
                    if (all || start+t==n) second->output(x,t,all,logits?logits+(all?mul(start,c.vocab):0):nullptr);
                    model->second->sync();
                }
                activation_bytes=add(activation_bytes,bytes);++handoffs;
            }
            model->first->source();model->second->source();first->position=second->position=initial+n;
        } catch (...) {
            if (started) poison();throw;
        }
    }
    void reset() {
        try {
            {std::lock_guard<std::mutex> lock(model->first->mutex);Device d(model->first->c.device);first->reset();}
            {std::lock_guard<std::mutex> lock(model->second->mutex);Device d(model->second->c.device);second->reset();}
        } catch (...) {poison();throw;}
    }
};
}

extern "C" void* venturi_qwen_stage_create(const VRConfig* c,const VRSpan* spans,int count,const VRLayer* layers,const VRMatrix* embedding,const VRMatrix* head,const void* norm,const float* inverse,int begin,int end) {
    try {vr::require(c && embedding && head,"null pipeline stage descriptor");vr::Device d(c->device);
        cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,c->device));vr::require(prop.major*10+prop.minor==VENTURI_BLOCKS_ARCH,"pipeline architecture mismatch");
        return new vr::Handle(std::make_shared<vr::Model>(*c,spans,count,layers,*embedding,*head,norm,inverse,begin,end,true));
    } catch (const std::exception& e) {error_text=e.what();return nullptr;}
}
extern "C" void* venturi_qwen_pipeline_create(void* a,void* b) {
    try {vr::require(a && b,"null pipeline stage handle");return new vr::PipelineHandle(std::make_shared<vr::Pipeline>(*static_cast<vr::Handle*>(a),*static_cast<vr::Handle*>(b)));
    } catch (const std::exception& e) {error_text=e.what();return nullptr;}
}
extern "C" void venturi_qwen_pipeline_destroy(void* h) {delete static_cast<vr::PipelineHandle*>(h);}
extern "C" void* venturi_qwen_pipeline_session(void* h,int capacity) {
    try {vr::require(h,"null pipeline model");return new vr::PipelineSession(*static_cast<vr::PipelineHandle*>(h),capacity);
    } catch (const std::exception& e) {error_text=e.what();return nullptr;}
}
extern "C" void venturi_qwen_pipeline_session_destroy(void* h) {delete static_cast<vr::PipelineSession*>(h);}
extern "C" int venturi_qwen_pipeline_forward(void* h,const std::int64_t* ids,int n,float* logits,std::uint64_t elements,int all,void* trace,std::uint64_t trace_elements) {
    try {vr::require(h,"null pipeline session");auto& s=*static_cast<vr::PipelineSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);
        const auto& c=s.model->first->c;
        vr::require((all==0 || all==1) && logits && n>0 && elements==vr::mul(all?n:1,c.vocab) &&
            ((!trace && !trace_elements) || (trace && trace_elements==vr::mul(vr::mul(n,c.hidden),c.layers))),"native pipeline output/trace shape mismatch");
        s.forward(ids,n,logits,all,trace);return 0;
    } catch (const std::exception& e) {error_text=e.what();return -1;}
}
extern "C" int venturi_qwen_pipeline_generate(void* h,const std::int64_t* prompt,int n,int maximum,const std::int64_t* eos,int eos_count,std::int64_t* tokens,int* count) {
    try {vr::require(h,"null pipeline session");auto& s=*static_cast<vr::PipelineSession*>(h);std::lock_guard<std::mutex> request(s.mutex);
        auto& m=*s.model->second;
        vr::require(maximum>0 && tokens && count && eos_count>=0 && (eos || !eos_count),"invalid pipeline generation arguments");*count=0;s.input(prompt,n);
        vr::require(vr::add(n,maximum)<=static_cast<std::uint64_t>(s.first->capacity-s.first->position),"native pipeline generation exceeds capacity");
        for (int i=0;i<eos_count;++i) vr::require(eos[i]>=0 && eos[i]<m.c.vocab,"native pipeline EOS out of range");
        try {
            for (int step=0;step<maximum;++step) {
                s.forward(step?tokens+step-1:prompt,step?1:n,nullptr,false,nullptr);
                std::int64_t selected[2];
                {
                    std::lock_guard<std::mutex> lock(m.mutex);vr::Device d(m.c.device);
                    vr::op(venturi_greedy_run(s.second->p("logits"),static_cast<std::int64_t*>(s.second->p("selection")),s.second->p("select_scratch"),vr::Session::min_tiles(m.c.vocab)*16,1,m.c.vocab,m.c.dtype,m.stream.ptr));
                    check(cudaMemcpyAsync(selected,s.second->p("selection"),16,cudaMemcpyDeviceToHost,m.stream.ptr));m.sync();
                }
                s.selection_bytes=vr::add(s.selection_bytes,16);
                if (selected[1] || selected[0]<0 || selected[0]>=m.c.vocab) throw std::runtime_error("native pipeline greedy rejected nonfinite logits");
                tokens[step]=selected[0];*count=step+1;
                if (eos_count && std::find(eos,eos+eos_count,selected[0])!=eos+eos_count) break;
            }
            return 0;
        } catch (...) {s.poison();throw;}
    } catch (const std::exception& e) {error_text=e.what();return -1;}
}
extern "C" int venturi_qwen_pipeline_info(void* h,std::uint64_t* out) {
    try {vr::require(h && out,"null pipeline info");auto& s=*static_cast<vr::PipelineSession*>(h);std::lock_guard<std::mutex> request(s.mutex);
        auto& a=*s.model->first;auto& b=*s.model->second;std::scoped_lock locks(a.mutex,b.mutex);
        const std::uint64_t v[]={static_cast<std::uint64_t>(s.first->position),s.failed()?1ULL:0ULL,a.device_bytes,s.first->device_bytes,
            a.stage_bytes+b.stage_bytes+s.activation.bytes,a.payload,vr::add(a.uploaded,b.uploaded),0};std::copy(v,v+8,out);return 0;
    } catch (const std::exception& e) {error_text=e.what();return -1;}
}
extern "C" int venturi_qwen_pipeline_parallel_info(void* h,std::uint64_t* out) {
    try {vr::require(h && out,"null pipeline parallel info");auto& s=*static_cast<vr::PipelineSession*>(h);std::lock_guard<std::mutex> request(s.mutex);
        const std::uint64_t v[]={static_cast<std::uint64_t>(s.model->second->c.device+1),s.model->second->device_bytes,s.activation_bytes,s.selection_bytes};std::copy(v,v+4,out);return 0;
    } catch (const std::exception& e) {error_text=e.what();return -1;}
}
extern "C" int venturi_qwen_pipeline_details(void* h,std::uint64_t* out) {
    try {vr::require(h && out,"null pipeline details");auto& s=*static_cast<vr::PipelineSession*>(h);std::lock_guard<std::mutex> request(s.mutex);
        const std::uint64_t v[]={static_cast<std::uint64_t>(s.model->first->end),s.second->device_bytes,s.handoffs,s.selection_bytes};std::copy(v,v+4,out);return 0;
    } catch (const std::exception& e) {error_text=e.what();return -1;}
}
extern "C" int venturi_qwen_pipeline_reset(void* h) {
    try {vr::require(h,"null pipeline reset");auto& s=*static_cast<vr::PipelineSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);s.reset();return 0;
    } catch (const std::exception& e) {error_text=e.what();return -1;}
}
extern "C" int venturi_qwen_pipeline_snapshot(void* h,int layer,int kind,void* target,std::uint64_t bytes) {
    try {vr::require(h,"null pipeline snapshot");auto& s=*static_cast<vr::PipelineSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);
        const auto& c=s.model->first->c;vr::require(layer>=0 && layer<c.layers && kind>=0 && kind<=3,"invalid native pipeline snapshot slot");
        auto& stage=layer<s.model->first->end?*s.first:*s.second;auto& m=*stage.model;std::lock_guard<std::mutex> gpu(m.mutex);vr::Device d(m.c.device);
        const auto* src=kind==0?stage.history[layer].get():kind==1?stage.state[layer].get():kind==2?stage.keys[layer].get():stage.values[layer].get();vr::require(src,"native pipeline snapshot slot absent");
        const auto n=kind<2?src->bytes:vr::mul(vr::mul(stage.position,c.hkv*c.dim),2);vr::require(bytes==n && (target || !n),"native pipeline snapshot byte size mismatch");
        if (n) check(cudaMemcpyAsync(target,src->ptr,n,cudaMemcpyDeviceToHost,m.stream.ptr));m.sync();return 0;
    } catch (const std::exception& e) {error_text=e.what();return -1;}
}
extern "C" int venturi_qwen_pipeline_debug_failure(void* h,int layer,int position) {
    try {vr::require(h,"null pipeline debug session");auto& s=*static_cast<vr::PipelineSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);
        vr::require(layer>=-1 && layer<s.model->first->c.layers && position>=0,"invalid native pipeline debug failure");
        s.first->fault_layer=s.second->fault_layer=-1;
        if (layer>=0) {auto& stage=layer<s.model->first->end?*s.first:*s.second;stage.fault_layer=layer;stage.fault_position=position;}return 0;
    } catch (const std::exception& e) {error_text=e.what();return -1;}
}
