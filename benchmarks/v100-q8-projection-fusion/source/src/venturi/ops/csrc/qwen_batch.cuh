// Grouped projections, independently owned sequence caches and native generation.
#pragma once

namespace vr {
struct BatchSegment {int slot,row,count;};
struct Batch {
    std::shared_ptr<Model> model;
    std::vector<std::unique_ptr<Session>> requests;
    std::unique_ptr<Session> work;
    Buffer value,last_hidden,head_input,selection,select_scratch;
    std::uint64_t extra_bytes=0,group_calls=0,input_rows=0,decode_calls=0,prefill_calls=0,head_calls=0,head_rows=0,max_rows=0;
    std::mutex mutex;
    Batch(std::shared_ptr<Model> m,const int* capacities,int count):model(std::move(m)) {
        const auto& c=model->c;
        require(!model->stage && c.resident==1 && capacities && count>0 && count<=32 && count<=c.max_tokens,"batch requires a complete resident target and 1..min(32,max_tokens) slots");
        std::uint64_t needed=Session::estimate(*model,std::min(c.max_tokens,c.max_position),false);
        for(int i=0;i<count;++i)needed=add(needed,Session::estimate(*model,capacities[i]));
        const auto additional=add(mul(mul(c.max_tokens,c.hkv*c.dim),2),add(mul(mul(count,c.hidden),4),mul(count,16+Session::min_tiles(c.vocab)*16)));
        model->guard(add(needed,additional));
        work=std::make_unique<Session>(model,std::min(c.max_tokens,c.max_position),false);
        for(int i=0;i<count;++i)requests.push_back(std::make_unique<Session>(model,capacities[i]));
        auto alloc=[&](Buffer& b,std::uint64_t n){b.allocate(n);extra_bytes=add(extra_bytes,n);};
        alloc(value,mul(mul(c.max_tokens,c.hkv*c.dim),2));
        alloc(last_hidden,mul(mul(count,c.hidden),2));alloc(head_input,last_hidden.bytes);
        alloc(selection,mul(count,16));alloc(select_scratch,mul(count,Session::min_tiles(c.vocab)*16));
        require(extra_bytes==additional,"batch memory estimate mismatch");
    }
    std::vector<int> slots(const int* p,int count) {
        require(p && count>0 && count<=static_cast<int>(requests.size()),"invalid selected batch slot count");
        std::vector<int> ids(p,p+count);std::set<int> unique;
        for(auto i:ids)require(i>=0 && i<static_cast<int>(requests.size()) && unique.insert(i).second,"invalid or duplicate batch slot");
        return ids;
    }
    void input(const std::vector<int>& ids,const std::int64_t* tokens,const int* offsets) {
        require(tokens && offsets && offsets[0]==0,"invalid batch input offsets");
        for(std::size_t j=0;j<ids.size();++j) {
            require(offsets[j]>=0 && offsets[j+1]>offsets[j],"batch prompts must be nonempty with increasing offsets");
            requests[ids[j]]->input(tokens+offsets[j],offsets[j+1]-offsets[j]);
        }
    }
    void poison(const std::vector<int>& ids) noexcept {
        for(auto i:ids)requests[i]->failed=true;
        try {model->sync();}catch(...){}
    }
    char* row(const char* name,int offset,int width,int bytes=2) {return static_cast<char*>(work->p(name))+mul(mul(offset,width),bytes);}
    // Each segment is a contiguous chunk of ONE sequence. Grouped linear/norm
    // rows are independent; convolution/GDN/attention always use its own cache.
    void group(const std::vector<BatchSegment>& segments,const std::vector<std::int64_t>& tokens,bool decode) {
        auto& m=*model;auto& w=*work;const auto& c=m.c;auto stream=m.stream.ptr;
        const int n=static_cast<int>(tokens.size()),width=2*c.gqk*c.gkd+c.gvh*c.gvd,qw=c.hq*c.dim,kw=c.hkv*c.dim;
        require(n>0 && n<=c.max_tokens,"batch row tile exceeds model workspace");
        w.embedding(tokens.data(),n,w.p("x"));void* x=w.p("x");void* out=w.p("out");
        for(int index=0;index<c.layers;++index) {
            const auto& l=m.layers[index];
            op(venturi_rms_run(x,static_cast<const float*>(l.norm),w.p("normalized"),n,c.hidden,c.dtype,c.epsilon,stream));
            if(l.kind==0) {
                m.shared_linears(w.p("normalized"),n,{{&l.matrix[0],w.p("qkv")},{&l.matrix[1],w.p("z")}});
                for(int j=0;j<2;++j) {
                    auto weight=j?l.b:l.a;auto output=w.p(j?"b":"a");
                    if(l.ab_fp32)op(venturi_fp32_linear_run(w.p("normalized"),static_cast<const float*>(weight),output,n,c.gvh,c.hidden,c.dtype,stream));
                    else {static_cast<Linear*>(m.dense.ptr)->tokens=n;op(venturi_dense_run(m.dense.ptr,w.p("normalized"),weight,output,m.blas_workspace.ptr,m.blas_workspace.bytes,stream));}
                }
                op(venturi_gates_run(w.p("a"),w.p("b"),static_cast<const float*>(l.rate),static_cast<const float*>(l.bias),static_cast<float*>(w.p("decay")),static_cast<float*>(w.p("beta")),n,c.gvh,c.dtype,1,stream));
                for(const auto& segment:segments) {
                    auto& request=*requests[segment.slot];const int r=segment.row,t=segment.count;
                    op(venturi_conv_run(row("qkv",r,width),l.conv,request.history[index]->ptr,row("convolved",r,width),request.history[index]->ptr,t,width,c.convolution,c.dtype,l.conv_fp32,stream));
                    auto* q=row("convolved",r,width);const auto stride=mul(c.gqk*c.gkd,2);
                    op(venturi_gdn_run(q,q+stride,q+2*stride,reinterpret_cast<const float*>(row("decay",r,c.gvh,4)),reinterpret_cast<const float*>(row("beta",r,c.gvh,4)),static_cast<const float*>(request.state[index]->ptr),row("recurrent",r,c.gvh*c.gvd),static_cast<float*>(request.state[index]->ptr),t,c.gqk,c.gvh,c.gkd,c.gvd,1,1,c.dtype,c.gdn_scale,stream));
                }
                op(venturi_gated_rms_run(w.p("recurrent"),w.p("z"),l.gamma,w.p("mixed"),n,c.gvh,c.gvd,c.dtype,l.gamma_fp32,c.epsilon,stream));
                m.linear(l.matrix[2],w.p("mixed"),w.p("projected"),n);
            } else {
                m.shared_linears(w.p("normalized"),n,{{&l.matrix[0],w.p("qg")},{&l.matrix[1],w.p("key")},{&l.matrix[2],value.ptr}});
                for(const auto& segment:segments) {
                    auto& request=*requests[segment.slot];const int r=segment.row,t=segment.count,pos=request.position;
                    auto* kt=request.keys[index]->at(mul(mul(pos,kw),2));auto* vt=request.values[index]->at(mul(mul(pos,kw),2));
                    check(cudaMemcpyAsync(vt,value.at(mul(mul(r,kw),2)),mul(mul(t,kw),2),cudaMemcpyDeviceToDevice,stream));
                    auto* cos=static_cast<const float*>(request.p("cos"))+mul(pos,c.rotary/2);auto* sin=static_cast<const float*>(request.p("sin"))+mul(pos,c.rotary/2);
                    op(venturi_qwen_qk_run(row("qg",r,2*qw),row("key",r,kw),static_cast<const float*>(l.qgamma),static_cast<const float*>(l.kgamma),cos,sin,row("query",r,qw),kt,t,c.hq,c.hkv,c.dim,c.rotary,c.dtype,c.epsilon,stream));
                    op(venturi_attention_run(row("query",r,qw),request.keys[index]->ptr,request.values[index]->ptr,row("attended",r,qw),t,pos+t,c.hq,c.hkv,c.dim,c.dtype,c.attention_scale,stream));
                }
                op(venturi_qwen_gate_run(w.p("attended"),w.p("qg"),w.p("attended"),n,c.hq,c.dim,c.dtype,stream));
                m.linear(l.matrix[3],w.p("attended"),w.p("projected"),n);
            }
            op(venturi_add_run(x,w.p("projected"),w.p("residual"),mul(n,c.hidden),c.dtype,stream));
            op(venturi_rms_run(w.p("residual"),static_cast<const float*>(l.post),w.p("mlp_norm"),n,c.hidden,c.dtype,c.epsilon,stream));
            m.shared_linears(w.p("mlp_norm"),n,{{&l.matrix[4],w.p("mlp_gate")},{&l.matrix[5],w.p("mlp_up")}});
            op(venturi_swiglu_run(w.p("mlp_gate"),w.p("mlp_up"),w.p("mlp_gate"),mul(n,c.intermediate),c.dtype,stream));
            m.linear(l.matrix[6],w.p("mlp_gate"),w.p("projected"),n);op(venturi_add_run(w.p("residual"),w.p("projected"),out,mul(n,c.hidden),c.dtype,stream));
            std::swap(x,out);
            for(const auto& segment:segments) {
                auto& request=*requests[segment.slot];
                if(index==request.fault_layer && request.position==request.fault_position){request.fault_layer=-1;throw std::runtime_error("injected native batch layer failure");}
            }
        }
        op(venturi_rms_run(x,m.norm,w.p("normalized"),n,c.hidden,c.dtype,c.epsilon,stream));
        for(const auto& segment:segments)check(cudaMemcpyAsync(last_hidden.at(mul(mul(segment.slot,c.hidden),2)),row("normalized",segment.row+segment.count-1,c.hidden),mul(c.hidden,2),cudaMemcpyDeviceToDevice,stream));
        m.sync();m.source();
        for(const auto& segment:segments)requests[segment.slot]->position+=segment.count;
        ++group_calls;input_rows=add(input_rows,n);max_rows=std::max(max_rows,static_cast<std::uint64_t>(n));if(decode)++decode_calls;else ++prefill_calls;
    }
    void prefill(const std::vector<int>& ids,const std::int64_t* tokens,const int* offsets) {
        std::vector<int> done(ids.size());
        for(;;) {
            int active=0;for(std::size_t i=0;i<ids.size();++i)if(done[i]<offsets[i+1]-offsets[i])++active;
            if(!active)break;const int share=std::max(1,model->c.max_tokens/active);
            std::vector<BatchSegment> segments;std::vector<std::int64_t> packed;
            for(std::size_t i=0;i<ids.size();++i) {
                const int count=std::min(share,offsets[i+1]-offsets[i]-done[i]);if(!count)continue;
                segments.push_back({ids[i],static_cast<int>(packed.size()),count});
                packed.insert(packed.end(),tokens+offsets[i]+done[i],tokens+offsets[i]+done[i]+count);done[i]+=count;
            }
            group(segments,packed,false);
        }
    }
    void head(const std::vector<int>& ids,std::vector<std::int64_t>* chosen,float* logits) {
        auto& m=*model;auto& w=*work;const auto& c=m.c;const int n=static_cast<int>(ids.size());
        for(int i=0;i<n;++i)check(cudaMemcpyAsync(head_input.at(mul(mul(i,c.hidden),2)),last_hidden.at(mul(mul(ids[i],c.hidden),2)),mul(c.hidden,2),cudaMemcpyDeviceToDevice,m.stream.ptr));
        m.linear(m.head,head_input.ptr,w.p("logits"),n);
        if(chosen) {
            chosen->resize(2*n);op(venturi_greedy_run(w.p("logits"),static_cast<std::int64_t*>(selection.ptr),select_scratch.ptr,select_scratch.bytes,n,c.vocab,c.dtype,m.stream.ptr));
            check(cudaMemcpyAsync(chosen->data(),selection.ptr,mul(n,16),cudaMemcpyDeviceToHost,m.stream.ptr));
        }
        if(logits)check(cudaMemcpyAsync(w.host_logits.data(),w.p("logits"),mul(mul(n,c.vocab),2),cudaMemcpyDeviceToHost,m.stream.ptr));
        m.sync();m.source();
        if(chosen)for(int i=0;i<n;++i)require((*chosen)[2*i]>=0 && (*chosen)[2*i]<c.vocab && !(*chosen)[2*i+1],"native batch greedy rejected nonfinite logits");
        if(logits)for(std::uint64_t j=0;j<mul(n,c.vocab);++j)logits[j]=c.dtype==0?__half2float(reinterpret_cast<const __half*>(w.host_logits.data())[j]):__bfloat162float(reinterpret_cast<const __nv_bfloat16*>(w.host_logits.data())[j]);
        ++head_calls;head_rows=add(head_rows,n);
    }
    void counters(){group_calls=input_rows=decode_calls=prefill_calls=head_calls=head_rows=max_rows=0;}
    void generate(const std::vector<int>& ids,const std::int64_t* tokens,const int* offsets,const int* maximum,const std::int64_t* eos,const int* eos_offsets,std::int64_t* out,int stride,int* counts) {
        input(ids,tokens,offsets);const auto& c=model->c;
        require(maximum && eos_offsets && eos_offsets[0]==0 && out && counts && stride>0,"invalid batch generation arguments");
        for(std::size_t j=0;j<ids.size();++j) {
            require(maximum[j]>0 && maximum[j]<=stride && add(offsets[j+1]-offsets[j],maximum[j])<=static_cast<std::uint64_t>(requests[ids[j]]->capacity-requests[ids[j]]->position),"batch generation exceeds slot capacity");
            require(eos_offsets[j]>=0 && eos_offsets[j+1]>=eos_offsets[j] && (eos || !eos_offsets[j+1]),"invalid batch EOS offsets");
            for(int k=eos_offsets[j];k<eos_offsets[j+1];++k)require(eos[k]>=0 && eos[k]<c.vocab,"batch EOS token out of range");
            counts[j]=0;
        }
        counters();const double start=native_clock();bool begun=false;
        auto stopped=[&](int j,std::int64_t value){for(int k=eos_offsets[j];k<eos_offsets[j+1];++k)if(eos[k]==value)return true;return false;};
        auto emit=[&](int j,std::int64_t value){out[mul(j,stride)+counts[j]++]=value;if(counts[j]==maximum[j] || stopped(j,value))requests[ids[j]]->generation_seconds=native_clock()-start;};
        try {
            begun=true;prefill(ids,tokens,offsets);std::vector<std::int64_t> chosen;head(ids,&chosen,nullptr);
            const auto first=native_clock()-start;
            for(std::size_t j=0;j<ids.size();++j){requests[ids[j]]->first_token_seconds=first;requests[ids[j]]->generation_seconds=0;emit(j,chosen[2*j]);}
            for(;;) {
                std::vector<BatchSegment> segments;std::vector<std::int64_t> packed;std::vector<int> active,indices;
                for(std::size_t j=0;j<ids.size();++j)if(counts[j]<maximum[j] && !stopped(j,out[mul(j,stride)+counts[j]-1])) {
                    segments.push_back({ids[j],static_cast<int>(packed.size()),1});packed.push_back(out[mul(j,stride)+counts[j]-1]);active.push_back(ids[j]);indices.push_back(j);
                }
                if(active.empty())break;group(segments,packed,true);head(active,&chosen,nullptr);
                for(std::size_t j=0;j<active.size();++j)emit(indices[j],chosen[2*j]);
            }
        }catch(...){if(begun)poison(ids);throw;}
    }
};
}

extern "C" void* venturi_qwen_batch_create(void* owner,const int* capacities,int count) {
    try {vr::require(owner,"null batch target");auto m=*static_cast<vr::Handle*>(owner);vr::Device device(m->c.device);std::lock_guard<std::mutex> lock(m->mutex);m->source();return new vr::Batch(m,capacities,count);}
    catch(const std::exception& e){error_text=e.what();return nullptr;}
}
extern "C" void venturi_qwen_batch_destroy(void* h){delete static_cast<vr::Batch*>(h);}
extern "C" int venturi_qwen_batch_forward(void* h,const int* slots,int count,const std::int64_t* tokens,const int* offsets,float* logits,std::uint64_t elements) {
    try {vr::require(h && logits,"null batch forward");auto& b=*static_cast<vr::Batch*>(h);std::lock_guard<std::mutex> a(b.mutex),lock(b.model->mutex);vr::Device device(b.model->c.device);auto ids=b.slots(slots,count);vr::require(elements==vr::mul(count,b.model->c.vocab),"batch logit shape mismatch");b.input(ids,tokens,offsets);
        try {b.prefill(ids,tokens,offsets);b.head(ids,nullptr,logits);}catch(...){b.poison(ids);throw;}return 0;}
    catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" int venturi_qwen_batch_generate(void* h,const int* slots,int count,const std::int64_t* tokens,const int* offsets,const int* maximum,const std::int64_t* eos,const int* eos_offsets,std::int64_t* out,int stride,int* counts) {
    try {vr::require(h,"null batch generation");auto& b=*static_cast<vr::Batch*>(h);std::lock_guard<std::mutex> a(b.mutex),lock(b.model->mutex);vr::Device device(b.model->c.device);b.generate(b.slots(slots,count),tokens,offsets,maximum,eos,eos_offsets,out,stride,counts);return 0;}
    catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" int venturi_qwen_batch_reset(void* h,const int* slots,int count) {
    try {vr::require(h,"null batch reset");auto& b=*static_cast<vr::Batch*>(h);std::lock_guard<std::mutex> a(b.mutex),lock(b.model->mutex);vr::Device device(b.model->c.device);auto ids=b.slots(slots,count);b.model->source();for(auto i:ids){b.requests[i]->reset();b.requests[i]->first_token_seconds=b.requests[i]->generation_seconds=0;}return 0;}
    catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" int venturi_qwen_batch_info(void* h,std::uint64_t* shared,std::uint64_t* rows,double* timings) {
    try {vr::require(h && shared && rows && timings,"null batch info");auto& b=*static_cast<vr::Batch*>(h);std::lock_guard<std::mutex> a(b.mutex),lock(b.model->mutex);auto& m=*b.model;std::uint64_t total=0;
        for(std::size_t i=0;i<b.requests.size();++i){auto& r=*b.requests[i];rows[4*i]=r.position;rows[4*i+1]=r.failed;rows[4*i+2]=r.capacity;rows[4*i+3]=r.device_bytes;timings[2*i]=r.first_token_seconds;timings[2*i+1]=r.generation_seconds;total=vr::add(total,r.device_bytes);}
        const std::uint64_t values[]={b.requests.size(),b.work->device_bytes+b.extra_bytes,total,m.device_bytes,m.second_device_bytes,m.stage_bytes+m.host_input.bytes+m.host_output.bytes,b.group_calls,b.input_rows,b.decode_calls,b.prefill_calls,b.head_calls,b.head_rows,b.max_rows,m.peer_input_bytes,m.peer_output_bytes};std::copy(values,values+15,shared);return 0;}
    catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" int venturi_qwen_batch_snapshot(void* h,int slot,int layer,int kind,void* target,std::uint64_t bytes) {
    try {vr::require(h,"null batch snapshot");auto& b=*static_cast<vr::Batch*>(h);std::lock_guard<std::mutex> a(b.mutex),lock(b.model->mutex);vr::Device device(b.model->c.device);vr::require(slot>=0 && slot<static_cast<int>(b.requests.size()) && layer>=0 && layer<b.model->c.layers && kind>=0 && kind<=3,"invalid batch snapshot slot/layer/kind");
        auto& s=*b.requests[slot];const auto* source=kind==0?s.history[layer].get():kind==1?s.state[layer].get():kind==2?s.keys[layer].get():s.values[layer].get();vr::require(source,"batch snapshot state kind absent");
        const auto expected=kind<2?source->bytes:vr::mul(vr::mul(s.position,b.model->c.hkv*b.model->c.dim),2);vr::require(bytes==expected && (target || !bytes),"batch snapshot byte shape mismatch");
        if(bytes)check(cudaMemcpyAsync(target,source->ptr,bytes,cudaMemcpyDeviceToHost,b.model->stream.ptr));b.model->sync();return 0;}
    catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" int venturi_qwen_batch_debug_failure(void* h,int slot,int layer,int position) {
    try {vr::require(h,"null batch failure hook");auto& b=*static_cast<vr::Batch*>(h);std::lock_guard<std::mutex> a(b.mutex),lock(b.model->mutex);vr::require(slot>=0 && slot<static_cast<int>(b.requests.size()) && layer>=0 && layer<b.model->c.layers && position>=0,"invalid batch failure hook");b.requests[slot]->fault_layer=layer;b.requests[slot]->fault_position=position;return 0;}
    catch(const std::exception& e){error_text=e.what();return -1;}
}
