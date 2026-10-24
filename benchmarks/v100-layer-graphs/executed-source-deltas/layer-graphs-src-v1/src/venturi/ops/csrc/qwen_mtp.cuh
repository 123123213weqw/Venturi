// Native greedy MTP with shared target vocabulary and transactional GDN.
#pragma once
#include <chrono>

namespace vr {
inline double mtp_clock() {return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct MTPModel {
    std::shared_ptr<Model> target,first,last,base;
    std::shared_ptr<Pipeline> pipeline;
    VRMatrix fc;
    const float *enorm=nullptr,*hnorm=nullptr;
    MTPModel(void* owner,bool pp,const VRConfig& cfg,const VRSpan* spans,int count,const VRLayer& layer,
             const VRMatrix& projection,const void* norm,const void* embnorm,const void* hiddennorm,const float* inv):fc(projection) {
        require(owner,"null MTP target");
        if (pp) {pipeline=*static_cast<PipelineHandle*>(owner);first=pipeline->first;last=pipeline->second;}
        else {target=*static_cast<Handle*>(owner);first=last=target;require(!target->stage,"MTP requires complete target");}
        const auto& c=first->c;
        require(c.resident==1 && c.max_tokens>=2 && cfg.device==last->c.device && cfg.second_device==-1 && cfg.resident==1 && cfg.layers==1 &&
            cfg.dtype==c.dtype && cfg.hidden==c.hidden && cfg.vocab==c.vocab && cfg.intermediate==c.intermediate && cfg.hq==c.hq && cfg.hkv==c.hkv && cfg.dim==c.dim && cfg.rotary==c.rotary &&
            cfg.epsilon==c.epsilon && cfg.max_tokens==c.max_tokens && cfg.max_position==c.max_position && layer.kind==1,"MTP/target geometry or residency mismatch");
        Device d(cfg.device);VRMatrix unused{0,0,c.vocab,c.hidden};
        base=std::make_shared<Model>(cfg,spans,count,&layer,unused,unused,norm,inv,0,1,false,&fc);
        base->guard(mul(c.hidden,8));
        enorm=static_cast<const float*>(base->copy_constant(embnorm,mul(c.hidden,4)));
        hnorm=static_cast<const float*>(base->copy_constant(hiddennorm,mul(c.hidden,4)));base->sync();
    }
    void source() {first->source();if (last!=first)last->source();base->source();}
};
using MTPHandle=std::shared_ptr<MTPModel>;

struct MTPCheckpoint {
    Session& session;
    struct Slot {Buffer* source;std::unique_ptr<Buffer> saved;};
    std::vector<Slot> slots;
    int position=0;std::uint64_t bytes=0;
    explicit MTPCheckpoint(Session& s):session(s) {
        auto& m=*s.model;std::lock_guard<std::mutex> lock(m.mutex);Device d(m.c.device);
        for (int i=m.begin;i<m.end;++i) for (auto* p:{s.history[i].get(),s.state[i].get()}) if (p) bytes=add(bytes,p->bytes);
        m.guard(bytes);
        for (int i=m.begin;i<m.end;++i) for (auto* p:{s.history[i].get(),s.state[i].get()}) if (p) {
            auto saved=std::make_unique<Buffer>();saved->allocate(p->bytes);slots.push_back({p,std::move(saved)});
        }
    }
    void copy(bool save) {
        auto& m=*session.model;std::lock_guard<std::mutex> lock(m.mutex);Device d(m.c.device);
        if (save) position=session.position;
        for (const auto& slot:slots) if (slot.source->bytes)
            check(cudaMemcpyAsync(save?slot.saved->ptr:slot.source->ptr,save?slot.source->ptr:slot.saved->ptr,slot.source->bytes,cudaMemcpyDeviceToDevice,m.stream.ptr));
        m.sync();if (!save) session.position=position;
    }
};

// Prefix one reuses the existing backup allocation. Only prefix two needs
// additional storage; attention tails remain protected by the committed cursor.
struct MTPPrefixes {
    MTPCheckpoint& checkpoint;
    PrefixTrace trace;
    std::vector<std::unique_ptr<Buffer>> extra;
    std::uint64_t bytes=0;
    explicit MTPPrefixes(MTPCheckpoint& p):checkpoint(p) {
        auto& session=p.session;auto& m=*session.model;std::lock_guard<std::mutex> lock(m.mutex);Device d(m.c.device);m.guard(p.bytes);
        for(int j=0;j<2;++j){trace.history[j].resize(m.c.layers);trace.state[j].resize(m.c.layers);}
        std::size_t slot=0;
        for(int i=m.begin;i<m.end;++i)for(auto* source:{session.history[i].get(),session.state[i].get()})if(source) {
            require(p.slots[slot].source==source,"prefix backup mapping mismatch");
            auto b=std::make_unique<Buffer>();b->allocate(source->bytes);bytes=add(bytes,b->bytes);
            auto* pointers=source==session.history[i].get()?trace.history:trace.state;
            pointers[0][i]=p.slots[slot].saved->ptr;pointers[1][i]=b->ptr;extra.push_back(std::move(b));++slot;
        }
        require(bytes==p.bytes,"prefix memory estimate mismatch");
    }
    void restore(int prefix,int position) {
        require(prefix==1 || prefix==2,"invalid committed prefix");auto& session=checkpoint.session;auto& m=*session.model;
        std::lock_guard<std::mutex> lock(m.mutex);Device d(m.c.device);
        for(std::size_t i=0;i<checkpoint.slots.size();++i) {
            auto& slot=checkpoint.slots[i];if(slot.source->bytes)check(cudaMemcpyAsync(slot.source->ptr,prefix==1?slot.saved->ptr:extra[i]->ptr,slot.source->bytes,cudaMemcpyDeviceToDevice,m.stream.ptr));
        }
        m.sync();session.position=position+prefix;
    }
};

template<class T> __global__ void mtp_concat(const T* e,const T* h,T* out,int rows,int width) {
    const auto i=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i<static_cast<std::int64_t>(rows)*width) {const auto r=i/width,c=i%width;out[r*width*2+c]=e[i];out[r*width*2+width+c]=h[i];}
}
struct MTPSession {
    std::shared_ptr<MTPModel> model;
    std::unique_ptr<Session> target,draft;
    Session* borrowed_target=nullptr;
    std::unique_ptr<PipelineSession> pipeline;
    std::unique_ptr<MTPCheckpoint> checkpoint_first,checkpoint_second;
    std::unique_ptr<MTPPrefixes> prefix_first,prefix_second;
    bool last_prefix_recovery=false;
    Buffer hidden,last_hidden,shifted,embedding_rows,remote_embedding,remote_ids,concat,selection,scratch;
    Pinned bridge;
    std::mutex mutex;
    std::vector<std::int64_t> debug_proposals;
    std::uint64_t proposed=0,accepted=0,verify_calls=0,replayed=0,rollback_calls=0,target_calls=0,target_inputs=0,draft_calls=0,draft_inputs=0,emitted=0,forced=0;
    int last_depth=1;
    std::uint64_t depth_proposed[2]{},depth_verified[2]{},depth_accepted[2]{},depth_forced[2]{},draft_rewinds=0,verified_catchup=0,verify_width2=0,verify_width3=0;
    double draft_seconds=0,verify_seconds=0,rollback_seconds=0,prefill_seconds=0,checkpoint_seconds=0,first_token_seconds=0,generation_seconds=0;
    explicit MTPSession(std::shared_ptr<MTPModel> p,int capacity,Session* borrowed=nullptr):model(std::move(p)),borrowed_target(borrowed) {
        if(borrowed)require(!model->pipeline && borrowed->model==model->target && borrowed->capacity==capacity,"invalid borrowed MTP target");
        else if (model->pipeline) pipeline=std::make_unique<PipelineSession>(model->pipeline,capacity);
        else {std::lock_guard<std::mutex> lock(model->target->mutex);Device d(model->target->c.device);target=std::make_unique<Session>(model->target,capacity);}
        checkpoint_first=std::make_unique<MTPCheckpoint>(first());
        if (pipeline)checkpoint_second=std::make_unique<MTPCheckpoint>(last());
        auto& m=*model->base;std::lock_guard<std::mutex> lock(m.mutex);Device d(m.c.device);
        const int rows=m.c.max_tokens;const auto bytes=mul(mul(rows,m.c.hidden),2);
        m.guard(add(Session::estimate(m,capacity),add(mul(bytes,5),add(mul(m.c.hidden,2),mul(rows,16+Session::min_tiles(m.c.vocab)*16)))));
        draft=std::make_unique<Session>(model->base,capacity);
        hidden.allocate(bytes);last_hidden.allocate(mul(m.c.hidden,2));shifted.allocate(bytes);embedding_rows.allocate(bytes);concat.allocate(mul(bytes,2));
        selection.allocate(mul(rows,16));scratch.allocate(mul(rows,Session::min_tiles(m.c.vocab)*16));
        if (model->first->c.device!=m.c.device) {
            {Device e(model->first->c.device);model->first->guard(add(bytes,mul(rows,8)));remote_embedding.allocate(bytes);remote_ids.allocate(mul(rows,8));}
            bridge.allocate(bytes);
        }
    }
    ~MTPSession() {
        if(target && target->layer_graph) {auto& m=*target->model;std::lock_guard<std::mutex> lock(m.mutex);Device d(m.c.device);try{m.sync();}catch(...){}target->device_bytes-=4;target->layer_graph.reset();}
    }
    Session& first() {return pipeline?*pipeline->first:borrowed_target?*borrowed_target:*target;}
    Session& last() {return pipeline?*pipeline->second:borrowed_target?*borrowed_target:*target;}
    int position() {return first().position;}
    bool failed() {return first().failed || last().failed || draft->failed;}
    void poison() {first().failed=last().failed=draft->failed=true;}
    void settle_failure() {
        poison();for(auto m:{model->first,model->last,model->base})try {std::lock_guard<std::mutex> lock(m->mutex);Device d(m->c.device);m->sync();}catch(...){}
    }
    std::uint64_t request_bytes() {return draft->device_bytes+hidden.bytes+last_hidden.bytes+shifted.bytes+embedding_rows.bytes+concat.bytes+selection.bytes+scratch.bytes;}
    void select(Session& s,int rows,std::int64_t* out) {
        auto& m=*s.model;
        op(venturi_greedy_run(s.p("logits"),static_cast<std::int64_t*>(selection.ptr),scratch.ptr,scratch.bytes,rows,m.c.vocab,m.c.dtype,m.stream.ptr));
        check(cudaMemcpyAsync(out,selection.ptr,mul(rows,16),cudaMemcpyDeviceToHost,m.stream.ptr));m.sync();
    }
    std::int64_t token(const std::vector<std::int64_t>& values,int row) {
        const auto i=values[row*2];require(!values[row*2+1] && i>=0 && i<model->base->c.vocab,"MTP greedy rejected nonfinite/invalid logits");return i;
    }
    void target_forward(const std::int64_t* ids,int n,bool all,std::vector<std::int64_t>& out) {
        require(n<=model->base->c.max_tokens,"MTP target capture chunk exceeds workspace");out.resize(mul(all?n:1,2));
        if (pipeline) pipeline->forward(ids,n,nullptr,all,nullptr,out.data(),hidden.ptr,selection.ptr,scratch.ptr,scratch.bytes);
        else {auto& r=first();auto& m=*r.model;std::lock_guard<std::mutex> lock(m.mutex);Device d(m.c.device);
            r.forward(ids,n,nullptr,all,nullptr,hidden.ptr);select(r,all?n:1,out.data());}
        ++target_calls;target_inputs=add(target_inputs,n);
    }
    void draft_step(const std::int64_t* ids,int n,const void* h,bool logits,std::vector<std::int64_t>* selected=nullptr,float* diagnostic_logits=nullptr,void* diagnostic_hidden=nullptr) {
        auto& md=*model;auto& m=*md.base;const auto& c=m.c;
        require(!failed() && n>0 && n<=c.max_tokens && n<=draft->capacity-draft->position,"MTP draft chunk exceeds capacity or failed session");
        for(int i=0;i<n;++i)require(ids[i]>=0 && ids[i]<c.vocab,"MTP draft token out of range");
        const auto start=mtp_clock();std::lock_guard<std::mutex> lock(m.mutex);Device d(c.device);md.source();
        const auto bytes=mul(mul(n,c.hidden),2);
        check(cudaMemcpyAsync(draft->p("ids"),ids,mul(n,8),cudaMemcpyHostToDevice,m.stream.ptr));m.sync();
        {
            auto& v=*md.first;std::lock_guard<std::mutex> vocab(v.mutex);Device e(v.c.device);
            auto w=v.weight(v.embedding);
            if (v.c.device==c.device) {
                op(venturi_q8_embedding_run(w.first,w.second,static_cast<const std::int64_t*>(draft->p("ids")),embedding_rows.ptr,n,c.vocab,c.hidden,padded(v.embedding),c.dtype,v.stream.ptr));v.sync();
            } else {
                // IDs on the first stage use a small request-owned temporary allocation.
                check(cudaMemcpyAsync(remote_ids.ptr,ids,mul(n,8),cudaMemcpyHostToDevice,v.stream.ptr));
                op(venturi_q8_embedding_run(w.first,w.second,static_cast<const std::int64_t*>(remote_ids.ptr),remote_embedding.ptr,n,c.vocab,c.hidden,padded(v.embedding),c.dtype,v.stream.ptr));
                check(cudaMemcpyAsync(bridge.ptr,remote_embedding.ptr,bytes,cudaMemcpyDeviceToHost,v.stream.ptr));v.sync();
            }
        }
        if (bridge.ptr)check(cudaMemcpyAsync(embedding_rows.ptr,bridge.ptr,bytes,cudaMemcpyHostToDevice,m.stream.ptr));
        op(venturi_rms_run(embedding_rows.ptr,md.enorm,embedding_rows.ptr,n,c.hidden,c.dtype,c.epsilon,m.stream.ptr));
        op(venturi_rms_run(h,md.hnorm,shifted.ptr,n,c.hidden,c.dtype,c.epsilon,m.stream.ptr));
        const int grid=static_cast<int>((mul(n,c.hidden)+255)/256);
        if (c.dtype==0)mtp_concat<<<grid,256,0,m.stream.ptr>>>(static_cast<const __half*>(embedding_rows.ptr),static_cast<const __half*>(shifted.ptr),static_cast<__half*>(concat.ptr),n,c.hidden);
        else mtp_concat<<<grid,256,0,m.stream.ptr>>>(static_cast<const __nv_bfloat16*>(embedding_rows.ptr),static_cast<const __nv_bfloat16*>(shifted.ptr),static_cast<__nv_bfloat16*>(concat.ptr),n,c.hidden);
        check(cudaGetLastError());m.linear(md.fc,concat.ptr,draft->p("x"),n);
        auto* x=draft->run_layers(draft->position,n,draft->p("x"),nullptr,n,0);
        if (logits || diagnostic_hidden) {
            op(venturi_rms_run(x,m.norm,draft->p("normalized"),n,c.hidden,c.dtype,c.epsilon,m.stream.ptr));
            if (diagnostic_hidden)check(cudaMemcpyAsync(diagnostic_hidden,draft->p("normalized"),bytes,cudaMemcpyDeviceToHost,m.stream.ptr));
        }
        if (logits) {
            m.sync();auto& v=*md.last;std::lock_guard<std::mutex> vocab(v.mutex);Device e(v.c.device);
            v.linear(v.head,static_cast<char*>(draft->p("normalized"))+mul(mul(n-1,c.hidden),2),draft->p("logits"),1);v.sync();
            if(selected) {selected->resize(2);op(venturi_greedy_run(draft->p("logits"),static_cast<std::int64_t*>(selection.ptr),scratch.ptr,scratch.bytes,1,c.vocab,c.dtype,v.stream.ptr));check(cudaMemcpyAsync(selected->data(),selection.ptr,16,cudaMemcpyDeviceToHost,v.stream.ptr));v.sync();}
            if(diagnostic_logits) {
                check(cudaMemcpyAsync(draft->host_logits.data(),draft->p("logits"),mul(c.vocab,2),cudaMemcpyDeviceToHost,v.stream.ptr));v.sync();
                for(int i=0;i<c.vocab;++i)diagnostic_logits[i]=c.dtype==0?__half2float(reinterpret_cast<const __half*>(draft->host_logits.data())[i]):__bfloat162float(reinterpret_cast<const __nv_bfloat16*>(draft->host_logits.data())[i]);
            }
        }
        m.sync();md.source();draft->position+=n;++draft_calls;draft_inputs=add(draft_inputs,n);draft_seconds+=mtp_clock()-start;
    }
    void remember(int row) {
        auto& m=*model->base;std::lock_guard<std::mutex> lock(m.mutex);Device d(m.c.device);
        check(cudaMemcpyAsync(last_hidden.ptr,hidden.at(mul(mul(row,m.c.hidden),2)),last_hidden.bytes,cudaMemcpyDeviceToDevice,m.stream.ptr));m.sync();
    }
    void warm(const std::int64_t* ids,int n) {
        auto& m=*model->base;Device d(m.c.device);
        // Shift target states right: token 0 pairs with an explicit zero state.
        check(cudaMemcpyAsync(embedding_rows.ptr,last_hidden.ptr,last_hidden.bytes,cudaMemcpyDeviceToDevice,m.stream.ptr));
        if(n>1)check(cudaMemcpyAsync(embedding_rows.at(last_hidden.bytes),hidden.ptr,mul(n-1,last_hidden.bytes),cudaMemcpyDeviceToDevice,m.stream.ptr));m.sync();
        // embedding_rows is also draft scratch; stage the input in concat first.
        check(cudaMemcpyAsync(concat.ptr,embedding_rows.ptr,mul(n,last_hidden.bytes),cudaMemcpyDeviceToDevice,m.stream.ptr));m.sync();
        draft_step(ids,n,concat.ptr,false);remember(n-1);
    }
    void checkpoint(bool save) {checkpoint_first->copy(save);if(checkpoint_second)checkpoint_second->copy(save);}
    void reset() {
        if(pipeline)pipeline->reset();else {auto& r=first();auto& m=*r.model;std::lock_guard<std::mutex> lock(m.mutex);Device d(m.c.device);r.reset();}
        {auto& m=*model->base;std::lock_guard<std::mutex> lock(m.mutex);Device d(m.c.device);draft->reset();}
        proposed=accepted=verify_calls=replayed=rollback_calls=target_calls=target_inputs=draft_calls=draft_inputs=emitted=forced=0;
        draft_seconds=verify_seconds=rollback_seconds=prefill_seconds=checkpoint_seconds=first_token_seconds=generation_seconds=0;debug_proposals.clear();
        last_prefix_recovery=false;last_depth=1;for(int i=0;i<2;++i)depth_proposed[i]=depth_verified[i]=depth_accepted[i]=depth_forced[i]=0;draft_rewinds=verified_catchup=verify_width2=verify_width3=0;
    }
    void generate(const std::int64_t* prompt,int n,int maximum,const std::int64_t* eos,int eos_count,std::int64_t* out,int* count) {
        const double begun=mtp_clock();first_token_seconds=generation_seconds=0;
        const auto& c=model->base->c;require(!failed() && position()==0 && draft->position==0,"MTP generation requires fresh/reset request state");
        first().input(prompt,n);require(maximum>0 && out && count && eos_count>=0 && (eos || !eos_count) && add(n,maximum)<=static_cast<std::uint64_t>(first().capacity),"Invalid MTP generation capacity/arguments");
        for(int i=0;i<eos_count;++i)require(eos[i]>=0 && eos[i]<c.vocab,"MTP EOS out of range");*count=0;
        auto stopped=[&](std::int64_t t){return eos_count && std::find(eos,eos+eos_count,t)!=eos+eos_count;};
        auto emit=[&](std::int64_t t){out[*count]=t;++*count;++emitted;};
        std::vector<std::int64_t> selected,proposal;
        try {
            {Device d(c.device);check(cudaMemsetAsync(last_hidden.ptr,0,last_hidden.bytes,model->base->stream.ptr));model->base->sync();}
            auto start=mtp_clock();
            for(int i=0;i<n;i+=c.max_tokens) {int t=std::min(c.max_tokens,n-i);target_forward(prompt+i,t,false,selected);warm(prompt+i,t);}
            prefill_seconds+=mtp_clock()-start;emit(token(selected,0));first_token_seconds=mtp_clock()-begun;
            while(*count<maximum && !stopped(out[*count-1])) {
                const std::int64_t pending=out[*count-1];
                if(maximum-*count==1) {
                    target_forward(&pending,1,false,selected);draft_step(&pending,1,last_hidden.ptr,false);remember(0);emit(token(selected,0));continue;
                }
                draft_step(&pending,1,last_hidden.ptr,true,&proposal);auto predicted=token(proposal,0);
                if(static_cast<std::size_t>(*count)<debug_proposals.size()) {predicted=debug_proposals[*count];++forced;}
                ++proposed;start=mtp_clock();checkpoint(true);checkpoint_seconds+=mtp_clock()-start;std::int64_t inputs[]={pending,predicted};start=mtp_clock();
                target_forward(inputs,2,true,selected);verify_seconds+=mtp_clock()-start;++verify_calls;
                auto correct=token(selected,0);const bool match=predicted==correct;
                if(match)++accepted;
                if(!match || stopped(correct)) {
                    start=mtp_clock();checkpoint(false);++rollback_calls;std::vector<std::int64_t> replay;
                    target_forward(&pending,1,false,replay);++replayed;require(token(replay,0)==correct,"MTP batch/replay target greedy mismatch");
                    rollback_seconds+=mtp_clock()-start;remember(0);emit(correct);
                } else {
                    auto bonus=token(selected,1);draft_step(&predicted,1,hidden.ptr,false);remember(1);emit(correct);emit(bonus);
                }
                require(position()==draft->position,"MTP committed cursor mismatch");
            }
            model->source();generation_seconds=mtp_clock()-begun;
        } catch (...) {settle_failure();throw;}
    }
    void prepare_prefix() {
        require(!failed() && position()==0 && draft->position==0,"prefix preparation requires fresh/reset state");
        require(model->base->c.max_tokens>=3,"MTP2 requires at least three projection rows");
        if(!prefix_first) {
            auto a=std::make_unique<MTPPrefixes>(*checkpoint_first);std::unique_ptr<MTPPrefixes> b;
            if(checkpoint_second)b=std::make_unique<MTPPrefixes>(*checkpoint_second);
            prefix_first=std::move(a);prefix_second=std::move(b);
        }
    }
    void verify_prefix(const std::int64_t* inputs,int n,std::vector<std::int64_t>& selected) {
        require(prefix_first && !first().verification_trace && !last().verification_trace,"invalid prefix trace ownership");
        prefix_first->trace.prefixes=n-1;first().verification_trace=&prefix_first->trace;
        if(prefix_second){prefix_second->trace.prefixes=n-1;last().verification_trace=&prefix_second->trace;}
        try {target_forward(inputs,n,true,selected);}
        catch(...){first().verification_trace=nullptr;last().verification_trace=nullptr;throw;}
        first().verification_trace=nullptr;last().verification_trace=nullptr;
    }
    void generate2(const std::int64_t* prompt,int n,int maximum,const std::int64_t* eos,int eos_count,std::int64_t* out,int* count,bool prefixes=false) {
        const double begun=mtp_clock();
        const auto& c=model->base->c;
        require(c.max_tokens>=3,"MTP2 requires at least three projection rows");
        require(!failed() && position()==0 && draft->position==0,"MTP generation requires fresh/reset request state");
        first().input(prompt,n);
        require(maximum>0 && out && count && eos_count>=0 && (eos || !eos_count) && add(n,maximum)<=static_cast<std::uint64_t>(first().capacity),"Invalid MTP generation capacity/arguments");
        for(int i=0;i<eos_count;++i)require(eos[i]>=0 && eos[i]<c.vocab,"MTP EOS out of range");
        if(prefixes)prepare_prefix();last_prefix_recovery=prefixes;
        last_depth=2;first_token_seconds=generation_seconds=0;*count=0;
        auto stopped=[&](std::int64_t t){return eos_count && std::find(eos,eos+eos_count,t)!=eos+eos_count;};
        auto emit=[&](std::int64_t t){out[(*count)++]=t;++emitted;};
        std::vector<std::int64_t> selected,proposal;
        try {
            {Device d(c.device);check(cudaMemsetAsync(last_hidden.ptr,0,last_hidden.bytes,model->base->stream.ptr));model->base->sync();}
            auto start=mtp_clock();
            for(int i=0;i<n;i+=c.max_tokens) {int t=std::min(c.max_tokens,n-i);target_forward(prompt+i,t,false,selected);warm(prompt+i,t);}
            prefill_seconds+=mtp_clock()-start;emit(token(selected,0));first_token_seconds=mtp_clock()-begun;
            while(*count<maximum && !stopped(out[*count-1])) {
                const std::int64_t pending=out[*count-1];
                if(maximum-*count==1) {
                    target_forward(&pending,1,false,selected);draft_step(&pending,1,last_hidden.ptr,false);remember(0);emit(token(selected,0));continue;
                }
                const int depth=std::min(2,maximum-*count-1),base=draft->position;
                std::int64_t inputs[3]{pending,0,0};
                for(int i=0;i<depth;++i) {
                    // Only the first step has a verified target hidden state.
                    // Later speculation recurs on the normalized draft output.
                    const void* h=i?draft->p("normalized"):last_hidden.ptr;
                    draft_step(inputs+i,1,h,true,&proposal);inputs[i+1]=token(proposal,0);
                    if(static_cast<std::size_t>(*count+i)<debug_proposals.size()) {inputs[i+1]=debug_proposals[*count+i];++forced;++depth_forced[i];}
                    ++proposed;++depth_proposed[i];
                }
                if(!prefixes){start=mtp_clock();checkpoint(true);checkpoint_seconds+=mtp_clock()-start;}
                start=mtp_clock();if(prefixes)verify_prefix(inputs,depth+1,selected);else target_forward(inputs,depth+1,true,selected);verify_seconds+=mtp_clock()-start;++verify_calls;
                if(depth==2)++verify_width3;else ++verify_width2;
                int committed=0;
                for(int i=0;i<depth;++i) {
                    const auto correct=token(selected,i);const bool match=inputs[i+1]==correct;
                    ++depth_verified[i];if(match){++accepted;++depth_accepted[i];}
                    emit(correct);
                    if(!match || stopped(correct)) {committed=i+1;break;}
                }
                if(!committed) {committed=depth+1;emit(token(selected,depth));}
                if(committed!=depth+1) {
                    // Attention tails are hidden by the cursor; GDN/history
                    // require a real checkpoint restore and accepted-prefix replay.
                    start=mtp_clock();++rollback_calls;
                    if(prefixes) {prefix_first->restore(committed,base);if(prefix_second)prefix_second->restore(committed,base);}
                    else {
                        checkpoint(false);std::vector<std::int64_t> replay;target_forward(inputs,committed,false,replay);replayed=add(replayed,committed);
                        require(token(replay,0)==token(selected,committed-1),"MTP2 replay target greedy mismatch");
                    }
                    rollback_seconds+=mtp_clock()-start;
                }
                {
                    auto& m=*model->base;std::lock_guard<std::mutex> lock(m.mutex);Device d(c.device);m.sync();
                    require(draft->position==base+depth,"MTP2 speculative draft cursor mismatch");
                    if(draft->position>base+1)++draft_rewinds;
                    // Draft has attention only. Keep its correct pending-input
                    // prefix; speculative later KV entries must be overwritten.
                    draft->position=base+1;
                }
                for(int i=1;i<committed;++i) {
                    draft_step(inputs+i,1,hidden.at(mul(i-1,last_hidden.bytes)),false);++verified_catchup;
                }
                remember(committed-1);
                require(position()==base+committed && position()==draft->position,"MTP2 committed cursor mismatch");
            }
            model->source();generation_seconds=mtp_clock()-begun;
        }catch(...){settle_failure();throw;}
    }

};

// Fixed-cohort MTP1: serial native shifted draft warmup/proposals, grouped target
// verification, exact request-local prefix recovery. No target replay or extra
// recurrent snapshot beyond the existing per-request MTP backup.
struct MTPBatch {
    std::shared_ptr<MTPModel> model;
    std::unique_ptr<Batch> batch;
    std::vector<std::unique_ptr<MTPSession>> requests;
    std::vector<PrefixTrace> traces;
    Buffer selection,scratch;
    std::mutex mutex;
    double verification_seconds=0,prefill_seconds=0,generation_seconds=0;
    explicit MTPBatch(std::shared_ptr<MTPModel> m,const int* capacities,int count):model(std::move(m)) {
        require(!model->pipeline && model->target && model->target->c.resident && model->target->c.dtype==0,"MTP batch requires complete resident FP16 single/TP target, not PP");
        auto& target=*model->target;const auto& c=target.c;
        require(capacities && count>0 && count<=16 && count<=c.max_tokens/2,"MTP batch requires 1..min(16,max_tokens/2) slots");
        {std::lock_guard<std::mutex> lock(target.mutex);Device d(c.device);target.source();batch=std::make_unique<Batch>(model->target,capacities,count);}
        traces.resize(count);
        for(int slot=0;slot<count;++slot) {
            requests.push_back(std::make_unique<MTPSession>(model,capacities[slot],batch->requests[slot].get()));
            auto& s=*requests.back();auto& trace=traces[slot];trace.prefixes=1;
            for(int p=0;p<2;++p){trace.history[p].resize(c.layers);trace.state[p].resize(c.layers);}
            std::size_t j=0;
            for(int i=target.begin;i<target.end;++i)for(auto* src:{s.first().history[i].get(),s.first().state[i].get()})if(src) {
                auto& saved=s.checkpoint_first->slots[j++];require(saved.source==src,"MTP batch checkpoint mapping mismatch");
                (src==s.first().history[i].get()?trace.history[0]:trace.state[0])[i]=saved.saved->ptr;
            }
        }
        {std::lock_guard<std::mutex> lock(target.mutex);Device d(c.device);target.guard(mul(c.max_tokens,16+Session::min_tiles(c.vocab)*16));selection.allocate(mul(c.max_tokens,16));scratch.allocate(mul(c.max_tokens,Session::min_tiles(c.vocab)*16));}
    }
    MTPSession& at(int slot){require(slot>=0 && slot<static_cast<int>(requests.size()),"invalid MTP batch slot");return *requests[slot];}
    void clear_traces() noexcept {for(auto& r:requests)r->first().verification_trace=nullptr;}
    void poison(const std::vector<int>& ids) noexcept {
        clear_traces();for(int i:ids)at(i).poison();
        for(auto m:{model->target,model->base})try{std::lock_guard<std::mutex> lock(m->mutex);Device d(m->c.device);m->sync();}catch(...){}
    }
    struct Round {int output,slot,row,count,before;std::int64_t predicted;};
    void verify(const std::vector<Round>& rounds,const std::vector<BatchSegment>& segments,const std::vector<std::int64_t>& inputs,std::vector<std::int64_t>& chosen) {
        const double started=mtp_clock();auto& m=*model->target;const auto& c=m.c;
        try {
            std::lock_guard<std::mutex> lock(m.mutex);Device d(c.device);
            for(const auto& r:rounds)if(r.count==2) {
                auto& s=at(r.slot);require(!s.first().verification_trace,"MTP batch trace ownership violation");
                s.checkpoint_first->position=r.before+1;s.first().verification_trace=&traces[r.slot];
            }
            batch->group(segments,inputs,true);const int n=static_cast<int>(inputs.size());
            for(const auto& r:rounds)check(cudaMemcpyAsync(at(r.slot).hidden.ptr,batch->row("normalized",r.row,c.hidden),mul(mul(r.count,c.hidden),2),cudaMemcpyDeviceToDevice,m.stream.ptr));
            m.linear(m.head,batch->work->p("normalized"),batch->work->p("logits"),n);
            chosen.resize(2*n);op(venturi_greedy_run(batch->work->p("logits"),static_cast<std::int64_t*>(selection.ptr),scratch.ptr,scratch.bytes,n,c.vocab,c.dtype,m.stream.ptr));
            check(cudaMemcpyAsync(chosen.data(),selection.ptr,mul(n,16),cudaMemcpyDeviceToHost,m.stream.ptr));m.sync();m.source();
            for(int i=0;i<n;++i)require(chosen[2*i]>=0 && chosen[2*i]<c.vocab && !chosen[2*i+1],"MTP batch greedy rejected nonfinite/invalid logits");
            ++batch->head_calls;batch->head_rows=add(batch->head_rows,n);
        }catch(...){clear_traces();throw;}
        clear_traces();const auto elapsed=mtp_clock()-started;verification_seconds+=elapsed;
        for(const auto& r:rounds){auto& s=at(r.slot);++s.target_calls;s.target_inputs=add(s.target_inputs,r.count);if(r.count==2){++s.verify_calls;s.verify_seconds+=elapsed;}}
    }
    void generate(const std::vector<int>& ids,const std::int64_t* tokens,const int* offsets,const int* maximum,const std::int64_t* eos,const int* eo,std::int64_t* out,int stride,int* counts) {
        batch->input(ids,tokens,offsets);const auto& c=model->target->c;
        require(maximum && eo && eo[0]==0 && out && counts && stride>0,"invalid MTP batch generation arguments");
        for(std::size_t j=0;j<ids.size();++j) {
            auto& s=at(ids[j]);require(!s.failed() && s.position()==0 && s.draft->position==0,"MTP batch generation requires fresh/reset requests");
            require(maximum[j]>0 && maximum[j]<=stride && add(offsets[j+1]-offsets[j],maximum[j])<=static_cast<std::uint64_t>(s.first().capacity),"MTP batch generation exceeds slot capacity");
            require(eo[j]>=0 && eo[j+1]>=eo[j] && (eos || !eo[j+1]),"invalid MTP batch EOS offsets");
            for(int k=eo[j];k<eo[j+1];++k)require(eos[k]>=0 && eos[k]<c.vocab,"MTP batch EOS token out of range");counts[j]=0;
        }
        const double started=mtp_clock();batch->counters();verification_seconds=prefill_seconds=generation_seconds=0;
        auto stopped=[&](int j,std::int64_t token){for(int k=eo[j];k<eo[j+1];++k)if(eos[k]==token)return true;return false;};
        auto finished=[&](int j){return counts[j]>=maximum[j] || (counts[j] && stopped(j,out[mul(j,stride)+counts[j]-1]));};
        auto timing=[&](int j){auto& s=at(ids[j]);if(finished(j))s.generation_seconds=s.first().generation_seconds=mtp_clock()-started;};
        auto emit=[&](int j,std::int64_t token){out[mul(j,stride)+counts[j]++]=token;++at(ids[j]).emitted;timing(j);};
        try {
            // Keep the already qualified shifted warm protocol; steady-state
            // target projections below, not these per-slot prefills, are batched.
            for(std::size_t j=0;j<ids.size();++j) {
                auto& s=at(ids[j]);s.generate(tokens+offsets[j],offsets[j+1]-offsets[j],1,eos?eos+eo[j]:nullptr,eo[j+1]-eo[j],out+mul(j,stride),counts+j);
                s.first_token_seconds=s.first().first_token_seconds=mtp_clock()-started;s.generation_seconds=s.first().generation_seconds=0;s.last_prefix_recovery=true;timing(j);
            }
            prefill_seconds=mtp_clock()-started;
            for(;;) {
                std::vector<Round> rounds;std::vector<BatchSegment> segments;std::vector<std::int64_t> inputs,chosen;
                for(std::size_t j=0;j<ids.size();++j)if(!finished(j)) {
                    auto& s=at(ids[j]);const auto pending=out[mul(j,stride)+counts[j]-1];const int n=maximum[j]-counts[j]==1?1:2,before=s.position(),row=static_cast<int>(inputs.size());
                    std::vector<std::int64_t> proposal;s.draft_step(&pending,1,s.last_hidden.ptr,n==2,n==2?&proposal:nullptr);
                    auto predicted=n==2?s.token(proposal,0):0;
                    if(n==2){if(static_cast<std::size_t>(counts[j])<s.debug_proposals.size()){predicted=s.debug_proposals[counts[j]];++s.forced;}++s.proposed;}
                    rounds.push_back({static_cast<int>(j),ids[j],row,n,before,predicted});segments.push_back({ids[j],row,n});inputs.push_back(pending);if(n==2)inputs.push_back(predicted);
                }
                if(rounds.empty())break;verify(rounds,segments,inputs,chosen);
                for(const auto& r:rounds) {
                    auto& s=at(r.slot);const int j=r.output;const auto correct=chosen[2*r.row];
                    if(r.count==1){s.remember(0);emit(j,correct);}
                    else {
                        const bool match=r.predicted==correct;if(match)++s.accepted;
                        if(!match || stopped(j,correct)) {
                            const auto start=mtp_clock();s.checkpoint_first->copy(false);++s.rollback_calls;s.remember(0);s.rollback_seconds+=mtp_clock()-start;emit(j,correct);
                        } else {
                            s.draft_step(&r.predicted,1,s.hidden.ptr,false);s.remember(1);emit(j,correct);emit(j,chosen[2*(r.row+1)]);
                        }
                    }
                    require(s.position()==s.draft->position && s.position()==offsets[j+1]-offsets[j]+counts[j]-1,"MTP batch committed cursor mismatch");
                }
            }
            model->source();generation_seconds=mtp_clock()-started;
        }catch(...){poison(ids);throw;}
    }
};
}

extern "C" void* venturi_qwen_mtp_create(void* target,int pipeline,const VRConfig* c,const VRSpan* spans,int count,const VRLayer* layer,const VRMatrix* fc,const void* norm,const void* enorm,const void* hnorm,const float* inv) {
    try {vr::require(c && layer && fc && (pipeline==0 || pipeline==1),"null/invalid MTP descriptor");return new vr::MTPHandle(std::make_shared<vr::MTPModel>(target,pipeline,*c,spans,count,*layer,*fc,norm,enorm,hnorm,inv));}
    catch(const std::exception& e){error_text=e.what();return nullptr;}
}
extern "C" void venturi_qwen_mtp_destroy(void* h){delete static_cast<vr::MTPHandle*>(h);}
extern "C" void* venturi_qwen_mtp_session(void* h,int capacity){try{vr::require(h,"null MTP model");return new vr::MTPSession(*static_cast<vr::MTPHandle*>(h),capacity);}catch(const std::exception& e){error_text=e.what();return nullptr;}}
extern "C" void venturi_qwen_mtp_session_destroy(void* h){delete static_cast<vr::MTPSession*>(h);}
extern "C" int venturi_qwen_mtp_generate(void* h,const std::int64_t* ids,int n,int maximum,const std::int64_t* eos,int eos_count,std::int64_t* out,int* count){try{vr::require(h,"null MTP session");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);s.generate(ids,n,maximum,eos,eos_count,out,count);return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_generate2(void* h,const std::int64_t* ids,int n,int maximum,const std::int64_t* eos,int eos_count,std::int64_t* out,int* count){try{vr::require(h,"null MTP session");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);s.generate2(ids,n,maximum,eos,eos_count,out,count);return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_generate2_prefix(void* h,const std::int64_t* ids,int n,int maximum,const std::int64_t* eos,int eos_count,std::int64_t* out,int* count){try{vr::require(h,"null MTP session");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);s.generate2(ids,n,maximum,eos,eos_count,out,count,true);return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_prepare_prefix(void* h){try{vr::require(h,"null MTP session");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);s.prepare_prefix();return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_prefix_info(void* h,std::uint64_t* out){try{vr::require(h && out,"null prefix info");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);out[0]=s.prefix_first?s.prefix_first->bytes:0;out[1]=s.prefix_second?s.prefix_second->bytes:0;out[2]=s.last_prefix_recovery;return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_chain_stats(void* h,std::uint64_t* out){try{vr::require(h && out,"null MTP chain stats");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);
    const bool two=s.last_depth==2;
    const std::uint64_t values[]={static_cast<std::uint64_t>(s.last_depth),two?s.depth_proposed[0]:s.proposed,two?s.depth_proposed[1]:0,two?s.depth_verified[0]:s.proposed,two?s.depth_verified[1]:0,two?s.depth_accepted[0]:s.accepted,two?s.depth_accepted[1]:0,s.draft_rewinds,s.verified_catchup,two?s.verify_width2:s.verify_calls,s.verify_width3,two?s.depth_forced[0]:s.forced,two?s.depth_forced[1]:0};std::copy(values,values+13,out);return 0;
}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_reset(void* h){try{vr::require(h,"null MTP session");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);s.reset();return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_info(void* h,std::uint64_t* out){try{vr::require(h && out,"null MTP info");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);return s.pipeline?venturi_qwen_pipeline_info(s.pipeline.get(),out):venturi_qwen_info(&s.first(),out);}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_parallel_info(void* h,std::uint64_t* out){try{vr::require(h && out,"null MTP info");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);return s.pipeline?venturi_qwen_pipeline_parallel_info(s.pipeline.get(),out):venturi_qwen_parallel_info(&s.first(),out);}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_snapshot(void* h,int layer,int kind,void* out,std::uint64_t bytes){try{vr::require(h,"null MTP snapshot");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);return s.pipeline?venturi_qwen_pipeline_snapshot(s.pipeline.get(),layer,kind,out,bytes):venturi_qwen_snapshot(&s.first(),layer,kind,out,bytes);}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_debug_failure(void* h,int layer,int pos){try{vr::require(h,"null MTP fault");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);return s.pipeline?venturi_qwen_pipeline_debug_failure(s.pipeline.get(),layer,pos):venturi_qwen_debug_failure(&s.first(),layer,pos);}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_stats(void* h,std::uint64_t* out,double* seconds){try{vr::require(h && out && seconds,"null MTP stats");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);
    const std::uint64_t a[]={s.proposed,s.accepted,s.verify_calls,s.replayed,s.rollback_calls,s.target_calls,s.target_inputs,s.draft_calls,s.draft_inputs,s.emitted,s.forced,static_cast<std::uint64_t>(s.draft->position),s.model->base->device_bytes,s.request_bytes(),s.checkpoint_first->bytes,s.checkpoint_second?s.checkpoint_second->bytes:0,s.remote_embedding.bytes+s.remote_ids.bytes,s.bridge.bytes,static_cast<std::uint64_t>(s.model->base->c.device),s.pipeline?s.last().device_bytes:0};
    std::copy(a,a+20,out);const double b[]={s.draft_seconds,s.verify_seconds,s.rollback_seconds,s.prefill_seconds,s.checkpoint_seconds};std::copy(b,b+5,seconds);return 0;
}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_debug_proposals(void* h,const std::int64_t* ids,int n){try{vr::require(h && n>=0 && (ids || !n),"invalid MTP debug proposals");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);vr::require(s.position()==0,"debug proposals require fresh state");for(int i=0;i<n;++i)vr::require(ids[i]>=0 && ids[i]<s.model->base->c.vocab,"invalid forced proposal");s.debug_proposals.assign(ids,ids+n);return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_draft(void* h,const std::int64_t* ids,int n,const void* hidden,float* logits,void* output){try{vr::require(h && hidden && output,"null MTP diagnostic");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);auto& m=*s.model->base;vr::require(n>0 && n<=m.c.max_tokens,"MTP diagnostic chunk exceeds workspace");vr::Device d(m.c.device);check(cudaMemcpyAsync(s.hidden.ptr,hidden,vr::mul(vr::mul(n,m.c.hidden),2),cudaMemcpyHostToDevice,m.stream.ptr));m.sync();try{s.draft_step(ids,n,s.hidden.ptr,logits!=nullptr,nullptr,logits,output);}catch(...){s.settle_failure();throw;}return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_draft_snapshot(void* h,int kind,void* out,std::uint64_t bytes){try{vr::require(h && (kind==2 || kind==3),"invalid MTP KV snapshot");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);return venturi_qwen_snapshot(s.draft.get(),0,kind,out,bytes);}catch(const std::exception& e){error_text=e.what();return -1;}}

extern "C" int venturi_qwen_mtp_timings(void* h,double* out){try{vr::require(h && out,"null MTP timings");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);out[0]=s.first_token_seconds;out[1]=s.generation_seconds;return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}

extern "C" void* venturi_qwen_mtp_batch_create(void* h,const int* capacities,int count){try{vr::require(h,"null MTP batch model");return new vr::MTPBatch(*static_cast<vr::MTPHandle*>(h),capacities,count);}catch(const std::exception& e){error_text=e.what();return nullptr;}}
extern "C" void venturi_qwen_mtp_batch_destroy(void* h){delete static_cast<vr::MTPBatch*>(h);}
extern "C" int venturi_qwen_mtp_batch_generate(void* h,const int* slots,int count,const std::int64_t* ids,const int* offsets,const int* maximum,const std::int64_t* eos,const int* eo,std::int64_t* out,int stride,int* counts){try{vr::require(h,"null MTP batch");auto& b=*static_cast<vr::MTPBatch*>(h);std::lock_guard<std::mutex> lock(b.mutex);b.generate(b.batch->slots(slots,count),ids,offsets,maximum,eos,eo,out,stride,counts);return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_batch_info(void* h,std::uint64_t* shared,std::uint64_t* rows,double* times){try{vr::require(h,"null MTP batch info");auto& b=*static_cast<vr::MTPBatch*>(h);std::lock_guard<std::mutex> lock(b.mutex);return venturi_qwen_batch_info(b.batch.get(),shared,rows,times);}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_batch_reset(void* h,const int* slots,int count){try{vr::require(h,"null MTP batch reset");auto& b=*static_cast<vr::MTPBatch*>(h);std::lock_guard<std::mutex> lock(b.mutex);for(int i:b.batch->slots(slots,count)){auto& s=b.at(i);s.reset();s.first().first_token_seconds=s.first().generation_seconds=0;}return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_batch_snapshot(void* h,int slot,int layer,int kind,void* out,std::uint64_t bytes){try{vr::require(h,"null MTP batch snapshot");auto& b=*static_cast<vr::MTPBatch*>(h);std::lock_guard<std::mutex> lock(b.mutex);return venturi_qwen_batch_snapshot(b.batch.get(),slot,layer,kind,out,bytes);}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_batch_debug_failure(void* h,int slot,int layer,int position){try{vr::require(h,"null MTP batch failure");auto& b=*static_cast<vr::MTPBatch*>(h);std::lock_guard<std::mutex> lock(b.mutex);return venturi_qwen_batch_debug_failure(b.batch.get(),slot,layer,position);}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_batch_debug_proposals(void* h,int slot,const std::int64_t* ids,int count){try{vr::require(h,"null MTP batch proposals");auto& b=*static_cast<vr::MTPBatch*>(h);std::lock_guard<std::mutex> lock(b.mutex);return venturi_qwen_mtp_debug_proposals(&b.at(slot),ids,count);}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_batch_draft_snapshot(void* h,int slot,int kind,void* out,std::uint64_t bytes){try{vr::require(h,"null MTP batch draft snapshot");auto& b=*static_cast<vr::MTPBatch*>(h);std::lock_guard<std::mutex> lock(b.mutex);return venturi_qwen_mtp_draft_snapshot(&b.at(slot),kind,out,bytes);}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_batch_stats(void* h,int slot,std::uint64_t* out,double* seconds){try{vr::require(h,"null MTP batch stats");auto& b=*static_cast<vr::MTPBatch*>(h);std::lock_guard<std::mutex> lock(b.mutex);return venturi_qwen_mtp_stats(&b.at(slot),out,seconds);}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_batch_extra_info(void* h,std::uint64_t* out,double* seconds){try{vr::require(h && out && seconds,"null MTP batch extra info");auto& b=*static_cast<vr::MTPBatch*>(h);std::lock_guard<std::mutex> lock(b.mutex);std::uint64_t request=0,backup=0;for(auto& s:b.requests){request=vr::add(request,s->request_bytes());backup=vr::add(backup,s->checkpoint_first->bytes);}out[0]=b.model->base->device_bytes;out[1]=request;out[2]=backup;out[3]=b.selection.bytes+b.scratch.bytes;seconds[0]=b.verification_seconds;return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}

extern "C" int venturi_qwen_mtp_batch_cohort_timings(void* h,double* out){try{vr::require(h && out,"null MTP batch cohort timings");auto& b=*static_cast<vr::MTPBatch*>(h);std::lock_guard<std::mutex> lock(b.mutex);out[0]=b.prefill_seconds;out[1]=b.generation_seconds;return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}

// Additive opt-in only; ABI16 descriptors and existing MTP entry points stay intact.
extern "C" int venturi_qwen_mtp_layer_graph(void* h,int enabled) {
    try {vr::require(h && (enabled==0 || enabled==1),"invalid target layer graph configuration");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> a(s.mutex);
        vr::require(!s.pipeline && !s.borrowed_target,"scalar target layer graphs exclude PP and borrowed cohorts");auto& m=*s.first().model;std::lock_guard<std::mutex> b(m.mutex);vr::Device d(m.c.device);s.first().configure_layer_graph(enabled!=0);return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" int venturi_qwen_mtp_layer_graph_info(void* h,std::uint64_t* out) {
    try {vr::require(h && out,"null target layer graph info");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);auto* g=s.first().layer_graph.get();
        const std::uint64_t v[]={g?1ULL:0ULL,g&&g->executable?1ULL:0ULL,g?g->captures:0,g?g->replays:0,g?g->fallbacks:0,g?g->warm_calls:0,g?g->nodes:0,g?g->position.bytes:0,g?g->host_position.bytes:0,g?static_cast<std::uint64_t>(*static_cast<int*>(g->host_position.ptr)):0};std::copy(v,v+10,out);return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
