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

template<class T> __global__ void mtp_concat(const T* e,const T* h,T* out,int rows,int width) {
    const auto i=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i<static_cast<std::int64_t>(rows)*width) {const auto r=i/width,c=i%width;out[r*width*2+c]=e[i];out[r*width*2+width+c]=h[i];}
}
struct MTPSession {
    std::shared_ptr<MTPModel> model;
    std::unique_ptr<Session> target,draft;
    std::unique_ptr<PipelineSession> pipeline;
    std::unique_ptr<MTPCheckpoint> checkpoint_first,checkpoint_second;
    Buffer hidden,last_hidden,shifted,embedding_rows,remote_embedding,remote_ids,concat,selection,scratch;
    Pinned bridge;
    std::mutex mutex;
    std::vector<std::int64_t> debug_proposals;
    std::uint64_t proposed=0,accepted=0,verify_calls=0,replayed=0,rollback_calls=0,target_calls=0,target_inputs=0,draft_calls=0,draft_inputs=0,emitted=0,forced=0;
    int last_depth=1;
    std::uint64_t depth_proposed[2]{},depth_verified[2]{},depth_accepted[2]{},depth_forced[2]{},draft_rewinds=0,verified_catchup=0,verify_width2=0,verify_width3=0;
    double draft_seconds=0,verify_seconds=0,rollback_seconds=0,prefill_seconds=0,checkpoint_seconds=0,first_token_seconds=0,generation_seconds=0;
    explicit MTPSession(std::shared_ptr<MTPModel> p,int capacity):model(std::move(p)) {
        if (model->pipeline) pipeline=std::make_unique<PipelineSession>(model->pipeline,capacity);
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
    Session& first() {return pipeline?*pipeline->first:*target;}
    Session& last() {return pipeline?*pipeline->second:*target;}
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
        else {auto& m=*target->model;std::lock_guard<std::mutex> lock(m.mutex);Device d(m.c.device);
            target->forward(ids,n,nullptr,all,nullptr,hidden.ptr);select(*target,all?n:1,out.data());}
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
        if(pipeline)pipeline->reset();else {auto& m=*target->model;std::lock_guard<std::mutex> lock(m.mutex);Device d(m.c.device);target->reset();}
        {auto& m=*model->base;std::lock_guard<std::mutex> lock(m.mutex);Device d(m.c.device);draft->reset();}
        proposed=accepted=verify_calls=replayed=rollback_calls=target_calls=target_inputs=draft_calls=draft_inputs=emitted=forced=0;
        draft_seconds=verify_seconds=rollback_seconds=prefill_seconds=checkpoint_seconds=first_token_seconds=generation_seconds=0;debug_proposals.clear();
        last_depth=1;for(int i=0;i<2;++i)depth_proposed[i]=depth_verified[i]=depth_accepted[i]=depth_forced[i]=0;draft_rewinds=verified_catchup=verify_width2=verify_width3=0;
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
    void generate2(const std::int64_t* prompt,int n,int maximum,const std::int64_t* eos,int eos_count,std::int64_t* out,int* count) {
        const double begun=mtp_clock();
        const auto& c=model->base->c;
        require(c.max_tokens>=3,"MTP2 requires at least three projection rows");
        require(!failed() && position()==0 && draft->position==0,"MTP generation requires fresh/reset request state");
        first().input(prompt,n);
        require(maximum>0 && out && count && eos_count>=0 && (eos || !eos_count) && add(n,maximum)<=static_cast<std::uint64_t>(first().capacity),"Invalid MTP generation capacity/arguments");
        for(int i=0;i<eos_count;++i)require(eos[i]>=0 && eos[i]<c.vocab,"MTP EOS out of range");
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
                start=mtp_clock();checkpoint(true);checkpoint_seconds+=mtp_clock()-start;
                start=mtp_clock();target_forward(inputs,depth+1,true,selected);verify_seconds+=mtp_clock()-start;++verify_calls;
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
                    start=mtp_clock();checkpoint(false);++rollback_calls;std::vector<std::int64_t> replay;
                    target_forward(inputs,committed,false,replay);replayed=add(replayed,committed);
                    require(token(replay,0)==token(selected,committed-1),"MTP2 replay target greedy mismatch");
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
extern "C" int venturi_qwen_mtp_chain_stats(void* h,std::uint64_t* out){try{vr::require(h && out,"null MTP chain stats");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);
    const bool two=s.last_depth==2;
    const std::uint64_t values[]={static_cast<std::uint64_t>(s.last_depth),two?s.depth_proposed[0]:s.proposed,two?s.depth_proposed[1]:0,two?s.depth_verified[0]:s.proposed,two?s.depth_verified[1]:0,two?s.depth_accepted[0]:s.accepted,two?s.depth_accepted[1]:0,s.draft_rewinds,s.verified_catchup,two?s.verify_width2:s.verify_calls,s.verify_width3,two?s.depth_forced[0]:s.forced,two?s.depth_forced[1]:0};std::copy(values,values+13,out);return 0;
}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_reset(void* h){try{vr::require(h,"null MTP session");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);s.reset();return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_info(void* h,std::uint64_t* out){try{vr::require(h && out,"null MTP info");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);return s.pipeline?venturi_qwen_pipeline_info(s.pipeline.get(),out):venturi_qwen_info(s.target.get(),out);}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_parallel_info(void* h,std::uint64_t* out){try{vr::require(h && out,"null MTP info");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);return s.pipeline?venturi_qwen_pipeline_parallel_info(s.pipeline.get(),out):venturi_qwen_parallel_info(s.target.get(),out);}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_snapshot(void* h,int layer,int kind,void* out,std::uint64_t bytes){try{vr::require(h,"null MTP snapshot");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);return s.pipeline?venturi_qwen_pipeline_snapshot(s.pipeline.get(),layer,kind,out,bytes):venturi_qwen_snapshot(s.target.get(),layer,kind,out,bytes);}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_debug_failure(void* h,int layer,int pos){try{vr::require(h,"null MTP fault");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);return s.pipeline?venturi_qwen_pipeline_debug_failure(s.pipeline.get(),layer,pos):venturi_qwen_debug_failure(s.target.get(),layer,pos);}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_stats(void* h,std::uint64_t* out,double* seconds){try{vr::require(h && out && seconds,"null MTP stats");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);
    const std::uint64_t a[]={s.proposed,s.accepted,s.verify_calls,s.replayed,s.rollback_calls,s.target_calls,s.target_inputs,s.draft_calls,s.draft_inputs,s.emitted,s.forced,static_cast<std::uint64_t>(s.draft->position),s.model->base->device_bytes,s.request_bytes(),s.checkpoint_first->bytes,s.checkpoint_second?s.checkpoint_second->bytes:0,s.remote_embedding.bytes+s.remote_ids.bytes,s.bridge.bytes,static_cast<std::uint64_t>(s.model->base->c.device),s.pipeline?s.last().device_bytes:0};
    std::copy(a,a+20,out);const double b[]={s.draft_seconds,s.verify_seconds,s.rollback_seconds,s.prefill_seconds,s.checkpoint_seconds};std::copy(b,b+5,seconds);return 0;
}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_debug_proposals(void* h,const std::int64_t* ids,int n){try{vr::require(h && n>=0 && (ids || !n),"invalid MTP debug proposals");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);vr::require(s.position()==0,"debug proposals require fresh state");for(int i=0;i<n;++i)vr::require(ids[i]>=0 && ids[i]<s.model->base->c.vocab,"invalid forced proposal");s.debug_proposals.assign(ids,ids+n);return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_draft(void* h,const std::int64_t* ids,int n,const void* hidden,float* logits,void* output){try{vr::require(h && hidden && output,"null MTP diagnostic");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);auto& m=*s.model->base;vr::require(n>0 && n<=m.c.max_tokens,"MTP diagnostic chunk exceeds workspace");vr::Device d(m.c.device);check(cudaMemcpyAsync(s.hidden.ptr,hidden,vr::mul(vr::mul(n,m.c.hidden),2),cudaMemcpyHostToDevice,m.stream.ptr));m.sync();try{s.draft_step(ids,n,s.hidden.ptr,logits!=nullptr,nullptr,logits,output);}catch(...){s.settle_failure();throw;}return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}
extern "C" int venturi_qwen_mtp_draft_snapshot(void* h,int kind,void* out,std::uint64_t bytes){try{vr::require(h && (kind==2 || kind==3),"invalid MTP KV snapshot");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);return venturi_qwen_snapshot(s.draft.get(),0,kind,out,bytes);}catch(const std::exception& e){error_text=e.what();return -1;}}

extern "C" int venturi_qwen_mtp_timings(void* h,double* out){try{vr::require(h && out,"null MTP timings");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);out[0]=s.first_token_seconds;out[1]=s.generation_seconds;return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}
