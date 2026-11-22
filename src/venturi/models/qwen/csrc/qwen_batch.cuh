// Grouped projections, independently owned sequence caches and native generation.
#pragma once

namespace vr {
struct BatchSegment {int slot,row,count;};
enum class BatchAttention : int { Legacy=0, Descriptor=1, Graph=2 };
struct BatchFlowCounters {
    std::uint64_t v_direct=0,v_copies=0,hidden_direct=0,hidden_copies=0,fences=0;
    std::uint64_t descriptor_steps=0,descriptor_bytes=0,captures=0,replays=0,eager_layers=0,fallback_layers=0,capture_ns=0;
    std::uint64_t v_copy_bytes=0;
};
struct Batch {
    std::shared_ptr<Model> model;
    std::vector<std::unique_ptr<Session>> requests;
    std::unique_ptr<Session> work;
    Buffer value,last_hidden,head_input,selection,select_scratch;
    std::uint64_t extra_bytes=0,group_calls=0,input_rows=0,decode_calls=0,prefill_calls=0,head_calls=0,head_rows=0,max_rows=0;
    std::mutex mutex;
    const bool q5_prefill_reuse;
    const bool resident_flow;
    const BatchAttention attention;
    const int slot_count;
    std::vector<int> attention_layers;
    Buffer device_rows;
    Pinned host_rows;
    BatchFlowCounters flow;
    // Destroy graphs before any of their buffers; public batch calls are synchronous.
    std::map<std::pair<int,int>,std::unique_ptr<RankGraph>> attention_graphs;
    Batch(std::shared_ptr<Model> m,const int* capacities,int count,bool reuse=false,
          bool resident=false,BatchAttention execution=BatchAttention::Legacy):
        model(std::move(m)),q5_prefill_reuse(reuse),resident_flow(resident),attention(execution),slot_count(count) {
        const auto& c=model->c;
        require(!model->stage && c.resident==1 && capacities && count>0 && count<=32 && count<=c.max_tokens,"batch requires a complete resident target and 1..min(32,max_tokens) slots");
        require((!resident_flow && attention==BatchAttention::Legacy) ||
                (!model->parallel() && !model->kv_tier),
                "resident batch experiments require single-device contiguous KV");
        if(attention!=BatchAttention::Legacy) {
            require(c.dtype==0,"descriptor decode attention requires FP16");
            for(int i=0;i<c.layers;++i)if(model->layers[i].kind)attention_layers.push_back(i);
        }
        if(q5_prefill_reuse) {
            require(count==1 && c.max_tokens==32,"Q5 prefill reuse requires one slot and max_tokens=32");
            require(VENTURI_BLOCKS_ARCH==70 && c.dtype==0 && c.weight_bits==5 &&
                    c.weight_group==64 && !model->parallel() && !c.linear_implementation &&
                    !c.dequant_resident && !model->activation_passes,
                    "Q5 prefill reuse requires canonical resident SM70 Q5-G64 FP16");
            for(const auto& layer:model->layers) {
                require(layer.kind!=0 || layer.ab_fp32,"Q5 prefill reuse needs FP32 GDN side weights");
                for(int i=0;i<7;++i) {
                    if(layer.kind==0 && i==3)continue;
                    const auto& w=layer.matrix[i];
                    require(model->matrix_bits(w)==5 && model->matrix_group(w)==64 && w.k%64==0,
                            "Q5 prefill reuse needs aligned Q5-G64 projections");
                }
            }
        }
        std::uint64_t needed=Session::estimate(*model,std::min(c.max_tokens,c.max_position),false);
        for(int i=0;i<count;++i)needed=add(needed,Session::estimate(*model,capacities[i]));
        const auto descriptor_bytes=mul(mul(attention_layers.size(),count),sizeof(VDecodeAttentionRow));
        const auto additional=add(descriptor_bytes,add(mul(mul(c.max_tokens,c.hkv*c.dim),2),add(mul(mul(count,c.hidden),4),mul(count,16+Session::min_tiles(c.vocab)*16))));
        model->guard(add(needed,additional));
        if(model->head_gdn_enabled){Device device(c.second_device);model->guard(mul(count,Session::estimate_peer(*model)));}
        work=std::make_unique<Session>(model,std::min(c.max_tokens,c.max_position),false);
        for(int i=0;i<count;++i)requests.push_back(std::make_unique<Session>(model,capacities[i]));
        auto alloc=[&](Buffer& b,std::uint64_t n){b.allocate(n);extra_bytes=add(extra_bytes,n);};
        alloc(value,mul(mul(c.max_tokens,c.hkv*c.dim),2));
        alloc(last_hidden,mul(mul(count,c.hidden),2));alloc(head_input,last_hidden.bytes);
        alloc(selection,mul(count,16));alloc(select_scratch,mul(count,Session::min_tiles(c.vocab)*16));
        if(descriptor_bytes) {alloc(device_rows,descriptor_bytes);host_rows.allocate(descriptor_bytes);}
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
    void upload_decode_rows(const std::vector<BatchSegment>& segments) {
        auto* desc=static_cast<VDecodeAttentionRow*>(host_rows.ptr);
        for(std::size_t i=0;i<attention_layers.size();++i) {
            for(std::size_t j=0;j<segments.size();++j) {
                const auto& s=segments[j];auto& req=*requests[s.slot];const int l=attention_layers[i];
                require(s.count==1 && s.row==int(j) && req.position<req.capacity,"invalid indirect decode row");
                desc[i*slot_count+j]={req.kv_ptr(l,0),req.kv_ptr(l,1),
                    static_cast<const float*>(req.p("cos")),static_cast<const float*>(req.p("sin")),req.position,0};
            }
        }
        // Copy all fixed-stride layer tables in one transfer. The unselected
        // tail is initialized on each upload and is never accessed by kernels.
        for(std::size_t i=0;i<attention_layers.size();++i)
            std::fill(desc+i*slot_count+segments.size(),desc+(i+1)*slot_count,VDecodeAttentionRow{});
        check(cudaMemcpyAsync(device_rows.ptr,host_rows.ptr,device_rows.bytes,cudaMemcpyHostToDevice,model->stream.ptr));
        ++flow.descriptor_steps;flow.descriptor_bytes=add(flow.descriptor_bytes,device_rows.bytes);
    }
    void decode_attention(int index,int table,int n) {
        auto& m=*model;auto& w=*work;auto& c=m.c;const auto& l=m.layers[index];
        const auto stream=m.stream.ptr;
        const auto* desc=reinterpret_cast<const VDecodeAttentionRow*>(device_rows.at(mul(mul(table,slot_count),sizeof(VDecodeAttentionRow))));
        const auto body=[&] {
            op(venturi_qwen_decode_attention_f16(w.p("qg"),w.p("key"),
                static_cast<const float*>(l.qgamma),static_cast<const float*>(l.kgamma),desc,
                w.p("query"),w.p("attended"),n,c.hq,c.hkv,c.dim,c.rotary,c.epsilon,c.attention_scale,stream));
            op(venturi_qwen_gate_run(w.p("attended"),w.p("qg"),w.p("attended"),n,c.hq,c.dim,c.dtype,stream));
        };
        // Exact-size graphs, no padded queries. Other sizes run the same
        // descriptor kernels eagerly; the maximum cache is four widths/layer.
        if(attention!=BatchAttention::Graph || !(n==1 || n==2 || n==4 || n==8)) {
            body();++flow.eager_layers;if(attention==BatchAttention::Graph)++flow.fallback_layers;return;
        }
        const auto key=std::make_pair(index,n);
        auto found=attention_graphs.find(key);
        if(found==attention_graphs.end()) {
            const double start=native_clock();
            auto graph=std::make_unique<RankGraph>(c.device);
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            try {body();}
            catch(...) {
                cudaGraph_t discarded=nullptr;(void)cudaStreamEndCapture(stream,&discarded);
                if(discarded)(void)cudaGraphDestroy(discarded);throw;
            }
            check(cudaStreamEndCapture(stream,&graph->graph));
            check(cudaGraphInstantiate(&graph->exec,graph->graph,nullptr,nullptr,0));
            found=attention_graphs.emplace(key,std::move(graph)).first;
            ++flow.captures;flow.capture_ns=add(flow.capture_ns,std::uint64_t((native_clock()-start)*1e9));
        }
        check(cudaGraphLaunch(found->second->exec,stream));++flow.replays;
    }
    // Each segment is a contiguous chunk of ONE sequence. Grouped linear/norm
    // rows are independent; convolution/GDN/attention always use its own cache.
    void group(const std::vector<BatchSegment>& segments,const std::vector<std::int64_t>& tokens,bool decode,
               const std::vector<int>* sample_now=nullptr,std::vector<std::int64_t>* chosen=nullptr) {
        auto& m=*model;auto& w=*work;const auto& c=m.c;auto stream=m.stream.ptr;
        const int n=static_cast<int>(tokens.size()),width=2*c.gqk*c.gkd+c.gvh*c.gvd,qw=c.hq*c.dim,kw=c.hkv*c.dim;
        require(n>0 && n<=c.max_tokens,"batch row tile exceeds model workspace");
        const bool reuse=q5_prefill_reuse && !decode && segments.size()==1 && n>16;
        bool indirect=decode && !attention_layers.empty() && static_cast<int>(segments.size())==n;
        for(const auto& s:segments)indirect=indirect && s.count==1 && !requests[s.slot]->verification_trace;
        if(indirect)upload_decode_rows(segments);
        int table=0;
        if(m.kv_tier){std::vector<std::pair<KVTier::Entry*,std::uint64_t>> goals;for(const auto& g:segments)requests[g.slot]->kv_goals(goals,requests[g.slot]->position+g.count);m.kv_tier->prepare(goals);}
        w.embedding(tokens.data(),n,w.p("x"));void* x=w.p("x");void* out=w.p("out");
        std::vector<HeadGDNSegment> private_segments;if(m.head_gdn_enabled)private_segments.reserve(segments.size());
        for(int index=0;index<c.layers;++index) {
            const auto& l=m.layers[index];
            op(venturi_rms_run(x,static_cast<const float*>(l.norm),w.p("normalized"),n,c.hidden,c.dtype,c.epsilon,stream));
            if(l.kind==0 && m.head_gdn_enabled) {
                private_segments.clear();
                for(const auto& segment:segments)private_segments.push_back(requests[segment.slot]->head_segment(index,segment.row,segment.count));
                head_gdn_forward(m,index,w.p("normalized"),w.p("projected"),n,private_segments);
            } else if(l.kind==0) {
                m.shared_linears(w.p("normalized"),n,{{&l.matrix[0],w.p("qkv")},{&l.matrix[1],w.p("z")}},reuse);
                for(int j=0;j<2;++j) {
                    auto weight=j?l.b:l.a;auto output=w.p(j?"b":"a");
                    if(l.ab_fp32)op(venturi_fp32_linear_run(w.p("normalized"),static_cast<const float*>(weight),output,n,c.gvh,c.hidden,c.dtype,stream));
                    else {op(venturi_dense_run(m.dense.ptr,w.p("normalized"),weight,output,n,c.gvh,c.hidden,m.blas_workspace.ptr,m.blas_workspace.bytes,stream));}
                }
                op(venturi_gates_run(w.p("a"),w.p("b"),static_cast<const float*>(l.rate),static_cast<const float*>(l.bias),static_cast<float*>(w.p("decay")),static_cast<float*>(w.p("beta")),n,c.gvh,c.dtype,1,stream));
                bool any_trace=false;for(const auto& s:segments)if(requests[s.slot]->verification_trace)any_trace=true;
                // PR-A+1: promote the head-GDN batch decode kernels to the plain
                // path. Pure-decode cohorts (every segment one token, no trace,
                // <=32 segments, gkd 64/128) collapse the per-layer x per-request
                // serial conv+GDN launches into one conv_segments + one
                // gdn_segments launch. These kernels store FP16 and need history: BF16
                // and kernel-size-one use the unchanged serial reference.
                bool batched=c.dtype==0 && c.convolution>=2 && !any_trace && segments.size()>1 && segments.size()<=32 && (c.gkd==64 || c.gkd==128);
                for(const auto& s:segments)batched=batched&&s.count==1;
                if(batched){
                    DecodeSegments batch{};batch.count=static_cast<int>(segments.size());
                    for(int j=0;j<batch.count;++j)batch.slots[j]={segments[j].row,requests[segments[j].slot]->history[index]->ptr,requests[segments[j].slot]->state[index]->ptr};
                    const dim3 cgrid((width+255)/256,batch.count);
                    if(l.conv_fp32)conv_segments<float><<<cgrid,256,0,stream>>>(static_cast<const __half*>(w.p("qkv")),static_cast<const float*>(l.conv),static_cast<__half*>(w.p("convolved")),batch,width,c.convolution);
                    else conv_segments<__half><<<cgrid,256,0,stream>>>(static_cast<const __half*>(w.p("qkv")),static_cast<const __half*>(l.conv),static_cast<__half*>(w.p("convolved")),batch,width,c.convolution);
                    check(cudaGetLastError());
                    gdn_segments<<<dim3(c.gvh*c.gvd,batch.count),c.gkd,0,stream>>>(
                        static_cast<const __half*>(w.p("convolved")),
                        static_cast<const float*>(w.p("decay")),static_cast<const float*>(w.p("beta")),
                        static_cast<__half*>(w.p("recurrent")),batch,c.gqk,c.gvh,c.gkd,c.gvd,c.gdn_scale);
                    check(cudaGetLastError());
                } else for(const auto& segment:segments) {
                    auto& request=*requests[segment.slot];const int r=segment.row,t=segment.count;
                    auto* trace=request.verification_trace;
                    if(trace) {
                        require(t==trace->prefixes+1,"batch prefix trace width mismatch");
                        op(venturi_history_prefix_run(row("qkv",r,width),request.history[index]->ptr,trace->history[0][index],trace->history[1][index],width,c.convolution,trace->prefixes,c.dtype,stream));
                    }
                    op(venturi_conv_run(row("qkv",r,width),l.conv,request.history[index]->ptr,row("convolved",r,width),request.history[index]->ptr,t,width,c.convolution,c.dtype,l.conv_fp32,stream));
                    auto* q=row("convolved",r,width);const auto stride=mul(c.gqk*c.gkd,2);
                    if(trace)op(venturi_gdn_prefix_run(q,q+stride,q+2*stride,reinterpret_cast<const float*>(row("decay",r,c.gvh,4)),reinterpret_cast<const float*>(row("beta",r,c.gvh,4)),static_cast<const float*>(request.state[index]->ptr),row("recurrent",r,c.gvh*c.gvd),static_cast<float*>(request.state[index]->ptr),t,c.gqk,c.gvh,c.gkd,c.gvd,1,1,c.dtype,c.gdn_scale,trace->state[0][index],trace->state[1][index],trace->prefixes,stream));
                    else op((m.activation_passes==2?venturi_gdn_compact_run:venturi_gdn_run)(q,q+stride,q+2*stride,reinterpret_cast<const float*>(row("decay",r,c.gvh,4)),reinterpret_cast<const float*>(row("beta",r,c.gvh,4)),static_cast<const float*>(request.state[index]->ptr),row("recurrent",r,c.gvh*c.gvd),static_cast<float*>(request.state[index]->ptr),t,c.gqk,c.gvh,c.gkd,c.gvd,1,1,c.dtype,c.gdn_scale,stream));
                }
                op(venturi_gated_rms_run(w.p("recurrent"),w.p("z"),l.gamma,w.p("mixed"),n,c.gvh,c.gvd,c.dtype,l.gamma_fp32,c.epsilon,stream));
                m.linear(l.matrix[2],w.p("mixed"),w.p("projected"),n,false,false,reuse);
            } else {
                const bool direct_v=resident_flow && segments.size()==1;
                void* v_destination=value.ptr;
                if(direct_v) {
                    auto& request=*requests[segments[0].slot];
                    v_destination=request.kv_ptr(index,1,mul(mul(request.position,kw),2));
                    ++flow.v_direct;
                }
                m.shared_linears(w.p("normalized"),n,{{&l.matrix[0],w.p("qg")},{&l.matrix[1],w.p("key")},{&l.matrix[2],v_destination}},reuse);
                if(false && segments.size()>1) {   // night-tps-20261117: disabled with the linear path above
                    // night-tps-20261116: per-segment KV copy + RoPE + attention on
                    // parallel streams (disjoint KV ranges and workspace rows).
                    // kv_ensure runs on the main stream first so any pool growth is
                    // ordered before the parallel section.
                    const auto scount=static_cast<int>(segments.size());
                    m.ensure_segment_streams(scount);
                    for(const auto& segment:segments){auto& request=*requests[segment.slot];request.kv_ensure(index,request.position+segment.count);}
                    check(cudaEventRecord(m.group_ready.ptr,stream));
                    for(int si=0;si<scount;++si) {
                        const auto& segment=segments[si];auto& request=*requests[segment.slot];
                        const int r=segment.row,t=segment.count,pos=request.position;
                        cudaStream_t ss=m.segment_streams[si].ptr;
                        check(cudaStreamWaitEvent(ss,m.group_ready.ptr,0));
                        auto* kt=request.kv_ptr(index,0,mul(mul(pos,kw),2));auto* vt=request.kv_ptr(index,1,mul(mul(pos,kw),2));
                        check(cudaMemcpyAsync(vt,value.at(mul(mul(r,kw),2)),mul(mul(t,kw),2),cudaMemcpyDeviceToDevice,ss));
                        auto* cos=static_cast<const float*>(request.p("cos"))+mul(pos,c.rotary/2);auto* sin=static_cast<const float*>(request.p("sin"))+mul(pos,c.rotary/2);
                        op(venturi_qwen_qk_run(row("qg",r,2*qw),row("key",r,kw),static_cast<const float*>(l.qgamma),static_cast<const float*>(l.kgamma),cos,sin,row("query",r,qw),kt,t,c.hq,c.hkv,c.dim,c.rotary,c.dtype,c.epsilon,ss));
                        op(venturi_attention_run(row("query",r,qw),request.kv_ptr(index,0),request.kv_ptr(index,1),row("attended",r,qw),t,pos+t,c.hq,c.hkv,c.dim,c.dtype,c.attention_scale,ss));request.kv_written(index,pos+t);
                        check(cudaEventRecord(m.segment_done[si].ptr,ss));
                    }
                    for(int si=0;si<scount;++si)check(cudaStreamWaitEvent(stream,m.segment_done[si].ptr,0));
                } else for(const auto& segment:segments) {
                    auto& request=*requests[segment.slot];const int r=segment.row,t=segment.count,pos=request.position;
                    request.kv_ensure(index,pos+t);auto* kt=request.kv_ptr(index,0,mul(mul(pos,kw),2));auto* vt=request.kv_ptr(index,1,mul(mul(pos,kw),2));
                    if(!direct_v) {
                        const auto bytes=mul(mul(t,kw),2);
                        check(cudaMemcpyAsync(vt,value.at(mul(mul(r,kw),2)),bytes,cudaMemcpyDeviceToDevice,stream));
                        ++flow.v_copies;
                        flow.v_copy_bytes=add(flow.v_copy_bytes,bytes);
                    }
                    if(!indirect) {
                        auto* cos=static_cast<const float*>(request.p("cos"))+mul(pos,c.rotary/2);auto* sin=static_cast<const float*>(request.p("sin"))+mul(pos,c.rotary/2);
                        op(venturi_qwen_qk_run(row("qg",r,2*qw),row("key",r,kw),static_cast<const float*>(l.qgamma),static_cast<const float*>(l.kgamma),cos,sin,row("query",r,qw),kt,t,c.hq,c.hkv,c.dim,c.rotary,c.dtype,c.epsilon,stream));
                        op(venturi_attention_run(row("query",r,qw),request.kv_ptr(index,0),request.kv_ptr(index,1),row("attended",r,qw),t,pos+t,c.hq,c.hkv,c.dim,c.dtype,c.attention_scale,stream));
                    }
                    request.kv_written(index,pos+t); // descriptor mode admits only plain private KV
                }
                if(indirect)decode_attention(index,table++,n);
                else op(venturi_qwen_gate_run(w.p("attended"),w.p("qg"),w.p("attended"),n,c.hq,c.dim,c.dtype,stream));
                m.linear(l.matrix[3],w.p("attended"),w.p("projected"),n,false,false,reuse);
            }
            op(venturi_add_run(x,w.p("projected"),w.p("residual"),mul(n,c.hidden),c.dtype,stream));
            const bool packed_mlp=m.normalize_mlp(w.p("residual"),static_cast<const float*>(l.post),w.p("mlp_norm"),n);
            m.mlp(l,w.p("mlp_norm"),w.p("mlp_gate"),w.p("mlp_up"),w.p("projected"),n,false,packed_mlp,reuse);
            op(venturi_add_run(w.p("residual"),w.p("projected"),out,mul(n,c.hidden),c.dtype,stream));
            std::swap(x,out);
            for(const auto& segment:segments) {
                auto& request=*requests[segment.slot];
                if(index==request.fault_layer && request.position==request.fault_position){request.fault_layer=-1;throw std::runtime_error("injected native batch layer failure");}
            }
        }
        op(venturi_rms_run(x,m.norm,w.p("normalized"),n,c.hidden,c.dtype,c.epsilon,stream));
        if(sample_now) {
            require(resident_flow && chosen && !sample_now->empty(),"invalid combined batch head");
            std::array<int,32> rows{};
            int j=0;bool contiguous=true;
            for(int slot:*sample_now) {
                const auto found=std::find_if(segments.begin(),segments.end(),[&](const auto& s){return s.slot==slot;});
                require(found!=segments.end(),"combined head sample is not in the packed group");
                rows[j]=found->row+found->count-1;contiguous=contiguous && rows[j]==rows[0]+j;++j;
            }
            const auto width=mul(c.hidden,2);
            const void* head_rows_ptr;
            if(contiguous){head_rows_ptr=row("normalized",rows[0],c.hidden);++flow.hidden_direct;}
            else {
                for(int i=0;i<j;++i)check(cudaMemcpyAsync(head_input.at(mul(i,width)),row("normalized",rows[i],c.hidden),width,cudaMemcpyDeviceToDevice,stream));
                flow.hidden_copies=add(flow.hidden_copies,j);head_rows_ptr=head_input.ptr;
            }
            // Only publish positions after the head's completion fence and
            // result validation. Do not persist a hidden row used only here.
            head_from(head_rows_ptr,j,chosen,nullptr);
        } else {
            for(const auto& segment:segments)check(cudaMemcpyAsync(last_hidden.at(mul(mul(segment.slot,c.hidden),2)),row("normalized",segment.row+segment.count-1,c.hidden),mul(c.hidden,2),cudaMemcpyDeviceToDevice,stream));
            flow.hidden_copies=add(flow.hidden_copies,segments.size());
            m.sync();m.source();++flow.fences;
        }
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
        auto& m=*model;const auto& c=m.c;const int n=static_cast<int>(ids.size());
        for(int i=0;i<n;++i)check(cudaMemcpyAsync(head_input.at(mul(mul(i,c.hidden),2)),last_hidden.at(mul(mul(ids[i],c.hidden),2)),mul(c.hidden,2),cudaMemcpyDeviceToDevice,m.stream.ptr));
        flow.hidden_copies=add(flow.hidden_copies,n);
        head_from(head_input.ptr,n,chosen,logits);
    }
    void head_from(const void* input,int n,std::vector<std::int64_t>* chosen,float* logits) {
        auto& m=*model;auto& w=*work;const auto& c=m.c;
        m.linear(m.head,input,w.p("logits"),n);
        if(chosen) {
            chosen->resize(2*n);op(venturi_greedy_run(w.p("logits"),static_cast<std::int64_t*>(selection.ptr),select_scratch.ptr,select_scratch.bytes,n,c.vocab,c.dtype,m.stream.ptr));
            check(cudaMemcpyAsync(chosen->data(),selection.ptr,mul(n,16),cudaMemcpyDeviceToHost,m.stream.ptr));
        }
        if(logits)check(cudaMemcpyAsync(w.host_logits.data(),w.p("logits"),mul(mul(n,c.vocab),2),cudaMemcpyDeviceToHost,m.stream.ptr));
        m.sync();m.source();++flow.fences;
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
                if(active.empty())break;
                if(resident_flow)group(segments,packed,true,&active,&chosen);
                else {group(segments,packed,true);head(active,&chosen,nullptr);}
                for(std::size_t j=0;j<active.size();++j)emit(indices[j],chosen[2*j]);
            }
        }catch(...){if(begun)poison(ids);throw;}
    }
};
}

namespace {
void* create_qwen_batch(void* owner,const int* capacities,int count,bool q5_prefill_reuse,
                        bool flow=false,vr::BatchAttention attention=vr::BatchAttention::Legacy) {
    try {
        vr::require(owner,"null batch target");auto m=*static_cast<vr::Handle*>(owner);
        vr::Device device(m->c.device);std::lock_guard<std::mutex> lock(m->mutex);
        m->source();return new vr::Batch(m,capacities,count,q5_prefill_reuse,flow,attention);
    } catch(const std::exception& e){error_text=e.what();return nullptr;}
}
}
extern "C" VENTURI_EXPORT void* venturi_qwen_batch_create(void* owner,const int* capacities,int count) {
    return create_qwen_batch(owner,capacities,count,false);
}
extern "C" VENTURI_EXPORT void* venturi_qwen_batch_create_q5_prefill_reuse(void* owner,const int* capacities,int count) {
    return create_qwen_batch(owner,capacities,count,true);
}
extern "C" VENTURI_EXPORT void* venturi_qwen_batch_create_resident_v1(
    void* owner,const int* capacities,int count,int reuse,int flow,int attention) {
    if(reuse<0 || reuse>1 || flow<0 || flow>1 || attention<0 || attention>2) {
        error_text="invalid explicit resident batch mode";return nullptr;
    }
    return create_qwen_batch(owner,capacities,count,reuse==1,flow==1,static_cast<vr::BatchAttention>(attention));
}
extern "C" VENTURI_EXPORT int venturi_qwen_batch_resident_info_v1(void* h,std::uint64_t* out,int count) {
    try {
        vr::require(h && out && count==16,"invalid resident batch counters");
        auto& b=*static_cast<vr::Batch*>(h);std::lock_guard<std::mutex> lock(b.mutex);const auto& f=b.flow;
        const std::uint64_t values[]={f.v_direct,f.v_copies,f.hidden_direct,f.hidden_copies,f.fences,
            f.descriptor_steps,f.descriptor_bytes,f.captures,f.replays,f.eager_layers,f.fallback_layers,
            f.capture_ns,b.device_rows.bytes,b.host_rows.bytes,
            f.v_copy_bytes,vr::mul(f.hidden_copies,vr::mul(b.model->c.hidden,2))};
        std::copy(std::begin(values),std::end(values),out);return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" VENTURI_EXPORT int venturi_qwen_batch_prepare_dense_io_v1(void* h,int direct,std::uint64_t budget) {
    try {
        vr::require(h && (direct==0 || direct==1),"invalid explicit dense I/O settings");
        auto& b=*static_cast<vr::Batch*>(h);auto& m=*b.model;
        std::lock_guard<std::mutex> a(b.mutex),lock(m.mutex);vr::Device device(m.c.device);
        m.source();m.prepare_dense_io(direct==1,budget);return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" VENTURI_EXPORT int venturi_qwen_batch_dense_io_info_v1(void* h,std::uint64_t* out,int count) {
    try {
        vr::require(h && out && count==15,"invalid dense I/O counters");
        auto& b=*static_cast<vr::Batch*>(h);auto& m=*b.model;
        std::lock_guard<std::mutex> a(b.mutex),lock(m.mutex);const auto& s=m.dense_io;
        const std::uint64_t values[]={m.prior_dense_workspace_bytes,vr::add(m.dequant.bytes,m.q4_chunk.bytes),
            m.dequant.bytes,m.q4_chunk.bytes,s.budget,s.storage?s.storage->bytes:0,s.tiles.size(),
            s.setup_ns,s.gemms,s.direct_gemms,s.output_copy_bytes,s.unpack_calls,s.unpack_bytes,s.cached_calls,s.cached_bytes};
        std::copy(std::begin(values),std::end(values),out);return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" VENTURI_EXPORT void venturi_qwen_batch_destroy(void* h){delete static_cast<vr::Batch*>(h);}
extern "C" VENTURI_EXPORT int venturi_qwen_batch_forward(void* h,const int* slots,int count,const std::int64_t* tokens,const int* offsets,float* logits,std::uint64_t elements) {
    try {vr::require(h && logits,"null batch forward");auto& b=*static_cast<vr::Batch*>(h);std::lock_guard<std::mutex> a(b.mutex),lock(b.model->mutex);vr::Device device(b.model->c.device);auto ids=b.slots(slots,count);vr::require(elements==vr::mul(count,b.model->c.vocab),"batch logit shape mismatch");b.input(ids,tokens,offsets);
        try {b.prefill(ids,tokens,offsets);b.head(ids,nullptr,logits);}catch(...){b.poison(ids);throw;}return 0;}
    catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" VENTURI_EXPORT int venturi_qwen_batch_generate(void* h,const int* slots,int count,const std::int64_t* tokens,const int* offsets,const int* maximum,const std::int64_t* eos,const int* eos_offsets,std::int64_t* out,int stride,int* counts) {
    try {vr::require(h,"null batch generation");auto& b=*static_cast<vr::Batch*>(h);std::lock_guard<std::mutex> a(b.mutex),lock(b.model->mutex);vr::Device device(b.model->c.device);b.generate(b.slots(slots,count),tokens,offsets,maximum,eos,eos_offsets,out,stride,counts);return 0;}
    catch(const std::exception& e){error_text=e.what();return -1;}
}
// Additive entry point: one bounded mixed iteration, no new kernel arithmetic or
// implicit request admission. Preserve the existing cohort ABI and generation.
extern "C" VENTURI_EXPORT int venturi_qwen_batch_step(void* h,const int* slots,int count,const std::int64_t* tokens,const int* offsets,const int* sample_slots,int sample_count,const int* decode_slots,int decode_count,std::int64_t* out,std::uint64_t elements) {
    try {
        vr::require(h && sample_count>=0 && decode_count>=0,"invalid native step arguments");
        auto& b=*static_cast<vr::Batch*>(h);
        std::lock_guard<std::mutex> a(b.mutex),lock(b.model->mutex);
        vr::Device device(b.model->c.device);
        auto ids=b.slots(slots,count);
        vr::require(sample_count<=count && decode_count<=count && elements==static_cast<std::uint64_t>(sample_count) && (!sample_count || out),"invalid native step output extent");
        auto samples=sample_count?b.slots(sample_slots,sample_count):std::vector<int>{};
        auto decodes=decode_count?b.slots(decode_slots,decode_count):std::vector<int>{};
        std::set<int> selected(ids.begin(),ids.end()),decode_set(decodes.begin(),decodes.end());
        for(auto slot:samples)vr::require(selected.count(slot),"sample slot is outside this iteration");
        for(auto slot:decodes)vr::require(selected.count(slot),"decode slot is outside this iteration");
        b.model->source();b.input(ids,tokens,offsets);
        vr::require(offsets[count]<=b.model->c.max_tokens,"step exceeds prepared token rows");
        std::vector<vr::BatchSegment> segments;segments.reserve(ids.size());
        for(int j=0;j<count;++j) {
            const int length=offsets[j+1]-offsets[j];
            vr::require(!decode_set.count(ids[j]) || length==1,"decode slot requires one token");
            segments.push_back({ids[j],offsets[j],length});
        }
        std::vector<std::int64_t> packed(tokens,tokens+offsets[count]);
        try {
            std::vector<std::int64_t> chosen;
            if(sample_count && b.resident_flow)
                b.group(segments,packed,decode_count==count,&samples,&chosen);
            else b.group(segments,packed,decode_count==count);
            if(sample_count) {
                if(!b.resident_flow)b.head(samples,&chosen,nullptr);
                for(int j=0;j<sample_count;++j)out[j]=chosen[2*j];
            }
        }catch(...){b.poison(ids);throw;}
        return 0;
    }catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" VENTURI_EXPORT int venturi_qwen_batch_reset(void* h,const int* slots,int count) {
    try {vr::require(h,"null batch reset");auto& b=*static_cast<vr::Batch*>(h);std::lock_guard<std::mutex> a(b.mutex),lock(b.model->mutex);vr::Device device(b.model->c.device);auto ids=b.slots(slots,count);b.model->source();for(auto i:ids){b.requests[i]->reset();b.requests[i]->first_token_seconds=b.requests[i]->generation_seconds=0;}return 0;}
    catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" VENTURI_EXPORT int venturi_qwen_batch_info(void* h,std::uint64_t* shared,std::uint64_t* rows,double* timings) {
    try {vr::require(h && shared && rows && timings,"null batch info");auto& b=*static_cast<vr::Batch*>(h);std::lock_guard<std::mutex> a(b.mutex),lock(b.model->mutex);auto& m=*b.model;std::uint64_t total=0;
        for(std::size_t i=0;i<b.requests.size();++i){auto& r=*b.requests[i];rows[4*i]=r.position;rows[4*i+1]=r.failed;rows[4*i+2]=r.capacity;rows[4*i+3]=r.device_bytes;timings[2*i]=r.first_token_seconds;timings[2*i+1]=r.generation_seconds;total=vr::add(total,r.device_bytes);}
        const std::uint64_t values[]={b.requests.size(),b.work->device_bytes+b.extra_bytes,total,m.device_bytes,m.second_device_bytes,m.stage_bytes+m.host_input.bytes+m.host_output.bytes,b.group_calls,b.input_rows,b.decode_calls,b.prefill_calls,b.head_calls,b.head_rows,b.max_rows,m.peer_input_bytes,m.peer_output_bytes};std::copy(values,values+15,shared);return 0;}
    catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" VENTURI_EXPORT int venturi_qwen_batch_snapshot(void* h,int slot,int layer,int kind,void* target,std::uint64_t bytes) {
    try {vr::require(h,"null batch snapshot");auto& b=*static_cast<vr::Batch*>(h);std::lock_guard<std::mutex> a(b.mutex),lock(b.model->mutex);vr::Device device(b.model->c.device);vr::require(slot>=0 && slot<static_cast<int>(b.requests.size()) && layer>=0 && layer<b.model->c.layers && kind>=0 && kind<=3,"invalid batch snapshot slot/layer/kind");
        b.requests[slot]->snapshot(layer,kind,target,bytes);return 0;}
    catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" VENTURI_EXPORT int venturi_qwen_batch_debug_failure(void* h,int slot,int layer,int position) {
    try {vr::require(h,"null batch failure hook");auto& b=*static_cast<vr::Batch*>(h);std::lock_guard<std::mutex> a(b.mutex),lock(b.model->mutex);vr::require(slot>=0 && slot<static_cast<int>(b.requests.size()) && layer>=0 && layer<b.model->c.layers && position>=0,"invalid batch failure hook");b.requests[slot]->fault_layer=layer;b.requests[slot]->fault_position=position;return 0;}
    catch(const std::exception& e){error_text=e.what();return -1;}
}
extern "C" VENTURI_EXPORT int venturi_qwen_batch_head_gdn_memory(void* h,std::uint64_t* out) {
    try{vr::require(h && out,"null distributed batch memory info");auto& b=*static_cast<vr::Batch*>(h);std::lock_guard<std::mutex> lock(b.mutex);std::uint64_t peer=0;for(const auto& r:b.requests)peer=vr::add(peer,r->peer_device_bytes);out[0]=peer;out[1]=0;return 0;}catch(const std::exception& e){error_text=e.what();return -1;}
}
