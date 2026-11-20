from pathlib import Path
p=Path('src/venturi/ops/csrc/qwen_runtime.cuh');s=p.read_text()
s=s.replace('#include <map>','#include <map>\n#include <set>')
s=s.replace('VRConfig c; std::vector<VRLayer> layers; VRMatrix embedding,head;', 'VRConfig c; std::vector<VRLayer> layers; VRMatrix embedding,head;\n    int begin=0,end=0;bool stage=false;\n    bool owns(int layer) const { return begin<=layer && layer<end; }')
s=s.replace('const void* final_norm,const float* inv):c(cfg),embedding(emb),head(h) {','const void* final_norm,const float* inv,int first=0,int last=-1,bool staged=false):c(cfg),embedding(emb),head(h),begin(first),end(last<0?cfg.layers:last),stage(staged) {\n        require(begin>=0 && begin<end && end<=c.layers,"invalid native stage range");\n        require(!stage || (c.resident==1 && c.second_device==-1),"pipeline stages require resident non-TP weights");')
old='''        std::uint64_t all=0,other=0,maxk=0,maxn=0,peak=mul(c.max_tokens,add(padded(embedding),padded(embedding)/32*2));'''
new='''        if (stage) {
            std::set<std::uint64_t> used;
            if (begin==0) used.insert(embedding.offset);
            if (end==c.layers) used.insert(head.offset);
            for (int i=begin;i<end;++i) for (int j=0;j<7;++j)
                if (layers[i].kind==1 || j!=3) used.insert(layers[i].matrix[j].offset);
            for (auto it=matrices.begin();it!=matrices.end();)
                if (!used.count(it->first)) it=matrices.erase(it);else ++it;
        }
        std::uint64_t all=0,other=0,maxk=0,maxn=0,peak=mul(c.max_tokens,add(padded(embedding),padded(embedding)/32*2));'''
assert old in s;s=s.replace(old,new)
s=s.replace('''        norm=static_cast<const float*>(copy_constant(final_norm,mul(c.hidden,4)));
        for (auto& l:layers) {''','''        if (end==c.layers) norm=static_cast<const float*>(copy_constant(final_norm,mul(c.hidden,4)));
        for (int index=begin;index<end;++index) {auto& l=layers[index];''')
s=s.replace('for (const auto& l:m.layers) n=add(n,l.kind==0 ?', 'for (int i=m.begin;i<m.end;++i) {const auto& l=m.layers[i];n=add(n,l.kind==0 ?')
s=s.replace(' : mul(mul(capacity,kw),4));\n        return n;', ' : mul(mul(capacity,kw),4));}\n        return n;')
s=s.replace('for (int i=0;i<c.layers;++i) {\n            if (md.layers[i].kind==0)', 'for (int i=md.begin;i<md.end;++i) {\n            if (md.layers[i].kind==0)')
start=s.index('    void forward(const std::int64_t* tokens,int n,float* logits,bool all,void* trace) {')
stop=s.index('    void reset() {',start)
old=s[start:stop]
new='''    void* run_layers(int pos,int t,void* x,void* trace,int n,int start) {
        auto& m=*model;const auto& c=m.c;void* out=p("out");
        for (int i=m.begin;i<m.end;++i) {
            layer(i,pos,t,x,out);std::swap(x,out);
            if (trace) check(cudaMemcpyAsync(static_cast<char*>(trace)+mul(add(mul(i,n),start),mul(c.hidden,2)),x,mul(mul(t,c.hidden),2),cudaMemcpyDeviceToHost,m.stream.ptr));
            if (i==fault_layer && pos==fault_position) { fault_layer=-1;throw std::runtime_error("injected native Qwen layer failure"); }
        }
        return x;
    }
    void output(void* x,int t,bool all,float* logits) {
        auto& m=*model;const auto& c=m.c;
        op(venturi_rms_run(x,m.norm,p("normalized"),t,c.hidden,c.dtype,c.epsilon,m.stream.ptr));
        const int rows=all?t:1;
        m.linear(m.head,static_cast<char*>(p("normalized"))+(all?0:mul(mul(t-1,c.hidden),2)),p("logits"),rows);
        if (logits) {
            const auto elements=mul(rows,c.vocab);
            check(cudaMemcpyAsync(host_logits.data(),p("logits"),mul(elements,2),cudaMemcpyDeviceToHost,m.stream.ptr));m.sync();
            for (std::uint64_t j=0;j<elements;++j) logits[j]=c.dtype==0 ? __half2float(reinterpret_cast<const __half*>(host_logits.data())[j]) : __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(host_logits.data())[j]);
        }
    }
    void forward(const std::int64_t* tokens,int n,float* logits,bool all,void* trace) {
        input(tokens,n);auto& m=*model;const auto& c=m.c;const int initial=position;bool started=false;
        require(!m.stage,"internal pipeline stage cannot execute as a complete model");
        try {
            for (int start=0;start<n;start+=c.max_tokens) {
                const int t=std::min(c.max_tokens,n-start),pos=initial+start;
                started=true;embedding(tokens+start,t,p("x"));
                auto* x=run_layers(pos,t,p("x"),trace,n,start);
                if (all || start+t==n) output(x,t,all,logits?logits+(all?mul(start,c.vocab):0):nullptr);
            }
            m.sync();m.source();position=initial+n;
        } catch (...) { if (started) { failed=true;try { m.sync(); } catch (...) {} }throw; }
    }
'''
s=s[:start]+new+s[stop:]
p.write_text(s)
p=Path('src/venturi/ops/csrc/blocks.cu');s=p.read_text().replace('venturi_blocks_abi() { return 13; }','venturi_blocks_abi() { return 14; }');s+='\n#include "qwen_pipeline.cuh"\n';p.write_text(s)
p=Path('src/venturi/ops/blocks.py');s=p.read_text().replace('venturi_blocks_abi() != 13','venturi_blocks_abi() != 14');p.write_text(s)
p=Path('tests/test_residency.py');s=p.read_text().replace('[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]', '[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]');p.write_text(s)
p=Path('src/venturi/ops/build.py');s=p.read_text().replace('[source / "blocks.cu", source / "qwen_runtime.cuh"]','[source / "blocks.cu", source / "qwen_runtime.cuh", source / "qwen_pipeline.cuh"]').replace('("blocks.cu", "qwen_runtime.cuh")','("blocks.cu", "qwen_runtime.cuh", "qwen_pipeline.cuh")');p.write_text(s)
