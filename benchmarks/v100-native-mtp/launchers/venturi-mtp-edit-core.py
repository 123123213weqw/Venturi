from pathlib import Path
root=Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi')
p=root/'src/venturi/ops/csrc/qwen_runtime.cuh';s=p.read_text()
s=s.replace('int first=0,int last=-1,bool staged=false):c(cfg)', 'int first=0,int last=-1,bool staged=false,const VRMatrix* mtp_input=nullptr):c(cfg)')
s=s.replace('        validate_matrix(embedding,c.vocab,c.hidden);validate_matrix(head,c.vocab,c.hidden);','        if (mtp_input) {\n            require(!stage && !parallel() && c.layers==1 && c.resident==1,"MTP base requires one resident unsharded layer");\n            validate_matrix(*mtp_input,c.hidden,2*c.hidden);\n        } else {validate_matrix(embedding,c.vocab,c.hidden);validate_matrix(head,c.vocab,c.hidden);}')
s=s.replace('void forward(const std::int64_t* tokens,int n,float* logits,bool all,void* trace) {','void forward(const std::int64_t* tokens,int n,float* logits,bool all,void* trace,void* hidden=nullptr) {')
needle='                if (all || start+t==n) output(x,t,all,logits?logits+(all?mul(start,c.vocab):0):nullptr);'
assert s.count(needle)==1
s=s.replace(needle,needle+'\n                else if (hidden) op(venturi_rms_run(x,m.norm,p("normalized"),t,c.hidden,c.dtype,c.epsilon,m.stream.ptr));\n                if (hidden) check(cudaMemcpyAsync(static_cast<char*>(hidden)+mul(mul(start,c.hidden),2),p("normalized"),mul(mul(t,c.hidden),2),cudaMemcpyDeviceToDevice,m.stream.ptr));')
p.write_text(s)
p=root/'src/venturi/ops/csrc/qwen_pipeline.cuh';s=p.read_text();s=s.replace('std::int64_t* selected=nullptr) {','std::int64_t* selected=nullptr,void* hidden=nullptr,void* batch_selection=nullptr,void* batch_scratch=nullptr,std::uint64_t batch_scratch_bytes=0) {',1)
s=s.replace('                    if (selected && start+t==n) {','                    else if (hidden) op(venturi_rms_run(x,model->second->norm,second->p("normalized"),t,c.hidden,c.dtype,c.epsilon,model->second->stream.ptr));\n                    if (hidden) check(cudaMemcpyAsync(static_cast<char*>(hidden)+mul(mul(start,c.hidden),2),second->p("normalized"),mul(mul(t,c.hidden),2),cudaMemcpyDeviceToDevice,model->second->stream.ptr));\n                    if (selected && (all || start+t==n)) {')
a='                        op(venturi_greedy_run(second->p("logits"),static_cast<std::int64_t*>(second->p("selection")),second->p("select_scratch"),Session::min_tiles(c.vocab)*16,1,c.vocab,c.dtype,model->second->stream.ptr));\n                        check(cudaMemcpyAsync(selected,second->p("selection"),16,cudaMemcpyDeviceToHost,model->second->stream.ptr));\n                        selection_bytes=add(selection_bytes,16);'
b='''                        const int rows=all?t:1;
                        require(rows==1 || (batch_selection && batch_scratch),"Batched pipeline selection requires explicit storage");
                        auto* dest=batch_selection?batch_selection:second->p("selection");
                        auto* scratch=batch_scratch?batch_scratch:second->p("select_scratch");
                        op(venturi_greedy_run(second->p("logits"),static_cast<std::int64_t*>(dest),scratch,batch_scratch?batch_scratch_bytes:Session::min_tiles(c.vocab)*16,rows,c.vocab,c.dtype,model->second->stream.ptr));
                        check(cudaMemcpyAsync(selected+(all?start*2:0),dest,mul(rows,16),cudaMemcpyDeviceToHost,model->second->stream.ptr));
                        selection_bytes=add(selection_bytes,mul(rows,16));'''
assert a in s;s=s.replace(a,b);p.write_text(s)
p=root/'src/venturi/ops/csrc/blocks.cu';s=p.read_text().replace('venturi_blocks_abi() { return 14; }','venturi_blocks_abi() { return 15; }');s+='\n#include "qwen_mtp.cuh"\n';p.write_text(s)
p=root/'src/venturi/ops/blocks.py';s=p.read_text().replace('venturi_blocks_abi() != 14','venturi_blocks_abi() != 15');p.write_text(s)
p=root/'src/venturi/ops/build.py';s=p.read_text().replace('source / "qwen_pipeline.cuh"]','source / "qwen_pipeline.cuh", source / "qwen_mtp.cuh"]').replace('("blocks.cu", "qwen_runtime.cuh", "qwen_pipeline.cuh")','("blocks.cu", "qwen_runtime.cuh", "qwen_pipeline.cuh", "qwen_mtp.cuh")');p.write_text(s)
