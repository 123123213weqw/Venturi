from pathlib import Path
r=Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi')
p=r/'src/venturi/ops/csrc/qwen_runtime.cuh';s=p.read_text().replace('#include <cstring>','#include <cstring>\n#include <chrono>')
s=s.replace('namespace vr {','namespace vr {\ninline double native_clock() {return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}',1)
s=s.replace('    int capacity,position=0; bool failed=false;','    int capacity,position=0; bool failed=false;\n    double first_token_seconds=0,generation_seconds=0;')
a='extern "C" int venturi_qwen_generate(void* h,const std::int64_t* prompt,int n,int maximum,const std::int64_t* eos,int eos_count,std::int64_t* tokens,int* count) {\n'
assert a in s;s=s.replace(a,a+'    const double begun=vr::native_clock();\n')
a='        std::lock_guard<std::mutex> a(s.mutex),b(m.mutex);\n        vr::require(maximum>0'
assert a in s;s=s.replace(a,'        std::lock_guard<std::mutex> a(s.mutex),b(m.mutex);\n        s.first_token_seconds=s.generation_seconds=0;\n        vr::require(maximum>0',1)
s=s.replace('                tokens[step]=selected[0];*count=step+1;','                tokens[step]=selected[0];*count=step+1;\n                if(step==0)s.first_token_seconds=vr::native_clock()-begun;',1)
s=s.replace('            m.source();return 0;\n        } catch (...) { s.failed=true;throw; }','            m.source();s.generation_seconds=vr::native_clock()-begun;return 0;\n        } catch (...) { s.failed=true;throw; }',1)
s+='\nextern "C" int venturi_qwen_timings(void* h,double* out) {try{vr::require(h && out,"null native Qwen timings");auto& s=*static_cast<vr::Session*>(h);std::lock_guard<std::mutex> lock(s.mutex);out[0]=s.first_token_seconds;out[1]=s.generation_seconds;return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}\n'
p.write_text(s)
p=r/'src/venturi/ops/csrc/qwen_pipeline.cuh';s=p.read_text().replace('    std::uint64_t activation_bytes=0,selection_bytes=0,handoffs=0;','    std::uint64_t activation_bytes=0,selection_bytes=0,handoffs=0;\n    double first_token_seconds=0,generation_seconds=0;')
a='extern "C" int venturi_qwen_pipeline_generate(void* h,const std::int64_t* prompt,int n,int maximum,const std::int64_t* eos,int eos_count,std::int64_t* tokens,int* count) {\n';assert a in s;s=s.replace(a,a+'    const double begun=vr::native_clock();\n')
# Keep validation and existing numerical/control operations unchanged.
a='        const auto& m=*s.model->first;';assert a in s;s=s.replace(a,a+'s.first_token_seconds=s.generation_seconds=0;',1)
s=s.replace('                tokens[step]=selected[0];*count=step+1;','                tokens[step]=selected[0];*count=step+1;\n                if(step==0)s.first_token_seconds=vr::native_clock()-begun;',1)
s=s.replace('            return 0;\n        } catch (...) {s.poison();throw;}','            s.generation_seconds=vr::native_clock()-begun;return 0;\n        } catch (...) {s.poison();throw;}',1)
s+='\nextern "C" int venturi_qwen_pipeline_timings(void* h,double* out){try{vr::require(h && out,"null pipeline timings");auto& s=*static_cast<vr::PipelineSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);out[0]=s.first_token_seconds;out[1]=s.generation_seconds;return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}\n';p.write_text(s)
p=r/'src/venturi/ops/csrc/qwen_mtp.cuh';s=p.read_text();s=s.replace('double draft_seconds=0,verify_seconds=0,rollback_seconds=0,prefill_seconds=0;','double draft_seconds=0,verify_seconds=0,rollback_seconds=0,prefill_seconds=0,checkpoint_seconds=0,first_token_seconds=0,generation_seconds=0;')
s=s.replace('draft_seconds=verify_seconds=rollback_seconds=prefill_seconds=0;','draft_seconds=verify_seconds=rollback_seconds=prefill_seconds=checkpoint_seconds=first_token_seconds=generation_seconds=0;')
s=s.replace('        const auto& c=model->base->c;require(!failed()', '        const double begun=mtp_clock();first_token_seconds=generation_seconds=0;\n        const auto& c=model->base->c;require(!failed()',1)
s=s.replace('prefill_seconds+=mtp_clock()-start;emit(token(selected,0));','prefill_seconds+=mtp_clock()-start;emit(token(selected,0));first_token_seconds=mtp_clock()-begun;')
s=s.replace('++proposed;checkpoint(true);std::int64_t inputs[]=', '++proposed;start=mtp_clock();checkpoint(true);checkpoint_seconds+=mtp_clock()-start;std::int64_t inputs[]=')
s=s.replace('            model->source();\n        } catch (...) {settle_failure();throw;}','            model->source();generation_seconds=mtp_clock()-begun;\n        } catch (...) {settle_failure();throw;}')
s=s.replace('s.rollback_seconds,s.prefill_seconds};std::copy(b,b+4,seconds);','s.rollback_seconds,s.prefill_seconds,s.checkpoint_seconds};std::copy(b,b+5,seconds);')
s+='\nextern "C" int venturi_qwen_mtp_timings(void* h,double* out){try{vr::require(h && out,"null MTP timings");auto& s=*static_cast<vr::MTPSession*>(h);std::lock_guard<std::mutex> lock(s.mutex);out[0]=s.first_token_seconds;out[1]=s.generation_seconds;return 0;}catch(const std::exception& e){error_text=e.what();return -1;}}\n';p.write_text(s)
p=r/'src/venturi/models/qwen_native.py';s=p.read_text().replace('        "venturi_qwen_info": ([ptr, C.POINTER(u64)], integer),','        "venturi_qwen_info": ([ptr, C.POINTER(u64)], integer),\n        "venturi_qwen_timings": ([ptr, C.POINTER(C.c_double)], integer),')
needle='    @property\n    def position(self):';assert needle in s
s=s.replace(needle,'''    def generation_timings(self):
        """Native-call timestamps, including model queues but excluding Python/load."""
        with self._lock:
            self._check()
            values = (C.c_double * 2)()
            self.model._backend._check(self.model._native.venturi_qwen_timings(self._handle, values))
            return dict(native_time_to_first_token_seconds=values[0], native_generation_seconds=values[1])

'''+needle,1);p.write_text(s)
p=r/'src/venturi/models/qwen_pipeline.py';s=p.read_text().replace('        "info": ([ptr, C.POINTER(u64)], integer),','        "info": ([ptr, C.POINTER(u64)], integer),\n        "timings": ([ptr, C.POINTER(C.c_double)], integer),');p.write_text(s)
p=r/'src/venturi/models/qwen_mtp_native.py';s=p.read_text().replace('        "info": ([p, C.POINTER(u)], i),','        "info": ([p, C.POINTER(u)], i),\n        "timings": ([p, C.POINTER(C.c_double)], i),').replace('(C.c_double * 4)()', '(C.c_double * 5)()').replace('"prefill_wall_seconds",','"prefill_wall_seconds",\n                        "checkpoint_save_wall_seconds",')
p.write_text(s)
