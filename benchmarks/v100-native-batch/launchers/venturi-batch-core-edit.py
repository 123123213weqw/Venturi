from pathlib import Path
root=Path('/Users/wangyue/Documents/Codex/2026-10-15/wo/work/Venturi')
p=root/'src/venturi/ops/csrc/qwen_runtime.cuh';s=p.read_text();s=s.replace('static std::uint64_t estimate(const Model& m,int capacity) {','static std::uint64_t estimate(const Model& m,int capacity,bool caches=true) {',1);s=s.replace('for (int i=m.begin;i<m.end;++i) {const auto& l=m.layers[i];n=add(n,l.kind==0 ?', 'for (int i=m.begin;caches && i<m.end;++i) {const auto& l=m.layers[i];n=add(n,l.kind==0 ?',1);s=s.replace('explicit Session(std::shared_ptr<Model> m,int cap):model(std::move(m)),capacity(cap) {','explicit Session(std::shared_ptr<Model> m,int cap,bool caches=true):model(std::move(m)),capacity(cap) {',1);s=s.replace('md.guard(estimate(md,cap));','md.guard(estimate(md,cap,caches));',1);s=s.replace('for (int i=md.begin;i<md.end;++i) {\n            if (md.layers[i].kind==0)', 'for (int i=md.begin;caches && i<md.end;++i) {\n            if (md.layers[i].kind==0)',1);s=s.replace('device_bytes==estimate(md,cap),','device_bytes==estimate(md,cap,caches),',1);p.write_text(s)
p=root/'src/venturi/ops/csrc/blocks.cu';s=p.read_text().replace('venturi_blocks_abi() { return 15; }','venturi_blocks_abi() { return 16; }').replace('#include "qwen_pipeline.cuh"','#include "qwen_batch.cuh"\n\n#include "qwen_pipeline.cuh"',1);p.write_text(s)
p=root/'src/venturi/ops/blocks.py';p.write_text(p.read_text().replace('venturi_blocks_abi() != 15','venturi_blocks_abi() != 16',1))
p=root/'src/venturi/ops/build.py';s=p.read_text().replace('source / "qwen_mtp.cuh",','source / "qwen_mtp.cuh",\n            source / "qwen_batch.cuh",',1).replace('                "qwen_mtp.cuh",','                "qwen_mtp.cuh",\n                "qwen_batch.cuh",',1);p.write_text(s)
p=root/'tests/test_residency.py';p.write_text(p.read_text().replace('[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]', '[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]',1))
p=root/'src/venturi/models/qwen_native.py';s=p.read_text();point='    def close(self):\n        with self._lock:\n            self._finalizer()';assert s.count(point)>=1;s=s.replace(point,'''    def create_batch(self, capacities):
        """Explicit grouped target execution with independent request caches."""
        from .qwen_batch import NativeQwenBatchSession

        return NativeQwenBatchSession.create(self, capacities)

'''+point,1);p.write_text(s)
p=root/'src/venturi/models/__init__.py';s=p.read_text().replace('__all__ = [','__all__ = [\n    "NativeQwenBatchSession",',1);s+='\nfrom .qwen_batch import NativeQwenBatchSession\n';p.write_text(s)
