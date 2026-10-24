"""Separate MTP1 timeline: diagnostics only, not serving throughput."""
import argparse, json
from pathlib import Path
import torch
from venturi.models import NativeQwenModel, NativeQwenMTPModel, QwenQ8Weights, QwenMTPWeights
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--library',required=True);p.add_argument('--case',required=True);p.add_argument('--output',required=True);a=p.parse_args()
r=Path('/home/data/wangyue/repos/codex-build/Venturi');case=json.loads(Path(a.case).read_text());torch.set_num_threads(4)
target=NativeQwenModel.from_q8(QwenQ8Weights.from_ninfer(NInferArtifact(r/'models/qwen3.5-27b-text-q8.ninfer')),CudaBackend(a.library),weight_mode='resident',dtype=torch.float16,device='cuda:0',max_tokens=32,tensor_parallel_device='cuda:1',tensor_parallel_transport='host',projection_profile='q8-residual')
model=NativeQwenMTPModel.from_q8(target,QwenMTPWeights.from_ninfer(NInferArtifact(r/'models/qwen3.5-27b-mtp-q8.ninfer')))
session=model.create_session(256)
try:
 expected=case['expected_generated_ids'];prompt=case['prompt_token_ids'];assert session.generate(prompt,max_new_tokens=len(expected))==expected;session.reset()
 with torch.cuda.nvtx.range('native_unsplit'):
  actual=session.generate(prompt,max_new_tokens=len(expected))
 assert actual==expected
 Path(a.output).write_text(json.dumps(dict(scope=__doc__,profiled=True,logical_requests=1,mtp_depth=1,projection_profile='q8-residual',draft_projection_profile='fp16',output_ids=actual,mtp_stats=session.stats()),indent=2)+'\n')
finally:session.close();model.close();target.close()
