import ctypes as C,json,os,tempfile,pathlib,torch
from test_qwen_mtp import mtp_fixture
from venturi.models import NativeQwenModel,NativeQwenMTPModel
from venturi.ops import CudaBackend
library='/home/data/wangyue/artifacts/venturi-mtp-draft-residual-fixtures-v1/build/089e914d53d145f0/libventuri_blocks.so'
records=[]
with tempfile.TemporaryDirectory(prefix='venturi-residual-draft-guards-') as temp:
 for kind in ['metadata-scalar','metadata-cohort','source-scalar','source-cohort','native-source-scalar']:
  case_dir=pathlib.Path(temp)/kind;case_dir.mkdir();a,b,_,_=mtp_fixture(case_dir)
  backend=CudaBackend(library)
  target=NativeQwenModel.from_q8(a,backend,weight_mode='resident',dtype=torch.float16,device='cuda:0',max_tokens=8,reserve_bytes=0,tensor_parallel_device='cuda:1',tensor_parallel_transport='host',projection_profile='q8-residual-column-mlp')
  model=NativeQwenMTPModel.from_q8(target,b,reserve_bytes=0,draft_projection_profile='q8-residual')
  session=model.create_batch([24]*2) if 'cohort' in kind else model.create_session(24)
  try:
   if kind.startswith('metadata'):
    original=model._mtp_weights.plan.config['rms_norm_eps'];model._mtp_weights.plan.config['rms_norm_eps']=original*1.01
   else:
    p=model._mtp_weights.artifact.files[0].path;stamp=p.stat();os.utime(p,ns=(stamp.st_atime_ns,stamp.st_mtime_ns+1000000))
   if kind=='native-source-scalar':
    ids=(C.c_int64*3)(1,2,3);out=(C.c_int64*4)();count=C.c_int()
    rc=model._native.venturi_qwen_generate2_prefix(session._handle,ids,3,4,None,0,out,C.byref(count))
    assert rc!=0
    error=backend._native.venturi_blocks_error().decode();assert 'source' in error.lower() and ('changed' in error.lower() or 'replaced' in error.lower())
    records.append(dict(case=kind,rejected=True,native_guard_without_sdk=True,error=error))
   else:
    entered=[]
    def forbidden(*args):entered.append(True);raise AssertionError('Native generation reached after invalid metadata/source')
    owner=session._native if 'cohort' in kind else model._native
    attr='generate' if 'cohort' in kind else 'venturi_qwen_generate2_prefix';saved=getattr(owner,attr);setattr(owner,attr,forbidden)
    try:
     try:
      if 'cohort' in kind:session.generate([[1,2],[3,4]],max_new_tokens=4)
      else:session.generate([1,2,3],max_new_tokens=4,draft_tokens=2,recovery='prefix')
     except RuntimeError as e:assert 'metadata/source changed' in str(e);error=str(e)
     else:raise AssertionError('Changed source or metadata accepted')
     assert not entered
     records.append(dict(case=kind,rejected=True,before_native_generation=True,error=error))
    finally:setattr(owner,attr,saved)
   if kind.startswith('metadata'):
    model._mtp_weights.plan.config['rms_norm_eps']=original
    info=session.info();positions=[r['position'] for r in info['requests']] if 'cohort' in kind else [info['position']];assert not any(positions)
  finally:session.close();model.close();target.close()
pathlib.Path('/home/data/wangyue/artifacts/venturi-mtp-draft-residual-guards-v6/report.json').write_text(json.dumps(dict(completed=True,checks=records),indent=2)+chr(10))
