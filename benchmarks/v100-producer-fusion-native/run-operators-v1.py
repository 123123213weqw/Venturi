"""Correctness-first operator campaign after native OFF/ON build gates; production unchanged."""
import csv,hashlib,json,os,signal,statistics,subprocess,time
from pathlib import Path
# Same audit and bounded command runner as build/fixture stage.
bootstrap=Path('/home/data/wangyue/repos/codex-build/Venturi/producer-fusion-native-src-v4/build-and-test-v1.py').read_text()
exec(bootstrap[:bootstrap.index('out.mkdir(parents=True,exist_ok=False)')])
root=out
out=root/'operators';out.mkdir(exist_ok=False)
try:
 before=audit('production-before.json')
 for _ in range(450):
  if (root/'completed.json').exists():break
  if (root/'production-after.json').exists():raise RuntimeError('Build/fixture stage ended without success')
  time.sleep(2)
 else:raise TimeoutError('Build/fixture stage unfinished')
 libs=json.loads((root/'libraries.json').read_text())
 for name in ['off','on']:
  lib=Path(libs[name]);record=json.loads(lib.with_name('build.json').read_text())
  assert hashlib.sha256(lib.read_bytes()).hexdigest()==record['library_sha256']
  for file,h in record['source_sha256'].items():assert hashlib.sha256((source/'src/venturi/ops/csrc'/file).read_bytes()).hexdigest()==h
 save('executed-source-sha256.json',{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src','tests','tools'] for p in (source/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'})
 env['CUDA_VISIBLE_DEVICES']='0'
 gguf='/home/wzu/models/Qwen3.8-27B-Uncensored-GGUF/Qwen3.8-27B-Uncensored-Q8_0.gguf'
 common=['--gguf',gguf,'--control-library',libs['off'],'--repeats','7']
 idle();run('native-screen',[python,'tools/projection/benchmark_native_swiglu_quant.py','--library',libs['on'],*common,'--output-directory',str(out/'native-screen')],300)
 idle();run('native-confirm',[python,'tools/projection/benchmark_native_swiglu_quant.py','--library',libs['on'],*common,'--tokens','1,8','--layers','0','--output-directory',str(out/'native-confirm')],120)
 # The independent extra operator is never selected in the native/model paths.
 rms=out/'librms_quant_probe.so'
 run('rms-build',['/usr/local/cuda/bin/nvcc','-O3','-std=c++20','-lineinfo','-shared','--cudart=static','-Xcompiler=-fPIC','-DVENTURI_Q8_ROW_PAIR=1','-gencode=arch=compute_70,code=sm_70','-gencode=arch=compute_70,code=compute_70','-Xptxas=-v','-I'+str(source/'src/venturi/ops/csrc'),str(source/'tools/projection/rms_quant_probe.cu'),'-o',str(rms)],120)
 env.update(VENTURI_RMS_QUANT_LIBRARY=str(rms),VENTURI_RMS_QUANT_CONTROL=libs['off'])
 idle();run('rms-tests',[python,'-m','pytest','-q','tests/test_rms_quant.py'],90)
 idle();run('rms-screen',[python,'tools/projection/benchmark_rms_quant.py','--library',str(rms),*common,'--output-directory',str(out/'rms-screen')],300)
 report=json.loads((out/'rms-screen/report.json').read_text())
 gains={name:statistics.median(1-c['median_ms']['producer'][name]/c['median_ms']['producer']['separate'] for c in report['cases']) for name in ['chunk8','chunk32','chunk128']}
 best=max(gains,key=gains.get);save('rms-screen-selection.json',dict(best_by_screen_producer_median=best,gains=gains,selection_only=True,not_a_promotion=True))
 idle();run('rms-confirm',[python,'tools/projection/benchmark_rms_quant.py','--library',str(rms),*common,'--tokens','1,8','--layers','0','--output-directory',str(out/'rms-confirm')],120)
 metrics=','.join(['gpu__time_duration.sum','sm__throughput.avg.pct_of_peak_sustained_elapsed',
 'dram__bytes_read.sum','dram__bytes_write.sum','dram__throughput.avg.pct_of_peak_sustained_elapsed',
 'sm__warps_active.avg.pct_of_peak_sustained_active','smsp__issue_active.avg.pct_of_peak_sustained_active',
 'l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum','l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum',
 'lts__t_sectors_op_read.sum','lts__t_sectors_op_write.sum',
 'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
 'smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct'])
 for label,tool,lib,arms in [('native','benchmark_native_swiglu_quant.py',libs['on'],['native-off','native-on']),('rms','benchmark_rms_quant.py',str(rms),['separate',best])]:
  for arm in arms:
   idle();cmd=['sudo','-n','/usr/bin/timeout','100','/usr/bin/env']
   cmd += [k+'='+env[k] for k in ['CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONPATH']]
   cmd += ['/usr/local/cuda/bin/ncu','--clock-control','none','--cache-control','none','--target-processes','all','--profile-from-start','off','--kernel-name-base','demangled','--launch-count','2' if arm in ['native-off','separate'] else '1','--metrics',metrics,'--csv','--page','raw','--disable-extra-suffixes','--log-file',str(out/f'counter-{label}-{arm}.csv')]
   cmd += [python,'tools/projection/'+tool,'--library',lib,*common,'--tokens','8','--layers','0','--profile-arm',arm,'--output-directory',str(out/f'profile-{label}-{arm}')]
   run('counter-'+label+'-'+arm,cmd,115)
 save('completed.json',dict(completed=True,production_deployed=False,full_model_tps_measured=False,ended_unix=time.time()))
finally:
 after=audit('production-after.json')
 if 'before' in globals():assert all(before[k]==after[k] for k in ['container_id','started_at','restart_count','canonical_config_fingerprint','counter_policy'])
