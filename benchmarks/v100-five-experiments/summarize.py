"""Audit raw matched native experiments; never score a failed/profiled run."""
import csv,hashlib,json,sqlite3,statistics,sys
from collections import defaultdict
from pathlib import Path
R=Path(__file__).resolve().parent
D=R/'evidence'/'window'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def summarize(root=D):
 b=read(root/'build-record.json');assert all('-DVENTURI_Q8_'+f+'=1' in b['command'] for f in ['ROW_PAIR','SWIGLU_QUANT','RMS_QUANT'])
 masks=[0]+read(root/'qualified-masks.json');baseline=read(root/'qualify-0/report.json');assert baseline['completed']
 for mask in masks[1:]:
  q=read(root/f'qualify-{mask}/report.json');assert q['completed'] and q['exact_reference_bytes_equal'] and q['comparison_payload']==baseline['comparison_payload']
 variants={}
 reports={}
 for mask in masks:
  reports[mask]=[read(root/f'timing-{i}-{mask}/report.json') for i in range(3)]
  for i,r in enumerate(reports[mask]):
   assert r['completed'] and not r['profiled'] and r['args']['mask']==mask and r['args']['trial']==i
   for k in ['artifact_id','config','prompt_token_ids','projection_profile','new_tokens','library_sha256']:assert r[k]==baseline[k],k
  variants[mask]={}
  for count in [1,8]:
   cells=[next(x for x in r['trials'] if x['count']==count) for r in reports[mask]]
   for c in cells:
    expected=baseline['comparison_payload']['scalar'][0]['tokens'] if count==1 else baseline['comparison_payload']['cohort']['tokens']
    assert c['output_ids']==expected
    for scope,tokens in [('decode',31*count),('complete',32*count)]:
     x=c[scope];assert x['output_tokens']==tokens and abs(x['tps']-tokens/x['wall_seconds'])<1e-9
   variants[mask][count]={}
   for scope in ['decode','complete']:
    wall=sum(c[scope]['wall_seconds'] for c in cells);tokens=sum(c[scope]['output_tokens'] for c in cells)
    value=dict(aggregate_tps=tokens/wall,mean_wall_seconds=wall/3,mean_cpu_seconds=statistics.mean(c[scope]['process_cpu_seconds'] for c in cells),cpu_cores=sum(c[scope]['process_cpu_seconds'] for c in cells)/wall,trial_tps=[c[scope]['tps'] for c in cells])
    if mask:
     controls=[next(c for c in r['trials'] if c['count']==count)[scope] for r in reports[0]]
     value['gain_percent']=(value['aggregate_tps']/variants[0][count][scope]['aggregate_tps']-1)*100
     value['paired_gains_percent']=[(x[scope]['tps']/y['tps']-1)*100 for x,y in zip(cells,controls)]
    variants[mask][count][scope]=value
   variants[mask][count]['mean_ttft_seconds']=statistics.mean(c['ttft']['wall_seconds'] for c in cells)
 out=dict(variants=variants,scopes='post-first decode vs complete fresh cohort, aggregate tokens/global wall; three balanced-order repeats',correctness='Each qualified mask exactly matches baseline teacher logits, per-layer hidden trace, private GDN/KV state and 32-token trajectories on this case only',library_sha256=baseline['library_sha256'],artifact_id=baseline['artifact_id'])
 mtp=root/'mtp/rows.json'
 if mtp.exists():
  rows=read(mtp);stats=[]
  for count in [1,8]:
   for phase,depths in [('timed-fixed',([0,1,2] if count==1 else [0,1])),('timed-auto',[None])]:
    for depth in depths:
     rs=[r for r in rows if r['count']==count and r['phase']==phase and r['requested_depth']==depth]
     if len(rs)!=3:continue
     assert all(r['exact_target_state'] for r in rs)
     wall=sum(r['wall_seconds'] for r in rs);proposed=sum(r['info']['proposed_tokens'] for r in rs);accepted=sum(r['info']['accepted_tokens'] for r in rs)
     stats.append(dict(count=count,mode='adaptive' if depth is None else depth,whole_call_tps=sum(r['emitted_tokens'] for r in rs)/wall,mean_wall_seconds=wall/3,cpu_cores=sum(r['process_cpu_seconds'] for r in rs)/wall,selected_depths=[r['selected_depth'] for r in rs],proposed=proposed,accepted=accepted,acceptance=accepted/proposed if proposed else None))
  out['mtp']=dict(completed=(root/'mtp/completed.json').exists(),results=stats,cold_exploration_seconds=sum(r['wall_seconds'] for r in rows if r['phase']=='cold-exploration'))
 return out
if __name__=='__main__':
 out=summarize(Path(sys.argv[1]) if len(sys.argv)>1 else D);(R/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
