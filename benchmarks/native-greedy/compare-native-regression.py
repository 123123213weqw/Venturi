import hashlib,json
from pathlib import Path
import torch
base=Path('/home/wzu/codex-build/Venturi')
a=base/'streaming-validation/full-model-run1'; b=base/'greedy-validation-v3/full-model-run1'
old=json.loads((a/'report.json').read_text()); new=json.loads((b/'report.json').read_text())
assert old['artifact_sha256']==new['artifact_sha256'] and old['case']==new['case'] and old['dtype']==new['dtype']
assert new['native_full_forward_completed'] and new['native_greedy_selector']
assert old['native_generated_ids']==new['native_generated_ids']==[57590,248046]
report=dict(scope='Same trained-model native outputs/states: ABI10 host-greedy vs ABI11 native-greedy. This is regression parity, not original-BF16/corpus quality or a speed comparison.',old_report_sha256=hashlib.sha256((a/'report.json').read_bytes()).hexdigest(),steps=[])
def same(x,y):
 assert x.shape==y.shape and x.dtype==y.dtype
 return torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))
for step in range(2):
 x=torch.load(a/f'native-step-{step}.pt',map_location='cpu',weights_only=True);y=torch.load(b/f'native-step-{step}.pt',map_location='cpu',weights_only=True)
 assert x.keys()==y.keys() and len(x['trace'])==len(y['trace'])==64
 checks=[same(x['logits'],y['logits'])]
 for u,v in zip(x['trace'],y['trace']):
  assert (u['layer'],u['position'])==(v['layer'],v['position']);checks.append(same(u['hidden'],v['hidden']))
 states=0
 for kind in ('gdn','kv'):
  assert len(x['state'][kind])==len(y['state'][kind])==64
  for u,v in zip(x['state'][kind],y['state'][kind]):
   assert (u is None)==(v is None)
   if u is not None:
    assert len(u)==len(v)==2
    for s,t in zip(u,v):checks.append(same(s,t));states+=1
 report['steps'].append(dict(step=step,logits_bitwise_equal=checks[0],hidden_tensors=64,state_tensors=states,all_tensor_bytes_equal=all(checks)))
 assert all(checks),'Native regression differs; inspect before accepting'
report['passed']=True
(base/'greedy-validation-v3/native-abi10-regression.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
