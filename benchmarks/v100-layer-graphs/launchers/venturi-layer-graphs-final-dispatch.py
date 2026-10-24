from pathlib import Path
import runpy
s=Path('/tmp/venturi-token-order-final-v7-worker.py').read_text().replace('token-order-src-v2','layer-graphs-src-v3').replace('venturi-token-order-final-v7','venturi-layer-graphs-final-v6').replace('venturi-token-order-campaign-v6','venturi-layer-graphs-campaign-v5').replace('venturi-token-order-study-v4','venturi-layer-graphs-fixtures-v3').replace('deadline=time.monotonic()+1500','deadline=time.monotonic()+3600').replace('all nine dependency hashes','all ten dependency hashes')
a=s.index('  for name,cmd in [');b=s.index("   prefix=out/name;cmd+=",a)
s=s[:a]+'''  for name,cmd in [
   ('mtp2-eager-c1',[py,'tools/profile_qwen_mtp.py',str(target),'--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--library',library,'--case',str(out/'profile-case.json'),'--depth','2','--recovery','prefix','--count','1']),
   ('mtp2-graph-c1',[py,'tools/profile_qwen_mtp.py',str(target),'--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--library',library,'--case',str(out/'profile-case.json'),'--depth','2','--recovery','prefix','--count','1','--target-layer-graph']),
   ('mtp2-graph-c8',[py,'tools/profile_qwen_mtp.py',str(target),'--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--library',library,'--case',str(out/'profile-case.json'),'--depth','2','--recovery','prefix','--count','8','--target-layer-graph'])]:
''' +s[b:]
s=s.replace("'--cuda-event-trace=false','--force-overwrite=false'", "'--cuda-event-trace=false','--cuda-graph-trace=node','--force-overwrite=false'").replace('timeout=300)','timeout=600)').replace('timeout=120)','timeout=240)').replace('timeout=90)','timeout=240)')
runpy.run_path('/tmp/venturi-dispatch-helper.py')['dispatch']('venturi-layer-graphs-final-v6',s)
