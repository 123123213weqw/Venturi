from pathlib import Path
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns)
s=Path('/tmp/venturi-layer-graphs-final-v6-worker.py').read_text()
s=s.replace('layer-graphs-src-v3','native-column-mlp-src-v4').replace('venturi-layer-graphs-final-v6','venturi-native-column-mlp-profile-v7').replace('venturi-layer-graphs-campaign-v5','venturi-native-column-mlp-campaign-v6').replace('venturi-layer-graphs-fixtures-v3','venturi-native-column-mlp-fixtures-v1')
s=s.replace("if campaign['operator_promoted']:","if campaign['model_qualified']:")
a=s.index('  for name,cmd in [');b=s.index('   prefix=out/name;',a)
block=r'''  for name,cmd in [
   ('same-library-row-mtp2-c1',[py,'tools/profile_qwen_mtp.py',str(target),'--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--library',library,'--case',str(out/'profile-case.json'),'--depth','2','--recovery','prefix','--count','1','--projection-profile','q8-residual']),
   ('column-mtp2-c1',[py,'tools/profile_qwen_mtp.py',str(target),'--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--library',library,'--case',str(out/'profile-case.json'),'--depth','2','--recovery','prefix','--count','1','--projection-profile','q8-residual-column-mlp']),
   ('column-ordinary256-c8',[py,'tools/profile_qwen_batch.py',str(target),'--library',library,'--case',str(out/'profile-case.json'),'--count','8','--projection-rows','256','--projection-profile','q8-residual-column-mlp']),
   ('column-mtp-batch-c8',[py,'tools/profile_qwen_mtp_batch.py',str(target),'--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--library',library,'--case',str(out/'profile-case.json'),'--count','8','--projection-profile','q8-residual-column-mlp'])]:
'''
s=s[:a]+block+s[b:]
s=s.replace("scope='Exact native library and nine dependency hashes;", "scope='Exact native library and ten dependency hashes;")
ns['dispatch']('venturi-native-column-mlp-profile-v7',s)
