from pathlib import Path
import hashlib,json,runpy
root=Path.cwd();cap=root/'benchmarks/v100-layer-graphs';manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (root/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts};(cap/'source-manifest-v1.json').write_text(json.dumps(manifest,indent=2)+'\n')
s=Path('/tmp/venturi-activation-words-study-v1-worker.py').read_text().replace('activation-words-src-v1','layer-graphs-src-v1').replace('venturi-activation-words-study-v1','venturi-layer-graphs-fixtures-v1').replace('venturi-token-order-final-v7','venturi-token16-graph-probe-v9').replace('tests/test_q8_activation_words.py','tests/test_qwen_layer_graph.py')
s=s.replace(" run([py,'-m','venturi.ops.build'", " expected="+repr(manifest)+"\n actual={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}\n assert actual==expected,'Full source manifest mismatch before build'\n (out/'pre-build-source.json').write_text(json.dumps(actual,indent=2)+'\\n')\n run([py,'-m','venturi.ops.build'",1)
s=s.replace(" env.pop('VENTURI_TILE_ORDER_LIBRARY',None)", " env.pop('VENTURI_TILE_ORDER_LIBRARY',None);env.pop('VENTURI_T16_LIBRARY',None)\n env['VENTURI_LAYER_GRAPH_LIBRARY']=library")
a=s.index(" run([py,'-u','tools/benchmark_q8_activation_words.py'");b=s.index(" (out/'counter-policy.txt')",a);s=s[:a]+s[b:]
runpy.run_path('/tmp/venturi-dispatch-helper.py')['dispatch']('venturi-layer-graphs-fixtures-v1',s)
