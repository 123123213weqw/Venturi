from pathlib import Path
import runpy
s=Path('/tmp/venturi-token-order-model-v5-worker.py').read_text().replace('token-order-src-v2','layer-graphs-src-v3').replace('venturi-token-order-model-v5','venturi-layer-graphs-model-v4').replace('venturi-token-order-study-v4','venturi-layer-graphs-fixtures-v3')
a=s.index(' op=json.loads');b=s.index(" library=(gate/",a)
s=s[:a]+''' promoted=True
 (out/'operator-promotion.json').write_text(json.dumps(dict(promoted=True,rule='No microkernel score substituted: native graph fixtures, full source/build identity and trained gates precede a matched full-model screen.'),indent=2)+'\\n')
''' +s[b:]
s=s.replace("if promoted:\n  target", "if promoted:\n  target")
a=s.index("  run([py,'-u','tools/qualify_qwen_mtp_batch.py'");b=s.index("  run([py,'-u','tools/qualify_qwen_mtp.py'",a)
s=s[:a]+'''  run([py,'-u','tools/qualify_qwen_layer_graph.py',str(target),'--mtp-artifact',str(root/'repos/codex-build/Venturi/models/qwen3.5-27b-mtp-q8.ninfer'),'--library',library,'--cases',str(cases),'--prior-target-report',str(prior),'--prior-target-report-sha256','a281b7b4fc5528d0c46eb1f2506f1f59f75c6f4c806800e8ffd765e051d4e275','--output-directory',str(out/'graph-qualification')],'graph-qualification.log',timeout=900)
''' +s[b:]
s=s.replace("'--draft-tokens','2','--recovery','prefix'", "'--draft-tokens','2','--recovery','prefix','--target-layer-graph'")
runpy.run_path('/tmp/venturi-dispatch-helper.py')['dispatch']('venturi-layer-graphs-model-v4',s)
