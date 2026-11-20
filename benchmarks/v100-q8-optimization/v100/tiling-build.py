import json,pathlib
from venturi.ops.build import build_blocks_backend
p=pathlib.Path('/home/wzu/codex-build/Venturi/v100-resident-optimization-v1')
r=[]
for t,row in [(4,1),(4,4),(8,4),(2,4)]:
 lib=build_blocks_backend('/home/wzu/codex-build/Venturi/v100-q8-tiling-build',architecture='sm70',q8_token_tile=t,q8_row_tile=row)
 r.append(dict(token_tile=t,row_tile=row,library=str(lib),manifest=json.load(lib.with_name('build.json').open())))
 (p/'tiling-libraries.json').write_text(json.dumps(r,indent=2)+'\n')
 print(t,row,lib,flush=True)
