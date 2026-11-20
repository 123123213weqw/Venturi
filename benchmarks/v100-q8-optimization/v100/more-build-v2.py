import json,pathlib
from venturi.ops.build import build_blocks_backend
p=pathlib.Path('/home/wzu/codex-build/Venturi/v100-resident-optimization-v1')
r=json.load((p/'tiling-libraries.json').open())
for row in (2,8):
 lib=build_blocks_backend('/home/wzu/codex-build/Venturi/v100-q8-tiling-build',architecture='sm70',q8_token_tile=4,q8_row_tile=row)
 r.append(dict(token_tile=4,row_tile=row,library=str(lib),manifest=json.load(lib.with_name('build.json').open())))
 with (p/'more-tiling-libraries-v2.json').open('w') as f:json.dump(r,f,indent=2)
 print(row,lib,flush=True)
