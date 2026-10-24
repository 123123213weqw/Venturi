import pathlib,json
from venturi.ops.build import build_blocks_backend
out=pathlib.Path('/home/wzu/codex-build/Venturi/v100-warp-experiment-v1');r=[]
for mode in (1,2):
 lib=build_blocks_backend('/home/wzu/codex-build/Venturi/v100-warp-build',architecture='sm70',q8_token_tile=4,q8_row_tile=4,q8_decode_mode=mode)
 r.append(dict(name='warp'+str(mode),token_tile=4,row_tile=4,decode_mode=mode,library=str(lib),manifest=json.load(lib.with_name('build.json').open())))
 (out/'libraries.json').write_text(json.dumps(r,indent=2))
 print(mode,lib,flush=True)
