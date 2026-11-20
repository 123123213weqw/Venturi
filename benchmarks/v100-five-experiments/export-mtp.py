import hashlib,json,time,torch
from pathlib import Path
from venturi.qwen_export import export_qwen_mtp_q8
O=Path('/home/data/wangyue/artifacts/venturi-five-mtp-export-v1');O.mkdir(exist_ok=False)
torch.set_num_threads(4)
a=Path('/home/data/wangyue/repos/codex-build/Venturi/models/qwen3.8-27b-mtp-q8-v1.ninfer')
r=export_qwen_mtp_q8('/home/wzu/models/Qwen3.8-27B',a,source_manifest='/home/data/wangyue/artifacts/venturi-rms-native-qwen38-export-v2/source-evidence.json',rows_per_chunk=256)
r['sha256']=hashlib.sha256(a.read_bytes()).hexdigest()
(O/'export.json').write_text(json.dumps(r,indent=2)+'\n')
(O/'completed.json').write_text(json.dumps(dict(completed=True,artifact=str(a),sha256=r['sha256'],unix=time.time()))+'\n')
