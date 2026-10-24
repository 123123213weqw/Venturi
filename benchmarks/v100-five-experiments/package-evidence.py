"""Freeze completed/restored evidence; retain large profiler originals by hash."""
import hashlib,json,tarfile,time
from pathlib import Path
BASE=Path('/home/data/wangyue/artifacts');ROOT=BASE/'venturi-five-window-v4'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
assert read(ROOT/'completed.json')['completed'] and read(ROOT/'production-restored.json')['healthy'] and read(ROOT/'production-restored.json')['config_unchanged']
a=read(ROOT/'qualify-0/report.json');b=read(BASE/'venturi-rms-model-window-v4/qualify-on/report.json')
comparison=dict(previous_report_sha256=sha(BASE/'venturi-rms-model-window-v4/qualify-on/report.json'),current_report_sha256=sha(ROOT/'qualify-0/report.json'),exact_previous_baseline_payload_equal=a['comparison_payload']==b['comparison_payload'])
(ROOT/'baseline-previous-equality.json').write_text(json.dumps(comparison,indent=2)+'\n')
sources={'window':ROOT,'build-v2':BASE/'venturi-five-v2','build-v1':BASE/'venturi-five-v1','mtp-export':BASE/'venturi-five-mtp-export-v1','preflight-v1':BASE/'venturi-five-window-v1','preflight-v2':BASE/'venturi-five-window-v2'}
out=BASE/'venturi-five-evidence-v1.tgz';assert not out.exists();pins={};omitted={}
with tarfile.open(out,'w:gz',compresslevel=6) as tar:
 for label,source in sources.items():
  for p in sorted(source.rglob('*')):
   if not p.is_file():continue
   name=label+'/'+str(p.relative_to(source))
   info=dict(remote=str(p),bytes=p.stat().st_size,sha256=sha(p))
   if p.suffix in ['.so','.nsys-rep'] or p.name=='.build.lock':omitted[name]=info;continue
   pins[name]=info;tar.add(p,arcname=name,recursive=False)
 data=json.dumps(dict(included=pins,omitted_remote_only=omitted),indent=2).encode();import io
 entry=tarfile.TarInfo('manifest.json');entry.size=len(data);tar.addfile(entry,io.BytesIO(data))
record=dict(path=str(out),bytes=out.stat().st_size,sha256=sha(out),included=len(pins),omitted_remote_only=omitted,unix=time.time())
(BASE/'venturi-five-evidence-v1.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
