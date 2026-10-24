"""Read both long and wide raw Nsight Compute CSV; preserve metric units."""
import csv,json,math,sys
from pathlib import Path

def parse(p):
 lines=p.read_text().splitlines()
 starts=[]
 for i,line in enumerate(lines):
  try:fields=next(csv.reader([line]))
  except (csv.Error,StopIteration):continue
  if ('Metric Name' in fields and 'Metric Value' in fields) or fields[:2]==['ID','Process ID']:starts.append(i)
 if not starts:return dict(file=p.name,available=False,reason='No hardware metric table; inspect log')
 reader=csv.reader(lines[starts[0]:]);header=next(reader);raw=list(reader);out={}
 if 'Metric Name' in header:rows=[dict(zip(header,r)) for r in raw]
 else:
  units=raw[0];values=[r for r in raw[1:] if r and r[0]];assert len(values)==1,'Expected one profiled launch'
  values=values[0];meta=dict(zip(header,values))
  rows=[{'Metric Name':h,'Metric Value':v,'Metric Unit':u,'Kernel Name':meta.get('Kernel Name'),'Device':meta.get('Device'),'ID':meta.get('ID')} for h,u,v in zip(header,units,values) if h.startswith(('gpu__','sm__','smsp__','dram__','l1tex__','lts__'))]
 for r in rows:
  key=r.get('Metric Name');v=r.get('Metric Value','').replace(',','');unit=r.get('Metric Unit','')
  if not key:continue
  try:v=float(v)
  except ValueError:continue
  if not math.isfinite(v):continue
  out[key]=dict(value=v,unit=unit,kernel=r.get('Kernel Name'),device=r.get('Device'),id=r.get('ID'))
 duration=out.get('gpu__time_duration.sum');gbps=None
 if duration:
  units={'second':1.,'msecond':1e-3,'usecond':1e-6,'nsecond':1e-9,'ns':1e-9,'us':1e-6,'ms':1e-3,'s':1.}
  seconds=duration['value']*units[duration['unit']]
  def byte_value(name):
   q=out[name];factor={'byte':1,'Kbyte':1e3,'Mbyte':1e6,'Gbyte':1e9}[q['unit']];return q['value']*factor
  if seconds and all(k in out for k in ['dram__bytes_read.sum','dram__bytes_write.sum']):gbps=sum(byte_value(k) for k in ['dram__bytes_read.sum','dram__bytes_write.sum'])/seconds/1e9
 return dict(file=p.name,available=bool(out),metrics=out,dram_read_plus_write_GB_per_second=gbps,scope='One replay-profiled kernel, not model throughput. Baseline conv/GDN launch processes one request; batched launch processes eight. Compare equal request work, not raw per-launch durations. Uncontrolled caches/clocks warnings preserved.')
if __name__=='__main__':
 d=Path(sys.argv[1]);out=[parse(p) for p in sorted(d.glob('counter-*.csv'))];Path(sys.argv[2]).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(out,indent=2,allow_nan=False))
