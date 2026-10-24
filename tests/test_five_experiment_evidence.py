"""CPU-only checks for the five-experiment measurement parsers."""
import csv,importlib.util,sqlite3
from pathlib import Path
import pytest
ROOT=Path(__file__).parents[1]/'benchmarks/v100-five-experiments'
def load(name):
    s=importlib.util.spec_from_file_location('evidence_'+name.replace('-','_'),ROOT/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_wide_ncu_units_and_bandwidth(tmp_path):
    p=tmp_path/'counter-wide.csv'
    with p.open('w') as f:
        f.write('==PROF== connected\n');w=csv.writer(f)
        w.writerow(['ID','Process ID','Kernel Name','Device','gpu__time_duration.sum','dram__bytes_read.sum','dram__bytes_write.sum','sm__warps_active.avg.pct_of_peak_sustained_active'])
        w.writerow(['','','','','ns','byte','byte','%'])
        w.writerow(['0','42','gdn(int, int)','1','100,000','10,000,000','5,000,000','92.8'])
    r=load('parse-counters').parse(p)
    assert r['available'] and r['dram_read_plus_write_GB_per_second']==pytest.approx(150.)
    assert r['metrics']['sm__warps_active.avg.pct_of_peak_sustained_active']['value']==92.8
    assert r['metrics']['gpu__time_duration.sum']['device']=='1'

def test_long_ncu_units_and_bandwidth(tmp_path):
    p=tmp_path/'counter-long.csv'
    with p.open('w') as f:
        w=csv.writer(f);w.writerow(['Metric Name','Metric Unit','Metric Value'])
        w.writerows([['gpu__time_duration.sum','usecond','10'],['dram__bytes_read.sum','Mbyte','1'],['dram__bytes_write.sum','byte','0']])
    assert load('parse-counters').parse(p)['dram_read_plus_write_GB_per_second']==pytest.approx(100.)

def test_graph_level_trace_is_not_a_kernel_count_speedup(tmp_path):
    p=tmp_path/'trace.sqlite';db=sqlite3.connect(p)
    db.executescript('CREATE TABLE StringIds(id INTEGER,value TEXT);CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE(start INTEGER);INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES(1);');db.close()
    with pytest.raises(ValueError,match='omits kernel nodes'):load('summarize-trace').summarize(p)
