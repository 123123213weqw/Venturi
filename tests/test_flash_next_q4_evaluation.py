"""Cold-cache helper touches explicit temp files only; graders never accept truncation."""
import importlib.util,json,os,pathlib,subprocess,sys
import pytest

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from evaluate_flash_next_q4_bundle import chat,grade
from flash_next_file_cache import owned_files,evict_owned


def test_exact_official_literal_suffix_and_strict_graders():
 assert chat('x',thinking=False).endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')
 assert chat('x',thinking=True).endswith('<|im_start|>assistant\n<think>\n')
 q=dict(grader='json',answer={'answer':42})
 assert grade(q,'{"answer":42}') and not grade(q,'{"answer":42.0}')
 assert not grade(q,'{"answer":42,"extra":1}') and not grade(q,'```json\n{"answer":42}\n```')
 assert grade(dict(grader='exact',answer='apple'),'apple\n')
 assert not grade(dict(grader='exact',answer='apple'),'Apple')


def test_owned_files_reject_escape_symlinks_and_directories(tmp_path):
 own=tmp_path/'own';own.mkdir();(own/'x').write_bytes(b'x');(tmp_path/'other').write_bytes(b'y');(own/'link').symlink_to(tmp_path/'other')
 assert owned_files(own,['x'])==[own/'x']
 for names in (['../other'],['link'],['.']):
  with pytest.raises(ValueError):owned_files(own,names)


@pytest.mark.skipif(sys.platform!='linux',reason='Linux mincore/advisory eviction only, no global fallback')
def test_linux_temp_file_eviction_probe(tmp_path):
 p=tmp_path/'fixture';p.write_bytes(b'z'*2**20)
 with p.open('rb') as f:os.fsync(f.fileno())
 r=evict_owned(owned_files(tmp_path,['fixture']))
 assert r['after']['files']==1 and r['after']['pages']==256 and r['cold_verified']
 assert p.read_bytes()==b'z'*2**20


def test_eval_help_is_read_only():
 r=subprocess.run([sys.executable,str(ROOT/'tools/evaluate_flash_next_q4_bundle.py'),'--help'],env=dict(os.environ,PYTHONPATH=str(ROOT/'src')),capture_output=True,text=True,check=True,timeout=30)
 assert '--bundle' in r.stdout and '--stage' in r.stdout
