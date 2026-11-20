from pathlib import Path
ns={};exec(Path('/tmp/venturi-dispatch-helper.py').read_text(),ns)
s=Path('/tmp/venturi-column-tp-fixtures-v1-worker.py').read_text().replace("out=root/'artifacts/venturi-column-tp-fixtures-v1'","out=root/'artifacts/venturi-column-tp-fixtures-v2'")
start=s.index(" run([py,'-c',");end=s.index(" library=(out/'library.txt')",start)
replacement=r''' previous=root/'artifacts/venturi-column-tp-fixtures-v1';deadline=time.monotonic()+900
 while not (previous/'exit-code.txt').exists():
  assert time.monotonic()<deadline,'Bounded predecessor wait elapsed'
  time.sleep(10)
 assert (previous/'exit-code.txt').read_text().strip()=='1'
 assert 'focused-tests.log' in (previous/'failure.txt').read_text()
 library=(previous/'library.txt').read_text().removesuffix('\\n').strip()
 assert pathlib.Path(library).is_file(),repr(library)
 build=json.loads(pathlib.Path(library).with_name('build.json').read_text())
 assert len(build['source_sha256'])==10
 for rel,h in build['source_sha256'].items():assert hashlib.sha256((src/'src/venturi/ops/csrc'/rel).read_bytes()).hexdigest()==h
 (out/'library.txt').write_text(library+'\n')
 (out/'reused-build.json').write_text(json.dumps(dict(path=library,sha256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),source_sha256=build['source_sha256'],reason='Original command wrote a literal backslash-n in the library path; native compilation succeeded. Exact native source hashes; no repeated compilation.'),indent=2)+'\n')
'''
# The replacement is raw Python source; newline literals in the worker must be
# single escaped newlines, but removesuffix must match the original literal pair.
replacement=replacement.replace("removesuffix('\\\\n')","removesuffix('\\\\n')")
s=s[:start]+replacement+s[end:]
ns['dispatch']('venturi-column-tp-fixtures-v2',s)
