import pathlib,subprocess
payload=pathlib.Path('/tmp/venturi-dp4a-profile-payload.py').read_text();summary=pathlib.Path('/tmp/venturi-dp4a-profile-summary.py').read_text()
s=pathlib.Path('/tmp/venturi-dp4a-fixture-dispatch.py').read_text().replace('dp4a-src-v1','dp4a-src-v4').replace('venturi-dp4a-fixtures-v1','venturi-dp4a-final-v4');a=s.index(" run([py,'-m','venturi.ops.build'");b=s.index(" (out/'completed.json')",a)
body=r'''
 assert (root/'artifacts/venturi-llama-arithmetic-v2/exit-code.txt').read_text().strip()=='0'
 previous=root/'artifacts/venturi-dp4a-vector-v3';assert (previous/'exit-code.txt').read_text().strip()=='0';library=(previous/'library.txt').read_text().strip();(out/'library.txt').write_text(library+'\n');build=json.loads(pathlib.Path(library).with_name('build.json').read_text())
 for rel,sha in build['source_sha256'].items():assert hashlib.sha256((src/'src/venturi/ops/csrc'/rel).read_bytes()).hexdigest()==sha
 env.update(VENTURI_DP4A_LIBRARY=library,VENTURI_SM70_LIBRARY=library,VENTURI_Q8_BASELINE_LIBRARY=str(root/'artifacts/venturi-native-batch-v1/build/a4d0656f0f653e0a/libventuri_blocks.so'),VENTURI_BATCH_TEST='1',VENTURI_TP_TEST='1',VENTURI_PP_TEST='1',VENTURI_MTP_TEST='1',VENTURI_EXPECT_P2P_FAILURE='1')
 run([py,'-m','pytest','-q','--tb=short'],'tests.log',timeout=300)
 nsys=subprocess.check_output(['which','nsys'],text=True).strip();run([nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-event-trace=false','--force-overwrite=false','--output',str(out/'projection'),py,str(out/'profile_payload.py')],'profile.log',timeout=120)
 run([nsys,'export','--type','sqlite','--output',str(out/'projection.sqlite'),str(out/'projection.nsys-rep')],'export.log',timeout=90)
 run([py,str(out/'profile_summary.py')],'attribution.log',timeout=90)
 (out/'remote-retained-files.json').write_text(json.dumps({str(p):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in out.iterdir() if p.suffix in ['.sqlite','.nsys-rep']},indent=2)+'\n')
 run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],'gpu-after.txt')
 (out/'counter-policy.txt').write_text(next(s for s in pathlib.Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in s)+'\n')
 (out/'executed-source.json').write_text(json.dumps({str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','tools'] for p in (src/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts},indent=2)+'\n')
'''
s=s[:a]+body+s[b:]
s=s.replace("p.write_text({worker!r})", "p.write_text({worker!r});(out/'profile_payload.py').write_text("+repr(repr(payload))+");(out/'profile_summary.py').write_text("+repr(repr(summary))+")")
# The injected payload literals must remain Python source literals in the remote envelope.
s=s.replace(repr(repr(payload)),repr(payload)).replace(repr(repr(summary)),repr(summary))
pathlib.Path('/tmp/venturi-dp4a-final-envelope-v4.py').write_text(s)
