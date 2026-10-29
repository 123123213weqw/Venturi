"""Synthetic download protocol tests; no network or actual checkpoint claim."""

import hashlib
import importlib.util
import json
from pathlib import Path
import time
import ssl
import socket
import threading
import errno
import urllib.error

import pytest

spec = importlib.util.spec_from_file_location(
    "verified_download",
    Path(__file__).parents[1] / "tools/download_verified_checkpoint.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def manifest(data=b"abcdefghijk"):
    return dict(
        filename="model.safetensors",
        bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        url="https://example.com/model.safetensors",
    )


class Response:
    def __init__(self, data, status, content_range):
        self.data, self.status, self.headers = (
            data,
            status,
            {"Content-Range": content_range},
        )

    def read(self, limit):
        return self.data[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def run(tmp_path, data=b"abcdefghijk", *, bad=None, partial=b"", expected=None):
    if partial:
        (tmp_path / "model.safetensors.venturi.part").write_bytes(partial)
    calls, progress = [], []

    def opener(request, **options):
        lo, hi = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        calls.append((lo, hi))
        value = data[lo : hi + 1]
        cr = f"bytes {lo}-{hi}/{len(data)}"
        if bad == "range":
            cr = "bytes 0-0/1"
        if bad == "short":
            value = value[:-1]
        if bad == "long":
            value += b"!"
        return Response(value, 200 if bad == "status" else 206, cr)

    result = mod.download_one(
        expected or manifest(data),
        tmp_path,
        chunk_bytes=4,
        timeout=1,
        attempts=1,
        deadline=time.monotonic() + 10,
        opener=opener,
        progress=lambda *args: progress.append(args),
    )
    return result, calls, progress


def test_resume_exact_ranges_publish_only_complete_sha(tmp_path):
    data = b"abcdefghijk"
    result, calls, progress = run(tmp_path, data, partial=data[:3])
    assert calls == [(3, 6), (7, 10)]
    assert (tmp_path / "model.safetensors").read_bytes() == data
    assert not (tmp_path / "model.safetensors.venturi.part").exists()
    assert progress[-1] == ("model.safetensors", len(data), True)
    assert result["sha256"] == hashlib.sha256(data).hexdigest()


def test_zero_byte_interrupted_partial_is_resumable_not_new_exclusive_create(tmp_path):
    part = tmp_path / "model.safetensors.venturi.part"
    part.touch()
    result, calls, progress = run(tmp_path)
    assert calls == [(0, 3), (4, 7), (8, 10)]
    assert result["sha256"] == hashlib.sha256(b"abcdefghijk").hexdigest()
    assert progress[-1] == ("model.safetensors", 11, True)
    assert not part.exists()


@pytest.mark.parametrize("bad", ["status", "range", "short", "long"])
def test_invalid_response_never_published_or_appended(tmp_path, bad):
    with pytest.raises(ValueError):
        run(tmp_path, bad=bad, partial=b"abc")
    assert not (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "model.safetensors.venturi.part").read_bytes() == b"abc"


def test_full_sha_is_authority_not_equal_length_or_partial(tmp_path):
    spec = manifest()
    spec["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run(tmp_path, expected=spec)
    assert not (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "model.safetensors.venturi.part").read_bytes() == b"abcdefghijk"


def test_existing_complete_file_verified_never_overwritten(tmp_path):
    target = tmp_path / "model.safetensors"
    target.write_bytes(b"abcdefghijk")
    result, calls, _ = run(tmp_path)
    assert result["reused_verified"] and calls == []
    target.write_bytes(b"bad")
    with pytest.raises(ValueError, match="never overwrite"):
        run(tmp_path)
    assert target.read_bytes() == b"bad"


def test_symlink_and_overlong_part_refused(tmp_path):
    part = tmp_path / "model.safetensors.venturi.part"
    part.write_bytes(b"x" * 20)
    with pytest.raises(ValueError, match="Overlong"):
        run(tmp_path)
    part.unlink()
    source = tmp_path / "elsewhere"
    source.write_bytes(b"ab")
    part.symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        run(tmp_path)
    assert source.read_bytes() == b"ab"


@pytest.mark.parametrize(
    "change",
    [
        dict(filename="../bad"),
        dict(filename=".env"),
        dict(bytes=True),
        dict(bytes=0),
        dict(sha256="bad"),
        dict(url="http://example.com/a"),
        dict(url="https://user:pw@example.com/a"),
        dict(url=None),
    ],
)
def test_manifest_admission(change):
    item = dict(manifest(), **change)
    with pytest.raises(ValueError):
        mod.validate_manifest(dict(files=[item]))


def test_duplicate_and_redirect_scheme_rejected():
    with pytest.raises(ValueError, match="Unique"):
        mod.validate_manifest(dict(files=[manifest(), manifest()]))
    with pytest.raises(ValueError, match="non-HTTPS"):
        mod.HTTPSRedirect().redirect_request(
            None, None, None, None, None, "http://example.com/a"
        )


def test_large_gguf_requires_explicit_admission_without_changing_default():
    item = dict(manifest(), filename="Engram.gguf", bytes=110_595_290_816)
    with pytest.raises(ValueError):
        mod.validate_manifest(dict(files=[item]))
    assert mod.validate_manifest(dict(files=[item]), max_file_bytes=128*2**30) == [item]
    with pytest.raises(ValueError):
        mod.validate_manifest(dict(files=[dict(item, bytes=128*2**30+1)]),
                              max_file_bytes=128*2**30)


@pytest.mark.parametrize("limit", [0, -1, True, 129*2**30, 1.5])
def test_invalid_file_admission_limit(limit):
    with pytest.raises(ValueError):
        mod.validate_manifest(dict(files=[manifest()]), max_file_bytes=limit)


def test_mirror_changes_transport_only_not_authoritative_source_or_digest():
    url = 'https://huggingface.co/owner/repo/resolve/'+ 'a'*40 +'/model.gguf?x=1'
    out = mod.transport_url(url, 'https://hf-mirror.com')
    assert out == url.replace('huggingface.co', 'hf-mirror.com')
    assert mod.transport_url(url) == url
    with pytest.raises(ValueError):
        mod.transport_url(url, 'http://hf-mirror.com')
    assert mod.validate_resolutions(['hf-mirror.com=160.16.86.14'],
                                    transport_endpoint='https://hf-mirror.com') == {'hf-mirror.com':'160.16.86.14'}


@pytest.mark.parametrize('route', ['hf-mirror.com=127.0.0.1','hf-mirror.com=192.168.0.1',
    'elsewhere.com=160.16.86.14','hf-mirror.com=not-an-ip'])
def test_dns_override_must_be_public_selected_source(route):
    with pytest.raises(ValueError):
        mod.validate_resolutions([route], transport_endpoint='https://hf-mirror.com')


def test_connection_routes_tcp_without_replacing_tls_hostname(monkeypatch):
    calls = []
    monkeypatch.setattr(socket, 'create_connection', lambda address, *a, **k: calls.append(address))
    conn = mod.ResolvedHTTPSConnection('hf-mirror.com', resolutions={'hf-mirror.com':'160.16.86.14'})
    conn._create_connection(('hf-mirror.com',443), 5)
    assert calls == [('160.16.86.14',443)]
    assert conn.host == 'hf-mirror.com'
    assert conn._context.check_hostname and conn._context.verify_mode == ssl.CERT_REQUIRED
    conn._create_connection(('other-public.example',443),5)
    assert calls[-1] == ('other-public.example',443)


def test_handler_verification_uses_context_not_removed_python312_keyword(monkeypatch):
    handler = mod.ResolvedHTTPSHandler({})
    def do_open(factory, request, **options):
        assert set(options) == {'context'}
        assert options['context'].check_hostname
        assert options['context'].verify_mode == ssl.CERT_REQUIRED
        return 'verified-context'
    monkeypatch.setattr(handler,'do_open',do_open)
    assert handler.https_open(object()) == 'verified-context'


@pytest.mark.parametrize('expire', [False, True])
def test_ephemeral_redirect_reuse_preserves_ranges_and_sha_with_expiry(tmp_path, monkeypatch, expire):
    data=b'abcdefghijk';calls=[]
    class RedirectResponse(Response):
        def geturl(self):return 'https://cdn.example.com/immutable-blob?signature=private-ephemeral'
    def opener(request,**options):
        lo,hi=map(int,request.headers['Range'].removeprefix('bytes=').split('-'))
        calls.append((request.full_url,lo,hi))
        if expire and len(calls)==2:
            raise urllib.error.HTTPError(request.full_url,403,'Expired',{},None)
        return RedirectResponse(data[lo:hi+1],206,f'bytes {lo}-{hi}/{len(data)}')
    monkeypatch.setattr(mod.time,'sleep',lambda x:None)
    out=mod.download_one(manifest(data),tmp_path,chunk_bytes=4,timeout=1,attempts=2,
        deadline=time.monotonic()+10,opener=opener,progress=lambda *a:None,reuse_redirect_url=True)
    assert (tmp_path/'model.safetensors').read_bytes()==data
    assert out['sha256']==hashlib.sha256(data).hexdigest()
    assert 'example.com/model.safetensors?venturi_range=' in calls[0][0]
    assert calls[1][0]=='https://cdn.example.com/immutable-blob?signature=private-ephemeral'
    assert 'venturi_range' not in calls[1][0]
    if expire:
        assert 'example.com/model.safetensors?venturi_range=' in calls[2][0]
        assert calls[1][1:]==calls[2][1:] == (4,7)


@pytest.mark.parametrize("failure", ["timeout", "integrity", "persistent"])
def test_main_keeps_independent_files_and_bounds_failed_file_recovery(tmp_path, monkeypatch, failure):
    files = [dict(manifest(bytes([n]) * 5), filename=f"part{n}", url=f"https://example.com/part{n}") for n in range(3)]
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(dict(files=files)))
    out = tmp_path / "weights"
    calls = []

    def opener(request, **options):
        name = mod.urllib.parse.urlsplit(request.full_url).path.rsplit("/", 1)[1]
        calls.append(name)
        index = int(name[-1])
        if index == 0 and (failure == "persistent" or calls.count(name) == 1):
            if failure != "integrity":
                raise TimeoutError("synthetic transient socket timeout")
            return Response(b"bad!!", 206, "bytes 0-4/5")
        return Response(bytes([index]) * 5, 206, "bytes 0-4/5")

    class Opener:
        open = staticmethod(opener)

    sleeps = []
    monkeypatch.setattr(mod.urllib.request, "build_opener", lambda *a: Opener())
    monkeypatch.setattr(mod.time, "sleep", sleeps.append)
    args = ["--manifest", str(source), "--output", str(out), "--workers", "1",
            "--attempts", "1", "--file-recovery-rounds", "2", "--reserve-disk-gib", "0"]
    if failure == "timeout":
        mod.main(args)
        result = json.loads((out / ".venturi-complete.json").read_text())
        assert result["complete"] and not result["errors"]
        assert len(result["files"]) == 3 and len(result["recovered_errors"]) == 1
        assert calls == ["part0", "part1", "part2", "part0"]
        assert sleeps == [30]
    else:
        with pytest.raises(RuntimeError, match="Incomplete"):
            mod.main(args)
        result = json.loads((out / ".venturi-download-result.json").read_text())
        assert not result["complete"] and len(result["files"]) == 2
        assert not (out / ".venturi-complete.json").exists()
        assert len(result["errors"]) == 1 and not result["recovered_errors"]
        assert calls[:3] == ["part0", "part1", "part2"]
        assert calls.count("part0") == (1 if failure == "integrity" else 3)
        assert sleeps == ([] if failure == "integrity" else [30, 60])
        assert (out / "part0.venturi.part").read_bytes() == (b"bad!!" if failure == "integrity" else b"")


def test_parallel_prefetch_writes_only_contiguous_ordered_prefixes(tmp_path):
    data = bytes(range(64))
    barrier = threading.Barrier(4)
    release_first = threading.Event()
    done, prefixes = [], []
    lock = threading.Lock()

    def opener(request, **options):
        lo, hi = map(int, request.headers['Range'].removeprefix('bytes=').split('-'))
        if lo in (4, 8, 12, 16):
            barrier.wait(timeout=5)
            if lo == 4:
                assert release_first.wait(timeout=5)
            elif lo == 16:
                with lock:
                    done.append(lo)
                release_first.set()
                return Response(data[lo:hi+1], 206, f'bytes {lo}-{hi}/{len(data)}')
        with lock:
            done.append(lo)
        return Response(data[lo:hi+1], 206, f'bytes {lo}-{hi}/{len(data)}')

    def progress(name, offset, verified):
        partial = tmp_path/(name+'.venturi.part')
        if not verified:
            prefix = partial.read_bytes()
            assert prefix == data[:offset]
            prefixes.append(offset)

    mod.download_one(manifest(data), tmp_path, chunk_bytes=4, timeout=1, attempts=1,
        deadline=time.monotonic()+10, opener=opener, progress=progress, connections_per_file=4)
    assert done.index(16) < done.index(4)
    assert prefixes == list(range(4, 65, 4))
    assert (tmp_path/'model.safetensors').read_bytes() == data


def test_failed_out_of_order_range_retains_safe_prefix_and_resumes(tmp_path):
    data = bytes(range(32))
    failed = False

    def opener(request, **options):
        lo, hi = map(int, request.headers['Range'].removeprefix('bytes=').split('-'))
        cr = 'bytes 0-0/1' if lo == 8 and not failed else f'bytes {lo}-{hi}/{len(data)}'
        return Response(data[lo:hi+1], 206, cr)

    kwargs = dict(chunk_bytes=4, timeout=1, attempts=1, deadline=time.monotonic()+10,
        opener=opener, progress=lambda *a:None, connections_per_file=4)
    with pytest.raises(ValueError, match='exact byte range'):
        mod.download_one(manifest(data), tmp_path, **kwargs)
    assert not (tmp_path/'model.safetensors').exists()
    assert (tmp_path/'model.safetensors.venturi.part').read_bytes() == data[:8]
    failed = True
    result = mod.download_one(manifest(data), tmp_path, **kwargs)
    assert result['sha256'] == hashlib.sha256(data).hexdigest()
    assert (tmp_path/'model.safetensors').read_bytes() == data


@pytest.mark.parametrize('connections', [0, 9, True, 1.5])
def test_parallel_range_admission(connections, tmp_path):
    with pytest.raises(ValueError, match='connections'):
        mod.download_one(manifest(), tmp_path, chunk_bytes=4, timeout=1, attempts=1,
            deadline=time.monotonic()+10, opener=lambda *a:None, progress=lambda *a:None,
            connections_per_file=connections)


def test_parallel_memory_budget_rejected_before_filesystem_work(tmp_path):
    output = tmp_path/'output'
    with pytest.raises(ValueError, match='2048 MiB'):
        mod.main(['--manifest', str(tmp_path/'missing'), '--output', str(output),
            '--workers', '16', '--connections-per-file', '8', '--chunk-mib', '64'])
    assert not output.exists()


def test_numeric_rate_limit_retry_after_is_bounded(tmp_path, monkeypatch):
    calls, sleeps = [], []
    data=b'abcdef'
    def opener(request, **options):
        calls.append(request.full_url)
        if len(calls)==1:
            raise urllib.error.HTTPError(request.full_url,429,'Rate limited',{'Retry-After':'300'},None)
        return Response(data,206,'bytes 0-5/6')
    monkeypatch.setattr(mod.time,'sleep',sleeps.append)
    mod.download_one(manifest(data),tmp_path,chunk_bytes=8,timeout=1,attempts=2,
        deadline=time.monotonic()+1000,opener=opener,progress=lambda *a:None)
    assert sleeps==[60] and len(calls)==2
    assert (tmp_path/'model.safetensors').read_bytes()==data


def test_mount_parser_longest_match_and_escaped_space(tmp_path):
    table=tmp_path/'mounts'
    table.write_text('/dev/sda4 / ext4 rw,relatime,emergency_ro 0 0\n'
        '/dev/sdb1 /home ext4 rw,relatime 0 0\n'
        '/dev/sdc1 /mnt/weights\\040dir ext4 ro 0 0\n')
    assert 'emergency_ro' in mod.output_mount(Path('/var/tmp/weights'),table)['options']
    assert mod.output_mount(Path('/home/test/models'),table)['source']=='/dev/sdb1'
    assert mod.output_mount(Path('/home2/models'),table)['target']=='/'
    assert mod.output_mount(Path('/mnt/weights dir/model'),table)['options']==['ro']


def test_emergency_readonly_mount_refused_before_any_probe_or_mkdir(tmp_path, monkeypatch):
    # The real incident had ST_RDONLY clear despite ext4 emergency_ro. Do not
    # rely on statvfs alone or write a probe to that selected failed mount.
    source=tmp_path/'manifest.json';source.write_text(json.dumps(dict(files=[manifest()])))
    output=tmp_path/'new-output'
    monkeypatch.setattr(mod,'output_mount',lambda p:dict(target='/',options=['rw','emergency_ro']))
    with pytest.raises(mod.CheckpointStorageError,match='emergency_ro'):
        mod.main(['--manifest',str(source),'--output',str(output)])
    assert not output.exists()


def test_private_storage_probe_checks_fsync_readback_and_cleans_only_itself(tmp_path):
    keep=tmp_path/'keep';keep.write_bytes(b'original')
    result=mod.probe_output_storage(tmp_path,None)
    assert result['probe_bytes']==4096 and 'not SMART' in result['scope']
    assert list(tmp_path.iterdir())==[keep] and keep.read_bytes()==b'original'


def test_local_fsync_failure_aborts_without_retrying_source_or_publishing(tmp_path, monkeypatch):
    data=b'abcdefgh';calls=[];stop=threading.Event()
    def opener(request,**options):
        lo,hi=map(int,request.headers['Range'].removeprefix('bytes=').split('-'))
        calls.append((lo,hi));return Response(data[lo:hi+1],206,f'bytes {lo}-{hi}/8')
    def fail(fd):raise OSError(errno.EIO,'synthetic local fsync failure')
    monkeypatch.setattr(mod.os,'fsync',fail)
    with pytest.raises(mod.CheckpointStorageError,match='Local checkpoint I/O'):
        mod.download_one(manifest(data),tmp_path,chunk_bytes=4,timeout=1,attempts=3,
            deadline=time.monotonic()+10,opener=opener,progress=lambda *a:None,stop_event=stop)
    assert stop.is_set() and calls==[(0,3),(4,7)]
    assert not (tmp_path/'model.safetensors').exists()
    assert (tmp_path/'model.safetensors.venturi.part').read_bytes()==data


@pytest.mark.parametrize('persistent',[False,True])
def test_transport_errno_is_not_misclassified_as_local_storage(tmp_path, monkeypatch, persistent):
    stop=threading.Event();calls=[];data=b'abcd'
    def opener(request,**options):
        calls.append(1)
        if len(calls)==1 or persistent:raise OSError(errno.EIO,'synthetic source error')
        return Response(data,206,'bytes 0-3/4')
    monkeypatch.setattr(mod.time,'sleep',lambda n:None)
    kwargs=dict(chunk_bytes=4,timeout=1,attempts=2,deadline=time.monotonic()+10,
        opener=opener,progress=lambda *a:None,stop_event=stop)
    if persistent:
        with pytest.raises(mod.CheckpointTransportError):mod.download_one(manifest(data),tmp_path,**kwargs)
    else:mod.download_one(manifest(data),tmp_path,**kwargs)
    assert not stop.is_set() and len(calls)==2


def test_preexisting_abort_makes_no_source_or_partial_changes(tmp_path):
    stop=threading.Event();stop.set()
    p=tmp_path/'model.safetensors.venturi.part';p.write_bytes(b'prefix')
    with pytest.raises(mod.CancelledError):
        mod.download_one(manifest(),tmp_path,chunk_bytes=4,timeout=1,attempts=1,
            deadline=time.monotonic()+10,opener=lambda *a:pytest.fail('no network after abort'),
            progress=lambda *a:pytest.fail('no progress writes after abort'),stop_event=stop)
    assert p.read_bytes()==b'prefix'


def test_campaign_local_progress_write_failure_stops_queued_files_and_reports_off_disk(tmp_path, monkeypatch, capsys):
    files=[dict(manifest(bytes([n])*5),filename=f'part{n}',url=f'https://example.com/part{n}') for n in range(3)]
    source=tmp_path/'manifest.json';source.write_text(json.dumps(dict(files=files)))
    output=tmp_path/'weights';calls=[]
    def opener(request,**options):
        n=int(mod.urllib.parse.urlsplit(request.full_url).path[-1]);calls.append(n)
        return Response(bytes([n])*5,206,'bytes 0-4/5')
    class Opener:open=staticmethod(opener)
    write_text=Path.write_text
    def guarded_write(self,*a,**kw):
        if self.name=='.venturi-download-progress.tmp':raise OSError(errno.EROFS,'synthetic emergency_ro')
        return write_text(self,*a,**kw)
    monkeypatch.setattr(Path,'write_text',guarded_write)
    monkeypatch.setattr(mod.urllib.request,'build_opener',lambda *a:Opener())
    sleeps=[];monkeypatch.setattr(mod.time,'sleep',sleeps.append)
    with pytest.raises(mod.CheckpointStorageError):
        mod.main(['--manifest',str(source),'--output',str(output),'--workers','1',
            '--file-recovery-rounds','2','--reserve-disk-gib','0'])
    assert calls==[0] and sleeps==[]
    assert not (output/'.venturi-complete.json').exists()
    assert not (output/'.venturi-download-result.json').exists()
    assert (output/'part0.venturi.part').read_bytes()==bytes(5)
    assert '"fatal_storage": true' in capsys.readouterr().out
