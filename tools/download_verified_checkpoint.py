"""Bounded HTTPS range download from an explicit hash-pinned file manifest.

No model code/auth tokens/global network settings. A source mirror is a transport,
not the authority: each complete file must match the manifest's original SHA256.
Private partials are never published under checkpoint names before verification.
"""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError
from datetime import datetime, timezone
import fcntl
import errno
import hashlib
import http.client
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import ssl
import threading
import tempfile
import time
import urllib.parse
import urllib.request


class CheckpointIntegrityError(ValueError):
    """Preserve suspect bytes; network retries cannot repair integrity failures."""


class CheckpointStorageError(OSError):
    """Local storage failed: stop all writers, not a retryable source failure."""


class CheckpointTransportError(OSError):
    """A source/network failure, distinguished from local file I/O."""


def output_mount(path, mounts_file=Path('/proc/mounts')):
    """Linux emergency_ro is NOT reliably reflected in statvfs ST_RDONLY.

    Only inspect the selected output's longest matching mount. An intentionally
    read-only unrelated filesystem must not block writes to a healthy data disk.
    Other platforms fall back to statvfs plus the explicit write/fsync probe.
    """
    try:
        lines=mounts_file.read_text().splitlines()
    except FileNotFoundError:
        return None
    candidates=[]
    for line in lines:
        fields=line.split()
        if len(fields)<4:continue
        decode=lambda s:re.sub(r'\\([0-7]{3})',lambda m:chr(int(m[1],8)),s)
        mount=Path(decode(fields[1]))
        if path==mount or mount in path.parents:
            candidates.append(dict(target=str(mount),source=decode(fields[0]),
                filesystem=fields[2],options=fields[3].split(',')))
    return max(candidates,key=lambda m:len(m['target'])) if candidates else None


def require_writable_mount(path):
    mount=output_mount(path)
    if mount and {'ro','emergency_ro'} & set(mount['options']):
        raise CheckpointStorageError(errno.EROFS,'Output mount is read-only/emergency_ro; no writes attempted',str(path))
    return mount


def probe_output_storage(root, mount):
    """Tiny private fsync/read probe. Does not repair or remount anything."""
    stat=os.statvfs(root)
    if stat.f_flag & os.ST_RDONLY:
        raise CheckpointStorageError(errno.EROFS,'Output statvfs is read-only',str(root))
    fd,name=tempfile.mkstemp(prefix='.venturi-storage-probe-',dir=root)
    data=os.urandom(4096)
    # On a local I/O failure leave the private probe intact; do NOT try more
    # metadata writes to a failed filesystem in an error-cleanup loop.
    with os.fdopen(fd,'w+b') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno());stream.seek(0)
        if stream.read()!=data:
            raise CheckpointStorageError(errno.EIO,'Storage probe readback mismatch',name)
    Path(name).unlink()
    return dict(mount=mount,probe_bytes=len(data),probe_sha256=hashlib.sha256(data).hexdigest(),
        scope='Filesystem write/fsync/read probe only; not SMART or hardware certification')


class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != 'https':
            raise ValueError('Refuse non-HTTPS redirect')
        return super().redirect_request(request, fp, code, message, headers, newurl)


def transport_url(url, endpoint=None):
    if endpoint is None:
        return url
    if endpoint != 'https://hf-mirror.com':
        raise ValueError('Only the explicitly selected public HF mirror transport is supported')
    # Repository/revision/path and SHA authority remain in the original manifest.
    return urllib.parse.urlunsplit(urllib.parse.urlsplit(url)._replace(netloc='hf-mirror.com'))


def validate_resolutions(values, *, transport_endpoint=None):
    host = 'hf-mirror.com' if transport_endpoint else 'huggingface.co'
    result = {}
    for value in values:
        name, sep, address = value.partition('=')
        if not sep or name != host or name in result or not ipaddress.ip_address(address).is_global:
            raise ValueError('Resolution override requires the selected public source host and a global IP')
        result[name] = address
    return result


class ResolvedHTTPSConnection(http.client.HTTPSConnection):
    """Per-connection DNS routing only; TLS still verifies the original hostname."""
    def __init__(self, host, *, resolutions, **options):
        super().__init__(host, **options)
        original = self._create_connection
        def connect(address, *args, **kwargs):
            name, port = address
            return original((resolutions.get(name, name), port), *args, **kwargs)
        self._create_connection = connect


class ResolvedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, resolutions):
        self.resolutions = dict(resolutions)
        # Python 3.12+ removed the old check_hostname connection argument.
        # Carry verification in the SSLContext, supported on both versions.
        context = ssl.create_default_context()
        context.check_hostname = True
        super().__init__(context=context)

    def https_open(self, request):
        def connection(host, **options):
            return ResolvedHTTPSConnection(host, resolutions=self.resolutions, **options)
        return self.do_open(connection, request, context=self._context)


def validate_manifest(raw, *, max_file_bytes=8*2**30):
    # GGUF's large sparse tables can exceed a safetensors shard by an order of
    # magnitude. Keep the old 8 GiB default; larger assets require explicit
    # admission, not a silently widened global download limit.
    if type(max_file_bytes) is not int or not 0 < max_file_bytes <= 128*2**30:
        raise ValueError('Explicit file limit must be positive and at most 128 GiB')
    files = raw.get('files')
    if not isinstance(files, list) or not 1 <= len(files) <= 2048:
        raise ValueError('Bounded explicit file manifest required')
    names = set()
    for spec in files:
        name, size, digest, url = (spec.get(k) for k in ('filename', 'bytes', 'sha256', 'url'))
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,191}', name) or (
                name in names or '..' in name or name.startswith('.')):
            raise ValueError('Unique direct safe filenames required')
        if type(size) is not int or not 0 < size <= max_file_bytes or not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Explicit bounded file bytes and SHA256 required')
        if not isinstance(url, str):
            raise ValueError('Explicit HTTPS URL required')
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != 'https' or not parsed.netloc or parsed.username is not None or parsed.password is not None:
            raise ValueError('Public HTTPS source URLs required; no embedded credentials')
        names.add(name)
    return files


def file_hash(path,check_abort=lambda:None):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while True:
            check_abort()
            block=handle.read(8*2**20)
            if not block:break
            h.update(block)
    return h.hexdigest()


def download_one(*args, **kwargs):
    try:
        return _download_one(*args,**kwargs)
    except CheckpointTransportError:
        raise
    except OSError as error:
        if kwargs.get('stop_event') is not None:
            kwargs['stop_event'].set()
        raise CheckpointStorageError(error.errno,'Local checkpoint I/O failed; campaign aborted: '+str(error),
            getattr(error,'filename',None)) from error


def _download_one(spec, root, *, chunk_bytes, timeout, attempts, deadline, opener, progress,
                 transport_endpoint=None, reuse_redirect_url=False, connections_per_file=1,stop_event=None):
    stop_event=stop_event if stop_event is not None else threading.Event()
    def check_abort():
        if stop_event.is_set():
            raise CancelledError('Campaign stopped after local storage failure')
    check_abort()
    if type(connections_per_file) is not int or not 1 <= connections_per_file <= 8:
        raise ValueError("One to eight explicit connections per file required")
    name, size, expected = spec['filename'], spec['bytes'], spec['sha256']
    final, partial = root/name, root/(name+'.venturi.part')
    if final.is_symlink() or partial.is_symlink():
        raise CheckpointIntegrityError('Refuse symlink output/partial')
    if final.exists():
        if final.stat().st_size != size or file_hash(final,check_abort) != expected:
            raise CheckpointIntegrityError('Existing completed file differs; never overwrite: '+name)
        progress(name, size, True)
        return dict(filename=name, bytes=size, sha256=expected, reused_verified=True)
    existed = partial.exists()
    offset = partial.stat().st_size if existed else 0
    if offset > size:
        raise CheckpointIntegrityError('Overlong private partial: '+name)
    h = hashlib.sha256()
    resolved_url = None  # Ephemeral transport only: never log or persist signed URLs.
    if offset:
        with partial.open('rb') as handle:
            while True:
                check_abort()
                block=handle.read(8*2**20)
                if not block:break
                h.update(block)
    def fetch_range(lo, hi):
        nonlocal resolved_url
        check_abort()
        if time.monotonic() >= deadline:
            raise CheckpointTransportError('Explicit campaign wall budget reached')
        length = hi-lo+1
        # Offset-specific query prevents a stale intermediary range entry.
        parsed = urllib.parse.urlsplit(transport_url(spec['url'], transport_endpoint))
        query = parsed.query+('&' if parsed.query else '')+f'venturi_range={lo}-{hi}'
        canonical_range_url = urllib.parse.urlunsplit(parsed._replace(query=query))
        for attempt in range(attempts):
            check_abort()
            try:
                url = resolved_url or canonical_range_url
                request = urllib.request.Request(url, headers={
                    'Range': f'bytes={lo}-{hi}', 'Accept-Encoding': 'identity'})
                with opener(request, timeout=min(timeout, max(1, deadline-time.monotonic()))) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {lo}-{hi}/{size}':
                        raise ValueError('Source did not honor exact byte range: '+name)
                    data = response.read(length+1)
                    final_url = response.geturl() if hasattr(response, 'geturl') else None
                if len(data) != length:
                    raise ValueError('Truncated/overlong range: '+name)
                if reuse_redirect_url and final_url and final_url != canonical_range_url:
                    redirect = urllib.parse.urlsplit(final_url)
                    if redirect.scheme != 'https' or not redirect.netloc or redirect.username is not None or redirect.password is not None:
                        raise ValueError('Refuse non-public HTTPS redirect transport')
                    resolved_url = final_url
                break
            except (OSError, ValueError) as error:
                # A signed CDN URL may expire or bind a particular range.
                # Re-resolve through the SAME immutable canonical source.
                if isinstance(error, ValueError) or getattr(error, 'code', None) in (400, 401, 403, 404, 410, 416):
                    resolved_url = None
                if attempt+1 == attempts or time.monotonic() >= deadline:
                    if isinstance(error,OSError):
                        raise CheckpointTransportError('Source transport failed: '+str(error)) from error
                    raise
                check_abort()
                delay = 2**attempt
                if getattr(error, 'code', None) in (429, 503):
                    # A bounded numeric Retry-After prevents a busy mirror/CDN
                    # from being hammered. No URL/credential is persisted.
                    try:
                        delay = max(delay, min(60, float(error.headers.get('Retry-After', '0'))))
                    except (AttributeError, TypeError, ValueError):
                        pass
                time.sleep(min(delay, max(0, deadline-time.monotonic())))
        return data

    # A partial is ALWAYS a contiguous prefix, even in parallel mode. Fetch
    # futures only own bounded RAM buffers; a single ordered consumer writes and
    # hashes bytes. No sparse holes, bitmap trust, duplicate weight file or
    # "file length equals complete" assumption is introduced.
    with partial.open('ab' if existed else 'xb') as target:
        def append(data):
            nonlocal offset
            check_abort()
            target.write(data)
            target.flush()
            h.update(data)
            offset += len(data)
            progress(name, offset, False)

        if connections_per_file == 1:
            while offset < size:
                append(fetch_range(offset, min(offset+chunk_bytes, size)-1))
        elif offset < size:
            # Resolve one HTTPS CDN route first, preventing a burst of mirror
            # metadata requests. Subsequent ranges reuse it only when opted in.
            append(fetch_range(offset, min(offset+chunk_bytes, size)-1))
            pending = deque()
            next_offset = offset
            with ThreadPoolExecutor(max_workers=connections_per_file) as ranges:
                def fill():
                    nonlocal next_offset
                    while next_offset < size and len(pending) < connections_per_file:
                        hi = min(next_offset+chunk_bytes, size)-1
                        pending.append(ranges.submit(fetch_range, next_offset, hi))
                        next_offset = hi+1
                try:
                    fill()
                    while pending:
                        append(pending.popleft().result())
                        fill()
                finally:
                    for future in pending:
                        future.cancel()
        os.fsync(target.fileno())
    check_abort()
    if h.hexdigest() != expected:
        raise CheckpointIntegrityError('Official complete-file SHA256 mismatch; partial NOT published: '+name)
    # Atomic no-overwrite publication on the same filesystem. Retain .part on
    # hash failure; do not repeatedly re-download/erase a numerical counterexample.
    os.link(partial, final)
    partial.unlink()
    progress(name, size, True)
    return dict(filename=name, bytes=size, sha256=expected, reused_verified=False)


def main(argv=None):
    try:
        return _main(argv)
    except CheckpointStorageError:
        raise
    except OSError as error:
        raise CheckpointStorageError(error.errno,'Local campaign setup/publication failed: '+str(error),
            getattr(error,'filename',None)) from error


def _main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, choices=range(1, 17), default=2)
    p.add_argument('--connections-per-file', type=int, choices=range(1, 9), default=1,
                   help='Bounded parallel range prefetch; ordered writes keep partials contiguous')
    p.add_argument('--chunk-mib', type=int, choices=(4, 8, 16, 32, 64), default=32)
    p.add_argument('--timeout', type=int, default=45)
    p.add_argument('--attempts', type=int, choices=range(1, 9), default=3)
    p.add_argument('--file-recovery-rounds', type=int, choices=range(3), default=1)
    p.add_argument('--wall-seconds', type=int, default=7200)
    p.add_argument('--reserve-disk-gib', type=int, default=32)
    p.add_argument('--max-file-gib', type=int, choices=(8, 64, 128), default=8,
                   help='Explicit admission limit for large GGUF assets; default unchanged')
    p.add_argument('--transport-endpoint', choices=('https://hf-mirror.com',),
                   help='Explicit mirror transport; original manifest SHA256 remains the authority')
    p.add_argument('--resolve', action='append', default=[], metavar='HOST=PUBLIC_IP',
                   help='Per-process source DNS route, preserving TLS hostname checks; no system DNS edit')
    p.add_argument('--reuse-redirect-url', action='store_true',
                   help='Reuse an in-memory HTTPS CDN redirect between ranges; verify each range and full SHA as usual')
    p.add_argument('--largest-first', action='store_true',
                   help='Schedule large independent files first without changing manifest identity')
    a = p.parse_args(argv)
    if a.workers*a.connections_per_file*a.chunk_mib > 2048:
        raise ValueError('Explicit in-flight range buffers must not exceed 2048 MiB')
    if min(a.timeout, a.wall_seconds) <= 0 or a.reserve_disk_gib < 0:
        raise ValueError('Positive time and nonnegative disk reserve budgets required')
    source = Path(a.manifest).read_bytes()
    manifest, digest = json.loads(source), hashlib.sha256(source).hexdigest()
    files = validate_manifest(manifest, max_file_bytes=a.max_file_gib*2**30)
    resolutions = validate_resolutions(a.resolve, transport_endpoint=a.transport_endpoint)
    root = Path(a.output).expanduser().resolve()
    mount=require_writable_mount(root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root/'.venturi-download.lock'
    if lock_path.is_symlink():
        raise ValueError('Refuse symlink lock')
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another writer owns this output; do not duplicate downloads') from error
        storage_preflight=probe_output_storage(root,mount)
        manifest_path = root/'.venturi-download-manifest.json'
        if manifest_path.exists() and manifest_path.read_bytes() != source:
            raise ValueError('Existing download has different provenance; do not overwrite')
        manifest_path.write_bytes(source)
        remaining = sum(max(0, s['bytes']-((root/s['filename']).stat().st_size if (root/s['filename']).exists() else
            (root/(s['filename']+'.venturi.part')).stat().st_size if (root/(s['filename']+'.venturi.part')).exists() else 0)) for s in files)
        if remaining+a.reserve_disk_gib*2**30 > shutil.disk_usage(root).free:
            raise ValueError('Insufficient disk for complete campaign plus explicit reserve')
        started, deadline = time.monotonic(), time.monotonic()+a.wall_seconds
        states, guard = {}, threading.Lock()
        stop_event=threading.Event()
        fatal_storage=None
        progress_path = root/'.venturi-download-progress.json'
        def progress(name, offset, verified):
            with guard:
                if stop_event.is_set():
                    raise CancelledError('Storage failure: no more progress writes')
                states[name] = dict(bytes=offset, verified=verified)
                record = dict(manifest_sha256=digest, utc=datetime.now(timezone.utc).isoformat(),
                    transport_endpoint=a.transport_endpoint, resolutions=resolutions,
                    reuse_redirect_url=a.reuse_redirect_url,
                    workers=a.workers, connections_per_file=a.connections_per_file,
                    storage_preflight=storage_preflight,
                    seconds=time.monotonic()-started, verified_files=sum(v['verified'] for v in states.values()),
                    expected_files=len(files), verified_bytes=sum(s['bytes'] for s in files if states.get(s['filename'], {}).get('verified')),
                    expected_bytes=sum(s['bytes'] for s in files), states=states)
                temp = progress_path.with_suffix('.tmp')
                temp.write_text(json.dumps(record, indent=2)+'\n')
                temp.replace(progress_path)
        opener = urllib.request.build_opener(HTTPSRedirect(), ResolvedHTTPSHandler(resolutions)).open
        results, errors, history = {}, {}, []
        remaining_files = sorted(files, key=lambda s: s['bytes'], reverse=True) if a.largest_first else files
        for recovery_round in range(a.file_recovery_rounds+1):
            if recovery_round:
                if time.monotonic() >= deadline:
                    break
                time.sleep(min(30*recovery_round, max(0, deadline-time.monotonic())))
            retry_files = []
            with ThreadPoolExecutor(max_workers=a.workers) as pool:
                jobs = {pool.submit(download_one, spec, root, chunk_bytes=a.chunk_mib*2**20,
                    timeout=a.timeout, attempts=a.attempts, deadline=deadline, opener=opener,
                    progress=progress, transport_endpoint=a.transport_endpoint,
                    reuse_redirect_url=a.reuse_redirect_url,
                    connections_per_file=a.connections_per_file,stop_event=stop_event): spec for spec in remaining_files}
                for job in as_completed(jobs):
                    spec = jobs[job]
                    name = spec['filename']
                    try:
                        item = job.result()
                        results[name] = item
                        errors.pop(name, None)
                        print(json.dumps(item), flush=True)
                    except Exception as error:
                        record = dict(filename=name, error=f'{type(error).__name__}: {error}',
                            recovery_round=recovery_round, retryable=isinstance(error, (OSError, ValueError))
                            and not isinstance(error, (CheckpointIntegrityError,CheckpointStorageError)))
                        if isinstance(error,CheckpointStorageError):
                            fatal_storage=error
                            stop_event.set()
                            for other in jobs:other.cancel()
                        errors[name] = record
                        history.append(record)
                        print(json.dumps(record), flush=True)
                        # A transient failure in one file must not cancel other
                        # independent files. Retry only after the rest progressed,
                        # with finite rounds/backoff and the SAME wall deadline.
                        # Integrity evidence remains private and is never erased.
                        if record['retryable']:
                            retry_files.append(spec)
            if stop_event.is_set() or not retry_files:
                break
            remaining_files = retry_files
        result = dict(manifest_sha256=digest, complete=not errors and len(results)==len(files),
            transport_endpoint=a.transport_endpoint, resolutions=resolutions,
            reuse_redirect_url=a.reuse_redirect_url,
            workers=a.workers, connections_per_file=a.connections_per_file,
            storage_preflight=storage_preflight,
            seconds=time.monotonic()-started, files=list(results.values()), errors=list(errors.values()),
            attempt_errors=history, recovered_errors=[e for e in history if e['filename'] in results])
        if fatal_storage is not None:
            result['fatal_storage']=True
            # The operator's stdout may be on a separate healthy volume. Never
            # attempt to save a report back to the failed checkpoint mount.
            print(json.dumps(result),flush=True)
            raise fatal_storage
        (root/'.venturi-download-result.json').write_text(json.dumps(result,indent=2)+'\n')
        if not result['complete']:
            raise RuntimeError('Incomplete verified checkpoint; inspect preserved partial/error records')
        marker = root/'.venturi-complete.json'
        temporary = root/'.venturi-complete.tmp'
        temporary.write_text(json.dumps(result,indent=2)+'\n')
        temporary.replace(marker)


if __name__ == '__main__':
    main()
