"""Linux OWNED immutable model-file data cache probe/eviction; never global cache."""
import ctypes,math,mmap,os
from pathlib import Path


def owned_files(root,names):
 root=Path(root).resolve();out=[]
 for name in names:
  p=root/name
  if p.is_symlink() or not p.resolve().is_relative_to(root) or not p.is_file():
   raise ValueError('Only explicit owned regular model files inside root may be evicted')
  out.append(p)
 return out


def residency(files):
 if os.name!='posix' or not Path('/proc/self/status').exists():raise RuntimeError('Linux mincore required')
 libc=ctypes.CDLL(None,use_errno=True);libc.mincore.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p];libc.mincore.restype=ctypes.c_int
 pages=resident=0;page=os.sysconf('SC_PAGE_SIZE')
 for path in files:
  size=path.stat().st_size
  if not size:continue
  n=(size+page-1)//page
  with path.open('rb') as f:
   with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_COPY) as view:
    address=ctypes.addressof(ctypes.c_char.from_buffer(view));vec=(ctypes.c_ubyte*n)()
    if libc.mincore(address,size,vec):raise OSError(ctypes.get_errno(),'mincore')
    resident+=sum(bool(v&1) for v in vec);pages+=n
 return dict(files=len(files),pages=pages,resident_pages=resident,resident_fraction=resident/max(1,pages),scope='Linux file DATA pages, not metadata/SSD/controller caches')


def evict_owned(files):
 before=residency(files)
 for path in files:
  with path.open('rb') as f:
   os.fdatasync(f.fileno())  # only this owned immutable file, never global sync
   os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
 after=residency(files)
 return dict(before=before,after=after,cold_verified=after['resident_fraction']<=.01,method='per-file POSIX_FADV_DONTNEED only; no global drop_caches')
