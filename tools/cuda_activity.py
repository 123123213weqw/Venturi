"""Read-only NVML process observations for benchmark contamination checks.

No CUDA context, hardware counters, permission/monitor changes or process signals.
Sampling is not proof of exclusive use between observations.
"""

import ctypes as C
import threading
import time


class _Process(C.Structure):
    # nvmlProcessInfo_v2_t, CUDA 12.9 nvml.h (not the larger v3 struct).
    _fields_ = [
        ("pid", C.c_uint),
        ("usedGpuMemory", C.c_ulonglong),
        ("gpuInstanceId", C.c_uint),
        ("computeInstanceId", C.c_uint),
    ]


class NvmlProcesses:
    def __init__(self, uuids):
        self.library = C.CDLL("libnvidia-ml.so.1")
        self._check(self.library.nvmlInit_v2())
        lookup = self.library.nvmlDeviceGetHandleByUUID
        lookup.argtypes, lookup.restype = [C.c_char_p, C.POINTER(C.c_void_p)], C.c_int
        self.query = self.library.nvmlDeviceGetComputeRunningProcesses_v2
        self.query.argtypes, self.query.restype = [
            C.c_void_p,
            C.POINTER(C.c_uint),
            C.POINTER(_Process),
        ], C.c_int
        self.handles = {}
        for uuid in uuids:
            handle = C.c_void_p()
            self._check(lookup(uuid.encode(), C.byref(handle)))
            self.handles[uuid] = handle

    @staticmethod
    def _check(code):
        if code:
            raise RuntimeError(f"NVML process observation unavailable: {code}")

    def __call__(self, allowed):
        foreign = []
        for uuid, handle in self.handles.items():
            capacity = 128
            while True:
                count = C.c_uint(capacity)
                values = (_Process * capacity)()
                rc = self.query(handle, C.byref(count), values)
                if rc == 7:  # NVML_ERROR_INSUFFICIENT_SIZE
                    capacity = max(capacity * 2, count.value)
                    if capacity > 4096:
                        raise RuntimeError("Unbounded NVML process count")
                    continue
                self._check(rc)
                for entry in values[: count.value]:
                    if entry.pid not in allowed:
                        foreign.append(
                            dict(
                                gpu=uuid, pid=entry.pid, used_bytes=entry.usedGpuMemory
                            )
                        )
                break
        return foreign


def wait_quiet(query, allowed, *, timeout=180, stable=0.5, interval=0.05):
    began = time.monotonic()
    quiet_since = None
    observations = []
    while time.monotonic() - began < timeout:
        foreign = query(allowed)
        now = time.monotonic()
        if foreign:
            observations.append(dict(unix=time.time(), foreign=foreign))
            quiet_since = None
        else:
            quiet_since = now if quiet_since is None else quiet_since
            if now - quiet_since >= stable:
                return dict(wait_seconds=now - began, observations=observations)
        time.sleep(interval)
    raise TimeoutError("No uncontended GPU observation window; no jobs stopped")


class ObservedWindow:
    def __init__(self, query, allowed, interval=0.05):
        self.query, self.allowed, self.interval = query, set(allowed), interval
        self.observations, self.errors = [], []
        self.samples = 0
        self.stop = threading.Event()
        self.thread = None

    def sample(self):
        try:
            foreign = self.query(self.allowed)
            self.samples += 1
            if foreign:
                self.observations.append(dict(unix=time.time(), foreign=foreign))
        except Exception as exc:
            self.errors.append(repr(exc))

    def __enter__(self):
        self.sample()

        def run():
            while not self.stop.wait(self.interval):
                self.sample()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()

    @property
    def clean(self):
        return self.samples >= 2 and not self.errors and not self.observations

    def report(self):
        return dict(
            clean=self.clean,
            samples=self.samples,
            interval_s=self.interval,
            foreign_observations=self.observations,
            errors=self.errors,
            scope="Read-only NVML; no observed foreign compute contexts, not proof of exclusive use between samples",
        )
