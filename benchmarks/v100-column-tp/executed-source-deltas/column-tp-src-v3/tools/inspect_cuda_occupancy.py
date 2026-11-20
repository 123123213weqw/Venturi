"""Read compiled CUDA residency limits and join them to existing native traces.

This tool does not launch kernels, collect hardware counters or measure achieved
occupancy. It loads cubins extracted from the exact hashed library and uses the
driver occupancy API, including its register-allocation rules. Trace durations
are instrumented attribution, never unprofiled throughput or occupancy weights.
"""

import argparse
from collections import defaultdict
import ctypes as ct
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import tempfile


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resource_symbols(text):
    return sorted(set(re.findall(r"^ Function ([^:\s]+):", text, re.MULTILINE)))


def compiled_launch_threads(symbol):
    """Fixed launch geometry of known operators, not an occupancy recommendation."""
    if "3vct3sum" in symbol:
        return 256
    match = re.search(r"3vwt7group32ILi([1248])E", symbol)
    if match:
        return int(match.group(1)) * 32
    if "q8_warp_exact" in symbol or "3vdq3dotI" in symbol:
        return 128
    if "q8_tiled" in symbol or "3vdq8quantize" in symbol:
        return 256
    return None


def compiled_launch_geometry(symbol):
    """Known fixed K-tile shared bytes must accompany the thread count."""
    threads = compiled_launch_threads(symbol)
    if threads is not None:
        return threads, 0
    partial = re.search(r"3vct7partialILi(1|4|8)ELi(4|8|16)ELb([01])E", symbol)
    if partial:
        tokens, rows, staged = map(int, partial.groups())
        return rows * 32, tokens * 2 * 32 * 36 if staged else 0
    staged = re.search(r"3(?:vdt|vqf)12staged16_dotILi([12])ELi8E", symbol)
    if staged:
        return 256, 8 * int(staged[1]) * 32 * 36
    match = re.search(
        r"3(?:vdt3dot|vqf9tiled_dot|vqf16paired_tiled_dot)ILi(3|4|8|16)ELi([12])ELi(4|8|16)E",
        symbol,
    )
    if match:
        tokens, passes, rows = map(int, match.groups())
        return rows * 32, tokens * passes * 32 * 36
    return None


def grid_limits(grid, threads, resident_blocks, device):
    """Static per-kernel ceilings; not time-averaged or achieved occupancy."""
    if any(type(v) is not int or v <= 0 for v in (grid, threads, resident_blocks)):
        raise ValueError("Positive integer grid, threads and resident blocks required")
    warp = device["warp_size"]
    warps = (threads + warp - 1) // warp
    max_warps = device["threads_per_sm"] // warp
    capacity = resident_blocks * device["sm_count"]
    return dict(
        theoretical_resident_blocks_per_sm=resident_blocks,
        theoretical_active_warps_per_sm=resident_blocks * warps,
        theoretical_occupancy_fraction=resident_blocks * warps / max_warps,
        grid_capacity_waves=grid / capacity,
        grid_limited_device_average_residency_upper_bound=min(
            resident_blocks * warps / max_warps,
            grid * warps / (device["sm_count"] * max_warps),
        ),
    )


def trace_launches(path):
    """Keep request-thread/correlation ownership, rather than whole-process sums."""
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        strings = dict(db.execute("select id,value from StringIds"))
        kernels = list(db.execute("select * from CUPTI_ACTIVITY_KIND_KERNEL"))
        if len({r["globalPid"] for r in kernels}) != 1:
            raise ValueError("Expected one profiled process")
        tables = {
            r[0]
            for r in db.execute("select name from sqlite_master where type='table'")
        }
        apis = []
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            if table in tables:
                apis.extend(db.execute(f"select * from {table}"))
        ranges = defaultdict(list)
        for e in db.execute("select * from NVTX_EVENTS where end is not null"):
            name = e["text"] or strings.get(e["textId"])
            if name in ("native_ttft", "native_decode", "native_unsplit"):
                ranges[name].append(e)
        if not ranges:
            raise ValueError("No native request ranges found")
        result = []
        for name, events in ranges.items():
            ids = {
                a["correlationId"]
                for a in apis
                for e in events
                if a["globalTid"] == e["globalTid"]
                and e["start"] <= a["start"]
                and a["end"] <= e["end"]
            }
            groups = defaultdict(lambda: dict(launches=0, kernel_nanoseconds=0))
            for k in kernels:
                if k["correlationId"] not in ids:
                    continue
                key = (
                    strings[k["mangledName"]],
                    strings[k["demangledName"]],
                    k["deviceId"],
                    k["blockX"] * k["blockY"] * k["blockZ"],
                    k["gridX"] * k["gridY"] * k["gridZ"],
                    k["registersPerThread"],
                    k["staticSharedMemory"],
                    k["dynamicSharedMemory"],
                    k["localMemoryPerThread"],
                    k["cacheConfig"],
                )
                groups[key]["launches"] += 1
                groups[key]["kernel_nanoseconds"] += k["end"] - k["start"]
            rows = []
            for key, values in sorted(
                groups.items(), key=lambda x: -x[1]["kernel_nanoseconds"]
            ):
                fields = (
                    "symbol",
                    "name",
                    "trace_device_id",
                    "threads",
                    "grid_blocks",
                    "trace_registers_per_thread",
                    "trace_static_shared_bytes",
                    "dynamic_shared_bytes",
                    "trace_local_bytes_per_thread",
                    "trace_cache_config",
                )
                rows.append(dict(zip(fields, key), **values))
            total = sum(r["kernel_nanoseconds"] for r in rows)
            result.append(
                dict(name=name, launches=rows, summed_kernel_nanoseconds=total)
            )
        return result
    finally:
        db.close()


class Driver:
    """Small typed subset of the CUDA driver API; no Torch/runtime dependency."""

    def __init__(self):
        self.lib = ct.CDLL("libcuda.so.1")
        self.init = self.bind("cuInit", [ct.c_uint])
        self.version = self.bind("cuDriverGetVersion", [ct.POINTER(ct.c_int)])
        self.device_get = self.bind("cuDeviceGet", [ct.POINTER(ct.c_int), ct.c_int])
        self.device_name = self.bind(
            "cuDeviceGetName", [ct.c_char_p, ct.c_int, ct.c_int]
        )
        self.device_attr = self.bind(
            "cuDeviceGetAttribute", [ct.POINTER(ct.c_int), ct.c_int, ct.c_int]
        )
        self.retain = self.bind(
            "cuDevicePrimaryCtxRetain", [ct.POINTER(ct.c_void_p), ct.c_int]
        )
        self.release = self.bind("cuDevicePrimaryCtxRelease_v2", [ct.c_int])
        self.current = self.bind("cuCtxSetCurrent", [ct.c_void_p])
        self.load = self.bind("cuModuleLoad", [ct.POINTER(ct.c_void_p), ct.c_char_p])
        self.unload = self.bind("cuModuleUnload", [ct.c_void_p])
        self.function = self.bind(
            "cuModuleGetFunction", [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p]
        )
        self.function_attr = self.bind(
            "cuFuncGetAttribute", [ct.POINTER(ct.c_int), ct.c_int, ct.c_void_p]
        )
        self.occupancy = self.bind(
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
            [ct.POINTER(ct.c_int), ct.c_void_p, ct.c_int, ct.c_size_t],
        )
        self.check(self.init(0))

    def bind(self, name, args):
        f = getattr(self.lib, name)
        f.argtypes, f.restype = args, ct.c_int
        return f

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"CUDA driver API returned error {code}")

    def integer(self, func, *args):
        value = ct.c_int()
        self.check(func(ct.byref(value), *args))
        return value.value

    def hardware(self, ordinal):
        dev = self.integer(self.device_get, ordinal)
        name = ct.create_string_buffer(256)
        self.check(self.device_name(name, len(name), dev))
        attributes = dict(
            sm_count=16,
            warp_size=10,
            threads_per_sm=39,
            registers_per_sm=82,
            shared_bytes_per_sm=81,
            blocks_per_sm=106,
            l2_bytes=38,
            compute_major=75,
            compute_minor=76,
        )
        return dev, dict(
            ordinal=ordinal,
            name=name.value.decode(),
            **{
                k: self.integer(self.device_attr, attr, dev)
                for k, attr in attributes.items()
            },
        )

    def attributes(self, func, threads, dynamic):
        fields = dict(
            max_threads_per_block=0,
            static_shared_bytes=1,
            local_bytes_per_thread=3,
            registers_per_thread=4,
            binary_version=6,
        )
        attributes = {
            k: self.integer(self.function_attr, v, func) for k, v in fields.items()
        }
        blocks = self.integer(self.occupancy, func, threads, dynamic)
        return attributes, blocks


def inspect(library, traces, devices, cuobjdump, expected_sha256):
    digest = sha256(library)
    if digest != expected_sha256:
        raise ValueError(
            "Library hash mismatch; refusing to infer occupancy from a different binary"
        )
    resources = subprocess.run(
        [cuobjdump, "-res-usage", str(library)],
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    symbols = resource_symbols(resources)
    launches = {label: trace_launches(path) for label, path in traces.items()}
    driver = Driver()
    result = dict(
        library=str(library),
        library_sha256=digest,
        driver_version=driver.integer(driver.version),
        achieved_occupancy=None,
        hardware_counter_collection=False,
        kernel_launches_by_this_tool=0,
        resource_usage=resources,
        trace_files={k: dict(path=str(v), sha256=sha256(v)) for k, v in traces.items()},
        devices=[],
        limitations=[
            "CUDA occupancy API results are theoretical residency, NOT measured achieved occupancy.",
            "Recorded kernel durations are from previous separate instrumented runs, not fresh throughput timings.",
            "Grid ceilings ignore load balance, tail effects, dependencies, memory/issue stalls and other processes.",
            "The tool never changes launch geometry; several kernels require a fixed thread count for correctness.",
            "Local bytes/stack alone do not prove register spilling; consult compiler/SASS evidence.",
            "No occupancy averaging across kernels or devices is presented as achieved occupancy.",
        ],
    )
    with tempfile.TemporaryDirectory(prefix="venturi-cubin-occupancy-") as temp:
        subprocess.run(
            [cuobjdump, "-xelf", "all", str(Path(library).resolve())],
            cwd=temp,
            text=True,
            capture_output=True,
            check=True,
        )
        cubins = sorted(Path(temp).glob("*.cubin"))
        if not cubins:
            raise ValueError("No cubins extracted")
        result["cubins"] = {p.name: sha256(p) for p in cubins}
        for ordinal in devices:
            dev, hw = driver.hardware(ordinal)
            context, modules = ct.c_void_p(), []
            driver.check(driver.retain(ct.byref(context), dev))
            try:
                driver.check(driver.current(context))
                for path in cubins:
                    module = ct.c_void_p()
                    driver.check(driver.load(ct.byref(module), str(path).encode()))
                    modules.append(module)
                functions = {}
                for symbol in symbols:
                    for module in modules:
                        func = ct.c_void_p()
                        code = driver.function(ct.byref(func), module, symbol.encode())
                        if code == 0:
                            functions[symbol] = func
                            break
                        if code != 500:  # CUDA_ERROR_NOT_FOUND
                            driver.check(code)
                measured = dict(hardware=hw, compiled_q8_functions=[], traces={})
                for symbol, func in functions.items():
                    launch = compiled_launch_geometry(symbol)
                    if launch is not None:
                        threads, dynamic = launch
                        attrs, blocks = driver.attributes(func, threads, dynamic)
                        measured["compiled_q8_functions"].append(
                            dict(
                                symbol=symbol,
                                threads=threads,
                                dynamic_shared_bytes=dynamic,
                                **attrs,
                                theoretical_resident_blocks_per_sm=blocks,
                                theoretical_occupancy_fraction=blocks
                                * threads
                                / hw["threads_per_sm"],
                            )
                        )
                for label, cohorts in launches.items():
                    output = []
                    for cohort in cohorts:
                        rows = []
                        for original in cohort["launches"]:
                            row = original.copy()
                            func = functions.get(row["symbol"])
                            if func is None:
                                row["status"] = (
                                    "Not in supplied library; no occupancy inferred"
                                )
                            else:
                                attrs, blocks = driver.attributes(
                                    func, row["threads"], row["dynamic_shared_bytes"]
                                )
                                if (
                                    attrs["registers_per_thread"]
                                    != row["trace_registers_per_thread"]
                                    or attrs["static_shared_bytes"]
                                    != row["trace_static_shared_bytes"]
                                ):
                                    raise ValueError(
                                        "Trace/library kernel resource mismatch"
                                    )
                                row.update(
                                    status="Exact symbol and trace register/shared usage match",
                                    **attrs,
                                    **grid_limits(
                                        row["grid_blocks"], row["threads"], blocks, hw
                                    ),
                                )
                            rows.append(row)
                        output.append(
                            dict(
                                name=cohort["name"],
                                launches=rows,
                                summed_kernel_nanoseconds=cohort[
                                    "summed_kernel_nanoseconds"
                                ],
                            )
                        )
                    measured["traces"][label] = output
                result["devices"].append(measured)
            finally:
                for module in reversed(modules):
                    driver.check(driver.unload(module))
                driver.check(driver.current(None))
                driver.check(driver.release(dev))
    if sha256(library) != digest:
        raise ValueError("Library changed during inspection")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", required=True, type=Path)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument(
        "--trace", action="append", default=[], help="label=Nsight-SQLite-path"
    )
    p.add_argument("--devices", default="0,1")
    p.add_argument("--cuobjdump", default="/usr/local/cuda/bin/cuobjdump")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    traces = dict(item.split("=", 1) for item in a.trace)
    if len(traces) != len(a.trace):
        p.error("Trace labels must be unique")
    devices = [int(x) for x in a.devices.split(",")]
    if not devices or len(set(devices)) != len(devices) or min(devices) < 0:
        p.error("Device ordinals must be unique and nonnegative")
    if a.output.exists():
        p.error("Output already exists; preserve previous evidence")
    result = inspect(
        a.library.resolve(), traces, devices, a.cuobjdump, a.expected_sha256
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
