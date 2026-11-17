"""Explicit SM70 Q5 Linear plans and opt-in, measured route calibration.

W = FP16(signed_q5 * stored_FP16_scale); X/Y FP16, accumulation FP32.
No online timing, activation quantization, implicit CPU fallback or weight shadow.
An uncalibrated adaptive shape uses the bounded cuBLAS lane, never a guessed win.
"""

import ctypes as C
import copy
from dataclasses import dataclass, field
import hashlib
import statistics
import threading
import time
import weakref

import torch

from .blocks import ResidentLinear, _no_overlap, _shape, _tensor
from .weights import Q5RowSplitWeight

BLAS_BYTES = 4 * 1024**2
DEFAULT_WORKSPACE = 64 * 1024**2
POLICY = "q5-g64-f16decode-f32acc-v1"
ROUTES = ("cublas", "scalar", "wmma")


def workspace_geometry(n, k, budget):
    if type(budget) is not int or budget < BLAS_BYTES + ((2 * k + 255) // 256 * 256):
        raise ValueError(
            "Q5 workspace budget cannot hold one decoded row plus cuBLAS scratch"
        )
    chunk = min(n, 65535, (budget - BLAS_BYTES) // (2 * k))
    while (chunk * k * 2 + 255) // 256 * 256 + BLAS_BYTES > budget:
        chunk -= 1
    return chunk, (chunk * k * 2 + 255) // 256 * 256 + BLAS_BYTES


def bind_native(backend):
    lib = backend._native
    version = lib.venturi_q5_backend_version
    version.argtypes, version.restype = [], C.c_int
    if version() != 1:
        raise RuntimeError("Unsupported Q5 backend extension; rebuild explicitly")
    for name, args in {
        "venturi_q5_fused_run": [C.c_void_p] * 5 + [C.c_int] * 5 + [C.c_void_p],
        "venturi_q5_hmma_run": [C.c_void_p] * 5 + [C.c_int] * 4 + [C.c_void_p],
        "venturi_q5_cublas_run": [C.c_void_p] * 6
        + [C.c_int] * 4
        + [C.c_void_p, C.c_size_t, C.c_int, C.c_void_p],
    }.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, C.c_int


@dataclass
class ResidentQ5:
    backend: object
    weight: Q5RowSplitWeight
    identity: dict
    selections: dict = field(default_factory=dict, repr=False)
    lock: object = field(default_factory=threading.RLock, repr=False)

    @property
    def storage_tensors(self):
        return (self.weight.codes, self.weight.high, self.weight.scales)

    def prepare(
        self, *, tokens, route="adaptive", workspace_limit_bytes=DEFAULT_WORKSPACE
    ):
        _shape(tokens, self.weight.in_features)
        if tokens > 65535:
            raise ValueError("Q5 plans require 1..65535 token rows")
        if route not in (*ROUTES, "adaptive"):
            raise ValueError("Q5 route must be adaptive/cublas/scalar/wmma")
        key = (tokens, workspace_limit_bytes)
        with self.lock:
            record = self.selections.get(key) if route == "adaptive" else None
        selected = (
            record["selected"]
            if record
            else ("cublas" if route == "adaptive" else route)
        )
        reason = (
            "calibrated-for-exact-shape-and-budget"
            if record
            else (
                "uncalibrated-safe-baseline"
                if route == "adaptive"
                else "explicit-route"
            )
        )
        if selected == "wmma" and self.weight.in_features % 64:
            raise ValueError("Q5 WMMA requires K multiple of 64")
        chunk, size = workspace_geometry(
            self.weight.out_features, self.weight.in_features, workspace_limit_bytes
        )
        return Q5Plan(
            self, tokens, selected, reason, size if selected == "cublas" else 0, chunk
        )


def upload_q5(backend, weight, *, dtype, device):
    if backend.architecture != 70 or dtype != torch.float16:
        raise ValueError("Native Q5 Linear is limited to SM70 FP16 activations")
    if not getattr(backend, "_q5_available", False):
        raise RuntimeError(
            "Q5 extension missing; rebuild the explicit operator library"
        )
    weight.validate()
    if weight.codes.device.type != "cpu":
        raise ValueError("Q5 upload requires explicit CPU planes")
    if not bool(torch.isfinite(weight.scales).all()) or bool((weight.scales < 0).any()):
        raise ValueError("Q5 scales must be finite and nonnegative")
    target = backend.validate_device(dtype, device)
    digest = hashlib.sha256()
    for plane in (weight.codes, weight.high, weight.scales):
        digest.update(memoryview(plane.detach().numpy()).cast("B"))
    with torch.cuda.device(target):
        owned = Q5RowSplitWeight(
            *(
                p.detach().to(target, copy=True)
                for p in (weight.codes, weight.high, weight.scales)
            ),
            weight.in_features,
        )
        torch.cuda.current_stream(target).synchronize()
        prop = torch.cuda.get_device_properties(target)
    bind_native(backend)
    identity = dict(
        policy=POLICY,
        n=weight.out_features,
        k=weight.in_features,
        device_uuid=str(prop.uuid),
        weights_sha256=digest.hexdigest(),
        library_sha256=hashlib.sha256(backend.library.read_bytes()).hexdigest(),
    )
    payload = ResidentQ5(backend, owned, identity)
    return ResidentLinear(
        backend,
        payload,
        payload.storage_tensors,
        dtype,
        target,
        weight.in_features,
        weight.out_features,
    )


@dataclass(frozen=True)
class Q5Plan:
    resident: ResidentQ5
    tokens: int
    kernel: str
    reason: str
    workspace_bytes: int
    chunk_rows: int

    def __post_init__(self):
        object.__setattr__(self, "_lock", threading.Lock())
        handle = None
        if self.kernel == "cublas":
            with torch.cuda.device(self.device):
                handle = self.resident.backend._native.venturi_dense_create(0)
                if not handle:
                    self.resident.backend._check(-1)
        object.__setattr__(self, "_handle", handle)
        object.__setattr__(
            self,
            "_finalizer",
            weakref.finalize(
                self, self.resident.backend._native.venturi_dense_destroy, handle
            ),
        )

    @property
    def dtype(self):
        return torch.float16

    @property
    def device(self):
        return self.resident.weight.codes.device

    @property
    def in_features(self):
        return self.resident.weight.in_features

    @property
    def out_features(self):
        return self.resident.weight.out_features

    @property
    def storage_tensors(self):
        return self.resident.storage_tensors

    @property
    def route(self):
        return "q5-" + self.kernel

    def run(self, x, *, out, workspace=None):
        _tensor(x, (self.tokens, self.in_features), self.dtype, self.device, "x")
        _tensor(out, (self.tokens, self.out_features), self.dtype, self.device, "out")
        operands = (x,) + self.storage_tensors
        if self.workspace_bytes:
            if (
                not isinstance(workspace, torch.Tensor)
                or workspace.numel() < self.workspace_bytes
            ):
                raise ValueError("Q5 workspace is too small")
            _tensor(
                workspace, (workspace.numel(),), torch.uint8, self.device, "workspace"
            )
            if workspace.data_ptr() % 256:
                raise ValueError("Q5 workspace requires 256-byte alignment")
            _no_overlap(workspace, operands)
            operands += (workspace,)
        elif workspace is not None:
            raise ValueError("This Q5 kernel uses no external workspace")
        _no_overlap(out, operands)
        b = self.resident.backend
        w = self.resident.weight
        with self._lock, torch.cuda.device(self.device):
            if not self._finalizer.alive:
                raise RuntimeError("Q5 Linear plan has been closed")
            stream = torch.cuda.current_stream(self.device).cuda_stream
            args = (
                x.data_ptr(),
                w.codes.data_ptr(),
                w.high.data_ptr(),
                w.scales.data_ptr(),
                out.data_ptr(),
                self.tokens,
                self.out_features,
                self.in_features,
                w.padded_k,
            )
            if self.kernel == "cublas":
                status = b._native.venturi_q5_cublas_run(
                    self._handle,
                    *args,
                    workspace.data_ptr(),
                    self.workspace_bytes,
                    self.chunk_rows,
                    stream,
                )
            elif self.kernel == "wmma":
                status = b._native.venturi_q5_hmma_run(*args, stream)
            else:
                status = b._native.venturi_q5_fused_run(*args, 0, stream)
            b._check(status)
        return out

    def close(self):
        with self._lock:
            self._finalizer()


def _payload(resident):
    if not isinstance(resident, ResidentLinear) or not isinstance(
        resident._payload, ResidentQ5
    ):
        raise TypeError("Explicit resident native Q5 Linear required")
    resident._backend.validate_device(resident.dtype, resident.device)
    return resident._payload


def prepare_q5(
    resident, *, tokens, route="adaptive", workspace_limit_bytes=DEFAULT_WORKSPACE
):
    """Setup only; an already prepared plan's route never changes during run/replay."""
    return _payload(resident).prepare(
        tokens=tokens, route=route, workspace_limit_bytes=workspace_limit_bytes
    )


def time_q5(plan, x, out, workspace, *, iterations=10):
    """Explicit benchmark helper, never called by prepare/run."""
    with torch.cuda.device(x.device):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        torch.cuda.synchronize(x.device)
        wall, cpu = time.perf_counter(), time.process_time()
        started = time.time()
        start.record()
        for _ in range(iterations):
            plan.run(x, out=out, workspace=workspace)
        end.record()
        end.synchronize()
    return dict(
        gpu_ms=start.elapsed_time(end) / iterations,
        wall_ms=(time.perf_counter() - wall) * 1000 / iterations,
        cpu_ms=(time.process_time() - cpu) * 1000 / iterations,
        started_unix=started,
        finished_unix=time.time(),
    )


def tune_q5(
    resident,
    x,
    *,
    workspace_limit_bytes=DEFAULT_WORKSPACE,
    warmup=3,
    iterations=10,
    repeats=5,
    minimum_gain=0.03,
):
    """Explicit calibration on caller-supplied representative activations.

    Balanced route order, full-output cuBLAS check and sampled independent CPU
    FP64 oracle. Failed routes are retained in the report and never selected.
    A cuBLAS tie wins within minimum_gain, avoiding noise-driven route changes.
    """
    r = _payload(resident)
    for v in (warmup, iterations, repeats):
        if type(v) is not int or v <= 0:
            raise ValueError("Calibration counts must be positive integers")
    if not 0 <= minimum_gain < 1:
        raise ValueError("minimum_gain must be in [0,1)")
    _tensor(x, (len(x), resident.in_features), torch.float16, resident.device, "x")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Calibration input must be finite")
    plans, scratch, results = {}, {}, {}
    out = torch.empty((len(x), resident.out_features), device=x.device, dtype=x.dtype)
    reference = torch.empty_like(out)
    try:
        for route in ROUTES:
            if route == "wmma" and resident.in_features % 64:
                results[route] = dict(eligible=False, reason="K-not-multiple-of-64")
                continue
            p = prepare_q5(
                resident,
                tokens=len(x),
                route=route,
                workspace_limit_bytes=workspace_limit_bytes,
            )
            plans[route] = p
            scratch[route] = (
                torch.empty(p.workspace_bytes, dtype=torch.uint8, device=x.device)
                if p.workspace_bytes
                else None
            )
        plans["cublas"].run(x, out=reference, workspace=scratch["cublas"])
        torch.cuda.synchronize(x.device)
        if not bool(torch.isfinite(reference).all()):
            raise ValueError("Nonfinite Q5 reference output")
        # First/middle/last rows, decoded independently of the CUDA decoder.
        ids = sorted({0, resident.out_features // 2, resident.out_features - 1})
        w = r.weight
        cpu_weight = Q5RowSplitWeight(
            w.codes[ids].cpu(), w.high[ids].cpu(), w.scales[ids].cpu(), w.in_features
        )
        expected = (
            x.cpu().double() @ cpu_weight.dequantize_reference().half().double().T
        )
        torch.testing.assert_close(
            reference[:, ids].cpu().double(),
            expected.half().double(),
            atol=0.002,
            rtol=0.002,
        )
        for route, p in plans.items():
            p.run(x, out=out, workspace=scratch[route])
            torch.cuda.synchronize(x.device)
            error = float((out.float() - reference.float()).abs().max())
            valid = bool(torch.isfinite(out).all()) and torch.allclose(
                out, reference, atol=0.002, rtol=0.002
            )
            results[route] = dict(
                eligible=True,
                correct=valid,
                max_abs_error=error,
                workspace_bytes=p.workspace_bytes,
                samples=[],
            )
            if valid:
                for _ in range(warmup):
                    p.run(x, out=out, workspace=scratch[route])
        eligible = [k for k in plans if results[k]["correct"]]
        for rep in range(repeats):
            order = eligible[rep % len(eligible) :] + eligible[: rep % len(eligible)]
            if rep % 2:
                order = order[::-1]
            for route in order:
                results[route]["samples"].append(
                    time_q5(plans[route], x, out, scratch[route], iterations=iterations)
                )
        for route in eligible:
            for metric in ("gpu_ms", "wall_ms", "cpu_ms"):
                results[route][metric] = statistics.median(
                    s[metric] for s in results[route]["samples"]
                )
        best = min(eligible, key=lambda k: results[k]["gpu_ms"])
        if results[best]["gpu_ms"] >= results["cublas"]["gpu_ms"] * (1 - minimum_gain):
            best = "cublas"
        record = dict(
            identity=r.identity.copy(),
            tokens=len(x),
            workspace_limit_bytes=workspace_limit_bytes,
            selected=best,
            minimum_gain=minimum_gain,
            candidates=results,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
        with r.lock:
            r.selections[(len(x), workspace_limit_bytes)] = copy.deepcopy(record)
        return record
    finally:
        torch.cuda.synchronize(x.device)
        for p in plans.values():
            p.close()


def export_q5_calibration(resident):
    """JSON-serializable calibration; not a portable hardware/weight qualification."""
    r = _payload(resident)
    with r.lock:
        return copy.deepcopy(list(r.selections.values()))


def load_q5_calibration(resident, records):
    """Explicit startup import, pinned to device, library, weights and policy.

    No file IO, CUDA launches or timing. Invalid records fail atomically; changing
    libraries/weights/devices requires recalibration rather than a guessed match.
    """
    r = _payload(resident)
    pending = {}
    for rec in records:
        if rec.get("identity") != r.identity:
            raise ValueError("Q5 calibration identity mismatch; recalibrate explicitly")
        t, budget, route = (
            rec.get("tokens"),
            rec.get("workspace_limit_bytes"),
            rec.get("selected"),
        )
        _shape(t, r.weight.in_features)
        workspace_geometry(r.weight.out_features, r.weight.in_features, budget)
        if (
            t > 65535
            or route not in ROUTES
            or (route == "wmma" and r.weight.in_features % 64)
        ):
            raise ValueError("Invalid Q5 calibration route or token extent")
        if not rec.get("candidates", {}).get(route, {}).get("correct"):
            raise ValueError("Q5 calibration selected an unvalidated route")
        pending[(t, budget)] = copy.deepcopy(rec)
    with r.lock:
        r.selections.update(pending)


class Q5Dispatcher:
    """Prebound finite load buckets; run only looks up M, with no allocation/timing.

    Prepare after explicit tune_q5 calls. Unknown M fails instead of padding work,
    guessing a kernel or preparing a CUDA handle inside graph capture.
    """

    def __init__(
        self,
        resident,
        token_counts,
        *,
        route="adaptive",
        workspace_limit_bytes=DEFAULT_WORKSPACE,
    ):
        counts = tuple(dict.fromkeys(token_counts))
        if not counts:
            raise ValueError("At least one token bucket is required")
        self.plans = {}
        try:
            for t in counts:
                self.plans[t] = prepare_q5(
                    resident,
                    tokens=t,
                    route=route,
                    workspace_limit_bytes=workspace_limit_bytes,
                )
        except Exception:
            self.close()
            raise
        self.workspace_bytes = max(p.workspace_bytes for p in self.plans.values())
        self.routes = {t: p.route for t, p in self.plans.items()}

    def run(self, x, *, out, workspace=None):
        if x.ndim != 2 or len(x) not in self.plans:
            raise ValueError("Q5 load bucket was not explicitly prepared")
        p = self.plans[len(x)]
        return p.run(x, out=out, workspace=workspace if p.workspace_bytes else None)

    def close(self):
        for p in self.plans.values():
            p.close()
