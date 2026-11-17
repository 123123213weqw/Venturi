"""Optional native SM70 Q5 routing control; never called from the token loop."""

import ctypes as C

ROUTES = ("cublas", "scalar", "wmma")
MODES = ("legacy", "cublas", "adaptive")


def _bind(model):
    native = model._native
    signatures = {
        "configure": [C.c_void_p, C.c_int, C.c_uint64],
        "calibrate": [
            C.c_void_p,
            C.POINTER(C.c_int),
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_double,
        ],
        "info": [C.c_void_p, C.POINTER(C.c_uint64), C.POINTER(C.c_double), C.c_int],
    }
    result = {}
    for name, args in signatures.items():
        fn = getattr(native, "venturi_qwen_q5_" + name, None)
        if fn is None:
            raise RuntimeError("Qwen Q5 routing extension missing; rebuild explicitly")
        fn.argtypes, fn.restype = args, C.c_int
        result[name] = fn
    if not model._finalizer.alive:
        raise RuntimeError("Native Qwen model closed")
    model._metadata()
    version = (C.c_uint64 * 12)()
    model._runtime._check(result["info"](model._handle, version, None, 0))
    if version[0] != 2:
        raise RuntimeError(
            "Qwen Q5 head-only qualification ABI mismatch; rebuild explicitly"
        )
    return result


def configure(model, route, workspace_limit_bytes=256 * 1024**2):
    """Prepare before sessions, or switch a quiescent model with the same budget.

    Switching resets counters, not request state. Caller must reset requests for
    matched A/B work. Native mutex serializes this with every model forward.
    """
    if route not in MODES:
        raise ValueError("Q5 route must be legacy/cublas/adaptive")
    if (
        type(workspace_limit_bytes) is not int
        or not 8 * 1024**2 <= workspace_limit_bytes <= 1024**3
    ):
        raise ValueError("Q5 workspace budget must be 8..1024 MiB")
    with model._lock:
        fn = _bind(model)
        model._runtime._check(
            fn["configure"](model._handle, MODES.index(route), workspace_limit_bytes)
        )
    return info(model)


def calibrate(model, token_rows, *, repeats=5, iterations=5, minimum_gain=0.03):
    """Startup-only shape calibration using the actual resident vocabulary head.

    All-output cuBLAS checks use atol=rtol=.002. Independent operator-oracle and
    held-out whole-model correctness tests are still required. Scalar is only a
    candidate at M<=8; WMMA requires K%64==0. Whole-Qwen policy
    permits only the head at M<=8. All state-affecting projections stay on
    cuBLAS after held-out recurrent-state regressions. No cross-process cache is imported.
    """
    if (
        not isinstance(token_rows, (list, tuple))
        or not 1 <= len(token_rows) <= 128
        or any(
            type(t) is not int or not 1 <= t <= model._max_tokens for t in token_rows
        )
        or len(set(token_rows)) != len(token_rows)
    ):
        raise ValueError("Unique calibration rows must fit model workspace")
    if (
        type(repeats) is not int
        or not 1 <= repeats <= 31
        or type(iterations) is not int
        or not 1 <= iterations <= 100
    ):
        raise ValueError("Invalid calibration repeat/iteration count")
    if not 0 <= minimum_gain < 1:
        raise ValueError("minimum_gain must be in [0,1)")
    with model._lock:
        fn = _bind(model)
        values = (C.c_int * len(token_rows))(*token_rows)
        model._runtime._check(
            fn["calibrate"](
                model._handle, values, len(values), repeats, iterations, minimum_gain
            )
        )
    return info(model)


def info(model):
    with model._lock:
        fn = _bind(model)
        values = (C.c_uint64 * 12)()
        model._runtime._check(fn["info"](model._handle, values, None, 0))
        records = (C.c_double * (11 * values[3]))()
        model._runtime._check(fn["info"](model._handle, values, records, values[3]))
        rows = []
        for i in range(values[3]):
            row = records[i * 11 : (i + 1) * 11]
            rows.append(
                dict(
                    n=int(row[0]),
                    k=int(row[1]),
                    tokens=int(row[2]),
                    selected=ROUTES[int(row[3])],
                    routes={
                        r: dict(
                            eligible_and_correct=bool(int(row[4]) & (1 << j)),
                            model_qualified=j == 0 or int(row[2]) <= 8,
                            gpu_ms=row[5 + j],
                            max_abs_error=row[8 + j],
                        )
                        for j, r in enumerate(ROUTES)
                    },
                )
            )
        return dict(
            version=values[0],
            mode=MODES[values[1]],
            workspace_bytes=values[2],
            calls=dict(zip(ROUTES, values[4:7])),
            token_rows=dict(zip(ROUTES, values[7:10])),
            uncalibrated_fallback_calls=values[10],
            guarded_cublas_calls=values[11],
            calibration=rows,
            policy="q5-g64-f16decode-f32acc-qwen-head-smallm-v2",
            adaptive_max_token_rows=8,
            scope="Head-only model-local (N,K,M) table; trunk always cuBLAS; no online tuning",
        )
