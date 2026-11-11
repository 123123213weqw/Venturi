"""ctypes ABI descriptors and native symbol binding for the Qwen runtime.

Split out of qwen_native so the load-planning modules can share the exact
descriptor definitions without importing the model class. qwen_native
re-exports every name here, so historical import sites are unchanged.
"""

import ctypes as C

from ..qwen_metadata import source_digest


class Config(C.Structure):
    _fields_ = (
        [
            (k, C.c_int)
            for k in (
                "device",
                "dtype",
                "layers",
                "hidden",
                "intermediate",
                "vocab",
                "hq",
                "hkv",
                "dim",
                "rotary",
                "gqk",
                "gvh",
                "gkd",
                "gvd",
                "convolution",
                "max_tokens",
                "max_position",
                "resident",
                "weight_bits",
                "weight_group",
            )
        ]
        + [(k, C.c_float) for k in ("epsilon", "gdn_scale", "attention_scale")]
        + [
            ("second_device", C.c_int),
            ("peer_transport", C.c_int),
            ("reserve", C.c_uint64),
        ]
    )


class Span(C.Structure):
    _fields_ = [("path", C.c_char_p)] + [
        (k, C.c_uint64) for k in ("logical", "file_offset", "bytes", "file_bytes")
    ]


class Matrix(C.Structure):
    _fields_ = [
        ("offset", C.c_uint64),
        ("scales", C.c_uint64),
        ("n", C.c_int),
        ("k", C.c_int),
    ]


class Layer(C.Structure):
    _fields_ = (
        [(k, C.c_int) for k in ("kind", "conv_fp32", "gamma_fp32", "ab_fp32")]
        + [("matrix", Matrix * 7)]
        + [
            (k, C.c_void_p)
            for k in (
                "norm",
                "post",
                "conv",
                "rate",
                "bias",
                "gamma",
                "a",
                "b",
                "qgamma",
                "kgamma",
            )
        ]
    )


def _bind(backend):
    n = backend._native
    ptr, integer, u64 = C.c_void_p, C.c_int, C.c_uint64
    signatures = {
        "venturi_qwen_sizes": ([C.POINTER(u64)], integer),
        "venturi_qwen_create": (
            [
                C.POINTER(Config),
                C.POINTER(Span),
                integer,
                C.POINTER(Layer),
                C.POINTER(Matrix),
                C.POINTER(Matrix),
                ptr,
                ptr,
            ],
            ptr,
        ),
        "venturi_qwen_destroy": ([ptr], None),
        "venturi_qwen_session": ([ptr, integer], ptr),
        "venturi_qwen_session_destroy": ([ptr], None),
        "venturi_qwen_forward": (
            [ptr, ptr, integer, ptr, u64, integer, ptr, u64],
            integer,
        ),
        "venturi_qwen_generate": (
            [ptr, ptr, integer, integer, ptr, integer, ptr, ptr],
            integer,
        ),
        "venturi_qwen_info": ([ptr, C.POINTER(u64)], integer),
        "venturi_qwen_timings": ([ptr, C.POINTER(C.c_double)], integer),
        "venturi_qwen_parallel_info": ([ptr, C.POINTER(u64)], integer),
        "venturi_qwen_reset": ([ptr], integer),
        "venturi_qwen_snapshot": ([ptr, integer, integer, ptr, u64], integer),
        "venturi_qwen_debug_failure": ([ptr, integer, integer], integer),
    }
    fn = getattr(n, "venturi_qwen_head_gdn_session_info", None)
    if fn is not None:
        fn.argtypes, fn.restype = [ptr, C.POINTER(u64)], integer
    for name, (args, result) in signatures.items():
        fn = getattr(n, name)
        fn.argtypes, fn.restype = args, result
    stage = n.venturi_qwen_stage_create
    stage.argtypes = signatures["venturi_qwen_create"][0] + [integer, integer]
    stage.restype = ptr
    sizes = (u64 * 4)()
    backend._check(n.venturi_qwen_sizes(sizes))
    if list(sizes) != [C.sizeof(t) for t in (Config, Span, Matrix, Layer)]:
        raise RuntimeError("Native Qwen descriptor ABI mismatch; rebuild explicitly")
    return n


def _digest(weights):
    return source_digest(weights.config, weights.artifact.manifest)
