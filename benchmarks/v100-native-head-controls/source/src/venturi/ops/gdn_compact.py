"""Explicit research GDN launch geometry; original public prepare_gdn is unchanged."""

from dataclasses import replace
from .gdn import prepare_gdn


def prepare_gdn_compact(backend, **configuration):
    if backend.name != "venturi-blocks-cuda":
        raise ValueError("Compact GDN requires an explicit native CUDA backend")
    plan = prepare_gdn(backend, **configuration)
    try:
        fn = backend._native.venturi_gdn_compact_run
    except AttributeError as e:
        raise RuntimeError("Compact GDN export absent; rebuild explicitly") from e
    fn.argtypes = backend._native.venturi_gdn_run.argtypes
    fn.restype = backend._native.venturi_gdn_run.restype
    return replace(plan, _native_entry="venturi_gdn_compact_run")
