"""Public checkpoint API, loaded on demand.

Asset admission and command metadata must not initialize a tensor framework
merely by importing their package. Existing public checkpoint symbols
retain their original identities; requesting them loads the normal modules.
"""

from importlib import import_module

__all__ = [
    "Checkpoint",
    "CheckpointError",
    "TensorSpec",
    "LoadedCheckpoint",
    "LoadStats",
    "inspect_checkpoint",
    "load_checkpoint",
    "VerificationStats",
    "verify_loaded_checkpoint",
]

_EXPORTS = {
    **{name: ".checkpoint" for name in ("Checkpoint", "CheckpointError", "TensorSpec", "inspect_checkpoint")},
    **{name: ".loader" for name in ("LoadedCheckpoint", "LoadStats", "load_checkpoint")},
    **{name: ".verification" for name in ("VerificationStats", "verify_loaded_checkpoint")},
}


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
