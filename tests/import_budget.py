"""Bounded startup watchdog for isolated framework imports, not a speed criterion."""

import os


def framework_import_timeout(default=30):
    raw = os.environ.get("VENTURI_FRAMEWORK_IMPORT_TIMEOUT")
    if raw is None:
        return default
    value = int(raw)
    if not 30 <= value <= 300:
        raise ValueError("Framework import timeout must be 30..300 seconds")
    return value
