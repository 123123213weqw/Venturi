import importlib
import json
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["venturi", "venturi.model_assets", "venturi.text"])
def test_metadata_import_does_not_load_tensor_framework(module):
    code = f"""import importlib,sys,json
importlib.import_module({module!r})
print(json.dumps(dict(torch="torch" in sys.modules,safetensors="safetensors" in sys.modules)))
"""
    result = json.loads(
        subprocess.check_output([sys.executable, "-c", code], text=True)
    )
    assert result == dict(torch=False, safetensors=False)


def test_public_checkpoint_symbols_keep_identity_and_discoverability():
    import venturi

    for name, module in venturi._EXPORTS.items():
        assert name in venturi.__all__ and name in dir(venturi)
        value = getattr(venturi, name)
        assert value is getattr(importlib.import_module(module, "venturi"), name)
        assert getattr(venturi, name) is value
    with pytest.raises(AttributeError, match="has no attribute"):
        venturi.__nonexistent_public_api__
