"""Generic full-source oracle assembly on small complete synthetic checkpoints."""

import sys
import os
from pathlib import Path

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from test_flash_next_fp8_loader import fixture
from test_flash_next_loader import OPTIONS
from venturi.models.flash_next_loader import load_weight_decoded_fp8_text
from venturi.models.flash_next_ngram import PreparedNGramTable
from venturi.models.flash_next_request import FlashNextRequest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from flash_next_full_oracle import make_full_oracle, run_oracle_step, assert_full_state


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_generic_source_forward_hash_state_and_all_experts(tmp_path, dtype):
    cp, _, _ = fixture(tmp_path, execution_dtype=dtype)
    loaded = load_weight_decoded_fp8_text(cp, **dict(OPTIONS, execution_dtype=dtype))
    root = Path(
        os.environ.get(
            "VENTURI_FLASH_NEXT_REFERENCE_DIR",
            Path(__file__).parents[1]
            / "benchmarks/qwen38-flash-next-generation-v1/measured-source",
        )
    )
    with PreparedNGramTable(loaded.tables[1]) as reader:
        oracle, head, Cache, flags = make_full_oracle(
            cp,
            loaded.core,
            reader,
            modeling=root / "reference.py",
            cache_source=root / "reference_cache.py",
            sdpa_source=root / "reference_sdpa.py",
        )
        request, cache = FlashNextRequest(loaded.core, reader), Cache()
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for ids in ([2, 31, 5], [3], [7, 8]):
                actual = request.preview(ids)
                expected = run_oracle_step(oracle, head, cache, ids)
                torch.testing.assert_close(actual.logits, expected, atol=0, rtol=0)
                request.commit(actual)
                assert_full_state(request.state, cache)
