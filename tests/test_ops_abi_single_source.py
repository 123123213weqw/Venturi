"""One operator ABI number, asserted across every place that restates it.

The loaded library already refuses a mismatched ABI at run time. That check
cannot fail on a CPU-only runner, and it cannot see a half-finished version
bump in the sources. This guard is static: it reads text only, imports nothing
from Venturi, needs no compiler, no CUDA device and no built library.

What it pins:

  ops_api.h              the definition
  blocks.cu              returns that definition, not a second literal
  blocks.py              loader guard agrees
  test_native_split.py   runtime guard agrees

A version bump that touches fewer than all of these now fails here instead of
surfacing as a load-time refusal on a GPU host.
"""

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
HEADER = ROOT / "src/venturi/ops/csrc/ops_api.h"
IMPLEMENTATION = ROOT / "src/venturi/ops/csrc/blocks.cu"
LOADER = ROOT / "src/venturi/ops/blocks.py"
RUNTIME_GUARD = ROOT / "tests/test_native_split.py"


def _exactly_one(pattern, path, description):
    matches = re.findall(pattern, path.read_text(), re.MULTILINE)
    assert len(matches) == 1, (
        f"expected exactly one {description} in {path.relative_to(ROOT)}, "
        f"found {len(matches)}"
    )
    return int(matches[0])


def header_abi():
    return _exactly_one(
        r"^#define VENTURI_OPS_ABI (\d+)$", HEADER, "VENTURI_OPS_ABI definition"
    )


def test_header_declares_a_single_abi_number():
    assert header_abi() > 0


def test_implementation_returns_the_header_macro_not_a_literal():
    source = IMPLEMENTATION.read_text()
    body = re.search(
        r'extern "C" int venturi_blocks_abi\(\)\s*\{([^}]*)\}', source
    )
    assert body is not None, "venturi_blocks_abi definition not found"
    assert "VENTURI_OPS_ABI" in body.group(1), (
        "venturi_blocks_abi must return VENTURI_OPS_ABI; a separate literal can "
        "disagree with ops_api.h without any build or test failing"
    )
    assert not re.search(r"\breturn\s+\d+\s*;", body.group(1)), (
        "venturi_blocks_abi returns a numeric literal instead of VENTURI_OPS_ABI"
    )


def test_loader_guard_agrees_with_the_header():
    expected = header_abi()
    actual = _exactly_one(
        r"venturi_blocks_abi\(\)\s*!=\s*(\d+)", LOADER, "loader ABI guard"
    )
    assert actual == expected, (
        f"blocks.py guard expects ABI {actual} but ops_api.h defines {expected}"
    )


def test_runtime_guard_agrees_with_the_header():
    expected = header_abi()
    actual = _exactly_one(
        r"venturi_blocks_abi\(\)\s*==\s*(\d+)", RUNTIME_GUARD, "runtime ABI guard"
    )
    assert actual == expected, (
        f"test_native_split.py expects ABI {actual} but ops_api.h defines {expected}"
    )


def test_header_is_a_tracked_build_dependency():
    """A bump must invalidate the build cache, so blocks.cu has to include it."""
    assert '#include "ops_api.h"' in IMPLEMENTATION.read_text()
