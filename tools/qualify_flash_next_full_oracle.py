"""Tiny generic full-source assembly before any real-model/production lease."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from flash_next_full_oracle import make_full_oracle, run_oracle_step, assert_full_state
from venturi.models.flash_next_expert_policy import bind_fp8_expert_policy
from venturi.models.flash_next_loader import load_weight_decoded_fp8_text
from venturi.models.flash_next_ngram import PreparedNGramTable
from venturi.models.flash_next_request import FlashNextRequest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ["upstream-modeling", "upstream-cache", "upstream-sdpa", "output"]:
        p.add_argument("--" + flag, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--library")
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    device = torch.device(a.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            1024 * 2**20 / torch.cuda.get_device_properties(device).total_memory, device
        )
    sys.path.insert(0, str(Path(__file__).parents[1] / "tests"))
    from test_flash_next_fp8_loader import fixture
    from test_flash_next_loader import OPTIONS

    records = []
    for dtype in (
        (torch.float32, torch.float16)
        if device.type == "cuda"
        else (torch.float32, torch.bfloat16)
    ):
        cp, _, _ = fixture(out / str(dtype).split(".")[-1], execution_dtype=dtype)
        loaded = load_weight_decoded_fp8_text(
            cp, **dict(OPTIONS, execution_device=str(device), execution_dtype=dtype)
        )
        policy = bind_fp8_expert_policy(
            loaded.core,
            cp,
            backend="cuda-byte-decode" if device.type == "cuda" else "cpu-vectorized",
            retained_bytes_per_layer=(1024, 1024),
            library=a.library,
        )
        with PreparedNGramTable(loaded.tables[1]) as reader:
            oracle, head, Cache, flags = make_full_oracle(
                cp,
                loaded.core,
                reader,
                modeling=a.upstream_modeling,
                cache_source=a.upstream_cache,
                sdpa_source=a.upstream_sdpa,
            )
            request, cache = FlashNextRequest(loaded.core, reader), Cache()
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                for ids in ([2, 31, 5], [3], [7, 8]):
                    proposal = request.preview(ids)
                    expected = run_oracle_step(oracle, head, cache, ids)
                    torch.testing.assert_close(
                        proposal.logits, expected, atol=0, rtol=0
                    )
                    request.commit(proposal)
                    assert_full_state(request.state, cache)
                    records.append(
                        dict(
                            device=str(device),
                            dtype=str(dtype),
                            tokens=ids,
                            all_logits_exact=True,
                            all_states_exact=True,
                            expert_policy=policy,
                        )
                    )
    report = dict(
        scope=__doc__,
        cases=records,
        torch=str(torch.__version__),
        actual_model_loaded=False,
        throughput_measured=False,
        static_weights_alias_read_only=True,
    )
    (out / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
