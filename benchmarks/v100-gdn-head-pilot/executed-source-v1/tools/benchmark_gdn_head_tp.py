"""Native two-V100 GDN sublayer pilot, not whole-model or serving TPS.

Includes input norm, QKV/Z/A/B, convolution, gates, independent recurrence,
gated norm, output projection and residual. Both arms have primary-only FP16
input/output; distributed mutable state is joined only in untimed diagnostics.
"""

import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
from pathlib import Path
import resource
import statistics
import subprocess
import sys
from time import perf_counter, time
from unittest.mock import patch

import torch
from safetensors import safe_open

from gdn_head_pilot import Parts, Pilot
from venturi.models import QwenQ8Weights
from venturi.ninfer import NInferArtifact

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from qwen_gdn_reference import gdn_layer_oracle  # noqa: E402
from qualify_qwen_projection_variant import tensor_signature  # noqa: E402
from test_gdn_head_pilot import error, inputs, validate  # noqa: E402


def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def represented(source):
    """Independent original-source reconstruction, not the runtime/export helper."""
    v = source.double()
    groups = v.reshape(v.shape[0], -1, 32)
    scales = (groups.abs().amax(-1) / 127).half().double()
    divisor = torch.where(scales > 0, scales, torch.ones_like(scales))
    return (
        torch.round(groups / divisor[:, :, None]).clamp(-127, 127) * scales[:, :, None]
    ).reshape(v.shape)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--source", required=True)
    p.add_argument("--blocks", required=True)
    p.add_argument("--host-library", required=True)
    p.add_argument("--output-directory", required=True)
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    out = Path(a.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    w = QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact))
    layers = [
        i
        for i, kind in enumerate(w.config["layer_types"])
        if kind == "linear_attention"
    ]
    selected = [layers[0]] if a.profile else [layers[0], layers[-1]]
    report = dict(
        scope=__doc__,
        completed=False,
        profiled=a.profile,
        cells=[],
        model_integration=False,
        model_tps_claim=False,
        achieved_hardware_counters=None,
        artifact_id=w.artifact.artifact_id,
        blocks_sha256=hashlib.sha256(Path(a.blocks).read_bytes()).hexdigest(),
        host_library_sha256=hashlib.sha256(
            Path(a.host_library).read_bytes()
        ).hexdigest(),
        precision="Unchanged stored Q8/F16 scales and two residual-A8/F32-scale passes; original small-parameter flags, FP16 activations/KV-independent operator, packed FP32 GDN state. Head-local output uses FP32 column partial/transport/sum before one FP16 cast: changed association.",
    )

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    sampler = None
    logfile = (out / "gpu-samples.csv").open("wb")
    try:
        if not a.profile:
            sampler = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem",
                    "--format=csv,noheader,nounits",
                    "-lms",
                    "100",
                ],
                stdout=logfile,
                stderr=subprocess.STDOUT,
            )
        for layer in selected:
            parts = Parts.from_weights(w, layer)
            source = Path(a.source)
            index = json.loads((source / "model.safetensors.index.json").read_text())[
                "weight_map"
            ]
            prefix = f"model.language_model.layers.{layer}."
            names = [
                name
                for name in index
                if name.startswith(prefix + "linear_attn.")
                or name == prefix + "input_layernorm.weight"
            ]
            raw = {}
            for name in names:
                with safe_open(
                    str(source / index[name]), framework="pt", device="cpu"
                ) as f:
                    raw[name] = f.get_tensor(name)
            cache = {}

            def quantizer(value):
                key = (value.data_ptr(), tuple(value.shape))
                if key not in cache:
                    cache[key] = represented(value)
                return cache[key]

            for sequences, steps in [(1, 1), (1, 3), (8, 1)]:
                x, hist, state = inputs(parts, sequences, steps, seed=layer + 411)
                plans = {
                    mode: Pilot(
                        parts,
                        a.blocks,
                        a.host_library,
                        mode=mode,
                        sequences=sequences,
                        steps=steps,
                        inputs=x,
                        histories=hist,
                        states=state,
                    )
                    for mode in ("row", "head")
                }
                try:
                    for plan in plans.values():
                        plan.run()
                    row, head = validate(plans["row"], plans["head"])
                    comparisons = []
                    for s in range(sequences):
                        ideal, ih, ist, trace = gdn_layer_oracle(
                            raw,
                            w.config,
                            x[s * steps : (s + 1) * steps],
                            hist[s],
                            state[s],
                            layer=layer,
                            only_gdn=True,
                            quantizer=quantizer,
                        )
                        actual = head["projected"][s * steps : (s + 1) * steps]
                        l2 = error(actual, ideal)
                        cosine = float(
                            torch.nn.functional.cosine_similarity(
                                actual.double().flatten(),
                                ideal.double().flatten(),
                                dim=0,
                            )
                        )
                        assert l2 <= 0.02 and cosine >= 0.999, (
                            "Original-source block quality gate failed"
                        )
                        comparisons.append(
                            dict(
                                relative_l2=l2,
                                cosine=cosine,
                                source="Independent original-source represented-Q8/FP64 staged GDN equations; no BF16 corpus-quality claim",
                            )
                        )
                    cell = dict(
                        layer=layer,
                        sequences=sequences,
                        tokens_per_sequence=steps,
                        total_rows=sequences * steps,
                        prefix_count=2 if steps == 3 else 0,
                        exact_pre_projection_state_and_history=True,
                        output_relative_l2_vs_row=error(
                            head["projected"], row["projected"]
                        ),
                        independent_reference=comparisons,
                        final_state_signature=tensor_signature(head["state"]),
                        trials=[],
                    )
                    for trial in range(1 if a.profile else 3):
                        for mode in ["row", "head"] if trial != 1 else ["head", "row"]:
                            plan = plans[mode]
                            plan.reset()
                            plan.run(8)
                            plan.reset()
                            begin, start, used = time(), perf_counter(), cpu()

                            def forbidden(*args, **kwargs):
                                raise AssertionError(
                                    "Python numerical fallback in timed native GDN block"
                                )

                            with ExitStack() as stack:
                                for name in (
                                    "matmul",
                                    "mm",
                                    "empty",
                                    "zeros",
                                    "cat",
                                    "tensor",
                                ):
                                    stack.enter_context(
                                        patch.object(torch, name, forbidden)
                                    )
                                with torch.cuda.nvtx.range(
                                    f"gdn_{layer}_{sequences}x{steps}_{mode}"
                                ):
                                    result = plan.run(32 if a.profile else 2000)
                            elapsed, used, end = (
                                perf_counter() - start,
                                cpu() - used,
                                time(),
                            )
                            cell["trials"].append(
                                dict(
                                    mode=mode,
                                    trial=trial + 1,
                                    start_unix=begin,
                                    end_unix=end,
                                    host_wall_seconds=elapsed,
                                    process_cpu_core_seconds=used,
                                    **result,
                                )
                            )
                    # Equal recurrence count for both arms after the final trial.
                    validate(plans["row"], plans["head"])
                    medians = {
                        mode: statistics.median(
                            t["native_wall_seconds"] / t["iterations"]
                            for t in cell["trials"]
                            if t["mode"] == mode
                        )
                        for mode in ("row", "head")
                    }
                    cell.update(
                        median_block_seconds=medians,
                        speedup=medians["row"] / medians["head"],
                    )
                    report["cells"].append(cell)
                    save()
                    print(
                        json.dumps(
                            {
                                k: cell[k]
                                for k in (
                                    "layer",
                                    "sequences",
                                    "tokens_per_sequence",
                                    "speedup",
                                    "median_block_seconds",
                                )
                            }
                        ),
                        flush=True,
                    )
                finally:
                    for plan in plans.values():
                        plan.close()
                    del plan, plans
                    gc.collect()
            cache.clear()
            del raw, parts
            gc.collect()
        if not a.profile:
            t3 = [
                c["speedup"] for c in report["cells"] if c["tokens_per_sequence"] == 3
            ]
            others = [
                c["speedup"] for c in report["cells"] if c["tokens_per_sequence"] == 1
            ]
            report["integration_gate"] = dict(
                t3_at_least_110=all(v >= 1.10 for v in t3),
                no_t1_or_independent8_regression_above_two_percent=all(
                    v >= 0.98 for v in others
                ),
                passed=all(v >= 1.10 for v in t3) and all(v >= 0.98 for v in others),
            )
        report["completed"] = True
        save()
    finally:
        if sampler is not None:
            sampler.terminate()
            sampler.wait(timeout=20)
        logfile.close()


if __name__ == "__main__":
    main()
