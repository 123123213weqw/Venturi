"""Qualify the public single-slot SM70 Q5 unpack-reuse option on a real checkpoint."""

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import time

from cuda_activity import NvmlProcesses, ObservedWindow, wait_quiet
import torch
from venturi.models import NativeQwenModel, QwenWeights
from venturi.models.qwen.native_runtime import QwenRuntime
from venturi.ops import CudaBackend
from venturi.runtime import ContinuousScheduler


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identical(a, b):
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )
    )


def state_bits(a, b):
    result = {}
    for kind in ("gdn", "kv"):
        checks = []
        for x, y in zip(a[kind], b[kind], strict=True):
            assert (x is None) == (y is None)
            if x is not None:
                checks.extend(identical(u, v) for u, v in zip(x, y, strict=True))
        result[kind] = dict(exact=sum(checks), count=len(checks))
    return result


def capture(batch, prompt, chunk, forced=None):
    batch.reset()
    n = (len(prompt) - 1) // chunk * chunk
    for i in range(0, n, chunk):
        batch.step([prompt[i : i + chunk]], sample_slots=[])
    logits, ids = [batch.forward([prompt[n:]])], []
    for step in range(6):
        assert bool(torch.isfinite(logits[-1]).all())
        token = int(logits[-1].argmax())
        ids.append(token)
        if step < 5:
            logits.append(batch.forward([[token if forced is None else forced[step]]]))
    assert batch.info()["requests"][0]["position"] == len(prompt) + 5
    return logits, ids, batch.snapshot(0)


def replay(batch, prompts, chunk, outputs, *, snapshot=False):
    scheduler = ContinuousScheduler(
        batch, token_budget=32, prefill_chunk_size=chunk, prefill_policy="round-robin"
    )
    try:
        epoch = time.time()
        start, cpu = time.perf_counter(), time.process_time()
        tickets = [scheduler.submit(p, max_new_tokens=outputs) for p in prompts]
        while scheduler.has_work:
            scheduler.tick()
        results = [t.future.result() for t in tickets]
        wall, cpu = time.perf_counter() - start, time.process_time() - cpu
        ended = time.time()
        assert all(
            len(r.output_ids) == outputs and r.finish_reason == "length"
            for r in results
        )
        stats = scheduler.info()
        assert stats["prefill_rows"] == sum(map(len, prompts))
        assert stats["decode_rows"] == len(prompts) * (outputs - 1)
        data = dict(
            ttft_s=[r.time_to_first_token_seconds for r in results],
            max_itl_s=[r.max_inter_token_seconds for r in results],
            output_ids=[list(r.output_ids) for r in results],
            start=epoch,
            end=ended,
            host_cpu_s=cpu,
            wall_s=wall,
            tps=len(prompts) * outputs / wall,
            scheduler=stats,
        )
        return (
            (data, [batch.snapshot(i) for i in range(len(prompts))])
            if snapshot
            else data
        )
    finally:
        scheduler.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("artifact", "ops", "qwen", "cases", "gpu-uuid", "out"):
        p.add_argument("--" + arg, required=True)
    p.add_argument(
        "--reference-qwen", help="Frozen main Qwen DSO, paired with the same ops DSO"
    )
    p.add_argument(
        "--profile",
        choices=("fixed16", "reuse32"),
        help="Capture only a 128-token prefill",
    )
    a = p.parse_args()
    if not a.profile and not a.reference_qwen:
        p.error("the frozen reference Qwen library is required without --profile")
    torch.set_num_threads(4)
    raw = json.loads(Path(a.cases).read_text())["cases"]
    lengths = (48, 128, 320, 768)
    cases = {n: [r["prompt"] for r in raw if r["prompt_tokens"] == n] for n in lengths}
    assert all(
        len(rows) >= 2 and all(len(row) == n for row in rows)
        for n, rows in cases.items()
    )
    prompt = cases[768][0]
    weights = QwenWeights.open(a.artifact)
    ops = CudaBackend(a.ops)
    nvml = NvmlProcesses([a.gpu_uuid])
    allowed = {os.getpid()}
    out = Path(a.out)
    result = dict(
        artifact_name=Path(a.artifact).name,
        artifact_bytes=Path(a.artifact).stat().st_size,
        cases_sha256=digest(a.cases),
        ops_sha256=digest(a.ops),
        qwen_sha256=digest(a.qwen),
        reference_qwen_sha256=digest(a.reference_qwen) if a.reference_qwen else None,
        torch=str(torch.__version__),
        profile=a.profile,
        completed=False,
        c1=[],
        c4=None,
        waves=[],
    )

    def save():
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        tmp.replace(out)

    def load(path):
        runtime = QwenRuntime(path, ops)
        return NativeQwenModel.from_q8(
            weights,
            runtime,
            weight_mode="resident",
            dtype=torch.float16,
            device="cuda:0",
            max_tokens=32,
            metadata_mode="sealed",
            projection_profile="fp16",
            linear_implementation="auto",
        )

    def unchanged_c4(model):
        with closing(model.create_batch([1024] * 4)) as batch:
            return replay(batch, [cases[n][0] for n in lengths], 16, 16, snapshot=True)

    try:
        wait_quiet(nvml, allowed, timeout=10, stable=0.25)
        if a.profile:
            reuse = a.profile == "reuse32"
            chunk = 32 if reuse else 16
            with (
                closing(load(a.qwen)) as model,
                closing(model.create_batch([1024], q5_prefill_reuse=reuse)) as batch,
            ):
                replay(batch, [prompt[:48]], chunk, 1)
                batch.reset()
                torch.cuda.profiler.start()
                try:
                    for i in range(0, 128, chunk):
                        batch.step([prompt[i : i + chunk]], sample_slots=[])
                finally:
                    torch.cuda.profiler.stop()
                result["captured_input_rows"] = 128
        else:
            with closing(load(a.reference_qwen)) as old:
                old_c4, old_state = unchanged_c4(old)
            with closing(load(a.qwen)) as model:
                new_c4, new_state = unchanged_c4(model)
                result["c4"] = dict(
                    greedy_exact=old_c4["output_ids"] == new_c4["output_ids"],
                    states=[
                        state_bits(x, y)
                        for x, y in zip(old_state, new_state, strict=True)
                    ],
                    prompt_lengths=lengths,
                    outputs_each=16,
                    fixed_chunk=16,
                )
                q = result["c4"]
                assert q["greedy_exact"] and all(
                    x["exact"] == x["count"]
                    for slot in q["states"]
                    for x in slot.values()
                )
                print("C4 frozen main parity", q, flush=True)
                save()
                with (
                    closing(model.create_batch([1024])) as ref,
                    closing(model.create_batch([1024], q5_prefill_reuse=True)) as trial,
                ):
                    plans = (
                        cases[48][1],
                        cases[128][1][:127],
                        cases[320][1],
                        prompt,
                        cases[768][1],
                    )
                    for data in plans:
                        x, ids, astate = capture(ref, data, 16)
                        y, other, bstate = capture(trial, data, 32, ids)
                        s = state_bits(astate, bstate)
                        row = dict(
                            input_tokens=len(data),
                            greedy_exact=ids == other,
                            logits_exact=sum(
                                identical(u, v) for u, v in zip(x, y, strict=True)
                            ),
                            logits_checks=len(x),
                            states=s,
                        )
                        result["c1"].append(row)
                        save()
                        print("C1", row, flush=True)
                        assert row["greedy_exact"] and row["logits_exact"] == len(x)
                        assert all(x["exact"] == x["count"] for x in s.values())
                    batches = {"fixed16": (ref, 16), "reuse32": (trial, 32)}
                    for batch, chunk in batches.values():
                        replay(batch, [prompt[:128]], chunk, 2)
                    greedy = None
                    for mode in ("fixed16", "reuse32", "reuse32", "fixed16"):
                        batch, chunk = batches[mode]
                        with ObservedWindow(nvml, allowed, interval=0.1) as window:
                            wave = replay(batch, [prompt], chunk, 16)
                        assert window.clean
                        if greedy is None:
                            greedy = wave["output_ids"]
                        assert greedy == wave["output_ids"]
                        wave.update(mode=mode, gpu_window=window.report())
                        result["waves"].append(wave)
                        save()
                        print("A/B", mode, wave["ttft_s"], wave["tps"], flush=True)
            result["summary"] = {}
            for mode in ("fixed16", "reuse32"):
                frames = [r for r in result["waves"] if r["mode"] == mode]
                result["summary"][mode] = dict(
                    tps=16 * len(frames) / sum(r["wall_s"] for r in frames),
                    wall_s=median(r["wall_s"] for r in frames),
                    cpu_s=median(r["host_cpu_s"] for r in frames),
                    ttft_s=median(r["ttft_s"][0] for r in frames),
                    max_itl_s=median(r["max_itl_s"][0] for r in frames),
                )
        result["completed"] = True
    except BaseException as e:
        result["failure"] = repr(e)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
