"""Fixed-work native Qwen batch sweep. Timing runs never enable a profiler.

C1 is the scalar session; larger counts are independently owned fixed cohorts.
The original eight 16-token prompts repeat at C16/C32, without prefix sharing.
Weights, FP16 KV, residual-A8 projections, row budget and experiment mask stay
fixed. RAM backing is forbidden. No HTTP, MTP or Python token/layer loop.
"""

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import threading
from time import perf_counter, process_time, time


COUNTS = (1, 2, 4, 8, 16, 32)
UUIDS = (
    "GPU-18fbf081-3c6d-d65b-11b8-288a4ce1ab28",
    "GPU-f65db2e8-87bb-c88d-363f-23c79f793548",
)
PROFILE = "q8-residual-column-mlp-head-gdn"


def verify_gpu_uuids(actual):
    # NVML includes GPU-, while this PyTorch version exposes the bare UUID.
    # Normalize only that display prefix; never relax physical host selection.
    normalized = tuple(str(value).removeprefix("GPU-") for value in actual)
    if normalized != tuple(value.removeprefix("GPU-") for value in UUIDS):
        raise AssertionError("Unexpected physical GPU UUIDs: " + repr(actual))


def write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def repeat_prompts(prompts, count):
    if count not in COUNTS or len(prompts) != 8 or any(len(p) != 16 for p in prompts):
        raise ValueError("Expected eight pinned 16-token prompts and a supported count")
    return [list(prompts[i % len(prompts)]) for i in range(count)]


def summarize(trials, count, new_tokens):
    if count not in COUNTS or new_tokens < 2 or not trials:
        raise ValueError("A supported count and nonempty fixed-work trials are required")
    for trial in trials:
        for phase, tokens in (("decode", count * (new_tokens - 1)), ("complete", count * new_tokens)):
            item = trial[phase]
            if item["output_tokens"] != tokens or item["wall_seconds"] <= 0:
                raise ValueError("Throughput must use cohort wall time and actual fixed output count")
    rates = [t["decode"]["output_tokens"] / t["decode"]["wall_seconds"] for t in trials]
    return dict(
        repeats=len(trials),
        decode_aggregate_tps_mean=statistics.mean(rates),
        decode_aggregate_tps_range=[min(rates), max(rates)],
        decode_per_request_tps_mean=statistics.mean(r / count for r in rates),
        mean_intertoken_ms=statistics.mean(t["decode"]["wall_seconds"] / (new_tokens - 1) * 1000 for t in trials),
        ttft_seconds_mean=statistics.mean(t["ttft"]["wall_seconds"] for t in trials),
        complete_aggregate_tps_mean=statistics.mean(t["complete"]["output_tokens"] / t["complete"]["wall_seconds"] for t in trials),
        complete_cohort_wall_seconds_mean=statistics.mean(t["complete"]["wall_seconds"] for t in trials),
        decode_process_cpu_cores_mean=statistics.mean(t["decode"]["process_cpu_seconds"] / t["decode"]["wall_seconds"] for t in trials),
    )


def signature(value):
    import torch

    if torch.is_tensor(value):
        tensor = value.cpu().contiguous()
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise AssertionError("Nonfinite diagnostic/state tensor")
        return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest())
    if isinstance(value, dict):
        # Scalar snapshots also carry position; compare the canonical state bytes
        # to cohort snapshots and validate positions independently.
        return {k: signature(value[k]) for k in ("gdn", "kv") if k in value}
    if isinstance(value, (list, tuple)):
        return [signature(v) for v in value]
    return value


def measured(call, output_tokens=None):
    start, wall, cpu = time(), perf_counter(), process_time()
    value = call()
    result = dict(started_unix=start, wall_seconds=perf_counter() - wall, process_cpu_seconds=process_time() - cpu)
    if output_tokens is not None:
        result.update(output_tokens=output_tokens, tps=output_tokens / result["wall_seconds"])
    return value, result


def no_spill(info):
    assert info["mode"] == "vram-first"
    for key in ("cpu_backing_bytes", "evictions", "reloads", "device_to_host_bytes", "host_to_device_bytes", "allocation_ooms"):
        assert info[key] == 0, (key, info[key])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact")
    parser.add_argument("--library", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=COUNTS, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--original-reference", type=Path)
    parser.add_argument("--mode", choices=("timing", "profile"), default="timing")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--counter-start", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1 or (args.counter_start and args.mode != "profile"):
        parser.error("Positive repeats required; profiler API belongs only to separate profile runs")
    if not args.reference and (args.count != 8 or not args.original_reference):
        parser.error("First qualify C8 against the original report; other runs require that reference")

    import torch
    from venturi.models import NativeQwenModel, QwenQ8Weights
    from venturi.ninfer import NInferArtifact
    from venturi.ops import CudaBackend

    torch.set_num_threads(4)
    assert torch.cuda.device_count() == 2
    actual = tuple(str(torch.cuda.get_device_properties(i).uuid) for i in range(2))
    verify_gpu_uuids(actual)
    assert all(torch.cuda.get_device_capability(i) == (7, 0) for i in range(2))
    args.output_directory.mkdir(parents=True, exist_ok=False)
    prompts = json.loads(Path(args.prompts).read_text())["prompt_token_ids"]
    selected = repeat_prompts(prompts, args.count)
    original = json.loads(args.original_reference.read_text()) if args.original_reference else None
    reference = json.loads(args.reference.read_text()) if args.reference else None
    if reference:
        assert reference["completed"] and reference["count"] == 8 and reference["correctness"]["exact_reference_bytes_equal"]
        assert reference["base_prompts"] == prompts
    stat = Path(args.artifact).stat()
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(args.artifact))
    result = dict(
        scope=__doc__, completed=False, profiled=args.mode == "profile", count=args.count,
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        base_prompts=prompts, selected_prompts=selected, artifact_id=weights.artifact.artifact_id,
        artifact_stat=dict(bytes=stat.st_size, inode=stat.st_ino, mtime_ns=stat.st_mtime_ns),
        library_sha256=hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
        model="official-source Qwen3.8-27B native Q8, not production Uncensored GGUF",
        config=weights.config, devices=list(UUIDS), pytorch_uuid_strings=list(actual), precision="Q8 group32/FP16 scales; F16 values/KV; residual-A8 projections; FP32 GDN/accumulations",
        max_projection_rows=32, capacity_per_request=256, input_tokens_per_request=16,
        new_tokens=32, mtp=False, experimental_flags=0, trials=[],
        cohort_kind="scalar" if args.count == 1 else "fixed cohort",
    )
    if original:
        assert original["completed"] and original["mode"] == "qualify"
        assert original["artifact_id"] == result["artifact_id"] and original["prompt_token_ids"] == prompts
    if reference:
        for key in ("artifact_id", "library_sha256", "config", "precision", "max_projection_rows", "new_tokens"):
            assert result[key] == reference[key], key
    samples, stop = [], threading.Event()

    def sample():
        while not stop.is_set():
            start = time()
            try:
                p = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                samples.append(dict(sample_started_unix=start, unix=time(), returncode=p.returncode, lines=p.stdout.splitlines()))
            except Exception as error:
                samples.append(dict(unix=time(), error=repr(error)))
            stop.wait(0.2)

    thread = threading.Thread(target=sample)
    thread.start()
    model = obj = None
    try:
        model, result["load"] = measured(lambda: NativeQwenModel.from_q8(
            weights, CudaBackend(args.library), weight_mode="resident", dtype=torch.float16,
            device="cuda:0", max_tokens=32, reserve_bytes=256 * 2**20,
            tensor_parallel_device="cuda:1", tensor_parallel_transport="host",
            projection_profile=PROFILE, experimental_flags=0,
            kv_cache="vram-first", kv_gpu_budget_bytes=0, kv_ram_budget_bytes=0,
        ))
        obj = model.create_session(256) if args.count == 1 else model.create_batch([256] * args.count)
        inputs = selected[0] if args.count == 1 else selected

        def generate(values, n):
            value = obj.generate(values, max_new_tokens=n, eos_token_ids=[])
            return [value] if args.count == 1 else value

        def snapshots():
            if args.count == 1:
                assert obj.position in (16, 47)
                return [signature(obj.snapshot())]
            info = obj.info()
            assert all(not r["failed"] and r["position"] in (16, 47) for r in info["requests"])
            return [signature(obj.snapshot(i)) for i in range(args.count)]

        if args.mode == "timing":
            logits = obj.forward(inputs)
            logits = logits.unsqueeze(0) if args.count == 1 else logits
            teacher_rows = [signature(row) for row in logits]
            teacher_states = snapshots()
            if original:
                assert signature(logits) == original["comparison_payload"]["cohort"]["teacher_logits"]
                assert teacher_states == original["comparison_payload"]["cohort"]["teacher_states"]
            if reference:
                baseline = reference["reference_payload"]
                assert teacher_rows == [baseline["teacher_rows"][i % 8] for i in range(args.count)]
                assert teacher_states == [baseline["teacher_states"][i % 8] for i in range(args.count)]
            del logits
            obj.reset()
            generated = generate(inputs, 32)
            states = snapshots()
            expected = original["comparison_payload"]["cohort"] if original else reference["reference_payload"]
            assert generated == [expected["tokens"][i % 8] for i in range(args.count)]
            assert states == [expected["states"][i % 8] for i in range(args.count)]
            result["correctness"] = dict(exact_reference_bytes_equal=True, finite_teacher_logits_and_all_state_tensors=True, all_slots_checked=args.count)
            result["reference_payload"] = dict(teacher_rows=teacher_rows, teacher_states=teacher_states, tokens=generated, states=states)
            obj.reset()
            no_spill(model.kv_cache_info())
            write(args.output_directory / "report.json", result)
            print(json.dumps(dict(count=args.count, correctness="all logits/greedy tokens/KV/GDN bytes exact")), flush=True)
        else:
            expected = reference["reference_payload"]
            generated = [expected["tokens"][i % 8] for i in range(args.count)]

        # Correctness is not charged to timing; one additional full-call warmup
        # follows snapshots/reset before the scored calls.
        assert generate(inputs, 32) == generated
        obj.reset()
        if args.mode == "profile":
            first = generate(inputs, 1)
            continuation = [first[0][-1]] if args.count == 1 else [[row[-1]] for row in first]
            runtime = C.CDLL("/usr/local/cuda/lib64/libcudart.so.12")
            if args.counter_start:
                assert runtime.cudaProfilerStart() == 0
            try:
                with torch.cuda.nvtx.range("batch_sweep_post_first"):
                    following, timing = measured(lambda: generate(continuation, 4), 4 * args.count)
            finally:
                if args.counter_start:
                    assert runtime.cudaProfilerStop() == 0
            assert following == [row[1:5] for row in generated]
            result["profile_window"] = dict(timing=timing, decode_steps=4, throughput_score=False)
        else:
            for repeat in range(args.repeats):
                def complete_call():
                    value, timing = measured(lambda: generate(inputs, 32), 32 * args.count)
                    assert value == generated
                    info = obj.info()
                    obj.reset()
                    return timing, info

                if repeat % 2:
                    complete, complete_info = complete_call()
                first, ttft = measured(lambda: generate(inputs, 1))
                continuation = [first[0][-1]] if args.count == 1 else [[row[-1]] for row in first]
                before = model.kv_cache_info()
                following, decode = measured(lambda: generate(continuation, 31), 31 * args.count)
                after = model.kv_cache_info()
                no_spill(after)
                assert [a + b for a, b in zip(first, following)] == generated
                obj.reset()
                if not repeat % 2:
                    complete, complete_info = complete_call()
                result["trials"].append(dict(repeat=repeat, ttft=ttft, decode=decode, complete=complete,
                    complete_info=complete_info, kv_before_decode=before, kv_after_decode=after))
                write(args.output_directory / "report.json", result)
                print(json.dumps(dict(count=args.count, repeat=repeat, decode_tps=decode["tps"], complete_tps=complete["tps"])), flush=True)
            result["summary"] = summarize(result["trials"], args.count, 32)
        no_spill(model.kv_cache_info())
        result["kv_cache"] = model.kv_cache_info()
        after = Path(args.artifact).stat()
        assert (after.st_size, after.st_ino, after.st_mtime_ns) == (stat.st_size, stat.st_ino, stat.st_mtime_ns)
        result["completed"] = True
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        if obj is not None:
            obj.close()
        if model is not None:
            model.close()
        stop.set()
        thread.join(timeout=8)
        write(args.output_directory / "report.json", result)
        write(args.output_directory / "gpu-telemetry.json", samples)


if __name__ == "__main__":
    main()
