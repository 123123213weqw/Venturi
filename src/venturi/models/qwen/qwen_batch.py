"""Native grouped target execution with independently owned GDN/KV requests.

Python handles metadata/submission only. Prefill packing, grouped projections,
sequence-local state updates, active decode compaction and greedy control are C++.
The generate entry remains fixed-cohort. The optional step entry exposes one
bounded mixed prefill/decode iteration to an external owner; it is not MTP,
paging or a scheduler by itself.
"""

import ctypes as C
import threading
from types import SimpleNamespace
import weakref

import torch

from venturi.models.qwen.qwen_native import NativeQwenModel
from venturi.ops.vocabulary import _ids


_ATTENTION_EXECUTION = {"legacy": 0, "descriptor": 1, "graph": 2}
_RESIDENT_COUNTERS = (
    "direct_v",
    "v_copies",
    "direct_hidden",
    "hidden_copies",
    "completion_fences",
    "descriptor_steps",
    "descriptor_bytes_uploaded",
    "graph_captures",
    "graph_replays",
    "descriptor_eager_layers",
    "graph_fallback_layers",
    "capture_ns",
    "device_descriptor_bytes",
    "pinned_descriptor_bytes",
    "v_copy_bytes",
    "hidden_copy_bytes",
)
_DENSE_IO_COUNTERS = (
    "prior_workspace_bytes",
    "workspace_bytes",
    "weight_scratch_bytes",
    "output_scratch_bytes",
    "shadow_budget_bytes",
    "shadow_bytes",
    "shadow_tiles",
    "setup_ns",
    "gemms",
    "direct_gemms",
    "output_copy_bytes",
    "unpack_calls",
    "unpack_bytes",
    "shadow_hits",
    "shadow_read_bytes",
)
_SHARED_TILE_COUNTERS = (
    "shared_tiles",
    "tile_grouped_calls",
    "tile_wmma_calls",
    "tile_avoided_fp16_write_bytes",
)


def _bind(backend, *, q5_prefill_reuse=False, resident_options=False):
    ptr, integer, u64 = C.c_void_p, C.c_int, C.c_uint64
    signatures = {
        "create": ([ptr, ptr, integer], ptr),
        "destroy": ([ptr], None),
        "forward": ([ptr, ptr, integer, ptr, ptr, ptr, u64], integer),
        "generate": (
            [ptr, ptr, integer, ptr, ptr, ptr, ptr, ptr, ptr, integer, ptr],
            integer,
        ),
        "reset": ([ptr, ptr, integer], integer),
        "info": ([ptr, ptr, ptr, ptr], integer),
        "snapshot": ([ptr, integer, integer, integer, ptr, u64], integer),
        "debug_failure": ([ptr, integer, integer, integer], integer),
    }
    result = {}
    for name, (args, restype) in signatures.items():
        symbol = "venturi_qwen_batch_" + name
        if name == "create" and resident_options:
            symbol += "_resident_v1"
            args = [*args, integer, integer, integer]
            if not hasattr(backend._native, symbol):
                raise RuntimeError(
                    "Resident batch modes require a rebuilt Qwen library"
                )
        elif name == "create" and q5_prefill_reuse:
            symbol += "_q5_prefill_reuse"
            if not hasattr(backend._native, symbol):
                raise RuntimeError("Q5 prefill reuse requires a rebuilt Qwen library")
        fn = getattr(backend._native, symbol)
        fn.argtypes, fn.restype = args, restype
        result[name] = fn
    return SimpleNamespace(**result)


class NativeQwenBatchSession:
    @property
    def max_token_rows(self):
        return self.model._max_tokens

    @property
    def vocab_size(self):
        return self.model._cfg["vocab_size"]

    @classmethod
    def create(
        cls,
        model,
        capacities,
        *,
        q5_prefill_reuse=False,
        resident_dataflow=False,
        attention_execution="legacy",
    ):
        if type(model) is not NativeQwenModel or model._mode != "resident":
            raise ValueError(
                "Batch requires a complete resident native target; PP/MTP wrappers are not batch targets"
            )
        if not isinstance(capacities, (list, tuple)) or not 1 <= len(capacities) <= min(
            32, model._max_tokens
        ):
            raise ValueError("Batch slot count must be 1..min(32,max_tokens)")
        if any(
            type(c) is not int or not 0 < c <= model._cfg["max_position_embeddings"]
            for c in capacities
        ):
            raise ValueError("Batch capacities exceed the model context")
        if type(q5_prefill_reuse) is not bool:
            raise ValueError("q5_prefill_reuse must be an explicit bool")
        if q5_prefill_reuse and (len(capacities) != 1 or model._max_tokens != 32):
            raise ValueError("Q5 prefill reuse requires one slot and max_tokens=32")
        if type(resident_dataflow) is not bool:
            raise ValueError("resident_dataflow must be an explicit bool")
        if (
            type(attention_execution) is not str
            or attention_execution not in _ATTENTION_EXECUTION
        ):
            raise ValueError("attention_execution must be legacy, descriptor, or graph")
        mode = _ATTENTION_EXECUTION[attention_execution]
        resident_options = bool(resident_dataflow or mode)
        native = _bind(
            model._runtime,
            q5_prefill_reuse=q5_prefill_reuse,
            resident_options=resident_options,
        )
        with model._lock:
            if not model._finalizer.alive:
                raise RuntimeError("Native Qwen model closed")
            model._metadata()
            cap = (C.c_int * len(capacities))(*capacities)
            args = (model._handle, cap, len(capacities))
            handle = (
                native.create(
                    *args, int(q5_prefill_reuse), int(resident_dataflow), mode
                )
                if resident_options
                else native.create(*args)
            )
            if not handle:
                model._runtime._check(-1)
        self = cls()
        self.model, self.capacities, self._handle, self._native = (
            model,
            tuple(capacities),
            handle,
            native,
        )
        self._lock = threading.RLock()
        self._finalizer = weakref.finalize(self, native.destroy, handle)
        return self

    def resident_execution_info(self):
        """Cumulative ordinary batch submissions, transfers and graph setup counters."""
        if type(self) is not NativeQwenBatchSession:
            raise ValueError("Resident execution counters require an ordinary batch")
        with self._lock:
            self._check()
            fn = getattr(
                self.model._runtime._native, "venturi_qwen_batch_resident_info_v1", None
            )
            if fn is None:
                raise RuntimeError(
                    "Resident execution counters require a rebuilt Qwen library"
                )
            fn.argtypes, fn.restype = (
                [C.c_void_p, C.POINTER(C.c_uint64), C.c_int],
                C.c_int,
            )
            values = (C.c_uint64 * len(_RESIDENT_COUNTERS))()
            self.model._runtime._check(fn(self._handle, values, len(values)))
            return dict(zip(_RESIDENT_COUNTERS, map(int, values), strict=True))

    def prepare_dense_io(self, *, direct_outputs=False, shadow_budget_bytes=0):
        """Opt in once per model, before inference, after reserving this batch's KV.

        Workspace is geometry-sized independently of these options. Explicit
        shadows share immutable shape-identical weight tiles across sessions.
        """
        if type(self) is not NativeQwenBatchSession:
            raise ValueError("Dense I/O preparation requires an ordinary batch")
        if (
            type(direct_outputs) is not bool
            or type(shadow_budget_bytes) is not int
            or not 0 <= shadow_budget_bytes <= 2**64 - 1
        ):
            raise ValueError(
                "Explicit direct_outputs bool and uint64 shadow budget required"
            )
        with self._lock:
            self._check()
            fn = getattr(
                self.model._runtime._native,
                "venturi_qwen_batch_prepare_dense_io_v1",
                None,
            )
            if fn is None:
                raise RuntimeError("Dense I/O requires a rebuilt Qwen library")
            fn.argtypes, fn.restype = [C.c_void_p, C.c_int, C.c_uint64], C.c_int
            self.model._runtime._check(
                fn(self._handle, int(direct_outputs), shadow_budget_bytes)
            )

    def dense_io_info(self):
        """Model-level counters; copy and unpack payloads are not DRAM counters."""
        if type(self) is not NativeQwenBatchSession:
            raise ValueError("Dense I/O counters require an ordinary batch")
        with self._lock:
            self._check()
            lib = self.model._runtime._native
            fn = getattr(lib, "venturi_qwen_batch_dense_io_info_v2", None)
            names = _DENSE_IO_COUNTERS + _SHARED_TILE_COUNTERS
            if fn is None:
                fn = getattr(lib, "venturi_qwen_batch_dense_io_info_v1", None)
                names = _DENSE_IO_COUNTERS
            if fn is None:
                raise RuntimeError("Dense I/O counters require a rebuilt Qwen library")
            fn.argtypes, fn.restype = (
                [C.c_void_p, C.POINTER(C.c_uint64), C.c_int],
                C.c_int,
            )
            values = (C.c_uint64 * len(names))()
            self.model._runtime._check(fn(self._handle, values, len(values)))
        return dict(zip(names, map(int, values), strict=True))

    def prepare_shared_tiles(self, *, direct_outputs=False):
        """Explicit SM70 numerical mode; not interchangeable with the dense path.

        Select it once before executing the model. Small output matrices keep
        the dense route; codec admission and request ownership are unchanged.
        """
        if type(self) is not NativeQwenBatchSession or type(direct_outputs) is not bool:
            raise ValueError("Shared tiles require an ordinary batch and explicit bool")
        with self._lock:
            self._check()
            fn = getattr(
                self.model._runtime._native,
                "venturi_qwen_batch_prepare_shared_tiles_v1",
                None,
            )
            if fn is None:
                raise RuntimeError("Shared tiles require a rebuilt Qwen library")
            fn.argtypes, fn.restype = [C.c_void_p, C.c_int], C.c_int
            self.model._runtime._check(fn(self._handle, int(direct_outputs)))

    def prepare_ada_tiles(self, rules, *, direct_outputs=False):
        """Apply an explicit experimental SM89 Q5 table before this batch projects."""
        if type(self) is not NativeQwenBatchSession or type(direct_outputs) is not bool:
            raise ValueError("Ada tiles require an ordinary batch and explicit bool")
        if not isinstance(rules, (list, tuple)) or not 0 < len(rules) <= 4096:
            raise ValueError("Ada rules require an explicit bounded table")
        raw = []
        for row in rules:
            if (
                not isinstance(row, (tuple, list))
                or len(row) != 4
                or any(type(v) is not int or not 0 <= v < 2**31 for v in row)
            ):
                raise ValueError("Ada rule must be [M,N,K,mode]")
            raw.extend(row)
        values = (C.c_int * len(raw))(*raw)
        with self._lock:
            self._check()
            fn = getattr(
                self.model._runtime._native,
                "venturi_qwen_batch_prepare_ada_tiles_v1",
                None,
            )
            if fn is None:
                raise RuntimeError("Ada tiles require a rebuilt Qwen library")
            fn.argtypes, fn.restype = (
                [C.c_void_p, C.c_int, C.POINTER(C.c_int), C.c_int],
                C.c_int,
            )
            self.model._runtime._check(
                fn(self._handle, int(direct_outputs), values, len(rules))
            )

    def _check(self):
        if not self._finalizer.alive:
            raise RuntimeError("Native Qwen batch closed")
        self.model._metadata()

    def _slots(self, slots):
        if (
            not isinstance(slots, (list, tuple))
            or not slots
            or any(
                type(i) is not int or not 0 <= i < len(self.capacities) for i in slots
            )
            or len(set(slots)) != len(slots)
        ):
            raise ValueError("Explicit unique valid batch slots required")
        return (C.c_int * len(slots))(*slots)

    def _inputs(self, prompts, slots, *, max_rows=None):
        if not isinstance(prompts, (list, tuple)) or not prompts:
            raise ValueError("Nonempty list of tokenized prompts required")
        slots = list(range(len(prompts))) if slots is None else slots
        selected = self._slots(slots)
        if len(slots) != len(prompts):
            raise ValueError("One prompt per selected batch slot required")
        flat, offsets = [], [0]
        for prompt in prompts:
            _ids(prompt, self.model._cfg["vocab_size"])
            if max_rows is not None and len(flat) + len(prompt) > max_rows:
                raise ValueError("Step token rows exceed prepared model workspace")
            flat.extend(prompt)
            offsets.append(len(flat))
        if len(flat) >= 2**31:
            raise ValueError("Batch token offsets exceed int32")
        return (
            selected,
            (C.c_int64 * len(flat))(*flat),
            (C.c_int * len(offsets))(*offsets),
        )

    def step(self, prompts, *, slots=None, sample_slots=None, decode_slots=()):
        """Commit ONE bounded group and return greedy IDs for requested slots.

        Each prompt is a contiguous append to one private request. Only completed
        prefills and decode rows should be sampled; partial prefills may pass an
        empty sample list. Decode rows must each append exactly one token. No
        logits, tensor math, state migration or automatic strategy fallback is
        performed in Python. Failed native execution poisons participating slots
        until reset. Requires the additive native step export; old libraries are
        rejected for this method without breaking their existing cohort API.
        """
        if type(self) is not NativeQwenBatchSession:
            raise ValueError("Step requires an ordinary target batch, not MTP")
        selected, tokens, offsets = self._inputs(
            prompts, slots, max_rows=self.model._max_tokens
        )
        samples = list(selected) if sample_slots is None else sample_slots
        for name, values in (("sample", samples), ("decode", decode_slots)):
            if not isinstance(values, (list, tuple)):
                raise ValueError(f"Explicit {name} slot list required")
            if values:
                self._slots(values)
            if not set(values).issubset(set(selected)):
                raise ValueError(f"Step {name} slots must belong to this iteration")
        extents = {int(s): offsets[i + 1] - offsets[i] for i, s in enumerate(selected)}
        if any(extents[s] != 1 for s in decode_slots):
            raise ValueError("Each decode slot must append exactly one token")
        sample_ids = (C.c_int * len(samples))(*samples)
        decode_ids = (C.c_int * len(decode_slots))(*decode_slots)
        output = (C.c_int64 * len(samples))()
        with self._lock:
            self._check()
            fn = getattr(self, "_step_fn", None)
            if fn is None:
                fn = getattr(
                    self.model._runtime._native, "venturi_qwen_batch_step", None
                )
                if fn is None:
                    raise RuntimeError(
                        "Native batch step export absent; rebuild explicitly"
                    )
                p, i = C.c_void_p, C.c_int
                fn.argtypes = [p, p, i, p, p, p, i, p, i, p, C.c_uint64]
                fn.restype = i
                self._step_fn = fn
            self.model._runtime._check(
                fn(
                    self._handle,
                    selected,
                    len(selected),
                    tokens,
                    offsets,
                    sample_ids,
                    len(samples),
                    decode_ids,
                    len(decode_slots),
                    output,
                    len(samples),
                )
            )
        return list(output)

    def forward(self, prompts, *, slots=None):
        """Diagnostic owned CPU FP32 final-row logits per selected request."""
        selected, tokens, offsets = self._inputs(prompts, slots)
        with self._lock:
            self._check()
            out = torch.empty(
                (len(selected), self.model._cfg["vocab_size"]),
                dtype=torch.float32,
                device="cpu",
            )
            self.model._runtime._check(
                self._native.forward(
                    self._handle,
                    selected,
                    len(selected),
                    tokens,
                    offsets,
                    out.data_ptr(),
                    out.numel(),
                )
            )
            return out

    def generate(self, prompts, *, max_new_tokens, eos_token_ids=(), slots=None):
        selected, tokens, offsets = self._inputs(prompts, slots)
        count = len(selected)
        maximum = (
            [max_new_tokens] * count if type(max_new_tokens) is int else max_new_tokens
        )
        if (
            not isinstance(maximum, (list, tuple))
            or len(maximum) != count
            or any(type(v) is not int or not 1 <= v < 2**31 for v in maximum)
        ):
            raise ValueError("Positive int32 output budget per selected slot required")
        if not isinstance(eos_token_ids, (list, tuple)):
            raise ValueError("Explicit EOS list or per-request lists required")
        eos_rows = (
            eos_token_ids
            if eos_token_ids and isinstance(eos_token_ids[0], (list, tuple))
            else [eos_token_ids] * count
        )
        if len(eos_rows) != count:
            raise ValueError("One EOS list per selected slot required")
        flat, eos_offsets = [], [0]
        for row in eos_rows:
            if not isinstance(row, (list, tuple)):
                raise ValueError("Explicit EOS lists required")
            if row:
                _ids(row, self.model._cfg["vocab_size"])
            flat.extend(row)
            eos_offsets.append(len(flat))
        stride = max(maximum)
        if stride * count >= 2**31:
            raise ValueError("Batch output allocation exceeds int32")
        budgets = (C.c_int * count)(*maximum)
        eos = (C.c_int64 * len(flat))(*flat)
        eo = (C.c_int * len(eos_offsets))(*eos_offsets)
        with self._lock:
            self._check()
            # Capacity validation precedes potentially large host output allocation.
            info = self._generation_info(selected)
            for j, slot in enumerate(selected):
                row = info["requests"][slot]
                if row["failed"]:
                    raise RuntimeError("Native batch request failed; reset required")
                if (
                    offsets[j + 1] - offsets[j] + maximum[j]
                    > row["capacity"] - row["position"]
                ):
                    raise ValueError("Batch generation exceeds slot capacity")
            out, counts = (C.c_int64 * (stride * count))(), (C.c_int * count)()
            self.model._runtime._check(
                self._native.generate(
                    self._handle,
                    selected,
                    count,
                    tokens,
                    offsets,
                    budgets,
                    eos,
                    eo,
                    out,
                    stride,
                    counts,
                )
            )
            return [
                list(out[j * stride : j * stride + counts[j]]) for j in range(count)
            ]

    def _generation_info(self, selected):
        """Bounded preflight data; subclasses need not build verbose public stats."""
        return self.info()

    def info(self):
        with self._lock:
            if not self._finalizer.alive:
                raise RuntimeError("Native Qwen batch closed")
            shared, rows, times = (
                (C.c_uint64 * 15)(),
                (C.c_uint64 * (4 * len(self.capacities)))(),
                (C.c_double * (2 * len(self.capacities)))(),
            )
            self.model._runtime._check(
                self._native.info(self._handle, shared, rows, times)
            )
            result = dict(
                zip(
                    (
                        "slot_count",
                        "shared_batch_workspace_device_bytes",
                        "all_request_device_bytes",
                        "model_device_bytes",
                        "peer_model_device_bytes",
                        "shared_model_pinned_host_bytes",
                        "group_forward_calls",
                        "input_rows",
                        "decode_group_calls",
                        "prefill_group_calls",
                        "head_calls",
                        "head_rows",
                        "max_group_rows",
                        "peer_input_copy_bytes",
                        "peer_output_copy_bytes",
                    ),
                    map(int, shared),
                )
            )
            result["requests"] = [
                dict(
                    position=int(rows[4 * i]),
                    failed=bool(rows[4 * i + 1]),
                    capacity=int(rows[4 * i + 2]),
                    request_device_bytes=int(rows[4 * i + 3]),
                    native_time_to_first_token_seconds=times[2 * i],
                    native_generation_seconds=times[2 * i + 1],
                )
                for i in range(len(self.capacities))
            ]
            result.update(self.model.projection_info)
            if result.get("head_gdn_enabled"):
                name = (
                    "venturi_qwen_mtp_batch_head_gdn_memory"
                    if hasattr(self.model, "_mtp_weights")
                    else "venturi_qwen_batch_head_gdn_memory"
                )
                query = getattr(self.model._runtime._native, name)
                query.argtypes, query.restype = (
                    [C.c_void_p, C.POINTER(C.c_uint64)],
                    C.c_int,
                )
                memory = (C.c_uint64 * 2)()
                self.model._runtime._check(query(self._handle, memory))
                result.update(
                    all_peer_request_device_bytes=int(memory[0]),
                    all_peer_target_backup_device_bytes=int(memory[1]),
                    distributed_memory_scope="all_request_device_bytes is primary; all_peer_request_device_bytes is additional GDN cache. MTP backup totals already include their reported peer component.",
                )
            if getattr(self.model, "_kv_cache_mode", "static") != "static":
                result["shared_model_kv_cache"] = (
                    self.model._kv_cache_info_for_request()
                )
            result["scope"] = (
                "Shared model/workspace counts once; request bytes per slot. Generation counters are shared per most recent cohort, never summed per response. Fixed cohort admission, no MTP/continuous batching/paging."
            )
            return result

    def reset(self, slots=None):
        selected = self._slots(
            list(range(len(self.capacities))) if slots is None else slots
        )
        with self._lock:
            self._check()
            self.model._runtime._check(
                self._native.reset(self._handle, selected, len(selected))
            )

    def _inject_failure(self, *, slot, layer, position):
        self._slots([slot])
        if (
            type(layer) is not int
            or type(position) is not int
            or not 0 <= layer < self.model._cfg["num_hidden_layers"]
            or position < 0
        ):
            raise ValueError("Invalid batch failure hook")
        with self._lock:
            self._check()
            self.model._runtime._check(
                self._native.debug_failure(self._handle, slot, layer, position)
            )

    def snapshot(self, slot):
        self._slots([slot])
        with self._lock:
            self._check()
            cfg = self.model._cfg
            position = self.info()["requests"][slot]["position"]
            result = dict(gdn=[], kv=[])
            packed = (
                2 * cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
                + cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
            )
            for i, kind in enumerate(cfg["layer_types"]):
                shapes = (
                    (
                        (cfg["linear_conv_kernel_dim"] - 1, packed),
                        (
                            cfg["linear_num_value_heads"],
                            cfg["linear_value_head_dim"],
                            cfg["linear_key_head_dim"],
                        ),
                    )
                    if kind == "linear_attention"
                    else ((position, cfg["num_key_value_heads"], cfg["head_dim"]),) * 2
                )
                tensors = []
                for j, shape in enumerate(shapes):
                    tensor = torch.empty(
                        shape,
                        dtype=(
                            torch.float32
                            if kind == "linear_attention" and j == 1
                            else self.model.dtype
                        ),
                        device="cpu",
                    )
                    self.model._runtime._check(
                        self._native.snapshot(
                            self._handle,
                            slot,
                            i,
                            j if kind == "linear_attention" else j + 2,
                            tensor.data_ptr(),
                            tensor.numel() * tensor.element_size(),
                        )
                    )
                    tensors.append(tensor)
                result["gdn"].append(
                    tuple(tensors) if kind == "linear_attention" else None
                )
                result["kv"].append(
                    tuple(tensors) if kind == "full_attention" else None
                )
            return result

    def close(self):
        with self._lock:
            self._finalizer()
