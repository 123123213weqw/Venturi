"""Load-specification validation: argument checks, profile flags, device plan.

One pure validation pass over every from_q8 argument, in the historical order
and with the historical messages, producing a frozen LoadSpec that the rest of
the load pipeline consumes. No budget computation and no descriptor assembly
happen here.
"""

from dataclasses import dataclass
import ctypes as C

import torch

from ... import codecs
from ...ops import CudaBackend
from venturi.models.qwen.qwen_weights import QwenQ8Weights


@dataclass(frozen=True)
class LoadSpec:
    weights: object
    backend: object
    weight_mode: str
    dtype: object
    device: str
    target: object
    max_tokens: int
    reserve_bytes: int
    tensor_parallel_device: object | None
    tensor_parallel_transport: object | None
    projection_profile: str
    head_profile: bool
    column_profile: bool
    stage_range: object
    experimental_flags: int
    kv_cache: str
    kv_gpu_budget_bytes: int
    kv_ram_budget_bytes: int
    metadata_mode: str
    dequant_resident: bool
    profiled_create: object | None
    tier_enable: object | None


def ensure_execution_qualified(weights):
    """Loading qualification and execution qualification are separate tiers.

    A codec must be execution-qualified on at least one architecture before a
    native model may be created at all (pre-device gate); the per-architecture
    refinement happens in descriptor assembly. Fail explicitly, never
    silently re-encode.
    """
    allowed = set().union(*codecs.EXECUTION_QUALIFIED.values())
    for binding in weights.plan.bindings:
        if binding.format in allowed or binding.format in ("bf16", "fp32"):
            continue
        raise ValueError(
            f"Codec {binding.format!r} loads and storage-verifies; "
            "native execution qualification pending"
        )



def validate_load_spec(
    weights,
    backend,
    *,
    weight_mode,
    dtype,
    device,
    max_tokens,
    reserve_bytes,
    tensor_parallel_device,
    tensor_parallel_transport,
    projection_profile,
    _stage_range,
    experimental_flags,
    kv_cache,
    kv_gpu_budget_bytes,
    kv_ram_budget_bytes,
    metadata_mode,
    dequant_resident=False,
):
    if not isinstance(weights, QwenQ8Weights) or not isinstance(
        backend, CudaBackend
    ):
        raise TypeError(
            "Native Qwen requires preflighted Q8 weights and explicit CudaBackend"
        )
    if weight_mode not in ("streaming", "resident"):
        raise ValueError("Select explicit streaming or resident weight_mode")
    if metadata_mode not in ("checked-json", "sealed"):
        raise ValueError("Metadata mode must be explicit checked-json or sealed")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 1024:
        raise ValueError(
            "Native Qwen max_tokens must be 1..1024; longer prompts are chunked"
        )
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Native Qwen supports explicit FP16/BF16 only")
    head_profile = projection_profile == "q8-residual-column-mlp-head-gdn"
    column_profile = projection_profile in (
        "q8-residual-column-mlp",
        "q8-residual-column-mlp-head-gdn",
    )
    if projection_profile not in (
        "fp16",
        "q8-residual",
        "q8-residual-column-mlp",
        "q8-residual-column-mlp-head-gdn",
    ):
        raise ValueError("Projection profile must be explicit fp16/q8-residual/q8-residual-column-mlp/q8-residual-column-mlp-head-gdn")
    if column_profile and (
        tensor_parallel_device is None or tensor_parallel_transport != "host"
    ):
        raise ValueError("Column MLP requires explicit two-device host TP")
    if projection_profile != "fp16" and (
        dtype != torch.float16
        or weight_mode != "resident"
        or _stage_range is not None
    ):
        raise ValueError(
            "q8-residual requires a complete resident FP16 target, not PP"
        )
    if type(experimental_flags) is not int or not 0 <= experimental_flags <= 15:
        raise ValueError("experimental_flags must be an integer mask 0..15")
    if experimental_flags and not head_profile:
        raise ValueError("Experiments require the explicit head-GDN projection profile")
    if dequant_resident:
        quantized = {s.format for s in weights.plan.bindings} - {"bf16", "fp32"}
        if not quantized or not all(f.startswith(("q4_", "q5_", "q6_")) for f in quantized):
            raise ValueError("dequant_resident requires a row-codec artifact (q4/q5/q6)")
        if weight_mode != "resident" or tensor_parallel_device is not None or projection_profile != "fp16":
            raise ValueError("dequant_resident requires resident single-device plain profile")
    if kv_cache not in ("static", "vram-first"):
        raise ValueError("kv_cache must be explicit static or vram-first")
    if any(type(n) is not int or not 0 <= n < 2**63 for n in
           (kv_gpu_budget_bytes, kv_ram_budget_bytes)):
        raise ValueError("KV budgets must be nonnegative integer byte counts")
    if kv_cache == "static" and (kv_gpu_budget_bytes or kv_ram_budget_bytes):
        raise ValueError("KV budgets require kv_cache='vram-first'")
    tier_enable = None
    if kv_cache == "vram-first":
        if weight_mode != "resident" or _stage_range is not None:
            raise ValueError("VRAM-first KV requires a complete resident target, not PP")
        try:
            tier_enable = backend._native.venturi_qwen_enable_kv_tier
            backend._native.venturi_qwen_kv_tier_info
        except AttributeError as e:
            raise RuntimeError("VRAM-first KV exports absent; rebuild blocks") from e
        tier_enable.argtypes = [C.c_void_p, C.c_uint64, C.c_uint64]
        tier_enable.restype = C.c_int
    profiled_create = None
    if projection_profile != "fp16":
        try:
            profiled_create = getattr(
                backend._native,
                "venturi_qwen_create_experimental"
                if experimental_flags
                else "venturi_qwen_create_head_gdn"
                if head_profile
                else "venturi_qwen_create_column_mlp"
                if column_profile
                else "venturi_qwen_create_profiled",
            )
        except AttributeError as e:
            raise RuntimeError(
                "Explicit projection profile exports absent; rebuild blocks"
            ) from e
    ensure_execution_qualified(weights)
    target = backend.validate_device(dtype, device)
    if tensor_parallel_transport not in ("peer", "host"):
        raise ValueError("Tensor parallel transport must be explicit peer or host")
    if tensor_parallel_device is None and tensor_parallel_transport != "peer":
        raise ValueError(
            "Host transport requires an explicit tensor parallel device"
        )
    peer = None
    if tensor_parallel_device is not None:
        if weight_mode != "resident":
            raise ValueError("Tensor parallel requires explicit resident weights")
        peer = backend.validate_device(dtype, tensor_parallel_device)
        if peer == target:
            raise ValueError("Tensor parallel peer must be a different CUDA device")
        if tensor_parallel_transport == "peer" and not (
            torch.cuda.can_device_access_peer(target.index, peer.index)
            and torch.cuda.can_device_access_peer(peer.index, target.index)
        ):
            raise ValueError("Bidirectional CUDA peer access is required")
    if column_profile and torch.cuda.get_device_capability(target) != (7, 0):
        raise ValueError("Column MLP requires an explicit SM70 target")
    return LoadSpec(
        weights=weights,
        backend=backend,
        weight_mode=weight_mode,
        dtype=dtype,
        device=device,
        target=target,
        max_tokens=max_tokens,
        reserve_bytes=reserve_bytes,
        tensor_parallel_device=peer,
        tensor_parallel_transport=tensor_parallel_transport,
        projection_profile=projection_profile,
        head_profile=head_profile,
        column_profile=column_profile,
        stage_range=_stage_range,
        experimental_flags=experimental_flags,
        kv_cache=kv_cache,
        kv_gpu_budget_bytes=kv_gpu_budget_bytes,
        kv_ram_budget_bytes=kv_ram_budget_bytes,
        metadata_mode=metadata_mode,
        profiled_create=profiled_create,
        tier_enable=tier_enable,
        dequant_resident=dequant_resident,
    )


def _outside_capture_guard(target):
    from venturi.models.decoder_layer import _outside_capture

    _outside_capture(target)
