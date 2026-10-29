"""Explicit setup-time expert policy binding; NEVER engine/backend hot switching.

Call before creating/using requests, with immutable complete checkpoint files.
No automatic GPU placement or OOM CPU fallback. Cache budgets count retained
weights only, not live leases, allocator reservations or miss/attention scratch.
"""

import torch

from ..flash_next import FlashNextConfig
from .flash_next_cuda_fp8 import CUDABlockFP8ExpertStore
from .flash_next_expert_cache import CachedExpertStore
from .flash_next_fp8 import BlockFP8ExpertStore
from .flash_next_moe import MoEGeometry
from .flash_next_text import FlashNextTextCore
from ..ops.flash_next_fp8 import NativeBlockFP8Decoder


def bind_fp8_expert_policy(core, checkpoint, *, backend, retained_bytes_per_layer, library=None,
                           max_decoded_expert_bytes=64*2**20):
    """Prevalidate ALL new stores, then replace expert bindings together.

    `cpu-tiled`/`cpu-vectorized` decode on CPU and honor the requested execution
    device; `cuda-byte-decode` requires an already explicitly placed SM70 FP16/
    FP32 core and a prebuilt library. No model/device/default routing changes.
    A positive per-layer budget wraps that source in an explicit decoded LRU;
    zero leaves it uncached. This is not full-model native/quality qualification.
    """
    if not isinstance(core, FlashNextTextCore) or core.training:
        raise ValueError('Explicit eval floating text reference required')
    if backend not in ('cpu-tiled', 'cpu-vectorized', 'cuda-byte-decode'):
        raise ValueError('Explicit qualified expert policy required')
    budgets = tuple(retained_bytes_per_layer)
    if len(budgets) != len(core.layers) or any(type(n) is not int or n < 0 for n in budgets):
        raise ValueError('One explicit nonnegative retained-weight budget per layer required')
    c, geometry = FlashNextConfig.from_dict(dict(checkpoint.config)), MoEGeometry.from_dict(dict(checkpoint.config))
    if c.layers != len(core.layers) or c.hidden_size != core.embed_tokens.embedding_dim or (
            c.vocab_size != core.embed_tokens.num_embeddings or c.hc_count != core.hyper_connection_mixer.streams) or any(
            layer.mlp.geometry != geometry for layer in core.layers):
        raise ValueError('Expert policy checkpoint/core geometry mismatch')
    parameters = tuple(core.parameters())
    device, dtype = parameters[0].device, parameters[0].dtype
    if any(p.device != device or p.dtype != dtype for p in parameters):
        raise ValueError('Core must already have one explicitly selected device/dtype')
    decoder = None
    if backend == 'cuda-byte-decode':
        if device.type != 'cuda' or dtype not in (torch.float32, torch.float16):
            raise ValueError('Native policy requires explicitly placed CUDA FP16/FP32 core')
        properties = torch.cuda.get_device_properties(device)
        if (properties.major, properties.minor) != (7, 0):
            raise ValueError('This explicit native policy is qualified only for SM70')
        if library is None:
            raise ValueError('Explicit prebuilt SM70 decoder library required')
        decoder = NativeBlockFP8Decoder(library, max_output_bytes=max_decoded_expert_bytes)
    elif library is not None:
        raise ValueError('CPU policies do not load a native library')
    stores = []
    for i, budget in enumerate(budgets):
        prefix = f'model.language_model.layers.{i}.mlp.experts'
        if decoder is None:
            source = BlockFP8ExpertStore(checkpoint, geometry, prefix, max_decoded_expert_bytes=max_decoded_expert_bytes,
                decode_strategy='vectorized' if backend == 'cpu-vectorized' else 'tiled')
        else:
            source = CUDABlockFP8ExpertStore(checkpoint, geometry, prefix, decoder=decoder,
                                            max_decoded_expert_bytes=max_decoded_expert_bytes)
        stores.append(CachedExpertStore(source, max_resident_bytes=budget) if budget else source)
    # All fallible admission/library/schema work precedes publication. This
    # setup call still must NOT race active requests or mutable checkpoint files.
    for layer, store in zip(core.layers, stores):
        layer.mlp.expert_store = store
    return dict(backend=backend, execution_device=str(device), execution_dtype=str(dtype),
        retained_bytes_per_layer=budgets, total_cache_retained_budget=sum(budgets),
        native_weight_decoder=decoder is not None, native_full_model=False, automatic_oom_fallback=False,
        setup_only=True, memory_scope='Retained weights only; not total VRAM/RAM admission')
