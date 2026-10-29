"""Experimental floating text binding, never a native/production route.

Preflight all text schemas and resident parameter bytes before materializing
any weight. Fused floating or explicitly decoded FP8 experts/PLE remain CPU file-backed
readers; selected experts/rows are fetched on demand. CPU default or explicit
one-device CUDA placement. No Q8/Q4 or FP8-GEMM, automatic placement, hot switch
or whole-official-model qualification.
"""

from dataclasses import dataclass
import math
from types import MappingProxyType

from safetensors import safe_open
import torch
from torch import nn

from ..dtypes import torch_dtype
from ..flash_next import FlashNextConfig
from .flash_next_attention import FlashNextQSAAttention, QSAAttentionGeometry
from .flash_next_fp8 import BlockFP8ExpertStore
from .flash_next_gdn import FlashNextGDN, GDNGeometry
from .flash_next_layer import FlashNextDecoderLayer
from .flash_next_moe import FlashNextMoE, FusedFloatingExpertStore, MoEGeometry
from .flash_next_ngram import FP8ShardedNGramTable, ShardedNGramTable
from .flash_next_ple import FlashNextPLE, PLEGeometry
from .flash_next_residual import FlashNextGatedResidual
from .flash_next_text import FlashNextTextCore


@dataclass(frozen=True)
class LoadedFloatingText:
    core: FlashNextTextCore
    tables: object  # read-only mapping: zero-based decoder index -> CPU reader
    report: object


def _preflight(checkpoint, *, max_core_bytes, max_expert_bytes, max_sequence, attention_backend,
               weight_encoding='floating'):
    raw = dict(checkpoint.config)
    c = FlashNextConfig.from_dict(raw)
    text = raw['text_config']
    if weight_encoding not in ('floating', 'block-fp8-cpu', 'q4-g64-fp16'):
        raise ValueError('Explicit qualified floating or CPU block-FP8 weight encoding required')
    q4 = weight_encoding == 'q4-g64-fp16'
    fp8 = weight_encoding in ('block-fp8-cpu','q4-g64-fp16')
    if not fp8 and (any(raw.get(k) for k in ('quantization_config', 'compression_config')) or any(
            text.get(k) for k in ('quantization_config', 'compression_config'))):
        raise ValueError('Only original floating text binding; quantized checkpoints require a separate codec')
    if attention_backend not in ('eager', 'sdpa'):
        raise ValueError('Explicit eager/sdpa reference backend required')
    if any(type(v) is not int or v <= 0 for v in (max_core_bytes, max_expert_bytes, max_sequence)) or (
            max_sequence > c.max_context):
        raise ValueError('Positive explicit resident/expert/sequence budgets within model context required')
    if c.layers > 64:
        raise ValueError('Reference supports at most64 explicitly configured decoder layers')
    rope = text.get('rope_parameters')
    if not isinstance(rope, dict) or rope.get('rope_type') != 'default':
        raise ValueError('Only static/default contiguous text RoPE is qualified')
    theta, factor = rope.get('rope_theta'), rope.get('partial_rotary_factor', 1.0)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
           for v in (theta, factor)) or factor > 1:
        raise ValueError('Positive finite default theta and partial rotary factor required')
    rotary_dim = int(c.head_dim*factor)
    if not 2 <= rotary_dim <= 256 or rotary_dim % 2:
        raise ValueError('Bounded even default rotary width required')
    if text.get('mamba_ssm_dtype', 'float32') != 'float32':
        raise ValueError('Only FP32 recurrent state is qualified')
    pad = text.get('pad_token_id')
    if pad is not None and (type(pad) is not int or not 0 <= pad < c.vocab_size):
        raise ValueError('Invalid explicit embedding padding token')
    mg, pg = MoEGeometry.from_dict(raw), PLEGeometry.from_dict(raw)
    gg = GDNGeometry.from_dict(raw) if 'linear_attention' in c.layer_types else None
    qg = QSAAttentionGeometry.from_dict(raw) if 'full_attention' in c.layer_types else None
    root = 'model.language_model'
    # Store construction validates fused expert shape/dtype/one-expert byte
    # admission using metadata only; it does not get_tensor the whole bank.
    banks = [BlockFP8ExpertStore(checkpoint, mg, f'{root}.layers.{i}.mlp.experts',
        max_decoded_expert_bytes=max_expert_bytes) if fp8 else FusedFloatingExpertStore(
        checkpoint, mg, f'{root}.layers.{i}.mlp.experts', max_expert_bytes=max_expert_bytes) for i in range(c.layers)] if not q4 else None
    if q4:
        from .flash_next_q4_bundle import complete_stores
        if 3*mg.hidden_size*mg.intermediate_size*4 > max_expert_bytes:
            raise ValueError('Q4 expert decoded scratch exceeds explicit byte budget')
        banks = complete_stores(checkpoint,mg,c.layers)
    with torch.device('meta'):
        layers = []
        for i, kind in enumerate(c.layer_types):
            attention = FlashNextGDN(gg) if kind == 'linear_attention' else FlashNextQSAAttention(
                qg, attention_backend=attention_backend)
            layers.append(FlashNextDecoderLayer(attention, FlashNextMoE(mg, banks[i]),
                FlashNextGatedResidual(c.hidden_size, c.hc_count, c.hc_lowrank, eps=pg.eps),
                FlashNextGatedResidual(c.hidden_size, c.hc_count, c.hc_lowrank, eps=pg.eps),
                ple=FlashNextPLE(pg) if i+1 in c.ple_layers else None))
        embedding = nn.Embedding(c.vocab_size, c.hidden_size, padding_idx=pad)
        mixer = FlashNextGatedResidual(c.hidden_size, c.hc_count, c.hc_lowrank, eps=pg.eps, use_combine=False)
        head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
    # Outside the meta context: static/default inverse frequencies are CPU FP32.
    prototype = FlashNextTextCore(embedding, layers, mixer, head, rotary_dim=rotary_dim,
        rope_theta=theta, max_sequence=max_sequence)
    specs = {}
    for key, parameter in prototype.named_parameters():
        if key.startswith('lm_head.'):
            name = key
        else:
            name = root+'.'+key
            if key.startswith('layers.') and '.attention.' in key:
                index = int(key.split('.')[1])
                name = name.replace('.attention.', '.linear_attn.' if c.layer_types[index] == 'linear_attention'
                                    else '.self_attn.', 1)
        spec = checkpoint.tensors.get(name)
        if spec is None or spec.shape != tuple(parameter.shape) or spec.dtype not in ('BF16', 'F16', 'F32'):
            raise ValueError('Missing/incompatible original floating text weight: '+name)
        specs[key] = spec
    dtypes = {spec.dtype for spec in specs.values()}
    if not fp8:
        dtypes |= {s.dtype for bank in banks for s in bank.specs}
    if len(dtypes) != 1:
        raise ValueError('All bound floating text/experts must share one dtype')
    dtype = next(iter(dtypes))
    if fp8 and dtype != 'BF16':
        raise ValueError('Official FP8 weight bridge requires BF16 static text parameters')
    resident_bytes = sum(s.nbytes for s in specs.values())
    if resident_bytes > max_core_bytes:
        raise ValueError('Resident floating text parameters exceed explicit core byte budget')
    # Validate every PLE shard/multiplier before the first payload read. This
    # strict floating API never silently switches codecs; the separate explicit
    # FP8-weight bridge uses its own global-scale row reader, not an FP8 GEMM.
    table_bytes = 0
    for ple_index, layer in enumerate(c.ple_layers):
        prefix = f'{root}.layers.{layer-1}.ple.ple_embedding'
        _, _, padded = c.ngram_geometry(ple_index)
        for part in range(c.split_ngram_parts):
            name = f'{prefix}.ngram_embedding.shard_{part}.weight'
            spec = checkpoint.tensors.get(name)
            if spec is None or spec.shape != (padded//c.split_ngram_parts, c.ple_embed_dim//c.ngram_heads) or spec.dtype != ('F8_E4M3' if fp8 else dtype):
                raise ValueError('Original same-dtype floating PLE shard or explicit FP8 code required: '+name)
            table_bytes += spec.nbytes
        if fp8:
            scale = checkpoint.tensors.get(prefix+'.ngram_embedding.weight_scale')
            if scale is None or scale.dtype != 'BF16' or scale.shape != (1,):
                raise ValueError('Official FP8 PLE requires one BF16 global multiplier')
            table_bytes += scale.nbytes
        spec = checkpoint.tensors.get(prefix+'.layer_multipliers')
        if spec is None or spec.dtype != 'I64' or spec.shape != (c.ngram_size,):
            raise ValueError('Serialized I64 PLE multipliers required')
    report = dict(scope='CPU floating reference binding; not an official-model qualification or native engine',
        dtype=dtype, layers=c.layers, resident_parameter_bytes=resident_bytes,
        file_backed_expert_bytes=sum(bank.serialized_bytes for bank in banks) if fp8 else
            sum(s.nbytes for bank in banks for s in bank.specs),
        file_backed_ple_bytes=table_bytes, static_tensor_count=len(specs), max_sequence=max_sequence,
        attention_backend=attention_backend, rotary_dim=rotary_dim, rope_theta=theta,
        expert_policy='CPU file-backed one selected expert at a time; no residency cache/eviction',
        ple_policy='CPU file-backed addressed rows; no full table materialization',
        memory_scope='Resident parameter bytes only; not peak RAM/VRAM/runtime-cache/scratch admission',
        ignored_components=['vision', 'MTP'], native_generation_supported=False,
        whole_official_model_qualified=False, quantized_body_supported=False)
    if fp8:
        report.update(weight_encoding='E4M3FN expert blocks128/BF16 block scales + global-scale FP8 PLE',
            expert_policy='Selected expert CPU FP32 dequant products -> explicit floating execution dtype; no cache/eviction',
            ple_policy='Addressed CPU FP8 -> BF16 rows times BF16 global scale; no full table materialization',
            activation_quantization_supported=False, fp8_gemm_supported=False,
            fp8_backend_numerical_equivalence_qualified=False,
            quantized_body_supported=False, weight_decoding_reference_supported=True)
    if q4:
        report.update(weight_encoding='explicit complete Q4-g64-FP16 experts + original FP8 PLE/BF16 core',
                      inference_quantization_allowed=False,expert_policy='Complete immutable Q4 archive, no source checkpoint dependency')
    return prototype, specs, c, report


def plan_floating_text(checkpoint, **options):
    """Metadata-only strict preflight; no tensor payload materialization."""
    return _execution_plan(checkpoint, options, weight_encoding='floating')[-1]



def plan_weight_decoded_fp8_text(checkpoint, **options):
    """Strict metadata-only plan for the explicit CPU FP8-weight bridge."""
    return _execution_plan(checkpoint, options, weight_encoding='block-fp8-cpu')[-1]


def _execution_plan(checkpoint, options, *, weight_encoding):
    # Device parsing/byte arithmetic only: planning never initializes CUDA or
    # reads payloads. CPU remains the default; CUDA index/dtype are explicit.
    options = dict(options)
    device = torch.device(options.pop('execution_device', 'cpu'))
    dtype = options.pop('execution_dtype', None)
    budget = options.pop('max_execution_core_bytes', options['max_core_bytes'])
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and device.index is None) or (
            device.type == 'cpu' and device.index is not None):
        raise ValueError('Explicit cpu or indexed cuda:N execution device required')
    if dtype is not None and dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError('Explicit floating execution dtype required')
    if type(budget) is not int or budget <= 0:
        raise ValueError('Positive execution core parameter byte budget required')
    prototype, specs, c, report = _preflight(checkpoint, **options, weight_encoding=weight_encoding)
    dtype = torch_dtype(report['dtype']) if dtype is None else dtype
    size = torch.empty((), device='meta', dtype=dtype).element_size()
    execution_bytes = sum(s.numel for s in specs.values()) * size
    if execution_bytes > budget:
        raise ValueError('Execution core parameters exceed explicit byte budget')
    report.update(execution_device=str(device), execution_dtype=str(dtype),
        execution_core_parameter_bytes=execution_bytes, execution_core_parameter_budget=budget,
        placement_scope='One-device static text core; expert/PLE files stay CPU-backed; no TP/PP or runtime OOM admission',
        loading_scope='Allocate target static weights directly; no complete CPU core copy before CUDA placement')
    return prototype, specs, c, report


def load_floating_text(checkpoint, **options):
    """Own admitted original floating static weights; lazy CPU expert/PLE files.

    CPU default; optional explicit indexed CUDA/dtype. No download/route change.
    Files must stay immutable/available.
    Quantized checkpoints are rejected, never silently routed to an emulation.
    """
    return _load(checkpoint, options, weight_encoding='floating')


def load_weight_decoded_fp8_text(checkpoint, **options):
    """EXPLICIT experimental FP8 weight-file -> floating reference, CPU default.

    Decodes selected experts and addressed PLE rows, not activation quantization
    or a GPU FP8 kernel. Does NOT reproduce a W8A8 engine's activation rounding.
    Whole-original-model quality/performance and native support are unqualified.
    """
    return _load(checkpoint, options, weight_encoding='block-fp8-cpu')


def load_weight_decoded_q4_text(checkpoint, **options):
    """Complete self-contained mixed-precision text bundle, canonical FP16 only."""
    options.setdefault('execution_dtype',torch.float16)
    if options['execution_dtype'] != torch.float16:
        raise ValueError('Explicit FP16 Q4 execution required')
    return _load(checkpoint,options,weight_encoding='q4-g64-fp16')


def _load(checkpoint, options, *, weight_encoding):
    prototype, specs, c, report = _execution_plan(checkpoint, options, weight_encoding=weight_encoding)
    device = torch.device(report['execution_device'])
    dtype = getattr(torch, report['execution_dtype'].split('.')[-1])
    if device.type == 'cuda':
        if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
            raise ValueError('Requested CUDA device is unavailable; no CPU fallback')
        properties = torch.cuda.get_device_properties(device)
        if dtype == torch.bfloat16 and properties.major < 8:
            raise ValueError('BF16 reference GEMM is not supported on this GPU; explicitly select FP16/FP32')
        if report['execution_core_parameter_bytes'] > torch.cuda.mem_get_info(device)[0]:
            raise ValueError('Static core alone exceeds currently free CUDA memory; no CPU fallback')
    factory = FP8ShardedNGramTable if weight_encoding in ('block-fp8-cpu','q4-g64-fp16') else ShardedNGramTable
    tables = {layer-1: factory(checkpoint, c, ple_index=i, max_tokens=min(options['max_sequence'], 65536))
              for i, layer in enumerate(c.ple_layers)}
    # Cast META parameters BEFORE allocating on the explicit target. This avoids
    # a complete CPU core plus its converted CUDA copy being live together.
    model = prototype.to(dtype=dtype).to_empty(device=device)
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            spec = specs[key]
            with safe_open(str(spec.file), framework='pt', device='cpu') as reader:
                view = reader.get_slice(spec.name)
                if tuple(view.get_shape()) != spec.shape or view.get_dtype() != spec.dtype:
                    raise ValueError('Floating text backing metadata changed')
                parameter.copy_(reader.get_tensor(spec.name))
                if not bool(torch.isfinite(parameter).all()):
                    raise ValueError('Nonfinite/overflowing static text parameter after execution cast: '+spec.name)
    return LoadedFloatingText(model.eval(), MappingProxyType(tables), MappingProxyType(report))
