"""Complete mixed-precision TEXT bundle: Q4 experts, unchanged BF16 core/FP8 PLE.

Not a general HF/GGUF checkpoint, BF16-derived quantization or native full model.
All expert payloads are mandatory; inference NEVER converts missing weights.
The complete marker authenticates the manifest; full payload audit is explicit.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
from types import SimpleNamespace

from .flash_next_moe import MoEGeometry
from .flash_next_fp8 import BlockFP8ExpertStore
from .flash_next_q4 import CODEC, LazyQ4ExpertArchive

FORMAT = 'venturi-flash-next-text-q4-v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*2**20),b''):h.update(data)
    return h.hexdigest()


def atomic_json(path,value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.part')
    with temporary.open('w') as f:
        json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    temporary.replace(path)


def bundle_manifest(directory, *, verify_payloads=False):
    root = Path(directory).resolve()
    marker = json.loads((root/'.venturi-q4-complete.json').read_text())
    raw = (root/'bundle.json').read_bytes()
    if marker.get('complete') is not True or marker.get('manifest_sha256') != hashlib.sha256(raw).hexdigest():
        raise ValueError('Authenticated complete Q4 bundle marker required')
    m = json.loads(raw)
    if m.get('format') != FORMAT or m.get('codec') != CODEC or m.get('scope') != 'text-only':
        raise ValueError('Explicit qualified text Q4 bundle required')
    g = MoEGeometry(**m['geometry'])
    if type(m['layers']) is not int or not 1 <= m['layers'] <= 64:
        raise ValueError('Bounded layer geometry required')
    for name,entry in m['files'].items():
        path = root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file() or path.stat().st_size != entry['bytes']:
            raise ValueError('Missing/unsafe/changed Q4 bundle file: '+name)
        if verify_payloads and digest(path) != entry['sha256']:
            raise ValueError('Q4 bundle checksum mismatch: '+name)
    expected = {f'experts/layer-{i:02d}/expert-{e:04d}.safetensors'
                for i in range(m['layers']) for e in range(g.num_experts)}
    if {n for n in m['files'] if n.startswith('experts/') and n.endswith('.safetensors')} != expected:
        raise ValueError('Every configured Q4 expert must be present, no conversion fallback')
    for name in ('config.json','model.safetensors.index.json'):
        if name not in m['files'] or digest(root/name) != m['files'][name]['sha256']:
            raise ValueError('Bundle metadata checksum mismatch')
    return m


class CompleteQ4ExpertStore(LazyQ4ExpertArchive):
    """Read-only complete archive; no source checkpoint or lazy quantizer exists."""
    def __init__(self, directory, geometry, *, source_identity):
        super().__init__(SimpleNamespace(geometry=geometry),directory,source_identity=source_identity)
        if set(self.manifest['experts']) != {str(e) for e in range(geometry.num_experts)}:
            raise ValueError('Complete Q4 bank required; missing expert cannot be converted')
        for e,entry in self.manifest['experts'].items():
            p = self.directory/f'expert-{int(e):04d}.safetensors'
            if not p.is_file() or p.stat().st_size != entry['file_bytes']:
                raise ValueError('Complete Q4 expert payload missing/changed')
        self.serialized_bytes = sum(e['encoded_bytes'] for e in self.manifest['experts'].values())

    def packed(self,expert):
        if str(expert) not in self.manifest['experts']:
            raise ValueError('Missing Q4 expert; inference conversion is forbidden')
        return super().packed(expert)


def complete_stores(checkpoint, geometry, layers):
    m = bundle_manifest(checkpoint.directory)
    if m['geometry'] != vars(geometry) or m['layers'] != layers or checkpoint.config.get('quantization_config',{}).get('quant_method') != FORMAT:
        raise ValueError('Explicit Q4 checkpoint configuration/geometry mismatch')
    stores = []
    for i in range(layers):
        path = checkpoint.directory/f'experts/layer-{i:02d}'
        name = f'experts/layer-{i:02d}/manifest.json'
        if digest(path/'manifest.json') != m['files'][name]['sha256']:
            raise ValueError('Expert bank manifest checksum mismatch')
        stores.append(CompleteQ4ExpertStore(path,geometry,source_identity=dict(m['source_identity'],layer=i)))
    return stores


def copy_tensor_subset(source, destination, specs):
    """Raw bounded streaming copy, never materialize a whole PLE table in RAM."""
    source,destination = Path(source),Path(destination)
    with source.open('rb') as f:
        length = struct.unpack('<Q',f.read(8))[0]
        if not 2 <= length <= 64*2**20:raise ValueError('Bounded safetensors header required')
        original = json.loads(f.read(length));data_start = 8+length
        header,offset = {},0
        for spec in specs:
            s = original[spec.name]
            lo,hi = s['data_offsets']
            if s['shape'] != list(spec.shape) or s['dtype'] != spec.dtype or hi-lo != spec.nbytes:
                raise ValueError('Original tensor metadata changed during export')
            header[spec.name] = dict(dtype=spec.dtype,shape=list(spec.shape),data_offsets=[offset,offset+spec.nbytes])
            offset += spec.nbytes
        encoded = json.dumps(header,separators=(',',':')).encode()
        encoded += b' '*((-len(encoded))%8)
        temporary = destination.with_suffix('.safetensors.part')
        if destination.exists() or temporary.exists():raise ValueError('Refuse overwrite of exported tensor shard')
        with temporary.open('xb') as out:
            out.write(struct.pack('<Q',len(encoded)));out.write(encoded)
            for spec in specs:
                f.seek(data_start+original[spec.name]['data_offsets'][0]);remaining=spec.nbytes
                while remaining:
                    data=f.read(min(8*2**20,remaining))
                    if not data:raise ValueError('Truncated original tensor payload')
                    out.write(data);remaining-=len(data)
            out.flush();os.fsync(out.fileno())
        temporary.replace(destination)


def export_q4_text(checkpoint, destination, *, source_identity, reuse_archive=None, workers=8):
    """CPU-only offline export. Fresh destination, immutable source; no GPU lease.

    Parallelism is across layers; one serial writer per layer. Existing Q4 files
    may be HARDLINKED read-only after SHA verification; never modified. Static
    core/PLE/metadata are independent copies. Completion requires a full rehash.
    """
    if type(workers) is not int or not 1 <= workers <= 8:raise ValueError('Bounded1..8 export workers required')
    root = Path(destination);root.mkdir(parents=True,exist_ok=False)
    g = MoEGeometry.from_dict(dict(checkpoint.config));layers = checkpoint.config['text_config']['num_hidden_layers']
    source_identity = json.loads(json.dumps(source_identity))
    if set(source_identity) != {'revision','manifest_sha256','decode_dtype'} or source_identity['decode_dtype'] != 'FP16':
        raise ValueError('Explicit authenticated FP8-to-FP16 source identity required')
    state = dict(format=FORMAT,complete=False,finished_layers=[],workers=workers)
    atomic_json(root/'export-status.json',state)
    grouped = defaultdict(list)
    for name,spec in checkpoint.tensors.items():
        if (name.startswith('model.language_model.') or name == 'lm_head.weight') and '.mlp.experts.' not in name:
            grouped[spec.file].append(spec)
    weight_map = {}
    for i,(source,specs) in enumerate(sorted(grouped.items(),key=lambda x:str(x[0]))):
        name = f'body-{i:04d}.safetensors'
        copy_tensor_subset(source,root/name,sorted(specs,key=lambda s:s.name))
        weight_map.update({s.name:name for s in specs})
    config = json.loads(json.dumps(dict(checkpoint.config)))
    config['quantization_config'] = dict(quant_method=FORMAT,expert_codec=CODEC,ple='original-global-scale-FP8',static='original-BF16',activation='FP16-reference')
    atomic_json(root/'config.json',config)
    atomic_json(root/'model.safetensors.index.json',dict(metadata=dict(total_size=sum(s.nbytes for specs in grouped.values() for s in specs)),weight_map=weight_map))
    assets=[]
    for name in ('tokenizer.json','tokenizer_config.json','generation_config.json','LICENSE','README.md','chat_template.jinja','merges.txt','vocab.json'):
        p=checkpoint.directory/name
        if p.is_file():shutil.copyfile(p,root/name);assets.append(name)
    def layer_export(i):
        path = root/f'experts/layer-{i:02d}'
        source = BlockFP8ExpertStore(checkpoint,g,f'model.language_model.layers.{i}.mlp.experts',decode_strategy='vectorized')
        identity = dict(source_identity,layer=i)
        old = None if reuse_archive is None else Path(reuse_archive)/f'layer-{i:02d}'
        if old is not None and old.exists():
            original = LazyQ4ExpertArchive(source,old,source_identity=identity)
            path.mkdir(parents=True)
            for e,entry in original.manifest['experts'].items():
                f=old/f'expert-{int(e):04d}.safetensors'
                if digest(f) != entry['sha256']:raise ValueError('Reuse Q4 checksum mismatch')
                os.link(f,path/f.name)
            atomic_json(path/'manifest.json',original.manifest)
            archive = LazyQ4ExpertArchive(source,path,source_identity=identity)
        else:archive=LazyQ4ExpertArchive(source,path,source_identity=identity,create=True)
        for e in range(g.num_experts):archive.packed(e)
        return i,archive.stats
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(layer_export,i) for i in range(layers)]
        for future in as_completed(futures):
            i,stats=future.result();state['finished_layers'].append(dict(layer=i,stats=stats));atomic_json(root/'export-status.json',state)
    files={}
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.name != 'export-status.json':
            files[str(p.relative_to(root))]=dict(bytes=p.stat().st_size,sha256=digest(p))
    m=dict(format=FORMAT,codec=CODEC,scope='text-only',source_identity=source_identity,layers=layers,geometry=vars(g),
           expert_count=layers*g.num_experts,source_precision='official FP8 -> FP16 -> RTN Q4, not BF16-derived',
           body_tensor_count=len(weight_map),ignored_components=['vision','MTP'],assets=assets,files=files)
    atomic_json(root/'bundle.json',m)
    # Publish completion ONLY AFTER a second complete payload audit. Failed or
    # interrupted exports never have a usable completion marker.
    for name,entry in files.items():
        if digest(root/name) != entry['sha256']:
            raise ValueError('Prepublication Q4 checksum mismatch: '+name)
    atomic_json(root/'.venturi-q4-complete.json',dict(complete=True,manifest_sha256=digest(root/'bundle.json')))
    bundle_manifest(root)
    state.update(complete=True,files=len(files),bytes=sum(e['bytes'] for e in files.values()),expert_count=m['expert_count'])
    atomic_json(root/'export-status.json',state)
    return state
