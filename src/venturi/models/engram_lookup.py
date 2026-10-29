"""CPU Engram hash -> IQ4_NL rows component, NOT full Engram or generation.

Explicit opt-in prototype, separate from existing Qwen/production backends. The
hash component must already be bound; the lookup source is independently fully
SHA-verified and matched by model revision, table names, row counts and widths.
No wkv projection, gating, attention, GPU, scheduler or speculative decoding.
"""
from itertools import islice

import torch

from venturi.gguf_rows import VerifiedIQ4NLRows
from venturi.model_assets import ModelAsset, ModelBundle
from venturi.models.engram_bindings import EngramComponentBinding


class EngramIQ4NLLookup:
    """One fully verified table source, private request outputs and no history writes.

    Opening scans the ENTIRE lookup asset (real candidate about 110 GB). Only
    healthy, complete storage is admissible; partial/header-only files and failed
    SSD paths must NOT be used. Thereafter only selected packed rows are fetched.
    Use a context manager/close. No persistent row cache or whole-table decode.
    """

    def __init__(self, bundle, binding, lookup_asset, root, *, max_values=2**20):
        if (not isinstance(bundle,ModelBundle) or not isinstance(binding,EngramComponentBinding)
                or bundle.model!=binding.config_asset.repository
                or bundle.official_revision!=binding.config_asset.revision
                or binding.metadata_asset not in bundle.files
                or not binding.token_identity.compared_stored_map):
            raise ValueError('Matching source-bound Engram component and explicit bundle required')
        if (not isinstance(lookup_asset,ModelAsset) or lookup_asset not in bundle.files
                or lookup_asset.role!='sparse_lookup'):
            raise ValueError('Explicit sparse-lookup asset from the same bundle required')
        spec=binding.hash_spec
        self._spec=spec
        self._head_dim=binding.head_dim
        self._columns=(spec.max_ngram_size-1)*len(spec.moduli[0][0])
        self._layer_positions={layer:i for i,layer in enumerate(spec.layer_ids)}
        self._tables={layer:f'blk.{layer}.engram_embd.weight' for layer in spec.layer_ids}
        schemas={self._tables[layer]:(rows,binding.head_dim)
                 for layer,rows in zip(spec.layer_ids,spec.table_rows)}
        self._rows=VerifiedIQ4NLRows(lookup_asset,root,expected_shapes=schemas,
            required_metadata={'general.architecture':'deepseek41-engram',
                               'deepseek41.engram.source':bundle.official_revision},max_values=max_values)

    @property
    def model_supported(self):return False

    @property
    def last_read(self):return self._rows.last_read

    @property
    def closed(self):return self._rows.closed

    def close(self):self._rows.close()

    def __enter__(self):
        self._rows.__enter__()
        return self

    def __exit__(self,*exc):self.close()

    def lookup_layer(self,layer_id,hashes):
        return self.lookup_layer_many(layer_id,(hashes,))[0]

    def lookup_layer_many(self,layer_id,requests):
        """[B,T,layers,hash_columns] -> per-request [B,T,hash_columns,head_dim].

        Only the selected layer's addressed rows are read. Layer/head/request
        order is preserved. Independent/forked hash states remain caller-owned;
        a storage failure cannot advance history here. For transactional caller
        state, hash a private fork and commit it only after every needed I/O step.
        """
        if type(layer_id) is not int or layer_id not in self._layer_positions:
            raise ValueError('Explicit bound Engram layer ID required')
        if isinstance(requests,torch.Tensor):raise ValueError('Explicit request sequence required')
        requests=tuple(islice(requests,65))
        if not 1<=len(requests)<=64:raise ValueError('Bounded nonempty request sequence required')
        index=self._layer_positions[layer_id]
        selected=[]
        for hashes in requests:
            if (not isinstance(hashes,torch.Tensor) or hashes.device.type!='cpu'
                    or hashes.dtype!=torch.int64 or hashes.ndim!=4
                    or not 1<=hashes.shape[0]<=64
                    or hashes.shape[2:]!=(len(self._spec.layer_ids),self._columns)):
                raise ValueError('CPU int64 [B,T,bound_layers,hash_columns] hashes required')
            selected.append(hashes[:,:,index,:])
        return self._rows.lookup_many(self._tables[layer_id],selected)
