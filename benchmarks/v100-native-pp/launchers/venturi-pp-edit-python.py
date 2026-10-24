from pathlib import Path
p=Path('src/venturi/models/qwen_native.py');s=p.read_text()
s=s.replace('''    sizes = (u64 * 4)()''','''    stage = n.venturi_qwen_stage_create
    stage.argtypes = signatures["venturi_qwen_create"][0] + [integer, integer]
    stage.restype = ptr
    sizes = (u64 * 4)()''')
s=s.replace('''        tensor_parallel_transport="peer",
    ):''','''        tensor_parallel_transport="peer",
        _stage_range=None,
    ):''',1)
s=s.replace('''        cfg = copy.deepcopy(owned.config)
        rope''','''        cfg = copy.deepcopy(owned.config)
        if _stage_range is not None:
            if (
                not isinstance(_stage_range, tuple) or len(_stage_range) != 2
                or any(type(i) is not int for i in _stage_range)
                or not 0 <= _stage_range[0] < _stage_range[1] <= cfg["num_hidden_layers"]
                or weight_mode != "resident" or peer is not None
            ):
                raise ValueError("Internal pipeline stage requires a valid resident non-TP range")
        rope''')
s=s.replace('''        if peer is not None:
            primary =''','''        if _stage_range is not None:
            # A conservative bound for small loader/native constants remains
            # shared with the full adapter. Large Q8 payloads are stage-local.
            selected = {}
            first, last = _stage_range
            for spec in owned.plan.bindings:
                if spec.format != "q8_g32_fp16":
                    continue
                role = spec.binding
                include = (
                    role == "text/token_embedding" and first == 0
                    or role == "text/output_head" and last == cfg["num_hidden_layers"]
                    or role.startswith("text/layers/")
                    and first <= int(role.split("/")[2]) < last
                )
                if include:
                    obj = owned._object(role)
                    n, k = spec.shape
                    padded = (k + 127) // 128 * 128
                    selected[obj["offset"]] = n * (padded + padded // 32 * 2)
            required = sum(selected.values()) + 2 * small + 4 * 2**20 + rope.rotary_dim * 2
        if peer is not None:
            primary =''')
s=s.replace('''        handle = native.venturi_qwen_create(
''','''        create = native.venturi_qwen_stage_create if _stage_range is not None else native.venturi_qwen_create
        handle = create(
''')
s=s.replace('''            rope.inverse_frequency.data_ptr(),
        )
        if not handle:''','''            rope.inverse_frequency.data_ptr(),
            *(_stage_range or ()),
        )
        if not handle:''')
p.write_text(s)
