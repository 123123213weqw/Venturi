import hashlib

import pytest

from venturi.model_assets import ModelAsset, ModelBundle, StorageBudget, plan_moe_storage


def asset(role="backbone", size=100, **changes):
    raw = dict(filename=role+".gguf", bytes=size, sha256="a"*64,
               repository="owner/model", revision="b"*40,
               source_filename=role+".gguf", role=role,
               url=f'https://huggingface.co/owner/model/resolve/{"b"*40}/{role}.gguf')
    return ModelAsset(**(raw | changes))


def bundle():
    return ModelBundle("official/model", "c"*40, "test mixed quant",
                       (asset(), asset("sparse_lookup", 200), asset("draft", 10)))


def budget(**changes):
    return StorageBudget(**(dict(vram_free=40, ram_available=100, disk_free=500,
                                  vram_reserve=5, ram_reserve=10, disk_reserve=20) | changes))


@pytest.mark.parametrize("change", [dict(filename="../a"), dict(bytes=True), dict(bytes=-1),
    dict(sha256="0"), dict(revision="main"), dict(role="ngram_draft"),
    dict(source_filename="../backbone.gguf"), dict(source_filename="/backbone.gguf"),
    dict(url="https://huggingface.co/owner/model/resolve/main/backbone.gguf"),
    dict(url=f'https://user@huggingface.co/owner/model/resolve/{"b"*40}/backbone.gguf')])
def test_asset_identity_fail_closed(change):
    with pytest.raises(ValueError):
        asset(**change)


def test_bundle_role_bytes_and_roundtrip():
    b = bundle()
    assert b.disk_bytes == 310
    assert b.role_bytes("sparse_lookup") == 200
    assert ModelBundle.from_dict(dict(model=b.model, official_revision=b.official_revision,
        variant=b.variant, files=[vars(a) for a in b.files])) == b


def test_bundle_duplicates_or_mixed_source_revision_refused():
    b = bundle()
    with pytest.raises(ValueError, match="Duplicate"):
        ModelBundle(b.model, b.official_revision, b.variant, (b.files[0], b.files[0]))
    a = asset("draft", revision="d"*40,
        url=f'https://huggingface.co/owner/model/resolve/{"d"*40}/draft.gguf')
    with pytest.raises(ValueError, match="Mixed revisions"):
        ModelBundle(b.model, b.official_revision, b.variant, (b.files[0], a))


def test_exact_hash_not_equal_length_or_symlink(tmp_path):
    a = asset(size=3, sha256=hashlib.sha256(b"abc").hexdigest())
    path = tmp_path/a.filename
    path.write_bytes(b"abc")
    assert a.verify(tmp_path) == path
    path.write_bytes(b"xyz")
    with pytest.raises(ValueError, match="SHA256"):
        a.verify(tmp_path)
    path.unlink()
    (tmp_path/"other").write_bytes(b"abc")
    path.symlink_to(tmp_path/"other")
    with pytest.raises(ValueError, match="symlinked"):
        a.verify(tmp_path)


def test_capacity_partition_no_lookup_double_count_and_explicit_spill():
    p = plan_moe_storage(bundle(), budget(), dense_and_draft_bytes=10,
        routed_expert_bytes=120, expert_alignment=4, sparse_cache_bytes=8)
    assert p.hot_experts_vram == 24
    assert p.cold_experts_ram == 80
    assert p.cold_experts_disk == 16
    assert p.sparse_lookup_disk == 200 and p.sparse_lookup_ram_cache == 8
    assert p.additional_download_bytes == 310
    assert not p.experts_fully_memory_resident
    assert sum((p.hot_experts_vram,p.cold_experts_ram,p.cold_experts_disk)) == 120


def test_verified_existing_download_capacity_is_explicit():
    with pytest.raises(ValueError, match="Insufficient disk"):
        plan_moe_storage(bundle(), budget(disk_free=300), dense_and_draft_bytes=0,
                          routed_expert_bytes=0, expert_alignment=1)
    p = plan_moe_storage(bundle(), budget(disk_free=300), dense_and_draft_bytes=0,
            routed_expert_bytes=0, expert_alignment=1, verified_existing_bytes=100)
    assert p.additional_download_bytes == 210 and p.experts_fully_memory_resident


@pytest.mark.parametrize("change,match", [
    (dict(dense_and_draft_bytes=36),"Dense/draft"),
    (dict(expert_alignment=0),"positive"),
    (dict(routed_expert_bytes=3),"allocation unit"),
    (dict(sparse_cache_bytes=201),"exceeds lookup"),
    (dict(sparse_cache_bytes=91),"exceeds RAM"),
    (dict(verified_existing_bytes=311),"exceed the bundle"),
    (dict(dense_and_draft_bytes=True),"integer"),
])
def test_admission_rejects_invalid_plan(change, match):
    with pytest.raises(ValueError, match=match):
        plan_moe_storage(bundle(), budget(), **(dict(dense_and_draft_bytes=10,
            routed_expert_bytes=120, expert_alignment=4) | change))


@pytest.mark.parametrize("change", [dict(vram_free=-1),dict(ram_reserve=101),
                                    dict(vram_free=True),dict(disk_reserve=501)])
def test_free_memory_reservations_validated(change):
    with pytest.raises(ValueError):
        budget(**change)
