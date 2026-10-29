import json
import subprocess
import sys

import pytest

from venturi.tokenizer_assets import inspect_deepseek41_tokenizer,verify_deepseek41_token_map


TOKENS=['<unk>',' The','the','THE',' ','','é','Ｅ','e','E','\ufffdA','\ufffda','<special>','ß','ss','\t','\n',' \t \n','\ue000']
MAP=(0,1,1,1,2,3,4,4,4,4,5,6,7,8,9,2,2,2,2)


def tokenizer_json():
    from tokenizers import Tokenizer,models
    return json.loads(Tokenizer(models.WordLevel({token:i for i,token in enumerate(TOKENS)},unk_token='<unk>')).to_str())


def test_exact_source_normalization_cases_not_naive_casefold_or_strip():
    pytest.importorskip('tokenizers')
    j=tokenizer_json();check=verify_deepseek41_token_map(j,MAP,10)
    assert check.compared_stored_map and check.vocabulary_size==len(TOKENS)
    assert check.compressed_vocabulary_size==10 and len(check.map_i32_sha256)==64
    diagnostic=inspect_deepseek41_tokenizer(j)
    assert not diagnostic.compared_stored_map and diagnostic.map_i32_sha256==check.map_i32_sha256


def test_mismatch_is_not_repaired_even_with_same_map_shape_and_range():
    pytest.importorskip('tokenizers')
    wrong=list(MAP);wrong[2]=0
    with pytest.raises(ValueError,match='token 2.*no repair'):
        verify_deepseek41_token_map(tokenizer_json(),wrong,10)
    assert wrong[2]==0  # input was not rewritten to make the test pass


@pytest.mark.parametrize('stored,count',[(MAP,9),(MAP,11),(MAP[:-1],10),([True]*len(MAP),10),(MAP,True)])
def test_token_map_identity_failures(stored,count):
    pytest.importorskip('tokenizers')
    with pytest.raises(ValueError):verify_deepseek41_token_map(tokenizer_json(),stored,count)


def test_import_is_torch_and_optional_tokenizers_free():
    subprocess.check_call([sys.executable,'-c','import sys;import venturi.tokenizer_assets;assert "torch" not in sys.modules;assert "tokenizers" not in sys.modules'])
