"""Independent planted-fact scoring and deterministic long-context input construction."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from validate_qwen_long_context import LABELS, MISSING, build_prompt, compare_ids, grade  # noqa: E402


def test_registry_is_deterministic_bounded_and_does_not_leak_answers_in_question():
    def encode(text):
        # A simple independent byte tokenizer.
        return list(text.encode())

    a = build_prompt(encode, 8192, 17)
    b = build_prompt(encode, 8192, 17)
    other = build_prompt(encode, 8192, 43)
    assert a == b and a["prompt_sha256"] != other["prompt_sha256"]
    assert a["raw_prompt_tokens"] <= 8192
    assert a["token_ids"] == encode(a["prompt"])
    document, question = a["prompt"].split("</registry>")
    for label in LABELS:
        assert document.count(a["expected"][label]) == 1
        assert a["expected"][label] not in question
    assert MISSING not in document and a["expected"][MISSING] is None
    fractions = [n["fraction_of_raw_prompt"] for n in a["needles"]]
    assert fractions == sorted(fractions)
    assert all(
        abs(actual - target) < 0.09
        for actual, target in zip(fractions, (0.05, 0.5, 0.95))
    )


@pytest.mark.parametrize(
    "text,score,exact",
    [
        ('{"A":"ABCD","B":null}', 2, True),
        ('```json\n{"A":"ABCD","B":null}\n```', 2, True),
        ('{"A":"WRONG","B":null}', 1, False),
        ('{"A":"ABCD"}', 1, False),
        ('{"A":"ABCD","B":null,"extra":0}', 2, False),
        ("The answer is ABCD.", 0, False),
        ('["ABCD",null]', 0, False),
    ],
)
def test_quality_is_scored_against_known_facts_not_id_equality(text, score, exact):
    result = grade(text, dict(A="ABCD", B=None))
    assert result["fields_correct"] == score
    assert result["all_correct"] is exact


def test_matching_id_prefix_is_not_mistaken_for_answer_quality():
    assert compare_ids(dict(generated_ids=[1, 2]), dict(generated_ids=[1, 3])) == dict(
        equal=False, matching_prefix=1
    )
    with pytest.raises(ValueError):
        build_prompt(lambda text: list(text), 100, 0)
