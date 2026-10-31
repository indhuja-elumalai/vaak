import json
from pathlib import Path

import pytest

from eval.scoring import comparable_scripts, completion_ok, percentile, route_ok, word_error_rate

EVAL_SET = Path(__file__).resolve().parent.parent / "eval" / "study_eval_set.json"


def test_route_ok():
    assert route_ok(["calculate"], ["lookup_formula", "calculate"])
    assert route_ok(["direct"], [])
    assert not route_ok(["direct"], ["calculate"])
    assert route_ok(["direct", "convert_units"], ["convert_units"])


def test_completion_values_and_text():
    assert completion_ok("25 kg is about 55.12 pounds.", expect_values=[55.1156])
    assert not completion_ok("25 kg is about 57.3 pounds.", expect_values=[55.1156])
    assert completion_ok("It is V = I × R.", expect_text=["v=i×r"])
    assert completion_ok("Rabindranath Tagore wrote it.", expect_text=["tagore"])
    assert not completion_ok("anything")  # no expectation = no credit


def test_wer_and_script_comparability():
    assert word_error_rate("What is 37 times 48?", "what is 37 times 48") == 0
    assert word_error_rate("a b c d", "a x c") == pytest.approx(0.5)
    assert comparable_scripts("What is 37 times 48?", "What is 37 times 48?")
    assert not comparable_scripts("Kinetic energy ka formula kya hai?", "काइनेटिक एनर्जी का फार्मूला क्या है?")


def test_percentile_nearest_rank():
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert percentile([5.0], 50) == 5.0


def test_eval_set_is_well_formed():
    spec = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    ids = [c["id"] for c in spec["cases"]]
    assert len(ids) == len(set(ids)) >= 20
    for case in spec["cases"]:
        assert case["voice"] in {"en-IN", "hi-IN", "ta-IN"}
        for turn in case["turns"]:
            assert turn["query"] and turn["expect_route"]
            assert turn.get("expect_values") or turn.get("expect_text")
