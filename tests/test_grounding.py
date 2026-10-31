import pytest

from core.grounding import check_answer, collect_known, known_from_messages

CONVERSION = {"value": 25, "from_unit": "kg", "to_unit": "lb", "result": 55.1156}


@pytest.mark.parametrize("answer", [
    "25 kilograms is approximately 55.12 pounds.",
    "25 kg is 55.11 pounds.",                    # truncated
    "25 kg is about 55 pounds.",                 # rounded to integer
    "25 கிலோகிராம் சுமார் 55.12 பவுண்டு.",         # Tamil text, digits
])
def test_rounded_or_restated_numbers_are_verified(answer):
    assert check_answer(answer, collect_known([CONVERSION]), had_tools=True).verdict == "verified"


@pytest.mark.parametrize("answer,bad", [
    ("25 kg is 55.21 pounds.", "55.21"),         # misquote
    ("25 kg is 1,00,000 grams", "1,00,000"),     # invented, Indian digit grouping
])
def test_unsupported_numbers_are_flagged(answer, bad):
    g = check_answer(answer, collect_known([CONVERSION]), had_tools=True)
    assert g.verdict == "flagged" and g.unsupported == [bad]


def test_symbols_glued_to_letters_are_not_claims():
    known = collect_known([{"formula": "KE = (1/2) * m * v^2"}])
    assert check_answer("KE = (1/2) m v2", known, had_tools=True).verdict == "verified"


def test_direct_answers_are_unverified_not_flagged():
    g = check_answer("Tagore was born in 1861.", collect_known([CONVERSION]), had_tools=True, turn_used_tools=False)
    assert g.verdict == "unverified"


def test_direct_answer_restating_earlier_tool_result_is_verified():
    g = check_answer("That was 55.12 pounds.", collect_known([CONVERSION]), had_tools=True, turn_used_tools=False)
    assert g.verdict == "verified"


def test_known_from_messages_reads_user_tool_args_and_results():
    messages = [
        {"role": "user", "content": "25 kg in pounds?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "convert_units", "arguments": '{"value": 25, "from_unit": "kg", "to_unit": "lb"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"result": 55.1156}'},
    ]
    known, had_tools = known_from_messages(messages)
    assert had_tools and {25.0, 55.1156} <= known
