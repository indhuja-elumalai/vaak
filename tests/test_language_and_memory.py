import pytest

from core.agent_runtime import detect_language, reply_instruction
from core.conversation import Conversation, ConversationStore


@pytest.mark.parametrize("text,lang,script", [
    ("25 கிலோகிராம் எத்தனை பவுண்டு?", "ta-IN", "native"),
    ("नमस्ते, फोर्स क्या है?", "hi-IN", "native"),
    ("144 ka square root kya hai?", "hi-IN", "latin"),
    ("Ohm's law formula enna?", "ta-IN", "latin"),
    ("tamizhil ethana mei ezhuthukkal irukiradhu?", "ta-IN", "latin"),   # word shapes, not a list hit
    ("Newton oda moondraavadhu vidhi ennadhu?", "ta-IN", "latin"),
    ("Explain the chemical and physical properties of water", "unknown", "latin"),  # no false Tamil
    ("How many young people live among us until April?", "unknown", "latin"),
])
def test_detect_language(text, lang, script):
    assert detect_language(text) == (lang, script)


def test_stt_language_overrides_text_detection():
    lang, instruction = reply_instruction("some english words", "ta-IN")
    assert lang == "ta-IN" and "Tanglish" in instruction


def test_romanized_text_beats_an_en_stt_label():
    # STT tags romanized Tanglish audio as en-IN; the Tamil markers in the transcript win.
    assert reply_instruction("Ohm's Law Formula enna?", "en-IN")[0] == "ta-IN"
    assert reply_instruction("What is Ohm's law?", "en-IN")[0] == "en-IN"


def _turn(i):
    return [
        {"role": "user", "content": f"q{i}"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}"}]},
        {"role": "tool", "tool_call_id": f"c{i}", "content": "{}"},
        {"role": "assistant", "content": f"a{i}"},
    ]


def test_conversation_trims_whole_turns():
    conv = Conversation(max_turns=2)
    for i in range(3):
        conv.add_turn(_turn(i))
    msgs = conv.messages()
    assert len(conv) == 2 and msgs[0]["content"] == "q1"
    # A tool result is never kept without the assistant message that called it.
    assert msgs[0]["role"] == "user"


def test_store_evicts_least_recently_used():
    store = ConversationStore(max_sessions=2)
    a, b = store.get("session-a"), store.get("session-b")
    store.get("session-a")          # touch a
    store.get("session-c")          # evicts b
    assert store.get("session-a") is a
    assert store.get("session-b") is not b


@pytest.mark.parametrize("text,lang", [("వ్యాపారం అంటే ఏమిటి?", "te-IN"), ("ಕನ್ನಡ ಪ್ರಶ್ನೆ", "kn-IN"), ("മലയാളം", "ml-IN")])
def test_other_indic_scripts_are_recognised(text, lang):
    assert detect_language(text) == (lang, "native")


def test_reply_instruction_never_crashes_on_unsupported_language():
    assert reply_instruction("వ్యాపారం", "te-IN")[0] == "te-IN"


@pytest.mark.parametrize("text,stt_label,refused", [
    ("వ్యాపారం అంటే ఏమిటి?", "te-IN", True),   # Telugu script: refuse
    ("vyaparam ante enti", "te-IN", True),        # Latin text: the STT label decides
    ("व्यापार क्या है", "mr-IN", False),          # Devanagari is Hindi even if labelled Marathi
    ("Ohm's law formula enna?", "en-IN", False),
    ("25 கிலோ", "ta-IN", False),
])
def test_pipeline_refuses_unsupported_languages(text, stt_label, refused):
    from core.pipeline import UnsupportedLanguageError, _check_supported
    if refused:
        with pytest.raises(UnsupportedLanguageError):
            _check_supported(text, stt_label)
    else:
        _check_supported(text, stt_label)
