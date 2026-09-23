"""Manual routing check for feat/agent-core: 13 hardcoded text queries.

Each query has an expected route: a tool that must be called, or None for a direct answer.
This is a quick smoke test; the real scoring harness lives in eval/ (feat/eval-harness).

  python smoke_agent.py              # single-turn routing
  python smoke_agent.py --followups  # two-turn conversations (memory)
  python smoke_agent.py --grounding  # answer checking: verdicts + a forced misquote rewrite
  python smoke_agent.py -v           # also print tool calls and answers
"""
import argparse
from types import SimpleNamespace as NS

from core.agent_runtime import Agent, load_tools, print_result
from core.conversation import Conversation
from core.tool_registry import ToolRegistry

QUERIES = [
    ("What is 37 times 48?", "calculate"),
    ("144 ka square root kya hai?", "calculate"),
    ("5 kilometer ko miles mein convert karo", "convert_units"),
    ("100 degree Fahrenheit is how much in Celsius?", "convert_units"),
    ("25 கிலோகிராம் எத்தனை பவுண்டு?", "convert_units"),
    ("Kinetic energy ka formula kya hai?", "lookup_formula"),
    ("Radius 7 cm wale circle ka area kitna hoga?", "calculate"),
    ("Photosynthesis kya hota hai? Simple mein samjhao.", None),
    ("Who wrote India's national anthem?", None),
    ("ஒளிச்சேர்க்கை என்றால் என்ன?", None),
    ("Ohm's law formula enna?", "lookup_formula"),
    ("100 kilometer-ah miles-la convert pannu", "convert_units"),
    ("tamizhil ethana mei ezhuthukkal irukiradhu?", None),
]


# (first question, follow-up, tool the follow-up must call or None, value that must appear
#  in the follow-up's tool results or answer). Each follow-up only makes sense with memory.
FOLLOWUPS = [
    ("25 kg in pounds?", "and in grams?", "convert_units", "25000"),
    ("Kinetic energy ka formula kya hai?", "Ab 2 kg aur 3 m/s ke liye calculate karo", "calculate", "9"),
    ("7 cm ஆரம் கொண்ட வட்டத்தின் பரப்பளவு என்ன?", "அதன் சுற்றளவு என்ன?", "calculate", "43.98"),
    ("Who wrote India's national anthem?", "When was he born?", None, "1861"),
]


def run_followups(agent: Agent, verbose: bool) -> None:
    passed = 0
    for i, (first, followup, expected, must_contain) in enumerate(FOLLOWUPS, 1):
        conv = Conversation()
        agent.run(first, conversation=conv)
        r = agent.run(followup, conversation=conv)
        evidence = r.answer + " " + " ".join(str(c.result) for c in r.tool_calls)
        routed = expected in r.tools_used if expected else not r.tool_calls
        ok = routed and must_contain in evidence.replace(",", "")
        passed += ok
        got = ",".join(r.tools_used) or "direct"
        print(f"{'PASS' if ok else 'FAIL'}  #{i}  {first}  ->  {followup}")
        print(f"        expected={expected or 'direct'} containing {must_contain!r}; got={got}")
        if verbose or not ok:
            print_result(r)
        print()
    print(f"{passed}/{len(FOLLOWUPS)} follow-ups answered from context")


# (question, expected grounding verdict)
GROUNDING = [
    ("25 kg in pounds?", "verified"),
    ("What is 37 times 48?", "verified"),
    ("Who wrote India's national anthem?", "unverified"),
]


def run_grounding(agent: Agent, verbose: bool) -> None:
    passed = 0
    for i, (query, expected) in enumerate(GROUNDING, 1):
        r = agent.run(query)
        ok = r.grounding.verdict == expected
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  #{i}  expected={expected:<10} got={r.grounding.verdict:<10} {query}")
        if verbose or not ok:
            print_result(r)
            print()

    # Force a misquote: the first draft says 57.3 lb (the tool says 55.1156). The check must
    # catch it and the real LLM must rewrite it.
    real = agent.client
    scripted = [
        NS(tool_calls=[NS(id="call_forced", function=NS(
            name="convert_units", arguments='{"value": 25, "from_unit": "kg", "to_unit": "lb"}'))], content=""),
        NS(tool_calls=None, content="25 kg is about 57.3 pounds."),
    ]
    def create(**kw):
        return NS(choices=[NS(message=scripted.pop(0))]) if scripted else real.chat.completions.create(**kw)
    agent.client = NS(chat=NS(completions=NS(create=create)))
    try:
        r = agent.run("25 kg in pounds?")
    finally:
        agent.client = real
    ok = r.grounding.verdict == "corrected" and "57.3" not in r.answer
    passed += ok
    print(f"{'PASS' if ok else 'FAIL'}  #{len(GROUNDING) + 1}  forced misquote: draft={r.draft_answer!r}")
    print(f"        rewritten={r.answer!r}  verdict={r.grounding.verdict}")
    print(f"\n{passed}/{len(GROUNDING) + 1} grounding checks passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--followups", action="store_true", help="run two-turn memory checks")
    parser.add_argument("--grounding", action="store_true", help="run answer-checking checks")
    parser.add_argument("--provider", choices=["sarvam", "openai"], default=None)
    args = parser.parse_args()

    registry = ToolRegistry()
    agent = Agent(registry, system_prompt=load_tools("tools.study_tools", registry), provider=args.provider)
    if args.followups:
        return run_followups(agent, args.verbose)
    if args.grounding:
        return run_grounding(agent, args.verbose)

    passed = 0
    for i, (query, expected) in enumerate(QUERIES, 1):
        r = agent.run(query)
        ok = expected in r.tools_used if expected else not r.tool_calls
        passed += ok
        got = ",".join(r.tools_used) or "direct"
        print(f"{'PASS' if ok else 'FAIL'}  #{i:<2} expected={expected or 'direct':<15} "
              f"got={got:<30} {r.reply_language}  {r.latency_ms:>6.0f} ms  {query}")
        if args.verbose or not ok:
            print_result(r)
            print()

    print(f"\n{passed}/{len(QUERIES)} routed correctly")


if __name__ == "__main__":
    main()
