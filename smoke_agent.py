"""Manual routing check for feat/agent-core: 13 hardcoded text queries.

Each query has an expected route: a tool that must be called, or None for a direct answer.
This is a quick smoke test; the real scoring harness lives in eval/ (feat/eval-harness).

  python smoke_agent.py            # all queries
  python smoke_agent.py -v         # also print tool calls and answers
"""
import argparse

from core.agent_runtime import Agent, load_tools, print_result
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--provider", choices=["sarvam", "openai"], default=None)
    args = parser.parse_args()

    registry = ToolRegistry()
    agent = Agent(registry, system_prompt=load_tools("tools.study_tools", registry), provider=args.provider)

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
