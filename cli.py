"""Vaak CLI: the full voice loop, STT -> agent -> TTS, with a text-only fallback.

  python cli.py --audio samples/me.m4a     # spoken query from a file
  python cli.py --text "37 times 48?"      # typed query, spoken reply
  python cli.py --text "..." --text-only   # no voice at all
  python cli.py                            # interactive session

Interactive commands:
  <question>        ask by typing
  /audio <path>     ask with an audio file
  /voice on|off     toggle spoken replies
  /quit             exit

If speech-to-text fails, the CLI asks you to type the question instead.
If text-to-speech fails, the text answer is still printed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from core.agent_runtime import Agent, load_tools
from core.pipeline import PipelineError, TurnResult, VoicePipeline
from core.tool_registry import ToolRegistry

DEFAULT_TOOLS = "tools.study_tools"


def play(path: Path) -> None:
    for player in ("afplay", "aplay", "paplay"):
        if shutil.which(player):
            subprocess.run([player, str(path)], check=False)
            return
    print(f"  (no audio player found; reply saved at {path})")


def show(turn: TurnResult, autoplay: bool) -> None:
    if turn.transcript:
        t = turn.transcript
        print(f"  heard   : {t.text}  [{t.language_code}, {t.provider}]")
    for c in turn.agent.tool_calls:
        print(f"  tool    : {c.name}({json.dumps(c.arguments, ensure_ascii=False)}) -> "
              f"{json.dumps(c.result, ensure_ascii=False)}")
    route = "tool" if turn.agent.tool_calls else "direct"
    print(f"  route   : {route}   (reply language: {turn.agent.reply_language})")
    print(f"  answer  : {turn.answer}")
    timings = "  ".join(f"{k} {v:.0f}ms" for k, v in turn.timings_ms.items())
    print(f"  timing  : {timings}")
    for err in turn.errors:
        print(f"  warning : {err} (showing text answer only)")
    if turn.speech:
        print(f"  voice   : {turn.speech.path}  [{turn.speech.provider}]")
        if autoplay:
            play(turn.speech.path)


def ask_audio(pipeline: VoicePipeline, path: str, autoplay: bool, interactive: bool) -> None:
    try:
        show(pipeline.run_audio(path), autoplay)
    except (PipelineError, FileNotFoundError) as e:
        print(f"  error   : {e}")
        if not interactive:
            sys.exit(1)
        try:
            text = input("  Voice input failed. Type your question instead: ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if text:
            show(pipeline.run_text(text), autoplay)


def interactive(pipeline: VoicePipeline, autoplay: bool) -> None:
    print("Vaak — ask by typing, or /audio <path>.  /voice on|off, /quit")
    while True:
        try:
            line = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return
        if line.startswith("/voice"):
            pipeline.speak = line.endswith("on")
            print(f"  spoken replies {'on' if pipeline.speak else 'off'}")
        elif line.startswith("/audio"):
            ask_audio(pipeline, line.removeprefix("/audio").strip(), autoplay, interactive=True)
        else:
            show(pipeline.run_text(line), autoplay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vaak voice agent")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--audio", help="audio file with a spoken question")
    src.add_argument("--text", help="typed question")
    parser.add_argument("--text-only", action="store_true", help="don't synthesize spoken replies")
    parser.add_argument("--no-play", action="store_true", help="save the reply audio but don't play it")
    parser.add_argument("--tools", default=DEFAULT_TOOLS, help="tools module to load")
    parser.add_argument("--provider", choices=["sarvam", "openai"], default=None, help="LLM provider")
    args = parser.parse_args()

    registry = ToolRegistry()
    agent = Agent(registry, system_prompt=load_tools(args.tools, registry), provider=args.provider)
    pipeline = VoicePipeline(agent, speak=not args.text_only)
    autoplay = not args.no_play

    if args.audio:
        ask_audio(pipeline, args.audio, autoplay, interactive=sys.stdin.isatty())
    elif args.text:
        show(pipeline.run_text(args.text), autoplay)
    else:
        interactive(pipeline, autoplay)


if __name__ == "__main__":
    main()
