
from __future__ import annotations

import argparse
import sqlite3

try:
    import spacy
except ImportError:
    spacy=None

from memory import TypedMemory
from perception import Perceiver
from assistant_core import CognitiveAssistant
from llm import LocalLLM


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "--memory",
        default=r".\results\combined_cognitive_memory.sqlite",
    )
    ap.add_argument("--teacher",default="")
    ap.add_argument("--parser-model",default="en_core_web_sm")
    ap.add_argument("--no-trace",action="store_true")
    args=ap.parse_args()

    con=sqlite3.connect(args.memory)

    nlp=None
    if spacy:
        try:
            print(
                f"[PARSER] loading {args.parser_model}...",
                flush=True,
            )
            nlp=spacy.load(args.parser_model)
            print("[PARSER] ready",flush=True)
        except Exception as exc:
            print(
                f"[PARSER] unavailable: {exc}; "
                "using lightweight parser",
                flush=True,
            )

    llm=None
    if args.teacher:
        llm=LocalLLM(args.teacher,max_new_tokens=80)

    memory=TypedMemory(con)
    assistant=CognitiveAssistant(
        memory,
        Perceiver(nlp),
        llm=llm,
        trace=not args.no_trace,
    )

    print()
    print("V492 typed-memory cognitive assistant ready.")
    print("Architecture owns state, typed relevance, operators, planning and choice.")
    print("The LLM is optional participant + constrained realizer.")
    print("Commands: /new  /status  /quit")
    print()

    while True:
        try:
            text=input("You: ").strip()
        except (EOFError,KeyboardInterrupt):
            print()
            return

        if not text:
            continue
        low=text.lower()

        if low=="/quit":
            return
        if low=="/new":
            assistant.reset()
            print("New conversation.")
            continue
        if low=="/status":
            print({
                "session":memory.session_id,
                "topic":memory.topic,
                "goal":memory.goal,
                "turns":memory.turn_index,
                "live_facts":len(memory.facts()),
            })
            continue

        assistant.respond(text)


if __name__=="__main__":
    main()
