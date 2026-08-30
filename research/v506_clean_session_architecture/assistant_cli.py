
from __future__ import annotations

import argparse
import sqlite3
import sys

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
    ap.add_argument(
        "--freeze-knowledge",
        action="store_true",
        help="Disable static semantic/corpus knowledge",
    )
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
    memory.set_knowledge_frozen(args.freeze_knowledge)
    assistant=CognitiveAssistant(
        memory,
        Perceiver(nlp),
        llm=llm,
        trace=not args.no_trace,
    )

    print()
    print("V506 clean-session cognitive assistant ready.")
    print("Architecture owns state, typed relevance, operators, planning and choice.")
    print("The LLM is optional participant + constrained realizer.")
    print("Commands: /new  /status  /freeze  /unfreeze  /quit")
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
        if low in {"/new","//new"}:
            assistant.reset()
            memory.set_knowledge_frozen(
                args.freeze_knowledge
            )
            print(
                "New conversation. "
                f"knowledge_frozen={memory.knowledge_is_frozen()}",
                flush=True,
            )
            continue

        if low=="/freeze":
            memory.set_knowledge_frozen(True)
            print("Static knowledge frozen.",flush=True)
            continue

        if low=="/unfreeze":
            memory.set_knowledge_frozen(False)
            print("Static knowledge unfrozen.",flush=True)
            continue
        if low=="/status":
            print({
                "session":memory.session_id,
                "topic":memory.topic,
                "goal":memory.goal,
                "turns":memory.turn_index,
                "live_facts":len(memory.facts()),
                "knowledge_frozen":memory.knowledge_is_frozen(),
            })
            continue

        # Internal cognition trace is diagnostic output, never conversation
        # output. Capture it separately from the public response.
        import io
        from contextlib import redirect_stdout

        trace_buffer=io.StringIO()
        with redirect_stdout(trace_buffer):
            result=assistant.respond(text)

        trace=trace_buffer.getvalue()
        if trace:
            sys.stderr.write(trace)
            sys.stderr.flush()

        print(result["response"],flush=True)


if __name__=="__main__":
    main()
