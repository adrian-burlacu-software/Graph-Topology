from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def build(args):
    import spacy
    from memory import TypedMemory
    from perception import Perceiver
    from assistant_core import CognitiveAssistant
    from llm_backend import LocalSmolLM3
    from llm_interface import StructuredAnswerInterface
    from topic_catalog import topic_rows, format_topics, inspect_topic, format_topic_inspection

    con = sqlite3.connect(str(Path(args.memory)))
    con.row_factory = sqlite3.Row
    memory = TypedMemory(con)
    memory.set_knowledge_frozen(args.freeze_knowledge)

    try:
        nlp = spacy.load(os.environ.get("STARK_PARSER_MODEL", "en_core_web_sm"))
    except Exception:
        nlp = None

    llm = LocalSmolLM3(
        args.model,
        max_new_tokens=args.max_new_tokens,
        quantization=args.quantization,
        trace=not args.no_trace,
    )
    interface = StructuredAnswerInterface(
        llm,
        trace=not args.no_trace,
        show_request=not args.no_llm_debug,
    )
    assistant = CognitiveAssistant(
        memory,
        Perceiver(nlp),
        llm=llm,
        trace=not args.no_trace,
        interface=interface,
    )
    assistant._topic_rows = lambda limit=40: topic_rows(memory.con, limit)
    assistant._format_topics = lambda limit=25: format_topics(assistant._topic_rows(max(limit, 40)), limit)
    assistant._inspect_topic = lambda topic, depth=2: format_topic_inspection(inspect_topic(memory.con, topic, depth=depth, limit=16))
    return assistant, memory


def main():
    ap = argparse.ArgumentParser(description="V530 instrumented cognitive knowledge interface")
    ap.add_argument("--memory", required=True)
    ap.add_argument("--model", "--teacher", dest="model", required=True)
    ap.add_argument("--freeze-knowledge", action="store_true")
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="4bit")
    ap.add_argument("--no-llm-debug", action="store_true", help="hide the exact structured request and rendered prompt sent to the LLM")
    args = ap.parse_args()

    assistant, memory = build(args)
    print("V530 instrumented cognitive knowledge assistant ready.", flush=True)
    print("Architecture owns parsing, goals, state, deterministic logic, attention scope and choice.", flush=True)
    print("SmolLM3 receives only architecture-owned structured answer requests and renders them.", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Knowledge frozen: {memory.knowledge_is_frozen()}", flush=True)
    print("One-pass interface: architecture -> structured answer -> SmolLM3.", flush=True)
    print("Commands: /new  /status  /topics  /know  /freeze  /unfreeze  /quit", flush=True)
    print("Tip: /topics ranks strong areas; /know <topic> [depth] recursively inspects long-term knowledge without conversation state.", flush=True)

    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        if raw == "/quit":
            break
        if raw == "/new":
            assistant.reset()
            print("[SESSION] reset", flush=True)
            continue
        if raw == "/freeze":
            memory.set_knowledge_frozen(True)
            print("[KNOWLEDGE] frozen=True", flush=True)
            continue
        if raw == "/unfreeze":
            memory.set_knowledge_frozen(False)
            print("[KNOWLEDGE] frozen=False", flush=True)
            continue
        if raw == "/topics":
            print(assistant._format_topics(), flush=True)
            continue
        if raw.startswith("/know"):
            parts = raw.split()
            if len(parts) < 2:
                print("Usage: /know <topic> [depth]", flush=True)
                continue
            try:
                depth = int(parts[2]) if len(parts) > 2 else 2
            except ValueError:
                depth = 2
            print(assistant._inspect_topic(" ".join(parts[1:2]), depth), flush=True)
            continue
        if raw == "/status":
            print({"goal": memory.goal, "topic": memory.topic,
                   "knowledge_frozen": memory.knowledge_is_frozen(),
                   "facts": len(memory.facts()), "session": memory.session_id}, flush=True)
            continue
        try:
            result = assistant.respond(raw)
            print(result["response"], flush=True)
        except Exception as exc:
            print(f"[ERROR] {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
