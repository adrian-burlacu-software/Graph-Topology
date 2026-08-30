
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import spacy

from config import Config
from memory import ConversationMemory, digest
from perception import Perceiver
from knowledge import KnowledgeStore
from participant import LLMParticipant
from assistant import CognitiveAssistant


def load_parser(model_name):
    print(
        f"[PARSER] loading {model_name}...",
        flush=True,
    )
    try:
        nlp=spacy.load(model_name)
    except Exception as exc:
        raise SystemExit(
            f"Could not load spaCy model {model_name}: {exc}\n"
            f"Install with:\n"
            f"python -m spacy download {model_name}"
        )
    print("[PARSER] ready",flush=True)
    return nlp


def main():
    defaults=Config()

    ap=argparse.ArgumentParser(
        description="V481 modular goal-directed cognitive assistant"
    )
    ap.add_argument(
        "--memory",
        default=str(defaults.memory),
    )
    ap.add_argument(
        "--teacher",
        default="",
        help="Optional LLM participant/realizer model path.",
    )
    ap.add_argument(
        "--parser-model",
        default=defaults.parser_model,
    )
    ap.add_argument(
        "--freeze-learning",
        action="store_true",
        default=defaults.freeze_learning,
    )
    ap.add_argument(
        "--learn",
        action="store_true",
        help="Allow conversational learning/policies.",
    )
    ap.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable planner diagnostics.",
    )
    args=ap.parse_args()

    memory_path=Path(args.memory)
    if not memory_path.exists():
        raise SystemExit(
            f"Semantic memory not found: {memory_path}"
        )

    con=sqlite3.connect(str(memory_path))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")

    nlp=load_parser(args.parser_model)

    participant=None
    if args.teacher:
        participant=LLMParticipant(
            args.teacher,
            max_new_tokens=defaults.participant_max_tokens,
        )

    session_id=digest(
        "assistant_session",
        __file__,
        id(con),
    )

    memory=ConversationMemory(
        con,
        session_id=session_id,
        recent_turns=defaults.recent_turns,
        memory_facts=defaults.memory_facts,
        freeze_learning=not args.learn,
    )

    assistant=CognitiveAssistant(
        con=con,
        perceiver=Perceiver(nlp),
        memory=memory,
        knowledge=KnowledgeStore(con),
        participant=participant,
        trace=not args.no_trace,
    )

    print()
    print("V482 strict modular cognitive assistant ready.")
    print(
        "Architecture owns goal, memory, retrieval, evaluation, content selection and choice."
    )
    print(
        "LLM proposes content and realizes selected content; it is not the final authority."
    )
    print("Commands: /status  /new  /quit")
    print()

    while True:
        try:
            text=input("You: ").strip()
        except (EOFError,KeyboardInterrupt):
            print()
            break

        if not text:
            continue

        command=text.lower()

        if command=="/quit":
            break

        if command=="/new":
            session_id=digest(
                "assistant_session",
                __file__,
                id(con),
                assistant.turn_index,
            )
            memory=ConversationMemory(
                con,
                session_id=session_id,
                recent_turns=defaults.recent_turns,
                memory_facts=defaults.memory_facts,
                freeze_learning=not args.learn,
            )
            assistant.memory=memory
            assistant.planner.memory=memory
            assistant.turn_index=0
            print("New conversation.")
            continue

        if command=="/status":
            state=assistant.memory.state()
            print({
                "session":assistant.memory.session_id,
                "topic":state.topic,
                "goal":state.goal,
                "last_user":state.last_user_text,
                "last_assistant":state.last_assistant_text,
                "facts":len(assistant.memory.facts()),
                "turns":assistant.turn_index,
            })
            continue

        assistant.respond(text)


if __name__=="__main__":
    main()
