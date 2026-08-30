
from __future__ import annotations
import argparse,sqlite3,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--memory",required=True)
    ap.add_argument("--teacher",default="")
    ap.add_argument("--freeze-knowledge",action="store_true")
    ap.add_argument("--no-trace",action="store_true")
    args=ap.parse_args()

    import spacy
    from memory import TypedMemory
    from perception import Perceiver
    from bridge_assistant import BridgeAssistant

    con=sqlite3.connect(args.memory)
    con.row_factory=sqlite3.Row
    try: nlp=spacy.load("en_core_web_sm")
    except Exception: nlp=None

    memory=TypedMemory(con)
    memory.set_knowledge_frozen(args.freeze_knowledge)

    llm=None
    if args.teacher:
        from llm_backend import LocalLLM
        llm=LocalLLM(args.teacher,max_new_tokens=96)

    assistant=BridgeAssistant(
        memory,Perceiver(nlp),llm=llm,
        trace=not args.no_trace,
        freeze=args.freeze_knowledge,
    )

    print("V509 cognitive bridge assistant ready.")
    print("Architecture owns logic/state/content selection.")
    print("LLM is a participant/realizer only.")
    print("Commands: /new  /status  /freeze  /unfreeze  /quit")

    while True:
        try:
            text=input("You: ").strip()
        except (EOFError,KeyboardInterrupt):
            print()
            break

        if not text:
            continue
        low=text.lower()

        if low in {"/quit","/exit"}:
            break
        if low in {"/new","//new"}:
            assistant.reset()
            print(
                f"New conversation. "
                f"knowledge_frozen={memory.knowledge_is_frozen()}",
                flush=True,
            )
            continue
        if low=="/freeze":
            memory.set_knowledge_frozen(True);assistant.freeze=True
            print("Static knowledge frozen.",flush=True)
            continue
        if low=="/unfreeze":
            memory.set_knowledge_frozen(False);assistant.freeze=False
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
                "entities":assistant._entity_counts(),
            },flush=True)
            continue

        result=assistant.respond(text)
        print(result["response"],flush=True)

if __name__=="__main__":
    main()
