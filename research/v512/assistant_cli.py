from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


FALLBACKS = {
    "social_greeting": "Hello!",
    "social_thanks": "You're welcome!",
    "social_goodbye": "Bye!",
    "social_affection": "That's nice to hear.",
    "request_generation": "Sure.",
    "request_action": "Sure. What would you like me to do?",
    "challenge_claim": "I'm not certain yet.",
    "request_information": "I'm not sure yet.",
    "request_explanation": "I'm not sure yet.",
    "explore_assistant": "I'm thinking about what we should explore next.",
    "continue_conversation": "Tell me more.",
}


@dataclass(frozen=True)
class Fact:
    subject: str
    predicate: str
    object_text: str
    fact_type: str = "state"
    negated: int = 0
    confidence: float = 1.0
    turn_index: int = 0


@dataclass
class Frame:
    goal: str
    act: str
    target: dict[str, Any]
    state: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    action: str
    constraints: list[str]


class StateStore:
    def __init__(self) -> None:
        self.facts: list[Fact] = []

    def add(self, fact: Fact) -> None:
        key = (fact.subject, fact.predicate, fact.object_text, fact.fact_type, fact.negated)
        for old in self.facts:
            if (old.subject, old.predicate, old.object_text, old.fact_type, old.negated) == key:
                return
        self.facts.append(fact)

    def for_subject(self, subject: str) -> list[Fact]:
        return [f for f in self.facts if f.subject == subject]

    def entities(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.facts:
            out[f.subject] = out.get(f.subject, 0) + 1
        return out

    def as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(f) for f in self.facts]


class CognitiveBridge:
    """V512 bridge: architecture selects semantic content; participant only realizes it."""

    def __init__(self, memory_path: str | None = None) -> None:
        self.state = StateStore()
        self.turn = 0
        self.last_explicit_subject: str | None = None
        self.memory_path = memory_path
        self.memory = None
        if memory_path:
            p = Path(memory_path)
            if p.exists():
                try:
                    self.memory = sqlite3.connect(str(p))
                except sqlite3.Error:
                    self.memory = None

    def _explicit_subject(self, lower: str) -> str | None:
        for noun in ("universe", "dog", "dogs", "cat", "cats", "animal", "animals", "router", "moon"):
            if re.search(rf"\b{re.escape(noun)}\b", lower):
                return noun.rstrip("s")
        return None

    def parse(self, text: str) -> tuple[Frame, list[Fact], str | None]:
        raw = text.strip()
        lower = raw.lower()
        new_facts: list[Fact] = []
        explicit_subject = self._explicit_subject(lower)

        # State assertions. Explicit subjects always win over conversational context.
        m = re.search(r"\bthe\s+(dog|cat|universe|moon|router)\s+is\s+(?:a\s+)?([a-z][a-z -]*)\s*$", lower)
        if m:
            subject, value = m.group(1), m.group(2).strip(" .?!")
            new_facts.append(Fact(subject, "has_property", value, turn_index=self.turn))
            explicit_subject = subject

        m_count = re.search(r"\bthere\s+are\s+(\d+)\s+(dogs?|cats?|animals?)\b", lower)
        if m_count:
            n, noun = int(m_count.group(1)), m_count.group(2).rstrip("s")
            new_facts.append(Fact(noun, "count", str(n), "state", turn_index=self.turn))
            explicit_subject = noun

        # Explicit subject extraction for questions/requests.
        subject = explicit_subject
        for noun in ("universe", "dog", "dogs", "cat", "cats", "animal", "animals", "router", "moon"):
            if re.search(rf"\b{re.escape(noun)}\b", lower):
                candidate = noun.rstrip("s")
                subject = candidate
                explicit_subject = candidate
                break

        # Do not inherit a previous subject for unrelated generic requests.
        if subject is None and re.search(r"\b(help|what can you do|how are you)\b", lower):
            subject = None

        if re.match(r"^(hi|hello|hey)\b", lower):
            goal, act, kind = "social_greeting", "greeting", "general"
        elif re.search(r"\b(thanks|thank you)\b", lower):
            goal, act, kind = "social_thanks", "thanks", "general"
        elif re.search(r"\b(bye|goodbye|see you)\b", lower):
            goal, act, kind = "social_goodbye", "goodbye", "general"
        elif re.search(r"\b(i love you|love you)\b", lower):
            goal, act, kind = "social_affection", "affection", "general"
        elif re.search(r"\b(joke|story|poem)\b", lower) and re.search(r"\b(tell|give|write|make)\b", lower):
            goal, act, kind = "request_generation", "request", "generate"
        elif re.search(r"\b(help me|what can you do)\b", lower):
            goal, act, kind = "request_action", "request", "general"
        elif re.search(r"\b(what color|what colour|what is the color|what is the colour)\b", lower):
            goal, act, kind = "request_information", "question", "property"
        elif re.search(r"\bhow many\b", lower):
            goal, act, kind = "request_information", "question", "count"
        elif re.search(r"\b(what is|what's|what are|who is|who are)\b", lower):
            goal, act, kind = "request_information", "question", "definition"
        elif re.search(r"\b(describe|tell me about|tell me more about)\b", lower):
            goal, act, kind = "request_information", "request", "general"
        elif re.search(r"\b(how can you describe|explain why|why is|why are)\b", lower):
            goal, act, kind = "request_explanation", "question", "property"
        elif re.search(r"\b(is it|is the|are the|really|are you sure)\b", lower):
            goal, act, kind = "challenge_claim", "challenge", "general"
        else:
            goal, act, kind = "continue_conversation", "statement", "general"

        attribute = None
        if kind == "property":
            if re.search(r"\bcolor|colour\b", lower):
                attribute = "color"
            elif re.search(r"\bshape\b", lower):
                attribute = "shape"

        # Critical V512 reference rule: explicit noun wins. Pronouns may resolve to context.
        if explicit_subject is None and re.search(r"\b(it|they|them|that|this)\b", lower):
            subject = self.last_explicit_subject

        target = {
            "kind": kind,
            "subject": subject,
            "attribute": attribute,
            "value": None,
            "plural": bool(re.search(r"\b(dogs|cats|animals)\b", lower)),
            "qualifier": None,
            "explicit": explicit_subject is not None,
        }

        action = "generate" if kind == "generate" else ("answer" if goal == "request_information" else "respond")
        frame = Frame(
            goal=goal,
            act=act,
            target=target,
            state=self.state.as_dicts(),
            evidence=[],
            action=action,
            constraints=[
                "architecture owns semantic content",
                "do not invent unsupported factual claims",
                "realize only selected content",
            ],
        )
        return frame, new_facts, explicit_subject

    def ingest(self, facts: list[Fact]) -> None:
        for fact in facts:
            self.state.add(fact)

    def select(self, frame: Frame) -> tuple[str, str | None]:
        goal = frame.goal
        target = frame.target
        subject = target.get("subject")
        kind = target.get("kind")
        attribute = target.get("attribute")

        if goal in FALLBACKS and not subject:
            return FALLBACKS[goal], "fallback"

        if goal == "social_greeting":
            return "Hello!", "architecture"
        if goal == "social_thanks":
            return "You're welcome!", "architecture"
        if goal == "social_goodbye":
            return "Bye!", "architecture"
        if goal == "social_affection":
            return "That's nice to hear.", "architecture"
        if goal == "request_action":
            return "Sure. What would you like me to do?", "architecture"
        if goal == "request_generation" and subject == "joke":
            # The architecture owns the action; generation remains an explicit participant slot.
            return "Generate a joke for the user.", "participant"
        if goal == "explore_assistant":
            return "I'm thinking about what we should explore next.", "architecture"

        if subject:
            facts = self.state.for_subject(subject)
            if kind == "property" and attribute == "color":
                for f in reversed(facts):
                    if f.predicate == "has_property":
                        return f"The {subject} is {f.object_text}.", "logic"
            if kind == "count":
                for f in reversed(facts):
                    if f.predicate == "count":
                        n = f.object_text
                        noun = subject if n == "1" else subject + "s"
                        return f"There {'is' if n == '1' else 'are'} {n} {noun}.", "logic"
            if kind in ("general", "definition") and facts:
                pieces = []
                for f in facts:
                    if f.predicate == "has_property":
                        pieces.append(f"The {subject} is {f.object_text}.")
                    elif f.predicate == "count":
                        pieces.append(f"There are {f.object_text} {subject}s.")
                if pieces:
                    return " ".join(pieces), "state"

            # Knowledge DB is consulted only for candidate semantic content; it is never emitted blindly.
            candidate = self._memory_candidate(subject, kind)
            if candidate:
                return candidate, "knowledge"

        return FALLBACKS.get(goal, "I'm not sure yet."), "fallback"

    def _memory_candidate(self, subject: str, kind: str) -> str | None:
        if not self.memory:
            return None
        # Conservative schema probing for common Graph-Topology semantic DBs.
        try:
            tables = [r[0] for r in self.memory.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            for table in tables:
                cols = [r[1] for r in self.memory.execute(f"PRAGMA table_info([{table}])").fetchall()]
                lower = {c.lower(): c for c in cols}
                s_col = lower.get("start") or lower.get("subject")
                r_col = lower.get("relation") or lower.get("predicate")
                e_col = lower.get("end") or lower.get("object") or lower.get("object_text")
                if not (s_col and r_col and e_col):
                    continue
                rows = self.memory.execute(
                    f"SELECT [{r_col}], [{e_col}] FROM [{table}] WHERE lower([{s_col}])=? LIMIT 8",
                    (subject.lower(),),
                ).fetchall()
                if not rows:
                    continue
                parts = []
                for rel, obj in rows:
                    rel = str(rel).replace("_", " ").lower()
                    parts.append(f"{subject} {rel} {obj}")
                return ". ".join(parts) + "."
        except sqlite3.Error:
            return None
        return None

    def turn_once(self, text: str) -> str:
        frame, facts, explicit_subject = self.parse(text)
        self.ingest(facts)
        if explicit_subject is not None:
            self.last_explicit_subject = explicit_subject
        frame.state = self.state.as_dicts()
        response, source = self.select(frame)
        print("[FRAME] COGNITIVE_FRAME")
        print(json.dumps(asdict(frame), ensure_ascii=False, separators=(",", ":")))
        print("[LOGIC] None -> None")
        print(f"[DECISION] source={source} target={frame.target['kind']}")
        print(response)
        self.turn += 1
        return response


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", default=None)
    args = ap.parse_args()

    bridge = CognitiveBridge(args.memory)
    print("V512 cognitive bridge assistant ready.")
    print("Architecture owns logic/state/content selection.")
    print("LLM is a participant/realizer only.")
    print("Commands: /new  /status  /freeze  /unfreeze  /quit")

    frozen = False
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/new":
            bridge = CognitiveBridge(args.memory)
            print("[STATUS] new conversation")
            continue
        if text == "/status":
            print(json.dumps({"turn": bridge.turn, "facts": bridge.state.as_dicts(), "last_subject": bridge.last_explicit_subject, "frozen": frozen}, ensure_ascii=False, indent=2))
            continue
        if text == "/freeze":
            frozen = True
            print("[STATUS] frozen")
            continue
        if text == "/unfreeze":
            frozen = False
            print("[STATUS] unfrozen")
            continue
        if frozen:
            print("[STATUS] cognitive state is frozen")
            continue
        bridge.turn_once(text)


if __name__ == "__main__":
    main()
