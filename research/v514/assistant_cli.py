from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
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

KNOWN_SUBJECTS = (
    "universe",
    "dog",
    "dogs",
    "cat",
    "cats",
    "animal",
    "animals",
    "router",
    "moon",
    "python",
)

PROPERTY_MAP = {
    "red": "color",
    "blue": "color",
    "green": "color",
    "small": "size",
    "big": "size",
    "large": "size",
    "round": "shape",
    "square": "shape",
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
        key = (
            fact.subject,
            fact.predicate,
            fact.object_text,
            fact.fact_type,
            fact.negated,
        )
        for old in self.facts:
            old_key = (
                old.subject,
                old.predicate,
                old.object_text,
                old.fact_type,
                old.negated,
            )
            if old_key == key:
                return
        self.facts.append(fact)

    def for_subject(self, subject: str) -> list[Fact]:
        return [f for f in self.facts if f.subject == subject]

    def as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(f) for f in self.facts]

    def entities(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for fact in self.facts:
            out[fact.subject] = out.get(fact.subject, 0) + 1
        return out


class CognitiveBridge:
    """V513: semantic bridge with optional teacher and preserved ingestion DB access."""

    def __init__(
        self,
        memory_path: str | None = None,
        teacher: Any | None = None,
        freeze_knowledge: bool = False,
        trace: bool = True,
    ) -> None:
        self.state = StateStore()
        self.turn = 0
        self.last_explicit_subject: str | None = None
        self.last_target: dict[str, Any] | None = None
        self.memory_path = memory_path
        self.teacher = teacher
        self.freeze_knowledge = freeze_knowledge
        self.trace = trace
        self.memory: sqlite3.Connection | None = None
        if memory_path:
            path = Path(memory_path)
            if path.exists():
                try:
                    self.memory = sqlite3.connect(str(path))
                    self.memory.row_factory = sqlite3.Row
                except sqlite3.Error:
                    self.memory = None

    def reset(self) -> None:
        self.state = StateStore()
        self.turn = 0
        self.last_explicit_subject = None
        self.last_target = None

    def _explicit_subject(self, text: str) -> str | None:
        lower = text.lower()
        for noun in KNOWN_SUBJECTS:
            if re.search(rf"\b{re.escape(noun)}\b", lower):
                return noun.rstrip("s")
        return None

    def _classify(self, lower: str) -> tuple[str, str, str]:
        if re.match(r"^(hi|hello|hey)\b", lower):
            return "social_greeting", "greeting", "general"
        if re.search(r"\b(thanks|thank you)\b", lower):
            return "social_thanks", "thanks", "general"
        if re.search(r"\b(bye|goodbye|see you)\b", lower):
            return "social_goodbye", "goodbye", "general"
        if re.search(r"\b(i love you|love you)\b", lower):
            return "social_affection", "affection", "general"
        if re.search(r"\b(joke|story|poem)\b", lower) and re.search(r"\b(tell|give|write|make)\b", lower):
            return "request_generation", "request", "generate"
        if re.search(r"\b(help me|what can you do)\b", lower):
            return "request_action", "request", "general"
        if re.search(r"\b(what color|what colour|what is the color|what is the colour)\b", lower):
            return "request_information", "question", "property"
        if re.search(r"\bhow many\b", lower):
            return "request_information", "question", "count"
        if re.search(r"\b(how can you describe|how do you describe|describe|tell me about|tell me more about)\b", lower):
            return "request_information", "request", "general"
        if re.search(r"\b(explain why|why is|why are)\b", lower):
            return "request_explanation", "question", "property"
        if re.search(r"\b(what is|what's|what are|who is|who are)\b", lower):
            return "request_information", "question", "definition"
        if re.search(r"\b(is it|is the|are the|really|are you sure|also)\b", lower):
            return "challenge_claim", "challenge", "general"
        return "continue_conversation", "statement", "general"

    def parse(self, text: str) -> tuple[Frame, list[Fact], str | None]:
        lower = text.strip().lower()
        new_facts: list[Fact] = []
        explicit_subject = self._explicit_subject(lower)

        # Parse assertions before question routing so context is updated on the same turn.
        statement = re.search(
            r"\b(?:the|a|an)\s+(dog|cat|universe|moon|router)\s+is\s+(?:a\s+)?([a-z][a-z -]*)\s*$",
            lower,
        )
        if statement:
            subject = statement.group(1).rstrip("s")
            value = statement.group(2).strip(" .?!")
            new_facts.append(Fact(subject, "has_property", value, turn_index=self.turn))
            explicit_subject = subject

        count_match = re.search(
            r"\bthere\s+(?:is|are)\s+(\d+)\s+(dogs?|cats?|animals?|people|persons|books?)\b",
            lower,
        )
        if count_match:
            n = int(count_match.group(1))
            noun = count_match.group(2)
            canonical = {
                "dogs": "dog",
                "cats": "cat",
                "animals": "animal",
                "persons": "person",
                "people": "people",
                "books": "book",
            }.get(noun, noun)
            new_facts.append(Fact(canonical, "count", str(n), turn_index=self.turn))
            explicit_subject = canonical

        goal, act, kind = self._classify(lower)
        subject = explicit_subject

        # Explicitly named subjects always override stale conversational context.
        if explicit_subject is None:
            pronoun = re.search(r"\b(it|they|them|that|this)\b", lower)
            if pronoun:
                subject = self.last_explicit_subject

        # Generic utterances do not inherit a previous target accidentally.
        if subject is None and goal in {"request_action", "social_greeting", "social_thanks", "social_goodbye", "social_affection"}:
            subject = None

        attribute = None
        if kind == "property":
            if re.search(r"\bcolor|colour\b", lower):
                attribute = "color"
            elif re.search(r"\bshape\b", lower):
                attribute = "shape"
            elif re.search(r"\bsize\b", lower):
                attribute = "size"

        # Infer a property attribute from an explicit adjective when a question names it.
        if kind in {"property", "general"}:
            for value, mapped in PROPERTY_MAP.items():
                if re.search(rf"\b{re.escape(value)}\b", lower):
                    attribute = mapped
                    break

        target = {
            "kind": kind,
            "subject": subject,
            "attribute": attribute,
            "value": None,
            "plural": bool(re.search(r"\b(dogs|cats|animals|people|books)\b", lower)),
            "qualifier": None,
            "explicit": explicit_subject is not None,
        }
        action = "generate" if kind == "generate" else (
            "answer" if goal in {"request_information", "request_explanation"} else "respond"
        )
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

    def _memory_rows(self, subject: str, limit: int = 12) -> list[dict[str, Any]]:
        """Read typed static facts only; never mix session/live facts into LTM retrieval."""
        if self.memory is None or not subject:
            return []
        try:
            sql = """
                SELECT
                    c.canonical AS subject,
                    f.predicate,
                    COALESCE(o.canonical, f.object_text) AS object_text,
                    f.fact_type, f.domain, f.confidence, f.frequency,
                    f.answerable
                FROM facts f
                JOIN concepts c ON c.concept_id=f.subject_id
                LEFT JOIN concepts o ON o.concept_id=f.object_id
                WHERE lower(c.canonical)=lower(?)
                  AND f.answerable=1
                ORDER BY f.confidence DESC, f.frequency DESC
                LIMIT ?
            """
            return [dict(r) for r in self.memory.execute(sql, (subject, limit)).fetchall()]
        except sqlite3.Error:
            return []

    def _memory_candidate(
        self,
        subject: str,
        kind: str,
        attribute: str | None = None,
    ) -> tuple[str, list[dict[str, Any]]] | None:
        rows = self._memory_rows(subject)
        if not rows:
            return None

        selected = rows
        if kind == "definition":
            preferred = {"definedas", "defined_as", "definition", "isa", "is_a", "hypernym"}
            picked = [r for r in rows if str(r["predicate"]).lower().replace(" ", "_") in preferred]
            if picked:
                selected = picked[:4]
        elif kind == "property" and attribute:
            picked = []
            for r in rows:
                pred = str(r["predicate"]).lower().replace(" ", "_")
                value = str(r.get("object_text") or "").lower()
                mapped = PROPERTY_MAP.get(value)
                if pred in {"hasproperty", "has_property", "property"} and mapped == attribute:
                    picked.append(r)
                elif attribute == "color" and "color" in pred:
                    picked.append(r)
                elif attribute == "size" and "size" in pred:
                    picked.append(r)
                elif attribute == "shape" and "shape" in pred:
                    picked.append(r)
            if picked:
                selected = picked[:4]
            else:
                return None

        pieces = []
        for row in selected:
            value = str(row.get("object_text") or "").strip()
            pred = str(row.get("predicate") or "").lower().replace("_", " ")
            if not value:
                continue
            if pred in {"has property", "property"}:
                pieces.append(f"The {subject} is {value}.")
            elif pred in {"is a", "isa", "is_a", "hypernym", "defined as", "definition"}:
                pieces.append(f"The {subject} is {value}.")
            elif pred in {"capable of", "capable_of"}:
                pieces.append(f"The {subject} can {value}.")
            elif pred in {"used for", "used_for"}:
                pieces.append(f"The {subject} is used for {value}.")
            elif pred in {"at location", "at_location"}:
                pieces.append(f"The {subject} is at {value}.")
            else:
                pieces.append(f"The {subject} {pred} {value}.")

        if not pieces:
            return None
        return " ".join(dict.fromkeys(pieces)), selected

    def _teacher_candidate(self, frame: Frame, user_text: str) -> str | None:
        if self.teacher is None:
            return None
        system = (
            "You are a semantic candidate generator inside a cognitive architecture. "
            "Return only content that could answer the user's request. Do not invent "
            "facts that conflict with the supplied architecture state or evidence. "
            "The architecture will decide whether your candidate is accepted."
        )
        context = {
            "user": user_text,
            "goal": frame.goal,
            "target": frame.target,
            "state": frame.state,
            "evidence": frame.evidence,
        }
        try:
            return self.teacher.generate(system, json.dumps(context, ensure_ascii=False))
        except Exception as exc:
            if self.trace:
                print(f"[TEACHER] error={exc}", flush=True)
            return None

    def _select_state(self, frame: Frame) -> tuple[str, str] | None:
        target = frame.target
        subject = target.get("subject")
        kind = target.get("kind")
        attribute = target.get("attribute")
        if not subject:
            return None

        facts = self.state.for_subject(subject)
        if kind == "property" and attribute:
            for fact in reversed(facts):
                if fact.predicate != "has_property":
                    continue
                mapped = PROPERTY_MAP.get(fact.object_text.lower())
                if mapped == attribute:
                    return f"The {subject} is {fact.object_text}.", "logic"

        if kind == "count":
            for fact in reversed(facts):
                if fact.predicate == "count":
                    n = fact.object_text
                    if subject == "people":
                        noun = "people"
                    else:
                        noun = subject if n == "1" else subject + "s"
                    return f"There {'is' if n == '1' else 'are'} {n} {noun}.", "logic"

        if kind in {"general", "definition"} and facts:
            pieces = []
            for fact in facts:
                if fact.predicate == "has_property":
                    pieces.append(f"The {subject} is {fact.object_text}.")
                elif fact.predicate == "count":
                    n = fact.object_text
                    noun = subject if n == "1" else subject + "s"
                    pieces.append(f"There {'is' if n == '1' else 'are'} {n} {noun}.")
            if pieces:
                return " ".join(pieces), "state"
        return None

    def select(self, frame: Frame, user_text: str) -> tuple[str, str]:
        goal = frame.goal
        target = frame.target
        subject = target.get("subject")
        kind = target.get("kind")

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
            candidate = self._teacher_candidate(frame, user_text)
            if candidate:
                return candidate, "participant"
            return "Generate a joke for the user.", "participant"

        selected = self._select_state(frame)
        if selected:
            return selected

        if subject:
            memory_candidate = self._memory_candidate(subject, kind, target.get("attribute"))
            if memory_candidate:
                content, rows = memory_candidate
                frame.evidence.append({
                    "type": "semantic_memory",
                    "subject": subject,
                    "content": content,
                    "support": rows[:4],
                })
                return content, "knowledge"

        teacher_candidate = self._teacher_candidate(frame, user_text)
        if teacher_candidate:
            # Candidate generation is permitted, but only as a proposal. The architecture
            # retains the target contract and can reject obviously empty output.
            if teacher_candidate.strip():
                candidate = re.sub(r"\s+", " ", teacher_candidate).strip()
                frame.evidence.append({
                    "type": "teacher_candidate",
                    "content": candidate,
                })
                return candidate, "participant"

        return FALLBACKS.get(goal, "I'm not sure yet."), "fallback"

    def turn_once(self, text: str) -> str:
        frame, facts, explicit_subject = self.parse(text)
        self.ingest(facts)
        if explicit_subject is not None:
            self.last_explicit_subject = explicit_subject
        frame.state = self.state.as_dicts()
        response, source = self.select(frame, text)
        self.last_target = frame.target.copy()

        if self.trace:
            print("[FRAME] COGNITIVE_FRAME")
            print(json.dumps(asdict(frame), ensure_ascii=False, separators=(",", ":")))
            print(f"[LOGIC] source={source}")
            print(
                f"[DECISION] source={source} "
                f"target={frame.target['kind']} "
                f"subject={frame.target['subject']}"
            )
            if source == "participant":
                print("[PARTICIPANT] candidate accepted by bridge")
        print(response)
        self.turn += 1
        return response


def build_teacher(path: str | None) -> Any | None:
    if not path:
        return None
    from llm_backend import LocalLLM
    return LocalLLM(path, max_new_tokens=96)


def main() -> None:
    ap = argparse.ArgumentParser(description="V514 cognitive bridge assistant")
    ap.add_argument("--memory", default=None)
    ap.add_argument("--teacher", default="", help="local teacher/participant model path")
    ap.add_argument("--freeze-knowledge", action="store_true")
    ap.add_argument("--no-trace", action="store_true")
    args = ap.parse_args()

    teacher = build_teacher(args.teacher or None)
    bridge = CognitiveBridge(
        memory_path=args.memory,
        teacher=teacher,
        freeze_knowledge=args.freeze_knowledge,
        trace=not args.no_trace,
    )

    print("V514 cognitive bridge assistant ready.")
    print("Architecture owns logic/state/content selection.")
    print("LLM is an optional teacher/participant/realizer only.")
    print(f"Knowledge frozen: {args.freeze_knowledge}")
    print("Commands: /new  /status  /freeze  /unfreeze  /quit")

    frozen = args.freeze_knowledge
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        low = text.lower()
        if low in {"/quit", "/exit"}:
            break
        if low in {"/new", "//new"}:
            bridge.reset()
            print(f"[STATUS] new conversation frozen={frozen}")
            continue
        if low == "/freeze":
            frozen = True
            bridge.freeze_knowledge = True
            print("[STATUS] knowledge frozen")
            continue
        if low == "/unfreeze":
            frozen = False
            bridge.freeze_knowledge = False
            print("[STATUS] knowledge unfrozen")
            continue
        if low == "/status":
            print(json.dumps({
                "turn": bridge.turn,
                "facts": bridge.state.as_dicts(),
                "last_subject": bridge.last_explicit_subject,
                "last_target": bridge.last_target,
                "knowledge_frozen": frozen,
                "entities": bridge.state.entities(),
                "teacher": type(teacher).__name__ if teacher is not None else None,
            }, ensure_ascii=False, indent=2))
            continue
        if frozen and low.startswith("/set"):
            print("[STATUS] knowledge is frozen")
            continue
        bridge.turn_once(text)


if __name__ == "__main__":
    main()
