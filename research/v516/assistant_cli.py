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
    "request_opinion": "I'm not sure yet.",
    "explore_assistant": "I'm thinking about our conversation.",
    "continue_conversation": "Tell me more.",
}

KNOWN_SUBJECTS = (
    "universe", "dog", "dogs", "cat", "cats", "animal", "animals",
    "router", "moon", "python", "book", "books", "car", "cars",
    "planet", "planets", "computer", "phone", "people", "person",
)

PROPERTY_MAP = {
    "red": "color", "blue": "color", "green": "color", "yellow": "color",
    "black": "color", "white": "color", "brown": "color",
    "small": "size", "big": "size", "large": "size", "tiny": "size",
    "huge": "size", "round": "shape", "square": "shape",
}

# Relations that are useful as conversational semantic content.
SEMANTIC_RELATIONS = {
    "has_property", "hasproperty", "property", "defined_as", "definedas",
    "definition", "is_a", "isa", "hypernym", "capable_of", "capableof",
    "used_for", "usedfor", "at_location", "atlocation", "made_of", "madeof",
    "part_of", "partof", "causes", "causes_desire", "causesdesire",
}

# Relations that are useful only as weak lexical hints, not primary answers.
LEXICAL_RELATIONS = {
    "synonym", "antonym", "hyponym", "related_to", "relatedto", "similar_to",
    "similarto",
}

INTERNAL_GRAPH_MARKERS = (
    "-->", "computer_support", "candidate proposition", "the architecture",
    "the participant", "the realizer",
)


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
    """Small typed working-memory store with semantic deduplication."""

    def __init__(self) -> None:
        self.facts: list[Fact] = []

    def add(self, fact: Fact) -> None:
        key = (
            fact.subject.lower(), fact.predicate.lower(), fact.object_text.lower(),
            fact.fact_type.lower(), int(fact.negated),
        )
        for old in self.facts:
            old_key = (
                old.subject.lower(), old.predicate.lower(), old.object_text.lower(),
                old.fact_type.lower(), int(old.negated),
            )
            if old_key == key:
                return
        self.facts.append(fact)

    def for_subject(self, subject: str) -> list[Fact]:
        subject = subject.lower().strip()
        return [f for f in self.facts if f.subject.lower() == subject]

    def as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(f) for f in self.facts]


class CognitiveBridge:
    """
    V515: typed conversational bridge.

    Authority split:
      * working state owns live facts and conversational entities;
      * semantic memory supplies typed long-term evidence;
      * architecture selects content;
      * teacher can propose candidate content, but cannot silently become
        the source of a factual answer when architecture evidence is absent.
    """

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
        self.freeze_knowledge = bool(freeze_knowledge)
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

    @staticmethod
    def _canonical_subject(value: str | None) -> str | None:
        if not value:
            return None
        value = value.lower().strip()
        return {
            "dogs": "dog", "cats": "cat", "animals": "animal",
            "books": "book", "cars": "car", "planets": "planet",
            "persons": "person",
        }.get(value, value)

    def _explicit_subject(self, text: str) -> str | None:
        """Return the LAST explicitly mentioned concrete subject."""
        low = text.lower()
        matches: list[tuple[int, str]] = []
        for noun in KNOWN_SUBJECTS:
            for match in re.finditer(rf"\b{re.escape(noun)}\b", low):
                matches.append((match.start(), noun))
        if not matches:
            return None
        _, noun = max(matches, key=lambda x: x[0])
        return self._canonical_subject(noun)

    def _has_explicit_subject(self, text: str) -> bool:
        return self._explicit_subject(text) is not None

    def _classify(self, lower: str) -> tuple[str, str, str]:
        if re.match(r"^(hi|hello|hey)\b", lower):
            return "social_greeting", "greeting", "general"
        if re.search(r"\b(thanks|thank you)\b", lower):
            return "social_thanks", "thanks", "general"
        if re.search(r"\b(bye|goodbye|see you)\b", lower):
            return "social_goodbye", "goodbye", "general"
        if re.search(r"\b(i love you|love you)\b", lower):
            return "social_affection", "affection", "general"

        if re.search(r"\b(joke|story|poem)\b", lower) and re.search(
            r"\b(tell|give|write|make|create|generate)\b", lower
        ):
            return "request_generation", "request", "generate"

        if re.search(r"\b(help me|what can you do)\b", lower):
            return "request_action", "request", "general"

        if re.search(r"\b(how are you|how's it going|what are you curious about|"
                     r"what are you thinking|what's on your mind|what do you want to do)\b", lower):
            return "explore_assistant", "question", "general"

        if re.search(r"\b(how many|how much)\b", lower):
            return "request_information", "question", "count"

        if re.search(r"\b(what color|what colour|what is the color|what is the colour|"
                     r"what size|what shape|what age|what is its color|what is its colour|"
                     r"how big|how large)\b", lower):
            return "request_information", "question", "property"

        # Put "I want to know ..." before the generic continue path.
        if re.search(r"\b(i want to know|i'd like to know|i would like to know)\b", lower):
            return "request_information", "request", "general"

        if re.search(r"\b(explain|why is|why are|why does|why do|how does|how do|"
                     r"what causes|how do you know)\b", lower):
            return "request_explanation", "question", "general"

        if re.search(r"\b(what is|what's|what are|who is|who are|where is|when is)\b", lower):
            return "request_information", "question", "definition"

        if re.search(r"\b(is it|are they|are the|is the|really|are you sure|also)\b", lower):
            return "challenge_claim", "challenge", "general"

        if lower.endswith("?"):
            return "request_information", "question", "general"

        if re.match(r"^(tell me|show me|describe|find|check|open)\b", lower):
            return "request_information", "request", "general"

        return "continue_conversation", "statement", "general"

    def _infer_target(self, text: str, goal: str, kind: str, subject: str | None) -> dict[str, Any]:
        low = text.lower()
        attribute = None
        value = None

        if re.search(r"\bcolor|colour\b", low):
            attribute = "color"
        elif re.search(r"\b(size|big|large|small|tiny|huge)\b", low):
            attribute = "size"
        elif re.search(r"\bshape|round|square\b", low):
            attribute = "shape"

        # Explicit adjectives refine the property target. For challenge forms
        # such as "is it black?", keep both the semantic attribute and value.
        for candidate_value, mapped in PROPERTY_MAP.items():
            if re.search(rf"\b{re.escape(candidate_value)}\b", low):
                attribute = mapped
                value = candidate_value
                break

        plural = bool(re.search(r"\b(dogs|cats|animals|people|books|cars|planets)\b", low))
        explicit = self._has_explicit_subject(text)

        return {
            "kind": kind,
            "subject": subject,
            "attribute": attribute,
            "value": value,
            "plural": plural,
            "qualifier": None,
            "explicit": explicit,
        }

    def parse(self, text: str) -> tuple[Frame, list[Fact], str | None]:
        lower = text.strip().lower()
        new_facts: list[Fact] = []
        explicit_subject = self._explicit_subject(lower)

        # Property assertions. Keep the last explicitly mentioned subject.
        statement = re.search(
            r"\b(?:the|a|an)\s+([a-z][a-z0-9_-]*)\s+"
            r"(?:is|are|was|were)\s+(?:a\s+|an\s+|the\s+)?"
            r"([a-z][a-z -]*)\s*$",
            lower,
        )
        if statement and not lower.endswith("?"):
            subject = self._canonical_subject(statement.group(1))
            value = statement.group(2).strip(" .?!")
            if subject and value:
                new_facts.append(Fact(subject, "has_property", value, turn_index=self.turn))
                explicit_subject = subject

        count_match = re.search(
            r"\bthere\s+(?:is|are)\s+(\d+)\s+"
            r"(dogs?|cats?|animals?|people|persons|books?|cars?|planets?)\b",
            lower,
        )
        if count_match:
            n = int(count_match.group(1))
            noun = self._canonical_subject(count_match.group(2)) or count_match.group(2)
            new_facts.append(Fact(noun, "count", str(n), turn_index=self.turn))
            explicit_subject = noun

        goal, act, kind = self._classify(lower)
        subject = explicit_subject

        # Only a genuine pronoun can inherit the previous explicit subject.
        # Explicit nouns always win, even if they occur later in the sentence.
        if subject is None and re.search(r"\b(it|they|them|that|this)\b", lower):
            subject = self.last_explicit_subject

        target = self._infer_target(text, goal, kind, subject)
        action = "generate" if kind == "generate" else (
            "answer" if goal in {"request_information", "request_explanation", "request_opinion", "challenge_claim"}
            else "respond"
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

    @staticmethod
    def _normalize_predicate(predicate: str) -> str:
        return str(predicate or "").lower().strip().replace(" ", "_")

    @staticmethod
    def _fact_value(row: dict[str, Any]) -> str:
        return str(row.get("object_text") or "").strip()

    def _memory_rows(self, subject: str, limit: int = 32) -> list[dict[str, Any]]:
        """Read only typed long-term facts. Never read live session tables here."""
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
        value: str | None = None,
    ) -> tuple[str, list[dict[str, Any]]] | None:
        rows = self._memory_rows(subject)
        if not rows:
            return None

        def pred(row: dict[str, Any]) -> str:
            return self._normalize_predicate(row.get("predicate", ""))

        def score(row: dict[str, Any]) -> tuple[float, float, str]:
            p = pred(row)
            conf = float(row.get("confidence") or 0.0)
            freq = float(row.get("frequency") or 0.0)
            base = 0.0
            if p in {"defined_as", "definedas", "definition"}:
                base += 100
            elif p in {"has_property", "hasproperty", "property"}:
                base += 80
            elif p in {"is_a", "isa"}:
                base += 75
            elif p == "hypernym":
                base += 60
            elif p in {"capable_of", "capableof", "used_for", "usedfor", "at_location", "atlocation", "made_of", "madeof", "part_of", "partof"}:
                base += 55
            elif p in LEXICAL_RELATIONS:
                base += 20
            return (base + conf * 5.0 + min(freq, 10.0) * 0.25, conf, str(row.get("object_text") or ""))

        selected: list[dict[str, Any]] = []
        if kind == "property" and attribute:
            for row in rows:
                p = pred(row)
                value = self._fact_value(row).lower()
                mapped = PROPERTY_MAP.get(value)
                if value and value != str(row.get("object_text") or "").strip().lower():
                    continue
                if p in {"has_property", "hasproperty", "property"} and mapped == attribute:
                    selected.append(row)
                elif attribute == "color" and "color" in p:
                    selected.append(row)
                elif attribute == "size" and "size" in p:
                    selected.append(row)
                elif attribute == "shape" and "shape" in p:
                    selected.append(row)

            selected.sort(key=score, reverse=True)

        elif kind == "definition":
            preferred = {
                "defined_as", "definedas", "definition", "is_a", "isa",
                "hypernym", "has_property", "hasproperty", "property",
            }
            selected = [r for r in rows if pred(r) in preferred]
            selected.sort(key=score, reverse=True)

            # Never answer a definition using generic lexical noise like
            # "content"/"collection" when a better semantic fact exists.
            generic = {
                "content", "collection", "natural object", "thing", "entity",
                "object", "concept", "item", "whole", "class",
            }
            selected = [r for r in selected if self._fact_value(r).lower() not in generic] or selected

        else:
            # General descriptions use semantic relations first. Lexical
            # relations are only a last-resort hint and are never allowed to
            # outrank an actual semantic property/type fact.
            selected = [r for r in rows if pred(r) in SEMANTIC_RELATIONS]
            selected.sort(key=score, reverse=True)

            generic = {
                "content", "collection", "natural object", "thing", "entity",
                "object", "concept", "item", "whole", "class",
            }
            useful = [
                r for r in selected
                if self._fact_value(r).lower() not in generic
            ]
            if useful:
                selected = useful

            if not selected:
                selected = [r for r in rows if pred(r) in LEXICAL_RELATIONS]
                selected.sort(key=score, reverse=True)

        selected = selected[:3]
        if not selected:
            return None

        pieces: list[str] = []
        for row in selected:
            value = self._fact_value(row)
            if not value:
                continue
            p = pred(row)
            if p in {"has_property", "hasproperty", "property"}:
                pieces.append(f"The {subject} is {value}.")
            elif p in {"defined_as", "definedas", "definition"}:
                pieces.append(f"{subject.capitalize()} is defined as {value}.")
            elif p in {"is_a", "isa", "hypernym"}:
                article = "an" if value[:1].lower() in "aeiou" else "a"
                pieces.append(f"The {subject} is {article} {value}.")
            elif p in {"capable_of", "capableof"}:
                pieces.append(f"The {subject} can {value}.")
            elif p in {"used_for", "usedfor"}:
                pieces.append(f"The {subject} is used for {value}.")
            elif p in {"at_location", "atlocation"}:
                pieces.append(f"The {subject} is at {value}.")
            elif p in {"made_of", "madeof"}:
                pieces.append(f"The {subject} is made of {value}.")
            elif p in {"part_of", "partof"}:
                pieces.append(f"The {subject} is part of {value}.")
            elif p == "synonym":
                pieces.append(f"{subject.capitalize()} is also called {value}.")
            elif p == "antonym":
                pieces.append(f"The opposite of {subject} is {value}.")
            else:
                continue

        if not pieces:
            return None
        return " ".join(dict.fromkeys(pieces)), selected

    def _teacher_candidate(self, frame: Frame, user_text: str, memory_evidence: list[dict[str, Any]]) -> str | None:
        if self.teacher is None:
            return None

        goal = frame.goal
        # Teacher may create requested open-ended content or social replies.
        # Factual answers require architecture-owned evidence and are not
        # allowed to become free-form teacher facts.
        can_generate = goal == "request_generation"
        can_socialize = goal in {
            "social_greeting", "social_thanks", "social_goodbye", "social_affection",
            "explore_assistant", "continue_conversation",
        }
        if not (can_generate or can_socialize):
            return None

        system = (
            "You are an optional participant inside a cognitive architecture. "
            "Return exactly one candidate response. For generation requests you may "
            "create requested content. For conversation/social turns keep it concise. "
            "Do not mention the architecture, candidate, prompt, or reasoning."
        )
        context = {
            "user": user_text,
            "goal": frame.goal,
            "target": frame.target,
            "state": frame.state,
            "evidence": memory_evidence[:8],
        }
        try:
            raw = self.teacher.generate(system, json.dumps(context, ensure_ascii=False))
        except Exception as exc:
            if self.trace:
                print(f"[TEACHER] error={exc}", flush=True)
            return None

        text = str(raw or "").strip()
        text = re.sub(r"\s+", " ", text)
        if not text or self._looks_internal(text):
            return None
        return text

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
                    return f"The {subject} is {fact.object_text}.", "state"

        if kind == "count":
            for fact in reversed(facts):
                if fact.predicate != "count":
                    continue
                n = fact.object_text
                if subject == "people":
                    noun = "people"
                else:
                    noun = subject if n == "1" else subject + "s"
                return f"There {'is' if n == '1' else 'are'} {n} {noun}.", "state"

        if kind in {"general", "definition"} and facts:
            pieces: list[str] = []
            for fact in facts:
                if fact.predicate == "has_property":
                    pieces.append(f"The {subject} is {fact.object_text}.")
                elif fact.predicate == "count":
                    n = fact.object_text
                    noun = subject if n == "1" else subject + "s"
                    pieces.append(f"There {'is' if n == '1' else 'are'} {n} {noun}.")
            if pieces:
                return " ".join(dict.fromkeys(pieces)), "state"
        return None

    def _select_content(self, frame: Frame, user_text: str) -> tuple[str, str]:
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

        # Challenge claims are proposition checks, not invitations to return
        # any fact about the subject. An exact property/value match is required.
        if goal == "challenge_claim" and subject and target.get("value"):
            expected_value = str(target["value"]).lower()
            for fact in reversed(self.state.for_subject(subject)):
                if fact.predicate != "has_property":
                    continue
                if fact.object_text.lower() != expected_value:
                    continue
                if fact.negated:
                    return f"No, the {subject} is not {expected_value}.", "state"
                return f"Yes, the {subject} is {expected_value}.", "state"

            memory_candidate = self._memory_candidate(
                subject,
                "property",
                target.get("attribute"),
                target.get("value"),
            )
            if memory_candidate:
                content, rows = memory_candidate
                frame.evidence.append({
                    "type": "semantic_memory",
                    "subject": subject,
                    "content": content,
                    "support": rows,
                })
                return f"Yes, {content[0].lower() + content[1:] if content else content}", "knowledge"
            return FALLBACKS.get(goal, "I'm not certain yet."), "fallback"

        # Direct state beats every other source for ordinary information queries.
        state = self._select_state(frame)
        if state:
            return state

        memory_candidate = None
        if subject:
            memory_candidate = self._memory_candidate(
                subject,
                kind,
                target.get("attribute"),
                target.get("value"),
            )
            if memory_candidate:
                content, rows = memory_candidate
                frame.evidence.append({
                    "type": "semantic_memory",
                    "subject": subject,
                    "content": content,
                    "support": rows,
                })
                return content, "knowledge"

        # Teacher is consulted only where it is semantically permitted.
        teacher_candidate = self._teacher_candidate(frame, user_text, frame.evidence)
        if teacher_candidate:
            frame.evidence.append({
                "type": "teacher_candidate",
                "content": teacher_candidate,
            })
            return teacher_candidate, "participant"

        return FALLBACKS.get(goal, "I'm not sure yet."), "fallback"

    @staticmethod
    def _looks_internal(text: str) -> bool:
        low = str(text or "").lower()
        return any(marker in low for marker in INTERNAL_GRAPH_MARKERS)

    @staticmethod
    def _public_safe(text: str) -> bool:
        text = str(text or "").strip()
        if not text:
            return False
        if CognitiveBridge._looks_internal(text):
            return False
        if re.search(r"\[[A-Z][A-Z _→←-]*\]", text):
            return False
        return True

    def turn_once(self, text: str) -> str:
        text = text.strip()
        frame, facts, explicit_subject = self.parse(text)
        self.ingest(facts)

        # Update conversational subject ONLY from explicit noun mentions.
        if explicit_subject is not None:
            self.last_explicit_subject = explicit_subject

        frame.state = self.state.as_dicts()
        response, source = self._select_content(frame, text)

        self.last_target = dict(frame.target)

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

        if not self._public_safe(response):
            response = FALLBACKS.get(frame.goal, "I'm not sure yet.")

        print(response)
        self.turn += 1
        return response


def build_teacher(path: str | None) -> Any | None:
    if not path:
        return None
    from llm_backend import LocalLLM
    return LocalLLM(path, max_new_tokens=96)


def main() -> None:
    ap = argparse.ArgumentParser(description="V515 cognitive bridge assistant")
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

    print("V516 cognitive bridge assistant ready.")
    print("Architecture owns logic/state/content selection.")
    print("LLM is an optional teacher/participant/realizer only.")
    print(f"Knowledge frozen: {args.freeze_knowledge}")
    print("Commands: /new  /status  /freeze  /unfreeze  /quit")

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
            bridge.reset()
            print("[SESSION] reset")
            continue
        if text == "/freeze":
            bridge.freeze_knowledge = True
            print("[KNOWLEDGE] frozen")
            continue
        if text == "/unfreeze":
            bridge.freeze_knowledge = False
            print("[KNOWLEDGE] unfrozen")
            continue
        if text == "/status":
            print(json.dumps({
                "turn": bridge.turn,
                "last_explicit_subject": bridge.last_explicit_subject,
                "last_target": bridge.last_target,
                "working_facts": bridge.state.as_dicts(),
                "knowledge_frozen": bridge.freeze_knowledge,
                "memory_connected": bridge.memory is not None,
                "teacher_enabled": bridge.teacher is not None,
            }, ensure_ascii=False, indent=2))
            continue

        bridge.turn_once(text)


if __name__ == "__main__":
    main()
