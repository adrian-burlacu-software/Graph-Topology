"""The reasoner: walk the taxonomy, record every step, answer with provenance.

Two query shapes, both producing the same replayable trace:

    ask("can a dog fall down")   -> a claim to verify along the ancestor chain
    ask("what can a dog do")     -> an open question, answered by collecting

Every step the walk takes is recorded as a `rules.Step`, so the UI can replay
the derivation rather than presenting a conclusion and asking to be trusted.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from . import rules
from .rules import Step


@dataclass
class Fact:
    concept: str
    relation: str
    object: str
    source: str
    confidence: float
    sense_assumed: bool
    distance: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "concept": self.concept, "relation": self.relation,
            "object": self.object, "source": self.source,
            "confidence": round(self.confidence, 4),
            "sense_assumed": bool(self.sense_assumed), "distance": self.distance,
        }


@dataclass
class Answer:
    question: str
    verdict: str                       # VERIFIED | CONTRADICTED | UNKNOWN | LISTING
    concept: str | None
    concept_gloss: str | None
    senses: list[dict[str, Any]] = field(default_factory=list)
    chain: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    evidence: list[Fact] = field(default_factory=list)
    parse: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question, "verdict": self.verdict,
            "concept": self.concept, "concept_gloss": self.concept_gloss,
            "senses": self.senses, "chain": self.chain,
            "steps": [s.as_dict() for s in self.steps],
            "evidence": [e.as_dict() for e in self.evidence],
            "parse": self.parse, "note": self.note,
            "rules": rules.RULE_TEXT,
        }


class Reasoner:
    """Read-only inference over the store built by `build.py`."""

    MAX_DEPTH = 16
    MAX_ANCESTORS = 400

    def __init__(self, store: Path):
        self.store = store
        self.connection = sqlite3.connect(f"file:{store}?mode=ro", uri=True,
                                          check_same_thread=False)
        self.connection.row_factory = sqlite3.Row

    # -- lookup -----------------------------------------------------------
    def senses_of(self, lemma: str) -> list[dict[str, Any]]:
        """R6: a word is not a concept. Return every sense it could mean."""
        rows = self.connection.execute(
            "SELECT c.id, c.lemma, c.pos, c.sense, c.definition "
            "FROM lemmas l JOIN concepts c ON c.id = l.concept "
            "WHERE l.lemma = ? ORDER BY (c.lemma <> ?), "
            "CASE c.pos WHEN 'n' THEN 0 WHEN 'v' THEN 1 WHEN 'a' THEN 2 ELSE 3 END, "
            "c.sense", (lemma.lower().strip(), lemma.lower().strip())
        ).fetchall()
        return [
            {"id": r["id"], "lemma": r["lemma"], "pos": r["pos"],
             "sense": r["sense"], "definition": r["definition"],
             "fact_count": self.fact_count(r["id"])}
            for r in rows
        ]

    def fact_count(self, concept: str) -> int:
        return self.connection.execute(
            "SELECT COUNT(*) FROM facts WHERE concept = ?", (concept,)
        ).fetchone()[0]

    def gloss(self, concept: str) -> str | None:
        row = self.connection.execute(
            "SELECT definition FROM concepts WHERE id = ?", (concept,)
        ).fetchone()
        return row["definition"] if row else None

    def parents_of(self, concept: str) -> list[str]:
        return [r["parent"] for r in self.connection.execute(
            "SELECT parent FROM taxonomy WHERE child = ? ORDER BY parent", (concept,)
        )]

    def facts_of(self, concept: str, relation: str | None = None) -> list[Fact]:
        """R9: asking about one relation sees its whole family."""
        if relation:
            group = rules.family(relation)
            placeholders = ",".join("?" * len(group))
            rows = self.connection.execute(
                f"SELECT * FROM facts WHERE concept = ? AND relation IN "
                f"({placeholders}) ORDER BY confidence DESC", (concept, *group))
        else:
            rows = self.connection.execute(
                "SELECT * FROM facts WHERE concept = ? ORDER BY confidence DESC",
                (concept,))
        return [Fact(r["concept"], r["relation"], r["object"], r["source"],
                     r["confidence"], bool(r["sense_assumed"])) for r in rows]

    # -- R1: the ascent ---------------------------------------------------
    def ascend(self, concept: str) -> Iterator[tuple[str, int, list[str]]]:
        """Breadth-first up the taxonomy, nearest ancestors first (R1, R4).

        Yields (concept, distance, parents). Bounded by MAX_DEPTH and
        MAX_ANCESTORS so a pathological branch cannot hang the UI.
        """
        seen = {concept}
        frontier = [concept]
        yield concept, 0, self.parents_of(concept)
        for distance in range(1, self.MAX_DEPTH + 1):
            nxt: list[str] = []
            for node in frontier:
                for parent in self.parents_of(node):
                    if parent not in seen:
                        seen.add(parent)
                        nxt.append(parent)
            if not nxt or len(seen) > self.MAX_ANCESTORS:
                return
            for node in nxt:
                yield node, distance, self.parents_of(node)
            frontier = nxt

    # -- R8: the three query shapes --------------------------------------
    def classify(self, concept: str, target_lemma: str) -> Answer:
        """Is `concept` a kind of `target_lemma`? Answered by the taxonomy.

        This is the one question the taxonomy answers by itself, with no facts
        involved: R1 alone. It either finds the target among the ancestors or
        it does not, and the chain it walked is the proof.
        """
        answer = Answer(question="", verdict="UNKNOWN", concept=concept,
                        concept_gloss=self.gloss(concept))
        steps = answer.steps
        steps.append(Step(len(steps), "resolve", concept, 0, "R6",
                          f"Reading “{concept}” as this sense, not as a word."))
        wanted = {row["concept"] for row in self.connection.execute(
            "SELECT concept FROM lemmas WHERE lemma = ?", (target_lemma.lower(),))}
        if not wanted:
            answer.note = f"“{target_lemma}” is not a concept in this ontology."
            return answer
        steps.append(Step(len(steps), "resolve", target_lemma, 0, "R6",
                          f"“{target_lemma}” could mean any of "
                          f"{len(wanted)} sense(s); any of them counts."))

        for node, distance, parents in self.ascend(concept):
            answer.chain.append(node)
            if distance:
                steps.append(Step(len(steps), "ascend", node, distance, "R1",
                                  f"Generalise to {node.rsplit('.', 2)[0]}.",
                                  parents=parents))
            if node in wanted and distance > 0:
                answer.verdict = "VERIFIED"
                fact = Fact(concept, "is_a", node, "wordnet", 0.95, False, distance)
                fact.confidence = rules.confidence_at(0.95, 0)
                answer.evidence.append(fact)
                steps.append(Step(len(steps), "match", node, distance, "R1",
                                  f"{node.rsplit('.', 2)[0]} is an ancestor of "
                                  f"{concept.rsplit('.', 2)[0]}, {distance} step(s) "
                                  f"up. Subsumption is transitive, so yes.",
                                  matched=fact.as_dict()))
                return answer
        answer.note = (f"“{target_lemma}” is not among the "
                       f"{len(answer.chain)} ancestors of {concept}. "
                       f"Absent, not false.")
        return answer

    # -- R8: the two fact-based query shapes -----------------------------
    def verify(self, concept: str, relation: str, target: str,
               matcher) -> Answer:
        """Is `concept relation target` true? Walk up until something says so."""
        answer = Answer(question="", verdict="UNKNOWN", concept=concept,
                        concept_gloss=self.gloss(concept))
        steps = answer.steps
        blocked_by: Fact | None = None

        steps.append(Step(len(steps), "resolve", concept, 0, "R6",
                          f"Reading “{concept}” as this sense, not as a word."))
        negation = rules.POSITIVES.get(relation)

        for node, distance, parents in self.ascend(concept):
            answer.chain.append(node)
            if distance:
                steps.append(Step(len(steps), "ascend", node, distance, "R1",
                                  f"Generalise: everything true of "
                                  f"{node.rsplit('.', 2)[0]} is true of "
                                  f"{concept.rsplit('.', 2)[0]}.",
                                  parents=parents))

            if distance and not rules.inheritable(relation):
                steps.append(Step(len(steps), "stop", node, distance, "R2",
                                  f"`{relation}` does not descend — "
                                  f"{rules.why_not_inheritable(relation)}."))
                answer.note = (f"`{relation}` is not an inheritable relation, so "
                               f"only facts stated directly about "
                               f"{concept} count.")
                break

            # R3: an explicit negation at this level blocks the positive.
            if negation:
                for fact in self.facts_of(node, negation):
                    if matcher(fact.object, target):
                        fact.distance = distance
                        fact.confidence = rules.confidence_at(fact.confidence, distance)
                        blocked_by = fact
                        steps.append(Step(len(steps), "block", node, distance, "R3",
                                          f"{node.rsplit('.', 2)[0]} explicitly "
                                          f"{negation.replace('_', ' ')} "
                                          f"“{fact.object}”.",
                                          matched=fact.as_dict()))
                        break
            if blocked_by:
                answer.verdict = "CONTRADICTED"
                answer.evidence.append(blocked_by)
                break

            candidates = self.facts_of(node, relation)
            steps.append(Step(len(steps), "check", node, distance, "R2",
                              f"Check {len(candidates)} `{relation}` fact(s) on "
                              f"{node.rsplit('.', 2)[0]}.",
                              facts_checked=len(candidates)))
            for fact in candidates:
                if matcher(fact.object, target):
                    fact.distance = distance
                    fact.confidence = rules.confidence_at(fact.confidence, distance)
                    if fact.confidence < rules.FLOOR:
                        continue
                    answer.verdict = "VERIFIED"
                    answer.evidence.append(fact)
                    steps.append(Step(len(steps), "match", node, distance, "R4",
                                      f"Found it: {node.rsplit('.', 2)[0]} "
                                      f"{relation.replace('_', ' ')} "
                                      f"“{fact.object}”.",
                                      matched=fact.as_dict()))
                    return answer
        if answer.verdict == "UNKNOWN" and not answer.note:
            answer.note = (f"Walked {len(answer.chain)} concepts up from "
                           f"{concept} without finding it. Absent, not false.")
        return answer

    def describe(self, concept: str, relation: str | None, limit: int = 40) -> Answer:
        """What is true of this concept, directly or by inheritance?"""
        answer = Answer(question="", verdict="LISTING", concept=concept,
                        concept_gloss=self.gloss(concept))
        steps = answer.steps
        steps.append(Step(len(steps), "resolve", concept, 0, "R6",
                          f"Reading “{concept}” as this sense, not as a word."))
        seen: set[tuple[str, str]] = set()
        collected: list[Fact] = []
        for node, distance, parents in self.ascend(concept):
            answer.chain.append(node)
            if distance:
                steps.append(Step(len(steps), "ascend", node, distance, "R1",
                                  f"Generalise to {node.rsplit('.', 2)[0]}.",
                                  parents=parents))
            found = 0
            for fact in self.facts_of(node, relation):
                if distance and not rules.inheritable(fact.relation):
                    continue
                key = (fact.relation, fact.object.lower())
                if key in seen:            # R4: the nearest statement wins
                    continue
                confidence = rules.confidence_at(fact.confidence, distance)
                if confidence < rules.FLOOR:
                    continue
                seen.add(key)
                fact.distance = distance
                fact.confidence = confidence
                collected.append(fact)
                found += 1
            steps.append(Step(len(steps), "check", node, distance, "R2",
                              f"{found} new fact(s) from "
                              f"{node.rsplit('.', 2)[0]}.", facts_checked=found))
            if len(collected) >= limit * 3:
                break
        # R3: drop anything the concept explicitly denies.
        denied = {(rules.NEGATIONS[f.relation], f.object.lower())
                  for f in collected if f.relation in rules.NEGATIONS}
        kept = [f for f in collected if (f.relation, f.object.lower()) not in denied]
        if len(kept) != len(collected):
            steps.append(Step(len(steps), "block", concept, 0, "R3",
                              f"Dropped {len(collected) - len(kept)} fact(s) the "
                              f"concept explicitly denies."))
        kept.sort(key=lambda f: (-f.confidence, f.distance, f.relation))
        answer.evidence = kept[:limit]
        return answer

    def close(self) -> None:
        self.connection.close()
