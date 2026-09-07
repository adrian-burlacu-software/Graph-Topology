"""The engine that routes a question to the reasoning it needs.

The order is the whole design. Each layer is more specific than the one under
it, and anything a layer does not recognise falls through rather than being
forced into it:

    R18 gating      refuse what no rule covers, by name
    contrast        two concepts named, or a class to be typical of   (R21)
    causal          why, what happens, what explains                  (R23)
    analogy         a is to b as what is to d                         (R24)
    profile/verify  one named concept from the norms                  (R17)
    identify        a description with no name                        (R16)
    inverse         a relation and an object with no subject          (R22)
    v685 / v684     everything else, unchanged

Falling through is what keeps this a superset rather than a fifth fork: every
question v684, v685 and v686 answered is answered the same way, because none
of the layers above recognises it.
"""
from __future__ import annotations

import re
from pathlib import Path

from . import logic, rules as v684_rules
from .analogy import Analogies
from .causal import Causal
from .contrast import Contrast
from .inverse import Inverse
from .relevance import RULE_TEXT as V685_RULES
from .server import V686_RULES, V687_RULES, IdentifyingEngine


class ReasoningEngine(IdentifyingEngine):
    """v686's engine, with the reasoning the v686 audit found missing."""

    title = "V687 Reasoner"

    def __init__(self, store: Path, depth: int = 3, breadth: int = 60):
        super().__init__(store, depth=depth, breadth=breadth)
        self.contrast = Contrast(self.profiles)
        self.inverse = Inverse(self.reasoner)
        self.causal = Causal(self.reasoner, self.inverse, parser=self.parser)
        self.analogies = Analogies(self.profiles)

    ALL_RULES = {**v684_rules.RULE_TEXT, **V685_RULES, **V686_RULES,
                 **V687_RULES}

    def ask(self, question: str, concept: str | None = None) -> dict:
        if not concept:
            for attempt in (self._gated, self._contrast, self._causal,
                            self._analogy):
                answer = attempt(question or "")
                if answer is not None:
                    return answer
        payload = super().ask(question, concept)
        # The inverse is tried only after everything above has declined, so a
        # backwards question never takes one that names its own subject.
        if not concept and payload.get("verdict") in (
                "UNKNOWN", "UNPARSED", "UNKNOWN_WORD", "LISTING"):
            backwards = self._inverse(question or "")
            if backwards is not None:
                return backwards
        payload["rules"] = {**payload.get("rules", {}), **V687_RULES}
        return payload

    # -- R18: refuse by name -----------------------------------------------
    def _gated(self, question: str) -> dict | None:
        reason = logic.unsupported(question)
        if reason is None:
            return None
        return self._shell(question, "UNSUPPORTED", "R18",
                           note=f"This asks for {reason}.",
                           steps=[self._step(0, "R18", "refused", "",
                                             f"Not answerable here: {reason}.",
                                             kind="block")])

    # -- R21: two concepts at once -----------------------------------------
    DIFFERENCE = re.compile(r"\b(difference|differ|differs)\b")
    COMMON = re.compile(r"\b(in common|both|share|shared)\b")
    SIMILAR = re.compile(r"\b(similar to|like a|like an|resembles?|closest to)\b")
    TYPICAL = re.compile(r"\b(typical|ordinary|unusual|representative)\b")

    def _contrast(self, question: str) -> dict | None:
        text = question.strip().lower().rstrip("?")
        if self.TYPICAL.search(text):
            return self._typicality(question, text)
        if not (self.DIFFERENCE.search(text) or self.COMMON.search(text)
                or self.SIMILAR.search(text)):
            return None
        pair = self._two(text)
        if self.SIMILAR.search(text) and not pair:
            return self._nearest(question, text)
        if not pair:
            return None
        left, right = pair
        found = self.contrast.compare(left, right)
        if found is None:
            return None
        wants = "difference" if self.DIFFERENCE.search(text) else "common"
        note = (f"{found.shared_total} properties shared, of "
                f"{found.left_total} and {found.right_total}; overlap "
                f"{found.jaccard:.0%}. In the stored trie they walk "
                f"{len(found.together)} node(s) together before parting"
                + (f" at “{found.parted_at}”." if found.parted_at else "."))
        steps = [
            self._step(0, "R21", "shared", "", "Both carry: "
                       + ", ".join(found.shared[:6] or ["nothing"])),
            self._step(1, "R21", left, "", f"Only {left}: "
                       + ", ".join(found.only_left[:6] or ["nothing"]),
                       parents=["shared"]),
            self._step(2, "R21", right, "", f"Only {right}: "
                       + ", ".join(found.only_right[:6] or ["nothing"]),
                       parents=["shared"]),
        ]
        return self._shell(question, "LISTING", "R21", concept=left, note=note,
                           steps=steps,
                           extra={"contrast": {"mode": wants,
                                               **found.as_dict()}})

    def _nearest(self, question: str, text: str) -> dict | None:
        name = self.profiles.named(text)
        if not name:
            return None
        near = self.contrast.nearest(name)
        if not near:
            return None
        steps = [self._step(i, "R21", other["name"], "",
                            f"{other['name']}: {other['shared']} shared, "
                            f"overlap {other['jaccard']:.0%} — "
                            + ", ".join(other["because"]),
                            parents=[name] if i == 0 else [])
                 for i, other in enumerate(near)]
        return self._shell(
            question, "LISTING", "R21", concept=name,
            note="Nearest by shared properties, computed rather than stored: "
                 "the `similar_to` relation carries 21,877 facts and cannot "
                 "say why any two things are alike.",
            steps=steps,
            extra={"contrast": {"mode": "nearest", "name": name,
                                "nearest": near}})

    def _typicality(self, question: str, text: str) -> dict | None:
        pair = self._two(text)
        if not pair:
            return None
        member, klass = pair
        found = self.contrast.typicality(member, klass)
        if found is None:
            member, klass = klass, member
            found = self.contrast.typicality(member, klass)
        if found is None:
            return None
        verdict = "VERIFIED" if found.score >= 0.5 else "CONTRADICTED"
        note = (f"{member} carries {found.score:.0%} of what a {klass} "
                f"typically has, ranking {found.rank} of {found.of}.")
        if not found.sound:
            verdict = "UNKNOWN"
            note += (f" The measure is not sound here: the most ordinary "
                     f"{klass} reaches only {found.support:.0%}, so the class "
                     f"has no core to rank against. XCSLB elicitation is free "
                     f"and sparse, and no two people list the same things.")
        steps = [self._step(i, "R21", entry["name"], "",
                            f"{entry['name']} carries {entry['score']:.0%} of "
                            f"the core of {klass}")
                 for i, entry in enumerate(found.ranking)]
        return self._shell(question, verdict, "R21", concept=member, note=note,
                           steps=steps,
                           extra={"contrast": {"mode": "typicality",
                                               **found.as_dict()}})

    #: Classes the norms do not cover but whose kinds they do, so that
    #: "what do a dog and a bird have in common" has two sides.
    CLASSES = ("bird", "fish", "animal", "dog", "cat", "whale", "insect")

    def _two(self, text: str) -> tuple[str, str] | None:
        """The two concepts a comparison names, in the order they appear."""
        seen: dict[int, str] = {}
        for name in list(self.profiles.stated) + list(self.CLASSES):
            match = re.search(r"\b" + re.escape(name) + r"\b", text)
            if not match:
                continue
            if name in self.CLASSES and not self.profiles.subtypes(name):
                continue
            start = match.start()
            # Longest at each position: `blue whale` beats `whale`.
            if len(name) > len(seen.get(start, "")):
                seen[start] = name
        picked: list[str] = []
        for start in sorted(seen):
            name = seen[start]
            # `whale` inside `blue whale` is the same mention, not a second.
            if any(start >= other and start + len(name) <= other + len(seen[other])
                   for other in seen if other != start):
                continue
            picked.append(name)
        return (picked[0], picked[1]) if len(picked) >= 2 else None

    # -- R23: scripts and abduction ---------------------------------------
    def _causal(self, question: str) -> dict | None:
        read = self.causal.reads(question)
        if read is None:
            return None
        kind, phrase = read
        if kind == "explain":
            found = self.causal.explains(phrase)
            steps = [self._step(i, "R23", h.cause, "",
                                f"{h.fact} — score {h.score:.3f}, explains "
                                f"{h.explains} thing(s) in all")
                     for i, h in enumerate(found.hypotheses)]
            note = found.note or (
                f"{found.considered} candidate causes weighed. The best "
                f"accounts for the observation while committing to least "
                f"else, which is what makes this a ranking and not a list.")
            return self._shell(question,
                               "LISTING" if found.hypotheses else "UNKNOWN",
                               "R23", note=note, steps=steps,
                               extra={"causal": {"mode": "abduction",
                                                 **found.as_dict()}})
        found = (self.causal.why(phrase, question) if kind == "why"
                 else self.causal.script(phrase, question=question))
        steps = [self._step(i, "R23", entry["phase"], "",
                            f"{entry['phase']}: {entry['object']}")
                 for i, entry in enumerate(found.steps)]
        return self._shell(question,
                           "LISTING" if found.steps else "UNKNOWN", "R23",
                           concept=found.concept, note=found.note, steps=steps,
                           extra={"causal": {"mode": kind, **found.as_dict()}})

    # -- R24: analogy ------------------------------------------------------
    ANALOGY = re.compile(r"^(.+?)\s+(?:is|are)\s+to\s+(?:a\s+|an\s+|the\s+)?"
                         r"(.+?)\s+as\s+what\s+(?:is|are)\s+to\s+"
                         r"(?:a\s+|an\s+|the\s+)?(.+)$")

    def _analogy(self, question: str) -> dict | None:
        match = self.ANALOGY.search(question.strip().lower().rstrip("?"))
        if not match:
            return None
        prop, source, target = (part.strip() for part in match.groups())
        found = self.analogies.solve(prop, source, target)
        steps = [self._step(i, "R24", m.property, "",
                            f"{m.property} — {m.kind}, standing "
                            f"{m.standing:.0%}, carried by {m.carriers}")
                 for i, m in enumerate(found.mappings)]
        note = found.note or (
            f"“{found.source_property}” is a {found.kind} property standing "
            f"{found.standing:.0%} of the way up {source}'s own properties. "
            f"These are {target}'s at the same kind and standing.")
        return self._shell(question,
                           "LISTING" if found.mappings else "UNKNOWN", "R24",
                           concept=target, note=note, steps=steps,
                           extra={"analogy": found.as_dict()})

    # -- R22: the graph backwards -----------------------------------------
    def _inverse(self, question: str) -> dict | None:
        found = self.inverse.answer(question)
        if found is None or not found.subjects:
            return None
        steps = [self._step(i, "R22", row["concept"], "",
                            f"{row['concept']} "
                            f"{row['relation'].replace('_', ' ')} "
                            f"“{row['object']}”")
                 for i, row in enumerate(found.subjects)]
        evidence = [{"concept": row["concept"], "relation": row["relation"],
                     "object": row["object"], "source": row["source"],
                     "confidence": row["confidence"], "sense_assumed": False,
                     "distance": 0, "similarity": 0.0}
                    for row in found.subjects]
        return self._shell(
            question, "LISTING", "R22",
            note=f"Read backwards: {len(found.subjects)} concept(s) stand in "
                 f"front of “{found.phrase}” under "
                 f"{found.relation.replace('_', ' ')}.",
            steps=steps, evidence=evidence,
            extra={"backwards": found.as_dict()})

    # -- the shape every answer here shares --------------------------------
    @staticmethod
    def _step(index: int, rule: str, concept: str, label: str, detail: str,
              parents: list[str] | None = None, kind: str = "check") -> dict:
        return {"index": index, "kind": kind, "concept": str(concept),
                "distance": index, "rule": rule, "detail": detail,
                "facts_checked": 0, "matched": None,
                "parents": parents if parents is not None else []}

    def _shell(self, question: str, verdict: str, rule: str,
               concept: str | None = None, note: str = "",
               steps: list[dict] | None = None,
               evidence: list[dict] | None = None,
               extra: dict | None = None) -> dict:
        payload = {
            "question": question, "verdict": verdict, "concept": concept,
            "concept_gloss": None, "senses": [], "chain": [],
            "evidence": evidence or [], "suggestions": [],
            "parse": {"question": question, "subject": concept,
                      "relation": rule, "target": None, "polar": False,
                      "backend": self.parser.backend, "tokens": [],
                      "note": f"answered by {rule}"},
            "note": note, "steps": steps or [],
            "rules": self.ALL_RULES, "store": self.reasoner.store.name,
            "neighbourhood": {"nodes": [], "edges": []},
        }
        payload.update(extra or {})
        return payload
