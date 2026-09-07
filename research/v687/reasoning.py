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

from . import logic, pins, rules as v684_rules
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

    def ask(self, question: str, concept: str | None = None,
            pinned: dict[str, str] | None = None) -> dict:
        pins.use(pinned)
        # A pin on the word v684 would take as its subject is the same choice
        # the sense card always offered, made from the question instead of
        # from a list under the answer. It is kept *separate* from `concept`,
        # which is a gate: `concept` means "the reader clicked a sense on an
        # answer, so re-answer that one thing", and setting it from a pin
        # skipped every rule above v684 -- pinning `bark` to its verb sense
        # stopped `why does a dog bark` being a why-question at all.
        subject_sense = None
        if not concept and pinned:
            subject = (self.parser.parse(question or "").subject or "").lower()
            subject_sense = pins.of(subject)
        if not concept:
            for attempt in (self._gated, self._contrast, self._causal,
                            self._analogy):
                answer = attempt(question or "")
                if answer is not None:
                    return answer
        if not concept and self._is_backwards(question or ""):
            backwards = self._inverse(question or "")
            if backwards is not None:
                return backwards
        payload = super().ask(question, concept or subject_sense)
        payload["rules"] = {**payload.get("rules", {}), **V687_RULES}
        return payload

    #: Words that never name a concept, so never get a sense chip.
    #:
    #: Two groups. Grammar, which WordNet does hold senses for and which no
    #: reader wants to disambiguate; and the *cue* vocabulary the router
    #: matches on -- `kinds`, `many`, `difference`, `happens`. A word the
    #: router consumes to decide which rule answers is not a word the answer
    #: resolves to a sense, so offering a choice on it is noise: "how many
    #: kinds of mouse" was showing chips for `many` and `kinds` beside the one
    #: chip that mattered.
    NOT_A_WORD = frozenset("""
    a an the of to for from in on at with by is are was were be been am
    do does did has have had can could will would shall should may might must
    what which who whom whose why how when where and or not no there this that
    these those it its they them their you your we our i me my
    kind kinds type types sort sorts breed breeds species many much more most
    difference differ differs common share shared both similar like typical
    ordinary unusual representative happens explains attribute attributes
    property properties feature features about know tell exist
    between among within than
    """.split())

    def word_senses(self, question: str) -> dict:
        """Every content word of a question, with the senses it could carry.

        This is R6 turned around. The rules have always run per sense; what
        was missing was any way for the reader to see which sense a word was
        taken in, or to say it was the wrong one -- the sense card offered
        that for the subject alone, after the fact, and for one rule.
        """
        seen: set[str] = set()
        words: list[dict] = []
        for word in re.findall(r"[a-z][a-z'-]*", (question or "").lower()):
            if word in self.NOT_A_WORD or word in seen or len(word) < 2:
                continue
            seen.add(word)
            senses = self.reasoner.senses_of(word)
            if not senses:
                # `bites` is not a lemma but `bite` is, and the reader wrote
                # the inflected form.
                stem = word[:-1] if word.endswith("s") and len(word) > 3 else word
                senses = self.reasoner.senses_of(stem)
            if not senses:
                continue
            words.append({
                "word": word,
                "senses": [{"id": s["id"], "pos": s["pos"],
                            "definition": s["definition"],
                            "facts": s["fact_count"],
                            "default": s["chosen"]} for s in senses[:8]],
                "total": len(senses),
                "applies": pins.applies_to(word, self),
            })
        return {"question": question, "words": words}

    def _is_backwards(self, question: str) -> bool:
        """Is this a question for the object side, and is it unclaimed?

        Deciding this from the *answer* was tried and fails both ways. Judging
        an empty `evidence` list as "nobody answered" handed identification's
        questions to the inverse, because an identification carries its
        evidence somewhere else; and judging any non-empty one as "answered"
        took `what is made of wood` back from the inverse, because v684 will
        always find something to say about wood.

        Question shape settles it. The inverse asks about the object side and
        names no subject, so a question that describes an unnamed thing (R16)
        or names a concept the norms cover (R17) is not one, whatever either
        of them ends up answering.
        """
        if self.inverse.reads(question) is None:
            return False
        if self.identifier.describes(question):
            return False                       # R16 owns descriptions
        return self.profiles.route(question) is None

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

    #: "how many kinds of dog are there", "what kinds of dog are there".
    #:
    #: The question has to *end* after the noun. An optional tail of `.*` let
    #: this swallow `what kind of animal is furry and has spots and is big`,
    #: which is an identification and was being answered with a count of
    #: animals -- the same greed R18 exists to prevent, one rule over.
    COUNT_KINDS = re.compile(r"^(?:how many|what|which)\s+(?:kinds?|types?|"
                             r"sorts?|breeds?|species)\s+of\s+([a-z ]+?)"
                             r"\s*(?:(?:are|is)\s+there|there\s+(?:are|is)|"
                             r"exists?|are\s+known|do\s+we\s+know)?\s*$")

    def _count(self, question: str) -> dict | None:
        """R25. The count R18 promises when it refuses to count legs."""
        match = self.COUNT_KINDS.match(question.strip().lower().rstrip("?"))
        if not match:
            return None
        word = match.group(1).strip()
        found = self.contrast.kinds_of(word)
        if found is None:
            return None
        return self._shell(
            question, "LISTING", "R25", concept=found["concept"],
            note=(f"{found['total']:,} kinds of {word} in the taxonomy, "
                  f"{found['direct']} of them directly beneath it. The feature "
                  f"norms describe {found['described']}, and those are the "
                  f"ones every other rule here can reason about."),
            extra={"kinds": found, "identification": self._tree(
                "kinds", [(word, found["gloss"] or found["concept"])],
                {0: [(kind.split(".")[0], "a kind of " + word)
                     for kind in found["direct_kinds"][4:12]]},
                [(kind.split(".")[0], "a kind of " + word)
                 for kind in found["direct_kinds"][:4]])})

    def _contrast(self, question: str) -> dict | None:
        counted = self._count(question)
        if counted is not None:
            return counted
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
        # The drawing is the comparison: the shared properties are the trunk
        # both concepts walk, and each one's own hang off the point they part.
        spine = [(prop, f"both {left} and {right}")
                 for prop in found.shared[:4]]
        if not spine:
            spine = [("nothing shared",
                      "these two carry no property in common")]
        hanging = {len(spine) - 1:
                   [(prop, f"only {left}") for prop in found.only_left[:4]]
                   + [(prop, f"only {right}") for prop in found.only_right[:4]]}
        return self._shell(question, "LISTING", "R21", concept=left, note=note,
                           extra={"contrast": {"mode": wants,
                                               **found.as_dict()},
                                  "identification": self._tree(
                                      "contrast", spine, hanging,
                                      [(left, ""), (right, "")])})

    def _nearest(self, question: str, text: str) -> dict | None:
        name = self.profiles.named(text)
        if not name:
            return None
        near = self.contrast.nearest(name)
        if not near:
            return None
        return self._shell(
            question, "LISTING", "R21", concept=name,
            note="Nearest by shared properties, computed rather than stored: "
                 "the `similar_to` relation carries 21,877 facts and cannot "
                 "say why any two things are alike.",
            extra={"contrast": {"mode": "nearest", "name": name,
                                "nearest": near},
                   "identification": self._tree(
                       "nearest", [(name, "shared properties")],
                       {0: [(other["name"], f"{other['shared']} shared")
                            for other in near[4:]]},
                       [(other["name"],
                         f"{round(other['jaccard'] * 100)}% overlap")
                        for other in near[:4]])})

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
        return self._shell(question, verdict, "R21", concept=member, note=note,
                           extra={"contrast": {"mode": "typicality",
                                               **found.as_dict()},
                                  "identification": self._tree(
                                      "typicality",
                                      [(prop, f"core of {klass}")
                                       for prop in found.core[:4]],
                                      {0: [(other["name"],
                                            f"{round(other['score'] * 100)}%")
                                           for other in found.ranking
                                           if other["name"] != member][:6]},
                                      [(member,
                                        f"{round(found.score * 100)}% of the "
                                        f"core")])})

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
            tree = self._tree(
                "abduction", [(phrase, "the observation to be explained")],
                {0: [(h.cause.split(".")[0], f"score {h.score:.3f}")
                     for h in found.hypotheses[4:]]},
                [(h.cause.split(".")[0],
                  f"score {h.score:.3f}, also causes {h.explains}")
                 for h in found.hypotheses[:4]])
            note = found.note or (
                f"{found.considered} candidate causes weighed. The best "
                f"accounts for the observation while committing to least "
                f"else, which is what makes this a ranking and not a list.")
            return self._shell(question,
                               "LISTING" if found.hypotheses else "UNKNOWN",
                               "R23", note=note,
                               extra={"causal": {"mode": "abduction",
                                                 **found.as_dict()},
                                      "identification": tree})
        found = (self.causal.why(phrase, question) if kind == "why"
                 else self.causal.script(phrase, question=question))
        # A script is phases, and each phase is what happens in it. One node
        # per fact loses the only structure a script has.
        phases: dict[str, list[str]] = {}
        for entry in found.steps:
            phases.setdefault(entry["phase"], []).append(entry["object"])
        spine = [(phase, f"{len(objects)} recorded")
                 for phase, objects in phases.items()]
        hanging = {depth: [(text, phase) for text in objects[:5]]
                   for depth, (phase, objects) in enumerate(phases.items())}
        tree = self._tree("script", spine or [(found.concept, "nothing found")],
                          hanging, [(found.concept, kind)])
        return self._shell(question,
                           "LISTING" if found.steps else "UNKNOWN", "R23",
                           concept=found.concept, note=found.note,
                           extra={"causal": {"mode": kind, **found.as_dict()},
                                  "identification": tree})

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
        tree = self._tree(
            "analogy",
            [(found.source_property or prop, f"of {source}"),
             (found.kind or "same kind", f"standing {found.standing:.0%}")],
            {}, [(m.property, f"of {target}, standing {m.standing:.0%}")
                 for m in found.mappings])
        note = found.note or (
            f"“{found.source_property}” is a {found.kind} property standing "
            f"{found.standing:.0%} of the way up {source}'s own properties. "
            f"These are {target}'s at the same kind and standing.")
        return self._shell(question,
                           "LISTING" if found.mappings else "UNKNOWN", "R24",
                           concept=target, note=note,
                           extra={"analogy": found.as_dict(),
                                  "identification": tree})

    # -- R22: the graph backwards -----------------------------------------
    def _inverse(self, question: str) -> dict | None:
        found = self.inverse.answer(question)
        if found is None or not found.subjects:
            return None
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
            evidence=evidence,
            extra={"backwards": found.as_dict(),
                   "identification": self._tree(
                       "backwards",
                       [(found.phrase, "read from the object side")],
                       {0: [(row["concept"].split(".")[0], row["object"])
                            for row in found.subjects[4:]]},
                       [(row["concept"].split(".")[0], row["object"])
                        for row in found.subjects[:4]])})

    # -- drawing the walk --------------------------------------------------
    @staticmethod
    def _tree(mode: str, spine: list[tuple[str, str]],
              hanging: dict[int, list[tuple[str, str]]],
              answers: list[tuple[str, str]]) -> dict:
        """Put an answer in the shape the graph already knows how to draw.

        Every rule here produces the same picture underneath: a chain of steps
        with things hanging off each one and a result at the end. That is what
        `buildTriePath` draws for an identification, so rather than five more
        drawings, each rule says what its chain, its side nodes and its
        answers are, and the existing one draws them.

        Node keys have to match the `concept` of the replay steps, because
        that is what the replay looks up to light them -- keying them
        differently is why an earlier version highlighted nothing at all.
        """
        return {
            "mode": mode, "terms": [], "among": None, "among_concept": None,
            "verdict": "", "note": "",
            "steps": [{"rule": "", "kind": "walk", "term": term,
                       "detail": detail, "remaining": 0,
                       "eliminated": len(hanging.get(depth, [])),
                       "examples": [name for name, _ in
                                    hanging.get(depth, [])][:6]}
                      for depth, (term, detail) in enumerate(spine)],
            "considered": [{"name": name, "concept": None, "survived": False,
                            "depth": depth, "matched": {"": detail} if detail
                                                       else {}}
                           for depth, entries in hanging.items()
                           for name, detail in entries]
                          + [{"name": name, "concept": None, "survived": True,
                              "depth": len(spine), "matched": {}}
                             for name, _ in answers],
            "candidates": [{"name": name, "source": mode,
                            "matched": {"": detail} if detail else {},
                            "predicates": 0}
                           for name, detail in answers],
        }

    # -- the shape every answer here shares --------------------------------
    @staticmethod
    def _step(index: int, rule: str, concept: str, label: str, detail: str,
              parents: list[str] | None = None, kind: str = "check") -> dict:
        return {"index": index, "kind": kind, "concept": str(concept),
                "distance": index, "rule": rule, "detail": detail,
                "facts_checked": 0, "matched": None,
                "parents": parents if parents is not None else []}

    @staticmethod
    def _replay(tree: dict, rule: str) -> list[dict]:
        """The step list, derived from the drawing rather than written twice.

        Writing both by hand is how they drift: the replay looks a step's
        `concept` up among the drawn nodes, so a step naming `shared` while
        the node is named `has a tail` lights nothing at all. Deriving one
        from the other makes that impossible.
        """
        steps: list[dict] = []
        spine = [round_["term"] for round_ in tree["steps"]]
        for depth, round_ in enumerate(tree["steps"]):
            steps.append({
                "index": len(steps), "kind": "check", "concept": round_["term"],
                "distance": depth, "rule": rule, "detail": round_["detail"],
                "facts_checked": round_.get("eliminated", 0), "matched": None,
                "parents": [spine[depth - 1]] if depth else []})
            for entry in tree["considered"]:
                if entry["survived"] or entry["depth"] != depth:
                    continue
                detail = " ".join(entry["matched"].values()) or entry["name"]
                steps.append({
                    "index": len(steps), "kind": "check",
                    "concept": entry["name"], "distance": depth + 1,
                    "rule": rule, "detail": f"{entry['name']} — {detail}",
                    "facts_checked": 0, "matched": None,
                    "parents": [round_["term"]]})
        for candidate in tree["candidates"]:
            detail = " ".join(candidate["matched"].values()) or candidate["name"]
            steps.append({
                "index": len(steps), "kind": "match",
                "concept": candidate["name"], "distance": len(spine),
                "rule": rule, "detail": f"{candidate['name']} — {detail}",
                "facts_checked": 0, "matched": None,
                "parents": [spine[-1]] if spine else []})
        return steps

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
        tree = payload.get("identification")
        if tree and not steps:
            payload["steps"] = self._replay(tree, rule)
        return payload
