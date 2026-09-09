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
    #: How much of the question this fact covered, 0..1. Only set on
    #: suggestions, where the point is that it was not enough.
    similarity: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "concept": self.concept, "relation": self.relation,
            "object": self.object, "source": self.source,
            "confidence": round(self.confidence, 4),
            "sense_assumed": bool(self.sense_assumed), "distance": self.distance,
            "similarity": round(self.similarity, 3),
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
    #: Facts that answered part of the question but not enough of it. Kept
    #: separate from `evidence` on purpose: they are not the answer, and
    #: promoting one to be the answer is exactly the bug this replaced.
    suggestions: list[Fact] = field(default_factory=list)
    parse: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question, "verdict": self.verdict,
            "concept": self.concept, "concept_gloss": self.concept_gloss,
            "senses": self.senses, "chain": self.chain,
            "steps": [s.as_dict() for s in self.steps],
            "evidence": [e.as_dict() for e in self.evidence],
            "suggestions": [s.as_dict() for s in self.suggestions],
            "parse": self.parse, "note": self.note,
            "rules": rules.RULE_TEXT,
        }


class Reasoner:
    """Read-only inference over the store built by `build.py`."""

    MAX_DEPTH = 16
    MAX_ANCESTORS = 400
    #: A fact covering at least this much of the question is worth offering
    #: as a near miss; below it the overlap is a coincidence.
    SUGGEST_FLOOR = 0.34
    MAX_SUGGESTIONS = 5

    def __init__(self, store: Path):
        self.store = store
        self._has_rank: bool | None = None
        self.connection = sqlite3.connect(f"file:{store}?mode=ro", uri=True,
                                          check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._broad: set[str] | None = None      # R12, loaded on first use

    # -- lookup -----------------------------------------------------------
    def senses_of(self, lemma: str, pos: str | None = None
                  ) -> list[dict[str, Any]]:
        """R6: a word is not a concept. Return every sense it could mean.

        `pos` is the part of speech the word was *used* as, when a parser has
        said. Without it `fly` in `do pigs fly` offers four noun senses --
        a fisherman's lure first -- ahead of every verb sense, for a word the
        tagger has already called a verb. The order is a default, and a
        default should give way to evidence from the sentence in front of it.
        """
        # A store built before `ranks.py` existed has no rank column, and a
        # reasoner that cannot read an older store is a reasoner that breaks
        # every build anyone still has lying about.
        ranked = self.has_sense_rank
        rows = self.connection.execute(
            # The sense the build's evidence picked comes first. WordNet's own
            # order is not a usefulness ranking: it puts the part of a gunlock
            # ahead of the tool, so offering senses in it made every default
            # answer about `hammer` an answer about a gun.
            "SELECT c.id, c.lemma, c.pos, c.sense, c.definition, "
            "l.primary_sense" + (", l.sense_rank " if ranked else ", 99 AS sense_rank ") +
            "FROM lemmas l JOIN concepts c ON c.id = l.concept "
            # And a multi-word concept is demoted below every single-word
            # one when the question used a single word. `pig` is a lemma of
            # `pig bed.n.01`, a mould for casting pig iron, which the build's
            # evidence made primary because the crawl has more rows about
            # foundry beds than about pigs. Someone asking about a pig does
            # not mean a pig bed, whatever the evidence counts say.
            "WHERE l.lemma = ? ORDER BY "
            "(? <> '' AND c.pos <> ?), "
            "(instr(c.lemma, ' ') > 0 AND instr(?, ' ') = 0), "
            "l.primary_sense DESC, "
            # WordNet's own order for *this* lemma, below the sense the build
            # chose and above everything else. It is not a usefulness
            # ranking on its own -- it puts the part of a gunlock ahead of
            # the hammer -- which is why the build's choice still outranks
            # it. But it beats preferring a synset merely for being *named*
            # after the word: `pig` offered `pig.n.06`, a crude block of
            # metal with no facts at all and WordNet's sixth reading, ahead
            # of `hog.n.03`, domestic swine, which is WordNet's first and the
            # one the store knows something about.
            + ("l.sense_rank, " if ranked else "") +
            "(c.lemma <> ?), "
            "CASE c.pos WHEN 'n' THEN 0 WHEN 'v' THEN 1 WHEN 'a' THEN 2 ELSE 3 END, "
            "c.sense", (lemma.lower().strip(), (pos or ""), (pos or ""),
                        lemma.lower().strip(), lemma.lower().strip())
        ).fetchall()
        return [
            {"id": r["id"], "lemma": r["lemma"], "pos": r["pos"],
             "sense": r["sense"], "definition": r["definition"],
             "chosen": bool(r["primary_sense"]),
             "rank": r["sense_rank"],
             "fact_count": self.fact_count(r["id"])}
            for r in rows
        ]

    @property
    def has_sense_rank(self) -> bool:
        """Whether this store carries WordNet's per-lemma sense order."""
        if self._has_rank is None:
            self._has_rank = any(
                row[1] == "sense_rank" for row in
                self.connection.execute("PRAGMA table_info(lemmas)"))
        return self._has_rank

    def vocabulary(self) -> set[str]:
        """Every lemma the ontology knows, for the parser to find subjects with."""
        return {row[0] for row in self.connection.execute(
            "SELECT DISTINCT lemma FROM lemmas")}

    def noun_vocabulary(self) -> set[str]:
        """Every lemma with a noun sense, for telling a class from a property.

        `is a chair furniture` names a kind and `is a dog telepathic` names a
        property, and no part of speech tagger separates them reliably out of
        context -- the small model calls `furry` a noun and `telepathic` a
        proper noun. WordNet does separate them, because only one of the two
        has a noun sense at all.
        """
        return {row[0] for row in self.connection.execute(
            "SELECT DISTINCT lemmas.lemma FROM lemmas JOIN concepts "
            "ON concepts.id = lemmas.concept WHERE concepts.id LIKE '%.n.%'")}

    def fact_count(self, concept: str) -> int:
        return self.connection.execute(
            "SELECT COUNT(*) FROM facts WHERE concept = ?", (concept,)
        ).fetchone()[0]

    def gloss(self, concept: str) -> str | None:
        row = self.connection.execute(
            "SELECT definition FROM concepts WHERE id = ?", (concept,)
        ).fetchone()
        return row["definition"] if row else None

    #: Whether R28 applies to a fact stated of the asked concept itself.
    #:
    #: R28 exists because a qualified fact can be about a minority: `fish
    #: capable_of "walk on land"` is the mudskippers. That is a claim about a
    #: *class* being carried down to its members, which is inheritance, and at
    #: distance zero there is no inheritance to abuse -- `leopard capable_of
    #: "hunt at night"` is about leopards, and a leopard that hunts at night
    #: hunts.
    #:
    #: Set False to test that reading; `research/v688/audit.py` does, and the
    #: number is in AUDIT.md. It is True here because the audit says so: the
    #: relaxation is not safe, and the reason is that `rock capable_of "go for
    #: swim"` is stated of rock itself.
    R28_ON_STATED = True

    #: R27. Branches of the taxonomy that nothing belongs to two of. These
    #: are not guessed: `plant.n.02`, `animal.n.01`, `person.n.01`,
    #: `artifact.n.01` and `abstraction.n.06` were checked against each other
    #: and none is an ancestor of another.
    #:
    #: Kept deliberately small. WordNet's hypernym tree is incomplete in the
    #: middle -- it does not record that a dog is a pet, or a whale not a
    #: fish -- so exclusion is claimed only between these top branches, where
    #: the tree really does partition. Everything finer stays UNKNOWN, which
    #: is the answer the closed-world assumption licenses.
    PARTITIONS = ("plant.n.02", "animal.n.01", "person.n.01",
                  "artifact.n.01", "abstraction.n.06")

    #: The one pair that the tree separates and the world does not. WordNet
    #: files `person` beside `animal` rather than under it, so the partitions
    #: would have `a person is not an animal`. It reads one way only: a person
    #: is an animal, and a dog is still not a person.
    NOT_REALLY_DISJOINT = (("person.n.01", "animal.n.01"),)

    def partition_of(self, concept: str) -> str | None:
        """Which top branch a sense belongs to, if one of them."""
        above = {node for node, _, _ in self.ascend(concept)}
        for node in self.PARTITIONS:
            if node in above:
                return node
        return None

    def target_partitions(self, target_lemma: str) -> set:
        """The top branches every noun sense of a lemma lives in.

        `excludes` computes this to refuse; `engine.py` uses it to *choose*.
        A question names two things, and when one of them is unambiguous about
        its branch it says which reading of the other was meant: `mammal` is
        under `animal`, so `is a donkey a mammal` is about the donkey that is
        also under `animal` and not about the symbol of the Democratic Party.
        """
        forms = {target_lemma.lower(), target_lemma.lower().replace(" ", "_")}
        marks = ",".join("?" * len(forms))
        senses = [row["concept"] for row in self.connection.execute(
            f"SELECT concept FROM lemmas WHERE lemma IN ({marks})",
            tuple(forms)) if ".n." in row["concept"]]
        return {branch for branch in
                (self.partition_of(sense) for sense in senses) if branch}

    def excludes(self, concept: str, target_lemma: str) -> str | None:
        """R27. Is the target in a branch this concept cannot be in?

        Closed-world silence is the right answer to `is a dog a pet`: nothing
        recorded, and WordNet's middle is too full of holes to read absence as
        denial. It is the wrong answer to `is a dog a plant`. The difference
        is that plants and animals are different branches of the tree, and
        nothing is in both.

        Every sense of the target has to be excluded before this says no.
        `plant` also means a factory and a stooge in an audience, and a
        question is only settled if it is settled whichever was meant.
        """
        mine = self.partition_of(concept)
        if mine is None:
            return None
        forms = {target_lemma.lower(), target_lemma.lower().replace(" ", "_")}
        marks = ",".join("?" * len(forms))
        senses = [row["concept"] for row in self.connection.execute(
            f"SELECT concept FROM lemmas WHERE lemma IN ({marks})",
            tuple(forms)) if ".n." in row["concept"]]
        if not senses:
            return None
        theirs = [self.partition_of(sense) for sense in senses]
        if any(branch is None for branch in theirs):
            return None
        for branch in theirs:
            if branch == mine:
                return None
            if (mine, branch) in self.NOT_REALLY_DISJOINT:
                return None
        return mine

    def parents_of(self, concept: str) -> list[str]:
        return [r["parent"] for r in self.connection.execute(
            "SELECT parent FROM taxonomy WHERE child = ? ORDER BY parent", (concept,)
        )]

    def too_broad(self, concept: str) -> bool:
        """R12: is this concept so general that word-level facts stop applying?"""
        if self._broad is None:
            self._broad = {row[0] for row in self.connection.execute(
                "SELECT id FROM concepts WHERE descendants >= ?",
                (rules.BREADTH_LIMIT,))}
        return concept in self._broad

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
        # WordNet writes a multi-word lemma with underscores, and a question
        # writes it with spaces: `is a greeting a speech act` looked up
        # "speech act", found nothing, and reported that speech act was not
        # among greeting's ancestors -- which it is, directly.
        forms = {target_lemma.lower(), target_lemma.lower().replace(" ", "_")}
        marks = ",".join("?" * len(forms))
        wanted = {row["concept"] for row in self.connection.execute(
            f"SELECT concept FROM lemmas WHERE lemma IN ({marks})",
            tuple(forms))}
        if not wanted:
            answer.note = f"“{target_lemma}” is not a concept in this ontology."
            return answer
        steps.append(Step(len(steps), "resolve", target_lemma, 0, "R6",
                          f"“{target_lemma}” could mean any of "
                          f"{len(wanted)} sense(s); any of them counts."))

        for node, distance, parents in self.ascend(concept):
            answer.chain.append(node)
            steps.append(Step(len(steps), "ascend" if distance else "check",
                              node, distance, "R1",
                              f"Generalise to {node.rsplit('.', 2)[0]}." if distance
                              else f"Start from {node.rsplit('.', 2)[0]}.",
                              parents=parents))
            if node in wanted:
                answer.verdict = "VERIFIED"
                fact = Fact(concept, "is_a", node, "wordnet", 0.95, False, distance)
                fact.confidence = rules.confidence_at(0.95, 0)
                answer.evidence.append(fact)
                # Distance zero is identity, and subsumption is reflexive:
                # every bee is a bee. Excluding it left `is a bee a bee`
                # answering UNKNOWN after walking the whole taxonomy, which
                # is the one is_a question that needs no walk at all.
                why = (f"{node.rsplit('.', 2)[0]} is an ancestor of "
                       f"{concept.rsplit('.', 2)[0]}, {distance} step(s) up. "
                       f"Subsumption is transitive, so yes."
                       if distance else
                       f"This is {node.rsplit('.', 2)[0]} itself, nothing "
                       f"above it. Subsumption is reflexive, so yes.")
                steps.append(Step(len(steps), "match", node, distance, "R1",
                                  why, matched=fact.as_dict()))
                return answer
        branch = self.excludes(concept, target_lemma)
        if branch:
            # R27. Not silence -- exclusion. The two are in branches of the
            # taxonomy that share no members, and that is a real no.
            answer.verdict = "CONTRADICTED"
            answer.note = (
                f"“{target_lemma}” is not among the {len(answer.chain)} "
                f"ancestors of {concept}, and it cannot be: {concept} is "
                f"under {branch.rsplit('.', 2)[0]}, and every sense of "
                f"“{target_lemma}” is under a different top branch of the "
                f"taxonomy. Nothing belongs to two of them, so this is a no "
                f"with a reason rather than an absence.")
            steps.append(Step(len(steps), "check", concept,
                              len(answer.chain), "R27",
                              f"{concept.rsplit('.', 2)[0]} is under "
                              f"{branch.rsplit('.', 2)[0]}; “{target_lemma}” "
                              f"is not, in any sense. Disjoint branches."))
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
        nearby: list[Fact] = []
        # R28: matches that carry the target inside a wider claim. Held back
        # rather than dropped -- they are the whole of what the store knows,
        # and the note reports them.
        qualified: list[Fact] = []
        plain = getattr(matcher, "plain", None) or (lambda obj, want: True)

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
            if distance and self.too_broad(node):
                # R12: `person` and `artifact` carry thousands of facts stated
                # about the *words*. Inherited, they answer every question.
                dropped = [f for f in candidates if f.sense_assumed]
                candidates = [f for f in candidates if not f.sense_assumed]
                if dropped:
                    # `skip`, not `stop`: the walk carries on past this node,
                    # only its word-level facts are set aside. Marking it the
                    # same as a contradiction said the derivation halted here.
                    steps.append(Step(len(steps), "skip", node, distance, "R12",
                                      f"Ignoring {len(dropped)} word-level "
                                      f"fact(s) on {node.rsplit('.', 2)[0]}: "
                                      f"too general to inherit from."))
            # `parents` is carried on every step, not only on ascents: the asked
            # concept never produces an ascend step, so without this it would
            # have no outgoing edges in the graph view.
            steps.append(Step(len(steps), "check", node, distance, "R2",
                              f"Check {len(candidates)} `{relation}` fact(s) on "
                              f"{node.rsplit('.', 2)[0]}.",
                              facts_checked=len(candidates), parents=parents))

            # R3 again, from inside the object. The relation column says
            # `has_part` and the object column says "no legs", so the negation
            # pre-pass above -- which only ever looked at the relation --
            # never saw it, and lemma overlap read "no legs" as an answer to
            # "legs". `does a fish have legs` came back VERIFIED on the fact
            # that a fish has none.
            #
            # A whole pass over the node before any positive is considered,
            # not a test inside the positive loop, because `facts_of` orders
            # by confidence and a denial is routinely the weaker row. Snake
            # River carries both `has_a "leg"` at 0.46 and `has_a "no legs"`
            # at 0.35, and whichever is read first decides the answer. R3's
            # rule is that an explicit negation at a level blocks the
            # positive at that level, so the pass has to finish first.
            for fact in candidates:
                if not rules.denial_in(fact.object, target, matcher):
                    continue
                # Unless the same node states it plainly, from the same
                # source, at a higher confidence. `winter has_property cold`
                # is 0.87 and `winter has_property "never cold"` is 0.23 --
                # both Ascent++, and somebody's sentence about a mild winter
                # does not outweigh twenty saying it is cold.
                #
                # Same source is the whole of the comparison. Confidences do
                # not compare across sources here: Ascent++ has a real
                # distribution, ConceptNet is the constant 0.35 for every row
                # it holds. Snake River carries `has_a "leg"` at 0.46 from
                # the crawl and `has_a "no legs"` at 0.35 from ConceptNet,
                # and reading 0.46 > 0.35 as "the positive is better attested"
                # compares a percentile against a placeholder.
                louder = any(
                    other.source == fact.source
                    and other.confidence > fact.confidence
                    and plain(other.object, target)
                    and matcher(other.object, target)
                    and not rules.denial_in(other.object, target, matcher)
                    for other in candidates)
                if louder:
                    continue
                fact.distance = distance
                fact.confidence = rules.confidence_at(fact.confidence, distance)
                answer.verdict = "CONTRADICTED"
                answer.evidence.append(fact)
                steps.append(Step(len(steps), "block", node, distance, "R3",
                                  f"{node.rsplit('.', 2)[0]} "
                                  f"{relation.replace('_', ' ')} "
                                  f"“{fact.object}” — stated in "
                                  f"the negative, so it denies "
                                  f"“{target}” rather than answering "
                                  f"it.",
                                  matched=fact.as_dict()))
                answer.note = (
                    f"{node.rsplit('.', 2)[0]} is recorded as "
                    f"“{fact.object}”. That is a denial written into "
                    f"the object rather than the relation, and it is about "
                    f"the thing asked, so this is a no.")
                return answer

            for fact in candidates:
                if not matcher(fact.object, target):
                    # A partial hit is worth showing and worth not believing.
                    # `device capable_of "fall into wrong hands"` covers half
                    # of "fall into a hole" and answers none of it.
                    close = getattr(matcher, "score", None)
                    if close is not None:
                        share = close(fact.object, target)
                        if share >= self.SUGGEST_FLOOR:
                            near = Fact(fact.concept, fact.relation, fact.object,
                                        fact.source, fact.confidence,
                                        fact.sense_assumed, distance, share)
                            nearby.append(near)
                    continue
                fact.distance = distance
                fact.confidence = rules.confidence_at(fact.confidence, distance)
                if fact.confidence < rules.FLOOR:
                    continue
                if not plain(fact.object, target) and (
                        distance or self.R28_ON_STATED):
                    # R28. The fact says more than the question asked, and the
                    # surplus is doing the work: `rock capable_of "go for
                    # swim"` is about a place people swim, and `fish capable_of
                    # "walk on land"` is about the fish that do. Keep walking:
                    # a plainer statement further up still answers.
                    if len(qualified) < self.MAX_SUGGESTIONS:
                        steps.append(Step(len(steps), "skip", node, distance,
                                          "R28",
                                          f"{node.rsplit('.', 2)[0]} "
                                          f"{relation.replace('_', ' ')} "
                                          f"“{fact.object}” — that "
                                          f"is a narrower claim than "
                                          f"“{target}”, so it is not "
                                          f"an answer to it.",
                                          matched=fact.as_dict()))
                    qualified.append(fact)
                    continue
                answer.verdict = "VERIFIED"
                answer.evidence.append(fact)
                steps.append(Step(len(steps), "match", node, distance, "R4",
                                  f"Found it: {node.rsplit('.', 2)[0]} "
                                  f"{relation.replace('_', ' ')} "
                                  f"“{fact.object}”.",
                                  matched=fact.as_dict()))
                return answer
        if answer.verdict == "UNKNOWN":
            # Nearest first, and only the best few: a long list of things that
            # nearly answered reads as an answer again.
            seen_objects: set[str] = set()
            for near in sorted(nearby, key=lambda f: (-f.similarity, f.distance)):
                key = near.object.lower()
                if key in seen_objects:
                    continue
                seen_objects.add(key)
                answer.suggestions.append(near)
                if len(answer.suggestions) >= self.MAX_SUGGESTIONS:
                    break
            if qualified and not answer.note:
                # R28. Not silence: the store had something, and it was not
                # this. Saying so is the difference between "nothing is
                # recorded" and "what is recorded is narrower than the
                # question", and only the second names what to ask next.
                shown = ", ".join(f"“{fact.object}”"
                                  for fact in qualified[:3])
                where = qualified[0].concept.rsplit('.', 2)[0]
                answer.note = (
                    f"The store records {shown} of {where}, but each of those "
                    f"is a narrower claim than “{target}”. Affirming a "
                    f"qualified property does not affirm the property, so on "
                    f"the question as asked this is unsettled rather than "
                    f"yes.")
                answer.suggestions = (qualified[:self.MAX_SUGGESTIONS]
                                      + answer.suggestions)[:self.MAX_SUGGESTIONS]
            if not answer.note:
                answer.note = (f"Walked {len(answer.chain)} concepts up from "
                               f"{concept} without finding it. Absent, not false.")
        return answer

    #: R29: the relations whose objects are synsets rather than free text,
    #: and which store rows answer a question asked under each.
    #:
    #: Only WordNet writes an object as a synset id -- 22,187 rows each for
    #: `has_part` and `part_of`, 21,374 `similar_to`, 408 `entails`, 220
    #: `causes`, across 36,283 concepts. Every other relation in this store,
    #: `capable_of` and its 772,890 rows included, is free text from a crawl.
    #:
    #: The direction is the store's, not the name's. `(car.n.01, part_of,
    #: accelerator.n.01)` is in there and an accelerator is a part of a car,
    #: not the reverse; so are `(hand.n.01, part_of, finger.n.01)` and
    #: `(bird.n.01, part_of, beak.n.02)`. Both spellings are held and both
    #: read the same way round, so `part_of` is where a "does X have Y"
    #: question looks. Checked against four unambiguous pairs rather than
    #: assumed from the column name.
    #:
    #: (concept column, relation to search, object column) per asked relation.
    SENSE_TAGGED: dict[str, tuple[str, bool]] = {
        "has_part": ("part_of", False),   # the object is a part of the node
        "has_a": ("part_of", False),
        "part_of": ("part_of", True),     # the node is a part of the object
        "similar_to": ("similar_to", None),   # symmetric: either column
        "causes": ("causes", False),
        "entails": ("entails", False),
    }

    def verify_sense(self, concept: str, relation: str, sense: str) -> Answer:
        """R29: answer between two synsets, with no string matching at all.

        A pin on the object is the reader saying which thing they mean, and
        on the free-text relations there is nothing for it to bind to: `can a
        dog bark` is answered by matching the word "bark" against a norm
        predicate, and every one of `bark`'s nine senses is spelled the same.

        Where both ends are synsets there is something to bind to, and this
        is that path. It matches the pinned synset itself or one of its
        recorded kinds -- `accelerator.n.01` is answered by a gas pedal too
        -- and it inherits, so what is true of a car is true of a hatchback.

        UNKNOWN here is not an answer, it is this path declining; the caller
        falls back to the words and says which of the two spoke.
        """
        answer = Answer(question="", verdict="UNKNOWN", concept=concept,
                        concept_gloss=self.gloss(concept))
        route = self.SENSE_TAGGED.get(relation)
        if not route or not sense:
            return answer
        stored, reversed_ = route
        steps = answer.steps
        steps.append(Step(len(steps), "resolve", concept, 0, "R29",
                          f"Reading “{concept}” as this sense, not as a "
                          f"word, and “{sense}” as well — both ends of "
                          f"this are senses, so no word is matched."))
        wanted = {sense} | {row[0] for row in self.connection.execute(
            "SELECT child FROM taxonomy WHERE parent = ?", (sense,))}
        marks = ",".join("?" * len(wanted))
        for node, distance, parents in self.ascend(concept):
            answer.chain.append(node)
            if distance and not rules.inheritable(relation):
                break
            if reversed_:
                rows = self.connection.execute(
                    f"SELECT * FROM facts WHERE relation = ? AND object = ? "
                    f"AND concept IN ({marks})",
                    (stored, node, *wanted)).fetchall()
            elif reversed_ is None:
                rows = self.connection.execute(
                    f"SELECT * FROM facts WHERE relation = ? AND "
                    f"((concept = ? AND object IN ({marks})) OR "
                    f" (object = ? AND concept IN ({marks})))",
                    (stored, node, *wanted, node, *wanted)).fetchall()
            else:
                rows = self.connection.execute(
                    f"SELECT * FROM facts WHERE relation = ? AND concept = ? "
                    f"AND object IN ({marks})",
                    (stored, node, *wanted)).fetchall()
            steps.append(Step(len(steps), "check", node, distance, "R29",
                              f"Ask the graph whether "
                              f"{node.rsplit('.', 2)[0]} and "
                              f"{sense.rsplit('.', 2)[0]} are recorded as "
                              f"`{relation}`: {len(rows)} row(s).",
                              facts_checked=len(rows), parents=parents))
            if not rows:
                continue
            row = rows[0]
            fact = Fact(row["concept"], row["relation"], row["object"],
                        row["source"], row["confidence"],
                        bool(row["sense_assumed"]), distance)
            fact.confidence = rules.confidence_at(fact.confidence, distance)
            answer.verdict = "VERIFIED"
            answer.evidence.append(fact)
            answer.note = (
                f"Between senses, not between words: the graph records "
                f"{row['concept']} — {row['relation'].replace('_', ' ')} "
                f"→ {row['object']}"
                + (f", inherited {distance} level(s) down to "
                   f"{concept.rsplit('.', 2)[0]}" if distance else "") +
                f". Nothing here was matched as a string, so the reading you "
                f"pinned is the reading that answered.")
            steps.append(Step(len(steps), "match", node, distance, "R29",
                              f"Found it: {row['concept']} "
                              f"{row['relation'].replace('_', ' ')} "
                              f"{row['object']}.", matched=fact.as_dict()))
            return answer
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
            breadth = rules.BREADTH_LIMIT if self.too_broad(node) else 0
            for fact in self.facts_of(node, relation):
                if distance and not rules.inheritable_from(
                        fact.relation, breadth, fact.sense_assumed):
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
                              f"{node.rsplit('.', 2)[0]}.", facts_checked=found,
                              parents=parents))
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
        # R4 first, confidence second. Sorting on confidence alone let a
        # borrowed fact outrank a stated one: `violin capable_of run android`,
        # inherited from `device` four levels up, sat above `sound beautiful`
        # stated about violins, because Ascent++ scored the electronics higher.
        # What is said about the thing itself comes first, always.
        kept.sort(key=lambda f: (f.distance, -f.confidence, f.relation))
        answer.evidence = kept[:limit]
        return answer

    def close(self) -> None:
        self.connection.close()
