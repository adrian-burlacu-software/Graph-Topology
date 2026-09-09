"""One command: everything v685 serves, plus identification.

    python -m research.v687

The page, the store, the rules and the bridge are v685's, untouched. What is
added is the inverse question -- describe a thing and be told what it is:

    what kind of dog has spots
    what kind of cat has stripes
    what kind of bird is red

A question that describes rather than names is answered by `identify.py` and
comes back with its narrowing shown step by step, in the same shape the page
already replays.

A question that *names* one of those things is answered by `profile.py`, which
walks the same trie the other way -- from the individual's leaf back to the
origin, which is where its attributes are:

    what attributes does a blue whale have
    is a blue whale furry
    does a robin fly

Everything else falls through to v685, which falls through to v684, so this is
a superset of a superset rather than a third fork.
"""
from __future__ import annotations

import argparse
import re
import threading
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path

from . import build, compress, rules as v684_rules, engine as v684_server
from .relevance import RULE_TEXT as V685_RULES
from .bridged import BridgedEngine
from . import logic, profile
from .identify import Identifier
from .profile import (Profiles, CORROBORATION_FLOOR,
                      CORROBORATION_MIN_KINDS)

#: Rule text for the identification half, listed on the page beside the rest.
V686_RULES: dict[str, str] = {
    "R16": "Identification: a description is answered by walking the trie "
           "down instead of storing into it. Properties are taken general "
           "first, so each step narrows visibly -- round things, then the "
           "round thing with hexagons. Rarest-first would identify in fewer "
           "questions but answer the whole thing at step one. What a norm "
           "states about a thing outranks what it inherits.",
    "R17": "Retrieval is the same walk backwards. An individual sits at a "
           "leaf, so walking from that leaf to the origin recovers exactly "
           "the predicates it was stored with: the ones met first are shared "
           "with nothing, the ones met last with half the corpus. Nodes where "
           "nothing branched are collapsed. A property the norms scored false "
           "is a denial and not a silence; one they never mention is looked "
           "for above the leaf, in what the concept inherits.",
}

#: The badge a verdict from the norms wears on the page. `INHERITED` is a yes
#: like `HELD` is, but the note says which ancestor supplied it.
VERDICT_STYLE = {"HELD": "VERIFIED", "DENIED": "CONTRADICTED",
                 "INHERITED": "VERIFIED", "MIXED": "MIXED",
                 "UNRECORDED": "UNRECORDED"}

class IdentifyingEngine(BridgedEngine):
    """v685's engine, with descriptions answered by identification."""

    title = "V686 Reasoner"

    def __init__(self, store: Path, depth: int = 3, breadth: int = 60):
        super().__init__(store, depth=depth, breadth=breadth)
        # shares the open reasoner: the identifier needs the same taxonomy
        self.identifier = Identifier(store, reasoner=self.reasoner,
                                     parser=self.parser)
        # The same norms, the same reasoner and the same trie, read upwards.
        self.profiles = Profiles(self.identifier)

    def corroborate(self, answer, target: str):
        """R19 on the fact path: is the inherited fact true of the class?

        R19 was written for the norms path and wired only into it, so the two
        paths held opposite standards of proof. The norms path refuses `animal
        has a wing` because 20 of 143 animals bear it out. The fact path
        believed `mammal capable_of fly` on one crawled sentence, and that is
        why `do pigs fly` came back VERIFIED: nothing on hog, swine, ungulate
        or placental, and then a giant class five levels up that some of its
        members can indeed do.

        A crawled fact about a class summarises its members existentially --
        bats fly, monotremes lay eggs -- and inheriting it reads that
        existential as a universal. The norms are the only thing here that can
        tell the two apart, because they asked a fixed question of every
        concept they cover, so silence in them is informative where silence in
        a crawl is not.

        Only inherited facts are put to this test. A fact stated of the asked
        concept itself is not being generalised, so there is nothing to check.
        """
        if answer.verdict != "VERIFIED" or not answer.evidence:
            return answer
        fact = answer.evidence[0]
        if not getattr(fact, "distance", 0):
            return answer
        bearing, kinds = self.profiles.corroboration(fact.concept, target)
        if kinds < CORROBORATION_MIN_KINDS or bearing / kinds >= CORROBORATION_FLOOR:
            return answer
        name = fact.concept.rsplit(".", 2)[0]
        answer.verdict = "UNKNOWN"
        answer.suggestions = [fact] + list(answer.suggestions)
        answer.evidence = []
        answer.note = (
            f"{name} is recorded as “{fact.object}”, but only {bearing} "
            f"of the {kinds} kinds of {name} the norms cover bear that out. A "
            f"crawled sentence about a class says some of its members do this, "
            f"not that this one does, so it is not inherited down to "
            f"{answer.concept.rsplit('.', 2)[0]}. R19.")
        answer.steps.append(v684_rules.Step(
            len(answer.steps), "stop", fact.concept,
            getattr(fact, "distance", 0), "R19",
            f"Put “{fact.object}” to the other kinds of {name}: "
            f"{bearing} of {kinds} bear it out. Not inherited."))
        return answer

    def ask(self, question: str, concept: str | None = None) -> dict:
        if not concept:
            about = self.about(question or "")
            if about is not None:
                return about
        if concept or not self.identifier.describes(question or ""):
            payload = super().ask(question, concept)
            payload["rules"] = {**payload.get("rules", {}), **V686_RULES}
            return payload

        found = self.identifier.identify(question)
        if found.verdict == "NO_MATCH" and not found.terms:
            return super().ask(question, concept)

        payload = {
            "question": question,
            "verdict": found.verdict,
            "concept": found.candidates[0].name if found.candidates else None,
            "concept_gloss": None,
            "senses": [], "chain": [], "evidence": [], "suggestions": [],
            "parse": {"question": question, "subject": found.among,
                      "relation": "identify", "target": ", ".join(found.terms),
                      "polar": False, "backend": self.parser.backend,
                      "tokens": [], "note": ""},
            "note": found.note,
            "identification": found.as_dict(),
            "steps": self._steps(found),
            "rules": {**v684_rules.RULE_TEXT, **V685_RULES, **V686_RULES},
            "store": self.reasoner.store.name,
            "neighbourhood": {"nodes": [], "edges": []},
        }
        return payload

    # -- the trie read upwards --------------------------------------------
    def about(self, question: str) -> dict | None:
        """Answer a question about a *named* individual, or hand it back.

        Three ways out, and handing it back is two of them. v684 answers
        taxonomy questions with a derivation of its own, so `is a hammer a
        tool` is left alone however much the norms have to say about hammers;
        and a property the norms neither state nor deny is not an answer, so
        that goes back too and the fact graph gets its turn. What is kept is
        what the norms can actually settle.
        """
        routed = self.profiles.route(question)
        if routed is None:
            return None
        mode, name, words = routed
        if mode == "verify":
            parsed = self.parser.parse(question)
            if not words or (parsed.relation == "is_a" and not parsed.hedged):
                return None                    # v684 owns the taxonomy
            # A hedged `is_a` -- a bare noun predicate, no determiner -- is
            # only a taxonomy question if the norms have nothing to say.
            # `is a chair furniture` names a kind and `is a raccoon white`
            # names a property, and they are the same shape: the norms are
            # what tells them apart, so they are asked first.
        query = None
        if mode == "verify":
            # The question's own structure, not a bag of words: `a tail and
            # wings` is a conjunction, `furry or purple` a disjunction, and
            # `all birds` a quantifier over the kinds beneath.
            query = logic.parse(self._tail(question, name), profile.ASIDE)
        found = self.profiles.describe(name)
        if found is None and query is not None and query.quantifier:
            # A class the norms do not cover has no branch of its own, but its
            # kinds can still be counted.
            found = profile.Description(name=name,
                                        concept=self.profiles.synset.get(name))
        if found is None:
            return None
        if mode == "verify":
            found.asked = self.profiles.assess(name, query)
            # Unrecorded goes back to the fact graph -- but only when the
            # fact graph is going to be talking about the same thing. v684
            # read "is whale furry" with `furry` as its subject and answered
            # about `furred.a.01`, and "nothing is stored about that adjective"
            # is a worse answer than "here is everything a whale has, and
            # this is not among it".
            # Hand back only a *simple* unanswered question. When the question
            # had structure, v684 has no way to do better -- it would drop the
            # conjunction and answer half of it, which is the defect R20 was
            # written for -- so the three-valued answer is kept and says which
            # part is unknown.
            # A question is structured when it says so. Several *parts* is
            # not the same thing: `fall into a hole` is one claim that split
            # into words, and handing that to v684 is right because v684
            # matches the phrase whole. `a tail and wings` is a real
            # conjunction and must keep its three-valued answer, which is the
            # defect R20 was written for.
            tail = self._tail(question, name)
            structured = bool(found.asked.quantifier) or bool(
                re.search(r"\b(and|or|not|no|never)\b", tail))
            # An inherited answer the norms could not corroborate is not an
            # answer the norms gave. It came from one crawled sentence at an
            # ancestor, which is the fact graph's own material -- and the fact
            # graph reads it better, with the relation typed and the
            # confidence attached. `can a dog fall into a hole` was answered
            # here as `verify` over feature norms, on a card that promises
            # R1-R9, citing an ancestor the norms describe 7 kinds of.
            lean = (found.asked.verdict == "INHERITED"
                    and found.asked.corroborated is False)
            if ((found.asked.verdict == "UNRECORDED" or lean)
                    and not structured
                    and self.parser.parse(question).subject == name):
                return None
        return self._payload(question, mode, found)

    @staticmethod
    def _tail(question: str, name: str) -> str:
        """What the question says *about* the subject, with the subject gone.

        The connectives have to survive this: `route` strips them as noise
        because it only ever wanted content words, and a conjunction with its
        `and` removed is a list of two unrelated properties.
        """
        text = question.strip().lower().rstrip("?")
        text = re.sub(r"^(is|are|was|were|does|do|did|has|have|can|could)\b",
                      " ", text)
        # Both the name and its plural: the question says `birds` where the
        # class is `bird`. A subject left in the tree becomes a property to
        # test, and `is a whale furry` came back DENIED on the strength of
        # `is used to kill whales`.
        for form in (name, name + "s", name + "es"):
            text = re.sub(r"\b" + re.escape(form) + r"\b", " ", text)
        return text

    def _payload(self, question: str, mode: str, found) -> dict:
        """One answer carrying both halves: what is stored, and what is above.

        `evidence` is the whole profile as facts, distinctive attributes
        first, because the far end of the walk is the interesting one -- the
        predicates nothing else carries are what make this thing that thing.
        The inherited half follows, each fact attributed to the ancestor that
        supplies it rather than to the concept, so the two are never confused.
        """
        verdict = ("PROFILE" if found.asked is None
                   else VERDICT_STYLE.get(found.asked.verdict, "UNKNOWN"))
        stated = [{"concept": found.name, "relation": "stated",
                   "object": held.predicate, "source": found.source,
                   "confidence": 1.0, "sense_assumed": False,
                   "distance": 0, "similarity": 0.0}
                  for held in reversed(found.path)]
        inherited = [{"concept": level.concept, "relation": fact["relation"],
                      "object": fact["object"], "source": fact["source"],
                      "confidence": fact["confidence"],
                      "sense_assumed": fact["sense_assumed"],
                      "distance": level.distance, "similarity": 0.0}
                     for level in found.inherited for fact in level.facts]
        chain = ([found.concept] if found.concept else []) + [
            level.concept for level in found.inherited]
        above = sum(level.total for level in found.inherited)
        return {
            "question": question,
            "verdict": verdict,
            "concept": found.concept,
            "concept_gloss": found.gloss,
            "senses": [], "chain": chain,
            # Not `stated + inherited`. The profile card already lays both out
            # -- the walk with its sharing counts, the ancestry with its
            # attribution -- and repeating all 50 rows in the generic fact
            # table said the same thing twice, worse the second time.
            "evidence": [],
            "suggestions": [],
            "parse": {"question": question, "subject": found.name,
                      "relation": "profile" if mode == "profile" else "verify",
                      "target": found.asked.term if found.asked else None,
                      "polar": mode == "verify",
                      "backend": self.parser.backend, "tokens": [], "note": ""},
            "note": found.asked.detail if found.asked else
                    (f"{len(found.path)} attributes stated by the norms, "
                     f"{above:,} more inherited from "
                     f"{len(found.inherited)} level(s) above it."),
            "identification": self._as_identification(question, mode, found),
            "profile": found.as_dict(),
            "steps": self._walk_steps(found),
            "rules": {**v684_rules.RULE_TEXT, **V685_RULES, **V686_RULES},
            "store": self.reasoner.store.name,
            "neighbourhood": {"nodes": [], "edges": []},
        }

    @staticmethod
    def _label(segment: dict) -> str:
        """A segment's name on the tree: short enough to read on a node."""
        first = segment["predicates"][0]
        extra = len(segment["predicates"]) - 1
        return f"{first} +{extra}" if extra else first

    @staticmethod
    def _witnesses(asked) -> list[dict]:
        """What actually answered, as nodes the walk can stop on.

        The page promises every answer is replayable, and a verdict reached
        from four kinds below the concept was not: the steps walked the
        concept's own branch and then announced a denial with nothing in
        between. These are the things that voted -- the kinds, or the ancestor
        that lent the fact -- so each one is a step and a node of its own.
        """
        if asked is None:
            return []
        if asked.get("members"):
            return [{"name": member["name"], "verdict": member["verdict"],
                     "predicate": member["predicate"]}
                    for member in asked["members"]]
        if asked["source"] and asked["distance"]:
            return [{"name": re.sub(r"\.[nvar]\.\d+$", "", asked["source"]),
                     "verdict": asked["verdict"],
                     "predicate": asked["predicate"]}]
        return []

    @classmethod
    def _as_identification(cls, question: str, mode: str, found) -> dict:
        """The walk in the shape the tree drawing already knows.

        Identification narrows a field down a branch; this walks one branch
        that is already chosen. Both are a chain of trie nodes with things
        falling off the side, so the same drawing serves both, and the ghosts
        here are the concepts that shared the prefix this far and then went
        somewhere else -- `dolphin` leaves `blue whale` at the point where the
        branch commits to plankton.
        """
        segments = [segment.as_dict() for segment in found.segments]
        steps = [{"rule": "R17", "kind": "walk", "term": cls._label(segment),
                  "detail": ", ".join(segment["predicates"]),
                  "remaining": segment["shared"],
                  "eliminated": len(segment["dropped"]),
                  "examples": segment["dropped"][:6]}
                 for segment in segments]
        # A concept that answered the question is drawn where it answered it,
        # not where it left the branch: `dolphin` is both a trie neighbour of
        # `whale` and one of the four kinds that denied `furry`, and the tree
        # keys nodes by name, so drawing it twice left the replay lighting up
        # the wrong one.
        voted = {witness["name"] for witness in
                 cls._witnesses(found.asked.as_dict() if found.asked else None)}
        considered = [{"name": other, "concept": None, "survived": False,
                       "depth": position, "matched": {}}
                      for position, segment in enumerate(segments)
                      for other in segment["dropped"] if other not in voted]
        matched = ({found.asked.term: found.asked.predicate or "—"}
                   if found.asked and found.asked.predicate else {})
        # The question itself becomes the last node of the walk, so what
        # answered it has somewhere to hang.
        asked = found.asked.as_dict() if found.asked else None
        witnesses = cls._witnesses(asked)
        if asked:
            steps.append({
                "rule": "R17", "kind": "ask", "term": cls._asked(asked),
                "detail": asked["detail"], "remaining": 1,
                "eliminated": len(witnesses),
                "examples": [w["name"] for w in witnesses][:6]})
            considered.extend(
                {"name": witness["name"], "concept": None, "survived": False,
                 "depth": len(segments),
                 "matched": {asked["term"]: (witness["predicate"] or "—")
                                            + f" ({witness['verdict'].lower()})"}}
                for witness in witnesses)
        considered.append({"name": found.name, "concept": found.concept,
                           "survived": True, "depth": len(steps),
                           "matched": matched})
        return {
            "question": question, "mode": mode,
            "terms": [found.asked.term] if found.asked else [],
            "among": None, "among_concept": None,
            "verdict": found.asked.verdict if found.asked else "PROFILE",
            "note": found.asked.detail if found.asked else "",
            "candidates": [{"name": found.name, "source": found.source,
                            "matched": matched,
                            "predicates": len(found.path)}],
            "considered": considered, "steps": steps,
        }

    @staticmethod
    def _asked(asked: dict) -> str:
        """The node the question itself occupies on the walk."""
        return f"{asked['term']}?"

    @classmethod
    def _walk_steps(cls, found) -> list[dict]:
        """The replay: the branch, the question, what answered it, the verdict."""
        steps: list[dict] = []
        for position, segment in enumerate(found.segments):
            left = (f" — {len(segment.dropped)} went elsewhere"
                    if segment.dropped else " — nothing branched here")
            steps.append({
                "index": position, "kind": "check",
                "concept": cls._label(segment.as_dict()),
                "distance": position, "rule": "R17",
                "detail": (", ".join(segment.predicates)
                           + f" — shared with {segment.shared}" + left),
                "facts_checked": len(segment.predicates),
                "matched": None,
                "parents": [cls._label(found.segments[position - 1].as_dict())]
                           if position else [],
            })
        last = (cls._label(found.segments[-1].as_dict())
                if found.segments else None)
        asked = found.asked.as_dict() if found.asked else None
        if asked:
            # The question, then each thing that answered it. Without these
            # the replay walked the branch and then produced a verdict out of
            # nowhere -- "is a whale furry" showed six attribute nodes and no
            # sign of the four kinds that actually denied it.
            steps.append({
                "index": len(steps), "kind": "check",
                "concept": cls._asked(asked),
                "distance": len(found.segments), "rule": "R17",
                "detail": f"Ask “{asked['term']}” of {found.name}: "
                          + asked["detail"],
                "facts_checked": 0, "matched": None,
                "parents": [last] if last else [],
            })
            for witness in cls._witnesses(asked):
                steps.append({
                    "index": len(steps),
                    # `block` is the page's word for it, and the page paints
                    # it red. `blocked` would have quietly styled as nothing.
                    "kind": "block" if witness["verdict"] == "DENIED"
                            else "match",
                    "concept": witness["name"],
                    "distance": len(found.segments) + 1, "rule": "R17",
                    "detail": f"{witness['name']} — {witness['verdict']}"
                              + (f" via “{witness['predicate']}”"
                                 if witness["predicate"] else ""),
                    "facts_checked": 1, "matched": None,
                    "parents": [cls._asked(asked)],
                })
        steps.append({
            "index": len(steps), "kind": "match", "concept": found.name,
            "distance": len(steps), "rule": "R17",
            "detail": (asked["detail"] if asked else
                       f"{found.name}: {len(found.path)} stated attributes, "
                       f"read off the branch on the way back up."),
            "facts_checked": len(found.path), "matched": None,
            "parents": [cls._asked(asked)] if asked else
                       ([last] if last else []),
        })
        return steps

    @staticmethod
    def _steps(found) -> list[dict]:
        """One step per attribute on the branch, then the thing it identifies.

        The step's `concept` has to be the key the drawing used for its nodes,
        because that is what the replay looks up to highlight. Naming the
        rivals here instead put synset ids in the steps and attribute names on
        the tree, so every lookup missed and nothing lit up at all.
        """
        steps: list[dict] = []
        for position, round_ in enumerate(found.steps):
            dropped = round_.get("eliminated") or 0
            steps.append({
                "index": position,
                "kind": "check",
                "concept": round_["term"],
                "distance": position,
                "rule": round_.get("rule", "R16"),
                "detail": round_["detail"]
                          + (f" — {dropped} ruled out" if dropped else
                             " — nothing ruled out"),
                "facts_checked": dropped,
                "matched": None,
                "parents": [found.steps[position - 1]["term"]] if position else [],
            })
        for candidate in found.candidates[:4]:
            matched = "; ".join(f"“{term}” via {hit}"
                                for term, hit in candidate.matched.items())
            steps.append({
                "index": len(steps),
                "kind": "match",
                "concept": candidate.name,
                "distance": len(found.steps),
                "rule": "R16",
                "detail": f"{candidate.name}: {matched}",
                "facts_checked": len(candidate.matched),
                "matched": None,
                "parents": [found.steps[-1]["term"]] if found.steps else [],
            })
        return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=build.DEFAULT_STORE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8686)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--breadth", type=int, default=60)
    arguments = parser.parse_args()

    build.ensure(arguments.store)
    store = arguments.store
    if not arguments.raw:
        packed = compress.DEFAULT_COMPRESSED
        if not packed.exists():
            print("compressing by inheritance (first run)")
            compress.compress(arguments.store, packed)
        store = packed

    print("  loading feature norms and joining them to WordNet...")
    from .reasoning import ReasoningEngine
    engine = ReasoningEngine(store, arguments.depth, arguments.breadth)
    httpd = ThreadingHTTPServer((arguments.host, arguments.port),
                                v684_server.make_handler(engine))
    url = f"http://{arguments.host}:{httpd.server_port}/"
    print(f"\n  V687 reasoner  ->  {url}")
    print(f"  parser: {engine.parser.backend}")
    print(f"  {len(engine.identifier.stated):,} individuals from XCSLB + AwA2, "
          f"{len(engine.identifier.synset):,} joined to WordNet")
    print("  ctrl-c to stop\n")
    if not arguments.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("  stopped")
    finally:
        httpd.server_close()
        engine.bridged.graph.close()
        engine.reasoner.close()


if __name__ == "__main__":
    main()


#: Rule text for the reasoning the backlog added, listed with the rest.
V687_RULES: dict[str, str] = {
    "R18": "Question-shape gating: a construction no rule covers is refused "
           "by name, not answered from the part of it that happens to be "
           "understandable. Comparatives, superlatives, counts of parts, "
           "counterfactuals, dates, questions about words rather than senses, "
           "facts about named individuals and antonyms are named and "
           "declined -- every silent wrong answer found in the v686 and v687 "
           "audits came from answering an easier question than the one asked. "
           "A pinned sense whose part of speech cannot complete the sentence "
           "is refused the same way: `can a dog bark` answered VERIFIED with "
           "`bark` pinned to the covering of a tree, and to a three-masted "
           "sailing ship, because the norms match the word and never resolve "
           "it. Where a pin merely fails to bite rather than contradicting "
           "the sentence, the answer says which pins it did not use.",
    "R19": "Corroboration: an inherited fact is put to the ancestor's other "
           "kinds before it is believed. `bird capable_of fly` is borne out "
           "by 21 of 29 birds in the norms and is inherited; `animal has a "
           "wing` by 20 of 143 and is refused. One crawled sentence is not a "
           "property of a category. This governs both answering paths. It was "
           "written for the norms and wired only into them, so the fact store "
           "believed `mammal capable_of fly` on one sentence and answered `do "
           "pigs fly` yes; a crawled fact about a class says some of its "
           "members do this, and the norms are what tell an existential from "
           "a universal.",
    "R20": "Three-valued composition: a question with structure is evaluated "
           "in Kleene's logic, because silence is not falsehood. One false "
           "conjunct settles a conjunction, one true disjunct settles a "
           "disjunction, and an unknown part suspends the whole. Quantifiers "
           "ask every kind beneath a concept and count.",
    "R21": "Contrast: what two concepts share, where they part, how alike "
           "they are and how typical one is are one operation -- the lowest "
           "common ancestor of two branches. Semantic overlap and trie "
           "prefix are both reported, because they disagree: a trie built "
           "for storage does not group by similarity.",
    "R22": "Inverse traversal: the graph is read from the object as well as "
           "the subject, so `what is made of wood` is answerable and not only "
           "`what is a hammer made of`. Relations that pair (`has_part` and "
           "`part_of`) are read from both columns.",
    "R23": "Scripts and abduction: prerequisites, subevents and effects are "
           "walked in script order -- before, during, after. An observation "
           "is explained by ranking causes as competing hypotheses, scored by "
           "specificity, directness and confidence. A cause that causes forty "
           "things explains none of them.",
    "R28": "Qualified claims: a fact that carries the question inside a wider "
           "claim does not answer it. `rock capable_of “go for swim”` "
           "is about a place people swim and `fish capable_of “walk on "
           "land”` is about the fish that do, and both answered yes. "
           "v687 already declined the mirror of this -- denying a qualified "
           "property does not deny the property -- and this is the same "
           "reading applied to yes. The match itself is whole-word now: "
           "`fly` was named by “attract butterfly”, `walk` by "
           "“block the sidewalk” and `run` by “get drunk”, "
           "so `can a tree fly` was VERIFIED on a butterfly.",
    "R29": "Sense-to-sense lookup: where both ends of a question are synsets, "
           "it is answered between them and no string is matched. Only "
           "WordNet writes an object as a synset id -- has_part, part_of, "
           "similar_to, entails and causes, 36,283 concepts -- and that is "
           "the only place a pin on the object has anything to bind to. "
           "`does a car have an accelerator` is UNKNOWN through the words, "
           "because the accelerator is recorded only as a synset, and "
           "VERIFIED through the graph. It is tried first and it is not "
           "authoritative: the synset rows are patchy (they have a car's "
           "wheel and a dog's tail, and not a fish's gills), so when the "
           "graph is silent the words still answer and the note says which "
           "of the two spoke.",
    "R26": "Definition: `what is a robin` is answered from the taxonomy "
           "itself -- which sense is meant, the gloss WordNet gives it, what "
           "it is a kind of, and what kinds it has. Before this it fell "
           "through to a property listing and returned `helpful, passionate, "
           "professional` for a robin, because ConceptNet holds those of the "
           "name Robin. A definition is the one question a taxonomy answers "
           "by being a taxonomy.",
    "R27": "Taxonomic exclusion: absence is not denial, except between the "
           "top branches of the taxonomy, where it is. Nothing is both a "
           "plant and an animal, or both an artifact and an abstraction, so "
           "`is a dog a plant` is a no with a reason and not a silence. Held "
           "to those branches on purpose: WordNet's middle does not record "
           "that a dog is a pet, so `is a dog a pet` stays unknown. Every "
           "sense of the target has to be excluded, because `plant` also "
           "means a factory.",
    "R25": "Counting kinds: the ontology holds no numbers, so it cannot count "
           "a dog's legs -- but it can count what stands beneath a concept in "
           "the taxonomy, which is what `how many kinds of dog` asks. Two "
           "counts are reported: every descendant WordNet records, and the "
           "far smaller number the feature norms actually describe, because "
           "only the second can be reasoned about.",
    "R24": "Analogy over the norms only: a role is approximated by feature "
           "type plus standing within the concept. `bark : dog :: ? : cat` "
           "gives meow and purr. The scraped graph cannot support this and is "
           "not asked -- its `part_of` is largely taxonomy misfiled.",
}
