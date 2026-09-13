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
            clash = self._pin_fights_the_question(question or "", pinned)
            if clash is not None:
                return clash
            for attempt in (self._gated, self._define, self._contrast,
                            self._causal, self._analogy):
                answer = attempt(question or "")
                if answer is not None:
                    return answer
        if not concept and self._is_backwards(question or ""):
            backwards = self._inverse(question or "")
            if backwards is not None:
                return backwards
        payload = super().ask(question, concept or subject_sense)
        payload["rules"] = {**payload.get("rules", {}), **V687_RULES}
        self._report_unused_pins(payload, pinned)
        return payload

    #: The tagger's labels in WordNet's alphabet, for comparing a pin against
    #: the way its word was actually used.
    USED_AS = {"NOUN": "n", "PROPN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}

    def _pin_fights_the_question(self, question: str,
                                 pinned: dict[str, str] | None) -> dict | None:
        """R18: a pin whose part of speech cannot complete the sentence.

        `can a dog bark` used to answer VERIFIED with `bark` pinned to
        `bark.n.01`, the tough protective covering of a tree -- and to
        `bark.n.03`, a three-masted sailing ship. Every reading gave the same
        answer with the same note, because the norms match the *word* against
        their predicates and never resolve it to a sense at all.

        Reporting that is `_report_unused_pins` below. This is the stronger
        case: the pin does not merely fail to bite, it contradicts the
        sentence it was made on. `can a dog bark` completes an auxiliary, so
        `bark` there is a verb, and asking whether a dog can tough-protective-
        covering-of-a-tree is not a question anyone can answer. R18 refuses a
        construction by name rather than answering the easier question hiding
        inside it, and this is one.

        Only the one slot grammar is certain about, which is why this asks
        `verb_slot` rather than reading a tag. A pin is the reader overruling
        the engine, so it must not be refused on the strength of a label the
        engine got wrong -- and it did get them wrong: in `can dogs bark` the
        tagger calls `dogs` a VERB and `bark` a NOUN, so a check against the
        tags refused `bark.v.04`, the correct reading, on a question the
        reader had already corrected twice over. After `can`, `does` or
        `will` the auxiliary needs completing and only a verb can complete
        it; everywhere else the pin is let through and `_report_unused_pins`
        says what became of it.
        """
        if not pinned:
            return None
        parse = self.parser.parse(question or "")
        slot = (parse.verb_slot or "").lower()
        if not slot:
            return None
        forms = {slot}
        for token in parse.tokens or []:
            if (token.get("lemma") or "").lower() == slot:
                forms.add((token.get("text") or "").lower())
        for word, sense in sorted(pinned.items()):
            parts = (sense or "").split(".")
            if (word or "").lower() not in forms:
                continue
            was = "v"
            if len(parts) < 3 or parts[1] == was:
                continue
            spoken = {"n": "a noun", "v": "a verb", "a": "an adjective",
                      "r": "an adverb"}
            gloss = self.reasoner.gloss(sense) or ""
            reason = (f"“{word}” pinned to {sense}, "
                      f"{spoken.get(parts[1], parts[1])}"
                      + (f" — {gloss}" if gloss else "") +
                      f", where the question uses it as "
                      f"{spoken.get(was, was)}")
            return self._shell(
                question, "UNSUPPORTED", "R18", note=f"This asks for {reason}.",
                steps=[self._step(0, "R18", "refused", "",
                                  f"Not answerable here: {reason}. Pick a "
                                  f"reading the sentence can take, or drop "
                                  f"the pin.", kind="block")],
                extra={"pins_refused": {word: sense}})
        return None

    def _report_unused_pins(self, payload: dict,
                            pinned: dict[str, str] | None) -> None:
        """Say which pins the answer did not actually use.

        `pins.applies_to` already reports which *rules* can honour a pin on a
        word, but it is asked about a word and not about a question, so it
        cannot know that this particular answer came from R17 -- which reads
        the sense key XCSLB ships and has nothing to overrule -- or that the
        predicate reached the answer as a bare string.

        A control that looks as though it works everywhere and works in
        places is worse than no control, so the answer carries what happened:
        a pin is honoured when its synset is the one the derivation actually
        stood on, and listed as unused when it is not.
        """
        payload["pins_used"], payload["pins_unused"] = {}, {}
        if not pinned:
            return
        # Both columns. R29 answers between two senses and the pinned one is
        # the *object* of the row it stands on -- `car.n.01 part_of
        # accelerator.n.01` -- so reading concepts alone reported the pin
        # that had just decided the answer as unused.
        stood_on = {payload.get("concept") or ""}
        for fact in payload.get("evidence") or []:
            stood_on.update({fact.get("concept") or "",
                             fact.get("object") or ""})
        for step in payload.get("steps") or []:
            matched = step.get("matched") or {}
            stood_on.update({step.get("concept") or "",
                             matched.get("concept") or "",
                             matched.get("object") or ""})
        for word, sense in pinned.items():
            if sense in stood_on:
                payload["pins_used"][word] = sense
            else:
                payload["pins_unused"][word] = sense
        if payload["pins_unused"]:
            said = ", ".join(f"“{word}” to {sense}"
                             for word, sense in
                             sorted(payload["pins_unused"].items()))
            payload["note"] = ((payload.get("note") or "").rstrip() + " "
                               + f"This answer did not use the reading you "
                                 f"pinned: {said}. Nothing on the path that "
                                 f"reached it resolves that word to a sense — "
                                 f"the norms match it as a word against their "
                                 f"own predicates — so the answer would read "
                                 f"the same under any reading of it.").strip()

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

    def sense_roles(self, question: str) -> dict[str, list[str]]:
        """Which words *this* question resolves to a sense, and as what.

        Asking the question is the only way to answer it. Judging a word on
        its own got both directions wrong: `birds` was reported as resolving
        to nothing, when the router turns it into `bird` and walks 29 kinds,
        and `fly` was reported as pinnable when R20 matches it as a string
        against the norms and no synset is chosen for it at all.

        So each router is asked which word it would resolve, and only those
        words get a choice. A chip that offers a pin which changes nothing is
        worse than no chip.
        """
        roles: dict[str, list[str]] = {}

        def mark(word: str | None, role: str) -> None:
            if word:
                roles.setdefault(word.lower(), []).append(role)

        subject = self.parser.parse(question or "").subject
        mark(subject, pins.APPLIES["subject"])
        routed = self.profiles.route(question or "")
        if routed:
            mark(routed[1], pins.APPLIES["class"])
        if self.identifier.describes(question or ""):
            mark(self.identifier.terms_of(question or "")[1],
                 pins.APPLIES["class"])
        read = self.causal.reads(question or "")
        if read:
            if read[0] == "explain":
                for word in read[1].split():
                    mark(word, pins.APPLIES["phrase"])
            else:
                for word in self.causal._event_words(read[1], question or ""):
                    mark(word, pins.APPLIES["event"])
        backwards = self.inverse.reads(question or "")
        if backwards:
            for word in backwards[1].split():
                mark(word, pins.APPLIES["phrase"])
        counted = self.COUNT_KINDS.match(
            (question or "").strip().lower().rstrip("?"))
        if counted:
            mark(counted.group(1).strip(), pins.APPLIES["class"])
        text = (question or "").strip().lower().rstrip("?")
        pair = self._two(text)
        for name in pair or ():
            mark(name, pins.APPLIES["class"])
        if not pair and (self.DIFFERENCE.search(text) or self.COMMON.search(text)
                         or self.SIMILAR.search(text)):
            # The comparison could not be made, and the reason is often the
            # sense: the first `bitch` is a difficulty, not a female dog. The
            # words stay pinnable precisely because a pin is what fixes it --
            # `cat` blinking between pinnable and not, depending on whether
            # the *other* word happened to be covered, is the confusing part.
            for word in re.findall(r"[a-z][a-z'-]*", text):
                if (word not in self.NOT_A_WORD and len(word) > 2
                        and self.reasoner.senses_of(word)):
                    mark(word, pins.APPLIES["class"])
        return roles

    def word_senses(self, question: str) -> dict:
        """Every content word of a question, with the senses it could carry.

        This is R6 turned around. The rules have always run per sense; what
        was missing was any way for the reader to see which sense a word was
        taken in, or to say it was the wrong one -- the sense card offered
        that for the subject alone, after the fact, and for one rule.
        """
        roles = self.sense_roles(question)
        seen: set[str] = set()
        words: list[dict] = []
        for word in re.findall(r"[a-z][a-z'-]*", (question or "").lower()):
            if word in self.NOT_A_WORD or word in seen or len(word) < 2:
                continue
            seen.add(word)
            # The lemma the *router* resolves, which is not always the word
            # the reader wrote: `birds` is read as `bird`, and a pin has to be
            # filed under the form the rules look up or it is never found.
            lemma, senses = word, self.reasoner.senses_of(word)
            if not senses or (word not in roles and word.endswith("s")):
                stem = word[:-1] if word.endswith("s") and len(word) > 3 else word
                if self.reasoner.senses_of(stem) and (
                        not senses or stem in roles):
                    lemma, senses = stem, self.reasoner.senses_of(stem)
            if not senses:
                continue
            words.append({
                "word": word, "lemma": lemma,
                "senses": [{"id": s["id"], "pos": s["pos"],
                            "definition": s["definition"],
                            "facts": s["fact_count"],
                            "default": s["chosen"]} for s in senses[:8]],
                "total": len(senses),
                "applies": roles.get(lemma) or roles.get(word) or [],
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

    # -- R26: what a thing is ---------------------------------------------
    #: `what is a robin`, `what are dogs`, `define a hammer`. The question has
    #: to end after the noun: `what is a dog made of` is a relation question
    #: and belongs to v684, and swallowing it here would be the same greed
    #: R18 exists to prevent.
    DEFINE = re.compile(r"^(?:what\s+(?:is|are)|define|what\s+does\s+"
                        r"(?:a|an|the)?\s*[a-z '-]+\s+refer\s+to)"
                        r"(?:\s+(?:a|an|the))?\s+([a-z][a-z '-]*?)\s*$")

    #: How far up to read. Two levels is a definition; six is a lecture.
    GENUS = 3

    def _define(self, question: str) -> dict | None:
        """R26. What a thing is, answered from the taxonomy that holds it.

        This is the question a taxonomy is *for*, and until the audit it was
        the one question it did not answer: `what is a robin` fell through to
        a property listing and came back "helpful, passionate, professional",
        which ConceptNet records of the name Robin and not of the bird.

        A definition here is the classical one -- genus and differentia. The
        genus is the immediate hypernym, which the taxonomy has exactly; the
        differentia is approximated by what the norms record of the thing and
        by the kinds beneath it. Where WordNet supplies a gloss it is quoted,
        because a written definition beats a reconstructed one.
        """
        match = self.DEFINE.match(question.strip().lower().rstrip("?"))
        if not match:
            return None
        word = match.group(1).strip()
        senses = self.reasoner.senses_of(word)
        if not senses:
            # `what is` opens more than a definition. `what is a dog made of`
            # and `what is the difference between a dog and a cat` both match
            # the shape, and the tail is not a name -- so a phrase this
            # ontology does not hold is handed on rather than refused, and
            # only a single unknown word is refused as one.
            if " " in word:
                return None
            return self._shell(
                question, "UNKNOWN_WORD", "R26",
                note=f"“{word}” is not a word in this ontology. Nothing can "
                     f"be said about it here without first being told what "
                     f"it is.",
                steps=[self._step(0, "R26", word, "", f"No sense of “{word}” "
                                  f"is recorded.", kind="block")])
        pinned = pins.of(word)
        chosen = next((sense for sense in senses if sense["id"] == pinned),
                      senses[0])
        climb = [name for name, distance, _ in
                 self.reasoner.ascend(chosen["id"]) if distance]
        above = climb[:self.GENUS]
        genus = above[0] if above else None
        # The top of the chain is the sort: whether this is a thing, an act,
        # a state or an abstraction. A cognition deciding what to *ask* next
        # needs it -- you ask what an object is made of and what an act leads
        # to -- and it was the one part of the walk not reported, because the
        # genus alone says `a kind of thrush` and never says `a physical
        # object`. WordNet's own answer is the last node before `entity`.
        sort = None
        for name in reversed(climb):
            if not name.startswith("entity."):
                sort = name
                break
        kinds = self.contrast.kinds_of(word)
        known = self.profiles.knows(word)
        note = f"{chosen['id']} — {chosen['definition']}."
        if genus:
            note += f" A kind of {genus.rsplit('.', 2)[0]}"
            note += (f", and under {sort.rsplit('.', 2)[0]} at the top."
                     if sort and sort != genus else ".")
        individual = self._individual(chosen["id"], climb, kinds)
        if individual:
            note += " " + individual
        if kinds and kinds["total"]:
            note += (f" {kinds['total']:,} kinds of it are recorded, "
                     f"{kinds['described']} of them described by the norms.")
        if known:
            note += (f" The norms describe it directly, so every property "
                     f"question about it is answerable.")
        if len(senses) > 1:
            note += (f" “{word}” has {len(senses)} senses; this is the one "
                     f"carrying the most facts. Pin another to define that.")
        spine = [(name.rsplit(".", 2)[0], f"{step + 1} level(s) up")
                 for step, name in enumerate(above)]
        hanging = {0: [(kind.split(".")[0], "a kind of " + word)
                       for kind in (kinds["direct_kinds"][:6] if kinds else [])]}
        return self._shell(
            question, "DEFINED", "R26", concept=chosen["id"], note=note,
            extra={"definition": {
                       "word": word, "sense": chosen["id"],
                       "gloss": chosen["definition"], "genus": genus,
                       "above": above, "senses": len(senses),
                       "described_by_norms": known, "sort": sort,
                       "names_an_individual": bool(individual),
                       "kinds": kinds["total"] if kinds else 0},
                   "identification": self._tree(
                       "definition", spine or [(word, "nothing above it")],
                       hanging, [(word, chosen["definition"][:60])])})

    #: Sorts under which a leaf with no kinds is usually one named thing
    #: rather than a category: WordNet files Adrian, Mary and Peter here.
    INDIVIDUALS_UNDER = ("person.n.01", "location.n.01", "organization.n.01",
                         "group.n.01", "region.n.03")

    def _individual(self, sense: str, climb: list[str], kinds) -> str:
        """Warn when a word names one thing WordNet records, not a kind.

        This matters more than it looks for anything that has to be *told*
        something. A reader who says "I am Adrian" means a person the system
        has never met; the store resolves `adrian` to a 20th-century
        physiologist, answers `is adrian a person` with a confident yes, and
        the two are never the same Adrian. `john` resolves to a toilet.

        The store has no instance relation to read -- WordNet's
        `instance_hypernym` was not carried across in the build -- so this is
        three structural signals rather than a stored fact, and it is worth
        saying which. Nothing beneath it in the taxonomy; a place in it under
        people, places or organisations; and a longer name that contains this
        one -- `edgar douglas adrian`, `saint peter the apostle`, `albert
        einstein`. Individuals have full names and kinds do not.

        The third signal is what makes it usable. Without it every childless
        occupation is flagged: `concierge` and `apostle` also have nothing
        beneath them. With it they are not, while `paris` is missed, having
        no longer form. A heuristic, and a conservative one.

        It changes no verdict. It only stops a name being taken for a kind in
        silence, which is the failure that matters for anything being told
        something.
        """
        if kinds and kinds.get("total"):
            return ""
        if not any(node in self.INDIVIDUALS_UNDER for node in climb):
            return ""
        row = self.reasoner.connection.execute(
            "SELECT lemma, descendants FROM concepts WHERE id = ?",
            (sense,)).fetchone()
        if not row or row["descendants"]:
            return ""
        lemma = row["lemma"]
        longer = [alias["lemma"] for alias in self.reasoner.connection.execute(
            "SELECT lemma FROM lemmas WHERE concept = ?", (sense,))
            if alias["lemma"] != lemma and lemma in alias["lemma"].split()]
        if not longer:
            return ""
        return (f"This names one individual — WordNet also calls it "
                f"“{longer[0]}” — and not a kind of thing. The store holds no "
                f"instances of its own, so a person or place you introduce is "
                f"not in it and cannot be looked up here.")

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
            # `what kinds of dogs are there`: the kind is named in the
            # plural, and the taxonomy files it in the singular.
            nlp = getattr(self.parser, "nlp", None)
            pieces = word.split()
            last = (nlp(pieces[-1])[0].lemma_.lower() if nlp else
                    pieces[-1][:-1] if pieces[-1].endswith("s")
                    else pieces[-1])
            singular = " ".join(pieces[:-1] + [last])
            if singular != word:
                found = self.contrast.kinds_of(singular)
                if found is not None:
                    word = singular
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
            return self._nearest(question, text) or self._uncomparable(
                question, text)
        if not pair:
            # A comparison this cannot make must say so. Returning None sent
            # `what do a bitch and a cat have in common` down the chain to the
            # backwards reading, which answered it about the word *common* --
            # "12 concepts stand in front of common under has part". A
            # question that is plainly a comparison is R21's to decline.
            return self._uncomparable(question, text)
        left, right = pair
        found = self.contrast.compare(left, right)
        if found is None:
            return None
        wants = "difference" if self.DIFFERENCE.search(text) else "common"
        # "walk 0 nodes together before parting at X" is a sentence arguing
        # with itself: sharing nothing and having a parting point are not both
        # true. Zero is its own case, and it is the interesting one -- two
        # concepts this alike whose stored paths diverge at the first step.
        walked = (f"they walk {len(found.together)} node(s) together before "
                  f"parting" + (f" at “{found.parted_at}”."
                                if found.parted_at else ".")
                  if found.together else
                  f"their paths part at the very first node"
                  + (f", “{found.parted_at}”." if found.parted_at
                     else ", sharing no prefix at all."))
        note = (f"{found.shared_total} properties shared, of "
                f"{found.left_total} and {found.right_total}; overlap "
                f"{found.jaccard:.0%}. In the stored trie {walked}")
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

    def _uncomparable(self, question: str, text: str) -> dict:
        """Name the side the norms do not cover, and what they cover instead.

        The norms describe 541 things, and a comparison needs two of them. The
        answer worth giving is not "no" but "not this one, though it is a kind
        of that one" -- `bitch` is not covered and `dog`, a level up, is.
        """
        covered, missing = [], []
        for word in re.findall(r"[a-z][a-z'-]*", text):
            if word in self.NOT_A_WORD or len(word) < 3:
                continue
            if self.profiles.knows(word):
                covered.append(word)
            elif self.reasoner.senses_of(word):
                missing.append(word)
        instead = {word: self.contrast.nearest_covered(word)
                   for word in missing}
        offers = [f"“{word}” is not — the nearest they do cover to "
                  f"{found[2]} is “{found[0]}”, {found[1]} level(s) away"
                  for word, found in instead.items() if found]
        note = "A comparison needs two concepts the feature norms describe. "
        note += (f"They cover {', '.join(covered)}. " if covered else
                 "They cover 541 things. ")
        if offers:
            note += ("; ".join(offers) + ". Ask about that instead, or pin a "
                     "different sense of the word.")
        elif missing:
            note += (f"Neither {', '.join(missing)} nor anything above it is "
                     f"among them.")
        else:
            note += "This question names only one of them."
        return self._shell(
            question, "UNKNOWN", "R21", note=note,
            extra={"identification": self._tree(
                "uncomparable",
                [(word, "not described by the norms") for word in missing]
                or [(text, "nothing to compare")],
                {0: [(found[0], f"covered — {found[2]} is a kind of it, "
                                f"{found[1]} level(s) up")
                     for found in instead.values() if found]},
                [(word, "described by the norms") for word in covered])})

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
        left, right = pair
        # The class is whichever side has kinds beneath it, not whichever was
        # written second. Trying one order and then the other looked like a
        # fallback and was really a swap into nonsense: `is a dog a typical
        # animal` found no core for animal, flipped, and answered "animal
        # carries 0% of what a dog typically has, ranking 0 of 3".
        member, klass = ((left, right)
                         if len(self.profiles.subtypes(right))
                         >= len(self.profiles.subtypes(left))
                         else (right, left))
        found = self.contrast.typicality(member, klass)
        if found is None:
            return self._no_core(question, member, klass)
        verdict = "VERIFIED" if found.score >= 0.5 else "CONTRADICTED"
        note = (f"{member} carries {found.score:.0%} of what a {klass} "
                f"typically has, ranking {found.rank} of {found.of}.")
        if found.excluded:
            note += (f" Ranked against the {found.of} kinds of {klass} that "
                     f"{found.corpus} describes; {found.excluded} more are "
                     f"described by the other corpus, whose vocabulary does "
                     f"not overlap this one and cannot be scored against it.")
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

    def _no_core(self, question: str, member: str, klass: str) -> dict:
        """Say that a class has no core, rather than ranking against nothing.

        Typicality is only meaningful where the kinds of a class agree about
        something. Under free elicitation they often do not, and the honest
        answer is that the question has no footing here -- not a percentage
        computed from an empty set.
        """
        corpus = self.profiles.origin.get(member)
        kinds = [kind for kind in self.profiles.subtypes(klass)
                 if not corpus or self.profiles.origin.get(kind) == corpus]
        note = (f"“{klass}” has no core to be typical of. ")
        if corpus and kinds:
            note += (f"{len(kinds)} of its kinds are described in the same "
                     f"vocabulary as {member} ({corpus}), and no property is "
                     f"shared by even a quarter of them. Typicality is "
                     f"agreement among a class's members, and there is none "
                     f"here to measure against.")
        elif corpus:
            note += (f"None of its kinds are described in the same vocabulary "
                     f"as {member} ({corpus}), and the two corpora share no "
                     f"property names, so nothing can be scored across them.")
        else:
            note += (f"The norms do not describe {member}, so there is "
                     f"nothing to rank.")
        return self._shell(
            question, "UNKNOWN", "R21", concept=member, note=note,
            extra={"identification": self._tree(
                "typicality", [(klass, "no agreed core")],
                {0: [(kind, "a kind, but agreeing with no other")
                     for kind in kinds[:6]]},
                [(member, "cannot be ranked")])})

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
        if found is None:
            return None
        if not found.subjects:
            # Silence the reader caused looks exactly like silence in the
            # data, and only one of those is worth changing their mind about.
            # Without a pin the question goes on to the next rule; with one,
            # it stops here and says the pin is why.
            chosen = pins.of(found.phrase) or (
                pins.of(found.phrase.split()[-1]) if found.phrase else None)
            if not chosen:
                return None
            return self._shell(
                question, "UNKNOWN", "R22",
                note=f"Nothing stands in front of “{found.phrase}” under "
                     f"{found.relation.replace('_', ' ')} once the reading is "
                     f"held to {chosen}, the sense you pinned. Unpin it to "
                     f"read every sense of the word.",
                extra={"backwards": found.as_dict(),
                       "identification": self._tree(
                           "backwards",
                           [(found.phrase, f"held to {chosen}")], {},
                           [("nothing", "no subject at this sense")])})
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
