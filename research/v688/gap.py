"""What an answer did not settle.

v687 answers a question and says, in prose, why it could not do better. This
reads that back as a typed record, so something other than a human can act on
it. Nothing here edits v687: every field is recovered from the payload it
already returns.

Two kinds of incompleteness, and they are not the same thing:

    Gap     the answer is missing. Memory said so itself -- a word it does
            not have, a sense it cannot choose between, a term nothing covers,
            a construction no rule reads. Rows 3 and 4 of the verdict table.

    Doubt   the answer is present and should not be trusted as far as it
            looks. `does a beagle swim` returns VERIFIED on one crawled fact
            at confidence 0.42, inherited three levels up, on a sense that was
            assumed rather than confirmed. The verdict says none of that.

A Gap asks to be filled. A Doubt asks to be corroborated. They generate
different questions, which is the whole reason for separating them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

#: Verdicts sorted by what they are *about*, from the v687 audit. The loop
#: reads the kind, never the verdict string: `UNKNOWN` and `CONTRADICTED`
#: were interchangeable in three code paths before the audit separated them,
#: and a loop built on that confusion learns false things.
ABOUT_THE_WORLD = frozenset({"VERIFIED", "CONTRADICTED", "DENIED", "HELD",
                             "INHERITED"})
ABOUT_CONTENT = frozenset({"DEFINED", "PROFILE", "LISTING", "IDENTIFIED"})
ABOUT_COVERAGE = frozenset({"UNKNOWN", "UNRECORDED", "NO_MATCH"})
ABOUT_THE_QUESTION = frozenset({"UNSUPPORTED", "UNKNOWN_WORD", "UNPARSED",
                                "AMBIGUOUS"})
ABOUT_RELIABILITY = frozenset({"MIXED"})

#: Gap kind per verdict. The kind, not the verdict, chooses the repair.
KIND = {
    "UNKNOWN_WORD": "word",          # a word the ontology does not have
    "UNPARSED": "construction",      # no subject could be found at all
    "UNSUPPORTED": "construction",   # R18 refused it by name
    "AMBIGUOUS": "sense",            # several readings, none chosen
    "UNKNOWN": "coverage",           # absent, and said so: not false
    "UNRECORDED": "coverage",        # the norms do not cover it
    "NO_MATCH": "coverage",          # nothing satisfied the description
    "MIXED": "conflict",             # the parts disagree with each other
}

#: Below this a ConceptNet or Ascent++ fact is crawl noise as often as it is
#: knowledge. `dog has_property wild` sits at 0.54 and answers `is a dog wild`
#: with yes; `dog capable_of swim` sits at 0.42 and every kind of dog the
#: norms cover denies it. The number is a threshold on a corpus, not a truth,
#: so it is named here rather than buried in a comparison.
WEAK_CONFIDENCE = 0.60

#: The curly quotes v687 writes its notes with, so a blocker can be recovered
#: from prose when no field carries it.
QUOTED = re.compile("[“\"]([^”\"]{1,40})[”\"]")

#: A synset id as v687 stores it. Evidence objects come back as `dog.n.01`,
#: and a question built straight from one asks `is a beagle a dog.n.01`,
#: which the parser then reports as an unknown word -- a gap the loop
#: manufactured for itself.
SYNSET = re.compile(r"^(.+)\.[nvasr]\.\d+$")

#: A stored object that denies rather than states. `fish has_a "no legs"` is
#: how the store records that fish lack them, and v687's verify path matches
#: `legs` inside it and answers VERIFIED. Reading that as a yes, and then
#: putting `no legs` to the family, gives `does a carp have no legs` --
#: whose denial means the carp *has* legs. Two wrongs that read as a right.
DENYING = re.compile(r"^\s*(?:no|not|non|never)\s+", re.I)

#: Determiners a blocker can arrive wearing. `does a beagle have an agility`
#: leaves the target as `an agility`, and `what is an an agility` follows.
LEADING = re.compile(r"^(?:an?|the|some|any|its|their|his|her)\s+", re.I)


def plain(text: str) -> str:
    """A stored object read back as words a question can be built from."""
    text = (text or "").strip()
    found = SYNSET.match(text)
    if found:
        text = found.group(1)
    text = text.replace("_", " ")
    return LEADING.sub("", text).strip()


@dataclass(frozen=True)
class Gap:
    """Something memory says it does not have."""

    kind: str                       # word|sense|coverage|construction|conflict
    blocker: str                    # the word, term or construction responsible
    verdict: str
    question: str                   # the question that ran into it
    options: tuple[str, ...] = ()   # for `sense`: the readings on offer
    detail: str = ""

    @property
    def blocking(self) -> bool:
        """Does this stop the question being answered at all?

        A `word` or `construction` gap does: nothing downstream of it means
        anything. A `coverage` gap does not -- the question was understood and
        the answer is a real, informative absence.
        """
        return self.kind in ("word", "construction", "sense")

    def as_dict(self) -> dict:
        return {"kind": self.kind, "blocker": self.blocker,
                "verdict": self.verdict, "question": self.question,
                "options": list(self.options), "detail": self.detail,
                "blocking": self.blocking}


@dataclass(frozen=True)
class Doubt:
    """A reason not to take a positive verdict at face value."""

    reason: str        # weak | inherited | assumed_sense | uncorroborated
    question: str
    concept: str = ""
    predicate: str = ""
    relation: str = ""
    confidence: float = 0.0
    distance: int = 0
    detail: str = ""

    def as_dict(self) -> dict:
        return {"reason": self.reason, "question": self.question,
                "concept": self.concept, "predicate": self.predicate,
                "relation": self.relation, "confidence": self.confidence,
                "distance": self.distance, "detail": self.detail}


def kind_of(verdict: str) -> str:
    """Which of the five things a verdict is about."""
    if verdict in ABOUT_THE_WORLD:
        return "world"
    if verdict in ABOUT_CONTENT:
        return "content"
    if verdict in ABOUT_COVERAGE:
        return "coverage"
    if verdict in ABOUT_THE_QUESTION:
        return "question"
    if verdict in ABOUT_RELIABILITY:
        return "reliability"
    return "unknown"


def _quoted(note: str) -> str:
    """The first word v687 put in quotes, which is the one it is naming."""
    found = QUOTED.search(note or "")
    return found.group(1) if found else ""


def read_gap(payload: dict, question: str = "") -> Gap | None:
    """The typed gap in a v687 answer, or None when it answered."""
    verdict = payload.get("verdict") or ""
    kind = KIND.get(verdict)
    if kind is None:
        return None
    parse = payload.get("parse") or {}
    note = payload.get("note") or ""
    question = question or payload.get("question") or ""

    blocker = ""
    options: tuple[str, ...] = ()
    if kind == "word":
        # The parser records which word defeated it; that word is the answer
        # to "what should I ask about next", and answering about the next
        # noun along was the bug that field was added for.
        blocker = (parse.get("unknown") or _quoted(note)
                   or parse.get("subject") or "")
    elif kind == "sense":
        blocker = parse.get("subject") or _quoted(note) or ""
        found = payload.get("identification") or {}
        options = tuple(entry.get("name") or entry.get("concept") or ""
                        for entry in (found.get("candidates") or [])[:8])
        options = tuple(name for name in options if name)
    elif kind == "coverage":
        # What was not covered is what was asked *of* the subject, not the
        # subject: `is a whale a fish` knows perfectly well about whales.
        blocker = (parse.get("target") or _quoted(note)
                   or parse.get("subject") or "")
    elif kind == "construction":
        blocker = _quoted(note) or (parse.get("relation") or "")
    elif kind == "conflict":
        blocker = parse.get("target") or parse.get("subject") or ""

    return Gap(kind=kind, blocker=plain(blocker), verdict=verdict,
               question=question, options=options, detail=note.strip())


def off_target(payload: dict) -> Doubt | None:
    """A verdict reached on a predicate that is not what was asked about.

    v687 scores a question as a set of content terms. `is a violin made of
    wood` becomes `made` and `wood`; `made` matches the stored predicate
    `can be made of ebony`, which the norms deny of violins, and the answer
    comes back CONTRADICTED -- about ebony. `does a dog live on the ground`
    goes the same way through `lives in a stable`.

    It is detectable without touching v687: the predicate it cites shares no
    word with the thing the question asked about.
    """
    parse = payload.get("parse") or {}
    target = (parse.get("target") or "").lower()
    note = payload.get("note") or ""
    cited = QUOTED.findall(note)
    if not target or not cited:
        return None
    wanted = {word for word in re.findall(r"[a-z]+", target)
              if word not in FRAME_WORDS}
    for phrase in cited:
        words = set(re.findall(r"[a-z]+", phrase.lower()))
        if len(words) < 2:
            continue
        if wanted & words:
            return None
        return Doubt(
            "off_target", payload.get("question") or "",
            payload.get("concept") or "", phrase,
            parse.get("relation") or "", 0.0, 0,
            f"the verdict rests on “{phrase}”, which has no word in common "
            f"with “{target}” — v687 scores a question one term at a time, "
            f"and a verb can match a predicate that is not what you asked "
            f"about")
    return None


#: Words that carry no content when matching a cited predicate to a question.
FRAME_WORDS = frozenset("""
a an the of in on at to for is are was were be been do does did have has had
can could will would made make making live lives living found find its their
""".split())


def read_doubts(payload: dict, question: str = "") -> list[Doubt]:
    """Every reason this answer is weaker than its verdict looks.

    Only positive verdicts are worth doubting. A `CONTRADICTED` reached on
    thin evidence is thin too, but it is already the cautious answer, and
    doubting both directions equally would have the loop chase every no it
    ever got.
    """
    verdict = payload.get("verdict") or ""
    stray = off_target(payload)
    if verdict not in ("VERIFIED", "HELD", "INHERITED"):
        # A no reached on the wrong predicate is as wrong as a yes.
        return [stray] if stray else []
    question = question or payload.get("question") or ""
    parse = payload.get("parse") or {}
    doubts: list[Doubt] = [stray] if stray else []

    evidence = payload.get("evidence") or []
    lead = evidence[0] if evidence else {}
    concept = lead.get("concept") or payload.get("concept") or ""
    predicate = plain(lead.get("object") or parse.get("target") or "")
    confidence = float(lead.get("confidence") or 0.0)
    distance = int(lead.get("distance") or 0)
    # The relation to re-ask under is the one the *evidence* was stored with,
    # not the one the question was routed by. `is a dog wild` routes as `is_a`
    # and the fact behind it is `has_property`; corroborating under the routed
    # relation asked `is a collie a wild`.
    relation = lead.get("relation") or parse.get("relation") or ""

    predicate = DENYING.sub("", predicate).strip() or predicate
    if lead and confidence and confidence < WEAK_CONFIDENCE:
        doubts.append(Doubt(
            "weak", question, concept, predicate, relation, confidence,
            distance,
            "the fact this rests on comes from "
            f"{lead.get('source') or 'the crawl'} at confidence "
            f"{confidence:.2f}, under the {WEAK_CONFIDENCE:.2f} floor"))
    if distance > 0:
        doubts.append(Doubt(
            "inherited", question, concept, predicate, relation, confidence,
            distance,
            "nothing says this of the subject itself: it is inherited from "
            f"{concept}, {distance} level(s) up"))
    if lead.get("sense_assumed"):
        doubts.append(Doubt(
            "assumed_sense", question, concept, predicate, relation,
            confidence, distance,
            "the sense of the object was assumed rather than resolved"))
    if DENYING.match(lead.get("object") or ""):
        # The fact behind the yes says the opposite of the yes. Nothing
        # downstream should be built on the negated string, so the predicate
        # is handed on in its positive form and the fan-out asks `does a carp
        # have legs`, which the store answers CONTRADICTED -- correctly.
        positive = DENYING.sub("", lead["object"]).strip()
        doubts.append(Doubt(
            "negated_evidence", question, concept, positive, relation,
            confidence, distance,
            f"the only fact behind this yes is “{concept} {relation} "
            f"{lead['object']}” — which says the opposite of what the answer "
            f"says"))
        predicate = positive

    asked_about = (parse.get("subject") or "").strip().lower()
    resolved = plain(payload.get("concept") or "")
    if asked_about and resolved and resolved.lower() != asked_about:
        # v687 reads `pig` as `pig bed.n.01`, a mould for casting pig iron,
        # and answers correctly about that. The verdict is right and the
        # subject is not the one the question named, which is the single
        # most misleading thing an answer here can do.
        doubts.append(Doubt(
            "sense_mismatch", question, concept, predicate, relation,
            confidence, distance,
            f"the answer is about {payload.get('concept')}, and the question "
            f"was about “{asked_about}”"))
    if (payload.get("corroborated") is False
            or parse.get("corroborated") is False):
        doubts.append(Doubt(
            "uncorroborated", question, concept, predicate, relation,
            confidence, distance,
            "the norms were consulted and had too few kinds to check it"))
    return doubts
