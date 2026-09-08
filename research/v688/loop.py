"""The cycle: attend, generate, fan out, read, update, settle.

A cycle is *not* a conversational turn. It is an internal round the system
runs on its own to resolve one thing you said, and an utterance normally takes
three to five of them. That scoping is the point:

    it needs no episodic memory, because nothing has to outlive the utterance;
    it needs no executive control, because "which question next" is a sort.

Dialogue is what you get later by not clearing the buffer between utterances.
It is not required here and building for it first is what makes the whole
thing unmanageable.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from . import attention
from .buffer import Buffer
from .pool import Answer, EnginePool
from .question import Generator, Question, article

#: An utterance that opens with one of these is a question already.
ASKING = re.compile(
    r"^\s*(is|are|was|were|do|does|did|can|could|will|would|has|have|had|"
    r"what|which|who|whose|why|how|where|when|name|list|tell)\b", re.I)

#: `X is a Y` -> `is X a Y`, and the same for the other copular openers. A
#: statement is checked by asking it, which is what makes R27 an assertion
#: gate: VERIFIED means already known, CONTRADICTED means object, UNKNOWN
#: means new and safe to store.
STATEMENT = re.compile(
    r"^\s*(?:an?\s+|the\s+)?(?P<subject>.+?)\s+"
    r"(?P<verb>is|are|was|were|can|has|have|eats|lives|contains)\s+"
    r"(?P<rest>.+?)\s*$", re.I)

#: Clauses a statement can trail, each of which is its own claim. "a beagle is
#: a dog that hunts rabbits" is two claims and checking it as one checks
#: neither.
CLAUSE = re.compile(r"\s+(?:that|which|and|but|who)\s+", re.I)

#: `a dog is a kind of animal` is a claim about animals, not about kinds.
#: Left in, it asked `is a dog a kind of animal` and got UNKNOWN.
HEDGE = re.compile(r"^(?:an?\s+)?(?:kind|type|sort|form|species)\s+of\s+",
                   re.I)


@dataclass
class Cycle:
    """One round: what was asked, by whom, and what it opened up."""

    number: int
    questions: list[Question]
    answers: list[Answer]
    gaps_found: list = field(default_factory=list)
    doubts_found: list = field(default_factory=list)
    activation: dict = field(default_factory=dict)
    elapsed: float = 0.0

    def as_dict(self) -> dict:
        return {
            "number": self.number,
            "questions": [q.as_dict() for q in self.questions],
            "answers": [a.as_dict() for a in self.answers],
            "gaps_found": [g.as_dict() for g in self.gaps_found],
            "doubts_found": [d.as_dict() for d in self.doubts_found],
            "activation": self.activation,
            "elapsed": round(self.elapsed, 3),
            "workers_used": sorted({a.worker for a in self.answers}),
        }


@dataclass
class Run:
    """One utterance, start to settled, replayable."""

    utterance: str
    cycles: list[Cycle]
    summary: dict
    buffer: dict
    pool: dict
    settled: bool
    elapsed: float

    def as_dict(self) -> dict:
        return {"utterance": self.utterance,
                "cycles": [c.as_dict() for c in self.cycles],
                "summary": self.summary, "buffer": self.buffer,
                "pool": self.pool, "settled": self.settled,
                "elapsed": round(self.elapsed, 3),
                "asked": sum(len(c.answers) for c in self.cycles)}


def seed_questions(text: str, lemmatise=None) -> list[str]:
    """What to ask first, from what was said.

    A question is asked as it stands. A statement is turned back into the
    questions that would check it -- one per clause -- because the only honest
    thing to do with something you have been told is find out whether you
    already disagree with it.

    `lemmatise` un-conjugates the verb a trailing clause opens with: "...that
    hunts rabbits" has to become "does a beagle hunt rabbits", because `does a
    beagle hunts rabbits` is not a sentence and the parser reads it as one
    noun phrase.
    """
    lemmatise = lemmatise or (lambda word: word)
    text = (text or "").strip().rstrip("?.")
    if not text:
        return []
    if ASKING.match(text):
        return [text]

    claims = [part.strip() for part in CLAUSE.split(text) if part.strip()]
    subject = ""
    asked: list[str] = []
    for index, claim in enumerate(claims):
        found = STATEMENT.match(claim)
        if found:
            if index == 0:
                subject = found.group("subject").strip()
            verb = found.group("verb").lower()
            rest = found.group("rest").strip()
            who = found.group("subject").strip() if index == 0 else subject
            if not who:
                continue
            if HEDGE.match(rest):
                # `a kind of animal` is `an animal`: dropping the hedge has to
                # give the noun back its own article, or the claim reads
                # `is a dog animal`.
                rest = HEDGE.sub("", rest).strip()
                rest = f"{article(rest)} {rest}"
            if verb in ("is", "are", "was", "were"):
                asked.append(f"is {article(who)} {who} {rest}")
            elif verb == "can":
                asked.append(f"can {article(who)} {who} {rest}")
            elif verb in ("has", "have"):
                asked.append(f"does {article(who)} {who} have {rest}")
            else:
                asked.append(f"does {article(who)} {who} {verb} {rest}")
        elif subject:
            # A trailing clause with no verb of its own inherits the subject:
            # "...that hunts rabbits" is a claim about the beagle.
            words = claim.split()
            head = lemmatise(words[0])
            rest = " ".join(words[1:])
            asked.append(
                f"does {article(subject)} {subject} {head} {rest}".strip())
    if not asked:
        asked = [f"what is {article(text)} {text}"]
    return asked[:4]


class Loop:
    """attend -> generate -> fan out -> read -> update -> repeat."""

    def __init__(self, pool: EnginePool, curiosity: attention.Curiosity,
                 max_cycles: int = 6, width: int | None = None) -> None:
        self.pool = pool
        self.curiosity = curiosity
        self.max_cycles = max_cycles
        #: how many questions a cycle may put out at once. One per worker:
        #: a twentieth question would only wait for a nineteenth to finish.
        self.width = width or pool.workers

    def lemma(self, word: str) -> str:
        """The dictionary form of one word, via the parser already loaded."""
        parser = self.pool.engines[0].parser
        if not getattr(parser, "nlp", None):
            return word[:-1] if word.endswith("s") and not word.endswith(
                "ss") else word
        read = parser.nlp(word)
        return read[0].lemma_ if len(read) else word

    def run(self, utterance: str) -> Run:
        engine = self.pool.engines[0]
        buffer = Buffer(utterance, engine, self.curiosity)
        generator = Generator(engine, self.curiosity)
        started = time.time()
        cycles: list[Cycle] = []

        # -- cycle 0: what was actually said ------------------------------
        buffer.attend(utterance)
        seeds = [Question(text, "seed", why="what you said, asked as it stands"
                          if text == utterance.strip().rstrip("?.")
                          else "the claim you made, put as a question")
                 for text in seed_questions(utterance, self.lemma)]
        for seed in seeds:
            buffer.attend(seed.text)

        pending = seeds
        for number in range(self.max_cycles):
            if not pending:
                break
            # A cycle with carried questions still has work even if nothing
            # new opened, so `settled` has to account for them.
            buffer.cycle = number
            clock = time.time()
            for question in pending:
                buffer.note_depth(question.text, question.depth)
            answers = self.pool.ask_many(pending, buffer.pins or None)
            for answer in answers:
                answer.cycle = number
            gaps, doubts = buffer.record(answers)
            # Whatever an answer turned out to be about is now live. This is
            # how attention follows the reasoning instead of only the words:
            # `does a beagle swim` was about beagles, and comes back about
            # dogs.
            for answer in answers:
                concept = (answer.payload or {}).get("concept") or ""
                if not concept:
                    continue
                buffer.activation.bump(concept.split(".")[0], 0.5, number)
                # Looking something up on purpose is how it becomes a topic.
                # `fish` is the target of `a whale is a fish` and not what the
                # sentence is about; it earns curiosity once the loop has
                # chosen to ask `what is a fish` to close a gap.
                if answer.origin in ("gap", "chain"):
                    buffer.take_topic(concept)
            cycles.append(Cycle(
                number=number, questions=pending, answers=answers,
                gaps_found=gaps, doubts_found=doubts,
                activation=buffer.activation.as_dict(),
                elapsed=time.time() - clock))

            buffer.activation.decay()
            pending = generator.queue(buffer, self.width)

        return Run(utterance=utterance, cycles=cycles,
                   summary=self.summarise(buffer, cycles),
                   buffer=buffer.as_dict(), pool=self.pool.as_dict(),
                   settled=buffer.settled(), elapsed=time.time() - started)

    # -- what it all came to ----------------------------------------------
    def summarise(self, buffer: Buffer, cycles: list[Cycle]) -> dict:
        """The answer, and every reason to hold it more loosely than it reads.

        This is the part that a single `ask` cannot produce, because a single
        ask has nothing to compare its answer against.
        """
        seed = cycles[0].answers if cycles else []
        headline = seed[0] if seed else None
        conflicts = buffer.conflicts()
        telling = buffer.needs_telling()

        lines: list[str] = []
        if headline is not None:
            lines.append(f"{headline.verdict} — {headline.question}")
        for bad in conflicts:
            names = ", ".join(question for question, _ in bad.against)
            about = ("but not corroborated"
                     if headline is not None and bad.question == headline.question
                     else f"and along the way, “{bad.question}” did not hold up")
            lines.append(f"{about}: {bad.detail} ({names})")
        # The payoff of a requirement check, stated as an argument rather
        # than left for the reader to assemble out of a conflict list.
        for answer in buffer.answers.values():
            if answer.origin != "require":
                continue
            needs = answer.predicate
            denied = [bad for bad in conflicts if bad.question == answer.question]
            if denied and headline is not None:
                lines.append(
                    f"and that is the trouble: {needs} is what the store says "
                    f"doing this needs, and every kind of {answer.about} it "
                    f"could be put to is scored and denied on it — so the yes "
                    f"rests on nothing the family will support")
            elif (headline is not None
                  and headline.verdict in ("CONTRADICTED", "DENIED")
                  and answer.verdict in ("VERIFIED", "HELD", "INHERITED")):
                lines.append(
                    f"and the no is not about anatomy: "
                    f"{article(answer.about)} {answer.about} does have "
                    f"{needs}, which is what the store says doing this needs")

        for hole in telling:
            lines.append(
                f"“{hole.blocker}” is not something this ontology has a word "
                f"for; nothing about it can be settled until it is told")

        learned = [answer for answer in buffer.answers.values()
                   if answer.origin == "curiosity"
                   and answer.verdict in ("VERIFIED", "CONTRADICTED",
                                          "DENIED", "HELD")]
        return {
            "headline": headline.as_dict(False) if headline else None,
            "verdict": headline.verdict if headline else "",
            "lines": lines,
            "conflicts": [bad.as_dict() for bad in conflicts],
            "needs_telling": [hole.as_dict() for hole in telling],
            "doubts": [doubt.as_dict() for doubt in buffer.seen_doubts],
            "asked": len(buffer.answers),
            "curiosity_settled": len(learned),
            # How far the loop got from what was actually said. A depth of 3
            # means three answers had to come back before the last question
            # could even be formed, and no amount of workers shortens that.
            "depth": max(buffer.depths.values(), default=0),
            "thread": self.thread(buffer),
            "trust": self.trust(headline, conflicts, buffer),
        }

    @staticmethod
    def thread(buffer: Buffer) -> list[dict]:
        """The longest chain of thought this utterance ran.

        Each link could only be written once the link above it came back, so
        this is the part of the work that nineteen workers cannot shorten.
        """
        deepest, best = [], -1
        for question, depth in buffer.depths.items():
            if depth <= best:
                continue
            answer = buffer.answers.get(question)
            if answer is None or answer.origin != "chain":
                continue
            walk, seen = [], set()
            while answer is not None and answer.question not in seen:
                seen.add(answer.question)
                walk.append({"question": answer.question,
                             "verdict": answer.verdict,
                             "origin": answer.origin,
                             "why": answer.why,
                             "depth": buffer.depth_of(answer.question)})
                answer = buffer.answers.get(answer.parent)
            deepest, best = list(reversed(walk)), depth
        return deepest

    @staticmethod
    def trust(headline, conflicts, buffer: Buffer) -> str:
        """One word for how far the headline should be taken.

        The point of the whole loop, compressed: v687 answers questions, and
        this says whether the answer survived contact with the rest of the
        store.
        """
        if headline is None:
            return "none"
        if headline.verdict in ("UNKNOWN_WORD", "UNPARSED", "UNSUPPORTED"):
            return "unreadable"
        # Only a conflict about *this* claim overturns it. `is a shark a fish`
        # stays VERIFIED even when `does a shark have scales` -- asked on the
        # way past -- does not survive its own family. Reporting the second as
        # though it were the first says the shark is not a fish, which is both
        # wrong and not what any part of the system concluded.
        if any(bad.question == headline.question for bad in conflicts):
            return "contradicted by its own family"
        doubted = [d for d in buffer.seen_doubts if d.question ==
                   headline.question]
        if doubted:
            return "weakly held"
        if headline.verdict in ("UNKNOWN", "UNRECORDED", "NO_MATCH"):
            return "absent, not false"
        if conflicts:
            # Something else did not hold up. The headline is untouched, and
            # the page says both things rather than one loudly.
            return "holds, but something it passed does not"
        if headline.verdict in ("CONTRADICTED", "DENIED"):
            # A no that nothing disputed. Calling this "corroborated" reads as
            # though the claim were corroborated, which is the opposite of
            # what happened.
            return "denied, unchallenged"
        if headline.verdict in ("LISTING", "DEFINED", "PROFILE", "IDENTIFIED"):
            return "content, not a verdict"
        return "corroborated"
