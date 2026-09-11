"""A local model, asked the questions the store could not settle.

v687 answers by walking over what was recorded, and mostly nothing was:
`does a pig have wings` comes back UNKNOWN because no row settles it either
way. `AUDIT.md` §14 measured SmolLM3-3B on exactly that shape -- the question,
bare, no graph behind it -- and under `CAREFUL` at 0.99 it asserts 1.0% of
claims built to be false, the store's own rate, while reaching 41.5% of true
claims against the crawl's 17.2%. So when an answer comes back unsettled, the
question as asked is put to it.

## Why not adjudicate R28's facts, as this first did

The first version put each qualified fact R28 held back to the model and
asked whether it supports the claim. That suits `leopard capable_of "hunt at
night"`, and it is the wrong question nearly everywhere else, because R28's
facts are mostly crawled sentences filed several levels up. `does a pig have
wings` put `animal has_a "leathery bat-like wings"` -- a sentence about bats
-- and asked whether it supports pig wings. The answer is always no, it says
nothing about pigs, and the question the reader asked was never put at all.
Adjudication had also been measured on five examples and found unstable to
surface phrasing; the bare question is what §14 measured on hundreds.

## What a judgement may do

Adjudication was safe because the worst case was believing a row the store
already held. A direct answer has no row behind it, so the bound it lost is
put back as `SETTLING_FLOOR`:

* an answer **at the floor settles** an unsettled headline, yes or no,
  priced by `confidence.RATIFIED` as a model's word rather than a record;
* an answer **below it** is reported and settles nothing.

The no was not counted at first, for fear of `CAREFUL`'s lean: it tells the
model most claims are false. `AUDIT.md` §26 measured that fear, and it does
not survive the floor. At 0.99 the confident no is about as sound as the
confident yes; a prompt leaning the other way (`CREDULOUS`) agrees with every
one of them; and asking the question negated does not work at all, because
the model answers the negation no as well -- for 55% of AwA2's false claims.
The lean lives below the floor.

And the store outranks it: a headline the loop has overturned is not settled
by a model's word either way.

It **edits nothing in v687**, which is the rule `gap.py` states and this
follows: the answer is left as it was and the judgement travels beside it.

## Why one process

The model is 6.2 GB in float16 on one CUDA device. `EnginePool` runs nineteen
v687 engines because they are cheap and independent; there is one GPU, so the
teacher is a serial stage in a loop that already has four of them (`sense`,
`require`, `split`, `chain` all reason in a line). `judge` takes a lock, and
the page draws it as its own lane for that reason.

## Determinism

Greedy decoding, and every judgement is cached by (concept, fact, claim) in
`llm/adjudications.json`. A replayed run is the same run, which is the whole
premise of the page, and a warm cache costs nothing.
"""
from __future__ import annotations

import contextlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .gap import ABOUT_COVERAGE

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL = REPOSITORY_ROOT / "llm" / "SmolLM3-3B"
CACHE = REPOSITORY_ROOT / "llm" / "adjudications.json"

#: The system prompt, chosen by measurement rather than by design. Against
#: 150 true claims and 150 built by corruption (`does an arm have a bubble
#: tube`), asked four ways:
#:
#:     strategy      floor   says yes to TRUE   to FALSE   separation
#:     plain         0.99          98.9%           7.7%      +91.2%
#:     careful       0.99          96.0%           0.0%      +96.0%
#:     unsure        0.99         100.0%          10.7%      +89.3%
#:     challenged    0.90         100.0%          38.5%      +61.5%
#:
#: Three points of recall for essentially all of the false-assertion rate.
#:
#: Two things that did *not* work, recorded so they are not tried again.
#: Offering `unsure` as an answer does nothing -- the model puts no mass on
#: it even when invited, so abstention has to be imposed from outside with a
#: threshold. And asking "are you certain?" in a second turn makes it much
#: worse: it revises 1.3% of the time and capitulates the rest, so false
#: acceptance goes from 7.7% to 38.5%. Taking the lower of the two
#: confidences does not rescue it.
#:
#: A prompt written specially for adjudication -- "extra detail does not
#: defeat the claim" -- scored 5/6 against this one's 6/6, accepting `a fish
#: walks` at 0.88. The general instruction to be sceptical beats the specific
#: instruction to be lenient.
CAREFUL = (
    "You judge whether a claim is true of a kind of thing in general. "
    "Most claims put to you are false. Say yes only if the property is "
    "typical of that kind; if it is merely possible, unusual, or you are "
    "not sure, say no."
)

#: The second system prompt, for claims about what a thing **can do** rather
#: than what it is like.
#:
#: `CAREFUL` says *say yes only if the property is typical*, and that is what
#: `AUDIT.md` §21 ran into: all seventeen canine witnesses denied `can a
#: beagle fall into a hole`, because falling into holes is not characteristic
#: of beagles even though every one of them can. R19 then refused `can a dog
#: fall into a hole`, which is true.
#:
#: The scepticism has to survive the change or this is just a looser prompt:
#: `a fish walks`, `a rock swims` and `a pig flies` must still be no. So the
#: test offered is possibility **for a typical member**, which keeps
#: mudskippers from making fish walkers, rather than typicality of the act.
CAPABLE = (
    "You judge whether a kind of thing is able to do something, or can have "
    "something done to it. Say yes if it is possible for an ordinary member "
    "of that kind, even where it is unusual or incidental. Say no if it is "
    "impossible, or only possible for a rare exception."
)

#: The mirror of `CAREFUL`, for measurement only (`negation.py`). A no from a
#: model told most claims are true is a no that survived the lean. §26 found
#: it agrees with every confident `CAREFUL` no, so the loop does not ask it.
CREDULOUS = (
    "You judge whether a claim is true of a kind of thing in general. "
    "Most claims put to you are true. Say no only if the claim is false of "
    "that kind; if it holds for ordinary members, even if it is "
    "unremarkable, or you are not sure, say yes."
)

#: Prompt by name. The empty name is `CAREFUL` and must stay that way: it is
#: what every cached judgement was answered under.
STYLES = {"": CAREFUL, "careful": CAREFUL, "capable": CAPABLE,
          "credulous": CREDULOUS}

#: What a judgement must be worth before it may *write* a fact. Nothing
#: teaches yet; this is the floor the measurement above supports when
#: something does, and it is high because a wrong fact written into the store
#: is permanent and a refused one is merely absent.
TEACHING_FLOOR = 0.99

#: What an answer must be worth, yes or no, before it settles a question the
#: store left open.
#:
#: The same number as teaching, for the same reason: nothing recorded stands
#: behind the answer, so a wrong yes is the model inventing and not the store
#: being believed. §14 measures this floor against claims built to be false:
#: 10.2% asserted with none, 3.3% at 0.95, 1.0% at 0.99. Adjudication ran
#: without one because it only ever ratified a row the crawl had written.
SETTLING_FLOOR = 0.99

#: One question, and the shortest answer that settles it. The claim is spelled
#: out rather than implied because `walk on land` and `hunt at night` differ
#: only in whether the surplus changes who the subject is.
PROMPT = (
    'A knowledge base records this about {subject}: "{fact}".\n'
    'Does that support answering yes to: "{claim}?"\n'
    "Answer yes or no, one word only."
)

#: The same question with nothing behind it, for the control that asks
#: whether the store earns its keep at all.
BARE = "{claim}?\nAnswer yes or no, one word only."

#: How many questions one cycle may put to it. The loop can raise thirty in a
#: fan-out and the point is the answer in front of the reader, not every
#: corroboration question the walk turned up.
PER_CYCLE = 6


@dataclass
class Judgement:
    """One unsettled question, put to the teacher as it was asked."""

    question: str            # "does a pig have wings", exactly as v687 had it
    supports: bool
    confidence: float
    started: float
    elapsed: float
    cached: bool = False

    @property
    def settles(self) -> bool:
        """An answer at the floor, either way. `AUDIT.md` §26."""
        return self.confidence >= SETTLING_FLOOR

    def as_dict(self) -> dict:
        return {"question": self.question,
                "supports": self.supports,
                "confidence": round(self.confidence, 4),
                "settles": self.settles,
                "started": self.started, "elapsed": round(self.elapsed, 3),
                "cached": self.cached,
                "worker": "llm", "origin": "teacher"}


class Teacher:
    """SmolLM3, loaded once, asked one thing.

    `available` is False when torch, transformers or the checkpoint is
    missing, and every caller treats that as "no judgements this run"
    rather than as an error. The loop is not allowed to depend on it.
    """

    def __init__(self, model: Path | None = None, load: bool = True) -> None:
        self.path = Path(model or MODEL)
        self.lock = threading.Lock()
        self.model = None
        self.tokenizer = None
        self.error = ""
        self.cache: dict = {}
        self._holding = 0
        self._dirty = False
        self.load_seconds = 0.0
        self._read_cache()
        if load:
            self._load()

    # -- the model ---------------------------------------------------------
    def _load(self) -> None:
        started = time.time()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as bad:                    # noqa: BLE001
            self.error = f"{type(bad).__name__}: {bad}"
            return
        if not self.path.exists():
            self.error = f"no checkpoint at {self.path}"
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.path))
            where = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.path),
                dtype=torch.float16 if where == "cuda" else torch.float32,
                device_map=where)
            self.model.eval()
            self.device = where
        except Exception as bad:                    # noqa: BLE001
            self.error = f"{type(bad).__name__}: {bad}"
            self.model = None
        self.load_seconds = time.time() - started

    @property
    def available(self) -> bool:
        return self.model is not None

    # -- the cache ---------------------------------------------------------
    def _read_cache(self) -> None:
        try:
            self.cache = json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:                           # noqa: BLE001
            self.cache = {}

    def _write_cache(self) -> None:
        """Save, unless a `batch` is holding the file open."""
        if self._holding:
            self._dirty = True
            return
        self._save()

    def _save(self) -> None:
        try:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(self.cache, indent=1, sort_keys=True),
                             encoding="utf-8")
        except Exception:                           # noqa: BLE001
            pass

    @contextlib.contextmanager
    def batch(self):
        """Hold the cache open across many judgements, then write once.

        `judge` saves after every answer, which is right for the loop -- a
        run interrupted mid-cycle keeps what it learned. `screen.py` asks
        49,036 questions in one pass, and saving a 2.5 MB file that many
        times costs more than the model does. Nesting is allowed and only
        the outermost writer writes.
        """
        self._holding += 1
        try:
            yield self
        finally:
            self._holding -= 1
            if not self._holding and self._dirty:
                self._dirty = False
                self._save()

    @staticmethod
    def key(subject: str, fact: str, claim: str, style: str = "") -> str:
        """The cache key, namespaced by system prompt.

        `careful` gets the bare key it has always had, so the 99,000
        judgements already on disk stay valid; any other prompt is answering
        a different question and is filed under its own name.
        """
        stem = f"{subject}|{fact}|{claim}".lower()
        return f"{style}::{stem}" if style else stem

    # -- the judgement -----------------------------------------------------
    def judge(self, subject: str, fact: str, claim: str,
              style: str = "") -> tuple:
        """(supports, confidence, cached). Greedy, so it is reproducible.

        The confidence is the model's own: the probability it puts on `yes`
        against `no` at the first generated token, renormalised over the two.
        A number that comes out of the model beats a constant, for the reason
        `confidence.py` prices Ascent++ by percentile and ConceptNet by fiat --
        one of them carries information and the other does not.
        """
        cached = self.cache.get(self.key(subject, fact, claim, style))
        if cached is not None:
            return bool(cached["supports"]), float(cached["confidence"]), True
        if not self.available:
            return False, 0.0, False

        import torch

        # With no fact to weigh, the claim is put on its own. That is the
        # `llm` control in `audit.py`: the same question, no graph.
        text = (BARE.format(claim=claim) if not fact
                else PROMPT.format(subject=subject, fact=fact, claim=claim))
        with self.lock:
            enc = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": STYLES.get(style, CAREFUL)},
                 {"role": "user", "content": text}],
                add_generation_prompt=True, return_tensors="pt",
                return_dict=True, enable_thinking=False)
            enc = {name: value.to(self.device) for name, value in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
            logits = out.logits[0, -1]
            yes = self._token_score(logits, ("Yes", " Yes", "yes", " yes"))
            no = self._token_score(logits, ("No", " No", "no", " no"))
        total = yes + no
        supports = yes >= no
        confidence = float(max(yes, no) / total) if total else 0.0
        self.cache[self.key(subject, fact, claim, style)] = {
            "supports": supports, "confidence": confidence}
        self._write_cache()
        return supports, confidence, False

    def _token_score(self, logits, spellings) -> float:
        """The mass the model puts on a word, however it is spelled."""
        import torch

        probabilities = torch.softmax(logits.float(), dim=-1)
        total = 0.0
        for spelling in spellings:
            ids = self.tokenizer.encode(spelling, add_special_tokens=False)
            if len(ids) == 1:
                total += float(probabilities[ids[0]])
        return total

    # -- what the loop calls ----------------------------------------------
    def review(self, answers, limit: int = PER_CYCLE,
               done: set | None = None) -> list:
        """Put the questions a cycle left unsettled to the model, as asked.

        In the order they were asked, so the headline -- the first question
        of the first cycle -- is never crowded out by corroboration. `done`
        is what earlier cycles of the same run already put: a question
        re-asked under a pin reads identically to the model.
        """
        done = set() if done is None else done
        out: list = []
        for answer in answers:
            if len(out) >= limit:
                break
            if not unsettled(answer) or answer.question in done:
                continue
            done.add(answer.question)
            started = time.time()
            supports, confidence, cached = self.judge(
                "", "", answer.question.strip().rstrip("?"))
            out.append(Judgement(
                question=answer.question, supports=supports,
                confidence=confidence, started=started,
                elapsed=time.time() - started, cached=cached))
        return out

    def as_dict(self) -> dict:
        return {"available": self.available, "model": self.path.name,
                "device": getattr(self, "device", ""),
                "error": self.error,
                "cached": len(self.cache),
                "load_seconds": round(self.load_seconds, 1)}


def unsettled(answer) -> bool:
    """Is this a yes/no question the store left open?

    `gap.ABOUT_COVERAGE` -- UNKNOWN, UNRECORDED, NO_MATCH -- is what absent
    rather than false looks like, and only those are worth a model's word: a
    VERIFIED or CONTRADICTED answer already has a reason behind it, and one
    v687 could not read has no question in it to put. Polar only, because
    `what is a fish` has no yes to give.
    """
    parse = (getattr(answer, "payload", None) or {}).get("parse") or {}
    return (answer.verdict in ABOUT_COVERAGE and bool(parse.get("polar"))
            and bool((answer.question or "").strip()))


#: A relation as the verb a claim would use -- **for the reader, not for the
#: model.**
#:
#: The prompt gets the bare object, because that is what measures best, and
#: the difference is not small. The same three facts behind `can a person
#: run`, judged three ways:
#:
#:     fact as written                        short distances  to safety  shop
#:     run for short distances                 supports 0.86    sup 0.88  ref 0.51
#:     capable of run for short distances      refuses  0.77    sup 0.94  ref 0.59
#:     can run for short distances             refuses  0.96    sup 0.94  sup 0.93
#:
#: Bare is 3/3; the tidier English is 1/3, and the last row cheerfully
#: supports `running shop`. **Adjudication is unstable to surface phrasing**,
#: which is a real limit of asking a model to judge, and the answer is to use
#: the wording with evidence behind it rather than the one that reads best.
#:
#: So the adjudication prompt was given the bare object, and this was for the
#: page. The teacher no longer judges facts at all; what reads this now is
#: `densify` and `kinds`, which store predicates in this form and phrase each
#: into a whole question before any model sees it.
READS = {
    "capable_of": "can {}",
    "has_a": "has {}",
    "has_part": "has {}",
    "part_of": "has {}",
    "has_property": "is {}",
    "made_of": "is made of {}",
    "used_for": "is used for {}",
    "at_location": "is found in {}",
    "is_a": "is a kind of {}",
    "receives_action": "can be {}",
    "causes": "causes {}",
    "entails": "involves {}",
    "similar_to": "is similar to {}",
}


def stated(relation: str, obj: str) -> str:
    """A fact as something readable: `has_a` + `wing` -> `has a wing`."""
    obj = (obj or "").strip()
    frame = READS.get((relation or "").strip())
    if not frame:
        return obj
    return frame.format(obj)


def article(word: str) -> str:
    """`leopard` -> `a leopard`, for a prompt that reads like English."""
    word = (word or "").strip()
    if not word:
        return word
    if word.split()[0].lower() in ("a", "an", "the", "some"):
        return word
    return ("an " if word[0].lower() in "aeiou" else "a ") + word
