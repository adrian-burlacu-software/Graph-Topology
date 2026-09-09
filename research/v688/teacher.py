"""A local model that adjudicates R28 refusals, and nothing else.

R28 holds that a qualified fact does not affirm the bare claim. It is right
often enough to keep -- `fish capable_of "walk on land"` is the mudskippers,
and `rock capable_of "go for swim"` is a place people swim near -- and wrong
often enough to cost, because `leopard capable_of "hunt at night"` is a
leopard that hunts.

`AUDIT.md` §12 measured both halves of that. R28 costs a third of the store's
reachable coverage and buys three accuracy points, and no rule available here
separates the two cases: corroboration cannot clear a leaf, grammar reads
`hunt at night` and `walk on land` identically, and the only thing that does
distinguish them -- whether the class is homogeneous -- needs norms that cover
1.2% of concepts.

So this asks something that has read a great deal of English. Not to supply
facts: to judge a fact the store already holds.

    a leopard   "hunt at night"    -> a leopard hunts              yes
    a fish      "walk on land"     -> a fish walks                 no
    a rock      "go for swim"      -> a rock swims                 no
    a hose      "washing car"      -> a hose is used to wash       yes
    a person    "fly helicoptor"   -> a person flies               yes   <- wrong

Four of five, including both cases the over-affirmation audit was built from.
The fifth is the one this project's backlog already names as the hard one.

## Why this is adjudication and not teaching

It never writes a fact. Every claim it rules on is one the crawl already
made; the model decides only whether the surplus in the phrasing destroys it.
That bounds the damage a confident wrong answer can do -- the worst case is
believing a row the store already contains -- and it bounds the cost, because
the work is proportional to R28 refusals rather than to the concept space.

It also **edits nothing in v687**, which is the rule `gap.py` states and this
follows: the answer is left as it was and the adjudication travels beside it.

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

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

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

#: What a judgement must be worth before it may *write* a fact. Nothing
#: teaches yet; this is the floor the measurement above supports when
#: something does, and it is high because a wrong fact written into the store
#: is permanent and a refused one is merely absent.
TEACHING_FLOOR = 0.99

#: Adjudication has no floor, and the asymmetry is deliberate. It ratifies a
#: row the crawl already wrote, so the worst case is believing something the
#: store already contains; teaching invents. `careful` costs marginal truths
#: their confidence -- `a person runs` from "run for short distances" falls
#: from 0.79 to 0.55 -- and at 0.99 that answer would be refused, which is
#: exactly the coverage R28 already costs. The number travels with the
#: judgement instead, for `confidence.py` to price.

#: One question, and the shortest answer that settles it. The claim is spelled
#: out rather than implied because `walk on land` and `hunt at night` differ
#: only in whether the surplus changes who the subject is.
PROMPT = (
    'A knowledge base records this about {subject}: "{fact}".\n'
    'Does that support the plain claim "{claim}"?\n'
    "Answer yes or no, one word only."
)

#: The same question with nothing behind it, for the control that asks
#: whether the store earns its keep at all.
BARE = "{claim}?\nAnswer yes or no, one word only."

#: How many refusals one cycle may put to it. The loop can raise thirty in a
#: fan-out and the point is to adjudicate the answer in front of the reader,
#: not to grind through every near miss the walk turned up.
PER_CYCLE = 6


@dataclass
class Adjudication:
    """One R28 refusal, put to the teacher and ruled on."""

    question: str            # the v687 question this belongs to
    subject: str             # "a leopard"
    fact: str                # "hunt at night"
    claim: str               # "a leopard hunts"
    supports: bool
    confidence: float
    started: float
    elapsed: float
    cached: bool = False

    def as_dict(self) -> dict:
        return {"question": self.question, "subject": self.subject,
                "fact": self.fact, "claim": self.claim,
                "supports": self.supports,
                "confidence": round(self.confidence, 3),
                "started": self.started, "elapsed": round(self.elapsed, 3),
                "cached": self.cached,
                "worker": "llm", "origin": "adjudicate"}


class Teacher:
    """SmolLM3, loaded once, asked one thing.

    `available` is False when torch, transformers or the checkpoint is
    missing, and every caller treats that as "no adjudications this run"
    rather than as an error. The loop is not allowed to depend on it.
    """

    def __init__(self, model: Path | None = None, load: bool = True) -> None:
        self.path = Path(model or MODEL)
        self.lock = threading.Lock()
        self.model = None
        self.tokenizer = None
        self.error = ""
        self.cache: dict = {}
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
        try:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(self.cache, indent=1, sort_keys=True),
                             encoding="utf-8")
        except Exception:                           # noqa: BLE001
            pass

    @staticmethod
    def key(subject: str, fact: str, claim: str) -> str:
        return f"{subject}|{fact}|{claim}".lower()

    # -- the judgement -----------------------------------------------------
    def judge(self, subject: str, fact: str, claim: str) -> tuple:
        """(supports, confidence, cached). Greedy, so it is reproducible.

        The confidence is the model's own: the probability it puts on `yes`
        against `no` at the first generated token, renormalised over the two.
        A number that comes out of the model beats a constant, for the reason
        `confidence.py` prices Ascent++ by percentile and ConceptNet by fiat --
        one of them carries information and the other does not.
        """
        cached = self.cache.get(self.key(subject, fact, claim))
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
                [{"role": "system", "content": CAREFUL},
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
        self.cache[self.key(subject, fact, claim)] = {
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
    def review(self, answers, limit: int = PER_CYCLE) -> list:
        """Adjudicate the R28 refusals in a cycle's answers.

        Only answers R28 actually blocked: a question the store settled needs
        no adjudication, and one it never reached has nothing to adjudicate.
        """
        out: list = []
        for answer in answers:
            if len(out) >= limit:
                break
            for subject, fact, claim in refusals(answer):
                if len(out) >= limit:
                    break
                started = time.time()
                supports, confidence, cached = self.judge(subject, fact, claim)
                out.append(Adjudication(
                    question=answer.question, subject=subject, fact=fact,
                    claim=claim, supports=supports, confidence=confidence,
                    started=started, elapsed=time.time() - started,
                    cached=cached))
        return out

    def as_dict(self) -> dict:
        return {"available": self.available, "model": self.path.name,
                "device": getattr(self, "device", ""),
                "error": self.error,
                "cached": len(self.cache),
                "load_seconds": round(self.load_seconds, 1)}


def refusals(answer) -> list:
    """(subject, fact, claim) for every R28 skip behind one answer.

    R28 leaves two things in the payload: a `skip` step naming the rule, and
    the fact itself in `suggestions`, which is where `verify` puts what it
    held back. The suggestions are what carry the object, so they are what is
    read; the step is only how we know R28 is why.
    """
    payload = getattr(answer, "payload", None) or {}
    steps = payload.get("steps") or []
    if not any((step.get("rule") or "") == "R28" for step in steps):
        return []
    parse = payload.get("parse") or {}
    target = (parse.get("target") or "").strip()
    subject = (parse.get("subject") or "").strip()
    if not target or not subject:
        return []
    claim = (answer.question or "").strip().rstrip("?")
    out = []
    seen = set()
    for fact in (payload.get("suggestions") or [])[:3]:
        obj = (fact.get("object") or "").strip()
        if not obj or obj in seen:
            continue
        seen.add(obj)
        out.append((article(subject), obj, claim))
    return out


def article(word: str) -> str:
    """`leopard` -> `a leopard`, for a prompt that reads like English."""
    word = (word or "").strip()
    if not word:
        return word
    if word.split()[0].lower() in ("a", "an", "the", "some"):
        return word
    return ("an " if word[0].lower() in "aeiou" else "a ") + word
