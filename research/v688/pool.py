"""A pool of v687 engines, asked in parallel.

Nineteen workers, by default, each holding its own `ReasoningEngine`. Separate
engines rather than threads over one, and that is forced rather than chosen:
`Parser._spent` and `Parser._blocked` are mutable state written *during* a
parse, so two threads sharing a parser corrupt each other's reading of a
sentence. Sharing the store is fine -- the connection is opened read-only with
`check_same_thread=False` -- but sharing the parser is not.

What it costs, measured: 916 MB and 6.1s for the first engine, then 264 MB and
3.4s for each after, linearly. Nineteen is about 5.7 GB. The page reports what
it actually got, because a reader should be able to see the price.

What it buys, honestly: not speed. One `ask` is about 20ms, and no human
notices nineteen of them overlapping. It buys *breadth per cycle* -- the loop
putting a doubted claim to a concept's parent and all its siblings at the same
time, so a disagreement surfaces on the cycle it exists on rather than never.
"""
from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_WORKERS = 19


@dataclass
class Answer:
    """One question, its v687 payload, and who answered it."""

    question: str
    payload: dict
    worker: int
    started: float
    elapsed: float
    origin: str = "seed"
    pins: dict = field(default_factory=dict)
    about: str = ""
    predicate: str = ""
    why: str = ""
    parent: str = ""
    cycle: int = 0
    error: str = ""

    @property
    def key(self) -> str:
        """The same words under a different reading are a different answer."""
        if not self.pins:
            return self.question
        held = " ".join(f"{word}={sense}" for word, sense in sorted(
            self.pins.items()))
        return f"{self.question} ⟨{held}⟩"

    @property
    def verdict(self) -> str:
        return (self.payload or {}).get("verdict") or ("ERROR" if self.error
                                                       else "")

    @property
    def relation(self) -> str:
        return ((self.payload or {}).get("parse") or {}).get("relation") or ""

    @property
    def note(self) -> str:
        return (self.payload or {}).get("note") or ""

    @property
    def steps(self) -> list:
        return (self.payload or {}).get("steps") or []

    def as_dict(self, with_payload: bool = True) -> dict:
        # Four readings and a number, beside the seventeen-word verdict. The
        # verdict stays: it is what chooses the repair. This is what a person
        # reads.
        from .confidence import of_answer
        from .content import digest
        weighed = of_answer(self.payload, self.verdict)
        record = {"question": self.question, "pins": dict(self.pins),
                  "verdict": self.verdict,
                  "outcome": weighed.outcome,
                  "confidence": round(weighed.value, 2),
                  "band": weighed.band,
                  "relation": self.relation, "note": self.note,
                  "worker": self.worker, "elapsed": round(self.elapsed, 4),
                  "started": round(self.started, 4), "origin": self.origin,
                  "about": self.about, "predicate": self.predicate,
                  "why": self.why, "parent": self.parent, "cycle": self.cycle,
                  "error": self.error}
        if with_payload:
            record["factors"] = weighed.as_dict()["factors"]
            record["steps"] = self.steps
            record["evidence"] = (self.payload or {}).get("evidence") or []
            record["concept"] = (self.payload or {}).get("concept")
            record["parse"] = (self.payload or {}).get("parse") or {}
        return record


class EnginePool:
    """`workers` v687 engines, checked out one per question."""

    def __init__(self, store: Path, workers: int = DEFAULT_WORKERS,
                 build_parallel: bool = True, on_ready=None,
                 engine_class=None) -> None:
        # `engine_class` exists for `audit.py`, which asks the same questions
        # of subclasses with parts turned off. Nothing in the served system
        # passes it, and the default is the engine the server has always
        # built.
        from research.v687.reasoning import ReasoningEngine
        engine_class = engine_class or ReasoningEngine

        self.store = Path(store)
        self.requested = int(workers)
        self.free: queue.Queue = queue.Queue()
        self.engines: list = []
        self._lock = threading.Lock()
        started = time.time()

        def build(index: int):
            engine = engine_class(self.store)
            with self._lock:
                self.engines.append(engine)
                self.free.put((index, engine))
                if on_ready:
                    on_ready(len(self.engines), self.requested)
            return engine

        # The first engine is built alone: spaCy's model load is the 640 MB
        # part and running nineteen of them at the identical moment is how a
        # machine gets pushed into swap for no gain.
        build(0)
        if self.requested > 1:
            if build_parallel:
                with ThreadPoolExecutor(max_workers=6) as pool:
                    list(pool.map(build, range(1, self.requested)))
            else:
                for index in range(1, self.requested):
                    build(index)
        self.workers = len(self.engines)
        self.build_seconds = time.time() - started

    # -- asking ------------------------------------------------------------
    def ask_one(self, question: str, pinned: dict | None = None,
                origin: str = "seed", **about) -> Answer:
        index, engine = self.free.get()
        started = time.time()
        try:
            payload = engine.ask(question, None, pinned or None)
            error = ""
        except Exception as bad:                    # noqa: BLE001
            # A worker that dies takes its question with it, not the cycle.
            # A malformed question is a normal event here: curiosity builds
            # questions out of corpus predicates and some of them are not
            # sentences.
            payload, error = {}, f"{type(bad).__name__}: {bad}"
        finally:
            self.free.put((index, engine))
        return Answer(question=question, payload=payload, worker=index,
                      started=started, elapsed=time.time() - started,
                      origin=origin, error=error, **about)

    def ask_many(self, questions, pinned: dict | None = None) -> list[Answer]:
        """Ask everything at once, one worker each, in the order given back.

        `questions` is a list of `question.Question`. The fan-out is capped at
        the pool size by the queue itself: a twentieth question waits for a
        worker rather than opening a twentieth engine.
        """
        questions = list(questions)
        if not questions:
            return []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self.ask_one, item.text,
                                   dict(item.pins) or pinned,
                                   item.origin, pins=dict(item.pins),
                                   about=item.about,
                                   predicate=item.predicate, why=item.why,
                                   parent=item.parent)
                       for item in questions]
            return [future.result() for future in futures]

    def close(self) -> None:
        for engine in self.engines:
            try:
                engine.reasoner.close()
            except Exception:                       # noqa: BLE001
                pass

    def as_dict(self) -> dict:
        return {"requested": self.requested, "workers": self.workers,
                "build_seconds": round(self.build_seconds, 2),
                "store": self.store.name}
