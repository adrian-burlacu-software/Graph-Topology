"""The v688 page: one utterance, played cycle by cycle.

Run:  python -m research.v688 --workers 19 --port 8688

The whole run is computed server-side and handed to the page in one JSON
document, so stepping through cycles, opening a question and reading its v687
derivation are all local. A run is therefore replayable without asking
anything twice, which is the only way a page about a loop is legible.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from research.v687 import build

from .attention import Curiosity, DECAY, FLOOR, URGENCY, WEIGHTS
from .gap import WEAK_CONFIDENCE
from .loop import Loop
from .pool import DEFAULT_WORKERS, EnginePool

HERE = Path(__file__).parent

#: The examples the page ships with. Chosen by running forty candidates and
#: keeping the ones that make the machinery visible: each note says which part
#: it is there to show, and no two show the same part.
EXAMPLES = [
    # -- reasoning in a line: A -> B -> C, where more workers do not help ---
    {"text": "what is a beagle",
     "shows": "Four answers deep, in a line: beagle → hound → hunting dog → "
              "dog → canine. Each rung is inside the previous answer, so "
              "nothing after the first could have been asked in advance.",
     "expect": "content, not a verdict"},
    {"text": "what is a whale",
     "shows": "The same ladder somewhere else entirely — whale → cetacean → "
              "aquatic mammal → placental — and curiosity following it down "
              "to ask what a cetacean is like.",
     "expect": "content, not a verdict"},
    {"text": "does a cat purr",
     "shows": "Fan out, then reason in a line. The cat family splits on "
              "`active` — bobcat and siamese yes, persian no — and the "
              "question that raises is which side a cat is on.",
     "expect": "holds, but something it passed does not"},
    {"text": "what is a violin",
     "shows": "The ladder on the artifact half of the trie, and curiosity "
              "moving with it: once `stringed instrument` has been looked "
              "up, it becomes something to be curious about too.",
     "expect": "content, not a verdict"},
    {"text": "is a penguin a typical bird",
     "shows": "Four deep, because what a penguin is has to be settled before "
              "whether it is a typical one: penguin → sphenisciform seabird → "
              "seabird → aquatic bird → bird.",
     "expect": "absent, not false"},
    {"text": "is a spider an insect",
     "shows": "The seed question is unanswerable, so the loop goes after the "
              "word it could not place and finds `arthropod` on the way.",
     "expect": "absent, not false"},

    # -- breadth: what nineteen workers are actually for -------------------
    {"text": "does a beagle swim",
     "shows": "A yes that does not survive its own family. One Ascent++ fact "
              "at confidence 0.42, inherited three levels; every kind of dog "
              "the norms cover denies it, all asked in one cycle.",
     "expect": "contradicted by its own family"},
    {"text": "can a dog fall into a hole",
     "shows": "The question that broke v687's page sweep. Almost nothing "
              "about the answer is stated of dogs themselves, so the loop "
              "spends eighteen questions finding out whose claim it is.",
     "expect": "weakly held"},
    {"text": "is a dog wild",
     "shows": "Crawl noise at 0.54 answering yes. The three kinds of dog the "
              "norms cover cannot corroborate it, and nothing contradicts it "
              "either — so it is reported as a weak hold, not an objection.",
     "expect": "weakly held"},
    {"text": "a whale is a fish",
     "shows": "A false statement, put back as a question. Absent is not "
              "false: the loop chases the word it could not place rather "
              "than denying the claim it was handed.",
     "expect": "absent, not false"},
    {"text": "a dog is a kind of animal",
     "shows": "A statement checked as a claim, then put to the family. "
              "`a kind of` is dropped before asking — it is a hedge, not "
              "part of what was said.",
     "expect": "weakly held"},
    {"text": "a beagle is a dog that hunts rabbits",
     "shows": "A statement with two clauses, checked as two claims. Checking "
              "it as one would check neither.",
     "expect": "weakly held"},

    # -- the short ones, short for a reason --------------------------------
    {"text": "is a bat a bird",
     "shows": "The classic misconception. The loop looks up what a bird is "
              "rather than denying the claim, because the store records an "
              "absence and an absence is not a no.",
     "expect": "absent, not false"},
    {"text": "is a wemble a greeting",
     "shows": "A word gap. Two asks and it stops: nothing downstream of an "
              "unknown word means anything, and no third attempt is invented.",
     "expect": "unreadable"},
    {"text": "does a robin fly",
     "shows": "The control. Well recorded, well corroborated, no doubt "
              "raised — so the loop is curious for one cycle and settles.",
     "expect": "corroborated"},
]


class Service:
    """The pool, the trie index and the loop, built once and shared."""

    def __init__(self, store: Path, workers: int, max_cycles: int) -> None:
        self.ready = False
        self.progress = (0, workers)
        self.store = store
        self.max_cycles = max_cycles

        def note(done: int, total: int) -> None:
            self.progress = (done, total)

        started = time.time()
        self.pool = EnginePool(store, workers=workers, on_ready=note)
        self.curiosity = Curiosity(self.pool.engines[0].profiles.plan)
        self.loop = Loop(self.pool, self.curiosity, max_cycles=max_cycles)
        self.startup = time.time() - started
        self.ready = True
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}

    def run(self, utterance: str) -> dict:
        key = utterance.strip().lower()
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        answer = self.loop.run(utterance).as_dict()
        with self._lock:
            self._cache[key] = answer
        return answer

    def settings(self) -> dict:
        return {
            "pool": self.pool.as_dict(),
            "startup_seconds": round(self.startup, 1),
            "max_cycles": self.max_cycles,
            "trie": {"individuals": len(self.curiosity.universe),
                     "predicates": len(self.curiosity.holders)},
            "weights": WEIGHTS, "urgency": URGENCY,
            "decay": DECAY, "floor": FLOOR,
            "weak_confidence": WEAK_CONFIDENCE,
            "examples": EXAMPLES,
        }


class Handler(BaseHTTPRequestHandler):
    service: Service = None            # type: ignore[assignment]

    def log_message(self, *_args) -> None:      # quiet
        pass

    def _send(self, body: bytes, kind: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(json.dumps(payload).encode("utf-8"),
                   "application/json; charset=utf-8", code)

    def do_GET(self) -> None:                   # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        if parsed.path in ("/", "/index.html"):
            page = (HERE / "app.html").read_text(encoding="utf-8")
            self._send(page.encode("utf-8"), "text/html; charset=utf-8")
        elif parsed.path == "/api/settings":
            self._json(self.service.settings())
        elif parsed.path == "/api/run":
            utterance = (query.get("q") or [""])[0].strip()
            if not utterance:
                self._json({"error": "no utterance"}, 400)
                return
            if len(utterance) > 200:
                self._json({"error": "too long"}, 400)
                return
            try:
                self._json(self.service.run(utterance))
            except Exception as bad:            # noqa: BLE001
                self._json({"error": f"{type(bad).__name__}: {bad}"}, 500)
        else:
            self._json({"error": "not found"}, 404)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8688)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="engines in the pool; each costs about 264 MB "
                             "and 3.4s to build after the first")
    parser.add_argument("--cycles", type=int, default=4,
                        help="most internal cycles one utterance may run")
    parser.add_argument("--store", type=Path, default=build.DEFAULT_STORE)
    parser.add_argument("--warm", action="store_true",
                        help="run every example once at startup so the page "
                             "answers instantly")
    options = parser.parse_args()

    print(f"building {options.workers} engines from {options.store.name} ...")
    service = Service(options.store, options.workers, options.cycles)
    print(f"  {service.pool.workers} engines up in "
          f"{service.pool.build_seconds:.1f}s")
    if options.warm:
        for example in EXAMPLES:
            clock = time.time()
            service.run(example["text"])
            print(f"  warmed {example['text']!r} in {time.time()-clock:.1f}s")

    Handler.service = service
    httpd = ThreadingHTTPServer(("127.0.0.1", options.port), Handler)
    print(f"v688 loop on http://127.0.0.1:{options.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.pool.close()


if __name__ == "__main__":
    main()
