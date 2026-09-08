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


def pins_from(values: list[str]) -> dict:
    """Read `pin=pig:pig.n.06` parameters, the way v687's own page does."""
    pinned: dict[str, str] = {}
    for value in values:
        word, _, sense = value.partition(":")
        if word.strip() and sense.strip():
            pinned[word.strip().lower()] = sense.strip()
    return pinned

#: The examples the page ships with. Chosen by running forty candidates and
#: keeping the ones that make the machinery visible: each note says which part
#: it is there to show, and no two show the same part.
EXAMPLES = [
    # -- the answer was about a different word -----------------------------
    {"text": "do pigs fly",
     "shows": "Five bugs came out of this one question. v687 read `pig` as "
              "`pig bed.n.01`, a foundry mould, because the crawl holds more "
              "rows about those — fixed in v687, which now reaches domestic "
              "swine. What is left is a yes resting on `mammal capable_of "
              "fly`, and the family check finds that is a fact about bats.",
     "expect": "contradicted by its own family"},

    {"text": "is a mouse an animal",
     "shows": "v687 says no, correctly, about `mouse.n.04` — the device. "
              "R27's exclusion is sound and it is about the wrong mouse. The "
              "loop finds that another reading answers yes, asks again under "
              "a pin, and leads with that.",
     "expect": "weakly held"},

    # -- the claim checked against what the act needs ----------------------
    {"text": "do fish run",
     "shows": "One crawled row says fish can run. So: what do things that "
              "run have? A leg — 12 of 90 of them, 102× commoner than among "
              "concepts at large. Does a fish have legs? Every kind of fish "
              "the norms cover is scored and denied. Three steps, none of "
              "which nineteen workers shorten.",
     "expect": "weakly held"},
    {"text": "can a penguin fly",
     "shows": "The same check on a denial, and the answer is the interesting "
              "one: a penguin does have the wings flying needs, so the no is "
              "not about anatomy.",
     "expect": "denied, unchallenged"},
    {"text": "can a whale fly",
     "shows": "Two answers deep and denied at every level — what a clean "
              "refutation looks like when the store actually has the facts.",
     "expect": "denied, unchallenged"},

    # -- breadth: nineteen workers earning their keep ----------------------
    {"text": "does a beagle swim",
     "shows": "A yes that does not survive its own family. One Ascent++ fact "
              "at confidence 0.42, inherited three levels; every kind of dog "
              "the norms cover denies it, all asked in one cycle.",
     "expect": "contradicted by its own family"},
    {"text": "does a cat purr",
     "shows": "Fan out, then reason in a line. The cat family splits on "
              "`active` — bobcat and siamese yes, persian no — and the "
              "question that raises is which side a cat is on.",
     "expect": "holds, but something it passed does not"},
    {"text": "can a dog fall into a hole",
     "shows": "The question that broke v687's page sweep. Almost nothing "
              "about the answer is stated of dogs themselves, so eighteen "
              "questions go looking for whose claim it actually is.",
     "expect": "weakly held"},
    {"text": "is a dog wild",
     "shows": "Crawl noise at 0.54 answering yes. The kinds of dog the norms "
              "cover cannot corroborate it and nothing contradicts it, so it "
              "is reported as a weak hold rather than an objection.",
     "expect": "weakly held"},

    # -- statements, checked rather than believed --------------------------
    {"text": "a whale is a fish",
     "shows": "A false statement, put back as a question. Absent is not "
              "false: the loop goes after the word it could not place rather "
              "than denying what it was handed.",
     "expect": "absent, not false"},
    {"text": "a dog is a kind of animal",
     "shows": "`a kind of` is dropped before asking — a hedge, not part of "
              "the claim — and what is left is put to the family.",
     "expect": "weakly held"},
    {"text": "a beagle is a dog that hunts rabbits",
     "shows": "Two clauses, checked as two claims. Checking it as one would "
              "check neither.",
     "expect": "weakly held"},

    # -- and the short ones ------------------------------------------------
    {"text": "is a violin made of wood",
     "shows": "v687 says no. It reached that through `made`, which matched "
              "the stored predicate `can be made of ivory` — a fact about "
              "ivory, not about wood. The loop says which predicate the "
              "verdict actually rests on.",
     "expect": "reached on a different predicate"},
    {"text": "is a spider an insect",
     "shows": "Absent, not false — and the loop says which of the two it "
              "found rather than guessing between them.",
     "expect": "absent, not false"},
    {"text": "what is a beagle",
     "shows": "The one definition ladder kept as an example: beagle → hound "
              "→ hunting dog → dog → canine, four answers deep. It only runs "
              "when a definition is what you asked for.",
     "expect": "content, not a verdict"},
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
        self._senses: dict[tuple[str, str], list] = {}
        self._senses_lock = threading.Lock()

    def run(self, utterance: str, pinned: dict | None = None) -> dict:
        key = utterance.strip().lower() + "|" + repr(sorted(
            (pinned or {}).items()))
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        answer = self.loop.run(utterance, pinned).as_dict()
        with self._lock:
            self._cache[key] = answer
        return answer

    def graph_size(self) -> dict:
        """What the store actually holds, as against the slice of it the
        feature norms cover.

        The header used to read `trie 541 individuals`, which reads as though
        the whole semantic memory were 541 things. It is the XCSLB and AwA2
        norms -- 1.2% of the concepts the store has facts for.
        """
        read = self.pool.engines[0].reasoner.connection.execute
        return {
            "facts": read("SELECT COUNT(*) FROM facts").fetchone()[0],
            "subjects": read(
                "SELECT COUNT(DISTINCT concept) FROM facts").fetchone()[0],
        }

    def senses(self, word: str, used_as: str = "") -> list:
        """Every reading of a word, cached and served one at a time.

        The page asks for these while a run is being stepped through, and it
        asks for several words at once. `Reasoner` holds a single sqlite
        connection -- opened read-only and shared, which is fine until a
        dozen handler threads query it together, at which point the handler
        dies and the page shows a word with no senses at all.

        A lock and a cache fix it outright: the answer never changes for a
        given word, so it is computed once and handed out thereafter.
        """
        key = (word, used_as)
        with self._senses_lock:
            if key not in self._senses:
                reasoner = self.pool.engines[0].reasoner
                self._senses[key] = (
                    reasoner.senses_of(word, used_as or None) or [])[:12]
            return self._senses[key]

    def settings(self) -> dict:
        return {
            "pool": self.pool.as_dict(),
            "startup_seconds": round(self.startup, 1),
            "max_cycles": self.max_cycles,
            "trie": {"individuals": len(self.curiosity.universe),
                     "predicates": len(self.curiosity.holders)},
            "graph": self.graph_size(),
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
        elif parsed.path == "/api/senses":
            word = (query.get("w") or [""])[0].strip().lower()
            if not word or len(word) > 40:
                self._json({"error": "no word"}, 400)
                return
            # The tagger already knows whether the word was used as a noun or
            # a verb. Offering `fly.n.05`, a fisherman's lure, first for a
            # word tagged VERB is throwing that away.
            used_as = (query.get("as") or [""])[0].strip().lower()[:1]
            # `senses_of` already carries the definition, the part of
            # speech, how many facts the store holds about each sense and
            # which one v687 took. That is what v687's own sense card shows,
            # and a picker without the definitions is a list of numbers.
            offered = self.service.senses(word, used_as)
            self._json({"word": word, "senses": [
                {"id": sense["id"],
                 "definition": sense.get("definition") or "",
                 "pos": sense.get("pos") or "",
                 "facts": sense.get("fact_count") or 0,
                 "rank": sense.get("rank"),
                 # What v687 will actually use if nobody pins anything: it
                 # takes the first sense this ordering returns. That is a
                 # different thing from `default`, which is the row the
                 # build's evidence marked primary -- the two part company
                 # whenever the ordering demotes the primary, as it does for
                 # a multi-word concept.
                 "picks": index == 0,
                 # `chosen` is the store's default reading of the word, not
                 # a choice made for this question: v687 resolves the subject
                 # and matches everything else as a string.
                 "default": bool(sense.get("chosen")),
                 "named_after_the_word":
                     sense["id"].split(".")[0].replace("_", " ") == word}
                for index, sense in enumerate(offered)]})
        elif parsed.path == "/api/run":
            utterance = (query.get("q") or [""])[0].strip()
            if not utterance:
                self._json({"error": "no utterance"}, 400)
                return
            if len(utterance) > 200:
                self._json({"error": "too long"}, 400)
                return
            pinned = pins_from(query.get("pin") or [])
            try:
                self._json(self.service.run(utterance, pinned))
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
    parser.add_argument("--cycles", type=int, default=8,
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
