"""The v689 page: a conversation, who each phrase meant, and what it stored.

    python -m research.v689 --workers 19 --port 8689 --teacher

Conversations and what they taught are kept in `state/v689-memory.sqlite`
(`longterm.py`); `--no-memory` keeps nothing between runs.

One process serves both layers: the conversation at `/`, and v688's page,
unchanged, at `/v688`. v689 does not run beside v688, it runs *as* it, with
v688's `Service` underneath -- two processes cannot hold the store at once.
"""
from __future__ import annotations

import argparse
import urllib.parse
from http.server import ThreadingHTTPServer
from pathlib import Path

from research.v687 import build
from research.v688 import server as v688
from research.v688.pool import DEFAULT_WORKERS

from .asker import Asker
from .longterm import DEFAULT_PATH, Archive, Keeper

HERE = Path(__file__).resolve().parent

#: What a v689 turn carries of v688's run. The whole run is every cycle's
#: every answer, hundreds of kilobytes, and the page shows the summary; the
#: full derivation is one click away on v688's own page.
SUMMARY_KEYS = ("verdict", "as_asked", "outcome", "confidence", "band",
                "trust", "lines", "asked")

EXAMPLES = [
    {"title": "an exception",
     "lines": ["there is a beagle", "can it swim", "it can't swim",
               "can it swim", "i have another beagle", "can it swim",
               "can the first one swim"],
     "shows": "Nothing told, and v687's walk from the beagle passes up to "
              "dog. Told `it can't swim`, the same walk stops at the "
              "individual: R3, a negation at distance 0 blocks what beagles "
              "do. The second beagle is placed under the same kind and still "
              "swims."},
    {"title": "a flying pig",
     "lines": ["there was a pig", "he was flying", "can the pig fly",
               "it was in an airplane", "can the pig fly",
               "the airplane couldn't fly", "can the pig fly"],
     "shows": "Doing shows ability: `was flying` is stored as `capable_of "
              "fly`, and R4 answers from it. `in an airplane` puts one "
              "airplane on the table, and E2 asks it: airplanes fly, so the "
              "flying was the airplane's and the pig's is withdrawn. Told "
              "`the airplane couldn't fly`, E2 is recomputed from that "
              "airplane, and the pig was flying after all."},
    {"title": "who did what",
     "lines": ["there is a dog", "there is a cat", "the dog chased it",
               "there is another cat", "did the dog chase the first cat",
               "did the dog chase the second cat"],
     "shows": "`it`, as the object of `the dog chased it`, cannot be the dog, "
              "so it is the cat. What is stored is `capable_of “chase a "
              "cat”`, which v687's rules can read, with which cat kept "
              "beside it -- so nothing was said of the second cat."},
    {"title": "not doing",
     "lines": ["there is a pig", "it wasn't flying", "does it fly",
               "can it fly"],
     "shows": "Not doing is not inability: `wasn't flying` is `did_not`, "
              "which answers `does it fly` and which no rule reads, so "
              "`can it fly` is a question about pigs."},
    {"title": "teaching",
     "lines": ["a wemble is a kind of animal", "wembles can fly",
               "can a wemble breathe", "there is a wemble", "can it fly",
               "beagles can't swim", "there is a beagle", "can it swim"],
     "shows": "Taxonomy and norms go into episodic memory, never the store. "
              "`wemble` is a kind the store never had, placed under animal, "
              "so R1 walks from a wemble into what the store knows of "
              "animals. `beagles can't swim` sits on beagle.n.01 itself, and "
              "R3 finds it one level up from every beagle, before dog's "
              "row."},
    {"title": "which one?",
     "lines": ["there is a beagle", "there is another beagle",
               "does the beagle bark", "the first beagle is black",
               "is the other one black", "is the black one fast"],
     "shows": "`the beagle` walks the trie for `is_a beagle`, finds two, and "
              "asks. `the black one` walks it for `has_property black` and "
              "finds one. And E1: a quality is never inherited from the kind."},
    {"title": "kinds",
     "lines": ["there is a dog", "does the animal bark", "it is a beagle",
               "what is it", "is it a dog"],
     "shows": "`the animal` finds the dog because the dog is stored with "
              "every kind above it. `it is a beagle` moves it down the "
              "taxonomy, and `is it a dog` is R1 walking up from the "
              "individual."},
    {"title": "you",
     "lines": ["my name is Adrian", "i have a beagle", "its name is Rex",
               "i can't swim", "can Rex swim", "can i swim",
               "does my beagle bark", "what is my name"],
     "shows": "You are an individual too, placed under person, and never "
              "`it`. A name is told like anything else -- never looked up, so "
              "WordNet's physiologist called Adrian stays out of it."},
    {"title": "nothing to refer to",
     "lines": ["can it swim", "the cat is black", "does it purr"],
     "shows": "`it` with nothing before it is refused. `the cat`, said "
              "first, is taken to introduce a cat."},
]


class StoreAsker(Asker):
    """v687 from the pool's first engine, v688 from the service's loop.

    Used only while `Service._engines` is held -- `Conversations.say` takes
    it -- because the parser and `engines[0]` share the one sqlite connection
    that lock exists for. That is also why `run` goes to the loop directly
    rather than through `Service.run`, which takes the same lock.
    """

    def __init__(self, service) -> None:
        engine = service.pool.engines[0]
        super().__init__(engine.reasoner, engine.parser, engine.match)
        self.service = service

    def lemma(self, word: str) -> str:
        return self.service.loop.lemma(word)

    def run(self, question: str) -> dict:
        # The same cache key `Service.run` uses, so a question asked on
        # either page is answered once.
        key = question.strip().lower() + "|" + repr([])
        with self.service._lock:
            found = self.service._cache.get(key)
        if found is None:
            found = self.service.loop.run(question).as_dict()
            with self.service._lock:
                self.service._cache[key] = found
        return found


def trimmed(turn: dict) -> dict:
    """A turn as the page gets it and the archive keeps it: v688's run cut
    down to its summary."""
    summary = (turn.get("run") or {}).get("summary") or {}
    turn["run"] = ({key: summary.get(key) for key in SUMMARY_KEYS}
                   if turn.get("run") else None)
    return turn


class Conversations:
    """The page's conversations, kept by `longterm.Keeper`.

    Every call takes `Service._engines`: a turn reasons over the store's one
    sqlite connection, and so does building a conversation back from disk.
    """

    def __init__(self, service, archive: Archive | None = None) -> None:
        self.service = service
        with service._engines:
            self.keeper = Keeper(StoreAsker(service), archive)

    def say(self, sid: str, text: str, example: bool = False) -> dict:
        with self.service._engines:
            return self.keeper.say(sid, text, example, trim=trimmed)

    def history(self, sid: str) -> dict:
        with self.service._engines:
            return self.keeper.history(sid)

    def forget(self, sid: str) -> None:
        with self.service._engines:
            self.keeper.forget(sid)

    def unlearn(self) -> dict:
        with self.service._engines:
            self.keeper.unlearn()
            return self.keeper.summary()


class Handler(v688.Handler):
    """v688's handler, with the conversation in front of it."""

    conversations: Conversations = None     # type: ignore[assignment]

    def do_GET(self) -> None:                   # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        sid = (query.get("sid") or [""])[0].strip()[:64]
        if parsed.path in ("/", "/index.html"):
            page = (HERE / "app.html").read_text(encoding="utf-8")
            self._send(page.encode("utf-8"), "text/html; charset=utf-8")
        elif parsed.path in ("/v688", "/v688/"):
            page = (v688.HERE / "app.html").read_text(encoding="utf-8")
            self._send(page.encode("utf-8"), "text/html; charset=utf-8")
        elif parsed.path == "/api/examples":
            self._json({"examples": EXAMPLES})
        elif parsed.path == "/api/say":
            text = (query.get("q") or [""])[0].strip()
            if not sid or not text:
                self._json({"error": "a conversation id and something said"},
                           400)
                return
            if len(text) > 200:
                self._json({"error": "too long"}, 400)
                return
            try:
                example = (query.get("example") or [""])[0] == "1"
                self._json(self.conversations.say(sid, text, example))
            except Exception as bad:            # noqa: BLE001
                self._json({"error": f"{type(bad).__name__}: {bad}"}, 500)
        elif parsed.path == "/api/forget":
            self.conversations.forget(sid)
            self._json({"forgotten": True})
        elif parsed.path == "/api/history":
            if not sid:
                self._json({"error": "a conversation id"}, 400)
                return
            self._json(self.conversations.history(sid))
        elif parsed.path == "/api/unlearn":
            # A GET, like the rest of this API, so it asks to be meant: a
            # link preview or a prefetch must not empty long-term memory.
            if (query.get("confirm") or [""])[0] != "yes":
                self._json({"error": "unlearn needs confirm=yes"}, 400)
                return
            self._json({"knowledge": self.conversations.unlearn()})
        else:
            super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8689)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--store", type=Path, default=build.DEFAULT_STORE)
    parser.add_argument("--teacher", action="store_true",
                        help="load v688's teacher; about 6.2 GB of VRAM")
    parser.add_argument("--memory", type=Path, default=DEFAULT_PATH,
                        help="where conversations and what they taught are "
                             "kept between runs")
    parser.add_argument("--no-memory", action="store_true",
                        help="keep nothing between runs")
    options = parser.parse_args()

    print(f"building {options.workers} engines from {options.store.name} ...")
    service = v688.Service(options.store, options.workers, options.cycles,
                           teacher=options.teacher)
    print(f"  {service.pool.workers} engines up in "
          f"{service.pool.build_seconds:.1f}s")
    Handler.service = service
    archive = None if options.no_memory else Archive(options.memory)
    Handler.conversations = Conversations(service, archive)
    if archive is not None:
        known = Handler.conversations.keeper.summary()
        print(f"  long-term memory {archive.path}: {known['kinds']} taught "
              f"kind(s), {known['edges']} edge(s), {known['norms']} norm(s)")
    httpd = ThreadingHTTPServer(("127.0.0.1", options.port), Handler)
    print(f"v689 conversation on http://127.0.0.1:{options.port} "
          f"(v688 at /v688)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.pool.close()


if __name__ == "__main__":
    main()
