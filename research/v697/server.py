"""The v697 page: v690's conversation, and everything each turn did.

    python -m research.v697 --workers 19 --port 8697

Each turn is kept whole as it was made -- before v689's trim cuts v688's
run to its summary -- with every executive run, the walk, and for a request
for code everything v696 did for it (`coding.py`). The page asks for a turn
whole (`/api/turn?sid=&n=`) when its inspector is opened.

v690's page is at `/v690`, v689's at `/v689`, v688's at `/v688`, over the
same process; conversations are kept as v690 keeps them.
"""
from __future__ import annotations

import argparse
import json
import threading
import urllib.parse
from collections import OrderedDict
from http.server import ThreadingHTTPServer
from pathlib import Path

from research.v690 import server as v690
from research.v697 import page  # noqa: F401 (registers the act)
from research.v697.graph import graph_of
from research.v697.steps import steps_of

HERE = Path(__file__).resolve().parent
#: how many whole turns a conversation keeps (the archive keeps them all,
#: trimmed)
KEPT = 60
#: the longest utterance taken: a request for code is longer than a
#: sentence
LONGEST = 4000


def _plain(value):
    """A deep copy as JSON keeps it."""
    return json.loads(json.dumps(value, default=str))


class Conversations(v690.Conversations):
    """v690's conversations, each turn kept whole as well."""

    #: how a turn is told step by step: a later server's account in place
    #: of this one's (v698's adds the project)
    STEPS = staticmethod(steps_of)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.whole: dict = {}
        self.whole_lock = threading.Lock()
        self._current = threading.local()

    def _spoken(self, turn: dict) -> dict:
        whole = _plain(turn)
        turn = super()._spoken(turn)
        turn["steps"] = self.STEPS(turn, turn.get("reply"))
        whole["reply"], whole["steps"] = turn["reply"], turn["steps"]
        sid = getattr(self._current, "sid", None)
        if sid:
            with self.whole_lock:
                kept = self.whole.setdefault(sid, OrderedDict())
                kept[turn["number"]] = whole
                while len(kept) > KEPT:
                    kept.popitem(last=False)
        turn["whole"] = bool(sid)
        return turn

    def say(self, sid: str, text: str, example: bool = False) -> dict:
        self._current.sid = sid
        page.said_as_typed(text)
        try:
            return super().say(sid, text, example)
        finally:
            page.said_as_typed(None)
            self._current.sid = None

    def turn(self, sid: str, number: int) -> dict | None:
        """A turn whole, and the graph it touched (`graph.py`)."""
        with self.whole_lock:
            found = (self.whole.get(sid) or {}).get(number)
        if found is None:
            return None
        return {**found, "graph": graph_of(found)}

    def forget(self, sid: str) -> None:
        with self.whole_lock:
            self.whole.pop(sid, None)
        super().forget(sid)


class Handler(v690.Handler):
    conversations: Conversations = None     # type: ignore[assignment]

    def do_GET(self) -> None:                   # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)

        def one(name: str, limit: int = 200) -> str:
            return (query.get(name) or [""])[0].strip()[:limit]

        sid = one("sid", 64)
        if parsed.path in ("/", "/index.html"):
            self._send((HERE / "app.html").read_bytes(),
                       "text/html; charset=utf-8")
        elif parsed.path == "/v690":
            self._send((v690.HERE / "app.html").read_bytes(),
                       "text/html; charset=utf-8")
        elif parsed.path == "/api/say":
            text = (query.get("q") or [""])[0].strip()
            if not sid or not text:
                self._json({"error": "a conversation id and something said"},
                           400)
                return
            if len(text) > LONGEST:
                self._json({"error": f"too long (at most {LONGEST})"}, 400)
                return
            try:
                self._json(self.conversations.say(sid, text,
                                                  one("example") == "1"))
            except Exception as bad:            # noqa: BLE001
                self._json({"error": f"{type(bad).__name__}: {bad}"}, 500)
        elif parsed.path == "/api/turn":
            try:
                number = int(one("n", 10))
            except ValueError:
                number = -1
            found = self.conversations.turn(sid, number)
            if found is None:
                self._json({"error": "no such turn kept whole (they are "
                                     "kept while the server runs)"}, 404)
                return
            self._json(found)
        else:
            super().do_GET()


def main(handler=None, conversations=None, name: str = "v697",
         description: str = "") -> None:
    from research.v687 import build
    from research.v688 import server as v688
    from research.v688.pool import DEFAULT_WORKERS
    from research.v689.definitions import DefinitionMemory
    from research.v689.longterm import DEFINITIONS_PATH, Archive

    handler = handler or Handler
    conversations = conversations or Conversations
    parser = argparse.ArgumentParser(
        description=description or __doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8697)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--store", type=Path, default=build.DEFAULT_STORE)
    parser.add_argument("--memory", type=Path, default=v690.DEFAULT_PATH)
    parser.add_argument("--no-memory", action="store_true",
                        help="keep nothing between runs")
    parser.add_argument("--definitions", type=Path, default=DEFINITIONS_PATH)
    parser.add_argument("--silent", action="store_true",
                        help="no decoder: turns carry v689's own answer only")
    options = parser.parse_args()

    print(f"building {options.workers} engines from {options.store.name} ...")
    service = v688.Service(options.store, options.workers, options.cycles)
    print(f"  {service.pool.workers} engines up in "
          f"{service.pool.build_seconds:.1f}s")
    speaker = None
    if not options.silent:
        from research.v690.speaking import Speaker
        nlp = getattr(service.pool.engines[0].parser, "nlp", None)
        speaker = Speaker(nlp)
    handler.service = service
    archive = None if options.no_memory else Archive(options.memory)
    from research.v691.learned import DEFAULT_PATH as LEARNED_PATH
    v690.v691_page.keep(None if options.no_memory else LEARNED_PATH)
    definitions = DefinitionMemory(None if options.no_memory
                                   else options.definitions)
    handler.conversations = conversations(service, archive, definitions,
                                          speaker)
    httpd = ThreadingHTTPServer(("127.0.0.1", options.port), handler)
    print(f"{name} on http://127.0.0.1:{options.port} (v690 at /v690, v689 "
          f"at /v689, v688 at /v688)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.pool.close()


if __name__ == "__main__":
    main()
