"""One command: build if needed, serve the reasoner, open the browser.

    python -m research.v684

Stdlib only. The page is served from disk so it can be edited and reloaded
without restarting, and the reasoner is held open read-only across requests.
"""
from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import build
from .language import Parser
from .reason import Reasoner

HERE = Path(__file__).resolve().parent
PAGE = HERE / "app.html"


class Engine:
    """Parser plus reasoner, shared by every request."""

    def __init__(self, store: Path):
        self.reasoner = Reasoner(store)
        self.parser = Parser()
        self.match = self.parser.matcher()

    def ask(self, question: str, concept: str | None = None) -> dict:
        parse = self.parser.parse(question)
        if not parse.subject and not concept:
            return {"verdict": "UNPARSED", "question": question,
                    "parse": parse.as_dict(), "senses": [], "steps": [],
                    "evidence": [], "chain": [],
                    "note": "No noun found to reason about."}

        # Senses always come from the word in the question. `concept` selects
        # among them; it is a synset id, not something to look up as a lemma.
        senses = self.reasoner.senses_of(parse.subject or "")
        if not senses and not concept:
            return {"verdict": "UNKNOWN_WORD", "question": question,
                    "parse": parse.as_dict(), "senses": [], "steps": [],
                    "evidence": [], "chain": [],
                    "note": f"“{parse.subject}” is not in the ontology."}

        chosen = concept if concept else senses[0]["id"]

        if parse.relation == "is_a" and parse.target:
            answer = self.reasoner.classify(chosen, parse.target)
        elif parse.polar and parse.target and parse.relation:
            answer = self.reasoner.verify(chosen, parse.relation, parse.target,
                                          self.match)
        else:
            answer = self.reasoner.describe(chosen, parse.relation)

        answer.question = question
        answer.parse = parse.as_dict()
        answer.senses = senses
        return answer.as_dict()

    def concept(self, identifier: str) -> dict:
        """Everything stored directly about one concept, for the inspector."""
        return {
            "id": identifier,
            "gloss": self.reasoner.gloss(identifier),
            "parents": self.reasoner.parents_of(identifier),
            "facts": [f.as_dict() for f in self.reasoner.facts_of(identifier)][:200],
        }


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, payload: bytes, content_type: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):                                  # noqa: N802
            route = urlparse(self.path)
            query = parse_qs(route.query)
            try:
                if route.path in ("/", "/index.html"):
                    self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
                elif route.path == "/api/ask":
                    payload = engine.ask(query.get("q", [""])[0],
                                         (query.get("concept") or [None])[0])
                    self._send(json.dumps(payload).encode("utf-8"),
                               "application/json; charset=utf-8")
                elif route.path == "/api/concept":
                    payload = engine.concept(query.get("id", [""])[0])
                    self._send(json.dumps(payload).encode("utf-8"),
                               "application/json; charset=utf-8")
                else:
                    self._send(b"not found", "text/plain; charset=utf-8", 404)
            except Exception as error:                     # noqa: BLE001
                body = json.dumps({"error": str(error),
                                   "type": type(error).__name__}).encode("utf-8")
                self._send(body, "application/json; charset=utf-8", 500)

        def log_message(self, *args):                      # noqa: N802
            pass                                           # keep the console clean

    return Handler


def serve(store: Path, host: str, port: int, open_browser: bool = True) -> None:
    engine = Engine(store)
    server = ThreadingHTTPServer((host, port), make_handler(engine))
    url = f"http://{host}:{server.server_port}/"
    print(f"\n  V684 reasoner  ->  {url}")
    print(f"  parser: {engine.parser.backend}")
    print("  ctrl-c to stop\n")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("  stopped")
    finally:
        server.server_close()
        engine.reasoner.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=build.DEFAULT_STORE)
    parser.add_argument("--source", type=Path, default=build.SOURCE_DATABASE)
    parser.add_argument("--ascent", type=Path, default=build.ASCENT_CSV)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8684)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--rebuild", action="store_true",
                        help="discard and rebuild the reasoning store")
    arguments = parser.parse_args()
    if arguments.rebuild and arguments.store.exists():
        arguments.store.unlink()
    build.ensure(arguments.store, arguments.source, arguments.ascent)
    serve(arguments.store, arguments.host, arguments.port,
          not arguments.no_browser)


if __name__ == "__main__":
    main()
