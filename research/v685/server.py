"""One command: serve the bridged reasoner.

    python -m research.v685

Same page as v684, same store, same rules. The only difference is what
happens to a question with two subjects in it: v685 bridges them first and
hands the payload an extra `bridge` key, which the page renders as its own
card. A question without a possessive is answered by v684 untouched, so this
is a strict superset rather than a fork.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..v684 import build, compress, server as v684_server
from .ask import BridgedReasoner


class BridgedEngine(v684_server.Engine):
    """v684's engine, with the two-subject case routed through v685."""

    def __init__(self, store: Path, depth: int = 3, breadth: int = 60):
        super().__init__(store)
        self.bridged = BridgedReasoner(store, depth=depth, breadth=breadth)
        # one store, one parser, one graph: the bridged reasoner reuses what
        # is already open rather than holding a second connection to the same
        # file with a second copy of the lemma index.
        self.bridged.reasoner.close()
        self.bridged.reasoner = self.reasoner
        self.bridged.parser = self.parser

    def ask(self, question: str, concept: str | None = None) -> dict:
        anchor, role = self.bridged.two_subjects(question)
        if not role or concept:
            # No second subject, or the user pinned a sense by clicking: this
            # is an ordinary v684 question and stays one.
            return super().ask(question, concept)

        result = self.bridged.ask(question)
        if result.answer is None:
            return super().ask(question, concept)

        # Answer about the role, exactly as v684 would, so the derivation
        # replay and the globe keep working with no special case.
        payload = super().ask(question, result.role)
        payload["bridge"] = result.as_dict()
        payload["bridge"].pop("answer", None)     # already the payload itself

        # R14 has to be applied here too. `super().ask` re-derives the answer
        # from scratch, so the filtering done on `result.answer` does not
        # reach the payload and `play drum` came back at the top of the list.
        if result.anchor:
            relevance = self.bridged.relevance
            kept, aside = [], []
            for row in payload.get("evidence", []):
                judgement = relevance.judge(row.get("object", ""), result.anchor)
                row["anchor_relation"] = judgement.verdict
                (aside if judgement.verdict == "sibling" else kept).append(row)
            kept.sort(key=lambda r: 0 if r["anchor_relation"] == "anchor" else 1)
            payload["evidence"] = kept
            payload["bridge"]["excluded"] = aside
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=build.DEFAULT_STORE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8685)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--raw", action="store_true",
                        help="serve the uncompressed store")
    parser.add_argument("--depth", type=int, default=3,
                        help="how many hops a bridge may take")
    parser.add_argument("--breadth", type=int, default=60,
                        help="how many frontier nodes survive each round")
    arguments = parser.parse_args()

    build.ensure(arguments.store)
    store = arguments.store
    if not arguments.raw:
        packed = compress.DEFAULT_COMPRESSED
        if not packed.exists():
            print("compressing by inheritance (first run)")
            compress.compress(arguments.store, packed)
        store = packed

    engine = BridgedEngine(store, arguments.depth, arguments.breadth)
    import threading
    import webbrowser
    from http.server import ThreadingHTTPServer

    httpd = ThreadingHTTPServer((arguments.host, arguments.port),
                                v684_server.make_handler(engine))
    url = f"http://{arguments.host}:{httpd.server_port}/"
    print(f"\n  V685 bridged reasoner  ->  {url}")
    print(f"  parser: {engine.parser.backend}")
    print(f"  budget: depth {arguments.depth}, breadth {arguments.breadth}")
    print("  ctrl-c to stop\n")
    if not arguments.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("  stopped")
    finally:
        httpd.server_close()
        engine.bridged.graph.close()
        engine.reasoner.close()


if __name__ == "__main__":
    main()
