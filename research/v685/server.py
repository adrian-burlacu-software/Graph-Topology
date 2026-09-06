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
from .relevance import RULE_TEXT as V685_RULES


class BridgedEngine(v684_server.Engine):
    """v684's engine, with the two-subject case routed through v685."""

    title = "V685 Reasoner"

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
            return self.with_rules(super().ask(question, concept))

        result = self.bridged.ask(question)
        if result.answer is None:
            return self.with_rules(super().ask(question, concept))

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
            self.replayable(payload, result, aside)
        return self.with_rules(payload)

    @staticmethod
    def replayable(payload: dict, result, aside: list) -> None:
        """Put the bridge and the filtering into the derivation trace.

        Without this the page showed a single dot. The answer is about
        `musician`, so v684's trace begins and ends there -- and the two
        things that actually made it an answer about violin players, the hop
        from the violin and the twelve facts R14 removed, happened outside
        the trace entirely.

        They are expressed as ordinary steps rather than a second widget, so
        the existing replay, scrubber and graph animate them with no special
        case: the anchor at distance 0, one step per hop, and the role's own
        derivation shifted out to make room.
        """
        hops = (result.route.hops if result.route else [])
        shift = len(hops) if hops else 0
        steps = payload.get("steps", [])
        if shift:
            for step in steps:
                step["distance"] = step.get("distance", 0) + shift

        opening: list[dict] = []
        if hops:
            anchor_word = result.anchor_word or ""
            opening.append({
                "index": 0, "kind": "resolve", "concept": result.anchor,
                "distance": 0, "rule": "R6",
                "detail": f"Reading “{anchor_word}” as {result.anchor} — the "
                          f"anchor the question is asked from.",
                "facts_checked": 0, "matched": None, "parents": []})
            for position, hop in enumerate(hops):
                opening.append({
                    "index": 0, "kind": "ascend", "concept": hop.source,
                    "distance": position, "rule": "R15",
                    "detail": f"{hop.source.rsplit('.', 2)[0]} "
                              f"{hop.relation.replace('_', ' ')} "
                              f"“{hop.text}” — so "
                              f"{hop.target.rsplit('.', 2)[0]} is reachable.",
                    "facts_checked": 0, "matched": None,
                    "parents": [hop.target]})

        closing: list[dict] = []
        if aside:
            shared = (result.excluded_class or "").rsplit(".", 2)[0]
            closing.append({
                "index": 0, "kind": "skip", "concept": result.role,
                "distance": shift, "rule": "R14",
                "detail": f"Set aside {len(aside)} fact(s) naming another "
                          f"{shared or 'sibling'}: true of "
                          f"{(result.role or '').rsplit('.', 2)[0]}, not of a "
                          f"{result.anchor_word}.",
                "facts_checked": len(aside), "matched": None, "parents": []})

        combined = opening + steps + closing
        for position, step in enumerate(combined):
            step["index"] = position
        payload["steps"] = combined
        if hops and result.anchor:
            chain = payload.get("chain") or []
            payload["chain"] = [result.anchor] + [h.target for h in hops
                                                  if h.target not in chain] + chain

    @staticmethod
    def with_rules(payload: dict) -> dict:
        """v685 runs one rule v684 does not, so the page has to list it."""
        rules = dict(payload.get("rules") or {})
        rules.update(V685_RULES)
        payload["rules"] = rules
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
