"""One command: everything v685 serves, plus identification.

    python -m research.v686

The page, the store, the rules and the bridge are v685's, untouched. What is
added is the inverse question -- describe a thing and be told what it is:

    what kind of dog has spots
    what kind of cat has stripes
    what kind of bird is red

A question that describes rather than names is answered by `identify.py` and
comes back with its narrowing shown step by step, in the same shape the page
already replays. Everything else falls through to v685, which falls through to
v684, so this is a superset of a superset rather than a third fork.
"""
from __future__ import annotations

import argparse
import re
import threading
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path

from ..v684 import build, compress, rules as v684_rules, server as v684_server
from ..v685.relevance import RULE_TEXT as V685_RULES
from ..v685.server import BridgedEngine
from .identify import Identifier

#: Rule text for the identification half, listed on the page beside the rest.
V686_RULES: dict[str, str] = {
    "R16": "Identification: a description is answered by walking the trie "
           "down instead of storing into it. The property that eliminates "
           "most candidates is asked first -- the same coverage ordering "
           "Appendix 3 uses to choose a storage slot. What a norm states "
           "about a thing outranks what it inherits.",
}

#: A question that describes rather than names. `what kind of X ...` and
#: `what is <adjective> and <adjective>` are the two shapes; naming questions
#: like `what can a violin do` deliberately do not match, because they have a
#: subject and belong to v684.
DESCRIBES = re.compile(
    r"^\s*(?:what|which)\b(?!.*\b(?:a|an|the)\s+\w+(?:'s)?\s+\w*\s*(?:do|need)\b)"
    r".*\b(?:kind|type|sort)s?\s+of\b|"
    r"^\s*(?:what|which)\s+(?:is|are)\s+\w+\s+and\b",
    re.IGNORECASE)


class IdentifyingEngine(BridgedEngine):
    """v685's engine, with descriptions answered by identification."""

    title = "V686 Reasoner"

    def __init__(self, store: Path, depth: int = 3, breadth: int = 60):
        super().__init__(store, depth=depth, breadth=breadth)
        # shares the open reasoner: the identifier needs the same taxonomy
        self.identifier = Identifier(store, reasoner=self.reasoner)

    def ask(self, question: str, concept: str | None = None) -> dict:
        if concept or not DESCRIBES.search(question or ""):
            payload = super().ask(question, concept)
            payload["rules"] = {**payload.get("rules", {}), **V686_RULES}
            return payload

        found = self.identifier.identify(question)
        if found.verdict == "NO_MATCH" and not found.terms:
            return super().ask(question, concept)

        payload = {
            "question": question,
            "verdict": found.verdict,
            "concept": found.candidates[0].name if found.candidates else None,
            "concept_gloss": None,
            "senses": [], "chain": [], "evidence": [], "suggestions": [],
            "parse": {"question": question, "subject": found.among,
                      "relation": "identify", "target": ", ".join(found.terms),
                      "polar": False, "backend": self.parser.backend,
                      "tokens": [], "note": ""},
            "note": found.note,
            "identification": found.as_dict(),
            "steps": self._steps(found),
            "rules": {**v684_rules.RULE_TEXT, **V685_RULES, **V686_RULES},
            "store": self.reasoner.store.name,
            "neighbourhood": {"nodes": [], "edges": []},
        }
        return payload

    @staticmethod
    def _steps(found) -> list[dict]:
        """The narrowing, in the shape the page's replay already understands."""
        steps = []
        for position, step in enumerate(found.steps):
            remaining = step["remaining"]
            examples = ", ".join(step.get("examples", [])[:5])
            steps.append({
                "index": position,
                "kind": "match" if remaining == 1 else (
                    "stop" if remaining == 0 else "check"),
                "concept": step["term"],
                "distance": position,
                "rule": step["rule"],
                "detail": f"{step['detail']} — {remaining} left"
                          + (f": {examples}" if examples else ""),
                "facts_checked": remaining,
                "matched": None,
                "parents": [],
            })
        return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=build.DEFAULT_STORE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8686)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--breadth", type=int, default=60)
    arguments = parser.parse_args()

    build.ensure(arguments.store)
    store = arguments.store
    if not arguments.raw:
        packed = compress.DEFAULT_COMPRESSED
        if not packed.exists():
            print("compressing by inheritance (first run)")
            compress.compress(arguments.store, packed)
        store = packed

    print("  loading feature norms and joining them to WordNet...")
    engine = IdentifyingEngine(store, arguments.depth, arguments.breadth)
    httpd = ThreadingHTTPServer((arguments.host, arguments.port),
                                v684_server.make_handler(engine))
    url = f"http://{arguments.host}:{httpd.server_port}/"
    print(f"\n  V686 reasoner  ->  {url}")
    print(f"  parser: {engine.parser.backend}")
    print(f"  {len(engine.identifier.stated):,} individuals from XCSLB + AwA2, "
          f"{len(engine.identifier.synset):,} joined to WordNet")
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
