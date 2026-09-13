"""One command: build if needed, serve the reasoner, open the browser.

    python -m research.v687

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

from . import build, compress, pins, rules
from .language import Parser
from .reason import Reasoner

HERE = Path(__file__).resolve().parent
PAGE = HERE / "app.html"


#: The tagger's labels in WordNet's alphabet.
WORDNET_POS = {"NOUN": "n", "PROPN": "n", "VERB": "v", "ADJ": "a",
               "ADV": "r"}


def subject_pos(parse) -> str | None:
    """What part of speech the subject was used as, if the tagger said."""
    subject = (getattr(parse, "subject", "") or "").lower()
    for token in getattr(parse, "tokens", None) or []:
        if (token.get("lemma") or token.get("text") or "").lower() == subject:
            return WORDNET_POS.get(token.get("pos"))
    return None


class Engine:
    """Parser plus reasoner, shared by every request."""

    #: What the page calls itself. The page is shared, so the server names it
    #: rather than the file hard-coding one of them and mislabelling the other.
    title = "V684 Reasoner"

    def __init__(self, store: Path):
        self.reasoner = Reasoner(store)
        # The parser reads subjects better when it knows what exists: `fire
        # truck` is one concept, `hammer break` is not.
        self.parser = Parser(vocabulary=self.reasoner.vocabulary(),
                             nouns=self.reasoner.noun_vocabulary())
        self.match = self.parser.matcher()

    def ask(self, question: str, concept: str | None = None) -> dict:
        parse = self.parser.parse(question)
        if not parse.subject and not concept:
            # When the parser gave up on a particular word, that word is the
            # answer: "I do not know what a wemble is" is a real answer, and
            # answering about the next noun along was not.
            return {"verdict": "UNKNOWN_WORD" if parse.unknown else "UNPARSED",
                    "question": question,
                    "parse": parse.as_dict(), "senses": [], "steps": [],
                    "evidence": [], "chain": [],
                    "note": (f"“{parse.unknown}” is not a word in this "
                             f"ontology, and it is what the question is "
                             f"about. Nothing can be said about it here "
                             f"without first being told what it is."
                             if parse.unknown else
                             "No noun found to reason about.")}

        # Senses always come from the word in the question. `concept` selects
        # among them; it is a synset id, not something to look up as a lemma.
        #
        # The tagger has already said what part of speech the subject was
        # used as, and offering a noun sense for a word tagged as a verb
        # throws that away. It rarely changes a subject -- subjects are
        # mostly nouns -- and it is free and correct.
        senses = self.reasoner.senses_of(parse.subject or "",
                                         subject_pos(parse))
        if not senses and not concept:
            return {"verdict": "UNKNOWN_WORD", "question": question,
                    "parse": parse.as_dict(), "senses": [], "steps": [],
                    "evidence": [], "chain": [],
                    "note": f"“{parse.subject}” is not in the ontology."}

        chosen = concept if concept else senses[0]["id"]

        if parse.relation == "is_a" and parse.target:
            answer = self.reasoner.classify(chosen, parse.target)
            # The target is already read existentially -- "any of its senses
            # counts" -- and a classification question reads the same way on
            # the subject side. `is red a color` resolved `red` to the
            # tributary of the Mississippi, which has facts where the colour
            # has none, and answered no. Asking a word whether it names a kind
            # of something is asking whether *any* of its senses does, so the
            # other senses are tried and the one that answers is named.
            # An exclusion does not settle the word either, and gating this on
            # UNKNOWN meant it did. `is a donkey a mammal` came back
            # CONTRADICTED because `donkey.n.01` is the symbol of the
            # Democratic Party, under `emblem -> symbol -> abstraction`; the
            # animal is `domestic ass.n.01` and was never asked. R27 is right
            # about the sense it was given and that is exactly why its answer
            # cannot end the search -- the sentence above says a word names a
            # kind of something if *any* of its senses does, and a no about
            # one sense is not a no about the word.
            excluded = any(step.rule == "R27" for step in answer.steps)
            unsettled = answer.verdict == "UNKNOWN" or excluded
            if unsettled and concept is None and len(senses) > 1:
                # Where the target is unambiguous about its branch it says
                # which reading of the subject was meant. `flowering plant` is
                # `angiosperm.n.01`, under `plant`, so `is a hyacinth a
                # flowering plant` is about `hyacinth.n.02` and not about the
                # zircon -- and the answer is then UNKNOWN, because WordNet
                # files hyacinth under `vascular plant` and never reaches
                # `angiosperm`. A hole in the tree is an absence; the zircon
                # was a confident no about the wrong thing.
                #
                # Only a *match* moves the sense, never the mere absence of an
                # exclusion. `dog` has senses the partitions cannot place at
                # all -- a hot dog is under `substance` -- and taking one of
                # those as permission to withdraw would lose `is a dog a
                # plant`, which is a correct no.
                # One branch, or none of this applies. `plant` is a factory and
                # a stooge as well as a herb, so its partitions are three and
                # it says nothing about which dog was meant -- and taking the
                # andiron as a match there lost `is a dog a plant`. A target
                # that could be anywhere places nothing.
                wanted = (self.reasoner.target_partitions(parse.target)
                          if excluded else set())
                if len(wanted) != 1:
                    wanted = set()
                for other in senses[1:8]:
                    if other["id"] == chosen:
                        continue
                    # A verb sense is not a kind of anything. `is water wet`
                    # came back VERIFIED as `water.v.01`, to supply with
                    # water, under `wet.v.01` -- the retry answering about a
                    # word's verb when the question named a thing.
                    if other.get("pos") == "v":
                        continue
                    attempt = self.reasoner.classify(other["id"], parse.target)
                    fits = bool(wanted) and (
                        self.reasoner.partition_of(other["id"]) in wanted)
                    if attempt.verdict != "VERIFIED" and not fits:
                        continue
                    attempt.note = (
                        f"Not of {chosen}, the sense carrying the most facts, "
                        f"but of {other['id']} — {other['definition']}. A word "
                        f"names a kind of something if any of its senses does."
                        if attempt.verdict == "VERIFIED" else
                        f"Not of {chosen} — “{parse.target}” places this "
                        f"question in a branch {chosen} is not in, and "
                        f"{other['id']} is: {other['definition']}. "
                        f"{attempt.note}")
                    answer, chosen = attempt, other["id"]
                    break
            # A hedged `is_a` was a guess -- `is winter cold` has the shape of
            # `is a chair furniture`, and only the data tells them apart. When
            # the taxonomy has nothing, the property reading gets its turn,
            # and `winter has_property cold` was there the whole time.
            #
            # R27's exclusion counts as having nothing. It is a sound answer
            # to the taxonomy question, and on a hedged predicate the taxonomy
            # question is the guess: `modern`, `aquatic`, `cold` and
            # `feminine` all carry noun senses under a different top branch,
            # so `is a television modern` came back CONTRADICTED at 0.95 --
            # confident, and about a reading nobody asked for. Every
            # high-confidence false denial the COMPS audit found was this.
            #
            # The determiner is what separates the two readings and the parser
            # already has it: `an animal` is not hedged and `modern` is, which
            # is why this is gated on the grammar rather than on whether the
            # target happens to own an adjective sense. `animal` owns one too,
            # and gating on that would have taken `is a mouse an animal` with
            # it.
            excluded = (answer.verdict == "CONTRADICTED"
                        and any(step.rule == "R27" for step in answer.steps))
            if parse.hedged and (answer.verdict == "UNKNOWN" or excluded):
                attempt = self.reasoner.verify(chosen, "has_property",
                                               parse.target, self.match)
                # Only what the concept says of itself. A hedged reading is
                # already a guess about which question was asked, and letting
                # it inherit compounds one guess with another: `wild` is
                # recorded of canines, and `is a dog wild` came back yes.
                here = [fact for fact in attempt.evidence
                        if not getattr(fact, "distance", 0)]
                if attempt.verdict != "UNKNOWN" and here:
                    answer = attempt
                elif excluded:
                    # Neither reading has anything. The taxonomy one is still
                    # standing, and it answers a question that was not asked,
                    # so it goes back to what this codebase says everywhere
                    # else about a store that never recorded something.
                    answer.verdict = "UNKNOWN"
                    answer.note = (
                        f"“{parse.target}” has a noun sense in a branch of "
                        f"the taxonomy {chosen} cannot be in, but nothing "
                        f"here asked a taxonomy question: with no determiner "
                        f"this reads as a property, and nothing states "
                        f"{chosen.rsplit('.', 2)[0]} as “{parse.target}”. "
                        f"Absent, not false. R27 withdrawn.")
                    answer.steps.append(rules.Step(
                        len(answer.steps), "stop", chosen, 0, "R27",
                        f"“{parse.target}” is bare, so the is_a reading was a "
                        f"guess. The exclusion answers the other question and "
                        f"is withdrawn."))
        elif parse.polar and parse.target and parse.relation:
            answer = self.sense_first(chosen, parse)
            answer = self.corroborate(answer, parse.target)
        else:
            answer = self.reasoner.describe(chosen, parse.relation)

        answer.question = question
        answer.parse = parse.as_dict()
        answer.senses = senses
        payload = answer.as_dict()
        # Real distances come from the steps. `chain` is visit order, and using
        # its index as a distance spreads the graph over twice as many shells
        # as exist, leaving too few nodes in each to form a ring.
        distances: dict[str, int] = {}
        for step in answer.steps:
            distances.setdefault(step.concept, step.distance)
        payload["neighbourhood"] = self.neighbourhood(chosen, answer.chain, distances)
        payload["store"] = self.reasoner.store.name
        return payload

    def sense_first(self, concept: str, parse):
        """R29 before R2: ask the graph, then fall back to the words.

        A pin on the object only means something where there is a sense to
        bind it to, and in this store that is WordNet's synset-to-synset
        rows -- `has_part`, `part_of`, `similar_to`, `entails`, `causes`.
        Everything else, `capable_of` and its 772,890 rows included, is free
        text from a crawl, and `can a dog bark` is answered by matching the
        word "bark" against a norm predicate that could be any of its nine
        senses.

        So: if the reader pinned a word in the object and the relation has a
        sense-tagged form, put the question to the graph between the two
        synsets, with no string matching anywhere in it. `does a car have an
        accelerator` is UNKNOWN through the words -- the accelerator is
        recorded only as a synset -- and VERIFIED through the graph.

        When the graph has nothing the words still get their turn, and the
        answer says which of the two spoke, because "the graph does not
        record this" and "no word matched" are different things to know.
        """
        pinned = ""
        for word in (parse.target or "").replace("-", " ").split():
            pinned = pins.of(word.strip(".,;:").lower()) or pinned
            if pinned:
                break
        if pinned:
            found = self.reasoner.verify_sense(concept, parse.relation, pinned)
            if found.verdict != "UNKNOWN":
                return found
        answer = self.reasoner.verify(concept, parse.relation, parse.target,
                                      self.match)
        if pinned and parse.relation in self.reasoner.SENSE_TAGGED:
            answer.note = ((answer.note or "").rstrip() + " " + (
                f"Asked between senses first — the graph records no "
                f"`{parse.relation}` between {concept} and {pinned} — and "
                f"then between words, which is what answered.")).strip()
        return answer

    def corroborate(self, answer, target: str):
        """R19 hook: put an inherited fact to the ancestor's other kinds.

        A no-op here. This engine has the fact store and nothing to check it
        against; the norms arrive two subclasses up, and that is where the
        override lives. The hook is here because this is where the fact path
        produces its verdict, and R19 was written for exactly this path and
        then only ever wired into the other one.
        """
        return answer

    def neighbourhood(self, concept: str, chain: list[str],
                      distances: dict[str, int], per_level: int = 3) -> dict:
        """The graph around the derivation, so the walk has context to move in.

        Kept deliberately thin. A sibling earns its place by showing what a
        generalisation swept in -- the other things the answer now also covers
        -- and three per ancestor makes that point. Nine buried the path they
        were supposed to explain.
        """
        on_path = set(chain)
        nodes: dict[str, dict] = {}
        edges: set[tuple[str, str]] = set()

        for node in chain:
            nodes[node] = {"id": node, "on_path": True,
                           "distance": distances.get(node, 0)}
            for parent in self.reasoner.parents_of(node):
                edges.add((node, parent))

        # siblings: other children of each ancestor actually on the path
        for node in chain:
            if node == concept:
                continue
            siblings = self.reasoner.connection.execute(
                "SELECT child FROM taxonomy WHERE parent = ? LIMIT ?",
                (node, per_level * 3)).fetchall()
            picked = [r["child"] for r in siblings if r["child"] not in on_path]
            # A sibling is one step more specific than the ancestor it hangs
            # from: the other children of `canine` sit at dog's own level.
            for sibling in picked[:per_level]:
                nodes.setdefault(sibling, {
                    "id": sibling, "on_path": False,
                    "distance": max(0, nodes[node]["distance"] - 1),
                })
                edges.add((sibling, node))
        return {
            "nodes": list(nodes.values()),
            "edges": [{"from": a, "to": b} for a, b in edges
                      if a in nodes and b in nodes],
        }

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
                    page = PAGE.read_text(encoding="utf-8").replace(
                        "V684 Reasoner", engine.title)
                    self._send(page.encode("utf-8"), "text/html; charset=utf-8")
                elif route.path == "/api/ask":
                    # `pin=bark:bark.v.01`, repeatable: the reader's choice of
                    # sense for a word, which outranks the engine's own.
                    pinned = pins.parse(query.get("pin", []))
                    payload = engine.ask(query.get("q", [""])[0],
                                         (query.get("concept") or [None])[0],
                                         pinned or None)
                    self._send(json.dumps(payload).encode("utf-8"),
                               "application/json; charset=utf-8")
                elif route.path == "/api/senses":
                    payload = engine.word_senses(query.get("q", [""])[0])
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
    parser.add_argument("--raw", action="store_true",
                        help="serve the uncompressed store instead of the "
                             "inheritance-compressed one")
    arguments = parser.parse_args()
    if arguments.rebuild and arguments.store.exists():
        arguments.store.unlink()
    build.ensure(arguments.store, arguments.source, arguments.ascent)
    store = arguments.store
    if not arguments.raw:
        # Answer from the compressed store: what R10 dropped, R2 rebuilds.
        packed = compress.DEFAULT_COMPRESSED
        if arguments.rebuild and packed.exists():
            packed.unlink()
        if not packed.exists():
            print("compressing by inheritance (first run)")
            compress.compress(store, packed)
        store = packed
    serve(store, arguments.host, arguments.port, not arguments.no_browser)


if __name__ == "__main__":
    main()
