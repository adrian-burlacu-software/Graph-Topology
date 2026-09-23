"""The v690 page: a conversation in English both ways, every step it took,
and memory as it stands.

    python -m research.v690 --workers 19 --port 8690

What is said is read by the encoder and answered by v689 over v687's store
and v688's loop, as on v689's page; what is answered is said by the decoder
and read back before it is said (`speaking.py`). Each turn carries its steps
(`steps.py`), the senses its words were taken in, which a conversation can
pin, and the memory it left.

v689's page is at `/v689` and v688's at `/v688`, over the same process.
Conversations and what they taught are kept in `state/v690-memory.sqlite`;
definitions memory is shared with v689's.
"""
from __future__ import annotations

import argparse
import threading
import urllib.parse
from http.server import ThreadingHTTPServer
from pathlib import Path

from research.v687 import build
from research.v688 import server as v688
from research.v688.pool import DEFAULT_WORKERS
from research.v689 import server as v689
from research.v689.definitions import DefinitionMemory
from research.v689.longterm import DEFINITIONS_PATH, Archive, Keeper

from .steps import steps_of

# Registers v691's acts with the session (`v691/page.py`): the
# agent is an act like any other, and until a world is opened it
# proposes nothing.
from research.v691 import page as v691_page  # noqa: F401
# And v692's mathematics (`v692/page.py`): an act the encoder proposes when
# an utterance asks something of mathematics, and R33 for the reasoner.
from research.v692 import page as v692_page  # noqa: F401

HERE = Path(__file__).resolve().parent
STATE = Path(__file__).resolve().parents[2] / "state"
DEFAULT_PATH = STATE / "v690-memory.sqlite"

#: How many of a turn's words are offered senses to pick.
OFFERED = 6
#: Auxiliaries a verb is said after: what follows is offered as a verb.
VERBAL = frozenset({"can", "could", "do", "does", "did", "will", "would",
                    "should", "may", "might", "must", "shall"})
#: How many senses of a word are offered.
SENSES = 8

EXAMPLES = [
    {"title": "hello",
     "lines": ["hello", "what can you do", "who are you",
               "there is a dog", "can it swim", "thanks", "bye"],
     "shows": "Social acts are read by the encoder like everything else, "
              "and answered from what the conversation is: what it can do is "
              "the relations its operators answer. Every reply is written by "
              "the decoder and read back before it is said."},
    {"title": "common sense",
     "lines": ["can a penguin fly", "why", "what is a whale",
               "is a whale a fish", "what can a bird do",
               "what do a dog and a cat have in common",
               "what is the capital of France"],
     "shows": "Questions about kinds go to v688's loop over v687's store. The "
              "decoder says the verdict, what it rests on and how far it is "
              "trusted, and nothing the store did not say: a reason it "
              "invents reads back as an added word, and is not said."},
    {"title": "a morning at home",
     "lines": ["Mary went to the kitchen", "John picked up the football",
               "John went to the garden", "where is Mary",
               "who has the football", "where is the football",
               "is Mary in the garden", "what did John pick up",
               "what happened first"],
     "shows": "Episodic memory: each statement is an event, placed in story "
              "time; VerbNet says what each changes (T4); questions are goals "
              "answered by composing memory. Watch the timeline fill in."},
] + v689.EXAMPLES


class PinnedAsker(v689.StoreAsker):
    """v689's asker, with the senses a conversation pinned: a kind placed
    under the sense pinned for its word, a verb read first in its pinned
    sense, and v688 asked with every pin, as its own page asks it."""

    def __init__(self, service) -> None:
        super().__init__(service)
        self.pins: dict[str, str] = {}

    def sense(self, kind: str):
        pinned = self.pins.get((kind or "").lower())
        return pinned or super().sense(kind)

    def verb_senses(self, lemma: str) -> list[str]:
        found = super().verb_senses(lemma)
        pinned = self.pins.get((lemma or "").lower())
        if pinned and pinned.split(".")[-2:-1] == ["v"]:
            return [pinned] + [one for one in found if one != pinned]
        return found

    def run(self, question: str) -> dict:
        if not self.pins:
            return super().run(question)
        key = (question.strip().lower() + "|"
               + repr(sorted(self.pins.items())))
        with self.service._lock:
            found = self.service._cache.get(key)
        if found is None:
            found = self.service.loop.run(question, dict(self.pins)).as_dict()
            with self.service._lock:
                self.service._cache[key] = found
        return found


class Conversations:
    """The page's conversations: v689's keeper, each turn said in English
    and taken apart into steps before it is kept."""

    def __init__(self, service, archive: Archive | None,
                 definitions: DefinitionMemory | None, speaker) -> None:
        self.service = service
        self.archive = archive
        self.definitions = definitions
        self.speaker = speaker
        self.pins: dict[str, dict[str, str]] = {}
        self.pins_lock = threading.Lock()
        with service._engines:
            self.asker = PinnedAsker(service)
            self.keeper = Keeper(self.asker, archive, definitions=definitions)

    def _spoken(self, turn: dict) -> dict:
        """A turn trimmed as v689's page trims it, with the reply said to it
        (`reply` -- `said` is what was said to it) and its steps."""
        turn = v689.trimmed(turn)
        reply = None
        spoken = (turn.get("answer") or {}).get("spoken")
        if spoken:
            # An answer that already is a sentence is not said again. v691's
            # agent writes its reply from the plan it carried out, and the
            # decoder is trained to paraphrase v689's verdicts -- given a
            # story it was never shown, it would paraphrase it into one.
            reply = {"text": spoken, "traced": True, "candidates": [],
                     "source": "the agent's own words"}
        elif self.speaker is not None:
            try:
                reply = self.speaker.speak(turn).as_dict()
            except Exception as bad:            # noqa: BLE001
                reply = {"text": "", "traced": False,
                         "error": f"{type(bad).__name__}: {bad}",
                         "candidates": []}
        turn["reply"] = reply
        turn["steps"] = steps_of(turn, reply)
        return turn

    def say(self, sid: str, text: str, example: bool = False) -> dict:
        with self.pins_lock:
            pins = dict(self.pins.get(sid, {}))
        with self.service._engines:
            self.asker.pins = pins
            try:
                turn = self.keeper.say(sid, text, example, trim=self._spoken)
            finally:
                self.asker.pins = {}
        turn["senses"] = self.senses_of(turn, pins)
        turn["pins"] = pins
        return turn

    def _words(self, turn: dict) -> list[tuple[str, str]]:
        """The words a turn was about, and whether each was a noun: the
        kinds its phrases named, then what was said of them."""
        from .roundtrip import FUNCTION, NEGATIONS

        # Words with no sense to pick: what `roundtrip` counts as no content,
        # and the articles, prepositions and question words around it.
        plain = FUNCTION | NEGATIONS | {
            "the", "and", "for", "with", "from", "into", "onto", "about",
            "what", "where", "who", "whom", "when", "why", "how", "which",
            "there", "that", "this", "these", "those", "does", "did", "are",
            "was", "were", "has", "had", "been", "being", "its", "their"}
        reading = turn.get("reading") or {}
        found: list[tuple[str, str]] = []

        def add(word: str, used_as: str) -> None:
            word = (word or "").strip().lower()
            if (word and word not in plain and len(word) > 2
                    and all(one != word for one, _ in found)):
                found.append((word, used_as))

        for key in ("mention", "object"):
            add((reading.get(key) or {}).get("kind", ""), "n")
        # What was said of it: a verb after `can`, `does`, `will` (`can it
        # swim` is not swimming.n.01), a noun after an article.
        verbal = (reading.get("aux") or "") in VERBAL
        before = ""
        for word, role in (turn.get("heard") or {}).get("roles") or []:
            if role in ("REST", "VERB", "B-KIND", "I-KIND"):
                used_as = ("n" if before in ("a", "an", "the") or "KIND" in role
                           else "v" if verbal or role == "VERB" else "")
                add(word, used_as)
                verbal = False
            before = word
        asked = (turn.get("asked") or "").split()
        for at, word in enumerate(asked):
            add(word, "n" if at and asked[at - 1] in ("a", "an") else "")
        return found[:OFFERED]

    def senses_of(self, turn: dict, pins: dict) -> list[dict]:
        out = []
        for word, used_as in self._words(turn):
            try:
                offered = self.service.senses(word, used_as)
            except Exception:                   # noqa: BLE001
                continue
            if not offered:
                continue
            out.append({"word": word, "as": used_as,
                        "pinned": pins.get(word),
                        "senses": [{"id": one["id"],
                                    "definition": one.get("definition") or "",
                                    "pos": one.get("pos") or "",
                                    "facts": one.get("fact_count") or 0,
                                    "picks": index == 0}
                                   for index, one in
                                   enumerate(offered[:SENSES])]})
        return out

    def pin(self, sid: str, word: str, sense: str) -> dict:
        word = word.strip().lower()
        with self.pins_lock:
            pins = self.pins.setdefault(sid, {})
            if sense:
                pins[word] = sense.strip()
            else:
                pins.pop(word, None)
            return dict(pins)

    def history(self, sid: str) -> dict:
        with self.service._engines:
            found = self.keeper.history(sid)
        with self.pins_lock:
            found["pins"] = dict(self.pins.get(sid, {}))
        turns = found.get("turns") or []
        if turns:
            turns[-1]["senses"] = self.senses_of(turns[-1], found["pins"])
        return found

    def forget(self, sid: str) -> None:
        with self.service._engines:
            self.keeper.forget(sid)
        with self.pins_lock:
            self.pins.pop(sid, None)

    def unlearn(self) -> dict:
        with self.service._engines:
            self.keeper.unlearn()
            return self.keeper.summary()

    def longterm(self, like: str = "") -> dict:
        """Long-term memory as it stands: every kind, taxonomy edge and norm
        taught in any conversation, definitions memory, and the
        conversations kept."""
        with self.service._engines:
            state = self.keeper.knowledge.as_state()
            summary = self.keeper.summary()
        edges = [{"node": node, "parent": parent, "said": said,
                  "conversation": conversation, "when": when}
                 for node, parent, said, conversation, when in state["edges"]]
        norms = [{"node": node, "relation": relation, "object": obj,
                  "said": said, "mode": mode, "against": against,
                  "conversation": conversation, "when": when}
                 for (node, relation, obj, said, mode, against, conversation,
                      when) in state["norms"]]
        return {"summary": summary, "kinds": state["kinds"], "edges": edges,
                "norms": norms,
                "definitions": (self.definitions.recent(40, like)
                                if self.definitions is not None else []),
                "conversations": (self.archive.listed(30)
                                  if self.archive is not None else [])}


class Handler(v688.Handler):
    """v688's handler, with v690's conversation in front of it."""

    conversations: Conversations = None     # type: ignore[assignment]

    def do_GET(self) -> None:                   # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)

        def one(name: str, limit: int = 200) -> str:
            return (query.get(name) or [""])[0].strip()[:limit]

        sid = one("sid", 64)
        pages = {"/": HERE / "app.html", "/index.html": HERE / "app.html",
                 "/v689": v689.HERE / "app.html",
                 "/v688": v688.HERE / "app.html"}
        if parsed.path in pages:
            page = pages[parsed.path].read_text(encoding="utf-8")
            self._send(page.encode("utf-8"), "text/html; charset=utf-8")
        elif parsed.path == "/api/examples":
            self._json({"examples": EXAMPLES})
        elif parsed.path == "/api/say":
            text = one("q", 400)
            if not sid or not text:
                self._json({"error": "a conversation id and something said"},
                           400)
                return
            if len(text) > 200:
                self._json({"error": "too long"}, 400)
                return
            try:
                self._json(self.conversations.say(sid, text,
                                                  one("example") == "1"))
            except Exception as bad:            # noqa: BLE001
                self._json({"error": f"{type(bad).__name__}: {bad}"}, 500)
        elif parsed.path == "/api/history":
            if not sid:
                self._json({"error": "a conversation id"}, 400)
                return
            self._json(self.conversations.history(sid))
        elif parsed.path == "/api/forget":
            self.conversations.forget(sid)
            self._json({"forgotten": True})
        elif parsed.path == "/api/pin":
            word = one("w", 40)
            if not sid or not word:
                self._json({"error": "a conversation id and a word"}, 400)
                return
            self._json({"pins": self.conversations.pin(sid, word,
                                                       one("sense", 80))})
        elif parsed.path == "/api/longterm":
            self._json(self.conversations.longterm(one("like", 40)))
        elif parsed.path == "/api/unlearn":
            if one("confirm") != "yes":
                self._json({"error": "unlearn needs confirm=yes"}, 400)
                return
            self._json({"knowledge": self.conversations.unlearn()})
        else:
            super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8690)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--store", type=Path, default=build.DEFAULT_STORE)
    parser.add_argument("--teacher", action="store_true",
                        help="load v688's teacher; about 6.2 GB of VRAM, "
                             "which leaves the decoder too little")
    parser.add_argument("--memory", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--no-memory", action="store_true",
                        help="keep nothing between runs")
    parser.add_argument("--definitions", type=Path, default=DEFINITIONS_PATH)
    parser.add_argument("--silent", action="store_true",
                        help="no decoder: turns carry v689's own answer only")
    options = parser.parse_args()

    print(f"building {options.workers} engines from {options.store.name} ...")
    service = v688.Service(options.store, options.workers, options.cycles,
                           teacher=options.teacher)
    print(f"  {service.pool.workers} engines up in "
          f"{service.pool.build_seconds:.1f}s")
    speaker = None
    if not options.silent:
        from .speaking import Speaker

        nlp = getattr(service.pool.engines[0].parser, "nlp", None)
        speaker = Speaker(nlp)
        print(f"  decoder {speaker.decoder.config.get('base', '')} on "
              f"{speaker.decoder.device}, {len(speaker.framing)} framing "
              f"words")
    Handler.service = service
    archive = None if options.no_memory else Archive(options.memory)
    # What v691 learns about acting is kept on the same terms as the rest.
    from research.v691.learned import DEFAULT_PATH as LEARNED_PATH
    v691_page.keep(None if options.no_memory else LEARNED_PATH)
    definitions = DefinitionMemory(None if options.no_memory
                                   else options.definitions)
    Handler.conversations = Conversations(service, archive, definitions,
                                          speaker)
    httpd = ThreadingHTTPServer(("127.0.0.1", options.port), Handler)
    print(f"v690 conversation on http://127.0.0.1:{options.port} "
          f"(v689 at /v689, v688 at /v688)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.pool.close()


if __name__ == "__main__":
    main()
