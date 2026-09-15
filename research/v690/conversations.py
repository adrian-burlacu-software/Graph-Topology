"""Conversations for the decoder to learn from, played through v689.

    python -m research.v690.conversations --processes 12

Every conversation is said to a fresh v689 session a line at a time, over the
real store with v688 behind it and no teacher, and every turn is kept as the
page gets it -- the reading, who each phrase meant, what the operators
answered, v687's walk and v688's summary -- without the memory view.
`message.py` reads what a reply to each has to say; `teach_decoder.py` teaches
the decoder to say it.

    examples    the page's own conversations (`v689/server.py`)
    babi        bAbI's training stories, every task, never the test split
    story       who went where and carried what: rooms and things from
                WordNet, people from NLTK's names
    individual  one thing put down, told of and asked about, as the page's
                examples do
    kinds       questions about kinds, as v688 is asked them
    teaching    a kind nobody has heard of, taught and then asked about
    tests       the questions v687's and v688's own tests ask

Nothing here decides what an answer is: the lines are drawn, v689 answers
them. The output is resumable; a conversation already kept is not played again.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from functools import lru_cache
from pathlib import Path

from research.encoder import LLM

DATA = LLM / "decoder-data"
TURNS = DATA / "turns.jsonl"

#: What a kept turn keeps of v688's run content and v687's walk.
ITEMS = 16
STEPS = 12
EVIDENCE = 4


# -- words to draw from ---------------------------------------------------------
@lru_cache(maxsize=None)
def words() -> dict:
    """teach_reader's vocabulary, kept to words Brown mostly uses as what
    they are drawn as -- a noun, a verb -- and the things split by what they
    are: animals under animal.n.01 to talk about, artifacts and food to
    carry, rooms under room.n.01 to go to, and the kinds right above each."""
    from nltk.corpus import wordnet

    from research.v689 import teach_reader

    found = dict(teach_reader._vocabulary())
    counts, tags = teach_reader._brown(), teach_reader._brown_tags()

    def share(word: str, prefix: str) -> float:
        seen = tags.get(word) or {}
        total = sum(seen.values())
        return (sum(count for tag, count in seen.items()
                    if tag.startswith(prefix)) / total) if total else 0.0

    def first(word: str):
        senses = wordnet.synsets(word, "n")
        return senses[0] if senses else None

    animal = wordnet.synset("animal.n.01")
    person = wordnet.synset("person.n.01")

    def above(sense) -> set:
        return set(sense.closure(lambda one: one.hypernyms()))

    people = {one.lower() for one in teach_reader._people()}

    def below(roots, least: int, nouns: float) -> list[str]:
        """Words for what WordNet puts under `roots`, each one Brown has
        `least` times, mostly as a noun, whose first sense is under them."""
        kept = set()
        for root in roots:
            for sense in root.closure(lambda one: one.hyponyms()):
                for word in sense.lemma_names():
                    if (word.isalpha() and word.islower()
                            and wordnet.morphy(word, "n") == word
                            and word not in people and counts[word] >= least
                            and share(word, "NN") >= nouns
                            and first(word) is not None
                            and set(roots) & (above(first(word))
                                              | {first(word)})):
                        kept.add(word)
        return sorted(kept)

    found["animals"] = [one for one in below([animal], 8, 0.7)
                        if person not in above(first(one))]
    # What can be carried: the things WordNet puts under a container, a
    # device, a tool, clothing, equipment or food -- not a library.
    carried = [wordnet.synset(name) for name in (
        "container.n.01", "device.n.01", "implement.n.01",
        "covering.n.02", "equipment.n.01", "toy.n.01", "food.n.02",
        "foodstuff.n.02", "edible_fruit.n.01", "vegetable.n.01",
        "consumer_goods.n.01")]
    found["objects"] = below(carried, 15, 0.8)
    found["bare"] = sorted({one for one in found["bare"]
                            + found["transitive"]
                            if share(one, "VB") >= 0.6})
    found["adjectives"] = [one for one in found["adjectives"]
                           if share(one, "JJ") >= 0.6]
    rooms = set(teach_reader.ROOMS)
    for below in wordnet.synset("room.n.01").closure(lambda one:
                                                      one.hyponyms()):
        rooms.update(one for one in below.lemma_names()
                     if one.isalpha() and counts[one] >= 30
                     and share(one, "NN") >= 0.9 and one not in people
                     and first(one) == below)
    found["rooms"] = sorted(rooms)
    found["above"] = {
        one: sorted({name for sense in first(one).hypernyms()
                     for name in sense.lemma_names()
                     if name.isalpha() and counts[name] >= 10})
        for one in found["animals"] + found["objects"]}
    return found


def plural(word: str) -> str:
    from research.v689.teach_reader import _plural
    return _plural(word)


def articled(text: str) -> str:
    from research.v689.teach_reader import _articled
    return _articled(text)


def past(verb: str) -> str:
    from research.v689.teach_reader import inflect
    return inflect(verb, "VBD")


def gerund(verb: str) -> str:
    from research.v688.rephrase import gerund as said
    return said(verb)


SYLLABLES = ("ba", "blo", "da", "fen", "gri", "ka", "lum", "mo", "nib",
             "pra", "quo", "ri", "sna", "tul", "vek", "wim", "zor")


def invented(rng: random.Random) -> str:
    """A word nobody has taught a kind by, and WordNet does not have."""
    from nltk.corpus import wordnet

    while True:
        word = "".join(rng.sample(SYLLABLES, rng.choice((2, 3))))
        if not wordnet.synsets(word):
            return word


# -- conversations ----------------------------------------------------------------
MOVING = ("went to", "moved to", "journeyed to", "travelled to",
          "went back to")
TAKING = ("picked up", "got", "grabbed", "took")
LEAVING = ("dropped", "put down", "left", "discarded")


def story(rng: random.Random) -> list[str]:
    """Who went where and carried what, with questions about it as it goes."""
    vocabulary = words()
    people = [one.capitalize() for one in rng.sample(vocabulary["names"], 3)]
    rooms = rng.sample(vocabulary["rooms"], 4)
    things = rng.sample(vocabulary["objects"], 3)
    lines: list[str] = []
    holding: dict[str, str] = {}
    visited: dict[str, list[str]] = {}
    for _ in range(rng.randint(4, 8)):
        person = rng.choice(people)
        roll = rng.random()
        if roll < 0.45 or person not in visited:
            room = rng.choice(rooms)
            opening = rng.choice(("", "", "", "then ", "yesterday ",
                                  "this morning ", "after that "))
            lines.append(f"{opening}{person} {rng.choice(MOVING)} the {room}")
            visited.setdefault(person, []).append(room)
        elif roll < 0.65:
            thing = rng.choice(things)
            lines.append(f"{person} {rng.choice(TAKING)} the {thing}")
            holding[thing] = person
        elif roll < 0.75 and person in holding.values():
            thing = next(one for one, who in holding.items() if who == person)
            lines.append(f"{person} {rng.choice(LEAVING)} the {thing}")
            holding.pop(thing)
        elif roll < 0.85 and person in holding.values():
            thing = next(one for one, who in holding.items() if who == person)
            other = rng.choice([one for one in people if one != person])
            lines.append(f"{person} gave the {thing} to {other}")
            holding[thing] = other
        else:
            lines.append(rng.choice((
                f"{person} is in the {rng.choice(rooms)}",
                f"{person} is either in the {rooms[0]} or the {rooms[1]}",
                f"the {rooms[0]} is north of the {rooms[1]}")))
        if rng.random() < 0.55:
            lines.append(_story_question(rng, people, rooms, things,
                                         holding, visited))
    lines.append(_story_question(rng, people, rooms, things, holding,
                                 visited))
    return lines


def _story_question(rng, people, rooms, things, holding, visited) -> str:
    person = rng.choice(list(visited) or people)
    other = rng.choice([one for one in people if one != person])
    room = rng.choice((visited.get(person) or rooms))
    thing = rng.choice(list(holding) or things)
    return rng.choice((
        f"where is {person}", f"where is the {thing}",
        f"who has the {thing}", f"what is {person} carrying",
        f"is {person} in the {room}", f"is {person} in the {rooms[-1]}",
        f"how many objects is {person} carrying",
        f"where was {person} before the {room}",
        f"who gave the {thing} to {other}", f"what did {person} give to {other}",
        f"did {person} go to the {room}", "what happened",
        "what did i tell you first", f"when did {person} go to the {room}",
        f"is {person} carrying the {thing}", f"who is in the {room}",
        f"how many people are in the {room}", f"where did {person} go first",
        f"does {person} have the {thing}", f"what is north of the {rooms[1]}",
        f"why did {person} go to the {room}"))


def individual(rng: random.Random) -> list[str]:
    """One thing put down, told of and asked about."""
    vocabulary = words()
    thing = rng.choice(vocabulary["animals"] or vocabulary["things"])
    other = rng.choice(vocabulary["animals"] or vocabulary["things"])
    verbs = rng.sample(vocabulary["bare"], 3)
    adjective = rng.choice(vocabulary["adjectives"])
    kind = rng.choice(vocabulary["above"].get(thing) or vocabulary["kinds"])
    name = rng.choice(vocabulary["names"]).capitalize()
    lines = [rng.choice((f"there is a {thing}", f"i have a {thing}",
                         f"there was a {thing}"))]
    pool = [
        f"can it {verbs[0]}", f"can the {thing} {verbs[1]}",
        f"does it {verbs[2]}", f"is it {adjective}",
        f"it can't {verbs[0]}", f"it can {verbs[1]}", f"it is {adjective}",
        f"it isn't {adjective}", f"it was {gerund(verbs[2])}",
        f"it wasn't {gerund(verbs[0])}", f"its name is {name}",
        f"can {name} {verbs[0]}", "what is its name", "what is it",
        f"is it a {kind}", "is it an animal", f"why can it {verbs[1]}",
        "why", "how do you know that", "what do you know about it",
        f"there is another {thing}", f"does the {thing} {verbs[0]}",
        f"the first {thing} is {adjective}", f"is the other one {adjective}",
        f"it chased the {other}", f"did it chase the {other}",
        f"it {past(verbs[2]) or 'slept'}", f"did it {verbs[2]}",
        f"yesterday it was {adjective}", f"is it {adjective} today",
        f"what about a {other}", f"what can it do"]
    lines += rng.sample(pool, rng.randint(4, 8))
    return [articled(line) for line in lines]


def kinds(rng: random.Random) -> list[str]:
    """Questions about kinds, several to a conversation."""
    vocabulary = words()
    frames = [
        "can a {n} {v}", "can {ns} {v}", "do {ns} {v}", "is a {n} {a}",
        "what is a {n}", "what can a {n} do", "describe a {n}",
        "why can a {n} {v}", "why can't a {n} {v}",
        "what do a {n} and a {m} have in common",
        "how many kinds of {n} are there", "what is a {n} made of",
        "where do you find a {n}", "what is a {n} used for",
        "does a {n} have a {p}", "is a {n} a {k}", "tell me about {ns}",
        "what is the difference between a {n} and a {m}",
        "do you know if a {n} can {v}", "is a {n} bigger than a {m}",
        "which {ns} can {v}", "what kind of {n} is {a}", "why do {ns} {v}"]
    out = []
    for _ in range(rng.randint(3, 6)):
        thing, other = rng.sample(vocabulary["animals"] + vocabulary["objects"],
                                  2)
        filled = rng.choice(frames).format(
            n=thing, ns=plural(thing), m=other, v=rng.choice(vocabulary["bare"]),
            a=rng.choice(vocabulary["adjectives"]),
            p=rng.choice(vocabulary["parts"]),
            k=rng.choice(vocabulary["above"].get(thing)
                         or vocabulary["kinds"]))
        out.append(articled(filled))
        if rng.random() < 0.2:
            out.append(rng.choice(("why", "how do you know that",
                                   f"what about a {other}")))
    return out


def teaching(rng: random.Random) -> list[str]:
    """A kind nobody has heard of, taught, then asked about."""
    vocabulary = words()
    word = invented(rng)
    above = rng.choice(vocabulary["animals"] + ["animal", "plant", "tool"])
    verbs = rng.sample(vocabulary["bare"], 2)
    adjective = rng.choice(vocabulary["adjectives"])
    lines = [f"a {word} is a kind of {above}",
             rng.choice((f"{plural(word)} can {verbs[0]}",
                         f"{plural(word)} can't {verbs[0]}",
                         f"{plural(word)} are {adjective}"))]
    lines += rng.sample([
        f"can a {word} {verbs[0]}", f"can a {word} {verbs[1]}",
        f"is a {word} a {above}", f"what is a {word}", f"there is a {word}",
        f"can it {verbs[0]}", f"is it {adjective}", "what is it",
        f"what can a {word} do"], rng.randint(3, 5))
    return [articled(line) for line in lines]


def babi(per_task: int, longest: int = 24) -> list[tuple[str, list[str]]]:
    """bAbI's training stories, the first `per_task` of every task."""
    from research.v689 import babi as tasks

    found = []
    for task in tasks.TASKS:
        for one in tasks.stories(task, "train")[:per_task]:
            found.append((f"babi-{task}-{one.index}",
                          [line.text for line in one.lines[:longest]]))
    return found


def tests(size: int = 6) -> list[tuple[str, list[str]]]:
    """The questions v687's and v688's tests put, a few to a conversation."""
    from research.v689.reading import AUX, QUESTION_WORDS
    from research.v689.teach_reader import _test_strings

    asked = [text for text in _test_strings(("v687", "v688"))
             if text.split()[:1] and (text.split()[0].lower() in AUX
                                      or text.split()[0].lower()
                                      in QUESTION_WORDS)]
    return [(f"tests-{start}", asked[start:start + size])
            for start in range(0, len(asked), size)]


def examples() -> list[tuple[str, list[str]]]:
    from research.v689.server import EXAMPLES
    return [(f"example-{index}", one["lines"])
            for index, one in enumerate(EXAMPLES)]


DRAWN = {"story": story, "individual": individual, "kinds": kinds,
         "teaching": teaching}


def conversations(counts: dict, seed: int = 690) -> list[tuple]:
    """(key, source, lines) for every conversation to play."""
    found = [(key, "examples", lines) for key, lines in examples()]
    if counts.get("babi"):
        found += [(key, "babi", lines) for key, lines in babi(counts["babi"])]
    if counts.get("tests"):
        found += [(key, "tests", lines) for key, lines in tests()]
    for source, draw in DRAWN.items():
        rng = random.Random(f"{seed}-{source}")
        found += [(f"{source}-{index}", source, draw(rng))
                  for index in range(counts.get(source, 0))]
    return found


# -- playing them -------------------------------------------------------------------
_ASKER = None


def _start(store: str, cycles: int) -> None:
    global _ASKER
    from research.v688 import server as v688
    from research.v689.server import StoreAsker

    sys.stdout.reconfigure(errors="replace")
    service = v688.Service(Path(store), 1, cycles, teacher=False)
    _ASKER = StoreAsker(service)


def compact(turn: dict) -> dict:
    """A turn as the page gets it, without the memory view, with v688's run
    cut to its summary and v687's walk to its first steps."""
    from research.v689.server import trimmed

    turn = trimmed(dict(turn))
    discourse = turn.pop("discourse", None) or {}
    turn.pop("memory", None)
    turn["referents"] = {
        one["id"]: {key: one.get(key) for key in
                    ("description", "kind", "sense", "name", "owner",
                     "speaker", "addressee")}
        for one in discourse.get("referents") or []}
    turn["focus"] = discourse.get("focus")
    run = turn.get("run")
    if run and run.get("content"):
        content = dict(run["content"])
        content["more"] = max(0, len(content.get("items") or []) - ITEMS)
        content["items"] = (content.get("items") or [])[:ITEMS]
        run["content"] = content
    walk = turn.get("walk")
    if walk:
        turn["walk"] = {
            "verdict": walk.get("verdict"), "note": walk.get("note"),
            "steps": [{key: step.get(key) for key in
                       ("rule", "distance", "detail")}
                      for step in (walk.get("steps") or [])[:STEPS]],
            "evidence": (walk.get("evidence") or [])[:EVIDENCE]}
    for key in ("resolution", "object"):
        found = turn.get(key)
        if found and found.get("identification"):
            found["identification"] = {
                name: found["identification"].get(name)
                for name in ("wanted", "candidates", "visited")}
    return turn


def play(job: tuple) -> list[dict]:
    """One conversation, said to a fresh session a line at a time."""
    from research.v689.definitions import DefinitionMemory
    from research.v689.session import Session

    key, source, lines = job
    session = Session(_ASKER, conversation=key, example=True,
                      definitions=DefinitionMemory(None))
    out, said = [], []
    for index, line in enumerate(lines):
        started = time.time()
        try:
            turn = compact(session.say(line).as_dict())
        except Exception as bad:                    # noqa: BLE001
            turn = {"said": line, "error": f"{type(bad).__name__}: {bad}"}
        turn.update(conversation=key, source=source, index=index,
                    before=said[-4:],
                    seconds=round(time.time() - started, 3))
        said.append({"said": line,
                     "answer": (turn.get("answer") or {}).get("text", "")})
        out.append(turn)
    return out


def _kept(path: Path) -> set:
    if not path.exists():
        return set()
    found = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            try:
                found.add(json.loads(raw)["conversation"])
            except (ValueError, KeyError):
                continue
    return found


def run(counts: dict, processes: int, store: Path, cycles: int,
        path: Path = TURNS) -> Path:
    import multiprocessing

    path.parent.mkdir(parents=True, exist_ok=True)
    kept = _kept(path)
    jobs = [job for job in conversations(counts) if job[0] not in kept]
    # Sources interleaved, so a partial run has some of each.
    by_source: dict = {}
    for job in jobs:
        by_source.setdefault(job[1], []).append(job)
    jobs = [group[at] for at in range(max(map(len, by_source.values()),
                                          default=0))
            for group in by_source.values() if at < len(group)]
    print(f"{len(jobs)} conversations to play, {len(kept)} kept: "
          f"{ {source: len(group) for source, group in by_source.items()} }",
          flush=True)
    started, done, turns = time.time(), 0, 0
    with path.open("a", encoding="utf-8") as sink, multiprocessing.Pool(
            processes, initializer=_start,
            initargs=(str(store), cycles)) as pool:
        for records in pool.imap_unordered(play, jobs):
            for record in records:
                sink.write(json.dumps(record) + "\n")
            sink.flush()
            done += 1
            turns += len(records)
            if done % 25 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} conversations, {turns} turns, "
                      f"{time.time() - started:.0f}s", flush=True)
    return path


def main(argv=None) -> int:
    from research.v687 import build

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processes", type=int, default=12)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--store", type=Path, default=build.DEFAULT_STORE)
    parser.add_argument("--babi", type=int, default=15,
                        help="training stories per bAbI task")
    parser.add_argument("--story", type=int, default=500)
    parser.add_argument("--individual", type=int, default=500)
    parser.add_argument("--kinds", type=int, default=300)
    parser.add_argument("--teaching", type=int, default=120)
    parser.add_argument("--no-tests", action="store_true")
    parser.add_argument("--show", type=int, default=0,
                        help="print this many drawn conversations and stop")
    options = parser.parse_args(argv)
    counts = {"babi": options.babi, "story": options.story,
              "individual": options.individual, "kinds": options.kinds,
              "teaching": options.teaching, "tests": not options.no_tests}
    if options.show:
        rng = random.Random(1)
        for source, draw in DRAWN.items():
            for _ in range(options.show):
                print(source, draw(rng))
        return 0
    run(counts, options.processes, options.store, options.cycles)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
