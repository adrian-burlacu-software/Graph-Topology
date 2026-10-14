"""Thinking (v705): a scratchpad of thoughts in words, each the outcome of
a tool the architecture has, the next one chosen from what is on the pad
by a policy -- and every chain kept with what it came to, for a learned
policy to be taught from (expert iteration, later reinforcement).

    python -m research.v705.thinking run "make this code more generalizable"
    python -m research.v705.thinking chains          what thinking has come to

An LLM thinks by writing steps it then reads, choosing each next one from
what is written, kept or dropped by whether the line led somewhere. The
architecture's steps were in a fixed order (look, change, one call up):
it could not have an idea. Here each **tool** -- gather evidence, read how
each layer took a case, find where a case fails, look inside, build a call
of what is in reach, ask the editor, go one call up -- writes **thoughts**
(what it observed, what it supposes, what it did and what came of it), and
the **policy** reads the pad and picks the next tool. The first policy is
the order that worked, with some exploring; every chain is kept
(`state/v705-chains.jsonl`) with its **reward**: a fix that held, a cause
found, or nothing.

**Reading the layers** is how the compound-noun cause was found: the
question said `baby powder`, the parser's subject said `powder` -- a word
lost between two layers is a supposition with a place, without any check
written for it.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHAINS = ROOT / "state" / "v705-chains.jsonl"
SEED = 7053
#: the most thoughts a chain takes
LONGEST = 40
#: how often the policy tries another tool than the one it would pick
EXPLORE = 0.15
#: the cases a chain takes up at most
CASES = 3


# -- the pad ------------------------------------------------------------------------

class Pad:
    """The scratchpad: thoughts in order, each {kind, said, data}, and what
    the thinking is about now (`case`, `place`)."""

    def __init__(self, goal: str) -> None:
        self.goal = goal
        self.thoughts: list = []
        self.cases: list = []
        self.case: dict | None = None
        self.place: dict | None = None
        self.tried: list = []
        self.done_cases: list = []
        self.reward = 0.0
        self.done = False

    def think(self, kind: str, said: str, **data) -> None:
        self.thoughts.append({"kind": kind, "said": said, **data})

    def text(self) -> str:
        return "\n".join(f"[{one['kind']}] {one['said']}"
                         for one in self.thoughts)


# -- the tools ----------------------------------------------------------------------

def gather(pad: Pad, mind) -> None:
    """What fails, by its ways of looking (`evidence`): each case observed."""
    from research.v705 import evidence
    found = evidence.gather(pad.goal, mind.session)
    pad.cases = found["cases"]
    pad.think("observe", f"By {', '.join(what for _, what in found['ways'])}"
              f", {len(pad.cases)} cases fail.")
    for one in pad.cases[:4]:
        pad.think("observe", f"\"{one['said']}\" is answered "
                             f"\"{(one.get('answer') or '')[:80]}\".")
    if pad.cases:
        pad.case = pad.cases[0]


def layers(pad: Pad, mind) -> None:
    """How each layer took the case: the words it is about, as the parser,
    v688's run and the answer hold them -- a word the question said and a
    layer dropped is lost there, and that layer's function is a place."""
    case = pad.case
    if case is None:
        return
    said = case["said"]
    words = _things(said)
    parser = mind.service.pool.engines[0].parser
    parsed = parser.parse(said)
    pad.think("observe", f"The parser reads \"{said}\" as about "
                         f"\"{parsed.subject}\""
                         + (f", asked of \"{parsed.target}\"" if parsed.target
                            else "") + f" ({parsed.relation or 'no relation'}).")
    # each phrase it said, followed through every call of its own code:
    # the first to give back its last word without the rest drops it
    # (`inside.lost`) -- through any layer, an executive's too
    from research.v705 import inside
    turns = list(case.get("turns") or [said])
    for phrase in words:
        if len(phrase.split()) < 2:
            continue
        found = inside.lost(mind.session, turns, phrase)
        if found:
            place = {"function": found["function"], "file": found["file"]}
            pad.think("suppose", f"\"{phrase}\" is said, and "
                                 f"{found['function']} ({found['file']}) "
                                 f"gives back {found['returns'][:80]}: the "
                                 f"rest of it is lost there.", place=place)
            pad.place = {**place, "lost": phrase}
            return
    pad.think("observe", "No layer drops a word the question said.")


def _things(text: str) -> list:
    """The noun phrases a sentence says (`baby powder`), by spaCy."""
    from research.v691 import hearing
    words = hearing.parse(text)
    out, run = [], []
    for one in words:
        if one.tag in ("NN", "NNS", "NNP") or (
                one.dep == "compound" and one.tag.startswith("NN")):
            run.append(one.text)
        else:
            if run:
                out.append(" ".join(run))
            run = []
    if run:
        out.append(" ".join(run))
    return out


def diagnose(pad: Pad, mind) -> None:
    """Where the case fails, by the operators that took it (`doing`)."""
    from research.v704 import doing, selfmodel
    if pad.case is None:
        return
    found = doing.diagnose(pad.case, selfmodel.places())
    places = doing.findings([found])
    if not places:
        pad.think("observe", "No operator's place is found for it.")
        return
    one = places[0]
    pad.place = {"function": one["function"], "file": one["file"],
                 "case": found}
    pad.think("observe", f"It {one['how']} in {one['function']} "
                         f"({one['file']}).", place=pad.place)


def look(pad: Pad, mind) -> None:
    """Inside the place: the case said again, the function watched, a
    give-up followed down (`inside.look_inside`)."""
    from research.v705 import inside
    if pad.place is None or pad.case is None:
        return
    turns = list(pad.case.get("turns") or [pad.case["said"]])
    found = inside.look_inside(mind.session, turns, pad.place["file"],
                               pad.place["function"].split(".")[-1])
    if not found or not found.get("function"):
        pad.think("observe", f"{pad.place['function']} is not called for "
                             f"it.")
        return
    chain = []
    while found:
        chain.append(found)
        found = found.get("deeper")
    last = chain[-1]
    pad.place = {**pad.place, "function": last["function"],
                 "file": last["file"], "look": chain[0],
                 "callers": [{"function": one["function"],
                              "file": one["file"]} for one in chain[:-1]]}
    pad.think("observe", f"{last['function']} gives back {last['returns']}"
                         f" at line {last['line']} (`{last['code']}`).",
              place=pad.place)


def up(pad: Pad, mind) -> None:
    """The function that called the place."""
    callers = (pad.place or {}).get("callers") or []
    if not callers:
        pad.think("observe", "Nothing above it was followed.")
        return
    pad.place = {**pad.place, **callers[-1], "callers": callers[:-1]}
    pad.think("act", f"One call up: {callers[-1]['function']} "
                     f"({callers[-1]['file']}).")


def build(pad: Pad, mind) -> None:
    """A call of what is in reach, or the code's own way past a nothing
    elsewhere, put in where it gives up -- each run again
    (`inside.compositions`, `precedents`, `verify`)."""
    _fix(pad, mind, edit=False)


def edit(pad: Pad, mind) -> None:
    """The editor asked for the change, each answer run again."""
    _fix(pad, mind, edit=True)


def _fix(pad: Pad, mind, edit: bool) -> None:
    from research.v700 import fixing
    from research.v698.asking import Subject
    from research.v705 import inside
    place, case = pad.place, pad.case
    if not place or not case:
        return
    pad.tried.append((place["function"], "edit" if edit else "build"))
    function = inside.function_at(place["file"], place["function"])
    if function is None:
        pad.think("observe", f"{place['function']} is not one of its own "
                             f"functions it can change.")
        return
    checks = inside.cases_of(case, function)
    if not checks:
        pad.think("observe", "It has no check to run a change against.")
        return
    held = mind.held
    text = held.files.get(place["file"])
    if text is None:
        return

    def verify(after):
        return inside.verify(mind.session, checks, function, after)
    look_ = inside.looked_at(place.get("look"), function.__name__) or {}
    if edit:
        from research.v705 import talk as debug_talk
        from research.v705.teach_debugging import SAYING
        context = {"case": case["said"], "verify": verify,
                   "editor": (debug_talk.EDITOR.parent / "editor-debug2").name,
                   "saying": SAYING,
                   "said": inside.statement_of(look_, function)
                   if look_ else None,
                   "imports": inside.imports_of(function)}
        made = fixing.change(f"change {place['function']} so that it "
                             f"answers \"{case['said']}\"", held,
                             Subject("function", function.__qualname__,
                                     place["file"]), "thinking", context)
    else:
        candidates = (inside.precedents(function, look_, text) +
                      inside.compositions(function, look_, text)) \
            if look_ else []
        made = fixing.first_verified(held, place["file"], candidates, verify,
                                     "thinking", function.__qualname__) \
            if candidates else {"status": "unchanged",
                                "said": "Nothing in reach to build from."}
    met = made["status"] in ("changed", "partly")
    pad.think("result", " ".join(made["said"].split())[:300],
              met=met, place=dict(place))
    if met:
        pad.reward = 1.0
        pad.done = True


TOOLS = {"gather": gather, "layers": layers, "diagnose": diagnose,
         "look": look, "build": build, "edit": edit, "up": up}


# -- the policy -----------------------------------------------------------------------

def heuristic(pad: Pad, rng: random.Random) -> str | None:
    """The order that worked, read off the pad: evidence first; how the
    layers took the case; a place by its operators where no layer dropped
    a word; inside it; built, then edited; one call up, and again."""
    used = [one.get("tool") for one in pad.thoughts if one.get("tool")]
    if not pad.cases:
        return "gather" if "gather" not in used else None
    since = _since_case(pad)
    if "layers" not in since:
        return "layers"
    if pad.place is None:
        if "diagnose" not in since:
            return "diagnose"
        pad.place = {"function": "", "file": "", "callers": []}
        pad.tried = [("", "build"), ("", "edit")]
    if "look" not in since and not pad.place.get("lost") and             pad.place.get("function"):
        return "look"
    here = pad.place["function"]
    if (here, "build") not in pad.tried and not pad.place.get("lost"):
        return "build"
    if (here, "edit") not in pad.tried:
        return "edit"
    if pad.place.get("callers"):
        return "up"
    # nothing left to try for this case: the next one, thought about anew
    left = [one for one in pad.cases if one is not pad.case
            and one["said"] not in pad.done_cases]
    if left and len(pad.done_cases) < CASES:
        pad.done_cases.append(pad.case["said"])
        pad.case, pad.place, pad.tried = left[0], None, []
        pad.think("act", f"Next case: \"{pad.case['said']}\".")
        return "layers"
    return None


def _since_case(pad: Pad) -> list:
    """The tools used since the case now thought about was taken up."""
    out = []
    for one in reversed(pad.thoughts):
        if one["kind"] == "act" and one["said"].startswith("Next case"):
            break
        if one.get("tool"):
            out.append(one["tool"])
    return out


def choose(pad: Pad, rng: random.Random) -> str | None:
    picked = heuristic(pad, rng)
    if picked in ("build", "edit", "up", "look") and rng.random() < EXPLORE:
        # trying another than the order would: what a learned policy
        # needs to see the worth of
        others = [one for one in ("build", "edit", "up", "look")
                  if one != picked]
        picked = rng.choice(others)
    return picked


# -- thinking -------------------------------------------------------------------

class Mind:
    """What the tools use: a session to ask, the service under it, the
    code held (in memory: nothing it tries is written to disk)."""

    def __init__(self) -> None:
        from research.v687 import build as store
        from research.v688 import server as v688
        from research.v689.definitions import DefinitionMemory
        from research.v689.session import Session
        from research.v690.server import PinnedAsker
        from research.v698.project import Project
        self.service = v688.Service(store.DEFAULT_STORE, 1, 8, teacher=False)
        self.session = Session(PinnedAsker(self.service), None, "thinking",
                               True, DefinitionMemory(None))
        self.held = Project("thinking")
        self.held.put({path.relative_to(ROOT).as_posix():
                       path.read_text(encoding="utf-8")
                       for path in (ROOT / "research").rglob("*.py")
                       if "__pycache__" not in path.parts})


def think(goal: str, mind: Mind, rng: random.Random | None = None,
          policy=choose) -> Pad:
    """A goal thought about: tools chosen from the pad until it comes to
    something or nothing is left to try; the chain kept."""
    rng = rng or random.Random(SEED)
    pad = Pad(goal)
    pad.think("goal", goal)
    while not pad.done and len(pad.thoughts) < LONGEST:
        tool = policy(pad, rng)
        if tool is None:
            break
        before = len(pad.thoughts)
        try:
            TOOLS[tool](pad, mind)
        except Exception as bad:                    # noqa: BLE001
            pad.think("observe", f"{tool} failed: {type(bad).__name__}: "
                                 f"{bad}"[:200])
        if len(pad.thoughts) == before:
            pad.think("observe", f"{tool} found nothing.")
        pad.thoughts[before]["tool"] = tool
    if not pad.done and any(one["kind"] == "suppose"
                            for one in pad.thoughts):
        # a cause found, not fixed: worth something, less than a fix
        pad.reward = max(pad.reward, 0.3)
    kept(pad)
    return pad


def kept(pad: Pad) -> None:
    CHAINS.parent.mkdir(parents=True, exist_ok=True)
    with CHAINS.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "at": time.strftime("%Y-%m-%d %H:%M"), "goal": pad.goal,
            "reward": pad.reward,
            "thoughts": [{key: value for key, value in one.items()
                          if key in ("kind", "said", "tool", "met")}
                         for one in pad.thoughts]}) + "\n")


def examples(least: float = 0.3) -> list:
    """What a proposer of the next thought is taught (expert iteration):
    of each chain that came to something (`least`: a cause found, a fix
    that held), each tool it chose, with the pad as it was when chosen --
    {pad, tool, reward}. A chain that came to nothing teaches nothing yet;
    kept, it is what reinforcement will weigh against."""
    if not CHAINS.exists():
        return []
    out = []
    for line in CHAINS.open(encoding="utf-8"):
        chain = json.loads(line)
        if chain["reward"] < least:
            continue
        thoughts = chain["thoughts"]
        for at, one in enumerate(thoughts):
            if not one.get("tool"):
                continue
            pad = "\n".join(f"[{t['kind']}] {t['said']}"
                            for t in thoughts[:at])
            out.append({"pad": pad, "tool": one["tool"],
                        "reward": chain["reward"]})
    return out


def chains() -> dict:
    if not CHAINS.exists():
        return {}
    rows = [json.loads(line) for line in CHAINS.open(encoding="utf-8")]
    return {"chains": len(rows),
            "fixed": sum(one["reward"] >= 1 for one in rows),
            "cause found": sum(0 < one["reward"] < 1 for one in rows),
            "nothing": sum(one["reward"] == 0 for one in rows)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("run", "chains"))
    parser.add_argument("goal", nargs="?", default="")
    options = parser.parse_args(argv)
    if options.job == "chains":
        print(json.dumps(chains(), indent=1))
        return 0
    pad = think(options.goal, Mind())
    print(pad.text())
    print(f"reward {pad.reward}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
