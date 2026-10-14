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
#: the found tools a chain calls where nothing fails
USES = 4


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
        #: the found tools called (`use`)
        self.used: set = set()
        #: the places a case was localized to (`localize`)
        self.places: list = []
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


def source(pad: Pad, mind) -> None:
    """Where the reply's words come from (`measures.made`): each function
    of its own code that gave back a piece of it, the most first -- the
    first the place, the rest above it to go to."""
    from research.v705 import measures
    case = pad.case
    if case is None or not case.get("answer"):
        return
    found = measures.made(mind.session, list(case.get("turns") or
                                             [case["said"]]),
                          case["answer"])
    if not found:
        pad.think("observe", "No function of its own gave back a piece of "
                             "the reply.")
        return
    pad.think("observe", "The reply's words come from " + ", ".join(
        f"{one['function']} ({one['gave']} characters)" for one in found)
        + ".")
    # and what those were made from (`measures.feeding`): where what is
    # said is chosen, as templates filled later -- nearest the goal first
    import inspect
    from research.v701.likeness import nearest
    from research.v705 import inside
    seen = {(one["function"], one["file"]) for one in found}
    fed: dict = {}
    for one in found:
        function = inside.function_at(one["file"], one["function"])
        for made in measures.feeding(function) if function else []:
            key = (made.__qualname__,
                   inside._relative(inspect.getsourcefile(made)))
            if key not in seen and key not in fed:
                doc = " ".join((made.__doc__ or "").split("\n\n")[0].split())
                fed[key] = f"{made.__name__.replace('_', ' ')}: {doc}"
    near = dict((said, score) for score, said in
                nearest(pad.goal, sorted(set(fed.values())))) if fed else {}
    ranked = sorted(fed, key=lambda key: -near[fed[key]])
    if ranked:
        pad.think("observe", "They are made from " + ", ".join(
            function for function, _ in ranked[:measures.PLACES]) + ".")
    # the one that gave back the most (the reply joined), then what it is
    # made of, then the rest that only passed it on (`_reply`, `_act`)
    given = [{"function": one["function"], "file": one["file"]}
             for one in found]
    places = given[:1] + [{"function": function, "file": path}
                          for function, path in ranked] + given[1:]
    places = [one for one in places if not judging(one)][:measures.PLACES]
    pad.place = {**places[0], "callers": list(reversed(places[1:]))}
    pad.think("suppose", f"What it says is chosen in "
                         f"{places[0]['function']} or what it is made of.",
              place=dict(pad.place))


#: what judges a change, never a place to change: set to a bug from its
#: history, the thinking took out the line of its test runner that runs
#: the test (`commits.passes`), and every test passed -- the files that
#: think, check and measure, and the functions of `inside` a case is said
#: again and judged by
JUDGING_FILES = ("research/v705/thinking.py", "research/v705/commits.py",
                 "research/v705/measures.py", "research/v705/evidence.py",
                 "research/v705/tools.py")
JUDGING = {("research/v705/inside.py", name) for name in (
    "_scratch", "replay", "changed_function", "answered", "_as_page_turn",
    "cases_of", "_holds_now", "verify", "_module", "_owner")}


def judging(place: dict) -> bool:
    """Whether a place is of what judges a change."""
    return place.get("file") in JUDGING_FILES or (
        place.get("file"), place.get("function")) in JUDGING


def localize(pad: Pad, mind) -> None:
    """Where failing tests go wrong (`commits.localize`): the functions
    they run, against the passing tests beside them, by Ochiai -- the
    first the place, the rest to go to."""
    from research.v705 import commits
    found = [one for one in commits.localize(pad.case, most=12)
             if not judging(one)][:8]
    pad.places = found
    if not found:
        pad.think("observe", "The failing tests run nothing of its own.")
        return
    pad.place = {"function": found[0]["function"], "file": found[0]["file"],
                 "callers": [{"function": one["function"],
                              "file": one["file"]}
                             for one in reversed(found[1:])]}
    pad.think("suppose", "The failing tests run, more than the passing "
                         "ones, " + ", ".join(
                             f"{one['function']} ({one['score']})"
                             for one in found[:5]) + ".",
              place=dict(pad.place))


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
    if judging(place):
        pad.think("observe", f"{place['function']} judges the change; it "
                             f"is not changed to pass.")
        return
    function = inside.function_at(place["file"], place["function"])
    if function is None:
        pad.think("observe", f"{place['function']} is not one of its own "
                             f"functions it can change.")
        return
    held = mind.held
    text = held.files.get(place["file"])
    if text is None:
        return
    if case.get("tests"):
        # a case its tests say is right (`commits`): run again, they pass
        from research.v705 import commits
        verify = commits.verifier(case, function)
    else:
        checks = inside.cases_of(case, function)
        if not checks:
            pad.think("observe", "It has no check to run a change "
                                 "against.")
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
        # a goal and a place, as its history's changes are taught
        # (`commits.edits`): the message, and which function it is in
        if case.get("tests"):
            from research.v705 import commits
            asked = commits.statement(
                {"message": pad.goal, "function": place["function"],
                 "file": place["file"], "tests": case["tests"]},
                commits.failure(case["tests"][0]))
        asked = asked if case.get("tests") else (
            f"change {place['function']} so that it answers "
            f"\"{case['said']}\"")
        made = fixing.change(asked, held,
                             Subject("function", function.__qualname__,
                                     place["file"]), "thinking", context)
    else:
        if case.get("measure") or case.get("tests"):
            # how a reply reads, or what tests say: the function with each
            # thing it does made less, or taken out
            from research.v705 import measures
            candidates = measures.smaller(function, text) + \
                measures.dropped(function, text)
        else:
            candidates = (inside.precedents(function, look_, text) +
                          inside.compositions(function, look_, text)) \
                if look_ else []
        made = fixing.first_verified(held, place["file"], candidates, verify,
                                     "thinking", function.__qualname__,
                                     how="it is now") \
            if candidates else {"status": "unchanged",
                                "said": "Nothing in reach to build from."}
    met = made["status"] in ("changed", "partly")
    pad.think("result", " ".join(made["said"].split())[:300],
              met=met, place=dict(place))
    if met:
        pad.reward = 1.0
        pad.done = True


def use(pad: Pad, mind) -> None:
    """A tool found in its own code (`tools`): of those it can call with
    what the pad holds, the one whose docstring is nearest what the
    thinking is about, run guarded; what it gave back observed."""
    from research.v705 import tools
    held = tools.holds(pad, mind)
    about = (pad.case or {}).get("said") or pad.goal
    last = next((one["said"] for one in reversed(pad.thoughts)
                 if one["kind"] in ("observe", "suppose")), "")
    near = tools.ranked(f"{about} {last}"[:400], held, pad.used)
    if not near:
        pad.think("observe", "No tool of its own is left to call with what "
                             "it holds.")
        return
    likeness, tool = near[0]
    name = f"{tool['module']}.{tool['name']}"
    pad.used.add(name)
    arguments = tools.filled(tool, held)
    pad.think("act", f"Calling {name} ({likeness:.2f}): {tool['doc'][:120]}",
              called=name)
    out = tools.called(tool, arguments)
    if out.get("slow"):
        pad.think("observe", f"{name} took longer than {tools.SLOW:.0f}s; "
                             f"left.")
    elif "failed" in out:
        pad.think("observe", f"{name} failed: {out['failed']}"[:200])
    else:
        pad.think("observe", f"{name} gave back "
                             f"{tools.shown(out['gave'])}.", called=name)


TOOLS = {"gather": gather, "layers": layers, "diagnose": diagnose,
         "look": look, "build": build, "edit": edit, "up": up, "use": use,
         "source": source, "localize": localize}


# -- the policy -----------------------------------------------------------------------

def heuristic(pad: Pad, rng: random.Random) -> str | None:
    """The order that worked, read off the pad: evidence first; how the
    layers took the case; a place by its operators where no layer dropped
    a word; inside it; built, then edited; one call up, and again."""
    used = [one.get("tool") for one in pad.thoughts if one.get("tool")]
    if not pad.cases:
        # nothing fails by its ways of looking: its own tools, read with
        if "gather" not in used:
            return "gather"
        return "use" if used.count("use") < USES else None
    since = _since_case(pad)
    if pad.case.get("tests"):
        # failing tests: where by spectrum, each place built, then edited,
        # then the next
        if "localize" not in since:
            return "localize"
        if pad.place is None:
            return None
        here = pad.place["function"]
        if (here, "build") not in pad.tried:
            return "build"
        if (here, "edit") not in pad.tried:
            return "edit"
        return "up" if pad.place.get("callers") else None
    if pad.case.get("measure"):
        # a reply's measure: where its words come from, each place built
        # without one thing in turn, then the next place
        if "source" not in since:
            return "source"
        if pad.place is None:
            return None
        if (pad.place["function"], "build") not in pad.tried:
            return "build"
        return "up" if pad.place.get("callers") else None
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
    if "use" not in since:
        return "use"
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
    if pad.case and (pad.case.get("measure") or pad.case.get("tests")):
        # a measure's places are set by where its words come from: a look
        # sets places of its own, an editor answers a case it is not
        if picked in ("build", "up") and rng.random() < EXPLORE:
            picked = rng.choice([one for one in ("build", "up", "use")
                                 if one != picked])
        return picked
    if picked in ("build", "edit", "up", "look") and rng.random() < EXPLORE:
        # trying another than the order would: what a learned policy
        # needs to see the worth of
        others = [one for one in ("build", "edit", "up", "look", "use")
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


class Held:
    """The mind of a turn (`page`): the conversation's own session, the
    service under it and the code it holds."""

    def __init__(self, session, held) -> None:
        self.session = session
        self.service = session.asker.service
        self.held = held


def think(goal: str, mind: Mind, rng: random.Random | None = None,
          policy=choose, seconds: float | None = None) -> Pad:
    """A goal thought about: tools chosen from the pad until it comes to
    something, nothing is left to try or `seconds` are up; the chain
    kept."""
    rng = rng or random.Random(SEED)
    pad = Pad(goal)
    pad.think("goal", goal)
    return _thought(pad, mind, rng, policy, seconds)


def think_case(goal: str, case: dict, mind, rng: random.Random | None = None,
               seconds: float | None = None) -> Pad:
    """A case given, not gathered (a bug from its history, `commits`):
    thought about as `think` does, from the case on."""
    rng = rng or random.Random(SEED)
    pad = Pad(goal)
    pad.think("goal", goal)
    pad.cases, pad.case = [case], case
    pad.think("observe", f"{len(case.get('tests') or [])} tests fail: "
                         + ", ".join(one["test"] for one in
                                     case.get("tests") or []) + ".")
    return _thought(pad, mind, rng, choose, seconds)


def _thought(pad: Pad, mind, rng, policy, seconds) -> Pad:
    """Tools chosen from the pad until it comes to something, nothing is
    left to try or `seconds` are up; the chain kept."""
    until = time.monotonic() + seconds if seconds else None
    while not pad.done and len(pad.thoughts) < LONGEST and (
            until is None or time.monotonic() < until):
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
    commit = (pad.case or {}).get("commit")
    with CHAINS.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "at": time.strftime("%Y-%m-%d %H:%M"), "goal": pad.goal,
            "reward": pad.reward,
            "thoughts": [{key: value for key, value in one.items()
                          if key in ("kind", "said", "tool", "met")}
                         for one in pad.thoughts],
            # a bug from its history: where the commit says the fix was,
            # and where the thinking looked -- a chain in hindsight
            **({"commit": {"sha": commit["sha"], "file": commit["file"],
                           "function": commit["function"]},
                "places": [one["function"] for one in pad.places]}
               if commit else {})}) + "\n")


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
