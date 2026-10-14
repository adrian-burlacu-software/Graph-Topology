"""Ideas the architecture has about itself (v705): what might be wrong, or
missing, and a cheap way to find out -- each kept with what came of it, so
the ways of having ideas that pay are the ones it uses more.

    python -m research.v705.ideas knowledge     what it knows but cannot say
    python -m research.v705.ideas ledger        its ideas so far, by source

A fixed test (the conversation test, the suite) only knows what someone
wrote into it, and the suite passes. An **idea** is a guess with its own
evidence. The first source here is **its knowledge against what it can
say**: its store holds facts by relation (made of, used for, at location
...); asked of it, each should be answered as the store holds it -- so the
store is the answer key, and no one writes the checks.

**How it asks**: as it was taught questions are asked -- its reader's
taught questions of each relation (`llm/reader-data`, the `parse` task:
`what is a product made of`, roles marked) are its frames; the thing a
frame is about is put in their place, and a question is kept only where
its own reader reads it back as that relation and form. **What comes of
it**: of a kind of question (a relation, wh or yes/no) answered some of
the time, the misses are **faults** -- cases for the look inside
(`doing`, `inside`); of one never answered, an **ability** it lacks --
"answer what things are used for" -- with its cases as the measure.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAUGHT = ROOT / "llm" / "reader-data" / "train.jsonl"
LEDGER = ROOT / "state" / "v705-ideas.jsonl"
FOUND = ROOT / "state" / "v705-knowledge.json"
SEED = 7052
#: the relations asked of it, and how much of the store answers each
RELATIONS = ("made_of", "used_for", "at_location", "has_part", "part_of",
             "capable_of", "causes", "has_prerequisite", "desires",
             "has_property", "is_a")
#: frames of a relation and form at most, facts asked of each kind
FRAMES, ASKED = 6, 60
#: the shortest a frame is, and the longest
SHORTEST, LONGEST = 3, 8
#: how sure its reader must be that a question reads as the kind it is of
READ_FLOOR = 0.6
#: how sure its reader must be of a frame's relation
SURE = 0.9
#: a concept is asked about where WordNet's tagged text uses its word
USED = 3
#: a kind of question is an ability it lacks when none of this many is
#: answered
LACKING = 4
#: what it answered right of a kind, guarding a fix of the rest, at most
GUARDS = 4


# -- how it asks ----------------------------------------------------------------------

def frames(most: int = FRAMES) -> dict:
    """(relation, polar) -> the questions people wrote of that kind (the
    conversation test's, `v704.suite`), as its reader reads them: the
    relation read surely, the word it is about (`SUBJ`) and, asked yes or
    no, what is asked of it (`TARGET`) -- a singular noun it is about, so
    a thing put in its place reads as said. Its reader's own taught
    questions were the frames first; their labels are noisy (`is a hall
    to the pawnshop`, taught 236 times as is a) and it asked `What a
    commuter of person is she?`."""
    from research import encoder
    from research.v691 import hearing
    from research.v692.corpus import words
    from research.v704 import suite
    out: dict = {}
    for item in suite.SUITE:
        if item.project or len(item.turns) != 1:
            continue
        said = words(item.turns[0])
        found = encoder.read(said, heads=("relation", "polar", "parse"))
        relation, chance = found["relation"][0]
        polar = found["polar"][0][0]
        if relation not in RELATIONS or chance < SURE:
            continue
        roles = [str(one) for one in found["parse"]]
        subject = _span(roles, ("SUBJ",))
        target = _span([one.replace("TARGET-VERB", "TARGET")
                        for one in roles], ("TARGET",))
        if subject is None or (polar == "yes") != (target is not None):
            continue
        parsed = hearing.parse(" ".join(said))
        tags = [one.tag for one in parsed]
        # the word it is about is the question's subject, by its parse:
        # `candle` of `how can i light a candle` is what is lit, and put
        # in its place, `How can i light a trim?`
        if len(tags) != len(said) or not all(
                tag == "NN" for tag in tags[subject[0]:subject[1]]) or \
                not parsed[subject[1] - 1].dep.startswith("nsubj"):
            continue
        # nothing else in it is a thing: `what is the capital of france`
        # filled asked `What is a glass of france?`
        slots = set(range(*subject)) | (set(range(*target)) if target
                                        else set())
        if any(tag.startswith(("NN", "JJ")) for at, tag in enumerate(tags)
               if at not in slots):
            continue
        out.setdefault((relation, polar), []).append({
            "words": [one.lower() for one in said if one not in "?.!"],
            "subject": subject, "target": target,
            "determined": subject[0] > 0 and tags[subject[0] - 1] == "DT",
            "taught": 1})
    return {key: held[:most] for key, held in out.items()}


def _a_thing(tags: list, span: tuple) -> bool:
    start, end = span
    return start > 0 and tags[start - 1] == "DT" and \
        all(tag in ("NN", "NNS") for tag in tags[start:end])


def _span(roles: list, wanted: tuple) -> tuple | None:
    at = [index for index, role in enumerate(roles) if role in wanted]
    if not at or at != list(range(at[0], at[-1] + 1)):
        return None
    return at[0], at[-1] + 1


def filled(frame: dict, subject: str, target: str | None) -> str:
    words = list(frame["words"])
    spans = [(frame["subject"], subject)]
    if frame["target"] is not None and target is not None:
        spans.append((frame["target"], target))
    for (start, end), text in sorted(spans, reverse=True):
        words[start:end] = text.split()
    said = " ".join(words)
    return said[0].upper() + said[1:] + "?"


def reads_as(question: str, relation: str, polar: str, subject: str,
             target: str | None) -> bool:
    """Whether its own reader reads a question as the kind it was made as
    -- and as about what was put in it: the thing's words read as what it
    is about (`SUBJ`), the fact's as what is asked of it (`TARGET`).
    `How fast can a cheetah a detergent?` read as capable of, and was not
    about a detergent."""
    from research import encoder
    from research.v692.corpus import words
    said = words(question)
    try:
        found = encoder.read(said, heads=("relation", "polar", "parse"))
    except Exception:                               # noqa: BLE001
        return False
    (got, chance), = found["relation"][:1]
    if got != relation or chance < READ_FLOOR or \
            found["polar"][0][0] != polar:
        return False
    roles = [one[0] if isinstance(one, (list, tuple)) else one
             for one in found["parse"]]
    lowered = [one.lower() for one in said]

    def read_as(text: str, role: str) -> bool:
        # its head word: the reader marks `powder` of `baby powder`, as it
        # was taught (`beef plant`: `plant`)
        part = text.lower().split()
        for at in range(len(lowered) - len(part) + 1):
            if lowered[at:at + len(part)] == part:
                head = at + len(part) - 1
                return head < len(roles) and role in str(roles[head])
        return False
    return read_as(subject, "SUBJ") and (
        target is None or read_as(target, "TARGET"))


# -- what it knows ---------------------------------------------------------------

def _word(node: str) -> str:
    """`ashtray.n.01` -> `ashtray`; a plain object as it is."""
    return node.split(".")[0] if re.search(r"\.[nvas]\.\d\d$", node) \
        else node


def _used(word: str) -> bool:
    try:
        from nltk.corpus import wordnet as wn
    except Exception:                               # noqa: BLE001
        return True
    return sum(lemma.count() for sense in wn.synsets(word.replace(" ", "_"))
               for lemma in sense.lemmas()
               if lemma.name().lower() == word.replace(" ", "_")) >= USED


def facts(relation: str, most: int, rng: random.Random) -> list:
    """Facts of a relation the store holds surely, of words people use:
    [(concept, object)]."""
    from research.v687 import build
    store = sqlite3.connect(f"file:{build.DEFAULT_STORE}?mode=ro", uri=True)
    if relation == "is_a":
        rows = store.execute("SELECT child, parent FROM taxonomy WHERE "
                             "child LIKE '%.n.01' LIMIT 20000").fetchall()
    else:
        # WordNet's part rows run the other way round (`cell has_part
        # organism`): parts from the other sources only
        rows = store.execute(
            "SELECT concept, object FROM facts WHERE relation = ? AND "
            "confidence >= 0.9 AND (? NOT IN ('has_part', 'part_of') OR "
            "source != 'wordnet') LIMIT 20000",
            (relation, relation)).fetchall()
    store.close()
    rng.shuffle(rows)
    out, seen = [], set()
    for concept, object_ in rows:
        subject, target = _word(concept), _word(object_)
        if subject in seen or len(target.split()) > 2 or \
                len(subject.split()) > 2 or \
                not all(_used(one) for one in subject.split()):
            continue
        seen.add(subject)
        out.append((subject, target))
        if len(out) >= most:
            break
    return out


# -- asking itself ----------------------------------------------------------------

def _article(word: str) -> str:
    return ("an " if word[:1] in "aeiou" else "a ") + word


def questions(rng: random.Random) -> list:
    """Each kind of question, asked of facts it holds: [{relation, polar,
    question, subject, target}], only those its reader reads as asked."""
    made = frames()
    out = []
    for relation in RELATIONS:
        held = facts(relation, ASKED * 3, rng)
        for polar in ("no", "yes"):
            shapes = made.get((relation, polar)) or []
            if not shapes:
                continue
            kept = 0
            for subject, target in held:
                if kept >= ASKED:
                    break
                for frame in shapes:
                    # its own determiner made `a` or `an` for the thing put
                    # in; one with none (`what is ice made of`) given one
                    said_subject = _article(subject)
                    if frame.get("determined"):
                        frame = {**frame, "words": frame["words"][
                            :frame["subject"][0] - 1] + frame["words"][
                            frame["subject"][0]:], "subject": (
                            frame["subject"][0] - 1, frame["subject"][1] - 1),
                            "target": None if frame["target"] is None else (
                                frame["target"][0] - 1,
                                frame["target"][1] - 1)}
                    said_target = target if polar == "yes" else None
                    # the target's determiner made again too (`a act`)
                    at = (frame["target"] or (0, 0))[0]
                    if said_target and at and \
                            frame["words"][at - 1] in ("a", "an"):
                        frame = {**frame, "words": frame["words"][:at - 1] +
                                 frame["words"][at:], "target": (
                                     at - 1, frame["target"][1] - 1)}
                        said_target = _article(said_target)
                    question = filled(frame, said_subject, said_target)
                    if reads_as(question, relation, polar, subject,
                                said_target):
                        out.append({"relation": relation, "polar": polar,
                                    "question": question,
                                    "subject": subject, "target": target})
                        kept += 1
                        break
    return out


def misread() -> list:
    """Its reader against what it was taught: the ways each kind of
    question was taught most, read back as written -- where it reads one
    as another kind (`is a wheel part of a car`, taught 71 times, read as
    none), an idea about its reading: [{relation, polar, question, read}]."""
    from research import encoder
    from research.v692.corpus import words
    out = []
    for (relation, polar), shapes in sorted(frames().items()):
        for frame in shapes[:2]:
            question = " ".join(frame["words"])
            found = encoder.read(words(question),
                                 heads=("relation", "polar"))
            got, chance = found["relation"][0]
            if got != relation or found["polar"][0][0] != polar:
                out.append({"relation": relation, "polar": polar,
                            "question": question, "taught": frame["taught"],
                            "read": f"{got} ({chance:.2f}), "
                                    f"{found['polar'][0][0]}"})
    return out


def answered(session, one: dict) -> tuple:
    """(right, what it said): a wh-question answered with what the store
    holds, a yes or no question about a fact it holds answered yes."""
    from research.v704 import converse
    from research.v705 import inside
    check = converse.stance("yes") if one["polar"] == "yes" else \
        converse.says(*one["target"].split()[-1:])
    turn = inside.answered(session, [one["question"]])
    said = " ".join(((turn.answer or {}).get("text") or "").split())
    return bool(check(inside._as_page_turn(turn), None)), said[:120], \
        check, turn


def knowledge(session, rng: random.Random | None = None) -> list:
    """Its ideas from its knowledge against what it can say: each kind of
    question with what came of asking it -- a fault (some answered, the
    rest cases to fix) or an ability it lacks (none answered)."""
    from research.v705 import evidence
    rng = rng or random.Random(SEED)
    kinds: dict = {}
    for one in questions(rng):
        right, said, check, turn = answered(session, one)
        kind = kinds.setdefault((one["relation"], one["polar"]),
                                {"asked": 0, "right": 0, "cases": [],
                                 "held": []})
        kind["asked"] += 1
        kind["right"] += right
        if right:
            kind["held"].append(([one["question"]], check,
                                 one["polar"], one["target"]))
        else:
            kind["cases"].append(evidence.case_of(
                turn, [one["question"]], check, area=one["relation"],
                expected=one["target"], polar=one["polar"]))
    ideas = []
    for (relation, polar), kind in kinds.items():
        # what it answered right of the kind guards a fix of the rest: it
        # must still answer them (`inside.cases_of`)
        for case in kind["cases"]:
            case["guards"] = [(turns, check) for turns, check, _, _ in
                              kind["held"][:GUARDS]]
            case["guarded"] = [(turns[0], polar, target) for turns, _,
                               polar, target in kind["held"][:GUARDS]]
        form = "yes or no" if polar == "yes" else "asked what"
        if kind["right"] == kind["asked"]:
            continue
        lacking = kind["right"] == 0 and kind["asked"] >= LACKING
        ideas.append({
            "source": "knowledge", "kind": "ability" if lacking
            else "fault", "relation": relation, "polar": polar,
            "said": (f"answer {relation.replace('_', ' ')} questions "
                     f"({form}) -- none of {kind['asked']} answered"
                     if lacking else
                     f"{relation.replace('_', ' ')} questions ({form}): "
                     f"{kind['right']} of {kind['asked']} answered as its "
                     f"store holds"),
            "asked": kind["asked"], "right": kind["right"],
            "cases": kind["cases"]})
    for one in misread():
        ideas.append({
            "source": "knowledge", "kind": "reading",
            "relation": one["relation"], "polar": one["polar"],
            "said": (f"its reader reads \"{one['question']}\" -- taught "
                     f"{one['taught']} times as {one['relation']} -- as "
                     f"{one['read']}"),
            "asked": 1, "right": 0, "cases": []})
    ideas.sort(key=lambda one: (("fault", "ability", "reading").index(
        one["kind"]), one["right"] - one["asked"]))
    return ideas


# -- its own code: what the failing cases run that the passing do not ------------------

#: the lines a code idea is of at most, and the least a line must score
SUSPECTED, SUSPECT = 8, 0.5


def covered(session, turns: list) -> set:
    """The lines of its own code saying the turns runs: {(file, function,
    line)}, by `sys.monitoring` -- each line reported once, then let be."""
    from research.v705 import inside
    monitoring = sys.monitoring
    tool = monitoring.COVERAGE_ID
    seen: set = set()
    places: dict = {}

    def line(code, number):
        name = code.co_filename
        path = places.get(name, False)
        if path is False:
            path = inside._relative(name) if "research" in name else None
            if path and ("/test" in path or not path.startswith("research/")):
                path = None
            places[name] = path
        if path:
            seen.add((path, getattr(code, "co_qualname", code.co_name),
                      number))
        return monitoring.DISABLE
    monitoring.use_tool_id(tool, "v705-ideas")
    try:
        monitoring.register_callback(tool, monitoring.events.LINE, line)
        monitoring.set_events(tool, monitoring.events.LINE)
        monitoring.restart_events()
        inside.answered(session, turns)
    finally:
        monitoring.set_events(tool, 0)
        monitoring.register_callback(tool, monitoring.events.LINE, None)
        monitoring.free_tool_id(tool)
    return seen


def spectrum(session, most: int = 0) -> list:
    """Ideas from its own code: the conversation test's cases said again
    (those said without a project), the lines each runs, and which lines
    the failing ones run that the passing ones do not -- by Ochiai, as
    `debugging.suspects` scores a program's lines. Each function holding
    suspected lines an idea, with the failing cases that run them."""
    import math
    from research.v704 import suite
    from research.v705 import evidence, inside
    runs = []
    items = [one for one in suite.SUITE if not one.project]
    for item in items[:most or None]:
        try:
            lines = covered(session, item.turns)
            holds = inside._holds_now(session, (item.turns, item.check,
                                                True))
        except Exception:                           # noqa: BLE001
            continue
        runs.append((item, lines, holds))
    failing = sum(1 for _, _, holds in runs if not holds)
    if not failing or failing == len(runs):
        return []
    by_line: dict = {}
    for item, lines, holds in runs:
        for one in lines:
            counts = by_line.setdefault(one, [0, 0])
            counts[0 if not holds else 1] += 1
    scored = sorted(((bad / math.sqrt(failing * (bad + good)), one)
                     for one, (bad, good) in by_line.items() if bad),
                    reverse=True)
    by_function: dict = {}
    for score, (path, function, number) in scored:
        if score < SUSPECT:
            break
        held = by_function.setdefault((path, function), [])
        if len(held) < 3:
            held.append((number, round(score, 2)))
        if len(by_function) >= SUSPECTED:
            break
    ideas = []
    for (path, function), lines in by_function.items():
        cases = [evidence.case_of(inside.answered(session, item.turns),
                                  item.turns, item.check, area=item.area)
                 for item, ran, holds in runs if not holds and any(
                     (path, function, number) in ran
                     for number, _ in lines)][:4]
        ideas.append({
            "source": "code", "kind": "fault", "relation": "",
            "polar": "", "place": {"function": function, "file": path},
            "said": (f"{function} ({path}) runs for failing cases more "
                     f"than for passing ones: lines "
                     f"{', '.join(str(number) for number, _ in lines)} "
                     f"(Ochiai {lines[0][1]})"),
            "asked": len(runs), "right": len(runs) - failing,
            "cases": cases})
    return ideas


# -- what was found, kept for a goal ------------------------------------------------

def _check(polar: str, expected: str):
    from research.v704 import converse
    return converse.stance("yes") if polar == "yes" else \
        converse.says(*expected.split()[-1:])


def saved(found: list) -> None:
    """What asking itself found, kept for a goal to read
    (`evidence.known`): asking takes minutes, a turn cannot."""
    keep = ("said", "turns", "n", "fired", "acts", "answer", "part",
            "area", "expected", "polar", "guarded")
    FOUND.parent.mkdir(parents=True, exist_ok=True)
    FOUND.write_text(json.dumps({"at": time.strftime("%Y-%m-%d %H:%M"),
                                 "ideas": [{**{key: value for key, value in
                                               idea.items() if key != "cases"},
                                            "cases": [{key: one[key] for key in
                                                       keep if key in one}
                                                      for one in idea["cases"]]}
                                           for idea in found]}, indent=1),
                     encoding="utf-8")


def loaded() -> list | None:
    """What asking itself found last (`saved`), each case with its check
    and guards made again; None where it has not asked."""
    if not FOUND.exists():
        return None
    out = json.loads(FOUND.read_text(encoding="utf-8"))["ideas"]
    for idea in out:
        for one in idea["cases"]:
            one["check"] = _check(one["polar"], one["expected"])
            one["guards"] = [([question], _check(polar, expected))
                             for question, polar, expected in
                             one.get("guarded") or ()]
    return out


# -- the ledger --------------------------------------------------------------------

def kept(idea: dict, outcome: str = "found") -> None:
    """An idea and what came of it (`found`, `fixed`, `held`, `nothing`),
    for which ways of having ideas pay (`ledger`)."""
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "at": time.strftime("%Y-%m-%d %H:%M"), "outcome": outcome,
            **{key: value for key, value in idea.items() if key != "cases"},
            "examples": [one["said"] for one in idea.get("cases") or []][:3]
        }) + "\n")


def ledger() -> dict:
    """By source: ideas had, and how many came to a fix that held."""
    if not LEDGER.exists():
        return {}
    out: dict = {}
    for line in LEDGER.open(encoding="utf-8"):
        row = json.loads(line)
        one = out.setdefault(row["source"], {})
        one[row["outcome"]] = one.get(row["outcome"], 0) + 1
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("knowledge", "code", "ledger"))
    options = parser.parse_args(argv)
    if options.job == "ledger":
        print(json.dumps(ledger(), indent=1))
        return 0
    from research.v687 import build
    from research.v688 import server as v688
    from research.v689.definitions import DefinitionMemory
    from research.v689.session import Session
    from research.v690.server import PinnedAsker
    service = v688.Service(build.DEFAULT_STORE, 1, 8, teacher=False)
    session = Session(PinnedAsker(service), None, "ideas", True,
                      DefinitionMemory(None))
    if options.job == "code":
        for idea in spectrum(session):
            kept(idea)
            print(f"[{idea['kind']}] {idea['said']}")
            for one in idea["cases"][:3]:
                print(f"    {one['said']} -> {one['answer'][:90]}")
        return 0
    found = knowledge(session)
    saved(found)
    for idea in found:
        kept(idea)
        print(f"[{idea['kind']}] {idea['said']}")
        for one in idea["cases"][:3]:
            print(f"    {one['said']} -> {one['answer'][:90]}  "
                  f"(store: {one['expected']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
