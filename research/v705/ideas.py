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
SEED = 7052
#: the relations asked of it, and how much of the store answers each
RELATIONS = ("made_of", "used_for", "at_location", "has_part", "part_of",
             "capable_of", "causes", "has_prerequisite", "desires",
             "has_property", "is_a")
#: frames of a relation and form at most, facts asked of each kind
FRAMES, ASKED = 6, 6
#: the shortest a frame is, and the longest
SHORTEST, LONGEST = 3, 8
#: how sure its reader must be that a question reads as the kind it is of
READ_FLOOR = 0.6
#: a concept is asked about where WordNet's tagged text uses its word
USED = 3
#: a kind of question is an ability it lacks when none of this many is
#: answered
LACKING = 4


# -- how it asks ----------------------------------------------------------------------

def frames(most: int = FRAMES) -> dict:
    """(relation, polar) -> its taught questions of that kind that ask it
    plainly: begun by a question word, one span of what it is about
    (`SUBJ`) and, asked yes or no, one of what is asked of it (`TARGET`)."""
    out: dict = {}
    with TAUGHT.open(encoding="utf-8") as stream:
        for line in stream:
            if '"task": "parse"' not in line:
                continue
            row = json.loads(line)
            relation, polar = row["relation"], row["polar"]
            if relation not in RELATIONS:
                continue
            words, roles = row["words"], row["parse"]
            if not SHORTEST <= len(words) <= LONGEST or \
                    not row["tags"] or row["tags"][0] not in (
                        "WP", "WRB", "WDT", "VBZ", "VBP", "MD", "VBD"):
                continue
            subject = _span(roles, ("SUBJ",))
            target = _span(roles, ("TARGET",))
            if subject is None or (polar == "yes") != (target is not None):
                continue
            if any(role not in ("O", "SUBJ", "TARGET") for role in roles):
                continue
            # a frame said negated asks the opposite of the fact put in it
            # (`is a beagle that can not streak`), by its own parse
            if "neg" in (row.get("deps") or ()):
                continue
            # its shape: the words around what it is about
            shape = tuple("{S}" if subject[0] <= at < subject[1] else
                          "{T}" if target and target[0] <= at < target[1]
                          else word for at, word in enumerate(words))
            shape = tuple(one for at, one in enumerate(shape)
                          if not (at and one in ("{S}", "{T}")
                                  and shape[at - 1] == one))
            held = out.setdefault((relation, polar), {})
            if shape not in held:
                held[shape] = {"words": words, "subject": subject,
                               "target": target, "taught": 0}
            held[shape]["taught"] += 1
    # the ways it was taught to ask most often: `what is X made of` many
    # times over, `is a window huffed to have X` once
    return {key: sorted(held.values(), key=lambda one: -one["taught"])[
        :most] for key, held in out.items()}


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
                    before = frame["words"][frame["subject"][0] - 1] \
                        if frame["subject"][0] else ""
                    said_subject = subject if before in ("a", "an", "the") \
                        else _article(subject)
                    said_target = target if polar == "yes" else None
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
                                {"asked": 0, "right": 0, "cases": []})
        kind["asked"] += 1
        kind["right"] += right
        if not right:
            kind["cases"].append(evidence.case_of(
                turn, [one["question"]], check, area=one["relation"],
                expected=one["target"]))
    ideas = []
    for (relation, polar), kind in kinds.items():
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
    parser.add_argument("job", choices=("knowledge", "ledger"))
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
    for idea in knowledge(session):
        kept(idea)
        print(f"[{idea['kind']}] {idea['said']}")
        for one in idea["cases"][:3]:
            print(f"    {one['said']} -> {one['answer'][:90]}  "
                  f"(store: {one['expected']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
