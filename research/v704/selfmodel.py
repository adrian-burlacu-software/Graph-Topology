"""What the architecture knows of itself (v704, step 3 of planning goals):
what it does, how well, and where it fails -- from the conversation test.

    python -m research.v704.selfmodel build [label]   llm/plan-data/self.json
    python -m research.v704.selfmodel show "improve how you communicate"

Planned from how this project reached goals like it, "improve how you
communicate" came back as build, change, teach -- of nothing in
particular: what "it" does badly was known nowhere it could look. The
conversation test (`converse.py`) knows: every area it does, and each
conversation's verdict and the part that answered it.

**Built from a run, by construction**: each area's right / unsure /
wrong; the part that answers it when it is right (data questions: the
project's data); and its failures -- **routing** where another part
answered (`what is 7 squared?` went to the shell), else **answering**
(the right part, the wrong answer) and **not knowing** (unsure).
**Read for a goal**: whether the goal is about the assistant itself or a
project -- the nearer in meaning of the two (MiniLM, as pretrained); of
itself, the weaknesses its words are near -- or, where it is near none
more than the rest (`communicate in general`), the worst.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SELF = ROOT / "llm" / "plan-data" / "self.json"
RUNS = ROOT / "conversations-out"
#: each area the test asks, as the architecture says it does it
AREAS = {
    "code-ask": "answering questions about a project's code",
    "code-change": "changing a project's code as asked",
    "code-make": "writing new code from a description",
    "explain": "explaining code that is pasted in",
    "data-ask": "answering questions about a project's data files",
    "data-change": "changing a project's data files as asked",
    "shell": "running commands on the computer",
    "facts": "answering questions about the world",
    "maths": "working out arithmetic",
    "memory": "remembering what was said earlier in the conversation",
    "how-to": "saying how to do something",
    "follow-up": "following on from what was just said",
    "asks": "asking when a request is unclear, and declining what is "
            "unsafe",
}
ROUTING = "understanding which kind of thing is asked, and sending it to " \
          "the right part"
#: what a goal about the assistant itself, and one about a project, is
ABOUT = ("making the assistant itself better at understanding, answering "
         "and talking", "making a software project's code, server or "
         "features better")
#: how much nearer a goal must be to one weakness than to most, for its
#: words to choose; the weaknesses a goal is shown
CHOOSES, SHOWN = 0.08, 3


def family(part: str | None) -> str | None:
    """Which part of the architecture an answer came from: the project's
    data (data talk), a change or a plan of it, any other question of it
    (code talk), else the part as the test kept it (the shell, code
    written, where a verdict came from)."""
    if not part:
        return part
    if part.startswith("project data"):
        return "data talk"
    if part.startswith(("project change", "project plan", "project goal")):
        return "changing"
    if part.startswith("project"):
        return "code talk"
    return part


def places(root: Path = ROOT / "research") -> dict:
    """Where each of the architecture's operators lives in its own code, by
    construction: every `Operator(...)` made in it -- its name, rule, and
    the functions that propose it and carry it out, each a file, a
    function and a line. {name: [place, ...]}: a name made in more than
    one version is told apart by its rule."""
    import ast
    found: dict = {}
    for path in sorted(root.rglob("*.py")):
        if path.name.startswith("test_"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        defs = [(one.lineno, one.name) for one in ast.walk(tree)
                if isinstance(one, (ast.FunctionDef, ast.AsyncFunctionDef))]

        def at(node, line):
            # a name: the latest function of that name made before the
            # operator (a closure in the same function); a lambda: itself
            if isinstance(node, ast.Name):
                made = [one for one in defs if one[1] == node.id and
                        one[0] <= line]
                return {"function": node.id,
                        "line": max(made)[0]} if made else None
            if isinstance(node, ast.Lambda):
                return {"function": "lambda", "line": node.lineno}
            return None
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(
                    node.func, "id", None) == "Operator"):
                continue
            given = dict(zip(("name", "apply", "proposes"), node.args))
            given.update({one.arg: one.value for one in node.keywords})
            name = given.get("name")
            if not (isinstance(name, ast.Constant) and
                    isinstance(name.value, str)):
                continue
            rule = given.get("rule")
            found.setdefault(name.value, []).append({
                "file": path.relative_to(ROOT).as_posix(),
                "rule": rule.value if isinstance(rule, ast.Constant) and
                isinstance(rule.value, str) else "",
                "apply": at(given.get("apply"), node.lineno),
                "proposes": at(given.get("proposes"), node.lineno)})
    return found


def place_of(name: str | None, rule: str = "", found: dict | None = None):
    """The place of the operator of this name (and rule, where several
    versions made one of that name); None where it is made nowhere."""
    if not name:
        return None
    made = (found if found is not None else places()).get(name) or []
    same = [one for one in made if rule and one["rule"] == rule]
    return (same or made or [None])[-1]


def _latest() -> str | None:
    runs = [one for one in RUNS.glob("*.jsonl") if "queued" not in one.name]
    return max(runs, key=lambda one: one.stat().st_mtime).stem \
        if runs else None


def build(label: str | None = None) -> dict:
    label = label or _latest()
    if label is None:
        raise SystemExit("no conversation test run in conversations-out")
    rows = [json.loads(line) for line in
            (RUNS / f"{label}.jsonl").open(encoding="utf-8")]
    areas: dict = {}
    found_places = places()
    for row in rows:
        row["placed"] = _placed((row.get("operator"), row.get("rule", "")),
                                found_places)
    for area in AREAS:
        mine = [one for one in rows if one["area"] == area]
        if not mine:
            continue
        verdicts = Counter(one["verdict"] for one in mine)
        # the parts that answer it when right, by kind: code questions are
        # answered right by callers, explain, where -- one part, code talk
        right_parts = Counter(family(one.get("part")) for one in mine
                              if one["verdict"] == "right" and
                              one.get("part"))
        usual = right_parts.most_common(1)[0][0] if right_parts else None
        # the operator that answers it right: where in the architecture's
        # own code the area is done
        right_ops = Counter((one["operator"], one.get("rule", ""))
                            for one in mine if one["verdict"] == "right"
                            and one.get("operator"))
        doer = right_ops.most_common(1)[0][0] if right_ops else None
        failed = [one for one in mine if one["verdict"] != "right"]
        routed = [one for one in failed if right_parts and one.get("part")
                  and family(one["part"]) not in right_parts]
        areas[area] = {
            "what": AREAS[area], "right": verdicts["right"],
            "unsure": verdicts["unsure"], "wrong": verdicts["wrong"],
            "of": len(mine), "usual part": usual,
            "done by": _placed(doer, found_places),
            "routed": [{**_example(one), "should": _placed(doer,
                                                           found_places)}
                       for one in routed],
            "failed": [_example(one) for one in failed
                       if one not in routed][:4]}
    found = {"run": label, "built": time.strftime("%Y-%m-%d"),
             "areas": areas,
             "routing": [one for area in areas.values()
                         for one in area["routed"]]}
    SELF.parent.mkdir(parents=True, exist_ok=True)
    SELF.write_text(json.dumps(found, indent=1), encoding="utf-8")
    return found


def _placed(operator, found: dict) -> dict | None:
    """An operator (name, rule) with its place in the code."""
    if not operator or not operator[0]:
        return None
    return {"operator": operator[0],
            "place": place_of(operator[0], operator[1], found)}


def _example(row: dict) -> dict:
    return {"said": row["turns"][-1][:120], "part": row.get("part"),
            "verdict": row["verdict"], "by": row.get("placed"),
            "answer": ((row.get("said") or [""])[-1] or "")[:160]}


def held() -> dict | None:
    if not SELF.exists():
        return None
    return json.loads(SELF.read_text(encoding="utf-8"))


def about_itself(goal: str) -> bool:
    """Whether a goal is about the assistant itself, rather than a project:
    the nearer in meaning."""
    from research.v701.likeness import nearest
    ranked = nearest(goal, list(ABOUT))
    return ranked[0][1] == ABOUT[0]


def weaknesses(goal: str, model: dict | None = None) -> list:
    """The weaknesses a goal about itself is about: those its words are
    near, or the worst where it is near none more than the rest -- each
    {what, how it does, examples}, worst first."""
    from research.v701.likeness import nearest
    model = model or held()
    if not model:
        return []
    found = []
    routing = model.get("routing") or []
    if routing:
        found.append({"what": ROUTING, "bad": len(routing),
                      "count": len(routing),
                      "how": f"{len(routing)} messages went to the wrong "
                             f"part", "examples": routing[:2],
                      "cases": routing,
                      "where": _where([(one.get("by"), "proposes", "took")
                                       for one in routing] +
                                      [(one.get("should"), "proposes",
                                        "should take") for one in routing])})
    for area in model["areas"].values():
        bad = area["wrong"] + area["unsure"] / 2 - len(area["routed"])
        if area["wrong"] + area["unsure"] == 0:
            continue
        found.append({"what": area["what"], "bad": bad,
                      "right": area["right"], "of": area["of"],
                      "wrong": area["wrong"],
                      "how": f"{area['right']} right of {area['of']}, "
                             f"{area['wrong']} wrong",
                      "examples": area["failed"][:2],
                      "cases": area["failed"],
                      "where": _where([(area.get("done by"), "apply",
                                        "does")])})
    if not found:
        return []
    near = dict((what, score) for score, what in
                nearest(goal, [one["what"] for one in found]))
    scores = sorted(near.values(), reverse=True)
    middle = scores[len(scores) // 2]
    chosen = [one for one in found if near[one["what"]] - middle >= CHOOSES]
    if not chosen:
        # a goal near none more than the rest is of all of it: the worst
        chosen = found
        chosen.sort(key=lambda one: -one["bad"])
    else:
        chosen.sort(key=lambda one: -near[one["what"]])
    return chosen[:SHOWN]


def _where(placed: list) -> list:
    """The functions of the architecture's own code a weakness is in, each
    once: the operator, its function (that proposes it, or carries it out),
    file and line, and what it does in the weakness (`took` a message it
    should not have, `should take` it, `does` the area), most cases
    first."""
    counted: Counter = Counter()
    for operator, which, role in placed:
        place = (operator or {}).get("place") or {}
        function = place.get(which)
        if not function or function["function"] == "lambda":
            continue
        counted[(operator["operator"], function["function"], place["file"],
                 function["line"], role)] += 1
    return [{"operator": one[0], "function": one[1], "file": one[2],
             "line": one[3], "role": one[4], "cases": count}
            for one, count in counted.most_common()]


def said(found: list, model: dict) -> str:
    lines = [f"What I do worst now, by the conversation test of "
             f"{model['built']}:"]
    for at, one in enumerate(found, 1):
        line = f"{at}) {one['what']}: {one['how']}"
        if one["examples"]:
            example = one["examples"][0]
            line += (f" -- e.g. \"{example['said']}\" "
                     f"(answered by {example['part']})")
        lines.append(line)
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("build", "show"))
    parser.add_argument("text", nargs="?", default=None)
    options = parser.parse_args(argv)
    if options.job == "build":
        found = build(options.text)
        print(json.dumps({area: [one["right"], one["unsure"], one["wrong"],
                                 one["usual part"], len(one["routed"])]
                          for area, one in found["areas"].items()},
                         indent=1))
    else:
        model = held()
        print("about itself:", about_itself(options.text))
        print(said(weaknesses(options.text, model), model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
