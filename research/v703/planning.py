"""A change across a project (v703): planned, made step by step, checked
whole, and put back whole where it fails.

    found = planning.plan("report whether the shell asks in /api/health and "
                          "show it in the MCP health tool", held)
    found["steps"]   [{"path": "research/v698/server.py", "said": "..."},
                      {"path": "tools/graph-topology-mcp/server.py", ...}]
    made = planning.carry(found, held, key)

**The files shown** (`shortlist`): the project's files whose path, or a
function in them, a word of the request names -- looked up, the words the
parser reads as carrying meaning and those shaped as code.

**The plan**: the planner (`llm/planner`, `teach_plans.py`) is asked
several times; a plan is the files most answers change, two at least, with
what the first of those says to do in each.

**Carried out**: each step is one change, made as any change is
(`v700.fixing.change`: placed on the function the step names, written by
the editor, checked, agreed on). A step not made, or the project's tests
beside the files changed failing after (run by the shell, `v702`), puts
every step back: a plan is made whole or not at all.
"""
from __future__ import annotations

import re
from collections import Counter

PLANNER = "planner"
#: the judge of what a request changes, and how sure it must be; the most
#: it may choose
PICKER, PICKED, MOST_PICKED = "picker", 0.5, 6
#: how near the likeliest a file's best must be, and a unit its file's best
NEAR_FILES, NEAR_UNITS = 0.06, 0.02
#: the files shown the planner, and its answers
SHOWN, SAMPLES, LEAST_AGREED = 12, 6, 2
#: how long the tests beside a change may run
TESTS_TIMEOUT = 600


def available() -> bool:
    from research import encoder
    return (encoder.LLM / PLANNER / "config.json").exists() or \
        (encoder.LLM / PICKER / "config.json").exists()


def _words(request: str) -> set:
    from research.v700.fixing import _content_words
    content = _content_words(request) or set()
    shaped = {one.strip("?.,!'\"`") for one in request.split()
              if re.search(r"[._/()]|[a-z][A-Z]", one.strip("?.,!'\"`"))}
    out = set()
    for word in content | shaped:
        out.add(word.lower())
        out.update(part.lower() for part in re.split(r"[/._\-()]+", word)
                   if len(part) > 2)
    return out


def shortlist(request: str, held, most: int = SHOWN) -> list:
    """The project's files a request may touch: those whose path, or a
    function in them, its words name -- most named first."""
    said = _words(request)
    score: Counter = Counter()
    for path in held.files:
        parts = {one.lower() for one in re.split(r"[/._\-]+", path)
                 if len(one) > 2}
        score[path] += 2 * len(parts & said)
    for one in held.functions():
        name = one["name"].split(".")[-1].lower()
        if name in said or any(part in said for part in name.split("_")
                               if len(part) > 3):
            score[one["file"]] += 1
    # what is shaped as code is looked for as written in the files
    # (`/api/health` is a route in research/v698/server.py, no name)
    shaped = [one.strip("?.,!'\"`") for one in request.split()
              if re.search(r"[_/()]|[a-z][A-Z]|\w\.\w", one.strip("?.,!'\"`"))
              and len(one.strip("?.,!'\"`")) > 3]
    for path, text in held.files.items():
        score[path] += 3 * sum(one in text for one in shaped)
    named = [path for path, found in score.most_common(most) if found > 0]
    # and what is joined to the files named most (`structure`): the bridge
    # reads what the server's /api/health answers, though it says neither
    from research.v703 import structure
    seeds = named[:most // 2]
    joined = structure.linked(held, seeds, most // 3)
    return list(dict.fromkeys(seeds + joined + named))[:most]


def _steps(answer: str, shown: list) -> list:
    out = []
    for line in answer.splitlines():
        match = re.match(r"\s*(new\s+)?([\w./\-]+)\s*:\s*(.+)", line)
        if not match:
            continue
        new, path, said = bool(match.group(1)), match.group(2), \
            match.group(3).strip()
        if not new and path not in shown:
            continue
        if path not in [one["path"] for one in out]:
            out.append({"path": path, "said": said, "new": new})
    return out


def picker_available() -> bool:
    from research import encoder
    return (encoder.LLM / PICKER / "config.json").exists()


def plan(request: str, held) -> dict:
    """What a request changes, chosen among what the project has: each
    function of the files shown (and each file that has none) put before
    the picker with the request; those it reads as changed, most likely
    first -- never a name the project has not (Adrian: planning is
    choosing). Without the picker, the planner's written plan."""
    shown = shortlist(request, held)
    if not shown:
        return {"status": "unplanned", "said": "nothing in the project is "
                "named by what you asked", "shown": []}
    if picker_available():
        return _chosen(request, held, shown)
    return _written(request, held, shown)


def _chosen(request: str, held, shown: list) -> dict:
    from research.v700 import teach_judge as J
    from research.v703.teach_picker import unit
    units = []
    for path in shown:
        functions = (held.outline().get(path) or {}).get("functions", ())
        if path.endswith(".py") and functions:
            units += [(path, one) for one in functions]
        else:
            units.append((path, None))
    if "picker" not in _LOADED:
        from research import encoder
        _LOADED["picker"] = J.Judge(encoder.LLM / PICKER)
    chances = _LOADED["picker"].chances(request, [unit(path, one)
                                                  for path, one in units])
    ranked = sorted(zip(chances, range(len(units))), reverse=True)
    # the picker reads most of what is shown as likely (it was taught
    # commits, and a commit changes something of every file it names):
    # what is chosen is what it reads near its likeliest -- each file's
    # best, where near the best of all, and of a file what is near its own
    best_of: dict = {}
    for chance, at in ranked:
        best_of.setdefault(units[at][0], chance)
    top = ranked[0][0] if ranked else 0.0
    files = {path for path, chance in best_of.items()
             if chance >= max(PICKED, top - NEAR_FILES)}
    picked = [(chance, units[at]) for chance, at in ranked
              if units[at][0] in files and
              chance >= best_of[units[at][0]] - NEAR_UNITS][:MOST_PICKED]
    steps = [{"path": path, "function": one["name"] if one else None,
              "said": request, "new": False, "chance": round(chance, 3)}
             for chance, (path, one) in picked]
    return {"status": "planned" if steps else "unplanned", "steps": steps,
            "shown": shown, "by": "picker",
            "nearest": [{"unit": unit(*units[at]), "chance": round(chance, 3)}
                        for chance, at in ranked[:8]]}


_LOADED: dict = {}


def _written(request: str, held, shown: list) -> dict:
    """The files most of the planner's answers change, two at least, and
    what the first of those says to do in each."""
    import zlib
    from research.v697.coding import Tools
    from research.v700 import teach_editor as T
    from research.v703 import structure
    writer = Tools.get().writer(PLANNER)
    # each file said as what it is -- of the files shown alone, as the
    # planner was taught
    said = structure.shown(structure.subset(
        {path: held.files[path] for path in shown}), shown)
    asked = T.prompt(request, "plan", said)
    writer.torch.manual_seed(703 + zlib.crc32(asked.encode()))
    answers = writer.write([asked], samples=SAMPLES, longest=400,
                           greedy=True)[0]
    plans = [_steps(answer, shown) for answer in answers]
    sets = Counter(tuple(sorted(one["path"] for one in steps))
                   for steps in plans if steps)
    if not sets:
        return {"status": "unplanned", "shown": shown, "answers": answers,
                "said": "no answer named a file shown"}
    paths, agree = sets.most_common(1)[0]
    steps = next(steps for steps in plans
                 if tuple(sorted(one["path"] for one in steps)) == paths)
    return {"status": "planned" if agree >= LEAST_AGREED else "unsure",
            "steps": steps, "agree": agree, "of": len(answers),
            "shown": shown}


def carry(found: dict, held, key, tests: bool = True) -> dict:
    """Each step made as a change of its file; any not made, or the tests
    beside the files failing, and every step is put back."""
    from research.v698 import reading
    from research.v698.asking import Subject
    from research.v700 import fixing
    from research.v703 import structure
    made, done = [], []
    before = structure.signatures(held, [step["path"] for step in
                                         found["steps"] if not step["new"]])
    for step in found["steps"]:
        if step["new"]:
            done.append({**step, "status": "left", "said_back":
                         "a file to make is not made yet"})
            continue
        if step.get("function"):
            # chosen by the picker: the function itself, and the request
            # said of it
            statement = (f"in {step['path']}, {step['function']}: "
                         f"{step['said']}")
            subject = Subject("function", step["function"], step["path"])
        else:
            statement = f"in {step['path']}, {step['said']}"
            read = reading.read(statement, set())
            subject = fixing.placed(read, _only(held, step["path"]),
                                    statement) \
                or Subject("file", step["path"], step["path"])
        result = fixing.change(statement, held, subject, key)
        done.append({**step, "status": result["status"],
                     "said_back": result.get("said", "")})
        if result["status"] != "changed":
            break
        made.append(step["path"])
    failed = next((one for one in done if one["status"] not in
                   ("changed", "left")), None)
    ran = None
    if failed is None and made:
        # what the project's graph says the change left undone: a caller
        # elsewhere of a function whose parameters it changed
        broken = structure.broken_callers(held, before, made)
        if broken:
            failed = {"path": "callers", "status": "incomplete",
                      "said_back": "still called the old way: "
                                   + "; ".join(broken[:3])}
    if failed is None and tests and made:
        ran = _tests(made, held)
        if ran is not None and ran.get("code") != 0:
            failed = {"path": "tests", "status": "failed",
                      "said_back": ran.get("command", "")}
    if failed is not None:
        for _ in made:
            fixing.undo(held, key)
        return {"status": "undone", "steps": done, "tests": ran,
                "said": (f"I put every step back: {failed['path']} -- "
                         f"{failed['said_back']}")}
    return {"status": "changed", "steps": done, "tests": ran,
            "said": (f"Changed {len(made)} files: "
                     + ", ".join(made)
                     + ("; the tests beside them pass." if ran else "."))}


def _only(held, path: str):
    """The project as it holds one file: where a step is placed."""
    from research.v698.project import Project
    one = Project(held.name, held.root)
    one.files = {path: held.files[path]}
    return one


def _tests(paths: list, held):
    """The tests beside the files changed (`research/v702/test_v702.py`
    for `research/v702/shell.py`), run by the shell in the project's
    folder; None where there are none."""
    from research.v702 import shell
    found = []
    for path in paths:
        folder = path.rsplit("/", 1)[0] if "/" in path else ""
        for one in held.files:
            if one.rsplit("/", 1)[0] == folder and re.search(
                    r"(^|/)test_\w+\.py$", one):
                found.append(one[:-3].replace("/", "."))
    found = list(dict.fromkeys(found))
    if not found or not held.root:
        return None
    command = "python -W ignore -m unittest " + " ".join(found)
    result = shell.run(command, held.root, timeout=TESTS_TIMEOUT)
    result["command"] = command
    return result
