"""Rung 5: working in a project (`PLAN.md`).

    use(task)      5b  a function written with what the project has: its
                       exported functions are operators the search grows,
                       each carrying its body (read, not written)
    fix(task)      5c  a test failing in one file, the fault in another:
                       rung 4's edits over every function the test reaches,
                       made in whichever file, the project's tests judging
    change(task)   5d  an API changed and its callers left behind: the
                       compiler's errors are the impasses -- each a subgoal
                       on the stack, met by an edit at its place (the
                       export it now is, arguments in the order it now
                       takes); a step kept when the errors fall; then the
                       tests, and 5c for what still fails

Every answer is the project with its files edited, judged by running the
project's own test file.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

from research.v696 import editing
from research.v696 import program as P
from research.v696.checker import CheckerError, checker
from research.v696.parse import Unread, project as read_project
from research.v696.spec import Spec

#: How many compiler errors a change may meet before it is given up.
IMPASSES = 12


@dataclass
class Done:
    task: str
    #: solved | unsolved | unread
    route: str
    files: dict | None = None
    edits: list = field(default_factory=list)
    #: compiler errors met and resolved (5d)
    impasses: int = 0
    files_touched: int = 0
    evaluated: int = 0


def judged(files: dict, tests: str) -> bool:
    try:
        return checker().project(files, tests) is None
    except CheckerError:
        checker().restart()
        return False


def _touched(before: dict, after: dict) -> int:
    return sum(1 for name in after if after[name] != before.get(name))


# -- 5b: the project's functions as the library -----------------------------

def operators(files: dict, besides: str) -> list:
    """Every function the project's other files declare, as an operator
    with its body, typed as it is declared."""
    module = read_project(files)
    if module is None:
        return []
    out = []
    for key, read in module.functions.items():
        where = key.split("#")[0]
        if where == besides or where.endswith(".test.ts") or "unread" in read:
            continue
        try:
            op = module.helper(key, tuple(P._plain(kind) or kind
                                          for _, kind in read["params"]))
        except (Unread, KeyError, TypeError, IndexError):
            continue
        out.append(op)
    return out


def use(task, budget: int = 8000, with_project: bool = True) -> Done:
    from research.v696 import search as S
    spec = Spec(task.name, [tuple(one) for one in task.params], task.returns,
                [(list(args), want) for args, want in
                 zip(task.cases, task.wanted)], entry=task.entry)
    if with_project:
        spec.library = operators(task.files, task.file)
    solver = S.Solver(S.Switches(meet=True, coarse=True, repair=True,
                                 forms=True), budget=budget)
    got = solver.solve(spec)
    if got.program is None:
        return Done(task.name, "unsolved", evaluated=got.evaluated)
    files = dict(task.files)
    written = _with_body(files[task.file], task.entry,
                         f"return {got.program.source()};")
    if written is None:
        return Done(task.name, "unsolved", evaluated=got.evaluated)
    # the function's body, and only it: helpers are called by the names
    # the file already imports
    files[task.file] = written
    if P.prelude([got.program]) and not all(
            re.search(rf"\b{op.name}\b", files[task.file])
            for op in P.helpers([got.program])):
        return Done(task.name, "unsolved", evaluated=got.evaluated)
    ok = judged(files, task.tests)
    return Done(task.name, "solved" if ok else "unsolved", files,
                files_touched=1, evaluated=got.evaluated)


def _with_body(text: str, entry: str, body: str) -> str | None:
    """The file with `entry`'s body replaced -- whatever it was."""
    head = re.search(r"\bfunction\s+" + re.escape(entry)
                     + r"\s*\([^)]*\)[^{]*\{", text)
    if head is None:
        return None
    depth = 0
    for index in range(head.end() - 1, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[:head.end()] + "\n  " + body + "\n" \
                    + text[index:]
    return None


# -- 5c: a fault in another file ---------------------------------------------

def fix(task, files: dict | None = None) -> Done:
    files = files or task.files
    bug = editing.Bug(task.name, files, task.entry,
                      [tuple(one) for one in task.params], task.returns,
                      task.cases, task.wanted, tests=task.tests,
                      file=task.file)
    got = editing.repair(bug)
    if got.route == "unread":
        return Done(task.name, "unread")
    if got.route in ("fixed", "already"):
        return Done(task.name, "solved", got.source, got.edits,
                    files_touched=_touched(task.files, got.source),
                    evaluated=got.tried)
    return Done(task.name, "unsolved", evaluated=got.tried)


# -- 5d: a change planned, the compiler's errors its impasses ----------------

def _exports(files: dict) -> list:
    """Every name a file exports: what a name that no longer resolves may
    now be."""
    names = []
    for text in files.values():
        names += re.findall(r"\bexport\s+(?:function|const)\s+(\w+)", text)
    return names


def _call_around(text: str, start: int, end: int):
    """The call whose parentheses hold [start, end): (open, close) of its
    argument list, or None."""
    depth, at = 0, start - 1
    while at >= 0:
        char = text[at]
        if char == ")":
            depth += 1
        elif char == "(":
            if depth == 0:
                break
            depth -= 1
        at -= 1
    if at < 0:
        return None
    opened, depth, close = at, 0, None
    for index in range(opened, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                close = index
                break
    if close is None or close < end:
        return None
    return opened + 1, close


def _arguments(said: str) -> list:
    return [one.strip() for one in P._split_top(said)] if said.strip() else []


def _moves(files: dict, error: dict) -> list:
    """What an impasse may be met by, at its place: the name it cannot
    find as a name the project exports (or the export under the old name),
    the arguments of the call it is in reordered."""
    text = files[error["file"]]
    start, end = error["start"], error["end"]
    said = text[start:end]
    out = []
    if re.fullmatch(r"\w+", said):
        for name in _exports(files):
            if name == said:
                continue
            out.append(editing.Edit(start, end, name, "name",
                                    file=error["file"]))
            if re.search(r"\bimport\s*\{[^}]*$", text[:start]):
                # at an import: the export as the name the file uses
                out.append(editing.Edit(start, end, f"{name} as {said}",
                                        "name", file=error["file"]))
    around = _call_around(text, start, end)
    if around:
        opened, close = around
        args = _arguments(text[opened:close])
        if 1 < len(args) <= 4:
            for order in itertools.permutations(range(len(args))):
                if list(order) == list(range(len(args))):
                    continue
                out.append(editing.Edit(
                    opened, close, ", ".join(args[at] for at in order),
                    "arguments", file=error["file"]))
    return out


def change(task) -> Done:
    files = dict(task.files)
    made, impasses = [], 0
    errors = checker().diagnose(files)
    while errors:
        if impasses >= IMPASSES:
            return Done(task.name, "unsolved", edits=made, impasses=impasses)
        # the goal stack: the first error the compiler gives is the top
        impasse = errors[0]
        best = None
        for move in _moves(files, impasse):
            after = move.apply(files)
            left = checker().diagnose(after)
            if len(left) < len(errors) and (
                    best is None or len(left) < len(best[2])):
                best = (move, after, left)
        if best is None:
            return Done(task.name, "unsolved", edits=made, impasses=impasses)
        move, files, errors = best
        made.append(move)
        impasses += 1
    if judged(files, task.tests):
        return Done(task.name, "solved", files, made, impasses,
                    _touched(task.files, files))
    # it compiles and still fails: what is wrong is a fault, as in 5c
    fixed = fix(task, files)
    if fixed.route == "solved":
        return Done(task.name, "solved", fixed.files, made + fixed.edits,
                    impasses, _touched(task.files, fixed.files),
                    fixed.evaluated)
    return Done(task.name, "unsolved", edits=made, impasses=impasses)
