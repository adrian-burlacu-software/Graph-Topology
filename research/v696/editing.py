"""Rung 4: editing a program that exists (`PLAN.md`).

The program is read into the search's tree (`parse.py`) with where every
node came from in the source (`Expr.where`). An **edit** replaces the text
of one node's span -- or of its operator or member -- with other text, and
is made in the source, so the rest of the file is untouched; the edited
file is what is run and returned.

    sites(tree)                every node that came from the source
    edits(tree, source)        each single edit, the text it makes
    repair(bug)                fewest edits first, each run on the cases

The edits are general, over the tree's types, not over any kind of bug:

    operator   another operator the language has with the same needs and
               gives (`<` for `<=`, `-` for `+`), or a `!` removed
    member     another member with the same needs and gives (`floor` for
               `ceil`, `lastIndexOf` for `indexOf`)
    constant   a number nudged by one, negated, 0 or 1; a truth flipped; a
               string emptied
    name       another name the function reads, of the same type
    swap       an operator's two sides exchanged
    unwrap     a node replaced by a part of it of the same type
    wrap       a number made absolute or negated, a truth negated

Where the fault is (backward error-correction, 2d): a node whose values on
the cases that fail are never its values on those that pass is suspected
first; the rest follow in the order the source has them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from research.v696 import program as P
from research.v696.checker import CheckerError, checker
from research.v696.parse import parse

#: How many edited programs a repair may run.
BUDGET = 3000
#: How many of the most suspected single edits pairs are made from.
PAIRED = 40
#: What the search may spend writing again what no few edits fix.
REWRITE_BUDGET = 8000
#: ... and how long, in seconds: in Python a candidate may take its whole
#: timeout, and the budget counts candidates (v699)
REWRITE_SECONDS = 60


@dataclass
class Bug:
    name: str
    source: str
    entry: str
    params: list
    returns: str
    #: the cases the program must meet, and what it must give on them
    cases: list
    wanted: list
    #: the test file that judges a repair in the end ("" for none)
    tests: str = ""
    #: for a generated bug: the program it was before
    original: str = ""
    #: in a project (rung 5): `source` is {file: text}, this the file the
    #: function is in, `tests` the project's test file
    file: str = ""
    #: the language the program is in (v699): its checker runs it, and in
    #: Python its edits are read off its syntax (`pyediting.py`)
    language: str = "typescript"


@dataclass
class Edit:
    start: int
    end: int
    text: str
    kind: str
    #: how much the site is suspected (higher first)
    suspicion: float = 0.0
    #: in a project (rung 5): the file the span is in
    file: str = ""

    def apply(self, source):
        """The source edited: a text, or a project whose file this is."""
        if isinstance(source, dict):
            out = dict(source)
            text = out[self.file]
            out[self.file] = text[:self.start] + self.text + text[self.end:]
            return out
        return source[:self.start] + self.text + source[self.end:]


@dataclass
class Fix:
    bug: str
    #: fixed (edited) | rewritten (searched) | already | unread | unfixed
    route: str
    source: str | None = None
    edits: list = field(default_factory=list)
    tried: int = 0


def sites(tree: P.Expr) -> list:
    """Every node that came from the source: the tree and the bodies of
    the helpers it calls."""
    out, seen = [], set()

    def walk(expr: P.Expr) -> None:
        if expr.where and id(expr) not in seen:
            seen.add(id(expr))
            out.append(expr)
        for one in expr.args:
            walk(one)
    walk(tree)
    for op in P.helpers([tree]):
        walk(op.body)
    return out


def _alternatives(op: P.Op) -> list:
    """Other operators or members with the same needs and gives."""
    lib = P.library()
    return [one for one in lib.ops + lib.read_only
            if one.kind == op.kind and one.needs == op.needs
            and one.gives == op.gives and one.name != op.name
            and one.spread == op.spread]


#: Numbers any program may be one of from another: the small ones.
SMALL = (0, 1, 2, -1)


def _numbers(value, others=()) -> list:
    """What a number may have been meant to be: one either side of it,
    its negation, a small number, or another the program says."""
    out = [value + 1, value - 1, -value, *SMALL, *others]
    return [one for one in dict.fromkeys(out) if one != value]


def _declared(source, entry: str) -> dict:
    """Every name each function declares, by type (`tscheck.js` `tree`) --
    of a file, or of every file of a project."""
    from research.v696.parse import module, project
    read = project(source) if isinstance(source, dict) else module(source)
    out: dict = {}
    if read is None:
        return out
    for function in read.functions.values():
        for name, type_ in function.get("names", ()):
            out.setdefault(P._plain(type_) or type_, set()).add(name)
    return out


def edits(tree: P.Expr, source: str, entry: str = "") -> list:
    """Every single edit of the read program."""
    found = sites(tree)
    names = _declared(source, entry)
    for one in found:
        said = one.where.get("said")
        if said and one.kind == "param":
            names.setdefault(one.type, set()).add(said)
    said_numbers = sorted({one.value for one in found if one.kind == "const"
                           and isinstance(one.value, (int, float))
                           and not isinstance(one.value, bool)})
    out = []
    here = {"source": source}

    def text(span) -> str:
        return here["source"][span[0]:span[1]]

    for node in found:
        where = node.where
        start, end = where["span"]
        file = where.get("file", "")
        # a project's node is in one of its files: that file's text
        here["source"] = source[file] if isinstance(source, dict) else source
        made = len(out)
        _node_edits(node, where, start, end, out, text, here["source"],
                    names, said_numbers)
        for one in out[made:]:
            one.file = file
    unique, seen = [], set()
    for one in out:
        key = (one.file, one.start, one.end, one.text)
        if key not in seen and one.apply(source) != source:
            seen.add(key)
            unique.append(one)
    return unique


def _node_edits(node, where, start, end, out, text, source, names,
                said_numbers) -> None:
    """Every single edit of one node, into `out`."""
    if node.kind == "const":
        value = node.value
        if isinstance(value, bool):
            news = [json.dumps(not value)]
        elif isinstance(value, (int, float)):
            news = [json.dumps(one) for one in _numbers(value,
                                                         said_numbers)]
        elif isinstance(value, str) and value:
            news = ['""']
        else:
            news = []
        out += [Edit(start, end, one, "constant") for one in news]
        return
    if node.kind == "param" and where.get("said"):
        out += [Edit(start, end, other, "name")
                for other in sorted(names.get(node.type, ()))
                if other != where["said"]]
        return
    if node.kind != "apply" or node.op is None:
        return
    op = node.op
    if op.kind == "operator" and "opspan" in where:
        a, b = where["opspan"]
        if len(op.needs) == 1:
            # a unary operator removed
            out.append(Edit(a, b, "", "operator"))
        out += [Edit(a, b, alt.name, "operator")
                for alt in _alternatives(op)]
        left, right = node.args if len(node.args) == 2 else (None, None)
        if left is not None and left.where and right.where \
                and op.name not in ("+", "*", "===", "!==", "&&", "||"):
            (l0, l1), (r0, r1) = left.where["span"], right.where["span"]
            if l1 <= r0:
                out.append(Edit(l0, r1, text((r0, r1)) + source[l1:r0]
                                + text((l0, l1)), "swap"))
    if op.kind in ("method", "function", "property") \
            and "namespan" in where:
        a, b = where["namespan"]
        out += [Edit(a, b, alt.name.split(".")[-1], "member")
                for alt in _alternatives(op)]
    for part in node.args:
        if part.type == node.type and part.where:
            out.append(Edit(start, end, text(part.where["span"]),
                            "unwrap"))
    said = text((start, end))
    if node.type == "number":
        out += [Edit(start, end, f"Math.abs({said})", "wrap"),
                Edit(start, end, f"-({said})", "wrap")]
    elif node.type == "boolean":
        out.append(Edit(start, end, f"!({said})", "wrap"))


def flattened(files: dict, file: str) -> str:
    """A project as one source to run `file`'s functions in: every other
    file first, imports and exports gone (they only join the files)."""
    import re
    text = "".join(files[name] + "\n" for name in files
                   if name != file and not name.endswith(".test.ts"))
    text += files[file]
    text = re.sub(r"^\s*import .*$", "", text, flags=re.M)
    return re.sub(r"\bexport\s+(?=function|const)", "", text)


def _passes(bug: Bug, source) -> list:
    """For each case, whether the program as edited gives what it must."""
    if isinstance(source, dict):
        source = flattened(source, bug.file)
    running = checker(bug.language)
    try:
        got = running.run(source, bug.entry, bug.cases)
    except CheckerError:
        running.restart()
        return [False] * len(bug.cases)
    return [("value" in one and json.dumps(one["value"], sort_keys=True)
             == json.dumps(want, sort_keys=True))
            for one, want in zip(got, bug.wanted)]


def _suspect(bug: Bug, tree: P.Expr, found: list, passing: list) -> dict:
    """How much each site is suspected: 1 where its values on the failing
    cases are never its values on the passing ones (it separates them),
    0 where it is the same on both; unknown (0.5) where it cannot be run
    on its own -- inside a loop's body, over the loop's names."""
    out = {}
    closed = [one for one in found if one.kind == "apply"]
    if not closed or all(passing) or not any(passing):
        return {id(one): 0.5 for one in found}
    try:
        rows = checker().values([name for name, _ in bug.params],
                                bug.cases, [one.source() for one in closed],
                                prelude=P.prelude(closed))
    except CheckerError:
        checker().restart()
        return {id(one): 0.5 for one in found}
    for node, row in zip(closed, rows):
        if any("value" not in one for one in row):
            out[id(node)] = 0.5
            continue
        good = {json.dumps(v["value"]) for v, ok in zip(row, passing) if ok}
        bad = {json.dumps(v["value"]) for v, ok in zip(row, passing)
               if not ok}
        out[id(node)] = 1.0 if not good & bad else 0.0
    return out


def _judged(bug: Bug, source: str) -> bool:
    """The last word: the bug's own test file, when it has one."""
    if isinstance(source, dict):
        # a project: its own test file, run with the project
        try:
            return checker().project(source, bug.tests) is None
        except CheckerError:
            checker().restart()
            return False
    if not bug.tests:
        return all(_passes(bug, source))
    running = checker(bug.language)
    try:
        return running.tests(source + "\n" + bug.tests) is None
    except CheckerError:
        running.restart()
        return False


def repair(bug: Bug, budget: int = BUDGET, rewrite: bool = True) -> Fix:
    """`rewrite`: whether what no few edits fix is searched for again."""
    if bug.language == "python":
        return _repair_python(bug, budget, rewrite)
    if isinstance(bug.source, dict):
        from research.v696.parse import parse_project
        tree = parse_project(bug.source, bug.file, bug.entry, bug.params)
    else:
        tree = parse(bug.source, bug.entry, bug.params)
    if tree is None:
        return Fix(bug.name, "unread")
    passing = _passes(bug, bug.source)
    if all(passing):
        return Fix(bug.name, "already", bug.source)
    found = sites(tree)
    suspicion = _suspect(bug, tree, found, passing)
    single = edits(tree, bug.source, bug.entry)
    by_span = {}
    for node in found:
        by_span.setdefault(tuple(node.where["span"]), id(node))
    for one in single:
        node = next((key for span, key in by_span.items()
                     if span[0] <= one.start and one.end <= span[1]), None)
        one.suspicion = suspicion.get(node, 0.5)
    # the most suspected first; among equals, the order of the source
    single.sort(key=lambda one: (-one.suspicion, one.start))
    tried = 0
    for one in single[:budget]:
        tried += 1
        edited = one.apply(bug.source)
        if all(_passes(bug, edited)) and _judged(bug, edited):
            return Fix(bug.name, "fixed", edited, [one], tried)
    # two edits: of the most suspected, any two that do not overlap
    top = single[:PAIRED]
    for at, first in enumerate(top):
        for second in top[at + 1:]:
            if tried >= budget:
                return _rewritten(bug, tree, tried) if rewrite \
                    else Fix(bug.name, "unfixed", tried=tried)
            if not (first.end <= second.start or second.end <= first.start):
                continue
            later, earlier = sorted((first, second),
                                    key=lambda one: -one.start)
            edited = earlier.apply(later.apply(bug.source))
            tried += 1
            if all(_passes(bug, edited)) and _judged(bug, edited):
                return Fix(bug.name, "fixed", edited, [first, second], tried)
    return _rewritten(bug, tree, tried) if rewrite \
        else Fix(bug.name, "unfixed", tried=tried)


def _repair_python(bug: Bug, budget: int, rewrite: bool) -> Fix:
    """Python's: the edits off its syntax (`pyediting.edits`), in the order
    of the text -- no site is suspected before another, its nodes being
    the syntax's and not the search's -- one and then two at a time; the
    search's reading of it, where there is one, the sketch it is written
    again from."""
    from research.v696 import pyediting
    from research.v696.language import Python
    passing = _passes(bug, bug.source)
    if all(passing):
        return Fix(bug.name, "already", bug.source)
    single = pyediting.edits(bug.source, bug.entry)
    tried = 0
    for one in single[:budget]:
        tried += 1
        edited = one.apply(bug.source)
        if all(_passes(bug, edited)) and _judged(bug, edited):
            return Fix(bug.name, "fixed", edited, [one], tried)
    top = single[:PAIRED]
    for at, first in enumerate(top):
        for second in top[at + 1:]:
            if tried >= budget:
                break
            if not (first.end <= second.start or second.end <= first.start):
                continue
            later, earlier = sorted((first, second),
                                    key=lambda one: -one.start)
            edited = earlier.apply(later.apply(bug.source))
            tried += 1
            if all(_passes(bug, edited)) and _judged(bug, edited):
                return Fix(bug.name, "fixed", edited, [first, second], tried)
    if not rewrite:
        return Fix(bug.name, "unfixed", tried=tried)
    tree = Python().parse(bug.source, bug.entry, bug.params)
    return _rewritten(bug, tree, tried)


def _rewritten(bug: Bug, tree: P.Expr, tried: int) -> Fix:
    """What no few edits fix -- logic missing, not misused -- is searched:
    the program as it is is the search's sketch (its parts in the forward
    trie, itself a near miss to repair), and what is found is the function
    written again."""
    from research.v696 import search as S
    from research.v696.spec import Spec
    spec = Spec(bug.name, bug.params, bug.returns,
                [(list(args), want) for args, want in
                 zip(bug.cases, bug.wanted)], entry=bug.entry,
                language=bug.language)
    spec.proposals = [tree] if tree is not None else []
    solver = S.Solver(S.Switches(meet=True, coarse=True, repair=True,
                                 forms=True, proposals=True),
                      budget=REWRITE_BUDGET)
    solver.seconds = REWRITE_SECONDS
    got = solver.solve(spec)
    if got.program is None:
        return Fix(bug.name, "unfixed", tried=tried + got.evaluated)
    written = spec.function(got.program)
    if not _judged(bug, written):
        return Fix(bug.name, "unfixed", tried=tried + got.evaluated)
    return Fix(bug.name, "rewritten", written, tried=tried + got.evaluated)
