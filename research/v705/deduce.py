"""Deduction (v705): a value wanted, broken down by its shape into the
values its parts must be, until each part is small enough to find -- and
the parts put back together. What the maths side does with a spec and a
form (v696's holes as subgoals), done on any Python value.

A leaf of a design -- `read` must give a Python signature -- is one goal
to a model asked to write it, and a model wrote none. Broken down by what
is wanted, it is pieces: a tuple is a goal per field; a list is a goal for
what each element is made of (`[f(one) for one in L]`) or kept of
(`[one for one in L if p]`); a dict, a goal per key; a string, a piece of
an input (found between what is around it in every example), or pieces
joined (`sep.join(...)`) or put one after another (`"def " + name +
...`), each found again; a value its own code's tables or functions
give (`BACK.get(kind)`). What is left at the bottom is small enough for
search over what is in hand, kept apart by what it gives (observational
equivalence, v696 2b).

    solve(examples, library)   [Program], most general first

Every rule here is of the value's shape, not of a task: the same rules
read a signature, take fields apart, build a list of pairs.
"""
from __future__ import annotations

import functools
import itertools
import re
import sys
import threading
import time
from dataclasses import dataclass, field

#: how deep a goal is broken down, how long a whole solve may take
DEPTH, SECONDS = 4, 30.0
#: what a search grows at most at each level
GROWN = 6000
#: the separators a string is split or joined by
SEPARATORS = ("\n", ", ", ",", " ", ":", ";", "|", "/", ".", "-", "_", "=",
              "\t", ": ", " -> ", "->")
#: the plain values offered as constants (no string a test could say)
PLAIN = (None, True, False, 0, 1, -1, "", (), [])


@dataclass(frozen=True)
class Program:
    """An expression over the names of an example's inputs, and what it
    needs imported (`from research.v696.pytypes import BACK`)."""
    expr: str
    imports: tuple = ()
    size: int = 1

    def __call__(self, inputs: dict, space: dict):
        # a program is of its inputs alone: one that reads a file or the
        # store is not what they give (`known_of` read a file a call, and
        # the search ran out of time on it)
        _PURE.on = True
        try:
            return eval(_compiled(self.expr), space, dict(inputs))  # noqa: S307
        finally:
            _PURE.on = False


#: whether a program is being run, in this thread
_PURE = threading.local()
#: what a program running may not do: reach outside itself
WORLD = ("open", "os.", "subprocess.", "socket.", "sqlite3.", "shutil.",
         "urllib.", "http.", "ctypes.")


def _guard(event: str, args) -> None:
    if getattr(_PURE, "on", False) and event.startswith(WORLD):
        _PURE.touched = True
        raise PermissionError(f"a program may not {event}")


sys.addaudithook(_guard)


@functools.lru_cache(maxsize=200_000)
def _compiled(expr: str):
    return compile(expr, "<deduced>", "eval")


@dataclass
class Library:
    """What a program may call besides the builtins: its own code's
    functions and tables, by name -- {name: (value, import line)}."""
    held: dict = field(default_factory=dict)
    #: the text of the code it is for, whose words are its own
    said: str = ""

    def words(self) -> set:
        if "_words" not in self.__dict__:
            self.__dict__["_words"] = set(re.findall(r"\w+", self.said))
        return self.__dict__["_words"]

    def space(self) -> dict:
        import ast
        import builtins
        import json
        out = {"re": re, "ast": ast, "json": json, "__builtins__": builtins}
        out.update({name: value for name, (value, _) in self.held.items()})
        return out


class _Any:
    """A part of a wanted value nothing says (a key of an element the test
    never looked at): anything there will do."""

    def __repr__(self) -> str:
        return "ANY"


ANY = _Any()


class Prefix(list):
    """A wanted list known only by its first elements (`examples[0]`
    looked at, the rest not): it begins with these, whatever follows."""

    def __repr__(self) -> str:
        return "Prefix(" + list.__repr__(self) + ")"


def _partial(value) -> bool:
    """Whether a wanted value leaves some of itself open."""
    if value is ANY or isinstance(value, Prefix):
        return True
    if isinstance(value, dict):
        return any(_partial(one) for one in value.values())
    if isinstance(value, (list, tuple)):
        return any(_partial(one) for one in value)
    return False


def _same(one, other) -> bool:
    if other is ANY:
        return True
    if isinstance(other, Prefix):
        return isinstance(one, list) and len(one) >= len(other) and all(
            _same(a, b) for a, b in zip(one, other))
    if _partial(other):
        if isinstance(other, dict):
            return isinstance(one, dict) and set(one) == set(other) and all(
                _same(one[key], other[key]) for key in other)
        return type(one) is type(other) and len(one) == len(other) and all(
            _same(a, b) for a, b in zip(one, other))
    try:
        if type(one) is not type(other) and not (
                isinstance(one, (int, float)) and
                isinstance(other, (int, float)) and
                not isinstance(one, bool) and not isinstance(other, bool)):
            return False
        return bool(one == other)
    except Exception:                               # noqa: BLE001
        return False


def _fits(program: Program, examples: list, space: dict) -> bool:
    for inputs, wanted in examples:
        try:
            if not _same(program(inputs, space), wanted):
                return False
        except Exception:                           # noqa: BLE001
            return False
    return True


class _Clock:
    """How long one solve has, and what it has grown over which inputs
    (the same inputs are grown over for every part of a value)."""

    def __init__(self, seconds: float) -> None:
        self.until = time.time() + seconds
        self.grown: dict = {}
        self.impure: set = set()

    def over(self) -> bool:
        return time.time() > self.until


# -- search over what is in hand -------------------------------------------------

def _grown(examples: list, library: Library, clock: _Clock) -> list:
    """Expressions over the inputs, grown two levels -- names, their
    fields and items, string and list methods, the library's calls -- each
    with what it gives on every example, one of each set of values
    (observational equivalence): [(Program, values)]."""
    # the inputs themselves kept with the key, so their ids stay theirs
    key = (id(library), tuple(id(inputs) for inputs, _ in examples))
    if key in clock.grown:
        return clock.grown[key][1]
    out = _growing(examples, library, clock)
    if not clock.over():
        clock.grown[key] = ([inputs for inputs, _ in examples], out)
    return out


def _growing(examples: list, library: Library, clock: _Clock) -> list:
    space = library.space()
    names = sorted(set.intersection(*(set(inputs) for inputs, _ in
                                      examples))) if examples else []
    seen: dict = {}
    out: list = []

    def offer(expr: str, imports=(), size=1, called=None) -> None:
        if len(out) > GROWN or clock.over() or called in clock.impure:
            return
        program = Program(expr, tuple(imports), size)
        values = []
        for inputs, _ in examples:
            _PURE.touched = False
            try:
                values.append(program(inputs, space))
            except Exception:                       # noqa: BLE001
                # reached outside (refused, whatever it became -- an import
                # refused its file is "no module"): not tried again this
                # solve, where it was looked for afresh at every call
                if called and _PURE.touched:
                    clock.impure.add(called)
                return
        try:
            key = repr(values)[:3000]
        except Exception:                           # noqa: BLE001
            return
        if key in seen:
            return
        seen[key] = True
        out.append((program, values))

    for name in names:
        offer(name)
    for level in range(2):
        hand = list(out)
        for program, values in hand:
            if clock.over():
                return out
            first = values[0]
            said = program.expr
            if isinstance(first, str):
                for method in ("strip()", "lower()", "upper()",
                               "splitlines()", "split()"):
                    offer(f"{said}.{method}", program.imports,
                          program.size + 1)
                for sep in SEPARATORS:
                    offer(f"{said}.split({sep!r})", program.imports,
                          program.size + 1)
            if isinstance(first, (int, float)) and                     not isinstance(first, bool):
                for form in ("{} % 2 == 0", "{} % 2 == 1", "{} > 0",
                             "{} < 0", "{} == 0", "{} + 1", "{} - 1"):
                    offer(form.format(said), program.imports,
                          program.size + 1)
            if isinstance(first, bool):
                offer(f"not {said}", program.imports, program.size + 1)
            if isinstance(first, str):
                offer(f"bool({said})", program.imports, program.size + 1)
            if isinstance(first, (list, tuple)) and first:
                for at in (0, 1, -1):
                    offer(f"{said}[{at}]", program.imports, program.size + 1)
                offer(f"len({said})", program.imports, program.size + 1)
            if isinstance(first, dict):
                keys = set.intersection(*(set(value) for value in values
                                          if isinstance(value, dict))) \
                    if all(isinstance(value, dict) for value in values) \
                    else set()
                for key in sorted(map(repr, keys))[:20]:
                    offer(f"{said}[{key}]", program.imports,
                          program.size + 1)
            if hasattr(first, "__dict__") and not isinstance(first, type):
                for name in [one for one in vars(first)
                             if not one.startswith("_")][:20]:
                    offer(f"{said}.{name}", program.imports,
                          program.size + 1)
            for name, (value, line) in library.held.items():
                if isinstance(value, dict):
                    offer(f"{name}.get({said}, {said})",
                          program.imports + ((line,) if line else ()),
                          program.size + 1)
                elif callable(value):
                    offer(f"{name}({said})",
                          program.imports + ((line,) if line else ()),
                          program.size + 1, called=name)
    return out


def _searched(examples: list, library: Library, clock: _Clock) -> list:
    """What in hand gives every wanted value: [Program], smallest first."""
    wanted = [value for _, value in examples]
    return sorted((program for program, values in _grown(examples, library,
                                                         clock)
                   if all(_same(got, want)
                          for got, want in zip(values, wanted))),
                  key=lambda one: one.size)


# -- breaking the wanted value down ----------------------------------------------

def solve(examples: list, library: Library | None = None, depth: int = 0,
          clock: _Clock | None = None) -> list:
    """Programs giving each example's wanted value from its inputs, most
    general first: what is in hand; else the value broken down by its
    shape, each part solved, the parts put back."""
    library = library or Library()
    clock = clock or _Clock(SECONDS)
    if examples and all(want is ANY for _, want in examples):
        # nothing said of it anywhere: whatever is put there
        return [Program("None")]
    # an example that says nothing of this part does not bear on it
    examples = [one for one in examples if one[1] is not ANY]
    if not examples or depth > DEPTH or clock.over():
        return []
    space = library.space()
    found = _searched(examples, library, clock)
    wanted = [value for _, value in examples]
    if found and not (isinstance(wanted[0], str) and
                      any(_positional(one) for one in found[:1])):
        return found[:3]
    if found:
        # a piece picked by where it stands (`splitlines()[1]`) is kept
        # beside one picked by what is around it, the latter first
        anchored = _extracted(examples, library, clock)
        return sorted(found[:3] + anchored, key=_general)[:3]
    kinds = {list if isinstance(value, Prefix) else type(value)
             for value in wanted}
    out: list = []
    if kinds - {str} and not any(_partial(value) for value in wanted):
        # a value written out in a string in hand is what a reader of
        # literals makes of that piece of it -- one rule, before taking the
        # value apart into many
        out += _parsed(examples, library, depth, clock)
    if not out and len(kinds) == 1:
        kind = kinds.pop()
        if kind in (tuple, list):
            if not any(isinstance(value, Prefix) for value in wanted):
                out += _fields(examples, library, depth, clock, kind) or []
            if not out:
                out += _mapped(examples, library, depth, clock, kind)
        elif kind is dict:
            out += _keyed(examples, library, depth, clock)
        elif kind is str:
            out += _extracted(examples, library, clock)
            if not out:
                out += _concatenated(examples, library, depth, clock)
            if not out:
                out += _joined(examples, library, depth, clock)
    if not out and all(value in PLAIN for value in wanted) and \
            len({repr(value) for value in wanted}) == 1:
        out.append(Program(repr(wanted[0])))
    return sorted([one for one in out if _fits(one, examples, space)],
                  key=_general)[:3]


def _positional(program: Program) -> int:
    """How many places a program picks by number (`[1]`, `[-2]`): a fact
    of these examples more often than of what is meant."""
    return len(re.findall(r"\[\s*-?[1-9]\d*\s*\]", program.expr))


def _general(program: Program) -> tuple:
    """Most general first: fewest picks by place, then smallest."""
    return (_positional(program), program.size)


def _fields(examples, library, depth, clock, kind) -> list:
    """A tuple or list as long in every example: a goal per place."""
    lengths = {len(value) for _, value in examples}
    if len(lengths) != 1 or not 0 < next(iter(lengths)) <= 8:
        return []
    parts = []
    for at in range(next(iter(lengths))):
        found = solve([(inputs, value[at]) for inputs, value in examples],
                      library, depth + 1, clock)
        if not found:
            return []
        parts.append(found[0])
    said = ", ".join(one.expr for one in parts)
    imports = tuple(sorted({line for one in parts for line in one.imports}))
    expr = f"({said},)" if kind is tuple else f"[{said}]"
    return [Program(expr, imports, sum(one.size for one in parts) + 1)]


def _mapped(examples, library, depth, clock, kind) -> list:
    """A list made of the elements of another: each element a goal of the
    one it is made from (`[f(one) for one in L]`), or the elements kept
    (`[one for one in L if p(one)]`)."""
    out = []
    lists = [(program, values) for program, values in
             _grown(examples, library, clock)
             if all(isinstance(value, (list, tuple)) for value in values)]
    for program, values in lists:
        if clock.over():
            break
        if all(len(value) == len(want) or isinstance(want, Prefix) and
               len(value) >= len(want) for value, (_, want) in
               zip(values, examples)) and \
                any(not isinstance(want, Prefix) for _, want in examples):
            inner = [({**inputs, "one": item}, want_item)
                     for (inputs, want), value in zip(examples, values)
                     for item, want_item in zip(value, want)]
            if not inner:
                continue
            found = solve(inner, library, depth + 1, clock)
            if found:
                body = found[0]
                close = "]" if kind is list else ")"
                open_ = "[" if kind is list else "tuple("
                out.append(Program(
                    f"{open_}{body.expr} for one in {program.expr}{close}",
                    tuple(sorted(set(body.imports + program.imports))),
                    body.size + program.size + 1))
                break
        elif all(all(any(_same(item, one) for one in value) for item in want)
                 and len(value) > len(want)
                 for value, (_, want) in zip(values, examples)):
            inner = [({**inputs, "one": item},
                      any(_same(item, one) for one in want))
                     for (inputs, want), value in zip(examples, values)
                     for item in value]
            found = solve(inner, library, depth + 1, clock)
            if found:
                out.append(Program(
                    f"[one for one in {program.expr} if {found[0].expr}]",
                    tuple(sorted(set(found[0].imports + program.imports))),
                    found[0].size + program.size + 1))
                break
    if not out:
        out += _kept_and_made(examples, library, depth, clock, lists)
    return out


def _leaves(value) -> list:
    """The strings inside a wanted value, three characters or more."""
    if isinstance(value, str):
        return [value] if len(value.strip()) >= 3 else []
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        return [one for item in value for one in _leaves(item)]
    return []


def _kept_and_made(examples, library, depth, clock, lists) -> list:
    """A list made of some of another's elements (`[f(one) for one in L if
    p(one)]`): in the examples whose wanted list is whole, each wanted
    element is of the element of L that holds its words (the line a saying
    of it was cut from), in order; what tells those apart from the rest is
    found (p), and each wanted element found of its own (f) -- where the
    wanted list is known only by its first elements, as many as it says."""
    whole = [at for at, (_, want) in enumerate(examples)
             if not isinstance(want, Prefix) and want]
    if not whole:
        return []
    space = library.space()
    for program, values in lists[:40]:
        if clock.over():
            break
        labelled, ok = [], True
        for at in whole:
            items, want = values[at], examples[at][1]
            if not all(isinstance(item, str) for item in items):
                ok = False
                break
            chosen, start = [], 0
            for element in want:
                words = sorted(_leaves(element), key=len, reverse=True)
                where = next((index for index in range(start, len(items))
                              if words and words[0] in items[index]), None)
                if where is None:
                    ok = False
                    break
                chosen.append(where)
                start = where + 1
            if not ok:
                break
            labelled += [({**examples[at][0], "one": item}, index in chosen)
                         for index, item in enumerate(items)]
        if not ok or all(kept for _, kept in labelled):
            continue
        # what tells the elements kept from the rest: a flag in hand, or a
        # word the kept all hold (`'=' in one`) -- the first that keeps as
        # many as every example wants, those it knows only in part too
        for test in itertools.islice(conditions(
                [inputs for inputs, kept in labelled if kept],
                [inputs for inputs, kept in labelled if not kept],
                library), 8):
            inner = []
            for (inputs, want), value in zip(examples, values):
                taken = [item for item in value
                         if _safe(test, {**inputs, "one": item}, space)]
                if len(taken) < len(want) or not isinstance(
                        want, Prefix) and len(taken) != len(want):
                    inner = []
                    break
                inner += [({**inputs, "one": item}, element)
                          for item, element in zip(taken, want)]
            if not inner:
                continue
            made = solve(inner, library, depth + 1, clock)
            if made:
                return [Program(
                    f"[{made[0].expr} for one in {program.expr} "
                    f"if {test.expr}]",
                    tuple(sorted(set(made[0].imports + test.imports +
                                     program.imports))),
                    made[0].size + test.size + program.size + 1)]
    return []


def _safe(program: Program, inputs: dict, space: dict):
    try:
        return program(inputs, space)
    except Exception:                               # noqa: BLE001
        return None


#: how a piece of text is read as a value: as written, and as the items
#: of a call's arguments (`(1, 2), None` read as `[(1, 2), None]`)
READ_AS = ("{}", "[{}]")


def _plain(value):
    """A value read as data: tuples as lists (what JSON keeps)."""
    if isinstance(value, (list, tuple)):
        return [_plain(one) for one in value]
    if isinstance(value, dict):
        return {key: _plain(one) for key, one in value.items()}
    return value


def _read(text: str):
    import ast
    return _plain(ast.literal_eval(text))


def _windows(text: str, want, most: int) -> list:
    """The pieces of a text that, read as a literal, are the wanted value:
    [(piece, form)], cut where a word does not go on, first found first."""
    out = []
    # a literal's words are its value's (`(1, 2), None`: 1 2 None): only
    # where the text has those words in a row is a piece tried, widened
    # over the marks around them
    words = re.findall(r"\w+", repr(want))
    if not words:
        return out
    tokens = list(re.finditer(r"\w+", text))
    for at in range(len(tokens) - len(words) + 1):
        if [one.group() for one in tokens[at:at + len(words)]] != words:
            continue
        first, last = tokens[at], tokens[at + len(words) - 1]
        before = tokens[at - 1].end() if at else 0
        after = tokens[at + len(words)].start() \
            if at + len(words) < len(tokens) else len(text)
        for start in range(max(before, first.start() - 4),
                           first.start() + 1):
            for end in range(last.end(), min(after, last.end() + 6) + 1):
                piece = text[start:end]
                if piece != piece.strip() or len(piece) > most:
                    continue
                for form in READ_AS:
                    try:
                        if _same(_read(form.format(piece)), want):
                            out.append((piece, form))
                    except Exception:               # noqa: BLE001
                        pass
    return out


def _parsed(examples, library, depth, clock) -> list:
    """A value that is a piece of a string in hand read as a literal --
    the piece found again as a string goal (by what is around it), read by
    `ast.literal_eval` (tuples as lists)."""
    out = []
    for program, values in _strings(examples, library, clock)[:30]:
        if clock.over():
            break
        found = []
        for value, (_, want) in zip(values, examples):
            most = 3 * len(repr(want)) + 4
            found.append(_windows(value[:600], want, most))
        if not all(found):
            continue
        for form in READ_AS:
            for pick in (0, -1):
                pieces = [[one for one in each if one[1] == form]
                          for each in found]
                if not all(pieces):
                    continue
                goal = [(inputs, each[pick][0]) for (inputs, _), each in
                        zip(examples, pieces)]
                inner = solve(goal, library, depth + 1, clock)
                if not inner:
                    continue
                said = inner[0].expr if form == "{}" else \
                    f"'[' + {inner[0].expr} + ']'"
                out.append(Program(
                    f"json.loads(json.dumps(ast.literal_eval({said})))",
                    tuple(sorted(set(inner[0].imports +
                                     ("import ast", "import json")))),
                    inner[0].size + 1))
                return out
    return out


def _keyed(examples, library, depth, clock) -> list:
    """A dict with the same keys in every example: a goal per key."""
    keys = [set(value) for _, value in examples]
    if any(one != keys[0] for one in keys) or len(keys[0]) > 12:
        return []
    parts = []
    for key in sorted(keys[0], key=repr):
        found = solve([(inputs, value[key]) for inputs, value in examples],
                      library, depth + 1, clock)
        if not found:
            return []
        parts.append((key, found[0]))
    said = ", ".join(f"{key!r}: {one.expr}" for key, one in parts)
    imports = tuple(sorted({line for _, one in parts
                            for line in one.imports}))
    return [Program("{" + said + "}", imports,
                    sum(one.size for _, one in parts) + 1)]


def _strings(examples, library, clock) -> list:
    """The string-valued expressions in hand: [(Program, values)]."""
    return [(program, values) for program, values in
            _grown(examples, library, clock)
            if all(isinstance(value, str) for value in values)]


def _common_suffix(texts: list) -> str:
    out = ""
    for chars in zip(*(text[::-1] for text in texts)):
        if len(set(chars)) != 1:
            break
        out = chars[0] + out
    return out


def _common_prefix(texts: list) -> str:
    out = ""
    for chars in zip(*texts):
        if len(set(chars)) != 1:
            break
        out += chars[0]
    return out


def _extracted(examples, library, clock) -> list:
    """A string that is a piece of a string in hand -- by what it begins
    with first, in every string in hand (the line that begins with `def
    `); only then by what is around it in each (the same text before it,
    or the start; the same after, or the end). Tried the other way, a
    flattened copy of the text, read first, was cut by what happened to
    stand beside the piece (`s ` of `vowels`)."""
    space = library.space()
    holding = [(program, values) for program, values in
               _strings(examples, library, clock)
               if all(want and want in value for value, (_, want) in
                      zip(values, examples))]
    head = _common_prefix([want for _, want in examples])
    # where a word of it ends, the fewest words first: `def `, not `def
    # vowe` -- with one example, all of it is shared
    ends = [at for at in range(2, min(len(head), 12) + 1)
            if head[at - 1].isalnum() and
            (at == len(head) or not head[at].isalnum())]
    for length in ends:
        start = head[:length] + (" " if head[length:length + 1] == " "
                                 else "")
        pattern = f"^({re.escape(start)}.*?)$"
        for program, _ in holding:
            candidate = Program(
                f"re.search({pattern!r}, {program.expr}, re.M).group(1)",
                program.imports, program.size + 2)
            if _fits(candidate, examples, space):
                return [candidate]
    for program, values in holding:
        befores, afters = [], []
        for value, (_, want) in zip(values, examples):
            at = value.index(want)
            befores.append(value[:at])
            afters.append(value[at + len(want):])
        for left in _contexts(befores, suffix=True):
            for right in _contexts(afters, suffix=False):
                pattern = (("^" if left == "^" else re.escape(left))
                           + "(.*?)" +
                           ("$" if right == "$" else re.escape(right)))
                candidate = Program(
                    f"re.search({pattern!r}, {program.expr}, re.M | re.S)"
                    f".group(1)", program.imports, program.size + 2)
                if _fits(candidate, examples, space):
                    return [candidate]
    return []


def _contexts(texts: list, suffix: bool) -> list:
    """What is around a piece in every example, the tightest first: the
    shared text next to it (all of it, then less), else the start or end
    of a line, else of the whole."""
    shared = _common_suffix(texts) if suffix else _common_prefix(texts)
    out = []
    # the least that tells it apart first: a line's edge, then the
    # shortest shared text -- with one example all it shares is all of
    # it, and the whole of it is a fact of that example
    line = "\n"
    if all(text.endswith(line) if suffix else text.startswith(line)
           for text in texts):
        out.append(line)
    # one example shares all of itself with itself: any text beside the
    # piece fits it, and fit by chance (`'\ _(.*?)si'` cut a saying out of
    # a docstring); what two examples share says something of both
    # -- and one character shared by three or more is no chance (`(` before
    # the arguments of `vowels(`, `add(`, `f(`), where by two it was
    least = 1 if len(texts) >= 3 else 2
    for length in range(least, min(len(shared), 12) + 1) \
            if len(texts) >= 2 else ():
        piece = shared[-length:] if suffix else shared[:length]
        out.append(piece)
    out.append("^" if suffix else "$")
    return list(dict.fromkeys(out))


def _joined(examples, library, depth, clock) -> list:
    """A string of pieces between one separator: the pieces a list, found
    again (`", ".join(...)`)."""
    for sep in SEPARATORS:
        if not all(sep in want for _, want in examples):
            continue
        inner = [(inputs, want.split(sep)) for inputs, want in examples]
        if any(not piece for _, pieces in inner for piece in pieces):
            # `a:` is not `a` and nothing joined by `:`
            continue
        found = solve(inner, library, depth + 1, clock)
        if found:
            return [Program(f"{sep!r}.join({found[0].expr})",
                            found[0].imports, found[0].size + 1)]
    return []


def _pieces(want: str, held: list) -> list:
    """A string cut into what is in hand and what is not: at each place,
    the longest piece (two characters or more) of any string in hand, else
    one character of glue -- [(piece, in hand)]."""
    out: list = []
    at = 0
    while at < len(want):
        best = 0
        for one in held:
            for end in range(len(want), at + 1, -1):
                if end - at > best and want[at:end] in one:
                    best = end - at
                    break
        if best >= 2:
            out.append([want[at:at + best], True])
            at += best
        else:
            if out and not out[-1][1]:
                out[-1][0] += want[at]
            else:
                out.append([want[at], False])
            at += 1
    return out


def _concatenated(examples, library, depth, clock) -> list:
    """A string of pieces one after another: the pieces from what is in
    hand found again, the glue between them the same in every example."""
    strings = _strings(examples, library, clock)
    held = [[value for value in values] for _, values in strings]
    cuts = []
    for at, (_, want) in enumerate(examples):
        cuts.append(_pieces(want, [values[at] for values in held]))
    shapes = {tuple(kind for _, kind in cut) for cut in cuts}
    if len(shapes) != 1:
        return []
    shape = next(iter(shapes))
    if not any(shape):
        return []
    parts = []
    for place, in_hand in enumerate(shape):
        if not in_hand:
            glue = {cut[place][0] for cut in cuts}
            if len(glue) != 1:
                return []
            parts.append(Program(repr(next(iter(glue)))))
            continue
        found = solve([(inputs, cut[place][0]) for (inputs, _), cut in
                       zip(examples, cuts)], library, depth + 1, clock)
        if not found:
            return []
        parts.append(found[0])
    imports = tuple(sorted({line for one in parts for line in one.imports}))
    return [Program(" + ".join(one.expr for one in parts), imports,
                    sum(one.size for one in parts) + 1)]


# -- conditions -------------------------------------------------------------------

def condition(apart: list, kept: list, library: Library | None = None
              ) -> Program | None:
    """What tells two groups of inputs apart -- true of every one of
    `apart`, false of every one of `kept`: an expression in hand that
    gives a flag, or whether a string in hand holds a piece the first
    group's all hold and the second's none."""
    return next(conditions(apart, kept, library), None)


def conditions(apart: list, kept: list, library: Library | None = None):
    """Each thing that tells two groups apart, in `condition`'s order: the
    first of them may be true of these by chance (a quote mark held by
    every saying of one request's examples), and another case decides."""
    library = library or Library()
    clock = _Clock(SECONDS)
    examples = [(inputs, True) for inputs in apart] + \
        [(inputs, False) for inputs in kept]
    yield from _searched(examples, library, clock)[:5]
    space = library.space()
    # a word of what was given, not of something made of it: in a string
    # made of the inputs a word is there by how it was made (`'2' in
    # _norm(len(spans))`)
    for program, values in [one for one in _strings(examples, library,
                                                     clock)
                            if one[0].size == 1]:
        firsts = values[:len(apart)]
        rest = values[len(apart):]
        # the shortest piece the first group's strings all hold, none of
        # the second's: a word of theirs, by the line it is on
        # a word its own code uses before one only these examples hold (a
        # request about `day` set apart from the rest by `day`, where every
        # `def` request is the kind)
        for word in sorted({one for text in firsts[:1]
                            for one in re.findall(r"[\w']+|\S", text)},
                           key=lambda one: (not one[0].isalnum(),
                                            one not in library.words(),
                                            len(one), one)):
            if all(word in text for text in firsts) and \
                    not any(word in text for text in rest):
                candidate = Program(f"{word!r} in {program.expr}",
                                    program.imports, program.size + 1)
                if _fits(candidate, examples, space):
                    yield candidate
