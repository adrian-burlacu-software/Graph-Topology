"""What an answer lacks, and where the architecture already knows it (v705):
the first steps of changing its own code by reasoning instead of by writing
a change in one go.

A wrong answer is said again, with who wrote it found (`inside.author`). What
was wanted of it and is not in it is then looked for in what the author had in
hand when it answered, three ways, nearest first:

- **made and dropped** (`made`): a call of the turn gave the words back, and
  they went no further;
- **held** (`found_in`): a value in scope holds them, however deep
  (`held.index().symbols[*].name`);
- **said by a sibling** (`siblings`): another function of the author's module,
  filled by name from that same scope, says them (`_a_file`).

And two things beside where the words are, for a reply is often written by a
function doing its job, sent there by something earlier:

- **what decided** (`decisions`): each frame's branches and table lookups on
  the way to the author, with the values that took them and what else they
  could have been;
- **what tells them apart** (`selectors`): the records holding the wanted
  words, and the fields that pick out those and no others (`kind` is `class`,
  `file` is `rental/hires.py`) -- and whether the message says those values.

Nothing here changes any code: it says what is missing and where it is, for
what makes the change to start from. No word of a request is read here, and
the wanted words are given: what a request means is the reader's.

    known(turns, wanted)    {author, lacks, made, held, siblings}
"""
from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import itertools
import re
import sys
import time

from research.v705 import inside

#: how far into a value, and how many values of it, a search looks
DEEP = 5
LOOKED = 20000
#: how many frames up from the author its scope is read
FRAMES = 3
#: how long the siblings of an author are tried for, seconds
TRYING = 60
#: how much of what a sibling said is kept
KEPT = 300
#: how many frames up from the author the decisions are read
DECIDING = 6
#: how many fields together pick records out
PICKED_BY = 2


def says(text: str, word: str) -> bool:
    """Whether a text says a word whole: `7` is not said by `tools.py:17`."""
    return bool(re.search(rf"(?<![\w.:/]){re.escape(word)}(?![\w/]|\.\w)",
                          text))


def _spoken(value) -> str:
    """The sentences a value holds: not the source it shows beside them,
    which holds every word of the file and so says nothing of what it knows."""
    return " ".join(one for one in inside._texts(value, 4) if "\n" not in one)


def _its_own(value) -> bool:
    """Whether a value is of the architecture's own classes: gone into, where
    a library's or the runtime's is not."""
    return type(value).__module__.startswith("research.")


def _returns_something(method) -> bool:
    """A method called with nothing, and saying it gives something: reading,
    not doing."""
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    return not signature.parameters and signature.return_annotation not in (
        inspect.Signature.empty, None, "None")


def _parts(value):
    """(how it is said, what it holds) of a value's parts."""
    if isinstance(value, dict):
        for key, one in value.items():
            yield "[*]", one
            if isinstance(key, str):
                yield "[key]", key
    elif isinstance(value, (list, tuple, set, frozenset)):
        for one in value:
            yield "[*]", one
    elif _its_own(value) and not inspect.isclass(value):
        fields = dataclasses.fields(value) if \
            dataclasses.is_dataclass(value) else ()
        for name in [one.name for one in fields] or \
                list(getattr(value, "__dict__", ())):
            yield f".{name}", getattr(value, name, None)
        for name, method in inspect.getmembers(type(value),
                                               inspect.isfunction):
            if name.startswith("_"):
                continue
            bound = getattr(value, name)
            if _returns_something(bound):
                try:
                    yield f".{name}()", bound()
                except Exception:                   # noqa: BLE001
                    continue


def found_in(scope: dict, wanted: list) -> list:
    """Where the values of a scope hold the wanted words, whole: [(path, the
    words found there)], most of them first. Paths of one shape are one
    (`.symbols[*].name`), for what the words have in common is where they
    are."""
    places: dict = {}
    seen: set = set()
    count = 0
    stack = [(name, value, 0) for name, value in scope.items()]
    while stack and count < LOOKED:
        path, value, depth = stack.pop()
        count += 1
        if isinstance(value, str):
            for word in wanted:
                if word == value:
                    places.setdefault(path, set()).add(word)
            continue
        if value is None or isinstance(value, (bool, int, float, bytes)) or \
                inspect.ismodule(value) or inspect.isroutine(value) or \
                depth >= DEEP or id(value) in seen:
            continue
        seen.add(id(value))
        for said, part in _parts(value):
            stack.append((path + said, part, depth + 1))
    return sorted(((path, sorted(words)) for path, words in places.items()),
                  key=lambda one: (-len(one[1]), len(one[0]), one[0]))


def _module_of(file: str):
    return importlib.import_module(file.removesuffix(".py").replace("/", "."))


def filled(function, scope: dict) -> dict | None:
    """The arguments of a function by name from a scope; None where one it
    needs is not there."""
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return None
    out = {}
    for name, one in parameters.items():
        if one.kind in (one.VAR_POSITIONAL, one.VAR_KEYWORD):
            continue
        if name in scope:
            out[name] = scope[name]
        elif one.default is inspect.Parameter.empty:
            return None
    return out


def siblings(module, name: str, scope: dict, wanted: list,
             seconds: float = TRYING) -> list:
    """The other functions of a module that say the wanted words, called with
    what the scope holds: [{function, says, gave}], most of them first."""
    out = []
    end = time.monotonic() + seconds
    for other, function in vars(module).items():
        if time.monotonic() > end:
            break
        if other == name or not inspect.isfunction(function) or \
                function.__module__ != module.__name__:
            continue
        given = filled(function, scope)
        if not given:
            continue
        try:
            gave = function(**given)
        except Exception:                           # noqa: BLE001
            continue
        words = _spoken(gave)
        said = [one for one in wanted if says(words, one)]
        if said:
            out.append({"function": other, "says": said, "gave": words[:KEPT]})
    return sorted(out, key=lambda one: -len(one["says"]))


def _record_fields(record) -> dict:
    """The plain fields of a record: a dict's, or an object of its own
    classes'."""
    if isinstance(record, dict):
        fields = record
    elif _its_own(record) and not inspect.isclass(record):
        fields = {one.name: getattr(record, one.name, None) for one in
                  dataclasses.fields(record)} if \
            dataclasses.is_dataclass(record) else vars(record)
    else:
        return {}
    return {key: value for key, value in fields.items() if isinstance(key, str)
            and isinstance(value, (str, int, bool)) or
            isinstance(key, str) and value is None}


def collections_in(scope: dict) -> list:
    """The collections of records a scope holds, however deep: [(path,
    [record])]."""
    out = []
    seen: set = set()
    count = 0
    stack = [(name, value, 0) for name, value in scope.items()]
    while stack and count < LOOKED:
        path, value, depth = stack.pop()
        count += 1
        if value is None or isinstance(value, (str, bool, int, float, bytes)) \
                or inspect.ismodule(value) or inspect.isroutine(value) or \
                depth >= DEEP or id(value) in seen:
            continue
        seen.add(id(value))
        items = list(value.values()) if isinstance(value, dict) else \
            list(value) if isinstance(value, (list, tuple)) else []
        records = [one for one in items if _record_fields(one)]
        if len(records) >= 2:
            out.append((path, records))
        for said, part in _parts(value):
            if said != "[key]":
                stack.append((path + said, part, depth + 1))
    return out


def said_of(value, text: str) -> bool:
    """Whether a message says a field's value, in the form it has there
    (`class` in `classes`, `hires.py` of `rental/hires.py`)."""
    if not isinstance(value, str) or len(value) < 3:
        return False
    words = re.findall(r"[\w./-]+", text.lower())
    value = value.lower()
    return any(value in one or (len(one) >= 4 and one in value)
               for one in words)


def selectors(scope: dict, wanted: list, text: str = "",
              unwanted: tuple = ()) -> list:
    """What tells the wanted apart: for each collection with a field naming
    all of them, the fields whose values pick out those records and no
    others -- fewest fields first, those the message says first. [{path,
    named by, fields: {field: value}, said: [fields the message says]}]. A
    name's own field picks anything out, so says nothing and is left out."""
    out = []
    for path, records in collections_in(scope):
        rows = [_record_fields(one) for one in records]
        names = {key for row in rows for key, value in row.items()
                 if value in wanted}
        for name in sorted(names):
            hits = [row for row in rows if row.get(name) in wanted]
            if {row[name] for row in hits} != set(wanted):
                continue
            others = [row for row in rows if row not in hits]
            if not others or any(row.get(name) in unwanted for row in hits):
                continue
            common = {key: value for key, value in hits[0].items()
                      if key != name and value is not None and
                      all(row.get(key) == value for row in hits)}
            for size in range(1, PICKED_BY + 1):
                found = False
                for keys in itertools.combinations(sorted(common), size):
                    if any(all(row.get(key) == common[key] for key in keys)
                           for row in others):
                        continue
                    found = True
                    out.append({"path": path, "named by": name,
                                "fields": {key: common[key] for key in keys},
                                "said": [key for key in keys
                                         if said_of(common[key], text)]})
                if found:
                    break
    return sorted(out, key=lambda one: (len(one["said"]) -
                                        len(one["fields"]) < 0,
                                        len(one["fields"]), len(one["path"])))


def decided(source: str, first: int, line: int, values: dict) -> list:
    """What took a function to a line of it: the tests of the branches the
    line is under, and the table lookup on the line itself, each with the
    values of the names in it. `source` is the function's text and `first`
    the line it starts on in its file."""
    try:
        tree = ast.parse(inspect.cleandoc(source) if source[:1].isspace()
                         else source)
    except SyntaxError:
        return []
    at = line - first + 1
    out = []

    def named(node):
        return {one.id: inside._shown(values[one.id]) for one in ast.walk(node)
                if isinstance(one, ast.Name) and one.id in values and
                not callable(values[one.id])}

    def options(node):
        """What else the names of a test are compared with in this function."""
        names = {one.id for one in ast.walk(node) if isinstance(one, ast.Name)}
        found = set()
        for one in ast.walk(tree):
            if isinstance(one, ast.Compare) and isinstance(one.left, ast.Name) \
                    and one.left.id in names:
                for right in one.comparators:
                    for constant in ast.walk(right):
                        if isinstance(constant, ast.Constant) and \
                                isinstance(constant.value, str):
                            found.add(constant.value)
        return sorted(found)

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            for branch, body in (("then", node.body), ("else", node.orelse)):
                if body and body[0].lineno <= at <= body[-1].end_lineno:
                    out.append({"test": ast.get_source_segment(
                        source, node.test) or "", "took": branch,
                        "values": named(node.test),
                        "other": options(node.test)})
        elif isinstance(node, ast.Subscript) and node.lineno == at and \
                isinstance(node.slice, ast.Name) and \
                isinstance(node.value, ast.Name):
            table = values.get(node.value.id)
            out.append({"test": f"{node.value.id}[{node.slice.id}]",
                        "took": "table", "values": named(node.slice),
                        "other": sorted(map(str, table))
                        if isinstance(table, dict) else []})
    return out


def _decisions_of(frames: list) -> list:
    """`decided` of frames on the way to an author, nearest first."""
    import linecache
    out = []
    for name, path, first, line, values in frames:
        lines = linecache.getlines(path)
        block = inspect.getblock(lines[first - 1:]) if lines else []
        found = decided("".join(block), first, line, values)
        if found:
            out.append({"function": name, "line": line, "decided": found})
    return out


def known(turns: list, wanted: list, unwanted: tuple = ()) -> dict:
    """What the answer to a case lacks of `wanted`, and where it is known.
    Said twice: once for who wrote the reply, once to look around there."""
    wrote = inside.author(list(turns))
    out = {"author": wrote, "lacks": [], "excess": [], "made": [], "held": [],
           "siblings": [], "decisions": [], "selectors": [], "reply": ""}
    if not wrote:
        return out
    name = wrote["function"].rpartition(".")[2]
    file = wrote["file"]
    scope: dict = {}
    made: dict = {}
    frames: list = []
    root = inside._relative(sys.modules[__name__].__file__).split("/")[0]

    def local(frame, event, arg):
        if event != "return":
            return local
        code = frame.f_code
        words = _spoken(arg)
        for word in wanted:
            if says(words, word):
                made.setdefault(code.co_qualname, set()).add(word)
        if code.co_name == name and file in code.co_filename.replace(
                "\\", "/"):
            scope.clear()
            at = frame
            for _ in range(FRAMES):
                if at is None:
                    break
                # the nearest frame's name wins
                for key, value in at.f_locals.items():
                    scope.setdefault(key, value)
                at = at.f_back
            frames.clear()
            at = frame.f_back
            for _ in range(DECIDING):
                if at is None:
                    break
                frames.append((at.f_code.co_qualname, at.f_code.co_filename,
                               at.f_code.co_firstlineno, at.f_lineno,
                               {**at.f_globals, **at.f_locals}))
                at = at.f_back
        return local

    def tracer(frame, event, arg):
        path = inside._relative(frame.f_code.co_filename)
        if event != "call" or not path or not path.startswith(root + "/") or \
                inside.of_the_exam(path):
            return None
        frame.f_trace_lines = False
        return local

    def look(turn, root_):
        # what it finds by looking is no part of the answer
        sys.settrace(None)
        from research.v704.converse import text_of
        reply = " ".join(text_of(turn).split())
        out["reply"] = reply
        out["lacks"] = [one for one in wanted if not says(reply, one)]
        out["excess"] = [one for one in unwanted if says(reply, one)]
        lacks = out["lacks"]
        if scope:
            out["selectors"] = selectors(scope, list(wanted), turns[-1],
                                         tuple(unwanted))[:5]
            out["decisions"] = _decisions_of(frames)
        if lacks and scope:
            out["held"] = found_in(scope, lacks)[:5]
            out["siblings"] = siblings(_module_of(file), name, scope, lacks)
        return True

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        inside._project_turn(list(turns), look)
    finally:
        sys.settrace(previous)
    out["made"] = sorted(((who, sorted(words)) for who, words in made.items()),
                         key=lambda one: -len(one[1]))[:5]
    return out
