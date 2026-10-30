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

Nothing here changes any code: it says what is missing and where it is, for
what makes the change to start from. No word of a request is read here, and
the wanted words are given: what a request means is the reader's.

    known(turns, wanted)    {author, lacks, made, held, siblings}
"""
from __future__ import annotations

import dataclasses
import importlib
import inspect
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


def known(turns: list, wanted: list) -> dict:
    """What the answer to a case lacks of `wanted`, and where it is known.
    Said twice: once for who wrote the reply, once to look around there."""
    wrote = inside.author(list(turns))
    out = {"author": wrote, "lacks": [], "made": [], "held": [],
           "siblings": [], "reply": ""}
    if not wrote:
        return out
    name = wrote["function"].rpartition(".")[2]
    file = wrote["file"]
    scope: dict = {}
    made: dict = {}
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
        lacks = out["lacks"]
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
