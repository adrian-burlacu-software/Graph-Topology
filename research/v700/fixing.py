"""A change asked of the project, made in it: *`_turn` reads the number
from "n", the server sends "number" -- read "number"*.

The reader reads the message as a change (`change`) of something the
project has -- a function, a file (`asking.resolve`). Here the change is
made:

1. **The part changed** is what was named: the function (decorators and
   all), or the file. Where that is longer than the editor reads, the
   lines of it around the code the statement names (`math`,
   `PredicateTrie`) -- looked up among the names the code uses, not its
   comments or docs: a lookup, not a reading.
2. **The editor writes it** (`teach_editor.py`, `llm/editor`): the change
   said as blocks of the lines as they are and as they become -- greedy
   first, then sampled. An answer whose old lines are not found in the part
   once exactly is refused, not guessed into place.
3. **Each change is checked** before anything is written: the file still
   reads (it parses), and the compiler says nothing more is wrong with it
   than before. Of the changes that pass, the one the most answers agree
   on is taken, the greedy one first among equals.
4. **It is written** -- in the project, and on disk where the editor said
   where the project is (`Project.root`): the change is made, not handed
   back. What it was is kept, so *undo that* puts it back.
"""
from __future__ import annotations

import difflib
import re
import time
import zlib

from research.v700 import teach_editor as T

#: the editor used: taught again on faults found in real code and commits
#: of small changes inside a function (`teach_faults.py`, `mix`) -- the
#: first editor's 80 of 160 held faults are 155 of 160 -- then on changes
#: of data files (research/v701/teach_data_edits.py): held data edits 31
#: of 181 right are 160, faults 153 of 160, small commits 12 -> 15 of 200;
#: then on a record's field set (`rows-more`): held 21 of 71 are 62, data
#: edits 163, faults 150, small commits 17; then (v703 `teach_parts`) on
#: a function's part of a commit across files, what the project has, parts
#: left as they were (NOTHING) and what a message names by its words:
#: held parts 0 -> 12 of 300 (any of 5: 4 -> 18), commits 24 -> 20,
#: NOTHING on 115 of 200 unchanged, 14 of 200 changed -- the editor of a
#: plan's steps (`STEP_EDITOR`), given what a step is given; a change asked
#: alone keeps editor5 (attach: editor8 did not move pytypes' import out
#: of the loop, editor5 does)
EDITOR, STEP_EDITOR = "editor5", "editor8"
SEED = 700
#: answers sampled beside the greedy one, in each round; and the rounds,
#: asked again while nothing passes that changes all the statement names
SAMPLES, ROUNDS = 6, 3
#: the rounds asked where each change is tried by running its case again:
#: the run, not agreement, decides -- so more answers may be looked at
VERIFIED_ROUNDS = 10
#: of those, the rounds written in one batch -- one: three at once (18
#: answers) took 3.5 GB of the card beyond the editor, one 1.2, for 9.9 s
#: against 4.8 (the longest answer bounds a batch, not the count), and
#: beside a goal's other models the card spilled
CHUNKED = 1
#: below this the judge is sure a change is not what was said, and it is
#: refused; above it, the judge only weighs -- it gave removing an unused
#: import alone 0.18, and the same with a docstring rewritten 0.70
JUDGE_FLOOR = 0.1
#: how many answers, written apart, must make a change for it to be written
LEAST_AGREED = 2
#: what the editor reads at most (`teach_editor.LONGEST_PART`), and the
#: lines kept around what a statement names in what is longer
LONGEST_PART, AROUND = T.LONGEST_PART, 8
#: changes made, by conversation: (path, text before, text after)
MADE: dict = {}


def _editor(step: bool = False):
    from research.v697.coding import Tools
    return Tools.get().writer(STEP_EDITOR if step and (
        T.LLM / STEP_EDITOR / "config.json").exists() else EDITOR)


def available() -> bool:
    return (T.LLM / EDITOR / "config.json").exists()


def part_of(held, subject, statement: str = "") -> tuple | None:
    """(path, first line, last line) -- 1-based, inclusive -- of what is
    changed: a function, decorators included, or a file whole; where that
    is longer than the editor reads, the lines of it around the code the
    statement names (`window`)."""
    found = span(held, subject)
    if found is None:
        return None
    start, end = found
    from research.v698.project import is_data
    if subject.kind == "file" and is_data(subject.file):
        # a data file: the block the statement is of, as the editor was
        # taught its changes, never the file whole (v701 `placing`)
        from research.v701 import placing
        placed_ = placing.window(held.files[subject.file], statement,
                                 AROUND, LONGEST_PART,
                                 placing.tabular(subject.file))
        # a row the statement gives whole is added after the last, as the
        # editor was taught adding a record: the table's end shown
        if placing.appended(subject.file, held.files[subject.file],
                            statement):
            count = len(held.files[subject.file].splitlines())
            placed_ = (max(1, count - 2 * AROUND), count)
        if placed_ is not None:
            return (subject.file, *placed_)
    lines = held.files[subject.file].splitlines()
    if end - start + 1 > LONGEST_PART:
        found = window(lines, start, end, statement, subject.file)
        if found is None:
            return None
        start, end = found
    return subject.file, start, end


#: what a name is written as, to be one: in backquotes, called, or shaped
#: as code (`next_n_fibs`, `skipQuestion`, `header()`)
SHAPED = re.compile(r"`([A-Za-z_]\w*)(?:\([^`]*\))?`|\b([A-Za-z_]\w*)\(|"
                    r"\b([a-z]+_[a-z_\d]+|[a-z]+[A-Z]\w*)\b")


def added(statement: str, held) -> list:
    """The names a change asks for that the project has nowhere -- written
    as code are, in backquotes or shaped so (`add a method next_n_fibs`):
    what the change adds, not what it changes."""
    if held is None or not held.files:
        return []
    import builtins
    import keyword
    index = held.index()
    said = [next(one for one in found if one) for found in
            SHAPED.findall(statement)]
    text = "\n".join(held.files.values())
    out = []
    for name in dict.fromkeys(said):
        if keyword.iskeyword(name) or hasattr(builtins, name) or \
                index.find(name) or re.search(rf"\b{re.escape(name)}\b",
                                              text):
            continue
        out.append(name)
    return out


def several(statement: str, held) -> bool:
    """Whether a change names two or more of one file's functions or
    methods (`implement draw in the Deck class and receive_card in the
    Player class`): a change of that file, not of one of them. A class
    named beside one function is where that function's change looks
    (`lend should refuse a Member holding the most`), not another."""
    if held is None or not held.files:
        return False
    index = held.index()
    found: dict = {}
    for word in dict.fromkeys(re.findall(r"[A-Za-z_]\w*", statement)):
        for one in index.find(word):
            if one.kind in ("function", "method"):
                found.setdefault(one.file, set()).add(word)
    return any(len(words) >= 2 for words in found.values())


def adding(statement: str, held, subject):
    """Where a change that adds names goes: the file of a class it names,
    else of what it was placed on, else the project's one file of code."""
    from research.v698.asking import Subject
    index = held.index()
    classes = [one for word in re.findall(r"[A-Za-z_]\w*", statement)
               for one in index.find(word) if one.kind == "class"]
    if classes:
        return Subject("file", classes[0].file, classes[0].file)
    if subject is not None and subject.kind in ("function", "file"):
        return Subject("file", subject.file, subject.file)
    code = [path for path in held.files if path.endswith(".py")
            and not index.is_test(path)]
    if len(code) == 1:
        return Subject("file", code[0], code[0])
    return subject


def span(held, subject) -> tuple | None:
    """(first line, last line) of a function -- decorators included -- or
    of a file, whole."""
    text = held.files.get(subject.file or "", "")
    lines = text.splitlines()
    if subject.kind == "function":
        found = [one for one in held.find(subject.name)
                 if one["file"] == subject.file]
        if not found:
            return None
        start, end = found[0]["start"], found[0]["end"]
        while start > 1 and lines[start - 2].lstrip().startswith("@"):
            start -= 1
        return start, end
    if subject.kind == "file" and lines:
        return 1, len(lines)
    return None


def _content_words(statement: str) -> set | None:
    """The statement's words that carry meaning -- nouns, verbs, adjectives,
    names, as the parser reads them -- lowered; None where there is no
    parser."""
    from research.v691.hearing import nlp
    parser = nlp()
    if parser is None:
        return None
    return {one.text.lower() for one in parser(statement)
            if one.pos_ in ("NOUN", "PROPN", "VERB", "ADJ", "X")}


def placed(read, held, statement: str):
    """What a change is of: among what the statement names that the
    project has -- the encoder's subject phrases, and each word that is a
    function's or a file's name, looked up -- the one whose code holds the
    most of the other code names the statement says (`readings imports
    checker`: `readings` holds `checker`, not the other way round). A
    function inside a file named is what is changed, not the whole file;
    among equals, what the encoder marked."""
    from research.v698.asking import Subject
    if held is None or not held.files:
        return None
    marked = [phrase.split("(")[0].strip(" ?.,!'\"`")
              for phrase in read.spans.get("SUBJ", ())]
    said = [one for one in dict.fromkeys(
        re.findall(r"[A-Za-z_$][\w$./-]*[\w$]", statement))]
    # a plain word the encoder did not mark is a name only where it can be
    # one: `the` is a function here (`Open.the`), and a subsystem asked for
    # was placed on it -- the parser's part of speech says which words of
    # the statement carry meaning
    content = _content_words(statement)
    if content is not None:
        said = [one for one in said if one in marked or
                re.search(r"[_.$/()]|[a-z][A-Z]", one) or
                one.lower() in content]
    candidates = []
    for word in dict.fromkeys(marked + said):
        if not word:
            continue
        for path in held.files:
            if word == path or path.endswith("/" + word):
                candidates.append(Subject("file", path, path))
        for one in held.find(word.split(".")[-1]) if "/" not in word \
                else ():
            candidates.append(Subject("function", one["name"], one["file"]))
    if not candidates:
        return None
    files = {one.file for one in candidates if one.kind == "file"}
    inside = [one for one in candidates
              if one.kind == "function" and one.file in files]
    if inside:
        candidates = [one for one in candidates
                      if one.kind == "function" and (
                          one.file in files or not files)]
    def score(subject) -> tuple:
        where = span(held, subject)
        code = names_between(held.files[subject.file], subject.file,
                             *where) if where else {}
        own = subject.name.split(".")[-1]
        others = [one for one in said if one != own
                  and one.rsplit("/", 1)[-1] != subject.name.rsplit("/", 1)[-1]]
        # a field said is where what it is read off is (`spec.language`
        # where `spec` is: the `_passes` of experiment.py, not editing.py's
        # `_passes(bug, source)`)
        return (sum(one in code or ("." in one and one.split(".")[0] in code)
                    for one in others),
                own in marked or subject.name in marked)
    return max(candidates, key=score)


def names_in(text: str, path: str) -> dict:
    """The names the code itself uses -- not its comments, strings or
    docs -- each with the lines (1-based) it is used on: what a word of a
    statement is looked up in."""
    import io
    import keyword
    import tokenize
    out: dict = {}
    if path.endswith(".py"):
        try:
            for one in tokenize.generate_tokens(io.StringIO(text).readline):
                if one.type == tokenize.NAME and \
                        not keyword.iskeyword(one.string):
                    out.setdefault(one.string, []).append(one.start[0])
        except (tokenize.TokenError, SyntaxError, IndentationError):
            pass
        return out

    def blank(found) -> str:
        # what is not code, blanked: its lines kept, so lines still count
        return re.sub(r"[^\n]", " ", found.group(0))
    stripped = re.sub(r"//[^\n]*|/\*.*?\*/|(['\"`])(?:\\.|(?!\1).)*\1",
                      blank, text, flags=re.S)
    for at, line in enumerate(stripped.split("\n"), 1):
        for name in re.findall(r"[A-Za-z_$][\w$]*", line):
            if name not in TS_WORDS:
                out.setdefault(name, []).append(at)
    return out


def names_between(text: str, path: str, start: int, end: int) -> dict:
    """`names_in` of lines [start, end] of a file, read in the whole file:
    a part read alone can begin inside a docstring, and nothing after is
    read as code."""
    out = {}
    for name, lines in names_in(text, path).items():
        inside = [at for at in lines if start <= at <= end]
        if inside:
            out[name] = inside
    return out


#: TypeScript's own words: never what a statement names
TS_WORDS = {"const", "let", "var", "function", "return", "if", "else",
            "for", "while", "of", "in", "new", "class", "export", "import",
            "from", "this", "true", "false", "null", "undefined", "async",
            "await", "type", "interface", "extends", "implements"}


def window(lines: list, start: int, end: int, statement: str,
           path: str) -> tuple | None:
    """The `LONGEST_PART` lines of [start, end] holding the most of the
    code names the statement says (`math`, `PredicateTrie`), found by
    lookup in the code; None where it names none of them."""
    code = names_between("\n".join(lines), path, start, end)
    said = [one for one in dict.fromkeys(
        re.findall(r"[A-Za-z_$][\w$]*", statement)) if one in code]
    if not said:
        return None
    where = {name: code[name] for name in said}
    best, most = None, -1
    for first in range(start, max(start, end - LONGEST_PART + 1) + 1):
        last = min(end, first + LONGEST_PART - 1)
        covered = sum(any(first <= at <= last for at in found)
                      for found in where.values())
        if covered > most:
            best, most = (first, last), covered
    # around what it names, not from the top of the window
    hits = sorted(at for found in where.values() for at in found
                  if best[0] <= at <= best[1])
    if hits:
        first = max(start, hits[0] - AROUND)
        last = min(end, max(hits[-1] + AROUND, first + 1))
        if last - first + 1 <= LONGEST_PART:
            best = (first, last)
    return best


def _lines(text: str) -> list:
    return text.splitlines(keepends=True)


def _diagnosed(held, path: str, text: str) -> list:
    """What the compiler says is wrong in `path`, were it `text`."""
    from research.v696.checker import CheckerError, checker
    from research.v698 import project as Pj
    language = Pj.language_of(path)
    files = held.checked(language)
    files[Pj.ROOT + path] = text
    if language == "typescript":
        files[Pj.ROOT + Pj.AMBIENT_FILE] = Pj.AMBIENT
    try:
        found = checker(language).diagnose(files, False)
    except CheckerError as bad:
        # a check that did not run is not a check that passed
        raise Unchecked(str(bad)) from None
    return [one for one in found if one["file"] == Pj.ROOT + path]


def _flakes(path: str, text: str) -> list:
    """pyflakes' findings of a Python file -- names never used, used and
    never defined, defined twice -- by what they say, not where: what the
    compiler, told to pass over imports it cannot find, does not say."""
    if not path.endswith(".py"):
        return []
    import io
    try:
        from pyflakes.api import check
        from pyflakes.reporter import Reporter
    except ImportError:
        return []
    out, err = io.StringIO(), io.StringIO()
    check(text, path, Reporter(out, err))
    said = []
    for line in (out.getvalue() + err.getvalue()).splitlines():
        message = line.split(":", 3)[-1].strip() if line.count(":") >= 3 \
            else line.strip()
        said.append({"code": "pyflakes", "message": re.sub(
            r"\bline \d+", "line", message)})
    return said


def _same_code(made: str, part: str) -> bool:
    """Whether two versions of code are the same code: the same lines, but
    for blank ones, comments alone on a line, and spaces at their ends."""
    def code(text: str) -> list:
        return [line.rstrip() for line in text.splitlines()
                if line.strip() and not line.strip().startswith(("#", "//"))]
    return code(made) == code(part)


def _lines_changed(part: str, made: str) -> tuple:
    """(lines taken out, lines put in) from `part` to `made`, stripped."""
    out, put = [], []
    for line in difflib.ndiff(part.splitlines(), made.splitlines()):
        if line.startswith("- "):
            out.append(line[2:].strip())
        elif line.startswith("+ "):
            put.append(line[2:].strip())
    return out, put


#: of a change keeping what the code gives, how many calls the code as it
#: is must give a value for (not an error), for it to be the oracle
KEPT_CALLS = 3


def _kept_what_it_gives(rows: list, original=()) -> list:
    """A change keeping what the code gives (the reader's `same`: `make it
    loop once`): the changes giving exactly what the code as it is gives on
    every call -- the code the oracle, no test's guessed output asked --
    where it gives values for enough calls to be one."""
    original = tuple(original or ())
    errors = re.compile(r"^[A-Z]\w*(Error|Exception)$|^ran past$")
    if sum(1 for one in original if not errors.match(str(one))) < \
            KEPT_CALLS:
        return []
    return [row for row in rows if row.get("own ok") and
            tuple(row.get("gives") or ()) == original]


def _agreed_by_tests(rows: list, tests: int, original=(),
                     stated: int = 0, keeps: bool = False) -> list:
    """The changes the tests of what was asked choose, as CodeT chooses
    (Chen et al. 2022), and what they do, as AlphaCode clusters: the
    changes that keep the project's own tests, grouped by the tests of
    what was asked each passes and what each gives for the calls the tests
    and the code model's chosen inputs make (`.strip(".")` and
    `.replace(".", "")` pass the same weak tests and differ on a dot
    inside). A group giving what the code gives now, passing no test,
    changes nothing asked. A group weighed by how many answers wrote its
    changes, times one and the tests they pass -- a wrong test is passed
    by none. The heaviest group, where answers written apart agree on it
    (two or more), or one passes every test there is (two or more); else
    none: a change one answer wrote is a guess."""
    if keeps:
        return _kept_what_it_gives(rows, original)
    groups: dict = {}
    original = tuple(original or ())
    for row in rows:
        # what breaks the project's own tests, or lacks what the request
        # writes out (`requested.interface`), is no change asked
        if not row.get("own ok"):
            continue
        passes = tuple(row.get("passes") or ())
        gives = tuple(row.get("gives") or ())
        if not passes and (not gives or gives == original):
            continue
        groups.setdefault((passes, gives), []).append(row)
    if not groups:
        return []

    def answers(key) -> int:
        return sum(len(one.get("by") or {1}) for one in groups[key])
    best = max(groups, key=lambda one: (answers(one) * (1 + len(one[0])),
                                        len(one[0])))
    if tests and not best[0]:
        # tests of what was asked there, and the heaviest group passing none
        # of them: answers agreeing on doing something else. (Chosen by what
        # they do alone, past the tests -- 2026-10-22 -- answers of one
        # model agreeing were no evidence: 12 of 12 agreeing, wrong; on
        # CanItEdit 15 right and 26 wrong against 20 and 22.)
        return []
    if answers(best) >= 2 or (len(best[0]) == tests and tests >= 2):
        return groups[best]
    if stated and set(range(stated)) <= set(best[0]):
        # the examples the request says (`requested.stated`: the person's,
        # not the code model's guesses) held: chosen though one answer
        # wrote it
        return groups[best]
    return []


def _extends(part: str, core: str, more: str) -> bool:
    """Whether `more` makes the change `core` makes, and more besides:
    the same lines taken out, and each line `core` puts in put in by
    `more` too -- as it is, or with more after it on the line."""
    if core == more:
        return False
    core_out, core_in = _lines_changed(part, core)
    more_out, more_in = _lines_changed(part, more)
    if not core_in or sorted(core_out) != sorted(more_out):
        return False
    return all(any(one == line or one.startswith(line.rstrip(",")) and
                   len(one) > len(line.rstrip(","))
                   for one in more_in) for line in core_in)


def _with_imports(path: str, text: str, offered: dict | None) -> str:
    """The file, importing what a change uses and the file does not
    define, where the project has it and the change was told so (a plan's
    step: `shell.ASK -- research/v702/shell.py`): looked up, not written
    by the editor -- it is shown one function, not the file's imports."""
    if not offered or not path.endswith(".py"):
        return text
    missing = sorted({found.group(1) for one in _flakes(path, text)
                      for found in [re.match(r"undefined name '(\w+)'",
                                             one["message"])] if found})
    wanted = [offered[name] for name in missing if name in offered]
    if not wanted:
        return text
    import ast
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return text
    # after the file's last import at its top, or its docstring
    at = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            at = node.end_lineno
        elif at == 0 and isinstance(node, ast.Expr) and isinstance(
                getattr(node, "value", None), ast.Constant):
            at = node.end_lineno
        elif at:
            break
    lines = text.splitlines(keepends=True)
    ending = "\r\n" if "\r\n" in text else "\n"
    return "".join(lines[:at]) + "".join(one + ending for one in wanted) + \
        "".join(lines[at:])


def _mended_json(path: str, before: str, after: str) -> str | None:
    """A JSON file a change left unreadable only by its separators, mended:
    a comma put in where one is expected, one taken out before what closes
    -- the record the editor added was right, the line above it lacked its
    comma (all 19 answers to `add a member with id 5 ...` were refused
    for it). None where it read before and reads now, or no comma mends
    it."""
    import json
    if not path.lower().endswith(".json"):
        return None
    try:
        json.loads(before)
    except ValueError:
        return None
    text = after
    for _ in range(4):
        try:
            json.loads(text)
            return text if text != after else None
        except json.JSONDecodeError as bad:
            if bad.msg == "Expecting ',' delimiter":
                at = len(text[:bad.pos].rstrip())
                text = text[:at] + "," + text[at:]
            elif bad.msg in ("Expecting value",
                             "Expecting property name enclosed in double "
                             "quotes", "Illegal trailing comma before end "
                             "of array", "Illegal trailing comma before "
                             "end of object"):
                # at the comma, or after it (as the version says)
                at = len(text[:bad.pos].rstrip())
                if text[bad.pos:bad.pos + 1] == ",":
                    text = text[:bad.pos] + text[bad.pos + 1:]
                elif at and text[at - 1] == ",":
                    text = text[:at - 1] + text[at:]
                else:
                    return None
            else:
                return None
    return None


def data_wrong(path: str, before: str, after: str) -> str | None:
    """Why a change of a data file is not one, or None: it must still read
    as what it is, and put in no place whose name the file already has
    elsewhere (`server.server.port` beside `server.port`: what was meant
    is said twice)."""
    from research.v701 import datamodel
    old, new = datamodel.read(path, before), datamodel.read(path, after)
    if new.error and not old.error:
        return f"the file no longer reads as {new.format}: {new.error}"
    added = [one for one in new.places if one not in old.places]
    for place in added:
        own = place.rsplit(".", 1)[-1]
        twin = next((one for one in old.places if one != place
                     and one.rsplit(".", 1)[-1] == own
                     and not place.startswith(f"{one}.")
                     and "[]" not in place), None)
        if twin is not None:
            return f"it puts in {place}, which the file has as {twin}"
    # its records stay records of what they were: a field each had, each
    # has (`31,stark-db,Ottawa` under `name,age,team,city` has no city),
    # and what told them apart still does (a second `sam`, not sam changed)
    for collection, found in old.collections.items():
        now = new.collections.get(collection)
        if now is None or not now["count"]:
            continue
        def place_of(field: str) -> str:
            return f"{collection}.{field}" if collection else field
        group = collection or "the rows"
        for field in found["fields"]:
            had, has = old.places.get(place_of(field)), \
                new.places.get(place_of(field))
            if had is None or has is None or had.count != found["count"]:
                continue
            # a cell left empty is a field without a value, and a column
            # whose kind changes is one whose cells moved (a CSV's row a
            # cell short: its ages read as text)
            if has.count < now["count"] or ("null" in has.types and
                                             "null" not in had.types):
                return (f"it leaves a record of {group} without {field}, "
                        f"which every one had")
            if len(had.types) == 1 and set(has.types) != set(had.types):
                return (f"it makes {field} of {group} "
                        f"{'/'.join(sorted(has.types))}, where every one "
                        f"was {next(iter(had.types))}")
        # what tells records apart: their key -- or, too few to know one,
        # their first field where it differs in each (`name`)
        keys = old.keys(collection)
        first = old.places.get(place_of(found["fields"][0])) \
            if found["fields"] else None
        if not keys and first is not None and found["count"] >= 2 and \
                first.count == found["count"] == len(first.distinct):
            keys = [found["fields"][0]]
        for key in keys:
            has = new.places.get(place_of(key))
            if has is not None and len(has.distinct) < has.count:
                return (f"it repeats a {key} of {group}, which told each "
                        f"apart")
    return None


def _unfound_imports(held, before: str, after: str, path: str) -> list:
    """Modules of the project's own packages a change imports that the
    project has no file for (`research.v696.checker.field`)."""
    if not path.endswith(".py"):
        return []
    import ast

    def imported(text: str) -> set:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return set()
        return {node.module for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
                and not node.level} | {
            alias.name for node in ast.walk(tree)
            if isinstance(node, ast.Import) for alias in node.names}
    tops = {one.split("/")[0] for one in held.files if "/" in one}
    out = []
    for module in imported(after) - imported(before):
        parts = module.split(".")
        if parts[0] not in tops:
            continue
        stem = "/".join(parts)
        if f"{stem}.py" not in held.files and \
                f"{stem}/__init__.py" not in held.files and not any(
                    one.startswith(stem + "/") for one in held.files):
            out.append(module)
    return out


class Unchecked(Exception):
    """The compiler could not check the project."""


def _more_wrong(before: list, after: list) -> list:
    """What the compiler says after that it did not say before -- by what
    it says, not where: lines move when lines are taken out."""
    from collections import Counter
    said = Counter((one.get("code"), one.get("message")) for one in before)
    out = []
    for one in after:
        key = (one.get("code"), one.get("message"))
        if said[key] > 0:
            said[key] -= 1
        else:
            out.append(one)
    return out


def _reads(path: str, text: str) -> bool:
    if path.endswith(".py"):
        import ast
        try:
            ast.parse(text)
        except (SyntaxError, ValueError):
            return False
    return True


def said_names(statement: str, text: str, path: str, start: int, end: int,
               subject) -> set:
    """The code names the statement says that the part's code (lines
    [start, end] of the file) uses -- looked up, not read -- other than
    what is changed itself (`forward assigns torch`: `torch`)."""
    code = names_between(text, path, start, end)
    own = {subject.name, subject.name.split(".")[-1],
           (subject.file or "").rsplit("/", 1)[-1]}
    return {one for one in re.findall(r"[A-Za-z_$][\w$]*", statement)
            if one in code and one not in own}


def _outside(block: tuple, start: int, part: str, made: str) -> bool:
    """Whether a change of a data file takes out or puts in lines outside
    the block the statement is of (`placing.block`, 0-based): a record
    added after its last line is in it."""
    matcher = difflib.SequenceMatcher(None, _lines(part), _lines(made),
                                      autojunk=False)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        first, end = start - 1 + i1, start - 1 + i2
        if first < block[0] or end > block[1] + 1 or \
                (tag != "insert" and first > block[1]):
            return True
    return False


def _changed_names(part: str, made: str) -> set:
    """The names on the lines a change takes out or puts in."""
    out = set()
    matcher = difflib.SequenceMatcher(None, _lines(part), _lines(made),
                                      autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        for line in _lines(part)[i1:i2] + _lines(made)[j1:j2]:
            out.update(re.findall(r"[A-Za-z_$][\w$]*", line))
    return out


def _shifted(part: str, made: str) -> str | None:
    """A call a change gives a value before what it was given already, by
    position: every argument after it now goes to the next parameter (six
    of seven wrote `Spec(name, params, spec.language, body_type, ...)`
    for `pass spec.language on to Spec`: `body_type` given as examples).
    The call so changed, or None. Python only: what does not parse is
    not looked at."""
    import ast
    import textwrap
    try:
        before = ast.parse(textwrap.dedent(part))
        after = ast.parse(textwrap.dedent(made))
    except SyntaxError:
        return None

    def calls(tree) -> dict:
        out: dict = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                out.setdefault(ast.dump(node.func), []).append(
                    (node.func, [ast.dump(one) for one in node.args]))
        return out
    now = calls(after)
    for func, given in calls(before).items():
        for called, old in given:
            for _, new in now.get(func, []) if old else ():
                if len(new) <= len(old) or new[:len(old)] == old:
                    continue
                # the old arguments all there, in order, with more among
                # them: some given by position to another parameter now
                left = iter(new)
                if all(any(one == other for other in left) for one in old):
                    return ast.unparse(called)
    return None


def _put_before(part: str, made: str) -> str | None:
    """A function a change gives a parameter before ones it had, by
    position: what each caller gives by position now goes to the one
    before (`gained(original, sources, said="", commit="", held=None)`:
    a call giving `held` third gave it as `commit`). The function, or
    None. Python only."""
    import ast
    import textwrap
    try:
        before = ast.parse(textwrap.dedent(part))
        after = ast.parse(textwrap.dedent(made))
    except SyntaxError:
        return None
    was = {node.name: [one.arg for one in node.args.posonlyargs +
                       node.args.args] for node in ast.walk(before)
           if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for node in ast.walk(after):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                or node.name not in was:
            continue
        old = was[node.name]
        new = [one.arg for one in node.args.posonlyargs + node.args.args]
        if len(new) > len(old) and new[:len(old)] != old and \
                [one for one in new if one in old] == old:
            return node.name
    return None


def _dropped(part: str, made: str, text: str) -> str | None:
    """A function of the module that gives a value, called by a statement
    a change puts in and what it gives dropped (`_repair_python(bug)`,
    the `Fix` it returns thrown away and the repair carried on, for `hand
    python bugs to _repair_python`; two of seven wrote it). Its name, or
    None. Python only."""
    import ast
    import textwrap
    try:
        module = ast.parse(text)
        before = ast.parse(textwrap.dedent(part))
        after = ast.parse(textwrap.dedent(made))
    except SyntaxError:
        return None
    giving = {node.name for node in module.body
              if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
              and not (isinstance(node.returns, ast.Constant)
                       and node.returns.value is None)
              and (node.returns is not None or any(
                  isinstance(one, ast.Return) and one.value is not None
                  for one in ast.walk(node)))}

    def called(tree) -> list:
        return [ast.dump(node) for node in ast.walk(tree)
                if isinstance(node, ast.Expr) and isinstance(node.value,
                                                             ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in giving]
    was = called(before)
    for one in called(after):
        if one in was:
            was.remove(one)
            continue
        return next(node.value.func.id for node in ast.walk(after)
                    if isinstance(node, ast.Expr) and ast.dump(node) == one)
    return None


def diff(before: str, after: str, path: str) -> str:
    return "".join(difflib.unified_diff(
        _lines(before), _lines(after), f"a/{path}", f"b/{path}", n=2))


#: passes made of one statement, while it names what no pass has changed
#: (`delete both`: the editor makes one change at a time)
PASSES = 4


#: the functions a conversation was told already take what it asked them
#: to: asked again, the change is made (`_taken_already`)
TOLD: dict = {}


def _taken_already(statement: str, held, subject, key, spans) -> dict | None:
    """What a person says where a function is asked to take what it takes
    already (v705 `teach_already`): `gained` asked to take the commit
    message takes it as `said` -- "what the change is for, said before
    each window alike" -- and the change path added `commit` beside it.
    Asked, not refused: nothing is written, and asked again in the
    conversation, the change is made. None where it is not so."""
    if not (spans or {}).get("NEW") or subject is None or \
            subject.name in TOLD.get(key, set()):
        return None
    where = part_of(held, subject, statement)
    if where is None or not where[0].endswith(".py"):
        return None
    path, start, end = where
    part = "".join(_lines(held.files[path])[start - 1:end])
    from research.v705 import teach_already
    found = teach_already.already(part, statement)
    if found is None:
        return None
    import ast
    import textwrap
    name, chance = found
    try:
        function = ast.parse(textwrap.dedent(part)).body[0]
        what = teach_already.described(function, textwrap.dedent(part)).get(
            name, ("", []))[0]
    except (SyntaxError, IndexError, AttributeError):
        what = ""
    TOLD.setdefault(key, set()).add(subject.name)
    return {"status": "already", "subject": subject.name, "file": path,
            "lines": [start, end], "taken as": name,
            "chance": round(chance, 3),
            "said": (f"{subject.name} already takes that, I think: `{name}`"
                     + (f" -- {what.rstrip('.')}." if what else ".") +
                     f" I changed nothing. If you want a new parameter "
                     f"beside it, ask again and I will add it.")}


def change(statement: str, held, subject, key=None,
           context: dict | None = None, spans: dict | None = None) -> dict:
    """The change asked, made -- again, on what it made, while the
    statement names code no pass has changed yet, each pass changing some
    of what is left; the passes kept as one change (`undo` puts all back).
    `context`: what a plan's step is given beside the statement (v703
    `teach_parts`) -- `said` to the editor, and the `imports` the project
    offers for what that says."""
    found = _taken_already(statement, held, subject, key, spans)
    if found is not None:
        return found
    first = _change_once(statement, held, subject, key, context=context,
                         spans=spans)
    if first["status"] != "changed" or first.get("by kind"):
        # made as the kind the reader read: its roles say what the change
        # is, all of it -- the words the statement shares with the code are
        # not more of it to change (`a`, in `take a message`)
        return first
    named = set(first.get("named", ()))
    done = set(first.get("touched", ()))
    passes = 1
    from research.v698.project import is_data
    # a data file's names are its keys, every one of them a word the
    # statement may say: one change, as asked (a second pass put in
    # `server.server.port`)
    while passes < PASSES and named - done and not is_data(first["file"]):
        more = _change_once(statement, held, subject, key,
                            need=named - done, context=context, spans=spans)
        if more["status"] != "changed":
            break
        done |= set(more.get("touched", ()))
        passes += 1
    if passes > 1:
        made = MADE[key]
        path, before, after = made[-passes][0], made[-passes][1],             made[-1][2]
        del made[-passes:]
        made.append((path, before, after))
        first["diff"] = diff(before, after, path)
        first["passes"] = passes
        first["said"] += f" (in {passes} passes, one change at a time)"
    first["touched"] = sorted(done)
    # what it names that the code defines there and no pass changed: said,
    # not passed over (asked to delete element and outputs, it deleted
    # element and said it had changed deduced)
    start, end = first.get("lines") or (1, 10 ** 9)
    left = sorted((named - done) & defined_in(
        MADE[key][-1][1], first["file"], start, end))
    if left:
        first["left"] = left
        first["status"] = "partly"
        first["said"] += (f" I left {', '.join(left)} as "
                          f"{'it was' if len(left) == 1 else 'they were'}:"
                          f" no change of {'it' if len(left) == 1 else 'them'}"
                          f" passed the checks.")
    return first


def first_verified(held, path: str, candidates: list, verify, key=None,
                   name: str = "",
                   how: str = "where it gave up, it now asks",
                   more=None) -> dict:
    """Of changed texts of a file made by construction (v706
    `inside.compositions`: a call of what is in reach put in where a
    function gives up), the first that pyflakes finds nothing more wrong
    in and whose case, run again, holds (`verify`) -- written. {status,
    said, diff?}. `more`: what makes further candidates, asked only where
    none of these holds (the code model: loaded and run for it)."""
    text = held.files[path]
    flakes = _flakes(path, text)
    tried = []

    def every():
        yield from candidates
        if more is not None:
            try:
                yield from more()
            except Exception as bad:                # noqa: BLE001
                tried.append(f"the code model: {type(bad).__name__}: {bad}")
    for changed, call in every():
        more = _more_wrong(flakes, _flakes(path, changed))
        why = (f"pyflakes finds {more[0]['message']}" if more else
               verify(changed))
        if why:
            tried.append(f"{call}: {why}")
            continue
        written = held.write(path, changed)
        MADE.setdefault(key, []).append((path, text, changed))
        changed_ = diff(text, changed, path)
        return {"status": "changed", "diff": changed_, "call": call,
                "tried": tried, "said": (
                    f"Changed {name} in {path}: {how} {call} -- of "
                    f"{len(tried) + 1} changes made by construction, the "
                    f"first whose case holds when run again"
                    + (", in the file." if written else
                       " (in the project I hold).") + "\n" + changed_)}
    return {"status": "unchanged", "tried": tried, "said": (
        f"Nor did any of {len(tried)} calls of what gives what it should "
        f"answer the case ({'; '.join(tried[:2])})." if tried else "")}


def defined_in(text: str, path: str, start: int = 1,
               end: int = 10 ** 9) -> set:
    """The names a Python file defines on lines [start, end] -- assigns,
    imports, takes as parameters, defines as functions or classes: what a
    statement names that can be taken out or changed."""
    if not path.endswith(".py"):
        return set()
    import ast
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        if not start <= getattr(node, "lineno", start) <= end:
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out.update((one.asname or one.name).split(".")[0]
                       for one in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.arg):
            out.add(node.arg)
    return out


#: a change made by the kind the request names is written where the judge
#: reads it as what was said at least so, and above the next so made by
#: at least so much
KIND_JUDGED, KIND_MARGIN = 0.5, 0.1
#: the roles the reader marks of what a change names (v705
#: `teach_change_talk.ROLES`)
CHANGE_ROLES = ("NEW", "DEFAULT", "VALUE", "CALLEE", "CASE", "HELPER")


def _change_once(statement: str, held, subject, key=None,
                 need: set | None = None,
                 context: dict | None = None,
                 spans: dict | None = None) -> dict:
    """One change asked, made: what was tried, what was checked, what was
    written. `need`: names it must change some of (what earlier passes
    left)."""
    started = time.time()
    where = part_of(held, subject, statement)
    if where is None:
        return {"status": "unplaced", "subject": subject.name,
                "said": (f"{subject.name} is too long to change whole, "
                         f"and what you said names nothing in its code; "
                         f"name the function to change, or what in it.")}
    path, start, end = where
    text = held.files[path]
    lines = _lines(text)
    part = "".join(lines[start - 1:end])
    asked = T.prompt((context or {}).get("said") or statement, path, part)
    from research.v698.project import is_data
    data = is_data(path)
    try:
        # a data file (v701) has no compiler: it is checked as data
        before = [] if data else _diagnosed(held, path, text)
    except Unchecked as bad:
        return {"status": "unchecked", "subject": subject.name,
                "said": (f"I did not change {subject.name}: the compiler "
                         f"cannot check the project ({bad}), and I do not "
                         f"write what I cannot check.")}
    # what a data file's change must be in: the block the statement says
    # (v701 `placing`) -- what its words name is told by where, not by the
    # names on the lines changed (`go` added under `languages`)
    block = None
    if data:
        from research.v701 import placing
        block = placing.block(text.splitlines(), statement,
                              placing.tabular(path))
    named = set() if data else said_names(statement, text, path, start,
                                          end, subject)
    flakes_before = _flakes(path, text)
    tried, seen, passed, answers = [], {}, [], []
    # what the reader marks a change to name, as words and as the name
    # they would be in code (`commit message`, `commit_message`)
    from research.v705.shaped import _identifier
    marking = set() if data else {
        word for role in CHANGE_ROLES for one in (spans or {}).get(role, [])
        for word in re.findall(r"\w+", one) + [_identifier(one)]}
    def check(row) -> None:
        """An answer's change put to every check: kept in `passed`, or
        refused with why -- the editors' and those made by their kind
        (`shaped.asked`) alike."""
        # the case a debugging step is fixing, written into the code: a
        # fit to the case, not a fix (`return Mention("speaker",
        # text="what is my dog called?")` for every phrase _whose did
        # not know -- three agreed, it compiled)
        case = (context or {}).get("case")
        if case and case.lower() in row["made"].lower() and \
                case.lower() not in part.lower():
            row["refused"] = "it writes the case it is fixing into the code"
            return
        # what the request names of the project by its words (a plan's
        # step: `the shell asks` is shell.ASK) -- a change using none of
        # it is not the change asked (`"asked": parsed.path ==
        # "/api/shell"` compiled, two wrote it, the judge let it by)
        wanted = (context or {}).get("names")
        if wanted and not any(name in line for line in _lines_changed(
                part, row["made"])[1] for name in wanted):
            row["refused"] = ("it uses nothing the request names "
                              f"({', '.join(wanted)})")
            return
        # what the reader marks the change to name (a parameter to take, a
        # value passed on, the call, the case, the helper): a change putting
        # in none of it is not the change asked (`or isinstance(bug.source,
        # list)`, two of seven, for `hand python bugs to _repair_python`)
        # (a word of a name as well: `commit` of `commit_message`)
        if marking and not any(word in re.split(r"[^A-Za-z0-9]+", line) or
                               word in re.findall(r"\w+", line)
                               for line in _lines_changed(part, row["made"])[1]
                               for word in marking):
            row["refused"] = ("it puts in nothing the request marks "
                              f"({', '.join(sorted(marking))})")
            return
        if need and not set(row["touched"]) & need:
            row["refused"] = ("it changes none of what is left "
                              f"({', '.join(sorted(need))})")
            return
        if named and not row["touched"]:
            # a change of nothing the statement names is not the
            # change it asks for, however well it compiles
            row["refused"] = ("it changes nothing the statement names "
                              f"({', '.join(sorted(named))})")
            return
        if data:
            if block is not None and _outside(block, start, part,
                                              row["made"]):
                row["refused"] = ("it changes lines outside what the "
                                  f"statement is of (lines {block[0] + 1}"
                                  f"-{block[1] + 1})")
                return
            mended = _mended_json(path, text, row["text"])
            if mended is not None:
                row["text"] = mended
            wrong = data_wrong(path, text, row["text"])
            if wrong:
                row["refused"] = wrong
                return
            passed.append(row)
            return
        if not _reads(path, row["text"]):
            row["refused"] = "the file no longer reads"
            return
        try:
            found = _diagnosed(held, path, row["text"])
        except Unchecked as bad:
            row["refused"] = f"the compiler could not check it: {bad}"
            return
        row["diagnostics"] = len(found)
        more = _more_wrong(before, found)
        if more:
            row["refused"] = (f"the compiler finds {len(more)} more "
                              f"wrong: {more[0].get('message', '')}")
            return
        more = _more_wrong(flakes_before, _flakes(path, row["text"]))
        if more:
            row["refused"] = (f"pyflakes finds {len(more)} more wrong: "
                              f"{more[0]['message']}")
            return
        unfound = _unfound_imports(held, text, row["text"], path)
        if unfound:
            row["refused"] = (f"it imports what the project does not "
                              f"have: {', '.join(unfound)}")
            return
        shifted = _shifted(part, row["made"]) if path.endswith(".py") \
            else None
        if shifted:
            row["refused"] = (f"it gives {shifted} a value before what it "
                              f"was given, each after it now another "
                              f"parameter's")
            return
        before_ = _put_before(part, row["made"]) if path.endswith(".py") \
            else None
        if before_:
            row["refused"] = (f"it gives {before_} a parameter before ones "
                              f"it had, what callers give by position now "
                              f"another's")
            return
        dropped = _dropped(part, row["made"], row["text"]) \
            if path.endswith(".py") else None
        if dropped:
            row["refused"] = (f"it calls {dropped} and drops what it "
                              f"gives")
            return
        passed.append(row)
    rounds = VERIFIED_ROUNDS if (context or {}).get("verify") else ROUNDS
    ahead: list = []
    for round_ in range(rounds):
        # each request sampled by a seed of its own and the round's, as the
        # writers are (`sketcher.proposals`): asked again, the same answers
        # a fix from a run seen inside: the editor taught to fix from what
        # a run shows (v705 `editor-debug`), told so
        from research.v697.coding import Tools
        writer = Tools.get().writer(context["editor"]) \
            if (context or {}).get("editor") else \
            _editor(step=context is not None)
        if not ahead:
            # where each change is run again, rounds are mostly all asked
            # for (none answers its case): written CHUNKED at once, the card
            # kept busy -- six at a time it was idle more than half the
            # while, waiting on each small step
            many = min(CHUNKED, rounds - round_) \
                if (context or {}).get("verify") else 1
            writer.torch.manual_seed(SEED + zlib.crc32(asked.encode()) +
                                     round_)
            got = writer.write([asked], samples=SAMPLES * many, longest=400,
                               greedy=round_ == 0,
                               saying=(context or {}).get("saying"))[0]
            first = SAMPLES + (round_ == 0)
            ahead = [got[:first]] + [
                got[first + SAMPLES * at:first + SAMPLES * (at + 1)]
                for at in range(many - 1)]
        written_now = ahead.pop(0)
        answers += written_now
        fresh = []
        for number, answer in enumerate(written_now):
            greedy = round_ == 0 and number == 0
            if answer.strip() == T.NOTHING:
                # its part of a larger change is nothing (a plan's step)
                tried.append({"answer": answer, "greedy": greedy,
                              "round": round_, "nothing": True})
                continue
            made_all = T.made_each(part, answer)
            if not made_all:
                tried.append({"answer": answer, "greedy": greedy,
                              "round": round_, "refused":
                              "its old lines are not in the code once"})
                continue
            # all its changes, and each alone: each a change of its own
            for made in made_all:
                row = {"answer": answer, "greedy": greedy, "round": round_}
                tried.append(row)
                if _same_code(made, part):
                    # a blank line or a comment is no change of the code
                    # (`change reader.build so that ... is read as
                    # ask_name` added one blank line, five agreed on it)
                    row["refused"] = "it changes nothing"
                    continue
                if made in seen:
                    # agreeing are the answers that wrote it, each once (an
                    # answer is all its changes and each alone: counted per
                    # change, "9 of 7 changes written agree")
                    seen[made]["by"].add((round_, number))
                    seen[made]["agree"] = len(seen[made]["by"])
                    row["same as"] = seen[made]["index"]
                    continue
                after = _with_imports(path, "".join(lines[:start - 1]) +
                                      made + "".join(lines[end:]),
                                      (context or {}).get("imports"))
                row.update({"index": len(tried) - 1, "agree": 1,
                            "by": {(round_, number)},
                            "text": after, "made": made, "touched": sorted(
                                named & _changed_names(part, made))})
                seen[made] = row
                fresh.append(row)
        for row in fresh:
            check(row)
        # asked again only while nothing passed that answers agree on: the
        # words a statement shares with the code are not all code it names
        # (`it`, `never` are names in some code), so all of them touched is
        # no stop -- and a change one answer wrote is not written, so more
        # answers are asked for before saying so
        nothing = sum(1 for one in tried if one.get("nothing"))
        score = (context or {}).get("score")
        if score is not None:
            # each change run on the tests of what was asked (v698
            # `requested`), the choice made over all of them as CodeT
            # makes it: asked again until changes written apart agree on
            # the tests they pass
            for row in passed:
                if "passes" not in row:
                    row["own ok"], row["passes"], row["gives"] = score(
                        row["text"])
            chosen = _agreed_by_tests(passed, context.get("tests", 1),
                                      context.get("original"),
                                      context.get("stated", 0),
                                      context.get("keeps", False))
            if len(chosen) >= 2 or round_ + 1 >= context.get(
                    "rounds", VERIFIED_ROUNDS):
                break
            continue
        verify = (context or {}).get("verify")
        if verify is not None:
            # each change tried with its case said again as it is written
            # (`inside.verify`): asked again until one answers it -- what
            # is written is what the run shows, not what answers agree on
            for row in passed:
                if "verified" in row:
                    continue
                why = verify(row["text"])
                row["verified"] = not why
                if why:
                    row["refused"] = f"run again, {why}"
            if any(row["verified"] for row in passed) or \
                    round_ + 1 >= (context or {}).get("rounds",
                                                      VERIFIED_ROUNDS):
                break
            continue
        agreed_ = any(len(row["by"].union(*(
            other["by"] for other in seen.values() if other is not row
            and _extends(part, row["made"], other["made"])))) >= LEAST_AGREED
            for row in passed)
        if agreed_ or nothing >= LEAST_AGREED and nothing * 2 > len(answers):
            break
    # the change of the kind the request's roles name (the reader's spans,
    # `shaped.asked`): a parameter it is to take, a value passed on, a case
    # handed to a helper -- what the editors could not write ("gained
    # should take a message": twenty answers, none of the `def` line)
    kinds = []
    if data:
        # a value of the data set as said, made by lookup (v701
        # `placing.settings`: the record a value names, its field named, a
        # word of the field's kind) -- the editor, shown Dune's row among
        # others, set the last row's pages (1 of 19 answers right)
        from research.v701 import placing
        head, tail = "".join(lines[:start - 1]), "".join(lines[end:])
        for what, whole in placing.settings(path, text, statement):
            if whole.startswith(head) and whole.endswith(tail) and \
                    len(whole) >= len(head) + len(tail):
                kinds.append((what, whole[len(head):len(whole) - len(tail)]))
    elif spans:
        from research.v705 import shaped
        kinds = shaped.asked(spans, part, path, text, said=statement,
                             project=held.files)
    if kinds:
        for what, made in kinds:
            row = {"answer": f"(by its kind) {what}", "greedy": False,
                   "round": -1, "kind": what}
            tried.append(row)
            if made in seen:
                # an editor wrote it too: answers written apart agreeing
                seen[made]["by"].add("kind")
                seen[made]["agree"] = len(seen[made]["by"])
                seen[made]["kind"] = what
                row["same as"] = seen[made]["index"]
                continue
            after = _with_imports(path, "".join(lines[:start - 1]) + made +
                                  "".join(lines[end:]),
                                  (context or {}).get("imports"))
            # what it touches of what was said: the names the statement
            # shares with the code, and what the reader marked it to take,
            # pass or call -- in the lines it changes (`message`, taken; the
            # words the statement shares with the code are `gained` and `a`,
            # on lines a parameter put on the end does not change)
            marked = {word for role, found in (spans or {}).items()
                      if role != "SUBJ" for one in found
                      for word in re.findall(r"\w+", one)}
            changed_ = _changed_names(part, made)
            # (a word marked found as a word of a name changed as well:
            # `commit`, `message` of `commit_message`)
            parts = {piece for one in changed_ for piece in one.split("_")
                     if piece}
            row.update({"index": len(tried) - 1, "agree": 1, "by": {"kind"},
                        "text": after,
                        "made": made, "touched": sorted(
                            (named & changed_) | (marked & changed_) |
                            (marked & parts))})
            seen[made] = row
            check(row)
            verify_ = (context or {}).get("verify")
            if verify_ is not None and row in passed:
                # made after the rounds were verified: verified as they were
                why = verify_(row["text"])
                row["verified"] = not why
                if why:
                    row["refused"] = f"run again, {why}"
            score_ = (context or {}).get("score")
            if score_ is not None and row in passed:
                row["own ok"], row["passes"], row["gives"] = score_(
                    row["text"])
    # the code model's changes (v698 `requested.rewritten`): the function
    # written whole again -- where the editors write a line or two of it
    head_, tail_ = "".join(lines[:start - 1]), "".join(lines[end:])
    for number, whole in enumerate((context or {}).get("more") or ()):
        if not (whole.startswith(head_) and whole.endswith(tail_) and
                len(whole) >= len(head_) + len(tail_)):
            continue
        made = whole[len(head_):len(whole) - len(tail_)]
        row = {"answer": f"(the code model) answer {number + 1}",
               "greedy": False, "round": -2}
        tried.append(row)
        if made in seen:
            seen[made]["by"].add(("coder", number))
            seen[made]["agree"] = len(seen[made]["by"])
            row["same as"] = seen[made]["index"]
            continue
        row.update({"index": len(tried) - 1, "agree": 1,
                    "by": {("coder", number)},
                    "text": _with_imports(path, whole,
                                          (context or {}).get("imports")),
                    "made": made,
                    "touched": sorted(named & _changed_names(part, made))})
        seen[made] = row
        check(row)
        if (context or {}).get("score") is not None and row in passed:
            row["own ok"], row["passes"], row["gives"] = context["score"](
                row["text"])
    if (context or {}).get("score") is not None:
        # the changes the tests of what was asked choose, over all of them
        chosen = _agreed_by_tests(passed, context.get("tests", 1),
                                  context.get("original"),
                                  context.get("stated", 0),
                                  context.get("keeps", False))
        tests = context.get("tests", 1)
        for row in passed:
            row["verified"] = row in chosen
            if row not in chosen:
                row["refused"] = (
                    "run again, it breaks the project's tests"
                    if not row.get("own ok") else
                    f"run again, {len(row.get('passes', ()))} of {tests} "
                    f"tests of what was asked hold, and no changes written "
                    f"apart agree on more")
    out = {"subject": subject.name, "file": path, "lines": [start, end],
           "named": sorted(named),
           "tried": [{key_: value for key_, value in one.items()
                      if key_ not in ("text", "made", "by")} for one in tried],
           "diagnostics before": len(before)}
    # most answers written apart saying nothing here is to change: the step
    # is not this function's (2 of 7 said so of do_GET, where the change
    # was; 4 of 7 of the bridge's _health, where none is)
    # -- or answers agreeing on it where no change written holds up (the
    # bridge's _health: 4 said nothing, the 22 changes all refused)
    if nothing >= LEAST_AGREED and (nothing * 2 > len(answers) or
                                    not passed):
        out.update({"status": "nothing", "said": (
            f"Nothing to change in {subject.name}: {nothing} of "
            f"{len(answers)} answers written say so"
            + ("." if passed or nothing * 2 > len(answers) else
               ", and none of the changes written passed the checks."))})
        out["seconds"] = round(time.time() - started, 2)
        return out
    if not passed:
        out.update({"status": "unchanged",
                    "said": (f"I could not change {subject.name} as asked: "
                             f"of {len(answers)} changes written, none "
                             f"{'could be put in' if not seen else 'passed the checks'}"
                             f" -- nothing was written.")})
        out["seconds"] = round(time.time() - started, 2)
        return out
    # a change of the architecture's own code for a case it failed: tried
    # with the case said again, and what it answered right before (v706
    # `inside.verify`) -- it compiles is no evidence it answers (`return
    # super().run(question, [])`, 3 of 7 agreed: a TypeError for every
    # question) -- and one that does is claimed, as a debugged fix is
    if (context or {}).get("verify") is not None or \
            (context or {}).get("score") is not None:
        # tried as they were written (in the rounds above)
        out["tried"] = [{key_: value for key_, value in one.items()
                         if key_ not in ("text", "made", "by")} for one in tried]
        passed = [row for row in passed if row.get("verified")]
        if not passed:
            why = next((one["refused"] for one in tried
                        if str(one.get("refused", "")).startswith(
                            "run again")), "none passed the checks")
            out.update({"status": "unchanged", "said": (
                f"I did not change {subject.name}: of {len(answers)} "
                f"changes written, none answers the case when it is run "
                f"again ({why}) -- nothing was written.")})
            out["seconds"] = round(time.time() - started, 2)
            return out
    for row in passed:
        row["size"] = sum(1 for one in diff(text, row["text"], path)
                          .splitlines() if one[:1] in "+-"
                          and not one.startswith(("+++", "---")))
    # an answer that makes a change and more besides agrees with that
    # change: three of seven wrote `"asked": shell.ASK,`, two of them with
    # a key of their own after it -- the more is a guess, the change agreed
    for row in passed:
        within = set().union(*(other["by"] for other in seen.values()
                               if other is not row and _extends(
                                   part, row["made"], other["made"])))
        more = len(within - row["by"])
        if more:
            row["by"] = row["by"] | within
            row["agree"] = len(row["by"])
            row["within"] = more
    # whether each does what was said: the judge's reading of the statement
    # against the change (`teach_judge.py`) -- a change it does not believe
    # is not written, however well it compiles
    from research.v700 import teach_judge as J
    judging = J.judge()
    if judging is not None:
        chances = judging.chances(statement, [J.diff_of(part, row["made"])
                                              for row in passed])
        for row, chance in zip(passed, chances):
            row["judged"] = round(chance, 3)
            # what most answers, written apart, make is not the judge's to
            # refuse: it read removing an unused import that 7 of 7 wrote
            # as not what was said (0.04)
            most = row["agree"] * 2 > len(answers)
            # nor one that answers its case when run again
            if chance < JUDGE_FLOOR and not most and \
                    not row.get("verified"):
                row["refused"] = (f"the judge reads it as not what was "
                                  f"said ({chance:.2f})")
        passed = [row for row in passed if "refused" not in row]
        out["tried"] = [{key_: value for key_, value in one.items()
                         if key_ not in ("text", "made", "by")} for one in tried]
        if not passed:
            out.update({"status": "unchanged", "said": (
                f"I could not change {subject.name} as asked: of "
                f"{len(answers)} changes written, those that "
                f"{'still read as data' if data else 'compile'} do "
                f"not do what you said -- nothing was written.")})
            out["seconds"] = round(time.time() - started, 2)
            return out
    # a change one answer wrote is a guess: written into the file only
    # where answers written apart agree on it (`_meets` was rewritten,
    # its comparison with what it must give dropped, by 1 of 13)
    # -- or one made by the kind the request names, where none the editors
    # wrote agree: the judge reading it as what was said, and clearly
    # above the next made so (it is made, not sampled: two copies of it
    # agreeing would say nothing)
    # (above the next made so, not above the editors' too: the judge read
    # answers that wrote `spec.language(got.program)` at 0.92 and the
    # change asked at 0.85)
    by_kind = sorted((row for row in passed if row.get("kind")),
                     key=lambda one: -one.get("judged", 0.0))
    if by_kind and "judged" in by_kind[0] and not any(
            row["agree"] >= LEAST_AGREED for row in passed
            if not row.get("kind")):
        best_ = by_kind[0]
        second = by_kind[1].get("judged", 0.0) if len(by_kind) > 1 else 0.0
        if best_["judged"] >= KIND_JUDGED and \
                best_["judged"] - second >= KIND_MARGIN:
            best_["by kind"] = True
    agreed = [row for row in passed if row["agree"] >= LEAST_AGREED
              or row.get("verified") or row.get("by kind")]
    if (context or {}).get("verify") is not None or \
            (context or {}).get("score") is not None:
        # what was asked is a check that runs (v698 `requested`): only a
        # change it holds for is written -- answers agreeing on one it does
        # not hold for agree on a wrong change (`rename parse_isbn ...
        # everywhere`: 7 of 7, two lines of one file)
        agreed = [row for row in passed if row.get("verified")]
    if not agreed:
        best = max(passed, key=lambda one: one.get("judged", 0.0))
        out["guess"] = diff(text, best["text"], path)
        out.update({"status": "unsure", "said": (
            f"I did not change {subject.name}: of {len(answers)} changes "
            f"written, {len(passed)} pass the checks, and "
            + ("none does what the tests of what you asked check"
               if (context or {}).get("verify") is not None else
               "no two agree")
            + " -- I do not write a guess into your file. The likeliest is "
            f"under `guess`.")})
        out["seconds"] = round(time.time() - started, 2)
        return out
    passed = agreed
    # how many answers wrote it, weighed by the judge's belief that it is
    # what was said: answers written apart agreeing is evidence the judge
    # does not have (it prefers changes with more in them); then the
    # smallest -- what was not asked is not changed -- and the greedy one
    best = next((row for row in passed if row.get("by kind")), None) or max(
        passed, key=lambda one: (one["agree"] * one.get("judged", 1.0),
                                 -one["size"], one["greedy"]))
    written = held.write(path, best["text"])
    MADE.setdefault(key, []).append((path, text, best["text"]))
    changed = diff(text, best["text"], path)
    count = sum(1 for one in changed.splitlines()
                if one[:1] in "+-" and not one.startswith(("+++", "---")))
    where_ = "in the file" if written else "in the project I hold (no " \
        "folder on disk was given)"
    out.update({
        "status": "changed", "written": written, "diff": changed,
        "by kind": bool(best.get("by kind")),
        "agree": best["agree"], "of": len(answers),
        "touched": best["touched"],
        "said": (f"Changed "
                 f"{path if subject.kind == 'file' else subject.name + ' in ' + path}"
                 f" ({count} lines), "
                 f"{where_}. "
                 + (f"Made as what you asked for ({best['kind']}), the "
                    f"judge reading it so ({best.get('judged', 0):.2f}), and "
                    if best.get("by kind") else
                    (f"{sum(len(one.get('by') or {1}) for one in passed)} "
                     f"changes written apart do the same on what you asked "
                     f"about ({len(best.get('gives') or ())} calls run)"
                     + (f" and pass {len(best.get('passes') or ())} of "
                        f"{(context or {}).get('tests', 0)} tests of it"
                        if (context or {}).get("tests") else "")
                     + ", the project's own tests still pass, and "
                     if "gives" in best else
                     f"{best['agree']} of {len(answers)} changes written "
                     f"agree on it, and "))
                 + ("it still reads as data and repeats no place it has."
                    if data else "the compiler finds nothing more wrong "
                    "with the file."))})
    out["seconds"] = round(time.time() - started, 2)
    return out


def undo(held, key) -> dict | None:
    """The last change made in this conversation, put back."""
    made = MADE.get(key) or []
    if not made or held is None:
        return None
    path, before, after = made.pop()
    if held.files.get(path) != after:
        return {"status": "unchanged",
                "said": f"{path} has changed since; I left it as it is."}
    written = held.write(path, before)
    return {"status": "undone", "file": path, "written": written,
            "diff": diff(after, before, path),
            "said": f"Put {path} back as it was."}
