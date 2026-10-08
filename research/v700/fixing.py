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
#: first editor's 80 of 160 held faults are 155 of 160
EDITOR = "editor3"
SEED = 700
#: answers sampled beside the greedy one, in each round; and the rounds,
#: asked again while nothing passes that changes all the statement names
SAMPLES, ROUNDS = 6, 3
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


def _editor():
    from research.v697.coding import Tools
    return Tools.get().writer(EDITOR)


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
    lines = held.files[subject.file].splitlines()
    if end - start + 1 > LONGEST_PART:
        found = window(lines, start, end, statement, subject.file)
        if found is None:
            return None
        start, end = found
    return subject.file, start, end


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
        return (sum(one in code for one in others),
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


def diff(before: str, after: str, path: str) -> str:
    return "".join(difflib.unified_diff(
        _lines(before), _lines(after), f"a/{path}", f"b/{path}", n=2))


#: passes made of one statement, while it names what no pass has changed
#: (`delete both`: the editor makes one change at a time)
PASSES = 4


def change(statement: str, held, subject, key=None) -> dict:
    """The change asked, made -- again, on what it made, while the
    statement names code no pass has changed yet, each pass changing some
    of what is left; the passes kept as one change (`undo` puts all back)."""
    first = _change_once(statement, held, subject, key)
    if first["status"] != "changed":
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
                            need=named - done)
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


def _change_once(statement: str, held, subject, key=None,
                 need: set | None = None) -> dict:
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
    asked = T.prompt(statement, path, part)
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
    named = said_names(statement, text, path, start, end, subject)
    flakes_before = _flakes(path, text)
    tried, seen, passed, answers = [], {}, [], []
    for round_ in range(ROUNDS):
        # each request sampled by a seed of its own and the round's, as the
        # writers are (`sketcher.proposals`): asked again, the same answers
        writer = _editor()
        writer.torch.manual_seed(SEED + zlib.crc32(asked.encode()) + round_)
        written_now = writer.write([asked], samples=SAMPLES, longest=400,
                                   greedy=round_ == 0)[0]
        answers += written_now
        fresh = []
        for number, answer in enumerate(written_now):
            greedy = round_ == 0 and number == 0
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
                if made == part:
                    row["refused"] = "it changes nothing"
                    continue
                if made in seen:
                    seen[made]["agree"] += 1
                    row["same as"] = seen[made]["index"]
                    continue
                after = "".join(lines[:start - 1]) + made + \
                    "".join(lines[end:])
                row.update({"index": len(tried) - 1, "agree": 1,
                            "text": after, "made": made, "touched": sorted(
                                named & _changed_names(part, made))})
                seen[made] = row
                fresh.append(row)
        for row in fresh:
            if need and not set(row["touched"]) & need:
                row["refused"] = ("it changes none of what is left "
                                  f"({', '.join(sorted(need))})")
                continue
            if named and not row["touched"]:
                # a change of nothing the statement names is not the
                # change it asks for, however well it compiles
                row["refused"] = ("it changes nothing the statement names "
                                  f"({', '.join(sorted(named))})")
                continue
            if data:
                wrong = data_wrong(path, text, row["text"])
                if wrong:
                    row["refused"] = wrong
                    continue
                passed.append(row)
                continue
            if not _reads(path, row["text"]):
                row["refused"] = "the file no longer reads"
                continue
            try:
                found = _diagnosed(held, path, row["text"])
            except Unchecked as bad:
                row["refused"] = f"the compiler could not check it: {bad}"
                continue
            row["diagnostics"] = len(found)
            more = _more_wrong(before, found)
            if more:
                row["refused"] = (f"the compiler finds {len(more)} more "
                                  f"wrong: {more[0].get('message', '')}")
                continue
            more = _more_wrong(flakes_before, _flakes(path, row["text"]))
            if more:
                row["refused"] = (f"pyflakes finds {len(more)} more wrong: "
                                  f"{more[0]['message']}")
                continue
            unfound = _unfound_imports(held, text, row["text"], path)
            if unfound:
                row["refused"] = (f"it imports what the project does not "
                                  f"have: {', '.join(unfound)}")
                continue
            passed.append(row)
        # asked again only while nothing passed: the words a statement
        # shares with the code are not all code it names (`it`, `never`
        # are names in some code), so all of them touched is no stop
        if passed:
            break
    out = {"subject": subject.name, "file": path, "lines": [start, end],
           "named": sorted(named),
           "tried": [{key_: value for key_, value in one.items()
                      if key_ not in ("text", "made")} for one in tried],
           "diagnostics before": len(before)}
    if not passed:
        out.update({"status": "unchanged",
                    "said": (f"I could not change {subject.name} as asked: "
                             f"of {len(answers)} changes written, none "
                             f"{'could be put in' if not seen else 'passed the checks'}"
                             f" -- nothing was written.")})
        out["seconds"] = round(time.time() - started, 2)
        return out
    for row in passed:
        row["size"] = sum(1 for one in diff(text, row["text"], path)
                          .splitlines() if one[:1] in "+-"
                          and not one.startswith(("+++", "---")))
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
            if chance < JUDGE_FLOOR and not most:
                row["refused"] = (f"the judge reads it as not what was "
                                  f"said ({chance:.2f})")
        passed = [row for row in passed if "refused" not in row]
        out["tried"] = [{key_: value for key_, value in one.items()
                         if key_ not in ("text", "made")} for one in tried]
        if not passed:
            out.update({"status": "unchanged", "said": (
                f"I could not change {subject.name} as asked: of "
                f"{len(answers)} changes written, those that compile do "
                f"not do what you said -- nothing was written.")})
            out["seconds"] = round(time.time() - started, 2)
            return out
    # a change one answer wrote is a guess: written into the file only
    # where answers written apart agree on it (`_meets` was rewritten,
    # its comparison with what it must give dropped, by 1 of 13)
    agreed = [row for row in passed if row["agree"] >= LEAST_AGREED]
    if not agreed:
        best = max(passed, key=lambda one: one.get("judged", 0.0))
        out["guess"] = diff(text, best["text"], path)
        out.update({"status": "unsure", "said": (
            f"I did not change {subject.name}: of {len(answers)} changes "
            f"written, {len(passed)} pass the checks, and no two agree -- "
            f"I do not write a guess into your file. The likeliest is "
            f"under `guess`.")})
        out["seconds"] = round(time.time() - started, 2)
        return out
    passed = agreed
    # how many answers wrote it, weighed by the judge's belief that it is
    # what was said: answers written apart agreeing is evidence the judge
    # does not have (it prefers changes with more in them); then the
    # smallest -- what was not asked is not changed -- and the greedy one
    best = max(passed, key=lambda one: (one["agree"] * one.get("judged", 1.0),
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
        "agree": best["agree"], "of": len(answers),
        "touched": best["touched"],
        "said": (f"Changed "
                 f"{path if subject.kind == 'file' else subject.name + ' in ' + path}"
                 f" ({count} lines), "
                 f"{where_}. {best['agree']} of {len(answers)} changes "
                 f"written agree on it, and the compiler finds nothing "
                 f"more wrong with the file.")})
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
