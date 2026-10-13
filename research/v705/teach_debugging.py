"""Teaching the editor to fix what a run shows wrong (v705), in Python and
TypeScript alike.

    python -m research.v705.teach_debugging bugs      llm/debug-data/bugs-*.jsonl
    python -m research.v705.teach_debugging corpus    llm/debug-data/fixes.jsonl
    python -m research.v705.teach_debugging train     llm/editor-debug
    python -m research.v705.teach_debugging measure   held out, both languages

Looking inside its own code, the architecture found where "what is my
dog called?" is lost (`_whose` knows `my`, not `my` and a noun) -- and its
editor, told the line, wrote the case into the code. It was taught changes
from commit messages; never from what a run showed.

**Bugs, by construction**: each verified program (MBPP's for teaching,
HumanEval's held; TypeScript and Python) made wrong by one edit
(`editing.edits`, `pyediting.edits`) that fails one of its cases. **What
is said**: the call, what it gave and what it should have, and the run --
each line it took with what changed, as `debugging.said` says it -- the
traced run of the failing case. **What is answered**: the edit undone, as
the editor's blocks. **Measured** on bugs it never saw: HumanEval's made
the same way, and HumanEvalFix's -- people's bugs -- in both languages,
fixed where the program then passes its cases.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from research.v705 import debugging

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "llm" / "debug-data"
CORPUS = DATA / "fixes.jsonl"
EDITOR = ROOT / "llm" / "editor-debug"
SOLUTIONS = ROOT / "data" / "code-meaning"
SEED = 705
#: the bugs made of each program at most, and the cases a statement shows
EACH = 6
#: what the editor is told it does
SAYING = ("You fix code from what a run of it shows. Answer with each "
          "change as <<< the lines as they are === the lines as they "
          "become >>>, nothing else.")
SOURCES = {
    ("typescript", "train"): ("solutions.jsonl", "mbpp-ts"),
    ("typescript", "held"): ("solutions-held.jsonl", "humaneval-ts"),
    ("python", "train"): ("solutions-py.jsonl", "mbpp-py"),
    ("python", "held"): ("solutions-held-py.jsonl", "humaneval-py"),
}


def _pairs(task, language: str) -> list:
    from research.v696 import meaning as M
    if language == "python":
        return M.values_of(M.test_pairs_python(task.tests), "python")
    return M.values_of(M.test_pairs(task.tests))


def _edits(source: str, task, language: str) -> list:
    if language == "python":
        from research.v696.pyediting import edits
        return edits(source, task.entry)
    from research.v696.editing import edits
    from research.v696.parse import parse
    tree = parse(source, task.entry, task.params)
    return edits(tree, source, task.entry) if tree is not None else []


class Dropped:
    """A missing case (v706): a guarded branch -- an `if` and all it holds
    -- taken out, so the fix is putting code in, not changing a line. The
    edits taught before only ever changed a line, and the editor never
    added one: asked to make `_whose` give a Mention for `my dog`, 61
    answers each changed a line that was there."""
    kind = "missing"

    def __init__(self, first: int, last: int) -> None:
        self.first, self.last = first, last         # 0-based, inclusive

    def apply(self, source: str) -> str:
        lines = source.splitlines(keepends=True)
        return "".join(lines[:self.first] + lines[self.last + 1:])


def _dropped(source: str, language: str) -> list:
    """Each `if` block that can be taken out whole, leaving code that still
    reads (Python by its tree; TypeScript by its braces, read by the
    checker when run)."""
    lines = source.splitlines()
    out = []
    if language == "python":
        import ast
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        for node in ast.walk(tree):
            # an `if` block, or any one statement of a body: a missing
            # line is a missing case too (`count += 1` never done)
            if isinstance(node, ast.stmt) and not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef,
                           ast.ClassDef, ast.Import, ast.ImportFrom)) \
                    and not (isinstance(node, ast.If) and node.orelse):
                made = Dropped(node.lineno - 1, node.end_lineno - 1)
                try:
                    ast.parse(made.apply(source))
                except SyntaxError:
                    continue
                out.append(made)
        return out
    for at, line in enumerate(lines):
        stripped = line.strip()
        if stripped.endswith(";") and not stripped.startswith(
                ("return", "}", "import", "export", "function")) and \
                stripped.count("(") == stripped.count(")"):
            # one statement on its line
            out.append(Dropped(at, at))
            continue
        if not line.lstrip().startswith("if (") or \
                line.lstrip().startswith("if (") and "else" in line:
            continue
        depth, end = 0, None
        for until in range(at, len(lines)):
            depth += lines[until].count("{") - lines[until].count("}")
            if depth <= 0 and (until > at or "{" not in lines[at]):
                end = until
                break
        if end is None or end + 1 < len(lines) and \
                lines[end + 1].lstrip().startswith("else"):
            continue
        out.append(Dropped(at, end))
    return out


#: of the bugs made of each program, those that are a missing case at most
#: (v706; `--missing`: editor-debug2 was taught with 2, editor-debug none)
MISSING = 0
#: how often each missing case is taught
REPEATED = 3


def bugs(language: str, split: str, each: int = EACH) -> list:
    """Each verified program made wrong by single edits that fail a case:
    the bug, the failing case, what it gave, what it should, its run."""
    from research.v696 import tasks
    from research.v696.checker import checker
    name, config = SOURCES[(language, split)]
    by = {one.name: one for one in tasks.load(config)}
    rng = random.Random(SEED)
    running = checker(language)
    out = []
    for line in (SOLUTIONS / name).open(encoding="utf-8"):
        row = json.loads(line)
        task = by.get(row["name"])
        if task is None or not row.get("code"):
            continue
        pairs = _pairs(task, language)
        if not pairs:
            continue
        cases = [list(args) for args, _ in pairs]
        wanted = [want for _, want in pairs]
        choices = _edits(row["code"], task, language)
        rng.shuffle(choices)
        if MISSING:
            # the missing cases first, a few: a fix that puts code in
            dropped = _dropped(row["code"], language)
            rng.shuffle(dropped)
            choices = dropped[:MISSING] + choices
        made = 0
        for one in choices:
            if made >= each:
                break
            broken = one.apply(row["code"])
            if broken == row["code"]:
                continue
            try:
                got = running.run(broken, task.entry, cases)
            except Exception:                       # noqa: BLE001
                running.restart()
                continue
            failing = [at for at, (one_got, want) in enumerate(
                zip(got, wanted)) if debugging.wrong(one_got, want)]
            if not failing or len(failing) == len(cases) and any(
                    "error" in g and "SyntaxError" in g["error"]
                    for g in got):
                continue
            at = failing[0]
            traced = running.trace(broken, task.entry, cases[at])
            if not traced.get("ok", True) or "steps" not in traced:
                continue
            out.append({"name": f"{task.name}-{one.kind}-{made}",
                        "language": language, "split": split,
                        "entry": task.entry, "source": broken,
                        "params": [list(one) for one in task.params],
                        "returns": task.returns,
                        "original": row["code"], "case": cases[at],
                        "got": got[at], "wanted": wanted[at],
                        "cases": cases, "all wanted": wanted,
                        "run": debugging.said(traced, broken,
                                              language=language)})
            made += 1
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"bugs-{language}-{split}.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for one in out:
            stream.write(json.dumps(one) + "\n")
    print(f"{len(out)} {language} bugs ({split}) -> {path}", flush=True)
    return out


def _shown(value, language: str = "") -> str:
    said = debugging.as_written(json.dumps(value), language)
    return said if len(said) <= 120 else said[:120] + "..."


def statement(bug: dict, traced: bool = True) -> str:
    """What the editor is told: the call, what it gave and should have
    given, and -- traced -- how the run went."""
    language = bug["language"]
    args = ", ".join(_shown(one, language) for one in bug["case"])
    got = bug["got"]
    gave = (f"raises {got['error']}" if "error" in got else
            f"gives {_shown(got.get('value'), language)}")
    said = (f"{bug['entry']}({args}) {gave} but should give "
            f"{_shown(bug['wanted'], language)}. Fix it.")
    if traced:
        said += f"\nThe run: {bug['run']}"
    return said


def corpus() -> dict:
    """The teaching bugs as the editor's rows: the statement with its run,
    the program, the edit undone as blocks."""
    from research.v700 import teach_editor as T
    rows = []
    for language in ("typescript", "python"):
        path = DATA / f"bugs-{language}-train.jsonl"
        for line in path.open(encoding="utf-8"):
            bug = json.loads(line)
            old = bug["source"].splitlines(keepends=True)
            new = bug["original"].splitlines(keepends=True)
            blocks = T.blocks_of(old, new, 0, len(old))
            if not blocks:
                continue
            rows.append({"split": "train", "language": language,
                         "statement": statement(bug), "path":
                         f"{bug['entry']}.{'py' if language == 'python' else 'ts'}",
                         "part": bug["source"], "start": 1,
                         "target": T.said(blocks),
                         "missing": "-missing-" in bug["name"]})
    random.Random(SEED).shuffle(rows)
    # a share kept apart to watch the loss on
    for one in rows[:len(rows) // 20]:
        one["split"] = "dev"
    # the missing cases taught more than once (v706): a few among many
    # changed lines, and the only ones whose fix puts code in
    rows += [dict(one) for one in rows if one.get("missing") and
             one["split"] == "train"] * (REPEATED - 1)
    random.Random(SEED + 1).shuffle(rows)
    with CORPUS.open("w", encoding="utf-8") as stream:
        for one in rows:
            stream.write(json.dumps(one) + "\n")
    counts = {language: sum(one["language"] == language for one in rows)
              for language in ("typescript", "python")}
    print(json.dumps(counts))
    return counts


def train(epochs: int = 2) -> None:
    """The editor taught to fix from a run: from editor5 (which knows the
    blocks), on the traced bugs."""
    from research.v700 import teach_editor as T
    T.SAYING = SAYING
    T.train(EDITOR, epochs=epochs, corpus=CORPUS, longest=1024,
            base=T.LLM / "editor5")


def _fixed(bug: dict, answer: str) -> bool:
    """Whether an answer's blocks, applied, make the program pass every
    case."""
    from research.v696.checker import checker
    from research.v700 import teach_editor as T
    made = T.applied(bug["source"], answer)
    if made is None or made == bug["source"]:
        return False
    try:
        got = checker(bug["language"]).run(made, bug["entry"], bug["cases"])
    except Exception:                               # noqa: BLE001
        checker(bug["language"]).restart()
        return False
    return not any(debugging.wrong(one, want) for one, want in
                   zip(got, bug["all wanted"]))


def _typed(rows: list, config: str) -> None:
    """Each bug made before its parameters were kept: its task's."""
    from research.v696 import tasks
    by = {one.name: one for one in tasks.load(config)}
    for row in rows:
        if "params" in row:
            continue
        task = next((one for name, one in by.items()
                     if row["name"].startswith(name + "-")), None)
        if task is not None:
            row["params"] = [list(one) for one in task.params]
            row["returns"] = task.returns


def held_bugs(most: int = 80) -> dict:
    """The bugs never taught: HumanEval's made the same way, and
    HumanEvalFix's (people's), each in both languages."""
    out = {}
    for language in ("typescript", "python"):
        path = DATA / f"bugs-{language}-held.jsonl"
        if path.exists():
            rows = [json.loads(line) for line in
                    path.open(encoding="utf-8")][:most]
            _typed(rows, SOURCES[(language, "held")][1])
            out[f"humaneval {language}"] = rows
    from research.v696 import bugs as B
    for language, made in (("typescript", B.humanevalfix),
                           ("python", B.humanevalfix_python)):
        rows = []
        for bug in made():
            from research.v696.checker import checker
            running = checker(language)
            got = running.run(bug.source, bug.entry, bug.cases)
            failing = [at for at, (one, want) in enumerate(
                zip(got, bug.wanted)) if debugging.wrong(one, want)]
            if not failing:
                continue
            at = failing[0]
            traced = running.trace(bug.source, bug.entry, bug.cases[at])
            rows.append({"name": bug.name, "language": language,
                         "entry": bug.entry, "source": bug.source,
                         "params": [list(one) for one in bug.params],
                         "returns": bug.returns,
                         "case": bug.cases[at], "got": got[at],
                         "wanted": bug.wanted[at], "cases": bug.cases,
                         "all wanted": bug.wanted, "tests": bug.tests,
                         "run": debugging.said(traced, bug.source,
                                               language=language)})
        out[f"humanevalfix {language}"] = rows[:most]
    return out


def measure(editor: str = "editor-debug", traced: bool = True,
            samples: int = 2) -> dict:
    """Bugs fixed, greedy and by any of the samples, on each held set."""
    from research.v696.sketcher import Sketcher
    model = Sketcher(ROOT / "llm" / editor)
    found = {}
    for name, rows in held_bugs().items():
        prompts = [f"{statement(bug, traced).strip()}\n\n"
                   f"{bug['entry']}.{'py' if bug['language'] == 'python' else 'ts'}:"
                   f"\n{bug['source'].rstrip()}\n" for bug in rows]
        written = model.write(prompts, samples=samples, longest=400,
                              saying=SAYING if editor == "editor-debug"
                              else None)
        greedy = sum(_fixed(bug, answers[0]) for bug, answers in
                     zip(rows, written))
        any_ = sum(any(_fixed(bug, one) for one in answers) for bug, answers
                   in zip(rows, written))
        found[name] = f"{greedy}/{any_} of {len(rows)}"
        print(name, found[name], flush=True)
    print(json.dumps({"editor": editor, "traced": traced, **found}))
    return found


def _split(bug: dict) -> tuple:
    """The cases shown -- the failing one, and half of the rest -- and the
    cases held back to judge a fix by."""
    order = [at for at, case in enumerate(bug["cases"])
             if case == bug["case"]][:1]
    order += [at for at in range(len(bug["cases"])) if at not in order]
    shown = order[:max(1, (len(order) + 1) // 2)]
    return shown, [at for at in order if at not in shown]


def _passes(bug: dict, made: str, ats: list) -> bool:
    from research.v696.checker import checker
    if not ats:
        return True
    running = checker(bug["language"])
    try:
        got = running.run(made, bug["entry"], [bug["cases"][at]
                                               for at in ats])
    except Exception:                               # noqa: BLE001
        running.restart()
        return False
    return not any(debugging.wrong(one, bug["all wanted"][at])
                   for one, at in zip(got, ats))


def _judged(bug: dict, made: str) -> bool:
    """A fix as it is judged: every case held back, and the task's own
    test file where it has one."""
    from research.v696.checker import checker
    _, hidden = _split(bug)
    if not _passes(bug, made, hidden):
        return False
    if not bug.get("tests"):
        return True
    running = checker(bug["language"])
    try:
        return running.tests(made + "\n" + bug["tests"]) is None
    except Exception:                               # noqa: BLE001
        running.restart()
        return False


def claim(bug: dict, answers: list) -> str | None:
    """The fix claimed: the first answer (greedy first) whose program
    passes every case shown; None -- unsure -- where none does."""
    from research.v700 import teach_editor as T
    shown, _ = _split(bug)
    for answer in answers:
        made = T.applied(bug["source"], answer)
        if made is None or made == bug["source"]:
            continue
        if _passes(bug, made, shown):
            return made
    return None


def searched(bug: dict) -> str | None:
    """A fix found by v696's repair -- edits searched for, suspected sites
    first, then a rewrite -- given only the cases shown, never those held
    back: its source, else None."""
    from research.v696.editing import Bug, repair
    shown, _ = _split(bug)
    asked = Bug(bug["name"], bug["source"], bug["entry"],
                bug.get("params") or [], bug.get("returns") or "",
                [bug["cases"][at] for at in shown],
                [bug["all wanted"][at] for at in shown],
                language=bug["language"])
    # its edits only: a function written again to pass the cases shown fits
    # them, it does not fix it (on HumanEvalFix's TypeScript, rewrites were
    # right 2 times of 14 by the cases held back, one edit 41 of 47)
    try:
        found = repair(asked, rewrite=False)
    except Exception:                               # noqa: BLE001
        return None
    return found.source if found.route == "fixed" else None


def precision(editor: str = "editor-debug", traced: bool = True,
              samples: int = 4, search: bool = False) -> dict:
    """On each held set: of the fixes claimed (an answer passing the cases
    shown), how many are right by what was held back -- precision -- and
    of all the bugs, how many are claimed -- coverage."""
    from research.v696.sketcher import Sketcher
    model = Sketcher(ROOT / "llm" / editor)
    found = {}
    for name, rows in held_bugs().items():
        prompts = [f"{statement(bug, traced).strip()}\n\n"
                   f"{bug['entry']}."
                   f"{'py' if bug['language'] == 'python' else 'ts'}:"
                   f"\n{bug['source'].rstrip()}\n" for bug in rows]
        written = model.write(prompts, samples=samples, longest=400,
                              saying=SAYING if editor != "editor5"
                              else None)
        claimed = right = 0
        for bug, answers in zip(rows, written):
            made = claim(bug, answers)
            if made is None and search:
                made = searched(bug)
            if made is None:
                continue
            claimed += 1
            right += _judged(bug, made)
        found[name] = {"precision": round(right / claimed, 3)
                       if claimed else None, "claimed": claimed,
                       "right": right, "of": len(rows)}
        print(name, json.dumps(found[name]), flush=True)
    print(json.dumps({"editor": editor, "traced": traced,
                      "search": search, **found}))
    return found


def main(argv=None) -> int:
    global MISSING, DATA, CORPUS, EDITOR
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("bugs", "corpus", "train",
                                        "measure", "precision"))
    parser.add_argument("--editor", default="editor-debug")
    parser.add_argument("--untraced", action="store_true")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--missing", type=int, default=MISSING)
    parser.add_argument("--data", default=DATA.name,
                        help="the folder of llm/ the bugs and fixes are in")
    parser.add_argument("--out", default=EDITOR.name,
                        help="the editor taught, in llm/")
    options = parser.parse_args(argv)
    MISSING = options.missing
    DATA = ROOT / "llm" / options.data
    CORPUS = DATA / "fixes.jsonl"
    EDITOR = ROOT / "llm" / options.out
    if options.job == "bugs":
        for language in ("typescript", "python"):
            for split in ("train", "held"):
                bugs(language, split)
    elif options.job == "corpus":
        corpus()
    elif options.job == "train":
        train()
    elif options.job == "precision":
        precision(options.editor, not options.untraced,
                  search=options.search)
    else:
        measure(options.editor, not options.untraced)
    return 0


if __name__ == "__main__":
    sys.exit(main())
