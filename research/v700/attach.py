"""The attachment, measured: statements Claude would say to fix the
architecture's own code, each read, placed and made by the architecture --
in a copy of the files, so a run changes nothing of the repository.

    python -m research.v700.attach            every statement
    python -m research.v700.attach --only 3   one

Each statement is one a person says of a fault pyflakes finds in the code
(an import never used, a name assigned and never read): the change is right
where that finding is gone, no new one is found, and nothing else in the
file changed but the lines the finding is on or beside. Reading, placing
and making are each said: where it went wrong is what is measured.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
#: what the project is: the code the statements are about, and around it
FOLDERS = ("research/v696", "research/v697", "research/v698",
           "tools/graph-topology-mcp")

#: (statement, file, what pyflakes says that must be gone -- one, or all of
#: several)
STATEMENTS = [
    ("_meets imports checker but never uses it -- drop the import",
     "research/v696/sketcher.py", "'research.v696.checker.checker' imported"),
    ("forward assigns torch and never uses it; remove that line",
     "research/v696/reader.py", "local variable 'torch'"),
    ("deduced computes element and outputs but never uses either, delete "
     "both assignments", "research/v696/forms.py",
     ("local variable 'element'", "local variable 'outputs'")),
    ("readings imports checker without using it", "research/v696/risk.py",
     "'research.v696.checker.checker' imported"),
    ("search.py imports math, field, PredicateTrie, ROOT and checker but uses "
     "none of them; remove those imports", "research/v696/search.py",
     ("'math' imported", "'dataclasses.field' imported",
      "'research.v687.trie.PredicateTrie' imported",
      "'research.v687.trie.ROOT' imported",
      "'research.v696.checker.checker' imported")),
    ("meaning.py imports field from dataclasses and never uses it",
     "research/v696/meaning.py", "'dataclasses.field' imported"),
    ("drop the unused sys import in projects.py", "research/v696/projects.py",
     "'sys' imported"),
    ("pycorpus.py never uses math, remove the import",
     "research/v696/pycorpus.py", "'math' imported"),
    ("prelude imports pytypes again inside the loop over the helpers; the "
     "import in the branch above is enough -- move it before the loops "
     "and import it once", "research/v696/pyprint.py",
     "import 'pytypes' from line"),
]


def flakes(path: Path) -> list:
    import io
    from pyflakes.api import check
    from pyflakes.reporter import Reporter
    out, err = io.StringIO(), io.StringIO()
    check(path.read_text(encoding="utf-8"), str(path), Reporter(out, err))
    return [line.split(":", 3)[-1].strip() for line in
            (out.getvalue() + err.getvalue()).splitlines() if line.strip()]


def project(root: Path):
    from research.v698.project import READ, Project
    held = Project("graph-topology", str(root))
    files = {}
    for folder in FOLDERS:
        for path in sorted((root / folder).rglob("*")):
            name = path.relative_to(root).as_posix()
            if path.is_file() and name.endswith(READ) and \
                    "__pycache__" not in name:
                files[name] = path.read_text(encoding="utf-8")
    held.put(files)
    return held


def known(held) -> set:
    names = set()
    for one in held.functions():
        names.add(one["name"])
        names.add(one["name"].split(".")[-1])
    for path in held.files:
        names.add(path)
        names.add(path.rsplit("/", 1)[-1])
    return names


def one(statement: str, path: str, finding: str, held, root: Path) -> dict:
    from research.v698 import asking, reading
    from research.v700 import fixing
    out = {"said": statement, "file": path}
    before = flakes(root / path)
    findings = (finding,) if isinstance(finding, str) else finding
    if not any(one in line for line in before for one in findings):
        # fixed already (by the architecture, asked through the MCP): not a
        # statement about this code any more
        out["already so"] = True
        return out
    found = reading.read(statement, known(held))
    out["read"] = None if found is None else {
        "act": found.act, "chance": round(found.chance, 3),
        "subject": found.subject, "spans": found.spans}
    if found is None or found.act != "change" or found.chance < reading.FLOOR:
        out["went wrong"] = "reading"
        return out
    subject = fixing.placed(found, held, statement) or \
        asking.resolve(found, held, None, 0, "attach")
    out["placed"] = None if subject is None else [subject.kind, subject.name,
                                                  subject.file]
    if subject is None or subject.kind not in ("function", "file") or \
            subject.file != path:
        out["went wrong"] = "placing"
        return out
    made = fixing.change(statement, held, subject, "attach")
    out["made"] = {key: made.get(key) for key in ("status", "lines", "diff",
                                                  "said", "seconds")}
    if made["status"] != "changed":
        out["went wrong"] = "making"
        out["tried"] = made.get("tried")
        return out
    after = flakes(root / path)
    gone = not any(one in line for line in after for one in findings)
    new = [line for line in after if line not in before]
    out["finding gone"], out["new findings"] = gone, new
    if not gone or new:
        out["went wrong"] = "the change"
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=int, default=None)
    parser.add_argument("--out", default=None)
    options = parser.parse_args(argv)
    rows = []
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        for folder in FOLDERS:
            shutil.copytree(ROOT / folder, root / folder,
                            ignore=shutil.ignore_patterns("__pycache__"))
        held = project(root)
        for at, (statement, path, finding) in enumerate(STATEMENTS):
            if options.only is not None and at != options.only:
                continue
            row = one(statement, path, finding, held, root)
            row["at"] = at
            rows.append(row)
            print(json.dumps(row, indent=1), flush=True)
    rows_asked = [row for row in rows if not row.get("already so")]
    right = sum("went wrong" not in row for row in rows_asked)
    wrong = {}
    for row in rows:
        if "went wrong" in row:
            wrong[row["went wrong"]] = wrong.get(row["went wrong"], 0) + 1
    print(json.dumps({"right": right, "of": len(rows_asked), "wrong": wrong,
                      "already so": len(rows) - len(rows_asked)}))
    if options.out:
        Path(options.out).write_text(json.dumps(rows, indent=1),
                                     encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
