"""Faults found in real code, fixed by construction: what the editor is
taught beside the commits.

    python -m research.v700.teach_faults phrase   the teacher's statements
    python -m research.v700.teach_faults rows     llm/editor-data/faults.jsonl

Asked to drop `checker` from `from x import CheckerError, checker`, the
editor (taught on commits alone) took the whole line out: a change inside
a line is rare in commits, and a change of a function named in the message
is a few of them. Here such changes are made where they are true: the
Python files of CommitPackFT's commits (`old_contents`, of commits taught
-- `train` -- only), read by pyflakes for

- an import never used: the name taken out of its import line (the line,
  where it is the only one) -- of the file, or of a function;
- a local name assigned on one line and never used: the line taken out.

**What it is told** is a statement about the fault, as people say it: the
teacher (SmolLM3) writes them, from a statement with real names in it
(`phrase`), and each is said of every fault with its names put in -- not
one of mine, or the editor would be taught the attachment's own words
(`attach.py`). **What it is shown** is what the runtime shows it
(`fixing.part_of`): the function, or the lines of the file around the
import. **What it answers** is the change as blocks (`teach_editor.said`).
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys

from research.v700 import teach_editor as T

PHRASES = T.OUT / "fault-phrases.jsonl"
FAULTS = T.OUT / "faults.jsonl"
SEED = 700
#: the most faults of one kind taken from one file
EACH_FILE = 2

#: each kind of fault, said once with real names, for the teacher to say
#: again; the names are put back as `{f}` (file), `{n}` (function), `{x}`
#: (the name), `{y}` (a second name) in what it writes
SEEDS = {
    "import-file": ("payment_gateway.py imports hashlib but never uses "
                    "it; remove the import",
                    {"payment_gateway.py": "{f}", "hashlib": "{x}"}),
    "import-function": ("render_invoice imports hashlib and never uses "
                        "it -- take the import out",
                        {"render_invoice": "{n}", "hashlib": "{x}"}),
    "variable": ("render_invoice assigns subtotal but never reads it, "
                 "delete that line",
                 {"render_invoice": "{n}", "subtotal": "{x}"}),
    "variables": ("render_invoice sets subtotal and discount and uses "
                  "neither; remove both",
                  {"render_invoice": "{n}", "subtotal": "{x}",
                   "discount": "{y}"}),
}


def phrase(samples: int = 3) -> None:
    """The teacher says each seed again, many ways; kept with the names
    put back as places."""
    from research.v696.teach_meaning import Teacher
    T.OUT.mkdir(parents=True, exist_ok=True)
    teacher = Teacher()
    teacher.torch.manual_seed(SEED)
    prompts = [(kind, seed, names, (
        f"Say this request to a coding assistant in 15 different ways, as "
        f"a programmer would type it: short and long, casual and terse, "
        f"some with the fault first and the fix after, some just the fix. "
        f"Keep {', '.join(names)} exactly so in every one. One per line, "
        f"nothing else: \"{seed}\""))
        for kind, (seed, names) in SEEDS.items()]
    replies = teacher.write([one[3] for one in prompts], longest=900,
                            samples=samples, temperature=0.9)
    with PHRASES.open("w", encoding="utf-8") as out:
        for (kind, seed, names, _), written in zip(prompts, replies):
            kept = {seed}
            for reply in written:
                for line in reply.splitlines():
                    line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line)
                    line = line.strip().strip("\"'“”")
                    if 8 <= len(line) <= 200 and all(
                            re.search(r"\b" + re.escape(name) + r"\b", line)
                            for name in names):
                        kept.add(line)
            for line in sorted(kept):
                said = line
                for name, place in names.items():
                    said = re.sub(r"\b" + re.escape(name) + r"\b", place,
                                  said)
                out.write(json.dumps({"kind": kind, "phrase": said}) + "\n")
            print(f"{kind}: {len(kept)} phrases", flush=True)


#: what the seeds were about, beyond the fault: a phrase still saying it
#: drifted from the fault into the seed's story (`so {n} processes invoice
#: elements only`)
STORY = ("invoice", "payment", "gateway")


def _clean(phrase: str) -> str | None:
    """A phrase as a programmer types it: the teacher's labels (`casual:`)
    and quoting taken off; none where it drifted into the seed's story."""
    said = re.sub(r"^\s*(casual|terse|short|long|formal|fault first|fix "
                  r"first|just the fix)\s*[:\-]\s*", "", phrase, flags=re.I)
    said = said.strip().strip("`\"'“” ").strip()
    if any(word in said.lower() for word in STORY) or len(said) < 8:
        return None
    return said


# -- the faults, and their fixes ------------------------------------------------

def _unused(source: str, path: str) -> list:
    """pyflakes' messages: (kind, line, name)."""
    from pyflakes import checker, messages
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    found = checker.Checker(tree, filename=path)
    out = []
    for one in found.messages:
        if isinstance(one, messages.UnusedImport):
            out.append(("import", one.lineno, one.message_args[0]))
        elif isinstance(one, messages.UnusedVariable):
            out.append(("variable", one.lineno, one.message_args[0]))
    return out


def _functions(tree) -> list:
    return [node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _holder(tree, line: int):
    """The innermost function a line is in, or None."""
    best = None
    for node in _functions(tree):
        if node.lineno <= line <= node.end_lineno and (
                best is None or node.lineno > best.lineno):
            best = node
    return best


def _without_import(line: str, full: str) -> str | None:
    """The import line with the name pyflakes said (`a.b`, `a.b as c`)
    taken out, or "" where it was the only one; None where it cannot be
    told on one line."""
    try:
        node = ast.parse(line.strip()).body[0]
    except (SyntaxError, IndexError, ValueError):
        return None
    if not isinstance(node, (ast.Import, ast.ImportFrom)):
        return None
    module = getattr(node, "module", None) or ""
    keep, gone = [], None
    for alias in node.names:
        said = alias.name if not module else f"{module}.{alias.name}"
        if alias.asname:
            said += f" as {alias.asname}"
        if gone is None and (said == full or alias.name == full):
            gone = alias
        else:
            keep.append(alias)
    if gone is None:
        return None
    if not keep:
        return ""
    indent = line[:len(line) - len(line.lstrip())]
    names = ", ".join(one.name + (f" as {one.asname}" if one.asname else "")
                      for one in keep)
    if isinstance(node, ast.Import):
        return f"{indent}import {names}\n"
    dots = "." * node.level
    return f"{indent}from {dots}{module} import {names}\n"


def _one_line_assignment(line: str, name: str) -> bool:
    found = re.match(r"^\s*" + re.escape(name) + r"\s*=\s*(.+)$",
                     line.rstrip("\n"))
    if not found or found.group(1).rstrip().endswith(("\\", "(", "[", "{",
                                                       ",")):
        return False
    try:
        ast.parse(line.strip())
    except SyntaxError:
        return False
    return True


def faults_of(source: str, path: str) -> list:
    """(kind, names, holder function or None, changed lines {index: new
    line or ""}) of a file."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    lines = source.splitlines(True)
    by_function: dict = {}
    out = []
    for kind, line, name in _unused(source, path):
        if line < 1 or line > len(lines):
            continue
        text = lines[line - 1]
        holder = _holder(tree, line)
        if kind == "import":
            new = _without_import(text, name)
            if new is None:
                continue
            short = name.split(" as ")[-1].split(".")[-1]
            out.append(("import", [short], holder, {line - 1: new}))
        elif holder is not None and _one_line_assignment(text, name):
            by_function.setdefault(holder.lineno, (holder, []))[1].append(
                (line - 1, name))
    for holder, found in by_function.values():
        if holder.end_lineno - holder.lineno > 40:
            continue
        for at, name in found:
            out.append(("variable", [name], holder, {at: ""}))
        if len(found) >= 2:
            out.append(("variables", [name for _, name in found[:2]],
                        holder, {at: "" for at, _ in found[:2]}))
    return out


def row_of(record: dict, fault, phrases: dict, rng) -> dict | None:
    kind, names, holder, changes = fault
    source = record["old_contents"]
    path = record["old_file"]
    lines = source.splitlines(True)
    if not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    new = [changes.get(at, one) for at, one in enumerate(lines)]
    new = [one for one in new if one != ""]
    if holder is not None:
        start = min([holder.lineno] + [one.lineno for one in
                                       holder.decorator_list]) - 1
        end = holder.end_lineno
        if end - start > T.LONGEST_PART:
            return None
        said_kind = kind if kind != "import" else "import-function"
    else:
        at = min(changes)
        start, end = max(0, at - T.AROUND), min(len(lines), at + T.AROUND + 1)
        said_kind = "import-file"
    blocks = T.blocks_of(lines, new, start, end)
    if not blocks:
        return None
    pool = phrases.get(said_kind)
    if not pool:
        return None
    statement = rng.choice(pool).format(
        f=path.rsplit("/", 1)[-1], n=holder.name if holder else "",
        x=names[0], y=names[1] if len(names) > 1 else "")
    part = "".join(lines[start:end])
    target = T.said(blocks)
    if T.applied(part, target) is None:
        return None
    return {"commit": record["commit"], "split": T.split_of(record["commit"]),
            "language": "python", "statement": statement, "path": path,
            "part": part, "start": start + 1, "target": target,
            "source": f"fault {kind}"}


#: the most of each kind of fault: imports never used are in most files,
#: and would be all there is
MOST = {"fault import": 4000, "fault variable": 3000,
        "fault variables": 1000}


def rows() -> dict:
    phrases: dict = {}
    for line in PHRASES.open(encoding="utf-8"):
        row = json.loads(line)
        said = _clean(row["phrase"])
        if said:
            phrases.setdefault(row["kind"], []).append(said)
    rng = random.Random(SEED)
    counts: dict = {}
    kinds: dict = {}
    made = []
    for line in (T.DATA / "python.jsonl").open(encoding="utf-8"):
        record = json.loads(line)
        source = record.get("old_contents") or ""
        if not source or len(source) > 100_000:
            continue
        found = faults_of(source, record["old_file"])
        rng.shuffle(found)
        taken = 0
        for fault in found:
            if taken >= EACH_FILE:
                break
            row = row_of(record, fault, phrases, rng)
            if row is None or kinds.get(row["source"], 0) >=                     MOST[row["source"]]:
                continue
            taken += 1
            kinds[row["source"]] = kinds.get(row["source"], 0) + 1
            made.append(row)
            key = f"{row['source']}|{row['split']}"
            counts[key] = counts.get(key, 0) + 1
    with FAULTS.open("w", encoding="utf-8") as out:
        for row in made:
            out.write(json.dumps(row) + "\n")
    print(json.dumps(dict(sorted(counts.items()))))
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("phrase", "rows"))
    options = parser.parse_args(argv)
    if options.job == "phrase":
        phrase()
    else:
        rows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
