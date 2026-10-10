"""Teaching the editor its part of a larger change: a request said whole,
one function of the several it changes, and what the project has.

    python -m research.v703.teach_parts fetch     data/commit-repos/*.git
    python -m research.v703.teach_parts corpus    llm/plan-data/parts.jsonl
    python -m research.v703.teach_parts mix       llm/plan-data/parts-mix.jsonl
    python -m research.v703.teach_parts train     llm/editor6

Asked to "report whether the shell asks in /api/health and show it in the
MCP health tool", the plan chose the server's `Handler.do_GET`, and the
editor -- taught commits of one file, whose message is all of that file's
change -- wrote `"show": "/api/health"` into the health answer: the
request's words, for it was not told what in the request was its part, nor
that the project has `shell.ASK`, in a file it was not shown.

**What it is taught from**: commits that change two files or more -- this
project's, and those of Python projects whose whole history is cloned
(`fetch`: permissive licences, `data/commit-repos.SOURCE.md`), each file
whole at every commit, so a function is shown whole as it is at run time.
**Each row, by construction**: for each function a commit changes, the
commit's message said of it (`in path, Class.method: message`, as a plan's
step says it), the other functions the commit changes (`also changed`),
what the project has that the change could use -- the definitions it
does use (in its own file, in what the file imports, added by the commit
elsewhere), among others of the same files -- and the function; the
answer, that function's changes only.

**At run time** (`context`): the plan's other steps, and the definitions of
the files the plan was shown, ranked by the picker with the request; where
the change uses a module the project has and the file does not import, the
import is added by lookup (`fixing`), not written by the editor.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import json
import random
import re
import subprocess
import sys

from research.v700 import teach_editor as T
from research.v703 import teach_plans as P

REPOS_DIR = P.ROOT / "data" / "commit-repos"
#: Python projects whose whole history is taught: permissive licences
#: (MIT, BSD, Apache-2.0); history as it was before this date
REPOS = ("pallets/flask", "pallets/click", "pallets/jinja",
         "pallets/werkzeug", "psf/requests", "encode/httpx",
         "encode/starlette", "Textualize/rich", "python-attrs/attrs",
         "fastapi/fastapi", "fastapi/typer", "psf/black", "PyCQA/flake8",
         "pytest-dev/pytest", "marshmallow-code/marshmallow", "tox-dev/tox",
         "pypa/hatch", "httpie/cli", "scrapy/scrapy", "aio-libs/aiohttp",
         "python-poetry/poetry", "pypa/pip")
UNTIL = "2026-10-01"
PARTS = P.DATA / "parts.jsonl"
MIX = P.DATA / "parts-mix.jsonl"
SEED = 705
#: a commit taught: files changed, at least and at most
LEAST_FILES, MOST_FILES = 2, 8
#: what the project has, shown: at most, and at most of it used
MOST_UNITS, MOST_USED = 8, 4
#: the other functions changed, said at most
MOST_OTHERS = 4
#: how alike a file must stay to be taught as changed, not written again
SAME_AT_LEAST = 0.5
#: what the judge of uses is taught of a row besides what it used; and
#: of a row that uses nothing, how many
CANDIDATES, UNUSED = 24, 2
USES_PAIRS = P.DATA / "uses-{}.jsonl"
USES = P.LLM / "uses"
#: rows of a project at most (pip's history is not the rest's); of the
#: editor's earlier teaching, kept beside them
MOST_PER_REPO, KEPT = 5000, 12000


# -- the repositories -----------------------------------------------------------------

def fetch() -> None:
    """Each project's history, bare, as published."""
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    for name in REPOS:
        out = REPOS_DIR / (name.replace("/", "_") + ".git")
        if not out.exists():
            subprocess.run(["git", "clone", "-q", "--bare",
                            f"https://github.com/{name}.git", str(out)])
        print(out.name, out.exists(), flush=True)


class Repo:
    """A repository's commits and files, read through one `git cat-file`."""

    def __init__(self, git_dir, key: str):
        self.args = ["git", f"--git-dir={git_dir}"]
        self.key = key
        self.batch = subprocess.Popen(self.args + ["cat-file", "--batch"],
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE)

    def git(self, *args) -> str:
        return subprocess.run(self.args + list(args), capture_output=True,
                              encoding="utf-8", errors="replace").stdout or ""

    def blob(self, commit: str, path: str) -> str | None:
        self.batch.stdin.write(f"{commit}:{path}\n".encode())
        self.batch.stdin.flush()
        head = self.batch.stdout.readline().split()
        if len(head) < 3 or head[1] != b"blob":
            return None
        data = self.batch.stdout.read(int(head[2]) + 1)[:-1]
        return data.decode("utf-8", errors="replace")

    def close(self) -> None:
        self.batch.stdin.close()
        self.batch.wait()

    def commits(self):
        """(commit, parent, message, [(status, path)]) of the commits that
        change several files, newest first."""
        top = self.git("rev-list", "-1", f"--before={UNTIL}", "HEAD").strip()
        if not top:
            return
        listed = self.git("log", "--no-merges", "--format=%x1e%H %P%n%B%x1f",
                          "--name-status", "--no-renames", top)
        for record in listed.split("\x1e")[1:]:
            head, _, rest = record.partition("\n")
            message, _, names = rest.partition("\x1f")
            ids = head.split()
            if len(ids) != 2:
                continue
            changed = [tuple(line.split("\t", 1)) for line in
                       names.strip().splitlines() if "\t" in line]
            if LEAST_FILES <= len(changed) <= MOST_FILES:
                yield ids[0], ids[1], message.strip(), changed


# -- a file read: its functions, its definitions, its imports --------------------------

def functions(source: str) -> list:
    """(qualified name, first line, end, indent) 0-based, end exclusive, of
    every function and method -- decorators included -- innermost last."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    out = []

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = min([child.lineno] + [one.lineno for one in
                                              child.decorator_list]) - 1
                name = prefix + child.name
                out.append((name, start, child.end_lineno, child.col_offset))
                walk(child, name + ".")
            elif isinstance(child, ast.ClassDef):
                walk(child, prefix + child.name + ".")
    walk(tree, "")
    return out


def definitions(source: str) -> list:
    """(name, the line that says it) of what a module defines at its top:
    functions, classes, names assigned."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    lines = source.splitlines()
    out = []
    for node in tree.body:
        line = lines[node.lineno - 1].strip() if node.lineno <= len(lines) \
            else ""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            out.append((node.name, line))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) \
                else [node.target]
            for one in targets:
                if isinstance(one, ast.Name):
                    out.append((one.id, line))
    return [(name, said[:100]) for name, said in out]


def module_path(path: str, module: str, paths: set, level: int = 0):
    """The file of the project a module names, from `path`: `a.b` is
    a/b.py or a/b/__init__.py, at the project's top or beside a src/."""
    if level:
        base = path.split("/")[:-level]
        parts = base + ([one for one in module.split(".") if one]
                        if module else [])
        stems = ["/".join(parts)]
    else:
        parts = module.split(".")
        stems = ["/".join(parts), "src/" + "/".join(parts)]
    for stem in stems:
        for found in (stem + ".py", stem + "/__init__.py"):
            if found in paths:
                return found
    return None


def imports(source: str, path: str, paths: set) -> dict:
    """What a file imports of the project: alias -> (file, name or None
    for the module itself)."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                as_module = module_path(path, f"{node.module or ''}."
                                        f"{alias.name}".strip("."), paths,
                                        node.level)
                if as_module:
                    out[alias.asname or alias.name] = (as_module, None)
                    continue
                found = module_path(path, node.module or "", paths,
                                    node.level)
                if found:
                    out[alias.asname or alias.name] = (found, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                found = module_path(path, alias.name, paths)
                if found and alias.asname:
                    out[alias.asname] = (found, None)
                elif found and "." not in alias.name:
                    out[alias.name] = (found, None)
    return out


def importable(path: str) -> bool:
    """Whether a file can be imported as a module: each part of its path a
    name."""
    return all(one.isidentifier() for one in path[:-3].split("/"))


def module_of(path: str) -> str:
    """How a file's module is said: research/v702/shell.py is shell."""
    stem = path[:-3] if path.endswith(".py") else path
    stem = stem[:-len("/__init__")] if stem.endswith("/__init__") else stem
    return stem.rsplit("/", 1)[-1]


def import_line(path: str) -> str:
    """The line that imports a file of the project as a module, as this
    project's code does: `from research.v702 import shell`."""
    stem = path[:-3]
    stem = stem[:-len("/__init__")] if stem.endswith("/__init__") else stem
    package, _, name = stem.rpartition("/")
    return f"from {package.replace('/', '.')} import {name}" if package \
        else f"import {name}"


def units_of(path: str, source: str, files: dict, paths: set) -> list:
    """What the project has that code in `path` may use, as it would say
    it: (how it is said, the file, the line that defines it) -- its own
    file's (`_reply`), what it imports (`shell.ASK`, `Project`), and the
    other files given, as their module (`structure.links`)."""
    out, seen = [], set()
    imported = imports(source, path, paths)
    by_file: dict = {}
    for alias, (found, name) in imported.items():
        by_file.setdefault(found, []).append((alias, name))
    for other, text in files.items():
        # a file no code imports (`tools/graph-topology-mcp/server.py`: a
        # program of its own) has nothing another file may use
        if not other.endswith(".py") or other != path and \
                not importable(other):
            continue
        for name, line in definitions(text):
            if other == path:
                said = name
            else:
                said = None
                for alias, imported_name in by_file.get(other, ()):
                    if imported_name == name:
                        said = alias
                    elif imported_name is None and said is None:
                        said = f"{alias}.{name}"
                said = said or f"{module_of(other)}.{name}"
            if said not in seen:
                seen.add(said)
                out.append((said, other, line))
    return out


def unit_line(unit) -> str:
    said, path, line = unit
    return f"{said} -- {path}: {line}"


def asked(statement: str, others: list, units: list) -> str:
    """The statement as the editor is given it: the request said of one
    function, what else the change is, what the project has."""
    out = statement.strip()
    if others:
        out += "\nalso changed: " + "; ".join(others[:MOST_OTHERS])
    if units:
        out += "\nthe project has:\n" + "\n".join(unit_line(one)
                                                 for one in units)
    return out


# -- a commit, as rows ---------------------------------------------------------------

def _message(text: str) -> str:
    first = re.split(r"\n\s*\n", text.strip())[0]
    first = re.sub(r"\s+", " ", first).strip()
    return first[:T.LONGEST_MESSAGE]


def _owner(spans: list, group, new: list) -> tuple | None:
    """The innermost function a change of lines is in: what it takes out
    inside it, or what it adds inside it -- at its end, indented under it."""
    tag, i1, i2, j1, j2 = group
    best = None
    for one in spans:
        name, start, end, indent = one
        if i1 < i2:
            inside = start < i1 and i2 <= end
        else:
            added = [line for line in new[j1:j2] if line.strip()]
            deeper = bool(added) and len(added[0]) - len(added[0].lstrip()) \
                > indent
            inside = start < i1 < end or (i1 == end and deeper)
        if inside and (best is None or end - start < best[2] - best[1]):
            best = one
    return best


def _blocks(old: list, new: list, groups: list, offset: int, end: int):
    """The groups' changes as blocks found once in old[offset:end] --
    widened by the lines around them until they are (`teach_editor`)."""
    part = old[offset:end]
    out = []
    for tag, i1, i2, j1, j2 in groups:
        a, b = i1 - offset, i2 - offset
        before = after = 0
        if a == b:
            if a > 0:
                before = 1
            elif b < len(part):
                after = 1
            else:
                return None
        while not T._unique(part, a - before, b + after) and (
                a - before > 0 or b + after < len(part)):
            if a - before > 0:
                before += 1
            if not T._unique(part, a - before, b + after) and \
                    b + after < len(part):
                after += 1
        if not T._unique(part, a - before, b + after):
            return None
        out.append((part[a - before:b + after],
                    part[a - before:a] + new[j1:j2] + part[b:b + after]))
    return out


_NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?")


def rows_of(repo: Repo, commit: str, parent: str, message: str,
            changed: list, rng, paths_of) -> list:
    statement = _message(message)
    if len(statement.split()) < 4:
        return []
    modified = [path for status, path in changed
                if status == "M" and path.endswith(".py")]
    if not modified:
        return []
    olds, news = {}, {}
    for status, path in changed:
        if not path.endswith(".py"):
            continue
        if status != "A":
            olds[path] = repo.blob(parent, path)
        if status != "D":
            news[path] = repo.blob(commit, path)
    # each function changed, with its changes
    found = []
    for path in modified:
        old, new = olds.get(path), news.get(path)
        if not old or new is None or len(old) > 200_000:
            continue
        a, b = T._lines(old), T._lines(new)
        spans = functions(old)
        groups: dict = {}
        matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
        if matcher.ratio() < SAME_AT_LEAST:
            # a file written again: what its diff pairs is not a change
            continue
        for group in matcher.get_opcodes():
            if group[0] == "equal":
                continue
            owner = _owner(spans, group, b)
            groups.setdefault(owner, []).append(group)
        for owner, mine in groups.items():
            found.append((path, owner, mine, a, b))
    if not found:
        return []
    # the other functions changed, as a plan's other steps are said
    said_of = [f"{path} {owner[0]}" if owner else None
               for path, owner, *_ in found]
    paths = None
    rows = []
    for at, (path, owner, mine, a, b) in enumerate(found):
        if owner is None:
            continue
        name, start, end, _ = owner
        size = sum((i2 - i1) + (j2 - j1) for _, i1, i2, j1, j2 in mine)
        if end - start > T.LONGEST_PART or size > T.LONGEST_CHANGE:
            continue
        blocks = _blocks(a, b, mine, start, end)
        if not blocks:
            continue
        part = "".join(a[start:end])
        target = T.said(blocks)
        if T.applied(part, target) is None:
            continue
        # what the project has: the file itself, what it imports, the
        # commit's other files -- as they are after it
        if paths is None:
            paths = paths_of(commit)
        new_text = news[path]
        files = {path: new_text}
        files.update({other: text for other, text in news.items()
                      if text and other != path})
        for found_path, _ in imports(new_text, path, paths).values():
            if found_path not in files:
                text = repo.blob(commit, found_path)
                if text and len(text) < 200_000:
                    files[found_path] = text
        units = [one for one in units_of(path, new_text, files, paths)
                 if one[0].split(".")[-1] != name.split(".")[-1]]
        added = "".join(line for _, new_lines in blocks
                        for line in new_lines)
        written = set(_NAME.findall(added))
        before = set(_NAME.findall(part))
        used = [one for one in units if one[0] in written and
                one[0] not in before]
        if len(used) > MOST_USED:
            continue
        others = [one for one in units if one not in used]
        # beside what it uses, others of the same files first
        near = [one for one in others if one[1] in {u[1] for u in used}]
        rest = [one for one in others if one not in near]
        rng.shuffle(near)
        rng.shuffle(rest)
        room = MOST_UNITS - len(used)
        if not used and rng.random() < 0.3:
            room = 0
        shown = used + (near[:room // 2] + rest)[:room]
        rng.shuffle(shown)
        rows.append({
            "commit": f"{repo.key}:{commit}", "split": P.split_of(
                f"{repo.key}:{commit}"),
            "language": "python", "source": "part " + repo.key,
            "statement": asked(f"in {path}, {name}: {statement}",
                               list(dict.fromkeys(
                                   one for k, one in enumerate(said_of)
                                   if k != at and one)), shown),
            "path": path, "part": part, "start": start + 1,
            "target": target, "used": [one[0] for one in used],
            # what the judge of uses is taught from: what it used, and
            # what else it could have
            "candidates": [unit_line(one) for one in used] +
            [unit_line(one) for one in rng.sample(
                others, min(len(others), CANDIDATES))]})
    return rows


def corpus() -> dict:
    rng = random.Random(SEED)
    repos = [(P.ROOT / ".git", "ours")] + [
        (REPOS_DIR / (name.replace("/", "_") + ".git"), name)
        for name in REPOS]
    counts = {}
    with PARTS.open("w", encoding="utf-8") as out:
        for git_dir, key in repos:
            if not git_dir.exists():
                print(f"{key}: not fetched", flush=True)
                continue
            repo = Repo(git_dir, key)
            listing: dict = {}

            def paths_of(commit, repo=repo, listing=listing):
                if commit not in listing:
                    listing.clear()
                    listing[commit] = set(repo.git(
                        "ls-tree", "-r", "--name-only", commit).split("\n"))
                return listing[commit]
            made = 0
            for commit, parent, message, changed in repo.commits():
                try:
                    rows = rows_of(repo, commit, parent, message, changed,
                                   rng, paths_of)
                except RecursionError:
                    continue
                for row in rows:
                    out.write(json.dumps(row) + "\n")
                made += len(rows)
                if made >= MOST_PER_REPO:
                    break
            repo.close()
            counts[key] = made
            print(f"{key}: {made}", flush=True)
    print(json.dumps(counts))
    return counts


def mix(seed: int = SEED) -> dict:
    """The parts, and `KEPT` of what the editor was taught before (its
    mix: commits of one file, faults, data), so it keeps them."""
    rng = random.Random(seed)
    parts = [json.loads(line) for line in PARTS.open(encoding="utf-8")]
    before = [json.loads(line) for line in T.MIX.open(encoding="utf-8")]
    kept = [one for one in before if one["split"] == "train"]
    out = parts + rng.sample(kept, min(KEPT, len(kept)))
    rng.shuffle(out)
    with MIX.open("w", encoding="utf-8") as stream:
        for one in out:
            stream.write(json.dumps(one) + "\n")
    counts = {"parts": len(parts), "kept": min(KEPT, len(kept)),
              "train": sum(one["split"] == "train" for one in out)}
    print(json.dumps(counts))
    return counts


def uses_job() -> dict:
    """(statement, a definition, whether the change used it) -- what it
    used, beside as many it did not, and two of a row that used nothing."""
    rng = random.Random(SEED)
    found: dict = {"train": [], "dev": [], "test": []}
    for line in PARTS.open(encoding="utf-8"):
        row = json.loads(line)
        request = row["statement"].split("\n")[0]
        used = row["candidates"][:len(row["used"])]
        unused = row["candidates"][len(row["used"]):]
        rng.shuffle(unused)
        for one in used:
            found[row["split"]].append((request, one, 1.0, row["commit"]))
        for one in unused[:max(len(used) * 4, UNUSED)]:
            found[row["split"]].append((request, one, 0.0, row["commit"]))
    counts = {}
    for split, rows in found.items():
        rng.shuffle(rows)
        with open(str(USES_PAIRS).format(split), "w",
                  encoding="utf-8") as out:
            for one in rows:
                out.write(json.dumps(one) + "\n")
        counts[split] = {"pairs": len(rows),
                         "used": sum(one[2] for one in rows)}
    print(json.dumps(counts))
    return counts


def uses_pairs(split: str, seed: int = SEED) -> list:
    return [tuple(json.loads(line)) for line in
            open(str(USES_PAIRS).format(split), encoding="utf-8")]


def train_uses(epochs: int = 2) -> None:
    from research.v700 import teach_judge as J
    J.pairs = uses_pairs
    J.train(USES, epochs=epochs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("fetch", "corpus", "uses",
                                        "train-uses", "mix", "train",
                                        "measure"))
    parser.add_argument("--base", default="editor5")
    parser.add_argument("--out", default="editor6")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--most", type=int, default=300)
    options = parser.parse_args(argv)
    if options.job == "fetch":
        fetch()
    elif options.job == "corpus":
        corpus()
    elif options.job == "uses":
        uses_job()
    elif options.job == "train-uses":
        train_uses()
    elif options.job == "mix":
        mix()
    elif options.job == "train":
        T.train(T.LLM / options.out, epochs=options.epochs,
                base=T.LLM / options.base, corpus=MIX)
    else:
        T.measure(T.LLM / options.out, most=options.most, corpus=PARTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
