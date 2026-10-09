"""What the architecture knows of a project's code, for a plan (v703): which
files are joined to which, and what each file is.

A plan made from file names alone does not know that the MCP bridge reads
what the server's `/api/health` answers: the files shown a planner are
those a request names *and those joined to them* -- calling into them or
called from them (`Project.calls`), importing them -- and each is shown
with what it is: what it is about (its docstring), what it defines (most
called first), and which of the other files shown use it, or are used by
it. The same is said of the files a plan is taught from (`teach_plans`),
as the project was before each commit.
"""
from __future__ import annotations

import re
from collections import Counter

#: what a file is said with: this many of its functions, words of its about
DEFINED, ABOUT_WORDS = 6, 16


def _file_of(end: str) -> str:
    return end.split("#", 1)[0]


def links(held) -> Counter:
    """(file, file) -> how much joins them, either way: calls across files
    (`Project.calls`); imports anywhere in a file, and the calls made
    through them (`shell.run` in page.py, imported inside a function); and
    a route both say (`/api/health`: the bridge asks, the server answers)."""
    if getattr(held, "_links", None) is not None:
        return held._links
    out: Counter = Counter()
    for one in held.calls():
        mine, theirs = _file_of(one["from"]), _file_of(one["to"])
        if mine != theirs:
            out[tuple(sorted((mine, theirs)))] += 1
    for path, text in held.files.items():
        if not path.endswith(".py"):
            continue
        for found, count in _imported(held, path, text).items():
            if found != path:
                out[tuple(sorted((path, found)))] += count
    routes: dict = {}
    for path, text in held.files.items():
        for route in set(re.findall(r"[\"'`](/api/[\w/\-]+)", text)):
            routes.setdefault(route, set()).add(path)
    for paths in routes.values():
        paths = sorted(paths)
        for at, one in enumerate(paths):
            for other in paths[at + 1:]:
                out[(one, other)] += 1
    held._links = out
    return out


def _imported(held, path: str, text: str) -> Counter:
    """The project's files a Python file imports -- at its top or inside a
    function -- each with how often it is used through what was imported."""
    import ast
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return Counter()
    names: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            module = "." * node.level + node.module
            for alias in node.names:
                found = held._python_module(path, f"{module}.{alias.name}") \
                    or held._python_module(path, module)
                if found:
                    names[alias.asname or alias.name] = found
        elif isinstance(node, ast.Import):
            for alias in node.names:
                found = held._python_module(path, alias.name)
                if found:
                    names[alias.asname or alias.name.split(".")[0]] = found
    out: Counter = Counter({found: 1 for found in names.values()})
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value,
                                                          ast.Name):
            found = names.get(node.value.id)
            if found:
                out[found] += 1
    return out


def linked(held, seeds: list, most: int) -> list:
    """The files joined to the seeds, most joined first -- not the seeds."""
    joined: Counter = Counter()
    seeds_ = set(seeds)
    for (one, other), count in links(held).items():
        if one in seeds_ and other not in seeds_:
            joined[other] += count
        elif other in seeds_ and one not in seeds_:
            joined[one] += count
    return [path for path, _ in joined.most_common(most)]


def summary(held, path: str, shown: list) -> str:
    """What a file is, in a line: what it is about, what it defines, which
    of the files shown use it and which it uses."""
    read = held.outline().get(path) or {}
    parts = []
    about = (read.get("about") or "").strip().split("\n\n")[0]
    if about:
        words = " ".join(about.split()).split()
        parts.append(" ".join(words[:ABOUT_WORDS]) +
                     (" ..." if len(words) > ABOUT_WORDS else ""))
    called: Counter = Counter()
    for one in held.calls():
        if _file_of(one["to"]) == path:
            called[one["to"].split("#", 1)[1]] += 1
    names = [one["name"] for one in read.get("functions", ())
             if not one["name"].startswith("_") or called[one["name"]]]
    names.sort(key=lambda one: -called[one])
    if names:
        parts.append("defines " + ", ".join(names[:DEFINED]) +
                     (" ..." if len(names) > DEFINED else ""))
    # the files shown it is joined to: by calls, imports, a route both say
    joined = sorted({other for pair in links(held) if path in pair
                     for other in pair if other != path and other in shown})
    if joined:
        parts.append("joined to " + ", ".join(joined))
    return "; ".join(parts)


def subset(files: dict, root=None):
    """A project of just these files: what a plan is shown is said of them
    alone, at run time as when it was taught (as each commit found them)."""
    from research.v698.project import Project
    one = Project("shown", root)
    one.files = dict(files)
    return one


def shown(held, paths: list) -> str:
    """The files as a planner is shown them: each with what it is."""
    out = []
    for path in paths:
        said = summary(held, path, paths)
        out.append(f"{path} -- {said}" if said else path)
    return "\n".join(out)
