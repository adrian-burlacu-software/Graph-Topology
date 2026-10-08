"""A project, read whole, in a conversation (`PLAN.md`).

The editor sends a workspace's files (`/api/project`, then each save
`/api/file`); the conversation it belongs to holds them as a `Project`:

    outline       each file's functions with their lines and types, what
                  each calls by name, what each file imports
                  (`tscheck.js` `outline`: syntax, so a file that does not
                  compile is still read)
    calls         who calls whom across files: a name resolved through the
                  file's own functions, then its imports
    diagnostics   what the compiler says is wrong, file by file (asked for
                  when wanted: it type-checks the whole project)

Paths are the workspace's, relative (`src/app.ts`); the checker is given
them under one root (`/ws/src/app.ts`) so imports between them resolve.

A project is TypeScript and JavaScript, Python, or both (v699): each file
is read by its language's checker (`tscheck.js`, `pycheck.py`), and what
they say is put together -- a Python file's imports resolve as Python's
do (`from .b import x`, `import pkg.mod`, a package's `__init__.py`).
"""
from __future__ import annotations

import posixpath
import threading
import time

ROOT = "/ws/"
#: what is read: TypeScript and JavaScript
TS_READ = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts")
#: and Python (v699)
PY_READ = (".py",)
READ = TS_READ + PY_READ


def language_of(path: str) -> str:
    """A file's language, by its name."""
    return "python" if path.endswith(PY_READ) else "typescript"
#: what a project may hold, at most
MOST_FILES = 2000
MOST_BYTES = 30_000_000

#: the runtime's globals, for the compiler's diagnostics: a project's own
#: declarations (`@types/node`) are not sent, so these stand in for them
AMBIENT_FILE = "__runtime__.d.ts"
AMBIENT = """declare var console: { log(...a: any[]): void; error(...a: any[]): void;
  warn(...a: any[]): void; info(...a: any[]): void; debug(...a: any[]): void;
  table(...a: any[]): void };
declare var process: any;
declare var require: any;
declare var module: any;
declare var exports: any;
declare var __dirname: string;
declare var __filename: string;
declare function setTimeout(f: (...a: any[]) => void, ms?: number, ...a: any[]): any;
declare function clearTimeout(id: any): void;
declare function setInterval(f: (...a: any[]) => void, ms?: number, ...a: any[]): any;
declare function clearInterval(id: any): void;
"""

PROJECTS: dict = {}
_LOCK = threading.Lock()


def _inside(path: str) -> str:
    return ROOT + path.lstrip("/")


class Project:
    def __init__(self, name: str = "", root: str | None = None) -> None:
        self.name = name
        self.files: dict = {}
        #: where its files are on disk (v700), where the editor said: a
        #: change asked of the project is made there, not handed back
        self.root = root
        self.read_at = 0.0
        self._outline: dict | None = None
        self._diagnostics: list | None = None

    # -- what it holds ------------------------------------------------------
    def put(self, files: dict) -> dict:
        """Files added or replaced: {path: text}. What was refused, why."""
        refused = {}
        for path, text in files.items():
            path = path.replace("\\", "/").lstrip("/")
            if not path.endswith(READ):
                refused[path] = "not TypeScript, JavaScript or Python"
                continue
            if text is None:
                self.files.pop(path, None)
                continue
            if len(self.files) >= MOST_FILES and path not in self.files:
                refused[path] = f"the project holds {MOST_FILES} files"
                continue
            self.files[path] = text
        if sum(map(len, self.files.values())) > MOST_BYTES:
            raise ValueError(f"the project is over {MOST_BYTES} bytes")
        self._outline = self._diagnostics = None
        self.read_at = time.time()
        return refused

    def on_disk(self, path: str):
        """Where one of its files is on disk, or None: no root said, or a
        path that would leave it."""
        from pathlib import Path
        if not self.root:
            return None
        root = Path(self.root).resolve()
        found = (root / path.lstrip("/")).resolve()
        return found if found.is_relative_to(root) else None

    def write(self, path: str, text: str) -> bool:
        """One of its files changed, here and on disk (where it has a
        root): whether it was written there."""
        path = path.replace("\\", "/").lstrip("/")
        if path not in self.files:
            raise ValueError(f"{path} is not a file of the project")
        self.put({path: text})
        found = self.on_disk(path)
        if found is None:
            return False
        found.write_text(text, encoding="utf-8", newline="")
        return True

    def checked(self, language: str | None = None) -> dict:
        """The files under the checker's root -- of one language, if
        given."""
        return {_inside(path): text for path, text in self.files.items()
                if language is None or language_of(path) == language}

    def languages(self) -> list:
        """The languages it is written in, the most files' first."""
        count: dict = {}
        for path in self.files:
            count[language_of(path)] = count.get(language_of(path), 0) + 1
        return sorted(count, key=lambda one: -count[one])

    def language(self) -> str | None:
        """What it is mostly written in, if anything."""
        found = self.languages()
        return found[0] if found else None

    # -- what it is -------------------------------------------------------------
    def outline(self) -> dict:
        if self._outline is None:
            from research.v696.checker import checker
            self._outline = {}
            for language in self.languages():
                read = checker(language).outline(self.checked(language))
                self._outline.update({path[len(ROOT):]: one
                                      for path, one in read.items()})
        return self._outline

    def functions(self) -> list:
        """Every function, with its file: [{file, name, ...}]."""
        return [{"file": path, **one} for path, read in
                sorted(self.outline().items())
                for one in read["functions"]]

    def find(self, name: str) -> list:
        """The functions called `name` (or `Class.name`, or a method
        `name` of any class)."""
        return [one for one in self.functions()
                if one["name"] == name or one["name"].endswith("." + name)]

    def _resolved(self, path: str, called: str) -> list:
        """What `called` in `path` names: one of its own functions, else
        an imported one, followed to its file."""
        read = self.outline().get(path) or {}
        own = [one for one in read.get("functions", ())
               if one["name"] == called]
        if own:
            return [{"file": path, **own[0]}]
        python = language_of(path) == "python"
        for imported in read.get("imports", ()):
            if called not in imported["names"]:
                continue
            if python:
                found = self._python_module(path, imported["from"])
                if found is not None:
                    return [{"file": found, **one}
                            for one in self.outline()[found]["functions"]
                            if one["name"] == called]
                continue
            if not imported["from"].startswith("."):
                continue
            base = posixpath.normpath(posixpath.join(
                posixpath.dirname(path), imported["from"]))
            for ending in ("", *READ, *(f"/index{one}" for one in READ)):
                target = self.outline().get(base + ending)
                if target is None:
                    continue
                return [{"file": base + ending, **one}
                        for one in target["functions"]
                        if one["name"] == called]
        return []

    def _python_module(self, path: str, module: str) -> str | None:
        """The project's file a Python import names: `.b` beside the
        importer, `..pkg.mod` above it, `pkg.mod` from the root (or from
        the importer's top folder), a package's `__init__.py`."""
        dots = len(module) - len(module.lstrip("."))
        parts = [one for one in module.lstrip(".").split(".") if one]
        if dots:
            base = posixpath.dirname(path)
            for _ in range(dots - 1):
                base = posixpath.dirname(base)
            starts = [base]
        else:
            top = path.split("/")[0] if "/" in path else ""
            starts = ["", top, posixpath.dirname(path)]
        for start in starts:
            stem = posixpath.join(start, *parts) if parts else start
            for candidate in (stem + ".py", posixpath.join(stem,
                                                           "__init__.py")):
                candidate = candidate.lstrip("/")
                if candidate in self.outline():
                    return candidate
        return None

    def calls(self) -> list:
        """Who calls whom: [{from, to, call}] between the project's own
        functions, `file#name` at each end."""
        out = []
        for one in self.functions():
            for called in one["calls"]:
                name = called.split(".")[-1] if called.startswith(
                    ("this.", "self.")) else called
                for target in self._resolved(one["file"], name):
                    out.append({"from": f"{one['file']}#{one['name']}",
                                "to": f"{target['file']}#{target['name']}",
                                "call": called})
        return out

    def callers(self, name: str) -> list:
        ends = {f"{one['file']}#{one['name']}" for one in self.find(name)}
        return [one for one in self.calls() if one["to"] in ends]

    def diagnostics(self, strict: bool = False) -> list:
        """What the compiler says is wrong, each with its file and line;
        `strict`: also what finds bugs (`tscheck.js` `STRICTER`), each
        marked `strict` where the plain compile does not say it."""
        if self._diagnostics is None:
            self._diagnostics = {}
        if strict not in self._diagnostics:
            from research.v696.checker import checker
            # what every runtime a project runs in has, said once: without
            # it, `console.log` is an error in every file that prints
            found = []
            for language in self.languages():
                checked = self.checked(language)
                if language == "typescript":
                    checked[ROOT + AMBIENT_FILE] = AMBIENT
                found += [one for one in checker(language).diagnose(
                    checked, strict) if not one["file"].endswith(
                        AMBIENT_FILE)]
            for one in found:
                one["file"] = one["file"][len(ROOT):] if one["file"].startswith(
                    ROOT) else one["file"]
                text = self.files.get(one["file"], "")
                one["line"] = text.count("\n", 0, one.get("start", 0)) + 1
            if strict:
                plain = {(one["file"], one["start"], one["code"])
                         for one in self.diagnostics()}
                for one in found:
                    one["strict"] = (one["file"], one["start"],
                                     one["code"]) not in plain
            self._diagnostics[strict] = found
        return self._diagnostics[strict]

    def at(self, path: str, line: int) -> dict | None:
        """The function a line of a file is in."""
        return next((one for one in (self.outline().get(path) or {}).get(
            "functions", ()) if one["start"] <= line <= one["end"]), None)

    def summary(self) -> dict:
        functions = self.functions()
        called = {}
        for one in self.calls():
            called[one["to"]] = called.get(one["to"], 0) + 1
        return {"name": self.name, "files": len(self.files),
                "languages": self.languages(),
                "lines": sum(read["lines"] for read in
                             self.outline().values()),
                "functions": len(functions),
                "exported": sum(one["exported"] for one in functions),
                "most called": sorted(called.items(),
                                      key=lambda one: -one[1])[:8],
                "read at": self.read_at}

    def graph(self) -> dict:
        """The call graph, as the page draws graphs: each function a node,
        each call an edge (`via` project)."""
        nodes = [{"id": f"{one['file']}#{one['name']}", "label": one["name"],
                  "via": ["project"], "file": one["file"],
                  "line": one["start"]} for one in self.functions()]
        edges = [{"from": one["from"], "to": one["to"], "relation": "calls",
                  "via": "project"} for one in self.calls()]
        return {"nodes": nodes, "edges": edges}


def project(key) -> Project | None:
    with _LOCK:
        return PROJECTS.get(key)


def put(key, files: dict, name: str = "", whole: bool = False,
        root: str | None = None) -> dict:
    """The files of a conversation's project: all of them (`whole`), or
    some changed (`None` to remove one); `root`, where they are on disk."""
    with _LOCK:
        found = PROJECTS.get(key)
        if found is None or whole:
            found = PROJECTS[key] = Project(name, root)
        refused = found.put(files)
        return {"refused": refused, **found.summary()}
