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
"""
from __future__ import annotations

import posixpath
import threading
import time

ROOT = "/ws/"
#: what is read: TypeScript and JavaScript
READ = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts")
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
    def __init__(self, name: str = "") -> None:
        self.name = name
        self.files: dict = {}
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
                refused[path] = "not TypeScript or JavaScript"
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

    def checked(self) -> dict:
        return {_inside(path): text for path, text in self.files.items()}

    # -- what it is -------------------------------------------------------------
    def outline(self) -> dict:
        if self._outline is None:
            from research.v696.checker import checker
            read = checker().outline(self.checked())
            self._outline = {path[len(ROOT):]: one
                             for path, one in read.items()}
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
        for imported in read.get("imports", ()):
            if called not in imported["names"] or \
                    not imported["from"].startswith("."):
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

    def calls(self) -> list:
        """Who calls whom: [{from, to, call}] between the project's own
        functions, `file#name` at each end."""
        out = []
        for one in self.functions():
            for called in one["calls"]:
                name = called.split(".")[-1] if called.startswith(
                    "this.") else called
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
            checked = {**self.checked(), ROOT + AMBIENT_FILE: AMBIENT}
            found = [one for one in checker().diagnose(checked, strict)
                     if not one["file"].endswith(AMBIENT_FILE)]
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


def put(key, files: dict, name: str = "", whole: bool = False) -> dict:
    """The files of a conversation's project: all of them (`whole`), or
    some changed (`None` to remove one)."""
    with _LOCK:
        found = PROJECTS.get(key)
        if found is None or whole:
            found = PROJECTS[key] = Project(name)
        refused = found.put(files)
        return {"refused": refused, **found.summary()}
