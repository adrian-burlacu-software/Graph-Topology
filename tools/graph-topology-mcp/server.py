"""An MCP server (stdio) that puts the cognitive architecture's server in
front of an MCP client: Claude asks it for code, asks about code, and has it
change files.

    python tools/graph-topology-mcp/server.py [--url http://127.0.0.1:8697]

It is a bridge, nothing more: every tool is a call on the v698 server's HTTP
API (`research/v698/protocol.md`) -- the architecture reads, searches and
writes; this only carries the words there and the answer back, and (for
`change_file`, when asked to) puts the answer into the file.

Standard library only: JSON-RPC 2.0, one message per line on stdin/stdout.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

NAME = "graph-topology"
VERSION = "0.1.0"
#: a request for code may take minutes: writers, then search
LONG = 15 * 60
READ = re.compile(r"\.(ts|tsx|js|jsx|mjs|cjs|mts|cts|py)$", re.I)
SKIPPED = ("node_modules/", "dist/", "build/", "out/", ".git/",
           "__pycache__/", ".venv/", "venv/", "site-packages/")

URL = os.environ.get("GRAPH_TOPOLOGY_URL", "http://127.0.0.1:8697")
ROOT = Path(os.environ.get("GRAPH_TOPOLOGY_ROOT") or os.getcwd()).resolve()


def _sid(given: str | None) -> str:
    """One conversation per workspace, unless one is named."""
    if given:
        return str(given)[:64]
    return "mcp-" + hashlib.sha1(str(ROOT).lower().encode()).hexdigest()[:12]


class Down(RuntimeError):
    """No server answered."""


def _raw(path: str, body: dict | None = None, timeout: float = 10) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        URL.rstrip("/") + path, data=data, method="GET" if body is None
        else "POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as answer:
            return json.loads(answer.read() or b"{}")
    except urllib.error.HTTPError as bad:
        try:
            found = json.loads(bad.read() or b"{}")
        except ValueError:
            found = {}
        raise RuntimeError(found.get("error") or f"HTTP {bad.code}")
    except (urllib.error.URLError, OSError) as bad:
        raise Down(f"no architecture server at {URL} ({bad})")


# -- the server restarted, stopped, or never started ----------------------------
#
# As the editor does (`protocol.md`): a changed `instance` is a restart, and
# what the server held -- the projects sent -- is sent again. A server that
# is down is started (in this workspace, where `research/v698` is) and
# waited for; a call cut off by a restart is made again.

#: how long a server may take to load its engines and models
STARTUP = float(os.environ.get("GRAPH_TOPOLOGY_STARTUP", 600))
WORKERS = os.environ.get("GRAPH_TOPOLOGY_WORKERS", "19")
#: the instance last seen, and what each conversation was sent
SEEN: dict = {"instance": None}
SENT: dict = {}
STARTED: dict = {"process": None}


def _start() -> bool:
    """Start the server, where this workspace is the architecture's."""
    import subprocess
    import tempfile
    import urllib.parse
    if not (ROOT / "research" / "v698").is_dir() or \
            os.environ.get("GRAPH_TOPOLOGY_START", "1") == "0":
        return False
    running = STARTED["process"]
    if running is not None and running.poll() is None:
        return True                         # started, still loading
    port = str(urllib.parse.urlparse(URL).port or 8697)
    log = Path(tempfile.gettempdir()) / "graph-topology-server.log"
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | \
            subprocess.CREATE_NEW_PROCESS_GROUP
    with log.open("ab") as out:
        STARTED["process"] = subprocess.Popen(
            [sys.executable, "-m", "research.v698", "--workers", WORKERS,
             "--port", port], cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=out, stderr=out, creationflags=flags)
    return True


def _health(wait: bool = True) -> dict:
    """The server's health, starting it and waiting for it if it is down;
    a restart sends again what was sent."""
    import time
    try:
        found = _raw("/api/health", timeout=5)
    except Down:
        if not wait or not _start():
            raise Down(f"no architecture server at {URL}; start it with "
                       "`python -m research.v698 --workers 19 --port "
                       "8697`") from None
        until = time.time() + STARTUP
        while True:
            time.sleep(3)
            try:
                found = _raw("/api/health", timeout=5)
                break
            except Down:
                if time.time() > until:
                    raise Down(f"the server at {URL} did not come up in "
                               f"{STARTUP:.0f} s") from None
    if SEEN["instance"] not in (None, found.get("instance")):
        for sid, (files, name) in list(SENT.items()):
            _raw("/api/project", {"sid": sid, "files": files, "name": name},
                 timeout=LONG)
        found["sent again"] = sorted(SENT)
    SEEN["instance"] = found.get("instance")
    return found


def _call(path: str, body: dict | None = None, timeout: float = 10) -> dict:
    """A call on the server, as it is now: up, and holding what was sent.
    A call cut off by a restart is made once more."""
    _health()
    try:
        return _raw(path, body, timeout)
    except Down:
        _health()
        return _raw(path, body, timeout)


def _path(given: str) -> Path:
    path = Path(given)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _fence(path_or_language: str) -> str:
    return "python" if re.search(r"(\.py$|python)", path_or_language, re.I) \
        else "ts"


# -- what a turn said ---------------------------------------------------------

def _turn(turn: dict, raw: bool) -> dict:
    """The turn as the editor reads it: the words said, and the code."""
    found = ((turn.get("answer") or {}).get("code") or {})
    answer = found.get("answer") or {}
    request = found.get("request") or {}
    out = {
        "reply": (turn.get("reply") or {}).get("text")
        or (turn.get("answer") or {}).get("text") or "",
        "code": answer.get("code"),
        "entry": answer.get("entry"),
        "status": answer.get("status"),
        "language": request.get("language"),
        "ways": (found.get("ways") or request.get("ways")),
        "turn": turn.get("n", turn.get("turn")),
    }
    out = {key: value for key, value in out.items() if value not in
           (None, "", [], {})}
    if raw:
        out["raw"] = turn
    return out


def _say(text: str, sid: str | None, raw: bool = False) -> dict:
    return _turn(_call("/api/say", {"sid": _sid(sid), "q": text},
                       timeout=LONG), raw)


# -- putting code into a file -------------------------------------------------

def _python_span(source: str, entry: str):
    """The lines (0-based start, end exclusive) of a top-level def or class
    named `entry`, decorators included."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and node.name == entry:
            start = min([node.lineno] + [one.lineno for one in
                                         node.decorator_list]) - 1
            return start, node.end_lineno
    return None


def _ts_span(source: str, entry: str):
    """The lines of `function entry(...) {...}` or `const entry = ...`,
    `export` included, found by matching its braces."""
    pattern = re.compile(
        r"^[ \t]*(export\s+)?(default\s+)?(async\s+)?(function\s*\*?\s*"
        + re.escape(entry) + r"\b|(const|let|var)\s+" + re.escape(entry)
        + r"\b)", re.M)
    found = pattern.search(source)
    if not found:
        return None
    at, depth, began, quote = found.end(), 0, False, None
    while at < len(source):
        char = source[at]
        if quote:
            if char == "\\":
                at += 1
            elif char == quote:
                quote = None
        elif char in "'\"`":
            quote = char
        elif char == "{":
            depth, began = depth + 1, True
        elif char == "}":
            depth -= 1
            if began and depth == 0:
                break
        elif char == ";" and not began and depth == 0:
            break
        at += 1
    end = source.find("\n", at)
    end = len(source) if end < 0 else end
    first = source.count("\n", 0, found.start())
    return first, source.count("\n", 0, end) + 1


def _replaced(source: str, entry: str, code: str, python: bool):
    span = (_python_span if python else _ts_span)(source, entry)
    if span is None:
        return None
    lines = source.splitlines(keepends=True)
    start, end = span
    new = code.rstrip("\n") + "\n"
    return "".join(lines[:start]) + new + "".join(lines[end:])


def _diff(before: str, after: str, name: str) -> str:
    import difflib
    return "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        f"a/{name}", f"b/{name}", n=2))


# -- the tools ----------------------------------------------------------------

def t_health(args: dict) -> dict:
    return _health(wait=bool(args.get("start", True)))


def t_say(args: dict) -> dict:
    return _say(str(args["message"]), args.get("sid"),
                bool(args.get("raw")))


def t_request_code(args: dict) -> dict:
    text = str(args["request"]).strip()
    if args.get("signature"):
        text += "\n" + str(args["signature"]).strip()
    for one in args.get("examples") or ():
        text += "\n" + str(one).strip()
    return _say(text, args.get("sid"), bool(args.get("raw")))


def t_ask_about_code(args: dict) -> dict:
    code = str(args["code"]).rstrip()
    fence = _fence(str(args.get("language") or ""))
    text = f"{str(args['question']).strip()}\n```{fence}\n{code}\n```"
    return _say(text, args.get("sid"), bool(args.get("raw")))


def _files(paths) -> dict:
    files = {}
    for given in paths:
        path = _path(given)
        found = [path] if path.is_file() else sorted(path.rglob("*"))
        for one in found:
            name = _relative(one)
            if not one.is_file() or not READ.search(name) or any(
                    ("/" + skip) in "/" + name for skip in SKIPPED):
                continue
            files[name] = one.read_text(encoding="utf-8", errors="replace")
    return files


def t_send_project(args: dict) -> dict:
    files = _files(args.get("paths") or ["."])
    if not files:
        raise RuntimeError("no .py/.ts/.js files under those paths")
    sid, name = _sid(args.get("sid")), args.get("name") or ROOT.name
    found = _call("/api/project", {"sid": sid, "files": files, "name": name},
                  timeout=LONG)
    SENT[sid] = (files, name)
    return found


def t_project(args: dict) -> dict:
    return _call("/api/project?sid=" + _sid(args.get("sid")), timeout=60)


def t_change_file(args: dict) -> dict:
    path = _path(str(args["path"]))
    source = path.read_text(encoding="utf-8")
    python = path.suffix.lower() == ".py"
    entry = args.get("function")
    if entry:
        span = (_python_span if python else _ts_span)(source, entry)
        if span is None:
            raise RuntimeError(f"no top-level {entry} in {_relative(path)}")
        lines = source.splitlines(keepends=True)
        shown = "".join(lines[span[0]:span[1]])
    else:
        shown = source
    # as the architecture reads a request with your own function in it: the
    # words, the function (unfenced: `yours`, tried as one more writer's),
    # then the examples it must meet -- fenced, it is read as a question
    # about the code and nothing is written
    text = "\n".join([str(args["instruction"]).strip(), shown.rstrip()]
                     + [str(one).strip() for one in args.get("examples")
                        or ()])
    said = _say(text, args.get("sid"), bool(args.get("raw")))
    code, written = said.get("code"), said.get("entry") or entry
    out = {"file": _relative(path), **said}
    if not code:
        out["applied"] = False
        out["why not applied"] = "the architecture answered with no code"
        return out
    after = _replaced(source, written, code, python) if written else None
    if after is None:
        out["applied"] = False
        out["why not applied"] = (f"no top-level {written} in the file to "
                                  "replace")
        return out
    out["diff"] = _diff(source, after, _relative(path))
    if args.get("apply"):
        path.write_text(after, encoding="utf-8", newline="")
        out["applied"] = True
        held = SENT.get(_sid(args.get("sid")))
        if held and _relative(path) in held[0]:
            held[0][_relative(path)] = after
        try:
            _call("/api/file", {"sid": _sid(args.get("sid")),
                                "path": _relative(path), "text": after},
                  timeout=60)
        except RuntimeError:
            pass                    # no project sent: nothing to keep in step
    else:
        out["applied"] = False
    return out


SID = {"sid": {"type": "string", "description":
               "Conversation id (default: one per workspace). The "
               "architecture remembers earlier turns of a conversation."}}
RAW = {"raw": {"type": "boolean", "description":
               "Include the server's whole turn JSON (steps, timings)."}}

TOOLS = {
    "health": (t_health, "Whether the cognitive architecture's server is "
               "up: its version, capabilities and languages.", {}, []),
    "say": (t_say, "Say anything to the cognitive architecture, as one turn "
            "of a conversation: a request for code, a change to the last "
            "answer ('make it recursive', 'in Python'), a question about "
            "the project ('who calls main'), or a teaching. Returns its "
            "reply and any code it wrote.",
            {"message": {"type": "string"}, **SID, **RAW}, ["message"]),
    "request_code": (t_request_code, "Ask the cognitive architecture to "
                     "write a function. Give the request in English, "
                     "optionally a signature (`def f(xs: list[int]) -> "
                     "int:` or a TypeScript one) and examples "
                     "(`f([1, 2]) == 3`). Returns the code it wrote.",
                     {"request": {"type": "string"},
                      "signature": {"type": "string"},
                      "examples": {"type": "array",
                                   "items": {"type": "string"}},
                      **SID, **RAW}, ["request"]),
    "ask_about_code": (t_ask_about_code, "Ask the cognitive architecture "
                       "about a piece of code: 'explain this code', 'fix "
                       "the bugs', 'make it iterative', 'add comments'. "
                       "Returns its reply and any code.",
                       {"code": {"type": "string"},
                        "question": {"type": "string"},
                        "language": {"type": "string",
                                     "enum": ["python", "typescript"]},
                        **SID, **RAW}, ["code", "question"]),
    "send_project": (t_send_project, "Send files (or folders) of the "
                     "workspace to the architecture, so it can answer "
                     "about the project. Paths are relative to the "
                     "workspace; default the whole workspace.",
                     {"paths": {"type": "array",
                                "items": {"type": "string"}},
                      "name": {"type": "string"}, **SID}, []),
    "project": (t_project, "What the architecture holds of the project "
                "sent: summary, outline and call graph.", {**SID}, []),
    "change_file": (t_change_file, "Ask the cognitive architecture to "
                    "change a file: the instruction (what the function "
                    "should do, e.g. 'fix the bug: return the first item') "
                    "is sent with the function's code and the examples it "
                    "must meet; the architecture writes its own and tries "
                    "yours. Its answer replaces that top-level function. "
                    "Returns a diff; writes the file only when apply is "
                    "true. Examples are what make a fix work.",
                    {"path": {"type": "string"},
                     "instruction": {"type": "string"},
                     "examples": {"type": "array",
                                  "items": {"type": "string"},
                                  "description": "Calls with results, as "
                                  "`first([1, 2]) == 1`."},
                     "function": {"type": "string", "description":
                                  "Top-level function to send and "
                                  "replace (recommended)."},
                     "apply": {"type": "boolean", "description":
                               "Write the change to the file."},
                     **SID, **RAW}, ["path", "instruction"]),
}


# -- JSON-RPC over stdio -------------------------------------------------------

def _send(message: dict) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _handle(message: dict):
    method, params = message.get("method"), message.get("params") or {}
    if method == "initialize":
        return {"protocolVersion": params.get("protocolVersion",
                                              "2025-06-18"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": NAME, "version": VERSION}}
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": [
            {"name": name, "description": description,
             "inputSchema": {"type": "object", "properties": properties,
                             "required": required}}
            for name, (_, description, properties, required)
            in TOOLS.items()]}
    if method == "tools/call":
        name = params.get("name")
        if name not in TOOLS:
            raise LookupError(f"no tool {name}")
        try:
            found = TOOLS[name][0](params.get("arguments") or {})
            text, failed = json.dumps(found, indent=1,
                                      ensure_ascii=False), False
        except Exception as bad:                    # noqa: BLE001
            text, failed = f"{type(bad).__name__}: {bad}", True
        return {"content": [{"type": "text", "text": text}],
                "isError": failed}
    raise LookupError(f"no method {method}")


def main() -> None:
    global URL
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL)
    URL = parser.parse_args().url
    sys.stdout.reconfigure(encoding="utf-8")
    for line in sys.stdin.buffer:
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if "id" not in message:
            continue                                # a notification
        try:
            _send({"jsonrpc": "2.0", "id": message["id"],
                   "result": _handle(message)})
        except LookupError as bad:
            _send({"jsonrpc": "2.0", "id": message["id"],
                   "error": {"code": -32601, "message": str(bad)}})
        except Exception as bad:                    # noqa: BLE001
            _send({"jsonrpc": "2.0", "id": message["id"],
                   "error": {"code": -32603, "message": str(bad)}})


if __name__ == "__main__":
    main()
