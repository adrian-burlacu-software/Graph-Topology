"""An MCP server (stdio) that puts the cognitive architecture's server in
front of an MCP client: Claude asks it for code, asks about code, and has it
change files.

    python tools/graph-topology-mcp/server.py [--url http://127.0.0.1:8697]

It is a bridge, nothing more: every tool is a call on the v698 server's HTTP
API (`research/v698/protocol.md`) -- the architecture reads, searches,
writes and changes files (the project is sent with its root on disk); this
only carries the words there and the answer back.

Standard library only: JSON-RPC 2.0, one message per line on stdin/stdout.
"""
from __future__ import annotations

import argparse
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
#: code, and data (v701: read as what it holds and its schema)
READ = re.compile(r"\.(ts|tsx|js|jsx|mjs|cjs|mts|cts|py|json|ya?ml|csv|tsv)$",
                  re.I)
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
        # read again from disk: what the architecture changed since is
        # what is there now
        for sid, (paths, name) in list(SENT.items()):
            _raw("/api/project", {"sid": sid, "files": _files(paths),
                                  "name": name, "root": str(ROOT)},
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
        "turn": turn.get("number"),
    }
    made = found.get("change") or {}
    for key in ("file", "lines", "written", "diff"):
        if made.get(key) not in (None, ""):
            out[key] = made[key]
    out = {key: value for key, value in out.items() if value not in
           (None, "", [], {})}
    if raw:
        out["raw"] = turn
    return out


def _say(text: str, sid: str | None, raw: bool = False) -> dict:
    return _turn(_call("/api/say", {"sid": _sid(sid), "q": text},
                       timeout=LONG), raw)


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
        raise RuntimeError("no code or data files under those paths")
    sid, name = _sid(args.get("sid")), args.get("name") or ROOT.name
    # the root: where the files are, so that a change asked is made there
    found = _call("/api/project", {"sid": sid, "files": files, "name": name,
                                   "root": str(ROOT)}, timeout=LONG)
    SENT[sid] = (list(args.get("paths") or ["."]), name)
    return found


def t_project(args: dict) -> dict:
    return _call("/api/project?sid=" + _sid(args.get("sid")), timeout=60)


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
            "the project ('who calls main'), a change of the project's code "
            "('in _turn, read the number from \"number\"') -- which it "
            "makes in the files itself, where the project was sent -- or a "
            "teaching. Returns its reply, any code it wrote, and any change "
            "it made (file, lines, diff, written).",
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
