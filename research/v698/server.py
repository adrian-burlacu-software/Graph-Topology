"""The v698 server: v697's, and what an editor needs of it.

    python -m research.v698 --workers 19 --port 8697

What it adds (`protocol.md` says it whole):

    GET  /api/health       who it is, an id per start, what it can do
    POST /api/say          {sid, q, example}: a turn, said at any length
    POST /api/project      {sid, files: {path: text}, name}: a project, whole
    POST /api/file         {sid, path, text}: one file changed (null: gone)
    GET  /api/project      ?sid=: its summary, outline and call graph

An editor attached to it (`tools/vscode-graph-topology`) asks
`/api/health` to know it is there and whether it was restarted: a changed
`instance` is a restart, and the project is sent again.
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.parse
import uuid
from pathlib import Path

from research.v697 import server as v697
from research.v698 import page  # noqa: F401 (registers the acts)
from research.v698 import project as Pj
from research.v698.steps import steps_of

ROOT = Path(__file__).resolve().parents[2]
#: one id per start: what tells an editor the server was restarted
INSTANCE = uuid.uuid4().hex
STARTED = time.time()
#: the largest body taken: a project is sent whole
LARGEST = 40_000_000
CAPABILITIES = ("say", "turn", "page", "code", "project", "file", "pasted",
                "python", "edit")
#: the languages code is read and written in (v699)
LANGUAGES = ("typescript", "python")


def _version() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=ROOT, capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


VERSION = _version()


class Conversations(v697.Conversations):
    STEPS = staticmethod(steps_of)


class Handler(v697.Handler):
    conversations: Conversations = None     # type: ignore[assignment]

    def do_GET(self) -> None:                   # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        sid = (query.get("sid") or [""])[0][:64]
        if parsed.path == "/api/health":
            self._json({"name": "graph-topology", "server": "v698",
                        "version": VERSION, "instance": INSTANCE,
                        "started": STARTED, "capabilities": CAPABILITIES,
                        "languages": LANGUAGES,
                        "page": "/", "projects": len(Pj.PROJECTS)})
        elif parsed.path == "/api/project":
            held = Pj.project(sid)
            if held is None:
                self._json({"error": "no project for this conversation"},
                           404)
                return
            self._json({"summary": held.summary(), "outline": held.outline(),
                        "graph": held.graph()})
        else:
            super().do_GET()

    def do_POST(self) -> None:                  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        size = int(self.headers.get("Content-Length") or 0)
        if size > LARGEST:
            self._json({"error": f"too large (at most {LARGEST} bytes)"}, 413)
            return
        try:
            body = json.loads(self.rfile.read(size) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "the body is not JSON"}, 400)
            return
        sid = str(body.get("sid") or "")[:64]
        if not sid:
            self._json({"error": "a conversation id (sid)"}, 400)
            return
        try:
            if parsed.path == "/api/say":
                text = str(body.get("q") or "").strip()
                if not text or len(text) > v697.LONGEST * 4:
                    self._json({"error": "something said, not too long"},
                               400)
                    return
                self._json(self.conversations.say(sid, text,
                                                  bool(body.get("example"))))
            elif parsed.path == "/api/project":
                files = body.get("files") or {}
                # `root` (v700): where the files are on disk -- a change
                # asked of the project is made there
                root = body.get("root")
                self._json(Pj.put(sid, files, str(body.get("name") or ""),
                                  whole=True,
                                  root=str(root) if root else None))
            elif parsed.path == "/api/file":
                path = str(body.get("path") or "")
                if not Pj.project(sid):
                    self._json({"error": "no project for this "
                                         "conversation"}, 404)
                    return
                self._json(Pj.put(sid, {path: body.get("text")}))
            else:
                self._json({"error": f"no such endpoint {parsed.path}"}, 404)
        except ValueError as bad:
            self._json({"error": str(bad)}, 400)
        except Exception as bad:                # noqa: BLE001
            self._json({"error": f"{type(bad).__name__}: {bad}"}, 500)


def main() -> None:
    v697.main(Handler, Conversations, "v698",
              __doc__.splitlines()[0])


if __name__ == "__main__":
    main()
