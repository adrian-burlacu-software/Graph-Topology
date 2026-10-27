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
                "python", "edit", "shell")
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


def models() -> dict:
    """Every model this process holds: its kind, where its weights are
    (`cuda`, `cpu`), how many megabytes, and where it was loaded from --
    what fills the card, seen (a goal of itself filled 7.9 GB of 8, models
    loaded as they were needed and never let go)."""
    import gc
    import sys
    if "torch" not in sys.modules:
        return {"models": [], "cuda": None}
    import torch
    found, inner = [], set()
    modules = [one for one in gc.get_objects()
               if isinstance(one, torch.nn.Module)]
    for one in modules:
        for child in one.children():
            inner.add(id(child))
    for one in modules:
        if id(one) in inner:
            continue
        sizes: dict = {}
        for tensor in list(one.parameters()) + list(one.buffers()):
            device = tensor.device.type
            sizes[device] = sizes.get(device, 0) + \
                tensor.numel() * tensor.element_size()
        if sum(sizes.values()) < 2 ** 20:
            continue
        config = getattr(one, "config", None)
        found.append({"kind": type(one).__name__,
                      "from": getattr(config, "_name_or_path", None),
                      "mb": {key: round(value / 2 ** 20) for key, value
                             in sizes.items()}})
    found.sort(key=lambda one: -sum(one["mb"].values()))
    cuda = {"allocated mb": round(torch.cuda.memory_allocated() / 2 ** 20),
            "reserved mb": round(torch.cuda.memory_reserved() / 2 ** 20)} \
        if torch.cuda.is_available() else None
    return {"models": found, "cuda": cuda}


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
        elif parsed.path == "/api/models":
            self._json(models())
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
                said = self.conversations.say(sid, text,
                                              bool(body.get("example")))
                # what this turn loaded and no longer uses, off the card,
                # before the next (v705 `paging`): the models loaded for a
                # goal thought through filled 7.8 GB of 8 and none let go
                # -- never the turn's undoing: a sweep that fails is said in
                # the log, and the turn's reply is sent
                from research.v705 import paging
                try:
                    said["paged"] = paging.sweep()
                except Exception:               # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                self._json(said)
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
            # where it failed, in the server's log: the reply says only
            # what (`IndexError: list index out of range`)
            import traceback
            traceback.print_exc()
            if wedged(bad):
                self._json({"error": f"{type(bad).__name__}: {bad} -- the "
                                     f"card is lost to this process: it "
                                     f"stops, to be started again"}, 500)
                import os
                print("the card's state is lost (a CUDA fault): stopping "
                      f"with {RESTART}, to be started again", flush=True)
                os._exit(RESTART)
            self._json({"error": f"{type(bad).__name__}: {bad}"}, 500)


#: the exit of a server whose card is lost to it: start it again (the
#: editor's extension does)
RESTART = 75
#: what CUDA says of a fault that leaves the process's card unusable:
#: every call after it fails the same (2026-10-27: one illegal access, and
#: the next four turns failed in 0.0 s each)
_LOST = ("illegal memory access", "unspecified launch failure",
         "device-side assert", "cudaErrorLaunchFailure",
         "cudaErrorIllegalAddress")


def wedged(bad: BaseException) -> bool:
    """Whether an error says the process's card is lost to it."""
    said = f"{type(bad).__name__}: {bad}"
    return any(one in said for one in _LOST)


def main() -> None:
    import argparse
    import sys
    # v702: whether a command that may change something is shown first and
    # run only on yes (`--ask`, the default) or run at once (`--no-ask`);
    # what only reads runs either way
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ask", action=argparse.BooleanOptionalAction,
                        default=True)
    found, rest = parser.parse_known_args()
    from research.v702 import shell
    shell.ASK = found.ask
    sys.argv = sys.argv[:1] + rest
    print(f"shell: commands that may change something "
          f"{'are shown first' if shell.ASK else 'run at once'}")
    # room made on the card as each model comes onto it (v705)
    from research.v705 import paging
    paging.install()
    v697.main(Handler, Conversations, "v698",
              __doc__.splitlines()[0])


if __name__ == "__main__":
    main()
