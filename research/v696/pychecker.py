"""The client of the Python checker (`pycheck.py`): the methods
`checker.Checker` has, one persistent worker process, the same protocol.

A Python candidate can do what the worker's watchdog cannot stop (a long
call in C, memory without end): a reply that does not come in time is the
worker killed and started again, and the request is an error -- as Node's
is when a candidate kills it.
"""
from __future__ import annotations

import itertools
import json
import queue
import subprocess
import sys
import threading
from pathlib import Path

from research.v696.checker import CheckerError

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "pycheck.py"
#: how long past its own timeout a request may take before the worker is
#: taken to be stuck (seconds); mypy's first run is the slow one
GRACE = 5.0
MYPY = 180.0


class PyChecker:
    def __init__(self) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(SCRIPT)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            encoding="utf-8", bufsize=1)
        self.replies: queue.Queue = queue.Queue()
        threading.Thread(target=self._read, args=(self.process,),
                         daemon=True).start()
        self.lock = threading.Lock()
        self.ids = itertools.count(1)
        self.calls = 0

    def _read(self, process) -> None:
        for line in process.stdout:
            self.replies.put(line)
        self.replies.put("")

    def _ask(self, request: dict, wait: float) -> dict:
        with self.lock:
            request["id"] = next(self.ids)
            self.calls += 1
            try:
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
            except OSError as bad:
                self.restart()
                raise CheckerError("the checker stopped") from bad
            while True:
                try:
                    line = self.replies.get(timeout=wait)
                except queue.Empty:
                    self.restart()
                    raise CheckerError("the checker did not answer in "
                                       f"{wait:.0f} s") from None
                if not line:
                    self.restart()
                    raise CheckerError("the checker stopped")
                reply = json.loads(line)
                if reply.get("id") == request["id"]:
                    return reply

    def restart(self) -> None:
        try:
            self.process.kill()
        except OSError:
            pass
        self.__init__()

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.stdin.close()
            self.process.wait(timeout=5)

    # -- running ------------------------------------------------------------

    def run(self, source: str, entry: str, cases, timeout: int = 200
            ) -> list:
        cases = [list(one) for one in cases]
        try:
            reply = self._ask({"op": "run", "source": source, "entry": entry,
                               "cases": cases, "timeout": timeout},
                              wait=GRACE + timeout * (len(cases) + 5) / 1000)
        except CheckerError as bad:
            return [{"error": str(bad)} for _ in cases]
        if not reply.get("ok"):
            return [{"error": reply.get("error", "failed")} for _ in cases]
        return reply["outputs"]

    def values(self, params, cases, expressions, timeout: int = 50,
               prelude: str = "", types=None) -> list:
        if not expressions:
            return []
        cases = [list(one) for one in cases]
        try:
            reply = self._ask(
                {"op": "values", "params": list(params), "cases": cases,
                 "expressions": list(expressions), "timeout": timeout,
                 "prelude": prelude, "types": types},
                wait=GRACE + timeout * len(cases) * len(expressions) / 1000)
            if reply.get("error") == "interrupted":
                # the watchdog's interrupt landed past the candidate it was
                # for: one of them, as a worker stopped is
                raise CheckerError("interrupted")
        except CheckerError:
            # one of them stopped the worker: halve until it is alone
            if len(expressions) == 1:
                return [[{"error": "the checker stopped"} for _ in cases]]
            middle = len(expressions) // 2
            return (self.values(params, cases, expressions[:middle],
                                timeout, prelude, types)
                    + self.values(params, cases, expressions[middle:],
                                  timeout, prelude, types))
        if not reply.get("ok"):
            raise CheckerError(reply.get("error", "failed"))
        return reply["values"]

    def tests(self, source: str, timeout: int = 2000) -> str | None:
        try:
            reply = self._ask({"op": "tests", "source": source,
                               "timeout": timeout},
                              wait=GRACE + timeout / 1000)
        except CheckerError as bad:
            return str(bad)
        return None if reply.get("ok") else reply.get("error", "failed")

    def project(self, files: dict, main: str, timeout: int = 2000
                ) -> str | None:
        try:
            reply = self._ask({"op": "project", "files": files, "main": main,
                               "timeout": timeout},
                              wait=GRACE + timeout / 1000)
        except CheckerError as bad:
            return str(bad)
        return None if reply.get("ok") else reply.get("error", "failed")

    # -- checking -------------------------------------------------------------

    def check(self, source: str) -> list:
        reply = self._ask({"op": "check", "source": source}, wait=MYPY)
        if "errors" not in reply:
            raise CheckerError(reply.get("error", "no reply"))
        return reply["errors"]

    def diagnose(self, files: dict, strict: bool = False) -> list:
        reply = self._ask({"op": "diagnose", "files": files,
                           "strict": strict}, wait=MYPY)
        if not reply.get("ok"):
            raise CheckerError(reply.get("error", "failed"))
        return reply["errors"]

    # -- reading ------------------------------------------------------------

    def _simple(self, request: dict, key: str, wait: float = 30.0):
        reply = self._ask(request, wait=wait)
        if not reply.get("ok"):
            raise CheckerError(reply.get("error", "failed"))
        return reply[key]

    def shape(self, source: str, entry: str) -> dict:
        return self._simple({"op": "shape", "source": source,
                             "entry": entry}, "shape")

    def outline(self, files: dict) -> dict:
        return self._simple({"op": "outline", "files": files}, "outline")

    # read off the syntax, here: nothing of it is run (`pystructure.py`)

    def structure(self, source: str, entry: str) -> dict:
        from research.v696 import pystructure
        return pystructure.structure(source, entry)

    def qualities(self, source: str, entry: str) -> dict:
        from research.v696 import pystructure
        return pystructure.qualities(source, entry)

    def restyle(self, source: str, entry: str, way: str) -> str | None:
        reply = self._ask({"op": "restyle", "source": source,
                           "entry": entry, "way": way}, wait=30.0)
        if not reply.get("ok"):
            raise CheckerError(reply.get("error", "failed"))
        return reply.get("source")

    def signatures(self, receivers, globals_) -> list:
        return self._simple({"op": "signatures",
                             "receivers": list(receivers),
                             "globals": list(globals_)}, "signatures",
                            wait=MYPY)

    def tree(self, source: str, files: dict | None = None) -> dict:
        return self._simple({"op": "tree", "source": source,
                             "files": files}, "tree", wait=MYPY)
