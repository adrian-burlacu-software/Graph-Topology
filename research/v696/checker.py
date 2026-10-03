"""The exact checks of the code world: does it type-check, what does it do.

One persistent Node process (`tscheck.js`) runs TypeScript's compiler API
in memory, so a check costs milliseconds and not a `tsc` start. Both
answers are exact -- nothing here is a guess -- which is what makes code
the world in which search can be seen working (`PLAN.md`).

    checker().check(source)                    -> [errors]
    checker().run(source, entry, cases)        -> [value or {"error"}]
    checker().tests(source)                    -> None, or the failure
"""
from __future__ import annotations

import itertools
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "tscheck.js"


class CheckerError(RuntimeError):
    pass


def _node_path() -> str:
    """Where the global `typescript` package is."""
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if npm is None:
        return os.environ.get("NODE_PATH", "")
    found = subprocess.run([npm, "root", "-g"], capture_output=True,
                           text=True, shell=os.name == "nt")
    return found.stdout.strip()


class Checker:
    def __init__(self) -> None:
        node = shutil.which("node")
        if node is None:
            raise CheckerError("node is not installed")
        env = dict(os.environ, NODE_PATH=_node_path())
        self.process = subprocess.Popen(
            [node, "--max-old-space-size=1024", str(SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            encoding="utf-8", env=env, bufsize=1)
        self.lock = threading.Lock()
        self.ids = itertools.count(1)
        self.calls = 0

    def _ask(self, request: dict) -> dict:
        with self.lock:
            request["id"] = next(self.ids)
            self.calls += 1
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        if not line:
            raise CheckerError("the checker stopped")
        return json.loads(line)

    def check(self, source: str) -> list:
        """The type errors in a candidate; none means it type-checks."""
        reply = self._ask({"op": "check", "source": source})
        if "errors" not in reply:
            raise CheckerError(reply.get("error", "no reply"))
        return reply["errors"]

    def run(self, source: str, entry: str, cases, timeout: int = 200
            ) -> list:
        """What `entry` returns on each case: {"value"} or {"error"}."""
        reply = self._ask({"op": "run", "source": source, "entry": entry,
                           "cases": [list(one) for one in cases],
                           "timeout": timeout})
        if not reply.get("ok"):
            return [{"error": reply.get("error", "failed")}
                    for _ in cases]
        return reply["outputs"]

    def values(self, params, cases, expressions, timeout: int = 50,
               prelude: str = "") -> list:
        """For each expression, its value on each case -- {"value"} or
        {"error"} -- as Node computes it; `prelude` declares the helpers
        they call."""
        if not expressions:
            return []
        try:
            reply = self._ask({"op": "values", "params": list(params),
                               "cases": [list(one) for one in cases],
                               "expressions": list(expressions),
                               "timeout": timeout, "prelude": prelude})
        except CheckerError:
            # A candidate killed the process -- memory, most likely: a
            # search runs arbitrary code. Start again and halve the batch
            # until the one that does it is alone; it is an error, and
            # the rest are what they are.
            self.restart()
            if len(expressions) == 1:
                return [[{"error": "the checker stopped"} for _ in cases]]
            middle = len(expressions) // 2
            return (self.values(params, cases, expressions[:middle],
                                timeout, prelude)
                    + self.values(params, cases, expressions[middle:],
                                  timeout, prelude))
        if not reply.get("ok"):
            raise CheckerError(reply.get("error", "failed"))
        return reply["values"]

    def restart(self) -> None:
        try:
            self.process.kill()
        except OSError:
            pass
        self.__init__()

    def signatures(self, receivers, globals_) -> list:
        """The library as the compiler has it (`tscheck.js`)."""
        reply = self._ask({"op": "signatures", "receivers": list(receivers),
                           "globals": list(globals_)})
        if not reply.get("ok"):
            raise CheckerError(reply.get("error", "failed"))
        return reply["signatures"]

    def tree(self, source: str, files: dict | None = None) -> dict:
        """Every function in `source` as steps -- its parameters, the names
        it binds once, its return -- each node typed; {"unread": why} for
        a function that is more than that (`tscheck.js`). With `files`, a
        project: every file's functions, by "file#name"."""
        reply = self._ask({"op": "tree", "source": source, "files": files})
        if not reply.get("ok"):
            raise CheckerError(reply.get("error", "failed"))
        return reply["tree"]

    def structure(self, source: str, entry: str) -> dict:
        """What `entry` is made of, as the compiler resolves it:
        {"uses": {word: count}, "root": word} (`tscheck.js`)."""
        reply = self._ask({"op": "structure", "source": source,
                           "entry": entry})
        if not reply.get("ok"):
            raise CheckerError(reply.get("error", "failed"))
        return {"uses": reply["uses"], "root": reply["root"]}

    def qualities(self, source: str, entry: str) -> dict:
        """What a program's risks are made of, off its syntax: {names,
        decisions, unbounded, mutates, partial, loops, found}
        (`tscheck.js`, `risk.py`)."""
        reply = self._ask({"op": "qualities", "source": source,
                           "entry": entry})
        if not reply.get("ok"):
            raise CheckerError(reply.get("error", "failed"))
        return reply["qualities"]

    def diagnose(self, files: dict) -> list:
        """What the compiler says is wrong in a project: [{file, start,
        end, message, code}], spans in the files as given."""
        reply = self._ask({"op": "diagnose", "files": files})
        if not reply.get("ok"):
            raise CheckerError(reply.get("error", "failed"))
        return reply["errors"]

    def project(self, files: dict, main: str, timeout: int = 2000
                ) -> str | None:
        """Run a project's file `main` (its tests), the project's own
        imports resolved among its files: None if it ran through, else what
        stopped it."""
        reply = self._ask({"op": "project", "files": files, "main": main,
                           "timeout": timeout})
        return None if reply.get("ok") else reply.get("error", "failed")

    def tests(self, source: str, timeout: int = 2000) -> str | None:
        """Run a whole file (a candidate and its tests): None if it ran
        through, else what stopped it."""
        reply = self._ask({"op": "tests", "source": source,
                           "timeout": timeout})
        return None if reply.get("ok") else reply.get("error", "failed")

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.stdin.close()
            self.process.wait(timeout=5)


_CHECKER: Checker | None = None


def checker() -> Checker:
    global _CHECKER
    if _CHECKER is None or _CHECKER.process.poll() is not None:
        _CHECKER = Checker()
    return _CHECKER
