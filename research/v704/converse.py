"""The conversation test (v704): everything the architecture says it does,
asked as a person would, over HTTP, and each answer scored.

    python -m research.v704.converse                  every conversation
    python -m research.v704.converse --area data-ask  one area
    python -m research.v704.converse --only 12 40     some, by number
    python -m research.v704.converse --label before   output name

Each conversation (`suite.SUITE`) is a fresh one: its own id, and where it
is about a project, its own copy of `fixtures/library` on disk, sent with
its root -- so a change the architecture makes is made there, and looked
at there. Its last answer (or the files after it) is checked:

- **right**: the check holds;
- **unsure**: it does not, and the answer says it could not, does not
  know, or asks -- an honest no-answer;
- **wrong**: it does not, and the answer claims otherwise, or the files
  were changed wrongly, or the server failed or took too long.

Written once (2026-10-11) and never tuned on: a round adds conversations
and keeps the old ones. Run against a server on `--url` (8697).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "library"
OUT = ROOT / "conversations-out"
#: conversations at once, and how long one turn may take -- one: the
#: server says one turn at a time, whoever says it (v690's
#: `service._engines`), and six at once queued behind code being written
#: until every turn timed out
WORKERS, TURN_SECONDS = 1, 300


# -- what an answer said -----------------------------------------------------

def text_of(turn: dict) -> str:
    reply = turn.get("reply")
    if isinstance(reply, dict):
        reply = reply.get("text") or ""
    return str(reply or "")


def stance_of(turn: dict) -> str | None:
    reply = turn.get("reply")
    if isinstance(reply, dict):
        return (reply.get("message") or {}).get("stance")
    return None


def code_of(turn: dict) -> dict:
    code = (turn.get("answer") or {}).get("code")
    return code if isinstance(code, dict) else {}


def _norm(text: str) -> str:
    return " ".join(str(text).lower().split())


#: what an answer that does not answer says -- read here, by the test,
#: never by the architecture
NOT_ANSWERED = re.compile(
    r"not sure|unsure|i don.t know|do not know|don't know|could not|"
    r"cannot|can.t|did not change|i find nothing|no two agree|put every "
    r"step back|give me an example|which (one|function|file)|what do you "
    r"mean|say yes|nothing (was )?written|nothing the writers|i did not|"
    r"no answer|not found|no such|unknown|which do you mean|\?$")
NOT_ANSWERED_STATUS = {"unsure", "unchanged", "undone", "unsolved",
                       "unplaced", "unchecked", "unplanned"}


def unanswered(turn: dict) -> bool:
    status = (code_of(turn).get("answer") or {}).get("status")
    stance = stance_of(turn)
    return status in NOT_ANSWERED_STATUS or stance in (
        "unknown", "unsure", "doubt") or bool(NOT_ANSWERED.search(
            _norm(text_of(turn))))


# -- checks: each (turn, the conversation's root) -> True / False / why ------

def says(*words):
    """The answer says any of these (case and spacing ignored)."""
    def check(turn, root):
        said = _norm(text_of(turn))
        return any(_norm(one) in said for one in words)
    check.about = f"says any of {words}"
    return check


def says_all(*words):
    def check(turn, root):
        said = _norm(text_of(turn))
        return all(_norm(one) in said for one in words)
    check.about = f"says all of {words}"
    return check


def number(value):
    """The answer says this number, as a number."""
    wanted = str(value)

    def check(turn, root):
        found = re.findall(r"-?\d+(?:\.\d+)?", text_of(turn).replace(",", ""))
        return any(one == wanted or (re.fullmatch(r"-?\d+\.\d+", one) and
                                     float(one) == float(wanted))
                   for one in found)
    check.about = f"says the number {value}"
    return check


def stance(wanted: str):
    """Yes or no: the answer's stance where it has one, else how it
    begins."""
    def check(turn, root):
        found = stance_of(turn)
        if found in ("yes", "no"):
            return found == wanted
        said = _norm(text_of(turn))
        return said.startswith(wanted)
    check.about = f"answers {wanted}"
    return check


def py(code: str):
    """Python run in the conversation's copy of the project after it --
    what was changed, looked at (exit 0: holds)."""
    def check(turn, root):
        if root is None:
            return False
        done = subprocess.run([sys.executable, "-c", code], cwd=root,
                              capture_output=True, text=True, timeout=60)
        return done.returncode == 0
    check.about = f"holds in the project: {code}"
    check.changes = True
    return check


def untouched_and_asked():
    """Nothing of the project changed, and the answer asks, or declines."""
    def check(turn, root):
        return root is not None and _same_as_fixture(root) and \
            unanswered(turn)
    check.about = "changes nothing, and asks or declines"
    check.changes = True
    return check


def code_does(cases: list, entry: str | None = None):
    """The code it gave does this: (arguments, what it gives) each."""
    def check(turn, root):
        answer = code_of(turn).get("answer") or {}
        code = answer.get("code") or turn.get("code")
        name = entry or answer.get("entry") or turn.get("entry")
        if not code or not name:
            return False
        language = answer.get("language") or code_of(turn).get(
            "request", {}).get("language") or "python"
        from research.v696.checker import CheckerError, checker
        try:
            rows = checker(language).run(code, name,
                                         [args for args, _ in cases])
        except CheckerError:
            return False
        for row, (_, wanted) in zip(rows, cases):
            got = row.get("value", "!error")
            if isinstance(wanted, float) and isinstance(got, (int, float)):
                if abs(got - wanted) > 1e-6:
                    return False
            elif got != wanted:
                return False
        return True
    check.about = f"its code gives {cases}"
    return check


def replies():
    """Any answer at all, not a failure (hello, thanks)."""
    def check(turn, root):
        return bool(text_of(turn).strip())
    check.about = "answers"
    return check


def _same_as_fixture(root: Path) -> bool:
    for path in FIXTURE.rglob("*"):
        if "__pycache__" in path.parts or not path.is_file():
            continue
        other = root / path.relative_to(FIXTURE)
        if not other.exists() or other.read_bytes() != path.read_bytes():
            return False
    return True


# -- a conversation, held ---------------------------------------------------

def _post(url: str, path: str, body: dict) -> dict:
    request = urllib.request.Request(url + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=TURN_SECONDS) as reply:
        return json.loads(reply.read())


def converse(number_: int, item, url: str, run_id: str) -> dict:
    sid = f"v704-{run_id}-{number_}"
    root = None
    out = {"n": number_, "area": item.area, "turns": item.turns,
           "check": getattr(item.check, "about", "")}
    started = time.time()
    try:
        if item.project:
            root = Path(tempfile.mkdtemp(prefix="v704-")) / "library"
            shutil.copytree(FIXTURE, root,
                            ignore=shutil.ignore_patterns("__pycache__"))
            files = {one.relative_to(root).as_posix():
                     one.read_text(encoding="utf-8")
                     for one in root.rglob("*") if one.is_file()}
            _post(url, "/api/project", {"sid": sid, "files": files,
                                        "name": "library",
                                        "root": str(root)})
        said = []
        turn = {}
        for message in item.turns:
            turn = _post(url, "/api/say", {"sid": sid, "q": message})
            said.append(text_of(turn)[:600])
        out["said"] = said
        try:
            held = bool(item.check(turn, root))
        except Exception as bad:                # noqa: BLE001
            held, out["check error"] = False, f"{type(bad).__name__}: {bad}"
        if held:
            out["verdict"] = "right"
        elif getattr(item.check, "changes", False) and root is not None \
                and not _same_as_fixture(root) and \
                item.check.about != "changes nothing, and asks or declines":
            out["verdict"] = "wrong"
            out["why"] = "changed the project, not as asked"
        elif getattr(item.check, "about", "") == \
                "changes nothing, and asks or declines":
            out["verdict"] = "wrong"
            out["why"] = ("changed the project" if root is not None and
                          not _same_as_fixture(root) else
                          "did not ask or decline")
        else:
            out["verdict"] = "unsure" if unanswered(turn) else "wrong"
        status = (code_of(turn).get("answer") or {}).get("status")
        out["status"] = status
    except urllib.error.HTTPError as bad:
        out.update(verdict="wrong", why=f"server {bad.code}: "
                   f"{bad.read().decode(errors='replace')[:300]}")
    except Exception as bad:                    # noqa: BLE001
        out.update(verdict="wrong", why=f"{type(bad).__name__}: {bad}")
    finally:
        if root is not None:
            shutil.rmtree(root.parent, ignore_errors=True)
    out["seconds"] = round(time.time() - started, 1)
    return out


def summary(rows: list) -> dict:
    areas: dict = {}
    for row in rows:
        one = areas.setdefault(row["area"], {"right": 0, "unsure": 0,
                                             "wrong": 0})
        one[row["verdict"]] += 1
    total = {key: sum(one[key] for one in areas.values())
             for key in ("right", "unsure", "wrong")}
    return {"areas": areas, "total": total, "of": len(rows)}


def main(argv=None) -> int:
    from research.v704.suite import SUITE
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8697")
    parser.add_argument("--area", default=None)
    parser.add_argument("--only", type=int, nargs="*", default=None)
    parser.add_argument("--label", default=time.strftime("%Y%m%d-%H%M"))
    parser.add_argument("--workers", type=int, default=WORKERS)
    options = parser.parse_args(argv)
    chosen = [(at, one) for at, one in enumerate(SUITE)
              if (options.area is None or one.area == options.area) and
              (options.only is None or at in options.only)]
    OUT.mkdir(exist_ok=True)
    started = time.time()
    run_id = str(int(started))
    with ThreadPoolExecutor(options.workers) as pool:
        rows = list(pool.map(lambda pair: converse(pair[0], pair[1],
                                                   options.url, run_id),
                             chosen))
    rows.sort(key=lambda row: row["n"])
    with (OUT / f"{options.label}.jsonl").open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    found = summary(rows)
    found["seconds"] = round(time.time() - started)
    (OUT / f"{options.label}.summary.json").write_text(
        json.dumps(found, indent=1), encoding="utf-8")
    for area, one in sorted(found["areas"].items()):
        print(f"{area:16} right {one['right']:3}  unsure {one['unsure']:3}"
              f"  wrong {one['wrong']:3}")
    total = found["total"]
    print(f"{'all':16} right {total['right']:3}  unsure {total['unsure']:3}"
          f"  wrong {total['wrong']:3}  of {found['of']} in "
          f"{found['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
