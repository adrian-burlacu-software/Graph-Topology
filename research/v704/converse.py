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


def part_of(turn: dict) -> str:
    """Which of the architecture's parts answered: what the project was
    asked (`callers`, `data`, `plan`, `goal`), the shell, code written, or
    where a verdict came from (`kind`, `mathematics`, `told`)."""
    code = code_of(turn)
    if code.get("shell"):
        return "shell"
    asked = (code.get("project") or {}).get("asked")
    if asked:
        return f"project {asked}"
    if code.get("request") or code.get("writers"):
        return "code written"
    if code.get("change"):
        return "change"
    return str((turn.get("answer") or {}).get("source") or "none")


def operator_of(turn: dict) -> tuple:
    """The executive's operator that answered, its rule, and the others
    proposed in the cycle it was chosen in -- the place in the
    architecture's own code an answer came from (`selfmodel.places`)."""
    for trace in reversed(turn.get("trace") or []):
        name = trace.get("answered_by")
        if not name:
            continue
        chosen = next((one for one in trace.get("fired", [])
                       if one.get("operator") == name), {})
        return name, chosen.get("rule", ""), [
            one for one in chosen.get("candidates", []) if one != name]
    return None, "", []


def fired_of(turn: dict) -> list:
    """Every operator the executive tried in the last turn, in order, and
    what came of it (`answered`, `declined`, `continue`): where an answer
    was looked for before the one that gave it."""
    out = []
    for trace in turn.get("trace") or []:
        for one in trace.get("fired", []):
            out.append({"operator": one.get("operator"),
                        "rule": one.get("rule", ""),
                        "outcome": one.get("outcome")})
    return out


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


def says_only(right, *others):
    """The answer says `right` and none of the other candidates: a list of
    them all is no answer to which one (`which member joined first`,
    answered with all four members, was scored right)."""
    def check(turn, root):
        said = _norm(text_of(turn))
        return _norm(right) in said and not any(_norm(one) in said
                                                for one in others)
    check.about = f"says {right!r} and none of {others}"
    return check


def not_denied(check_, *words):
    """A check that holds, in an answer that does not deny it (`180 hours
    kilometers is not something measured in 1 hour` says 180)."""
    def check(turn, root):
        said = f" {_norm(text_of(turn))} "
        return check_(turn, root) and not any(f" {one} " in said
                                              for one in words)
    check.about = f"{check_.about}, not denied"
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
        return root is not None and \
            _same_as_fixture(root, FIXTURE.parent / root.name) and \
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


def _same_as_fixture(root: Path, fixture: Path = FIXTURE) -> bool:
    for path in fixture.rglob("*"):
        if "__pycache__" in path.parts or not path.is_file():
            continue
        other = root / path.relative_to(fixture)
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
    fixture = FIXTURE.parent / getattr(item, "fixture", "library")
    out = {"n": number_, "area": item.area, "turns": item.turns,
           "check": getattr(item.check, "about", "")}
    started = time.time()
    try:
        if item.project:
            root = Path(tempfile.mkdtemp(prefix="v704-")) / fixture.name
            shutil.copytree(fixture, root,
                            ignore=shutil.ignore_patterns("__pycache__"))
            files = {one.relative_to(root).as_posix():
                     one.read_text(encoding="utf-8")
                     for one in root.rglob("*") if one.is_file()}
            _post(url, "/api/project", {"sid": sid, "files": files,
                                        "name": fixture.name,
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
                and not _same_as_fixture(root, fixture) and \
                item.check.about != "changes nothing, and asks or declines":
            out["verdict"] = "wrong"
            out["why"] = "changed the project, not as asked"
        elif getattr(item.check, "about", "") == \
                "changes nothing, and asks or declines":
            out["verdict"] = "wrong"
            out["why"] = ("changed the project" if root is not None and
                          not _same_as_fixture(root, fixture) else
                          "did not ask or decline")
        else:
            out["verdict"] = "unsure" if unanswered(turn) else "wrong"
        status = (code_of(turn).get("answer") or {}).get("status")
        out["status"] = status
        out["part"] = part_of(turn)
        out["operator"], out["rule"], out["candidates"] = operator_of(turn)
        out["fired"] = fired_of(turn)
        out["acts"] = {"read": ((turn.get("heard") or {}).get("acts")
                                or [])[:2],
                       "taken": (turn.get("reading") or {}).get("act")}
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


def rescore(label: str) -> dict:
    """A run's answers scored again by the suite's checks as they are now
    -- only those that read what was said; a check of files or code needs
    the run again. After a check is corrected (2026-10-11: a list of every
    candidate passed `which ... first`)."""
    from research.v704.suite import SUITE
    path = OUT / f"{label}.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    changed = 0
    for row in rows:
        item = SUITE[row["n"]]
        about = getattr(item.check, "about", "")
        if not row.get("said") or getattr(item.check, "changes", False) \
                or about.startswith(("its code", "answers yes",
                                     "answers no")):
            continue
        turn = {"reply": {"text": row["said"][-1]},
                "answer": {"code": {"answer": {"status": row.get(
                    "status")}}}}
        verdict = "right" if item.check(turn, None) else (
            "unsure" if unanswered(turn) else "wrong")
        if verdict != row["verdict"] and not row.get("why"):
            row["verdict"], row["check"] = verdict, about
            changed += 1
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    found = summary(rows)
    found["rescored"] = changed
    (OUT / f"{label}.summary.json").write_text(json.dumps(found, indent=1),
                                               encoding="utf-8")
    return found


def _seconds() -> dict:
    """How long each conversation took in the latest run that says:
    {n: seconds}."""
    runs = sorted(OUT.glob("*.jsonl"), key=lambda one: one.stat().st_mtime)
    for path in reversed(runs[-5:]):
        try:
            rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        except (OSError, ValueError):
            continue
        if rows and all("seconds" in row for row in rows):
            return {row["n"]: row["seconds"] for row in rows}
    return {}


def main(argv=None) -> int:
    from research.v704.suite import CORE, SUITE
    if argv is None and len(sys.argv) > 2 and sys.argv[1] == "--rescore":
        for label in sys.argv[2:]:
            found = rescore(label)
            print(label, found["total"], "rescored", found["rescored"])
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8697")
    parser.add_argument("--area", default=None)
    parser.add_argument("--suite", default="suite",
                        choices=("suite", "heldout"),
                        help="heldout: fixtures/garden, kept apart from "
                             "the work it measures")
    parser.add_argument("--core", action="store_true",
                        help="only the core areas (suite.CORE)")
    parser.add_argument("--only", type=int, nargs="*", default=None)
    parser.add_argument("--label", default=time.strftime("%Y%m%d-%H%M"))
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--urls", nargs="*", default=None,
                        help="servers of the same code, each said one "
                             "conversation at a time (a server says one "
                             "turn at a time, whoever says it)")
    parser.add_argument("--heavy", nargs="*", default=None,
                        help="areas said to the first of --urls alone")
    options = parser.parse_args(argv)
    if options.suite == "heldout":
        from research.v704.heldout import HELDOUT as SUITE
    chosen = [(at, one) for at, one in enumerate(SUITE)
              if (options.area is None or one.area == options.area) and
              (not options.core or one.area in CORE) and
              (options.only is None or at in options.only)]
    OUT.mkdir(exist_ok=True)
    started = time.time()
    run_id = str(int(started))
    urls = options.urls or [options.url] * options.workers
    # the longest first, where a run before says how long: the last of
    # several servers waiting on one long conversation is time lost
    took = _seconds()
    chosen.sort(key=lambda pair: -took.get(pair[0], 0))
    # what writes code, said to the first server alone (most of the card:
    # its models are large, and two such turns at once filled it -- every
    # server waited, eighteen turns timed out); the rest to any -- the
    # first too, once its own are said (206 s of them, against 539 s of
    # the rest on the other)
    import collections
    heavy = set(options.heavy or ()) if len(urls) > 1 else set()
    own_first = collections.deque(one for one in chosen
                                  if one[1].area in heavy)
    shared = collections.deque(one for one in chosen
                               if one[1].area not in heavy)
    rows = []
    # each conversation kept as it ends: a run cut short (the machine went
    # down 2026-10-20, a whole measure lost) keeps what it said
    partial = OUT / f"{options.label}.partial.jsonl"
    partial.write_text("", encoding="utf-8")
    import threading
    kept = threading.Lock()

    def worker(at, url):
        while True:
            try:
                pair = own_first.popleft() if at == 0 and own_first \
                    else shared.popleft()
            except IndexError:
                return
            row = converse(pair[0], pair[1], url, run_id)
            rows.append(row)
            with kept, partial.open("a", encoding="utf-8") as out:
                out.write(json.dumps(row) + "\n")
    with ThreadPoolExecutor(len(urls)) as pool:
        list(pool.map(lambda one: worker(*one), enumerate(urls)))
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
