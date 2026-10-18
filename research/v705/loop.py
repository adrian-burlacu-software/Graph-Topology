"""The architecture improving itself, unattended (v705): its own measure
run, a goal of itself said to it, each step it proposes agreed to, and a
change it makes to its own code kept only where the measure says it is
better -- reverted where not.

    python -m research.v705.loop [--rounds N]     state/v705-loop.jsonl

Each round:

1. **measure** -- the conversation test (v704 `converse`, 197
   conversations) against the running server, and the self-model built
   from it (`selfmodel build`): what it does worst now, by its own test.
   Not run again where nothing changed since the last.
2. **goal** -- `I want you to make fewer mistakes`, said to it in a
   conversation of its own with this repository as the project, and `yes`
   to each step it proposes (look, look inside, change, one call up, the
   next weakness) until it says it is as far as it can take the goal. Which
   case, which place, which change, and whether a change answers its case
   are all its own (v704 `doing`, v705 `evidence`, `inside.verify`).
3. **keep or revert** -- where it wrote a change: the server started again
   on its code as changed, the unit tests (`run_tests`) and the
   conversation test run; kept, committed with what was measured, where the
   tests pass, more conversations are right, and none right before is not
   now; else every file it changed put back as it was.

What the driver does is say the goal and agree; what is decided is the
architecture's. A round where it changes nothing ends the loop: said the
same goal again, it would do the same.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "state" / "v705-loop.jsonl"
URL = "http://127.0.0.1:8697"
#: what it is told to want: read as a goal (1.0) and looked at by what it
#: gets wrong now (`evidence.ways`) -- `fix what you get wrong in
#: conversations` was read as a change of `conversations` (v690)
GOAL = "I want you to make fewer mistakes."
#: the steps agreed to at most in a round
STEPS = 16
#: the server's start, at most (it takes about three minutes)
STARTUP = 900


#: a turn of the goal waited for at most: thinking it through (240 s),
#: and a change step writing and running again sixty changes and more
TURN = 1800


def _post(path: str, body: dict) -> dict:
    import urllib.request
    request = urllib.request.Request(URL + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=TURN) as reply:
        return json.loads(reply.read())


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout


def _changed() -> list:
    """The repository's files not as committed."""
    return [line[3:].strip() for line in _git(
        "status", "--porcelain").splitlines() if line.strip()]


def _bridge():
    """The MCP bridge's module (`tools/graph-topology-mcp/server.py`): the
    files of a project as it sends them."""
    path = ROOT / "tools" / "graph-topology-mcp" / "server.py"
    spec = importlib.util.spec_from_file_location("graph_topology_bridge",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- the server ---------------------------------------------------------------

def _up() -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{URL}/api/health", timeout=5):
            return True
    except (urllib.error.URLError, OSError):
        return False


def _stop() -> None:
    """The server and its checker stopped, found by what they run (the
    bridge and anything else left alone)."""
    script = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
              "| Where-Object { $_.CommandLine -like "
              "'*-m research.v698 --workers*' -or $_.CommandLine -like "
              "'*v696\\pycheck.py*' } | ForEach-Object { Stop-Process -Id "
              "$_.ProcessId -Force }")
    subprocess.run(["powershell", "-NoProfile", "-Command", script],
                   capture_output=True)
    for _ in range(30):
        if not _up():
            return
        time.sleep(2)


def _start() -> None:
    """The server started on the code as it is on disk, waited for."""
    log = ROOT / "state" / "v705-loop-server.log"
    with log.open("ab") as out:
        out.write(f"\n== started {time.strftime('%Y-%m-%d %H:%M:%S')} by "
                  f"the loop\n".encode())
        subprocess.Popen([sys.executable, "-m", "research.v698",
                          "--workers", "19", "--port", "8697"], cwd=ROOT,
                         stdin=subprocess.DEVNULL, stdout=out, stderr=out,
                         creationflags=getattr(subprocess,
                                               "CREATE_NEW_PROCESS_GROUP", 0))
    until = time.time() + STARTUP
    while not _up():
        if time.time() > until:
            raise RuntimeError(f"the server did not come up in {STARTUP} s "
                               f"(state/{log.name})")
        time.sleep(10)


def _restart() -> None:
    _stop()
    _start()


# -- the steps ----------------------------------------------------------------

def measure(label: str) -> dict:
    """The conversation test run, and the self-model built from it:
    {n: verdict}."""
    from research.v704 import converse, selfmodel
    converse.main(["--url", URL, "--label", label])
    selfmodel.build(label)
    rows = (converse.OUT / f"{label}.jsonl").open(encoding="utf-8")
    return {one["n"]: one["verdict"] for one in map(json.loads, rows)}


def goal(sid: str) -> list:
    """The goal said to it, and each step it proposes agreed to: the turns,
    [(said, reply, status)]."""
    files = _bridge()._files(["research"])
    _post("/api/project", {"sid": sid, "files": files,
                           "name": ROOT.name, "root": str(ROOT)})
    turns = []
    said = GOAL
    for _ in range(STEPS):
        from research.v704.converse import code_of, text_of
        turn = _post("/api/say", {"sid": sid, "q": said})
        reply = " ".join(text_of(turn).split())
        status = (code_of(turn).get("answer") or {}).get("status")
        turns.append({"said": said, "reply": reply, "status": status})
        if status not in ("goal", "goal step") or \
                "as far as I can take" in reply or \
                "last step of the plan" in reply:
            break
        said = "yes"
    return turns


def tests() -> tuple:
    """The unit tests, all of them: (passed, the summary line)."""
    done = subprocess.run([sys.executable, "-m", "research.run_tests"],
                          cwd=ROOT, capture_output=True, text=True)
    lines = (done.stdout + done.stderr).strip().splitlines()
    summary = next((one for one in reversed(lines) if " tests in " in one),
                   lines[-1] if lines else "")
    return done.returncode == 0 and "all passed" in summary, summary


def compared(before: dict, after: dict) -> dict:
    """What a change did to the conversation test: right before and after,
    which became right and which stopped being."""
    gained = sorted(n for n, verdict in after.items()
                    if verdict == "right" and before.get(n) != "right")
    lost = sorted(n for n, verdict in before.items()
                  if verdict == "right" and after.get(n) != "right")
    return {"right before": sum(v == "right" for v in before.values()),
            "right after": sum(v == "right" for v in after.values()),
            "gained": gained, "lost": lost}


def _kept(row: dict, files: list) -> None:
    measured = row["measured"]
    subject = (f"v705 loop: its own change kept -- {len(measured['gained'])} "
               f"more conversations right ({measured['right before']} -> "
               f"{measured['right after']}), none lost, unit tests pass")
    body = (f"Made by the architecture itself, unattended, for \"{GOAL}\" "
            f"(research/v705/loop.py, round {row['round']}): conversations "
            f"now right {measured['gained']}.\n\n" + "\n".join(
                f"- {one['said']}: {one['reply'][:300]}"
                for one in row["turns"]) +
            "\n\nCo-Authored-By: Claude Opus 5.5 (1M context) "
            "<noreply@anthropic.com>")
    _git("add", "--", *files)
    _git("commit", "-q", "-m", subject, "-m", body)


def _reverted(files: list) -> None:
    tracked = set(_git("ls-files").splitlines())
    for one in files:
        if one in tracked:
            _git("checkout", "--", one)
        else:
            (ROOT / one).unlink(missing_ok=True)


def _logged(row: dict) -> None:
    with OUT.open("a", encoding="utf-8") as out:
        out.write(json.dumps(row) + "\n")


def run(rounds: int = 3, baseline: str | None = None) -> list:
    """`baseline`: a run of the conversation test already made on the code
    as it is, its self-model built -- not made again."""
    if _changed():
        raise SystemExit("the tree is not as committed: commit it first, so "
                         "what the loop changes is the loop's alone")
    # on its code as it is now
    _restart()
    stamp = time.strftime("%Y%m%d-%H%M")
    label = baseline or f"loop-{stamp}-0"
    if baseline:
        from research.v704 import converse, selfmodel
        selfmodel.build(baseline)
        before = {one["n"]: one["verdict"] for one in map(json.loads, (
            converse.OUT / f"{baseline}.jsonl").open(encoding="utf-8"))}
    else:
        before = measure(label)
    rows = []
    for round_ in range(1, rounds + 1):
        started = time.time()
        row = {"round": round_, "measure": label,
               "right": sum(v == "right" for v in before.values()),
               "turns": goal(f"loop-{stamp}-{round_}")}
        files = _changed()
        row["changed"] = files
        if not files:
            row.update(outcome="nothing changed",
                       seconds=round(time.time() - started))
            _logged(row)
            rows.append(row)
            break
        row["diff"] = _git("diff")[:20000]
        _stop()
        passed, said = tests()
        row["tests"] = said
        _start()
        after_label = f"loop-{stamp}-{round_}"
        after = measure(after_label) if passed else {}
        row["measured"] = compared(before, after) if passed else None
        better = passed and row["measured"]["gained"] and \
            not row["measured"]["lost"]
        if better:
            _kept(row, files)
            _git("push", "-q")
            row["outcome"] = "kept"
            before, label = after, after_label
        else:
            _reverted(files)
            row["outcome"] = ("reverted: the unit tests fail" if not passed
                              else "reverted: the measure is not better")
            # its code as it was, for the next round
            _restart()
        row["seconds"] = round(time.time() - started)
        _logged(row)
        rows.append(row)
        print(json.dumps({key: row.get(key) for key in (
            "round", "outcome", "changed", "measured", "tests")}),
            flush=True)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--baseline", default=None)
    options = parser.parse_args(argv)
    for row in run(options.rounds, options.baseline):
        print(row["round"], row["outcome"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
