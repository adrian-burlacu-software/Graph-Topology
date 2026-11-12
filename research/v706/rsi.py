"""The loop (v706): it learns ways from what it got wrong, and what it learnt
is kept only where measuring says it is better.

    python -m research.v706.rsi --baseline m1-seen m1-held m1-core --label r1

1. **Learn** (`learn`): from the failures of the SEEN measure, and from its
   own questions it gets wrong (`curriculum`) -- by search, no model.
2. **Measure**, one server of the code as it now is: SEEN, where it must gain
   and lose nothing; HELD, which it has never looked at, and the core, where
   it may lose nothing. Cheapest first, stopped at the first that fails.
3. **Keep or revert**: `ways.json` committed where all hold, put back as it
   was where one does not. What HELD gains chooses nothing: it is the
   evidence that what was learnt is general.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request

from research.v706 import learn, ways

PORT = 8697
URL = f"http://127.0.0.1:{PORT}"
#: how long the server may take to come up, and a measure to run, seconds
UP = 900
MEASURED = {"seen": 1800, "held": 1800, "core": 5400}
SUITES = {"seen": ["--suite", "probes"], "held": ["--suite", "probes-held"],
          "core": ["--core"]}


def verdicts(label: str) -> dict:
    path = learn.OUT / f"{label}.jsonl"
    return {json.dumps(row["turns"]): row["verdict"] for row in
            map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def compared(before: dict, after: dict) -> dict:
    gained = sorted(key for key, verdict in after.items()
                    if verdict == "right" and before.get(key) != "right")
    lost = sorted(key for key, verdict in before.items()
                  if verdict == "right" and after.get(key) != "right")
    return {"right before": sum(v == "right" for v in before.values()),
            "right after": sum(v == "right" for v in after.values()),
            "gained": gained, "lost": lost}


def _up() -> bool:
    try:
        urllib.request.urlopen(URL, timeout=5)
        return True
    except Exception:                               # noqa: BLE001
        return False


def served():
    """A server of the code as it is on disk, up; None where it never came."""
    if _up():
        raise SystemExit(f"a server is already on {PORT}: stop it first")
    log = (learn.OUT / "rsi-server.log").open("w", encoding="utf-8")
    server = subprocess.Popen(
        [sys.executable, "-m", "research.v698.server", "--port", str(PORT)],
        cwd=learn.ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
    end = time.monotonic() + UP
    while time.monotonic() < end:
        if _up():
            return server
        if server.poll() is not None:
            return None
        time.sleep(10)
    server.kill()
    return None


def measured(label: str, which: str) -> dict:
    subprocess.run([sys.executable, "-m", "research.v704.converse",
                    "--url", URL, "--label", label, *SUITES[which]],
                   cwd=learn.ROOT, stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   timeout=MEASURED[which], check=False)
    return verdicts(label)


def judged(label: str, baseline: dict) -> dict:
    """SEEN, HELD and the core measured, as far as each holds: {better, seen,
    held, core}."""
    out = {"better": False}
    server = served()
    if server is None:
        out["failed"] = "the server never came up"
        return out
    try:
        for which in ("seen", "held", "core"):
            found = compared(verdicts(baseline[which]),
                             measured(f"{label}-{which}", which))
            out[which] = found
            if found["lost"] or (which == "seen" and not found["gained"]):
                return out
        out["better"] = True
        return out
    finally:
        server.kill()


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs=3, required=True,
                        metavar=("SEEN", "HELD", "CORE"))
    parser.add_argument("--label", default=time.strftime("rsi-%Y%m%d-%H%M"))
    parser.add_argument("--curriculum", action="store_true")
    options = parser.parse_args(argv)
    baseline = dict(zip(("seen", "held", "core"), options.baseline))
    before = ways.PATH.read_text(encoding="utf-8") if ways.PATH.exists() \
        else None
    more = []
    if options.curriculum:
        from research.v706 import curriculum
        more = curriculum.failures()
    report = {"learn": learn.learn(baseline["seen"], more)}
    count = len(report["learn"]["learnt"])
    if count:
        report["judged"] = judged(options.label, baseline)
        if report["judged"]["better"]:
            said = ", ".join(
                f"{which} {report['judged'][which]['right before']}->"
                f"{report['judged'][which]['right after']}"
                for which in ("seen", "held", "core"))
            subprocess.run(["git", "add", str(ways.PATH)], cwd=learn.ROOT)
            subprocess.run(
                ["git", "commit", "-q", "-m",
                 f"v706: {count} ways learnt by search from its own "
                 f"failures, kept by measure ({said}; nothing lost)\n\n"
                 f"Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>"],
                cwd=learn.ROOT)
            report["kept"] = True
        else:
            if before is None:
                ways.PATH.unlink(missing_ok=True)
            else:
                ways.PATH.write_text(before, encoding="utf-8")
            report["kept"] = False
    (learn.OUT / f"{options.label}.rsi.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
