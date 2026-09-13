"""Run the test suites in parallel, one process per module.

    python -m research.run_tests
    python -m research.run_tests --workers 3 --only v688

Serially the suites take about five and a half minutes, and almost none of
that is the assertions. Every test class builds its own `ReasoningEngine`,
which loads spaCy's model -- 916 MB and about six seconds -- and v688 builds
a pool of five. The work is startup, and startup is what parallelises.

Module is the right grain. The classes inside a module share one engine
through `setUpClass`, so splitting finer would rebuild it per class and
spend more on loading than it saved on waiting. Splitting coarser leaves the
longest module setting the wall clock, which it does anyway:

    test_reasoning   147 tests   89s
    test_v687         65 tests   60s
    test_norms        59 tests   48s
    test_bridging     35 tests   20s
    test_trie         32 tests    0s
    v688/test_v688    66 tests  100s

Nothing shared has to be locked. Each process opens the store read-only and
writes nothing, so the modules are independent by construction rather than
by convention -- and if that ever stops being true, this runner is where it
will show up first, as a module that only fails when it has company.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

#: Longest first. With fewer workers than modules the schedule matters, and
#: starting the 89-second one last is how a 100-second run becomes 150.
SUITES = ("research.v687.test_reasoning", "research.v688.test_v688",
          "research.v687.test_v687", "research.v687.test_norms",
          "research.v687.test_bridging", "research.v687.test_trie",
          "research.v687.test_within", "research.v687.test_shapes",
          "research.v687.test_rated",
          "research.v688.test_audit", "research.v688.test_screen",
          "research.v688.test_distil", "research.v688.test_content",
          "research.v688.test_rephrase",
          "research.v689.test_v689")

COUNT = re.compile(r"^Ran (\d+) test")


def run_one(module: str) -> tuple[str, int, int, float, str]:
    """(module, tests, returncode, seconds, output)."""
    started = time.time()
    done = subprocess.run(
        [sys.executable, "-m", "unittest", module, "-q"],
        capture_output=True, text=True, cwd=ROOT,
        encoding="utf-8", errors="replace")
    output = (done.stdout or "") + (done.stderr or "")
    tests = 0
    for line in output.splitlines():
        found = COUNT.match(line)
        if found:
            tests = int(found.group(1))
    return module, tests, done.returncode, time.time() - started, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4,
                        help="processes at once (default 4). Each holds a "
                             "spaCy model, so this is bounded by memory "
                             "rather than by cores.")
    parser.add_argument("--only", default="",
                        help="substring: run only matching modules")
    options = parser.parse_args()

    wanted = [name for name in SUITES if options.only in name]
    if not wanted:
        print(f"no suite matches {options.only!r}")
        return 2

    started = time.time()
    failed: list[tuple[str, str]] = []
    total = 0
    print(f"{len(wanted)} suite(s), {options.workers} at a time\n")
    with concurrent.futures.ThreadPoolExecutor(options.workers) as pool:
        for module, tests, code, seconds, output in pool.map(run_one, wanted):
            total += tests
            mark = "ok  " if code == 0 else "FAIL"
            print(f"  {mark} {module:34} {tests:4} tests  {seconds:6.1f}s")
            if code != 0:
                failed.append((module, output))

    elapsed = time.time() - started
    print(f"\n{total} tests in {elapsed:.1f}s "
          f"({'all passed' if not failed else f'{len(failed)} suite(s) failed'})")
    for module, output in failed:
        print(f"\n{'=' * 70}\n{module}\n{'=' * 70}")
        print(output.strip()[-4000:])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
