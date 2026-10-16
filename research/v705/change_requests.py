"""Changes asked in words, measured end to end (v705): the request read
(`reading`), placed (`fixing.placed`), made (`fixing.change`) in a copy of
the file, and the function it leaves compared with the one the
repository's history wrote for it -- the change a person asked for, made
as they would have.

    python -m research.v705.change_requests     state/v705-change-requests.json

Each case is a function of this repository as it was before a commit
changed it (`commits` mined), and what a person would say to ask for that
change; a change made is right where the function is then the commit's
own (`truth._same_code`).
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from research.v705 import commits

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "state" / "v705-change-requests.json"
#: (file, function, the request as a person says it): each a change the
#: history made, of a kind the reader reads (`teach_change_talk`)
CASES = [
    ("research/v696/editing.py", "repair",
     "In editing.py, repair should hand python bugs to _repair_python."),
    ("research/v696/forms.py", "_sub_spec",
     "_sub_spec should pass spec.language on to Spec."),
    ("research/v696/search.py", "Solver._ops",
     "In Solver._ops, give library the spec.language."),
    ("research/v696/experiment.py", "_passes",
     "_passes should pass spec.language to checker."),
]


def _before(path: str, function: str) -> str | None:
    """The function as it was before the commit that changed it last in
    the history mined (`commits.MINED`)."""
    found = None
    for line in commits.MINED.open(encoding="utf-8"):
        one = json.loads(line)
        if one["file"] == path and one["function"] == function:
            found = one["before"]
    return found


def run() -> dict:
    from research.v698 import asking, reading
    from research.v698.project import Project
    from research.v700 import fixing
    from research.v705.truth import _same_code
    rows, counts = [], {"cases": 0, "read": 0, "placed": 0, "changed": 0,
                        "right": 0}
    for path, function, said in CASES:
        before = _before(path, function)
        disk = (ROOT / path).read_text(encoding="utf-8")
        true = commits.functions(disk).get(function)
        if before is None or true is None:
            continue
        counts["cases"] += 1
        row = {"said": said, "file": path, "function": function}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(disk.replace(true, before),
                                     encoding="utf-8")
            held = Project("change-requests", str(root))
            held.put({path: (root / path).read_text(encoding="utf-8")})
            known = {one["name"] for one in held.functions()} | {
                one["name"].split(".")[-1] for one in held.functions()} | {
                path, path.rsplit("/", 1)[-1]}
            read = reading.read(said, known)
            row["read"] = None if read is None else {
                "act": read.act, "chance": round(read.chance, 3),
                "spans": read.spans}
            if read is None or read.act != "change":
                rows.append(row)
                continue
            counts["read"] += 1
            subject = fixing.placed(read, held, said) or asking.resolve(
                read, held, None, 0, "requests")
            row["placed"] = None if subject is None else subject.name
            if subject is None or subject.name.split(".")[-1] != \
                    function.split(".")[-1]:
                rows.append(row)
                continue
            counts["placed"] += 1
            made = fixing.change(said, held, subject, "requests",
                                 spans=read.spans)
            row["status"], row["said back"] = made["status"], made["said"]
            row["diff"] = made.get("diff") or made.get("guess")
            if made["status"] in ("changed", "partly"):
                counts["changed"] += 1
                now = commits.functions((root / path).read_text(
                    encoding="utf-8")).get(function, "")
                row["right"] = _same_code(now, true)
                counts["right"] += row["right"]
            shutil.rmtree(root / "research", ignore_errors=True)
        rows.append(row)
        print(json.dumps({key: row.get(key) for key in (
            "function", "read", "placed", "status", "right")}), flush=True)
    OUT.write_text(json.dumps({**counts, "each": rows}, indent=1),
                   encoding="utf-8")
    print(json.dumps(counts))
    return counts


if __name__ == "__main__":
    run()
    sys.exit(0)
