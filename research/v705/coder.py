"""A code model at run time (v705): Qwen2.5-Coder-1.5B-Instruct, asked to
fix a function of its own code as the teacher was (`commits.FIX`) -- the
goal, the function, the failing test and how it fails, what is beside it
-- each answer judged by the tests and those beside them.

    python -m research.v705.coder measure     the 40 held-out history bugs

Every model the architecture taught or asked so far wrote code at 0 to 2
of 40 of its history's bugs with the place given (editor-debug2's two were
overfits). Asked first as it came: taught only if it writes something.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from research.v696.teach_meaning import Teacher
from research.v705 import commits

ROOT = Path(__file__).resolve().parents[2]
CODER = ROOT / "llm" / "Qwen2.5-Coder-1.5B-Instruct"
OUT = ROOT / "state" / "v705-coder-measure.json"
#: the answers written for each bug
TRIES = 8


class Coder(Teacher):
    """The code model in bf16 (3 GB), writing in batches as the teacher
    does."""

    def __init__(self, path: Path = CODER) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path), device_map="cuda", dtype=torch.bfloat16)
        self.model.eval()


_LOADED: dict = {}

#: what the code model is told of a change any case asks for
CHANGE = ("A function of a Python project should change. What is wanted: "
          "{goal}\nIn {path}, the function as it is:\n```python\n{before}\n"
          "```\n{shown}\nIn the same file, beside it:\n{reach}\nWrite the "
          "whole changed function -- the same name and parameters -- in one "
          "```python block, nothing else.")


def get() -> Coder:
    """The code model, loaded once."""
    if "coder" not in _LOADED:
        _LOADED["coder"] = Coder()
    return _LOADED["coder"]


def _shown(case: dict) -> str:
    """What shows the function wrong, as the case has it."""
    if case.get("tests"):
        test = case["tests"][0]
        source = commits.functions((ROOT / test["file"]).read_text(
            encoding="utf-8")).get(test["test"], "")
        return (f"This test fails:\n```python\n{source}\n```\nIt fails "
                f"with: {commits.failure(test)}")
    if case.get("measure"):
        one = case["measure"]
        way = "less" if one["way"] == "min" else "more"
        return (f"Told \"{case['said']}\", the program replies:\n"
                f"{case.get('answer', '')}\nThe reply should have {way} "
                f"{one['measure']}, saying the same.")
    return (f"Told \"{case['said']}\", the program answers "
            f"\"{(case.get('answer') or '')[:300]}\", which is wrong.")


def written(goal: str, case: dict, place: dict, function, text: str
            ) -> list:
    """The code model's answers as changed texts of the file:
    [(text, what)] -- each the function of its name, put in where it
    stands."""
    before = commits.functions(text).get(place["function"])
    if not before:
        return []
    asked = CHANGE.format(goal=goal, path=place["file"], before=before,
                          shown=_shown(case),
                          reach="\n".join(commits.reach(
                              {"file": place["file"],
                               "function": place["function"]})))
    replies = get().write([asked], longest=900, samples=TRIES,
                          temperature=0.8)[0]
    out, seen = [], set()
    for reply in replies:
        code = commits._code_of(reply)
        changed = commits._placed(code, before, text) if code else None
        if changed and changed != text and changed not in seen:
            seen.add(changed)
            out.append((changed, f"answer {len(out) + 1}"))
    return out


def measure(most: int = 0) -> dict:
    """Of the bugs editors are measured on (`commits.MEASURED`), each fixed
    where one of `TRIES` answers makes the failing tests pass and those
    beside them still pass."""
    import time
    mined = {(one["sha"], one["function"]): one for one in
             map(json.loads, commits.MINED.open(encoding="utf-8"))}
    bugs = [one for one in map(json.loads,
                               commits.CHECKED.open(encoding="utf-8"))
            if one["status"] == "bug"][:most or commits.MEASURED]
    coder = Coder()
    counts = {"bugs": 0, "fixed": 0, "answers": 0, "compiled": 0}
    out = []
    for bug in bugs:
        started = time.time()
        case = {**mined[(bug["sha"], bug["function"])], "tests": bug["fails"]}
        function = commits._current(case)
        made = commits.planted(case)
        disk = (ROOT / case["file"]).read_text(encoding="utf-8")
        old = disk.replace(commits.functions(disk)[case["function"]],
                           case["before"])
        test = case["tests"][0]
        source = commits.functions((ROOT / test["file"]).read_text(
            encoding="utf-8")).get(test["test"], "")
        why = []
        with commits.Put(function, made):
            failed = commits.failure(test)
            replies = coder.write([commits.FIX.format(
                message=case["message"], path=case["file"],
                before=case["before"], test=source, failure=failed,
                reach="\n".join(commits.reach(case)))],
                longest=900, samples=TRIES, temperature=0.8)[0]
            verify = commits.verifier(case, commits._current(case))
            seen = set()
            for reply in replies:
                counts["answers"] += 1
                code = commits._code_of(reply)
                text = commits._placed(code, case["before"], old) \
                    if code else None
                if text is None:
                    why.append("no function of its name")
                    continue
                counts["compiled"] += 1
                if text in seen or text == old:
                    why.append("the same again" if text in seen
                               else "unchanged")
                    continue
                seen.add(text)
                said = verify(text)
                why.append(said or "holds")
                if said is None:
                    break
        fixed = "holds" in why
        counts["bugs"] += 1
        counts["fixed"] += fixed
        out.append({"sha": case["sha"], "function": case["function"],
                    "fixed": fixed, "why": why,
                    "seconds": round(time.time() - started),
                    "fix": next((reply for reply, said in zip(replies, why)
                                 if said == "holds"), None)})
        print(f"{case['sha'][:7]} {case['function']}: "
              f"{'FIXED' if fixed else 'no'} ({why[:3]}) "
              f"{round(time.time() - started)}s", flush=True)
    OUT.write_text(json.dumps({**counts, "each": out}, indent=1),
                   encoding="utf-8")
    print(json.dumps(counts))
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("measure",))
    parser.add_argument("--most", type=int, default=0)
    options = parser.parse_args(argv)
    measure(options.most)
    return 0


if __name__ == "__main__":
    sys.exit(main())
