"""Teaching the shared reader what asks for debugging (v705), in Python's
and TypeScript's own spelling.

    python -m research.v705.teach_debug_talk write    the teacher writes requests
    python -m research.v705.teach_debug_talk check    yes or no, each
    python -m research.v705.teach_debug_talk corpus   llm/code-talk-data/*-debug

The architecture traces a call and fixes from what the run shows (v705
`debugging`, `teach_debugging`); a person asks for it in their own words:
"`oldest([])` raises a TypeError but should give None", "why does
due_date(date(2026, 1, 1)) return the 15th?", "debug parse('1,2')".

**A code-talk act, `debug`**, the call marked `CALL`, what it gave `GOT`,
what it should have given `WANT`. **By construction**: the teacher
(SmolLM3, offline) writes requests in each form with `{CALL}`, `{GOT}`,
`{WANT}` where they go -- it never sees a call, so it invents none --
checked yes-or-no; each filled with real calls and values of the bugs
made for teaching (`teach_debugging.bugs`): MBPP's for training, HumanEval's
held out, in both languages, each written as its language writes it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys

from research.v698 import teach_code_talk as K

DATA = K.DATA
WRITTEN = DATA / "debug-written.jsonl"
CHECKED = DATA / "debug-checked.jsonl"
SEED = 7051
#: each form a request takes: the slots it names, and how it is asked
FORMS = {
    "report": ("{CALL}, {GOT}, {WANT}",
               "says that calling {CALL} gives {GOT} but should give "
               "{WANT}, and wants it fixed or looked into -- like "
               "'{CALL} returns {GOT}, expected {WANT}', 'why does {CALL} "
               "give {GOT} instead of {WANT}?', '{CALL} should be {WANT} "
               "but I get {GOT}'"),
    "fails": ("{CALL}",
              "says that calling {CALL} crashes, hangs or gives the wrong "
              "answer, without saying what it gives -- like '{CALL} "
              "crashes', '{CALL} is wrong, can you fix it?', 'something "
              "is off with {CALL}'"),
    "why": ("{CALL}, {GOT}",
            "asks why calling {CALL} gives {GOT} -- like 'why does {CALL} "
            "return {GOT}?', 'how come {CALL} is {GOT}?'"),
    "trace": ("{CALL}",
              "asks the assistant to debug, trace or step through {CALL} -- "
              "like 'debug {CALL}', 'step through {CALL} for me', 'can you "
              "trace {CALL}?'"),
}
QUESTION = ("Is the person saying that some code gives a wrong result or "
            "fails, or asking for it to be debugged?")
#: requests asked of each form, in rounds of two samples
EACH, ROUNDS = 12, 8
#: the calls each kept request is filled with (three were too few beside
#: code talk's thousands: reader-code38 read 3 of 6 requests as debug)
FILLED = 8
#: how often what a call gave, or should give, is an expression
EXPRESSIONS = 0.35
VALID = K.VALID


def write() -> None:
    from research.v696.teach_meaning import Teacher
    teacher = Teacher()
    teacher.torch.manual_seed(SEED)
    with WRITTEN.open("w", encoding="utf-8") as out:
        for round_ in range(ROUNDS):
            forms = list(FORMS)
            prompts = [
                f"Write {EACH} different messages a programmer types to an "
                f"assistant that works on their code, in which the person "
                f"{FORMS[form][1]}. Write {FORMS[form][0]} exactly where "
                f"they go -- do not fill them in. Vary them a lot: short "
                f"and long, casual and formal. One per line, nothing else."
                for form in forms]
            replies = teacher.write(prompts, longest=600, samples=2,
                                    temperature=0.95)
            for form, written in zip(forms, replies):
                wanted = set(_slots(FORMS[form][0]))
                lines = []
                for reply in written:
                    for line in reply.splitlines():
                        line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line)
                        line = line.strip().strip("\"'“”")
                        if set(_slots(line)) == wanted and all(
                                line.count("{" + one + "}") == 1
                                for one in wanted) and \
                                not re.search(r"\{(?!CALL\}|GOT\}|WANT\})",
                                              line) and \
                                8 <= len(line) <= 220:
                            lines.append(line)
                out.write(json.dumps({"form": form, "lines": lines}) + "\n")
            print(f"round {round_ + 1} of {ROUNDS}", flush=True)


def _slots(text: str) -> list:
    # by their own marks: a reply may hold braces of its own (JSON), which
    # a format string refuses
    return re.findall(r"\{(CALL|GOT|WANT)\}", text)


def check(batch: int = 24) -> dict:
    from research.v696.teach_meaning import Teacher
    rows = [json.loads(line) for line in WRITTEN.open(encoding="utf-8")]
    todo = [(row["form"], line) for row in rows for line in row["lines"]]
    teacher = Teacher()
    counts: dict = {}
    with CHECKED.open("w", encoding="utf-8") as out:
        for at in range(0, len(todo), batch):
            chunk = todo[at:at + batch]
            replies = teacher.write(
                [f"A programmer typed this to an assistant: \"" +
                 line.replace("{CALL}", "total([1, 2])").replace(
                     "{GOT}", "4").replace("{WANT}", "3") +
                 f"\"\n{QUESTION} Answer yes or no only."
                 for _, line in chunk], longest=3)
            for (form, line), reply in zip(chunk, replies):
                kept = reply[0].strip().lower().startswith("yes")
                key = f"{form} {'kept' if kept else 'left'}"
                counts[key] = counts.get(key, 0) + 1
                out.write(json.dumps({"form": form, "template": line,
                                      "kept": kept}) + "\n")
    print(json.dumps(counts))
    return counts


def _written(value, language: str) -> str:
    """A value as its language writes it."""
    if language == "python":
        return repr(value)
    return json.dumps(value)


def calls(split: str) -> list:
    """(call, got, wanted) of the bugs made for teaching (`train`) or held
    (`held`), each in its language's spelling."""
    from research.v705.teach_debugging import DATA as BUGS
    out = []
    for language in ("python", "typescript"):
        path = BUGS / f"bugs-{language}-{split}.jsonl"
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            bug = json.loads(line)
            args = ", ".join(_written(one, language) for one in bug["case"])
            call = f"{bug['entry']}({args})"
            if len(call) > 80:
                continue
            got = bug["got"]
            gave = ("an error" if "error" in got else
                    _written(got.get("value"), language))
            out.append((call, gave, _written(bug["wanted"], language)))
    return out


def corpus() -> dict:
    """The kept requests as code-talk records: act `debug`, the call, what
    it gave and what it should have marked; the split by the request as
    written, so a template is never in both, and the values by the bugs'
    own split."""
    rows = [json.loads(line) for line in CHECKED.open(encoding="utf-8")]
    kept = [one for one in rows if one["kept"]]
    rng = random.Random(SEED)
    values = {"train": calls("train"), "valid": calls("held")}
    out = {"train": [], "valid": []}
    for one in kept:
        split = "valid" if hashlib.sha1(one["template"].lower().encode()
                                        ).digest()[0] / 256 < VALID \
            else "train"
        for call, got, want in rng.sample(values[split],
                                          min(FILLED, len(values[split]))):
            # a value may be an expression as well as a literal: another
            # call's text (`date(2026, 1, 22)`; the bugs' values are
            # literals, and reader-code39 never marked what a call should
            # give when it was one)
            pool = values[split]
            if rng.random() < EXPRESSIONS:
                got = rng.choice(pool)[0]
            if rng.random() < EXPRESSIONS:
                want = rng.choice(pool)[0]
            filled = {"CALL": call, "GOT": got, "WANT": want}
            said_text = one["template"]
            for name, value in filled.items():
                said_text = said_text.replace("{" + name + "}", value)
            said = K.words(said_text)
            roles = ["O"] * len(said)
            fine = True
            for name in _slots(one["template"]):
                found = K._match(said, filled[name])
                if found is None:
                    fine = False
                    break
                begin, end = found
                for at in range(begin, end):
                    roles[at] = ("B-" if at == begin else "I-") + name
            if not fine or not said:
                continue
            out[split].append(K.record(said, roles, "debug", "none", "none",
                                       f"debug {one['form']}"))
    counts = {}
    for split, records in out.items():
        rng.shuffle(records)
        with (DATA / f"{split}-debug.jsonl").open(
                "w", encoding="utf-8") as stream:
            for one in records:
                stream.write(json.dumps(one) + "\n")
        counts[split] = len(records)
    counts["kept templates"] = len(kept)
    print(json.dumps(counts))
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("write", "check", "corpus"))
    options = parser.parse_args(argv)
    {"write": write, "check": check, "corpus": corpus}[options.job]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
