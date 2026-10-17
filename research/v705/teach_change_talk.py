"""Teaching the shared reader what a change asks to be made (v705): its
kind, by the words it marks -- a function taking something new, a value
passed on to a call, a case handed to a helper.

    python -m research.v705.teach_change_talk write    the teacher writes requests
    python -m research.v705.teach_change_talk check    yes or no, each
    python -m research.v705.teach_change_talk corpus   llm/code-talk-data/*-change

The change path read "gained should take a message, empty by default" as
a change of `gained` -- and its editors wrote twenty changes, none of the
`def` line; a change of the right kind was there to be made (`shaped`:
a parameter, a value passed on, a case handed to a helper), and nothing
read which kind was asked. The reader reads it, as it reads a debugging
request's call and values (`teach_debug_talk`): **roles** for what the
change names -- `NEW` what the function is to take, `DEFAULT` what it is
when not given, `VALUE` what is passed on, `CALLEE` the call it is passed
to, `CASE` the case a helper is for, `HELPER` the helper -- the function
changed `SUBJ`, the act `change`. **By construction**: the teacher
(SmolLM3, offline) writes requests in each form with the slots where they
go, never filled -- it invents no names -- checked yes-or-no; each filled
with names this repository's own code uses: its functions, their
parameters and defaults, the values its calls are given, the cases its
code compares, its helpers; a template in train or valid by its own hash.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from research.v698 import teach_code_talk as K

ROOT = Path(__file__).resolve().parents[2]
DATA = K.DATA
WRITTEN = DATA / "change-written.jsonl"
CHECKED = DATA / "change-checked.jsonl"
SEED = 7052
#: each form a request takes: the slots it names, what the person asks,
#: and the question its check asks
FORMS = {
    "takes": ("{SUBJ}, {NEW}",
              "asks that the function {SUBJ} take something new, {NEW}, as "
              "an input -- like 'make {SUBJ} take {NEW}', '{SUBJ} should "
              "accept {NEW} too', 'add a {NEW} parameter to {SUBJ}'",
              "asking that a function get a new input or parameter"),
    "takes default": ("{SUBJ}, {NEW}, {DEFAULT}",
                      "asks that the function {SUBJ} take something new, "
                      "{NEW}, which is {DEFAULT} when it is not given -- like "
                      "'{SUBJ} should take {NEW}, {DEFAULT} by default', "
                      "'give {SUBJ} an optional {NEW} defaulting to "
                      "{DEFAULT}'",
                      "asking that a function get a new input with a "
                      "default value"),
    # what is taken said with what it is for (`gained should also take
    # the commit message and score each window after it, so ...`: read
    # with `window` the call, and nothing taken)
    "takes and uses": ("{SUBJ}, {NEW}",
                       "asks that the function {SUBJ} take something new, "
                       "{NEW}, and says in the same message what {SUBJ} "
                       "should do with it and why -- like 'make {SUBJ} take "
                       "{NEW} and print it before each line, so the log "
                       "says whose it is', '{SUBJ} should also take {NEW} "
                       "and skip the rows it names'",
                       "asking that a function get a new input and use it"),
    "passes": ("{SUBJ}, {VALUE}, {CALLEE}",
               "asks that in the function {SUBJ}, the call to {CALLEE} be "
               "given {VALUE} -- like 'in {SUBJ}, pass {VALUE} to {CALLEE}', "
               "'{SUBJ} should give {CALLEE} the {VALUE}', 'have {SUBJ} "
               "hand {VALUE} on to {CALLEE}'",
               "asking that a value be passed on to a call inside a "
               "function"),
    "hands": ("{SUBJ}, {CASE}, {HELPER}",
              "asks that the function {SUBJ} hand the case {CASE} to the "
              "helper {HELPER} -- like 'when it is {CASE}, {SUBJ} should "
              "use {HELPER}', 'for {CASE}, have {SUBJ} call {HELPER} "
              "instead', '{SUBJ} should send {CASE} to {HELPER}'",
              # said of the example filled in: asked abstractly, the check
              # left 140 of 155 out, as plain as 'for {CASE}, have {SUBJ}
              # call {HELPER} instead'
              "asking that {SUBJ} call {HELPER} when it is {CASE}"),
}
#: the roles of what a change names, after the reader's own (`K.ROLES`)
ROLES = ("NEW", "DEFAULT", "VALUE", "CALLEE", "CASE", "HELPER")
SLOTS = ("SUBJ",) + ROLES
#: requests asked of each form, in rounds of two samples
EACH, ROUNDS = 12, 8
#: the fillings each kept request is given (six were too few beside the
#: reader's 450 000 records: reader-code44 marked no role of a change)
FILLED = 30
VALID = K.VALID
#: how often a request says first which file the function is in
IN_FILE = 0.4
#: how many times as often a request to take something is filled
TAKES_MORE = 3
#: how often what a function is to take is said in words, after an article
WORDED = 0.5
#: the roles naming what the project has: known to be code where it is read
KNOWN_ROLES = ("SUBJ", "CALLEE", "HELPER")


def write(only: list | None = None) -> None:
    """Every form's requests written again -- or `only` those forms', put
    after what was written (a form added: the others as they were)."""
    from research.v696.teach_meaning import Teacher
    teacher = Teacher()
    teacher.torch.manual_seed(SEED + (len(only) if only else 0))
    with WRITTEN.open("a" if only else "w", encoding="utf-8") as out:
        for round_ in range(ROUNDS):
            forms = list(only or FORMS)
            prompts = [
                f"Write {EACH} different messages a programmer types to an "
                f"assistant that changes their code, in which the person "
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
                                not re.search(r"\{(?!(%s)\})" % "|".join(
                                    SLOTS), line) and 8 <= len(line) <= 220:
                            lines.append(line)
                out.write(json.dumps({"form": form, "lines": lines}) + "\n")
            print(f"round {round_ + 1} of {ROUNDS}", flush=True)


def _slots(text: str) -> list:
    return re.findall(r"\{(%s)\}" % "|".join(SLOTS), text)


#: the kinds a request is checked against, and each form's
CHOICES = ("that the function parse take a new input or parameter",
           "that inside parse, a value be passed on to another call",
           "that parse call another function for one particular case",
           "something else")
KIND_OF = {"takes": 0, "takes default": 0, "takes and uses": 0,
           "passes": 1, "hands": 2}
#: what each slot is shown as when a request is checked
SHOWN = {"SUBJ": "parse", "NEW": "language", "DEFAULT": "None",
         "VALUE": "spec.language", "CALLEE": "checker", "CASE": "python",
         "HELPER": "_parse_python"}


def check(batch: int = 24) -> dict:
    from research.v696.teach_meaning import Teacher
    rows = [json.loads(line) for line in WRITTEN.open(encoding="utf-8")]
    todo = [(row["form"], line) for row in rows for line in row["lines"]]
    teacher = Teacher()
    counts: dict = {}
    with CHECKED.open("w", encoding="utf-8") as out:
        for at in range(0, len(todo), batch):
            chunk = todo[at:at + batch]
            prompts = []
            for form, line in chunk:
                shown = line
                for name, value in SHOWN.items():
                    shown = shown.replace("{" + name + "}", value)
                # which of the kinds it asks, as letters: asked yes or no of
                # one kind, the teacher said no to 'make parse take
                # language' and to 'for python, have parse call
                # _parse_python instead' alike
                prompts.append(f"A programmer typed this to an assistant: "
                               f"\"{shown}\"\nWhat does the person ask "
                               f"for?\n" + "\n".join(
                                   f"{letter}) {said}" for letter, said in
                                   zip("ABCD", CHOICES)) +
                               "\nAnswer with the letter only.")
            replies = teacher.write(prompts, longest=3)
            for (form, line), reply in zip(chunk, replies):
                letter = reply[0].strip()[:1].upper()
                kept = letter == "ABCD"[KIND_OF[form]]
                key = f"{form} {'kept' if kept else 'left'}"
                counts[key] = counts.get(key, 0) + 1
                out.write(json.dumps({"form": form, "template": line,
                                      "kept": kept}) + "\n")
    print(json.dumps(counts))
    return counts


def names() -> dict:
    """What this repository's own code names, by slot: its functions, their
    parameters and defaults, the values its calls are given (a name, or a
    field of one), the calls, the strings it compares, its helpers."""
    out = {slot: set() for slot in SLOTS}
    for path in sorted((ROOT / "research").rglob("*.py")):
        if "test" in path.name:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if len(node.name) >= 3:
                    out["SUBJ"].add(node.name)
                if node.name.startswith("_") and not node.name.startswith(
                        "__"):
                    out["HELPER"].add(node.name)
                for one in node.args.args + node.args.kwonlyargs:
                    if one.arg not in ("self", "cls") and len(one.arg) >= 3:
                        out["NEW"].add(one.arg)
                for one in node.args.defaults + node.args.kw_defaults:
                    if isinstance(one, ast.Constant) and len(repr(
                            one.value)) <= 12:
                        out["DEFAULT"].add(repr(one.value))
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    out["CALLEE"].add(node.func.id)
                for one in node.args + [one.value for one in node.keywords]:
                    said = ast.unparse(one)
                    if isinstance(one, (ast.Name, ast.Attribute)) and \
                            len(said) <= 24 and said.count(".") <= 1:
                        out["VALUE"].add(said)
            elif isinstance(node, ast.Compare) and len(node.ops) == 1 and \
                    isinstance(node.comparators[0], ast.Constant) and \
                    isinstance(node.comparators[0].value, str) and \
                    re.fullmatch(r"[a-z]{3,12}", node.comparators[0].value):
                out["CASE"].add(node.comparators[0].value)
    return {slot: sorted(found) for slot, found in out.items()}


#: what a value is passed on to said as people go on to say it (`pass
#: spec.language on to Spec`: one kept template in 81 said `on to`, and
#: reader-code47 read the value as a parameter taken)
ON_TO = ("on to", "along to", "through to")
#: the verbs a call can be given a value with, the call said first (`give
#: library the spec.language`: three templates said the call first)
GIVEN = ("pass", "give", "hand", "send", "provide")


def _said_otherwise(template: str) -> list:
    """A request to pass a value on, said the other ways people say it:
    its `to` as `on to` and the like, and the call before the value."""
    out = []
    joined = re.search(r"\{VALUE\}\s+to\s+\{CALLEE\}", template)
    if joined:
        for said in ON_TO:
            out.append(template[:joined.start()] + "{VALUE} " + said +
                       " {CALLEE}" + template[joined.end():])
    first = re.search(r"\b(%s)\s+(?:the\s+)?\{VALUE\}\s+(?:on\s+|over\s+)?"
                      r"(?:to|into)\s+\{CALLEE\}" % "|".join(GIVEN), template,
                      re.IGNORECASE)
    if first:
        verb = first.group(1).lower()
        out.append(template[:first.start()] + ("give" if verb in (
            "pass", "provide") else verb) + " {CALLEE} the {VALUE}" +
            template[first.end():])
    return out


def corpus() -> dict:
    """The kept requests as code-talk records: act `change`, the function
    changed `SUBJ`, what the change names in its roles; the split by the
    request as written, the names by a hash of their own."""
    rows = [json.loads(line) for line in CHECKED.open(encoding="utf-8")]
    kept = [one for one in rows if one["kept"]]
    rng = random.Random(SEED)
    found = names()

    def split_of(text: str) -> str:
        return "valid" if hashlib.sha1(text.lower().encode()).digest()[0] / \
            256 < VALID else "train"
    pools = {split: {slot: [one for one in values if split_of(one) == split]
                     for slot, values in found.items()}
             for split in ("train", "valid")}
    files = {split: sorted({path.name for path in (ROOT / "research").rglob(
        "*.py") if split_of(path.name) == split}) for split in ("train",
                                                                "valid")}
    out = {"train": [], "valid": []}
    # each said otherwise as well, in its own split
    kept = kept + [{**one, "template": said, "split": split_of(
        one["template"])} for one in kept if one["form"] == "passes"
        for said in _said_otherwise(one["template"])]
    for one in kept:
        split = one.get("split") or split_of(one["template"])
        slots = _slots(one["template"])
        # `takes X` is said of every function asked to be written (`a
        # function that takes a list`, its X no role): what a change is to
        # take is taught more for it -- reader-code46 marked it in its own
        # templates and in no request beside; and a value passed on as
        # often (`_sub_spec should pass spec.language on to Spec` read as
        # one taken where it was not)
        for _ in range(FILLED * TAKES_MORE if one["form"] in (
                "takes", "takes default", "takes and uses", "passes") else FILLED):
            filled = {slot: rng.choice(pools[split][slot]) for slot in slots
                      if pools[split][slot]}
            if len(filled) != len(slots):
                break
            said_text = one["template"]
            # what a function is to take, said as people say it as often as
            # not: `take a message`, `the commit message` -- a filler was
            # always a name as code writes it, and reader-code47 marked no
            # NEW of `gained should take a message`
            if "NEW" in filled and rng.random() < WORDED:
                words_ = [one_ for one_ in filled["NEW"].split("_") if one_]
                if words_ and all(one_.isalpha() for one_ in words_):
                    filled["NEW"] = " ".join(words_)
                    article = rng.choice(("a ", "the ", "an " if words_[0][
                        0] in "aeiou" else "a "))
                    said_text = re.sub(r"(?<![A-Za-z])(a |an |the )?\{NEW\}",
                                       article + "{NEW}", said_text, count=1)
            for name, value in filled.items():
                said_text = said_text.replace("{" + name + "}", value)
            # where the function is, said first as people do: `In
            # natural.py, gained should take a message` -- never said so in
            # a template, and reader-code45 marked no role of a change once
            # it was
            put_first = None
            if files[split] and rng.random() < IN_FILE:
                put_first = rng.choice(files[split])
                said_text = f"In {put_first}, " + said_text[:1].lower() + \
                    said_text[1:]
            said = K.words(said_text)
            roles = ["O"] * len(said)
            if put_first is not None:
                span = K._match(said, put_first)
                if span is not None:
                    for at in range(*span):
                        roles[at] = ("B-" if at == span[0] else "I-") + "SUBJ"
            fine = True
            for name in slots:
                span = K._match(said, filled[name])
                if span is None or any(roles[at] != "O" for at in range(
                        *span)):
                    fine = False
                    break
                for at in range(*span):
                    roles[at] = ("B-" if at == span[0] else "I-") + name
            if not fine or not said:
                continue
            made = K.record(said, roles, "change", "none", "named",
                            f"change {one['form']}")
            # what the project has is known to be code (`reading._known`),
            # as the code corpus teaches it (`K.KNOWN`): the function, its
            # file, the call, the helper -- untagged, reader-code48 read
            # `parse should take language` with no NEW where parse was
            # known, and with it where it was not
            if rng.random() < K.KNOWN:
                made["tags"] = ["CODE" if role[2:] in KNOWN_ROLES else ""
                                for role in roles]
            out[split].append(made)
    counts = {}
    for split, records in out.items():
        rng.shuffle(records)
        with (DATA / f"{split}-change.jsonl").open(
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
    parser.add_argument("--forms", nargs="*", default=None,
                        help="write only these forms, after what is written")
    options = parser.parse_args(argv)
    if options.job == "write":
        write(options.forms)
    else:
        {"check": check, "corpus": corpus}[options.job]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
