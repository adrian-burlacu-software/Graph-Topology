"""Teaching the shared reader what is said back to a plan it holds (v704):
which of its options a reply picks, asks about, or answers what it did.

    python -m research.v704.teach_plan_talk write    the teacher writes replies
    python -m research.v704.teach_plan_talk check    and chooses what each is
    python -m research.v704.teach_plan_talk corpus   llm/code-talk-data/*-plans

Holding a plan, the architecture understood yes and no only. "ok, start by
looking at what fails in remembering" went to the change reader ("which
function or file is it in?"); "that change only added a blank line to
build" was answered about another `build`. Matched by meaning alone, the
plan's options came back wrong for "the routing problem" and "the dog
question", and feedback was taken for a pick.

**Three acts of code talk** (`teach_code_talk.ACTS`), each with the words
that point at an option marked `OPTION` (`ROLES`):

    pick      which option to start with, or do next  `let's start with
              the remembering one`
    query     a question about one                      `why did the dog
              question fail?`
    feedback  what the person makes of what it just did `that change to
              build added nothing`

**By construction**: the teacher (SmolLM3, offline) writes replies with
`{X}` where the option is named -- it never sees an option, so it invents
none -- kept where `{X}` is there once. Each is filled with references to
real options of this project's plans (`references`): a weakness by its
words, a step by its number or kind, a case by its words or what it is
about, a change by the function it changed. **Checked**: each filled
reply put back to the teacher as a choice among every code-talk act and
these three; kept where it chooses the act it was written as.
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
WRITTEN = DATA / "plans-written.jsonl"
CHECKED = DATA / "plans-checked.jsonl"
SEED = 7043
#: what is said back to a plan, as the teacher is asked to write it
ASKED = {
    "pick": "tells the assistant which one to start with or to do next",
    "query": "asks the assistant a short question about one of them -- "
             "why it fails, what it is, what went wrong with it, or what "
             "the assistant would do about it",
    "feedback": "reacts to what the assistant just did about it -- that it "
                "did nothing useful, changed the wrong thing, missed the "
                "point, or did well",
}
SITUATION = {
    "pick": "An assistant that works on software has just shown a person "
            "a plan: what it does worst, the steps it would take, and "
            "cases that fail now.",
    "query": "An assistant that works on software has just shown a person "
             "a plan: what it does worst, the steps it would take, and "
             "cases that fail now.",
    "feedback": "An assistant that works on software has just carried out "
                "a step of its plan and said what it did.",
}
#: the acts' choices, as the teacher is told them
CHOICES_ADDED = (
    ("pick", "choosing which of the options it was shown to start with or "
             "do next"),
    ("query", "asking about one of the options it was shown"),
    ("feedback", "reacting to what the assistant just did"),
)
#: replies asked of each act, in rounds of two samples
EACH, ROUNDS = 12, 6
VALID = K.VALID


def write(acts=tuple(ASKED), rounds: int = ROUNDS, seed: int = SEED,
          mode: str = "w") -> None:
    from research.v696.teach_meaning import Teacher
    teacher = Teacher()
    teacher.torch.manual_seed(seed)
    with WRITTEN.open(mode, encoding="utf-8") as out:
        for round_ in range(rounds):
            acts = list(acts)
            prompts = [
                f"{SITUATION[act]} Write {EACH} different replies in which "
                f"the person {ASKED[act]}, naming it as {{X}}. Write {{X}} "
                f"exactly, where the thing is named -- do not say what it "
                f"is. Vary them a lot: short and long, casual and formal, "
                f"questions and orders. One per line, nothing else."
                for act in acts]
            replies = teacher.write(prompts, longest=600, samples=2,
                                    temperature=0.95)
            for act, written in zip(acts, replies):
                lines = []
                for reply in written:
                    for line in reply.splitlines():
                        line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line)
                        line = line.strip().strip("\"'“”")
                        if line.count("{X}") == 1 and 8 <= len(line) <= 200:
                            lines.append(line)
                out.write(json.dumps({"act": act, "lines": lines}) + "\n")
            print(f"round {round_ + 1} of {rounds}", flush=True)


def references(rng: random.Random) -> dict:
    """What an option is called, by construction from this project's own
    plans: each weakness's words and the start of them, each step by its
    number and its kind, each case by its words and the thing it is about,
    each change by the function and file changed."""
    from research.v704 import history, selfmodel
    out: dict = {"pick": [], "query": [], "feedback": []}
    model = selfmodel.held() or {"areas": {}, "routing": []}
    whats = [one["what"] for one in model["areas"].values()] + \
        [selfmodel.ROUTING]
    for name in list(model["areas"]) + ["routing"]:
        said = name.replace("-", " ")
        out["pick"] += [said, f"the {said} one", f"the {said} problem",
                        f"{said} questions"]
    for what in whats:
        words_ = what.split(",")[0].split()
        out["pick"] += [what, " ".join(words_[:2]), " ".join(words_[:3]),
                        f"the {words_[0]} one", f"{words_[0]}"]
    for kind, said in history.SAYING.items():
        out["pick"] += [f"the {kind} step", said.split(",")[0].lower()]
    for number in range(1, 7):
        out["pick"] += [f"step {number}", f"number {number}",
                        f"option {number}"]
    cases = [one["said"].split("\n")[0] for area in model["areas"].values()
             for one in area["failed"]] + \
        [one["said"].split("\n")[0] for one in model["routing"]]
    nouns = _nouns(cases)
    for case, noun in zip(cases, nouns):
        out["pick"] += [f"\"{case}\""] + ([f"the {noun} question",
                                          f"the {noun} one"] if noun else [])
    out["query"] = list(out["pick"])
    functions = sorted({one for one in _functions(rng)})
    for name in functions:
        out["feedback"] += [name, f"that change to {name}", f"{name}"]
    out["feedback"] += ["that change", "that step", "what you just did",
                        "the last change"]
    return out


def _nouns(cases: list) -> list:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    return [next((word.lemma_.lower() for word in nlp(case)
                  if word.pos_ == "NOUN"), None) for case in cases]


def _functions(rng: random.Random, most: int = 60) -> list:
    import ast
    from research.v704 import history
    names = []
    for path in sorted((history.ROOT / "research").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        names += [f"{path.stem}.{one.name}" if rng.random() < 0.5 else
                  one.name for one in tree.body
                  if isinstance(one, ast.FunctionDef)]
    return rng.sample(names, min(most, len(names)))


def filled(rng: random.Random, each: int = 3) -> list:
    """Each written reply filled with `each` references of its act."""
    rows = [json.loads(line) for line in WRITTEN.open(encoding="utf-8")]
    names = references(rng)
    out, seen = [], set()
    for row in rows:
        for line in row["lines"]:
            for name in rng.sample(names[row["act"]],
                                   min(each, len(names[row["act"]]))):
                said = line.replace("{X}", name)
                if said not in seen:
                    seen.add(said)
                    out.append({"act": row["act"], "template": line,
                                "option": name, "said": said})
    return out


#: what the teacher is asked of each reply, yes or no, for each act -- a
#: choice among every act was answered `feedback` for most (picks kept: 18
#: of 274)
QUESTIONS = {
    "pick": "Is the person choosing which item to start with or to do "
            "next?",
    # `about an item` was answered no for `What is the root cause of the
    # problem with option 2?`: a question is enough, picks kept apart by
    # their own
    "query": "Is the person asking a question?",
    "feedback": "Is the person reacting to something the assistant just "
                "did?",
}


#: the act a reply of each is nearest being instead: a no to it is asked
#: too (`feedback`'s question is yes for most replies of any act)
CLOSEST = {"pick": "query", "query": "pick", "feedback": "pick"}


def check(batch: int = 24) -> dict:
    """Each written reply, with its item said as `option 2`, put to the
    teacher as three yes-or-no questions: kept where it says yes to its own
    act's and no to the others'."""
    from research.v696.teach_meaning import Teacher
    rows = [json.loads(line) for line in WRITTEN.open(encoding="utf-8")]
    todo = [(row["act"], line, asked) for row in rows
            for line in row["lines"] for asked in QUESTIONS]
    teacher = Teacher()
    answers: dict = {}
    for at in range(0, len(todo), batch):
        chunk = todo[at:at + batch]
        replies = teacher.write(
            [f"An assistant showed a person a list of items: what it does "
             f"worst, steps of a plan, failing cases, or a change it just "
             f"made. The person replied: \"{line.replace('{X}', 'option 2')}"
             f"\"\n{QUESTIONS[asked]} Answer yes or no only."
             for _, line, asked in chunk], longest=3)
        for (act, line, asked), reply in zip(chunk, replies):
            answers.setdefault((act, line), {})[asked] = \
                reply[0].strip().lower().startswith("yes")
    counts: dict = {}
    with CHECKED.open("w", encoding="utf-8") as out:
        for (act, line), said in answers.items():
            kept = said.get(act) and not said.get(CLOSEST[act])
            counts[f"{act} {'kept' if kept else 'left'}"] = counts.get(
                f"{act} {'kept' if kept else 'left'}", 0) + 1
            out.write(json.dumps({"act": act, "template": line,
                                  "answers": said, "kept": bool(kept)}) +
                      "\n")
    print(json.dumps(counts))
    return counts


def split_of(text: str) -> str:
    at = hashlib.sha1(text.lower().encode()).digest()[0] / 256
    return "valid" if at < VALID else "train"


def corpus() -> dict:
    """The kept replies as code-talk records: the act, and the option's
    words marked; the split by the reply as written, so a template is
    never in both."""
    rows = [json.loads(line) for line in CHECKED.open(encoding="utf-8")]
    templates = {(one["act"], one["template"]) for one in rows
                 if one["kept"]}
    rng = random.Random(SEED)
    kept = [one for one in filled(rng) if (one["act"], one["template"])
            in templates]
    rng.shuffle(kept)
    out = {"train": [], "valid": []}
    for one in kept:
        said = K.words(one["said"])
        found = K._match(said, one["option"])
        if not said or found is None:
            continue
        roles = ["O"] * len(said)
        begin, end = found
        for at in range(begin, end):
            roles[at] = ("B-" if at == begin else "I-") + "OPTION"
        out[split_of(one["template"])].append(K.record(
            said, roles, one["act"], "none", "none",
            f"plan {one['act']}"))
    counts = {}
    for split, records in out.items():
        with (DATA / f"{split}-plans.jsonl").open(
                "w", encoding="utf-8") as stream:
            for one in records:
                stream.write(json.dumps(one) + "\n")
        counts[split] = len(records)
    counts["checked"] = len(rows)
    counts["kept"] = len(kept)
    print(json.dumps(counts))
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("write", "check", "corpus",
                                        "more-queries"))
    options = parser.parse_args(argv)
    {"write": write, "check": check, "corpus": corpus,
     # queries, asked again more plainly: the first were questions about
     # products, not about an item shown (15 of 92 said yes to)
     "more-queries": lambda: write(("query",), 6, SEED + 1, "a")}[
        options.job]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
