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
    "correction": "says the assistant got something wrong about it, or "
                  "corrects what it said -- a count, a place in the code, "
                  "a cause, which case belongs where",
}
#: what each kind of reply is taught as: a correction of what it said is
#: feedback too, on its words rather than its deeds (`build loses 2 of the
#: 4, not 1` was read as a query, 0.58)
LABEL = {"correction": "feedback"}
SITUATION = {
    "pick": "An assistant that works on software has just shown a person "
            "a plan: what it does worst, the steps it would take, and "
            "cases that fail now.",
    "query": "An assistant that works on software has just shown a person "
             "a plan: what it does worst, the steps it would take, and "
             "cases that fail now.",
    "feedback": "An assistant that works on software has just carried out "
                "a step of its plan and said what it did.",
    "correction": "An assistant that works on software has just told a "
                  "person what it found about the items of its plan: which "
                  "part fails, how many cases, where in the code.",
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
#: what a message is read with while a plan is held: the conversation's
#: state, as a word the reader sees -- `start with the routing problem`
#: is a pick only where a plan was shown, a change anywhere else
#: (reader-code34, taught without it: 0 of 5 real picks read as picks)
MARKER = "[plan]"
#: the acts only a plan held makes possible
PLAN_ACTS = ("pick", "query", "feedback")
#: the forms each act is asked in, besides the first rounds' (whose
#: picks put {X} first, and said why it was a good start)
FORMS = {
    "pick": "short orders and choices like 'start with {X}', '{X} first', "
            "'go with {X}', 'let's do {X} next', 'ok, look at {X}', 'try "
            "{X}', 'begin with {X} please', 'I'd pick {X}', 'skip that, go "
            "to {X}', 'move on to {X}', 'do {X}'",
    "query": "short questions like 'why does {X} fail?', 'what is {X}?', "
             "'what went wrong with {X}?', 'what would you do about {X}?', "
             "'how come {X} broke?', 'what's {X} about?'",
    "feedback": "short reactions like 'that change to {X} did nothing', "
                "'you changed the wrong thing in {X}', '{X} is still "
                "broken', 'nice, {X} works now', 'that's not what I meant "
                "by {X}', 'undo {X}'",
    "correction": "short corrections like 'no, {X} fails too', 'you "
                  "counted wrong, {X} is lost there as well', '{X} goes "
                  "wrong in the same place', 'that's not right about {X}', "
                  "'actually {X} is the cause', '{X} isn't what fails'",
}
#: of other code talk, how many said with the marker for each reply
#: taught: what a plan held does not change
CONTRAST = 2
VALID = K.VALID


def write(acts=tuple(ASKED), rounds: int = ROUNDS, seed: int = SEED,
          mode: str = "w", forms: bool = False) -> None:
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
                + (f"{FORMS[act]}. " if forms else "questions and orders. ")
                + "One per line, nothing else."
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
        # and by its doing (`the teaching step`, `measuring`), as Brown has
        # it
        for doing in _doing(kind):
            out["pick"] += [f"the {doing} step", doing, f"the {doing}"]
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
    out["correction"] = out["pick"] + out["feedback"]
    return out


def _doing(kind: str) -> list:
    """A step's verb as its doing: the words Brown has that end in -ing
    and are, to WordNet, that verb."""
    if "doing" not in _DOING:
        from nltk.corpus import brown
        _DOING["doing"] = {one.lower() for one in brown.words()
                           if one.lower().endswith("ing")}
    from nltk.corpus import wordnet
    return sorted(one for one in _DOING["doing"]
                  if wordnet.morphy(one, "v") == kind)


_DOING: dict = {}


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
    "correction": "Is the person saying the assistant was wrong, or "
                  "correcting it?",
}


#: the act a reply of each is nearest being instead: a no to it is asked
#: too (`feedback`'s question is yes for most replies of any act)
CLOSEST = {"pick": "query", "query": "pick", "feedback": "pick",
           "correction": "query"}


#: the questions a reply of each kind is kept on a yes to: a correction
#: is feedback (what it is taught as), and the teacher says no to `is the
#: person correcting it?` for `No, {X} fails too.` (102 of 153 left)
YES_TO = {"correction": ("correction", "feedback")}


def kept_reply(act: str, line: str, said: dict) -> bool:
    """Kept: a yes to its own kind's question (or one it is taught as), a
    no to the closest kind's, and no other slot than {X}."""
    if re.search(r"\{(?!X\})", line):
        return False
    return any(said.get(one) for one in YES_TO.get(act, (act,))) and \
        not said.get(CLOSEST[act])


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
            kept = kept_reply(act, line, said)
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
                 if kept_reply(one["act"], one["template"], one["answers"])}
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
            [MARKER] + said, ["O"] + roles,
            LABEL.get(one["act"], one["act"]), "none", "none",
            f"plan {one['act']}"))
    # the rest of code talk, said while a plan is held: its own act still
    # (a change named, a question of code, a new goal)
    for split in out:
        others = []
        for name in ("code", "goals"):
            path = DATA / f"{split}-{name}.jsonl"
            if path.exists():
                others += [json.loads(line) for line in
                           path.open(encoding="utf-8")]
        for one in rng.sample(others, min(len(others), CONTRAST *
                                          len(out[split]))):
            out[split].append({**one, "words": [MARKER] + one["words"],
                               "roles": ["O"] + one["roles"],
                               "tags": [""] + one["tags"],
                               "deps": [""] + one["deps"],
                               "said": f"{MARKER} {one['said']}",
                               "source": f"plan held, {one['source']}",
                               "how": "plan held"})
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
                                        "more-queries", "more",
                                        "corrections"))
    options = parser.parse_args(argv)
    {"write": write, "check": check, "corpus": corpus,
     # queries, asked again more plainly: the first were questions about
     # products, not about an item shown (15 of 92 said yes to)
     "more-queries": lambda: write(("query",), 6, SEED + 1, "a"),
     # every act again in short everyday forms (`FORMS`)
     "more": lambda: write(("pick", "query", "feedback"), 10, SEED + 2,
                           "a", forms=True),
     # corrections of what it said, and picks again with skips and
     # moves (`skip that, go to measuring` was read as none)
     "corrections": lambda: write(("correction", "pick"), 8, SEED + 3, "a",
                                  forms=True)}[
        options.job]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
