"""Teaching the architecture Bash: what is asked of the computer, and the
command that does it.

    python -m research.v702.teach_shell fetch     data/nl2bash (NL2Bash, as published)
    python -m research.v702.teach_shell phrase    the teacher says requests again
    python -m research.v702.teach_shell corpus    llm/shell-data: the reader's rows, the writer's
    python -m research.v702.teach_shell train     llm/shell-writer: a request -> its command

Asked `run ls research/v701`, the architecture said it did not know what
that was; asked `what files are in research/v701`, it looked in the
project's data; `$ git log --oneline -3` it took as a request for a
TypeScript function. It had no shell.

**The reader** (`shell_act`, `shell_role`): a message gives a command to
run as written (`run ls -la`, `$ git status`, a command alone), asks for
something a command does (`what files are in research`, `how big is
data/`), answers whether to run one it was shown (`yes`, `go ahead`,
`no`), or is none of these. The command's words are marked (`CMD`).

**The writer**: SmolLM2-360M taught NL2Bash's 12,607 descriptions and
their commands (Lin et al. 2018, `TellinaTool/nl2bash`), and the same
requests said again by the teacher (SmolLM3, offline) as people ask them --
questions and requests, keeping every name, path and quoted word exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LLM = ROOT / "llm"
SOURCE = ROOT / "data" / "nl2bash"
URL = ("https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/"
       "bash/{}")
DATA = LLM / "shell-data"
PHRASED = DATA / "phrased.jsonl"
COMMANDS = DATA / "commands.jsonl"
WRITER = LLM / "shell-writer"
SEED = 702

SAYING = ("You write Bash: one command that does what is asked, run in the "
          "project's folder. Answer with the command alone.")
ACTS = ("none", "command", "task", "yes", "no")
ROLES = ("O", "B-CMD", "I-CMD")
#: requests said again by the teacher, and how many ways each
PHRASE_MOST, PHRASE_WAYS = 1500, 5


def words(text: str) -> list:
    from research.v692.corpus import words as said
    return said(text)


def split_of(command: str) -> str:
    at = hashlib.sha1(command.encode()).digest()[0]
    return "test" if at < 8 else "dev" if at < 16 else "train"


# -- NL2Bash ------------------------------------------------------------------------

def fetch() -> None:
    import urllib.request
    SOURCE.mkdir(parents=True, exist_ok=True)
    for name in ("all.nl", "all.cm"):
        path = SOURCE / name
        if not path.exists():
            urllib.request.urlretrieve(URL.format(name), path)
    (SOURCE.parent / "nl2bash.SOURCE.md").write_text(
        "NL2Bash (Lin et al., LREC 2018): English descriptions and the Bash "
        "commands they describe.\nFetched from "
        "https://github.com/TellinaTool/nl2bash (data/bash/all.nl, all.cm) by "
        "`python -m research.v702.teach_shell fetch`.\n", encoding="utf-8")
    print(len(pairs()), "pairs")


def pairs() -> list:
    """(description, command), as published, each once."""
    said = (SOURCE / "all.nl").read_text(encoding="utf-8").splitlines()
    done = (SOURCE / "all.cm").read_text(encoding="utf-8").splitlines()
    seen, out = set(), []
    for one, command in zip(said, done):
        one, command = one.strip(), command.strip()
        if one and command and (one, command) not in seen and \
                len(command) <= 300:
            seen.add((one, command))
            out.append((one, command))
    return out


# -- the teacher says each again ----------------------------------------------------

def _names(text: str) -> list:
    """What a request must keep exactly: quoted words, paths, file names,
    variables and numbers."""
    found = re.findall(r"\"[^\"]+\"|'[^']+'|`[^`]+`", text)
    rest = re.sub(r"\"[^\"]+\"|'[^']+'|`[^`]+`", " ", text)
    found += [one for one in re.findall(r"\S+", rest)
              if re.search(r"[/$.*_\d]", one.strip(".,;:()"))]
    return list(dict.fromkeys(one.strip(".,;:()") for one in found
                              if one.strip(".,;:()")))


def _commandlike(line: str) -> bool:
    """A line the teacher wrote as the command, not as a request: options
    (` -name`), pipes, redirections."""
    return bool(re.search(r"(^|\s)--?[a-zA-Z]|\||\s>{1,2}\s|\$\(|`", line))


def phrase(batch: int = 24) -> None:
    """Requests said again as people ask the computer -- questions and
    requests -- every name kept."""
    from research.v696.teach_meaning import Teacher
    rng = random.Random(SEED)
    chosen = [one for one in pairs() if split_of(one[1]) == "train"]
    rng.shuffle(chosen)
    chosen = chosen[:PHRASE_MOST]
    DATA.mkdir(parents=True, exist_ok=True)
    done = set()
    if PHRASED.exists():
        done = {json.loads(line)["said"] for line in
                PHRASED.open(encoding="utf-8")}
    todo = [one for one in chosen if one[0] not in done]
    print(f"{len(todo)} requests to say again", flush=True)
    teacher = Teacher()
    teacher.torch.manual_seed(SEED)
    for at in range(0, len(todo), batch):
        chunk = todo[at:at + batch]
        prompts = []
        for said, _ in chunk:
            names = _names(said)
            keep = (f" Keep {', '.join(names)} exactly so in every one."
                    if names else "")
            # the same request, said otherwise -- not another question:
            # told to ask `how many`, the teacher asked how many PHP files
            # need `foo` replaced, of a command that replaces it
            prompts.append(
                f"Say this request to an assistant that runs commands on "
                f"your computer in {PHRASE_WAYS} different ways, as a person "
                f"would type it -- short and casual, as a request or as a "
                f"question -- asking for exactly the same thing, nothing "
                f"more or less.{keep} One per line, nothing else: "
                f"\"{said}\"")
        # sampled: greedy, it says the request back as it was
        replies = teacher.write(prompts, longest=400, samples=2,
                                temperature=0.9)
        with PHRASED.open("a", encoding="utf-8") as out:
            for (said, command), written in zip(chunk, replies):
                names = _names(said)
                lines = []
                for reply in written:
                    for line in reply.splitlines():
                        line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line)
                        line = line.strip().strip("\"'“”")
                        if 6 <= len(line) <= 200 and all(
                                name in line for name in names) and                                 line.lower() != said.lower() and                                 not _commandlike(line):
                            lines.append(line)
                lines = list(dict.fromkeys(lines))
                out.write(json.dumps({"said": said, "command": command,
                                      "lines": lines}) + "\n")
        print(f"  {at + len(chunk)}/{len(todo)}", flush=True)


# -- the corpus ---------------------------------------------------------------------------

#: a command given to run, as people give one (the command's words marked)
GIVEN = ("{c}", "{c}", "run {c}", "$ {c}", "can you run {c}",
         "please run {c}", "execute {c}", "run `{c}`", "try {c}",
         "what does {c} say?", "run {c} for me", "> {c}")
YES = ("yes", "yes, run it", "go ahead", "do it", "ok run it", "sure",
       "yep", "yes please", "run it", "okay, go")
NO = ("no", "don't run it", "cancel", "no thanks", "stop", "nope",
      "don't", "leave it", "no, don't")


def _record(said: list, roles: list, act: str, source: str) -> dict:
    return {"task": "shell", "words": said, "roles": roles, "act": act,
            "tags": [""] * len(said), "deps": [""] * len(said),
            "names": [], "said": " ".join(said), "source": source,
            "how": source}


def _given(command: str, rng) -> dict | None:
    template = rng.choice(GIVEN)
    head, _, tail = template.partition("{c}")
    before, inside, after = words(head), words(command), words(tail)
    if not inside:
        return None
    said = before + inside + after
    roles = ["O"] * len(before) + ["B-CMD"] + ["I-CMD"] * (
        len(inside) - 1) + ["O"] * len(after)
    return _record([one.lower() for one in said], roles, "command",
                   "given")


def corpus(negatives: int = 9000) -> dict:
    rng = random.Random(SEED)
    rows = {"train": [], "valid": []}
    writer = []
    phrased = {}
    if PHRASED.exists():
        for line in PHRASED.open(encoding="utf-8"):
            one = json.loads(line)
            phrased[(one["said"], one["command"])] = one["lines"]
    for said, command in pairs():
        split = split_of(command)
        part = "train" if split == "train" else "valid"
        asked = [said] + phrased.get((said, command), [])
        for line in asked:
            said_words = words(line)
            if said_words:
                rows[part].append(_record(said_words, ["O"] * len(said_words),
                                          "task", "nl2bash" if line == said
                                          else "teacher"))
            writer.append({"statement": line, "path": "shell", "part": "",
                           "target": command, "split": split,
                           "source": "nl2bash" if line == said
                           else "teacher"})
        made = _given(command, rng)
        if made is not None:
            rows[part].append(made)
    for part in rows:
        for _ in range(400 if part == "train" else 60):
            for kind, pool in (("yes", YES), ("no", NO)):
                said = words(rng.choice(pool))
                rows[part].append(_record(said, ["O"] * len(said), kind,
                                          "answer"))
    # what is none of these: code talk (a function's call is no command),
    # questions about the project's data, everyday sentences
    others = []
    for folder, share in (("code-talk-data", 1.0), ("data-talk-data", 1.0),
                          ("reader-data", 0.03)):
        for path in sorted((LLM / folder).glob("train*.jsonl")):
            for line in path.open(encoding="utf-8"):
                found = json.loads(line).get("words")
                if found and rng.random() < share:
                    others.append(found)
    for said in rng.sample(others, min(negatives, len(others))):
        said = [str(one).lower() for one in said]
        part = "valid" if rng.random() < 0.1 else "train"
        rows[part].append(_record(said, ["O"] * len(said), "none",
                                  "not shell"))
    DATA.mkdir(parents=True, exist_ok=True)
    stats = {}
    for part, made in rows.items():
        rng.shuffle(made)
        with (DATA / f"{part}-shell.jsonl").open("w", encoding="utf-8") as out:
            for one in made:
                out.write(json.dumps(one) + "\n")
        count: dict = {}
        for one in made:
            count[one["act"]] = count.get(one["act"], 0) + 1
        stats[part] = count
    rng.shuffle(writer)
    with COMMANDS.open("w", encoding="utf-8") as out:
        for one in writer:
            out.write(json.dumps(one) + "\n")
    stats["writer"] = {split: sum(one["split"] == split for one in writer)
                       for split in ("train", "dev", "test")}
    (DATA / "stats.json").write_text(json.dumps(stats, indent=1),
                                     encoding="utf-8")
    print(json.dumps(stats))
    return stats


def train(epochs: int = 2, seed: int = SEED) -> None:
    """The writer: SmolLM2-360M as it came, taught a request and the
    command that does it (`teach_editor.train`, its prompt: the request,
    then `shell:`)."""
    from research.v700 import teach_editor as T
    # what it is told it writes, kept with it (`sketcher.json`) and so told
    # again when it writes
    T.SAYING = SAYING
    T.train(WRITER, epochs=epochs, rate=1e-4, seed=seed, corpus=COMMANDS,
            longest=512)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("fetch", "phrase", "corpus",
                                        "train"))
    parser.add_argument("--epochs", type=int, default=2)
    options = parser.parse_args(argv)
    if options.job == "train":
        train(options.epochs)
    else:
        {"fetch": fetch, "phrase": phrase, "corpus": corpus}[options.job]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
