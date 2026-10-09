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
import shlex
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
COMMANDS_MORE = DATA / "commands-more.jsonl"
WRITER_MORE = LLM / "shell-writer2"
#: how often a row's words are told as naming code
TAGGED = 0.25
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
    note = SOURCE.parent / "nl2bash.SOURCE.md"
    # the one committed says more: written only where there is none
    if not note.exists():
        note.write_text(
            "NL2Bash (Lin et al., LREC 2018): English descriptions and the "
            "Bash commands they describe.\nFetched from "
            "https://github.com/TellinaTool/nl2bash (data/bash/all.nl, "
            "all.cm) by `python -m research.v702.teach_shell fetch`.\n",
            encoding="utf-8")
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


# -- everyday requests, and requests that are not the shell's -----------------------------

EVERYDAY = DATA / "everyday.jsonl"
FEATURES = DATA / "features.jsonl"
#: requests to change a program that talk of shells and commands: what is
#: asked of the developer, not of the computer (`integrate writing and
#: reading Bash commands into the architecture` was written `brew install`)
FEATURE_SEEDS = (
    "add a feature that runs shell commands", "make the server log every "
    "command it runs", "write a function that parses ls output",
    "integrate a terminal into the app", "support running Bash scripts "
    "in the editor", "refactor the command runner", "add tests for the "
    "shell module", "teach the reader to understand commands",
    "fix the bug where git output is cut off", "document how commands are "
    "checked before they run")


#: what people ask of a computer in a project's folder -- NL2Bash has
#: little of it (`git log -3`, `du -sh data`, the current branch): the
#: writer agreed on `git log -n 3 | tail -1` for the last three commits.
#: Commands made here, over this project's own folders, files and words
#: (labels by construction); the teacher says what a person asks for each.
#: (Told to write requests and commands both, the teacher wrote `request
#: ||| ls -l -d */` and `ls -lt | grep -o \d+` for the newest folder.)
EVERYDAY_PER = 12


def _project():
    """The project's folders, files and words, as git keeps them."""
    import subprocess
    listed = subprocess.run(["git", "-C", str(ROOT), "ls-files"],
                            capture_output=True, text=True).stdout.split()
    files = [one for one in listed if (ROOT / one).is_file() and
             (ROOT / one).stat().st_size < 400_000]
    folders = sorted({one.rsplit("/", 1)[0] for one in files if "/" in one})
    names = []
    for one in files:
        if one.endswith(".py"):
            text = (ROOT / one).read_text(encoding="utf-8", errors="replace")
            names += re.findall(r"^def ([a-z_]\w{3,})", text, re.M)
    return files, folders, sorted(set(names))


def everyday_commands(rng) -> list:
    """(command, reads only) over the project, each kind once or more."""
    files, folders, names = _project()
    exts = ("py", "md", "json", "ts", "csv", "yaml")
    def f():
        return rng.choice(files)
    def d():
        return rng.choice(folders)
    def k():
        return rng.choice((1, 2, 3, 5, 10, 20))
    def w():
        return rng.choice(names)
    def x():
        return rng.choice(exts)
    def base(path):
        return path.rsplit("/", 1)[-1]
    makers = [
        lambda: f"ls {d()}", lambda: f"ls -la {d()}",
        lambda: f"ls {d()} | wc -l", lambda: f"du -sh {d()}",
        lambda: f"du -sh {f()}", lambda: f"find {d()} -name '*.{x()}' | wc -l",
        lambda: f"find {d()} -name '*.{x()}'", lambda: f"find . -name {base(f())}",
        lambda: f"wc -l {f()}", lambda: f"head -n {k()} {f()}",
        lambda: f"tail -n {k()} {f()}", lambda: f"cat {f()}",
        lambda: f"grep -rn '{w()}' {d()}", lambda: f"grep -rl '{w()}' .",
        lambda: f"grep -c '{w()}' {f()}",
        lambda: f"git log -n {k()} --oneline", lambda: f"git log -n {k()}",
        lambda: f"git log --oneline -- {f()}", lambda: "git log -1 --format=%cd",
        lambda: "git status --short", lambda: "git status",
        lambda: "git branch --show-current", lambda: "git branch -a",
        lambda: "git diff --stat", lambda: "git diff --name-only",
        lambda: "git show --stat HEAD", lambda: f"git log -n {k()} --format='%an %s'",
        lambda: "df -h .", lambda: "date", lambda: "whoami", lambda: "pwd",
        lambda: "python --version", lambda: "git --version",
        lambda: f"ls -lt {d()} | head -n {k()}", lambda: f"ls -S {d()} | head -n {k()}",
        lambda: f"find {d()} -type f -newer {f()}",
        lambda: f"mkdir -p {d()}/notes", lambda: f"rm {f()}",
        lambda: f"cp {f()} {f()}.bak", lambda: f"touch {d()}/TODO.md",
        lambda: f"tar -czf {base(d())}.tar.gz {d()}", lambda: f"git add {f()}",
        lambda: f"mv {f()} {d()}/",
    ]
    from research.v702 import shell
    out, seen = [], set()
    for _ in range(EVERYDAY_PER):
        for make in makers:
            command = make()
            if command in seen:
                continue
            seen.add(command)
            reads, _ = shell.reads_only(command)
            if reads:
                ran = shell.run(command, ROOT, timeout=10)
                if ran.get("code") != 0 or not ran["out"].strip():
                    continue
            out.append((command, reads))
    return out


def everyday(batch: int = 24) -> None:
    """The project's everyday commands, each said by the teacher as people
    ask for it -- every name in the command kept exactly."""
    from research.v696.teach_meaning import Teacher
    rng = random.Random(SEED + 1)
    commands = everyday_commands(rng)
    print(f"{len(commands)} commands", flush=True)
    teacher = Teacher()
    teacher.torch.manual_seed(SEED + 1)
    kept = 0
    with EVERYDAY.open("w", encoding="utf-8") as out:
        for at in range(0, len(commands), batch):
            chunk = commands[at:at + batch]
            prompts = []
            for command, _ in chunk:
                names = _command_names(command)
                keep = (f" Keep {', '.join(names)} exactly so in every one."
                        if names else "")
                prompts.append(
                    f"In a project's folder, this Bash command was run: "
                    f"`{command}`. Write 4 different things a person might "
                    f"type to an assistant that runs commands for them, "
                    f"asking for exactly what this command does or shows -- "
                    f"questions or requests, short and casual, in plain "
                    f"words, not the command.{keep} One per line, nothing "
                    f"else.")
            replies = teacher.write(prompts, longest=300, samples=1)
            for (command, reads), written in zip(chunk, replies):
                names = _command_names(command)
                for line in written[0].splitlines():
                    line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line)
                    line = line.strip().strip("\"'“”")
                    if 8 <= len(line) <= 200 and all(
                            name in line for name in names) and \
                            not _commandlike(line) and command not in line:
                        kept += 1
                        out.write(json.dumps({"said": line, "command": command,
                                              "reads": reads}) + "\n")
            print(f"  {at + len(chunk)}/{len(commands)}", flush=True)
    print(f"everyday: {kept} requests")


def _command_names(command: str) -> list:
    """What a request for the command must say: its paths, files, numbers
    and quoted words."""
    found = []
    for one in shlex.split(command.replace("|", " ")):
        if one.startswith("-") or one in (".", "wc", "head", "tail"):
            continue
        if re.search(r"[/.]", one) and one not in (".",) or \
                re.fullmatch(r"\d+", one) or re.fullmatch(r"[a-z_]\w{3,}",
                                                          one) and "_" in one:
            found.append(one.strip("'"))
    return list(dict.fromkeys(found))


def features(per: int = 12) -> None:
    """Requests to change a program that talk of shells and commands --
    not the shell's to carry out -- said by the teacher from seeds."""
    from research.v696.teach_meaning import Teacher
    teacher = Teacher()
    teacher.torch.manual_seed(SEED + 2)
    prompts = [(f"Write {per} different requests a developer gives a "
                f"programmer working on their program, like: \"{seed}\". "
                f"Each asks for a change to the program's code, about "
                f"commands, shells, terminals, scripts or git. One per line, "
                f"nothing else.") for seed in FEATURE_SEEDS]
    replies = teacher.write(prompts, longest=700, samples=2, temperature=0.9)
    lines = set(FEATURE_SEEDS)
    for written in replies:
        for reply in written:
            for line in reply.splitlines():
                line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line)
                line = line.strip().strip("\"'“”")
                if 10 <= len(line) <= 200 and not _commandlike(line):
                    lines.add(line)
    with FEATURES.open("w", encoding="utf-8") as out:
        for line in sorted(lines):
            out.write(json.dumps({"said": line}) + "\n")
    print(f"features: {len(lines)}")


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
    # everyday requests (the teacher's, each command checked by running
    # it): taught as tasks, and to the writer three times over; their
    # commands given bare, as `git status` is typed
    more = []
    if EVERYDAY.exists():
        for line in EVERYDAY.open(encoding="utf-8"):
            one = json.loads(line)
            split = split_of(one["command"])
            part = "train" if split == "train" else "valid"
            said_words = words(one["said"])
            if said_words:
                rows[part].append(_record(said_words, ["O"] * len(said_words),
                                          "task", "everyday"))
            row = {"statement": one["said"], "path": "shell", "part": "",
                   "target": one["command"], "split": split,
                   "source": "everyday"}
            writer.append(row)
            more += [row] * 3
            bare = words(one["command"])
            if bare:
                rows[part].append(_record(
                    [w.lower() for w in bare],
                    ["B-CMD"] + ["I-CMD"] * (len(bare) - 1), "command",
                    "given bare"))
            made = _given(one["command"], rng)
            if made is not None:
                rows[part].append(made)
    # requests to change a program that talk of commands: the developer's,
    # not the shell's
    # and the requests plans are taught from (v703: commits of several
    # files) -- `report whether the shell asks in /api/health and show it
    # in the MCP health tool` was written `curl -sIq /api/health`
    planned = LLM / "plan-data" / "described.jsonl"
    asked = [json.loads(line)["message"].splitlines()[0]
             for line in planned.open(encoding="utf-8")] \
        if planned.exists() else []
    if FEATURES.exists():
        asked += [json.loads(line)["said"]
                  for line in FEATURES.open(encoding="utf-8")]
    for one in asked:
        if one:
            said = words(one)
            part = "valid" if rng.random() < 0.1 else "train"
            for _ in range(3):
                rows[part].append(_record(said, ["O"] * len(said), "none",
                                          "feature"))
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
    # the words naming the project's code told beside them, as at run time
    # (`the` is a function's name here: told so, `delete the __pycache__
    # folders` fell under the floor) -- now and then, on any row, so the tag
    # alone decides nothing
    for made in rows.values():
        for one in made:
            if one["words"] and rng.random() < TAGGED:
                marked = set(rng.sample(range(len(one["words"])),
                                        min(2, len(one["words"]))))
                one["tags"] = ["CODE" if at in marked else ""
                               for at in range(len(one["words"]))]
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
    # what the second writer is taught from the first: the everyday
    # requests, and as many of the rest, so what it knew is kept
    rest = [one for one in writer if one["source"] != "everyday"]
    more += rng.sample(rest, min(len(rest), max(len(more), 4000)))
    rng.shuffle(more)
    with COMMANDS_MORE.open("w", encoding="utf-8") as out:
        for one in more:
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


def train_more(seed: int = SEED + 1) -> None:
    """The second writer: the first taught the everyday requests."""
    from research.v700 import teach_editor as T
    T.SAYING = SAYING
    T.train(WRITER_MORE, epochs=1, rate=5e-5, seed=seed,
            corpus=COMMANDS_MORE, longest=512, base=WRITER)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("fetch", "phrase", "everyday",
                                        "features", "corpus", "train",
                                        "train-more"))
    parser.add_argument("--epochs", type=int, default=2)
    options = parser.parse_args(argv)
    if options.job == "train":
        train(options.epochs)
    else:
        {"fetch": fetch, "phrase": phrase, "everyday": everyday,
         "features": features, "corpus": corpus,
         "train-more": train_more}[options.job]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
