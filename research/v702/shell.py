"""Bash, for basic computer use (v702): a command read or written, checked,
run in the project's folder, and its output read back.

    found = shell.read("what files are in research/v701", known)
    found.act      "task"
    made = shell.written("what files are in research/v701")
    made["command"]   "ls research/v701"
    shell.reads_only("ls research/v701")   (True, "")
    shell.run("ls research/v701", root)    {"code": 0, "out": "...", ...}

What runs without asking is what only reads (`reads_only`): a command is
parsed into the programs it runs, and each must be one that reads -- `ls`,
`cat`, `git log`, `du`, `grep`, `find` without `-delete` or `-exec` -- with
no output written to a file. Anything else is shown and run when the person
says yes; `ASK` (the server's `--no-ask`) runs it at once.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from research import encoder

HEADS = ("shell_act", "shell_role")
#: how sure the reader must be that a message is about the shell
FLOOR = 0.6
#: whether a command that may change something is shown first, and run
#: only when the person says yes (the server's `--no-ask` sets it False)
ASK = True
#: how long a command may run, and how much of its output is kept
TIMEOUT, MOST_OUTPUT = 60, 20_000
#: the writer, its answers, and how many must agree
WRITER = "shell-writer"
SAMPLES, LEAST_AGREED = 6, 2


@dataclass
class Asked:
    act: str
    chance: float
    words: list
    #: the command as written in the message, where one is given
    command: str = ""
    acts: list = field(default_factory=list)


def available() -> bool:
    try:
        return "shell_act" in encoder.LOADED.get().heads
    except Exception:                               # noqa: BLE001
        return False


def read(text: str, known=()) -> Asked | None:
    """What a message asks of the computer, as the encoder reads it; the
    command it gives cut from the message as written (its case, its
    quoting) -- not from the reader's words."""
    from research.v692.corpus import words
    if not available():
        return None
    said = words(text)
    if not said:
        return None
    lower = {str(one).lower() for one in known}
    tags = ["CODE" if word.lower() in lower else "" for word in said]
    found = encoder.read(said, tags=tags, heads=HEADS)
    act, chance = found["shell_act"][0]
    marked = [at for at, role in enumerate(found["shell_role"])
              if role in ("B-CMD", "I-CMD")]
    command = ""
    if act == "command" and marked:
        command = _cut(text, said, marked[0], marked[-1])
    return Asked(act, chance, said, command,
                 [name for name, _ in found["shell_act"]])


def _cut(text: str, said: list, first: int, last: int) -> str:
    """The message's own text from its `first` word to its `last`: each
    word found in order in the lowered message."""
    lowered, at, spans = text.lower(), 0, []
    for word in said:
        where = lowered.find(word.lower(), at)
        if where < 0:
            return " ".join(said[first:last + 1])
        spans.append((where, where + len(word)))
        at = where + len(word)
    out = text[spans[first][0]:spans[last][1]].strip()
    # a command given in backquotes is the command inside them
    return out.strip("`").strip()


# -- what only reads ---------------------------------------------------------------

#: programs that only read, whatever they are given -- but for what each
#: may not be given (`find -delete`, `sed -i`, `sort -o`)
READERS = {
    "ls": (), "dir": (), "cat": (), "head": (), "tail": (), "wc": (),
    "grep": (), "egrep": (), "fgrep": (), "rg": (), "du": (), "df": (),
    "pwd": (), "echo": (), "printf": (), "date": (), "whoami": (),
    "hostname": (), "uname": (), "which": (), "type": (), "file": (),
    "stat": (), "tree": (), "sort": ("-o", "--output"), "uniq": (),
    "cut": (), "tr": (), "nl": (), "less": (), "more": (), "basename": (),
    "dirname": (), "realpath": (), "readlink": (), "printenv": (),
    "md5sum": (), "sha1sum": (), "sha256sum": (), "cmp": (), "diff": (),
    "comm": (), "column": (), "jq": (), "sed": ("-i",),
    "find": ("-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint",
             "-fprintf", "-fls"),
    "ps": (), "free": (), "uptime": (), "id": (), "true": (), "test": (),
    "seq": (), "python": (), "python3": (), "node": (),
}
# (not `env`, which runs what it is given; not `awk`, whose `system()` runs
# anything)
#: of git, what only reads
GIT_READS = ("log", "status", "diff", "show", "ls-files", "rev-parse",
             "blame", "shortlog", "describe", "grep", "branch", "tag",
             "remote")
#: of those, what makes one where it is given a name (`git branch new`,
#: `git tag v1`, `git remote add`): only listing reads
GIT_LISTS = ("branch", "tag", "remote")
#: what git may not be given there (`branch -d`, `config --add`)
GIT_WRITES = ("-d", "-D", "-m", "-M", "--delete", "--add", "--unset",
              "--set-upstream-to", "add", "remove", "rm", "rename",
              "set-url", "--replace-all")


def reads_only(command: str) -> tuple:
    """(True, "") where every program the command runs only reads and it
    writes no file; else (False, why)."""
    if re.search(r"`|\$\(", command):
        return False, "it runs a command inside it"
    # output to a file (not to /dev/null, nor one stream into another)
    for found in re.finditer(r"(\d?)>>?\s*(&?\S+)", command):
        target = found.group(2)
        if target != "/dev/null" and not target.startswith("&"):
            return False, f"it writes to {target}"
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError as bad:
        return False, f"it does not parse ({bad})"
    programs, current = [], []
    for token in tokens:
        if token and set(token) <= set(";&|"):
            programs.append(current)
            current = []
        elif token and set(token) <= set("<>"):
            continue
        else:
            current.append(token)
    programs.append(current)
    for words_ in programs:
        words_ = [one for one in words_ if not re.fullmatch(
            r"[A-Za-z_]\w*=.*", one)] or words_
        if not words_:
            continue
        name = Path(words_[0]).name.lower()
        if name in ("python", "python3", "node") and len(words_) > 1:
            return False, f"it runs a program ({' '.join(words_[:2])})"
        if name == "git":
            sub = next((one for one in words_[1:] if not one.startswith("-")),
                       "")
            after = words_[words_.index(sub) + 1:] if sub else []
            if sub not in GIT_READS or any(one in GIT_WRITES
                                           for one in after) or (
                    sub in GIT_LISTS and any(
                        not one.startswith("-") for one in after)
                    and not (sub == "remote" and after[:1] == ["show"])):
                return False, f"it runs git {' '.join([sub] + after[:1])}"                     .rstrip()
            continue
        if name not in READERS:
            return False, f"it runs {name}"
        refused = [one for one in words_[1:] if any(
            one == bad or one.startswith(bad + "=") or (
                bad == "-i" and re.fullmatch(r"-i\S*", one))
            for bad in READERS[name])]
        if refused:
            return False, f"{name} is given {refused[0]}"
    return True, ""


# -- writing and running ---------------------------------------------------------------

def bash() -> str | None:
    """Bash: Git for Windows' where it is there, else the one on the path."""
    for one in (r"C:\Program Files\Git\bin\bash.exe",
                r"C:\Program Files (x86)\Git\bin\bash.exe"):
        if os.name == "nt" and Path(one).exists():
            return one
    return shutil.which("bash")


def parses(command: str) -> bool:
    found = bash()
    if found is None:
        return False
    checked = subprocess.run([found, "-n", "-c", command],
                             capture_output=True, timeout=10)
    return checked.returncode == 0


def run(command: str, root: str | Path, timeout: int | None = None) -> dict:
    """The command run by Bash in `root`: its exit code, what it printed
    (the first `MOST_OUTPUT` characters of each stream) and how long."""
    found = bash()
    if found is None:
        return {"command": command, "code": None, "out": "",
                "error": "no bash on this computer", "seconds": 0}
    started = time.time()
    limit = timeout or TIMEOUT
    # its own process group: stopped, all it started stops with it -- a
    # `find` under a stopped bash kept its output open, and the run waited
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    # and stopped from inside: GNU timeout stops all it started (Windows'
    # taskkill reaches Git Bash, not the MSYS programs under it)
    process = subprocess.Popen(
        [found, "-c", f"timeout -k 2 {limit} bash -c {shlex.quote(command)}"],
        cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL, creationflags=flags,
        start_new_session=os.name != "nt")
    try:
        stdout, stderr = process.communicate(timeout=limit + 10)
    except subprocess.TimeoutExpired:
        _stop(process)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = b"", b""
        return {"command": command, "code": None, "out": _text(stdout or b""),
                "error": f"stopped after {limit} s", "seconds": limit}
    if process.returncode in (124, 137):
        return {"command": command, "code": None, "out": _text(stdout),
                "error": f"stopped after {limit} s", "seconds": limit}
    out, error = _text(stdout), _text(stderr)
    return {"command": command, "code": process.returncode, "out": out,
            "error": error, "cut": len(stdout) > MOST_OUTPUT or
            len(stderr) > MOST_OUTPUT,
            "seconds": round(time.time() - started, 2)}


def _stop(process) -> None:
    """The process and everything it started, stopped."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)],
                       capture_output=True)
    else:
        import signal
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
    process.kill()


def _text(raw: bytes) -> str:
    # a CRLF file's lines said as lines
    return raw[:MOST_OUTPUT].decode("utf-8", "replace").replace("\r\n", "\n")


def writer_available() -> bool:
    return (encoder.LLM / WRITER / "config.json").exists()


def _grounded(command: str, request: str) -> str:
    """The paths the request names, as it names them: taught on NL2Bash's
    `/path/to/dir`, the writer made `research/v701` `/research/v701`, a
    folder at the computer's root, not the project's."""
    said = {one.strip("\"'`.,?!") for one in request.split()}
    def own(found) -> str:
        path = found.group(2)
        return found.group(1) + (path[1:] if path[1:].rstrip("/") in {
            one.rstrip("/") for one in said} else path)
    return re.sub(r"(^|[\s=\"'])(/[\w.\-/]+)", own, command)


def written(request: str, root: str | Path | None = None) -> dict:
    """The command a request asks for: the writer's answers, those that
    parse, the one most of them write -- where at least two agree."""
    from research.v697.coding import Tools
    from research.v700 import teach_editor as T
    import zlib
    writer = Tools.get().writer(WRITER)
    asked = T.prompt(request, "shell", "")
    writer.torch.manual_seed(702 + zlib.crc32(asked.encode()))
    answers = writer.write([asked], samples=SAMPLES, longest=120,
                           greedy=True)[0]
    counts: dict = {}
    for answer in answers:
        command = answer.strip().strip("`").strip()
        if command.startswith("$ "):
            command = command[2:]
        command = command.splitlines()[0].strip() if command else ""
        command = _grounded(command, request)
        if command and parses(command):
            key = " ".join(command.split())
            counts.setdefault(key, [command, 0])[1] += 1
    ranked = sorted(counts.values(), key=lambda one: -one[1])
    if not ranked:
        return {"status": "unwritten", "of": len(answers)}
    command, agree = ranked[0]
    out = {"status": "written" if agree >= LEAST_AGREED else "unsure",
           "command": command, "agree": agree, "of": len(answers),
           "others": [one[0] for one in ranked[1:4]]}
    if agree < LEAST_AGREED and root is not None:
        found = _agreed_by_output(ranked, root)
        if found is not None:
            out.update(found)
    return out


#: how long a command written apart may run to be compared by its output
COMPARED = 10


def _agreed_by_output(ranked: list, root) -> dict | None:
    """Commands written apart that only read, run, and compared by what
    they print: `git log -n 3` and `git log -3` say the same, and agree
    though written otherwise. The most of them printing the same (two at
    least, and something), the most written of those."""
    printed: dict = {}
    for command, times in ranked[:SAMPLES]:
        if not reads_only(command)[0]:
            continue
        found = run(command, root, timeout=COMPARED)
        if found.get("code") != 0 or not found["out"].strip():
            continue
        printed.setdefault(found["out"], []).append((command, times))
    groups = sorted(printed.values(), key=lambda one: -len(one))
    if not groups or len(groups[0]) < LEAST_AGREED:
        return None
    best = max(groups[0], key=lambda one: one[1])[0]
    return {"status": "written", "command": best, "agree": len(groups[0]),
            "by": "output", "same": [one[0] for one in groups[0]]}
