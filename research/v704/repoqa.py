"""RepoQA as conversations: a function asked for by what it does, of a
project holding its repository's code (2026-10-25, the core measured beyond
its own test: a subject said by what it does).

    python -m research.v704.repoqa fetch       data/repoqa/repoqa-2024-06-23.json
    python -m research.v704.repoqa projects    a project of each needle's code
    python -m research.v704.converse --suite repoqa ...   the architecture
    python -m research.v704.repoqa alone       the code model alone

RepoQA (Liu et al. 2024, "RepoQA: Evaluating Long Context Code
Understanding"; evalplus/repoqa, its `dev-dataset` release): Search Needle
Function -- 10 repositories in each of 6 languages, 10 functions ("needles")
of each described in words that avoid their names (purpose, input, output,
procedure). Python and TypeScript are what is said (200 needles): the
languages the architecture reads.

Each needle is a conversation of one turn (`code-ask`): its description,
asked as which function does it, of a project holding the code RepoQA gives
a model -- the needle's file and the files after it, whole, to about 16 000
tokens (`CONTEXT` characters) -- right where the reply names the function.
Evaluation only, read as totals.

**The code model alone** (`alone`): the 3B code model given the same code
and description, asked for the function's name (greedy) -- where the
needle's file alone is longer than `CONTEXT`, the window of it around the
needle (39 of 200; the project holds those files whole).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "repoqa"
SOURCE = DATA / "repoqa-2024-06-23.json"
URL = ("https://github.com/evalplus/repoqa/releases/download/dev-dataset/"
       "repoqa-2024-06-23.json")
LANGUAGES = ("python", "typescript")
#: the code given, in characters: RepoQA's 16k tokens, about
CONTEXT = 48000


def fetch() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(URL, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(request, timeout=600) as reply:
        SOURCE.write_bytes(reply.read())
    print(f"{SOURCE.stat().st_size} bytes -> {SOURCE}")
    return 0


def needles() -> list:
    """Each needle said: {n, language, repo, name, path, description,
    files} -- `files` the code given, the needle's file first."""
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    out = []
    for language in LANGUAGES:
        for repo in data[language]:
            content = repo["content"]
            for needle in repo["needles"]:
                files = {needle["path"]: content[needle["path"]]}
                size = len(content[needle["path"]])
                ordered = sorted(content)
                start = ordered.index(needle["path"])
                for path in ordered[start + 1:] + ordered[:start]:
                    if size + len(content[path]) > CONTEXT:
                        continue
                    files[path] = content[path]
                    size += len(content[path])
                out.append({"n": len(out), "language": language,
                            "repo": repo["repo"], "name": needle["name"],
                            "path": needle["path"],
                            "at": int(needle["start_byte"]),
                            "description": needle["description"].strip(),
                            "files": files})
    return out


def _folder(one: dict) -> Path:
    return DATA / "projects" / f"{one['n']:03d}"


def projects() -> int:
    made = needles()
    for one in made:
        for path, text in one["files"].items():
            target = _folder(one) / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
    print(f"{len(made)} projects -> {DATA / 'projects'}")
    return len(made)


def asked(one: dict) -> str:
    return f"Which function in the project does this?\n{one['description']}"


def _check(one: dict):
    name = one["name"]

    def check(turn, root):
        from research.v704.converse import text_of
        return re.search(rf"(?<![\w.]){re.escape(name)}(?!\w)",
                         text_of(turn)) is not None
    check.about = f"names {name}"
    return check


def suite() -> list:
    from research.v704.suite import Conversation
    return [Conversation("code-ask", [asked(one)], _check(one), True,
                         str(_folder(one))) for one in needles()]


ALONE = ("Here is code from a repository:\n{code}\n\n{asked}\nAnswer with "
         "the function's name alone.")


def alone(label: str = "alone") -> dict:
    from research.v705 import coder
    model = coder.big()
    out = DATA / f"{label}.jsonl"
    done = {}
    if out.exists():
        done = {one["n"]: one for one in map(json.loads,
                                             out.open(encoding="utf-8"))}
    todo = [one for one in needles() if one["n"] not in done]
    with out.open("a", encoding="utf-8") as stream:
        for one in todo:
            code = "\n\n".join(f"# file: {path}\n{text}"
                               for path, text in one["files"].items())
            if len(code) > CONTEXT:
                # a needle's file longer than the context by itself: the
                # window around the needle, as RepoQA cuts its context (the
                # project keeps the file whole: cut, it does not parse)
                text = one["files"][one["path"]]
                start = max(0, min(one["at"] - CONTEXT // 2,
                                   len(text) - CONTEXT))
                code = f"# file: {one['path']}\n" + text[start:start +
                                                          CONTEXT]
            reply = model.write([ALONE.format(code=code, asked=asked(one))],
                                longest=40)[0]
            reply = reply[0] if isinstance(reply, list) else reply
            right = _check(one)({"reply": {"text": reply}}, None)
            row = {"n": one["n"], "verdict": "right" if right else "wrong",
                   "said": reply[:120]}
            done[one["n"]] = row
            stream.write(json.dumps(row) + "\n")
            stream.flush()
    right = sum(one["verdict"] == "right" for one in done.values())
    print(json.dumps({"right": right, "of": len(done)}))
    return {"right": right, "of": len(done)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("fetch", "projects", "alone"))
    options = parser.parse_args(argv)
    {"fetch": fetch, "projects": projects, "alone": alone}[options.job]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
