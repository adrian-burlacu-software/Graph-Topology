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
a model -- the needle's file and the files after it, whole, to about 5 000
tokens (`CONTEXT` characters: a third of RepoQA's 16k, what the card holds) -- right where the reply names the function.
Evaluation only, read as totals.

**The code model alone** (`alone`): the 3B code model given the same code
and description, asked for the function's name (greedy) -- where the
needle's file alone is longer than `CONTEXT`, the window of it around the
needle (80 of 200; the project holds those files whole).
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
#: the code given, in characters: about 5k tokens -- a third of RepoQA's
#: 16k, what the 8 GB card attends over beside the model with room left
#: (16k spilled 25 GB into the machine's memory; 8k peaked at 7.9 GB)
CONTEXT = 16000


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


#: a function's name where its code defines one: `def f`, `function f`,
#: `f(...) {` (a method), `f = function`, `f = (...) =>`
DEFINED = re.compile(r"\bdef\s+(\w+)|\bfunction\s*\*?\s*(\w+)|^\s*(?:async\s+"
                     r")?(?:static\s+)?(?:get\s+|set\s+)?(\w+)\s*\([^)]*\)\s*"
                     r"\{|\b(\w+)\s*[:=]\s*(?:async\s*)?(?:function|\([^)]*\)"
                     r"\s*=>)", re.M)


def _defined(one: dict) -> set:
    """The names of the functions its code defines."""
    found = set()
    for text in one["files"].values():
        for match in DEFINED.finditer(text):
            found.update(name for name in match.groups() if name)
    return found - {"if", "for", "while", "switch", "catch", "return",
                    "function"}


#: words of a sentence that some code also names a function (cheerio's
#: `is`, three.js's `that`): no function a reply answers with
SAID = {"is", "it", "that", "this", "does", "the", "a", "an", "to", "of",
        "in", "on", "for", "and", "or", "not", "be", "has", "get", "set"}


def _check(one: dict):
    name = one["name"]
    names = (_defined(one) - SAID) | {name}

    def check(turn, root):
        from research.v704.converse import text_of
        # the function answered: the first of the code's functions the
        # reply names -- the name, or as its class or module says it
        # (`InitCommand._find_best_version_for_package`) -- not one named
        # after it, as what calls it (`transform_line ... called by
        # run_transformer`)
        for word in re.findall(r"[A-Za-z_$][\w$]*", text_of(turn)):
            if word in names:
                return word == name
        return False
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
    from research.v696.teach_meaning import MOST_TOKENS
    with out.open("a", encoding="utf-8") as stream:
        for one in todo:
            code = "\n\n".join(f"# file: {path}\n{text}"
                               for path, text in one["files"].items())
            # where it is longer than the context, or than the card takes
            # (dense code: 16 000 characters were 6 173 tokens), the window
            # around the needle, as RepoQA cuts its context (the project
            # keeps its files whole: cut, they do not parse)
            at = len(f"# file: {one['path']}\n") + one["at"]
            size = CONTEXT
            while True:
                start = max(0, min(at - size // 2, len(code) - size))
                shown = code[start:start + size]
                prompt = ALONE.format(code=shown, asked=asked(one))
                if len(model.tokenizer(prompt)["input_ids"]) <= \
                        MOST_TOKENS - 200:
                    break
                size = int(size * 0.85)
            reply = model.write([prompt], longest=40)[0]
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
