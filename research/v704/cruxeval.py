"""CRUXEval as conversations: code pasted, asked what it gives for an input,
and what input gives an output (2026-10-23, the core measured beyond its
own test).

    python -m research.v704.cruxeval fetch    data/cruxeval/test.jsonl
    python -m research.v704.converse --suite cruxeval-o|cruxeval-i ...
    python -m research.v704.cruxeval alone o|i    the code model alone

CRUXEval (cruxeval-org/cruxeval, Gu et al. 2024, "CRUXEval: A Benchmark
for Code Reasoning, Understanding and Execution"): 800 short Python
functions `f`, each with an input and the output it gives. Output
prediction (O): `f(input)` asked, the output said. Input prediction (I):
the output given, an input asked that gives it -- any such input is right,
checked by running `f` on it.

Each item a conversation of one turn, the function pasted in a fence
(`explain`): what the architecture does with pasted Python is its own
(v698 `observe`: it runs it). Evaluation only, read as totals.

**The code model alone** (`alone`): the 3B code model given the same code
and question, one answer (greedy), as CRUXEval asks it of a model.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "cruxeval"
ROWS = ("https://datasets-server.huggingface.co/rows?dataset=cruxeval-org/"
        "cruxeval&config=default&split=test&offset={offset}&length=100")
COUNT = 800
#: how many are said, of each task (every item: 800)
SAID = 800


def fetch() -> int:
    rows = []
    for offset in range(0, COUNT, 100):
        request = urllib.request.Request(ROWS.format(offset=offset),
                                         headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(request, timeout=300) as reply:
            rows += [one["row"] for one in json.load(reply)["rows"]]
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "test.jsonl").open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    print(f"{len(rows)} items -> {DATA / 'test.jsonl'}")
    return len(rows)


def items() -> list:
    return [json.loads(line) for line in
            (DATA / "test.jsonl").open(encoding="utf-8")][:SAID]


def _value(text: str):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


def _leading(text: str):
    """The Python literal a text begins with, and only it -- `[(4, 1)]. In
    words: ...` is `[(4, 1)]` (the sentence after it read as part of it
    made every value said wrong: 79 of 800 right, the architecture right
    on most of the rest)."""
    text = text.strip()
    for end in range(len(text), 0, -1):
        if end < len(text) and text[end] not in " .;,\n":
            continue
        piece = text[:end].strip()
        if not piece:
            continue
        # (`2.` ending a sentence is 2, not the float 2.0)
        for one in dict.fromkeys((piece.rstrip("."), piece)):
            try:
                return True, ast.literal_eval(one)
            except (ValueError, SyntaxError, TypeError, MemoryError,
                    RecursionError):
                continue
    return False, None


def _said_values(said: str) -> list:
    """The Python literals a reply says of what is asked: the reply whole,
    or after what says a value (`gives`, `returns`, `==`), in a fence or
    in backquotes too."""
    import re
    out = []
    pieces = re.findall(r"```(?:python)?\n(.*?)```", said, re.S) + \
        re.findall(r"`([^`]+)`", said) + [said]
    for piece in pieces:
        found, value = _leading(piece) if len(piece) < 2000 else (False,
                                                                   None)
        if found:
            out.append(value)
        for marker in re.finditer(r"\b(gives|returns|return|is|==)\s",
                                  piece):
            found, value = _leading(piece[marker.end():][:2000])
            if found:
                out.append(value)
    return out


def output_check(row: dict):
    """Right where the reply says the output CRUXEval gives."""
    wanted = _value(row["output"])

    def check(turn, root):
        from research.v704.converse import text_of
        return any(one == wanted and type(one) is type(wanted)
                   for one in _said_values(text_of(turn)))
    check.about = f"says {row['output'][:60]}"
    return check


RUNS = """import json, sys
{code}
try:
    print(json.dumps(repr(f({args}))))
except BaseException as bad:
    print(json.dumps(type(bad).__name__))
"""


def gives(code: str, args: str) -> str | None:
    try:
        done = subprocess.run([sys.executable, "-I", "-c",
                               RUNS.format(code=code, args=args)],
                              capture_output=True, text=True, timeout=10)
        return json.loads(done.stdout.strip().splitlines()[-1])
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def input_check(row: dict):
    """Right where an input the reply says, `f` run on it, gives the output
    asked: `f(...)` written out, or the arguments alone."""
    import re
    wanted = repr(_value(row["output"]))

    def check(turn, root):
        from research.v698.observe import _written_call
        from research.v704.converse import text_of
        said = text_of(turn)
        # each call written out, read whole by Python's parser (a greedy
        # `f\((.*)\)` took the line to its last bracket: inputs found and
        # right scored wrong)
        calls, rest = [], said
        for _ in range(6):
            call = _written_call(rest, "f")
            if call is None:
                break
            calls.append(call)
            rest = rest[rest.index(call) + len(call):]
        for call in calls:
            if gives(row["code"], call[call.index("(") + 1:-1]) == wanted:
                return True
        return False
    check.about = f"an input f gives {row['output'][:60]} for"
    return check


def suite(task: str) -> list:
    from research.v704.suite import Conversation
    out = []
    for row in items():
        code = row["code"].rstrip()
        if task == "o":
            said = (f"```python\n{code}\n```\nwhat does f({row['input']}) "
                    f"return?")
            out.append(Conversation("explain", [said], output_check(row)))
        else:
            said = (f"```python\n{code}\n```\nwhat input to f gives "
                    f"{row['output']}?")
            out.append(Conversation("explain", [said], input_check(row)))
    return out


ALONE = {"o": ("Here is a Python function:\n```python\n{code}\n```\nWhat "
               "does f({input}) return? Answer with the value alone, as "
               "Python writes it."),
         "i": ("Here is a Python function:\n```python\n{code}\n```\nGive an "
               "input for which f returns {output}. Answer with the call "
               "alone, as f(...).")}


def alone(task: str, label: str = "") -> dict:
    from research.v705 import coder
    model = coder.big()
    out = DATA / f"alone-{task}{label}.jsonl"
    done = {}
    if out.exists():
        done = {one["id"]: one for one in map(json.loads,
                                              out.open(encoding="utf-8"))}
    todo = [row for row in items() if row["id"] not in done]
    with out.open("a", encoding="utf-8") as stream:
        for at in range(0, len(todo), 8):
            chunk = todo[at:at + 8]
            asked = [ALONE[task].format(code=row["code"].rstrip(),
                                        input=row["input"],
                                        output=row["output"])
                     for row in chunk]
            replies = model.write(asked, longest=200)
            for row, reply in zip(chunk, replies):
                reply = reply[0] if isinstance(reply, list) else reply
                check = (output_check if task == "o" else input_check)(row)
                one = {"id": row["id"], "verdict": "right" if check(
                    {"reply": {"text": reply}}, None) else "wrong",
                    "said": reply[:300]}
                done[row["id"]] = one
                stream.write(json.dumps(one) + "\n")
                stream.flush()
    right = sum(one["verdict"] == "right" for one in done.values())
    print(json.dumps({"task": task, "right": right, "of": len(done)}))
    return {"right": right, "of": len(done)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("fetch", "alone"))
    parser.add_argument("task", nargs="?", default="o", choices=("o", "i"))
    options = parser.parse_args(argv)
    if options.job == "fetch":
        fetch()
    else:
        alone(options.task)
    return 0


if __name__ == "__main__":
    sys.exit(main())
