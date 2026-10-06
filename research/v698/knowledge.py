"""What talking about code teaches, kept for every conversation.

*A vowel is one of a, e, i, o, u*: the encoder reads the act `teach`, the
concept and each member (`reading.py`); it is kept here, long-term and
shared, as taught kinds are (`state/v698-knowledge.json`) -- and used when
code is written (`coding.py`: the writers are told what a request's
concepts are, and the search gets each as something to build with).
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "state" / "v698-knowledge.json"
_LOCK = threading.Lock()


def load(path: Path = PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"concepts": {}}


def save(found: dict, path: Path = PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(found, indent=1), encoding="utf-8")


def teach(concept: str, members: list, said: str,
          path: Path = PATH) -> dict:
    """A concept and its members, kept; what it replaces, if anything."""
    with _LOCK:
        found = load(path)
        before = found["concepts"].get(concept)
        found["concepts"][concept] = {"members": members, "said": said,
                                      "at": time.time()}
        save(found, path)
    return {"concept": concept, "members": members, "replaced": before}


def concepts(path: Path = PATH) -> dict:
    return load(path)["concepts"]


def _named(concept: str) -> str:
    """`primary colour` -> `isPrimaryColour`: what a concept is asked as."""
    import re
    parts = [one for one in re.split(r"[^A-Za-z0-9]+", concept) if one]
    return "is" + "".join(one[:1].upper() + one[1:].lower() for one in parts)


def helper(concept: str, members: list) -> tuple:
    """(name, TypeScript source) of the function a concept is: whether a
    value is one of its members."""
    numbers = all(re.fullmatch(r"-?\d+(\.\d+)?", one) for one in members)
    kind = "number" if numbers else "string"
    values = ", ".join(one if numbers else json.dumps(one)
                       for one in members)
    name = _named(concept)
    return name, (f"// {concept}: one of {', '.join(members)} (taught)\n"
                  f"function {name}(x: {kind}): boolean {{\n"
                  f"  return [{values}].includes(x);\n}}\n")


def helper_python(concept: str, members: list) -> tuple:
    """(name, Python source) of the function a concept is, as Python names
    and writes it: `is_vowel(x)`."""
    numbers = all(re.fullmatch(r"-?\d+(\.\d+)?", one) for one in members)
    kind = "float" if numbers and any("." in one for one in members) else \
        "int" if numbers else "str"
    values = ", ".join(one if numbers else json.dumps(one)
                       for one in members)
    parts = [one.lower() for one in re.split(r"[^A-Za-z0-9]+", concept)
             if one]
    name = "is_" + "_".join(parts)
    return name, (f"def {name}(x: {kind}) -> bool:\n"
                  f'    """{concept}: one of {", ".join(members)} '
                  f'(taught)"""\n'
                  f"    return x in ({values},)\n")


def _python_operator(name: str, source: str):
    """A Python helper as an operator the search can use: its body read
    into the engine's tree (`pyparse`)."""
    import ast

    from research.v696 import program as P
    from research.v696 import pyparse, pytypes
    tree = ast.parse(source)
    function = tree.body[0]
    params = [(one.arg, pytypes.engine(ast.unparse(one.annotation)))
              for one in function.args.args]
    body = pyparse.Reader(tree).function(name, params)
    return P.Op(name, "helper", tuple(kind for _, kind in params),
                body.type, params=tuple(n for n, _ in params), body=body)


def relevant(text: str, path: Path = PATH) -> list:
    """The taught concepts a request names -- looked up in what was taught,
    as a function is looked up in a project: [(concept, members)]."""
    from research.v698.teach_code_talk import _match, words
    said = words(text)
    return [(concept, row["members"]) for concept, row in
            concepts(path).items() if _match(said, concept) is not None]


def context(asked: dict, path: Path = PATH) -> dict:
    """What is known of a request's concepts, for writing its code
    (`v697.coding.CONTEXT`): said to the writers, and each concept's
    function offered to the search."""
    from research.v696.changing import operators
    found = relevant(asked.get("english") or "", path)
    if not found:
        return {}
    if asked.get("language") == "python":
        library, said, used = [], [], []
        for concept, members in found:
            name, source = helper_python(concept, members)
            try:
                library.append(_python_operator(name, source))
            except Exception:                       # noqa: BLE001
                pass
            said.append(f"a {concept} is one of {', '.join(members)}")
            used.append({"concept": concept, "members": members,
                         "function": name})
        return {"english": "(" + "; ".join(said) + ")", "library": library,
                "used": used}
    files, said = {}, []
    for concept, members in found:
        name, source = helper(concept, members)
        files[f"/knowledge/{name}.ts"] = "export " + source.split("\n", 1)[1]
        said.append(f"a {concept} is one of {', '.join(members)}")
    return {"english": "(" + "; ".join(said) + ")",
            "library": operators(files, besides=""),
            "used": [{"concept": concept, "members": members,
                      "function": helper(concept, members)[0]}
                     for concept, members in found]}


def taught(reading, text: str, path: Path = PATH) -> dict:
    """The answer to a message the encoder read as teaching."""
    concept = " ".join(reading.spans.get("CONCEPT") or []).strip().lower()
    members = [one.strip() for one in reading.spans.get("MEMBER") or []
               if one.strip()]
    if not concept or not members:
        said = ("I read that as teaching me something, but could not tell "
                "what is taught and what belongs to it.")
        return {"outcome": "acted", "source": "knowledge", "text": said,
                "spoken": said, "code": {"knowledge": {
                    "read": reading.json(), "kept": None},
                    "answer": {"status": "knowledge", "code": None}}}
    kept = teach(concept, members, text, path)
    before = (kept["replaced"] or {}).get("members")
    said = ((f"I knew that already: {concept} is one of "
             f"{', '.join(members)}." if before == members else
             f"Kept: {concept} is one of {', '.join(members)}"
             + (f" (it was {', '.join(before)})" if before else "") + ".")
            + " I will use it when I write code.")
    return {"outcome": "acted", "source": "knowledge", "text": said,
            "spoken": said, "code": {
                "knowledge": {"read": reading.json(), "kept": kept},
                "answer": {"status": "knowledge", "code": None}},
            "suggest": [{"text": f"count the {concept}s in a string",
                         "send": False}]}
