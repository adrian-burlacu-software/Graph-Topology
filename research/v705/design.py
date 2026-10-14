"""New design as steps (v705): a change its history made by inventing a
helper -- `determines`, "the output is fixed by this value" -- written,
and called from the function that needed it. One edit cannot hold that;
three steps can, each its own: **propose** the helper (a name, its
parameters, what it gives: a def line and a docstring), **write** it, and
**wire** it in (a local change, once it exists).

    python -m research.v705.design mine     state/v705-designs.jsonl

**Mined** (`mine`): of each commit, each function it added to a source
file that a function it changed in the same file calls in the commit's
version and did not before -- {message, file, helper (its def line,
docstring and source), caller, caller before, caller after}. What the
proposer is taught (goal + caller before -> the helper's def line and
docstring) and what the steps are measured on.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

from research.v705 import commits

ROOT = Path(__file__).resolve().parents[2]
DESIGNS = ROOT / "state" / "v705-designs.jsonl"


def _calls(source: str) -> set:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                out.add(func.id)
            elif isinstance(func, ast.Attribute):
                out.add(func.attr)
    return out


def _head(source: str) -> str:
    """A function's def line, on one line."""
    return " ".join(source.split(":\n")[0].split()) + ":"


def mine() -> dict:
    shas = commits._git("rev-list", "HEAD", "--", "research").split()
    rows = []
    for sha in shas:
        message = commits._git("log", "-1", "--format=%s", sha).strip()
        files = [one for one in commits._git(
            "diff-tree", "--no-commit-id", "--name-only", "-r", sha).split()
            if one.startswith("research/") and one.endswith(".py")
            and not commits._test(one)]
        for path in files:
            before = commits._show(f"{sha}^", path)
            after = commits._show(sha, path)
            if before is None or after is None:
                continue
            old, new = commits.functions(before), commits.functions(after)
            added = {name: source for name, source in new.items()
                     if name not in old}
            if not added:
                continue
            for caller, source in new.items():
                if caller not in old or old[caller] == source:
                    continue
                gained = _calls(source) - _calls(old[caller])
                for helper, body in added.items():
                    if helper.split(".")[-1] not in gained:
                        continue
                    tree = ast.parse(body.strip()) if body else None
                    doc = ast.get_docstring(tree.body[0]) if tree and \
                        tree.body and isinstance(
                            tree.body[0], (ast.FunctionDef,
                                           ast.AsyncFunctionDef)) else None
                    rows.append({"sha": sha, "message": message,
                                 "file": path, "helper": helper,
                                 "head": _head(body), "doc": doc or "",
                                 "helper source": body, "caller": caller,
                                 "before": old[caller], "after": source})
    DESIGNS.parent.mkdir(parents=True, exist_ok=True)
    with DESIGNS.open("w", encoding="utf-8") as out:
        for one in rows:
            out.write(json.dumps(one) + "\n")
    found = {"designs": len(rows), "commits": len({one["sha"]
                                                   for one in rows}),
             "with a docstring": sum(bool(one["doc"]) for one in rows)}
    print(json.dumps(found))
    return found


CORPUS = commits.HISTORY / "design-corpus.jsonl"
DESIGNER = ROOT / "llm" / "designer"
#: what the designer is told it does
DESIGNING = ("You write the new function a change needs: its def line, a "
             "docstring saying what it gives, and its body. Answer with the "
             "function only.")
#: a helper taught at most this many lines long; a caller shown at most
HELPER_LINES, CALLER_LINES = 40, 80
#: the helpers proposed for each bug, and the editor's rounds wiring each
PROPOSED, WIRING_ROUNDS = 4, 2


def asked(message: str, caller: str, file: str) -> str:
    """What the designer is told: the goal, and which function needs a
    new one -- taught and asked alike."""
    return (f"{message}\nWrite the new function {caller} in {file} needs "
            f"for it.")


def corpus() -> dict:
    """Each design of a commit not held out (`commits._held_out`): the
    goal and the caller as it was -> the helper the commit wrote."""
    import hashlib
    held = commits._held_out()
    rows = []
    for one in map(json.loads, DESIGNS.open(encoding="utf-8")):
        if one["sha"] in held or "." in one["helper"]:
            continue
        helper = one["helper source"].strip("\n")
        if len(helper.splitlines()) > HELPER_LINES or \
                len(one["before"].splitlines()) > CALLER_LINES:
            continue
        at = hashlib.sha1(one["sha"].encode()).digest()[0] / 256
        rows.append({"split": "dev" if at < 0.05 else "train",
                     "language": "python",
                     "statement": asked(one["message"], one["caller"],
                                        one["file"]),
                     "path": one["file"], "part": one["before"],
                     "start": 1, "target": helper})
    commits.HISTORY.mkdir(parents=True, exist_ok=True)
    with CORPUS.open("w", encoding="utf-8") as out:
        for one in rows:
            out.write(json.dumps(one) + "\n")
    found = {"rows": len(rows), "dev": sum(one["split"] == "dev"
                                           for one in rows)}
    print(json.dumps(found))
    return found


def train(epochs: int = 3) -> None:
    """The designer: from the editor taught its own code (`editor-own`),
    taught to write the helper a change needs."""
    from research.v700 import teach_editor as T
    T.SAYING = DESIGNING
    T.train(DESIGNER, epochs=epochs, corpus=CORPUS, longest=1024,
            base=ROOT / "llm" / "editor-own", rate=5e-5)


def _uses(name: str, text: str) -> int:
    """How many functions of a file call `name`."""
    return sum(name in _calls(source) for source in
               commits.functions(text).values())


def cases() -> list:
    """The held-out bugs that are designs: the caller a commit changed to
    call a helper it added, the helper called by that caller alone (taken
    out, nothing else breaks for want of it)."""
    held = commits._held_out()
    designs = {(one["sha"], one["caller"]): one for one in
               map(json.loads, DESIGNS.open(encoding="utf-8"))}
    mined = {(one["sha"], one["function"]): one for one in
             map(json.loads, commits.MINED.open(encoding="utf-8"))}
    out = []
    for bug in map(json.loads, commits.CHECKED.open(encoding="utf-8")):
        key = (bug["sha"], bug["function"])
        if bug["status"] != "bug" or bug["sha"] not in held or \
                key not in designs or "." in designs[key]["helper"]:
            continue
        design = designs[key]
        text = (ROOT / design["file"]).read_text(encoding="utf-8")
        helper = commits.functions(text).get(design["helper"])
        if not helper or _uses(design["helper"], text) != 1:
            continue
        out.append({**mined[key], "tests": bug["fails"],
                    "helper": design["helper"], "helper now": helper})
    return out


def _without(text: str, helper: str) -> str:
    """The file's text without a function, and the blank lines after it."""
    at = text.index(helper)
    end = at + len(helper)
    while text[end:end + 1] == "\n":
        end += 1
    return text[:at] + text[end:]


def _with(text: str, helper: str, caller: str) -> str:
    """The file's text with a module-level function put in before the
    top-level definition holding the caller."""
    tree = ast.parse(text)
    name = caller.split(".")[0]
    node = next(one for one in tree.body if isinstance(
        one, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and one.name == name)
    first = node.decorator_list[0].lineno if getattr(
        node, "decorator_list", None) else node.lineno
    lines = text.splitlines(keepends=True)
    return "".join(lines[:first - 1]) + helper.strip("\n") + "\n\n\n" + \
        "".join(lines[first - 1:])


def chain(most: int = 0) -> dict:
    """Each held-out design bug, the helper taken out as well as the caller
    put back: one edit by the editor (as before), then the chain -- the
    designer proposes `PROPOSED` helpers, each put in and wired by the
    editor -- each counted where the commit's tests and those beside them
    pass."""
    import importlib
    import linecache
    from research.v697.coding import Tools
    from research.v698.asking import Subject
    from research.v698.project import Project
    from research.v700 import fixing
    from research.v700 import teach_editor as T
    from research.v705 import inside
    from research.v705.teach_debugging import SAYING
    fixing.VERIFIED_ROUNDS = WIRING_ROUNDS
    found = cases()[:most or None]
    counts = {"bugs": len(found), "one edit": 0, "proposed": 0,
              "compiled": 0, "chain": 0}
    out = []
    for case in found:
        module = importlib.import_module(".".join(
            Path(case["file"]).with_suffix("").parts))
        kept_helper = getattr(module, case["helper"])
        disk = (ROOT / case["file"]).read_text(encoding="utf-8")
        old = _without(disk.replace(commits.functions(disk)[
            case["function"]], case["before"]), case["helper now"])
        function = commits._current(case)
        name = str(ROOT / case["file"]) + commits.PLANTED
        linecache.cache[name] = (len(old), None, old.splitlines(True), name)
        row = {"sha": case["sha"], "function": case["function"],
               "helper": case["helper"], "proposals": []}
        try:
            delattr(module, case["helper"])
            made = inside.changed_function(function, old, filename=name)
            with commits.Put(function, made):
                now = commits._current(case)
                failed = commits.failure(case["tests"][0])
                verify = commits.verifier(case, now)
                held = Project("design")
                held.put({case["file"]: old})
                single = fixing.change(
                    commits.statement(case, failed), held,
                    Subject("function", case["function"], case["file"]),
                    "design", {"editor": "editor-own", "saying": SAYING,
                               "verify": verify,
                               "imports": inside.imports_of(now)})
                row["one edit"] = single.get("status") in ("changed",
                                                           "partly")
                writer = Tools.get().writer(DESIGNER.name)
                proposals = writer.write(
                    [T.prompt(asked(case["message"], case["function"],
                                    case["file"]), case["file"],
                              case["before"])],
                    samples=PROPOSED, longest=600,
                    saying=DESIGNING)[0]
                for proposal in proposals:
                    counts["proposed"] += 1
                    entry = {"proposal": proposal}
                    row["proposals"].append(entry)
                    try:
                        tree = ast.parse(proposal.strip())
                        helper = tree.body[0]
                        assert len(tree.body) == 1 and isinstance(
                            helper, ast.FunctionDef)
                        space = dict(vars(module))
                        exec(compile(proposal.strip(), name, "exec"),
                             space)                  # noqa: S102
                        setattr(module, helper.name, space[helper.name])
                    except Exception:               # noqa: BLE001
                        entry["status"] = "does not compile"
                        continue
                    counts["compiled"] += 1
                    try:
                        wired = _with(old, proposal, case["function"])
                        held = Project("design")
                        held.put({case["file"]: wired})
                        said = commits.statement(case, failed) + (
                            f"\nCall {helper.name} from it: "
                            f"{ast.get_docstring(helper) or ''}")
                        found_ = fixing.change(
                            said, held, Subject("function", case["function"],
                                                case["file"]),
                            "design", {"editor": "editor-own",
                                       "saying": SAYING, "verify": verify,
                                       "imports": inside.imports_of(now)})
                        entry["status"] = found_.get("status")
                        if found_.get("status") in ("changed", "partly"):
                            entry["diff"] = found_.get("diff")
                    finally:
                        if hasattr(module, helper.name) and \
                                helper.name != case["helper"]:
                            delattr(module, helper.name)
                        elif helper.name == case["helper"]:
                            delattr(module, helper.name)
                    if entry["status"] in ("changed", "partly"):
                        break
        except Exception as bad:                    # noqa: BLE001
            row["failed"] = f"{type(bad).__name__}: {bad}"[:200]
        finally:
            setattr(module, case["helper"], kept_helper)
        row["chain"] = any(one.get("status") in ("changed", "partly")
                           for one in row["proposals"])
        counts["one edit"] += bool(row.get("one edit"))
        counts["chain"] += row["chain"]
        out.append(row)
        print(f"{case['sha'][:7]} {case['function']} (needs "
              f"{case['helper']}): one edit {row.get('one edit')}, chain "
              f"{row['chain']}", flush=True)
    (ROOT / "state" / "v705-design-chain.json").write_text(
        json.dumps({**counts, "each": out}, indent=1), encoding="utf-8")
    print(json.dumps(counts))
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("mine", "corpus", "train", "chain"))
    parser.add_argument("--most", type=int, default=0)
    options = parser.parse_args(argv)
    if options.job == "chain":
        chain(options.most)
    else:
        {"mine": mine, "corpus": corpus, "train": train}[options.job]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
