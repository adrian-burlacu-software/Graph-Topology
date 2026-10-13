"""A debugging request answered (v705): the call run in its project, what
the run did said, and a fix written only where the call then gives what
it should.

Read as code talk's act `debug` (`teach_debug_talk`): the call as written
(`CALL`), what it gave (`GOT`), what it should give (`WANT`). The function
is the one the call names, looked up in the project; the call is run in
its module with the project around it, in its own language (the checkers'
`trace project`, in a folder of their own: the project's files on disk are
not run). Said: each line the run took, what changed, what it gave. Where
what it should give was said and the run does not give it, the debugging
editor (`editor-debug`) is asked for the fix, from the call, what it gave
and the run; an answer is written into the project only where the call,
run again on the changed file, gives what it should -- and none that
writes the call it was asked about into the code. Else: unsure, and the
run said.
"""
from __future__ import annotations

import re
from pathlib import Path

from research.v705 import debugging

ROOT = Path(__file__).resolve().parents[2]
EDITOR = ROOT / "llm" / "editor-debug"
#: the editor's answers asked for a fix
SAMPLES = 4
_LOADED: dict = {}


def _language(path: str) -> str | None:
    if path.endswith(".py"):
        return "python"
    if path.endswith((".ts", ".js", ".tsx", ".mts")):
        return "typescript"
    return None


def _editor():
    if not (EDITOR / "config.json").exists():
        return None
    if "editor" not in _LOADED:
        from research.v696.sketcher import Sketcher
        _LOADED["editor"] = Sketcher(EDITOR)
    return _LOADED["editor"]


def called(call: str) -> str | None:
    """The function a call names: the name before its first bracket."""
    found = re.match(r"\s*([A-Za-z_$][\w$.]*)\s*\(", call)
    return found.group(1).split(".")[-1] if found else None


def run(held, file: str, call: str, wanted: str | None) -> dict:
    from research.v696.checker import checker
    language = _language(file)
    running = checker(language)
    try:
        return running.trace_project(dict(held.files), file, call, wanted)
    except Exception as bad:                        # noqa: BLE001
        running.restart()
        return {"ok": False, "error": str(bad)}


def debug(held, call: str, got: str | None, wanted: str | None,
          key=None) -> dict:
    """{said, status, file, function, run, fixed?}: the call run, said;
    fixed where it can be and checked."""
    from research.v698.asking import Subject
    from research.v700 import fixing
    name = called(call)
    found = [one for one in (held.find(name) if held and name else [])
             if _language(one["file"])]
    if not found:
        return {"status": "unplaced", "said":
                f"I couldn't find {name or 'the function'} in the project, "
                f"so I couldn't run {call}."}
    one = found[0]
    file = one["file"]
    traced = run(held, file, call, wanted)
    if not traced.get("ok", True) or "steps" not in traced:
        return {"status": "unrun", "file": file, "said":
                f"I couldn't run {call}: {traced.get('error', 'it failed')}."}
    output = traced["output"]
    text = held.files[file]
    language = _language(file)
    said_run = debugging.said(traced, text, language=language)
    gave = (f"raises {output['error']}" if "error" in output else
            f"gives {output.get('said') or output.get('value')}")
    lines = [f"I ran {call} in {file}: it {gave}.", f"The run: {said_run}."]
    out = {"file": file, "function": one["name"], "run": traced}
    if output.get("right"):
        out.update(status="right", said="\n".join(
            lines + [f"That is what it should give ({wanted})."]))
        return out
    if not wanted:
        out.update(status="traced", said="\n".join(
            lines + ["What should it give? Tell me, and I'll look for the "
                     "fix."]))
        return out
    # the fix: the debugging editor asked, each answer checked by the call
    subject = Subject("function", one["name"], file)
    where = fixing.span(held, subject)
    editor = _editor()
    if where is None or editor is None:
        out.update(status="unsure", said="\n".join(
            lines + ["I can't look for a fix here."]))
        return out
    start, end = where
    all_lines = text.splitlines(keepends=True)
    part = "".join(all_lines[start - 1:end])
    asked = (f"{call} {gave} but should give {wanted}. Fix it.\n"
             f"The run: {said_run}")
    prompt = (f"{asked.strip()}\n\n{Path(file).name}:\n{part.rstrip()}\n")
    from research.v705.teach_debugging import SAYING
    answers = editor.write([prompt], samples=SAMPLES, longest=400,
                           saying=SAYING)[0]
    from research.v700 import teach_editor as T
    for answer in answers:
        made_part = T.applied(part, answer)
        if made_part is None or made_part == part:
            continue
        if call.lower() in made_part.lower():
            # the case written into the code is a fit, not a fix
            continue
        if not made_part.endswith("\n"):
            made_part += "\n"
        changed = "".join(all_lines[:start - 1]) + made_part + \
            "".join(all_lines[end:])
        files = dict(held.files)
        files[file] = changed
        again = run_files(files, file, call, wanted, language)
        if (again.get("output") or {}).get("right"):
            written = held.write(file, changed)
            fixing.MADE.setdefault(key, []).append((file, text, changed))
            out.update(status="changed", fixed=changed, said="\n".join(
                lines + [f"Fixed {one['name']} in {file}"
                         + (", in the file" if written else
                            " (in the project I hold)")
                         + f": {call} now gives {wanted}.",
                         fixing.diff(text, changed, file)]))
            return out
    out.update(status="unsure", said="\n".join(
        lines + ["I didn't find a fix that makes it give "
                 f"{wanted} -- nothing was written."]))
    return out


def run_files(files: dict, file: str, call: str, wanted: str | None,
              language: str) -> dict:
    from research.v696.checker import checker
    running = checker(language)
    try:
        return running.trace_project(files, file, call, wanted)
    except Exception as bad:                        # noqa: BLE001
        running.restart()
        return {"ok": False, "error": str(bad)}
