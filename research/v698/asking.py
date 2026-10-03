"""Questions about a project, and about code pasted into the conversation.

Answered from what the compiler reads -- the outline, the calls resolved
across files, the diagnostics, a function's structure -- never made up:

    overview    what is in the project
    files       which files, how big
    in file     the functions of one file
    where       where a function is defined
    callers     who calls it
    calls       what it calls (the project's, then the library's)
    explain     what a function is: its signature, lines, what it is made
                of and what gives its result, who calls it, what it calls
    errors      what the compiler says is wrong

Code pasted with a question about it (*explain the selection*) is read the
same way, and becomes the conversation's code (`v697.conversation`): *what
does it do*, a call, *how sure* follow it as they follow code written here.
"""
from __future__ import annotations

import re

from research.v698 import project as Pj

IDENT = re.compile(r"[A-Za-z_$][\w$]*")
NAMED = (
    ("callers", re.compile(r"\b(?:who|what|which(?: functions?)?) "
                           r"(?:calls|uses)\s+([\w$.]+)|\bcallers of "
                           r"([\w$.]+)|\bwhere is ([\w$.]+) (?:called|used)",
                           re.I)),
    ("calls", re.compile(r"\bwhat does ([\w$.]+) (?:call|use)\b", re.I)),
    ("explain", re.compile(r"\bwhat does ([\w$.]+) do\b|\bexplain "
                           r"([\w$.]+)|\bdescribe ([\w$.]+)|\bwhat is "
                           r"([\w$.]+)\b", re.I)),
    ("where", re.compile(r"\bwhere(?:'s| is| are)? ([\w$.]+)\b|\bfind "
                         r"([\w$.]+)|\bgo to ([\w$.]+)|\bshow me "
                         r"([\w$.]+)", re.I)),
)
IN_FILE = re.compile(r"\b(?:what(?:'s| is) in|functions (?:in|of)|outline "
                     r"(?:of )?)\s*([\w./-]+\.(?:[cm]?[jt]sx?))\b", re.I)
FILES = re.compile(r"\b(which|what) files\b|\blist (the )?files\b", re.I)
ERRORS = re.compile(r"\b(compile|compiler|type errors?|diagnostics|"
                    r"errors? in the (project|code))\b|\bwhat(?:'s| is) "
                    r"(wrong|broken)\b", re.I)
#: the project, by any of the names people give it
PROJECT = r"(?:(?:the|this|my|our|your) (?:project|workspace|repo(?:sitory)?|code ?base|app(?:lication)?))"
#: a question whose subject is the project: what it is, what it does, what
#: is in it, how big, tell me / describe / summarise / explain it
ABOUT = re.compile(r"\b(what|which|how|tell me|describe|summari[sz]e|"
                   r"explain|overview|show me|walk me through)\b", re.I)
#: pasted code with a question about it
FENCE = re.compile(r"```[\w-]*\n(.*?)```", re.S)
ABOUT_CODE = re.compile(r"\b(explain|what does|what is|describe|how does|"
                        r"read)\b", re.I)


def classify(text: str, held) -> tuple:
    """(kind, detail) of a question about the project `held`, or (None,
    None)."""
    if held is None or not held.files:
        return None, None
    for kind, pattern in NAMED:
        found = pattern.search(text)
        if found:
            name = next(one for one in found.groups() if one)
            if held.find(name):
                return kind, name
    found = IN_FILE.search(text)
    if found:
        path = _file(held, found.group(1))
        if path:
            return "in file", path
    if FILES.search(text):
        return "files", None
    if ERRORS.search(text):
        return "errors", None
    if ABOUT.search(text) and (re.search(rf"\b{PROJECT}\b", text, re.I)
                               or _named(held, text)):
        return "overview", None
    return None, None


def _named(held, text: str) -> bool:
    """Whether the project is named by its folder's name (`hello-world`,
    `hello world`)."""
    name = (held.name or "").lower()
    if len(name) < 3:
        return False
    said = re.sub(r"[\s_-]+", " ", text.lower())
    return re.sub(r"[\s_-]+", " ", name) in said


def _file(held, said: str) -> str | None:
    said = said.replace("\\", "/").lstrip("./")
    if said in held.files:
        return said
    return next((one for one in sorted(held.files)
                 if one.endswith("/" + said) or one == said), None)


def _reply(kind: str, text: str, spoken: str, looked: dict, blocks=(),
           name: str = "") -> dict:
    return {"outcome": "acted", "source": "the project", "text": spoken,
            "spoken": spoken, "code": {
                "project": {"asked": kind, "said": text, "looked": looked,
                            "name": name},
                "blocks": list(blocks),
                "answer": {"status": "project", "code": None}}}


def _at(one: dict) -> str:
    return f"{one['file']}:{one['start']}"


def _signature(one: dict) -> str:
    params = ", ".join(f"{name}: {kind}" if kind else name
                       for name, kind in one["params"])
    return f"{one['name']}({params})" + (f": {one['returns']}"
                                         if one["returns"] else "")


def _source(held, one: dict) -> str:
    lines = held.files[one["file"]].splitlines()
    return "\n".join(lines[one["start"] - 1:one["end"]])


def answered(text: str, held) -> dict | None:
    kind, detail = classify(text, held)
    if kind is None:
        return None
    return {"overview": _overview, "files": _files, "in file": _in_file,
            "where": _where, "callers": _callers, "calls": _calls,
            "explain": _explain, "errors": _errors}[kind](held, text, detail)


def _overview(held, text, _):
    """What the project is: what its files say of themselves, what runs
    when they are run, the functions nothing else calls (where it is
    entered), the most called -- and how big it is. Read, not guessed:
    the comments are the project's own words."""
    summary = held.summary()
    outline = held.outline()
    called = {one["to"] for one in held.calls()}
    parts = [f"{held.name or 'The project'}: {summary['files']} files, "
             f"{summary['functions']} functions, {summary['lines']} lines."]
    said_of = [(path, read["about"]) for path, read in sorted(outline.items())
               if read.get("about")]
    for path, about in said_of[:6]:
        parts.append(f"{path} says of itself: \u201c{about}\u201d")
    runs = [(path, one) for path, read in sorted(outline.items())
            for one in read.get("runs", ())]
    for path, one in runs[:4]:
        doc = next((f["doc"] for f in read_functions(outline, path)
                    if f["name"] == one["call"] and f.get("doc")), None)
        parts.append(f"Running {path} calls {one['call']}()"
                     + (f": \u201c{doc.rstrip('.')}\u201d" if doc else "")
                     + ".")
    # run at the top of its file: called, by running it
    ran = {(path, one["call"]) for path, one in runs}
    entered = [one for one in held.functions()
               if f"{one['file']}#{one['name']}" not in called
               and (one["file"], one["name"]) not in ran
               and one["exported"]]
    if entered:
        parts.append("Nothing in it calls " + ", ".join(
            one["name"] for one in entered[:8]) + (" …" if len(entered) > 8
                                                   else "")
                     + ": that is where it is used from.")
    most = ", ".join(f"{end.split('#')[1]} ({count})"
                     for end, count in summary["most called"][:4])
    if most:
        parts.append(f"Most called: {most}.")
    documented = [one for one in held.functions() if one.get("doc")]
    return _reply("overview", text, " ".join(parts),
                  {**summary, "files said": dict(said_of),
                   "runs": [{"file": path, **one} for path, one in runs],
                   "entered from": [f"{one['file']}#{one['name']}"
                                    for one in entered],
                   "documented": [{"function": one["name"],
                                   "file": one["file"], "doc": one["doc"]}
                                  for one in documented]})


def read_functions(outline: dict, path: str) -> list:
    return (outline.get(path) or {}).get("functions", [])


def _files(held, text, _):
    outline = held.outline()
    rows = [{"file": path, "lines": read["lines"],
             "functions": len(read["functions"])}
            for path, read in sorted(outline.items())]
    said = f"{len(rows)} files: " + ", ".join(
        f"{one['file']} ({one['functions']})" for one in rows[:20]) + (
        " …" if len(rows) > 20 else "") + "."
    return _reply("files", text, said, {"files": rows})


def _in_file(held, text, path):
    read = held.outline()[path]
    said = (f"{path} has {read['lines']} lines and "
            f"{len(read['functions'])} functions: "
            + ", ".join(f"{one['name']} (line {one['start']})"
                        for one in read["functions"][:30]) + "."
            + (f" It imports from {', '.join(one['from'] for one in read['imports'])}."
               if read["imports"] else ""))
    return _reply("in file", text, said, read)


def _where(held, text, name):
    found = held.find(name)
    said = f"{name} is defined in " + "; ".join(
        f"{_at(one)}-{one['end']} as {_signature(one)}"
        + (" (exported)" if one["exported"] else "") for one in found) + "."
    return _reply("where", text, said, {"found": found},
                  [{"title": f"{one['file']}:{one['start']}",
                    "kind": one["name"], "code": _source(held, one),
                    "file": one["file"], "line": one["start"]}
                   for one in found[:3]], name)


def _callers(held, text, name):
    found = held.callers(name)
    if not found:
        said = f"Nothing in the project calls {name}."
    else:
        said = f"{name} is called by " + ", ".join(
            f"{one['from'].split('#')[1]} ({one['from'].split('#')[0]})"
            for one in found) + "."
    return _reply("callers", text, said, {"callers": found}, (), name)


def _calls(held, text, name):
    rows = []
    for one in held.find(name):
        mine = {call["call"]: call["to"] for call in held.calls()
                if call["from"] == f"{one['file']}#{one['name']}"}
        rows.append({"function": _at(one), "project": sorted(mine.items()),
                     "library": [c for c in one["calls"] if c not in mine]})
    first = rows[0]
    said = (f"{name} calls "
            + (", ".join(f"{call} ({end.split('#')[0]})" for call, end in
                         first["project"]) or "nothing of the project's")
            + (f"; and from the language and libraries: "
               f"{', '.join(first['library'])}" if first["library"] else "")
            + ".")
    return _reply("calls", text, said, {"calls": rows}, (), name)


def _explain(held, text, name):
    """A function as the compiler reads it: where it is, its signature,
    what it is made of and what gives its result, who calls it and what
    it calls."""
    from research.v696 import meaning as M
    from research.v696.checker import CheckerError
    parts, looked, blocks = [], [], []
    for one in held.find(name)[:3]:
        try:
            uses, root = M.structure(held.files[one["file"]], one["name"])
        except CheckerError:
            uses, root = frozenset(), None
        made = [word for word in sorted(uses) if word not in
                ("a name", "a literal", "let", "const", "return")]
        callers = held.callers(one["name"])
        said = (f"{_signature(one)}, at {_at(one)}-{one['end']}"
                + (", exported" if one["exported"] else ""))
        if made:
            said += f"; made of {', '.join(made[:14])}"
        if root:
            said += f"; its result comes from {root}"
        said += (f"; called by {len(callers)} function(s)" if callers
                 else "; nothing in the project calls it")
        parts.append(said)
        looked.append({**one, "uses": made, "root": root,
                       "callers": callers})
        blocks.append({"title": f"{one['file']}:{one['start']}",
                       "kind": one["name"], "code": _source(held, one),
                       "file": one["file"], "line": one["start"]})
    return _reply("explain", text, " ".join(f"{name}: {one}." for one in
                                            parts), {"functions": looked},
                  blocks, name)


def _errors(held, text, _):
    found = held.diagnostics()
    if not found:
        said = "The compiler finds nothing wrong in the project."
    else:
        files = sorted({one["file"] for one in found})
        said = (f"The compiler finds {len(found)} problem(s) in "
                f"{len(files)} file(s): " + "; ".join(
                    f"{one['file']}:{one['line']} "
                    f"{one['message'].rstrip('.')}"
                    for one in found[:6]) + ("; …" if len(found) > 6
                                             else "") + ".")
    return _reply("errors", text, said, {"diagnostics": found[:200]})


# -- code pasted, with a question about it -------------------------------------

def pasted(text: str) -> str | None:
    """The code in an utterance that pastes some and asks about it."""
    found = FENCE.search(text)
    if found and ABOUT_CODE.search(text[:found.start()] + " "
                                   + text[found.end():]):
        return found.group(1)
    return None


def read_pasted(text: str, code: str, space) -> dict:
    """Pasted code read as the conversation's code, then explained as
    code written here is (`v697.conversation._explain`)."""
    from research.v696.checker import checker
    from research.v697 import coding
    from research.v697 import conversation as C
    source, signature = coding._function(code)
    if signature is None:
        return _reply("pasted", text, "I could not find a function in what "
                      "you pasted.", {})
    entry = signature.group(1)
    from research.v696.tasks import _params
    params = [kind for _, kind in _params(signature.group(2))]
    inputs = [[coding.DEFAULTS.get(kind, [None])[at % len(
        coding.DEFAULTS.get(kind, [None]))] for kind in params]
        for at in range(4)] if params else [[]]
    space.request = {"entry": entry, "english": "(pasted)", "said": text,
                     "signature": None, "examples": [], "made": [],
                     "params": [], "returns": None}
    space.found = {"request": space.request,
                   "answer": {"code": source, "entry": entry,
                              "status": "pasted"},
                   "open": {"inputs": inputs}}
    answer = C._explain(space, text)
    outline = checker().outline({"/pasted.ts": source})["/pasted.ts"]
    problems = checker().diagnose({"/pasted.ts": source})
    found = answer["code"]
    found["followup"]["asked"] = "pasted"
    found["followup"]["looked"].update(outline=outline,
                                       diagnostics=problems[:50])
    if problems:
        answer["text"] = answer["spoken"] = (
            answer["spoken"] + f" Alone, the compiler finds {len(problems)} "
            f"problem(s) in it (names from elsewhere, most likely).")
    return answer
