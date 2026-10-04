"""Questions about code: a subject and an aspect, each read on its own.

Not one pattern a question. What is asked is two things, read apart, and
any aspect is answered for any subject it means something for:

    subject   the project (by the names people give one, or its folder's);
              a file of it; a function of it; the conversation's own code
              (the answer written last, or pasted); `it` / `this` / `the
              code` -- whatever was talked about last (`FOCUS`)
    aspect    a small closed vocabulary (`ASPECTS`): what it is, where, how
              big, who calls it, what it calls, its gaps (bugs), how sure,
              why that one, what else, its risks, its files, its functions

Where an aspect means nothing for a subject, the answer says what can be
asked of it -- never a turn left to the reasoner about kinds, which reads
*any bugs in this project* as a question about insects. Every answer offers
the other aspects of its subject to ask next (`suggest`).

**Gaps** are what *bugs* is asked as: what the code promises -- its
declared types, its name and doc comment, the examples given, being used at
all -- against what is found -- the compiler (strict, too), running it, the
call graph. A gap is *contradicted* (a bug), *open* (promised, nothing yet
checks it: with the question that would settle it, `greeting("Ann") == ?`),
or *unused*. Answered from what is read and run, never made up.

Code pasted with a question about it (*explain the selection*) is read the
same way and becomes the conversation's code.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from research.v698 import project as Pj

IDENT = re.compile(r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?")
PATH = re.compile(r"[\w./-]+\.(?:[cm]?[jt]sx?)\b")
WORD = re.compile(r"[a-z']+")
#: the project, by any of the names people give it
PROJECT = re.compile(r"\b(?:the|this|my|our|your|whole) (?:project|workspace|"
                     r"repo(?:sitory)?|code ?base|app(?:lication)?)\b", re.I)
#: what was talked about last
PRONOUN = re.compile(r"\b(?:it|its|this|that|these|those|the (?:code|"
                     r"function|program|answer|method|script))\b", re.I)

#: the aspects, in the order they are preferred where several are said
ASPECTS = (
    ("bugs", {"bug", "bugs", "buggy", "error", "errors", "wrong", "broken",
              "problem", "problems", "issue", "issues", "mistake",
              "mistakes", "compile", "compiles", "lint", "review", "gap",
              "gaps", "fix", "flaw", "flaws", "faulty"}),
    ("sure", {"sure", "confident", "confidence", "trust", "confirmed",
              "certain", "verified", "reliable", "correct", "right"}),
    ("why", {"why", "reason", "chose", "choose", "chosen", "picked",
             "wrote", "written"}),
    ("others", {"else", "other", "others", "alternative", "alternatives",
                "instead", "different"}),
    ("risk", {"risk", "risks", "risky", "danger", "dangerous"}),
    ("where", {"where", "located", "location", "defined", "declared",
               "find", "locate", "open"}),
    ("size", {"big", "large", "size", "long", "lines", "many", "count"}),
    ("files", {"files", "modules"}),
    ("functions", {"functions", "methods", "procedures"}),
)
CALLING = {"call", "calls", "called", "caller", "callers", "calling", "use",
           "uses", "used", "user", "users", "import", "imports", "imported",
           "depend", "depends", "dependency", "dependencies", "dependents",
           "references", "referenced"}
#: what asks what a thing is, when no other aspect is said
CUES = {"what", "describe", "explain", "tell", "summarize", "summarise",
        "overview", "purpose", "about", "show", "whats", "what's"}
#: what makes an aspect said with no subject the project's: `are there any
#: bugs` is; `do bugs have legs` is not
DEFAULTS_TO_PROJECT = {"any", "there", "find", "check", "my", "our", "this",
                       "the", "in"}
#: the words a question is built of, beside its subject and its aspect: an
#: utterance with any other word has a subject of its own (`do bugs have
#: legs`, `where is Mary`) and does not borrow the last one
FRAME = {"a", "an", "the", "this", "that", "these", "those", "it", "its",
         "my", "our", "your", "you", "i", "me", "we", "us", "they", "them",
         "is", "are", "was", "were", "be", "been", "am", "do", "does", "did",
         "have", "has", "had", "can", "could", "would", "should", "will",
         "may", "might", "must", "shall", "there", "any", "some", "all",
         "of", "in", "on", "at", "for", "to", "from", "with", "by", "into",
         "how", "who", "whom", "which", "when", "and", "or", "but", "not",
         "no", "so", "if", "then", "please", "now", "again", "also", "just",
         "really", "very", "much", "more", "most", "find", "check", "look",
         "see", "tell", "show", "give", "get", "think", "know", "say",
         "said", "write", "wrote", "make", "made", "got", "one", "ones",
         "thing", "things", "code", "function", "program", "answer",
         "method", "script", "project", "file", "you're", "it's", "is't",
         "isn't", "aren't", "don't", "doesn't", "didn't", "can't", "won't"}


def _own_subject(words: list) -> bool:
    """Whether the utterance says something besides question, aspect and
    the words for code: a subject of its own."""
    vocabulary = set().union(*(one for _, one in ASPECTS), CALLING, CUES,
                             FRAME)
    return any(word not in vocabulary and len(word) > 1 for word in words)


@dataclass(frozen=True)
class Subject:
    #: project | file | function | code
    kind: str
    name: str
    file: str = ""


#: conversation -> (Subject, turn): what was talked about last
FOCUS: dict = {}


# -- reading: the subject, the aspect ----------------------------------------------

def _explicit(text: str, held, space) -> Subject | None:
    """A subject named outright: a function of the project, a file, the
    project, the conversation's code by its function's name."""
    entry = ((space.found or {}).get("answer") or {}).get("entry") \
        if space is not None else None
    if held is not None and held.files:
        for found in IDENT.finditer(text):
            name = found.group(0)
            if name == entry:
                return Subject("code", name)
            if len(name) > 1 and held.find(name):
                one = held.find(name)[0]
                return Subject("function", one["name"], one["file"])
        for found in PATH.finditer(text):
            path = _file(held, found.group(0))
            if path:
                return Subject("file", path, path)
        if PROJECT.search(text) or _named(held, text):
            return Subject("project", held.name or "the project")
    elif entry and re.search(rf"(?<![\w$]){re.escape(entry)}(?![\w$])",
                             text):
        return Subject("code", entry)
    return None


def _aspect(words: list, text: str, subject: Subject | None) -> str | None:
    said = set(words)
    if said & CALLING and subject is not None:
        # `what does main call`: the subject does the calling; `who calls
        # main`: it is called
        at = text.lower().find(subject.name.lower())
        verb = min((text.lower().find(w) for w in said & CALLING), default=-1)
        return ("calls" if at != -1 and verb > at and re.search(
            r"\b(does|do|did)\b", text, re.I) else "callers")
    for name, vocabulary in ASPECTS:
        if said & vocabulary:
            return name
    return None


def route(text: str, held, space, turn: int, key) -> tuple:
    """(aspect, subject) of a question about code, or (None, None) where it
    is not one -- or is a request, more of one, or a call (`v697`)."""
    from research.v697 import conversation as C
    if pasted(text) is not None:
        return None, None
    if space is not None and space.request is not None:
        kind = C.classify(text, space, turn)[0]
        if kind in ("request", "more", "call"):
            return None, None
    else:
        from research.v697 import coding
        if coding.read(text).get("asked"):
            return None, None
    words = WORD.findall(text.lower())
    subject = _explicit(text, held, space)
    pronoun = subject is None and PRONOUN.search(text) is not None
    if pronoun:
        subject = _focus(key, space, turn)
    aspect = _aspect(words, text, subject)
    if subject is None and _own_subject(words):
        return None, None
    if subject is None:
        recent = _focus(key, space, turn)
        if aspect and recent is not None and not pronoun:
            subject = recent
        elif held is not None and held.files and (
                aspect in ("files", "functions") or (
                    aspect in ("bugs", "size")
                    and set(words) & DEFAULTS_TO_PROJECT)):
            subject = Subject("project", held.name or "the project")
    if subject is None:
        return None, None
    if aspect is None:
        if set(words) & CUES:
            aspect = "explain"
        elif text.rstrip().endswith("?") and not pronoun:
            # something named outright, asked about: what can be asked of it
            aspect = "capabilities"
        else:
            # `can it swim?` just after code: not about the code
            return None, None
    return aspect, subject


def _focus(key, space, turn: int) -> Subject | None:
    """What `it` is: the last thing talked about, two turns back at most --
    the conversation's code where that was answered last."""
    held, at = FOCUS.get(key, (None, -10))
    code_at = getattr(space, "turn", -10) if space is not None and \
        space.found else -10
    if code_at >= at and turn - code_at <= 2:
        entry = (space.found.get("answer") or {}).get("entry") or "the code"
        return Subject("code", entry)
    if held is not None and turn - at <= 2:
        return held
    return None


# -- answering -------------------------------------------------------------------------

#: what can be asked of each kind of subject
CAN = {
    "project": ("explain", "bugs", "files", "functions", "size"),
    "file": ("explain", "bugs", "functions", "size", "callers", "calls",
             "where"),
    "function": ("explain", "bugs", "where", "callers", "calls", "size"),
    "code": ("explain", "bugs", "sure", "why", "others", "risk", "size"),
}
#: an aspect asked of a subject, as a question to offer
ASKING = {"explain": "what does {n} do", "bugs": "are there any bugs in {n}",
          "files": "which files are in {n}", "functions": "which functions "
          "are in {n}", "size": "how big is {n}", "callers": "who calls {n}",
          "calls": "what does {n} call", "where": "where is {n}",
          "sure": "how sure are you of {n}", "why": "why {n}",
          "others": "what else did you write", "risk": "what are the risks "
          "of {n}"}


def answered(text: str, held, space, turn: int, key) -> dict | None:
    aspect, subject = route(text, held, space, turn, key)
    if aspect is None:
        return None
    return about(aspect, subject, text, held, space, turn, key)


# -- what the encoder read, resolved exactly ---------------------------------------

def resolve(reading, held, space, turn: int, key) -> Subject | None:
    """What the encoder's subject is: its kind as read, and what a phrase
    it marked names -- a function of the project or the conversation's
    code by name, a file by its path -- looked up, not read."""
    entry = ((space.found or {}).get("answer") or {}).get("entry") \
        if space is not None else None
    phrase = (reading.spans.get("SUBJ") or [""])[0].strip()
    if not phrase and reading.subject in ("named", "file") and \
            len(getattr(reading, "known", ())) == 1:
        # the encoder read a subject named but marked no words: the one word
        # here known to be code -- found by lookup, not by reading -- is it
        phrase = reading.known[0]
    name = phrase.split("(")[0].strip(" ?.,!'\"`")
    if reading.subject == "project" and held is not None:
        return Subject("project", held.name or "the project")
    if reading.subject in ("file", "named") and name:
        if name == entry:
            return Subject("code", name)
        if held is not None and held.files:
            path = _file(held, name)
            if path:
                return Subject("file", path, path)
            found = held.find(name)
            if found:
                return Subject("function", found[0]["name"],
                               found[0]["file"])
        return Subject("unknown", name)
    if reading.subject == "last" or reading.subject == "none":
        recent = _focus(key, space, turn)
        if recent is not None:
            return recent
        if held is not None and held.files:
            return Subject("project", held.name or "the project")
    return None


def answered_read(reading, text: str, held, space, turn: int,
                  key) -> dict | None:
    """A question the encoder read (`reading.Reading`, act `ask`)."""
    subject = resolve(reading, held, space, turn, key)
    if subject is None:
        return None
    if subject.kind == "unknown":
        said = (f"I don't know {subject.name}: it is not a function or "
                f"file of the project, nor the code we were writing."
                + ("" if held is not None else
                   " (No project is read: Read the Project in the editor.)"))
        return _reply("unknown", text, said, {"read": reading.json()},
                      name=subject.name)
    aspect = reading.aspect if reading.aspect != "none" else "explain"
    found = about(aspect, subject, text, held, space, turn, key)
    found["code"]["read"] = reading.json()
    return found


def about(aspect: str, subject: Subject, text: str, held, space, turn: int,
          key) -> dict:
    """An aspect of a subject, answered -- however it was read."""
    if aspect not in CAN[subject.kind] or aspect == "capabilities":
        found = _capabilities(held, space, subject, aspect, text)
    elif subject.kind == "code":
        found = _code(held, space, subject, aspect, text)
    else:
        found = ANSWERS[aspect](held, subject, text)
    FOCUS[key] = (subject, turn)
    found.setdefault("suggest", _suggest(held, subject, aspect))
    found["code"]["subject"] = {"kind": subject.kind, "name": subject.name,
                                "file": subject.file, "aspect": aspect}
    return found


def _suggest(held, subject: Subject, aspect: str) -> list:
    name = ("the project" if subject.kind == "project" else
            "it" if subject.kind == "code" else subject.name)
    out = [{"text": ASKING[one].format(n=name), "send": True}
           for one in CAN[subject.kind] if one != aspect][:4]
    if subject.kind == "function" and held is not None:
        one = held.find(subject.name)
        if one:
            out.append({"text": _template(one[0]), "send": False})
    return out


def _reply(kind: str, text: str, spoken: str, looked: dict, blocks=(),
           name: str = "", suggest=None) -> dict:
    found = {"outcome": "acted", "source": "the project", "text": spoken,
             "spoken": spoken, "code": {
                 "project": {"asked": kind, "said": text, "looked": looked,
                             "name": name},
                 "blocks": list(blocks),
                 "answer": {"status": "project", "code": None}}}
    if suggest is not None:
        found["suggest"] = suggest
    return found


def _capabilities(held, space, subject: Subject, aspect, text: str) -> dict:
    what = {"project": "the project", "file": subject.name,
            "function": subject.name, "code": subject.name}[subject.kind]
    can = ", ".join(ASKING[one].format(n="it") for one in CAN[subject.kind])
    said = (f"I don't know what {aspect} would be for {what}" if aspect and
            aspect != "capabilities" else f"About {what}") + \
        f", I can tell you: {can}."
    return _reply("capabilities", text, said, {"can": CAN[subject.kind]},
                  name=subject.name)


def _file(held, said: str) -> str | None:
    said = said.replace("\\", "/").lstrip("./")
    if said in held.files:
        return said
    return next((one for one in sorted(held.files)
                 if one.endswith("/" + said) or one == said), None)


def _named(held, text: str) -> bool:
    """Whether the project is named by its folder's name (`hello-world`,
    `hello world`)."""
    name = (held.name or "").lower()
    if len(name) < 3:
        return False
    said = re.sub(r"[\s_-]+", " ", text.lower())
    return re.sub(r"[\s_-]+", " ", name) in said


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


def _block(held, one: dict) -> dict:
    return {"title": f"{one['file']}:{one['start']}", "kind": one["name"],
            "code": _source(held, one), "file": one["file"],
            "line": one["start"]}


def _template(one: dict) -> str:
    """A call of a function with inputs of its types, its result asked:
    what an example of it would be."""
    from research.v697.coding import DEFAULTS
    args = []
    for _, kind in one["params"]:
        values = DEFAULTS.get((kind or "").replace(" ", ""))
        args.append(json.dumps(values[min(2, len(values) - 1)])
                    if values else "…")
    return f"{one['name'].split('.')[-1]}({', '.join(args)}) == ?"


def _scope(held, subject: Subject) -> list:
    """The functions a subject holds."""
    if subject.kind == "function":
        return held.find(subject.name)[:1]
    if subject.kind == "file":
        return [one for one in held.functions() if one["file"] == subject.file]
    return held.functions()


def _within(held, subject: Subject, path: str, line: int) -> bool:
    if subject.kind == "project":
        return True
    if subject.kind == "file":
        return path == subject.file
    one = held.find(subject.name)[0]
    return path == one["file"] and one["start"] <= line <= one["end"]


# -- what it is, where, how big, who calls it, what it calls ---------------------

def _explain(held, subject: Subject, text: str) -> dict:
    if subject.kind == "project":
        return _overview(held, text)
    if subject.kind == "file":
        return _a_file(held, subject, text)
    return _a_function(held, subject, text)


def _overview(held, text: str) -> dict:
    """What the project is: what its files say of themselves, what runs
    when they are run, where it is entered from, the most called -- and how
    big it is. The comments are the project's own words."""
    summary = held.summary()
    outline = held.outline()
    called = {one["to"] for one in held.calls()}
    parts = [f"{held.name or 'The project'}: {summary['files']} files, "
             f"{summary['functions']} functions, {summary['lines']} lines."]
    said_of = [(path, read["about"]) for path, read in sorted(outline.items())
               if read.get("about")]
    for path, about in said_of[:6]:
        parts.append(f"{path} says of itself: “{about}”")
    runs = [(path, one) for path, read in sorted(outline.items())
            for one in read.get("runs", ())]
    for path, one in runs[:4]:
        doc = next((f["doc"] for f in outline[path]["functions"]
                    if f["name"] == one["call"] and f.get("doc")), None)
        parts.append(f"Running {path} calls {one['call']}()"
                     + (f": “{doc.rstrip('.')}”" if doc else "")
                     + ".")
    ran = {(path, one["call"]) for path, one in runs}
    entered = [one for one in held.functions()
               if f"{one['file']}#{one['name']}" not in called
               and (one["file"], one["name"]) not in ran and one["exported"]]
    if entered:
        parts.append("Nothing in it calls " + ", ".join(
            one["name"] for one in entered[:8]) + (" …" if len(entered) > 8
                                                   else "")
                     + ": that is where it is used from.")
    most = ", ".join(f"{end.split('#')[1]} ({count})"
                     for end, count in summary["most called"][:4])
    if most:
        parts.append(f"Most called: {most}.")
    return _reply("explain", text, " ".join(parts),
                  {**summary, "files said": dict(said_of),
                   "runs": [{"file": path, **one} for path, one in runs],
                   "entered from": [f"{one['file']}#{one['name']}"
                                    for one in entered]},
                  name=held.name)


def _a_file(held, subject: Subject, text: str) -> dict:
    read = held.outline()[subject.file]
    importers = sorted({path for path, other in held.outline().items()
                        for one in other["imports"]
                        if _imports(held, path, one["from"], subject.file)})
    said = (f"{subject.file}: {read['lines']} lines"
            + (f", saying of itself “{read['about']}”"
               if read.get("about") else "")
            + f"; {len(read['functions'])} functions: "
            + (", ".join(f"{one['name']} (line {one['start']})"
                         for one in read["functions"][:20]) or "none")
            + "." + (f" It imports from "
                     f"{', '.join(one['from'] for one in read['imports'])}."
                     if read["imports"] else "")
            + (f" Imported by {', '.join(importers)}." if importers else ""))
    return _reply("explain", text, said, {**read, "imported by": importers},
                  name=subject.file)


def _imports(held, path: str, said: str, target: str) -> bool:
    import posixpath
    if not said.startswith("."):
        return False
    base = posixpath.normpath(posixpath.join(posixpath.dirname(path), said))
    return target in {base + ending for ending in ("", *Pj.READ)}


def _a_function(held, subject: Subject, text: str) -> dict:
    """A function as the compiler reads it: where it is, its signature, its
    doc, what it is made of and what gives its result, who calls it."""
    from research.v696 import meaning as M
    from research.v696.checker import CheckerError
    one = held.find(subject.name)[0]
    try:
        uses, root = M.structure(held.files[one["file"]], one["name"])
    except CheckerError:
        uses, root = frozenset(), None
    made = [word for word in sorted(uses) if word not in
            ("a name", "a literal", "let", "const", "return")]
    callers = held.callers(one["name"])
    said = (f"{_signature(one)}, at {_at(one)}-{one['end']}"
            + (", exported" if one["exported"] else "")
            + (f"; its doc says “{one['doc']}”" if one.get("doc")
               else "")
            + (f"; made of {', '.join(made[:14])}" if made else "")
            + (f"; its result comes from {root}" if root else "")
            + (f"; called by {len(callers)} function(s)" if callers
               else "; nothing in the project calls it") + ".")
    return _reply("explain", text, said, {**one, "uses": made, "root": root,
                                          "callers": callers},
                  [_block(held, one)], subject.name)


def _where(held, subject: Subject, text: str) -> dict:
    if subject.kind == "file":
        read = held.outline()[subject.file]
        return _reply("where", text, f"{subject.file}, {read['lines']} "
                      f"lines.", {"file": subject.file},
                      [{"title": subject.file, "kind": "the file",
                        "code": "\n".join(held.files[subject.file]
                                          .splitlines()[:30]),
                        "file": subject.file, "line": 1}], subject.name)
    found = held.find(subject.name)
    said = f"{subject.name} is defined in " + "; ".join(
        f"{_at(one)}-{one['end']} as {_signature(one)}"
        + (" (exported)" if one["exported"] else "") for one in found) + "."
    return _reply("where", text, said, {"found": found},
                  [_block(held, one) for one in found[:3]], subject.name)


def _size(held, subject: Subject, text: str) -> dict:
    if subject.kind == "project":
        summary = held.summary()
        said = (f"{summary['files']} files, {summary['lines']} lines, "
                f"{summary['functions']} functions ({summary['exported']} "
                f"exported).")
        return _reply("size", text, said, summary, name=subject.name)
    if subject.kind == "file":
        read = held.outline()[subject.file]
        return _reply("size", text, f"{subject.file}: {read['lines']} lines, "
                      f"{len(read['functions'])} functions.", read,
                      name=subject.name)
    one = held.find(subject.name)[0]
    return _reply("size", text, f"{subject.name}: {one['end'] - one['start'] + 1}"
                  f" lines, {len(one['params'])} parameter(s), "
                  f"{len(one['calls'])} distinct call(s).", one,
                  name=subject.name)


def _files(held, subject: Subject, text: str) -> dict:
    rows = [{"file": path, "lines": read["lines"],
             "functions": len(read["functions"])}
            for path, read in sorted(held.outline().items())]
    said = f"{len(rows)} files: " + ", ".join(
        f"{one['file']} ({one['functions']})" for one in rows[:20]) + (
        " …" if len(rows) > 20 else "") + "."
    return _reply("files", text, said, {"files": rows}, name=subject.name)


def _functions(held, subject: Subject, text: str) -> dict:
    found = _scope(held, subject)
    said = (f"{len(found)} function(s): " + ", ".join(
        f"{one['name']} ({_at(one)})" for one in found[:30])
        + (" …" if len(found) > 30 else "") + ".")
    return _reply("functions", text, said, {"functions": found},
                  name=subject.name)


def _callers(held, subject: Subject, text: str) -> dict:
    if subject.kind == "file":
        return _a_file(held, subject, text)
    found = held.callers(subject.name)
    said = (f"Nothing in the project calls {subject.name}." if not found
            else f"{subject.name} is called by " + ", ".join(
                f"{one['from'].split('#')[1]} ({one['from'].split('#')[0]})"
                for one in found) + ".")
    return _reply("callers", text, said, {"callers": found},
                  name=subject.name)


def _calls(held, subject: Subject, text: str) -> dict:
    if subject.kind == "file":
        return _a_file(held, subject, text)
    one = held.find(subject.name)[0]
    mine = {call["call"]: call["to"] for call in held.calls()
            if call["from"] == f"{one['file']}#{one['name']}"}
    library = [c for c in one["calls"] if c not in mine]
    said = (f"{subject.name} calls "
            + (", ".join(f"{call} ({end.split('#')[0]})" for call, end in
                         sorted(mine.items())) or "nothing of the project's")
            + (f"; and from the language and libraries: "
               f"{', '.join(library)}" if library else "") + ".")
    return _reply("calls", text, said, {"project": mine, "library": library},
                  name=subject.name)


# -- gaps: what it promises against what is found ----------------------------------

#: what a strict-only diagnostic means, by the compiler's code
STRICT_SAYS = {7030: "a path returns nothing", 6133: "declared, never read",
               6196: "declared, never used", 6198: "nothing of it is read",
               7027: "code never reached", 7029: "a case falls through"}


def _gaps(held, subject: Subject, text: str) -> dict:
    """The project's (a file's, a function's) gaps: the compiler against
    the declared types (strict too: what finds bugs), the call graph
    against being used, and what nothing checks yet -- each exported
    function whose result nothing gives an example of."""
    contradicted, unused, open_ = [], [], []
    for one in held.diagnostics(strict=True):
        if not _within(held, subject, one["file"], one["line"]):
            continue
        where = held.at(one["file"], one["line"])
        contradicted.append({
            "where": f"{one['file']}:{one['line']}",
            "in": where["name"] if where else None,
            "promise": "its declared types" if not one.get("strict")
            else "what the code says it does",
            "found": one["message"].rstrip("."),
            "kind": STRICT_SAYS.get(one["code"], "a type error")
            if one.get("strict") else "a type error",
            "file": one["file"], "line": one["line"]})
    called = {one["to"] for one in held.calls()}
    ran = {(path, one["call"]) for path, read in held.outline().items()
           for one in read.get("runs", ())}
    for one in _scope(held, subject):
        end = f"{one['file']}#{one['name']}"
        if end not in called and (one["file"], one["name"]) not in ran \
                and not one["exported"]:
            unused.append({"where": _at(one), "in": one["name"],
                           "promise": "to be used", "found": "nothing calls "
                           "it, and it is not exported"})
        elif one["exported"] and one["params"]:
            open_.append({"where": _at(one), "in": one["name"],
                          "promise": (f"its doc: “{one['doc']}”"
                                      if one.get("doc") else
                                      f"its name: {one['name']}"),
                          "found": "nothing gives an example to check it by",
                          "question": _template(one)})
    name = "the project" if subject.kind == "project" else subject.name
    parts = [f"I compared what {name} promises -- its types, its names and "
             f"docs, being used -- with what the compiler and its calls "
             f"show."]
    if contradicted:
        parts.append(f"{len(contradicted)} contradicted: " + "; ".join(
            f"{one['where']} ({one['kind']}): {one['found']}"
            for one in contradicted[:4]) + ("; …" if len(contradicted) > 4
                                            else "") + ".")
    else:
        parts.append("Nothing contradicts them.")
    if unused:
        parts.append(f"{len(unused)} unused: " + ", ".join(
            one["in"] for one in unused[:6]) + ".")
    if open_:
        parts.append(f"{len(open_)} open: nothing checks what "
                     + ", ".join(one["in"] for one in open_[:5])
                     + " should give -- an example settles it (e.g. "
                     + open_[0]["question"] + ").")
    blocks, seen = [], set()
    for one in contradicted:
        where = held.at(one["file"], one["line"])
        if where and where["name"] not in seen and len(blocks) < 3:
            seen.add(where["name"])
            blocks.append(_block(held, {**where, "file": one["file"]}))
    suggest = [{"text": one["question"], "send": False} for one in open_[:3]]
    suggest += [{"text": ASKING[asp].format(
        n="the project" if subject.kind == "project" else subject.name),
        "send": True} for asp in CAN[subject.kind] if asp != "bugs"][:3]
    return _reply("bugs", text, " ".join(parts),
                  {"contradicted": contradicted, "unused": unused,
                   "open": open_}, blocks, subject.name, suggest)


ANSWERS = {"explain": _explain, "where": _where, "size": _size,
           "files": _files, "functions": _functions, "callers": _callers,
           "calls": _calls, "bugs": _gaps}


# -- the conversation's own code -----------------------------------------------------

def _code(held, space, subject: Subject, aspect: str, text: str) -> dict:
    """An aspect of the code the conversation wrote or was given: v697's
    answers, and its gaps found by running it."""
    from research.v697 import conversation as C
    if aspect == "bugs":
        return _code_gaps(space, text)
    if aspect == "size":
        code = (space.found.get("answer") or {}).get("code") or ""
        lines = len(code.splitlines())
        said = f"{subject.name}: {lines} lines."
        return C._reply(space, text, "size", said, {"lines": lines})
    return C._question(space, aspect, text)


def _code_gaps(space, text: str) -> dict:
    """The conversation's code against what it promises: the examples
    given (run), not throwing on inputs at the edges of its types (run),
    the other programs written for it (where they part, an open question),
    a second writer's agreement (confirmed or not)."""
    from research.v696 import meaning as M
    from research.v696.checker import checker
    from research.v697 import conversation as C
    found = space.found
    answer = found.get("answer") or {}
    code, entry = answer.get("code"), answer.get("entry")
    request = space.request or {}
    pairs = [(list(one["args"]), one["value"]) for one in
             request.get("examples") or () if "(" in one.get("said", "(")]
    contradicted, open_ = [], []
    if code:
        if pairs:
            rows = checker().run(code, entry, [args for args, _ in pairs])
            for (args, want), row in zip(pairs, rows):
                if row.get("value") != want or "error" in row:
                    contradicted.append({
                        "where": f"{entry}({', '.join(map(json.dumps, args))})",
                        "promise": f"your example: {json.dumps(want)}",
                        "found": row.get("error") or json.dumps(
                            row.get("value"))})
        edges = (M.edge_probes(pairs) + M.probes(pairs)) if pairs else \
            (found.get("open") or {}).get("inputs") or []
        # and inputs of its types, far from the examples too: what is
        # varied from `day(0)` is `day(1)`, and `day(7)` is the gap
        for args in M.wide_probes([kind for _, kind in
                                   request.get("params") or ()]):
            if args not in edges:
                edges.append(args)
        rows = checker().run(code, entry, edges) if edges else []
        returns = request.get("returns") or ""
        # a type that promises a value: nothing given back breaks it
        promised = returns and returns not in ("void", "undefined") and \
            "undefined" not in returns
        for args, row in zip(edges, rows):
            if "error" in row:
                contradicted.append({
                    "where": f"{entry}({', '.join(map(json.dumps, args))})",
                    "promise": "to give a result for any input of its types",
                    "found": f"throws: {row['error']}"})
            elif promised and row.get("value") is None:
                contradicted.append({
                    "where": f"{entry}({', '.join(map(json.dumps, args))})",
                    "promise": f"its type: a {returns}",
                    "found": "gives nothing (undefined)"})
        # and what the compiler, strict, finds of it
        try:
            said = checker().diagnose({"/answer.ts": code}, strict=True)
        except Exception:                           # noqa: BLE001
            said = []
        for one in said:
            line = code[:one.get("start", 0)].count("\n") + 1
            contradicted.append({
                "where": f"line {line}", "promise": "the compiler, strict",
                "found": one.get("message", "").rstrip(".")})
    others = C._others(space, text)["code"]["followup"]["looked"].get(
        "others") or []
    for one in others:
        if one.get("differs at") is not None:
            call = f"{entry}({', '.join(map(json.dumps, one['differs at']))})"
            open_.append({"where": call, "promise": "one answer",
                          "found": f"programs written for it part here: "
                                   f"{one['the answer gives']} or "
                                   f"{one['it gives']}",
                          "question": f"{call} == ?"})
    status = answer.get("status")
    if status not in ("confirmed", "pasted") and not open_:
        open_.append({"where": entry, "promise": "a second reading",
                      "found": "nothing written apart confirms it",
                      "question": (f"{entry}(…) == ?")})
    parts = [f"I ran {entry} against what it promises: "
             + (f"your {len(pairs)} example(s), " if pairs else "")
             + "inputs at the edges of its types, and the other programs "
               "written for it."]
    parts.append(f"{len(contradicted)} contradicted: " + "; ".join(
        f"{one['where']} -- {one['found']}" for one in contradicted[:4])
        + "." if contradicted else "Nothing contradicts it.")
    if open_:
        parts.append(f"{len(open_)} open: " + "; ".join(
            f"{one['where']} ({one['found']})" for one in open_[:3]) + ".")
    suggest = [{"text": one["question"], "send": False} for one in open_[:3]]
    return C._reply(space, text, "bugs", " ".join(parts),
                    {"contradicted": contradicted, "open": open_}) | {
        "suggest": suggest + [{"text": "what does it do", "send": True},
                              {"text": "how sure are you", "send": True}]}


# -- code pasted, with a question about it -------------------------------------------

#: pasted code with a question about it
FENCE = re.compile(r"```[\w-]*\n(.*?)```", re.S)
ABOUT_CODE = re.compile(r"\b(explain|what does|what is|describe|how does|"
                        r"read|bugs?|wrong)\b", re.I)


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
    answer["suggest"] = [{"text": "are there any bugs in it", "send": True},
                         {"text": f"{entry}(…)", "send": False}]
    return answer
