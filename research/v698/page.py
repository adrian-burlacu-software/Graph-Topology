"""Code talk in the conversation: v698's acts in v689's executive.

Where the shared reader was taught code talk (`reading.taught`), one act
reads every message about code with it (`reading.code_talk`) and does what
the encoder read: write what is asked for (`make`), add to what was asked
(`change`), run it (`run`), answer a question about the project, a file, a
function or the conversation's code (`ask`), keep a concept taught
(`teach`). v697's code act, which reads by hand rules, is withdrawn: no
word list decides what a message is (Adrian, 2026-10-03).

Where it was not, the hand rules stand in as before (`asking.route`): a
reader without code talk must not leave code unanswered.

Code pasted with a question about it is read and kept (`asking.pasted`):
code, not natural language, is what marks it.

All read the utterance as typed (`v697.page.said_as_typed`).
"""
from __future__ import annotations

from research.v687.executive import ANSWERED, Operator
from research.v689 import session as v689
from research.v697 import coding
from research.v697 import conversation
from research.v697 import page as v697_page
from research.v698 import asking, knowledge, reading
from research.v698 import project as Pj

PASTED, CODE_TALK = 212.0, 211.0
#: a question about the project's data (v701): before code talk, where it is
#: one and what it names is in the data
DATA_TALK = 211.5
#: data said in a message: before code pasted (a JSON object reads as a
#: TypeScript one)
PASTED_DATA = 212.5
#: what is asked of the computer (v702): a command given, a task a command
#: does, yes or no to one shown -- before code talk (`run ls` is no
#: function's call) and data talk
SHELL_TALK = 212.25
#: a command shown and waiting for yes, by conversation
WAITING: dict = {}


def _key(session):
    return getattr(session, "conversation", "") or id(session)


def known(key) -> set:
    """The names known to be code in a conversation: its project's
    functions (and a method by its own name) and files (and their base
    names), and the code written or pasted in it -- what the reader is told
    of each word (`reading._known`)."""
    names = set()
    held = Pj.project(key)
    if held is not None and held.files:
        for one in held.functions():
            names.add(one["name"])
            names.add(one["name"].split(".")[-1])
        for path in held.files:
            names.add(path)
            names.add(path.rsplit("/", 1)[-1])
    entry = ((conversation.workspace(key).found or {}).get("answer")
             or {}).get("entry")
    if entry:
        names.add(entry)
    return names


def _arguments(text: str, read) -> str | None:
    """The arguments of a call the encoder read: inside the brackets of the
    subject as written (`rev("ab")`), else the longest end of the message
    that reads as TypeScript arguments (`run it on [1, 2]`). Code, not
    words, is what is read here."""
    phrase = (read.spans.get("SUBJ") or [""])[0]
    if "(" in phrase:
        at = text.find(phrase)
        start = text.find("(", at if at >= 0 else 0)
        close = coding._balanced(text, start + 1, ")")
        if close > start:
            return text[start + 1:close]
    words = text.rstrip(" ?.!").split()
    for begin in range(len(words)):
        tail = " ".join(words[begin:])
        try:
            coding._literal(f"[{tail}]")
            return tail
        except ValueError:
            continue
    return None


def _act(read, text: str, session, memory) -> dict | None:
    key = _key(session)
    space = conversation.workspace(key)
    held = Pj.project(key)
    turn = v697_page._turn(memory)
    # the language code is asked in (v699): what the message asks (*in
    # Python*, read by the encoder), else the project's, else the
    # conversation's -- and the code's own, where it is written (`read`)
    from research.v698 import ways as W
    language = (W.asked_language(read.ways)
                or (held.language() if held is not None else None)
                or space.language)
    if read.act == "make":
        asked = coding.read(text, language)
        if asked.get("yours") and space.request and \
                asked.get("entry") == space.request.get("entry"):
            # the conversation's own function, written out by the person
            # (`what about function day(n) {...}`): more of what was asked
            # -- its examples kept -- not a new request
            return conversation.respond("more", asked, text, space, turn,
                                        ways=read.ways, language=language)
        return conversation.respond("request", asked, text,
                                    space, turn, ways=read.ways,
                                    language=language)
    if read.act == "change":
        # a change of the project's own code -- a function or file named,
        # or the one just talked about (v700): made in the project, and in
        # its files on disk
        from research.v700 import fixing
        subject = (fixing.placed(read, held, text)
                   or asking.resolve(read, held, space, turn, key)) \
            if held is not None else None
        if subject is not None and subject.kind in ("function", "file"):
            if fixing.available():
                made = fixing.change(text, held, subject, key)
                asking.FOCUS[key] = (subject, turn)
                found = asking._reply("change", text, made["said"],
                                      {"read": read.json()},
                                      name=subject.name)
                found["code"]["change"] = made
                found["code"]["answer"]["status"] = made["status"]
                return found
        asked = W.asked_language(read.ways)
        if asked and space.request and asked != conversation.language_of(
                space):
            # *now write it in Python*: the same request, in the language
            # asked -- its words and its examples said again as that
            # language writes them, not its old signature
            said = conversation.restated(space.request, asked)
            return conversation.respond(
                "request", coding.read(said, asked), said, space, turn,
                ways=read.ways, language=asked)
        return conversation.respond("more", coding.read(text, language),
                                    text, space, turn, ways=read.ways,
                                    language=language)
    if read.act == "teach":
        return knowledge.taught(read, text)
    if read.act == "run":
        subject = asking.resolve(read, held, space, turn, key)
        args = _arguments(text, read)
        if subject is not None and subject.kind == "code" and args is not None:
            return conversation.respond("call", args, text, space, turn)
        if subject is not None and subject.kind == "function":
            said = (f"{subject.name} is the project's: I read the project, "
                    f"I do not run it (it may do anything a program can). "
                    f"Paste it with a question, and I will run it here.")
            return asking._reply("run", text, said, {"read": read.json()},
                                 name=subject.name)
        return None
    return asking.answered_read(read, text, held, space, turn, key)


def _shell_turn(asked, text: str, key) -> dict:
    """A message to the shell, carried out: the command given, or the one
    written for the task; run where it only reads (or the person said yes,
    or `shell.ASK` is off), else shown and kept waiting for yes."""
    from research.v702 import shell
    looked: dict = {"act": asked.act, "chance": round(asked.chance, 3)}
    if asked.act == "no":
        waiting = WAITING.pop(key, None)
        said = f"I did not run `{waiting['command']}`." if waiting else \
            "Nothing was waiting to run."
        return _shell_reply(text, said, looked)
    if asked.act == "yes":
        waiting = WAITING.pop(key)
        return _shell_ran(text, waiting["command"], key, looked,
                          asked_yes=True)
    if asked.act == "command":
        command = asked.command
        looked["given"] = command
    else:
        held = Pj.project(key)
        made = shell.written(text, held.root if held is not None and
                             held.root else None)
        looked["written"] = made
        if made["status"] == "unwritten":
            return _shell_reply(text, (
                f"I could not write a command for that: of {made['of']} "
                f"written, none parses."), looked)
        command = made["command"]
        if made["status"] == "unsure":
            WAITING[key] = {"command": command}
            return _shell_reply(text, (
                f"I am not sure of the command: of {made['of']} written, no "
                f"two agree. The likeliest is `{command}` -- say yes to run "
                f"it."), looked, command)
    if not shell.parses(command):
        return _shell_reply(text, f"`{command}` does not parse as Bash.",
                            looked, command)
    reads, why = shell.reads_only(command)
    looked["reads only"] = reads
    if not reads and shell.ASK:
        WAITING[key] = {"command": command}
        return _shell_reply(text, (
            f"I would run `{command}`, but {why}, which may change "
            f"something -- say yes to run it."), looked, command)
    return _shell_ran(text, command, key, looked)


def _shell_ran(text: str, command: str, key, looked: dict,
               asked_yes: bool = False) -> dict:
    """The command run in the project's folder (else the server's), what it
    printed said, and what parses as data held as the conversation's."""
    import os
    from research.v701 import pasting
    from research.v702 import shell
    held = Pj.project(key)
    root = (held.root if held is not None and held.root else os.getcwd())
    result = shell.run(command, root)
    looked["ran"] = {one: result.get(one) for one in ("code", "seconds",
                                                      "cut")}
    shown = (result["out"] or "").rstrip()
    if result.get("error") and (result.get("code") or not shown):
        shown = (shown + "\n" + result["error"].rstrip()).strip()
    lines = shown.count("\n") + 1 if shown else 0
    said = (f"Ran `{command}`" + (" (as you said)" if asked_yes else "")
            + (f": exit {result['code']}" if result.get("code") else "")
            + (f", {result['error']}" if result.get("code") is None else "")
            + (f" -- {lines} lines:" if shown else ", nothing printed."))
    if shown:
        said += "\n```\n" + shown + ("\n..." if result.get("cut") else "") \
            + "\n```"
    # what it printed, where it reads as data, held: asked about next
    if shown and pasting.found(shown) is not None:
        found = pasting.hold(shown, key)
        if found is not None:
            looked["held"] = found["path"]
            said += f"\nI hold its output as {found['path']}."
    reply = _shell_reply(text, said, looked, command)
    reply["code"]["shell"] = result
    return reply


def _shell_reply(text: str, said: str, looked: dict,
                 command: str = "") -> dict:
    reply = asking._reply("shell", text, said, looked)
    reply["code"]["shell"] = {"command": command}
    return reply


def replies(session) -> list:
    def typed(memory) -> str:
        return v697_page._typed(memory)

    def pastes(memory) -> bool:
        return asking.pasted(typed(memory)) is not None

    def explain(memory):
        text = typed(memory)
        space = conversation.workspace(_key(session))
        memory["turn"].answer = asking.read_pasted(
            text, asking.pasted(text), space)
        space.turn = v697_page._turn(memory)
        return ANSWERED

    out = [Operator(name="pasted code", apply=explain, proposes=pastes,
                    utility=PASTED,
                    rule="code pasted with a question about it: read by "
                         "the compiler, run, kept as the conversation's")]

    # a question about the project's data (v701): read by the encoder's data
    # heads, its fields looked up in the data's schema, carried out over
    # what the data holds -- where it is one, and what it names is there
    from research.v701 import pasting, querying

    def says_data(memory) -> bool:
        return pasting.found(typed(memory)) is not None

    def hold_data(memory):
        text = typed(memory)
        key = _key(session)
        held = pasting.hold(text, key)
        if held is None:
            return None
        said = f"I hold {held['path']} now. {held['model'].said()}"
        found = None
        if held["rest"] and querying.available():
            found = querying.answer(held["rest"], Pj.project(key))
            if found is not None:
                said = f"{found.said} (I hold {held['path']} now.)"
        reply = asking._reply("data", text, said,
                              {"held": held["model"].json(),
                               "query": found.json() if found else None})
        reply["code"]["data"] = {"held": held["path"],
                                 "query": found.json() if found else None}
        memory["turn"].answer = reply
        return ANSWERED

    out.append(Operator(
        name="pasted data", apply=hold_data, proposes=says_data,
        utility=PASTED_DATA,
        rule="data said in a message (it parses as JSON, YAML or CSV): "
             "held as the conversation's, and what is asked of it answered "
             "from it"))

    from research.v702 import shell

    if shell.available():
        asked_for: dict = {}

        def shell_read(memory):
            text = typed(memory)
            if text not in asked_for:
                asked_for.clear()
                asked_for[text] = shell.read(text, known(_key(session)))
            return asked_for[text]

        def asks_shell(memory) -> bool:
            asked = shell_read(memory)
            if asked is None or asked.act == "none" or \
                    asked.chance < shell.FLOOR or pastes(memory):
                return False
            if asked.act in ("yes", "no"):
                return _key(session) in WAITING
            return asked.act == "task" and shell.writer_available() or \
                asked.act == "command" and bool(asked.command)

        def do_shell(memory):
            asked = shell_read(memory)
            text, key = typed(memory), _key(session)
            memory["turn"].answer = _shell_turn(asked, text, key)
            return ANSWERED

        out.append(Operator(
            name="shell", apply=do_shell, proposes=asks_shell,
            utility=SHELL_TALK,
            rule="what is asked of the computer, read by the encoder: a "
                 "command given, or written for a task; run in the "
                 "project's folder where it only reads, shown first where "
                 "it may change something"))

    if querying.available():
        found_for: dict = {}

        def data_found(memory):
            text = typed(memory)
            if text not in found_for:
                found_for.clear()
                found_for[text] = querying.answer(
                    text, Pj.project(_key(session)))
            return found_for[text]

        def asks_data(memory) -> bool:
            return not pastes(memory) and data_found(memory) is not None

        def answer_data(memory):
            found = data_found(memory)
            if found is None:
                return None
            reply = asking._reply("data", typed(memory), found.said,
                                  {"query": found.json()})
            reply["code"]["data"] = found.json()
            memory["turn"].answer = reply
            return ANSWERED

        out.append(Operator(
            name="data talk", apply=answer_data, proposes=asks_data,
            utility=DATA_TALK,
            rule="a question about the project's data, read by the encoder, "
                 "looked up in the data's schema, carried out over what it "
                 "holds"))

    if reading.taught():
        def talks(memory) -> bool:
            return not pastes(memory) and reading.code_talk(
                typed(memory), known(_key(session))) is not None

        def answer(memory):
            text = typed(memory)
            read = reading.code_talk(text, known(_key(session)))
            found = _act(read, text, session, memory) if read else None
            if found is None:
                return None
            found.setdefault("code", {})["read"] = read.json()
            memory["turn"].answer = found
            return ANSWERED

        out.append(Operator(
            name="code talk", apply=answer, proposes=talks,
            utility=CODE_TALK,
            rule="a message about code, read by the encoder: what it does "
                 "(make, change, run, ask, teach), what it asks, of what"))
        return out

    # a reader not taught code talk: the hand rules, as before
    def about(memory) -> bool:
        key = _key(session)
        return asking.route(typed(memory), Pj.project(key),
                            conversation.workspace(key),
                            v697_page._turn(memory), key)[0] is not None

    def answer_by_hand(memory):
        key = _key(session)
        found = asking.answered(typed(memory), Pj.project(key),
                                conversation.workspace(key),
                                v697_page._turn(memory), key)
        if found is None:
            return None
        memory["turn"].answer = found
        return ANSWERED

    out.append(Operator(name="about code", apply=answer_by_hand,
                        proposes=about, utility=CODE_TALK,
                        rule="a question about code, read by hand rules "
                             "(a reader not taught code talk)"))
    return out


v689.contributes(replies)
# what is taught, used in writing code: told to the writers, offered to
# the search (`knowledge.context`)
if knowledge.context not in coding.CONTEXT:
    coding.CONTEXT.append(knowledge.context)
if reading.taught():
    # v697's act reads code by hand rules: the encoder reads it now
    if v697_page.replies in v689.LAYERS:
        v689.LAYERS.remove(v697_page.replies)
