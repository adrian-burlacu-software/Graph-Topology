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

import json
import re

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
#: a question of the project's data nothing here answers: after every part
#: (code talk 211, the page's code 210), before the conversation's own
DATA_UNANSWERED = 209.5
#: how sure code talk must be a question is its own for a guess of the
#: data's to give way to it
SURE_CODE = 0.9
#: what code talk reads as asking something done, never a question of data
DOINGS = ("change", "make", "run", "teach", "goal")
#: data said in a message: before code pasted (a JSON object reads as a
#: TypeScript one)
PASTED_DATA = 212.5
#: what is asked of the computer (v702): a command given, a task a command
#: does, yes or no to one shown -- before code talk (`run ls` is no
#: function's call) and data talk
SHELL_TALK = 212.25
#: a command shown and waiting for yes, by conversation
WAITING: dict = {}
#: a yes or no to a goal's plan comes before one to a command
GOAL_STEP = 212.3
#: the call run last to be debugged, by conversation (v705)
DEBUGGED: dict = {}
#: what is said back to a plan held, before the shell and code talk
PLAN_TALK = 212.28
#: a goal said back with its plan, by conversation: a yes takes its first
#: step (v704)
GOALS: dict = {}
#: the one record a data answer was of, by conversation: (file, record,
#: turn) -- what a follow-up's `it`, `their` is
RECORDS: dict = {}
#: the records a data answer ended with, by conversation: (kept, turn) --
#: what a follow-up's `these`, `how many are there?` asks of
#: (`querying._kept`)
KEPT: dict = {}


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
        # its classes and constants (`Book`, `LOAN_DAYS`) -- not fields:
        # `title`, `pages` are words before they are code
        names.update(one.name for one in held.index().symbols.values()
                     if one.kind in ("class", "constant"))
    entry =((conversation.workspace(key).found or {}).get("answer")
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


def _shown(value):
    """What a goal holds, as the page shows it: a case's check said by what
    it checks (v705 cases carry their checks), never kept as a function."""
    return json.loads(json.dumps(value, default=lambda one: getattr(
        one, "about", None) or type(one).__name__))


#: how long a goal is thought about before it is answered (seconds)
THINKING = 240


def _goal(text: str, held, read, key: str = "", session=None) -> dict:
    """Something wanted, not something to do (v704): never acted on -- told
    `I want to improve the ability for it to communicate in general`, the
    architecture once changed a function called `plan` -- but said back as
    a goal, with what the project has that its words point to, and the
    first step asked for."""
    looked: dict = {"read": read.json()}
    # a goal about the assistant itself: what it does worst now, by the
    # conversation test (v704 `selfmodel`) -- what the steps are of
    from research.v704 import history, selfmodel, teach_saying
    model = selfmodel.held()
    # of itself: what its own ways of looking find (v705 `evidence`), each
    # place it fails at a subgoal carried out in turn (`doing.findings`)
    from research.v705 import evidence, inside
    itself = bool(model) and selfmodel.about_itself(text)
    # of itself or of the code it holds alike -- never while it is asking
    # itself (a scratch session's answer)
    found_by = bool(model) and (itself or held is not None) and \
        session is not None and not inside.SCRATCH.get()
    if found_by:
        weak = evidence.weaknesses(text, session, model)
        looked["ways"] = [what for _, what in evidence.ways(text)]
    else:
        weak = selfmodel.weaknesses(text, model) if itself else []
    if weak:
        looked["weaknesses"] = _shown(weak)
    # its steps, chosen from how this project reached goals like it (v704),
    # each said of what it is done to: the weakness it starts with, or what
    # in the project the goal's words point to
    near: list = []
    if held is not None and not weak:
        from research.v703 import planning
        near = planning.shortlist(text, held, most=3)
        looked["near"] = near
    planned = history.plan(text)
    if planned:
        planned = history.ground(planned, weak[0] if weak else None, near)
        looked["plan"] = planned
    # the reply in pieces, each said by the plan sayer where its saying
    # reads back to it, else as the architecture holds it
    pieces = teach_saying.pieces_of(weak, model, planned)
    sayings, by_sayer = teach_saying.say(pieces)
    looked["said by the sayer"] = f"{by_sayer} of {len(pieces)}"
    said = teach_saying.joined(pieces, sayings)
    if found_by:
        # thought about before it is answered (v705 `thinking`): tools
        # chosen from the pad, its own found ones too, each step on the
        # page; the reply says what the thinking came to
        from research.v705 import thinking
        pad = thinking.think(text, thinking.Held(session, held),
                             seconds=THINKING)
        looked["thinking"] = [f"[{one['kind']}] {one['said']}"
                              for one in pad.thoughts]
        came = next((one["said"] for one in reversed(pad.thoughts)
                     if one["kind"] in ("result", "suppose")), None)
        said += f" I thought it through in {len(pad.thoughts)} steps" + (
            f": {came}" if came else " and came to nothing yet.")
    if near and not planned:
        said += " What your words point to: " + ", ".join(near) + "."
    if key and (planned or (found_by and weak)) and not inside.SCRATCH.get():
        # kept for a yes: the first step taken (`_first_step`)
        GOALS[key] = {"goal": text, "weak": weak, "plan": planned}
        if found_by and weak:
            # carried place by place; looked at again when a round met any
            GOALS[key].update(subgoals=[], again=lambda: evidence.gather(
                text, session)["cases"])
    found = asking._reply("goal", text, said, looked)
    found["code"]["answer"]["status"] = "goal"
    return found


#: the cases a first step shows at most
FIRST_CASES = 6


def _next_step(key: str, held, change, debug=None) -> dict:
    """A held goal's next step, on a yes (v704 `doing`): of itself, each
    step carried out where it can be; of a project, where it would go
    first (`_first_step`), then the same."""
    from research.v704 import doing
    goal = GOALS[key]
    if not goal["weak"] and goal.get("at") is None:
        found = _first_step(key, held)
        goal["at"] = 0
        lines = []
        steps = (goal.get("plan") or {}).get("steps") or []
        if steps:
            lines.append(f"Next: 1. {steps[0]['said']}. Shall I go on?")
        found["text"] = found["spoken"] = "\n".join(
            [found["text"]] + lines)
        return found
    lines = doing.carry(goal, change=change, held=held, debug=debug)
    # a plan carried out is kept, done, for what is said back to it
    # (`plan talk`): a yes is no longer its
    found = asking._reply("goal step", goal["goal"], "\n".join(lines),
                          _shown({"goal": goal["goal"], "at": goal.get("at"),
                                  "found": goal.get("found"),
                                  "finding": goal.get("finding")}))
    found["code"]["answer"]["status"] = "goal step"
    return found


def _first_step(key: str, held) -> dict:
    """A goal's first step, on a yes: a look, never a change -- of itself,
    what fails in its worst weakness now (the conversation test's cases,
    each with what answered it); of a project, where in it the goal would
    go (v703's plan, nothing changed)."""
    goal = GOALS[key]
    lines = []
    looked: dict = {"goal": goal["goal"]}
    if goal["weak"]:
        weak = goal["weak"][0]
        cases = weak.get("cases") or weak["examples"]
        looked["cases"] = cases
        lines.append(f"First, a look at {weak['what']} -- what fails now:")
        for at, one in enumerate(cases[:FIRST_CASES], 1):
            answer = " ".join((one.get("answer") or "").split())[:100]
            lines.append(f"{at}. \"{one['said']}\" went to "
                         f"{one['part']}" + (f", which said: {answer}"
                                             if answer else ""))
        if len(cases) > FIRST_CASES:
            lines.append(f"(and {len(cases) - FIRST_CASES} more)")
        lines.append("Which of these should I look into first?")
    elif held is not None:
        from research.v703 import planning
        found = planning.plan(goal["goal"], held)
        looked["plan"] = found
        steps = found.get("steps") or []
        if steps:
            lines.append("First, where it would go in the project -- "
                         "nothing changed:")
            for at, one in enumerate(steps[:FIRST_CASES], 1):
                lines.append(f"{at}. {one['function'] or 'the file'} in "
                             f"{one['path']}")
            lines.append("Which should change first, and how?")
        else:
            lines.append("I couldn't find where in the project it would go. "
                         "Which file or function is it about?")
    else:
        lines.append("For that I need the project: send it, and I'll look "
                     "at where it would go.")
    found = asking._reply("goal step", goal["goal"], "\n".join(lines),
                          looked)
    found["code"]["answer"]["status"] = "goal step"
    return found


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
    if read.act == "goal":
        return _goal(text, held, read, key, session)
    if read.act == "make" and held is not None and held.files:
        from research.v700 import fixing
        if fixing.added(text, held) and any(
                one.kind == "class" for word in re.findall(
                    r"[A-Za-z_]\w*", text)
                for one in held.index().find(word)):
            # something new asked of a class the project has (`create a
            # method in Graph with the signature fibonacci(x: Node)`): a
            # change of the project, not code apart from it
            import dataclasses
            return _act(dataclasses.replace(read, act="change"), text,
                        session, memory)
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
        # a change of the project's data naming no file -- a value it holds,
        # a record described, a setting the code has no name of -- is of
        # that data file (v701 `placing.data_file`): `set the port to 9090`
        # was planned across the code, or placed nowhere
        data, asked = None, text
        if held is not None and not _named_outright(read, text, held):
            from research.v701 import placing
            data = placing.data_file(held, text)
            # -- or the data just answered of, where it says none of its own
            # (`change it to 9000`): its place said with it, what `it` is
            recent = asking._focus(key, space, turn) if data is None \
                else None
            if recent is not None and recent.kind == "file" and \
                    Pj.is_data(recent.file or "") and \
                    recent.file in held.files and \
                    placing.block(held.files[recent.file].splitlines(),
                                  text) is None:
                data = recent.file
                file_, place = asking.PLACES.get(key, (None, ""))
                words = " ".join(part for part in (place or "").replace(
                    "[]", " ").split(".") if part.strip())
                if file_ == data and words:
                    asked = f"{text} ({words})"
        if data is not None and fixing.available():
            subject = asking.Subject("file", data, data)
            made = fixing.change(asked, held, subject, key, spans=read.spans)
            asking.FOCUS[key] = (subject, turn)
            found = asking._reply("change", text, made["said"],
                                  {"read": read.json()}, name=data)
            found["code"]["change"] = made
            found["code"]["answer"]["status"] = made["status"]
            return found
        # a change naming no function or file of the project outright --
        # its words name what it is about, not where (`report whether the
        # shell asks in /api/health and show it in the MCP health tool`) --
        # is planned across the project (v703), where a plan is agreed
        # (across files: a project of one file of code has no plan to make
        # -- 11 of CanItEdit's changes were planned in one file, and put back)
        if held is not None and not _named_outright(read, text, held) and \
                sum(path.endswith(".py") or path.endswith(".ts")
                    for path in held.files) > 1:
            from research.v703 import planning
            if planning.available():
                planned = planning.plan(text, held)
                # across files: two steps or more, or one whose answer
                # another file passes through as it came (the MCP's
                # _health shows what do_GET adds to /api/health)
                if planned["status"] == "planned" and (
                        len(planned["steps"]) >= 2 or
                        planned.get("passed through")):
                    made = planning.carry(planned, held, key)
                    found = asking._reply("plan", text, made["said"],
                                          {"read": read.json(),
                                           "plan": planned})
                    found["code"]["plan"] = {**planned, **made}
                    found["code"]["answer"]["status"] = made["status"]
                    return found
        subject = (fixing.placed(read, held, text)
                   or asking.resolve(read, held, space, turn, key)) \
            if held is not None else None
        if subject is not None and subject.kind == "symbol":
            # a constant or a class named (`MAX_LOANS should be 5`): the
            # change is of its file
            subject = asking.Subject("file", subject.file, subject.file)
        if held is not None and (subject is None or subject.kind not in (
                "function", "file")) and any(
                path.endswith(".py") for path in held.files):
            # named nowhere (`let members hold up to 5 books at once`):
            # what the project has nearest what is said, by meaning -- a
            # function, or the file of a constant or class
            near = asking._described(held, text, "where")
            if near is not None:
                subject = near if near.kind == "function" else \
                    asking.Subject("file", near.file, near.file)
        if held is not None and (fixing.added(text, held) or
                                 subject is None or
                                 fixing.several(text, held)):
            # a change adding what the project has not (`add a method
            # next_n_fibs`): of the file that will hold it, not of the
            # function nearest -- 26 of CanItEdit's 91 changed a method
            # beside it; and a change placed nowhere, of a project with one
            # file of code, is of that file
            subject = fixing.adding(text, held, subject)
        if subject is not None and subject.kind in ("function", "file"):
            if fixing.available():
                # what was asked, made a check that runs: tests of the
                # request the project as it is fails, and its own tests
                # (`requested.py`) -- a change is written where it holds
                context = None
                path_ = subject.file or ""
                if path_.endswith(".py") and not Pj.is_data(path_):
                    from research.v698 import requested
                    # what `it` is, said to the code model where the
                    # request does not name it (`what does parse_isbn
                    # do?` ... `make it remove dots as well`: no test of
                    # parse_isbn written)
                    stated = text if subject.kind != "function" or re.search(
                        rf"(?<!\w){re.escape(subject.name.split('.')[-1])}"
                        rf"(?!\w)", text) else \
                        f"{subject.name.split('.')[-1]}: {text}"
                    check = requested.checker(
                        stated, held, path_, keeps=read.aspect == "same")
                    # a name made another everywhere it is used, where
                    # the request's tests hold for it (`rename
                    # parse_isbn to clean_isbn everywhere`: two lines of
                    # one file, 7 of 7 editors agreeing)
                    found_renames = requested.renames(text, held)
                    if not check.get("kept"):
                        # no test of the request to tell which name is
                        # meant: the one name written as code writes names
                        # (`clean_isbn`), or none -- "parse_isbn to to" was
                        # made, `to` passing the project's own tests
                        shaped = [one for one in found_renames
                                  if requested.code_shaped(one[1])]
                        found_renames = shaped if len(
                            {one[1] for one in shaped}) == 1 else []
                    for old, new, changed in found_renames:
                        # made, not guessed: its own tests kept and,
                        # where tests of what was asked are kept, as many
                        # of them as any change must pass -- not one alone
                        # (`kl_div` renamed `num_samples` passed one test
                        # of a new way of working it out)
                        if check.get("verify files") is None or \
                                check["verify files"](
                                    changed, least=None if check.get("kept")
                                    else 0):
                            continue
                        diffs = []
                        for path__, after in changed.items():
                            before = held.files[path__]
                            held.write(path__, after)
                            fixing.MADE.setdefault(key, []).append(
                                (path__, before, after))
                            diffs.append(fixing.diff(before, after,
                                                     path__))
                        said = (f"Renamed {old} to {new} in "
                                f"{', '.join(changed)}: every use of "
                                f"it, the project's own tests still "
                                f"passing.")
                        found = asking._reply("change", text, said,
                                              {"read": read.json()},
                                              name=new)
                        found["code"]["change"] = {
                            "status": "changed", "renamed": [old, new],
                            "files": sorted(changed),
                            "diff": "\n".join(diffs),
                            "requested": {"kept": check.get("kept", []),
                                          "need": check.get("need", 0)}}
                        found["code"]["answer"]["status"] = "changed"
                        return found
                    if check.get("score") is None:
                        # nothing runs that tells the project as it is from
                        # the change: answers agreeing are no evidence it
                        # does what was asked (`overdue should use today's
                        # date`, 3 of 13 agreeing on a wrong change) -- the
                        # likeliest shown, not written
                        context = {"verify": lambda changed: (
                            "nothing I could run tells the project as it "
                            "is from the change asked")}
                    if check.get("score") is not None:
                        # three rounds of the editors: each candidate is
                        # run, ten rounds passed a turn's 300 s
                        # the change written whole by the code model too,
                        # beside the editors' lines: the tests choose
                        try:
                            more = requested.rewritten(
                                stated, held, path_, subject.name
                                if subject.kind == "function" else None)
                        except Exception:       # noqa: BLE001
                            more = []
                        # (one round beside the code model's: three rounds
                        # took 104 s of #38's 255, near the test's 300)
                        context = {"score": check["score"],
                                   "rounds": 1 if more else 3,
                                   "tests": len(check["kept"]),
                                   "stated": check.get("stated", 0),
                                   "keeps": check.get("keeps", False),
                                   "original": check["original"],
                                   "more": more,
                                   "requested": {
                                       "kept": check["kept"],
                                       "need": check["need"]}}
                try:
                    made = fixing.change(text, held, subject, key,
                                         spans=read.spans, context=context)
                except Exception as bad:            # noqa: BLE001
                    # the card run out writing the change (CUDA out of
                    # memory: the turn was lost, the server said 500) -- no
                    # change, said so, the card emptied for the next
                    if "OutOfMemoryError" not in type(bad).__name__:
                        raise
                    import torch
                    torch.cuda.empty_cache()
                    made = {"status": "unchanged", "subject": subject.name,
                            "said": (f"I did not change {subject.name}: the "
                                     f"card ran out of memory writing the "
                                     f"change -- nothing was written.")}
                if context is not None and "requested" in context:
                    made["requested"] = context["requested"]
                asking.FOCUS[key] = (subject, turn)
                found = asking._reply("change", text, made["said"],
                                      {"read": read.json()},
                                      name=subject.name)
                found["code"]["change"] = made
                found["code"]["answer"]["status"] = made["status"]
                return found
        if held is not None and not space.request:
            # a change of the project, placed nowhere in it, with no code
            # of the conversation's to change: said, not written as new
            # code (`bump the server version to 1.3.0` became a function)
            found = asking._reply(
                "change", text, "I couldn't find what in the project to "
                "change for that -- which function or file is it in?",
                {"read": read.json()})
            found["code"]["answer"]["status"] = "unplaced"
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
    if read.act == "debug":
        # a run asked to be looked into (v705): the call run in its
        # project, said, and fixed only where the call then gives what it
        # should
        call = " ".join((read.spans.get("CALL") or [""])[0].split())
        got = (read.spans.get("GOT") or [None])[0]
        wanted = (read.spans.get("WANT") or [None])[0]
        if call:
            call = _as_written(text, call)
        from research.v705 import talk
        if key in DEBUGGED and (not call or held is None or not held.find(
                talk.called(call) or "")):
            # the call run last, answered (`it should give date(2026, 1,
            # 22)`): what was marked a call names nothing of the project's
            # -- it is what the last call should give
            if call and not wanted:
                wanted = call
            call = DEBUGGED[key]
        if held is None or not call:
            said = ("Send me the project and name the call -- like "
                    "`f(3) gives 7 but should give 9` -- and I'll run it "
                    "and look." if held is None else
                    "Which call should I run? Say it as you'd write it, "
                    "like `f(3)`.")
            found = asking._reply("debug", text, said, {"read": read.json()})
            found["code"]["answer"]["status"] = "unplaced"
            return found
        DEBUGGED[key] = call
        # each part found after the one before it: the call, what it gave,
        # what it should give
        at = text.find(call) + len(call) if call in text else 0
        if got:
            got = _as_written(text, got, at)
            at = text.find(got, at) + len(got) if got in text[at:] else at
        if wanted:
            wanted = _as_written(text, wanted, at)
        made = talk.debug(held, call,
                          got, wanted, key)
        found = asking._reply("debug", text, made["said"],
                              {"read": read.json(), "file": made.get("file"),
                               "function": made.get("function")},
                              name=made.get("function") or "")
        found["code"]["answer"]["status"] = made["status"]
        if made.get("function"):
            asking.FOCUS[key] = (asking.Subject(
                "function", made["function"], made["file"]), turn)
        return found
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


def _as_written(text: str, phrase: str, after: int = 0) -> str:
    """A phrase the reader marked, as the message wrote it: its words found
    in order in the message, from `after` on (the reader's words are
    lowered and split at marks -- `due_date ( date ( 2026 , 1 , 1 ) )`;
    what a call should give comes after the call, which may hold the same
    words), else the phrase."""
    words = phrase.split()
    if not words:
        return phrase
    pattern = r"\s*".join(re.escape(one) for one in words)
    found = re.compile(pattern, re.I).search(text, after)
    if not found:
        return phrase
    # code's own extent: an opened bracket or quote runs on to its close
    # (`due_date(date(2026,` was marked of `due_date(date(2026, 1, 1))`)
    end, depth, quote = found.end(), 0, None
    for at, mark in enumerate(text[found.start():found.end()]):
        if quote:
            quote = None if mark == quote else quote
        elif mark in "\"'":
            quote = mark
        elif mark in "([{":
            depth += 1
        elif mark in ")]}":
            depth -= 1
    at = end
    while (depth > 0 or quote) and at < len(text):
        mark = text[at]
        if quote:
            quote = None if mark == quote else quote
        elif mark in "\"'":
            quote = mark
        elif mark in "([{":
            depth += 1
        elif mark in ")]}":
            depth -= 1
        at += 1
    return text[found.start():at].strip(" .?!")


def _named_outright(read, text: str, held) -> bool:
    """Whether a change names what it changes: a subject the encoder marked,
    or a word shaped as code (`_turn`, `server.py`, `Open.the`), that the
    project has -- where it does, it is one change of that; where it does
    not, a plan says where."""
    names = set()
    for one in held.functions():
        names.update({one["name"], one["name"].split(".")[-1]})
    names.update(held.files)
    names.update(path.rsplit("/", 1)[-1] for path in held.files)
    marked = [phrase.split("(")[0].strip(" ?.,!'\"`")
              for phrase in read.spans.get("SUBJ", ())]
    shaped = [one.strip("?.,!'\"`") for one in text.split()
              if re.search(r"[_.()]|[a-z][A-Z]", one.strip("?.,!'\"`"))]
    return any(one in names or one.split("(")[0].split(".")[-1] in names
               for one in marked + shaped if one)


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
    if shown and shell.tabular(shown) and pasting.found(shown) is not None:
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

    # what the parts below can do, said to `what can you do?`
    from research.v689 import social
    social.can("write code, change your project's code, explain or debug "
               "it, and answer questions about its functions, classes and "
               "settings")
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
        social.can("run a command on this computer, or write one for a task")
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
            if asked.act == "command":
                # a command is one this computer can run (`what is 7
                # squared?` runs `what`, which it has not)
                return bool(asked.command) and \
                    shell.runs_programs(asked.command)
            if asked.act == "task" and (_of_the_code(typed(memory)) or
                                        _claimed(memory)):
                return False
            return asked.act == "task" and shell.writer_available()

        def _claimed(memory) -> bool:
            # a task a command could do, that a part made for it reads as
            # its own, is that part's: the shell is what is left (`make
            # loans last 21 days instead of 14` is a change of the
            # project's code; `what is 7 squared?` mathematics; `what is
            # the newest book?` the project's data)
            text, key = typed(memory), _key(session)
            from research.v692 import page as maths
            if maths.asked_of_mathematics(session, text):
                return True
            held = Pj.project(key)
            if held is None:
                return False
            read = reading.code_talk(text, known(key))
            if read is not None and read.act in ("change", "make", "goal"):
                return True
            return querying.available() and \
                querying.answer(text, held) is not None

        def _of_the_code(text: str) -> bool:
            # a task that code talk reads as of something the project has
            # -- its subject a function or file found there -- is code
            # talk's (`who calls shell.run`: `who` is a command too)
            read = reading.code_talk(text, known(_key(session)))
            if read is None or read.act in ("none", "teach"):
                return False
            names = known(_key(session))
            if read.aspect == "where" and read.subject == "file":
                # a file shown, as the shell would do it (`show me the first
                # line of books.csv`: code talk shows a file whole, the
                # shell the lines asked)
                return False
            # its subject as marked, else the message's words shaped as
            # code (`shell.run`, `_turn`, `main()`) -- looked up
            shaped = [one.strip("?.,!'\"`") for one in text.split()
                      if re.search(r"[._(]", one.strip("?.,!'\"`"))]
            if any(one in names
                   for phrase in list(read.spans.get("SUBJ", ())) + shaped
                   for word in [phrase] + phrase.split()
                   for one in (word, word.split("(")[0],
                               word.split("(")[0].rsplit(".", 1)[-1])):
                return True
            # a change asked of something the project's code says -- a word
            # shaped as code found in its files' text, not a path of it
            # (`report whether the shell asks in /api/health ...`: the
            # server answers /api/health) -- is a change of the project
            held = Pj.project(_key(session))
            if read.act != "change" or held is None:
                return False
            routes = [one.strip("?.,!'\"`") for one in text.split()
                      if re.search(r"[_/()]|[a-z][A-Z]", one.strip(
                          "?.,!'\"`")) and len(one) > 3]
            return any(one not in held.files and not any(
                path.startswith(one.strip("/")) for path in held.files)
                and any(one in body for body in held.files.values())
                for one in routes)

        def do_shell(memory):
            asked = shell_read(memory)
            text, key = typed(memory), _key(session)
            memory["turn"].answer = _shell_turn(asked, text, key)
            return ANSWERED

        def asks_goal_step(memory) -> bool:
            # yes or no to a goal said back with its plan (v704) -- read as
            # the shell's yes and no to a command shown are
            if _key(session) not in GOALS or \
                    GOALS[_key(session)].get("done"):
                return False
            asked = shell_read(memory)
            return asked is not None and asked.act in ("yes", "no") and \
                asked.chance >= shell.FLOOR

        def goal_step(memory):
            key = _key(session)
            if shell_read(memory).act == "no":
                GOALS.pop(key, None)
                reply = asking._reply("goal step", typed(memory),
                                      "All right -- I've left it.", {})
            else:
                reply = _next_step(key, Pj.project(key),
                                   lambda text, function=None, file=None,
                                   case=None: _by_myself(
                                       text, memory, function, file, case),
                                   _inside)
            memory["turn"].answer = reply
            return ANSWERED

        def _by_myself(text: str, memory, function: str | None = None,
                       file: str | None = None,
                       case: str | None = None) -> tuple:
            # a change a plan's step asks for: of a function the look
            # found, made there -- what it is a change of is known, not
            # read back from its own words (`in build, build "what is my
            # dog called?" as ask_name` read as none: `reader`, `reads`
            # are this project's names); of a project, read as anyone's
            key = _key(session)
            held = Pj.project(key)
            if function and file:
                from research.v700 import fixing
                found = [one for one in (held.find(function) if held else [])
                         if one["file"] == file]
                if not found or not fixing.available():
                    return f"{function} is not in the project I hold.", False
                subject = asking.Subject("function", found[0]["name"], file)
                # the case it fixes, kept out of the code (`fixing`) -- and
                # where the function is its own, said again with each
                # change put in, beside what it answered right (v706)
                context = {"case": case} if case else None
                whole = ((GOALS.get(key) or {}).get("finding") or {}).get(
                    "case") or {}
                if case and whole.get("said") == case:
                    from research.v705 import inside
                    own = inside.function_at(file, found[0]["name"])
                    checks = inside.cases_of(whole, own) if own else []
                    if checks:
                        context["verify"] = lambda after: inside.verify(
                            session, checks, own, after)
                    else:
                        # a change of its own code for a case it cannot say
                        # again is a guess at its own code: not made (four
                        # of seven wrote `if at == name` into reader.first
                        # for "how many times does this loop run?", which
                        # was not found among the test's cases -- the unit
                        # tests failed, and it was put back)
                        return (f"I can't run \"{case}\" again to check a "
                                f"change of {found[0]['name']} against it "
                                f"(it is not one of my test's cases as I "
                                f"have it, or {found[0]['name']} is not a "
                                f"function I can try a change of), so I left "
                                f"{found[0]['name']} as it is."), False
                    # seen inside: asked of the editor taught to fix from
                    # a run, as it was taught (v705 `editor-debug`)
                    look = inside.looked_at((GOALS.get(key) or {}).get(
                        "finding", {}).get("inside"), found[0]["name"])
                    if own and look:
                        from research.v705 import talk as debug_talk
                        from research.v705.teach_debugging import SAYING
                        # editor-debug2 was taught missing cases too (v706:
                        # a give-up is mostly a case no branch handles);
                        # people's bugs it fixes a little less precisely
                        # (81.8/85.2% against 84.9/87.1%), so it is this
                        # path's alone
                        editor = next((one for one in (
                            debug_talk.EDITOR.parent / "editor-debug2",
                            debug_talk.EDITOR) if one.exists()), None)
                        if editor is not None:
                            context.update(
                                said=inside.statement_of(look, own),
                                imports=inside.imports_of(own),
                                editor=editor.name, saying=SAYING)
                made = fixing.change(text, held, subject, key, context)
                if made["status"] not in ("changed", "partly") and \
                        context and context.get("verify") and \
                        context.get("editor"):
                    # what the editor did not write, built: a call of what
                    # in reach gives what it should, filled from what the
                    # function holds, put in where it gave up -- each run
                    # again (v706 `inside.compositions`)
                    look = inside.looked_at((GOALS.get(key) or {}).get(
                        "finding", {}).get("inside"), found[0]["name"])
                    # -- and the code's own way past such a nothing
                    # elsewhere (`_whose(...) or mention(...)`)
                    # -- and repair's own edits of it, the likeliest first
                    # -- and, last, the function written whole again by
                    # the code model (`inside.written`): what one line or
                    # one edit cannot be
                    def wholes():
                        return inside.written(
                            own, held.files[file], file, case,
                            answer=whole.get("answer") or "")
                    built = fixing.first_verified(
                        held, file, inside.precedents(
                            own, look, held.files[file]) +
                        inside.compositions(own, look, held.files[file]) +
                        inside.repaired(own, held.files[file], text),
                        context["verify"], key, found[0]["name"],
                        more=wholes)
                    if built["said"]:
                        made = {**made, "status": built["status"],
                                "said": made["said"] + " " + built["said"]}
                asking.FOCUS[key] = (subject, v697_page._turn(memory))
                return made["said"], made["status"] == "changed"
            read = reading.read(text, known(key)) \
                if reading.taught() else None
            if read is None or read.act != "change" or \
                    read.chance < reading.FLOOR:
                return (f"I read my own request as "
                        f"{read.act if read else 'nothing'}"
                        + (f" ({int(read.chance * 100)}%)" if read else "")
                        + ", not a change, so nothing was done."), False
            found = _act(read, text, session, memory)
            status = ((found or {}).get("code") or {}).get(
                "answer", {}).get("status")
            return (found["text"] if found else "Nothing came of it.",
                    status in ("changed", "planned"))

        def _inside(turns, file, function, act=None):
            # one of its own functions watched while a failing case is
            # said again, to a scratch session like this one (v705)
            from research.v705 import inside
            return inside.look_inside(session, turns, file, function, act)

        def asks_plan_talk(memory) -> bool:
            # what is said back to a plan held: an option picked or asked
            # about, or what was just done answered (v704)
            key = _key(session)
            if key not in GOALS or pastes(memory) or not reading.taught():
                return False
            read = _plan_read(memory)
            return read is not None and read.act in ("pick", "query",
                                                     "feedback")

        def _plan_read(memory):
            # read with the conversation's state: a plan is held
            # (`teach_plan_talk.MARKER`) -- `start with the routing
            # problem` is a pick here, a change anywhere else
            # -- its words naming the plan's options, not the project's
            # code: read without the project's names marked (with them,
            # `start by looking at what fails in remembering` marked
            # `looking`, one of this project's 6,000 names)
            from research.v704.teach_plan_talk import MARKER
            return reading.code_talk(f"{MARKER} {typed(memory)}")

        def plan_talk(memory):
            from research.v700 import fixing
            from research.v704 import doing
            key = _key(session)
            held = Pj.project(key)
            read = _plan_read(memory)
            phrase = " ".join((read.spans.get("OPTION") or [""])[0].split())
            goal = GOALS[key]
            goal.pop("done", None)
            lines = doing.talk(
                goal, read.act, phrase,
                change=lambda text, function=None, file=None, case=None:
                _by_myself(text, memory, function, file, case),
                held=held,
                undo=lambda: (fixing.undo(held, key) or {}).get("said"),
                debug=_inside)
            reply = asking._reply("plan talk", typed(memory),
                                  "\n".join(lines),
                                  {"act": read.act, "option": phrase,
                                   "read": read.json()})
            reply["code"]["answer"]["status"] = "plan talk"
            memory["turn"].answer = reply
            return ANSWERED

        out.append(Operator(
            name="plan talk", apply=plan_talk, proposes=asks_plan_talk,
            utility=PLAN_TALK,
            rule="what is said back to a plan held, read by the encoder: an "
                 "option picked is done, one asked about is said, what was "
                 "just done answered -- a change put back where found "
                 "wanting"))
        out.append(Operator(
            name="goal step", apply=goal_step, proposes=asks_goal_step,
            utility=GOAL_STEP,
            rule="yes or no to a goal's plan: on yes its first step, a look "
                 "-- what fails now, or where in the project it would go -- "
                 "never a change"))
        out.append(Operator(
            name="shell", apply=do_shell, proposes=asks_shell,
            utility=SHELL_TALK,
            rule="what is asked of the computer, read by the encoder: a "
                 "command given, or written for a task; run in the "
                 "project's folder where it only reads, shown first where "
                 "it may change something"))

    if querying.available():
        social.can("answer questions about your project's data files, or "
                   "change them")
        found_for: dict = {}

        def data_found(memory):
            text = typed(memory)
            if text not in found_for:
                found_for.clear()
                key = _key(session)
                held = Pj.project(key)
                kept, after = KEPT.get(key, (None, -1))
                found = querying.answer(
                    text, held, kept if after == v697_page._turn(memory) - 1
                    else None)
                # a question naming nothing of its own, just after the data
                # answered of one record (`which book is the oldest?` ...
                # `who wrote it?`): asked of that record
                file_, record, at = RECORDS.get(key, (None, None, -1))
                if (found is None or found.looked.get("nothing") or
                        found.looked.get("read as") == "no question") and \
                        record is not None and \
                        at == v697_page._turn(memory) - 1 and \
                        held is not None:
                    of = querying.of_record(text, held, file_, record)
                    found = of if of is not None else found
                found_for[text] = found
            return found_for[text]

        def asks_data(memory) -> bool:
            found = None if pastes(memory) else data_found(memory)
            if found is None:
                return False
            if found.looked.get("of the record"):
                return True
            # a name shaped as code (`ORDER_LIMIT`), said exactly as the
            # project's code defines it, is the code's question where the
            # data's answer is a place spelled otherwise (`where is
            # ORDER_LIMIT set?` answered `nursery.order_limit in
            # settings.yaml`)
            held_ = Pj.project(_key(session))
            if held_ is not None and any(
                    re.fullmatch(r"[A-Z][A-Z0-9]*(_[A-Z0-9]+)+|[a-z]\w*_\w+|"
                                 r"[a-z]+[A-Z]\w*", word) and
                    held_.index().find(word) and
                    word not in str(found.place or "")
                    for word in (one.strip("?.,!'\"`") for one in
                                 typed(memory).split())):
                return False
            # an answer whole of the records the question names (`which book
            # is the oldest?`: Solaris, the earliest year) is no guess,
            # however surely code talk reads the word `oldest`, a function's
            # name, as its own -- where the data reader read a question of
            # data at all (`what does oldest return when there are no
            # books?` it reads as none: the code's)
            whole = found.value is not None and found.looked.get("target") \
                and not found.looked.get("ask which") and \
                not found.looked.get("nothing") and \
                found.looked.get("read act") != "none"
            if not whole and (found.looked.get("ask which") or
                              found.looked.get("read as") == "no question"):
                # a guess of the data (made by lookup, or asking which field
                # is meant) is the code's question where code talk reads it
                # surely as its own (`what does oldest return when there
                # are no books?`: "several fields of numbers -- which do you
                # mean?")
                read = reading.code_talk(typed(memory), known(_key(session))) \
                    if reading.taught() else None
                # -- of something it names: one it only finds by meaning
                # is a guess too (`which book is the longest?` found the
                # field Loan.book)
                if read is not None and (read.act in DOINGS or (
                        read.act == "ask" and read.chance >= SURE_CODE and
                        read.subject != "described")):
                    return False
            if found.looked.get("read as") != "no question" and \
                    not found.looked.get("nothing"):
                return True
            if found.looked.get("nothing"):
                # nothing of the data so called (`which routes does handle
                # answer`: "I find nothing called routes"): code talk's,
                # where the reader reads it as code talk
                read = reading.code_talk(typed(memory), known(_key(session))) \
                    if reading.taught() else None
                return read is None
            # read as no question of data, and answered from it all the
            # same: a doing is no question, whatever it names (`set Dune's
            # pages to 896` names a record by a value, and asks to change
            # it)
            read = reading.code_talk(typed(memory), known(_key(session))) \
                if reading.taught() else None
            return read is None or read.act not in DOINGS

        def answer_data(memory):
            found = data_found(memory)
            if found is None:
                return None
            if found.looked.get("beyond"):
                # a question of the data asking what is not looked up yet:
                # said so, unsure
                path = of_the_data(memory) or "the project's data"
                reply = asking._reply("data", typed(memory), (
                    f"I can't answer that from {path} yet: it asks to "
                    f"group, sort or combine the records in a way I don't "
                    f"look up."), {"query": found.json()})
                reply["code"]["answer"]["status"] = "unsure"
                memory["turn"].answer = reply
                return ANSWERED
            reply = asking._reply("data", typed(memory), found.said,
                                  {"query": found.json()})
            reply["code"]["data"] = found.json()
            memory["turn"].answer = reply
            # what was answered of is what is talked about: `change it to
            # 9000` after `server.port in config.yaml is 8080`
            if found.file:
                key = _key(session)
                asking.FOCUS[key] = (asking.Subject("file", found.file,
                                                    found.file),
                                     v697_page._turn(memory))
                asking.PLACES[key] = (found.file, found.place)
                record = found.looked.get("record")
                RECORDS[key] = (found.file, record, v697_page._turn(memory)) \
                    if isinstance(record, dict) else (None, None, -1)
                KEPT[key] = (found.looked.get("kept"),
                             v697_page._turn(memory)) \
                    if found.looked.get("kept") else (None, -1)
            return ANSWERED

        out.append(Operator(
            name="data talk", apply=answer_data, proposes=asks_data,
            utility=DATA_TALK,
            rule="a question about the project's data, read by the encoder, "
                 "looked up in the data's schema, carried out over what it "
                 "holds"))

        def of_the_data(memory) -> str | None:
            """The data file a question names a field or a value of, where
            no part answers it -- or None."""
            text, held = typed(memory), Pj.project(_key(session))
            if pastes(memory) or held is None or not held.data():
                return None
            # what the data reader marks a question asking of (its target,
            # field, value) that the data has -- not every word that is
            # also a field's name (`what is my name?`)
            asked = querying.read(text, querying.known(held))
            if asked is None:
                return None
            files = {name for model in held.data().values() for name in (
                model.path.rsplit("/", 1)[-1].lower(),
                model.path.rsplit("/", 1)[-1].lower().rsplit(".", 1)[0])}
            names = querying.known(held) - files
            marked = [word.lower() for role in ("TARGET", "FIELD", "VALUE")
                      for phrase in asked.spans.get(role, ())
                      for word in str(phrase).split()]
            # the file that holds what it names (not the first file the
            # project has: `which member joined first?` "from config.yaml")
            for model in held.data().values():
                own = set()
                for path in list(model.places) + list(model.collections):
                    for part in querying._parts(path):
                        own.update({part, querying._one(part)})
                if any(one in own or querying._one(one) in own
                       for one in marked):
                    return model.path
            found = querying._said_whole(held, [str(one).lower()
                                                for one in asked.words])
            return found[1].path if found else None

        def unanswered_data(memory):
            path = of_the_data(memory)
            reply = asking._reply("data", typed(memory), (
                f"I can't answer that from {path}: what it asks is more than "
                f"I can look up in it."), {"query": None})
            reply["code"]["answer"]["status"] = "unsure"
            memory["turn"].answer = reply
            return ANSWERED

        # a question of the project's data that no part answers: said so,
        # not left to the conversation's own (`what was the last season
        # played?` of a table of seasons: "The last season was good. It was
        # short, long, and special.")
        out.append(Operator(
            name="data unanswered", apply=unanswered_data,
            proposes=lambda memory: of_the_data(memory) is not None,
            utility=DATA_UNANSWERED,
            rule="a question naming the project's data that nothing here "
                 "answers: said so, not answered of the world"))

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
