"""Telling the agent what to do, in English, and having it do it.

v691a proved the executive can plan and act. This is the part that makes it
usable: a conversation over a scene, where you describe what is there, say
what you want, and it works out the actions and takes them.

    > there is a red block on a green block, and a blue block on the table
    > put the green block on the blue block and the red block on the green
      block
    I took the red block off the green block and put it on the table, put
    the green block on the blue block, then put the red block on the green
    block.

That exchange is the Sussman anomaly, said in English.

## Why this has its own reader

v689's reader is the one that knows about dogs and stories, and pointing it
at a new act means changing the grammar -- which is where ToMi and StepGame
died. This reader knows twelve phrasings and a colour list, and that is the
point: it is a **thin shell over a thick agent**, so what the conversation
demonstrates is the planning and not the parsing. Everything it cannot read
it says it cannot read, by name, rather than guessing (`Heard.trouble`).

The acts are an `Executive`, the same as v689's `say` (`session.py:445`),
because a conversation choosing what to do with an utterance is the same
kind of choice as a planner choosing what to do with a goal, and there is no
reason for it to be a second mechanism.

## What it can do

    describing      there is a red block on the table; there are three
                    blocks; the red block is on the green block
    wanting         put the red block on the green block; put it on the
                    table; make a tower of red on green on blue
    asking          what is on the red block; where is the green block;
                    what do you see
    why             why did you put the blue block down
    meddling        actually the red block is on the table now  -- changes
                    the world behind its back, which is how a surprise is
                    provoked on purpose
    starting over   reset
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field

from research.v687.executive import (ANSWERED, CONTINUE, DECLINED, Executive,
                                     Operator, Working, episode)
from research.v691 import acting, world as W

#: The blocks that can be talked about. Colour is the whole of a block's
#: identity here: no sizes, no shapes, no two blocks of a colour. That is a
#: deliberate floor -- reference resolution is v689's problem and it is
#: solved there; repeating it badly here would only obscure the planning.
COLOURS = ("red", "green", "blue", "yellow", "orange", "purple", "black",
           "white", "grey", "pink", "brown")

#: Said of the table rather than of a block.
TABLE = ("the table", "table", "the floor", "the ground")

WORDS = re.compile(r"[a-z]+")

#: `put A on B`, and everything that means it.
WANT = re.compile(r"\b(put|move|place|stack|set)\b")
#: `there is`, and everything that means a description rather than an order.
TELL = re.compile(r"\b(there is|there are|there's|i have|you have|"
                  r"imagine|suppose)\b")


def colours_in(text: str) -> list:
    """The colours named, in the order they were said. A colour twice is a
    colour twice: `put the red block on the red block` has to be refused
    rather than quietly deduplicated."""
    return [word for word in WORDS.findall(text.lower()) if word in COLOURS]


def on_the_table(text: str) -> bool:
    return any(one in text.lower() for one in TABLE)


@dataclass
class Heard:
    """What an utterance was taken to be."""

    said: str
    act: str = "puzzled"
    blocks: list = field(default_factory=list)
    #: the world facts an order or a description comes to
    facts: list = field(default_factory=list)
    #: what could not be read, by name, so a refusal can say so
    trouble: str = ""


def read(text: str, known: list) -> Heard:
    """Twelve phrasings, and a name for anything else.

    Order matters: `what is on the red block` contains no `put`, but
    `put the red block on the table and tell me what is on the green one`
    contains both, and the order here is the order a person means them in.
    """
    plain = " ".join(text.lower().replace(",", " , ").split())
    said = Heard(text)
    named = colours_in(plain)
    said.blocks = named

    if not plain or plain in ("quit", "exit", "bye"):
        said.act = "leave"
    elif plain.startswith("reset") or plain.startswith("start over"):
        said.act = "reset"
    elif plain.startswith("why"):
        said.act = "why"
    elif plain.startswith(("actually", "now ", "in fact")):
        said.act = "meddle"
        said.facts = _placings(plain, named)
    elif plain.startswith("where"):
        said.act = "where"
    elif plain.startswith("what") and ("see" in plain or "there" in plain
                                       or "look" in plain):
        said.act = "look"
    elif plain.startswith("what") and named:
        said.act = "on top of" if " on " in plain else "where"
    elif TELL.search(plain) or (named and " is on " in plain
                                and not WANT.search(plain)):
        said.act = "describe"
        said.facts = _placings(plain, named)
    elif WANT.search(plain) or "tower" in plain:
        said.act = "want"
        said.facts = _placings(plain, named)
    if said.act in ("want", "meddle") and not said.facts:
        said.trouble = ("I could not work out what should go where"
                        if named else "you did not say which blocks")
    unknown = [one for one in named if one not in known]
    if said.act in ("want", "where", "on top of") and unknown:
        said.trouble = f"I do not know of a {unknown[0]} block"
    return said


def _placings(text: str, named: list) -> list:
    """The `on` relations an utterance states, as world facts.

    `red on green on blue` is a tower and chains; `red on green and blue on
    yellow` is two placings and does not. The difference is whether a colour
    is the right-hand side of one relation and the left of the next, which
    is decided by where `and` falls.
    """
    facts: list = []
    for clause in re.split(r"\band\b|,|;", text):
        here = colours_in(clause)
        if not here:
            continue
        if " on " not in clause and "tower" not in clause:
            # `there is a red block` -- it is somewhere, and the table is
            # the only somewhere that needs nothing else said.
            facts += [f"table {one}" for one in here]
            continue
        if len(here) == 1:
            if on_the_table(clause.split(" on ", 1)[-1]):
                facts.append(f"table {here[0]}")
            continue
        for upper, lower in zip(here, here[1:]):
            facts.append(f"on {upper} {lower}")
    return facts


# -- saying back what was done ---------------------------------------------

def phrase(name: str) -> str:
    """One action, in English. `stack green blue` -> `put the green block on
    the blue block`."""
    parts = name.split()
    if parts[0] == "take":
        return f"pick up the {parts[1]} block"
    if parts[0] == "drop":
        return f"put the {parts[1]} block on the table"
    if parts[0] == "unstack":
        return f"take the {parts[1]} block off the {parts[2]} block"
    return f"put the {parts[1]} block on the {parts[2]} block"


def in_words(fact: str) -> str:
    """One world fact, in English -- what a goal or a precondition *is*,
    so that `why` can say what was needed rather than quoting a slot."""
    parts = fact.split()
    if parts[0] == "clear":
        return f"the {parts[1]} block clear"
    if parts[0] == "held":
        return f"the {parts[1]} block in my hand"
    if parts[0] == "table":
        return f"the {parts[1]} block on the table"
    if parts[0] == "on":
        return f"the {parts[1]} block on the {parts[2]} block"
    return "my hand empty"


def _listed(facts) -> str:
    return ", ".join(in_words(one) for one in sorted(facts))


def narrate(actions: list) -> str:
    """A plan as a sentence. The pairs are joined because nobody says `I
    picked up the red block, then I put the red block on the green block`."""
    if not actions:
        return "nothing needed doing"
    parts: list = []
    index = 0
    names = [one.name.split() for one in actions]
    while index < len(names):
        one = names[index]
        following = names[index + 1] if index + 1 < len(names) else None
        if one[0] == "take" and following and following[0] == "stack":
            parts.append(f"put the {one[1]} block on the {following[2]} "
                         f"block")
            index += 2
        elif one[0] == "unstack" and following and following[0] == "stack":
            parts.append(f"moved the {one[1]} block from the {one[2]} block "
                         f"to the {following[2]} block")
            index += 2
        elif one[0] == "unstack" and following and following[0] == "drop":
            parts.append(f"took the {one[1]} block off the {one[2]} block "
                         f"and put it on the table")
            index += 2
        elif one[0] == "take":
            parts.append(f"picked up the {one[1]} block")
            index += 1
        elif one[0] == "drop":
            parts.append(f"put the {one[1]} block on the table")
            index += 1
        elif one[0] == "unstack":
            parts.append(f"took the {one[1]} block off the {one[2]} block")
            index += 1
        else:
            parts.append(f"put the {one[1]} block on the {one[2]} block")
            index += 1
    if len(parts) == 1:
        return f"I {parts[0]}"
    return f"I {', '.join(parts[:-1])}, then {parts[-1]}"


def describe(world: W.World) -> str:
    """The scene, as towers."""
    towers = world.towers()
    if not towers:
        return "the table is empty"
    said = []
    for stack in sorted(towers):
        if len(stack) == 1:
            said.append(f"the {stack[0]} block is on the table")
        else:
            # Top down, each resting on the next: `the red block is on the
            # green block, which is on the blue block, which is on the
            # table`. Bottom up reads as a list and not as a tower.
            falling = list(reversed(stack))
            line = f"the {falling[0]} block is on the {falling[1]} block"
            for one in falling[2:]:
                line += f", which is on the {one} block"
            said.append(line + ", which is on the table")
    held = [one.split()[1] for one in world.facts if one.startswith("held ")]
    if held:
        said.append(f"I am holding the {held[0]} block")
    return "; ".join(said)


def reasons(trace) -> list:
    """(operator, the goal it was fired for), for every operator that did
    something, subgoals included -- which is what `why did you ...` is
    asking. The goal names come from means-ends: `achieve clear red for
    stack green red` is the answer to why the red block was cleared."""
    out = []
    for step in trace.fired:
        if step.outcome != DECLINED:
            out.append((step.operator, trace.goal))
    for inner in trace.subgoals:
        out.extend(reasons(inner))
    return out


# -- the conversation ------------------------------------------------------

class Gremlin(W.World):
    """A world with something else in it.

    Between turns a person can move a block and the agent simply plans
    again from what it finds -- it is never *surprised*, because it was not
    expecting anything at the time. A surprise needs the world to move
    **while an action is being taken**, so this is a world that lets
    something do that: after each action, whatever `meddling` is gets a
    turn. It is how `--gremlin` makes the impasse in `acting.agent`
    reachable from a conversation rather than only from a test.
    """

    def __init__(self, facts, meddling=None) -> None:
        super().__init__(facts)
        self.meddling = meddling

    def do(self, action) -> bool:
        done = super().do(action)
        if done and self.meddling is not None:
            self.meddling(self)
        return done


def topple(world: "Gremlin") -> None:
    """The simplest thing that can go wrong: the block most recently put on
    another one falls off onto the table. Once, so a conversation stays
    readable."""
    if getattr(world, "toppled", False):
        return
    for fact in sorted(world.facts):
        if fact.startswith("on "):
            _, upper, lower = fact.split()
            world.toppled = True
            world.facts = ((world.facts - {fact})
                           | {f"table {upper}", f"clear {lower}"})
            return


class Table:
    """A scene, and everything that can be said about it.

    The world is real -- `World`, changed only by `do`, every change
    announced with `effect` -- so the agent that acts here is the same
    agent, unmodified, that `acting.solve` runs on the fixed suite. Nothing
    about the conversation is special-cased into it.
    """

    def __init__(self, meddling=None) -> None:
        #: what moves a block while the agent is acting, or None
        self.meddling = meddling
        self.world = Gremlin({"empty"}, meddling)
        self.blocks: list = []
        #: the last thing planned and done, for `why`
        self.last: acting.Attempt | None = None
        self.last_plan: list = []
        self.last_reasons: list = []
        #: every turn as (said, act, reply), for a test to read back
        self.turns: list = []

    # -- the world ---------------------------------------------------------
    def add(self, colour: str) -> None:
        if colour in self.blocks:
            return
        self.blocks.append(colour)
        self.world.facts = (self.world.facts
                            | {f"table {colour}", f"clear {colour}"})

    def place(self, facts: list) -> None:
        """Put the scene into a described arrangement, directly -- this is
        the description of a world, not an action in it, so it does not go
        through `do` and is not something the agent did."""
        for fact in facts:
            for colour in fact.split()[1:]:
                self.add(colour)
        for fact in facts:
            if not fact.startswith("on "):
                continue
            _, upper, lower = fact.split()
            self.world.facts = ((self.world.facts
                                 - {f"table {upper}", f"clear {lower}"})
                                | {fact})

    def problem(self, goal) -> W.Problem:
        return W.Problem("what you asked for", tuple(self.blocks),
                         frozenset(self.world.facts), frozenset(goal))

    # -- one utterance -----------------------------------------------------
    def say(self, text: str) -> str:
        heard = read(text, self.blocks)
        memory = Working({"heard": heard}, goal=f"do something with: {text}")
        with episode():
            self.acts().run(memory)
        reply = memory.get("reply", "I did not follow that")
        self.turns.append((text, heard.act, reply))
        return reply

    def acts(self) -> Executive:
        """One operator per act, chosen by what was heard -- v689's shape,
        because deciding what an utterance is for is the same kind of choice
        as deciding what to do about a goal."""
        def answering(name, handler, effects=()):
            def apply(memory):
                memory["reply"] = handler(memory["heard"])
                return ANSWERED
            return Operator(name=name, apply=apply, gives=("reply",),
                            effects=effects,
                            proposes=lambda memory, name=name:
                            memory["heard"].act == name,
                            rule=f"the utterance is a {name}")

        return Executive([
            answering("leave", lambda heard: "goodbye"),
            answering("reset", self._reset),
            answering("describe", self._describe),
            answering("want", self._want, effects=(W.WORLD,)),
            answering("meddle", self._meddle),
            answering("look", lambda heard: describe(self.world)),
            answering("where", self._where),
            answering("on top of", self._on_top_of),
            answering("why", self._why),
            answering("puzzled", lambda heard:
                      "I only know about blocks on a table -- try `there is "
                      "a red block on a green block`, or `put the red block "
                      "on the table`"),
        ], name="talking")

    # -- the acts ----------------------------------------------------------
    def _reset(self, heard: Heard) -> str:
        self.__init__(self.meddling)
        return "the table is empty"

    def _describe(self, heard: Heard) -> str:
        self.place(heard.facts)
        for colour in heard.blocks:
            self.add(colour)
        return f"all right: {describe(self.world)}"

    def _meddle(self, heard: Heard) -> str:
        """Change the world without the agent acting -- what provokes a
        surprise, and the only way a conversation can."""
        if heard.trouble:
            return f"sorry -- {heard.trouble}"
        before = describe(self.world)
        for fact in heard.facts:
            parts = fact.split()
            self._put(parts[1], parts[2] if parts[0] == "on" else None)
        after = describe(self.world)
        return (f"I see -- {after}" if after != before
                else "that is how it already was")

    def _put(self, block: str, under: str | None) -> None:
        """Move a block by fiat, leaving the world consistent.

        Not an action: nothing in `Action` is being applied, and the agent
        is not told. Keeping the rest of the world true is fiddly in exactly
        the way a delete list makes unnecessary -- whatever the block was
        on becomes clear, whatever it is put on stops being clear, and
        anything that was on top of *it* goes to the table -- which is one
        more argument for the world being a set of facts with actions over
        it rather than a picture that is edited.
        """
        facts = set(self.world.facts)
        for one in list(facts):
            if one.startswith(f"on {block} "):
                facts.discard(one)
                facts.add(f"clear {one.split()[2]}")
            elif one.startswith("on ") and one.endswith(f" {block}"):
                # Whatever was resting on it has nowhere to be but the table.
                facts.discard(one)
                facts.add(f"table {one.split()[1]}")
                facts.add(f"clear {one.split()[1]}")
        facts.discard(f"held {block}")
        facts.discard(f"table {block}")
        facts.add("empty")
        facts.add(f"clear {block}")
        if under is None:
            facts.add(f"table {block}")
        else:
            for one in list(facts):
                if one.startswith("on ") and one.endswith(f" {under}"):
                    facts.discard(one)
                    facts.add(f"table {one.split()[1]}")
                    facts.add(f"clear {one.split()[1]}")
            facts.discard(f"clear {under}")
            facts.discard(f"held {under}")
            facts.add(f"on {block} {under}")
        self.world.facts = frozenset(facts)

    def _want(self, heard: Heard) -> str:
        if heard.trouble:
            return f"sorry -- {heard.trouble}"
        problem = self.problem(heard.facts)
        report = acting.Attempt(name=problem.name)
        before = len(self.world.did)
        trace = acting.agent(problem, self.world, None, report).run(
            Working(goal=f"do: {heard.said}"))
        self.last = report
        # The agent's own trace says it planned and acted; the *planner's*
        # trace says what each action was for, and that is what `why` wants.
        self.last_reasons = (reasons(report.search.trace)
                             if report.search.trace is not None else [])
        self.last_plan = list(self.world.did[before:])
        if not report.solved:
            return ("I could not see a way to do that"
                    + (f", and I have {describe(self.world)}"
                       if self.last_plan else ""))
        return self._story(report)

    def _story(self, report: acting.Attempt) -> str:
        """What was done, broken where the world did something unexpected.

        Told in stretches rather than as one list, because after a surprise
        the agent goes back over ground it has already covered and `I put
        the green block on the blue block, put the green block on the blue
        block` is true and unreadable. Each stretch is what one plan came
        to.
        """
        cuts = [gap.after for gap in report.gaps]
        pieces, at = [], 0
        for cut in cuts + [len(self.last_plan)]:
            if cut > at:
                pieces.append(narrate(self.last_plan[at:cut]))
            at = cut
        if not pieces:
            return "nothing needed doing"
        said = pieces[0]
        for gap, piece in zip(report.gaps, pieces[1:]):
            said += (f". Then something moved while I was working: I "
                     f"expected {_listed(gap.missing) or 'no change'}, and "
                     f"found {_listed(gap.extra) or 'nothing'}. So I "
                     f"planned again, and {piece[2:]}")
        if len(pieces) == len(report.gaps):
            gap = report.gaps[-1]
            said += (f". Then something moved: I expected "
                     f"{_listed(gap.missing) or 'no change'} and found "
                     f"{_listed(gap.extra) or 'nothing'}, but what you "
                     f"asked for held anyway")
        return said

    def _where(self, heard: Heard) -> str:
        if heard.trouble:
            return f"sorry -- {heard.trouble}"
        if not heard.blocks:
            return "which block?"
        block = heard.blocks[0]
        for fact in sorted(self.world.facts):
            if fact.startswith(f"on {block} "):
                return f"the {block} block is on the {fact.split()[2]} block"
        if f"held {block}" in self.world.facts:
            return f"I am holding the {block} block"
        return f"the {block} block is on the table"

    def _on_top_of(self, heard: Heard) -> str:
        if heard.trouble:
            return f"sorry -- {heard.trouble}"
        if not heard.blocks:
            return "on which block?"
        block = heard.blocks[-1]
        above = [fact.split()[1] for fact in sorted(self.world.facts)
                 if fact.startswith("on ") and fact.endswith(f" {block}")]
        if not above:
            return f"nothing is on the {block} block"
        return f"the {above[0]} block is on the {block} block"

    def _why(self, heard: Heard) -> str:
        """What the last thing done was for. The means-ends subgoal names
        are already the answer -- `achieve clear red for stack green red` --
        so this reads them rather than inventing an explanation."""
        if not self.last_plan:
            return "I have not done anything yet"
        wanted = heard.blocks
        for action in self.last_plan:
            parts = action.name.split()
            if wanted and parts[1] not in wanted:
                continue
            for name, goal in self.last_reasons:
                if name == action.name and goal.startswith("achieve "):
                    wanted_for = goal.split(" for ", 1)[-1]
                    needed = goal[len("achieve "):].split(" for ")[0]
                    needed = " and ".join(in_words(one) for one
                                          in needed.split(", "))
                    return (f"I {narrate([action])[2:]} because I needed "
                            f"{needed} before I could "
                            f"{phrase(wanted_for)}")
        return (f"{narrate(self.last_plan)} -- that was the shortest way I "
                f"found to what you asked for")


BANNER = """A table with blocks on it. Tell me what is there, then what you
want. `reset` starts over, `quit` leaves.

  there is a red block on a green block, and a blue block on the table
  put the green block on the blue block and the red block on the green block
  what is on the blue block
  why did you move the red block
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gremlin", action="store_true",
                        help="let a block fall off, once, while the agent "
                             "is working: a real surprise rather than a "
                             "change between turns")
    options = parser.parse_args(argv)
    table = Table(topple if options.gremlin else None)
    print(BANNER)
    while True:
        try:
            said = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        reply = table.say(said)
        print(f"  {reply}\n")
        if table.turns and table.turns[-1][1] == "leave":
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
