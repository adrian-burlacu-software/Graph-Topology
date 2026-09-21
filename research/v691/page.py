"""The agent on the page: v691's acts, in v689's conversation.

Importing this registers v691's acts with `v689.session`, so a page that
serves a conversation can also be given something to do. Nothing in v689
imports v691 -- a later layer reaching into an earlier one is how the
cascades this executive replaced became impossible to follow -- so the
direction is the other way round: `session.contributes` takes a callable,
the operators join the act executive's conflict set on their own utility,
and a turn's trace on the page shows what they were chosen over.

That is the whole of the wiring, and it matters that it is: **the agent is
an act like any other.** It reads an utterance, does something, and answers.
What the page then shows for one of its turns is what it shows for any --
except that the runs nested inside are a planner's, so the goal stack, the
means-ends subgoals and the actions taken are all there, because
`Executive.run` records every run of an open episode (E3) and the planner
is an executive.

## What it takes to be about a scene

**A world has to have been opened, by asking for one.** Until then this
layer proposes nothing at all, and the page behaves exactly as it did.
That is not caution for its own sake: `the dog is on the mat` reads as
`on dog mat` in the blocks domain without any trouble, and a layer that
took it would break every question v687 to v690 answer. So:

    > what worlds do you have
    > use the blocks world
    > there is a red block on a green block

and from then on, an utterance that reads as facts of *that* domain is
this layer's. `ORDER` and `ASK` put those acts above `generic` -- what an
unread utterance falls to -- so the page still answers what it can answer,
and only what nothing else claims becomes an order.
"""
from __future__ import annotations

import re

from research.v687.executive import ANSWERED, Operator
from research.v689 import session as v689
from research.v691.domains import DOMAINS
from research.v691.scene import Heard, Scene

#: A scene per conversation, so two tabs are two tables.
SCENES: dict = {}

#: Utilities. An `Executive` numbers operators that carry none from the
#: order they were written -- `len - index` -- so v689's run from about
#: forty down to one, with `generic` last. These sit above all of them.
#:
#: Being above is right, and it is only right because of the gate: while no
#: world is open none of these propose at all, and once one is open, an
#: utterance that reads as facts of *that domain* is about the scene rather
#: than about what is true of dogs. `what is on the blue block` has to
#: outrank v689's `located`, which otherwise answers it from episodic
#: memory and says nothing was told -- true, and not what was asked.
SETUP = 210.0
ORDER = 200.0
ASK = 190.0

WANTED = re.compile(r"\b(put|move|place|get|bring|take|make|build|set|"
                    r"deliver|fetch|carry|stack|send)\b")
TOLD = re.compile(r"\b(there is|there are|there's|i have|imagine|suppose|"
                  r"is at|is in|is on|are at|are in|are on)\b")
MEDDLED = re.compile(r"^(actually|in fact|now )")
LOOKED = re.compile(r"\b(what do you see|what is there|what is on the "
                    r"table|describe the scene|what does it look like)\b")
WORLDS = re.compile(r"\b(what worlds|which worlds|what domains)\b")
USING = re.compile(r"\buse the (\w+) world\b|\bswitch to (\w+)\b")


def scene_for(session) -> Scene:
    key = getattr(session, "conversation", "") or id(session)
    if key not in SCENES:
        SCENES[key] = Scene()
    return SCENES[key]


def hear(text: str, scene: Scene) -> Heard:
    """What v691 makes of an utterance, and how sure it is.

    `weight` is what the act executive ranks on, and it is zero for anything
    this should not touch. Being wrong here is expensive in one direction
    only: an order missed is a turn wasted, and a question stolen is the
    page broken.
    """
    plain = " ".join(text.lower().replace(",", " , ").split())
    said = Heard(said=text)
    found = USING.search(plain)
    if found:
        said.domain = found.group(1) or found.group(2)
        said.act = "use a world"
        said.weight = SETUP
        return said
    if WORLDS.search(plain):
        said.act = "which worlds"
        said.weight = SETUP
        return said
    if not scene.open:
        # No world open: this layer has nothing to say, and says nothing.
        return said
    facts = scene.reader.facts_in(plain)
    wants = facts
    try:
        wants = scene.reader.facts_in(plain, wanting=True) or facts
    except TypeError:
        pass
    said.facts = facts
    said.wants = wants
    # Every thing the scene knows of that the utterance mentions, in the
    # order said, plus anything a fact named. `where is the book` states no
    # fact, so the facts alone would leave it with nothing to answer about.
    words = re.findall(r"[a-z][a-z0-9-]*", plain)
    said.names = [one for one in words if one in scene.objects]
    said.names += [one for fact in facts for one in fact.split()[1:]
                   if one not in said.names]
    known = bool(scene.objects)
    if LOOKED.search(plain) and known:
        said.act = "look"
        said.weight = ASK
    elif plain.startswith("why") and scene.last_plan:
        said.act = "why"
        said.weight = ASK
    elif MEDDLED.search(plain) and facts and known:
        said.act = "meddle"
        said.weight = ORDER
    elif plain.startswith(("where is", "where are")) and known:
        said.act = "where"
        said.weight = ASK
    elif plain.startswith("what is on") and known:
        said.act = "upon"
        said.weight = ASK
    elif (WANTED.search(plain) or _an_order(plain)) and any(
            one.split()[0] in scene.domain.goalish for one in wants):
        said.act = "want"
        said.weight = ORDER
    elif facts and (TOLD.search(plain) or not known
                    or not ASKING.match(plain)):
        said.act = "tell"
        said.weight = ORDER
    return said


#: What a question or a query opens with. Everything else that states a fact
#: states it, which is what makes `the door is closed` something told.
ASKING = re.compile(r"^(what|where|why|who|when|how|is|are|was|were|can|"
                    r"could|does|do|did|will|would|should|may|might)")


def _an_order(plain: str) -> bool:
    """Whether it opens with a verb, asked of VerbNet rather than of a list.

    `open the door` is an order and `put the book down` is an order, and
    the only thing they have in common is that English puts a verb first.
    Which words are verbs is what VerbNet is; keeping a list here would be
    the hand-written domain coming back in through the reader.
    """
    first = plain.split()[0] if plain.split() else ""
    if not first or first in ("the", "a", "an", "there"):
        return False
    from research.v689 import change
    return first in change.frames()


#: The last utterance read, per conversation. `proposes` runs every cycle
#: and for every operator, and it must *read* working memory rather than
#: write to it -- a condition that left something behind would spend it
#: before the action ran. So what was heard is kept here instead, worked out
#: once per utterance.
_HEARD: dict = {}


def heard_for(session, said: str) -> Heard:
    key = getattr(session, "conversation", "") or id(session)
    was = _HEARD.get(key)
    if was is None or was[0] != said:
        was = (said, hear(said, scene_for(session)))
        _HEARD[key] = was
    return was[1]


def used(scene: Scene, heard: Heard) -> str:
    if heard.domain in ("open", "anything", "general", "real"):
        from research.v691.openworld import Open, resolver
        scene.__init__(Open(resolver()))
        return ("all right -- a world with nothing declared about it. What "
                "can be done comes from what verbs mean, and what a thing "
                "is comes from the store. Tell me what is there.")
    if heard.domain not in DOMAINS:
        return (f"I do not have a {heard.domain} world. I have "
                f"{', '.join(sorted(DOMAINS))}.")
    scene.use(heard.domain)
    return f"all right, the {heard.domain} world. Tell me what is there."


def which(scene: Scene, heard: Heard) -> str:
    said = [f"{name} ({', '.join(domain.kinds)}; "
            f"{len(domain.schemas)} kinds of action)"
            for name, domain in sorted(DOMAINS.items())]
    return ("I can work in any of these, and the planner is the same one "
            "in each: " + "; ".join(said)
            + ". There is also **open**, which has nothing declared about "
              "it at all: what can be done comes from what verbs mean and "
              "what things are comes from the store.")


#: act -> (handler, utility, what the operator's rule says, what it changes).
#: One table, read by the page's operators and by the REPL, so a turn cannot
#: mean two things depending on where it was said.
ACTS = {
    "use a world": (used, SETUP, "change the world talked of", ()),
    "which worlds": (which, SETUP, "the worlds it has", ()),
    "tell": (lambda scene, heard: scene.tell(heard), ORDER,
             "what is in the scene", ()),
    "want": (lambda scene, heard: scene.want(heard), ORDER,
             "plan and act: means-ends over the world", ("world",)),
    "meddle": (lambda scene, heard: scene.meddle(heard), ORDER,
               "the scene changed without it acting", ()),
    "look": (lambda scene, heard: scene.look(), ASK,
             "the scene as it stands", ()),
    "where": (lambda scene, heard: scene.where(heard), ASK,
              "where one thing is", ()),
    "upon": (lambda scene, heard: scene.upon(heard), ASK,
             "what is on one thing", ()),
    "why": (lambda scene, heard: scene.why(heard), ASK,
            "what the last thing done was for", ()),
}


def replies(session) -> list:
    """v691's acts, as operators for v689's act executive."""
    def answering(name: str, handler, utility: float, rule: str,
                  effects=()) -> Operator:
        def proposes(memory) -> bool:
            return heard_for(session, memory["reading"].said).act == name

        def apply(memory):
            scene = scene_for(session)
            heard = heard_for(session, memory["reading"].said)
            text = handler(scene, heard)
            memory["turn"].answer = {
                "outcome": "acted", "source": "world", "act": name,
                "text": text,
                # Said as it stands: this is already English written from a
                # plan, and the decoder is trained to paraphrase v689's
                # verdicts, not to re-tell a story it was not given.
                "spoken": text,
                "scene": sorted(scene.world.facts),
                "domain": scene.domain.name if scene.open else ""}
            return ANSWERED

        return Operator(name=name, apply=apply, proposes=proposes,
                        utility=utility, rule=rule, effects=effects)

    return [answering(name, handler, utility, rule, effects)
            for name, (handler, utility, rule, effects) in ACTS.items()]


def say_to(scene: Scene, text: str) -> str:
    """One utterance against one scene, outside any page.

    The same acts the page runs, chosen the same way -- by the highest
    weight -- so the REPL in `talking.py` and a turn on the page cannot
    behave differently.
    """
    heard = hear(text, scene)
    if not heard.act:
        return ("I did not take that as being about the world in front of "
                "me. Tell me what is there, or what you want done with it.")
    return ACTS[heard.act][0](scene, heard)


v689.contributes(replies)
