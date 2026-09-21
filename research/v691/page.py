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

from research.v687.executive import ANSWERED, CONTINUE, Operator
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
#: Above everything, because it does not answer: it writes down what the
#: utterance said about the world and lets the cycle go on.
NOTING = 300.0

WANTED = re.compile(r"\b(put|move|place|get|bring|take|make|build|set|"
                    r"deliver|fetch|carry|stack|send)\b")
TOLD = re.compile(r"\b(there is|there are|there's|i have|imagine|suppose|"
                  r"is at|is in|is on|are at|are in|are on)\b")
MEDDLED = re.compile(r"^(actually|in fact|now )")
LOOKED = re.compile(r"\b(what do you see|what is there|what is on the "
                    r"table|describe the scene|what does it look like)\b")
WORLDS = re.compile(r"\b(what worlds|which worlds|what domains)\b")
USING = re.compile(r"\buse the (\w+) world\b|\bswitch to (\w+)\b")


#: What has been worked out about acting, shared by every conversation and
#: kept on disk. One store, because `a door cannot be open and closed` is
#: not true only of the conversation it was said in.
_LEARNED = None


def store():
    global _LEARNED
    if _LEARNED is None:
        from research.v691.learned import Learned
        _LEARNED = Learned()
    return _LEARNED


def scene_for(session) -> Scene:
    """The scene this conversation is about.

    **A conversation starts in the open world**, which is the one with
    nothing declared about it: what can be done comes from what verbs mean
    and what things are comes from the store. Saying `use the blocks world`
    still gets a declared one, and that is now the special case rather than
    the way in.
    """
    key = getattr(session, "conversation", "") or id(session)
    if key not in SCENES:
        from research.v691.openworld import Open, resolver
        SCENES[key] = Scene(Open(resolver(), store()))
    scene = SCENES[key]
    if scene.open and getattr(scene.domain, "can", False) is None:
        scene.domain.can = able(session)
    return scene


def able(session):
    """Whether a thing of a kind can do something itself, asked of what the
    conversation knows (`Session.can`: v688 on the kind, and v687's walk
    for a kind taught here). A pig cannot fly; a plane can."""
    judged = getattr(session, "can", None)
    if judged is None:
        return None
    known: dict = {}

    def can(kind: str, verb: str) -> bool:
        if (kind, verb) not in known:
            known[kind, verb] = bool(judged(kind, verb))
        return known[kind, verb]
    return can


#: conversation -> how much of its session's experience has been learned.
_ABSORBED: dict = {}


def absorb(session) -> list:
    """Learn what the story showed about doing things, into long-term
    memory: v689 keeps it as experience (`Session.experience`), and this is
    where it stops being about one pig and one plane. Returns what was new.
    """
    key = getattr(session, "conversation", "") or id(session)
    seen = getattr(session, "experience", [])
    fresh, _ABSORBED[key] = seen[_ABSORBED.get(key, 0):], len(seen)
    learned, out = store(), []
    for one in fresh:
        if one.get("kind") == "carried" and learned.carry(
                one["verb"], one["thing_sense"] or one["thing"],
                one["carrier_sense"] or one["carrier"], one["carrier"],
                one.get("way", ""), bool(one.get("own")),
                " / ".join(one.get("said", ()))):
            out.append(one)
    return out


def scene_ish(facts, scene) -> bool:
    """Whether these facts are about the world in front of it.

    With the open world as the default this is the whole of the guard, and
    it has to be strict in one direction: `a whale is a mammal` reads as
    `mammal whale` and is v688's question, not a scene. A fact is the
    scene's if it says where something is, if it says a thing is in a state
    -- asked of WordNet, because `closed` is an adjective and `mammal` is
    not -- or if it is about something already being talked about.
    """
    from research.v689.change import adjective
    if not facts:
        return False
    if getattr(scene.domain, "says", None):
        # A declared domain was read through its own `say` lines, so
        # anything they matched is about it by construction. The guard is
        # for the open world, where the patterns are English's and not any
        # world's.
        return True
    for fact in facts:
        parts = fact.split()
        if len(parts) > 1 and parts[1] in scene.objects:
            continue
        if parts[0] == "at":
            continue
        if parts[0] == "with":
            # `does a table have legs` reads as `with legs table`, and it is
            # v688's question about a kind. A holder already being talked
            # about is what makes it this scene's instead, so `with` alone
            # never introduces anything.
            if len(parts) > 2 and parts[2] in scene.objects:
                continue
            return False
        if len(parts) == 2 and adjective(parts[0]):
            continue
        return False
    return True


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
    from research.v691.learned import teaching
    taught = teaching(plain)
    if taught and getattr(scene.domain, "learned", None) is not None:
        said.act = "learn"
        said.weight = SETUP
        said.taught = taught
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
    doing = getattr(scene.domain, "a_doing", None)
    asked_for = [one for one in wants
                 if one.split()[0] in scene.domain.goalish
                 or (doing is not None and doing(one.split()[0]))]
    if LOOKED.search(plain) and known:
        said.act = "look"
        said.weight = ASK
    elif plain.startswith("why") and scene.last_plan:
        said.act = "why"
        said.weight = ASK
    elif (MEDDLED.search(plain) and facts and known
          and scene_ish(facts, scene)):
        said.act = "meddle"
        said.weight = ORDER
    elif plain.startswith(("where is", "where are")) and known:
        said.act = "where"
        said.weight = ASK
    elif plain.startswith("what is on") and known:
        said.act = "upon"
        said.weight = ASK
    elif ((ASKING.match(plain) or plain.rstrip().endswith("?"))
          and not REQUEST.match(plain)
          and (WANTED.search(plain) or _an_order(plain)) and asked_for):
        # `what steps are required to make a pig fly`, `how would I get the
        # book to the kitchen`: a question about an order is asking what it
        # would take, not asking for it done.
        said.act = "how"
        said.weight = ASK
    elif ordered(plain) and any(
            one.split()[0] in scene.domain.goalish for one in wants):
        said.act = "want"
        said.weight = ORDER
    elif (facts and not ASKING.match(plain)
          and scene_ish(facts, scene)):
        said.act = "tell"
        said.weight = ORDER
    return said


#: What a question or a query opens with. Everything else that states a fact
#: states it, which is what makes `the door is closed` something told.
ASKING = re.compile(r"^(what|which|where|why|who|when|whose|how|is|are|was|"
                    r"were|can|could|does|do|did|will|would|should|may|"
                    r"might)\s")


#: A question that is an order said politely: `can you put the book on the
#: table` asks for it done, and is not asking what it would take.
REQUEST = re.compile(r"^(can|could|would|will)\s+you\s")


def ordered(plain: str) -> bool:
    """Whether it is said as an order: a verb first (`put the book on the
    table`), `please`, or `can you`. `i put the key in the drawer` has the
    same verb and tells what happened -- taken for an order, it was done
    instead of remembered, and v689 never heard the story."""
    words = plain.split()
    if words[:1] == ["please"]:
        plain = " ".join(words[1:])
    return bool(REQUEST.match(plain)) or _an_order(plain)


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


def taught(scene: Scene, heard: Heard) -> str:
    """Learn what was said about acting, into long-term memory.

    The two things `DESIGN.md` §9 said were missing and could not be read
    off a verb, because they are facts about doors and hands rather than
    about words: which states exclude each other, and what else has to be
    true. Both are ordinary things to say while somebody is getting it
    wrong, and both are kept until they are taken back.
    """
    learned = scene.domain.learned
    said = []
    for one in getattr(heard, "taught", ()):
        if one[0] == "exclude" and learned.exclude(one[1], one[2],
                                                   heard.said):
            said.append(f"a thing cannot be {one[1]} and {one[2]}")
        elif one[0] == "require" and learned.require(one[1], one[2],
                                                     heard.said):
            plain = one[2].replace("?subject", "you").replace(
                "?object", "it")
            said.append(f"to {one[1]} something, "
                        f"{scene.domain.in_words(plain)}")
    if not said:
        return "I already knew that"
    return "noted, and I will remember: " + "; ".join(said)


#: act -> (handler, utility, what the operator's rule says, what it changes).
#: One table, read by the page's operators and by the REPL, so a turn cannot
#: mean two things depending on where it was said.
ACTS = {
    "learn": (taught, SETUP, "learn something about acting", ()),
    "use a world": (used, SETUP, "change the world talked of", ()),
    "which worlds": (which, SETUP, "the worlds it has", ()),
    "tell": (lambda scene, heard: scene.tell(heard), ORDER,
             "what is in the scene", ()),
    "want": (lambda scene, heard: scene.want(heard), ORDER,
             "plan and act: means-ends over the world", ("world",)),
    "how": (lambda scene, heard: scene.how(heard), ASK,
            "what it would take: planned, not done", ()),
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


#: conversation -> the number of the last turn this layer answered.
_DONE: dict = {}


def whole(memory) -> str:
    """What was said this turn, all of it.

    v689 splits `get the cup to the shop and the book to the garden` into
    two claims and runs the act executive once for each. An order is not
    two orders: planned claim by claim, the second goal was lost and the
    second claim, read on its own as `the book to the garden`, was taken
    for a fact. So this layer hears the whole utterance, once.
    """
    turn = memory.get("turn")
    return getattr(turn, "said", "") or memory["reading"].said


def recorded(scene: Scene, heard: Heard) -> bool:
    """Whether a statement is only written down here, and answered by v689.

    In the open world, `the pig is in a field` is something v689 has to
    remember -- where the pig is today, against where it was yesterday (T3)
    -- and an act here that answered it took it away from episodic memory
    altogether. So in the open world a statement is noted, and the turn goes
    on; a declared world (`use the blocks world`) still answers its own.
    """
    return heard.act == "tell" and not getattr(scene.domain, "says", None)


def replies(session) -> list:
    """v691's acts, as operators for v689's act executive."""
    key = getattr(session, "conversation", "") or id(session)

    def answering(name: str, handler, utility: float, rule: str,
                  effects=()) -> Operator:
        def proposes(memory) -> bool:
            heard = heard_for(session, whole(memory))
            return heard.act == name and not recorded(scene_for(session),
                                                      heard)

        def apply(memory):
            scene = scene_for(session)
            absorb(session)
            heard = heard_for(session, whole(memory))
            _DONE[key] = getattr(memory.get("turn"), "number", None)
            text = handler(scene, heard)
            planning = (scene.planning() if name in ("want", "how")
                        else {})
            memory["turn"].answer = {
                "outcome": "acted", "source": "world", "act": name,
                "text": text, "planning": planning,
                # Said as it stands: this is already English written from a
                # plan, and the decoder is trained to paraphrase v689's
                # verdicts, not to re-tell a story it was not given.
                "spoken": text,
                "scene": sorted(scene.world.facts),
                "domain": scene.domain.name if scene.open else ""}
            return ANSWERED

        return Operator(name=name, apply=apply, proposes=proposes,
                        utility=utility, rule=rule, effects=effects)

    def noting(memory):
        """Record what an utterance says about the world, and go on.

        This is the one operator here that does not answer. It returns
        CONTINUE, so the cycle carries on and v689 answers the turn as it
        always did -- which is what lets the open world be the default
        without taking `a whale is a mammal` away from v688. The scene
        stays current either way, so an order two turns later knows where
        things are.
        """
        scene = scene_for(session)
        absorb(session)
        heard = heard_for(session, whole(memory))
        memory["noted"] = True
        if not scene.open or (heard.act and not recorded(scene, heard)):
            # Something here is going to act on it; it will record its own.
            return CONTINUE
        if scene_ish(heard.facts, scene):
            scene.tell(heard)
        return CONTINUE

    def settled(memory):
        """A later claim of a turn this layer has already answered whole.

        It answers with nothing, so the merged reply is the one reply and
        not the same plan told twice -- and so nothing else reads the
        leftover claim as something new.
        """
        memory["turn"].answer = {"outcome": "acted", "source": "world",
                                 "text": ""}
        return ANSWERED

    return [Operator(name="settled", apply=settled, utility=NOTING + 1,
                     rule="the rest of an utterance already acted on",
                     proposes=lambda memory: _DONE.get(key) is not None
                     and _DONE.get(key) == getattr(memory.get("turn"),
                                                   "number", None)),
            Operator(name="noting", apply=noting, gives=("noted",),
                     utility=NOTING, rule="what the utterance says is there",
                     proposes=lambda memory: scene_for(session).open)] + [
        answering(name, handler, utility, rule, effects)
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
