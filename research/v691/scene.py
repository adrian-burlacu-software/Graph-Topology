"""A scene you can talk about, in whatever domain it is in.

`talking.py` was blocks: it knew colours, it said *block* in eleven places,
and the only thing it could be asked for was a tower. This is the same
conversation with the domain taken out of it and put in `domains.py`, where
it is a string. What is left here is what is true of *any* world you can
discuss -- the things in it, what holds, what you want, and what was done --
and the only reason it can say `the red block is on the green block` is that
the blocks domain has a `say on` line.

## Reading is the `say` lines, backwards

A domain says how a fact reads:

    say on     the {0} block is on the {1} block
    reads on   {0} on {1}

and both lines are compiled into patterns, so *there is a red block on a
green block*, *the red block is on the green block* and *put the red block
on the green block* all come to `on red green`. One template per direction
and no second grammar, so the words it understands and the words it uses
cannot drift apart.

A placeholder matches a name with the optional trimmings a person puts round
it: `the`, `a`, and the kind word afterwards. So `{0}` reads *red*, *the
red*, *a red block* and *the red block* the same way, in any domain, because
the kinds come from the domain too.

## What it can do

    telling     there is a red block on a green block; the book is at the
                shop; the blue van is in york
    wanting     put the red block on the green block; get the book to the
                office -- anything whose facts are `goalish` in the domain
    asking      what is on the red block; where is the book; what do you see
    why         why did you move the red block
    meddling    actually the red block is on the table now
    reporting   the door is still closed, it is locked -- what it did did
                not happen, and what the person said is why (`failed`)
    asking      an order it cannot plan is not dropped: it asks for what is
                missing, and the answer resumes it (`missing`, `resume`)
    changing    use the errands world; what worlds do you have
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from research.v687.executive import (ANSWERED, DECLINED, Executive, Operator,
                                     Working)
from research.v691 import acting, lessons, world as W
from research.v691.domains import DOMAINS

#: Words that are never an object's name, however a pattern falls.
NEVER = frozenset("""a an the and or is are was were be do does did to of on
in at it its this that there here what where why who how i you me my your
please now actually then so put move place get take bring make build set
tell say left right up down with from into onto for all some any thing
things world worlds use using""".split())


#: How English says something just done did not happen (`Scene.reported`).
#: `the door did not open`, `the door is not open`: the thing and what it
#: should have come to.
DENIED = re.compile(r"\b(?:the |a |an )?(\w+)\s+(?:is not|is n't|isn't|"
                    r"are not|aren't|did not|didn't|does not|doesn't|"
                    r"would not|wouldn't|won't|will not|could not|"
                    r"couldn't)\s+(?:be\s+)?(\w+)")
#: `the door is still closed`: a state the action should have ended.
STILL = re.compile(r"\bstill\b")
#: `that did not work`, `it failed`.
FAILED = re.compile(r"\b(?:that|it|this)\s+(?:did not|didn't|does not|"
                    r"doesn't)\s+work\b|\b(?:that|it)\s+failed\b")


def _participle(verb: str) -> str:
    from research.v691.verbs import participle
    return participle(verb)


#: A yes, a no, and being told to leave it, to a question asked
#: (`Scene.resume`).
YES = re.compile(r"^(?:yes|yeah|yep|it is|they are|sure|correct|right)\b")
NO = re.compile(r"^(?:no|nope|it is not|it isn't|it's not|they are not|"
                r"they aren't|not)\b")
DROP = re.compile(r"\b(?:never mind|forget it|leave it|don't bother|"
                  r"stop)\b")


#: A copula a person puts between a name and where it is. Optional, and
#: allowed after every placeholder, so a domain does not need a `reads` line
#: per verb: `the keys are at the office` and `the keys at the office` are
#: the same fact said twice.
COPULA = r"(?:\s+(?:is|are|was|were|'s|'re))?"


def names_of(domain) -> re.Pattern:
    """What an object may be called: a word, with `the`/`a` and the kind
    word optional round it. Built from the domain, so nothing here knows
    what a block or a van is."""
    kinds = "|".join(sorted(domain.kinds, key=len, reverse=True)) or "x^"
    return re.compile(rf"(?:the |a |an )?([a-z][a-z0-9-]*)(?:\s+(?:{kinds}))?")


@dataclass
class Heard:
    """What an utterance was taken to be."""

    said: str = ""
    act: str = ""
    facts: list = field(default_factory=list)
    names: list = field(default_factory=list)
    #: the facts an *order* states, where those are read differently from
    #: the facts a statement states (`openworld.WANTINGS`)
    wants: list = field(default_factory=list)
    domain: str = ""
    #: what a sentence teaches about acting (`learned.teaching`)
    taught: list = field(default_factory=list)
    trouble: str = ""
    #: how sure, so the act executive can rank it against v689's own acts
    weight: float = 0.0


class Reader:
    """One domain's phrasings, compiled once."""

    def __init__(self, domain) -> None:
        self.domain = domain
        self.name = names_of(domain)
        self.patterns = self._patterns()

    def _patterns(self) -> list:
        """(regex, predicate, arity), the most specific first.

        Specificity is how much of the template is literal words: `{0} on
        the table` has three and `{0} on {1}` has one, so *put the red block
        on the table* is not read as putting it on a block called `table`.
        """
        out = []
        body = r"(?:the |a |an )?([a-z][a-z0-9-]*)"
        kinds = "|".join(sorted(self.domain.kinds, key=len, reverse=True))
        if kinds:
            body += rf"(?:\s+(?:{kinds}))?"
        # A copula after a name is English, not a domain fact: `the keys
        # **are** at the office` against `the book **is** at the office`,
        # and neither is worth a line per verb in every domain.
        body += COPULA
        for predicate, templates in self.domain.readings().items():
            for template in templates:
                pattern = re.escape(" ".join(template.split()))
                arity = 0
                while re.escape("{%d}" % arity) in pattern:
                    pattern = pattern.replace(re.escape("{%d}" % arity),
                                              body, 1)
                    arity += 1
                literal = len(re.sub(r"\{\d+\}", "", template).split())
                out.append((re.compile(pattern), predicate, arity, literal))
        return [one[:3] for one in sorted(out, key=lambda one: -one[3])]

    def mentions(self, text: str) -> list:
        """(name, kind) for every thing named with its kind beside it.

        `a red block, a green block and a blue block on the table` states
        one fact and names three things, and a reader that only collected
        facts would put one block on the table and lose the others. A kind
        word is the domain's, so this is as general as the domain list is.
        """
        out = []
        for kind in self.domain.kinds:
            for found in re.finditer(
                    rf"\b([a-z][a-z0-9-]*)\s+{re.escape(kind)}\b", text):
                if found.group(1) not in NEVER:
                    out.append((found.group(1), kind))
            for found in re.finditer(
                    rf"\b{re.escape(kind)}\s+([a-z][a-z0-9-]*)\b", text):
                if found.group(1) not in NEVER:
                    out.append((found.group(1), kind))
        return out

    def facts_in(self, text: str) -> list:
        """Every fact the sentence states, left to right, without letting
        two readings overlap -- the first (most specific) wins its span."""
        taken: list = []
        found: list = []
        for pattern, predicate, arity in self.patterns:
            for match in pattern.finditer(text):
                span = match.span()
                if any(span[0] < end and start < span[1]
                       for start, end in taken):
                    continue
                args = [match.group(index + 1) for index in range(arity)]
                if any(one in NEVER for one in args):
                    continue
                taken.append(span)
                found.append((span[0], predicate + "".join(
                    f" {one}" for one in args)))
        return [fact for _, fact in sorted(found)]


#: How much an act is worth against v689's own, in the page's conflict set.
#: Above `generic` (which is the lowest) and below anything v689 reads as a
#: question it can answer, because *what is a dog* must not become a query
#: about a scene just because a scene is open.
WEIGHT = 40.0


class Scene:
    """Things, where they are, and what has been done about it."""

    def __init__(self, domain=None) -> None:
        #: None until a world is opened. **Nothing about a scene is read
        #: until then**, because the page is a conversation about what is
        #: true of the world before it is one about a table of blocks: `the
        #: dog is on the mat` parses perfectly well as `on dog mat`, and
        #: answering it as a scene instead of as something told would break
        #: the page for everything v687 to v690 do. So a world is opened by
        #: asking for one, and `open` is what every act is gated on.
        self.domain = domain
        # A declared domain is read through its own `say` lines; a world
        # with nothing declared about it brings its own reader
        # (`openworld.OpenReader`), because there are no templates to read
        # it by.
        self.reader = None
        if domain is not None:
            own = getattr(domain, "reader", None)
            self.reader = own() if own is not None else Reader(domain)
        self.objects: dict = dict(domain.always) if domain is not None else {}
        self.world = W.World(domain.begin(self.objects)
                             if domain is not None else ())
        self.last: acting.Attempt | None = None
        self.last_plan: list = []
        self.last_reasons: list = []
        #: what the last order asked for, and what the planner had to
        #: choose from, so the page can show the search and not only the
        #: answer
        self.wanted: list = []
        self.offered: int = 0
        self.offered_names: list = []
        #: each action of the last order, with the world just before it:
        #: what a report that one of them did not happen is checked against
        self.steps: list = []
        #: draws lessons from what happened when acting (`lessons.py`),
        #: where there is somewhere to keep them
        learned = getattr(domain, "learned", None)
        self.learner = None
        if learned is not None:
            # What kind a thing is, where the domain can say: a lesson can
            # then be about doors and survive a box that does not follow it.
            things = getattr(domain, "things", None)
            self.learner = lessons.Learner(
                learned,
                kind_of=(lambda name: things.kinds.get(name, name))
                if things is not None else None,
                is_a=getattr(domain, "is_a", None))
        #: what the last report taught, for the page to show
        self.taught: list = []
        #: an order that could not be planned, and what was asked about it:
        #: {goal, said, asked, kind, thing}. The goal is **suspended**, not
        #: dropped -- the impasse's way out is a question to the person, and
        #: the answer, when it comes, is a later turn (`resume`)
        self.pending: dict | None = None

    @property
    def open(self) -> bool:
        return self.domain is not None

    # -- the domain --------------------------------------------------------
    def use(self, name: str) -> None:
        self.__init__(DOMAINS[name])

    def kind_of(self, name: str, predicate: str, position: int,
                text: str) -> str:
        """What kind a newly named thing is.

        The fact it was named in usually settles it -- `in book shop` is a
        thing and then a place, because the domain's schemas say so -- and
        where the predicate is untyped, the kind word beside it in the
        sentence does. Neither knows what a block is.
        """
        kinds = self.domain.typing().get(predicate, ())
        if position < len(kinds) and kinds[position]:
            return kinds[position]
        for kind in self.domain.kinds:
            if re.search(rf"\b{re.escape(name)}\s+{re.escape(kind)}\b", text):
                return kind
            if re.search(rf"\b{re.escape(kind)}\s+{re.escape(name)}\b", text):
                return kind
        if getattr(self.domain, "things", None) is not None:
            # A world with nothing declared about it: a thing is what it was
            # called, and the store says what that is. Falling back on the
            # first kind here made a house a person.
            return ""
        return self.domain.kinds[0] if self.domain.kinds else ""

    def told(self, name: str, kind: str) -> None:
        """Tell the domain a thing exists, where it wants to know.

        A declared domain does not: its things are whatever fills its
        schemas. A world with nothing declared about it has to be told,
        because what it can do is worked out *from* the things
        (`openworld.Open.toward`).
        """
        note = getattr(self.domain, "note", None)
        if note is not None:
            note(name, kind)

    def introduce(self, facts: list, text: str) -> None:
        for fact in facts:
            parts = fact.split()
            for position, name in enumerate(parts[1:]):
                if name in self.objects:
                    continue
                self.objects[name] = self.kind_of(name, parts[0], position,
                                                  text)
                self.told(name, self.objects[name])
                self.world.facts = self.world.facts | frozenset(
                    one.replace("?x", name) for one in
                    self.domain.starts.get(self.objects[name], ()))
        self.world.facts = self.world.facts | self.domain.world

    # -- the acts ----------------------------------------------------------
    def tell(self, heard: Heard) -> str:
        # Things named with their kind but placed nowhere -- `a red block,
        # a green block and a blue block on the table` -- start wherever
        # the domain says a thing of that kind starts.
        # What a fact says a thing *is* -- `closed` in `the door is closed`
        # -- is not a thing, however generously `mentions` reads.
        stated = {fact.split()[0] for fact in heard.facts}
        for name, kind in self.reader.mentions(heard.said.lower()):
            if name not in self.objects and name not in stated:
                self.objects[name] = kind
                self.told(name, kind)
                self.world.facts = self.world.facts | frozenset(
                    one.replace("?x", name)
                    for one in self.domain.starts.get(kind, ()))
        self.introduce(heard.facts, heard.said.lower())
        for fact in heard.facts:
            self.put(fact)
        return f"all right: {self.look()}"

    def put(self, fact: str) -> None:
        """Assert a fact and keep the world consistent: whatever it
        contradicts gives way, and whatever that leaves loose goes back to
        however the domain says a thing starts.

        Editing a world by hand is fiddly in exactly the way a delete list
        makes unnecessary, which is the argument for a world being facts
        with actions over them rather than a picture.
        """
        parts = fact.split()
        subject = parts[1] if len(parts) > 1 else ""
        facts = set(self.world.facts)
        for one in list(facts):
            bits = one.split()
            if bits[0] not in self.domain.goalish or len(bits) < 2:
                continue
            if bits[1] == subject:
                # A thing is in one place: whatever was said before about
                # where this one is gives way. `goalish` is what the domain
                # calls a placement, which is the same list an order may
                # ask for, and that is not a coincidence.
                facts.discard(one)
            elif (len(bits) > 2 and len(parts) > 2 and bits[2] == parts[2]
                  and 2 in self.domain.taken.values()):
                # Something else was where this is going -- but only where
                # the domain says a target holds one thing. `taken clear 2`
                # is blocks saying that: whatever is on a block stops it
                # being clear, so one block sits on another. A kitchen holds
                # John and the cup at once, and applying this everywhere
                # took John out of the kitchen when the cup came in.
                facts.discard(one)
                facts |= {two.replace("?x", bits[1]) for two in
                          self.domain.starts.get(self.objects.get(bits[1]),
                                                 ())}
        # Whatever is known to exclude the new state stops holding. This
        # is what being taught an antonym buys: `open` arriving takes
        # `closed` away, and nothing here knows which words those are.
        learned = getattr(self.domain, "learned", None)
        if learned is not None and len(parts) == 2:
            for other in learned.excluded(parts[0]):
                facts.discard(f"{other} {subject}")
        facts.add(fact)
        self.world.facts = frozenset(self._settled(facts))

    def _settled(self, facts: set) -> set:
        """`clear`-like facts put right after a fact was asserted by hand.

        A predicate that is true exactly when nothing else claims the thing
        -- `clear` in blocks -- cannot be maintained by asserting one fact,
        so the domain says which predicate it is and which argument of a
        goalish fact takes it away (`taken clear 2`). Nothing here knows
        that the word is `clear`.
        """
        for predicate, position in self.domain.taken.items():
            claimed = {one.split()[position] for one in facts
                       if one.split()[0] in self.domain.goalish
                       and len(one.split()) > position}
            for name in self.objects:
                fact = f"{predicate} {name}"
                if name in claimed:
                    facts.discard(fact)
                else:
                    facts.add(fact)
        return facts

    def want(self, heard: Heard) -> str:
        goal = [one for one in (heard.wants or heard.facts)
                if one.split()[0] in self.domain.goalish]
        if not goal:
            return ("I understood that as a scene rather than as something "
                    "to do -- tell me where something should end up")
        unplaced = self.unplaced(goal)
        self.introduce(goal, heard.said.lower())
        if unplaced:
            # Moving a thing nobody has placed: VerbNet has readings of
            # `leave` and `send` with no precondition at all, and a plan
            # built on one says it moved a book it never found. Where the
            # thing is, is the question.
            self.wanted = list(goal)
            self.last_plan, self.steps = [], []
            self.pending = self._asking(
                "where", f"at {unplaced} ?", unplaced,
                f"where is {self.domain.the(unplaced)}?")
            self.pending.update(goal=list(goal), said=heard.said)
            return ("I will need to know where it is first. "
                    + self.pending["question"])
        # A world with nothing declared about it works out what can be done
        # only once it knows what is wanted: 7,796 operators over the things
        # in a conversation is not a search space (`verbs.useful`).
        toward = getattr(self.domain, "toward", None)
        if toward is not None:
            toward(goal)
        self.wanted = list(goal)
        offered = self.domain.ground(self.objects)
        self.offered = len(offered)
        self.offered_names = [one.name for one in offered]
        problem = W.Problem("what you asked for",
                            frozenset(self.world.facts),
                            frozenset(goal),
                            tuple(offered), dict(self.objects))
        report = acting.Attempt(name=problem.name)
        before = len(self.world.did)
        start = frozenset(self.world.facts)
        acting.agent(problem, self.world, None, report,
                     learner=self.learner).run(
            Working(goal=f"do: {heard.said}"))
        self.last = report
        self.last_reasons = (reasons(report.search.trace)
                             if report.search.trace is not None else [])
        self.last_plan = list(self.world.did[before:])
        self.steps, facts = [], start
        for one in self.last_plan:
            self.steps.append((one, facts))
            facts = one.on(facts)
        agents = getattr(self.domain, "agents", None)
        if agents is not None:
            # Whoever filled the role VerbNet restricts to the animate is
            # the doer, and narration says "I" for them rather than their
            # name again.
            # Only a *name* can be the doer here. The store's categories
            # are the union over every sense a word has, and some sense of
            # `cup` is animate enough to pass -- which made the cup the one
            # doing the carrying, and the narration say "I left the kitchen".
            names = getattr(self.domain, "names", set())
            for one in self.last_plan:
                parts = one.name.split()
                if (len(parts) > 2 and parts[1] in names
                        and "animate" in self.domain.things.categories(
                            parts[1])):
                    agents.add(parts[1])
        if not report.solved:
            asked = self.missing(offered)
            if asked is not None:
                self.pending = dict(asked, goal=list(goal), said=heard.said)
                return ("I could not see a way to do that yet"
                        + self.stopped(offered) + ". " + asked["question"])
            self.pending = None
            return ("I could not see a way to do that"
                    + self.stopped(offered) + ". As it stands, "
                    + self.look())
        self.pending = None
        return self.story(report)

    # -- asking for what is missing ----------------------------------------
    def missing(self, offered: list) -> dict | None:
        """What stands between the world and the goal that nothing offered
        can bring about, as a question: {question, asked, kind, thing}.

        Worked out the way a planner's heuristic is: first what is
        reachable at all if nothing were ever undone (a relaxed plan), and
        if the goal is not, back from the goal through whichever way of
        getting each fact leaves least unreached, to a fact nothing brings
        about. That fact is what is missing, and **in a world nobody
        declared, a fact not said is not known to be false** -- so a state
        is asked about (`is the park lit?`) before it is called impossible,
        a thing nobody placed is asked after (`where is the book?`), and a
        blocker nothing undoes is put as a question of how.

        None when the goal is reachable on that relaxed reading: then what
        failed was the search, and a question would be asking the person
        to do the planner's job.
        """
        compiled, facts, _ = acting.negated(offered, self.world.facts)
        reach = set(facts)
        grown = True
        while grown:
            grown = False
            for one in compiled:
                if one.needs <= reach and not one.adds <= reach:
                    reach |= one.adds
                    grown = True
        goal = [one for one in self.wanted if one not in self.world.facts]
        if not goal or set(goal) <= reach:
            return None
        placed = {one.split()[1] for one in self.world.facts
                  if one.split()[0] == "at" and len(one.split()) == 3}
        for literal in goal:
            parts = literal.split()
            if parts[0] == "at" and len(parts) == 3 and \
                    parts[1] not in placed:
                return self._asking("where", literal, parts[1],
                                    f"where is {self.domain.the(parts[1])}?")
        leaves: list = []
        seen: set = set()

        def back(literal: str, depth: int) -> None:
            if literal in reach or literal in seen or depth > 6:
                return
            seen.add(literal)
            ways = [one for one in compiled if literal in one.adds]
            if not ways:
                leaves.append((depth, literal))
                return
            best = min(ways, key=lambda one: (len(one.needs - reach),
                                              one.name))
            for need in sorted(best.needs - reach):
                back(need, depth + 1)

        for literal in sorted(set(goal) - reach):
            back(literal, 0)
        if not leaves:
            return None
        literal = sorted(leaves, key=lambda one: (-one[0], one[1]))[0][1]
        if literal.startswith(acting.NOT):
            fact = literal[len(acting.NOT):]
            thing = fact.split()[1] if len(fact.split()) > 1 else ""
            words = self.domain.in_words(fact)
            return self._asking("how", literal, thing,
                                f"{words[:1].upper()}{words[1:]} -- what "
                                f"would change that?")
        parts = literal.split()
        thing = parts[1] if len(parts) > 1 else ""
        if len(parts) == 2 and not getattr(self.domain, "a_doing",
                                           lambda _: False)(parts[0]):
            return self._asking("whether", literal, thing,
                                f"is {self.domain.the(thing)} {parts[0]}?")
        words = self.domain.in_words(literal)
        return self._asking("how", literal, thing,
                            f"how would I make it so that {words}?")

    def unplaced(self, goal: list) -> str:
        """A thing an order moves that nothing says is anywhere, in a world
        with nothing declared about it -- or empty. A declared domain
        places everything it has, and an agent is where it is."""
        if getattr(self.domain, "things", None) is None:
            return ""
        placed = {one.split()[1] for one in self.world.facts
                  if len(one.split()) == 3}
        agents = getattr(self.domain, "agents", set())
        for literal in goal:
            parts = literal.split()
            if (parts[0] == "at" and len(parts) == 3
                    and parts[1] not in placed and parts[1] not in agents
                    and parts[1] not in getattr(self.domain, "names",
                                                set())):
                return parts[1]
        return ""

    @staticmethod
    def _asking(kind: str, literal: str, thing: str, question: str) -> dict:
        return {"kind": kind, "asked": literal, "thing": thing,
                "question": question[:1].upper() + question[1:]}

    def answers(self, plain: str, facts: list, taught=()) -> bool:
        """Whether an utterance is the answer to what was asked: a yes or
        a no to a question of whether, anything said about the thing asked
        after, something taught, or being told to drop it."""
        pending = self.pending
        if pending is None:
            return False
        if DROP.search(plain) or taught:
            return True
        if pending["kind"] == "whether" and (YES.match(plain)
                                             or NO.match(plain)):
            return True
        return any(pending["thing"] in fact.split()[1:] for fact in facts)

    def resume(self, heard: Heard, taught: str = "") -> str:
        """The answer came: take it in, and try the suspended order again.
        Trying again can ask again -- the next thing missing -- which is how
        a plan is put together over several turns."""
        pending = self.pending
        if pending is None:
            return "I was not waiting on anything"
        plain = " ".join(heard.said.lower().split())
        if DROP.search(plain):
            self.pending = None
            return "all right, I will leave it"
        if pending["kind"] == "whether" and NO.match(plain):
            # Said to be false: now it is a thing to bring about, and
            # asking how is the question left.
            words = self.domain.in_words(pending["asked"])
            self.pending = dict(pending, kind="how", question=(
                f"Then how would I make it so that {words}?"))
            return self.pending["question"]
        if pending["kind"] == "whether" and YES.match(plain):
            self.introduce([pending["asked"]], plain)
            self.put(pending["asked"])
        elif heard.facts:
            self.tell(heard)
        again = Heard(said=pending["said"], wants=list(pending["goal"]),
                      facts=list(pending["goal"]))
        done = self.want(again)
        return (f"{taught}; " if taught else "") + ("then " + done
                                                    if self.last_plan
                                                    else done)

    def stopped(self, offered: list) -> str:
        """What stands in the way, where something learned says: an action
        that would have brought a wanted fact about is blocked by a state
        that holds, and nothing on offer undoes it."""
        said = []
        for action in offered:
            if not set(action.adds) & set(self.wanted):
                continue
            for fact in sorted(getattr(action, "forbids", ())):
                if fact in self.world.facts:
                    words = self.domain.in_words(fact)
                    if words not in said:
                        said.append(words)
        if not said:
            return ""
        return (": " + " and ".join(said) + ", and I know of nothing I can "
                "do about that")

    # -- being told it did not work ----------------------------------------
    def reported(self, plain: str) -> dict | None:
        """Whether an utterance says something just done did not happen,
        and if so, which step and what was said with it.

        Three ways English says it: a state *still* holds that the action
        should have ended (`the door is still closed`), what it should have
        brought about is denied (`the door did not open`, `the door is not
        open`), or plainly (`that did not work`). `it` is the thing last
        acted on. Nothing here knows a door or a lock.
        """
        if not self.steps:
            return None
        last = self.steps[-1][0].name.split()
        thing = last[1] if len(last) > 1 else ""
        text = re.sub(r"\bit\b", thing, plain) if thing else plain
        things = {one for action, _ in self.steps
                  for one in action.name.split()[1:]}
        denied = []
        for found in DENIED.finditer(text):
            if found.group(1) in things:
                denied.append(f"{found.group(2)} {found.group(1)}")
        rest = DENIED.sub(" ", text)
        still = bool(STILL.search(rest))
        rest = STILL.sub(" ", rest)
        failed = bool(FAILED.search(text))
        if not (denied or still or failed):
            return None
        facts = [one for one in self.reader.facts_in(rest)
                 if set(one.split()[1:]) & things]
        if not (denied or facts or failed):
            return None
        step = None
        for index, (action, _) in enumerate(self.steps):
            if set(action.adds) & set(denied):
                step = index
                break
        if step is None and still:
            about = {one for fact in facts for one in fact.split()[1:]}
            step = next((index for index, (action, _) in
                         enumerate(self.steps)
                         if set(action.name.split()[1:]) & about), None)
        if step is None and (failed or denied):
            step = len(self.steps) - 1
        if step is None:
            return None
        return {"step": step, "facts": facts, "denied": denied}

    def failed(self, heard: Heard) -> str:
        """Something done did not happen: put the world back to before it,
        take in what was said, and learn what can be learned.

        The world goes back to just before the step that failed, because
        everything after it was done in a world that was not so. What the
        person said with it is taken in, and whatever was not already
        known then is **revealed** -- *it is locked* -- and handed to the
        learner as the reason (`lessons.Learner.failed`).
        """
        plain = " ".join(heard.said.lower().replace(",", " , ").split())
        found = self.reported(plain)
        if found is None:
            return "I did not catch what did not work"
        action, before = self.steps[found["step"]]
        for later, _ in self.steps[found["step"]:]:
            # Done and believed done, and it was not: the successes they
            # were recorded as go, or they would count both ways.
            row = (self.learner.rows.pop(later.name, None)
                   if self.learner is not None else None)
            if row is not None:
                self.learner.learned.untry(row)
        self.world.facts = frozenset(before)
        self.introduce(found["facts"], plain)
        revealed = [one for one in found["facts"] if one not in before]
        for fact in found["facts"]:
            self.put(fact)
        self.taught = []
        if self.learner is not None:
            gap = acting.Gap(action, frozenset(action.adds) - before,
                             frozenset(revealed), found["step"],
                             frozenset(before), True)
            self.taught = self.learner.failed(gap, self.world.facts,
                                              revealed, heard.said)
        self.last_plan = [one for one, _ in self.steps[:found["step"]]]
        self.steps = self.steps[:found["step"]]
        said = f"I see -- I could not {self.domain.doing(action.name)}"
        if revealed:
            said += ", because " + " and ".join(
                self.domain.in_words(one) for one in revealed)
        lessons_said = [self.lesson(one) for one in self.taught]
        lessons_said = [one for one in lessons_said if one]
        if lessons_said:
            said += ". I did not know " + "; ".join(lessons_said) + \
                ", and I will remember it"
        elif self.learner is not None and not revealed:
            said += ". I do not know why yet"
        return said + ". As it stands, " + self.look()

    def lesson(self, one) -> str:
        """A lesson in words, about any thing and not the one it was
        learned from."""
        parts = one.literal.split()
        thing = "it" if len(parts) == 2 else "something"
        state = " ".join(word for word in parts if not word.startswith("?"))
        if one.kind == "blocks" and len(parts) == 2:
            return f"that nothing {state} can be {_participle(one.verb)}"
        if one.kind == "requires" and len(parts) == 2:
            return (f"that a thing has to be {state} before it can be "
                    f"{_participle(one.verb)}")
        if one.kind in ("blocks", "requires"):
            return f"what it takes to {one.verb} {thing}"
        return ""

    def how(self, heard: Heard) -> str:
        """What it would take: planned, and not done.

        `what steps are required to make a pig fly` asks for a plan, not
        for a pig in the air, so the plan is made in a copy of the world --
        the same planner, the same actions, and nothing in the scene moves.
        Whoever would do it is the one asking, when nobody in the scene can
        (`you`), and what the plan needs that nobody said is there -- the
        plane -- is said as needed rather than assumed.
        """
        doing = getattr(self.domain, "a_doing", None)
        goal = [one for one in (heard.wants or heard.facts)
                if one.split()[0] in self.domain.goalish
                or (doing is not None and doing(one.split()[0]))]
        if not goal:
            return ("I did not catch what it should come to -- say what "
                    "should be true, or what something should do")
        self.introduce(goal, heard.said.lower())
        things = getattr(self.domain, "things", None)
        if things is not None:
            # Whoever asks what it would take is who would do it -- always,
            # not only when nobody else here could: a pig named a turn ago
            # is animate, and was otherwise the one loading the piano.
            self.told("you", "person")
            self.domain.names.add("you")
            self.domain.agents.add("you")
        toward = getattr(self.domain, "toward", None)
        if toward is not None:
            toward(goal)
        self.wanted = list(goal)
        offered = self.domain.ground(self.objects)
        self.offered = len(offered)
        self.offered_names = [one.name for one in offered]
        objects = dict(things.kinds) if things is not None else {}
        objects.update(self.objects)
        problem = W.Problem("what it would take",
                            frozenset(self.world.facts), frozenset(goal),
                            tuple(offered), objects)
        report = acting.Attempt(name=problem.name)
        pretend = W.Imagined(self.world.facts)
        acting.agent(problem, pretend, None, report).run(
            Working(goal=f"how: {heard.said}"))
        self.last = report
        self.last_reasons = (reasons(report.search.trace)
                             if report.search.trace is not None else [])
        self.last_plan = list(pretend.did)
        return self._how_said(goal, report)

    def _how_said(self, goal: list, report) -> str:
        """The steps, and why they are the steps: what the thing cannot do
        itself, and what was seen that shows the way it can."""
        domain = self.domain
        wanted = " and ".join(domain.in_words(one) for one in goal)
        carried = getattr(domain, "carried", {})
        own = getattr(domain, "own", set())
        why = []
        for literal in goal:
            verb, *rest = literal.split()
            if len(rest) != 1 or literal in own or not (
                    getattr(domain, "a_doing", None)
                    and domain.a_doing(verb)):
                continue
            thing = domain.the(rest[0])
            # The carrier the plan used, before any it could have.
            used = {one.name for one in self.last_plan}
            ways = [(said, fits) for name, (one, _, doing, said, fits)
                    in sorted(carried.items(),
                              key=lambda item: item[0] not in used)
                    if one == rest[0] and doing == verb]
            if ways:
                said, fits = ways[0]
                seen = " and ".join(f"“{one}”" for one in said.split(" / "))
                if fits and not said:
                    why.append(f"{thing} cannot {verb} by itself, but {fits}"
                               f": what is aboard it goes where it goes")
                elif fits:
                    why.append(f"{thing} cannot {verb} by itself, but {fits}"
                               f", so it fits, and I have seen what is "
                               f"carried on one {verb} with it: {seen}")
                else:
                    why.append(f"{thing} cannot {verb} by itself, but I "
                               f"have seen it done by being carried: {seen}")
            elif getattr(domain, "refused", None):
                why.append(f"{thing} cannot {verb} by itself, and nothing "
                           f"that can will carry it: " + "; ".join(
                               dict.fromkeys(domain.refused)))
            else:
                why.append(f"{thing} cannot {verb} by itself, and I have "
                           f"not seen any other way for it to")
        if not report.solved:
            said = f"I could not see a way for {wanted}"
        else:
            steps = [domain.doing(one.name) for one in self.last_plan]
            said = (f"for {wanted}: " + "; ".join(
                f"{index}. {step}" for index, step in enumerate(steps, 1)))
            needed = sorted(getattr(domain, "supposed", set())
                            & {part for one in self.last_plan
                               for part in one.name.split()[1:]})
            if needed:
                said += (". You would need " + " and ".join(
                    f"{'an' if one[:1] in 'aeiou' else 'a'} {one}"
                    for one in needed))
        said += (". " + "; ".join(why) if why else "")
        # Each sentence opened as one, outside what is quoted.
        return re.sub(r"(^|\. )([a-z])",
                      lambda found: found.group(1) + found.group(2).upper(),
                      said)

    def planning(self) -> dict:
        """The last plan, as the page shows it.

        Small and curated on purpose. A turn's full `executed` is every run
        of every executive -- thirty of them for an ordinary question --
        and putting that on every turn would bloat the archive to show
        nothing most of the time. This is the planner's own account: what
        was wanted, what it had to choose from, the goal stack it pushed,
        and what it did.
        """
        report = self.last
        if report is None:
            return {}
        search = report.search
        return {
            "goal": [self.domain.in_words(one) for one in self.wanted],
            "offered": self.offered,
            "verbs": len({one.split()[0] for one in self.offered_names}),
            "plan": [self.domain.phrase(one.name) for one in self.last_plan],
            "actions": [one.name for one in self.last_plan],
            "solved": report.solved,
            "fired": search.fired, "subgoals": search.subgoals,
            "deep": search.depth, "plans": report.plans,
            "stack": _stack(search.trace) if search.trace is not None
            else None,
            "surprises": [{"action": str(gap.action),
                           "expected": sorted(gap.missing),
                           "found": sorted(gap.extra)}
                          for gap in report.gaps],
        }

    def meddle(self, heard: Heard) -> str:
        if not heard.facts:
            return "I did not catch what changed"
        self.introduce(heard.facts, heard.said.lower())
        learnt = self.corrected(heard)
        for fact in heard.facts:
            self.put(fact)
        said = f"I see -- {self.look()}"
        if learnt:
            said += (". I did not know " + " and ".join(
                f"a thing cannot be {one} and {other}"
                for one, other in learnt) + "; I do now")
        return said

    def corrected(self, heard: Heard) -> list:
        """What a correction teaches about which states exclude each other.

        The scene holds `closed door`; the person says *the door is open
        now*. Two states of one thing, one right after the other, the
        second marked as a change: that is what incompatibility looks like
        from the inside, and it is the only evidence for it there is. Two
        facts holding at once is evidence they *do not* exclude and never
        evidence that they do, so nothing is learned from co-occurrence.

        Both have to be adjectives, asked of WordNet, because `mammal
        whale` and `closed door` are the same shape and only one of them is
        a state something can stop being in.
        """
        learned = getattr(self.domain, "learned", None)
        if learned is None:
            return []
        from research.v689.change import adjective
        found = []
        for fact in heard.facts:
            parts = fact.split()
            if len(parts) != 2 or not adjective(parts[0]):
                continue
            for one in sorted(self.world.facts):
                was = one.split()
                if (len(was) == 2 and was[1] == parts[1]
                        and was[0] != parts[0] and adjective(was[0])
                        and learned.exclude(parts[0], was[0], heard.said)):
                    found.append((parts[0], was[0]))
        return found

    def look(self) -> str:
        """The scene, in the domain's own words.

        Every fact the domain calls worth telling, said with its own `say`
        line. An earlier version chained blocks into towers -- *the red
        block is on the green block, which is on the table* -- and that
        read better and was blocks. This is what any domain can say.
        """
        said = [self.domain.in_words(one) for one in sorted(self.world.facts)
                if one.split()[0] in self.domain.tellable()]
        return "; ".join(said) or "there is nothing here yet"

    def where(self, heard: Heard) -> str:
        if not heard.names:
            return "which one?"
        name = heard.names[0]
        about = [self.domain.in_words(one) for one in sorted(self.world.facts)
                 if len(one.split()) > 1 and one.split()[1] == name
                 and one.split()[0] in self.domain.tellable()]
        return "; ".join(about) or f"I know nothing about the {name}"

    def upon(self, heard: Heard) -> str:
        if not heard.names:
            return "on which one?"
        name = heard.names[-1]
        about = [self.domain.in_words(one) for one in sorted(self.world.facts)
                 if len(one.split()) > 2 and one.split()[2] == name]
        if about:
            return "; ".join(about)
        # Said in the domain's own words rather than glued together here:
        # `clear red` is exactly *nothing is on the red block* in blocks,
        # and whatever the domain calls it elsewhere.
        for predicate in self.domain.taken:
            return self.domain.in_words(f"{predicate} {name}")
        return f"nothing is on the {name}"

    def why(self, heard: Heard) -> str:
        """What the last thing done was for.

        The planner's means-ends subgoals are already named `achieve clear
        green for stack green blue`, so this reads the goal stack back
        rather than composing an explanation. An explanation invented apart
        from the search would be a story about the agent; this one is the
        search.
        """
        if not self.last_plan:
            return "I have not done anything yet"
        for action in self.last_plan:
            if heard.names and not set(action.name.split()[1:]) & set(
                    heard.names):
                continue
            for name, goal in self.last_reasons:
                if name == action.name and goal.startswith("achieve "):
                    needed, _, for_what = goal[len("achieve "):].partition(
                        " for ")
                    words = " and ".join(self.domain.in_words(one)
                                         for one in needed.split(", "))
                    toward = ("do what you asked" if for_what == "done"
                              else self.domain.doing(for_what))
                    return (f"I {self.domain.phrase(action.name)} because "
                            f"I could not {toward} until {words}")
        return (f"{self.narrate(self.last_plan)} -- that was the shortest "
                f"way I found to what you asked for")

    # -- saying it ---------------------------------------------------------
    def narrate(self, actions: list) -> str:
        if not actions:
            return "nothing needed doing"
        parts = [self.domain.phrase(one.name) for one in actions]
        parts = self.domain.joined(parts, actions)
        if len(parts) == 1:
            return f"I {parts[0]}"
        return f"I {', '.join(parts[:-1])}, then {parts[-1]}"

    def story(self, report: acting.Attempt) -> str:
        """What was done, broken where the world did something unexpected --
        after a surprise the agent covers ground it has already covered, and
        one flat list reads as a stutter."""
        cuts = [gap.after for gap in report.gaps]
        pieces, at = [], 0
        for cut in cuts + [len(self.last_plan)]:
            if cut > at:
                pieces.append(self.narrate(self.last_plan[at:cut]))
            at = cut
        if not pieces:
            return "nothing needed doing"
        said = pieces[0]
        for gap, piece in zip(report.gaps, pieces[1:]):
            said += (f". Then something moved while I was working: I "
                     f"expected {self._listed(gap.missing) or 'no change'}, "
                     f"and found {self._listed(gap.extra) or 'nothing'}. So "
                     f"I planned again, and {piece[2:]}")
        return said

    def _listed(self, facts) -> str:
        return ", ".join(self.domain.in_words(one) for one in sorted(facts))


#: The most goals shown of one plan's stack. A plan in the open world can
#: push a hundred subgoals, and a page that drew all of them would be a
#: wall; the count is always right even where the tree is cut.
SHOWN = 60


def _stack(trace, budget=None) -> dict:
    """A planner's run as a tree of goals: what each one wanted, what fired
    in it, and what it pushed."""
    if budget is None:
        budget = [SHOWN]
    budget[0] -= 1
    out = {"goal": trace.goal,
           "fired": [{"operator": one.operator, "outcome": one.outcome,
                      "rule": one.rule,
                      "candidates": len(one.candidates or ())}
                     for one in trace.fired],
           "answered": trace.answered_by,
           "subgoals": [], "more": 0}
    for inner in trace.subgoals:
        if budget[0] <= 0:
            out["more"] += 1
            continue
        out["subgoals"].append(_stack(inner, budget))
    return out


def reasons(trace) -> list:
    """(operator, the goal it was fired for), for everything that did
    something, subgoals included."""
    out = []
    for step in trace.fired:
        if step.outcome != DECLINED:
            out.append((step.operator, trace.goal))
    for inner in trace.subgoals:
        out.extend(reasons(inner))
    return out
