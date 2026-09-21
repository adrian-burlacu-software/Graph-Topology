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
    changing    use the errands world; what worlds do you have
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from research.v687.executive import (ANSWERED, DECLINED, Executive, Operator,
                                     Working)
from research.v691 import acting, world as W
from research.v691.domains import DOMAINS

#: Words that are never an object's name, however a pattern falls.
NEVER = frozenset("""a an the and or is are was were be do does did to of on
in at it its this that there here what where why who how i you me my your
please now actually then so put move place get take bring make build set
tell say left right up down with from into onto for all some any thing
things world worlds use using""".split())


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
        for name, kind in self.reader.mentions(heard.said.lower()):
            if name not in self.objects:
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
            elif len(bits) > 2 and len(parts) > 2 and bits[2] == parts[2]:
                # Something else was where this is going.
                facts.discard(one)
                facts |= {two.replace("?x", bits[1]) for two in
                          self.domain.starts.get(self.objects.get(bits[1]),
                                                 ())}
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
        self.introduce(goal, heard.said.lower())
        # A world with nothing declared about it works out what can be done
        # only once it knows what is wanted: 7,796 operators over the things
        # in a conversation is not a search space (`verbs.useful`).
        toward = getattr(self.domain, "toward", None)
        if toward is not None:
            toward(goal)
        problem = W.Problem("what you asked for", frozenset(self.world.facts),
                            frozenset(goal),
                            tuple(self.domain.ground(self.objects)),
                            dict(self.objects))
        report = acting.Attempt(name=problem.name)
        before = len(self.world.did)
        acting.agent(problem, self.world, None, report).run(
            Working(goal=f"do: {heard.said}"))
        self.last = report
        self.last_reasons = (reasons(report.search.trace)
                             if report.search.trace is not None else [])
        self.last_plan = list(self.world.did[before:])
        if not report.solved:
            return ("I could not see a way to do that. As it stands, "
                    + self.look())
        return self.story(report)

    def meddle(self, heard: Heard) -> str:
        if not heard.facts:
            return "I did not catch what changed"
        self.introduce(heard.facts, heard.said.lower())
        for fact in heard.facts:
            self.put(fact)
        return f"I see -- {self.look()}"

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
