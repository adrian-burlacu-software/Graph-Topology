"""A domain is data: what there can be, what can be done, and how to say it.

v691a's blocks world was written in Python, and that hid the question it was
supposed to answer. `world.blocks()` built ground actions with a loop, the
planner's two utility signals mentioned `stack` and `clear` by name, and the
conversation knew what a colour was -- so "the executive can plan" was a
claim about one domain that happened to be the one everything was tuned on.

Here a domain is a **string**, and everything else reads it:

    action stack ?x:block ?y:block
      needs held ?x, clear ?y
      adds  empty, clear ?x, on ?x ?y
      dels  held ?x, clear ?y

    say on ?x ?y   the {0} block is on the {1} block

One text gives the planner its actions, the reader its phrasings, and the
narrator its words, because they are three views of the same thing. Adding a
domain is writing one of these; nothing in `acting.py`, `world.py` or
`talking.py` mentions a block.

## The lines

    kind <name>            a kind of thing the domain has
    start <kind> ?x        what holds when a thing of that kind appears
    world <facts>          what holds before anything appears
    object <name> <kind>   a thing that is always there
    action <name> ?a:k ?b:k
      needs / adds / dels  comma-separated literals over the parameters
    say <literal>  <template>   how a fact reads, `{0}` per parameter
    do <action>    <template>   how an action reads, in the past tense
    doing <action> <template>   the same, in the present, for `why`
    goalish <predicate>    a predicate a person would ask *for* -- what an
                           order is allowed to mean (`on`, not `clear`)
    reads <literal> <template>  another way that fact can be *said to* it
    tell <predicates>      what is worth reporting when asked what there is
    taken <predicate> <n>  `clear`-like: false for whatever is argument n
                           of a goalish fact
    join <a> <b> <template>     two actions that are one thing said

`say` templates are read backwards as well as forwards: the same line that
turns `on red green` into *the red block is on the green block* is what
recognises that sentence and turns it back into the fact (`Reading`). One
template, so they cannot drift apart.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

from research.v691.world import Action

#: A parameter in a schema: `?x` or `?x:block`.
PARAM = re.compile(r"\?([a-z][a-z0-9]*)(?::([a-z][a-z0-9-]*))?")
#: What an object may be called in a sentence.
NAME = r"[a-z][a-z0-9-]*"


@dataclass(frozen=True)
class Schema:
    """An action with variables in it, and the types they take."""

    name: str
    params: tuple
    types: tuple
    needs: tuple
    adds: tuple
    deletes: tuple

    def ground(self, binding: dict) -> Action:
        def fill(literal: str) -> str:
            return " ".join(binding.get(word[1:], word) if
                            word.startswith("?") else word
                            for word in literal.split())

        name = " ".join([self.name] + [binding[one] for one in self.params])
        return Action(name,
                      frozenset(fill(one) for one in self.needs),
                      frozenset(fill(one) for one in self.adds),
                      frozenset(fill(one) for one in self.deletes))


@dataclass
class Domain:
    """Everything one world is, read from one text."""

    name: str
    kinds: tuple = ()
    schemas: tuple = ()
    #: what holds as soon as a thing of a kind exists
    starts: dict = field(default_factory=dict)
    #: what holds before anything does
    world: frozenset = frozenset()
    #: things that are always there, as name -> kind: `home` in `errands`,
    #: because an errand has to start somewhere and nobody says so
    always: dict = field(default_factory=dict)
    #: predicate -> (arity, template)
    says: dict = field(default_factory=dict)
    #: action name -> template, in the past tense
    does: dict = field(default_factory=dict)
    #: action name -> template, in the present: what `why` says it could do
    doings: dict = field(default_factory=dict)
    #: the predicates an order may ask for
    goalish: tuple = ()
    #: predicate -> extra ways it can be said, for reading only
    reads: dict = field(default_factory=dict)
    #: the predicates worth reporting when asked what there is
    tells: tuple = ()
    #: (first action, second action) -> one phrase for the pair
    joins: dict = field(default_factory=dict)
    #: predicate -> which argument of a goalish fact stops it being true
    taken: dict = field(default_factory=dict)
    source: str = ""

    def ground(self, objects: dict) -> list:
        """Every action over these objects, as `name -> kind`.

        Grounding the whole domain up front is what lets an action be an
        `Operator` with no binding at plan time (`acting.operator_of`). It
        is also why nothing here is a scaling claim: the count is the
        product of the typed object counts, per schema.
        """
        out = []
        for schema in self.schemas:
            choices = [[name for name, kind in objects.items()
                        if kind == wanted or wanted is None]
                       for wanted in schema.types]
            for picked in itertools.product(*choices):
                if len(set(picked)) != len(picked):
                    continue
                out.append(schema.ground(dict(zip(schema.params, picked))))
        return out

    def begin(self, objects: dict) -> frozenset:
        """The facts a set of objects starts with, before anything is said
        about where they are."""
        facts = set(self.world)
        for name, kind in objects.items():
            for literal in self.starts.get(kind, ()):
                facts.add(literal.replace("?x", name))
        return frozenset(facts)

    # -- saying ------------------------------------------------------------
    def in_words(self, fact: str) -> str:
        parts = fact.split()
        arity, template = self.says.get(parts[0], (0, ""))
        if not template:
            return fact
        return template.format(*parts[1:])

    def phrase(self, action: str) -> str:
        """What was done, in the past tense: `picked up the red block`."""
        parts = action.split()
        template = self.does.get(parts[0], "")
        return template.format(*parts[1:]) if template else action

    def doing(self, action: str) -> str:
        """What is to be done, after `could` or `had to`: `pick up the red
        block`. A `doing` line where the domain has one, and otherwise the
        past tense, which reads badly and is at least true."""
        parts = action.split()
        template = self.doings.get(parts[0])
        return (template.format(*parts[1:]) if template
                else self.phrase(action))

    def typing(self) -> dict:
        """predicate -> the kind of each of its arguments.

        Nowhere is this written down, and it does not need to be: the action
        schemas already say that `fetch ?what:thing ?where:place` needs
        `in ?what ?where`, so `in` is a thing and a place. Reading it off
        the schemas means a domain cannot declare its types twice and get
        them different.
        """
        out: dict = {}
        for schema in self.schemas:
            types = dict(zip(schema.params, schema.types))
            for literal in schema.needs + schema.adds + schema.deletes:
                parts = literal.split()
                kinds = tuple(types.get(one[1:]) if one.startswith("?")
                              else None for one in parts[1:])
                known = out.get(parts[0])
                if known is None or (None in known and None not in kinds):
                    out[parts[0]] = kinds
        return out

    def tellable(self) -> tuple:
        """What is worth saying when asked what there is: everything by
        default, and whatever `tell` names where a domain has facts that are
        bookkeeping rather than news."""
        return self.tells or tuple(self.says)

    def joined(self, parts: list, actions: list) -> list:
        """Actions said in pairs where the domain says they belong in one.

        Nobody says *I picked up the red block, then I put the red block on
        the green block*, and which pairs read that way is a fact about the
        domain, so it is a `join` line and not a rule here.
        """
        out, index = [], 0
        while index < len(actions):
            first = actions[index].name.split()
            second = (actions[index + 1].name.split()
                      if index + 1 < len(actions) else None)
            key = (first[0], second[0]) if second else None
            if key in self.joins and first[1:2] == second[1:2]:
                out.append(self.joins[key].format(
                    *(first[1:] + second[1:])))
                index += 2
            else:
                out.append(parts[index])
                index += 1
        return out

    # -- reading -----------------------------------------------------------
    def readings(self) -> dict:
        """predicate -> every template it can be recognised by: the `say`
        line it is written with, and any `reads` lines beside it.

        The same template both ways is the point -- what it understands and
        what it says cannot drift apart -- and `reads` is only for the
        phrasings a person uses that a statement does not, such as *put the
        red block on the green block* against *the red block **is** on the
        green block*.
        """
        out = {}
        for predicate, (_, template) in self.says.items():
            out[predicate] = [template]
        for predicate, extra in self.reads.items():
            out.setdefault(predicate, []).extend(extra)
        return out


def parse(text: str) -> Domain:
    """A domain from its text. Unknown lines raise rather than being
    skipped: a domain that silently lost an action would fail as a planning
    result, which is the most expensive way to find a typo."""
    domain = Domain(name="")
    kinds: list = []
    schemas: list = []
    goalish: list = []
    current: dict | None = None

    def close():
        if current is not None:
            schemas.append(Schema(current["name"], tuple(current["params"]),
                                  tuple(current["types"]),
                                  tuple(current.get("needs", ())),
                                  tuple(current.get("adds", ())),
                                  tuple(current.get("dels", ()))))

    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        head, _, rest = line.partition(" ")
        rest = rest.strip()
        if head == "domain":
            domain.name = rest
        elif head == "kind":
            kinds.append(rest)
        elif head == "object":
            name, _, kind = rest.partition(" ")
            domain.always[name] = kind.strip()
        elif head == "world":
            domain.world = frozenset(_literals(rest))
        elif head == "goalish":
            goalish += rest.split()
        elif head == "start":
            kind, _, body = rest.partition(" ")
            domain.starts[kind] = tuple(_literals(body))
        elif head == "say":
            predicate, _, template = rest.partition(" ")
            template = template.strip()
            arity = len(re.findall(r"\{\d+\}", template))
            domain.says[predicate] = (arity, template)
        elif head == "doing":
            name, _, template = rest.partition(" ")
            domain.doings[name] = template.strip()
        elif head == "reads":
            predicate, _, template = rest.partition(" ")
            domain.reads.setdefault(predicate, []).append(template.strip())
        elif head == "tell":
            domain.tells = tuple(rest.split())
        elif head == "taken":
            predicate, _, where = rest.partition(" ")
            domain.taken[predicate] = int(where)
        elif head == "join":
            first, second, _, template = rest.split(" ", 3)
            domain.joins[(first, second)] = template.strip()
        elif head == "do":
            name, _, template = rest.partition(" ")
            domain.does[name] = template.strip()
        elif head == "action":
            close()
            name, *params = rest.split()
            found = [PARAM.fullmatch(one) for one in params]
            if not all(found):
                raise ValueError(f"{line!r}: a parameter must be ?x or ?x:kind")
            current = {"name": name,
                       "params": [one.group(1) for one in found],
                       "types": [one.group(2) for one in found]}
        elif head in ("needs", "adds", "dels"):
            if current is None:
                raise ValueError(f"{line!r}: outside an action")
            current[head] = _literals(rest)
        else:
            raise ValueError(f"{line!r}: not a line this reads")
    close()
    domain.kinds = tuple(kinds)
    domain.schemas = tuple(schemas)
    domain.goalish = tuple(goalish)
    domain.source = text
    return domain


def _literals(text: str) -> list:
    return [" ".join(one.split()) for one in text.split(",") if one.strip()]


# -- the domains that ship -------------------------------------------------

BLOCKS = """
domain blocks
kind block
start block table ?x, clear ?x
world empty
goalish on table

action take ?x:block
  needs clear ?x, table ?x, empty
  adds  held ?x
  dels  clear ?x, table ?x, empty
action drop ?x:block
  needs held ?x
  adds  clear ?x, table ?x, empty
  dels  held ?x
action stack ?x:block ?y:block
  needs held ?x, clear ?y
  adds  empty, clear ?x, on ?x ?y
  dels  held ?x, clear ?y
action unstack ?x:block ?y:block
  needs clear ?x, on ?x ?y, empty
  adds  held ?x, clear ?y
  dels  clear ?x, on ?x ?y, empty

say on     the {0} block is on the {1} block
say table  the {0} block is on the table
say clear  nothing is on the {0} block
say held   I am holding the {0} block
say empty  my hand is empty

do take     picked up the {0} block
do drop     put the {0} block on the table
do stack    put the {0} block on the {1} block
do unstack  took the {0} block off the {1} block

doing take     pick up the {0} block
doing drop     put the {0} block on the table
doing stack    put the {0} block on the {1} block
doing unstack  take the {0} block off the {1} block

reads table  {0} on the table
reads on     {0} on {1}
reads on     {0} onto {1}
tell  on table held
taken clear 2
join take stack     put the {0} block on the {2} block
join unstack stack  moved the {0} block from the {1} block to the {3} block
join unstack drop   took the {0} block off the {1} block and put it on the table
"""

ERRANDS = """
domain errands
kind place
kind thing
object home place
world free, at home

goalish in

action go ?from:place ?to:place
  needs at ?from
  adds  at ?to
  dels  at ?from
action fetch ?what:thing ?where:place
  needs at ?where, in ?what ?where, free
  adds  carrying ?what
  dels  in ?what ?where, free
action leave ?what:thing ?where:place
  needs at ?where, carrying ?what
  adds  in ?what ?where, free
  dels  carrying ?what

say at         I am at {0}
say in         the {0} is at {1}
say carrying   I am carrying the {0}
say free       my hands are empty

do go      went to {1}
do fetch   picked up the {0} at {1}
do leave   left the {0} at {1}

doing go      go to {1}
doing fetch   pick up the {0} at {1}
doing leave   leave the {0} at {1}

reads in   {0} at {1}
reads in   {0} to {1}
reads in   {0} is in {1}
tell  in at carrying
join go fetch  went to {1} and picked up the {2}
join go leave  went to {1} and left the {2} there
"""

DELIVERY = """
domain delivery
kind parcel
kind town
kind van
start van empty ?x

goalish at

action drive ?v:van ?from:town ?to:town
  needs parked ?v ?from
  adds  parked ?v ?to
  dels  parked ?v ?from
action load ?p:parcel ?v:van ?where:town
  needs at ?p ?where, parked ?v ?where, empty ?v
  adds  aboard ?p ?v
  dels  at ?p ?where, empty ?v
action unload ?p:parcel ?v:van ?where:town
  needs aboard ?p ?v, parked ?v ?where
  adds  at ?p ?where, empty ?v
  dels  aboard ?p ?v

say at       the {0} is in {1}
say parked   the {0} van is in {1}
say aboard   the {0} is in the {1} van
say empty    the {0} van is empty

do drive   drove the {0} van from {1} to {2}
do load    loaded the {0} into the {1} van at {2}
do unload  unloaded the {0} from the {1} van at {2}

doing drive   drive the {0} van from {1} to {2}
doing load    load the {0} into the {1} van at {2}
doing unload  unload the {0} from the {1} van at {2}

reads at   {0} to {1}
reads at   {0} is in {1}
tell  at parked aboard
join drive unload  drove the {0} van to {2} and unloaded the {3}
join load drive    loaded the {0} into the {1} van and drove to {5}
"""

#: Every domain that ships, by name. The blocks world is first because it is
#: the one with an oracle and a measured suite; the others are here to say
#: that nothing in the planner is about blocks, which is a claim that means
#: nothing until a second domain exists.
DOMAINS = {one.name: one for one in
           (parse(BLOCKS), parse(ERRANDS), parse(DELIVERY))}
