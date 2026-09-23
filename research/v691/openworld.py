"""A world with nothing declared about it.

`domains.py` holds worlds somebody wrote. This is the one nobody wrote: its
things are whatever the conversation names, what can be done with them comes
from VerbNet (`verbs.py`), and whether a thing may fill a role comes from
the store's taxonomy. **No file anywhere says what a block, a van or a door
is**, and adding a new kind of thing is saying its name.

    > use the open world
    > john is in the kitchen and the book is in the shop
    > get the book to the kitchen

It is a `domains.Domain` so that `scene.py`, `page.py` and `talking.py` work
on it unchanged -- the conversation was already general, and this is what it
was general *for* -- but every method is answered rather than looked up:

    kinds      whatever nouns have been said
    ground     `verbs.useful`, backwards from the goal
    in_words   from the shape of the fact, not from a template
    phrase     from the verb and its roles, as VerbNet orders them
    goalish    any predicate any verb brings about

## Reading with no templates

A declared domain is read through its own `say` lines. Here there are none,
so what is read are the shapes English states a fact in at all:

    the book is in the shop        at book shop
    john has the book              with book john
    the door is open               open door
    get the book to the kitchen    at book kitchen
    open the door                  open door

Five patterns, and the last two are the same shapes read as something wanted
rather than something true. This is emphatically **not** a grammar of
English -- v689's reader is that, and joining them is the work `DESIGN.md`
§9 says has not been done. It is enough to state a situation and ask for
something, which is what it takes to show that the planning is general.
"""
from __future__ import annotations

import re

from research.v691 import hearing, learned as L, numbers, quantities as Q, \
    verbs
from research.v691.domains import Domain
from research.v691.world import Action

#: The shapes a fact is stated in, as (pattern, how to read the groups).
#: `is <word>` is a state and `is in/at <thing>` is a place, which is the
#: one distinction that has to be made and the only one made here.
SAYINGS = (
    (re.compile(r"\b(?:the |a |an )?(\w+) (?:is|are|'s) (?:in|at|on|inside|"
                r"within) (?:the |a |an )?(\w+)"), "at {0} {1}"),
    (re.compile(r"\b(?:the |a |an )?(\w+) (?:has|have|holds|is holding|"
                r"is carrying|carries) (?:the |a |an )?(\w+)"),
     "with {1} {0}"),
    (re.compile(r"\b(?:the |a |an )?(\w+) (?:is|are|'s) (\w+)"),
     "{1} {0}"),
)

#: The shapes something is asked for in. A goal is a fact said in the
#: imperative, so these are the same two shapes with a verb in front.
WANTINGS = (
    (re.compile(r"\b(?:get|put|move|take|bring|carry|send|place)\s+"
                r"(?:the |a |an )?(\w+)\s+(?:to|into|in|onto|on|at)\s+"
                r"(?:the |a |an )?(\w+)"), "at {0} {1}"),
    (re.compile(r"\b(?:give|hand|pass)\s+(?:the |a |an )?(\w+)\s+to\s+"
                r"(?:the |a |an )?(\w+)"), "with {0} {1}"),
    (re.compile(r"\b(?:make|leave)\s+(?:the |a |an )?(\w+)\s+(\w+)"),
     "{1} {0}"),
    (re.compile(r"^\s*(\w+)\s+(?:the |a |an )?(\w+)\s*$"), "{0} {1}"),
    # `get the cup to the shop and the book to the garden`: the second
    # errand borrows the first one's verb, as English lets it.
    (re.compile(r"\band\s+(?:the |a |an )?(\w+)\s+(?:to|into|onto)\s+"
                r"(?:the |a |an )?(\w+)"), "at {0} {1}"),
)

#: `make X V`: what a thing is made to do.
CAUSED = re.compile(r"\bmake\s+(?:the |a |an )?(\w+)\s+(\w+)")

#: Words that are a verb or a filler rather than the name of a thing.
NOT_A_THING = frozenset("""the a an and or is are was were be been it its
this that there here what which where why who whose when how i you me
my your please now
actually then so to of in on at into onto with from for all some any thing
things world worlds use using get put move take bring carry send place give
hand pass make leave open close do does did can could would should""".split())


_PAST: dict | None = None


def past(verb: str) -> str:
    """The past tense of a verb, from WordNet's own list of irregular forms.

    WordNet keeps every irregular inflection it knows so that it can find
    the lemma of `took`; read the other way, it says that the past of `take`
    is `took`. Where it lists two -- `took` and `taken`, `went` and `gone`
    -- the participle is the one ending in `n`. Everything it does not list
    is regular. A table of irregular verbs written here would have been the
    hand-written thing this module exists not to have.
    """
    global _PAST
    if _PAST is None:
        _PAST = {}
        try:
            from nltk.corpus import wordnet
            wordnet.ensure_loaded()
            for form, lemmas in wordnet._exception_map["v"].items():
                if form.endswith("ing") or form.endswith("s"):
                    continue
                for lemma in lemmas:
                    _PAST.setdefault(lemma, set()).add(form)
        except Exception:                          # noqa: BLE001
            pass
    forms = sorted(_PAST.get(verb, ()), key=lambda one: (
        one.endswith(("n", "ne")), len(one)))
    if forms:
        return forms[0]
    if verb.endswith("e"):
        return verb + "d"
    if verb.endswith("y") and len(verb) > 1 and verb[-2] not in "aeiou":
        return verb[:-1] + "ied"
    if (len(verb) <= 4 and len(verb) >= 3 and verb[-1] in "td"
            and verb[-2] in "aeiou" and verb[-3] not in "aeiou"):
        # put, cut, set, hit, let, shut, rid: a short verb ending in a
        # single vowel and a t or d keeps its form, and WordNet does not
        # list it because there is nothing to undo.
        return verb
    return verb + "ed"


_RESOLVER = None

_DOERS: dict = {}


def doers(verb: str) -> list:
    """Every sense the store says can do `verb`, best attested first."""
    if verb not in _DOERS:
        found: list = []
        try:
            import sqlite3

            from research.v687 import build
            connection = sqlite3.connect(
                f"file:{build.DEFAULT_STORE}?mode=ro", uri=True)
            try:
                found = [row[0] for row in connection.execute(
                    "SELECT concept, MAX(confidence) FROM facts WHERE "
                    "relation = 'capable_of' AND object = ? "
                    "GROUP BY concept ORDER BY MAX(confidence) DESC", (verb,))]
            finally:
                connection.close()
        except Exception:                          # noqa: BLE001
            found = []
        _DOERS[verb] = found
    return _DOERS[verb]


def article(word: str) -> str:
    return "an" if word[:1] in "aeiou" else "a"


def resolver():
    """The store's taxonomy, opened once. It is what decides whether a
    thing may fill a role, so a world with no store is a world where
    anything can do anything -- which `Things.allows` says plainly."""
    global _RESOLVER
    if _RESOLVER is not None:
        return _RESOLVER
    try:
        from research.v687 import build
        from research.v687.graph import FactGraph
        _RESOLVER = FactGraph(build.DEFAULT_STORE).resolver
    except Exception:                              # noqa: BLE001
        _RESOLVER = False
    return _RESOLVER or None


class Open(Domain):
    """A domain whose every answer is worked out rather than declared."""

    def __init__(self, resolver=None, learned=None) -> None:
        super().__init__(name="open")
        self.things = verbs.Things(resolver)
        #: what has been worked out about acting that VerbNet does not say
        #: -- which states exclude each other, what else has to be true --
        #: kept between conversations (`learned.py`)
        self.learned = learned
        #: every predicate seen, so `goalish` and `tellable` can answer
        self.seen: set = set()
        #: things named without an article -- `john`, not `the book` --
        #: which is what the reader saw of the difference between a name
        #: and a common noun. A name is someone: the same set is what
        #: `Things.acts` reads, because WordNet makes `john` a toilet
        self.names: set = set()
        self.things.agents = self.names
        #: things that have been the subject of an action someone did,
        #: so narration does not say the doer twice
        self.agents: set = set()
        #: whether a thing of a kind can do something itself: `can(kind,
        #: verb)`, asked of what the conversation knows -- v687's walk and
        #: v688 -- by whoever opened the world. None is no opinion, and a
        #: thing may then try anything, as `Things.allows` lets it
        self.can = None
        #: the doings of the last goal: an action's name -> (thing,
        #: carrier, verb, what was said) for a doing by being carried, and
        #: the names of doings done by the thing itself
        self.carried: dict = {}
        self.own: set = set()
        #: an action done the way it was seen done -> the preposition it
        #: was said with: `put the pig on the plane`, not `to`
        self.ways: dict = {}
        #: things the plan needs that nobody said were there: the plane a
        #: pig would have to be put on
        self.supposed: set = set()
        #: why a carrier that does it was not a way: `a house is not
        #: smaller than a plane`, for saying why there is no plan
        self.refused: list = []
        #: predicates asked for as something a thing does: `make a chair
        #: bark` has the chair bark, and is not a state to leave it in,
        #: whatever VerbNet's *bark your shin* says (`OpenReader`)
        self.doings: set = set()
        #: the verbs the last order was said with: tried first
        self.said_verbs: list = []
        #: place -> the preposition a person put before it in an order:
        #: said back the same way (`put the book on the table`, not `to`)
        self.said_preps: dict = {}
        #: whether a thing of one kind is smaller than one of another:
        #: `fits(kind, carrier)`, asked of what the conversation knows
        #: (v688's R31) -- True, False, or None for not known
        self.fits = None
        #: the kind last counted and the person last named: what `she gave
        #: 2 to john` counts, and who she is (`hearing.hear`)
        self.counted = ""
        self.person = ""
        #: a counted action's name -> how it changes counts, for saying
        #: it: whether the basket is where the apples went or came from
        self.counts_moved: dict = {}

    # -- what there is -----------------------------------------------------
    @property
    def kinds(self) -> tuple:                      # type: ignore[override]
        return tuple(sorted(set(self.things.kinds.values())))

    @kinds.setter
    def kinds(self, value) -> None:
        pass

    def note(self, name: str, kind: str = "") -> None:
        self.things.add(name, kind or name)

    def ground(self, objects: dict) -> list:
        """Whatever `verbs.useful` last worked out. A world with no domain
        cannot ground everything -- 7,796 operators over the things in a
        conversation is a memory error -- so the actions are found per goal
        and kept here for the turn."""
        return list(getattr(self, "_actions", ()))

    def toward(self, goal, per_verb: int = 8, facts=frozenset()) -> list:
        """The actions worth grounding for a goal, with its doings.

        A goal can ask for a state -- `open door`, `at book kitchen`, which
        a verb brings about -- or for a **doing**: `fly pig`, which no verb
        brings about, because flying is what a thing does and not a state
        something leaves it in. A doing is done one of two ways:

        - **by the thing itself**, if its kind can (`can`, asked of what the
          conversation knows): `fly bird` is one step;
        - **by being carried**, when the doing is a motion (`change.moves`:
          what is aboard a thing goes where it goes, E2) and there is a
          carrier (`carriers`) that does it, that the thing can be put
          aboard and fits on. `fly pig` needs the pig on a plane and is the
          plane's doing; `sail piano` needs it on a ship. A carrier seen in
          what the story showed (`Learned.carry`) is tried first, and the
          way it was loaded then is the way it is loaded now (`prefer`).
        """
        goal = list(goal)
        counted = [one for one in goal if Q.read(one) is not None]
        goal = [one for one in goal if Q.read(one) is None]
        # A doing is done, not brought about: nothing that leaves a thing
        # in a state of that name is asked for it.
        wanted = [one for one in goal if not (
            len(one.split()) == 2 and one.split()[0] in self.doings)]
        extra, prefer, seen = [], {}, []
        self.carried, self.own, self.ways = {}, set(), {}
        self.refused = []
        for literal in goal:
            parts = literal.split()
            if len(parts) != 2 or not self.a_doing(parts[0]):
                continue
            verb, thing = parts
            if self.able(thing, verb):
                extra.append(Action(literal, frozenset(),
                                    frozenset({literal}), frozenset()))
                self.own.add(literal)
            for kind, carrier, named, way, said in self.carriers(verb):
                # What was seen is a fact about the carrier: what is aboard
                # it goes where it goes. The thing seen aboard is evidence,
                # and anything else that can be put aboard it is carried the
                # same way (`aboard`).
                if not said and not self.able(named, verb):
                    # A store row is a candidate: `pig bed capable_of fly`
                    # is not a carrier, and v688 says so before size is
                    # asked about at all.
                    continue
                fits = ("" if kind and self.is_a(thing, kind)
                        else self.aboard(thing, named, way))
                if fits is None:
                    continue
                if not said:
                    fits += f", and {article(named)} {named} can {verb}"
                # The carrier by the word it was seen as first -- the plane,
                # not a jet a previous answer happened to name -- then any
                # thing here that is one.
                ride = (named if named in self.things.kinds
                        and named != thing else
                        next((one for one in sorted(self.things.kinds)
                              if one != thing and self.is_a(one, carrier)),
                             None))
                if ride is None:
                    ride = named
                    if not self.able(ride, verb):
                        continue
                    self.note(ride, named)
                    self.supposed.add(ride)
                elif not self.able(ride, verb):
                    continue
                aboard = f"at {thing} {ride}"
                name = f"{verb} {thing} {ride}"
                if name in self.carried:
                    # The plane already offered is an aircraft too: the
                    # first way found for it stands, with its reason.
                    continue
                extra.append(Action(name, frozenset({aboard}),
                                    frozenset({literal, f"{verb} {ride}"}),
                                    frozenset()))
                self.carried[name] = (thing, ride, verb, said, fits)
                wanted.append(aboard)
                if way:
                    doing, *by = way.split()
                    prefer.setdefault("at", []).append(doing)
                    # Done once, so it can be done, whatever VerbNet says
                    # the verb's roles must be -- and offered first, so the
                    # way it was seen done is the way tried first.
                    doer = "you" if "you" in self.agents else None
                    for one in verbs.seen_done(doing, aboard, self.things,
                                               doer):
                        seen.append(one)
                        self.ways[one.name] = by[0] if by else ""
        # Which verb, of the several VerbNet has for a fact: the one the
        # order was said with, then the ones people were seen using --
        # before how central a verb is to VerbNet, which is what decided
        # it before anything was seen.
        for literal in wanted:
            predicate = literal.split()[0]
            ways = list(self.said_verbs)
            if self.learned is not None:
                ways += self.learned.preferred(predicate)
            ways += prefer.get(predicate, [])
            if ways:
                prefer[predicate] = list(dict.fromkeys(ways))
        self._actions = self._with_remedies(
            wanted, L.applied(
                seen + verbs.useful(wanted, self.things, per_verb=per_verb,
                                    prefer=prefer) + extra, self.learned),
            per_verb, prefer) if wanted else []
        if counted:
            self._actions += self._counting(counted, facts, per_verb)
        return self._actions

    #: Rounds of what counts need: giving 5 needs 5, and having 5 may need
    #: taking some from the basket first.
    ROUNDS = 2
    #: Verbs per count. Give, pass, hand and leave move apples alike, and
    #: each is another copy of every holder at every amount.
    COUNTING_VERBS = 4

    def _counting(self, wanted: list, facts, per_verb: int) -> list:
        """Actions that move amounts toward counted goals: what VerbNet
        says moves one thing, lifted to `n` of a kind (`quantities.lifted`)
        at the amounts each goal is short of, and then again for what those
        need.

        Which verbs: the one the order was said with, the ones people were
        seen using, and then the verbs that move a thing from one holder to
        another -- give, take, get -- whichever relation each keeps it
        under (`verbs.connecting`). The goal says where apples should end
        up, and apples in a basket are got out of it by a verb the goal does
        not name. Every reading of those verbs that moves a thing to a
        holder is ground with the kind as the thing moved, not the two
        `useful` offers of each: *take from* is neither take's most nor its
        least committed reading, and it is the one that empties a basket.
        """
        chosen: list = list(self.said_verbs)
        for literal in wanted:
            if self.learned is not None:
                chosen += self.learned.preferred(Q.read(literal).predicate)
        chosen += verbs.connecting(Q.HOLDING, Q.HOLDING,
                                   min(per_verb, self.COUNTING_VERBS))
        chosen = list(dict.fromkeys(chosen))
        out: list = []
        names: set = set()
        asked = list(wanted)
        canon = Q.kept(facts)
        for round in range(self.ROUNDS):
            amounts = Q.amounts_for(asked, facts, [
                Q.read(one).value for one in wanted
                if Q.read(one).op in ("+=", "-=")])
            fresh: list = []
            for kind in sorted({Q.read(one).kind for one in asked}):
                ground: list = []
                for verb in chosen:
                    for ability in verbs.abilities().get(verb, ()):
                        theme = verbs.moved(ability, Q.HOLDING)
                        if theme is not None:
                            ground += verbs.ground(ability, self.things,
                                                   {theme: kind})
                for action in L.applied(ground, self.learned):
                    if action.doer and self.things.acts(action.doer) is False:
                        # A basket does not give John six apples: whoever
                        # moves a count is somebody (`Things.acts`).
                        continue
                    for n in amounts:
                        made = Q.lifted(action, kind, n, canon)
                        if made is not None and not self._mine(made):
                            continue
                        if made is not None and made.name not in names:
                            names.add(made.name)
                            fresh.append(made)
                            self.counts_moved[made.name] = made.changes
            if round == 0:
                fresh += self._consuming(wanted, amounts, names)
            if round == 0 and self.said_verbs:
                # **The order says how.** `give john 6 apples` is not
                # answered by John taking six from the basket himself,
                # however short that plan is: where the verb said can change
                # the count asked about, only it may. Getting what it needs
                # -- the apples to give -- is anybody's way.
                goal = {Q.read(one).fluent for one in wanted}

                def touches(one) -> bool:
                    return any(fluent in goal for fluent, _ in one.changes)

                said = [one for one in fresh if touches(one)
                        and one.name.split()[0] in self.said_verbs]
                if said:
                    fresh = [one for one in fresh
                             if not touches(one) or one in said]
            # Whose to move, first: mine, then yours, then anyone else's.
            # Asked to give John four apples, handing over Mary's is a plan
            # and not the one anybody meant.
            fresh.sort(key=lambda one: (one.doer != "me", one.doer != "you"))
            out += fresh
            asked = [one for one in Q.needed(fresh)
                     if not Q.holds(one, facts)]
            if not asked:
                break
        return out

    def _consuming(self, wanted: list, amounts: list, names: set) -> list:
        """`eat 2 pears`: a count used up by a verb VerbNet says takes a
        thing in (`take_in`: eat-39.1, drink) -- where it goes is nowhere,
        so no reading of the verb moves it and `lifted` has nothing to
        lift. Done by me, or with yours on your behalf."""
        out = []
        for verb in self.said_verbs:
            if not hearing._consumes(verb):
                continue
            for literal in wanted:
                found = Q.read(literal)
                if found is None or found.holder not in self.SELVES:
                    continue
                for n in amounts:
                    name = f"{verb} {found.holder} {found.kind} #{Q._number(n)}"
                    if name in names:
                        continue
                    names.add(name)
                    fluent = f"{found.predicate} {found.kind} {found.holder}"
                    made = Action(name, frozenset({Q.condition(fluent, ">=",
                                                               n)}),
                                  frozenset(), frozenset(),
                                  changes=((fluent, -n),),
                                  doer=found.holder)
                    self.counts_moved[name] = made.changes
                    out.append(made)
        return out

    #: Who a plan may act as, and whose things it may give away: the one
    #: being asked, and the one asking, on their behalf.
    SELVES = ("me", "you")

    def _mine(self, action) -> bool:
        """Whether a plan may move a count this way: done by me or for you,
        and taking only from me, from you, or from what is nobody's -- a
        basket, a shelf. Mary's apples are Mary's: giving them to John is
        not a thing to do because it was asked, and taking them is worse."""
        if action.doer and action.doer not in self.SELVES:
            return False
        for fluent, delta in action.changes:
            holder = fluent.split()[-1]
            if (delta < 0 and holder not in self.SELVES
                    and self.things.acts(holder) is not False):
                return False
        return True

    def _with_remedies(self, wanted: list, actions: list, per_verb: int,
                       prefer: dict) -> list:
        """The actions, and what it takes to make them possible.

        Two things VerbNet cannot supply. A state no frame brings about --
        `unlocked door` -- is made by the verb WordNet derives it from
        (`verbs.making`). And a state something has been learned to be
        *blocked* by (`lessons.py`: a locked door does not open) has to be
        undone before it, so the opposite state is wanted too: `unlocked`
        for `locked`, which is the same derivation again. One round, so
        the remedy of a remedy is not chased; that is the planner's job
        once the actions exist.
        """
        more: list = []
        for literal in wanted:
            parts = literal.split()
            # A doing is not a state, whatever WordNet's adjectives say:
            # *fly* is one (slang for alert), and a pig is not made to fly
            # by being left in it.
            if (len(parts) == 2 and not self.a_doing(parts[0])
                    and parts[0] not in verbs.brought_about()):
                more += verbs.making(literal, self.things)
        undo: list = []
        for action in actions:
            # A state a taught or learned requirement asks for is made the
            # same way as one the goal asks for: `unlocked box`, by unlock.
            for need in sorted(action.needs):
                parts = need.split()
                if (len(parts) == 2 and need not in wanted
                        and need not in undo and not self.a_doing(parts[0])
                        and parts[0] not in verbs.brought_about()
                        and verbs.making(need, self.things)):
                    undo.append(need)
            for blocked in getattr(action, "forbids", ()):
                parts = blocked.split()
                if len(parts) != 2:
                    continue
                for other in sorted(verbs.opposites(parts[0])):
                    one = f"{other} {parts[1]}"
                    if one not in wanted and one not in undo:
                        undo.append(one)
        if undo:
            actions = actions + L.applied(
                verbs.useful(undo, self.things, per_verb=per_verb,
                             prefer=prefer), self.learned)
            for literal in undo:
                more += verbs.making(literal, self.things)
        names = {one.name for one in actions}
        return actions + [one for one in L.applied(more, self.learned)
                          if one.name not in names]

    #: How many carriers the store is asked about for one doing. Each is
    #: two questions to v688 -- can it do it, and does the thing fit.
    CARRIERS = 4

    def carriers(self, verb: str) -> list:
        """(kind seen aboard, carrier sense, its word, how it was loaded,
        what was said) for everything that might carry a thing doing `verb`.

        First what the story showed (`Learned.carry`). Then, for a motion
        (`change.moves`), what the store says does it (`capable_of verb`)
        that is a vehicle or a container -- VerbNet's own words for what
        things ride in, read off the sense's own ancestors -- best attested
        first. Whether each really can, and whether the thing fits, is asked
        of v688 afterwards: the rows are candidates, not answers.
        """
        out = list(self.learned.carriers(verb)) if self.learned else []
        from research.v689 import change
        if not change.moves(verb):
            return out
        seen = {one[1] for one in out}
        for concept in doers(verb):
            if len([one for one in out if not one[4]]) >= self.CARRIERS:
                break
            if concept in seen or not self.carries(concept):
                continue
            seen.add(concept)
            out.append(("", concept,
                        concept.split(".")[0].replace(" ", "-"), "", ""))
        return out

    def carries(self, concept: str) -> bool:
        """Whether a sense is something things ride in or on: a vehicle or
        a container, by its own ancestors in the store."""
        senses = self.things.senses
        if senses is None:
            return False
        try:
            names = {concept.split(".")[0].replace(" ", "_")} | {
                one.split(".")[0].replace(" ", "_")
                for one in senses.ancestors(concept)}
        except Exception:                          # noqa: BLE001
            return False
        return bool(names & (set(verbs.KINDS["vehicle"])
                             | set(verbs.KINDS["container"])))

    def aboard(self, thing: str, carrier: str, way: str) -> str | None:
        """Whether a thing can be put aboard a carrier, and why: the verb
        that put something aboard it must take the thing (`verbs.takes`:
        put-9.1's Theme is concrete), and the thing must fit -- smaller than
        the carrier, on the scale people rated. The reason as words, or
        None where it cannot, or where its size is not known."""
        doing = way.split()[0] if way else ""
        kind = self.things.kinds.get(thing, thing)
        a, one = f"{article(kind)} {kind}", f"{article(carrier)} {carrier}"
        if doing and not verbs.takes(doing, "at", thing, self.things):
            self.refused.append(f"{a} is not something one can {way} {one}")
            return None
        if self.fits is None:
            return None
        fits = self.fits(kind, carrier)
        if fits is not True:
            self.refused.append(f"{a} is not smaller than {one}"
                                if fits is False else
                                f"I do not know whether {a} fits on {one}")
            return None
        return f"{a} is smaller than {one}"

    def a_doing(self, predicate: str) -> bool:
        """A doing, not a state: a verb VerbNet has frames for that no
        verb brings about (`fly`, `swim`; `open` is brought about, so `open
        door` asks for a state), or one asked for as what a thing does
        (`make a chair bark`)."""
        from research.v689 import change
        return predicate in self.doings or (
            predicate in change.frames()
            and predicate not in verbs.brought_about())

    def able(self, name: str, verb: str) -> bool:
        if self.can is None:
            return True
        try:
            return bool(self.can(self.things.kinds.get(name, name), verb))
        except Exception:                          # noqa: BLE001
            return False

    def is_a(self, name: str, kind: str) -> bool:
        """Whether a thing is of a kind something was learned about: the
        word it was said as, or any of its senses or their ancestors in the
        store -- so what was seen of one pig holds of pigs."""
        said = self.things.kinds.get(name, name)
        if kind in (said, name):
            return True
        senses = self.things.senses
        if senses is None:
            return False
        try:
            return any(one == kind or kind in senses.ancestors(one)
                       for one in senses.denotes(said)[:verbs.Things.SENSES])
        except Exception:                          # noqa: BLE001
            return False

    def begin(self, objects: dict) -> frozenset:
        return frozenset()

    # -- saying it ---------------------------------------------------------
    def in_words(self, fact: str) -> str:
        """A fact in English, from its shape. `at X Y` is a place because
        `at` is what `verbs.py` calls a place, and everything of two
        arguments reads the same way whatever the predicate turns out to
        be -- which is what lets a word nobody has seen before be said."""
        counted = Q.in_words(fact, self.the)
        if counted is not None:
            return counted
        parts = fact.split()
        if len(parts) == 3 and parts[0] == "at":
            return f"{self.the(parts[1])} is in {self.the(parts[2])}"
        if len(parts) == 3 and parts[0] == "with":
            return f"{self.the(parts[2])} has {self.the(parts[1])}"
        if len(parts) == 3:
            return (f"{self.the(parts[1])} is {parts[0]} "
                    f"{self.the(parts[2])}")
        if len(parts) == 2 and self.a_doing(parts[0]):
            return f"{self.the(parts[1])} to {parts[0]}"
        if len(parts) == 2:
            return f"{self.the(parts[1])} is {parts[0]}"
        return fact

    def the(self, name: str) -> str:
        """A thing as it is said: `the book`, but `john`. A thing first
        named with an article is a common noun and one named without is a
        name, which is all the reader saw and all it needs."""
        said = name.replace("-", " ")
        return said if name in self.names else f"the {said}"

    def phrase(self, action: str) -> str:
        """What was done, in the past tense: the verb and the things it was
        done to, in the order VerbNet puts its roles."""
        return self._said(action, past)

    def doing(self, action: str) -> str:
        """What is to be done, after `could`: the bare verb."""
        return self._said(action, lambda verb: verb)

    def _said(self, action: str, tense) -> str:
        action, amount = Q.action_words(action)
        if amount is not None:
            return self._moved(action, amount, tense)
        if action in self.carried:
            thing, ride, verb = self.carried[action][:3]
            if tense is past:
                return (f"{self.the(ride)} {past(verb)}, carrying "
                        f"{self.the(thing)}")
            return f"have {self.the(ride)} {verb} with {self.the(thing)} on it"
        if action in self.own:
            verb, thing = action.split()
            if tense is past:
                return f"{self.the(thing)} {past(verb)}"
            return f"let {self.the(thing)} {verb}"
        parts = action.split()
        if len(parts) == 1:
            return tense(parts[0])
        rest = [one for one in parts[1:]]
        # The doer is the subject and is not said again after the verb: `I
        # took the book`, not `I took john the book`.
        if len(rest) > 1 and rest[0] in self.agents:
            rest = rest[1:]
        said = f"{tense(parts[0])} {self.the(rest[0])}"
        if len(rest) > 1:
            # The preposition it was asked with, where it is done with the
            # verb it was asked with: `carried the box into the garden`,
            # but `took the book to the table` when `put` was not the way.
            asked = (self.said_preps.get(rest[-1])
                     if parts[0] in self.said_verbs else None)
            said += (f" {self.ways.get(action) or asked or 'to'} "
                     + self.the(rest[-1]))
        return said

    def _moved(self, action: str, amount, tense) -> str:
        """`gave 2 apples to mary`: an amount moved, said with its count.
        The doer is the subject and is not said again after the verb."""
        verb, *rest = action.split()
        doer = rest[0] if len(rest) > 1 and (
            rest[0] in self.names or rest[0] in ("you", "me")) else ""
        if doer:
            rest = rest[1:]
        if not rest:
            return tense(verb)
        changes = dict(self.counts_moved.get(f"{action} #{amount}", ())) or             dict(self.counts_moved.get(
                f"{action} #{Q._number(amount)}", ()))
        count = numbers.counted(amount, rest[0])
        if doer == "you" and any(delta < 0 and fluent.split()[-1] == "you"
                                 for fluent, delta in changes.items()):
            # Done for the person, with what is theirs.
            count = (f"{numbers.said(amount)} of your "
                     f"{numbers.plural(rest[0], amount)}")
        said = f"{tense(verb)} {count}"
        if doer not in ("", "you", "me"):
            # Somebody else's, and they did it: I only saw to it.
            said = (f"{'had' if tense is past else 'have'} "
                    f"{self.the(doer)} {verb} {count}")
        if len(rest) > 1:
            other = rest[-1]
            # Where the count went from or to, by which way it moved.
            lost = any(delta < 0 and fluent.split()[-1] == other
                       for fluent, delta in changes.items())
            asked = (self.said_preps.get(other)
                     if verb in self.said_verbs and not lost else None)
            said += (f" {'from' if lost else asked or 'to'} "
                     f"{self.the(other)}")
        return said

    @property
    def goalish(self) -> tuple:                    # type: ignore[override]
        """Anything a verb can bring about, plus anything said.

        A declared domain lists what an order may ask for. Here the list is
        the verbs': if something makes a door open, then `open the door` is
        a thing to want, and nobody had to say so.
        """
        return tuple(sorted(self.seen | set(verbs.brought_about())))

    @goalish.setter
    def goalish(self, value) -> None:
        pass

    def tellable(self) -> tuple:
        """Only what has been said -- everything a verb *could* bring about
        is not a description of this scene."""
        return tuple(sorted(self.seen))

    def typing(self) -> dict:
        return {}

    def joined(self, parts: list, actions: list) -> list:
        return parts

    def reader(self) -> "OpenReader":
        return OpenReader(self)


class OpenReader:
    """Facts out of English, with no templates to read them by."""

    def __init__(self, domain: Open) -> None:
        self.domain = domain

    def _names(self, plain: str) -> None:
        """Which words were said with no article in front of them, where a
        thing could have been named: those are names. A word said with one
        anywhere is a common noun from then on."""
        words = plain.replace(",", " ").split()
        for index, word in enumerate(words):
            if word in NOT_A_THING or not word.isalpha():
                continue
            before = words[index - 1] if index else ""
            if before in ("the", "a", "an", "some", "my", "your", "his",
                          "her", "their", "its"):
                self.domain.names.discard(word)
            elif index == 0 or before in ("and", ",", "to", "with"):
                if word not in self.domain.things.kinds or \
                        word in self.domain.names:
                    self.domain.names.add(word)

    @staticmethod
    def _result(fact: str) -> str:
        """`unlock door` asked for is `unlocked door` wanted: an order whose
        verb no frame brings about, and whose participle is a state
        (WordNet's adjective), asks for the thing to be left in that state.
        `fly pig` stays a doing -- *flown* is not a state."""
        parts = fact.split()
        if len(parts) != 2 or parts[0] in verbs.brought_about():
            return fact
        from research.v689 import change
        state = verbs.participle(parts[0])
        if (parts[0] in change.frames() and change.adjective(state)
                and verbs.maker(state) == parts[0]):
            return f"{state} {parts[1]}"
        return fact

    def mentions(self, text: str) -> list:
        """Every word that could be the name of a thing. Generous on
        purpose: a name it wrongly admits is a thing nothing can be done
        with, and a name it misses is a thing the goal cannot mention."""
        return [(one, one) for one in re.findall(r"[a-z][a-z0-9-]*", text)
                if one not in NOT_A_THING]

    def asked(self, text: str) -> bool:
        """Whether it asks what something would take: `how would a pig
        fly`, `what would it take to get the book home` (`hearing.py`)."""
        if hearing.nlp() is None:
            return False
        return self.heard(text).asked

    #: The utterances last heard, with what they were heard in the light
    #: of. The page reads one utterance several times -- as facts, as an
    #: order, as a question -- and it must be the same utterance each time,
    #: not one that has already taught itself what `she` is.
    KEPT = 8

    def heard(self, text: str) -> "hearing.Heard":
        """One utterance, read once, with what was last counted and who was
        last named as what a bare number and `she` refer to."""
        kept = self.__dict__.setdefault("_heard", {})
        if text not in kept:
            if len(kept) >= self.KEPT:
                kept.pop(next(iter(kept)))
            found = hearing.hear(text, verbs.stated,
                                 kind=self.domain.counted,
                                 who=self.domain.person)
            kept[text] = found
            self._context(found)
        return kept[text]

    def _context(self, found) -> None:
        """What this utterance leaves `she` and a bare number meaning."""
        if found.counted:
            self.domain.counted = found.counted
        for name in found.names:
            if name not in NOT_A_THING:
                self.domain.names.add(name)
        literals = list(found.facts) + [fluent for fluent, _ in
                                        found.changes] + list(found.wants)
        for literal in literals:
            for name in literal.split()[1:]:
                if name in ("you", "me") or name in hearing.PERSONS:
                    continue
                # `the farmer had 25 eggs. he sold 10`: a person is named
                # by a name, or by a word whose first sense is someone.
                if name in self.domain.names or (
                        Q.read(literal) is not None
                        and name == literal.split()[-1]
                        and self.domain.things.acts(name)):
                    self.domain.person = name
        for literal in literals:
            if Q.read(literal) is not None or len(literal.split()) == 3:
                for name in literal.split()[2:]:
                    if name in ("you", "me"):
                        self.domain.names.add(name)

    def counts(self, text: str) -> tuple:
        """(how counts changed, a question of how many, counts no longer to
        be trusted) in an utterance."""
        if hearing.nlp() is None:
            return [], None, []
        found = self.heard(text)
        return list(found.changes), found.count, list(found.unsure)

    def facts_in(self, text: str, wanting: bool = False) -> list:
        plain = " ".join(text.lower().replace(",", " , ").split())
        self._names(plain)
        if hearing.nlp() is not None:
            # Read off the parse, with VerbNet saying what an order's verb
            # does (`hearing.py`). The patterns below are what is left when
            # no parser is installed.
            heard = self.heard(text)
            out = list(heard.wants if wanting else heard.facts)
            if wanting:
                self.domain.doings |= heard.doings
                self.domain.said_verbs = list(heard.verbs)
                self.domain.said_preps.update(heard.preps)
                out = [self._result(fact) for fact in out]
            # `you` and `me` are not things -- except as who has a count: `i
            # have five apples` is the person's. And a word the parse reads
            # as a noun in a thing's place is a thing whatever else it can
            # be: *open the can*.
            out = [fact for fact in dict.fromkeys(out)
                   if not any(one in NOT_A_THING
                              and one not in heard.nouns
                              and not (Q.read(fact) is not None
                                       and one in ("you", "me"))
                              for one in fact.split()[1:])]
            for fact in out:
                if Q.read(fact) is None:
                    self.domain.seen.add(fact.split()[0])
            return out
        found: list = []
        taken: list = []
        for pattern, shape in (WANTINGS if wanting else SAYINGS):
            for match in pattern.finditer(plain):
                span = match.span()
                if any(span[0] < end and start < span[1]
                       for start, end in taken):
                    continue
                groups = [one for one in match.groups() if one]
                fact = shape.format(*groups)
                # Checked after the shape is filled, not before: in `open
                # the door` the first group is the *predicate*, and a verb
                # is exactly what a predicate is allowed to be. Only the
                # arguments have to be things.
                if any(one in NOT_A_THING for one in fact.split()[1:]):
                    continue
                taken.append(span)
                found.append((span[0], fact))
        out = [fact for _, fact in sorted(found)]
        if wanting:
            out = [self._result(fact) for fact in out]
        for fact in out:
            self.domain.seen.add(fact.split()[0])
        if wanting:
            # `make a pig fly`, `make a chair bark`: the thing is the one
            # doing it, so it asks for a doing -- unless the word names a
            # state (`make the door open`: WordNet's adjective, as T4 has
            # it).
            for match in CAUSED.finditer(plain):
                thing, verb = match.group(1), match.group(2)
                if (f"{verb} {thing}" in out and verb not in NOT_A_THING
                        and not verbs.stated(verb)):
                    self.domain.doings.add(verb)
        return out
