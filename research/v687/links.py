"""The link types: what each relation in the graph is, said once, as data.

Every rule that depends on what kind of relation it is looking at used to
keep its own list: R2's inheritable relations and R7's gated ones in
`rules.py`, R29's stored directions and R27's partitions in `reason.py`, R22's
pairs in `inverse.py`, E1's qualities in v689's `episodic.py`, T3's relations
told of a time in `timeline.py`, and the dimensions S1 to S4 walk in
`relations.py`. They are one table now, and each of those names is read off
it (`v690/DESIGN.md` §4.1).

A row says, for one relation:

    inherited      R2   descends the taxonomy from a kind to its kinds
    why_not        R2   the stated reason it does not
    gated          R7   carries no usable semantics and never participates
    denies         R3   the relation this one is the negation of
    family         R9   relations that answer for it, itself included
    range          R13  what its object must fall under
    converse       R22, S1  the same fact read from the other end
    stored         R29  where the store keeps it between two senses, and
                        which way round
    transitive, antisymmetric   R1, T2, S2: walked as a partial order
    disjoint       R27  branches of it nothing belongs to two of
    quality        E1   what a thing is like: not inherited by an individual
    in_time        T3   told of a time, and held within its episode
    exclusive      T3   one value at a time for an individual
    episodic            written only by a conversation, never by the store
    axis, across, compass   S1-S4: the dimension it orders along, the one
                        it puts two things level on, and whether it is a map

Nothing here reads the store or a conversation. A relation with no row is a
plain link: not inherited, not gated, no converse.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LinkType:
    name: str
    inherited: bool = False
    why_not: str = ""
    gated: bool = False
    denies: str = ""
    family: frozenset = frozenset()
    range: str = ""
    converse: str = ""
    #: (relation to search, reversed): False when the node is the concept
    #: column, True when it is the object column, None when either
    stored: tuple | None = None
    transitive: bool = False
    antisymmetric: bool = False
    disjoint: tuple = ()
    #: (branch, branch) pairs the tree separates and the world does not
    not_disjoint: tuple = ()
    quality: bool = False
    in_time: bool = False
    exclusive: bool = False
    episodic: bool = False
    axis: str = ""
    across: str = ""
    compass: bool = False


LINKS: dict[str, LinkType] = {}


def define(*types: LinkType) -> None:
    for one in types:
        LINKS[one.name] = one


def link(name: str) -> LinkType:
    """The row for a relation, or a plain link when it has none."""
    return LINKS.get(name) or LinkType(name)


def named(test) -> frozenset:
    """Every relation whose row passes `test`."""
    return frozenset(name for name, one in LINKS.items() if test(one))


# -- the taxonomy --------------------------------------------------------------
define(LinkType(
    "is_a", transitive=True, antisymmetric=True,
    # R27. Branches of the taxonomy that nothing belongs to two of. These are
    # not guessed: `plant.n.02`, `animal.n.01`, `person.n.01`, `artifact.n.01`
    # and `abstraction.n.06` were checked against each other and none is an
    # ancestor of another. Kept deliberately small. WordNet's hypernym tree is
    # incomplete in the middle -- it does not record that a dog is a pet, or
    # a whale not a fish -- so exclusion is claimed only between these top
    # branches, where the tree really does partition.
    disjoint=("plant.n.02", "animal.n.01", "person.n.01", "artifact.n.01",
              "abstraction.n.06"),
    # The one pair that the tree separates and the world does not. WordNet
    # files `person` beside `animal` rather than under it. It reads one way
    # only: a person is an animal, and a dog is still not a person.
    not_disjoint=(("person.n.01", "animal.n.01"),)))

# -- R9: relations that answer for each other ------------------------------------
# Sources disagree about which relation a fact belongs under. WordNet files "a
# dog has a tail" as `has_part`; Ascent++ files it as `has_a`. A question
# asking about one must see the other, or the answer depends on which source
# happened to record it. Only genuinely interchangeable relations share a
# family -- `part_of` is NOT a member of `has_part`'s, it is its converse.
PARTS = frozenset({"has_a", "has_part"})
PLACES = frozenset({"at_location", "located_near"})
QUALITY = frozenset({"has_property", "has_attribute"})

# R13: what a relation's object is allowed to be. 26.9% of the `at_location`
# rows name something that is not a place at all -- `concept at_location
# play`, `obligation at_location writing` -- and asking where a hammer is
# returned `communication` and `high quality` because of it. WordNet's own
# top-level split is the check, so it needs no new data.
PHYSICAL = "physical entity.n.01"

# -- R2: what descends -----------------------------------------------------------
# A subtype has whatever the supertype has. If mammals can breathe, dogs can.
define(
    LinkType("capable_of", inherited=True),
    LinkType("has_property", inherited=True, family=QUALITY, quality=True,
             in_time=True),
    # R29 routes through `part_of` because only WordNet writes an object as a
    # synset id, and it writes this relation there, whichever way it is asked.
    LinkType("has_a", inherited=True, family=PARTS, converse="part_of",
             stored=("part_of", False)),
    LinkType("has_part", inherited=True, family=PARTS, converse="part_of",
             stored=("part_of", False)),
    LinkType("receives_action", inherited=True),
    LinkType("used_for", inherited=True),
    LinkType("desires", inherited=True),
    LinkType("not_desires", inherited=True, denies="desires"),
    LinkType("not_capable_of", inherited=True, denies="capable_of"),
    LinkType("not_has_property", inherited=True, denies="has_property",
             quality=True, in_time=True),
    LinkType("has_prerequisite", inherited=True),
    LinkType("has_subevent", inherited=True),
    LinkType("motivated_by_goal", inherited=True),
    LinkType("causes", inherited=True, stored=("causes", False)),
    LinkType("at_location", inherited=True, family=PLACES, range=PHYSICAL,
             in_time=True, exclusive=True),
    # The store's direction is not the name's: `(car.n.01, part_of,
    # accelerator.n.01)` is in there and an accelerator is a part of a car, so
    # a "does X have Y" question looks here with X in the concept column.
    # Checked against four unambiguous pairs rather than assumed.
    LinkType("part_of", inherited=True, converse="has_part",
             stored=("part_of", True)),
    LinkType("has_attribute", inherited=True, family=QUALITY, quality=True,
             in_time=True),
    LinkType("entails", inherited=True, stored=("entails", False)),
    LinkType("located_near", inherited=True, family=PLACES, range=PHYSICAL),
)

# -- R2: what does not, and why --------------------------------------------------
define(
    LinkType("made_of", why_not="a subtype may be made of something else "
             "entirely -- a chair is furniture, but furniture is not made of "
             "wood"),
    LinkType("similar_to", why_not="similarity is not transitive through "
             "subtyping", stored=("similar_to", None)),
    LinkType("instance_of", why_not="an instance's membership says nothing "
             "about a subclass"),
    LinkType("created_by", why_not="the maker of a kind is not the maker of "
             "every subkind"),
    LinkType("symbol_of", why_not="symbolism attaches to the specific thing, "
             "not the category"),
    LinkType("defined_as", why_not="a definition is about that concept alone"),
    LinkType("manner_of", why_not="manner relates two actions, it does not "
             "descend a hierarchy"),
)

# -- R7: relations that never participate ----------------------------------------
# `related_to` is 1,678,150 of v633's 3.9M edges and carries no semantics -- no
# direction, no relation type, just co-occurrence. Inheriting it floods every
# answer. It is excluded from storage and from inference.
define(*(LinkType(name, gated=True) for name in (
    "related_to", "has_context", "form_of", "derived_from",
    "etymologically_related_to", "synonym", "antonym", "has_sense",
    "definition", "usage_count", "distinct_from", "verb_group")))

# -- written only by a conversation (v689) ---------------------------------------
define(
    # Not doing a thing is not being unable to: `it wasn't flying`.
    LinkType("did_not", in_time=True, episodic=True),
    # What E2 withdrew. Read by nothing; kept so the page can say what happened.
    LinkType("carried", episodic=True),
    # T2: before is a partial order, walked like the taxonomy.
    LinkType("before", converse="after", transitive=True, antisymmetric=True,
             episodic=True),
)

# S1-S4: a relation places one individual on one side of another along a
# dimension. Each row is the side up the order; its converse is the WordNet
# antonym on the other side. Which axis is at right angles to which is
# geometry, not vocabulary, and is the one thing here not read out of a
# resource.
define(*(LinkType(more, converse=less, transitive=True, antisymmetric=True,
                  episodic=True, axis=axis, across=across, compass=compass)
         for more, less, axis, across, compass in (
             ("north", "south", "north-south", "east-west", True),
             ("east", "west", "east-west", "north-south", True),
             ("above", "below", "vertical", "lateral", False),
             ("right", "left", "lateral", "vertical", False),
             ("bigger", "smaller", "size", "", False))))
