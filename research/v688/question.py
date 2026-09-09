"""Where the next question comes from.

The executive never invents a question. There are exactly three sources, and
each one is a mechanical transform of something already on the table:

    gap        the last answer named what it did not have. `UNKNOWN_WORD`
               naming *which* word is what makes the follow-up derivable
               rather than guessed.
    doubt      the last answer is weaker than its verdict looks, so the same
               claim is put to the concept's parent and its siblings. This is
               R19 corroboration run *across* queries instead of inside one,
               which is the belief revision the subsystem audit lists as
               present but unrun.
    curiosity  nothing is wrong; the trie has a child here that would split
               the field, and asking it is how the field gets split.

Gap questions are serial by nature -- you cannot know the second before the
first comes back. That is fine, and it is why curiosity exists: it supplies
breadth on cycle one with nothing to wait for, which is what a pool of
workers can actually be given.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import attention
from .gap import Doubt, Gap
from .graph import GraphCuriosity, Kinds, Requirements

VOWELS = "aeiou"

#: A blocker opening with one of these is the tail of a phrase v687 handed
#: back whole, not a term: `for rabbits`, `into a hole`, `build nests`.
#: `what is a for rabbits` is a question the loop invented for itself.
FRAGMENT = frozenset("""
for into onto out off up down over under through across around at in on to
from by with and or not no never that which who be is are was were do does
did have has had can could will would
""".split())

#: How far up the taxonomy curiosity will look for a field to compare a
#: concept against. Beyond about three levels the ancestor is `organism` or
#: `artifact` and its kinds have nothing to do with the concept asked about:
#: it is how `hunt` ended up being asked whether it was a small bird.
MAX_ANCHOR_DISTANCE = 3

#: Words a stored object should never open with when it is being put to a
#: sibling as a thing the sibling has.
AUXILIARY = frozenset("can could will would do does did is are was were "
                      "has have had must should may might".split())

#: Body parts come out of the store singular -- `leg`, `wing` -- and the
#: feature norms record them plural: `has legs` is what XCSLB scores and
#: denies. `does a fish have a leg` comes back UNRECORDED where `does a fish
#: have legs` reaches the norms and the family that denies it.
IRREGULAR = {"tooth": "teeth", "foot": "feet", "mouse": "mice",
             "goose": "geese", "child": "children"}


def plural(word: str) -> str:
    word = (word or "").strip()
    if word in IRREGULAR:
        return IRREGULAR[word]
    if not word or word.endswith("s"):
        return word
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        return word[:-1] + "ies"
    if word.endswith(("ch", "sh", "x", "z")):
        return word + "es"
    return word + "s"

#: How many kinds a concept may have and still count as a family. `dog` has
#: three and putting a claim to all three is a check; `device` has fifty-nine
#: -- accordions, arrows, bayonets -- and putting a claim to those is a
#: different question wearing the same words. It is what turned `what has
#: wings` into `can an accordion play video`: `wing` resolved to the aircraft
#: sense, whose nearest ancestor with kinds is `device`.
MAX_FAMILY = 12

#: How far a claim may have been inherited and still be worth putting to the
#: family it came from. It was 4, on the reasoning that a distant ancestor is
#: an unrelated one -- but distance is not what makes an ancestor unrelated,
#: family size is, and `Kinds` already bounds that. Meanwhile the cap was
#: hiding the worst case it exists for: `do pigs fly` rests on `mammal
#: capable_of fly` at five levels, which is a fact about bats.
MAX_DOUBT_DISTANCE = 7

#: How much the store has to hold about a word before curiosity will take it
#: as a topic. `purr`, `hunt` and `hello` each have one recorded fact and are
#: nouns in WordNet, which was enough to produce `can a purr be airborne`.
#: `cat` has 1312, `shark` 923, `beagle` 136, `greeting` 60. Being interested
#: in something requires already knowing enough about it to compare it to
#: anything.
MIN_FACTS_FOR_CURIOSITY = 20

#: How many answers deep a chain of thought may run. Each step's subject comes
#: out of the previous step's *content*, so nothing after the first can be
#: asked until the one before it comes back: a chain uses one worker and as
#: many cycles as it has steps. That is the honest limit of the pool, and the
#: reason curiosity exists to fill the other eighteen lanes.
MAX_CHAIN_DEPTH = 4

#: Where a definition ladder stops being about anything. `beagle -> hound ->
#: hunting dog -> dog -> canine -> carnivore` is a chain of thought;
#: `-> organism -> physical entity` is the ontology's plumbing.
LADDER_FLOOR = frozenset({
    "entity", "physical entity", "abstraction", "object", "whole", "unit",
    "living thing", "organism", "thing", "matter", "substance", "causal agent",
    "artifact", "instrumentality", "attribute", "psychological feature",
    "abstract entity", "relation", "measure", "group", "act", "event",
    "state", "phenomenon", "process",
    # The rungs a taxonomy passes through on its way to saying nothing.
    # `violin -> bowed stringed instrument -> stringed instrument -> musical
    # instrument` is a chain of thought; `-> device` is filing. Likewise
    # `bat -> vertebrate -> chordate -> animal`: the first rung answers the
    # question and the rest are the ontology's own scaffolding.
    "animal", "chordate", "vertebrate", "invertebrate", "device",
    "food", "material", "part", "structure", "body part", "person",
})

#: How a stored relation reads back as a question. The loop has to rebuild a
#: question from (concept, relation, object) to put a doubted claim to a
#: sibling, and the phrasing has to be one v687's parser routes to the same
#: relation it came from -- otherwise the corroboration answers a different
#: question than the one it is corroborating.
FRAMES = {
    "capable_of": "can {article} {concept} {object}",
    "receives_action": "can {article} {concept} be {object}",
    "has_property": "is {article} {concept} {object}",
    "has_a": "does {article} {concept} have {object}",
    "has_part": "does {article} {concept} have {object}",
    "part_of": "is {article} {concept} part of {an_object}",
    "is_a": "is {article} {concept} {an_object}",
    "defined_as": "is {article} {concept} {an_object}",
    "similar_to": "is {article} {concept} similar to {an_object}",
    "made_of": "is {article} {concept} made of {object}",
    "at_location": "is {article} {concept} found in {object}",
    "located_near": "is {article} {concept} found near {object}",
    "used_for": "is {article} {concept} used for {object}",
    "created_by": "is {article} {concept} created by {object}",
    "has_prerequisite": "does {article} {concept} require {object}",
    "has_subevent": "does {article} {concept} involve {object}",
    "causes": "does {article} {concept} cause {object}",
    "desires": "does {article} {concept} want {object}",
    "entails": "does {article} {concept} entail {object}",
    "verify": "is {article} {concept} {object}",
}

#: The relations whose object is a thing the concept has, where the article
#: depends on whether that thing is singular. `has_a scales` is `have scales`;
#: `has_part tail` is `have a tail`.
HAVING = ("has_a", "has_part", "has_property", "verify")

#: Bare AwA2 attribute tokens carry no frame of their own, so the frame comes
#: from the part of speech: `furry` is something a collie is, `tail` is
#: something it has. Without this, `is a collie tail` gets asked.
BY_POS = {"a": "is {article} {concept} {object}",
          "s": "is {article} {concept} {object}",
          "n": "does {article} {concept} have {an_object}",
          "np": "does {article} {concept} have {object}",
          "v": "can {article} {concept} {object}",
          "r": "is {article} {concept} {object}"}

#: The AwA2 columns are attributes, so where a token reads as both an
#: adjective and a noun the adjective is the one meant. `fast` resolves to
#: `fast.n.01`, a religious fast, and asked `does a beagle have fast`.
POS_PREFERENCE = ("a", "s", "n", "v", "r")

#: AwA2 habitat columns. They are nouns, and read as nouns they become `does
#: a beagle have a ground`; they are places, and the frame they need is
#: locative. This list is corpus-specific and says so.
#: A surface you live *on*, not in. `is a dog found in the ground` was asked
#: of every animal, and it is a question about burial. v687 strips the
#: preposition before matching, so it answers `on` and `in` identically and
#: the mistake is invisible from the answer -- the phrasing has to be right
#: going in.
#: And with no verb of its own. v687 splits a question into content terms
#: and scores each: `does a dog live on the ground` becomes `live` and
#: `ground`, and `live` matches the stored predicate `lives in a stable`,
#: which the norms deny of dogs. `ground` alone is HELD, which is right, and
#: the conjunction of the two comes out CONTRADICTED, which is not. Checked
#: against the norms across dog, fish, whale, bird and lion, the verbless
#: form agrees with them every time and `live` agrees on three of five.
LIVES_ON = frozenset("ground plains fields".split())
LIVES_ON_FRAME = "is {article} {concept} on the {object}"

HABITAT = frozenset("""
water tree cave coastal desert bush forest jungle mountains ocean arctic
swamp
""".split())
HABITAT_FRAME = "is {article} {concept} in the {object}"

#: AwA2 diet columns, which have the same problem one frame over: read as
#: nouns they become `does a wolf have a fish`.
DIET = frozenset("""
meat fish plankton insects vegetation grass fruit seeds nuts leaves
""".split())
DIET_FRAME = "does {article} {concept} eat {object}"

#: AwA2 column names that are not English words in any frame. They are real
#: columns of a real corpus and nonsense as questions, so curiosity skips them
#: rather than asking something nobody could answer.
NOT_A_QUESTION = frozenset("""
oldworld newworld quadrapedal bipedal nestspot agility chewteeth meatteeth
buckteeth strainteeth toughskin hairless bulbous flys smelly patches
""".split())

#: Ancestors too general for corroboration to mean anything. Putting `is X
#: furry` to alligators and ants is not checking a claim about beagles, it is
#: asking a different question -- and `animal.n.01` is where a 0.11-confidence
#: crawl fact about fur actually sits.
TOO_GENERAL = frozenset({
    "entity", "physical entity", "abstraction", "object", "whole", "unit",
    "living thing", "organism", "animal", "artifact", "instrumentality",
    "matter", "substance", "attribute", "thing", "causal agent", "person",
})

#: Predicate phrases that already carry their own verb, keyed by the word they
#: open with. `can be played` becomes `can an accordion be played`; `has a
#: reed` has to become `does an accordion have a reed`, because `has an
#: accordion a reed` is not a sentence anyone parses.
OPENERS = {
    "is": "is {article} {concept} {rest}",
    "can": "can {article} {concept} {rest}",
    "has": "does {article} {concept} have {rest}",
    "does": "does {article} {concept} {rest}",
    "was": "was {article} {concept} {rest}",
}
#: Anything else that opens a predicate is a present-tense verb: `eats meat`,
#: `lives in water`, `contains aluminum`.
VERB_OPENER = "does {article} {concept} {opener} {rest}"


def article(word: str) -> str:
    word = (word or "").strip()
    return "an" if word[:1].lower() in VOWELS else "a"


@dataclass(frozen=True)
class Question:
    """One thing to ask, and everything about why it is being asked."""

    text: str
    origin: str            # seed | gap | doubt | curiosity
    about: str = ""        # the concept it is about
    predicate: str = ""    # the trie predicate or claim under test
    why: str = ""          # one line, shown on the page
    parent: str = ""       # the question that produced this one
    salience: float = 1.0
    gain: float = 1.0
    urgency: float = 1.0
    novelty: float = 0.0
    #: How many answers had to come back before this question could be
    #: formed. 0 for anything askable from the utterance alone.
    depth: int = 0
    #: word -> synset, held for this question only. v687 takes these; the
    #: buffer keeps them for the rest of the utterance once one is set.
    pins: tuple[tuple[str, str], ...] = ()

    @property
    def key(self) -> str:
        """What counts as the same question. The same words under a different
        reading are a different question, and the whole point of asking."""
        if not self.pins:
            return self.text
        held = " ".join(f"{word}={sense}" for word, sense in self.pins)
        return f"{self.text} ⟨{held}⟩"

    @property
    def rank(self) -> float:
        return attention.rank(self.salience, self.gain, self.urgency)

    def as_dict(self) -> dict:
        return {"text": self.text, "origin": self.origin, "about": self.about,
                "predicate": self.predicate, "why": self.why,
                "parent": self.parent, "salience": round(self.salience, 4),
                "gain": round(self.gain, 4), "urgency": round(self.urgency, 4),
                "novelty": self.novelty, "depth": self.depth,
                "pins": {word: sense for word, sense in self.pins},
                "rank": round(self.rank, 5)}


def phrase(concept: str, relation: str, obj: str) -> str:
    """A question that puts `obj` to `concept` under `relation`."""
    frame = FRAMES.get(relation, FRAMES["verify"])
    return frame.format(article=article(concept), concept=concept,
                        object=obj, an_object=f"{article(obj)} {obj}").strip()


def phrase_predicate(concept: str, predicate: str, pos: str = "",
                     singular: str = "", opener_lemma: str = "") -> str:
    """A question that puts a trie predicate to a concept.

    The trie holds two vocabularies at once -- XCSLB phrases that carry their
    own verb (`can be played`, `is played by musicians`) and AwA2 attribute
    tokens that carry nothing (`furry`, `tail`, `oldworld`). One function
    reads both, because the trie does not distinguish them and neither should
    the question.
    """
    words = (predicate or "").split()
    art = article(concept)
    if not words:
        return ""
    if len(words) == 1:
        token = words[0]
        if token in NOT_A_QUESTION:
            return ""
        if pos == "vs":
            return f"can {art} {concept} {singular or token}"
        if token in LIVES_ON:
            frame = LIVES_ON_FRAME
        elif token in HABITAT:
            frame = HABITAT_FRAME
        elif token in DIET:
            frame = DIET_FRAME
        else:
            frame = BY_POS.get(pos or "a", BY_POS["a"])
        return frame.format(article=art, concept=concept, object=token,
                            an_object=f"{article(token)} {token}")
    opener, rest = words[0], " ".join(words[1:])
    if opener in OPENERS:
        return OPENERS[opener].format(article=art, concept=concept, rest=rest)
    # `contains`, `requires`, `carries`, `goes` -- every predicate that opens
    # with its own verb opens with the third person, and `does a violin
    # requires skill` is not a sentence.
    return VERB_OPENER.format(article=art, concept=concept,
                              opener=opener_lemma or opener, rest=rest)


def bare(concept: str) -> str:
    """`dog.n.01` read back as the word a question can be built from."""
    name = (concept or "").split(".")[0]
    return name.replace("_", " ")


class Generator:
    """The three sources, and the one ranked queue they feed.

    Holds no state of its own: everything it needs comes from the buffer it
    is handed, so a cycle can be replayed by handing it the same buffer.
    """

    def __init__(self, engine, curiosity: attention.Curiosity,
                 requirements=None) -> None:
        self.engine = engine
        self.curiosity = curiosity
        self.requirements = requirements or Requirements(engine.reasoner)
        self.wider = GraphCuriosity(engine.reasoner)
        self.kinds = Kinds(engine.reasoner, engine.profiles)
        self._pos: dict[str, str] = {}
        self._facts: dict[str, bool] = {}
        self._singular: dict[str, str] = {}
        self._lemma: dict[str, str] = {}
        self._verbish: dict[str, bool] = {}

    def lemma(self, word: str) -> str:
        """The dictionary form, from the spaCy model the engine already holds.

        Needed in two places and got wrong by rule in both: `scales` has to
        become `scale` before WordNet will admit it is a noun, and `requires
        skill to play` has to become `require skill to play` before `does a
        violin ...` is a sentence. Stripping an `s` gives `scal` and `goe`.
        """
        if word in self._lemma:
            return self._lemma[word]
        parser = self.engine.parser
        found = word
        if getattr(parser, "nlp", None):
            read = parser.nlp(word)
            if len(read):
                found = read[0].lemma_ or word
        elif word.endswith("ies"):
            found = word[:-3] + "y"
        elif word.endswith("s") and not word.endswith("ss"):
            found = word[:-1]
        self._lemma[word] = found
        return found

    # -- part of speech, for the bare AwA2 tokens -------------------------
    def pos_of(self, token: str) -> str:
        """The part of speech WordNet gives a bare attribute token.

        Cached: the same 85 AwA2 attributes recur across every animal, and
        `senses_of` is a store query.
        """
        if token in self._pos:
            return self._pos[token]
        # A word whose dictionary form is not itself is a plural, and takes
        # no article: `does a carp have a legs` was asked because `legs` is
        # a WordNet lemma in its own right and resolved as a singular noun.
        dictionary = self.lemma(token)
        if dictionary != token and any(
                (sense.get("id") or "").split(".")[-2:-1] == ["n"]
                for sense in self.engine.reasoner.senses_of(dictionary) or []):
            self._singular[token] = dictionary
            self._pos[token] = "np"
            return "np"
        senses = self.engine.reasoner.senses_of(token) or []
        available = set()
        for sense in senses:
            parts = (sense.get("id") or "").split(".")
            if len(parts) >= 2:
                available.add(parts[-2])
        if not available and token.endswith("s") and len(token) > 3:
            # `scales` is not a WordNet lemma; `scale` is. Without this the
            # token reads as no part of speech at all and comes out as
            # `is a carp scales`.
            singular = self.lemma(token)
            for sense in self.engine.reasoner.senses_of(singular) or []:
                parts = (sense.get("id") or "").split(".")
                if len(parts) >= 2:
                    available.add(parts[-2])
            if "v" in available:
                # `walks` is the third person of a verb far more often than
                # it is a plural noun, and `does a cat have walks` is what
                # reading it as the noun produces.
                self._singular[token] = singular
                self._pos[token] = "vs"
                return "vs"
            if "n" in available:
                self._singular[token] = singular
                self._pos[token] = "np"       # a plural noun
                return "np"
        found = next((tag for tag in POS_PREFERENCE if tag in available), "")
        self._pos[token] = found
        return found

    def could_be_a_verb(self, word: str) -> bool:
        """Any verb sense at all, not the preferred one.

        `pos_of` prefers the adjective and then the noun, which is right for
        an AwA2 attribute column and wrong here: `make` has a noun sense (a
        brand) and `used_for make music` is plainly a purpose.
        """
        if word in self._verbish:
            return self._verbish[word]
        found = any((sense.get("id") or "").split(".")[-2:-1] == ["v"]
                    for sense in self.engine.reasoner.senses_of(word) or [])
        self._verbish[word] = found
        return found

    def substantial(self, concept: str) -> bool:
        """Does the store hold enough about this word to compare it to
        anything? See MIN_FACTS_FOR_CURIOSITY."""
        if concept in self._facts:
            return self._facts[concept]
        held = False
        for sense in self.engine.reasoner.senses_of(concept) or []:
            name = sense.get("id") or ""
            if bare(name) == concept and name.split(".")[-2:-1] == ["n"]:
                held = len(self.engine.reasoner.facts_of(name)
                           ) >= MIN_FACTS_FOR_CURIOSITY
                break
        self._facts[concept] = held
        return held

    # -- where a question about this concept is aimed ---------------------
    def pool_for(self, concept: str) -> tuple[str, frozenset[str]]:
        """The field a question about `concept` is trying to split.

        For one of the 541 the trie holds, that is its nearest rivals. For
        anything else it is the nearest ancestor whose kinds the norms *do*
        cover: `beagle` is not in the norms, so the field is the kinds of dog
        that are, and the ranking means something again. Returns the anchor it
        settled on so the page can say which field was used.
        """
        if self.curiosity.knows(concept):
            return concept, self.curiosity.rivals(concept)
        senses = self.engine.reasoner.senses_of(concept) or []
        # The word has to name a thing under its own name. `swim` resolves to
        # `swimming.n.01`, `hunt` to `hunt.n.01` by way of a chase -- both are
        # nouns, and neither is a thing with attributes to ask after. Where
        # the synset does not carry the word itself, curiosity declines.
        itself = next((sense["id"] for sense in senses
                       if bare(sense.get("id") or "") == concept
                       and (sense.get("id") or "").split(".")[-2:-1] == ["n"]),
                      "")
        if itself:
            for name, distance, _parents in self.engine.reasoner.ascend(itself):
                if distance > MAX_ANCHOR_DISTANCE:
                    break
                word = bare(name)
                if self.curiosity.knows(word):
                    return word, self.curiosity.rivals(word)
                kin = [kind for kind in self.engine.profiles.subtypes(word)
                       if self.curiosity.knows(kind)]
                if 2 <= len(kin) <= MAX_FAMILY:
                    return word, frozenset(kin)
                if len(kin) > MAX_FAMILY:
                    return word, frozenset(kin[:MAX_FAMILY])
        return "", self.curiosity.universe

    # -- source 1: the answer named what it lacked ------------------------
    def from_gap(self, hole: Gap, buffer) -> list[Question]:
        """The repairs a gap licenses. A closed set per kind, never a search."""
        urgency = attention.URGENCY.get(hole.kind, 0.5)
        salience = max(buffer.activation.salience(hole.blocker), 0.4)
        asked: list[Question] = []

        if hole.kind == "word":
            # Nothing downstream of an unknown word means anything, and there
            # is exactly one repair the system can run itself: look the word
            # up as something other than the reading that failed. If that
            # comes back empty too, the honest move is to be told, and the
            # loop reports that rather than inventing a third attempt.
            asked.append(Question(
                f"what is {article(hole.blocker)} {hole.blocker}", "gap",
                hole.blocker, why=f"“{hole.blocker}” stopped the question; "
                                  f"nothing after it can be read until it is "
                                  f"named",
                parent=hole.question, salience=salience, gain=1.0,
                urgency=urgency, novelty=self.curiosity.novelty(hole.blocker)))
        elif hole.kind == "sense":
            for option in hole.options[:3]:
                name = bare(option)
                asked.append(Question(
                    f"what is {article(name)} {name}", "gap", name,
                    why=f"“{hole.blocker}” has more than one reading; this "
                        f"asks what {option} is so the readings can be told "
                        f"apart",
                    parent=hole.question, salience=salience, gain=0.9,
                    urgency=urgency))
        elif hole.kind == "coverage":
            # An absence is informative and not blocking, so the repair is to
            # find out where the absence starts: define the missing term, and
            # put the same question one level up. `is a whale a fish` is not
            # answered by knowing more about whales.
            #
            # Only when the blocker is a word, though. v687 hands back whole
            # trailing phrases as the target -- `for rabbits`, `build nests`
            # -- and `what is a for rabbits` is a question the loop invented
            # and nobody asked.
            words = hole.blocker.split()
            # And only when the blocker is a *noun*. `is a dog wild` leaves
            # `wild` uncovered; `wild` has a noun sense -- a wild region --
            # so `what is a wild` came back defining wilderness, and the
            # ladder then walked geographical area -> region -> location.
            # The word was an adjective in the question that raised it.
            blocker = self.lemma(hole.blocker) if len(words) == 1                 else hole.blocker
            if (words and len(words) <= 2
                    and hole.blocker.replace(" ", "").isalpha()
                    and words[0] not in FRAGMENT
                    and self.pos_of(words[-1]) == "n"):
                asked.append(Question(
                    f"what is {article(blocker)} {blocker}", "gap", blocker,
                    why=f"nothing covers “{hole.blocker}”; this asks what it "
                        f"is before asking anything of it",
                    parent=hole.question, salience=salience, gain=0.8,
                    urgency=urgency,
                    novelty=self.curiosity.novelty(hole.blocker)))
        elif hole.kind == "conflict":
            asked.append(Question(
                f"what is {article(hole.blocker)} {hole.blocker}", "gap",
                hole.blocker,
                why="the parts of the answer disagree; this asks after the "
                    "part they disagree about",
                parent=hole.question, salience=salience, gain=0.7,
                urgency=urgency))
        # `construction` licenses nothing: no rule reads the sentence, and
        # asking a differently-worded version of a sentence nobody could read
        # is how a loop talks itself into an answer. It is reported instead.
        return asked

    def reask(self, concept: str, relation: str, obj: str) -> str:
        """Rebuild a stored claim as a question, choosing the frame by hand.

        `phrase` reads the relation only, and `has_property` with a noun
        phrase for an object comes out as `is a carp distinctive fishy smell`.
        A property whose head word is a noun is a thing the concept *has*.
        """
        words = obj.split()
        # An object that carries its own auxiliary is a claim, not a thing.
        # The store records `capable_of "can walk"`, and putting that to a
        # sibling under a having relation asked `does a bobcat have a can
        # walk`. Strip the auxiliary and it is a capability again.
        if words and words[0].lower() in AUXILIARY and len(words) > 1:
            rest = " ".join(words[1:])
            return f"{words[0].lower()} {article(concept)} {concept} {rest}"
        if relation == "used_for" and words and self.could_be_a_verb(words[0]):
            # `used_for make music` is a purpose, not a thing: `is a violin
            # used for make music` has to become `used to make music`.
            return f"is {article(concept)} {concept} used to {obj}"
        if words and words[0].lower() in ("no", "not"):
            # The store records `fish has_a "no legs"`, and an article in
            # front of it gives `does a carp have a no legs`.
            return f"does {article(concept)} {concept} have {obj}"
        if relation in HAVING and words:
            tag = self.pos_of(words[-1])
            if tag == "vs":
                # `scales` is a verb and a plural noun, and `pos_of` prefers
                # the verb because `walks` is one far more often. As the
                # object of `has_a` it is the noun that was meant.
                tag = "np"
            if tag == "np":
                return f"does {article(concept)} {concept} have {obj}"
            if tag == "n":
                return (f"does {article(concept)} {concept} have "
                        f"{article(obj)} {obj}")
            if relation in ("has_a", "has_part"):
                # An object with no noun head under a having relation is a
                # property in disguise: `has_a soft` reads as `is a cat soft`.
                return f"is {article(concept)} {concept} {obj}"
        return phrase(concept, relation, obj)

    # -- source 2: the answer is thinner than it looks --------------------
    def from_doubt(self, doubt: Doubt, buffer) -> list[Question]:
        """Put a doubted claim to the concept it was inherited from, and to
        that concept's other kinds.

        This is the one that finds things. `does a beagle swim` returns
        VERIFIED from a single crawled fact at dog.n.01; asking the same of
        dog itself and of the three kinds of dog the norms cover returns
        DENIED three times over. Neither answer is wrong. The disagreement is
        invisible to anything that asks once.
        """
        parent = bare(doubt.concept)
        if not parent or not doubt.predicate:
            return []
        if doubt.distance > MAX_DOUBT_DISTANCE:
            return []
        kinds = self.kinds.of(parent, doubt.concept)
        if len(kinds) > MAX_FAMILY:
            return []
        if parent in TOO_GENERAL:
            # `is a beagle furry` is VERIFIED off a 0.11-confidence fact at
            # animal.n.01, five levels up. The doubt is real and worth
            # reporting; putting it to alligators and ants is not a check on
            # it. R19 already corroborates within a level, and this would be
            # a worse copy of it.
            return []
        if len(doubt.predicate.split()) > 3 or "," in doubt.predicate:
            # Crawled objects like `curved, cheerful tails` are sentences, not
            # claims. Asking a sibling about one asks something nobody meant.
            return []
        urgency = attention.URGENCY["doubt"]
        salience = max(buffer.activation.salience(parent), 0.5)
        asked: list[Question] = []
        if doubt.distance > 0:
            # Only worth asking when the claim was inherited. At distance 0
            # the fact is stated of the subject itself, and re-asking the
            # subject is re-asking the question that raised the doubt.
            asked.append(Question(
                self.reask(parent, doubt.relation, doubt.predicate), "doubt",
                parent, doubt.predicate,
                why=f"the yes came from {doubt.concept}, so this asks "
                    f"{doubt.concept} directly instead of inheriting it",
                parent=doubt.question, salience=salience, gain=0.9,
                urgency=urgency))

        for kind in kinds[:6]:
            asked.append(Question(
                self.reask(kind, doubt.relation, doubt.predicate), "doubt",
                kind, doubt.predicate,
                why=f"{kind} is a kind of {parent} the norms cover: if the "
                    f"claim holds of {parent} it should hold here",
                parent=doubt.question,
                salience=salience * 0.9, gain=0.85, urgency=urgency))
        return asked

    # -- source 3: read it the other way and see whether it still holds ----
    def from_sense(self, doubt, buffer) -> list[Question]:
        """Re-ask the question with the subject held to the sense you meant.

        `do pigs fly` comes back VERIFIED about `pig bed.n.01`, a mould for
        casting pig iron. The verdict is correct and the subject is not the
        one you named. v687 already takes a pin -- `pins.py` exists for
        exactly this and has been request-scoped and unused since it was
        written -- so the loop pins the sense that carries the word itself
        and asks again.

        Which sense to read a word in is v687's to decide, and this does not
        overrule it: it asks a *second* question and reports both readings.
        When they disagree, that disagreement is the answer.
        """
        word = (doubt.question and buffer.subject_of(doubt.question)) or ""
        if not word:
            return []
        chosen = self.better_sense(word, doubt.concept)
        if not chosen or word in buffer.pins:
            return []
        buffer.pins[word] = chosen
        return [Question(
            doubt.question, "sense", word, chosen, pins=((word, chosen),),
            why=f"asked again with “{word}” held to {chosen}, the sense that "
                f"carries the word itself. The first reading resolved it to "
                f"{doubt.concept or 'something else'}, and an answer about "
                f"the wrong thing is worse than no answer.",
            parent=doubt.question, salience=1.0, gain=1.0,
            urgency=attention.URGENCY["sense"],
            depth=buffer.depth_of(doubt.question) + 1)]

    def other_sense_answers(self, answer, buffer) -> list[Question]:
        """A no that another reading of the same word would have said yes to.

        `is a mouse an animal` comes back CONTRADICTED, correctly, about
        `mouse.n.04` -- the device. R27 excludes it because an artifact is
        not an animal, and the exclusion is sound. It is also about the
        wrong mouse: `mouse.n.01` is under animal, and nothing said so.

        The name-matching check misses this because both senses are called
        `mouse`. What separates them is that one of them answers the question
        and the other does not, which costs one call to v687's classifier per
        sense and is worth it on a denial.
        """
        if answer.origin != "seed" or answer.verdict not in ("CONTRADICTED",
                                                             "DENIED"):
            return []
        parse = (answer.payload or {}).get("parse") or {}
        subject, target = parse.get("subject"), parse.get("target")
        if not subject or not target or parse.get("relation") != "is_a":
            return []
        chosen = (answer.payload or {}).get("concept") or ""
        for sense in (self.engine.reasoner.senses_of(subject) or [])[:6]:
            name = sense.get("id") or ""
            if name == chosen or name.split(".")[-2:-1] != ["n"]:
                continue
            found = self.engine.reasoner.classify(name, target)
            if getattr(found, "verdict", "") not in ("VERIFIED", "HELD"):
                continue
            if subject in buffer.pins:
                return []
            buffer.pins[subject] = name
            return [Question(
                answer.question, "sense", subject, name,
                pins=((subject, name),),
                why=f"read as {chosen} the answer is no, and correctly so. "
                    f"But {name} — “{sense.get('definition') or name}” — is "
                    f"another reading of “{subject}”, and it answers yes. "
                    f"Which of the two you meant is not something the store "
                    f"can tell.",
                parent=answer.question, salience=1.0, gain=1.0,
                urgency=attention.URGENCY["sense"],
                depth=buffer.depth_of(answer.question) + 1)]
        return []

    def better_sense(self, word: str, taken: str) -> str:
        """A reading of the word more obvious than the one v687 took.

        This used to prefer whichever sense was *named* after the word, which
        was right while `pig` resolved to `pig bed.n.01` and became wrong the
        moment v687 was fixed: it then overruled `hog.n.03`, domestic swine
        and WordNet's first reading, in favour of `pig.n.06`, a crude block
        of metal that happens to be spelt like the word.

        The test is WordNet's own order for the lemma. Only a sense the
        dictionary lists *earlier* than the one v687 took is a candidate, so
        a fixed default is never argued with.
        """
        senses = self.engine.reasoner.senses_of(word) or []
        here = next((one for one in senses if one.get("id") == taken), None)
        floor = here.get("rank") if here else None
        if floor is None or floor >= 90:
            return ""
        for sense in senses:
            rank = sense.get("rank")
            if rank is None or rank >= floor:
                continue
            if (sense.get("id") or "").split(".")[-2:-1] == ["n"]:
                return sense["id"]
        return ""

    # -- source 4: does the subject have what the act needs? ---------------
    def from_requirement(self, answer, buffer) -> list[Question]:
        """Check a capability against what doing it turns out to need.

        `do fish run` used to come back VERIFIED, on `animal capable_of "could
        run"`. R19 and R28 stopped that, and it is UNKNOWN now. Rather than
        take that or leave it, work out what running needs -- the things the
        store says can run have a **leg**, 12 of 90 of them, 102 times
        commoner than among concepts at large -- and put that to the fish.

        Three steps, and each needs the one before it: the claim has to come
        back before it is worth grounding, the requirement has to be derived
        before it can be asked, and the answer to `does a fish have legs` is
        itself inherited from `animal`, so the family check takes over from
        there and finds that sharks, salmon and goldfish are all scored and
        denied. Nineteen workers shorten none of it.
        """
        if answer.origin not in ("seed", "gap"):
            return []
        # A denial is worth grounding too, and differently: a penguin cannot
        # fly and *does* have the wings flying needs, which says the answer
        # is not about anatomy.
        #
        # And silence most of all. When v687 has nothing to say about whether
        # a fish runs, what running *needs* is the only route left to an
        # answer, and it is the route a person would take. Excluding UNKNOWN
        # here meant the one question this generator was written for stopped
        # reaching it the moment v687 stopped over-affirming: the loop spent
        # its cycles defining "run" and wondering whether a fish is slimy.
        if answer.verdict not in ("VERIFIED", "HELD", "INHERITED",
                                  "CONTRADICTED", "DENIED", "UNKNOWN"):
            return []
        parse = (answer.payload or {}).get("parse") or {}
        # The word that was said, not the sense v687 resolved it to. Asked
        # `do pigs fly`, v687 takes `pig` to mean `pig bed.n.01` -- a foundry
        # mould -- and building the follow-up from the concept asked `does a
        # pig bed have wings`, which opened a gap on `bed` and spent a cycle
        # wondering whether a bed is furry. Which sense to read a word in is
        # v687's to decide; the question text is not.
        subject = bare(parse.get("subject") or answer.about
                       or (answer.payload or {}).get("concept") or "")
        action = self.lemma((parse.get("target") or "").split()[0]
                            if parse.get("target") else "")
        if not subject or not action:
            return []
        needs = self.requirements.of(action)
        if needs is None or needs.part == subject:
            return []
        # A check that cannot come back other than yes is not a check.
        # `does a beagle swim` derived `tooth` -- swimmers have teeth,
        # 15 of 54, because swimmers are animals -- and asked `does a beagle
        # have teeth`, which the store settles before a worker is spent on
        # it and which grounds nothing when it comes back.
        #
        # A denial is the exception, and it is the reason this is not simply
        # dropped: `can a penguin fly` is a no, and confirming the penguin
        # *does* have the wings flying needs is what says the no is not about
        # anatomy. There the foregone yes is the finding.
        settled = self.requirements.recorded_of(
            (answer.payload or {}).get("concept") or "", needs.part)
        if settled and answer.verdict not in ("CONTRADICTED", "DENIED"):
            return []
        many = plural(needs.part)
        return [Question(
            f"does {article(subject)} {subject} have {many}", "require",
            subject, needs.part,
            why=f"the things the store says can {action} have "
                f"{article(needs.part)} {needs.part} — {needs.holders} of "
                f"{needs.doers} of them, {needs.lift:.0f} times commoner than "
                f"among concepts at large. That is what they have in common, "
                f"not a proof that {action}ning needs it; this asks whether "
                f"{article(subject)} {subject} has {many} too.",
            parent=answer.question, salience=1.0, gain=0.95,
            urgency=attention.URGENCY["require"],
            depth=buffer.depth_of(answer.question) + 1)]

    # -- source 4: the family disagreed, so ask which side the subject is on
    def from_split(self, split: dict, buffer) -> list[Question]:
        """Settle a divided family by comparing the subject with the dissent.

        `is a shark a fish` puts `has scales` to six kinds of fish. Goldfish,
        minnow and salmon have them; a seahorse does not; the shark itself is
        unrecorded. Counting that up as "1 of 6 deny it" is not an answer --
        the question it actually raises is whether the shark is more like the
        minnow or more like the seahorse, and R21 is the rule that answers it.

        This is the level that has to be serial. The comparison cannot be
        named until the fan-out comes back, so nineteen workers buy the split
        in one cycle and then wait while one worker resolves it.
        """
        subject = split["subject"]
        if not subject:
            return []
        urgency = attention.URGENCY["split"]
        salience = max(buffer.activation.salience(subject), 0.6)
        depth = buffer.depth_of(split["parent"]) + 1
        asked: list[Question] = []
        for other in (split["differ"][:1] + split["agree"][:1]):
            if other == subject:
                continue
            asked.append(Question(
                f"what is the difference between {article(subject)} "
                f"{subject} and {article(other)} {other}", "split", subject,
                split["claim"],
                why=f"the family split on “{split['claim']}” — "
                    f"{', '.join(split['agree'][:3]) or 'some'} yes, "
                    f"{', '.join(split['differ'][:3])} no. Which side "
                    f"{subject} is on is the question that answers, and it "
                    f"could not be asked before the split came back",
                parent=split["parent"], salience=salience, gain=0.95,
                urgency=urgency, depth=depth))
        return asked

    # -- source 4: the answer's own content names the next question --------
    def from_content(self, answer, buffer) -> list[Question]:
        """A chain of thought: the next subject comes out of the last answer.

        This is the one source the pool cannot help with. A gap or a doubt can
        be fanned out -- the family goes to nineteen workers at once -- but
        here the *terms* of the next question are inside the previous answer,
        so step three cannot be formed until step two comes back. One worker,
        one step per cycle, however many engines are idle.

        Three kinds of content are worth following, and each is a different
        rule of v687 answering:

            a definition    its genus is the next thing to ask about, and the
                            ladder beagle -> hound -> hunting dog -> dog ->
                            canine is a chain nothing could have predicted
                            from the utterance.       (R26)
        Only the definition ladder is left of the three that were tried.
        Grounding an inheritance asked a taxonomic question nothing doubted;
        following an explanation walked into whichever sense the crawl had
        used. Both are recorded below where they were removed.
        """
        depth = buffer.depth_of(answer.question)
        if depth >= MAX_CHAIN_DEPTH or answer.error:
            return []
        if answer.origin not in ("seed", "gap", "chain", "require"):
            # A chain is a line of reasoning about what was said. Starting one
            # from a corroboration answer walks away from the utterance: a run
            # about whales went `does a goldfish have a gill` -> `is a
            # goldfish a bony fish`, which is true and about nothing.
            return []
        payload = answer.payload or {}
        salience = max(buffer.activation.salience(answer.about), 0.5)
        urgency = attention.URGENCY["chain"]
        # A chain narrows as it goes: each step is one level further from what
        # was actually said, and should not outrank a fresh doubt about it.
        # Floored, though: a line of reasoning already under way should not be
        # interrupted by a new interest, and without the floor a wide pool
        # reached one rung less than a narrow one -- more curiosity questions
        # to outbid the fourth rung, purely because there were more workers.
        fade = max(0.85 ** (depth + 1), 0.62)
        asked: list[Question] = []

        definition = payload.get("definition") or {}
        genus = bare(definition.get("genus") or "")
        # A ladder that changes sort has stopped being about its subject.
        # `why does a dog bark` returned an explanation mentioning vomiting,
        # and the ladder walked vomiting -> expulsion -> propulsion -> force:
        # four real definitions, none of them about dogs. R26 already reports
        # which top partition a word sits under, so the rung has to match the
        # one the chain started in.
        sort = bare(definition.get("sort") or "")
        started = buffer.sort_of_chain(answer.question) or sort
        # And only when a definition is what was asked for. `does a snake
        # run` produced a coverage gap on `snake`, whose definition v687 read
        # as the winding-river sense, and the ladder walked river -> stream ->
        # body of water: three real definitions, none about snakes. Defining
        # the words in your own follow-ups is not reasoning.
        if (genus and genus not in LADDER_FLOOR and sort == started
                and buffer.wants_a_definition):
            asked.append(Question(
                f"what is {article(genus)} {genus}", "chain", genus,
                why=f"“{definition.get('word') or answer.about}” was defined "
                    f"as a kind of {genus}, so {genus} is what the definition "
                    f"leans on and nothing here has said what it is",
                parent=answer.question, salience=salience, gain=fade,
                urgency=urgency, depth=depth + 1))
            buffer.note_sort(f"what is {article(genus)} {genus}", started)

        # The inheritance grounding used to live here -- `does a goldfish
        # have a gill` rests on `bony fish has_a single gill`, so ask whether
        # a goldfish is a bony fish. It is factually right and asks the wrong
        # half. Goldfish being a bony fish is WordNet taxonomy at 0.95 and
        # was never in doubt; what is doubtful is the fact itself, at
        # confidence 0.37 and five levels up. The doubt generator already
        # puts that to the ancestor, which is the half that can be wrong.
        return asked

    # -- source 4: nothing is wrong, but the trie has a child --------------
    def from_curiosity(self, concept: str, buffer, limit: int = 6
                       ) -> list[Question]:
        """The predicates that would tell this concept from its rivals."""
        salience = buffer.activation.salience(concept)
        if salience < attention.FLOOR or not self.substantial(concept):
            return []
        if buffer.barren(concept):
            return []
        urgency = attention.URGENCY["curiosity"]
        anchor, pool = self.pool_for(concept)
        if not anchor and not self.wider.expectations(
                self.wider.sense_of(concept) or "", 1):
            # Nothing in the norms is near this word, so there is no field for
            # a question to split and the gain scores are meaningless. `swim`
            # became live as the object of `does a beagle swim` and produced
            # `can a swim be airborne`. Being interested in something requires
            # having something to be interested about.
            return []
        field_note = ("" if anchor == concept else
                      f", judged against the kinds of {anchor}"
                      if anchor else ", judged against the whole corpus")
        asked: list[Question] = []
        # The family first, where there is one. The feature norms cover 541
        # concepts and give the same six columns to all of them; the graph
        # covers 45,219 and gives this concept's own relatives. An
        # expectation is a better question than a slot: `every other bowed
        # instrument is used to make music -- is a violin?`
        sense = self.wider.sense_of(concept)
        for expects in (self.wider.expectations(sense, 3) if sense else []):
            text = self.reask(concept, expects.relation, expects.object)
            if not text:
                continue
            asked.append(Question(
                text, "curiosity", concept, expects.object,
                why=f"{round(expects.share * 100)}% of the other kinds of "
                    f"{bare(expects.parent)} are recorded as this, and "
                    f"{concept} is not — so it is either an exception or a "
                    f"hole in the store",
                salience=salience, gain=0.9 * expects.share, urgency=urgency,
                novelty=self.curiosity.novelty(concept)))

        for predicate, gain in self.curiosity.candidates(
                concept, limit * 3, pool):
            pos = self.pos_of(predicate) if " " not in predicate else ""
            if " " not in predicate and not pos:
                # An AwA2 coinage the ontology has no word for -- `oldworld`,
                # `quadrapedal`. It is a real column of the corpus and a
                # nonsense question, so it is skipped rather than asked.
                continue
            words = predicate.split()
            head = self.lemma(words[0]) if len(words) > 1 else ""
            text = phrase_predicate(concept, predicate, pos,
                                    self._singular.get(predicate, ""), head)
            if not text:
                continue
            asked.append(Question(
                text, "curiosity", concept, predicate,
                why=f"“{predicate}” splits the field most evenly{field_note}, "
                    f"so the answer removes the most candidates",
                salience=salience, gain=gain, urgency=urgency,
                novelty=self.curiosity.novelty(concept)))
            if len(asked) >= limit:
                break
        return asked

    # -- the queue --------------------------------------------------------
    def queue(self, buffer, width: int) -> list[Question]:
        """Everything worth asking this cycle, best first, deduplicated.

        Gaps and doubts are generated from what came back last cycle;
        curiosity fills whatever room is left. That ordering is the whole of
        executive control at this stage -- a sort, not a subsystem -- and it
        is what stops curiosity drowning out a blocked word.
        """
        candidates: list[Question] = list(buffer.carried)
        for hole in buffer.open_gaps():
            candidates.extend(self.from_gap(hole, buffer))
        for doubt in buffer.open_doubts():
            candidates.extend(self.from_doubt(doubt, buffer))
        for doubt in buffer.open_senses():
            candidates.extend(self.from_sense(doubt, buffer))
        for answer in buffer.recent:
            candidates.extend(self.other_sense_answers(answer, buffer))
        for answer in buffer.recent:
            candidates.extend(self.from_requirement(answer, buffer))
        for split in buffer.splits():
            candidates.extend(self.from_split(split, buffer))
        for answer in buffer.recent:
            candidates.extend(self.from_content(answer, buffer))

        ranked = sorted(candidates, key=lambda q: -q.rank)
        chosen: list[Question] = []
        seen: set[str] = set()
        for question in ranked:
            if question.key in seen or buffer.already_asked(question.key):
                continue
            seen.add(question.key)
            chosen.append(question)
        # Whatever a cycle could not hold waits for the next one, ahead of any
        # new curiosity. Without this the pool size silently decides which
        # members of a family get asked, and the conflict a wide pool finds is
        # one a narrow pool never sees.
        buffer.carried = chosen[width:]
        chosen = chosen[:width]

        # Curiosity fills the rest of the pool. It is generated last and
        # ranked last on purpose: the workers would otherwise be busy being
        # interested while something was blocked.
        if len(chosen) < width:
            spare: list[Question] = []
            for concept in buffer.topics[:4]:
                spare.extend(self.from_curiosity(concept, buffer))
            for question in sorted(spare, key=lambda q: -q.rank):
                if question.key in seen or buffer.already_asked(question.key):
                    continue
                seen.add(question.key)
                chosen.append(question)
                if len(chosen) >= width:
                    break
        return chosen[:width]
