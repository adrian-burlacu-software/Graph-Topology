# V687: the reasoning the audit found missing

```bash
python -m research.v687             # the page, on port 8687
python -m research.v687.run_v687    # the compression experiment, unchanged
```

Self-contained: v683 through v686 are copied in, not imported, because the
backlog changes things at every level — the parser, the rules, the trie reader
and the page. All 200 of their tests come along and pass unchanged.

The v686 audit put 34 question types to the engine and found what it could not
answer. This closes that list. Each item is a rule, and the page has a
selector with worked examples for every one.

## The defect that started it

Three questions came back **VERIFIED** that the system could not answer at
all: `does a dog have a tail and wings`, `is a dog furry or hairless`, `is a
whale bigger than a dolphin`. Each was answered from whichever part of the
question happened to be understandable. Silent wrong answers are worse than
refusals, and every one had the same shape.

## R18 — refuse by name

A construction no rule covers is declined and named, rather than answered from
the half of it that parses. Comparatives, superlatives, counts of parts and
counterfactuals are refused with the reason.

The reason matters: **there are no magnitudes anywhere in this data**. AwA2
records `big` as a yes or no; XCSLB records `is the largest animal` as a
string. There is no scale to compare on, and saying so is the answer.

`how many kinds of dog` is *not* refused — the taxonomy can count kinds. Only
counting parts is impossible.

## R20 — three values, not two

A question has structure, so it is parsed into a tree and evaluated in
Kleene's strong three-valued logic. Two values will not do: this system has
three real answers — stated, denied, silent — and silence is not falsehood.

| question | answer | why |
| --- | --- | --- |
| `does a dog have a tail and wings` | UNSETTLED | one unknown conjunct suspends the whole |
| `is a dog furry or purple` | YES | one true disjunct settles it |
| `is a dog furry and telepathic` | UNSETTLED | not false — unknown |
| `do all birds fly` | NO | 21 of 29 do; emu and chicken do not |
| `do most birds fly` | YES | the same count, a different question |

Quantifiers ask every kind beneath a concept and count. `bird` is not one of
XCSLB's 521 concepts, so a class the norms do not cover but WordNet does now
resolves — without it the showcase question fell through to a single crawled
sentence that answered it wrongly.

## R19 — corroboration

Found while testing R20, and worse than the defect it interrupted:
`animal.n.01 has a wing` made **every dog winged**. An inherited fact is now
put to the ancestor's other kinds, which the norms did elicit:

| fact | borne out by | verdict |
| --- | --- | --- |
| `bird.n.01` — fly | 21 of 29 kinds (72%) | inherited |
| `animal.n.01` — wings | 20 of 143 (14%) | refused |
| `carnivore.n.01` — wings | 0 of 24 (0%) | refused |

Below eight kinds nothing is refused: four whales are not evidence about
whales. And **votes from below must be independent** — three dog breeds
re-inheriting that same sentence had counted as three witnesses, so induction
from the kinds now counts stated evidence only.

## R21 — contrast

Four question types that look unrelated are one operation, the lowest common
ancestor of two branches:

    what do a dog and a cat have in common     the shared prefix
    what is the difference between them        where the branch splits
    what is similar to a dog                   how far others walk with it
    is a dalmatian a typical dog               how far it walks with its class

Two measures are reported because they disagree, and **the disagreement is the
finding**: `killer whale` and `blue whale` share 46% of their properties and
part at the *first node* of the stored trie. A trie built for storage does not
group by similarity — the same result as v686's storing-against-asking, one
axis over.

Typicality reports whether it is sound. `dalmatian` scores 93% of what a dog
is, ranking 1 of 3. `robin` against `bird` is refused: the most ordinary bird
reaches only 44%, because XCSLB elicitation is free and sparse and no two
people list the same things. Dense corpora support the measure and sparse ones
do not, which is the same split as everywhere else here.

## R22 — the graph backwards

Every question v684 through v686 could answer starts from a named subject. The
store is a forward index, and that shape was baked into the questions. The
audit counted 46,883 facts reachable by no question; the larger number is that
**all 1.7M were reachable from one direction only**.

    what is made of wood        baseball bat, hilt, picket fence, chopstick
    what is found in a toolbox  drill, file, power saw, screwdriver, hammer
    who makes a car             bmw, ford, audi
    what has wings              bird, bat, angel, turkey

No new index was needed, which was worth checking rather than assuming:
`idx_facts_relation` plus a `LIKE` prefilter answers in about 0.15s. Pulling
every `capable_of` row into Python instead took 2.1s.

Two things had to be right. A paired relation is read from the *other column*
— searching `part_of` for an object of "wings" answered `what has wings` with
the aileron and the flight feather, which are parts of a wing. And a fact that
*denies* the phrase is not an answer: `vegetarian` turned up under `what eats
meat` on the strength of not eating any.

## R23 — scripts and abduction

The eventive relations nothing had ever asked: 12,270 `has_subevent`, 11,970
`has_prerequisite`, 11,114 `causes`, 2,158 `motivated_by_goal`.

    what happens when you drive a car
      before   buy car, driving permit, fasten seatbelt, get car
    why do people sleep
      in order to   body requires, dream, exhausted

**Scripts are word-level, and this says so.** Choosing a sense was tried twice
and is not recoverable: ConceptNet records these of a *word*, and the build
attached them to whatever synset it could find, so `cook.n.02` is Captain Cook
and carries "buy cookbook", while `bark.n.01` is the covering of a tree. Lesk
against the question's own words barely moves the ranking, because the flaw is
upstream. Between two mistaken senses there is nothing to choose, so every
sense of the word is read together and the answer is labelled word-level — the
distinction R12 already draws for inheritance.

**Abduction was already here, unnamed.** `what is round with hexagons →
football` is inference to the best explanation of observed features: R16
restricted to identity. What is added is abduction over causes, ranked as
competing hypotheses rather than listed:

| cause of a fire | score | also causes |
| --- | --- | --- |
| space heater | 0.240 | 1 |
| wiring | 0.206 | 1 |
| cigarette butt | 0.191 | 1 |
| short circuit | 0.159 | 6 |

Scored by specificity (a cause that causes forty things explains none of them
— `anti_coverage`, the same measure that makes `hexagons` a better question
than `round`), directness, and confidence.

## R24 — analogy, over the norms only

The v686 audit put analogy on the skip list, asserting the data could not
support it. That was asserted without testing and the reason was wrong.
Testing separates two cases cleanly:

**The scraped graph cannot.** Its `part_of` is largely taxonomy misfiled as
mereology — `acanthisitta.n.01 part_of rifleman bird.n.01` — and `bird
has_part` returns `band`, `broken wing`, `enough room`, `feast`.

**The norms can**, because XCSLB ships a *feature type* for every property,
which is the missing half of a role. A mapping needs the same kind of property
at the same standing within its own concept:

    bark : dog :: ? : cat        ->  can meow, can purr
    fins : fish :: ? : bird      ->  has a beak, has big wings, has feathers

Same dense-and-closed versus sparse-and-open split as compression, typicality
and everything else here. That consistency is the point.

## R25 — counting kinds

R18 refuses `how many legs does a dog have` and says in the same breath that
the taxonomy *can* count kinds. It could not: nothing implemented it, so the
question its own refusal offers as the answerable one fell through to a
listing about the word `kind`. A rule that advertises an exception it does not
have is worse than one that refuses everything.

    how many kinds of dog are there
      189 in the taxonomy, 18 directly beneath it, 3 described by the norms

Two counts, because they are different questions. WordNet names 189 kinds of
dog; the feature norms describe three of them, and those three are the only
ones any other rule here can reason about. Reporting the big number alone
would imply a coverage that does not exist.

## R26 — what a thing is

A taxonomy's own question, and the one it did not answer. `what is a robin`
fell through to a property listing and came back **helpful, passionate,
professional, kind** — which ConceptNet records of the *name* Robin. The
answer was about a person called Robin and nothing said so.

    what is a robin
      robin.n.01 — small Old World songbird with a reddish breast
      a kind of thrush; the norms describe it directly

Genus and differentia, both read off the structure: the genus is the immediate
hypernym, which the taxonomy has exactly, and the rest is the gloss, the kinds
beneath, and whether the norms cover it — because "every property question
about this is answerable" is part of knowing what a thing is here.

A word with no sense in the store is named as unknown rather than answered
about something else. That matters for a dialogue that has to *ask*: a system
that cannot say "I do not know what a wemble is" cannot be told.

## Rated norms — a no that is not a silence

The crawl cannot say a chair is not alive, and nothing else could either:
XCSLB is a free listing and AwA2 covers 50 animals. `rated.py` reads two
sources that put one fixed question to every object they cover:

    THINGSplus   1,854 objects rated 1-7 on lives, manmade, natural, heavy;
                 real-world size; 53 everyday categories
    NEWTON       777 household objects voted Low or High on softness,
                 sharpness, brittleness, elasticity, malleability, stiffness,
                 surface smoothness and surface hardness

    is a chair alive      no: people rated a chair 1.5 of 7 as something that lives
    is a pillow sharp     no: NEWTON's annotators voted it Low, 100% agreement
    is a tomato a fruit   yes, as people sort things -- WordNet files it elsewhere

**They are a layer of their own, not more norms.** `identify.stated` is what
R19 counts, and merging 2,000 rated objects would put THINGSplus's birds in
`bird.n.01`'s denominator without anyone having asked them about flying: fly
would fall from 28 of 29 to about 28 of 50, under the floor. So the ratings are
keyed by synset and read only for the predicates they rated, in
`Profiles.verify_one` beside AwA2's zeros and in `reasoning._rated` for
subjects the norms do not name. Folk categories answer only after the taxonomy
has found nothing, and only yes.

The join and every threshold are measured in `rated.py`'s docstring and in
`data/thingsplus.SOURCE.md` and `data/newton.SOURCE.md`. THINGS rates `mouse`
twice, the animal and the device, so a word whose rated readings disagree gets
no answer from them at all.

## R31 — bigger and heavier

    is an elephant bigger than a mouse       yes: 376 against 191
    which is heavier, a feather or a brick   a brick: 5.5 of 7 against 1.1
    is a cheetah faster than a turtle        refused by R18: no scale for speed

R18 refused every comparative because nothing had a magnitude. THINGSplus has
two: a size scale anchored from a grain of sand (109) to an aircraft carrier
(421), and heaviness rated 1-7. Two objects rated within 20 units of size or a
point of weight are called too close. Anything not two rated objects on one of
those scales still reaches R18, whose note now says what R31 would have
needed.

## R32 — only the living

    does a rock breathe    no: VerbNet gives `breathe` a living doer, and a
                           rock is rated 1.5 of 7 as something that lives
    can a computer think   left to the store, which says it can

VerbNet restricts who can be a verb's subject, and for `breathe`, `eat`,
`drink` and `think` every class the verb is in wants an animate doer. Joined to
THINGSplus's "not alive" that is a no neither source states. It stands down
when any reading of the word is alive, and when the store says the thing does
exactly that -- the disagreement test AwA2's zeros already face.

## The answer audit

Ninety-two questions across seventeen question shapes, plus five sweeps that
need no hand labels — take the corpus's own claims, ask them back as questions,
and see whether the answers agree with the data they came from. Six findings,
all now fixed and each pinned by its own test in `AnswerAuditTests`.

**F1 — a denial is not a denial of every word inside it.** `_hit` matched any
whole word inside a predicate, and the denied set was matched by the same
function as the held set. The norms record `has small ears` as false of a
beaver, so:

    does a beaver have ears   ->  CONTRADICTED
    does a horse have teeth   ->  CONTRADICTED
    is a wheel part of a car  ->  CONTRADICTED   (`has two wheels`)

**20,359** denied predicates across all 541 concepts had a modifier and a head
noun appearing in no held predicate of that concept — every one would answer a
question about the head noun with a confident no. This was the only place the
system was confident *and* wrong, which is what R8's three values exist to
prevent.

A denial now answers a question only when the question covers what the denial
claims. The exception is a locative tail: `has spots on its body` says *where*,
not *which*, so it still denies spots — which is the difference between a dog
and a dalmatian, and the example this whole module is built on.

**F2 — the subject is the thing asked about, or nothing.** The subject scan
required a noun and would read past what it could not place, so an
unrecognised subject was quietly answered about the thing it was compared to:

    is hello a greeting     ->  a list of the properties of greetings
    is a wemble an animal   ->  a list of the properties of animals
    is hello a word         ->  an answer about bible.n.01

Two fixes. A subject need not be a noun — `hello` tags as an interjection and
`running` as a verb, and both are concepts here. And a determiner marks the
seam of a copula, so the scan stops at the predicate and names the word it
could not place instead of reading on. `is a wemble an animal` now answers
UNKNOWN_WORD, naming `wemble`.

The same seam settles a bare noun predicate. `is a chair furniture` names a
kind and `is a raccoon white` names a property, and they are the same shape:
the reading is hedged, the norms are asked first, and the taxonomy takes it
only if they are silent.

**F3 — a class is only a class within one vocabulary.** `animal`'s core came
out `oldworld, quadrapedal, ground` — pure AwA2, because AwA2 gives every one
of its 50 animals the same 85 attributes while XCSLB is free elicitation. All
**100** of the 143 kinds of animal that come from XCSLB scored zero against it:

    is a dog a typical animal  ->  CONTRADICTED, 0%, ranking 67 of 143

Typicality is now measured within one corpus, and a class whose kinds agree
about nothing says so rather than ranking against an empty set.

**F4 — subsumption is reflexive.** `is a bee a bee` answered UNKNOWN after
walking the whole taxonomy. Zero of forty reflexive questions verified.

**F5 — a script cannot ignore who the question named.** `what happens when a
beaver moves` and `what happens when a piano moves` returned byte-identical
answers: the event is what ConceptNet records these relations of, and the doer
was dropped without trace. There is no beaver-specific script to invent, so
the doer is accounted for instead — the facts tying it to the event, or the
plain statement that nothing does and this is the generic script.

**F6 — R18 refused four constructions; the open set was larger.** Dates,
questions about words rather than senses, facts about named individuals and
antonyms are now refused by name too. `who invented the telephone` was being
answered `genius.n.04 capable_of invent telephone`.

### What the sweeps say

| sweep | before | after |
| --- | --- | --- |
| stated property asked back | 60/60 | 60/60 |
| denied property asked back | 57/60 | 57/60 |
| subject resolution | 51/60 | 51/60 |
| reflexivity | 0/40 | **37/40** |
| invented words given a confident answer | 4/36 | **0/36** |

The nine subject-resolution "misses" are synonym normalisation — `donkey`
becomes `domestic ass`, `budgie` becomes `budgerigar` — which is correct. The
three remaining reflexive failures are the sweep feeding corpus names verbatim:
`is a glove a glove` and `is a roller skate a roller skate` both verify, while
`rollerskate` is not a WordNet lemma and is correctly refused as an unknown
word.

## R6, the other way round: choosing the sense

Every rule here has always run per sense rather than per word string — R6, and
the reason v684 has evidence-weighted disambiguation at all is that picking
badly once sent 348 facts about carpenters' toolboxes to the part of a gunlock.
What was missing was any way for the reader to *see* which sense a word was
taken in, or to say it was the wrong one. The old sense card offered that for
the subject alone, after the answer, and only for the rules that accepted a
`concept` parameter — which none of the new ones do.

Under the question box there is now one chip per content word, showing the
sense it is being read in. Hovering opens the senses it could carry, with each
one's definition and how many facts it holds; clicking one pins it and re-asks.
Pins persist, because meaning `mouse` the animal is a fact about the reader
rather than about one question.

    how many kinds of mouse are there
      unpinned   mouse.n.04, the device        0 kinds
      pinned     mouse.n.01, the animal       11 kinds

    is a hammer a tool
      unpinned   hammer.n.02, the tool         VERIFIED
      pinned     hammer.n.01, the gunlock      "tool is not among the 11
                                                ancestors of hammer.n.01"

**Where a pin bites, and where it does not.** A control that looks as though
it works everywhere and only works in places is worse than none, so the flyout
says which it is for each word:

| | pinnable |
| --- | --- |
| v684's subject | yes — the choice the old card made |
| R23's event | yes — this is the one that reads `bark` as tree bark |
| R16's class, R21, R24, R25 | yes — `class_concept` was taking the primary sense |
| R22's phrase | yes — as an identity on the subject side, as extra spellings on the object side |
| R17's profile | **no**, and deliberately: XCSLB ships a sense key with each of its concepts, so the join is given rather than guessed and there is nothing to overrule |

Pins are request-scoped through a context variable, because the server is
threaded and two readers pinning different senses of `mouse` at the same
moment must not see each other's choice.

One bug is worth recording. Feeding a pin in as v684's `concept` parameter
looked obviously right and disabled every rule above v684: `concept` is a
*gate* meaning "the reader clicked a sense under an answer, so re-answer that
one thing", and setting it from a pin stopped `why does a dog bark` being a
why-question at all. The pin now travels separately and only reaches the
fallback.

## The audit that produced these

Every question the page offers was put back through the engine and read, not
spot-checked. Four routing errors turned up, each of which looked like a
working answer:

| question | was answered by | should be |
| --- | --- | --- |
| `what is a hammer made of` | R22, listing things made *of* hammers | v684, forward |
| `what is a hammer used for` | R22, the same way | v684, forward |
| `what does a car driver need` | R22, one fact about cars | v685's bridge |
| `what kind of animal is furry…` | R25, counting animals | R16, identification |

The first two came from letting a backwards question take its phrase from
either side of the cue. A named subject *before* the cue means the question is
forward, so the object side is now read only from the tail. The third came
from deciding "did anyone answer this" by looking at the answer — an empty
evidence list meant nobody had, which is false for identification, and a
non-empty one meant somebody had, which is false for v684's willingness to say
something about any noun. Question shape settles it instead. The fourth was
R25's own greed, fixed by requiring the question to end where a count ends.

## The page

One card, `How this was answered`, replaced three. Identification, the
attribute walk and the four new rules each had their own, and any two showing
at once said the same thing twice — a verify answer about one property came
with the whole 28-row attribute walk beside it. The walk is still there, folded
away, because it is context for a verdict and the subject of a profile.

The generic fact table no longer repeats a profile's 50 rows underneath the
card that already lays them out. `Set aside` holds both kinds of
deliberately-unused evidence — R14's sibling facts and the near misses — since
they are the same kind of thing and neither is an answer.

**Every rule now draws its own derivation.** The graph used to appear only for
an identification; the other seven produced a wall of text. Each rule now says
what its chain, its side nodes and its result are, and the drawing that
already existed renders it — the difference between a dog and a wolf comes out
as a trunk of shared properties with each animal's own hanging off the point
they part.

The steps are *derived from* that drawing rather than written beside it. The
replay lights a node by looking its name up among the drawn ones, so writing
both by hand is how they drift apart and light nothing; a test asserts every
step of every new answer names a node that exists.

## Layout

| file | role |
| --- | --- |
| `logic.py` | the query tree, Kleene's three values, R18's refusals |
| `contrast.py` | difference, commonality, similarity, typicality |
| `inverse.py` | the fact graph read from the object side |
| `causal.py` | scripts in script order, and abduction as a ranking |
| `analogy.py` | role mapping over the closed norm vocabulary |
| `reasoning.py` | the engine, and the order questions are tried in |
| `rated.py` | THINGSplus and NEWTON joined to the store: a no that is not a silence, and R32 |
| `magnitudes.py` | R31, comparatives on the scales THINGSplus rated |
| `test_rated.py` | the rated norms, R31 and R32 |
| `pins.py` | the reader's chosen sense, for one request |
| `test_v687.py` | 58 tests for the above |
| `test_trie/reasoning/bridging/norms.py` | v683–v686's 200 tests, unchanged |

Everything else is v683 through v686, copied and re-imported locally.
