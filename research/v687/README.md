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
| `test_v687.py` | 49 tests for the above |
| `test_trie/reasoning/bridging/norms.py` | v683–v686's 200 tests, unchanged |

Everything else is v683 through v686, copied and re-imported locally.
