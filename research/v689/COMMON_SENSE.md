# Common sense about kinds

2026-09-13, on `v689/common-sense`. What the page could not say about ordinary
things -- is a chair alive, is an elephant bigger than a mouse, can you cut
bread with a knife -- where each broke, the data that was brought in, and what
changed. Time and events are left out on purpose: there is no architecture for
them yet.

## Method

114 single questions about kinds, sent to the running page (`/api/say`) in
the example sandbox and forgotten afterwards, in 25 families: size and weight
said and compared, fitting inside, material, temperature, buoyancy, state,
colour, shape, texture, what a thing is made of, what it can be used for and
with, edibility and safety, needs, likes, where things are kept, living and
made things, perception and feeling, abilities, parts, and the categories
people use. The probe and its diff are session scripts (`probe_common.py`,
`diff_common.py`); each run is compared to the one before it.

## What it came to

1. **Nothing could say no.** `is a chair alive`, `does a rock breathe`, `can a
   table think`, `is a whale a fish`: every one came back UNKNOWN, "not
   recorded; the teacher says no". The crawl records what things do, never what
   they do not, XCSLB is a free listing whose silence is what nobody happened
   to say, and AwA2's closed matrix covers 50 animals.
2. **Every comparative was refused**, correctly -- nothing held a magnitude.
3. **`you` was the program.** `can you cut bread with a knife` was asked as
   `can a computer program cut bread with a knife`, and `what do you use to
   cut paper` was a listing about paper.
4. **Being somewhere was a property.** `is a fridge in a kitchen` asked whether
   a fridge has the property `in a kitchen`; the store has `electric
   refrigerator at_location kitchen`.
5. **Folk categories were absent.** `is a tomato a fruit` is UNKNOWN because
   WordNet files the tomato under `solanaceous vegetable`.

## The data

| dataset | what it is | taken | why |
| --- | --- | --- | --- |
| THINGSplus | 1,854 objects rated 1-7 on lives, manmade, natural, heavy; real-world size; 53 categories with typicality | yes, `data/thingsplus` | a closed question put to every object, so a low rating is a no; the only magnitudes available |
| NEWTON | 777 household objects voted Low/High on eight physical attributes | yes, `data/newton` | the same shape for softness, sharpness, brittleness and the rest |
| VerbNet 3.3 | who can be a verb's subject | already in `data/verbnet3.3` | turns "not alive" into "does not breathe" |
| VerbPhysics | relative size, weight, strength of verb arguments | downloaded, removed | its 217 objects are `person`, `head`, `way` -- not kinds anyone asks about |
| DoQ | measured distributions over size, weight, duration | not available | the bucket is not publicly listable |
| Quasimodo | commonsense from query logs, 944 MB | not taken | a crawl like Ascent++; the candidate for likes (`does a cat like milk`) |
| ATOMIC 2020 | if-then knowledge about events | not taken | events, for which there is no architecture yet |

Provenance, licences and the numbers each join depends on are in
`data/thingsplus.SOURCE.md` and `data/newton.SOURCE.md`.

## The design, and the trap it avoids

**The ratings are a layer of their own, not more norms** (`research/v687/
rated.py`). `identify.stated` is what the trie is built from and what R19
counts: `Profiles.corroboration` takes every norm-covered kind beneath an
ancestor as its denominator. Merging 2,000 rated objects would have put
THINGSplus's birds into `bird.n.01`'s denominator without anyone having asked
them about flying -- `fly` would fall from 28 of 29 to about 28 of 50, under
R19's floor, and `do all birds fly` would name them as exceptions. So the
ratings are keyed by synset and read only for the predicates they rated: in
`Profiles.verify_one`, beside AwA2's zeros, and in `reasoning._rated` for
subjects the norms do not name. `do all birds fly` still reads 24 of 27.

**The join goes through the gloss, never the word.** THINGS rates `mouse1`
(alive) and `mouse2` (the device) under one word, and the store's first
`mouse` is the device; readings of a word that disagree give no answer from
the ratings at all. A rating is borrowed from another sense of the word only on
the same side of alive -- the store's first `turkey` is the meat.

| rule | answers | from |
| --- | --- | --- |
| rated norms (R17) | `is a chair alive` no; `is a pillow sharp` no; `is an ant tiny` yes | THINGSplus, NEWTON |
| R31 | `is an elephant bigger than a mouse` yes, 376 against 191; `which is heavier, a feather or a brick` a brick | THINGSplus size and heaviness |
| R32 | `does a rock breathe` no | VerbNet's animate-only verbs joined to "not alive" |
| folk categories | `is a tomato a fruit` yes, after the taxonomy found nothing | THINGSplus categories |
| reader | `can you cut bread with a knife` → `is a knife used for cutting bread`; `is a fridge in a kitchen` → `found in`, read as at_location | rephrase, parser |

Every threshold was checked before it was used: `lives <= 2.5` is outside
`organism.n.01` for 1,392 of 1,569 rated objects and inside it for one;
size agrees with AwA2's `big`; NEWTON never contradicts XCSLB on the 140
objects both cover. The measurements are in `rated.py`'s docstring.

## Wrong answers found on the way

Each was caught by the probe or the suite and fixed before it stood:

- **`can a cup hold water` came back no.** The norms split it into `hold` and
  `water`, and R32 asked whether a cup waters (VerbNet's butter-9.9). R32
  answers only the verb alone now.
- **`can dogs eat chocolate` came back no.** The norms router takes the
  longest norm name in a question, and `dogs` is not `dog`, so it asked
  whether chocolate eats. Same fix; the routing itself is older and open.
- **`is a cat bigger than a horse` was refused.** Every rated reading of a word
  had to agree, and `horse` has a sawhorse at 284 beside the horse at 342.
- **A pin was overridden.** `hammer.n.01` pinned came back as the hammer
  THINGSplus sorted under tools.
- **`is a tree alive` was headlined as about a different sense of the word**,
  because the new answers put a synset where v688 looks for the word.
- **`is a mouse alive` is CONTRADICTED, with or without the ratings.** Found,
  not caused: `Identifier._join` keeps only the lemma of XCSLB's sense key, so
  the norms' `mouse` is the device.

## What changed

Re-running the 114 questions on `9161b9e` changed 22 answers, and none of the
changes is a new wrong answer:

| family | before | after |
| --- | --- | --- |
| `is a chair alive`, `does a rock breathe`, `can a table think` | UNKNOWN, the teacher says no | CONTRADICTED, on a rating (and VerbNet) |
| `is an ant tiny` | UNKNOWN | VERIFIED, 116 on the size scale |
| `is an elephant bigger than a mouse`, `is a cat bigger than a horse`, `is a car heavier than a bicycle` | refused | yes, no, yes, with both ratings |
| `which is bigger, a dog or an ant`, `which is heavier, a feather or a brick` | refused | the dog; the brick |
| `is a cheetah faster than a turtle` | refused | still refused, and the note says what R31 would need |
| `can you drink from a cup`, `can you sit on a chair`, `can you eat an apple` | asked of a computer program | VERIFIED, asked of the thing |
| `can you cut bread with a knife`, `can you write with a pencil`, `can you eat soup with a fork` | asked of a computer program | asked of the thing; unrecorded |
| `what do you use to cut paper`, `what can you use to open a door` | a listing about paper, and about doors | scissors, razor ...; handle, key, knob ... |
| `is a fridge in a kitchen` | UNKNOWN, the teacher says yes | VERIFIED on `at_location kitchen` |
| `is a tomato a fruit` | UNKNOWN | VERIFIED, as people sort things |

The other 92 answers are the same as before, among them `can a cup hold water`,
`can dogs eat chocolate`, `can a bird fly` and the three exceptions `do all
birds fly` names.

## Still open

- **Taxonomic no's**: `is a whale a fish`, `is a spider an insect`, `is a bat a
  bird`. WordNet does not say that siblings exclude each other, and THINGSplus's
  categories were assigned, not scored, so neither can deny.
- **`is fire hot` is CONTRADICTED**: `fire.n.03` carries ConceptNet's
  `not_has_property hot` beside two positives, and R3 lets the denial win.
- **Unique things**: `is the sun hot`, `is the sky blue` are read by v689 as
  individuals, and E1 keeps qualities from descending to them.
- **Likes and needs**: `does a cat like milk`, `do people need sleep` --
  Quasimodo is the candidate source.
- **Fitting inside** (`can an elephant fit in a car`): THINGSplus sizes could
  answer it; no rule reads containment yet.
- **Speed, age, strength**: no scale in any source here.
- **Norms routing by the longest name**: `can dogs eat chocolate` is routed to
  chocolate.
