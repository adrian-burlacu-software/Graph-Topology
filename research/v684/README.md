# V684: an interactive reasoner over a clean lexical ontology

```bash
pip install -r research/v684/requirements.txt
python -m spacy download en_core_web_sm
python -m research.v684
```

One command. It builds the reasoning store on first run (a few minutes), then
serves <http://127.0.0.1:8684/> and opens it. Ask a question, watch the
derivation replay on the graph, expand the steps if you want the detail.

```bash
python -m unittest research.v684.test_v684 -v   # 86 tests
python -m research.v684 --rebuild               # discard and rebuild the store
python -m research.v684 --port 9000 --no-browser
python -m research.v684 --raw                   # serve the uncompressed store
python -m research.v684.compress                # compress and verify losslessness
```

## What it is made of, and why

The composition is the conclusion of `research/v683/ontologies.py`, which
scored four candidate ontologies on the same label-free metrics:

| source | acyclic | components | ancestors med/mean/max | role here |
| --- | --- | --- | --- | --- |
| WordNet | yes | 317 | 8 / 8.2 / 28 | **the taxonomy** |
| ConceptNet | NO (5) | 5,911 | 6451 / 5946.7 / 6,546 | facts only, low confidence |
| YAGO 4.6 | yes | 3 | 9 / 9.8 / 41 | not used — entity records, not a lexicon |
| Ascent++ | NO (5) | 2,254 | 3 / 16.4 / 353 | facts, graded by typicality |

ConceptNet's median concept has 6,451 ancestors. Its five cycles close the
taxonomy over one giant component, so generalising anything reaches nearly
everything — that is the inheritance explosion `research/v683/diagnose.py`
measured, and no downstream gate fixes it. Its `is_a` is dropped entirely here.

WordNet supplies the hierarchy instead: a clean DAG after one repair, since
exactly one pair in 97,666 (`restrain.v.01` / `inhibit.v.04`) is asserted as
both `is_a` and `has_subtype` in the same direction.

## The rules

`rules.py` holds all of them, and every step in the UI names the rule that
produced it.

| | rule | why |
| --- | --- | --- |
| R1 | Subsumption closure | `is_a` is transitive over an acyclic taxonomy |
| R2 | Property lift | a subtype inherits, **for inheritable relations only** |
| R3 | Exception blocking | a closer statement, or an explicit negation, overrides an inherited fact |
| R4 | Specificity preference | the nearest ancestor that answers wins |
| R5 | Confidence decay | each level of borrowing multiplies confidence by 0.85 |
| R6 | Sense scoping | inference runs per WordNet sense, never per word |
| R7 | Relation gating | contentless relations never participate |
| R8 | Answer synthesis | VERIFIED, CONTRADICTED or UNKNOWN — never "probably" |
| R9 | Relation families | `has_a` and `has_part` answer for each other |
| R10 | Redundancy elimination | a fact an ancestor already states is not stored — R2 rebuilds it |
| R11 | Hoisting | a fact *every* child states moves up to the parent |
| R12 | Breadth gating | a word-level fact does not inherit from a concept with 8,000+ descendants |
| R13 | Range typing | a relation's object must be the kind of thing the relation takes |

**R2 is the one that matters.** Without it, trusting every relation over every
edge yields a mean of 1,548 derived facts per concept, most of them wrong. Each
non-inheritable relation states its reason: `made_of` does not descend because
a chair is furniture but furniture is not made of wood.

**R7 is where `related_to` dies.** It is 1,678,150 of v633's 3.9M edges and
asserts only that two words co-occur — no direction, no relation type.
Inheriting it floods every answer. It is excluded from storage *and* from
inference.

**R9 exists because the sources disagree.** WordNet files "a dog has a tail"
under `has_part`; Ascent++ files it under `has_a`. Without the family, the
answer depends on which source happened to record it. `part_of` is deliberately
**not** a family member of `has_part` — it is the inverse, and conflating them
reverses facts.

**R10 and R11 make the compression native.** They are the paper's claim
executed: storing a fact once at the most general level where it holds shrinks
the store 12.5% (1,966,921 → 1,720,748), and the mechanism that shrinks it is
the same R2 that answers questions. `compress.verify` re-derives every dropped
fact and reports 0 of 169,492 lost. R11 is different in kind and marked as
such — hoisting a fact every child states is induction, not deduction, so it
requires unanimity and the rows it creates are flagged `hoisted`.

**R4 orders the listing, not just the walk.** What is stated about the thing
outranks what is borrowed from above, whatever the confidence: `violin
capable_of run android`, inherited from `device` four levels up, used to sit
above `sound beautiful` stated about violins.

**R12 is where the word/sense join stops being trusted.** Ascent++ and
ConceptNet talk about words, and a word in an open-domain corpus never means a
top-of-taxonomy abstraction: their `artifact` is a thing dug out of a tomb, not
WordNet's "man-made object taken as a whole". Six concepts sit above the limit
— `entity`, `living thing`, `organism`, `causal agent`, `artifact`, `person` —
carrying 8,815 word-level facts that each reached 10,000+ descendants. That is
why asking where a hammer is returned `at_location tomb`. `animal` (4,016
descendants) and `plant` (4,487) sit below the line and keep inheriting, which
is the point: there the word and the class still mean the same thing.

## Which sense do word-level facts belong to?

This is the join between two ontologies that disagree about what a subject is,
and it is the single most consequential decision in the build. WordNet's sense
numbering is not a usefulness ranking:

    hammer.n.01   the part of a gunlock that strikes the percussion cap
    hammer.n.02   a hand tool with a heavy rigid head and a handle

Ranking senses by a fixed rule — eponymous, then noun, then `.01` — gets `dog`
right and sends all 348 of hammer's facts to a gun part. 30% of the words
carrying facts have two or more eponymous noun senses, where that rule is a
coin flip.

`senses.py` lets the facts choose instead: each candidate sense is scored
against the vocabulary of everything the sources say about the word, using its
gloss, its synonyms and its place in the taxonomy. `hammer` is stated to be in
a toolbelt and a hardware store; `hammer.n.02` sits under `hand tool`. It
reassigns 3,090 of 7,635 ambiguous words, including

| word | was | now |
| --- | --- | --- |
| hammer | the part of a gunlock | a hand tool |
| bank | a sloping land beside water | a financial institution |
| mouse | a small rodent | a hand-operated pointing device |
| crane | a large wading bird | a lifting machine |
| gold | coins made of gold | a soft yellow metallic element |
| ram | a tool for driving by impact | random-access memory |

Two guards, both found by reading what it changed. Nouns win when the word has
a noun sense, or `dog` goes to `chase.v.01` — these sources state facts about
things. And leaving the synset named for the word takes a decisive margin, or
`seal` goes to `navy seal.n.01` on a 5% edge.

**Routing each fact individually was tried and rejected.** It moved 77,950 of
1.9M facts and most moves were wrong: `fire capable_of burn clothing` went to
`burn.v.01`, `change capable_of become official` to `deepen.v.04`. A fact's
object describes the *predicate*, not the subject, so a verb in the object
drags the subject to a verb sense. Pooled over hundreds of objects that noise
averages out — which is why the aggregate works and per-fact routing cannot.

## What a question does

    can a dog fall into a hole
      R6  read "dog" as dog.n.01, not as a word          (8 senses available)
      R2  check 704 capable_of facts on dog              no match
      R1  generalise to canine, domestic animal          ...
      R1  generalise to carnivore, animal
      ...
      R4  found: object capable_of "fall into hole"      6 levels up, conf 0.132

`is a dog an animal` is answered by the taxonomy alone (R1): `animal.n.01` is
an ancestor two steps up, and the chain walked is the proof.

## Honest limits

- **All of a word's facts still go to one sense.** The choice is now made by
  the facts rather than by sense number, but a word that is genuinely used in
  two senses keeps both sets on one synset: `hammer.n.02` holds `at_location
  price charts of financial assets`, which belongs to the candlestick pattern
  and has no WordNet sense to go to. Every such row is flagged `sense_assumed`
  and shown as such. The join is visible, not silent.
- **UNKNOWN means absent, not false.** The ontology asserts nothing by omission.
- **`can a dog fall down` returns UNKNOWN.** No ancestor asserts it; "fall into
  a hole" is there and "fall down" is not. The matcher was not loosened to make
  the example work.
- **A general class donates facts that are only typically true of it.**
  `device.n.01` subsumes both smartphones and violins, so Ascent++'s
  `capable_of run android` — a true statement about the modal device —
  reaches the violin. This is not a sense error and no available signal
  separates it from a good generalisation: Ascent++'s own typicality scores
  `device run android` at 0.757, *above* `bird fly` at 0.716, because it
  measures typicality among sentences about the subject rather than share of
  the class. Corpus generics are not universally quantified, and inheritance
  treats them as if they were. R4's ordering keeps them below what is stated
  directly; nothing here removes them.
- **No accuracy number.** There are still no labels for derived facts. The
  confidence shown is a decayed source score, not a measured probability.

## Layout

| file | role |
| --- | --- |
| `build.py` | assembles the store: WordNet taxonomy, Ascent++ and ConceptNet facts |
| `senses.py` | which sense a word's facts belong to, decided by the facts |
| `rules.py` | the thirteen rules, each with its stated reason |
| `reason.py` | the walk: `classify`, `verify`, `describe`, all step-recorded |
| `compress.py` | R10/R11, and the verification that R10 is lossless |
| `language.py` | spaCy question parsing, lemma matching, regex fallback |
| `server.py` | stdlib HTTP server |
| `app.html` | the page: graph animation, collapsible steps, sense picker |
| `test_v684.py` | 86 tests; store-dependent ones skip if it is not built |

Data lives in `data/` (gitignored): `v633_full_semantic.sqlite` and
`ascentpp.csv` in, `v684_reasoning.sqlite` out, plus
`v684_reasoning_compressed.sqlite` — which is what the server answers from
unless you pass `--raw`.
