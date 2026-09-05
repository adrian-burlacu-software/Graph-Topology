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
python -m unittest research.v684.test_v684 -v   # 32 tests
python -m research.v684 --rebuild               # discard and rebuild the store
python -m research.v684 --port 9000 --no-browser
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

- **Facts are attached to a lemma's primary sense.** Ascent++ and ConceptNet
  state facts about word strings. They are attached to the eponymous synset —
  `dog` to `dog.n.01`, not `andiron.n.01` — and flagged `sense_assumed`, shown
  in the UI. A wrong attachment is visible, not silent.
- **UNKNOWN means absent, not false.** The ontology asserts nothing by omission.
- **`can a dog fall down` returns UNKNOWN.** No ancestor asserts it; "fall into
  a hole" is there and "fall down" is not. The matcher was not loosened to make
  the example work.
- **No accuracy number.** There are still no labels for derived facts. The
  confidence shown is a decayed source score, not a measured probability.

## Layout

| file | role |
| --- | --- |
| `build.py` | assembles the store: WordNet taxonomy, Ascent++ and ConceptNet facts |
| `rules.py` | the nine rules, each with its stated reason |
| `reason.py` | the walk: `classify`, `verify`, `describe`, all step-recorded |
| `language.py` | spaCy question parsing, lemma matching, regex fallback |
| `server.py` | stdlib HTTP server |
| `app.html` | the page: graph animation, collapsible steps, sense picker |
| `test_v684.py` | 32 tests; store-dependent ones skip if it is not built |

Data lives in `data/` (gitignored): `v633_full_semantic.sqlite` and
`ascentpp.csv` in, `v684_reasoning.sqlite` out.
