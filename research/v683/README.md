# V683: ontology minimization by predicate trie

Does storing an ontology in a trie *minimize* it, or does the trie just hold
what it was given? Appendix 3 of *Cognitive Network Topology and Optimization
for the Mental Lexicon* says storage order decides, and that ordering
predicates by how many individuals they capture is what makes storage cheap.
This tests that against all of WordNet plus all English ConceptNet.

```bash
python -m research.v683.run_v683                    # full experiment
python -m unittest research.v683.test_v683 -v       # regression suite
```

The experiment needs `data/v633_full_semantic.sqlite`, which it opens
read-only. It writes `results/v683/v683_results.json` and `v683_report.md`.

## What is being measured

An **individual** is a subject concept. A **predicate** is a `(relation,
object)` pair. That is Appendix 3's framing, where `Big` and `Spots` are the
alphabet and the dogs are the goal nodes.

Storing individual *i* costs one trie node per predicate that no earlier
individual already put on that path. Flat storage costs one cell per
individual-predicate pair and shares nothing. The gap between them is the
minimization, and predicate order decides how large it is.

Two numbers are kept apart, because the paper makes two separate claims:

- **Growth** — nodes allocated versus flat cells. Section 15: "the
  topographical growth of tries is driven by allocation."
- **Access** — how deep an individual sits. Section 2.b: "performance is
  centred around the length of the word and not the size of the vocabulary."

## The hypotheses

| | Claim | Falsified when |
|---|---|---|
| H1 | Coverage ordering allocates fewer nodes than an arbitrary order | `global_coverage` does not beat `shuffled` |
| H2 | Branch-local ordering beats one global order | `adaptive_coverage` does not beat `global_coverage` |
| H3 | Access depth is set by the individual, not the vocabulary size | mean depth climbs as the corpus grows |
| H4 | The greedy heuristic lands near the true optimum | mean excess over exhaustive search is large |

H2 exists because the paper describes its algorithm twice and the two
descriptions differ. The Appendix 3 *text* says predicates "are aggregated and
fired in decreasing order" — one global order. Figure 20 draws something a
global order cannot produce: inside the `Big` branch it puts `Long Hair` before
`Spots`, though both cover two individuals overall, because *within that
branch* `Long Hair` covers two and `Spots` covers one. `adaptive_coverage`
implements the figure, `global_coverage` implements the text.

The paper's own five-dog example cannot tell them apart — both reach five
nodes, which exhaustive search confirms is minimal. That is why H2 needs the
full graph.

## Result

All four hold. Full output in `results/v683/v683_report.md`.

On the `attributes` slice — 288,035 individuals, 642,738 flat cells, 264,656
distinct predicates:

| ordering | nodes | reuse |
|---|---|---|
| `adaptive_coverage` | 385,626 | 40.00% |
| `global_coverage` | 392,183 | 38.98% |
| `lexical` | 417,843 | 34.99% |
| `shuffled` | 423,744 | 34.07% |
| `anti_coverage` | 448,311 | 30.25% |

- **H1 holds** on all four slices: coverage ordering beats an arbitrary one by
  6.46%–11.83% of allocated nodes. Ordering alone decides 62,685 nodes between
  the best and worst arrangement of the same facts.
- **H2 holds** on all four slices, but narrowly: branch-local ordering wins
  1.51%–3.62%, for 20–40x the compute. On this data the Appendix 3 text is
  nearly as good as its own figure.
- **H3 holds** cleanly. Across a 10x vocabulary the node count grew 7.9x while
  mean access depth moved from 2.2349 to 2.2315 — a factor of 0.9985. Storage
  grows; access does not.
- **H4 holds**: coverage ordering hit the exact optimum on 199 of 200
  exhaustively-searched samples, mean excess 0.005 nodes. The heuristic is not
  leaving anything on the table; the ~40% ceiling is what the data affords.

The honest caveat is H2. The paper's figure does beat the paper's text, on
every slice, in the predicted direction — but by a margin small enough that the
simple global order is the better engineering default.

## Layout

| file | role |
|---|---|
| `trie.py` | the router: predicates are the alphabet, `ensure` reports allocation |
| `ordering.py` | six orderings, all reduced to one interface: corpus in, predicate paths out |
| `substrate.py` | corpora from the V633 database, plus Table 1 verbatim |
| `measure.py` | node count, reuse rate, access depth, scaling curve |
| `run_v683.py` | the experiment and its report |
| `test_v683.py` | regression suite, anchored on Figure 20 |

Node identity is an integer path, never the predicate symbol, so one predicate
may appear in several branches. Figure 20 requires this: `Spots` is drawn twice,
once under `Big` and once under the root.

## Orderings

| name | rule |
|---|---|
| `lexical` | sort predicates by their own identity — no optimization |
| `shuffled` | fixed-seed arbitrary order — the control for H1 |
| `global_coverage` | Appendix 3 as written: one decreasing-coverage order |
| `adaptive_coverage` | Figure 20: re-rank by coverage within each branch |
| `anti_coverage` | increasing coverage — bounds what bad ordering costs |
| `optimal` | exhaustive over every global order; small corpora only |

`adaptive_coverage` builds an inverted predicate index per branch so the chosen
predicate's members are a lookup rather than a scan. The naive recount-and-scan
version is quadratic: 195s at 80k individuals against 2.7s here, for identical
output. Equivalence to the naive version is checked on random corpora.

## Which relations count

`substrate.py` names three slices explicitly, and the runner reports all of
them, including the ones that weaken the result:

- `attributes` — relations asserting something about the individual
- `taxonomy` — the strict `is_a`/`part_of` backbone
- `attributes_plus_related_to` — adds ConceptNet's dense, low-information
  `related_to`

Word-form relations (`form_of`, `derived_from`, `has_sense`, …) are excluded
from `attributes` because they classify a string rather than a thing: that
"dogged" is `derived_from` "dog" says nothing about the animal.

## Why this database and not the focused one

`data/v673_focused_semantic.sqlite` was seeded as a two-hop crawl around
`{animal, bear, dog}`. 87% of its subjects carry exactly one predicate, and
1,423 of its 4,276 edges are the single predicate `related_to → en:animal`.
Coverage ordering beats alphabetical order there by 49 nodes out of 1,842 — and
by 17 out of 592 on its taxonomic subset. There is nothing for minimization to
act on, so any architecture built on it can only use the data as given. That is
a property of the crawl, not of the theory, and it is why the full graph is the
substrate here.
