# V685: questions with two subjects

```bash
python -m research.v685          # same page, same store, port 8685
```

Everything v684 does, plus the shape it could not do:

    what does a dog's owner need        what does a dog owner need
    what can a violin's player do       what can a violin player do
    where does a dog's owner live       what does a car driver need

These are not questions about dogs. They are questions about owners — but only
the owners a dog has, and an answer that skips the first half is an answer
about owners in general with the dog as decoration. So v685 answers in two
parts and shows both.

## The graph was already there

V684 answers by going **up**. Every question it takes is about one concept and
the only edge it follows is `is_a`. That is why its search is 15 concepts and
one millisecond — and why it cannot reach an owner from a dog.

But the edge exists, hidden inside the text:

    dog.n.01  has_a  "owner"

`owner` is a concept. **80.1%** of the store's fact objects name one, so
resolving the object text turns 1.7M free-text facts into a directed
multi-relation graph over the same 117,659 concepts. Nothing is rebuilt and no
new data is downloaded; `graph.py` is a different reading of the same rows.

## Which is exactly why the budget is needed

| from `dog.n.01` | concepts reached |
| --- | --- |
| up the `is_a` DAG (v684, exhaustive, ~1 ms) | **15** |
| one fact hop | **870** |
| two fact hops | **8,064** (and that is a sample of the first hop) |

An exhaustive walk stops being possible somewhere inside the first hop. This
is the measurement that answers the question v684's README left open — whether
the depth/breadth controller from the earlier experiments has anything to do.
Over the taxonomy it did not: the branching factor there is 1.02 and there is
no frontier to rank. Here there is.

`bridge.py` is a bounded best-first search with both budgets explicit:

    depth    how many hops a route may take          (default 3)
    breadth  how many frontier nodes survive a round (default 60)

and three signals deciding what survives, none of them learned:

| signal | why |
| --- | --- |
| confidence | a hop through a 0.59 fact beats one through a 0.12 fact |
| hubness | `person` reaches 3,476 concepts; routing through it proves nothing |
| kinship | shared ancestry, counted only over ancestors specific enough to mean it — everything shares `entity` |

**What the breadth budget costs, measured.** From 20 to 1000, over six pairs,
the route that gets used is identical every time. Widening it 50× changes
nothing about the answer. It does change the *alternatives* below the best
route — at 20 the third route for violin runs through `mozart`, at 400 through
`music` — so the test pins the best route only. The first version of that test
claimed the whole list was stable and was wrong.

Pairs it does not find, it reports as absent rather than inventing:
`hammer → carpenter` and `car → mechanic` have no route at any breadth tried.

## The graph disambiguates the role

`player` defaults to the sports sense, so the first working version answered
"what can a violin's player do" with *suffer concussion* and *stand for the
anthem*. The fix needed no new machinery: of the senses `player` might mean,
take one the violin can actually reach.

    violin  --has_prerequisite["musician"]-->  musician.n.01

which then answers *play drum*, *play violin*, *play piano*, *make living*.
A word's meaning here is which other concepts it is joined to, so the graph is
the right thing to ask. `car's driver` resolves to the operator of a motor
vehicle, and `house's owner` is reached in two hops through `dwelling`.

## What is shown

The page is v684's, unchanged, with one card added. A bridged question renders
the route hop by hop with the stored sentence that licenses each one, plus what
the search cost — expanded, generated, pruned. Then v684 takes over completely:
the role concept, the relation the question asked for, R1–R13, and the same
derivation replay and globe as always.

A question with no possessive is answered by v684 untouched, so this is a
superset rather than a fork.

## Layout

| file | role |
| --- | --- |
| `graph.py` | v684's facts read as a multi-relation graph, with the licensing fact on every edge |
| `bridge.py` | bounded best-first search, depth and breadth budgets, reported not assumed |
| `ask.py` | possessive questions: bridge, then hand the second half to v684 |
| `server.py` | v684's engine and page, with the bridge card fed |
| `test_v685.py` | 24 tests; store-dependent ones skip if it is not built |

## Honest limits

- **Two shapes: the possessive and the compound.** `dog's owner` comes from
  spaCy's `poss` dependency and `violin player` from `compound`, both reliable
  across the question forms tried. The compound needs one extra decision the
  possessive does not — `fire truck` is a single concept and must not be split
  — and the ontology settles it: a compound is two subjects exactly when the
  lemma index has no name for the whole of it. `fire truck`, `police dog` and
  `musical instrument` stay whole; `violin player`, `dog owner` and `car
  driver` are bridged. Anything else falls back to v684.
- **A route is evidence, not proof.** `car → commodity → driver` is found
  through `car has_property "good"`, which is a weak link that happened to
  reach the right sense. The route is shown so it can be judged.
- **Everything v684's own limits say still applies**, including that corpus
  generics are not universally quantified. Bridging does not make an
  inherited fact truer.
