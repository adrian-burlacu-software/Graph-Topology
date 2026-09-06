# V686: procedural knowledge, and the trie read both ways

```bash
python -m research.v686.run_v686     # the compression experiment
python -m research.v686              # everything v685 serves, plus the trie both ways
```

V684 and V685 gave the architecture an idea of *what things are*. This adds
what the paper's Appendix 3 is actually about: the properties that **identify**
a thing, and what happens when you compress them.

## The data

Both are elicited, which is the point. A fixed question was put to people about
every concept, so the predicate vocabulary is closed and two robins really do
carry overlapping predicate sets — unlike a scraped ontology, where predicates
are whatever a crawler happened to record.

| | individuals | predicates | cells | shape |
| --- | --- | --- | --- | --- |
| **AwA2** | 50 animal classes | 85 attributes | 1,562 | dense, closed |
| **XCSLB** | 521 concepts | 3,592 properties | 12,335 | sparse, open |

Provenance in `data/awa2.SOURCE.md` and `data/xcslb.SOURCE.md`. AwA2's base
archive is 32 KB — no images are involved anywhere in this.

XCSLB also ships two things v684 never had: a **negation** for every property
(`has keys` / `does not have keys`), and a **WordNet sense key** for 530 of its
concepts. The second is the join v684 and v685 spent their time guessing at;
here it is given, and 541 of 541 individuals resolve into v684's taxonomy.

**The published `concept_matrix.txt` is not used.** It ships without column
labels and its column order is not the lexicon's — lining them up in file order
gives `budgie` the property `can be covered in lip balm`. Recovering the labels
by matching each column's concept set against the COMPS pairs identified only
**31.2%** of columns uniquely, because hundreds of rare properties are held by
exactly one concept and share a column signature. The COMPS pair file needs
none of that and carries 98.6% of the properties.

## Compression — Appendix 3, unmodified

`research/v683/trie.py` and `research/v683/ordering.py` are imported untouched.
Only the corpus is new: if the compression is a property of the paper's
structure rather than of one dataset, it has to show up without editing it.

| corpus | individuals | cells | best nodes | **compression** |
| --- | --- | --- | --- | --- |
| xcslb/taxonomic | 422 | 857 | 401 | **53.2%** |
| awa2 | 50 | 1,562 | 883 | **43.5%** |
| xcslb/visual perceptual | 520 | 4,072 | 3,153 | **22.6%** |
| xcslb/visual, birds only | 36 | 274 | 220 | **19.7%** |
| xcslb/animal | 59 | 1,557 | 1,324 | **15.0%** |
| xcslb (all) | 521 | 12,335 | 10,794 | **12.5%** |
| xcslb/bird | 36 | 752 | 672 | **10.6%** |
| *v683's scraped ontology* | *288,023* | *641,648* | *384,575* | *40.1%* |

Three things fall out of this:

**Dense and closed compresses; sparse and open does not.** AwA2 at 43.5%
against XCSLB's 12.5% is the same mechanism on the same day — the difference is
entirely the shape of the data. XCSLB's median concept carries 23 of 3,592
properties, so there is very little prefix to share.

**Taxonomic properties compress best of all — 53.2%.** `is a bird`, `is an
animal` are carried by many concepts at once, which is exactly the redundancy
prefix sharing removes. That is the same finding v684's R10 reached from the
other direction, and it is the paper's thesis in miniature.

**`adaptive_coverage` wins on every single corpus, without exception.** The
ordering is identical everywhere: adaptive > global > shuffled ≈ lexical >
anti. Appendix 3 describes its algorithm twice and the two differ — one global
decreasing-coverage order in the text, branch-local ordering in Figure 20 — and
the figure wins on scraped ontologies and elicited norms alike.

### Branch-local beats the best possible global order

On a slice small enough to brute-force every one of the 8! = 40,320 global
orders:

| ordering | nodes | compression |
| --- | --- | --- |
| **adaptive_coverage** | **34** | **89.7%** |
| optimal (best global) | 42 | 87.2% |
| global_coverage | 44 | 86.6% |
| lexical | 53 | 83.9% |

`optimal` is exhaustive over *global* orders — one sequence for every
individual. `adaptive_coverage` orders predicates differently in different
branches, so it searches a strictly larger space and lands **19% below the
global optimum**. Figure 20 does not merely beat the text's heuristic; it beats
anything the text's formulation can express.

### Compression grows with the corpus

| | 10% | 25% | 50% | all |
| --- | --- | --- | --- | --- |
| awa2 | 22.0% | 30.4% | 36.9% | **43.5%** |
| xcslb | 6.5% | 8.9% | 12.5% | **12.5%** |

More individuals, more shared prefix — as the paper predicts. XCSLB plateaus at
12.5%, which is what a sparse open vocabulary looks like when it runs out of
overlap to find.

## Storing against asking

The same trie is for two things, and they pull in opposite directions.

Storing wants **shared prefixes**: put the predicate the most individuals carry
first, and everyone walks the same corridor before splitting. Identifying wants
the **opposite**: the predicate almost nobody carries splits the field on the
first question. So `global_coverage` and `anti_coverage` are not a good idea
and a bad one — they are the two ends of one axis.

Measured on the same trie, where *questions* is how many predicates of an
individual's path must be read before no other individual shares it:

| ordering | nodes | compressed | questions to identify |
| --- | --- | --- | --- |
| anti_coverage | 1,521 | 2.6% | **2.46** |
| cue_validity | 1,478 | 5.4% | 3.64 |
| shuffled | 1,447 | 7.4% | 4.18 |
| lexical | 1,405 | 10.1% | 4.92 |
| global_coverage | 1,177 | 24.6% | 10.06 |
| **adaptive_coverage** | **883** | **43.5%** | **18.66** |

*(AwA2; XCSLB and Buchanan give the same ordering.)*

**The rank correlation between compression and questions-to-identify is
+1.000 on all three corpora.** Not a tendency — an ordering, with no
exceptions among six policies. The ordering that stores AwA2 in 883 nodes
needs 18.66 questions to name an animal; the one that names it in 2.46 needs
1,521 nodes and saves almost nothing.

That is worth stating plainly, because Appendix 3 optimises one end of it. A
system that has to both hold knowledge cheaply and recognise things quickly
cannot use one ordering for both, and the trie does not have to: storage order
and question order are separate choices over the same structure.

### On McRae

McRae et al. (2005) report the two statistics that name this axis —
**distinctiveness** (1 / concepts carrying the feature) and **cue validity**
(P(concept | feature), weighted by production frequency). Their file is behind
a Google login and is not here.

It turns out not to matter, and the reason is worth being precise about: both
are *derived statistics over a feature-norm corpus*, not independent human
ratings. Ranking by distinctiveness is provably identical to ranking by
ascending coverage — `anti_coverage`, already implemented and tested to agree.
And `load_buchanan` carries the production frequencies cue validity weights
by, over seven times as many concepts, so `cue_validity` in
`identifiability.py` is the weighted version McRae would have supplied
unweighted. The experiment McRae suggested runs; McRae's file is not what it
needed.

Weighting by agreement buys a little compression for a few more questions
(5.4% / 3.64 against 2.6% / 2.46) — the same trade-off in miniature.

### What cannot be told apart

Across 3,722 Buchanan concepts exactly **one pair** never becomes unique no
matter how much of its path is read: `percent` and `percentage`. The only
things the predicates cannot separate are two words that mean the same thing,
which is the right failure.

## Identification — the same trie, read downwards

Storing an individual walks *down* the trie until its predicate set runs out.
Identifying one walks the same structure until only one individual is left
underneath. Same mechanism, same ordering question: ask the property that
eliminates most candidates first, which is `adaptive_coverage` choosing a
question instead of a storage slot.

    what kind of dog has spots
      R1   restrict to kinds of dog        4 left: collie, dalmatian, dog, german shepherd
      R16  keep only what is "spots"       1 left: dalmatian          IDENTIFIED

    what kind of cat has stripes
      R1   restrict to kinds of cat        9 left: bobcat, cat, cheetah, leopard, lion
      R16  keep only what is "stripes"     1 left: tiger              IDENTIFIED

    what kind of bird is red
      R1   restrict to kinds of bird      30 left
      R16  keep only what is "red"         2 left: robin, cockerel    AMBIGUOUS
           robin   "has a red breast"
           cockerel "has a red crest"

The last is a *correct* ambiguity: both birds really are red, and the trace says
which stored property matched.

**The question is routed by grammar, not a pattern list.** A description
leaves the thing unnamed and says what it is like, which shows up as a
relative clause (`an object *that is round*`), an adjectival complement (`what
is *round* with spots`), or `what` used as a determiner (`*what animal* has
stripes`). A naming question has a subject and asks what it does — that is
v684's and must not be taken. So all of these are answered:

    what is an object that is round with spots
    what is round with hexagons          -> football
    what animal has stripes
    which animal is big and furry

while `what can a violin do`, `what is a hammer used for` and `what does a
dog's owner need` still belong to v684 and v685.

**The rivals shown are the nearest misses.** Each attribute displays the
candidates *it* removed, ranked by how many properties they share with the
answer. `what is round with hexagons` puts `ball, balloon, frisbee, boomerang,
dice` under `hexagons` — sorting by name instead gave `accordion, ambulance,
antelope`, which is the alphabet rather than the near misses.

**Stated evidence outranks inherited.** v684's inherited facts are corpus free
text, and matching one word inside them is how `collie` acquires "spots" from
`capable of spot movement`, or `marble` learns to fly via `capable of fly
through the air`. Inheritance is consulted only when nothing is stated at all.

**Two morphology traps, both found by reading the output.** Prefix matching at
four characters puts cheetahs under "stripes" (`striven`) and desks under "has
a trunk" (`truncated`), so matching is whole-word on a stem. And AwA2 ships no
sense keys, so its classes are looked up by name — without that `dalmatian` is
not a kind of dog and the flagship question finds only the concept `dog`.

## Retrieval — the same trie, read upwards

Identification walks the trie down. The check on its answer walks the same
structure back up, and that is where the attributes are: an individual sits at
a leaf, so the path from that leaf to the origin **is** the predicate set it
was stored with. Nothing else has to be kept.

    what attributes does a blue whale have
    is a blue whale furry           -> no, and the norms say so
    does a robin fly                -> not in the norms; birds do, 4 levels up

A test asserts the retrieval claim on all 541 individuals: walking each leaf
back to the origin returns exactly the set it was stored with. If it did not,
the trie would be a lossy index rather than the storage itself, and every
compression figure above would be measuring the wrong thing.

**The order carries the compression.** Read upwards, `shared` — how many
concepts are still on the branch at that node — never decreases:

| depth | attribute | company |
| --- | --- | --- |
| 28 | strainteeth | unique to it |
| 21 | blue | unique to it |
| 20 | slow | 3 on the branch |
| 11 | flippers | 4 on the branch |
| 2 | fast | 5 on the branch |
| 1 | oldworld | 44 on the branch |

The tail is what makes it a blue whale; the head is the corridor it shares
with forty-three other animals. Same fact as the 43.5% compression, seen from
inside one branch instead of counted over the corpus.

**Most of the path is pass-through.** Nothing branches at a node whose members
are unchanged, so consecutive ones collapse the way a radix tree collapses
them: the median concept carries 24 predicates and has **4** branch points,
and **80.8%** of path nodes disappear. That is what the page draws — six nodes
for a blue whale, not twenty-eight — with the concepts that shared the prefix
and left shown beside the point they left at. `dolphin` stays with the blue
whale until the branch commits to plankton.

**A "no" needs a denial, and both corpora have one.** The trie stores what a
thing has, so absence is silence and nothing more. AwA2 scored all 50 classes
on all 85 attributes, so a zero is a denial — that is what answers "is a blue
whale furry". COMPS ships an unacceptable concept beside every acceptable one,
which denies all 521 XCSLB concepts a median of 67 properties each. And a
property can be phrased as a denial: the norms state `cannot fly` of a
penguin, so matching "fly" against it and reporting a yes is the one way this
could be confidently wrong. It is read, not just matched.

| | answer | evidence |
| --- | --- | --- |
| `is a blue whale furry` | **DENIED** | AwA2 scored it zero |
| `is a lion striped` | **DENIED** | AwA2 scored it zero |
| `does a penguin fly` | **DENIED** | the norms state `cannot fly` |
| `is a dalmatian spotted` | **HELD** | `spots`, depth 28, unique to it |
| `does a killer whale have flippers` | **HELD** | `flippers`, depth 17 |
| `does a robin fly` | **INHERITED** | `bird.n.01`, 4 levels up |
| `is a whale furry` | **DENIED** | all 4 kinds of whale deny it |
| `is a blue whale telephonic` | **UNRECORDED** | absent, not false |

**A class is answered by the kinds beneath it.** `whale` is a concept in its
own right and carries 27 properties, not one about fur — so "is a whale furry"
was silence while four kinds of whale sat underneath with the attribute scored
and denied. Asking them is induction rather than inheritance, and it is placed
*above* the taxonomy walk on purpose: four whales scored `hairless` is better
evidence than whatever `mammal.n.01` happens to say about fur. When the kinds
disagree the answer says so rather than picking one.

What the concept states about itself still outranks them both. A dalmatian has
spots and a dog does not, and both are right: the question is about the
typical dog, and the norms scored that directly.

**And the replay has to show it.** The page's claim is that every answer is
replayable, and this one was not: the steps walked whale's own branch and then
announced a denial with nothing in between. The question is now a node of the
walk and each thing that answered it is a step of its own — the four kinds, or
the ancestor that lent the fact:

    is hunted                      shared with 10
    can be found in seas           shared with 1
    can dive underwater +24
    furry?                         ask it of whale
      blue whale       DENIED      humpback whale   DENIED
      dolphin          DENIED      killer whale     DENIED
    whale                          DENIED

**Above the leaf, the taxonomy carries on.** The norms stop at the concept;
v684 does not. `blue whale` inherits 777 facts from six levels, and each is
attributed to the level that supplies it, nearest first, with a fact met twice
kept only at the nearer one — R10 applied to the display. Order matters here:
what a norm states about the thing outranks what it inherits, and only when
the norms neither state nor deny is the taxonomy asked.

Two smaller things that had to be right. `spots` is what the norms store and
`spotted` is what a person asks, so the stemmer lost its participle blindness
— while keeping the whole-word matching that stopped `striven` matching
`stripes`. And when several inherited facts match, the nearest ancestor wins,
then the plainest fact: "does a robin fly" was being answered by `desires fly`
rather than `capable of fly high`.

**Questions about a named thing are routed by name, not by pattern.** The
longest concept the norms know that the question mentions is the subject, so
`blue whale` beats `whale`. Everything else is handed back: v684 owns `is a
hammer a tool`, because it answers taxonomy with a derivation of its own, and
a property the norms neither state nor deny goes back to the fact graph rather
than being reported as absent.

## What did not work

**"What is red and flies" has no clean answer here.** `cardinal` is not among
the 521 concepts, and `robin` — which should answer it — lists 14 properties
without one mentioning flight. Falling back to inherited properties for both
words produces noise (`marble`, `jellyfish`). The mechanism is fine; the
coverage is not. `what kind of bird is red` is the same question in a form the
data supports, and it answers correctly.

## Layout

| file | role |
| --- | --- |
| `corpora.py` | XCSLB and AwA2 as v683 `Corpus` objects, sliceable by feature type and category |
| `run_v686.py` | the compression experiment: orderings, the optimal floor, the scaling curve |
| `identify.py` | the trie read downwards, with the narrowing recorded |
| `profile.py` | the trie read upwards: attributes, denials, and the taxonomy above the leaf |
| `identifiability.py` | questions-to-identify on the same trie, and the cue-validity ordering |
| `server.py` | everything v685 serves, plus identification, on the same page |
| `test_v686.py` | 59 tests; they skip if the norms are not downloaded |
