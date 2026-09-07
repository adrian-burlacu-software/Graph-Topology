# Graph Topology

Does storing knowledge in a trie *minimize* it — and can you reason over what
comes out?

Appendix 3 of *Cognitive Network Topology and Optimization for the Mental
Lexicon* (Burlacu & West, 2021) says storage order decides: order predicates by
how many individuals carry them, and the trie shares prefixes instead of
repeating them. **[research/v687/](research/v687/)** is that structure, built,
measured, and then read back in every direction a question can come from.

```bash
python -m research.v687.run_v687              # the compression experiment
python -m research.v687                       # the interactive reasoner
python -m unittest research.v687.test_v687 -v # 285 tests, in five suites
```

## What holds

**Prefix sharing compresses, and the paper's figure beats the paper's text.**
Across a scraped ontology of 288,023 individuals and two corpora of elicited
feature norms, `adaptive_coverage` — Figure 20's branch-local ordering — wins
on every corpus without exception, and lands 19% *below* the best possible
single global order, which is what the appendix's prose describes.

**Dense and closed compresses; sparse and open does not.** AwA2 at 43.5%
against XCSLB's 12.5% is the same mechanism on the same day. The difference is
the shape of the data, and that split recurs everywhere in this repo:
typicality and analogy work on the dense corpus and fail honestly on the
sparse one.

**Storing and asking pull in opposite directions.** The rank correlation
between how well an ordering compresses and how many questions it needs to
identify a thing is **+1.000** on all three corpora. Not a tendency — an
ordering, with no exceptions among six policies. A system that must hold
knowledge cheaply *and* recognise things quickly cannot use one order for
both, and the trie does not have to.

## What it answers

The same trie, read in every direction, plus the fact graph around it:
taxonomy and inheritance, two-subject bridging, identification from a
description, retrieval of a thing's attributes, three-valued logic with
quantifiers, contrast and counting, the graph read backwards, causal scripts
and abduction, analogy over a closed vocabulary, and definition from the
taxonomy itself — twenty-six rules, each one visible in the derivation the page
draws for every answer.

Full detail, every measurement, and an honest account of what does not work:
**[research/v687/README.md](research/v687/README.md)**.

## Layout

| | |
| --- | --- |
| `research/v687/` | everything: the trie, the reasoner, the page, the tests |
| `data/` | WordNet, ConceptNet, Ascent++, XCSLB, AwA2, Buchanan, with a `SOURCE.md` for each |
| `llm/` | a separate line of work |

Earlier versions (v683–v686) were folded into v687, which is self-contained
and imports nothing from them; they remain in the git history.

The experiments read `data/v633_full_semantic.sqlite` read-only and never
write to it.
