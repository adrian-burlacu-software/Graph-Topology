# Graph Topology

Does storing knowledge in a trie *minimize* it?

Appendix 3 of *Cognitive Network Topology and Optimization for the Mental
Lexicon* (Burlacu & West, 2021) says storage order decides: order predicates by
how many individuals they capture, and the trie shares prefixes instead of
repeating them. **[research/v683/](research/v683/)** tests that against all of
WordNet plus all English ConceptNet — 288,035 individuals, 642,738 facts.

```bash
python -m research.v683.run_v683                 # full experiment
python -m unittest research.v683.test_v683 -v    # regression suite
```

It reproduces the paper's Figure 20 from its Table 1, then reports four
falsifiable claims. All four hold:

| | Claim | Result |
| --- | --- | --- |
| H1 | Coverage ordering allocates fewer nodes than an arbitrary order | holds — 4.96%–10.62% fewer, across four relation slices |
| H2 | Branch-local ordering beats one global order | holds, narrowly — 1.64%–2.19%, for 20–40x the compute |
| H3 | Access depth is set by the individual, not the vocabulary size | holds — vocabulary 10x, nodes 8.0x, mean depth 1.0085x |
| H4 | The greedy heuristic lands near the true optimum | holds — exact optimum on 200 of 200 exhaustive samples |

Ordering alone decides 49,878 nodes between the best and worst arrangement of
the same facts.

Relation and node normalization is applied first and reported as its own
ablation, because it moves the numbers more than ordering does: `has_subtype`
is `is_a` stored backwards and `has_part` is `part_of` inverted, so counting
both spellings stored the WordNet hierarchy twice and inflated the raw taxonomy
slice by 76% (259,703 nodes against 147,148). Every hypothesis survives every
normalization preset. Details and caveats in
[research/v683/README.md](research/v683/README.md); generated output in
`results/v683/`.

The experiment reads `data/v633_full_semantic.sqlite` read-only and never
writes to it.
