# V683: ontology minimization by predicate trie

Database: `C:\Users\adria\Desktop\dev\Graph-Topology\data\v633_full_semantic.sqlite`

## Appendix 3 reproduction

| ordering | reproduces Figure 20 | nodes |
|---|---|---|
| adaptive_coverage | yes | 5 |
| global_coverage | yes | 5 |
| optimal | yes | 5 |

All three agree here at five nodes, which exhaustive search confirms is minimal. The paper's own example cannot separate its global description from its branch-local figure, so H2 needs real data.

## Normalization

Applied before any ordering question, and reported separately because it moves the numbers more than ordering does.

| slice | normalization | individuals | flat cells | nodes | reuse | H1 |
| --- | --- | --- | --- | --- | --- | --- |
| attributes | `raw` | 288,023 | 641,648 | 391,109 | 39.05% | 7.25% |
| attributes | `safe` | 277,806 | 596,288 | 360,881 | 39.48% | 6.43% |
| attributes | `sense_merged` | 255,620 | 592,973 | 371,994 | 37.27% | 6.30% |
| attributes | `lemma_bridged` | 182,114 | 515,353 | 347,459 | 32.58% | 6.80% |
| taxonomy | `raw` | 238,021 | 479,975 | 259,703 | 45.89% | 11.52% |
| taxonomy | `safe` | 237,684 | 360,120 | 147,148 | 59.14% | 10.62% |
| taxonomy | `sense_merged` | 216,878 | 357,150 | 158,020 | 55.76% | 9.96% |
| taxonomy | `lemma_bridged` | 151,930 | 286,044 | 138,052 | 51.74% | 11.23% |

`raw` is unnormalized apart from self-loop removal, which is unconditional: the database holds 43,033 edges whose subject and object are the same node, and a self-predicate allocates a trie node while saying nothing about the individual.

Everything below uses `safe`.

## Orderings

### attributes

277,806 individuals, 596,288 flat cells, 241,616 distinct predicates

| ordering | nodes | reuse | seconds |
|---|---|---|---|
| adaptive_coverage | 354,977 | 40.47% | 20.592 |
| global_coverage | 360,881 | 39.48% | 0.9 |
| lexical | 378,658 | 36.50% | 0.311 |
| shuffled | 385,677 | 35.32% | 0.901 |
| anti_coverage | 404,855 | 32.10% | 1.117 |

Depth is identical for every ordering here — median 1.0, mean 2.1464, p99 20, max 6230 — because an ordering permutes an individual's predicates without adding or dropping any. Ordering decides how much storage is *shared*, not how deep anyone sits.

- **H1** coverage vs arbitrary: 24,796 nodes (6.43%) - holds
- **H2** branch-local vs global: 5,904 nodes (1.64%) - holds
- ordering spread, worst minus best: 49,878 nodes

### attributes_multi_predicate

74,068 individuals, 392,550 flat cells, 200,489 distinct predicates

| ordering | nodes | reuse | seconds |
|---|---|---|---|
| adaptive_coverage | 302,768 | 22.87% | 18.807 |
| global_coverage | 309,055 | 21.27% | 0.697 |
| lexical | 327,878 | 16.47% | 0.16 |
| shuffled | 332,681 | 15.25% | 0.702 |
| anti_coverage | 354,448 | 9.71% | 0.584 |

Depth is identical for every ordering here — median 2.0, mean 5.2999, p99 71, max 6230 — because an ordering permutes an individual's predicates without adding or dropping any. Ordering decides how much storage is *shared*, not how deep anyone sits.

- **H1** coverage vs arbitrary: 23,626 nodes (7.10%) - holds
- **H2** branch-local vs global: 6,287 nodes (2.03%) - holds
- ordering spread, worst minus best: 51,680 nodes

### taxonomy

237,684 individuals, 360,120 flat cells, 92,396 distinct predicates

| ordering | nodes | reuse | seconds |
|---|---|---|---|
| adaptive_coverage | 143,924 | 60.03% | 1.298 |
| global_coverage | 147,148 | 59.14% | 0.466 |
| lexical | 157,909 | 56.15% | 0.187 |
| shuffled | 164,641 | 54.28% | 0.461 |
| anti_coverage | 178,587 | 50.41% | 0.479 |

Depth is identical for every ordering here — median 1.0, mean 1.5151, p99 7, max 372 — because an ordering permutes an individual's predicates without adding or dropping any. Ordering decides how much storage is *shared*, not how deep anyone sits.

- **H1** coverage vs arbitrary: 17,493 nodes (10.62%) - holds
- **H2** branch-local vs global: 3,224 nodes (2.19%) - holds
- ordering spread, worst minus best: 34,663 nodes

### attributes_plus_related_to

563,362 individuals, 2,210,178 flat cells, 702,485 distinct predicates

| ordering | nodes | reuse | seconds |
|---|---|---|---|
| adaptive_coverage | 1,698,742 | 23.14% | 73.823 |
| global_coverage | 1,730,537 | 21.70% | 3.406 |
| shuffled | 1,820,838 | 17.62% | 3.611 |
| lexical | 1,830,581 | 17.18% | 0.939 |
| anti_coverage | 1,914,273 | 13.39% | 4.181 |

Depth is identical for every ordering here — median 1.0, mean 3.9232, p99 48, max 6615 — because an ordering permutes an individual's predicates without adding or dropping any. Ordering decides how much storage is *shared*, not how deep anyone sits.

- **H1** coverage vs arbitrary: 90,301 nodes (4.96%) - holds
- **H2** branch-local vs global: 31,795 nodes (1.84%) - holds
- ordering spread, worst minus best: 215,531 nodes

## H3: access depth against vocabulary size

Individuals are drawn in a fixed random order, so each row is a sample of the ontology rather than a prefix of the alphabet.

| individuals | nodes | median depth | mean depth | p99 depth | max depth |
|---|---|---|---|---|---|
| 27,780 | 45,197 | 1.0 | 2.1283 | 20 | 283 |
| 55,561 | 83,662 | 1 | 2.1141 | 20 | 283 |
| 83,341 | 120,169 | 1 | 2.1132 | 20 | 283 |
| 111,122 | 154,701 | 1.0 | 2.1093 | 20 | 283 |
| 138,903 | 190,202 | 1 | 2.1204 | 21 | 391 |
| 166,683 | 223,176 | 1 | 2.1166 | 20 | 391 |
| 194,464 | 258,549 | 1.0 | 2.1295 | 20 | 1937 |
| 222,244 | 292,238 | 1.0 | 2.1337 | 21 | 1937 |
| 250,025 | 329,174 | 1 | 2.1505 | 20 | 6230 |
| 277,806 | 360,881 | 1.0 | 2.1464 | 20 | 6230 |

Vocabulary grew 10.0002x and nodes grew 7.9846x, while median depth moved 1.0x and mean depth 1.0085x.

**H3 holds** - median access depth changes by less than 10% while the vocabulary grows by the factor reported above.

## H4: greedy against the exhaustive optimum

200 samples, predicate universe at most 8.

Coverage ordering hit the exact optimum on 200 of them (100.00%); mean excess 0.0 nodes, worst 0.

**H4 holds** - mean excess over the exhaustive optimum below 0.5 nodes.

