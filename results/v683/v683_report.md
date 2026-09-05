# V683: ontology minimization by predicate trie

Database: `C:\Users\adria\Desktop\dev\Graph-Topology\data\v633_full_semantic.sqlite`

## Appendix 3 reproduction

| ordering | reproduces Figure 20 | nodes |
|---|---|---|
| adaptive_coverage | yes | 5 |
| global_coverage | yes | 5 |
| optimal | yes | 5 |

All three agree here at five nodes, which exhaustive search confirms is minimal. The paper's own example cannot separate its global description from its branch-local figure, so H2 needs real data.

## Orderings

### attributes

288,035 individuals, 642,738 flat cells, 264,656 distinct predicates

| ordering | nodes | reuse | seconds |
|---|---|---|---|
| adaptive_coverage | 385,626 | 40.00% | 19.427 |
| global_coverage | 392,183 | 38.98% | 0.988 |
| lexical | 417,843 | 34.99% | 0.293 |
| shuffled | 423,744 | 34.07% | 0.974 |
| anti_coverage | 448,311 | 30.25% | 0.986 |

Depth is identical for every ordering here — median 1, mean 2.2315, p99 20, max 6231 — because an ordering permutes an individual's predicates without adding or dropping any. Ordering decides how much storage is *shared*, not how deep anyone sits.

- **H1** coverage vs arbitrary: 31,561 nodes (7.45%) - holds
- **H2** branch-local vs global: 6,557 nodes (1.67%) - holds
- ordering spread, worst minus best: 62,685 nodes

### attributes_multi_predicate

89,840 individuals, 444,543 flat cells, 224,624 distinct predicates

| ordering | nodes | reuse | seconds |
|---|---|---|---|
| adaptive_coverage | 334,434 | 24.77% | 18.479 |
| global_coverage | 341,340 | 23.22% | 0.668 |
| lexical | 367,033 | 17.44% | 0.178 |
| shuffled | 372,201 | 16.27% | 0.64 |
| anti_coverage | 398,992 | 10.25% | 0.625 |

Depth is identical for every ordering here — median 2.0, mean 4.9482, p99 65, max 6231 — because an ordering permutes an individual's predicates without adding or dropping any. Ordering decides how much storage is *shared*, not how deep anyone sits.

- **H1** coverage vs arbitrary: 30,861 nodes (8.29%) - holds
- **H2** branch-local vs global: 6,906 nodes (2.02%) - holds
- ordering spread, worst minus best: 64,558 nodes

### taxonomy

238,033 individuals, 480,785 flat cells, 197,771 distinct predicates

| ordering | nodes | reuse | seconds |
|---|---|---|---|
| adaptive_coverage | 256,556 | 46.64% | 2.914 |
| global_coverage | 260,500 | 45.82% | 0.718 |
| lexical | 294,946 | 38.65% | 0.241 |
| shuffled | 295,462 | 38.55% | 0.68 |
| anti_coverage | 319,777 | 33.49% | 0.748 |

Depth is identical for every ordering here — median 1, mean 2.0198, p99 12, max 671 — because an ordering permutes an individual's predicates without adding or dropping any. Ordering decides how much storage is *shared*, not how deep anyone sits.

- **H1** coverage vs arbitrary: 34,962 nodes (11.83%) - holds
- **H2** branch-local vs global: 3,944 nodes (1.51%) - holds
- ordering spread, worst minus best: 63,221 nodes

### attributes_plus_related_to

704,575 individuals, 2,320,888 flat cells, 540,336 distinct predicates

| ordering | nodes | reuse | seconds |
|---|---|---|---|
| adaptive_coverage | 1,587,839 | 31.58% | 38.069 |
| global_coverage | 1,647,451 | 29.02% | 2.946 |
| lexical | 1,753,240 | 24.46% | 0.899 |
| shuffled | 1,761,286 | 24.11% | 3.02 |
| anti_coverage | 1,872,979 | 19.30% | 2.989 |

Depth is identical for every ordering here — median 2, mean 3.294, p99 24, max 6354 — because an ordering permutes an individual's predicates without adding or dropping any. Ordering decides how much storage is *shared*, not how deep anyone sits.

- **H1** coverage vs arbitrary: 113,835 nodes (6.46%) - holds
- **H2** branch-local vs global: 59,612 nodes (3.62%) - holds
- ordering spread, worst minus best: 285,140 nodes

## H3: access depth against vocabulary size

Individuals are drawn in a fixed random order, so each row is a sample of the ontology rather than a prefix of the alphabet.

| individuals | nodes | median depth | mean depth | p99 depth | max depth |
|---|---|---|---|---|---|
| 28,803 | 49,549 | 1 | 2.2349 | 22 | 249 |
| 57,607 | 92,484 | 1 | 2.2266 | 21 | 415 |
| 86,410 | 132,599 | 1.0 | 2.2214 | 20 | 541 |
| 115,214 | 170,913 | 1.0 | 2.216 | 20 | 541 |
| 144,017 | 214,803 | 1 | 2.2583 | 21 | 6231 |
| 172,821 | 251,419 | 1 | 2.2499 | 21 | 6231 |
| 201,624 | 289,553 | 1.0 | 2.2546 | 21 | 6231 |
| 230,428 | 324,112 | 1.0 | 2.2442 | 21 | 6231 |
| 259,231 | 359,012 | 1 | 2.2398 | 21 | 6231 |
| 288,035 | 392,183 | 1 | 2.2315 | 20 | 6231 |

Vocabulary grew 10.0002x and nodes grew 7.9151x, while median depth moved 1.0x and mean depth 0.9985x.

**H3 holds** - median access depth changes by less than 10% while the vocabulary grows by the factor reported above.

## H4: greedy against the exhaustive optimum

200 samples, predicate universe at most 8.

Coverage ordering hit the exact optimum on 199 of them (99.50%); mean excess 0.005 nodes, worst 1.

**H4 holds** - mean excess over the exhaustive optimum below 0.5 nodes.

