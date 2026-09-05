# Which ontology can be reasoned over?

| source | nodes | taxonomy edges | facts | acyclic | components | largest | mean parents | ancestors med/mean/max | grounded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wordnet | 95,657 | 97,665 | 67,666 | yes | 317 | 85.8% | 1.025 | 8 / 8.2 / 28 | 25.1% |
| conceptnet | 155,594 | 223,354 | 240,488 | NO (5) | 5,911 | 91.1% | 1.61 | 6451 / 5946.7 / 6,546 | 13.3% |
| yago | 206,339 | 232,531 | 0 | yes | 3 | 100.0% | 1.127 | 9 / 9.8 / 41 | 0.0% |
| ascentpp | 22,261 | 24,244 | 1,884,031 | NO (5) | 2,254 | 73.9% | 3.233 | 3 / 16.4 / 353 | 33.5% |

## Machine-checkable errors

- **wordnet**: not checkable — source ships no disjointness or domain constraints.
- **conceptnet**: not checkable — source ships no disjointness or domain constraints.
- **yago**: 2 of 4,000 sampled concepts inherit from classes the schema declares disjoint (0.050%).
      yago:Honor_board: schema:Intangible vs schema:Place
      yago:Airport_authority: schema:Organization vs schema:Product
- **ascentpp**: not checkable — source ships no disjointness or domain constraints.

## Capabilities

| source | senses disambiguated | confidence scores | ships constraints |
| --- | --- | --- | --- |
| wordnet | yes | no | no |
| conceptnet | no | no | no |
| yago | yes | no | yes |
| ascentpp | no | yes | no |
