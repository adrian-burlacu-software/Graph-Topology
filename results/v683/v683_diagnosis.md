# V683 diagnosis: can this graph answer general questions?

`C:\Users\adria\Desktop\dev\Graph-Topology\data\v633_full_semantic.sqlite` under `safe`, 277,806 individuals.

## 1. Is the taxonomy navigable?

- 251,251 nodes, 321,020 `is_a` edges
- **cycles: 5** — `is_a` depth is not a valid generality measure
- 6,227 components; largest holds 56.4%, second is 82,115

      abstraction -> theorization -> conjecture -> theory -> explanation -> thinking
      condition -> good health -> physiological state -> condition
      localized spatial thing -> boundary -> area -> extent -> degree -> intangible object describing predicate

## 2. Do properties inherit?

- `en:dog`: 265 direct, 31 inherited at depth 1, from ['artwork', 'chap', 'domesticated animal', 'good friend', 'mammal', 'pet', 'thing']
- `en:hammer`: 84 direct, 82 inherited at depth 1, from ['blow', 'field event', 'sports equipment', 'striker', 'tool']
- `en:violin`: 26 direct, 19 inherited at depth 1, from ['stringed instrument']

Over 1,200 concepts: direct facts median 0 / mean 0.437; derived median 24 / mean 1548.2 / max 12,923.
87.6% of concepts have no direct facts at all.

## 3. Can bad inheritance be gated?

| gate | median | mean | p99 | max | fact-less concepts covered |
| --- | --- | --- | --- | --- | --- |
| none | 24 | 1548.2 | 9,355 | 12,923 | 74.7% |
| information>=8bits | 8 | 323.8 | 3,061 | 6,669 | 70.7% |
| agreement>=0.02 | 0 | 1.2 | 37 | 298 | 2.6% |
| wordnet_confirmed | 0 | 565.8 | 6,903 | 9,129 | 25.5% |
| sense_scoped | 0 | 21.9 | 149 | 6,217 | 13.6% |

`en:violin` parents reached at depth 2, by gate:

- `none`: ['instrument', 'musical instrument', 'stringed instrument']
- `information>=8bits`: ['instrument', 'musical instrument', 'stringed instrument']
- `agreement>=0.02`: ['musical instrument', 'stringed instrument']
- `wordnet_confirmed`: ['musical instrument', 'stringed instrument']
- `sense_scoped`: ['stringed instrument']

## 4. Can edges be cross-validated against WordNet?

20,000 ConceptNet `is_a` edges sampled of 221,566: 7,598 confirmed, 1,453 contradicted, 10,949 unjudgeable.

Precision among judgeable: 84.0%

- `en:dog is_a en:mammal` → confirmed
- `en:dog is_a en:thing` → REJECTED
- `en:hammer is_a en:tool` → confirmed
- `en:hammer is_a en:sports equipment` → confirmed
- `en:hammer is_a en:match` → REJECTED
- `en:violin is_a en:stringed instrument` → confirmed

## Verdict

The graph supports direct, provenance-carrying lookup over 3.9M edges. It does
not yet support multi-hop inference, and no gate tested here fixes that.

Inheritance is either explosive or over-pruned, with nothing usable between:
trusting every is_a edge yields a mean of ~1,550 derived facts per concept and
a maximum near 13,000, most of it wrong; sense-scoping cuts the mean to ~22 but
drops coverage of fact-less concepts from ~75% to ~14%, and it removes correct
parents (violin loses `musical instrument`) while keeping wrong ones (hammer
keeps `sports equipment`, correct only for the throwing sense).

Information content and profile agreement both fail outright. The database is
too sparse for any statistical agreement measure: `dog is_a mammal` scores
0.0034 agreement, below the junk it is meant to outrank.

What is missing is not another heuristic. It is a labelled set saying which
derived facts are true. Without one, every gate above is tuned against an
unmeasurable target -- which is exactly how V682's benchmark ended up asserting
`actual = expected` and reporting 1.0 accuracy.

