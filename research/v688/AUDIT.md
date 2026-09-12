# The v688 audit — results

2026-09-09. Run with:

```
python -m research.v688.audit --limit 1200 --loop-limit 500 --shards 5 --workers 4
python -m research.v688.audit --phrasing        # the transform, no engines
```

Five configurations differing only in what is switched off, asked COMPS'
minimal pairs over XCSLB's 521 concepts. 4,800 pairs (1,200 per rung of the
foil ladder); the loop at 2,000. `audit.py`'s docstring carries the method and
the caveats; this file is what came back.

---

## 1. The gold set is not what it was documented to be

Found before any accuracy number, and the most consequential result here.

`concept_matrix.txt` is 521 × 3,644 binary and **1.58% dense** — 30,009 ones
in 1.9M cells. That is a free-listing norm: a zero means no participant
mentioned the feature, not that anyone judged it false. COMPS draws its foils
from those zeros, so they are not denials:

```
stocking  NOT absorbs sweat     (taxonomic)
potato    NOT absorbs water     (co-occurrence)
```

So `corpora.denied_xcslb`'s docstring — *"the only place in any of this data
where absence is stated rather than merely observed"* — is **false**, and this
is the AwA2-zeros trap one level up. `profile.py` merges it into
`Profiles.denied`, which is what **R17 answers DENIED from**: a live
false-denial source in the shipped system.

Not fixed here. The audit works around it by never scoring a foil as a
denial. Fixing it in v687 is a separate change and it will cost real denials.

---

## 2. The crawl is not wrong. It is absent.

Positives only, where a listed feature *is* an assertion (n = 1,200):

| config | coverage | confirmed | contradicted |
| --- | --- | --- | --- |
| shipped | 92.3% | 89.9% | 2.2% |
| **crawl** | **19.4%** | 16.9% | 0.8% |
| pinned | 19.9% | 17.7% | 0.6% |
| corroborated | 14.8% | 12.3% | 0.8% |
| loop | 14.8% | 12.2% | 0.8% |

1.9M crawled facts reach **19.4%** of what people list about 521 everyday
concepts. Where the crawl does settle a positive it is essentially never
wrong — `R4` 100% (n=165), `ascentpp` 100% (n=147), `R1` 100% (n=47).

`shipped` is a control, not a result: R17 answers from the rows the questions
were built from, so its 92.3% says the harness works. Anything much below
that would have meant the phrasing was broken.

**"Sparsity is the binding constraint" now has a number, and the number says
the rules are fine.**

---

## 3. The loop adds nothing R19 was not already adding

Matched on the identical 2,000-pair sample — the first comparison put the loop
and `corroborated` on different strides, which is not a comparison:

| config | decided | accuracy |
| --- | --- | --- |
| crawl | 20.9% | 75.4% |
| corroborated | 14.1% | 82.6% |
| **loop** | **14.0%** | **83.2%** |

83.2% against 82.6% is about two pairs of 282 decided, and two rungs of the
foil ladder agree to three decimal places. The loop does not decide more
either.

**The case for v688 cannot rest on accuracy.** The honest reading is not that
the loop is useless: 86% of these pairs are undecided because the store holds
no row either way, and no amount of asking around repairs an absence. What the
loop does is explain and expose — and this benchmark measures neither.

---

## 4. What R19 costs and what it buys

A measured trade, where before there was one example:

| | coverage | accuracy | random | co-occur | overlap | taxonomic |
| --- | --- | --- | --- | --- | --- | --- |
| crawl | 19.4% | 81.6% | 92.0% | 81.8% | 79.2% | 72.7% |
| corroborated | 14.8% | 87.3% | 94.0% | 85.5% | 87.0% | 82.2% |

R19 refuses about a quarter of what the crawl reached and returns **+5.7
points**, of which **+9.5 lands on the near foils it was written for and +2.0
on random ones**. Near foils are exactly where an existential class fact gets
wrongly inherited down, so it does its designed job and now there is evidence
rather than an anecdote.

**Caveat that limits this row.** R19 corroborates against the ancestor's other
kinds, and on this gold set those siblings are other rows of the same
instrument. The delta describes the shipped system on these 521 concepts and
says nothing about the other 44,678.

---

## 5. The foil ladder falls monotonically — the audit's own control

Accuracy against how near the foil is, in every configuration:

```
                random   co-occur   overlap   taxonomic
crawl            92.0%     81.8%     79.2%      72.7%
pinned           93.9%     83.5%     82.3%      74.8%
corroborated     94.0%     85.5%     87.0%      82.2%
loop             95.6%     80.6%     83.8%      73.6%
shipped          99.6%     99.2%     99.4%      99.2%
```

A near foil is harder than a distant one, consistently. Had this come out flat
the whole method would have been measuring noise. `shipped` is flat *because*
it is at ceiling.

---

## 6. `confidence.py` holds up

Tested the sound way: the margin between the two sides of a pair against how
often the listed concept won. This needs no absolute label for the foil, which
is what makes it valid here.

```
margin        0.0-0.2  0.2-0.4  0.4-0.6  0.6-0.8  0.8-1.0
crawl           68.5%    67.7%    87.6%    89.0%    91.8%
pinned          72.8%    68.9%    88.2%    90.4%    95.5%
corroborated    77.5%    72.6%    89.3%    88.5%    91.5%
loop            60.0%    45.5%    69.2%    92.9%    93.7%
```

23 points of spread for `crawl`, the same shape four times independently. The
`0.2-0.4` dip recurs in every configuration and is unexplained.

The per-answer reliability curve looks *inverted* (0.8–1.0 at 88–91%, every
lower band at 100%) and that is an artefact, not a finding: scored on
positives only, the sole way to be wrong is to answer DENIED, so that curve
measures false denials. Which is how §7 was found.

---

## 7. The one real bug: R27 at 0.95 on a misread sense

Every confident false denial is R27, all at 0.95, and R27 is reasoning
correctly:

```
is a donkey a mammal              donkey.n.01 = "symbol of the Democratic
                                  Party", under emblem -> symbol ->
                                  abstraction. The animal is
                                  domestic ass.n.01.
is a hyacinth a flowering plant   hyacinth.n.01 = a zircon gemstone.
                                  The plant is hyacinth.n.02.
```

A sound inference about the wrong sense, delivered at maximum confidence,
which is worse than being unsure. This is the backlog's open
"context-sensitive sense choice" item with a cost attached.

**Pinning the sense fixes some of it and buys almost no coverage.** XCSLB
ships a sense key per concept; all 530 resolve through `nltk`
(`donkey%1:05:00::` → `domestic ass.n.01`). `identify.py` makes the same join
but keeps only the lemma half and drops the sense index.

```
crawl -> pinned    coverage 19.4% -> 19.9%   (+0.5)
                   accuracy 81.6% -> 83.9%   (+2.3)
                   false denials 10 -> 7
```

**This is the important negative.** If the 80% silence had been the reasoner
looking in the wrong place, pinning would have moved coverage a long way. It
moved it half a point. The silence is genuine absence, and §2 stands
unqualified.

### The residue is a different bug

What survives pinning is not about senses:

```
is a television modern
is a mussel aquatic
is a calf an infant
is pliers a garden tool     (weak gold — arguably correct)
```

`modern`, `aquatic`, `feminine`, `cold` are used adjectivally and all have
noun synsets, so R27 reads `is X <adjective>` as a taxonomy question and
excludes on it. **R27 should not answer a polar `is X Y` by taxonomy
exclusion when Y is being used as a property.** Distinct from the sense bug,
same 0.95 confidence.

---

## 8. Acted on: R27 withdrawn on a hedged predicate

Committed `8aaa0df`. `engine.py` already tried the property reading for a
hedged `is_a` and never reached it, because that fallback fires on UNKNOWN and
R27 answers CONTRADICTED. The exclusion now counts as having nothing, and is
withdrawn to UNKNOWN if the property reading finds nothing either.

Gated on the **determiner**, not on whether the target owns an adjective
sense. `animal` owns `animal.a.01`, so the data-side guard would have taken
`is a mouse an animal` with it. `sense_rank` does separate them (live
adjectival readings at 1-3, `animal.a.01` at 99) but that is a threshold
picked off six points.

`crawl`, same 1,200-per-rung sample, before and after:

| | coverage | confirmed | contradicted | accuracy | false denials |
| --- | --- | --- | --- | --- | --- |
| before | 19.4% | 16.9% | 0.8% | 81.6% | 10 |
| after | 19.2% | 17.1% | **0.4%** | 81.9% | **5** |

Half the false denials gone, and they were the 0.95-confidence half. The
aggregate barely moves because ten items in 1,200 cannot move it — the point
was never the average, it was that the errors were confident.

**R17 was masking this.** `is a television modern` reads VERIFIED on the
shipped engine and read CONTRADICTED on the crawl path, so the bug was
invisible on the 521 concepts the examples exercise and live on the other
44,678. Same shape as the R19 finding: a defect on the path that generalises,
hidden by the path that is tested.

The five that remain are not this bug:

```
is a hyacinth a flowering plant  |  sense selection -- pinning fixes all three
is a donkey a mammal             |
is a worm an invertebrate        |
is a calf an infant              |  weak gold: WordNet's infant.n.01 is human
is pliers a garden tool          |  weak gold, arguably correct
```

---

## 9. Acted on: COMPS foils are no longer denials

Committed after §8. `Profiles.denied` merged `denied_xcslb()` — 36,701 rows
— and none of them was a judgement. AwA2 stays, guarded, and the distinction
is now written into `Profiles.__init__`: **AwA2's matrix is closed**, every
class scored on all 85 attributes, so its zeros are real judgements and the
only question is what they mean. **XCSLB's is 1.58% dense**, so its zeros were
never judgements and there is nothing to guard.

What it cost, and what it bought:

| | before | after |
| --- | --- | --- |
| `is a violin made of wood` | CONTRADICTED | **absent, not false** |
| `do all birds fly` exceptions | 7: chicken, cockerel, emu, magpie… | **3: chicken, emu, penguin** |
| `does a boat have sails` | family disagreement | reached on a different predicate |

Cockerels and magpies fly. Four of the seven flightless birds were foils, and
the violin was denied because `violin` is COMPS' foil for `can be made of
ivory`, matched on the single term `made`. Two bugs stacked.

**Four v687 tests were encoding the bug rather than catching it.** Each
asserted a rule through one row of data, and when the row turned out to be a
foil the rule looked broken — `denial_hit`'s locative-tail logic was correct
throughout. They now assert the mechanism directly.

### And the machinery has never had a sound example

`does a boat have sails` was the **second** flagship example of v688's
family-disagreement claim to rest on absence read as denial, after
`does a beagle swim` rested on AwA2's zeros. `canoe NOT has sails` is a foil
in three flavours, and the norms positively assert `boat has sails`.

So the store was searched for a third. Sixty-eight questions through the loop;
four fired, and all four are artefacts:

```
does a rat swim     AwA2 zero on `swims`                  rats swim
does a zebra run    AwA2 zero                             zebras run
does a deer run     ("red" and "run") matched `red`,      deer run
                    and AwA2 says deer are not red
does a moth fly     denied by `can a tineoid fly in may`  moths fly
```

Enumerating what the machinery *could* fire on gives the structural reason:
with XCSLB's foils gone, the denial side is AwA2's 85 attributes plus crawled
qualified text, and the candidate pairs are things like
`container.n.01 / chewteeth` at 1/39. **Every family disagreement the loop can
currently report is grounded in something that is not a denial.**

Four tests asserted this claim. They now assert the mechanism, plus
`test_no_family_disagreement_in_the_store_is_currently_sound`, which records
the finding so it is noticed if it stops being true.

What survives is real and is not that claim: the boat's yes is **correct** —
the norms assert `boat has sails` — and it is reached through `can sail`,
which shares no word with `sails`. The loop catches a right answer arriving by
a wrong route, which one ask cannot do.

---

## 10. Acted on: sense selection, and the pattern it completes

The same structural bug, a third time. `engine.py` retries the other senses of
a subject only when the answer is UNKNOWN, and R27 answers CONTRADICTED — so
`is a donkey a mammal` denied on `donkey.n.01` (the symbol of the Democratic
Party) and never tried `domestic ass.n.01`. The comment directly above the
gate states the principle the gate breaks: *a word names a kind of something
if any of its senses does*.

**All three defects fixed today have one shape: a confident wrong answer
blocking the repair path a silent one would have taken.** R27's exclusion is
sound every time; what is wrong is treating soundness on one reading as
settling the question.

Retrying was not enough on its own. `hyacinth.n.02` is under `vascular plant`
and WordNet never joins it to `angiosperm.n.01`, so the right sense answers
UNKNOWN and the wrong sense's CONTRADICTED stood. The rule that works is the
one the question already carries: **the target picks the subject's sense.**
`mammal` is under `animal`, so `is a donkey a mammal` is about the donkey
under `animal`.

Gated on the target naming **exactly one** branch. `plant` is a factory and a
stooge as well as a herb, and matching on any of its three took `is a dog a
plant` — an andiron is an artifact, and so is a factory.

```
is a donkey a mammal              CONTRADICTED -> VERIFIED
is a worm an invertebrate         CONTRADICTED -> VERIFIED
is a hyacinth a flowering plant   CONTRADICTED -> UNKNOWN   (a hole in the tree)
is a dog a plant                  CONTRADICTED (unchanged)
```

### And it cost v688 its fourth example in one sitting

`is a mouse an animal` was on the page because v687 answered CONTRADICTED
about `mouse.n.04`, the device, and **the loop** re-asked it pinned to
`mouse.n.01`. v687 now picks the rodent itself — and the pin the loop used to
supply is the sense v687 now chooses, which is the point: the repair was real,
and so is its being unnecessary.

That was the `sense` generator's only instance. Twenty-four candidates were
tried for a replacement (`why does a dog bark`, `is a crane a bird`, `is a
bass a fish`, `is a date a fruit` …) and **none fires it**.

Running total for one session:

| v688 machinery | its example | what happened |
| --- | --- | --- |
| family disagreement | `does a beagle swim` | AwA2 zeros are not denials |
| family disagreement | `does a boat have sails` | COMPS foils are not denials |
| `scored_apart` doubt | `is a violin made of wood` | the denial was a foil |
| `sense` re-ask | `is a mouse an animal` | v687 resolves it now |

Four demonstrations, four removals, and no replacement found for three of
them despite deliberate search. **Each v687 defect fixed removes a
demonstration of v688's value, because the demonstrations were v687's
defects.**

---

## 11. Where the three fixes left it, measured

Same 1,200-per-rung sample, before and after R27's withdrawal, the foils, and
the sense rule:

| config | coverage | confirmed | contradicted | accuracy |
| --- | --- | --- | --- | --- |
| crawl before | 19.4% | 16.9% | 0.8% | 81.6% |
| **crawl after** | 19.1% | 17.2% | **0.2%** | **82.8%** |
| pinned before | 19.9% | 17.7% | 0.6% | 83.9% |
| pinned after | 19.7% | 17.8% | 0.2% | 84.1% |

**False denials on positives: 10 -> 2**, and both survivors are weak gold
rather than defects — `is pliers a garden tool`, and `is a calf an infant`
where WordNet's `infant.n.01` is human. Every false denial that was actually
wrong is gone.

**The gap `pinned` measured has half closed.** Being told the sense was worth
2.3 points over guessing it; it is worth 1.3 now. The remaining 1.3 is what a
better sense rule could still buy, and it is a smaller prize than it looked.

**Coverage did not move**: 19.4% -> 19.1%, which is noise plus a few
CONTRADICTED becoming UNKNOWN. Three real fixes, and the ceiling is exactly
where it was. That is the whole finding of this audit in one line — the rules
were not what was wrong.

---

## 12. Not acted on: R28, and what the benchmark cannot see

The verdict in §3 said the missing 79% is absence. That rested on `pinned`,
and pinning tests *sense selection* only — it says nothing about a fact
sitting in the store under different words. Sampling 120 missed positives:
**54% have a row sharing a content word**, and many are one shape.

```
bra      / "can be fastened"     capable_of: fasten in the front
leopard  / "can hunt"            capable_of: hunt at night, hunt monkey
hose     / "is used to wash"     used_for: washing car
```

R28 refuses those. So two ablations, both against `crawl`:

| config | coverage | confirmed | accuracy | foil ladder |
| --- | --- | --- | --- | --- |
| crawl | 19.1% | 17.2% | 82.8% | .933 .836 .802 .736 |
| stated (R28 off at distance 0) | 22.5% | 20.7% | 82.3% | .933 .815 .797 .744 |
| lenient (R28 off entirely) | 25.4% | 23.6% | 79.8% | .910 .786 .773 .722 |

**R28 costs a third of the reachable coverage for three points of accuracy.**
That bounds §3 rather than overturning it: 6.3 points of an 80.9-point gap is
8%, so data is still 92% of the problem — but "the store does not have it" was
doing work that "the store has it and R28 will not take it" should have done.

### The benchmark said ship it. The benchmark was wrong.

`stated` looks like a clear win: half of `lenient`'s coverage for a tenth of
its accuracy cost, and the taxonomic rung *improves*. It is not a win. It
resurrects the bugs this whole line of work started from:

```
                        strict     relaxed
can a rock swim        UNKNOWN    VERIFIED
can a fish walk        UNKNOWN    VERIFIED
can a person fly       UNKNOWN    VERIFIED
```

`rock capable_of "go for swim"` is stated **of rock, at distance zero**, so
the qualification problem is not about inheritance and the distance-0
hypothesis was simply wrong.

**XCSLB's 521 concepts are almost all leaves** — accordion, barrel, beagle —
so the gold set barely contains the class-level concepts where the relaxation
fails, and cannot see the failure. The v687 suite caught it, because those
cases were added by hand after the over-affirmation audit. *The benchmark is
necessary and not sufficient, and the hand-picked adversarial cases are still
earning their place.*

### Why there is no cheap fix

The failures are classes and the wins are leaves — rock 31 descendants, fish
617, person 10,296, against hose 5, leopard 2, bra 1 — and a threshold
anywhere between would separate all seven. That is a threshold picked off
seven hand-chosen points, which is the move this file exists to discourage.
The principled alternatives do not work either:

- **R19's corroboration** cannot clear a leaf: `leopard` has no norm-covered
  kinds, so it would refuse the wins along with the failures.
- **Grammar** cannot separate them: `hunt at night` and `walk on land` are
  both VERB + PP.
- The existing width constant (`WIDE = 1000`) does not separate them; rock
  and fish are both far below it.

What actually distinguishes `leopard hunt at night` from `fish walk on land`
is whether the class is homogeneous, which is corroboration, which needs the
norms, which cover 1.2% of concepts. **R28's bluntness is downstream of norm
sparsity, not a defect of the rule.** `R28_ON_STATED` is left True and carries
this reasoning.

> **Revised by §16.** The "three accuracy points" here were measured on foils
> that are absence, not denial, and two-thirds of them were the benchmark
> punishing the store for correctly affirming a true foil. On screened
> material relaxing R28 at distance zero costs **0.4 accuracy points**, not
> three. The rule survives anyway, on a different column: it buys 0.9 points
> of over-affirmation overall and 1.1 on the taxonomic rung, which is what it
> was written for and what nothing here could see until §16. The conclusion
> is unchanged and the reason for it is not.

---

## 13. GenericsKB, measured

570,717 facts over 21,424 concepts, from AI2's GenericsKB-Best. Loaded into a
copy of the store by `ingestion/load.py`; the baseline differs from it only by
this source. `crawl`, same 1,200-per-rung sample:

| store | coverage | contradicted | accuracy | foil ladder |
| --- | --- | --- | --- | --- |
| baseline | 19.1% | 0.2% | 82.8% | .933 .836 .802 .736 |
| + GenericsKB (all) | 23.9% | 0.5% | 79.1% | .890 .791 .767 .705 |
| **+ GenericsKB (score >= p50)** | **23.3%** | **0.2%** | **80.7%** | .905 .810 .781 .722 |

**I predicted 7 to 12 coverage points and got 4.8**, and did not predict an
accuracy cost at all.

Filtering to GenericsKB's own median score is most of the fix: half the data
(259,791 facts) gets 88% of the coverage gain, recovers half the accuracy, and
puts false denials back at baseline. The net is **+4.2 coverage for -2.1
accuracy** — which is roughly the same trade as turning R28 off (+6.3 for
-3.0), and that was rejected.

**The adversarial set is clean.** `can a rock swim`, `does a cat lay eggs`,
`can a fish walk`, `do pigs fly`, `can a person fly` all stay UNKNOWN. Per
§12, this check is not optional.

### It does not relieve R28, which was the argument for it

```
can a leopard hunt             UNKNOWN -> UNKNOWN
does a crocodile ambush prey   UNKNOWN -> UNKNOWN
```

GenericsKB holds *Leopards hunt at night.* and *Crocodiles ambush large
prey.* R28 refuses both, for the surplus. The corpus is generic in its
*subject*, not bare in its predicate, and the 15%-to-9% measurement in
`ingestion/genericskb.py` is that fact in advance.

### Part of the accuracy cost is the benchmark, not the data

Of the 30 pairs the baseline got right and GenericsKB got wrong:

```
"contains water"   ketchup vs tomato       tomato now verified
"contains water"   ketchup vs oil_tanker   oil_tanker now verified
```

Tomatoes contain water. GenericsKB is correctly affirming a true thing about
a *foil*, and the scoring counts it as an error, because the foils are
absence-derived (§1). But the ladder drops on `random` foils too (.933 ->
.905), where foil-truth is least likely, so real noise is present as well.

**Verdict: marginal, and a values call rather than a factual one.** It buys
reach at the price of precision, at about the same rate as the R28 relaxation
this file already rejected, and precision is what the three fixes above spent
the day buying back. Not merged into `build.py`. The branch is
`data/genericskb` and the numbers are here for when that trade looks
different.

---

## 14. Corrupted claims, and the measure that was missing

Every negative this file had was a COMPS foil, and §1 established those are
absence rather than denial — `carp can be a trophy` is a foil and is true.
So a **false-assertion rate was unmeasurable**, which is the one number
over-affirmation needs, and over-affirmation is what the audit that started
all this found by hand.

`audit.corrupted()` builds negatives instead: take a property only one XCSLB
category ever holds, held by at least three of its members, and put it on a
concept from another category. `can jam drool`. `does a puppet have a deck`.
`is a mitten an appliance`. 521 of them, one per concept, asked of every
configuration alongside the pairs.

This also lifts §12's limitation. XCSLB is leaf-heavy and cannot see
class-level over-generalisation; a corrupted claim can be built for any
concept at all.

| config | asserted | refused | silent | wrong when it spoke |
| --- | --- | --- | --- | --- |
| crawl | **1.0%** | 2.1% | 95.2% | 31.2% |
| llm (no floor) | 10.2% | 89.8% | 0.0% | 10.2% |

**The graph almost never over-affirms because it almost never speaks.** When
it does settle a false claim it is wrong a third of the time, on n=16.

### What the confidence floor does

| floor | false asserted | true asserted | true claims it is sure about |
| --- | --- | --- | --- |
| 0.00 | 10.2% | 82.5% | 100% |
| 0.90 | 4.6% | 89.0% | 75.2% |
| 0.95 | 3.3% | 91.8% | 66.8% |
| **0.99** | **1.0%** | **95.2%** | **41.5%** |

**At 0.99 the model matches the graph's over-affirmation rate exactly, while
reaching 41.5% of true claims against the graph's 17.2%.** Same safety, 2.3
times the coverage.

And the five corrupted claims it still asserts there — `is turnip used for
eating`, `can raspberry be served with ice cream`, `can an owl cling to
rocks` — are all true. The residue is this file's test set, not the model.

### The prompt was worth more than the threshold

Measured on 150 true and 150 corrupted claims:

    strategy      floor   says yes to TRUE   to FALSE   separation
    plain         0.99          98.9%           7.7%      +91.2%
    careful       0.99          96.0%           0.0%      +96.0%
    unsure        0.99         100.0%          10.7%      +89.3%
    challenged    0.90         100.0%          38.5%      +61.5%

Telling it *most claims put to you are false; say yes only if the property is
typical* costs three points of recall and removes essentially all the false
assertion. Two things that failed, recorded so they are not retried:

- **Offering `unsure` does nothing.** The model puts no probability mass on
  it even when invited, so abstention must be imposed from outside.
- **A second turn asking "are you certain?" makes it much worse.** It revises
  1.3% of the time and capitulates otherwise; false acceptance goes 7.7% ->
  38.5%. Taking the lower of the two confidences does not rescue it.

A prompt written specially for adjudication — "extra detail does not defeat
the claim" — scored 5/6 against `careful`'s 6/6. The general instruction to
be sceptical beat the specific instruction to be lenient.

---

## 15. A model, measured four ways

### It beats the store, and it never shuts up

`--config llm` puts the gold questions to SmolLM3-3B with no graph at all.

| | coverage | confirm | contra | accuracy | foil ladder |
| --- | --- | --- | --- | --- | --- |
| crawl | 19.1% | 17.2% | 0.2% | 82.8% | .933 .836 .802 .736 |
| **LLM alone** | **100%** | **86.9%** | 13.1% | **86.4%** | .936 .860 .824 .829 |

A 3B model on a laptop beats 1.9M facts on every measure, by 9 points on the
hardest rung. What the store still has is precision when it speaks — 0.2%
confident falsehoods against 13.1% — and the honesty to say nothing, which
the model never does. **A better answerer and a worse epistemic agent.**

### The prompt was worth more than the threshold

150 true claims against 150 built by corruption:

    strategy      floor   says yes to TRUE   to FALSE   separation
    plain         0.99          98.9%           7.7%      +91.2%
    careful       0.99          96.0%           0.0%      +96.0%
    unsure        0.99         100.0%          10.7%      +89.3%
    challenged    0.90         100.0%          38.5%      +61.5%

Two failures worth not repeating. **Offering `unsure` does nothing** — the
model puts no mass on it even when invited, so abstention must be imposed from
outside. **A second turn asking "are you certain?" makes it much worse**: it
revises 1.3% of the time and capitulates otherwise, 7.7% -> 38.5% false
acceptance, and taking the lower confidence does not rescue it.

At `careful` + 0.99 it asserts **1.0%** of corrupted claims — exactly the
store's own rate — while reaching 41.5% of true claims against 17.2%.

### Judgement is unstable to surface phrasing

The same three facts behind `can a person run`, worded three ways:

    run for short distances              supports 0.86   sup 0.88  ref 0.51   3/3
    capable of run for short distances   refuses  0.77   sup 0.94  ref 0.59   2/3
    can run for short distances          refuses  0.96   sup 0.94  sup 0.93   1/3

Tidier English scores worse and the tidiest supports `running shop`. This is
a limit of asking a model to judge, not a thing to tune away.

### Teaching: safe, and nearly useless

23,304 questions, 3,363 written at 0.99 (14.5% accepted), 409 concepts, 25
GPU-minutes.

| | coverage | confirm | accuracy | over-affirmed |
| --- | --- | --- | --- | --- |
| before | 19.1% | 17.2% | 82.8% | 1.0% |
| after | 19.8% | 18.0% | 76.4% | 1.0% |

**+0.7 coverage for -6.4 accuracy** — a worse trade than GenericsKB's
+4.2/-2.1, which was already marginal. The good half: **over-affirmation did
not move.** The floor and the prompt did their job; 3,363 written facts added
no measurable falsehood.

And the holdout, `bird`/`tool`/`fruit`, 95 concepts never written to:

| holdout only | coverage | confirm | accuracy | over-affirmed |
| --- | --- | --- | --- | --- |
| before | 21.6% | 17.8% | 73.5% | 2.1% |
| after | 21.6% | 17.8% | 73.1% | 2.1% |

**The control held** — nothing leaked — and **teaching does not generalise**.
409 taught concepts bought 95 untaught ones nothing. Covering 45,219 concepts
means asking about 45,219 concepts: ~2.6M questions, ~36 GPU-hours, no
dividend to wait for.

### And the benchmark blocked a third conclusion

Teaching wrote facts for all 409 non-held XCSLB concepts, and **COMPS foils
are XCSLB concepts**. Teaching a foil a *true* property makes it compete with
the held concept and the pair scores wrong. So part of that -6.4 is §1 again,
the same confound that inflated GenericsKB's cost and that made the R28
distance-zero relaxation look like a win.

Three conclusions this file could not settle cleanly, all for one reason:
**the foils are absence, not denial.** The corrupted claims of §14 are the
only unconfounded measure here, and they say teaching changed nothing.

---

## 16. Acted on: the foils are screened, and 40% of them were never negatives

2026-09-10. `research/v688/screen.py`, and `audit.py --gold screened`.

§1 found the flaw and every section since has worked around it. This fixes
it, and the first thing the fix produced was a measurement of how bad it was.

### The calibration, which is the whole argument

Screening a benchmark with a language model looks like the mistake §13 warns
about — the model is a *worse* epistemic agent than the store, 13.1% confident
falsehoods and never silent. The difference is that a screen is one bit, and
its two error rates can be **counted** against material whose truth is known
independently of both the model and XCSLB:

- **AwA2's continuous matrix**, which is *closed* — 50 classes each scored on
  every one of 85 attributes — so a low score is somebody saying no, which is
  exactly what XCSLB has never had. Taking only the ends (≤ 5 denied, ≥ 60
  held, against a published binary threshold at 20.8) gives 1,766 claims known
  false and 448 known true. `corpora.denied_awa2` has said this all along;
  what was missing was phrasing the 85 attributes as questions, which is the
  bulk of `screen.py`.
- **XCSLB's own ones** — 1,500 claims known true.
- **§14's corrupted claims** — 521 known false, and the only negative set no
  model had a hand in.

```
theta   denies awa2-F   denies corrupt-F   denies awa2-T   denies xcslb-T
0.50           91.9%              89.8%           17.6%           20.4%
0.75           88.3%              85.4%           13.2%           14.5%
0.90           81.4%              78.1%            9.2%            9.4%
0.95           75.8%              74.7%            8.3%            6.4%
0.99           58.3%              54.3%            4.5%            2.3%
```

Two things to read here. The separation is wide — 72 points at 0.90. And
**the two negative sets agree**: 81.4% against 78.1%, natural claims about
real animals against synthetic corruptions. §14 worried that `does an arm
have a bubble tube` was too easy to calibrate anything. It is not much easier
than `is a beaver blue`, which is the first evidence that the corrupted set
was measuring something after all.

### What COMPS is actually made of

With sensitivity R and false-alarm rate K known, the share p of an unlabelled
pool that is genuinely false follows from the rate F the screen fires at it:
`p = (F - K) / (R - K)`. That is prevalence from a screening test, and it is
what turns "the foils are suspect" into a number. All 36,701 distinct foil
claims, at theta 0.90:

| rung | n | screen denies | est. really false | **est. not a negative** |
| --- | --- | --- | --- | --- |
| random | 12,149 | 74.1% | 89.8% | **10.2%** |
| co-occurrence | 9,929 | 46.8% | 51.9% | **48.1%** |
| overlap | 7,830 | 38.6% | 40.6% | **59.4%** |
| taxonomic | 6,793 | 38.8% | 40.8% | **59.2%** |

**About 40% of COMPS' foils are not false, and on the three near rungs it is
about half.** The rung that matters most — taxonomic, the near miss the whole
ladder was built to test — is roughly a coin flip between a denial and a true
statement nobody happened to list.

That is the confound, quantified. A system scored on the base set is being
marked down, on the near rungs, for about half of what it correctly affirms.

**The estimate is a range, not a point.** It drifts with theta — the taxonomic
rung reads 55.3% false at 0.50 and 26.8% at 0.99 — and the drift has a
direction that means something: the screen is *less* confident denying foils
than denying AwA2 negatives, which is what near misses being harder looks
like. The true sensitivity on the taxonomic rung is therefore below the 81.4%
plugged in, and a lower R raises p. So **these figures overstate how broken
the benchmark is, and it is still this broken.** Sensitivity on the hardest
rung is not directly measurable with anything in this repository; that is the
honest limit of §16 and it is why the screen is a filter and not a label.

### The artifact

`data/xcslb/comps_screened.jsonl`, built at theta 0.95: **20,925 of 49,340
pairs** (random 8,317, co-occurrence 4,684, taxonomic 4,107, overlap 3,817),
each carrying the question the judge was asked and how sure it was. It is a
build product — `screen.py --build`, free from cache — and `audit.py` falls
back to `base` with a warning when it is absent, so a fresh clone still runs.

0.95 leaves about 6% of kept foils still true against roughly 40% before,
a sevenfold reduction, and keeps the near rungs populated. 0.99 gives 4.1%
contamination at half the size and is one flag away; the table is in
`screen.py` so the choice can be remade without another GPU hour.

### What this changes about reading the audit

A fourth column appears, and it is the one §14 was a stand-in for:

    denials   the foil side, scored as denial. `verified` is now an error.

That is over-affirmation on thousands of **natural near misses sorted by how
near**, where `corrupted` is 521 synthetic ones at no distance at all. Class
level over-affirmation that only shows up on the taxonomic rung — the kind
this project keeps finding by hand, `does a pig have wings` — is visible in a
column for the first time.

Two rules for using it:

1. **`base` and `screened` levels are not comparable.** They are different
   rows, and the screened set is easier by selection. Differences *between
   configurations* on one gold are what carry.
2. **`llm` is excluded from screened runs, in code.** It screened this gold,
   so it would be marking its own paper. `audit.py` drops it from the sweep
   and says why rather than reporting it with an asterisk. §13 is its honest
   measurement.

And the standing caveat: the screen shares a model with `teacher.py`.
Measuring a **taught** store against a model-screened benchmark is circular
in a way measuring an untaught one is not — the same judge decided what to
write and what counts as false. §14's corrupted claims stay in the report for
exactly that case and remain the arbiter whenever teaching is what is being
weighed.

### What it changed, measured

Five configurations, `--limit 600 --shards 5 --workers 4 --corrupt 0`, run
twice over the same store, differing only in which gold they were scored
against:

| config | gold | coverage | rel. acc | corrupted | **denials** | random | co-occ | overlap | taxonomic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crawl | base | 18.2% | 83.9% | 1.0% | — | | | | |
| crawl | screened | 25.8% | 87.1% | 1.0% | **6.7%** | 0.8% | 7.8% | 7.9% | 10.7% |
| stated | base | 21.2% | 82.7% | 1.7% | — | | | | |
| stated | screened | 28.1% | 86.7% | 1.7% | **7.6%** | 1.2% | 8.7% | 9.2% | 11.8% |
| lenient | base | 24.0% | 79.6% | 3.5% | — | | | | |
| lenient | screened | 31.2% | 85.7% | 3.5% | **9.2%** | 2.7% | 10.3% | 10.4% | 14.0% |
| pinned | base | 18.3% | 83.7% | 1.0% | — | | | | |
| pinned | screened | 26.2% | 87.3% | 1.0% | **6.8%** | 0.8% | 7.2% | 9.0% | 10.8% |
| corroborated | base | 13.3% | 87.6% | 0.4% | — | | | | |
| corroborated | screened | 19.2% | 90.9% | 0.4% | **2.5%** | 0.2% | 3.2% | 2.3% | 4.4% |

**Four findings.**

**1. §14's corrupted claims understated over-affirmation about sixfold.**
`crawl` asserts 1.0% of the synthetic corruptions and **6.7%** of screened
foils. `does an arm have a bubble tube` is not the shape of mistake this
system makes. The *ordering* of configurations survives intact — 0.4 < 1.0 <
1.7 < 3.5 against 2.5 < 6.7 < 7.6 < 9.2 — so §14 was reading the right
direction at the wrong scale, which is the most useful way for a proxy to be
wrong. The corrupted set keeps its job as the model-free arbiter; it is not a
level.

**2. Over-affirmation has a distance gradient, and it is the one the ladder
predicts.** `crawl` runs 0.8% → 7.8% → 7.9% → **10.7%** from the farthest
foil to the nearest. This is the first column in this file that can see class
level over-generalisation sorted by how near the miss is — the `does a pig
have wings` shape the project keeps finding by hand — and it rises
monotonically, which is the audit's own control saying the measurement is
real rather than noise.

**3. Most of R28's apparent accuracy benefit was the benchmark.** The
comparison that matters is within one gold, on one fixed sample:

```
                      base gold        screened gold
crawl -> stated       -1.2 acc          -0.4 acc
crawl -> lenient      -4.3 acc          -1.4 acc
```

**Two-thirds of the accuracy R28 appeared to buy was COMPS punishing the
store for correctly affirming true foils.** §12 priced R28 at "a third of the
reachable coverage for three accuracy points" and recommended keeping it on
that basis. On material where the foils are denials, relaxing R28 at distance
zero costs **0.4 points**, for +2.3 coverage.

**4. And R28 still earns its keep — on the axis it was written for.** The
relative score cannot see over-affirmation and the denials column can:
`crawl` 6.7% → `stated` 7.6% → `lenient` 9.2%, and on the taxonomic rung
10.7% → 11.8% → **14.0%**. R28's real product was never pair accuracy. It was
confident falsehood on near misses, which is exactly what "a qualified fact
does not affirm the bare claim" is a rule against. **The trade is now +2.3
coverage and −0.4 accuracy against +0.9 points of confident falsehood, +1.1
of it on the nearest rung.** That is a decision someone can actually make.
It was not one before, and I would still not take it.

The other two results are unchanged and now better supported. `pinned` is
`crawl` to within a rounding error on every column (26.2/87.3/6.8 against
25.8/87.1/6.7), so supplying the reader's sense still does not move the
aggregate — §11's finding, on cleaner material. And `corroborated` more than
halves over-affirmation, 6.7% → **2.5%**, and cuts the taxonomic rung from
10.7% to 4.4%, at 6.6 points of coverage; R19 remains advantaged here for the
reason the module docstring gives, so read it as the shape and not the size.

**One thing the table cannot say.** Levels are not comparable across the two
golds — screened coverage of 25.8% against base 18.2% is a different sample
of held claims, not an improvement, and the 3.2-point accuracy rise from
83.9% to 87.1% is an *upper* bound on the confound: removing contaminated
pairs also removes pairs that were hard for reasons the screen correlates
with. The within-gold deltas in findings 3 and 4 do not have this problem,
which is why the argument rests on them.

### The third blocked conclusion, settled: teaching cost nothing

§15 priced teaching at **+0.7 coverage for −6.4 accuracy** and could not say
how much of that loss was the foils — teaching wrote facts about the foil
concepts themselves, because COMPS' foils *are* XCSLB concepts. Same taught
store (`data/v684_taught.sqlite`), same `crawl` configuration, screened gold:

| all concepts | coverage | rel. acc | denials | taxonomic | corrupted |
| --- | --- | --- | --- | --- | --- |
| before | 25.8% | 87.1% | 6.7% | 10.7% | 1.0% |
| after teaching | 26.1% | 87.2% | 6.7% | 10.7% | 1.0% |

**The −6.4 was the benchmark. All of it.** On material where the foils are
denials, teaching costs **+0.1 accuracy** — nothing, within noise — and moves
over-affirmation not at all, on the near rungs or anywhere else.

The holdout is identical to the last decimal on every column, before and
after (n=360, 28.1% / 84.6% / 8.4% / 2.1%). The control held, and teaching
still **generalises not at all**, which §15 already had right.

**How much of this to believe.** The screen and the teacher are the same
model, and a claim it would teach is a claim it would not have denied, so
taught facts and screened foils are close to disjoint by construction. The
denials column therefore *cannot* show teaching's over-affirmation and its
0.0 change is not evidence. What is evidence is the **corrupted** column,
which no model touched: 1.0% → 1.0%, agreeing with §15's own base-gold
reading. Two independent measures say teaching did not make the store more
confidently wrong.

So the verdict on teaching changes, and only in one direction. It was
"+0.7 coverage for −6.4 accuracy", which is a bad trade. It is
**+0.3 to +0.7 coverage for nothing at all**, which is a cheap trade for a
small gain — and the reason not to scale it is unchanged and is §15's: no
transfer, and 45,219 concepts is ~36 GPU-hours. `TEACHING_FLOOR` at 0.99
threw away 85% of what the model knew; §14 says 0.95 keeps 79% at 3.3% false
assertion. That lever now has a clean measurement behind it and is worth
pulling, which it was not yesterday.

---

## 17. The hours should buy norms, not facts

2026-09-10. `research/v688/norms.py`.

§15 priced a flat teaching sweep: covering 45,219 concepts means ~2.6M
questions and ~36 GPU-hours, and §16 showed the +0.7 coverage it bought was
real but small. The question this section answers is what those hours should
be spent on instead.

**A fact about a beagle helps beagles.** A norm is different in kind: it is
not an answer, it is the evidence **R19** weighs before believing an
inherited fact, and one norm-covered sibling serves every question about
every member of its class. So the proposal is to distil norms. Three
measurements had to come first, and the first one kills the obvious version.

### R19 does not need more reach. It needs less altitude.

R19 can already fire for **70.5%** of the store's 45,219 fact-bearing
concepts — 86 ancestors carry eight or more norm-covered kinds. Norming every
remaining concept would take that to **71.5%**. One point. The 86 firing
ancestors are the huge ones and they sit above nearly everything.

But firing is not the same as saying something, and R19's own docstring is
the authority on that: `bird.n.01 "fly" 21 of 29` is signal, `animal.n.01
"wings" 20 of 143` is noise. So the real question is how *specific* the
ancestor is that each concept actually gets:

| narrowest corroborating ancestor | now | if fully normed |
| --- | --- | --- |
| under 300 kinds — says something | **1,709 (3.8%)** | 23,919 (52.9%) |
| 300–1,000 | 2,571 | 5,229 |
| 1,000+ — `animal.n.01`-level noise | **27,610 (61.0%)** | 3,193 |
| no corroborating ancestor at all | 13,329 | 12,878 |

**61% of concepts can only be corroborated against a thousand-plus
siblings**, which is the case the rule was written to avoid. Zero concepts
currently have an ancestor under 30 kinds; fully normed, 5,280 would.

### R19 is running on the bug §16 just fixed one level up

`corroboration` counts a kind as *not bearing a term out* when the norms
never asked. XCSLB averages **23.7 properties per concept out of a 3,592
feature lexicon — 0.66% dense.** AwA2 averages 31.2 of 85, **36.8%**.

On the 30 concepts in both corpora, with exact feature matching:

```
XCSLB feature      free-listed    AwA2 (closed) says    R19 verdict
has four legs         4/30              26/30             REFUSED
has a tail            3/30              25/30             REFUSED
can be fast           2/30              25/30             REFUSED
can walk              2/30              23/30             REFUSED
is agile              1/30              20/30             REFUSED
has sharp teeth       5/30              11/30             REFUSED
is domesticated       2/30              10/30             REFUSED
can swim              2/30               5/30             REFUSED
```

Every probe flips, all one direction. **Free listers mention what is
distinctive and never what is typical** — describing a leopard they say `can
pounce`, not `has four legs` — and typicality is the entire thing R19 tests.
This is absence-as-denial, live, in the shipped reasoner. It is also why
`corroborated` costs five points of coverage in §16's table.

### The experiment

AwA2 is the only closed gold here, so it can validate a norm source the way
it validated the screen. **4,150 cells — 50 classes × 83 phrasable
attributes — asked of SmolLM3 at 20.1/s**, roughly three minutes, most of it
already cached because `screen.py` asked the extremes of this same grid.

**Are distilled norms any good?** Against the closed matrix:

| floor | density | accuracy | precision | recall | accuracy on clear cells |
| --- | --- | --- | --- | --- | --- |
| 0.50 | 100.0% | 77.7% | 71.5% | 62.2% | **90.0%** |
| 0.90 | 76.2% | 83.5% | 78.0% | 47.7% | 94.0% |
| 0.99 | 46.9% | 89.3% | 84.8% | 29.0% | 96.8% |

77.7% raw looks poor until the last column: on cells humans were not split on
(continuous rating ≤ 5 or ≥ 60) it is **90.0% at full density**. Most of the
disagreement is the ambiguous middle — `is a leopard big`, `is a rat active`
— where AwA2's own raters were divided.

For contrast, on the 19 attributes XCSLB can express at all: of 238 cells the
closed matrix says are true, free listing mentions **58 — 24.4% recall.**

### Does it change what R19 concludes?

The measurement that matters, and it is not the same question. A source can
be individually noisy and still give R19 the right verdict, because a third
of eight siblings is a blunt threshold. Verdicts computed three ways over the
same ancestors and attributes, scored against AwA2:

```
four ancestors narrower than 400 (ruminant, even-toed ungulate,
ungulate, carnivore):

evidence                  pairs   agreed     lost     over   silent
distilled                   332    78.6%    14.5%     6.9%     0.0%
free listing                 76    46.1%    27.6%     1.3%    25.0%
distilled, same subset       76    81.6%    11.8%     6.6%     0.0%
```

**On the identical 76 comparisons, distilled norms agree with the closed
human matrix 81.6% of the time and free listing 46.1%.** Distilled nearly
doubles it, and the error profile is exactly what the sparsity argument
predicts:

- **Free listing's errors are almost entirely `lost`** — 27.6% here and 40.4%
  across all ancestors — true inheritances refused, with essentially no
  over-affirmation (1.3%). Sparsity makes R19 say no.
- **Distilled halves the losses and pays for it in over-affirmation**, 11.8%
  lost against 6.6% over. That is a real cost and it is the thing to watch.
- Free listing is **silent 25% of the time** even where it applies, because
  XCSLB covers 30 of the 50 classes and cannot reach eight kinds.

The result holds across all 15 ancestors (76.0% distilled, 46.0% free
listing), so it is not an artefact of the two narrow ones.

### What this does not show

**The per-ancestor ladder does not measure the altitude effect**, and it
would be easy to read it as though it did. Every row draws its kinds from the
same 50 classes, so `animal.n.01` is scored over 50 mammals rather than the
143 diverse kinds real R19 finds there. The dilution cannot appear in that
table by construction. The altitude evidence is the table at the top of this
section, and it is separate.

The simulation also computes R19's arithmetic rather than calling
`profile.corroboration`, deliberately: real R19 reaches its evidence through
`_hit`, a whole-word stem match between a fact's object and a norm's
predicate. Feeding it AwA2 attributes would measure that matcher as much as
the norms. **The evidence question is answered here; the matching question is
not, and it is the next one.**

And 50 classes over 83 attributes of one domain is a narrow base. Animals are
where free listing's typicality gap is most obvious; artefacts may differ.

### What it would cost

Not norms for concepts — **dense norms for eight members of each narrow
class**, chosen by greedy set cover over the 1,539 classes that are narrower
than 300 and have eight members available:

| concepts distilled | narrow classes firing | concepts with a real corroborator |
| --- | --- | --- |
| today | 26 | 3.8% |
| 1,000 | 510 | 19.7% |
| 2,000 | 810 | **31.2%** |
| 5,000 | 1,326 | 49.3% |

Median 26 questions to norm one member — the union of inheritable terms on
its narrow ancestors — with a heavy tail that needs capping; a few classes
carry 4,000+. At 20/s, **2,000 concepts is roughly two GPU-hours**, about 4%
of the flat sweep, aimed at the mechanism §16 measured as the largest single
quality lever (`corroborated` cuts over-affirmation 6.7% → 2.5%, taxonomic
10.7% → 4.4%).

### Verdict

**Worth doing, and the two things that had to be true are true.** Distilled
norms are 90% accurate where humans agree, and they nearly double R19's
agreement with a closed matrix against the free listing it uses today.

Three conditions on doing it:

1. **Keep distilled norms provenance-separable** from elicited ones.
   `identify.origin` already tracks `awa2` against `xcslb` and a third value
   belongs beside them. Without it the `shipped` control in this file stops
   meaning anything.
2. **Watch over-affirmation, not accuracy.** Density buys back refused
   inheritances and pays in confident falsehood — 1.3% → 6.6% here. That is
   the number that decides whether the floor should be 0.5 or 0.9.
3. **Answer the matching question before shipping.** `_hit` is a stem match,
   and a distilled norm phrased as `walks` has to meet a crawled fact
   phrased as `walk on land`. Nothing here tested that.

---

## 18. Acted on: R19's evidence is dense now, and it barely matters

2026-09-10. `research/v688/densify.py`, `corpora.load_distilled`,
`Profiles.corroboration`.

§17 said distil norms and gave three conditions. All three are met, the
mechanism works exactly as predicted, and **the end-to-end result is much
smaller than §17 implied.** Recording that gap is the point of this section.

### Condition 3 first: what R19 is actually asked

§17 simulated R19's arithmetic with AwA2 attribute names on both sides, so it
never tested `_hit`. Recording a live engine over 1,468 audit questions --
**1,503 consultations, 553 distinct (ancestor, term), 118 ancestors** --
changed the plan before it was built:

```
animal.n.01      317 calls   143 norm-covered kinds
plant.n.02       142          52
furniture.n.01   131          17
device.n.01       87          59
bird.n.01         85          29
```

The ancestors R19 reaches are **basic level**, chosen by where the crawl put
the fact rather than by how narrow the class is, and **not one has more than
150 norm-covered kinds**. So §17's greedy set cover over narrow classes was
the wrong target and no new concepts were needed: the 477 concepts XCSLB and
AwA2 already cover *are* the kinds R19 consults, and they are simply empty.
Of the 17,607 cells in the 8-to-150-kind band, the existing norms bear out
**10.1%**.

`_hit` matches a query word against any word of a predicate, so storing
`teacher.stated(relation, object)` -- `has_a` + `wing` becomes `has a wing`
-- makes the term reachable **by construction**. Condition 3 answered.

### The distillation

17,607 cells, **14.5 GPU-minutes at 19.8/s**, asked as the bare claim
`audit.phrase` builds. 4,137 predicates written at 0.90 and **1,408 at 0.99**
over 328 concepts. Relations `teacher.READS` cannot frame are dropped (100
cells): `has_prerequisite` + `cold or warm water` asks `does a gown cold or
warm water`, and the model answers such a question rather than refusing it.

### The mechanism works. 34.3% of R19's verdicts flip.

At 0.90, of the 391 recorded calls R19 could speak on, **134 flip — all
refused → believed**, and the named ones are precisely what free listing
misses:

```
mammal   blooded     4 -> 55 of 65        mammal   four      8 -> 59 of 65
mammal   warm        3 -> 55              mammal   legs     15 -> 59
mammal   fur        13 -> 50              bird     food      2 -> 27 of 29
```

`does a robin fly` goes from *21 of 29 kinds bear it out* to **28 of 29**.
§17's prediction is confirmed at scale on the real store.

### And it buys almost nothing

`corroborated`, screened gold, on the sample the fill was recorded from:

| distilled norms | coverage | confirmed | accuracy | denials | taxonomic | corrupted |
| --- | --- | --- | --- | --- | --- | --- |
| off | 20.3% | 18.2% | 88.9% | 3.8% | 5.8% | 0.4% |
| **on, floor 0.99** | **20.5%** | **18.4%** | **89.0%** | 3.8% | 5.8% | 0.4% |
| on, floor 0.90 | 20.5% | 18.4% | 88.0% | 4.0% | 5.8% | 0.4% |

**+0.2 coverage and +0.1 accuracy.** Not the transfer §17 was reaching for.
A larger sample (`--limit 600`, partly outside the recorded fill) reads
+0.6 coverage and −0.8 accuracy at 0.90; the matched sample above is the
honest one, and dilution was not the explanation — the effect is simply
small.

**Why §17 over-promised.** §17 measured R19's verdict over **AwA2's
attributes** — curated typicality claims like `has four legs`, `is furry`,
`can walk`, which are exactly what free listing omits. Real R19 is asked
about **the crawl's fact objects**, and those are mostly not typicality
claims at all: the recorded terms include `used`, `body`, `filled`,
`touching`, `conceal`, `riddled with bullet`. Densifying the evidence about
noise lets more noise through. At 0.90 `animal capable_of used` goes from 14
of 143 to 129, which is the whole of the accuracy loss.

So §17's limitation was not the one it declared. It said "50 classes of one
domain is a narrow base"; the real gap was the **property distribution**, and
that is the more useful lesson: *validating a mechanism on curated properties
says little about a workload made of crawled ones.*

### Where it leaves things

**Shipped at 0.99, on by default**, because it is a strict small improvement
with over-affirmation flat on both the screened foils and §14's model-free
corrupted claims. `V687_NO_DISTILLED_NORMS=1` ablates it in one run.

The three conditions held:

1. **Provenance is separable.** `identify.stated` does not merge distilled
   norms — a test asserts it — so the predicate trie, `_from_below`, the
   profile display and the `shipped` control are untouched. Only
   `Profiles.corroboration` reads them, and only the numerator: `kinds` still
   comes from `stated`, so the change is one term of one ratio.
2. **Over-affirmation was the number watched**, and the floor is where it
   lives. 0.90 costs a point of accuracy for the same coverage; 0.99 does
   not.
3. **`_hit` is answered by construction**, above.

**And pigs still do not fly.** `do pigs fly` and `does a pig have wings` are
UNKNOWN before and after, for the same reason as before — `only 1 of the 65
kinds of mammal` and `20 of the 143 kinds of animal` bear it out. The
distillation was asked about those cells and correctly declined to add
bearing, which is the negative control this whole change most needed to pass.

**The real conclusion is about the crawl, not about R19.** R19's evidence was
genuinely broken and is now genuinely fixed, and the store barely moved,
because R19 spends most of its time adjudicating claims like `an animal can
be riddled with bullet`. **Fact quality is upstream of everything measured
here**, and it is the first thing this file has pointed at that no amount of
better evidence or better rules can reach.

---

## 19. Three attacks on the crawl, two of which failed

2026-09-10. §18 said fact quality is upstream of everything and named no fix.
Three were tried. **One shipped, and the two failures are the more useful
half of the section**, because each failed for a reason worth not repeating.

### First, where the errors actually are

Screened gold, `crawl`, split by whether the leading evidence was stated of
the concept asked about or borrowed from an ancestor:

| affirmation | true | false | precision |
| --- | --- | --- | --- |
| stated (d=0) | 163 | 23 | **87.6%** |
| inherited (d≥1) | 343 | 132 | **72.2%** |

**The crawl is not wrong where it speaks.** Inheritance carries 68% of the
correct affirmations and 85% of the errors. And it is not evenly to blame:

| source of inherited evidence | right | wrong | precision |
| --- | --- | --- | --- |
| wordnet | 144 | 6 | 96.0% |
| conceptnet | 9 | 2 | 81.8% |
| **ascentpp** | 190 | 124 | **60.5%** |

`record_inherited.py` then narrowed it further. Over 1,468 questions,
inheritance supplies only **116 answers using 93 distinct facts**, and
**29 facts — every one of them Ascent++ — account for all the errors**. The
problem is not 1.9 million facts. It is dozens.

### Attack 1: a better dataset. Rejected, and the old verdict stands.

§13 scored GenericsKB at −2.1 accuracy on base gold, and §16 raised the
possibility that the foils had done that. Re-measured on screened gold:

| store | coverage | accuracy | over-affirmed | **corrupted** |
| --- | --- | --- | --- | --- |
| baseline | 25.8% | 87.1% | 6.7% | 1.0% |
| + GenericsKB | 31.2% | 85.0% | 9.7% | **3.3%** |

+5.4 coverage for −2.1 accuracy — **the same trade as on base gold**, so the
confound was never the explanation and my reason for re-opening it was
wrong. The model-free corrupted column **triples**, and nothing about a
benchmark can explain that. A second crawl of the same character adds
coverage and adds proportionally more falsehood.

### Attack 2: prune the class-level facts. No effect, for three reasons.

`prune.py` asked SmolLM3 whether each fact on a wide node is a claim about
the class, and demoted the ones it confidently denied: 21,235 candidates, 18
GPU-minutes, **2,985 demoted** — `animal at_location "black lagoon"`,
`"heaven"`, `"judas iscariot"`. `reason.py` gained R30, which skips a demoted
fact when `distance` is non-zero and keeps it where it was stated.

**The audit did not move by a single decimal.** Not one demoted fact was ever
the evidence for an answer. Three separate causes, all worth recording:

1. **Blast radius is not the same as being read.** The band was "1,000+
   descendants" on the reasoning that a wrong fact there reaches the most
   concepts. The facts that actually supply errors sit on `clothing.n.01`
   (28 kinds), `tree.n.01` (16), `boat.n.01` (8) — mid-sized nodes.
2. **The judge is unreliable about superordinates.** §17 validated it on
   concrete classes — leopards, beavers. Asked about abstractions it comes
   apart: `does a mammal have four legs` → **no**, 0.518; `is a tree
   deciduous` → **no**, 0.718; `can an animal kill people` → **yes**, 0.996.
   The same model, asked about *members* rather than the class, gets `mammal
   four legs` right at 56 of 65. **This is §18's lesson a second time, on a
   different axis**: a mechanism validated on one distribution says nothing
   about another, and abstraction level is a distribution.
3. **R19 already catches most of it.** `animal beak` is 31 of 143, `animal
   stealthy` 5 of 143, `animal kill` 10 of 143 — all refused before any
   pruning.

The machinery and `V687_NO_DEMOTION` are kept, and the artifact it built is
inert. Re-aiming it at `record_inherited.py`'s output is the version that
might work; asking a model about `animal.n.01` is not.

### Attack 3: R19 as a precondition. Measured, not shipped.

R19 is silent when the ancestor has fewer than 8 norm-covered kinds — 162 of
553 recorded calls, 29% — and inheritance then proceeds unchecked.
`V687_CORROBORATION_REQUIRED=1` inverts that.

It reads well: **18.1% coverage, 92.2% accuracy, 1.9% over-affirmed** against
19.8/90.1/2.8. And it is not shippable, for a reason the benchmark cannot
see: it turns **every gap in norm coverage into a refusal**. `does a beagle
bark` becomes UNKNOWN because none of the three dogs XCSLB covers was ever
asked whether it barks. Two attempts to rescue it — believing on any positive
support below the minimum, then applying the burden only to `CRAWLED`
sources — fixed `does a beagle swim` (3 of 3 kinds, refused for having too
small a sample, which is absurd) but not `bark`, because `bark` was never a
recorded term and so was never distilled. **The blocker is norm coverage, not
the rule.** Kept as a flag, off.

### Attack 4, which fell out of the diagnosis and did ship

The 29 culprits are not exotic. The ones R19 lets through sit just above its
floor: `clothing has_a sleeve` at **11 of 28** and `tree has_property
deciduous` at **7 of 16**. A third of a class bearing a property is not the
class bearing it, and a third was a guess. Swept:

| R19 floor | coverage | accuracy | over-affirmed | taxonomic rung |
| --- | --- | --- | --- | --- |
| 0.333 (was) | 19.8% | 90.1% | 2.8% | 4.8% |
| 0.400 | 19.2% | 90.8% | 2.4% | 4.0% |
| **0.500** | **18.7%** | **91.6%** | **2.2%** | **3.7%** |
| 0.600 | 18.4% | 91.7% | 2.1% | 3.7% |

**`CORROBORATION_FLOOR` is now 0.5.** −1.1 coverage for +1.5 accuracy and a
quarter off over-affirmation, and **no page example changes at any floor in
the sweep**. 0.6 buys a tenth of a point for another third of a point of
coverage, so 0.5 is the knee. `V687_CORROBORATION_FLOOR` overrides it.

That is the whole of what four attacks on the crawl produced: one constant.
It is also the first change in this file that improves accuracy *and*
over-affirmation together, rather than trading one for the other.

### What the three failures have in common

Each one assumed the problem was where the *volume* was — 1.9M facts, 1,000+
descendant nodes, 29% of R19 calls — and the measurements kept saying the
damage is somewhere small and specific instead. `record_inherited.py` exists
because of that and is the tool to reach for first next time: **find the
facts that were read before proposing to fix the ones that were written.**

---

## 20. Kinds, not just properties — and what still blocks the precondition

2026-09-10. `research/v688/kinds.py`, `corpora.load_distilled_kinds`,
`Profiles.witnesses`.

§18 added *properties* to the 571 concepts the norms cover. This adds
*concepts*, which is the half deferred twice, and it was deferred because
`Profiles.corroboration` draws its **denominator** from those 571 alone. When
an ancestor has fewer than eight beneath it R19 cannot speak: **83 of the 118
consulted ancestors, 195 of 1,503 calls.** `dog.n.01` has three covered
kinds. `turtle.n.02` has one.

### The design decision that mattered

A new kind would enter the denominator for **every** term asked at that
ancestor, including ones it was never asked about, where it contributes
nothing. Five thin kinds under `dog.n.01` take `bearing / 3` to `bearing / 8`
and make refusal *more* likely — §17's sparsity trap, self-inflicted.

Asking each new kind about everything avoids that and costs **11.2 GPU-hours**
(806,597 cells; a median of 2,621 inheritable facts per concept's ancestry).
Capping is arbitrary and only bounds the damage.

So a distilled kind **counts only for terms it was actually asked about**. The
artifact records `asked` beside `predicates`, and a witness joins the
denominator only where it has testimony:

```
dog.n.01  transport    0 of 3  ->  0 of 8    R19 can now speak
dog.n.01  bark         0 of 3  ->  0 of 3    no witness was asked; still silent
bird.n.01 fly         28 of 29 -> 30 of 35
```

That is R19 doing what it should have all along: speaking where there is
evidence, silent where there is none.

### What it cost and bought

319 concepts by greedy cover close all 66 feedable ancestors. **10,914 claims
asked, 9 GPU-minutes, 594 affirmed**, 312 witnesses with any testimony.
`corroborated`, screened gold:

| | coverage | confirmed | accuracy | over-affirmed | taxonomic |
| --- | --- | --- | --- | --- | --- |
| kinds off | 18.7% | 16.8% | 91.6% | 2.2% | 3.7% |
| **kinds on** | **18.4%** | **16.6%** | **92.2%** | **2.0%** | **3.5%** |
| kinds on + precondition | 17.8% | 15.9% | 93.9% | 1.5% | 2.4% |

−0.3 coverage for +0.6 accuracy and a tenth off over-affirmation, and **no
page example changes**. Shipped on; `V687_NO_DISTILLED_KINDS=1` ablates.

### The precondition is still blocked, and now by something else

With witnesses loaded it reads better than ever — **+1.7 accuracy, −0.5
over-affirmation, −0.6 coverage against kinds-on** — and it still turns `does
a beagle bark` into UNKNOWN. §19 said the blocker was *kind* coverage. It was
not: `beagle.n.01` is now a witness under `dog.n.01`. The blocker is **term**
coverage — `bark` is not a term COMPS ever asks, so nobody was ever asked it,
so there is no testimony either way.

And there is a deeper reason it cannot simply be widened. A free-listing norm
has **no record of what was asked**, so for XCSLB-derived kinds "nobody listed
it" and "nobody was asked" are indistinguishable. The precondition is only
safe over witnesses whose `asked` set is known, and those are exactly the 312
distilled ones. **Running R19 as a precondition would mean running it over
distilled witnesses alone** — a coherent design, and a different one from
what is here.

### The mistake worth recording

`test_reasoning` pinned `21 of 29` for `does a robin fly`. §18 moved it to
`28 of 29`, so the fix read the count from `corroboration` — and pinned the
*denominator* instead. §20 moved that to 35 and it broke again. Both readings
were correct on some checkout, because both artifacts are build products a
fresh clone lacks. **A test over a derived artifact has to assert the claim,
not the number**: R19 can speak here, a majority bear it out, and the note
quotes what it counted.

---

## 21. The overnight run: the largest gain in the audit, shipped off

2026-09-11. `research/v688/kinds.py --dense`, 8.42 GPU-hours.

§20 built witnesses cheaply — each asked only the terms the recording saw —
and noted the limit: `CORROBORATION_MIN_KINDS` needs **eight witnesses able
to speak to the same term at once**, and witnesses each covering a random
half of an ancestor's facts give an expected four. Density per witness is not
density they share.

So `--dense` organised the work per ancestor: ten concepts under each
consulted ancestor, each asked **every** inheritable fact on it.
**643,440 claims, 8.42 hours, 94 of 118 ancestors, 691 witnesses, 25,701
affirmed.** It ran to completion with no intervention.

### It works, and it is the biggest single gain measured

`corroborated`, screened gold:

| | coverage | confirmed | accuracy | over-affirmed | taxonomic |
| --- | --- | --- | --- | --- | --- |
| no witnesses | 18.7% | 16.8% | 91.6% | 2.2% | 3.7% |
| sparse witnesses (312) | 18.4% | 16.6% | 92.2% | 2.0% | 3.5% |
| **dense witnesses (691)** | 17.6% | 15.8% | **93.6%** | **1.6%** | **2.8%** |
| dense + precondition | 17.5% | 15.6% | 94.0% | 1.4% | 2.4% |

**−1.1 coverage for +2.0 accuracy, with over-affirmation down 27% and the
taxonomic rung down 24%.** Nothing else in this file bought that much.

R19's arithmetic moved exactly where it was supposed to:

```
dog.n.01   bark         0 of 3   ->  10 of 13   silent -> believes
clothing   sleeve      11 of 28  ->  12 of 38   believes -> refuses
tree       deciduous    7 of 16  ->   7 of 26   believes -> refuses
mammal     fly          1 of 65  ->   1 of 119
```

**And it settled the §20 question in the negative.** The precondition rule
was supposed to be what witnesses unlocked; with dense witnesses it adds
**+0.4 accuracy for −0.1 coverage** and is now nearly redundant. Dense
witnesses subsume it, because letting R19 *speak* where it was silent is what
the precondition was trying to force by fiat. The rule stays off, and now for
a better reason than "it breaks an example".

### And it ships off

`V687_DENSE_WITNESSES=1` opts in. Default is §20's behaviour.

Dense witnesses take **`can a dog fall into a hole` from VERIFIED to
UNKNOWN**: 0 of 17 canines on record bear the claim out. The judge is asked
whether a property is **typical**, and falling into a hole is something a dog
can do without being characteristic of dogs. **R19 tests typicality; a
`capable_of` question asks capability.** That mismatch was always there —
dense witnesses surface it at scale precisely because they let R19 speak
where it used to decline.

That example is not incidental. It is a card on **both** the v687 and v688
pages and it is load-bearing in four tests, one of which —
`test_an_ancestor_fact_is_ranked_by_what_it_accounts_for` — stops exercising
fact ranking at all once the fact it ranks is refused. Turning this on means
weakening a mechanism test with no replacement to hand.

**The benchmark and the example disagree, and both are right.** Screened
gold's claims are mostly property-shaped, so it rewards testing typicality;
`can a dog fall into a hole` is capability-shaped, where typicality is the
wrong test. §16's lesson applies to itself here: when a measurement and an
example disagree, ask what the measurement is made of.

Two things to settle before flipping it, in order:

1. **Separate capability from typicality.** The `careful` prompt drives this
   — "say yes only if the property is typical" — so it is a second prompt and
   a second calibration, not a phrasing tweak. §13's table is the template.
2. **Find a replacement ranking example**, or the test goes from asserting
   ranking to asserting a refusal.

### The cost of finding out

Eight and a half GPU-hours to learn that the mechanism works, is worth two
accuracy points, makes a rule we spent two sections on redundant, and cannot
ship until a prompt question is answered. The artifact is 1.1 MB, tracked,
and rebuilds free from cache. Nothing about it has to be run again.

---

## 22. Going after §21's gate found a bug under three sections of tuning

2026-09-11. §21 named two things to settle before dense witnesses could ship,
the first being that R19 tests typicality while a `capable_of` question asks
capability. Measuring **what dense witnesses actually lose** was meant to size
that. It found something else.

### The losses were not capability questions

The 23 true answers dense witnesses cost, listed:

```
does a dolphin have a blowhole     does a tortoise have a hard shell
does an airplane have an engine    does an ambulance have four wheels
is a bone porous                   does a violin make a screeching sound
```

Properties, not capabilities, and plainly true. R19's reason for refusing
them was `0 of the 14 kinds of whale on record bear that out` — about whales
and blowholes. The model had in fact affirmed them: `does a baleen whale have
blowhole`, yes at 0.996.

### The bug

`_hit` stems a term **as a whole** and compares it to each *word* of a
predicate, so it can only match a single word. R19 has two callers:

- `profile.verify` passes a term `route` has already reduced to content words.
- **`server.corroborate` passes the parsed target untouched** — `a blowhole`.

```
corroboration("whale.n.02", "blowhole")    ->  (6, 14)
corroboration("whale.n.02", "a blowhole")  ->  (0, 14)
```

Nothing matched, bearing was 0, and R19 refused. **As old as that call site**,
and invisible while the norms were sparse enough that a low count looked
ordinary. §21's witnesses made the denominator big enough for `0 of 14` to
read as absurd rather than unremarkable.

`Profiles.bears` now retries a multi-word term word by word with function
words dropped. It can only ever *raise* bearing, so it can only turn a
refusal into an acceptance.

### It made the benchmark worse, and it ships anyway

| | coverage | accuracy | over-affirmed | taxonomic |
| --- | --- | --- | --- | --- |
| before the fix | 18.4% | 92.2% | 2.0% | 3.5% |
| after the fix | 20.7% | 90.3% | 3.1% | 5.3% |

**The bug had been acting as a filter** — failing to match is indistinguishable
from demanding more evidence — and it was filtering well. Removing it costs
1.9 accuracy points and raises over-affirmation by half.

It ships because the mechanism was lying in its own notes. A page whose whole
claim is that you can read why it answered cannot tell you that no whale on
record has a blowhole. **A benchmark gain bought by a broken mechanism is not
a gain**, and the numbers it produced were never measuring what they said.

### Which means §19's floor was tuned against it

`CORROBORATION_FLOOR` went 1/3 → 0.5 in §19 and that sweep, and §20's and
§21's, all ran with the matcher broken. Re-swept:

| floor | coverage | accuracy | over-affirmed | taxonomic |
| --- | --- | --- | --- | --- |
| 0.5 | 20.7% | 90.3% | 3.1% | 5.3% |
| 0.7 | 19.0% | 91.0% | 2.5% | 4.6% |
| 0.8 | 18.4% | 92.2% | 2.2% | 4.0% |

0.8 recovers the pre-fix numbers exactly, which confirms what the bug was
doing. No page example changes anywhere in the range — **but 0.7 breaks an
answer the benchmark cannot see**: `does a dog have legs`. It is corroborated
at `animal.n.01`, where `leg` is borne out by **156 of 244 kinds — 64%**,
which is true, because fish and snakes have none.

So the floor stays at 0.5, now for a reason instead of a knee in a curve, and
the reason is **altitude again**. R19 checks the level the crawl attached the
sentence to, not the nearest ancestor that could speak: `canine.n.02` is 12 of
12 for legs and would clear any floor. **Until R19 corroborates where the
evidence is sharpest, the floor is bounded by the vaguest level at which a
true property can be stated.** That is the sharpest statement of the altitude
problem this file has reached, and it is now a concrete piece of work rather
than an observation.

### Where §21's gates stand

Gate 1 is **not** what §21 said it was. The capability/typicality mismatch is
real and explains `can a dog fall into a hole`, but it explained none of the
23 losses, which were this bug. Dense witnesses are still off and still break
that one example.

Gate 2 — a replacement example for the ranking test — is untouched and stays
untouched while dense is off.

**The lesson is the one §16 taught about the benchmark, applied to a rule:**
three sections of constant-tuning sat on a mechanism nobody had checked. The
constants were fitted to the bug. Measuring what a change *loses*, one
example at a time, is what found it — the aggregate never would have.

---

## 23. Acted on: corroborate where the evidence is, not where the crawl put it

2026-09-11. `Profiles.sharpest`.

§22 ended on the sharpest statement of the altitude problem this file had
reached: **the corroboration floor is bounded by the vaguest level at which a
true property can be stated.** `does a dog have legs` is checked at
`animal.n.01`, where `leg` is borne out by 156 of 244 kinds — 64%, and true,
because fish and snakes have none. `canine.n.02` is 12 of 12. `dog.n.01` is 8
of 8. The floor could not go past 0.64 without the dog losing its legs.

R19 checked `animal.n.01` because that is where the crawl attached the
sentence. That is not the level that knows most about the question.

### The change

Walk from the asked concept up to the attached level and take the **first**
rung with enough kinds to speak. Two constraints keep it from becoming a way
of shopping for a verdict: it stops at the first rung that can speak rather
than the most agreeable one, and it **never climbs above where the fact was
attached**, so a fact stated of animals is still judged as a claim about
animals whenever nothing narrower can be asked.

Checked by hand before implementing, which is what made it worth building:

```
                              attached            sharpest
does a dog have legs          animal 156/244      dog 8 of 8
do pigs fly                   mammal 1/111        even-toed ungulate 0/18
does a pig have wings         animal 20/143       even-toed ungulate 0/11
does a flamingo have talons   bird 15/29          aquatic bird 3 of 8
```

The refusals that should hold, hold — and hold on **sharper** evidence. `0 of
18 even-toed ungulates fly` is a better reason than `1 of 111 mammals do`,
and the note now says so.

### It is not a benchmark win. It is what makes the floor movable.

| level | floor | coverage | accuracy | over-affirmed | taxonomic |
| --- | --- | --- | --- | --- | --- |
| attached | 0.5 | 20.7% | 90.3% | 3.1% | 5.3% |
| attached | 0.8 | 18.4% | 92.2% | 2.2% | 4.0% |
| sharpest | 0.5 | 21.4% | 90.5% | 3.4% | 6.2% |
| **sharpest** | **0.8** | **18.6%** | **92.1%** | **2.2%** | **3.9%** |

At every floor, sharpest is worth about two tenths of a point. **Its value is
that `attached + 0.8` breaks `does a dog have legs` and `sharpest + 0.8` does
not.** The gain is not in the row; it is in which rows are reachable.

**Shipped: `CORROBORATION_FLOOR` 0.5 → 0.8, sharpest on.** Against the
mechanism as it stood at the start of today:

```
              coverage  accuracy  over-affirmed  taxonomic
before          20.7%     90.3%          3.1%       5.3%
after           18.6%     92.1%          2.2%       3.9%
```

−2.1 coverage for **+1.8 accuracy, 29% less over-affirmation and 26% less on
the taxonomic rung**, with 535 tests and every page example unchanged. That
recovers what §22's matcher fix cost, and recovers it honestly — the numbers
before the fix came from a mechanism that reported `0 of the 14 kinds of
whale` about blowholes.

### Two tests stopped asserting things that had become incidental

`test_a_class_fact_is_put_to_the_class_before_it_is_inherited` pinned the
word `mammal` in the note. The note now says `even-toed ungulate` and
`feline` — the same refusals, named at the level that produced them. It
asserts instead that R19 ran, refused, and named **a class the concept
actually belongs to**, which is what the test was for.

That is the third test this week to pin a number or a name that a better
mechanism moved. The pattern is worth naming: **a test over a derived
quantity should assert the claim, not the reading.**

### What this does not fix

`sharpest` needs a level with eight norm-covered kinds between the concept
and the fact. Where there is none it falls back to the attached level and
altitude still bites — `does a dolphin have a blowhole` is checked at
`whale.n.02`, 1 of 4, and only survives because four is too thin to refuse
on. More witnesses would help there; §21's dense run is sitting behind a flag
for an unrelated reason.

---

## 24. R19 does not want a capability judge

2026-09-11. `research/v688/capability.py`, `teacher.CAPABLE`.

§21's remaining gate. `teacher.CAREFUL` says *say yes only if the property is
typical of that kind*, which is right for `has_a` and looked wrong for
`capable_of`: falling into a hole is not characteristic of beagles though
every one of them can.

### Gold nobody had to build

§16's screened benchmark is labelled on both sides and **4,766 of its 20,925
pairs are capability-shaped** — the property begins with `can`. That gives
**2,775 true and 4,008 false capability claims** for free.

AwA2 was the obvious alternative and is unusable here: its capability
attributes are scored for typicality too — `swims` is 0 for collie and
dalmatian, and dogs swim — so it would measure the prompt against the belief
the prompt is meant to correct.

### The prompt is much better at its job

400 true and 400 false claims, both prompts:

| prompt | floor | yes to true | yes to false | separation |
| --- | --- | --- | --- | --- |
| careful | 0.99 | 33.8% | 0.0% | 33.8% |
| **capable** | 0.99 | **73.0%** | **0.0%** | **73.0%** |

At the floor these cells are written at, the capability prompt affirms more
than twice as many true capabilities with **zero** false positives. Both
answer all eight of R19's guard claims correctly — `a fish walks`, `a rock
swims` and `a pig flies` stay no under both.

### And it makes the system worse

Re-distilled with it, every artifact grew about threefold — 1,454 predicates
to 4,789 — and `corroborated` on screened gold went:

| prompt | floor | coverage | accuracy | over-affirmed | taxonomic |
| --- | --- | --- | --- | --- | --- |
| careful | 0.80 | 18.6% | **92.1%** | **2.2%** | 3.9% |
| capable | 0.80 | 18.9% | 90.9% | 2.5% | 4.0% |
| capable | 0.85 | 18.8% | 91.1% | 2.4% | 3.9% |
| capable | 0.90 | 18.5% | 91.4% | 2.3% | 3.9% |

It loses at every floor. Sweeping to 0.9 recovers most of the coverage
difference and none of the accuracy.

### Why, and it is not a subtlety

**R19's question is not "can this thing do that".** It is *is this inherited
fact a claim about the class* — and `can fall into a hole` is not a property
of canines in any useful sense, however true it is of every canine. A judge
that affirms every true capability affirms every vacuous one too, and R19
then has nothing to discriminate with.

So **typicality is the right test even for `capable_of`**, and `careful` was
answering the right question all along. §21 predicted the opposite and §21
was wrong.

That also retires §21's reading of `can a dog fall into a hole`. Dense
witnesses refusing it is not a prompt defect to be fixed; it is R19 declining
to treat a vacuous capability as a class generalisation. Whether that is the
*right* answer for a reader is a separate question from whether the mechanism
is behaving as designed, and the mechanism is.

**Reverted to `careful`.** The prompt, the namespaced cache and
`capability.py` are kept, because §21 predicted the opposite and somebody
will want to try it again — the table above is so they do not have to.

### One thing nearly lost

`kinds --build` covers 319 concepts and writes what it touched; `--dense`
covers 691 and gives each an `asked_at`. Running the first after the second
**silently discarded the overnight run**, and only a backup taken for an
unrelated reason saved it. It merges now, and a test asserts the artifact
still carries more than 500 dense witnesses.

---

## 25. Dense witnesses on, and the floor goes down to pay for it

2026-09-11. Asked for directly: turn dense witnesses on and accept losing
`can a dog fall into a hole`. §24 had already retired the objection to that
answer — R19 asks whether a fact is a claim about the class, and falling into
holes is no more a property of canines than of chairs.

Turning it on found a second casualty nobody had sanctioned, and it is the
more instructive one.

### `does a beagle swim` broke, and not for the reason §21 gave

That is the example v688 was built around, with a test named after it
(`test_an_awa2_zero_is_not_a_denial`) and a finding behind it: AwA2 scores
collie, dalmatian and german shepherd 0 for `swims`, and a 0 there means *not
characteristic*, not *false*. Beagles swim.

The cause was **not** dense witnesses denying it. Nine of the thirteen dogs on
record bear it out. The cause was the **0.8 floor refusing a 69% majority.**

```
dog.n.01   swim       9 of 13   69%
dog.n.01   breathe    8 of 13   62%
```

Every dog breathes. A `careful` witness is asked whether a property is
*typical*, and for a property every member has but none is known for, it says
no — so the ratio lands in the sixties. §24 already established that a
capability prompt makes the aggregate worse, so this is not fixable by asking
differently.

And refusing at 62% would print `8 of the 13 kinds of dog on record bear that
out` and then call the claim absent, which reads as the system arguing with
itself. That is the incoherence this project has been caught by twice before.

**So the floor sits under the majorities it means to believe: 0.6.** 0.65
costs the beagle its breath.

### What that costs

| | coverage | accuracy | over-affirmed | taxonomic |
| --- | --- | --- | --- | --- |
| start of the day | 20.7% | 90.3% | 3.1% | 5.3% |
| sparse witnesses, floor 0.8 | 18.6% | **92.1%** | **2.2%** | 3.9% |
| **shipped: dense, floor 0.6** | **19.0%** | 91.5% | 2.4% | 4.2% |
| dense, floor 0.8 | 17.7% | 93.5% | 1.6% | 2.9% |

**Turning dense witnesses on is, on the benchmark, a wash or slightly
negative** — it forces the floor down by more than the extra evidence buys
back. The best-scoring row is dense at 0.8, and it refuses two true
majorities and explains itself incoherently while doing so.

What is shipped is better *epistemically* and not on the scoreboard: R19 can
now speak at 94 ancestors it was silent at, and every page example holds.
`sparse + 0.8` scores marginally better and is one flag away
(`V687_SPARSE_WITNESSES=1 V687_CORROBORATION_FLOOR=0.8`); the two are close
enough that the choice is a judgement about what the store is for.

### The pattern, stated plainly

Three sections running, a mechanism improved and the aggregate did not:

- §18 dense norms: mechanism flips 34.3% of verdicts, +0.2 points.
- §24 capability prompt: twice the judge, −1.2 accuracy.
- §25 dense witnesses: R19 speaks in 94 more places, a wash.

Each time the improvement was real and each time something downstream — a
constant, a prompt's question, a floor — had been fitted to the defect. **The
system's numbers are not a sum of its parts' quality**, and a component
measured alone predicts almost nothing about it.

### Cards and tests that moved

`can a dog fall into a hole` now answers UNKNOWN on both pages, with cards
saying why. It was swapped off v687's `Taxonomy & facts` card, which promises
a settled verdict, and `does a beagle swim` took its place.

Two v687 tests changed what they assert without changing what they protect.
`test_the_norms_hand_back_what_they_did_not_answer` checks the hand-back, not
the verdict after it, and now checks the fact is still found and cited behind
the refusal. `test_an_ancestor_fact_is_ranked_by_what_it_accounts_for` read
the ranking off `predicate`, which tested the sort **and** the corroboration;
it reads the detail now, which is where the ranking is visible whether or not
R19 then refuses the winner. Both assert on `asked=["fall","hole"]` giving
`fall into hole` and `asked=["fall"]` giving `fall victim` — the sort key,
which is what the docstring was ever about.

That is the **fourth** test this week to pin a reading rather than a claim,
including one of mine from §23 that asserted the floor was above 64%.

---

## 26. Counting the teacher's no: negation fails, and the no was sound

2026-09-11. `research/v688/negation.py`, and a read-back of the judgement
cache.

The loop now puts an unsettled question to the model bare (`teacher.py`), and
at first counted only a yes at 0.99. The no was left out for fear of
`CAREFUL`'s lean — it tells the model most claims are false — so the proposal
was to ask each question negated too, and trust a no only when the negation
came back yes. A second way was measured beside it: the same question under
`CREDULOUS`, a prompt leaning the other way.

400 claims from each of `screen.truth_sets`, each asked three ways, floor
0.99. Cells are right / wrong, in percent; *right* is yes on a true set and
no on a false one.

| rule | AwA2 false | AwA2 true | XCSLB true | corrupted |
| --- | --- | --- | --- | --- |
| yes only (as shipped) | 0.0 / 1.2 | 49.8 / 0.0 | 29.5 / 0.0 | 0.0 / 0.8 |
| `careful`, counting its no | **56.0** / 1.2 | 49.8 / **4.5** | 29.5 / **1.3** | **53.3** / 0.8 |
| negation | 1.5 / 1.2 | 49.8 / 0.5 | 29.5 / 0.0 | 0.0 / 0.8 |
| mirror | 14.5 / 1.2 | 49.8 / 1.0 | 29.5 / 0.5 | 19.1 / 0.8 |

### Negation does not work under this prompt

The negated question is answered no as well: **55.0%** of AwA2's false claims
and **68.3%** of the corrupted ones get no to both. `CAREFUL` says yes only to
what is typical, and *not having flippers* is not a typical property of
anything, so the rule abstains on nearly everything and counts almost no no.
The phrasing was not the problem — `is a dog unable to swim`, not `can a dog
not swim`, which asks whether it can refrain.

### The mirror adds nothing at the floor

Every one of `careful`'s confident no's, on all four sets, is one `CREDULOUS`
also leans no on. The lean only moves answers below 0.99. A second GPU call
per question, for no change.

### And the no was sound all along

With equal numbers of true and false claims, a confident no is wrong 7.4% of
the time on AwA2 and 2.4% on XCSLB against corrupted, where the confident yes
is wrong 2.4% and 2.6%. The wrong no's on AwA2 are mostly colour asked of a
partly coloured animal — `is a zebra black`, `is a dalmatian white`, `is a
killer whale white` — and XCSLB's include `is a hyena a large cat`, which is
the gold being wrong. The genuine misses are few, `is a lion furry` and `can
an elephant squirt out water` among them, and the yes has noise of the same
shape: `does a deer have paw pads`.

The fear came from §25, where `careful` witnesses doubted that 5 of 13 kinds
of dog breathe. Those doubts were never confident. Of the 63 cached bare
questions about breathing, swimming and legs that it denies at 0.99, none
denies breathing to an animal (unless `can kiwi breathe` meant the bird), and
the clear errors are `can a giant panda swim` and `can an eagle swim`. `does a
dog breathe`, asked directly, is yes at 0.999.

**Shipped:** an answer at 0.99 settles an unsettled headline either way,
priced the same by `confidence.RATIFIED`. Negation and the mirror are kept as
measurements, not used.

---

## 27. Challenging a yes that nothing bore out

2026-09-11. `research/v688/challenge.py`.

The teacher was asked only what the store left open (§26), so the store's own
over-affirmation never reached it. `can a fish walk on land` is VERIFIED on
one Ascent++ row, `fish capable_of "walk on land"`, which is about
mudskippers. The run finds nothing for or against it, and the page read
`unchallenged`.

`Teacher.challenge` now puts that headline to the model bare, after the last
cycle, when three things hold: it is a polar yes; its evidence is a crawled
row (Ascent++ or ConceptNet, not WordNet, the norms or a row the teacher wrote
itself); and the run's trust would otherwise read `unchallenged`.

What a confident no may do was measured first. Every polar yes on a crawled
row in the most recent audit run (`audit-out/dense-now`, screened gold) was
joined to the model's bare answer, 139 of them newly asked. That is a
superset of what the loop challenges -- the loop skips a yes its run bore out
-- so the cost below is an upper bound. Floor 0.99.

| set | n | disputed | agreed | below the floor |
| --- | --- | --- | --- | --- |
| listed features (true) | 181 | **2.2%** | 46.4% | 51.4% |
| COMPS foils (truth open) | 31 | 51.6% | 0.0% | 48.4% |
| corrupted (false) | 0 | | | |

### The cost is small, and every one of it is an error

The four true claims disputed are `can a flamingo build nests`, `can a bottle
store milk`, `is a rose sharp` and `does a flamingo eat algae`. All are true
and all are stated at distance 0. On gold, every dispute is wrong, at 2.2% of
the yes answers the store gives on crawled rows. Half the time the model is
below the floor and does nothing.

### The gain is where the gold cannot look

No corrupted claim is verified on a crawled row, so the one model-free set
has nothing to say. The foils do: 16 of 31 disputed, and they read like
over-affirmation caught -- `can an emu fly`, `can a dolphin lay eggs`, `does
an apple grow on palm trees`, `is a pine tree deciduous`, `does a canoe have
sails`. But screened gold is the set this same model denied at 0.95 (§16), so
half of it clearing 0.99 is partly built in. It is described, not scored.

The class-level questions are where the motivation lives, and XCSLB is almost
all leaves, so they are read one at a time:

| question | truth | teacher | reading |
| --- | --- | --- | --- |
| can a fish walk on land | false | no, 0.996 | disputed |
| can a fish walk | false | no, 0.998 | disputed |
| does a cat lay eggs | false | no, 0.999 | disputed |
| can a rock swim | false | no, 0.998 | disputed |
| can a person fly | false | no, 0.977 | below |
| can a mammal fly | false | no, 0.931 | below |
| does an animal have wings | false | no, 0.898 | below |
| can an animal fly | false | **yes**, 0.962 | below |
| can a bird fly, does an animal breathe, can a fish swim, can a person walk, does a dog have a tail, can a horse run | true | yes, 0.992 to 0.998 | agreed |
| does a dog bark, does a mammal have fur | true | yes, 0.989 | below |

Four of eight false class-level claims disputed, and none of eight true ones.
`can an animal fly` is not caught and cannot be: the model believes it.

### What a dispute does

A confident no against an unchallenged yes **unsettles** it. The outcome reads
`unknown`, the trust `disputed by the teacher`, and the lines name the row and
what the model said. It is not turned into a no: a crawled row and a model
disagree, nothing corroborates either, and neither outranks the other the way
the store's own argument outranks a model over an overturned headline (§26).
The cost is paid in abstention rather than falsehood -- at most 2.2% of
correct yes answers become `unknown`, and nothing false is asserted.

A confident yes changes the trust phrase to `unchallenged; the teacher
agrees` and nothing else. A model agreeing is not a family bearing a claim
out, and pricing it as one would be counting the teacher twice.

`audit.py` does not see any of this: its loop configuration runs without a
teacher, so no benchmark number above moves.

---

## 28. Definitions memory against the audit

2026-09-12. `research/v689/learn_definitions.py`, `ingestion.load --source
definitions`, `research/v689/definition_guards.py`.

All 82,115 noun glosses in the store were read into facts -- 69,235 of them,
with a genus found in 87.3% of glosses and agreeing with the taxonomy in 64.5%
-- and loaded into a copy of the store as one more source, `definition`, the
way GenericsKB was in §13. A second copy took only the 51,492 facts from
glosses whose genus agreed with the taxonomy.

The teacher check was skipped by decision. The errors in this data are the
parser misreading curated text, which a judge of truth and typicality only
partly sees -- `can an african daisy have daisylike flowers` is true and
misfiled -- so the audit was asked first whether there was anything to catch.
Three shapes the check was meant for were settled by rule before loading.

Screened gold, `--limit 600`, every store measured in the same session:

| store | config | coverage | confirmed | contradicted | pairs decided | pair accuracy | corrupted asserted | denials asserted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| store | corroborated | 19.02% | 17.17% | 0.09% | 446 | 91.48% | 0.58% | 55 |
| + definitions | corroborated | **19.58%** | 17.74% | 0.09% | 459 | **91.94%** | 0.58% | 55 |
| + genus agrees | corroborated | 19.54% | 17.69% | 0.09% | 457 | 91.90% | 0.58% | 55 |
| store | crawl | 25.78% | 23.94% | 0.09% | 610 | 87.05% | 0.96% | 155 |
| + definitions | crawl | **26.25%** | 24.41% | 0.09% | 622 | **87.46%** | 0.96% | 155 |
| + genus agrees | crawl | 26.21% | 24.36% | 0.09% | 620 | 87.42% | 0.96% | 155 |

### A small gain, and no cost anywhere it was measured

About half a coverage point in both configurations, with accuracy up beside
it, because every answer that rested on a definition fact was right: 17 of 17
under `corroborated` and 23 of 23 under `crawl`. Every over-affirmation
measure is identical to the row: corrupted claims asserted, screened denials
asserted, contradicted. Nothing the definitions added made the store assert
a false thing it did not already assert.

The gain is small for a reason worth stating. XCSLB lists what people say
about a concept -- `has four legs`, `is used for cutting` -- and a gloss says
what distinguishes it from its genus. The two overlap in a sliver of the gold.

### The class-level guards

31 questions, v687 alone over each store. Right 17 -> 19, wrong 3 -> 3,
unknown 11 -> 9. Both changes are right: `can a cat roar` is denied, from `no
ability to roar`, and `can a testicle secrete androgens` is verified. The
three wrong answers are the store's own and unchanged: `can a fish walk on
land`, `can an animal fly`, `does an animal have wings` (§27). `is a kitten
old` stays unknown, because v687 has no antonymy; the v689 page, which reads
WordNet antonyms, says the definition rules it out.

### What this decides

- **The teacher check is not worth its GPU hour now.** Every measure it would
  have protected shows nothing to protect. It would remove garbled facts the
  gold never asks about, at §27's measured cost of disputing about 2% of true
  facts that are unusual rather than typical.
- **The genus filter buys nothing.** It drops a quarter of the facts and a
  tenth of the gain and is exactly as safe. The mark stays in the data and
  does not gate what is read.
- **Definitions memory stays on**, read by v689 after what was told and
  before the store rows. It is not built into the store: a rebuild is where a
  source goes once it has earned its place, and half a point has not yet.

---

## 29. More definitions: Open English WordNet, Wikipedia, Wiktionary

2026-09-12. `ingestion/oewn.py`; `ingestion/wikipedia.py` and
`research/v689/articles.py`; `research/v689/learn_wiktionary.py`. Each source
is its own definitions memory, layered onto §28's store copy with
`ingestion.load --extra` and audited as §28 was: screened gold, `--limit 600`.

| store | config | coverage | confirmed | contradicted | pairs decided | pair accuracy | corrupted asserted | denials asserted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| §28 definitions | corroborated | 19.58% | 17.74% | 0.09% | 459 | 91.94% | 0.58% | 55 |
| + OEWN re-reads | corroborated | 19.58% | 17.74% | 0.09% | 459 | 91.94% | 0.58% | 55 |
| + Wikipedia leads | corroborated | **19.73%** | 17.88% | 0.09% | 462 | **92.21%** | 0.58% | 55 |
| §28 definitions | crawl | 26.25% | 24.41% | 0.09% | 622 | 87.46% | 0.96% | 155 |
| + OEWN re-reads | crawl | 26.25% | 24.41% | 0.09% | 622 | 87.46% | 0.96% | 155 |
| + Wikipedia leads | crawl | **26.40%** | 24.55% | 0.09% | 625 | **87.68%** | 0.96% | 155 |

### Open English WordNet: nothing the gold can see

The store is Princeton WordNet 3.0. Its maintained successor's 2025 edition
renumbers every synset, so the two were joined through sense keys: of 71,864
noun synsets, 67,511 define in the same words, 2,803 are new and have nothing
in the store to hang on, and 1,550 changed in substance. Those 1,550 were
re-read into definitions memory, 1,117 facts becoming 1,418. Both
configurations are identical to §28's to the last pair: no re-read synset is
asked about. The re-reads stay as the more current text, not as a gain.

### Wikipedia leads: a little more, at no measured cost

485 lead paragraphs, read one sentence at a time and only where the subject is
the kind itself, unquantified, unhedged and in the present tense, gave 832
facts, about six in ten good by hand. The gold reached four of them and all
four were right; three more pairs were decided and accuracy rose with them.
Every over-affirmation measure is unchanged, and the guards are 19 right and 3
wrong, as §28's store was.

That is also the limit of what this audit says about it. Four facts in 832 are
asked about, so a four-in-ten junk rate among the rest is invisible here. The
measure shows the facts the gold reaches do no harm; it cannot show the rest
are true.

---

## What to do with this

Ranked by evidence, not by appeal:

1. ~~**Stop R27 excluding on adjectival predicates.**~~ Done, §8.
2. ~~**Decide what to do about `denied_xcslb` feeding `Profiles.denied`**~~
   Done, §9 — it was cut out. §16 now makes the other half available:
   `comps_screened.jsonl` **is** a denial source, 16,910 claims a calibrated
   judge denied at 0.95, and wiring the screened half back into
   `Profiles.denied` would give R17 something real to answer DENIED from for
   the first time. Not done here. It changes what the shipped system says,
   the residual contamination is about 6%, and it would need the page
   examples re-checked after — a separate pass, on its own evidence.
3. **Do not tune the loop for accuracy.** §3 says there is nothing there to
   win. If v688's claim is explanation, the page should say that and this file
   should be cited for why.
4. **Sense selection is now the largest measured defect.** Three of the five
   surviving false denials, and `pinned` shows they go away when the reader's
   sense is supplied. This is the backlog's open item, and it is the one with
   a number behind it.
   **Superseded as the top item by §17**: R19's evidence base is worse. Sense
   selection costs three false denials in a sample; free listing gives R19
   the wrong verdict on more than half of what it is asked.
5. **Re-read the coverage number before any new rule.** 19.2%, unmoved by
   pinning, is the ceiling every rule is working under.
6. **Report over-affirmation from the denials column, not the corrupted
   set** (§16). 6.7% against 1.0% for the same store: the synthetic claims
   were six times too easy, and they have no notion of how near the miss
   was. Keep `--corrupt` running — it is the only model-free negative here
   and it is the arbiter for anything about teaching — but quote it as an
   ordering, not a level.
7. ~~**Re-measure teaching against screened gold.**~~ Done, §16. The −6.4
   accuracy was the benchmark; on screened gold teaching costs **+0.1**, and
   the model-free corrupted column agrees it added no over-affirmation.
   **`TEACHING_FLOOR` is now the live lever**: 0.99 threw away 85% of what
   the model knew, §14 says 0.95 keeps 79% at 3.3% false assertion, and the
   thing that made the trade look bad no longer exists. The argument against
   scaling teaching is now only §15's — no transfer, ~36 GPU-hours for the
   remaining 45,219 concepts — which is a cost argument, not a safety one.
8. ~~**Spend the hours on norms rather than facts**~~ Done, §18. Built in 15
   GPU-minutes rather than two hours, because recording what R19 is asked
   showed no new concepts were needed. It flips 34.3% of R19's verdicts and
   buys **+0.2 coverage and +0.1 accuracy**. Shipped at floor 0.99 because it
   costs nothing, not because it achieved much.
9. ~~**Fact quality is the next thing**~~ Attacked four ways in §19. A
   better dataset makes it worse (GenericsKB triples the model-free
   over-affirmation). Pruning class-node facts changed nothing, because the
   facts that get *read* are not the ones with the widest reach. Making R19 a
   precondition reads well on the benchmark and turns every gap in norm
   coverage into a refusal. What worked was one constant: the corroboration
   floor, a third to a half. **29 Ascent++ facts account for every wrong
   inherited answer in a 1,468-question sample** — the remaining problem is
   small and specific, and `record_inherited.py` is how to find it.
10. **Find the facts that were read before fixing the ones that were
   written** (§19). All three failed attacks aimed at where the volume was.
11. **Fact quality is still upstream of everything** (§18). R19's evidence was broken and is now fixed, and the
   store barely moved, because R19 spends its time adjudicating claims like
   `an animal can be riddled with bullet`. Every lever this file has pulled —
   rules, corroboration, evidence density, teaching — sits downstream of what
   the crawl put in the store. The 19% coverage ceiling and the noise are the
   same problem seen from two sides.
12. ~~**Make R19 corroborate at the sharpest level**~~ Done, §23. It let
   the floor go to 0.8: +1.8 accuracy and 29% less over-affirmation against
   this morning, no example changed. Altitude still bites where no level
   between the concept and the fact has eight covered kinds.
13. ~~**Answer the typicality-versus-capability question**~~ Done, §24, and
   the answer is that R19 wants typicality. The capability prompt is twice
   the judge on capability claims and a worse one for R19, whose question is
   whether a fact is a claim about the class. It is the
   one thing standing between the audit and its largest measured gain: +2.0
   accuracy and 27% less over-affirmation, sitting behind
   `V687_DENSE_WITNESSES` because R19 tests typicality and `can a dog fall
   into a hole` asks capability.
14. **Validate a mechanism on the distribution it will face.** §17 measured
   R19 over AwA2's curated typicality attributes and predicted a large win;
   R19's real workload is crawled free text and the win was 0.2 points. The
   limitation §17 declared — one domain — was not the one that mattered.

## Reproducing

The screen, which needs the GPU once and then never again:

```
python -m research.v688.screen --questions              # the claim sets, no model
python -m research.v688.screen --calibrate              # §16's error rates,  ~4 min
python -m research.v688.screen --survey                 # what the foils are, ~31 min
python -m research.v688.screen --build                  # comps_screened.jsonl, free from cache
python -m research.v688.audit --gold screened --limit 600 --shards 5 --workers 4
```

§17's norm distillation, which needs the GPU for about three minutes:

```
python -m research.v688.norms --questions                # the 50 x 83 grid, no model
python -m research.v688.norms --distil --validate        # against the closed matrix
python -m research.v688.norms --corroborate --narrow 400 # R19 three ways
```

Every judgement caches to `llm/adjudications.json` by claim, so the survey is
paid once: 36,701 foils at 19.7/s, and a rerun is a file read. The cache is
flushed every 2,000 fresh judgements rather than held to the end, so an
interruption at minute thirty costs two minutes.

Numbers above are `--limit 1200` (`--loop-limit 500`), sharded 5 × 4 engines;
`crawl`/`corroborated`/`loop` at `--limit 500` for the matched §3 comparison.
Sampling is a strided walk over a sorted list, not a random draw, so a given
`--limit` gives the same items on every machine. Output lands in
`audit-out/` (git-ignored): `audit.txt`, `audit.json`, and per-config JSONL
with one row per question.
