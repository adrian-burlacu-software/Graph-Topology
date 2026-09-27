# v695: common-sense actions — how a goal is reached, and who can

*How would I get over a fence?* was refused: v694's designer takes goals a
tool or a place reaches, and nothing read a goal that is the asker's own
movement. The knowledge was there — ConceptNet says climbing is done to get
over a fence, and that goats, deer and horses jump fences — but kept as
noun senses (*ascent* motivated by the goal), where no reader looked.

## 1. The mined layer (`mined.py`)

ConceptNet 5.7's English edges between doings, kept as the phrases they
were said in, with each end normalised by WordNet's morphology:
MotivatedByGoal, HasPrerequisite, HasFirst/Last/Subevent, CapableOf,
NotCapableOf, UsedFor — **126,454 edges** in `data/v695_actions.sqlite`,
built by `python -m regenerate --only actions` in about a minute.

## 2. Who can (`can.py`) — any subject

`can(kind, verb)` asks of what the subject *is*, so the one talking (a
person), the agent (a computer program), a dog or John the farmer are asked
the same way:

1. **VerbNet's restrictions on the doer.** Read from the XML with subclass
   inheritance (the loader in v689 does not keep them): run-51.3.2's Theme
   is `+animate | +machine`, eat-39.1's Agent `+animate`; mapped to WordNet
   (animate = animal or person, concrete = physical entity, machine,
   vehicle, organisation). No class of the verb's common senses admits the
   kind → **no**. A program cannot jump, eat or see; neither can a rock.
2. **Unaided evidence, near the kind.** The store's `capable_of` and
   ConceptNet's `CapableOf` rows about the kind and the one above it — not
   further: *a carnivore can climb trees* is true of cats and not of dogs.
   A row counts only in the shape of doing it oneself: `jump`, `jump over
   puddle`, `climb tree`, `swim in water` — not `fly helicopter`, `fly with
   machines` or `fall in love`.
3. Otherwise **unattested**, which is not no.

v694's designer now asks it about its own doer (`knowing.person_can(verb,
patient, subject)`), so what is unaided depends on who does it. Its banks
did not move: BANK 39/44, HELD2 22/31.

## 3. Reaching a goal (`achieving.py`)

A goal is a doing with a place: verb, place words, thing (*get · over ·
fence*), and a subject. Actions come from

- **done for it** — MotivatedByGoal (*climb* → *get over fence*);
- **seen doing it** — rows where something animate does a motion verb to
  the thing in the same place (*goat, horse, kangaroo jump a fence*); a
  bare object counts only when VerbNet's role for it matches the place
  (Location/Trajectory for *over*, Initial_Location for *out of*,
  Destination for *into*) and not its opposite (*leave the car* is not
  getting into it); one witness counts only if it is the subject's kind;
- **the language** — a motion verb whose gloss puts the place right before
  a kind of the thing (*vault: leap over (an obstacle)*).

Each is kept only if `can(subject, verb)`. Tried and dropped: WordNet's
phrasal verbs (*climb down*, *file out*) — mostly not ways to the place.

## 4. On the page (`page.py`)

    how would I get over a fence    You could jump the fence or climb over the fence -- ...
    how would a hen get over a fence  A hen could fly over the fence -- ...
    could you jump over a fence     No, I can't jump over a fence -- I am a computer program, ...
    can i jump                      Yes, you can jump -- a person can jump (ConceptNet).

## 5. Measured

**PIQA** (`piqa.py`; Bisk et al. 2020): a goal and two solutions. The graph
chooses by the words only one solution has — how the store and the mined
edges tie them to the goal (`store`, `actions`), how well the designer's
`serving` says they serve the goal's doing (`design`), and verbs no person
can do (`can`). Train was looked at while building; **dev run once**:

| sources | answered | right when answered | overall (abstain = ½) |
|---|---|---|---|
| store | 83.9% | 51.4% | 51.2% |
| actions | 63.2% | 51.7% | 51.1% |
| design | 37.5% | 51.5% | 50.6% |
| store+actions | 85.8% | 52.3% | 52.0% |
| all | 86.0% | 52.3% | **52.0%** |

Chance is 50%. On train, the most confident quarter of decisions is right
55–57%: the signal is real and weak. PIQA's wrong solutions were filtered
to be as related to the goal as the right ones, so relatedness — which is
what these sources measure — cannot choose.

### 5b. Matching requirements (`requiring.py`)

The mechanism that should choose where relatedness cannot: what the goal
requires of its means, and whether each solution's own words have it —
the properties that the things serving the goal's doing *agree* on (at
least two of them; people excluded — they attach things too), what the
goal's things are made of, and WordNet antonyms against (hot water for a
sunburn that wants cool). Evaluations said of more than 1% of concepts
(`great`, `safe`) are not properties; every fit is bounded. Rarity as the
weight was tried first and rewarded `hermaphroditic` and `accusative`.

Asked alone it chooses half the items; summed with the others it is
drowned, so the sources are also asked in turn (`--cascade`):
requirements, then serving, then relatedness. **Dev, run once:**

| | answered | right when answered | overall |
|---|---|---|---|
| require | 48.2% | 51.3% | 50.6% |
| all, summed | 86.5% | 52.2% | 51.9% |
| cascade | 86.6% | 52.3% | **52.0%** |

On the 500 train items it was built on, `require` was right 58.4% and the
cascade 55.0% — **none of it held on dev**. The weighting was fixed while
looking at the first 40 of those items, and what it gained there was
theirs. The mechanism is not what fails: the store's properties are.
What serves a doing is found (hammers crush), but what those things are
*like* is mostly evaluation and noise — the means of *attach* agree on
`elastic` and `sore`, of *boil* on `cloudy` and `alkaline`. Rated physical
norms (NEWTON, THINGSplus: hardness, sharpness, weight) cover some 1,800
household objects and would be the honest next source for requirements;
the store's free-listed properties are not enough.

**Reaching a goal** (`held_out.py`): PIQA has almost none of these (77 of
3,000 train goals parse as a doing with a place, most not movements), so
ConceptNet itself was used — hide each MotivatedByGoal edge whose goal is a
movement past a physical thing, and see if the other sources find it. Only
**41** such goals exist; 2 of 24 proposed come back in the top three. Too
small to be a benchmark, and the finding is the size: ConceptNet has the
fence by luck, not breadth. A generative benchmark needs goal–step data at
scale — proScript or wikiHow — which is the next step if wanted.
