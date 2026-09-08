# V688 — the loop around v687

v687 answers questions. v688 notices when its own answer does not survive
contact with the rest of the store.

```
python -m research.v688 --workers 19 --port 8688
```

Then open http://127.0.0.1:8688 and pick an example.

`--workers` defaults to 19. Each engine costs about **264 MB and 3.4s** to
build after the first (which costs 916 MB and 6.1s, because it loads spaCy),
so nineteen is roughly **5.7 GB and 25s** with the parallel build. Use
`--workers 6` on a smaller machine; the page reports what it actually got.

## The example this exists for

Ask v687 `does a beagle swim` and it says **VERIFIED**. Behind that yes:

```
concept: dog.n.01   relation: capable_of   object: swim
source: ascentpp    confidence: 0.4168
distance: 3         sense_assumed: True
```

One crawled fact at confidence 0.42, inherited from three levels up, on a
sense that was assumed rather than confirmed. The verdict shows none of it.

Now ask the same store a question nobody asked — the trie's other children
under `dog`:

```
dog        DENIED  swims   "all 3 of the 3 kinds of dog the norms cover deny it"
collie     DENIED  swims   "scored and denied, not merely absent"
chihuahua  DENIED  swims   "scored and denied, not merely absent"
beagle     UNRECORDED      "beagle is not one of the concepts the norms cover"
```

**v687 is not wrong at any point here.** Every answer is correct about its own
evidence. The contradiction is invisible only because one question gets one
route, and nothing in v687 ever asks two ways at once.

v688 does, and reports: *weakly yes, and I don't believe it.*

## What is here

| file | what it does |
| --- | --- |
| `gap.py` | types a v687 payload's incompleteness — a **Gap** (memory says it lacks something) or a **Doubt** (the answer is thinner than its verdict looks). Reads the payload v687 already returns; **edits nothing in v687**. |
| `attention.py` | salience (activation, decaying per cycle) · gain (bits of the candidate set a question removes, on the predicate trie) · urgency (what kind of hole). Ranked as a **product**, so a zero anywhere is fatal. |
| `question.py` | the five generators, and the queue they feed. |
| `buffer.py` | one utterance's state: cycle, activation, topics, open gaps and doubts, carried questions, and the conflicts the answers add up to. |
| `pool.py` | N engines, asked in parallel, with per-question worker and timing. |
| `loop.py` | attend → generate → fan out → read → update → settle. Emits a replayable `Run`. |
| `server.py` + `app.html` | the page: step or play cycle by cycle, click any question for its v687 derivation. |
| `test_v688.py` | 49 tests, including every page example against the claim its card makes. |

## Where a question comes from

The executive never invents one. Five sources, each a mechanical transform of
something already on the table. Two of them fan out and three run in a line.

**Breadth — what the pool is for:**

- **gap** — the last answer named its own blocker. `UNKNOWN_WORD` naming
  *which* word is what makes the follow-up derivable rather than guessed.
- **doubt** — the claim is put to the concept it was inherited from and that
  concept's other kinds, all in one cycle. This is R19 corroboration run
  *across* queries instead of inside one, which the subsystem audit lists as
  present but unrun.
- **curiosity** — the trie has a child here that would split the field.

**Depth — what the pool cannot help with:**

- **chain** — the next question's *terms* are inside the last answer, so it
  cannot be formed until that answer comes back. Two kinds are reliable
  enough to follow: a **definition ladder** (R26 hands back the genus, so
  `beagle → hound → hunting dog → dog → canine` is four questions nothing
  could have predicted from the utterance) and an **inheritance grounding**
  (`does a robin fly` rests on a robin being a bird, which is a claim of its
  own and nobody checked it).
- **split** — a family check that *disagreed with itself*. `1 of 6 deny it`
  is a count, not an answer; the question it raises is which side the subject
  is on, and R21 answers that. It cannot be named until the fan-out returns.

That last pair is the honest limit of nineteen workers. A chain uses **one**
worker and as many cycles as it has steps, however many engines are idle. The
page shows the longest one as a thread, and the `depth` figure in the summary
is how many answers had to come back before the last question was askable.

**Attention is the bound.** Curiosity over a trie with no attention is a
breadth-first crawl of 11,707 nodes — every question defensible on its own,
none of them about anything. Three rules keep the frontier small and moving,
and each one was written after watching the loop wander:

- **Only the subject is a topic.** Everything else a sentence names is
  activated, but is not something to be curious about: the object of a
  question is not what the question is about. Without this, `does a snake
  have legs` asked `is a leg furry` and `what eats meat` asked
  `can a meat walk`.
- **A concept becomes a topic by being looked up on purpose.** `fish` is the
  target of `a whale is a fish`, not its subject; it earns curiosity once the
  loop has chosen to ask `what is a fish` to close a gap.
- **Attention withdraws from a topic that yields nothing.** `meat` really is
  one of the corpus concepts, so the questions are legitimate — and all six
  come back UNKNOWN. The loop asks once and stops. This is the only thing it
  learns within an utterance.

**Follow-ups are only built on deliberate answers.** A corroboration question
is a means, not a topic, and a curiosity question is a guess: chaining off
either walks away from the utterance. A run about whales went
`does a goldfish have a gill` → `is a goldfish a bony fish` — true, and about
nothing — and `is a shark a fish` spent 37 questions on gills and slime. The
exception is curiosity about the subject you actually named, which is how
`does a cat purr` found that the cat family disagrees about being active.

Salience-with-decay is also, quietly, the **activation dynamics** the
subsystem audit lists as the deepest missing piece. It arrives as a side
effect of having to bound curiosity, not as a feature anyone set out to build.

## Why a cycle is not a conversational turn

A cycle is an internal round the system runs on its own to resolve **one**
question or statement. An utterance takes three to five of them. That scoping
is the whole reason this is buildable:

- **Episodic memory** is this buffer log surviving past the utterance. Scoped
  to one utterance there is nothing to remember, so it is not needed and not
  built. Later it is "don't clear it, index by utterance".
- **Executive control** is `sorted(queue, key=rank)` in `question.py` —
  thirty lines, not a subsystem, because v687 already routes deterministically
  and there is no production conflict to resolve.

Dialogue is what you get by not clearing the buffer between utterances. It is
what *would* need those two subsystems, which is the strongest argument for
being utterance-bound first.

## The trie does two jobs, and they pull apart

`identifiability.py` in v687 already measures identification depth over the
predicate trie and calls it twenty questions:

> Storing wants shared prefixes: put the predicate the most individuals carry
> first and everyone walks the same corridor before splitting. Identifying
> wants the opposite.

v688 takes that seriously as an architecture claim. The **same structure that
compresses the ontology schedules the questions**: storage keeps
`adaptive_coverage`, and interrogation ranks by expected bits removed, which
is maximised by a predicate splitting the field in half — not by the rarest
one, which is `anti_coverage`'s greedy approximation.

## Containment

`research/v688/` imports `research.v687` and **never edits it**. The semantic
system stays frozen behind `ReasoningEngine.ask()`; v687's 317 tests are
untouched. Every typed gap is recovered from the payload v687 already returns,
which is why `gap.py` exists at all rather than a field being added upstream.

## What is deliberately not here

- **No instance layer.** The store holds kinds. "I am Adrian" still needs an
  instance layer and an assertion channel, and neither is a rule.
- **No learning.** `PredicateTrie.ensure()` reports whether it allocated and
  nothing reads that yet.
- **No persistence.** Runs are cached in memory for the life of the server.
- **Parallelism buys breadth, not speed.** One `ask` is about 20ms. Nineteen
  workers make a fan-out land in one cycle instead of five; they do not make
  anything faster that a person would notice.

## Known rough edges

- **Curiosity phrasing is corpus-specific.** The trie holds two vocabularies:
  XCSLB phrases that carry their own verb (`can be played`) and AwA2 attribute
  columns that carry nothing (`furry`, `ground`, `meat`). Habitat and diet
  columns get locative and eating frames; a short list of AwA2 coinages
  (`oldworld`, `quadrapedal`) is skipped as unaskable. This is a corpus
  problem wearing a phrasing costume.
- **Sense choice is still v687's**, so `wing` reads as an aircraft wing and
  `mouse` as the device. The buffer holds pins but nothing sets them yet;
  widening `pins.py` from request-scoped to utterance-scoped is done, using
  it is not.
- **A conflict is reported, never resolved.** The loop says the family
  disagrees and asks which side the subject is on; it does not then decide.
  Deciding needs belief revision, and that needs somewhere to write the
  answer down.
- **Explanations are not followed.** R23's objects were a third chain source
  and are dropped: `why does a dog bark` resolves `bark` to a sense whose
  recorded causes are nausea and vomiting, and the ladder ran
  `nausea → symptom → evidence → information` — four correct definitions,
  none about dogs. Following a crawled explanation is only as good as the
  sense it was crawled under, and nothing here chooses that sense yet.
- **One conflict on this page turned out to be a phrasing bug.** `is a shark
  a fish` reported a family split on scales because the generated question
  read `is a seahorse scales`. With the grammar fixed every fish agrees and
  the conflict is gone — a reminder that a disagreement between generated
  questions is evidence about the generator first.
