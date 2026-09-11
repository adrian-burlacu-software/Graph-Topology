# V689 — who is who

v688 answers questions about kinds: *does a beagle swim*. A conversation is
about individuals — *the* beagle, *that* one, *the second* one, *it*, *me* —
and nothing in the words names which. v689 keeps an **episodic memory** of
them, reasons over it with **v687's own rules**, and stores it in **the paper's
trie**, grown live as the conversation goes.

```
python -m research.v689 --workers 19 --port 8689 --teacher
```

Then open http://127.0.0.1:8689. One process serves both layers: the
conversation at `/` and v688's page, unchanged, at `/v688`. Two processes
cannot hold the store at once, so v689 runs *as* v688 rather than beside it.
What conversations were told and taught is kept in
`state/v689-memory.sqlite`; `--memory PATH` moves it and `--no-memory` keeps
nothing.

## Semantic memory and episodic memory, one set of rules

v687's store is semantic memory: what is true of kinds. An individual is one
more node at the bottom of that taxonomy — the pig you mentioned sits under
`hog.n.03` — and what you told me about it is one more fact of the same shape,
with `told` as its source. `EpisodicReasoner` is v687's `Reasoner` with its two
lookups extended, so every rule runs on an individual unchanged:

| you said | stored | answered by |
| --- | --- | --- |
| `it can't swim` | `not_capable_of swim` | **R3** — a negation at this level blocks what beagles do |
| `he was flying` | `capable_of fly` | **R4** at distance 0 — doing a thing shows it can |
| `it has no tail` | `has_a "no tail"` | **R3**'s denial-in-the-object, as the crawl writes them |
| *(nothing)* | — | **R1** up to the kind, **R2**, **R4**, **R5**; then v688 on the kind |
| `is it a dog` | — | **R1** walking up from the individual |

One rule is added, and it is R2's question asked one level lower:

**E1 — a quality does not descend from a kind to an individual.** `is a beagle
black` is recorded; `is the second beagle black` is about one dog. So a
quality is answered from what was said about that individual, or not at all,
with the kind's tendency shown beside it.

One relation is added, and no rule reads it: **`did_not`**. `it wasn't flying`
says nothing about whether it can, so it is not `not_capable_of` — stored as
that, R3 would deny a grounded pig the ability to fly. It answers `does it
fly`, and nothing else.

## The trie, live

Appendix 3 stores individuals as goal nodes under ordered predicate paths. The
episodic trie is that structure over the conversation: an individual's
predicates are its kind and **every kind above it**, what it was told, what it
is called and whose it is. Every change re-plans it with `adaptive_coverage`
and reports what it allocated — *"the topographical growth of tries is driven
by allocation"*, measured per turn instead of per corpus. A second beagle
allocates nothing: it shares every predicate with the first until one of them
is told something.

Resolving a description is **identification, the trie read downwards**, as
`identify.py` reads it: walk down until the description is exhausted, and
everyone stored beneath fits. Because a beagle is stored with `is_a dog`,
`the dog` finds it by R1's closure rather than by a special case.

| phrase | resolved by |
| --- | --- |
| `a beagle`, `another beagle` | placing a new individual under `beagle.n.01` |
| `the beagle`, `the dog`, `the black one`, `my beagle` | identification on `is_a …`, `has_property …`, `owner you` — then salience, or *which one?* |
| `rex` | identification on `name rex`; two Rexes are a question, because a name is told, not an identity |
| `it`, `that one` | the most salient individual |
| `the second one`, `the other one` | order of introduction; the one not in focus |
| `i`, `me`, `my` | you — a person, kept apart from `it` |

**Salience** is v688's `Activation` over individuals rather than concepts,
decaying every turn, with one change: a mention *refreshes* rather than adds,
so `it` follows the conversation, not a tally.

## Teaching: taxonomy and norms, episodic only

Episodic memory is an overlay on the whole store, not only a place for
individuals. A conversation can teach kinds, taxonomy and norms, and all of
it is held beside the store and never written into it:

| you say | held as | and then |
| --- | --- | --- |
| `a wemble is a kind of animal` | a node `wemble`, under `animal.n.01` | `can a wemble breathe` is R1 walking from a wemble into what the store knows of animals |
| `wembles can fly` | `capable_of fly` on `wemble` | every wemble inherits it (R4) |
| `beagles can't swim` | `not_capable_of swim` on `beagle.n.01` itself | R3 finds it one level up from every beagle, before dog's row |
| `dogs are animals` | nothing new | R1 already walks there |

`EpisodicReasoner.parents_of` and `facts_of` read both memories, so a taught
edge, a taught kind and a taught norm are the same to every rule as the
store's own. A question about a kind is answered from episodic memory when
anything taught bears on the walk, and by v688 otherwise.

## E2: what carried it did it

**E2. An action done while carried belongs to what carries it.** `he was
flying` is `capable_of fly` on the pig, and R4 answers `can the pig fly` yes
from it. `it was in an airplane` is `at_location airplane` -- v687's parser
reads the phrase as a quality, so v689 reads `in`, `on`, `inside` and
`aboard` itself -- and an airplane flies, so the flying was the airplane's:
the pig's `capable_of fly` is withdrawn, kept as `carried fly`, which no rule
reads, and `can the pig fly` is the kind's answer again.

- **Either order.** The rule runs whenever either fact is told.
- **Only a doing.** `it can fly`, said outright, is a claim about the pig and
  is never withdrawn.
- **Only a carrier that does it.** A pig on a cat was still flying. Whether
  the carrier does it is v687's walk first, and v688 only if the store has
  nothing.
- **Recomputed, not applied once.** Told `the airplane couldn't fly`, E2 runs
  again for everyone in or on that airplane, and a doing no carrier does is
  given back.

## Objects: the other individual

`the dog chased it` and `it was in the plane` name two individuals. The one
after the verb is resolved by the same identification and salience as a
subject, with two differences. It is never the subject: `it`, as the object
of `the dog chased it`, cannot be the dog. And it refreshes salience only to
half, without taking the focus, so `it` in the next sentence still means the
dog.

What is stored is still a fact about kinds, because that is what v687's rules
and matcher read -- with **which one** kept beside it:

| said | stored |
| --- | --- |
| `the dog chased it` | `capable_of "chase a cat"`, bound to that cat |
| `it chased a cat` | `capable_of "chase a cat"`, about any cat |
| `it was in an airplane` | a new airplane, and `at_location airplane` bound to it |

So `did the dog chase the second cat` meets the fact, sees it was about the
first cat, and says it was not told rather than yes. An indefinite object is
a kind except after `in`, `on`, `inside` and `aboard`, where what carried it
is one particular thing E2 has to ask about -- and E2 asks *it*: a carrier
that is an individual is walked from itself, so what was told of that
airplane comes before what airplanes do.

## Where an answer comes from

- **The walk decides at the individual** (R3 or R4 at distance 0): what you
  told me, and when the kind says otherwise, that it is an exception.
- **The walk meets something taught** further up: a norm on its kind, a
  taught edge, or a kind the store never had. What you taught answers.
- **The walk stops there by E1**: not known of this one; the kind's tendency
  beside it.
- **The walk passes the individual**: nothing was told, so it is a question
  about the kind, and v688 — corroboration, R19, the teacher — answers it.

The store is never written. Everything told lives in the conversation.

## Long-term memory

A conversation is how this layer learns anything, so it is kept on disk
(`longterm.py`), and it is kept as two memories, because they are two kinds:

| said | belongs to | survives |
| --- | --- | --- |
| `a wemble is a kind of animal`, `beagles can't swim` | every conversation | a restart, `start over` |
| `there is a beagle`, `its name is Rex` | this conversation | a restart |

**Knowledge** -- taught kinds, taxonomy and norms -- is about no one in
particular, so it is no one conversation's. One copy is shared by every
conversation (`Knowledge`, read through `Layered` beside each conversation's
own individuals), rewritten to disk after every turn, and an answer drawn
from it says when it was taught in an earlier conversation. It is still never
written into the store. `unlearn` empties it.

**A conversation** -- individuals, what was told of them, names, bindings,
what E2 withdrew, salience -- is snapshotted after every turn with the turn as
the page showed it. A page that comes back after a restart finds its
conversation where it left it. `start over` forgets the conversation and
keeps the knowledge.

**Examples teach nothing that is kept.** They teach on purpose -- `beagles
can't swim` is there to show R3 -- and a demonstration that wrote a false norm
into what every later conversation starts from would be a bug with a
permanent address. An example runs in a conversation of its own, with
knowledge of its own, and leaves yours alone.

## What it does not do

- **One object at most, and it ends the sentence.** `the dog chased the cat
  in the garden` binds the garden and leaves the cat a kind.
- **No plural references** (`the beagles`) -- plurals teach a kind
  (`beagles can't swim`) but never pick out individuals -- and names of one
  word only for mentioning.
- **A lower-case `i am adrian` is not a name** — capitalisation is the only
  evidence; `my name is adrian` works in any case.
- **Only being carried explains a doing away (E2).** Nothing else about the
  situation is reasoned over: `it was in a storm` withdraws nothing.
- **Negated taxonomy is not stored** (`a whale is not a fish`): v687 keeps no such relation.
- **An unknown kind must be one word**, and a bare unknown singular (`Adrian can swim`) is read as someone, not a kind.
- **The v688 loop does not run over individuals.** It answers the kind; the
  individual is v687's rules over episodic memory.
- **Knowledge is one for the whole server.** It records which conversation
  taught each thing, not who; there is no notion of two people disagreeing.

## Files

| file | what it does |
| --- | --- |
| `reading.py` | which words pick out an individual, and what the utterance does |
| `episodic.py` | episodic memory, `EpisodicReasoner` and E1, the live trie and identification |
| `discourse.py` | attention: salience, order, focus, and resolving a phrase to one individual |
| `session.py` | one conversation: told facts into memory, questions to v687's walk, the kind to v688 |
| `asker.py` | what a session needs from v687 |
| `longterm.py` | the knowledge every conversation shares, and every conversation, kept on disk |
| `server.py` + `app.html` | the page, over v688's `Service` |
| `test_v689.py` | v687's real reasoner and parser over a nine-concept store built in the test |
