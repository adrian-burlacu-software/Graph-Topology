# v691: an agent that acts

v690 §8c closed the executive as a **reasoner**: E1–E6 built, ProofWriter
answered to depth 5, EntailmentBank's remaining gap shown to be about
meaning rather than control. The conclusion was that the architecture was
finished and the capability idle.

v691 is what it was idle *for*. An agent has to do three things the
executive had never been asked to do — hold a world that changes only by
acting, execute against it rather than return a plan nobody runs, and notice
when what happened is not what it expected — and one thing the first version
of this file did not do: **work anywhere but where it was fitted**.

§1-§7 are that story told twice -- a blocks world, then three worlds
declared as data. **§8 is where it stops being somebody's world at all**:
the actions come from what verbs mean, and what may fill a role from what
the store knows. If you read one section, read that one.

## 1. A domain is a string

The first version of v691 was a blocks world written in Python, and that hid
the question it was supposed to answer. `world.blocks()` built ground
actions with a loop; the planner's utility signals mentioned `stack` and
`clear` by name; the conversation knew what a colour was. Every number was
measured on blocks, so "the executive can plan" was a claim about the one
domain everything had been tuned on.

Now a domain is a **string** (`domains.py`), and four things read it:

```
action stack ?x:block ?y:block
  needs held ?x, clear ?y
  adds  empty, clear ?x, on ?x ?y
  dels  held ?x, clear ?y

say    on     the {0} block is on the {1} block
reads  on     {0} on {1}
do     stack  put the {0} block on the {1} block
```

One text gives the planner its actions, the reader its phrasings and the
narrator its words, because they are three views of the same thing. The
`say` line is compiled **both ways**, so what it understands and what it
says cannot drift apart. Argument types are not declared twice: `in` is a
thing and then a place because the schema that needs it says so
(`Domain.typing`).

Three domains ship — `blocks`, `errands` (fetch things between places) and
`delivery` (parcels, vans, towns). Adding a fourth is writing one of these
texts. Nothing in `world.py`, `acting.py`, `scene.py` or `page.py` names a
predicate, an action or a kind of thing.

## 2. The planner is the executive, unchanged

An `Action`'s preconditions are an `Operator`'s `needs` and its adds are its
`gives`, so `_means_ends` — taking the most useful waiting operator and
pushing a subgoal for the slots it lacks — **is** goal-stack planning, which
is what a STRIPS planner of the period did. The whole translation is
`acting.operator_of`, fifteen lines, and nothing in `executive.py`'s cycle
changed.

| domain | solved | shortest | actions | optimal | subgoals |
|---|---|---|---|---|---|
| blocks (10 fixed + 14 sampled) | 23/24 | 19 | 150 | 134 | 1157 |
| blocks, held out (3 unseen seeds) | 76/97 | 67 | | | |
| errands (30 sampled) | 30/30 | 1 | 398 | 218 | 1865 |
| delivery (30 sampled) | 30/30 | 3 | 465 | 163 | 617 |

The Sussman anomaly is solved in six, which is optimal. There is no
backtracking anywhere: every problem is one forward pass.

**Control transfers; plan quality does not.** Both new domains are solved
outright, and almost none of those plans are the shortest — the agent drives
a van to the wrong town and back before fetching the parcel. Blocks is
nearly all optimal because almost every action there changes something that
matters; `go` and `drive` can be repeated at no cost, and means-ends counts
no cost. A planner that cared about length would need an evaluation
function, which is a different thing from the control this was built to
test, and is not smuggled in here.

## 3. The three things a world broke, and what each cost

### `gives` only ever adds

Nothing in the executive removes a slot, because knowing something does not
stop you knowing something else. **Acting does.** `Executive.plan` is honest
about what it is — *what could be, not what will* — and is kept unchanged as
the baseline `by_regression`, so the gap is a number:

```
22 four-block problems: a plan for 22, executable to the end for 2,
49 actions applied in total before one did not apply
```

Past four blocks it does not return, which is the same fact from the other
side: with nothing ever becoming false, every action always *could* be
added, so the regression has nothing to prune on.

### `Working` un-does a failed subgoal, and a world does not

`Working` scopes a subgoal so that one which fails leaves nothing behind.
That is exactly right for belief and exactly wrong for a world: a subgoal
that unstacked a block and then gave up has still unstacked it.
`acting.Situation` is `Working` with that one difference — the facts are one
set the whole goal stack shares, and `retract` reaches into every frame
beneath.

**This is the irreversibility problem at its smallest.** Here it is honest
because blocks can be put back, so `effect(..., undo)` can really undo. The
moment a world contains anything that cannot — pouring, sending, deleting —
planning against a model and executing against the world stop being the same
activity, and only the split this code already makes (`think` searches a
`Situation`; `agent` acts on a `World`) survives it.

### chunking is keyed on the wrong thing

An E6 chunk is keyed on (executive, slots wanted, slots in hand). In a
world, "slots in hand" is the state, and the state differs after every
action, so the key never comes round: **38 chunks learned, 1 hit, 784
misses**, every plan identical with and without. Not a bug — the design says
a chunk can only save search, never change an answer, and it holds exactly.
But a chunk that fits a world would have to be keyed on *what the impasse
turned on* rather than on everything that was true at the time.

## 4. What the utilities turned out to be, twice

Four obvious **local** signals were written first: prefer an action that
achieves a goal fact, avoid one that undoes a goal fact, prefer one that
frees a block the goal buries, prefer putting a block on the table. An
exhaustive search over all sixteen subsets kept only a fifth that is not
local — build the goal tower from the bottom — and, held out, `frees a goal
block` beside it.

Then all of them were replaced by three that name nothing:

| | the general form of |
|---|---|
| a goal fact that others wait on | build from the bottom |
| undoing what a goal's achiever needs | do not stack onto a block that has to move |
| something an achiever will need | clear the block the goal has to sit on |

All three are read off `needs`/`adds`/`deletes` alone. *Enabling* — one goal
fact enabling another when something achieving the first produces something
the second's achiever needs — reproduces the tower ordering exactly on
blocks, and in `errands` puts being at the shop before having what is kept
there. On the held-out 97 blocks problems:

| | solved | shortest |
|---|---|---|
| as shipped, general | 76 | 67 |
| the blocks-specific version | 78 | 68 |
| without `undoing what a goal's achiever needs` | 70 | 61 |
| without `a goal fact that others wait on` | 73 | 60 |
| without `something an achiever will need` | 76 | 66 |
| none (the domain's own order) | 56 | 47 |
| as shipped, without goal protection | 58 | 51 |

**Generality costs two problems of the 97.** That is the honest price: the
fitted signals knew `clear` was special, and these only know that something
is a precondition. What they buy is that the same numbers can be asked of
`errands` and `delivery` — which the fitted ones could not have been asked
of at all.

Three things in that history are worth more than the winner.
`frees a goal block` was measured *out* and then back *in*: with the other
three it cost points, and beside the tower ordering alone it was worth six
problems and the Sussman anomaly — so **ablate against the final feature
set, not the first one**. `achieves a goal fact`, the most obvious signal of
all, is worth nothing. And magnitudes are irrelevant — nine pairs of
coefficients gave identical results — because means-ends reads a rank, not a
score, which is worth remembering before any of this meets utility learning.

## 5. Goal protection: one read added to the executive

The failure that cost the most was textbook clobbering: a subgoal for
`clear c, held b` gets `held b` by unstacking b, then goes after `clear c`,
and on the way `drop b` looks like progress and throws away the block it is
holding.

The executive already knew which slots its open subgoals were achieving —
`_PURSUING` — and only could not say so. `executive.pursuing()` is a pure
read, changes no behaviour, and lets an operator decline. Worth **eighteen
problems of the held-out 97** (58 → 76). The rule is asymmetric on purpose:

> Never throw away the means. The ends may be undone and redone.

Protecting *everything* pursued, goal facts included, is what makes the
Sussman anomaly unsolvable — the only way to `on b c` there is to take
`on a b` apart again.

## 6. A surprise is an impasse

When `look` finds the world is not what the action was expected to leave, it
does not quietly plan again. It **names an impasse**, and nothing can be
proposed until the subgoal `make sense of it` has run and handed back a
plan. That is E2's mechanism doing the job v691 was built for, and it
matters because the gap between prediction and observation now has somewhere
to live.

`acting.Gap` is that gap: the action, what was expected and not found, what
was found and not expected, and how far in. **It is the first thing in this
project a learner could be given** — everywhere else the signal was whether
an answer was right, judged from outside; this is a prediction the agent
made itself, falsified by the world. Nothing learns from it yet.

Two traps, recorded: the executive resolves each impasse **name** once per
run, rightly, so surprises are numbered and `Surprises` maps every
`surprise N` to the one substate; and `plan it` has to refuse to propose
while a surprise is open, or the impasse never happens and the substate is
decoration.

## 7. Talking to it, and on the page

```
> what worlds do you have
> use the blocks world
> there is a red block on a green block, and a blue block on the table
> put the green block on the blue block and the red block on the green block
  I took the red block off the green block and put it on the table, put the
  green block on the blue block, then put the red block on the green block
> why did you move the red block
  I took the red block off the green block because I could not pick up the
  green block until nothing is on the green block
```

That works at a terminal (`python -m research.v691`) **and in a conversation
on the v690 page**, over the same `Scene` and the same acts (`page.say_to`),
so the two cannot behave differently.

**The agent is an act like any other.** `v689.session` grew a registration
hook — `contributes`, because a later layer importing into an earlier one is
how the cascades this executive replaced became impossible to follow — and
v691's acts join the act executive's conflict set on their own utility. A
turn's trace then shows what they were chosen over, and the runs nested
inside a `want` turn are the planner's, so the goal stack, the means-ends
subgoals and the actions taken are already on the page, because
`Executive.run` records every run of an open episode (E3).

**Nothing is read until a world is opened.** `the dog is on the mat` parses
perfectly well as `on dog mat`, and a layer that took it would break every
question v687 to v690 answer. So the gate is explicit: ask for a world, and
only then is an utterance that reads as facts of *that domain* this layer's.
With a world open, `can a penguin fly` and `what is a whale` are answered
exactly as before — 1179 tests pass and the probe is unchanged at 114/114.

`why` is worth one more line: **nothing generates it.** The planner's
subgoals are already named `achieve clear green for take green`, so the
answer is read off the goal stack. An explanation invented apart from the
search would be a story about the agent; this one is the search.

## 8. A world with nothing declared about it

§1 was a half-step. A domain as data is better than a domain as code, but it
is still a world somebody wrote, and "the executive can plan" was still a
claim about the worlds I had written. `verbs.py` is the other half: **the
actions come from the knowledge already in the repository.**

### VerbNet is already a STRIPS domain

VerbNet 3.3 writes the meaning of every frame as predicates over the phases
of an event, and v689 has read it since T4 for what an occurrence *changed*
(`v689/change.py`). Read forwards instead of backwards, those phases are a
precondition and an effect:

```
put-9.1      path_rel(end(E), Theme, Destination, ch_of_loc)
             -> adds   at ?Theme ?Destination
carry-11.4   path_rel(start(E), Theme, Initial_Location, ch_of_loc) ...
             -> needs  at ?Theme ?Initial_Location, at ?Agent ?Initial_Location
                adds   at ?Theme ?Destination,      at ?Agent ?Destination
murder-42.1  alive(start(E), Patient); !alive(result(E), Patient)
             -> needs  alive ?Patient    deletes  alive ?Patient
```

`start` is what had to hold, `end` and `result` are what holds afterwards, a
`!` is a delete. **4,569 verbs have frames; 2,749 of them yield at least one
operator, 7,796 operators in all.** The rest change nothing a planner can
bring about — they say that something happened, or how.

Two things are read off the data rather than declared:

- **A thing is in one place at a time** is *learned*. One frame that says a
  thing is somewhere at the start and somewhere else at the end is saying
  that the relation is a function of its first argument (`functional`). No
  axiom was written; `carry` was read.
- **Which verb to try first** is counted, not chosen: a verb in many VerbNet
  classes is one English uses for many things (`take`, `go`, `put`), and one
  in a single class is specialised (`ferry`, `barge`). Alphabetical order is
  not a decision and this is (`central`).

### What may fill a role is a question for the graph

An operator that will melt anyone is a joke, so VerbNet's `SELRESTRS` are
read too — `Agent +animate`, `Theme +concrete`, `Destination +location` —
and a thing may fill a role only if the store's taxonomy says so
(`Things.categories` over `senses.Ranges.ancestors`). **So what is possible
is a question about knowledge**, and a fact about a thing changes what can
be done with it.

Two traps, both found by being wrong:

- `<SELRESTRS logic="or">` is a *choice*: `bring`'s Destination is
  `+animate` **or** `+location`, because you can bring a thing to a place or
  to a person. Read as a conjunction it refuses every destination there is,
  and a kitchen stopped being somewhere to go.
- **Any** sense may satisfy a restriction, which is `senses.Ranges`' own
  rule. Reading the first sense only makes a dog an andiron and a shop a
  class in woodwork, because that is what the store's first sense for each
  of them happens to be.

### What it comes to

`errands.py` is twelve everyday situations stated as facts and goals, with
nothing between them and a plan written for them (`python -m
research.v691.errands`):

```
reached 12/12, sensible 8/12
```

**Reached** is whether a plan was found and executed. **Sensible** is
whether a person would have chosen those actions, judged against a list of
verbs each errand will accept. They are different questions and the gap
between them is the result:

```
open a door      yes  yes   open door
break a vase     yes  yes   break vase
kill the fly     yes  yes   kill john fly
two errands      yes   no   leave book shop kitchen; take cup book kitchen shop
fetch a book     yes   no   leave book shop kitchen
```

The control is general and it works. **The choice of verb is sometimes not
the one anybody means** — a book that leaves the shop for the kitchen by
itself satisfies `at book kitchen` and is not what was asked. That is the
same wall as EntailmentBank (`v690/DESIGN.md` §8c): a judgement about
meaning, not about control, reached this time from the other side.

One thing was tried against it and **measured out**: preferring the reading
of a verb whose roles carry restrictions the things satisfy — `carry` as
carry-11.4 says an animate agent moves a concrete thing, `leave` as
become-109.1 says a patient becomes a result, of anything. It took sensible
from 8 to 7, which on twelve hand-judged errands is noise, so it is not
shipped. What would settle it is a reason to prefer one verb over another
for a purpose, and nothing in the store has one.

### Talking to it

`openworld.py` makes this a world you can open like any other, so
`scene.py`, `page.py` and `talking.py` work on it unchanged:

```
> use the open world
> john is in the kitchen and the book is in the shop
  all right: the book is in the shop; the john is in the kitchen
> get the book to the kitchen
  I leave the book to the kitchen
> why did you move the book
  I leave the book to the kitchen because I could not do what you asked
  until the book is in the kitchen
```

Nothing declares what a book, a kitchen or a shop is; nothing lists what may
be wanted — **anything a verb brings about can be asked for**, so `open the
door` is an order because VerbNet has verbs that make things open. Five
patterns read a situation and four read an order, and that is deliberately
not a grammar of English: v689's reader is that, and §9 says why they are
not joined yet.

## 9. What this does not do

Said plainly, because the gap is the interesting part.

- **VerbNet says what changes and not what else must be true.** That you
  must be where a thing is to pick it up, that a hand holds one thing, that
  a door must be unlocked before it opens: these are facts about bodies and
  rooms, true of every verb and therefore written on none of them. Some of
  it is derivable — being in one place is, above — and some is not.
- **Antonymy is missing.** Opening a door does not retract `closed door`,
  because nothing in the action model says they are opposites. WordNet knows
  it; the store as built has no antonym table, so this is a data step and
  not a hard problem.
- **A negative precondition cannot be said.** `needs` is a list of slots
  that must be present, so *the door is not already open* is dropped.
  Opening an open door is a wasted action, not a wrong one.
- **Nothing is inflected.** The narration says *I leave the book to the
  kitchen* because no morphology is available in the repository and a table
  of irregular verbs written here would be exactly the hand-written thing
  this was built to avoid. UD_GUM has lemma-and-form pairs and would settle
  it.
- **The reader is nine patterns.** Enough to state a situation and ask for
  something, which is what it takes to show the planning is general. Joining
  it to v689's grammar is the work ToMi and StepGame showed is not free.

## 10. What is next


- **a reason to prefer one verb over another**, per §8. This is the one
  that matters: reached is 12/12 and sensible is 8/12, and closing that is
  a knowledge problem of exactly the shape EntailmentBank left open.
- **antonyms, and the axioms about bodies**, per §9 — both are data steps
  rather than hard problems, and both make plans less silly.
- **something that learns from a `Gap`**, per §6.
- **plan quality**, per §2 — the first thing it needs and does not have is a
  notion of cost.
- **the chunk key**, per §3.
- **joining the reader to v689's grammar**, per §9.
- **v691c — a real text environment** (ScienceWorld over ALFWorld: it is
  science, so the graph's knowledge is relevant), where §3's irreversibility
  stops being theoretical.

Nothing here changes v687–v690 except one pure read in `executive.py` and
the registration hook in `session.py`. v691 is a layer: v687 knowledge, v688
asking, v689 conversation and events, v690 generation, v691 world, actions,
execution and monitoring.
