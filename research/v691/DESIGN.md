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

## 8. What is next

- **something that learns from a `Gap`**, per §6.
- **plan quality**, per §2 — the first thing it needs and does not have is a
  notion of cost.
- **the chunk key**, per §3.
- **a reader that is not the domains' own templates.** `scene.py` reads
  `say` and `reads` lines, which is enough to demonstrate planning and is
  not language. v689's grammar is where language lives; joining them is the
  work ToMi and StepGame showed is not free.
- **v691c — a real text environment** (ScienceWorld over ALFWorld: it is
  science, so the graph's knowledge is relevant), where §3's irreversibility
  stops being theoretical.

Nothing here changes v687–v690 except one pure read in `executive.py` and
the registration hook in `session.py`. v691 is a layer: v687 knowledge, v688
asking, v689 conversation and events, v690 generation, v691 world, actions,
execution and monitoring.
