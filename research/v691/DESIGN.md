# v691: an agent that acts

v690 §8c closed the executive as a **reasoner**: E1–E6 built, ProofWriter
answered to depth 5, EntailmentBank's remaining gap shown to be about
meaning rather than about control. The conclusion there was that the
architecture was finished and the capability was idle — there was nothing
left for it to do that it was not already doing.

v691 is what it was idle *for*. An agent has to do three things the
executive had never been asked to do:

| | what was missing | where it is |
|---|---|---|
| a world | facts that change only by acting, and are not undone by giving up | `world.World`, `acting.Situation` |
| execution | one action at a time against the world, not a plan nobody runs | `acting.agent` |
| surprise | noticing that what happened is not what was expected | `acting.agent`'s `look` |

**v691a is a blocks world with no reader anywhere in it.** That is
deliberate, and it is the ProofWriter move: instrument before task. ToMi and
StepGame died at the parser, so a domain where
there is nothing to parse makes any failure the architecture's. It also
makes the suite run in a tenth of a second, which is why there are
twenty-four measured problems and three held-out seeds instead of an
anecdote.

## 1. What was built, and what it comes to

`world.py` is the domain: ground `Action`s with preconditions, adds and
**deletes**; a `World` whose only mutator announces itself with `effect`; a
fixed suite of ten problems, a sampler, and a breadth-first oracle that says
what the shortest plan was.

`acting.py` is the agent. The planner is **the executive itself** — nothing
in `executive.py` was changed to make this work:

```
an Action's preconditions   ->  an Operator's `needs`
an Action's adds            ->  an Operator's `gives`
_means_ends at an impasse   ->  goal-stack planning
```

Means-ends taking the most useful waiting operator and pushing a subgoal for
the slots it lacks **is** what a STRIPS planner of the period did. The whole
translation is `acting.operator_of`, which is fifteen lines.

Measured on `world.SUITE` plus 20 sampled four-block problems (24 after the
degenerate ones are dropped), with a breadth-first oracle for the optimum:

```
solved 21/24, of which 18 in the fewest actions;
118 actions, 187 operators fired, 799 subgoals, deepest goal stack 9
```

and held out on 97 sampled problems over three unseen seeds, **78 solved,
68 of them optimal**. The Sussman anomaly is solved in six, which is
optimal. There is no backtracking anywhere: every problem is one forward
pass, and the three that fail, fail because the first choice was the only
one.

## 2. The three things a world broke, and what each cost

### `gives` only ever adds

Nothing in the executive ever removes a slot, because knowing something does
not stop you knowing something else. **Acting does.** `Executive.plan` is
honest about what it is — *what could be, not what will* — and it is kept,
unchanged, as the baseline `by_regression`, because the size of the gap is
worth a number rather than an argument:

```
22 four-block problems: a plan for 22, executable to the end for 2,
49 actions applied in total before one did not apply
```

It also does not terminate in reasonable time past four blocks, and that is
the same fact from the other side: with nothing ever becoming false, every
action always *could* be added, so the regression has nothing to prune on.

### `Working` un-does a failed subgoal, and a world does not

`Working` scopes a subgoal so that one which fails leaves nothing behind.
That is exactly right for belief and exactly wrong for a world: a subgoal
that unstacked a block and then gave up has still unstacked it.
`acting.Situation` is `Working` with that one difference — the facts are one
set the whole goal stack shares, and `retract` reaches into every frame
beneath. Everything else about `Working` is untouched and still does its
job.

**This is the irreversibility problem at its smallest.** Here it is honest
because blocks can be put back, so `effect(..., undo)` can really undo. The
moment the world contains anything that cannot — pouring, sending,
deleting — planning against a model and executing against the world stop
being the same activity, and only the split this file already makes
(`think` searches a `Situation`; `agent` acts on a `World`) survives it.
That is the v691c risk, and it is architectural, not incidental.

### chunking is keyed on the wrong thing

E6's chunk is keyed on (executive, slots wanted, slots in hand). In a world,
"slots in hand" is the state, and the state is different after every action,
so the key never comes round: **38 chunks learned, 1 hit, 784 misses**
across the suite. Every plan is identical with and without, 21 solved and 18
optimal either way, and the one hit saves 14 subgoals out of 799. Not a bug
— the design says a chunk can only save search, never change an answer, and
it holds exactly. But a chunk that fits a world would have to be keyed
on *what the impasse turned on* rather than on everything that was true at
the time, and that is a real piece of work that v691a has now motivated with
a number instead of a hunch.

## 3. Goal protection: one read added to the executive

The failure that cost the most was textbook clobbering. A subgoal for
`clear c, held b` gets `held b` by unstacking b, then goes after `clear c`,
and on the way `drop b` looks like progress and throws away the block it is
holding.

The executive already knew which slots its open subgoals were achieving —
`_PURSUING` — and only could not say so. `executive.pursuing()` is a pure
read, changes no behaviour, and lets an operator decline to undo what a goal
beneath it has got. It is worth **eighteen problems of the held-out 97**
(60 → 78).

The rule is asymmetric on purpose:

> Never throw away the means. The ends may be undone and redone.

Protecting *everything* pursued, goal facts included, is what makes the
Sussman anomaly unsolvable — the only way to `on b c` there is to take
`on a b` apart again. So a fact the goal itself asked for is not protected;
a fact the search built as a step is.

## 4. What the utilities turned out to be

Four obvious local signals were written first — prefer an action that
achieves a goal fact, avoid one that undoes a goal fact, prefer one that
frees a block the goal buries, prefer putting a block on the table. An
exhaustive search over all sixteen subsets found the **goal-tower ordering**
(build from the bottom) on its own as good as any of them, and held out,
`frees a goal block` beside it was better again. On the held-out 97:

| | solved | shortest | subgoals |
|---|---|---|---|
| both, as shipped | 78 | 68 | 3052 |
| the tower ordering alone | 72 | 65 | 2833 |
| all five signals | 67 | 57 | 3686 |
| none (the domain's generation order) | 58 | 49 | 4579 |
| as shipped, without goal protection | 60 | 51 | 2731 |

Two things in that table matter more than the winner.

**`frees a goal block` was measured out and then back in.** With the other
three it cost points; with the tower ordering alone it is worth six problems
and it is what solves the Sussman anomaly. A subset search on the first
feature set would have thrown away the signal that mattered — the same shape
of mistake as v11's matrix, found the other way round.

**`achieves a goal fact`, the most obvious of the four, is worth nothing at
all.** What decides a blocks problem is not which action helps now but the
structure of the goal: build from the bottom, and clear what the goal has to
sit on. Both surviving signals are about the goal; none of the discarded
ones were.

Magnitudes are irrelevant — 0.25 and 2.0 give identical results — because
means-ends reads an order, not a score. That is worth remembering before any
of this is handed to utility learning.

## 5. A surprise is an impasse (v691b)

When `look` finds the world is not what the action was expected to leave,
it does not quietly plan again. It **names an impasse**, and nothing can be
proposed until the subgoal `make sense of it` has run and handed back a
plan: `noticed` writes down the difference, `plan again` replans from the
world as it is now. That is E2's mechanism doing the job v691 was built
for, and the reason it matters is not tidiness — it is that the gap between
what was predicted and what was observed now has somewhere to live.

`acting.Gap` is that gap: the action, what was expected and not found, what
was found and not expected, and how far in it happened. **It is the first
thing in this project a learner could be given.** Everywhere else the signal
was whether an answer was right, judged from outside; this is a prediction
the agent made itself, falsified by the world, with the action that made it
still in hand. Nothing learns from it yet, and that is the next piece of
work rather than an oversight.

Two small things the mechanism needed, both recorded because they are
traps:

- The executive resolves each impasse **name** once per run, rightly — a
  second unresolved impasse of the same name is the first one still
  standing. Two surprises are not that, so they are numbered, and
  `Surprises` maps every `surprise N` to the one substate.
- `plan it` has to refuse to propose while a surprise is open, or the
  impasse never happens and the substate is decoration.

## 6. Talking to it (`talking.py`)

The point of an agent is that you can give it a problem. `talking.py` is a
conversation over a scene: you say what is on the table, you say what you
want, and it plans, acts, and tells you what it did.

```
> there is a red block on a green block, and a blue block on the table
> put the green block on the blue block and the red block on the green block
  I took the red block off the green block and put it on the table, put the
  green block on the blue block, then put the red block on the green block
> why did you move the red block
  I took the red block off the green block because I needed the green block
  clear before I could pick up the green block
```

That second line is the Sussman anomaly said in English, and the third is
worth more than it looks: **nothing generates an explanation.** The planner's
means-ends subgoals are already named `achieve clear green for take green`,
so `why` reads the goal stack back and puts it into words. An explanation
that had to be invented separately from the search would be a story about
the agent; this one is the search.

Three deliberate limits:

- **The reader is thin on purpose.** Twelve phrasings and a colour list.
  v689's grammar is where the knowledge of language lives, and pointing it
  at a new act is where ToMi and StepGame died; a thin shell over a thick
  agent keeps the demonstration about the planning. Anything it cannot read
  it refuses *by name* (`Heard.trouble`) rather than guessing.
- **The acts are an `Executive`**, the same shape as v689's `say`, because
  choosing what an utterance is for is the same kind of choice as choosing
  what to do about a goal.
- **A change between turns is not a surprise.** Say `actually the red block
  is on the table now` and the agent simply plans from what it finds — it
  was not expecting anything at the time. A surprise needs the world to move
  *during* an action, so `--gremlin` puts something in the world that does
  that, and the reply then says what it expected, what it found, and what it
  did about it.

## 7. What is next

- **something that learns from a `Gap`**, per §5. It is the first signal
  this project has had that is the agent's own prediction rather than an
  external judgement.
- **the chunk key**, per §2. A chunk keyed on the state is useless in a
  world; keyed on what the impasse turned on it might not be.
- **more than one domain.** `world.py` is generic and only `blocks()` is
  not; a second domain would say so rather than assert it.
- **v691c — a real text environment** (ScienceWorld over ALFWorld: it is
  science, so the graph's knowledge is relevant). This is where the reader
  becomes load-bearing again, and §2's irreversibility stops being
  theoretical.

Nothing here touches v687–v690. v691 is a layer: v687 knowledge, v688
asking, v689 conversation and events, v690 generation, v691 world, actions,
execution and monitoring.
