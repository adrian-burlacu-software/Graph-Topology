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

## 9. The open world is the default, and it learns

Two things were still wrong with §8. You had to *ask* for the world with
nothing declared about it, which made the general case the special one. And
what it could not do was a fixed list -- the gaps in §10 were gaps for good,
because nothing could tell it otherwise.

### Starting there

A conversation now starts in the open world. `use the blocks world` still
gets a declared one, and that is the special case.

The guard that made this safe before was *no world is open yet*, and that is
gone, so the guard is now about the utterance. **This layer answers a turn
only when what was said is about the world in front of it**, and the test is
`scene_ish`: a fact says where something is, or says a thing is in a state
(asked of WordNet, because `closed` is an adjective and `mammal` is not), or
is about something already being talked about. `does a table have legs`
reads perfectly well as `with legs table` and is v688's question, so `with`
never introduces anything -- the holder has to be known already.

One operator here does not answer. **`noting` fires first, records what the
utterance said about the world, and returns CONTINUE**, so the cycle goes on
and v689 answers the turn as it always did. That is what lets the open world
be the default without taking anything away: the scene stays current whether
or not this layer was the one to reply.

### Learning what VerbNet cannot say

`learned.py` is a sqlite file in `state/`, beside v689's memory and for the
same reason -- it is what someone told a running server. Three things:

```
excludes   two states of a thing that cannot both hold
requires   something else that has to be true to do a thing
brings     an effect VerbNet did not mention
```

**Negative preconditions come free with exclusion**, which is why exclusion
is worth learning first. `needs` is a list of slots that must be *present*,
so *the door is not already open* cannot be said — but once `open` and
`closed` are known to exclude each other, *the door is closed* says the same
thing and is positive. One thing learned closes two of §10's gaps, and
`learned.applied` folds it into the actions at grounding, where the facts
are ground and the store is current.

It is learned two ways.

**By being corrected.** The scene holds `closed door`; the person says *the
door is open now*. Two states of one thing, one right after the other, the
second marked as a change — that is what incompatibility looks like from the
inside. Both have to be adjectives, asked of WordNet, because `mammal whale`
and `closed door` are the same shape and only one of them is a state
something can stop being in.

**By being told.** `a door cannot be open and closed`, `open and closed are
opposites`, `you can only drop it if you are holding it`, `you must be
holding it to put it down`, `it must be closed before you can open it`. A
taught condition is over `?subject` and `?object` rather than over thematic
roles, because a person says *you* and *it*, and which role those are is
different for every verb.

Nothing is inferred from co-occurrence: two facts holding at once is
evidence that they *do not* exclude and never evidence that they do. So
`forget` is a plain part of the interface — both a correction and a lesson
can be wrong.

```
> the door is closed
  all right: the door is closed
> actually the door is open now
  I see -- the door is open. I did not know a thing cannot be open and
  closed; I do now
> the window is closed
> open the window
  I open the window
> what do you see
  the door is open; the window is open
```

Learned of a door and used on a window, and still there next time the server
starts. That is what makes it knowledge rather than a note about one thing.

### The trap this cost

A regex word boundary (backslash-b) written through a shell heredoc became a literal backspace, so
`ASKING` — the pattern that decides an utterance is a question — could never
match. Invisible in the file and in every `grep`. It shipped in the previous
commit and did nothing, because no world was open by default; the moment one
was, the agent started answering *does a table have legs* out of a scene and
the probe moved by three. **This is the third time in this project, and the
rule is already written down: use a file, never a heredoc, for anything
containing an escape.** The repair was by line index with no escapes at all.

## 10. On the page, step by step

`python -m research.v690` is the page, and it is the same page it was: text
in, text out, v689 answering what it answers, the decoder writing those
replies, every step behind a turn one click away. What is new is that an
order is carried out, and the turn shows how.

A turn that asked for something gets a step of its own, **planned**, between
*reasoned* and *remembered*:

```
[reasoned] The want operator answered; noting went first and let it go on.
[planned ] You asked for the cup is in the shop, the book is in the garden;
           961 actions were possible, over 8 verbs; 2 actions did it, off a
           goal stack 2 deep (1 subgoals, 2 operators fired).
```

Its details are what was wanted, what it had to choose from, the plan with
each action as it was carried out, any surprise and what it did about it,
and **the goal stack**: each goal, what fired in it, and what it pushed, as
an outline you open a level at a time. A turn that asked for nothing has no
such step and reads exactly as it did.

What had to change to get there, each found by running the page rather than
a harness:

- **`executed` never reached the page.** `Turn.as_dict` has no key for it,
  so no executive run of any layer has ever been visible. The planning is
  put on the answer instead, curated (`Scene.planning`): a turn's full
  `executed` is thirty runs for an ordinary question, and archiving that on
  every turn would show nothing most of the time.
- **v689 splits `A and B` into claims** and runs the act executive once per
  claim, so the second goal of *get the cup to the shop and the book to the
  garden* was lost, and the leftover claim, read alone, garbled the scene.
  v691 now hears the **whole** utterance once per turn, and a later claim
  of a turn it answered is `settled` with nothing. v689's merge of claim
  answers keeps what a later layer put on the first (`LAYER_KEYS`).
- **Never do what would change nothing.** After the cup was in the shop the
  planner put it there five more times, by five verbs, before turning to
  the book. An action whose every effect already holds does not propose.
  Every measured number is unchanged by it: declared domains never had such
  an action.
- **A kitchen holds more than one thing.** Putting the cup there took John
  out, because a blocks rule -- one block sits on another -- was applied to
  every place. It applies now only where a domain says a target is taken
  (`taken clear 2`).
- **The narration.** Past tense comes from WordNet's own list of irregular
  forms, read backwards (`took`, `went`, `left`); names are said without an
  article because they were said without one; the doer is only ever a name,
  because the store's categories are a union over senses and some sense of
  `cup` is animate enough to carry things.
- **The account said `noting` had nothing.** An operator that went on did
  its work; *reasoned* now says so, *answered* says a scene reply is not a
  verdict, and *said* says the reply is the agent's own words rather than
  reporting a decoder that never ran.

## 10b. A doing, learned from what the story showed

*"It should have learned that a pig requires to be carried by a plane to
fly."* Four pieces, none of them about pigs or planes:

- **Experience, in v689.** When the story tells a doer doing something one
  word long -- *the plane was flying* -- whoever it last put aboard that doer
  did it too, carried (E2 turned round). `Session.experience` keeps it: the
  thing's kind and sense, the carrier's, the verb and preposition that put it
  aboard (`put on`), and whether the thing could have done it itself
  (`Session.can`: v688 on the kind first, because the raw walk climbs from a
  pig to a crawled `animal capable_of fly` about bats).
- **Long-term memory, in v691.** `page.absorb` moves experience into
  `Learned.carry` -- kinds, not individuals, so what was seen of one pig is
  kept of pigs -- in `state/v691-learned.sqlite`, where it survives the
  conversation and the server.
- **A doing is a goal.** A verb VerbNet has frames for and no verb brings
  about (`Open.a_doing`: `fly`, not `open`) is done by the thing itself if
  its kind can, or by a learned carrier: an action that needs the thing
  aboard and adds the doing. What put it aboard is offered first and grounded
  **whatever VerbNet restricts its roles to** (`verbs.seen_done`): put-9.1
  wants a location, and a pig was put on a plane. That is also the first
  reason here to prefer one verb over another that is not a class count.
- **Asking is not ordering.** *What steps are required to…* is planned in
  a `world.Imagined`, which changes and announces nothing, and answered with
  the steps and where the way came from.

**Then generalised past pigs, planes and flying.** What was seen is a fact
about the *carrier* -- what is aboard it goes where it goes -- and the pig
was only evidence. So for a doing `V X`:

- only a **motion** is carried (`change.moves`: some frame of V says
  `motion(E, subject)`): fly, sail, swim, travel, roll; never bark or eat,
  in v689's E2 as well as here;
- a **carrier** is one seen in experience, or any sense the store says does
  V (`capable_of`) that is a vehicle or container by its own ancestors
  (VerbNet's restriction words), best attested first, each judged by v688
  (`Session.can`) before it is used -- the store's rows are candidates;
- **X goes aboard** if the verb that loads it takes it (`verbs.takes`:
  put-9.1's Theme is concrete, so an idea does not) and X is smaller than
  the carrier on the scale people rated (`Session.smaller`, v688's R31).
  Unrated or too close is *not known*, and said so.

`make X V` asks X to do V (`OpenReader`, `Open.doings`) unless V names a
state -- WordNet's adjective, `verbs.stated`, T4's own rule -- so *make a
chair bark* is not VerbNet's *bark your shin*. A piano flies on the plane
it learned about and sails on a boat it never saw; a sofa sails; a chair
does not bark; a house's size against a plane is not known, and it says so.
Errands unchanged at 12/12 reached, 8/12 sensible. Found on the way: in the
open world a new thing with no kind beside it took the scene's first kind,
which made a house a person.

Two older faults surfaced on the way, both a v691 act taking a turn v689
needed: *the pig is in a field* was answered by `tell` and never remembered
(T3 then answered from yesterday), and *i put the key in the drawer* was
taken for an order and carried out. Now an open-world statement is only
`noting`'s, and an order must be said as one (`page.ordered`).

## 10c. What a surprise teaches

§6 left `acting.Gap` as the first learning signal here that is not an
external judgement -- a prediction the agent made and the world falsified --
and nothing learned from it. `lessons.py` does.

**What it learns.** Three kinds of thing, all over the action's positions so
that what one door taught holds of every door:

```
requires   held every time it worked, and not when it failed
blocks     held when it failed, and never when it worked
brings     came about, about the things acted on, and nobody predicted it
```

**The evidence rule.** A failure is compared with every success of the same
verb on record (`learned.tried`, kept across conversations). One candidate,
and it is learned; more than one, and nothing is, and the failure waits for
the next success. A version space with the hypothesis language cut down to
one literal, which is what a single surprise can support. **A lesson learned
by trying is taken back by the first success that contradicts it**; one a
person taught is not, because a success that did not show it may just not
have shown it. The one exception to "one candidate" is a person saying why
-- *the door is still closed, it is locked* -- where what was revealed with
the failure is the candidate set, filtered by the successes on record.

**The thing changed is `?it`.** A lesson over name positions broke at once:
VerbNet has `open door` and `open john door`, the door is in a different
place in each, and a blocker learned from one missed the other. So the thing
an action's one-place effects are about is `?it` wherever it is named
(`learned.changed`), and a lesson about doors finds the door in every
reading.

**Negative preconditions, natively.** `Action.forbids`, checked by the
world, and compiled for the planner into a positive `not X` slot -- true at
the start where X is not, brought about by whatever deletes X
(`acting.negated`). Means-ends then plans to *remove* a blocker exactly as it
plans to bring about a need, and a domain string can say `forbids`.

**Removing a blocker needs to know what removes it.** VerbNet has no frame
in which `unlock` ends `locked`. Two derivations from WordNet, neither a
list: the **opposite** of a state (`verbs.opposites`: a word's own antonyms,
and the `un-` that negates -- `locked`/`unlocked`), and the **verb that
leaves a thing in a state** (`verbs.maker`: `unlocked` is `unlock`'s
participle). A learned blocker makes its opposite wanted, and the opposite
is made by its verb. `unlock the door` asked for is `unlocked door` wanted,
by the same derivation read the other way.

**Measured, in a world with rules nobody told it** (`hidden.py`): `go` needs
the destination lit, `fetch` cannot be done to a wrapped thing, and the
model knows `light` and `unwrap` but not that they are ever needed. Four
distracting properties (busy, quiet, red, old) vary at random. Over eight
seeds, thirty problems each:

| | solved, first 10 | solved, last 10 | surprises, last 10 |
| --- | --- | --- | --- |
| no learning | 1 or fewer | 0 | ~40 |
| learning | 3 to 10 | **10, every seed** | **0, every seed** |

Every seed learns exactly `go requires lit ?it` and `fetch blocks wrapped
?it`, and nothing else -- no false lesson, no effect it was not shown. How
long it takes depends on how soon the successes rule the distractors out.

**On the page:**

```
> the door is closed
> open the door
  I opened the door
> the door is still closed, it is locked
  I see -- I could not open the door, because the door is locked. I did not
  know that nothing locked can be opened, and I will remember it.
> open the door
  I unlocked the door, then opened the door
> the gate is locked
> open the gate
  I unlocked the gate, then opened the gate
```

A report is read three ways -- a state *still* holds, what it should have
brought about is denied, or *that did not work* -- and the world goes back to
just before the step that failed, because everything after it was done in a
world that was not so. When there is no reason and nothing to compare, it
says it does not know why yet; that is where asking (§12) begins.

## 10d. Asking when it is stuck

An order it could not plan used to end the conversation about it: *I could
not see a way to do that*. Now the order is **suspended**, what is missing
is asked for, and the answer resumes it (`Scene.missing`, `Scene.resume`).

**What is missing** is worked out the way a planner's heuristic is. First,
what is reachable at all if nothing were ever undone -- a relaxed plan over
the actions on offer, with negative preconditions compiled as in §10c. If
the goal is reachable on that reading, what failed was the search, and no
question is asked: that would be asking the person to do the planner's job.
If it is not, back from the goal through whichever way of getting each fact
leaves least unreached, to a fact nothing brings about. That fact is the
question, and its shape says how to ask it:

```
a thing an order moves, that nothing places     where is the book?
a state nothing it knows can bring about        is the box safe?
a blocker nothing it knows can undo             the door is locked -- what
                                                would change that?
```

**A fact not said is not known to be false.** In a world nobody declared,
the planner's closed world is a convenience, not a belief, so a state is
asked about before it is called impossible. *No* turns the question of
whether into one of how; *never mind* drops the order.

**Where a thing is, is asked first.** VerbNet has readings of `leave` and
`send` with no precondition at all, and a plan built on one said it moved a
book it had never found. So an order that moves a thing nobody has placed
asks where it is before planning at all (`Scene.unplaced`).

**Only what nothing known can do is asked.** A taught requirement --
*you can only open it if it is unlocked* -- is a state `unlock` makes
(§10c), so it is simply done: *I unlocked the box, then opened the box*.

**The answer is still the conversation's.** *The book is in the garden*,
said as an answer, resumes the order -- and v689 must still remember it, or
episodic memory loses a statement (the bug `recorded` fixed in §10). So the
`resume` act runs v689's own `tell` first and declares its effects (E4c):
over HTTP, *was the book in the garden?* is answered *yes -- you told me*.

```
> john is in the kitchen
> get the book to the kitchen
  I will need to know where it is first. Where is the book?
> the book is in the garden
  then I left the book to the kitchen
> you can only open it if it is safe
> the box is closed
> open the box
  I could not see a way to do that yet. Is the box safe?
> no
  Then how would I make it so that the box is safe?
> never mind
  all right, I will leave it
```

Resuming can ask again -- the next thing missing -- which is how a plan is
put together over several turns. The goal is held in the scene rather than
in a live executive, because a turn is one run of the act executive and a
suspended subgoal that outlived it would be state no trace could show.

## 10e. One reader, off the parse

§9 called the reader "nine patterns" and said joining it to v689's grammar
was the work. The patterns had a list of the verbs that move things
(`get|put|move|take|bring|carry|send|place`), so the planner could reach only
as far as the list: `how would a pig fly?` was not read at all, and `the
door is still closed` came out as a door in the state *still*.

`hearing.py` reads the dependency parse instead, into v689's own `Word`
type, and **an order's goal is what the sentence says, with VerbNet deciding
the one thing a list decided before**:

```
be + place / state / participle      at book shop, closed door, locked door
have / hold                          with book john
there is X on Y                      at cup table
V X  PREP Y                          at X Y -- any verb
V X to Y, where VerbNet's Theme      with X Y  (give-13.1: the Theme leaves
  leaves the Agent                   the Agent)
V X                                  V X, or the state V leaves (unlocked)
make X V / how would X V             a doing
```

**What it will not take**, each found by the probe or the shared test run
and each a turn stolen from v689 or v688 when it was wrong:

- a question states nothing (`what steps are required`, `does a table have
  legs`);
- a plan is asked for only by `how` on the verb itself with a modal -- *how
  would*, *how can*, *how do I* -- or by *what would it take*: `how many
  times did the dog bark` is v689's count, and `what do you need to bake a
  cake` is v688's;
- `can you V X` is a request only when what it names is in the scene: *can
  you close the box* with a box here, and not *can you eat an apple*.

**The parse is spaCy's transformer model**, the same one every layer now
reads with (`v687.language.load`: loaded once per process, shared by every
engine under one lock, on the GPU where there is one). The small one tagged
`fly` in *how would a pig fly* as a noun and `mary` in *give the cup to mary*
as a verb. The store's own fragments are still lemmatised by the small model
(`language.lemmas_of`): matching a fact to a question is lemma overlap over
thousands of fragments a turn, which is lookup, not reading, and on the
transformer it cost one question 128 seconds. Moving v689 over took three
fixes, each a place the small model's quirks had been relied on: a word
looked up alone is lemmatised by WordNet's noun forms (`mice`, `wembles`);
a claim that is only the words placing it in time (`following that`) is
not a claim; and the ops a GPU model runs with are set per thread. The
patterns remain only for when no parser is installed.

Being able to ask `how would a piano fly?` on the page exposed two older
faults in the same conversation: the pig named a turn earlier is animate, so
VerbNet let *it* load the piano; and a jet an earlier answer had named was
picked over the plane actually seen, by alphabetical order. Now whoever asks
what it would take is who would do it (`seen_done`'s `doer`), and the
carrier is the one seen, by its word, first.

**Not fixed, and not the reader's:** *give the cup to mary* is read right
(`with cup mary`) and planned wrong -- a VerbNet reading of `take` with a
box as its agent. That is choosing the verb, §8's open problem.

## 10f. What a lesson is about: scope, exceptions, widening

§10c's lessons were about everything: a locked door taught that nothing
locked opens, and one success against a lesson deleted it. Both are wrong
the moment kinds differ. **If a wrapped hat can be picked up and a wrapped
book cannot, comparing across everything hides the rule for good** -- the
hat's success held `wrapped`, so `wrapped` is never the one candidate.

So a lesson has a **scope** (a kind, or everything) and **exceptions**, and
every try records the kind of thing it was about (`learned.tried.kind`):

- **Compared across everything first**, as before. If that leaves other
  than one candidate, **compared again within the failing thing's kind**,
  and a lesson found that way holds of that kind.
- **Widened** when a failure of another kind is explained by it: two kinds
  are the evidence it was never about the kind.
- **Narrowed, not dropped,** by a success of a kind no failure ever
  supported: that kind becomes an exception. A success of a kind that *did*
  support it refutes it, and it goes.

Kinds come from wherever a domain can say: a declared domain's object
types, the open world's words and their senses (`Open.is_a`). With nothing
to say what kind a thing is, every lesson is about everything, as before.

**Measured** (`hidden.py --kinds`: hats are garments, and the wrapping rule
is not theirs), eight seeds:

| | seeds ending with 0 surprises | wrong lessons kept |
| --- | --- | --- |
| compared across everything | 6 of 8 | 0 |
| compared within kind too | **8 of 8** | 0 |

In the two seeds where it matters, the global comparison never finds the
rule. Within the kind, it is found and held of things only. (Measured with
plans shortened, §10h. Before shortening the global comparison ended clean
on 5 of 8 and in one seed settled on a coincidence -- `fetch` needs a
*quiet* place -- which is the risk a one-candidate rule carries.)

**What it costs, said plainly.** In three seeds the rule is learned of
everything *before* any hat shows otherwise, and from then on hats are
unwrapped for nothing (2 to 4 needless steps in the last ten problems). A
lesson that is always planned around is never tested, so the exception is
never found. That is exploration against exploitation, and the choice here
is to generalise and pay in needless steps, not to fail again on purpose --
because generalising is what was asked for (§10b: *any* object that fits on
a plane can fly), and a door's lesson should hold of a window without the
window being tried.

## 10g. Choosing the verb: who acts, and what people say

§8 left *sensible 8/12* as a knowledge problem, and the roadmap called it
verb preference. **Measured first, it was mostly not about the verb.** A
preference for `carry`, `take` or `go` left it at 8 (one version dropped it
to 7), because the plans that were not sensible were wrong about *who
acts*: the book went from the shop to the kitchen by itself, and the book
took the cup there. VerbNet allows both, because `Things.allows` reads every
sense of a word -- as it has to, or a dog is an andiron -- and some sense of
`book` passes a restriction to the animate.

**Who acts is a question about the thing, not the word.** `Things.acts`
reads the thing's most common sense, which is WordNet's first (WordNet
orders senses by use): a man, a dog, a pig and a fly are living things; a
book, a cup and a plane are not. Where anything in the scene acts, an
agent's role -- a restriction to the animate or to something with control --
and a subject that moves *itself* are filled only by what acts. Where
nothing does, nothing is taken away, and a door still opens by itself. A
name is someone (the open world's `names`), because WordNet's `john` is a
toilet. This is §11's first gap -- *facts about bodies, true of every verb
and so written on none* -- closed for one fact by a derivation, not a list.

**Then the verb, learned.** Three sources, in order:

- the verb the order was said with (`carry the box into the garden` is
  carried);
- what people say they did: *sam went to the park* is a way `at` comes
  about (`hearing.done`, VerbNet's location effect), kept in long-term
  memory by every statement the page hears (`page.seen_done`);
- corrections: *no, carry it* after a plan, weighted twice a sighting.

| errands | sensible |
| --- | --- |
| before | 8/12 |
| who acts | **10/12** |
| who acts, and five sentences about other people doing other things | **11/12** |

The five sentences (`errands.SEEN`) share no person, thing or place with the
errands; only the verbs carry over. The one left is *put it down*, where
the person is already holding the book and `put` is what anyone would say:
a preference keyed only on the fact brought about cannot see that, and one
keyed on what already holds is the next refinement.

```
> get the box to the garden
  I took the box to the garden
> no, carry it
  All right -- next time I will carry it
> get the cup to the garden
  I carried the cup to the garden
```

## 10h. What a plan costs

§2 said control transfers and plan quality does not: means-ends takes the
first way to each subgoal, counts nothing, and `go` repeats. The fix is not
a cost function inside the search. It is to **cut the plan it found, in the
model, before acting on it** (`acting.shortened`). Three cuts, none of which
knows a domain, and none of which can make a plan wrong, because every
candidate is run in the model before it is kept:

- a stretch that comes back to a state already passed through did nothing;
- an action whose removal leaves the rest applicable and the goal reached
  was not needed, tried last first;
- a stretch that one available action takes from the same state to the same
  state is a detour -- going home and then to the shop, where going to the
  shop was on offer.

| | shortest, before | shortest, after | steps, after (oracle) |
| --- | --- | --- | --- |
| blocks suite (10 solved) | 9 | **10** | 60 (60) |
| blocks held out (67 solved) | 63 | **67** | 302 (302) |
| errands (30) | 1 | **24** | 231 (218) |
| delivery (30) | 3 | **14** | 237 (163) |

Nothing that was solved stops being solved. Blocks plans are now all as
short as the oracle's. What is left in delivery is **interleaving** -- one
trip carrying two parcels -- which no cut of a found plan can make: the
plan has to be found that way, and that is search with a cost, the one part
of this item left.

## 10i. Saying it, and remembering how

**Narration keeps the person's preposition.** A place said in an order is
said back with the preposition it was said with, when it is done with the
verb it was asked with: *I carried the box into the garden*, not *to*. Done
another way, it is *to*: *took the book on the table* is worse than saying
nothing about it.

**A reply may not say nothing was told when something was.** The decoder --
a trained model, not to be retrained -- would answer *was my pig flying*
with *you didn't tell me anything about it*, where v689's own answer quotes
what was told. The read-back check (`v690/roundtrip.trace`) already refused
*you told me* of what the store said; it now refuses the mirror image too,
*you didn't tell me* when v689's answer says *you told me*, and the speaker
takes the next candidate. Nothing about the model changed; what it may get
away with did.

**A chunk is keyed on what the impasse was about** (E6, §3). Keyed on every
fact, a chunk in a world never came round again. The executive now asks
working memory what an impasse is about where it can say
(`Situation.about`: the facts about the things the missing facts are
about), which is one read added to `executive.py`, like `pursuing`:

| shared chunks | hits, keyed on the world | hits, keyed on the impasse | subgoals saved |
| --- | --- | --- | --- |
| blocks (46) | 6 | 26 | none |
| errands (30) | 4 | 66 | 14% |
| delivery (30) | 15 | 57 | 18% |

Nothing solved changes. Blocks gains little, because there the whole tower
decides how a block is cleared; more recalled chunks are also forgotten
there, being tried and not fitting.

**Not done: a text environment** (ScienceWorld/TextWorld). None is
installed, ScienceWorld needs a JVM, and the standing preference is for
small evaluations of a mechanism over benchmark harnesses. `hidden.py` is
that world for now: rules the agent was not told, found by acting.

## 11. What this does not do

Said plainly, because the gap is the interesting part.

- **VerbNet says what changes and not what else must be true.** That you
  must be where a thing is to pick it up, that a hand holds one thing, that
  a door must be unlocked before it opens: these are facts about bodies and
  rooms, true of every verb and therefore written on none of them. Some of
  it is derivable — being in one place is, above — and some is not.
- **Antonymy and negative preconditions are no longer missing**, but they
  are *learned* rather than known: until somebody says so, opening a door
  does not retract `closed door` (§9). Nothing seeds them from WordNet,
  which would be a data step and not a hard problem.
- **The guard is a seam.** With the open world as the default, `scene_ish`
  decides what this layer may answer, and a question form it does not
  recognise is a turn taken from v688. The probe is the test that catches
  it, and it has caught it twice.
- **Nothing is inflected.** The narration says *I leave the book to the
  kitchen* because no morphology is available in the repository and a table
  of irregular verbs written here would be exactly the hand-written thing
  this was built to avoid. UD_GUM has lemma-and-form pairs and would settle
  it.
- **The reader is nine patterns.** Enough to state a situation and ask for
  something, which is what it takes to show the planning is general. Joining
  it to v689's grammar is the work ToMi and StepGame showed is not free.

## 12. What is next


- **a verb preference keyed on what already holds**, per §10g: *put it
  down* when the book is in hand.
- **the axioms about bodies**, per §10 — that you must be where a thing
  is to touch it. Antonymy is no longer on this list: §9 learns it.
- **plans found interleaved**, per §10h: cost inside the search.
- **v691c — a real text environment** (ScienceWorld over ALFWorld: it is
  science, so the graph's knowledge is relevant), where §3's irreversibility
  stops being theoretical.

Outside v691, this touches four places, each small: two pure reads in
`executive.py` (`pursuing`, and asking memory what an impasse is `about`),
the registration hook in `session.py`, and in v690 the read-back check
(§10i) and the page server saying where learned memory is kept. v691 is a layer: v687 knowledge, v688
asking, v689 conversation and events, v690 generation, v691 world, actions,
execution and monitoring.
