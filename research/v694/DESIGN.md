# v694: designing in the open world

v693 gave the system a designer for mathematics: propose a form with
holes, fill the holes from the goal, check the result exactly, learn from
what failed, and prefer the simplest design (Occam). This is the same
designer taken out of mathematics. The question is whether the loop ever
depended on maths. **It did not.** The agent over a draft, the learner,
curiosity, Occam and the learned proposer run unchanged. What had to be
built is what maths supplied for free:

| v693 (maths) | v694 (open world) |
|---|---|
| a specification: clauses with residuals and exact checks | a goal: v691 literals (`cut rope`, `cold milk`) in a scene |
| forms with unknown coefficients | ways with a hole for a *thing*: tool, place, wear, bring, helper |
| sympy solves the unknowns | the store fills the hole (`used_for`, `capable_of`, ...), the scene first |
| an exact check against every clause | the steps are carried out in a `W.Imagined` world from the scene |
| fewest unknowns | rely on nobody, suppose nothing, fewest steps, then the store's support |

## 1. What a goal is (`goals.py`)

A goal is v691's own literals, in a scene of v691 facts:

    cut rope          a doing (VerbNet's `cut`)
    cold milk         a state a thing is to be left in
    fixed car         a doing said as its result, as v691's reader writes it
    at book kitchen   a place, as v691 plans it

`features` says what a clause is, for the ways and the proposer to use:
its kind, whether its thing is alive or can't be moved (`patient-fixed`:
a room, or a floor that is part of one), and whether VerbNet gives its
verb an Instrument.

## 2. What the store knows (`knowing.py`)

The store's free-text rows are read **word by word, each word by its
lemma**, with no parser. A row *serves* a goal in two cases. Its first
word can be the goal's verb, as itself or its -ing form (`cutting paper`),
but not as something already done to a thing: an axle *fixed to the
vehicle* does not fix cars. Or it can keep, make or get a thing in the
goal's state (`keep food cold`), where the state's words come from its
first WordNet sense and that sense's own nouns and verbs. Neighbouring
adjectives count, but their verbs don't: *snappy* weather is cold, and
snapping is not how milk is made so.

How far a row is **about the goal's thing** (`about`):

| Row | Score |
|---|---|
| Names the thing itself (`kill flies`, for a fly) | 3 |
| Names a kind of it (`keep food cold`, for milk; `wake people up`, for john) | 2 |
| Names nothing (`cutting`, `keep things cold`) | 1 |
| Names something else (`cut hair`, for a rope) | ⅓ |

A few kinds are too general to count as a match, because everything is one:
*object*, *whole*, *artifact*, *part*. Without that exclusion, a tractor
that pulls heavy objects was offered for pulling teeth.

A means' score is its best row, plus a quarter of each of its next five.
Summed evenly, a gun that kills animals in six phrasings beat a
flyswatter that kills flies.

Four other things come from WordNet:

- `named_tools`: tools the language itself names. A noun derived from the
  verb gives `mow` → mower and `heat` → heater. What the verb's definition
  says it is done *with* gives `iron` (*with a heated iron*) and `sweep`.
- `has_parts`: WordNet's part meronyms, plus the store's `has_part` and,
  read the other way, `part_of`. A knob is part of a door, and the store
  says so only that way round.
- `category`: a word with more than 40 kinds under it is not a thing to
  get: *a kitchen utensil*.
- `kept_at`: whether the store says things like this one are found at a
  place: milk in a refrigerator.

## 3. The ways (`ways.py`)

| Way | Shape | Hole filled by |
|---|---|---|
| unaided | do it | the store attests people do it, *and* no tool for this thing is better attested, unless a part of the thing does it (a door's knob) |
| tool | get ?T, do it with ?T | `used_for` / `capable_of` rows, weighted (an artifact's `capable_of` counts half), plus tools the verb names |
| place | put the thing in ?P | a place *for* keeping things so, where things like it are kept (`kept_at`); not for fixed things |
| bring | put ?M in the thing | the place form turned round, for a fixed thing: a heater into a room |
| wear | give them ?W | for a living thing: a blanket |
| inside | do it with the thing in ?P | a place made for doing it to this thing: an oven (§6) |
| helper | ask ?H to do it | a person whose `capable_of` rows are about this thing: a plumber fixes toilets |
| move / hand | take it there | v691's `at` and `with`, as v691 plans them |

Where VerbNet gives the verb no Instrument (*fix*, *unlock*, *write*), a
tool is offered only on a `used_for` row about this very thing, or when
the verb names it. So a key unlocks doors, but duct tape, which "fixes
everything", doesn't fix cars.

**Composition.** Every tool, place or thing to wear is itself a goal:
*have one*. `obtain` meets it:

1. What is held costs nothing.
2. What is at hand is taken. A kind of what serves counts: a carving knife
   is a knife.
3. Otherwise one is supposed, with where one is usually found
   (`found_in`).

A goal of several clauses is designed clause by clause, each in the world
the ways before it leave. The knife fetched to cut the rope is held when
the string is cut, and costs nothing the second time.

**Another one.** For every means at hand, a second candidate supposes
another like it (`_twice`). It costs more, so it is only chosen when the
one at hand fails the check, for example when a lesson says a broken
fridge keeps nothing cold.

**Offered at all.** Candidates are the store's best four plus anything at
hand that serves. A candidate is dropped in each of these cases:

- every row about it is about something else;
- it is a category;
- it is the goal's own thing, or a kind of it;
- it is a general kind of another candidate the store backs nearly as
  well (*kitchen utensil* when *can opener* is there);
- the store backs it less than a quarter as well as the best candidate.
  Without this, a knife in hand "that digs" beat a shovel.

## 4. Designing (`designing.py`)

This is v693's loop, over ways:

1. v691's agent plans in a `Draft` whose facts are the clause's features,
   one plan per run.
2. A way that fails leaves the table for this draft.
3. Curiosity (0.15) and a last resort try ways a lesson holds back.
4. `checked` does each candidate's steps in a `W.Imagined` world from the
   scene, with what has been learned folded in first
   (`learner.applied`). Every step must apply in turn, and the clause must
   hold at the end.

**Occam** (`Candidate.cost`) ranks by, in order:

1. relying on nobody else (helpers);
2. supposing nothing;
3. fewest steps;
4. doing the verb that was asked, rather than something that brings the
   state about some other way (cleaning the floor, not laying a mat to
   keep it clean);
5. the store's support.

`_simpler` fits every way that could cost less than the one found, the
same rule as v693's.

## 5. Carrying out, learning, remembering (`carrying.py`)

`carry_out` works in a loop:

1. Design.
2. Do each step in a `W.World`.
3. When a step is refused, turn it into v691's `Gap` and pass it to
   `Learner.failed`. What a person says came with the failure (*the fridge
   is broken*) is the one case where a single failure is enough to learn
   from.
4. Design again from where things now stand.

The lesson is over the step's positions: `put` is blocked while
`broken ?object` holds. So it applies to every design after it, for
anything put anywhere.

`Remembered` keeps the ways that worked, and the recipes it has been
taught or has seen (§6). A way a person *teaches* is kept the same way as
one that worked. That is how a means the store never heard of gets into a
design at all: *cut the rope with a saw*, "you told me".

## 6. Learning from similar examples

What one example teaches is carried to examples **like it**. Likeness is
WordNet's, computed, never listed:

- `similar(a, b)`: 1 for the same thing or a kind of it (a carving knife
  is a knife), and a half for close siblings. Siblings share a kind no more
  than two steps up from each, deep enough to say something: rope and cord
  are *lines*, knife and scissors are *edge tools*, cloth and rag are
  *fabric*. Everything else is 0: a knife and a fork share nothing that
  close.
- `similar_verb(a, b)`: 1 for synonyms (*fix* and *repair* share a
  sense), and a half where one is a way of doing the other. *Slice*,
  *chop* and *saw* are ways to cut, *toast* is heating, and *bake* is
  cooking.

Similarity is used in three places, always as weaker evidence than the
thing itself and never as a decision:

1. **Ways that worked or were taught** (`Remembered.ways_for`) are offered
   for a similar doing on a similar thing, at the product of the two
   likenesses. Taught *you can cut the rope with an axe*, the designer
   cuts a cord with the axe in the shed, "you told me". Untaught, it
   fetches scissors.
2. **The store's rows** (`about`): a row about a close sibling of the thing
   counts 1.5. That is between a row about nothing in particular (1) and
   one about a kind of it (2). A helper whose rows are about siblings may
   be asked.
3. **The parts of a recipe** (below): a thing at hand stands in for a part
   if it is one, or a close sibling of one.

Occam still decides. What similarity offers must still check in the
imagined world, and among equals a row about this very thing beats one
about its kind (`Candidate.cost` compares how exactly the rows fit before
it compares support). A saucepan that cooks rice beats a grill that cooks
food.

### Building a tool from what is at hand

`Recipe(product, parts)`, taught or seen. **None comes from the store.**
The store's `made_of` lists what a thing *can* be made of (a blanket of
wool, or cotton, or fleece), not what it takes, and inventing recipes by
hand would be exactly the overfitting to avoid.

- `obtain` tries, in order: held, at hand, **built from a recipe whose
  every part is at hand** (by `similar`), and only then supposed. Building
  supposes nothing, so Occam prefers it to fetching one from somewhere.
- A thing that can be built is offered like one at hand (`_offered`). It
  is not dropped as too general, and it is not cut off by the store's top
  four.
- A recipe for a product serves wherever the product is a kind of what the
  store named. A torch is a light: taught *you can make a torch from a
  stick and a cloth*, with a stick and a rag about, *light the room* comes
  to

      I took the rag from the drawer, then took the stick from the shed,
      then made a torch from the rag and the stick (you told me how), then
      lit the room with the torch -- a torch is a light, and I know how to
      make one.

- A recipe used in a design that worked counts as having worked
  (`Remembered.made`).

### Doings done in a place

The `inside` way: a doing done with the thing in somewhere made for it
(`places_for`), such as bread in an oven or clothes in a sink. It is a
single step, as using a tool is: putting the bread in the oven and baking
it is baking it in the oven. The place must be one a thing is put into:

- not a room (pasta is not put in the kitchen to cook);
- not something held (that is a tool);
- never with a living thing inside (john is *taken to* a fireplace, which
  is the place form's to say).

A place made for doing it to this very thing also stops the doing being
called unaided. People bake bread, and they bake it in an oven.

## 7. English and the page (`page.py`, `teaching.py`, v691 changes)

v691's reader (`hearing.py`) is the open world's reader. It gained three
general parse rules, and no word lists:

- A verb's object left in the state its adjective names, where the parser
  makes the thing the adjective's own subject: *make the milk cold*,
  *keep the food cold* (`RESULTING`).
- Things joined by *and*: *cut the rope and the string*, *make the milk
  and the beer cold*.
- A word the parse reads as a noun in a thing's place is a thing, whatever
  else it can be: *open the **can***.

`teaching.read` reads ways and recipes off the same parse, by the
sentence's shape:

- **A way:** a verb with an object and a `with` phrase, or *use X to V Y*.
- **A recipe:** *make* or *build* with an object and a `from` or *out of*
  phrase, or *X is made from Y*.

It must be said of anyone (*you can*, *one can*), as advice, as what a
thing is made from, or as what was done (*i cut*, *i made*). So an order
(*cut the rope with a saw*) and a request (*can you ...*) teach nothing.

`v691/page.py` gained three hooks, and nothing else:

- `contributes(act, handler)`: v694 answers `want`, `how` and `why` first.
- `adds(act, recognise, handler)`: v694 adds `teach` (*you can make a
  torch from ...* → "I will remember that ...").
- `observes(fn)`: runs as every utterance is noted. v694 learns from what
  was **seen done** (*i made a raft from the logs and the rope*) quietly,
  so the turn is still v689's to keep as something that happened.

The designer takes an order only when its way is a tool, a place made for
it, a place that keeps a thing so, something to wear or bring, or a
helper (`TAKES`). Moving a book is still v691's planner's job, and so is
anything the designer cannot design.

Tried over HTTP on the real page (v690 server, `--no-memory`):

    cut the rope          I got a knife (there is usually one in a drawer), then
                          cut the rope with the knife -- a knife is used for cut rope.
    make the milk cold    I put the milk in the fridge -- a refrigerator is used for
                          keep food cold.            (the fridge the person mentioned)
    how can i make the beer cold   I would put the beer in the fridge -- ...
    get the book to the kitchen    I got the book to the kitchen   (v691's planner)
    fix the car           I asked a technician to fix the car -- ...
    why did you get the knife   I used the knife to cut the string with the knife -- ...
    what is a dog / find a quadratic with roots 2 and -3 / is a whale a mammal
                          answered as before

## 8. The learned proposer (`proposer.py`, `generating.py`)

This is v693's proposer, with one logistic model per way.

- **Features:** the clause's features, plus whether people do it
  themselves and whether the verb names a tool.
- **Training data:** 400 goals made from the store's own rows (a
  `used_for`/`capable_of` row "VERB NOUN" with a physical noun and a verb
  VerbNet gives an Instrument; "keep NOUN ADJ"), each labelled by the
  designer.
- **Stored in:** `llm/open-designer-proposer2`, rebuilt by the
  `open-designer-proposer` step of `regenerate.py`. The first model,
  trained before `inside` existed, stays in `llm/open-designer-proposer`
  untouched, because models in `llm/` are never written over.

| First way tried is the one kept | Proposer | Table order |
|---|---|---|
| held-out generated goals | 83/136 | 28/136 |
| the bank | 32/40 | 10/40 |
| the held-out bank | 24/31 | 7/31 |

It changes what is tried first, never what is chosen. The banks give the
same results with it and without it. The total number of ways fitted is
not lower, because Occam still fits every way that could be cheaper.

## 9. Numbers

`python -m research.v694.bank [--held | --held2] [--proposer]`

| Bank | Goals | Before §6 | After §6 | Status |
|---|---|---|---|---|
| `BANK` | 44 | 39 | 39 | development: rules were made while looking at it (first run 33) |
| `HELD` | 34 | 28 | 29 | first-round held-out; it showed what §6 should add, so no longer clean |
| `HELD2` | 31 | 19 | **21** (22) | written before §6, run once after it |

The **21** is the generalisation number for §6. The 22 is after one bug
its misses exposed was fixed: the inside way had put pasta in a kitchen,
and rooms are now excluded. "Sensible" means one of the ways and things
written down before the run. On all three banks every sensible design was
also the simplest the scene allowed.

`HELD` still scores *bake the bread → the oven, inside* as a miss, because
its expectations were written before `inside` existed and have not been
changed.

Tests: `test_v694.py`, 42 tests. For the full suite see the commit.

The misses that remain, and why:

| Goal | What happened | Why |
|---|---|---|
| open the can | a screwdriver | the store: screwdrivers open paint cans. True, but not the bank's answer |
| pull the tooth | a string | the store's folk method, "used for pulling loose tooth" |
| kill the fly | a bullet | rows about killing animals, and none about flyswatters |
| make the room bright | nothing | *bright* has no verb of its own, and no row keeps a room bright |
| sweep the kitchen | nothing | WordNet lists no parts of a kitchen, so the floor is not reached |
| fix the sink | nothing | WordNet's first *toilet* is a room, so the plumber's toilet rows are not siblings |
| grate the cheese, make the water cold | nothing | no row the reading accepts |
| whisk the eggs | a meringue | a `capable_of` row about egg whites |
| cut the wire (even with pliers in the drawer) | a laser | the store has no pliers-cut-wire row it accepts |
| wash the car | by hand | people wash cars, and no tool is better attested for cars |

## 10. What this is not

- **Not an LLM's breadth.** A way exists only where VerbNet has the verb
  and the store, the language or a person has said something. Physical,
  locational and change-of-state goals are covered. Social and abstract
  goals (*make john trust me*) have almost nothing, and fail by saying so.
- **Checks are plausible, not proven.** The imagined world checks the
  steps. That a knife cuts is the store's word, carried as support and
  said with the design.
- **Recipes only from people.** Nothing is built that nobody has shown or
  told it how to build.
- **Similarity is WordNet's first sense.** Where that sense is odd (a
  toilet is a room, a grill a grillroom), what is carried over is odd too,
  or nothing is.

## 11. Next

- Parts of the patient from the store's `part_of` too, once there is a way
  to tell its good rows from its noise (a fire truck is not part of a
  roof).
- Recipes the store can vouch for: `has_part` rows that list components,
  not materials.
- Teaching that is corrected: *no, a saw won't cut a rope*, taking a way
  back.
