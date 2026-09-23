# v692: mathematics as a subject

v691 gave the system an agent: a world that changes only by acting, a plan
carried out a step at a time, and a surprise treated as an impasse to make
sense of. What it did not have was a **subject** — a body of exact knowledge
deep enough that every layer has something to do with it, and shallow enough
in its inputs that nothing has to be guessed.

Mathematics is that subject, and it is not a separate system. It is one
package, `research/v692`, that **registers into the layers already built**:

```
v687  reasoning.contributes  ->  R33, a layer of the reasoner
v688  (through the pool)     ->  the loop gets R33 for free
v689  session.contributes    ->  an act on the page, with a workspace
v690  speaking.contributes   ->  the round trip that checks a maths reply
v691  acting.agent           ->  equations solved, measures planned
```

Nothing above knows mathematics exists until v692 is imported, and v692
never reaches up. This is the same direction v691 registers in, and it is
the reason the subject can be removed by deleting one import.

## 1. There is no parser, anywhere

The first version of this had a hand-written reader for mathematical
English. It was deleted. The rule now is Adrian's:

> i don't want a custom parser, I want the decoder and encoder trained with
> math terms

So the only thing that turns words into mathematics is the **encoder**, and
the only thing that teaches the encoder is a **generator that goes the other
way**. `saying.py` takes a sympy object and says it aloud, and as it says it
it records, for every word, the symbols that word stands for:

```
x^2 - 5x + 6     x   squared  minus  five  x  plus  six
                 x   ^2       -      5     x  +     6
```

Because the object comes first, no alignment is guessed and no grammar is
written: `squared` stands for `^2` because it was said *for* `^2`. A word
that stands for nothing is `DROP`; a word that is its own symbols is `KEEP`;
a group closes on the word that ends it, so in *the square root of the
quantity x plus one*, `one` carries `1 ))`.

The same object is said many ways — *x squared*, *x^2*, *x to the power of
2*, *the quantity x plus one*, *open bracket x plus one close bracket* — and
which way is drawn per sentence, because an encoder taught one phrasing
reads only that one.

**Every record is checked by putting it back together.** The symbols of each
part are handed to sympy and must come back as the object the part was said
for; a saying that does not come back is not taught (`corpus.record`). That
is the round trip this project holds every corpus to, and here it is exact
rather than approximate: `same()` is equality after simplification, not
string equality.

## 2. sympy's parser, behind a whitelist

Symbols become objects in one place (`symbols.parsed`), and that place is
sympy's own `parse_expr` — which works by `eval`. So what reaches it is
checked first: only the characters mathematics is written in, and only names
the labels can say (functions and constants, single letters, Greek, units,
`f`/`g`/`h` where something is applied to them). Anything else is
`Unreadable`. Relations are split off before parsing, because `=` is
assignment to Python, and are built unevaluated, because *2 + 2 = 5* is a
claim to check and not `False`.

This is the whole of the "reading" side, and it is about 130 lines. The
encoder does the language; sympy does the mathematics; the whitelist is what
lets the two meet without `eval` being reachable from the page.

## 3. One table: the curriculum

`curriculum.py` is the subject as a table, and everything else reads it.

**Kinds** are where the store and computation meet. `prime number.n.01` is
in the store, with its gloss and its place under `integer`; whether 91 is
one is not a fact anybody could store, it is a computation. So a kind is a
store concept (or, where WordNet has none — *even number* — the concept it
is a kind of) plus the exact test for membership. 31 kinds: numbers,
expressions, equations, matrices, sets, propositions, sequences.

**Acts** are what an utterance asks, with the ways people ask it. 52 acts
across 14 branches — arithmetic, number theory, algebra, calculus,
statistics, combinatorics, linear algebra, sets, geometry, measures,
sequences, logic, probability, complex numbers. Adding a branch is adding
rows: a filler that draws an object, templates that ask about it, and a
handler that does it exactly (`doing.py`). The encoder corpus, the
conversation act, the decoder's pairs and the evaluation bank all follow
from the table without being told about the new branch.

**Nothing in the curriculum is learned.** The tests and the handlers are
sympy, in the same spirit as v691's `numbers.py`: what is exact is written
once and never approximated.

## 4. What the encoder was taught

Three heads on the shared MiniLM encoder (`research/encoder.py`), taught as
a new task beside the others (`v689/teach_reader.py`, `--subject math`):

```
math_act     what is asked: value, solve, factor, is kind, measure, ... or none
math_role    each word's part: EXPR VAR A B LIST KIND OTHER NAME VALUE WANTED SHAPE, or O
math_symbol  the symbols each word stands for: ^2, sqrt(, 5, KEEP, DROP, radius, circle
```

`math_act` includes `none`, and three records in ten are utterances with no
mathematics to do: the reader's own corpus, and near misses that belong to
another layer — *what is a prime number* is a definition (R26), *mary has 3
apples* is a count in a world (v691), *how many sides does a hexagon have*
is the store, *the pentagon is in washington* is not mathematics at all.
Leaving things alone is a thing the encoder is taught, not a threshold.

Two roles are worth naming. `NAME` and `VALUE` are how a measure's givens
are read: a run of `VALUE` words belongs to the quantity `NAME`d last before
it, so *legs 6 and 8* gives the leg and the other leg, and the quantity
itself is the **label** the encoder gives the word (`radius`, `surface-area`),
not the word. The same for `SHAPE`: *right-angled triangle* is labelled
`right-triangle`. So the vocabulary of shapes and quantities is learned,
and `reading.py` never looks a word up.

## 5. R33: mathematics in the reasoner

`semantics.py` registers a layer with v687's `ReasoningEngine` through a new
`reasoning.contributes` hook, and v688's loop gets it through the pool.
It answers two kinds of thing:

- **truths about objects**: *is 91 prime* — no, 7 times 13;
- **claims about kinds**, checked on the members: *are all primes odd* — no,
  2. The members are drawn from the kind's own test over a range, and the
  quantifier is R20's, so "all", "some" and "no" are not re-implemented.

This pre-empts R18's refusal to answer about numbers it has no facts for. It
is a layer like any other, and it says nothing when the question is not
about a kind it knows.

## 6. The page: a conversation that keeps what it was told

`page.py` registers one act with v689's session, proposed when the encoder
reads an utterance as mathematics. What it answers is shaped as the rest of
the system speaks: a value is `retrieved`, a yes or a no `verified` or
`denied`, something taken in `noted`.

A conversation has a **workspace**: what its variables were said to be (*let
x be 5*), what its functions were defined as (*let f(x) = x^2 + 1*, then
*what is f(3)* — 10), and one measure left open. Those are the
conversation's, not the world's: v689's event types are closed, and what `x`
is here is not a fact about anything.

## 7. Planning, twice

**Equations** (`solving.py`). The state is what the equation is like —
brackets, fractions, the variable on both sides, linear or quadratic, zero
on the right, factored, and whether the discriminant is a perfect square —
worked out by the world after every step. The actions are algebra's moves,
each saying what it needs and what it is for. What a move *does* is sympy's,
so the plan is always tried against mathematics rather than against the
model.

That gap is where the learning happens. Factoring `x^2 + x - 1 = 0` does
nothing over the rationals: the planner expected `factored eq`, the world
says no, and that is v691's impasse. The learner compares it with the times
factoring worked — every one had a square discriminant, this one had not —
and learns that factoring **requires** it. Nothing in the file says when
factoring works; the replan goes by the formula.

**Measures** (`measuring.py`). Each shape's formulas are actions: from the
quantities a formula relates, all but one known, the last becomes known. So
`area = side^2` is *side from area* and *area from side*, and the question
*what is the perimeter of a square with area 49* is a planning problem whose
plan is the derivation — side from area, then perimeter from side. Units are
carried through (`25pi square centimeters`), and mixed units are put in the
first one's.

**What cannot be planned is asked for.** *The area of a rectangle with
length 4* has no plan; which one more quantity would give one is found by
planning again with each supposed known, and the answer names them: *I need
the width of the rectangle (or its diagonal, or its perimeter)*. The page
keeps the question open, so *the width is 6* finishes it. That is the
asking-for-what-is-missing that v691's roadmap wanted, in a domain where
what is missing is exactly definable.

**How many sides a polygon has is read from WordNet**, not written here:
*hexagon* is `a six-sided polygon`, *decagon* `a polygon with 10 sides and
10 angles`. So the angles of any polygon WordNet names can be worked out,
and the store is doing what the store is for.

## 8. Saying it back, and checking it as mathematics

`speaking.py` builds the decoder's pairs through v690's own `message.of_turn`
and `prompt`, so the prompt the decoder learns from is the prompt it will be
given. The reply says the result in words, the way `saying.py` says any
object — *the derivative is three x squared*, *no, 91 is not prime: it is 7
times 13*.

The same saying is also an **answer record** for the encoder, which is what
lets a reply be read back. v690's round trip compares words; for mathematics
that is too strict and too weak at once, so v692 registers its own check
(`speaking.traced`, through a new `speaking.CHECKS` hook): the encoder reads
the decoder's reply into an object, and **sympy decides whether it is the
same object** the message was about. *x = 2 or x = 3* and *x equals 3 or 2*
both pass; *x = 2 or x = 4* does not.

## 9. What was measured

`bank.py` is 98 questions written by hand, in phrasings that are not the
curriculum's templates, and written **before** the encoder was trained on
any of them. Each is scored three ways: *read* (the act is the one asked),
*right* (the answer is the expected one), *honest* (right, or said not to be
known — never a wrong answer). Six of them are not mathematics and must be
left alone.

| encoder | read | right | honest |
|---|---|---|---|
| `llm/reader-math2`, taught the first 30 acts only | 66/98 | 65/98 | 87/98 |
| `llm/reader-maths2`, taught all 52 | **98/98** | **98/98** | **98/98** |

The first row is the baseline and the reason to believe the second. An
encoder never taught the later branches reads the earlier ones perfectly
(59/59 of the items that existed when it was trained) and, of the 32 it had
never seen, gets 6 right and leaves 21 alone rather than guessing. The
second row is the same questions after the branches were added.

Two things were learned from the training runs themselves, and both are in
the code as comments so they are not learned twice:

- **A subject's words are the rest of the system's words.** The first
  maths encoder said units short — `5 m`, `3 in`, `2 s` — and the reader
  that came out dropped the preposition from *expand in warm temperatures*.
  Eight tests of ordinary reading moved. Units are now said in full
  wherever the short form is an English word (`saying.UNIT_WORDS`), and
  they do not move.
- **A shipped model is not retrained by accident.** `teach_reader` and
  `teach_decoder` take `--subject math` and read the maths folders only
  when asked, so rebuilding `llm/reader` rebuilds the reader that shipped.
  The maths models are their own folders, and the runtime prefers them
  where they exist.

**What the decoder says.** 200 fresh questions (a seed the corpus was not
built with), each made into the message the page would build, answered by
`llm/decoder-maths`, and the reply read back by the encoder and compared
with sympy: **197 of 200** say the same mathematics the message says. The
three that do not are one garbled reply and two the encoder misread
(`thirteen` as 10, `at most` as greater-than). On v690's own held-out
messages, which have nothing to do with mathematics, the maths decoder
traces 280/300 against the shipped decoder's 281/300 — it learned a subject
without forgetting its work.

**On the page, over HTTP** (`v690/server.py`, port 8695, `--no-memory`),
the whole path runs: *what is 17 times 4* — "It comes to 68."; *is 91
prime* — "No, 91 is not prime: 91 is 7 times 13."; *what is the derivative
of x cubed* — "It's three x squared."; *what is the area of a circle with
radius 3* — "The area is nine pi."; *what is the area of a rectangle with
length 4* — "I need the width of the rectangle (or its diagonal, or its
perimeter)", and then *the width is 6* — "The area of the rectangle is
24."; *are all prime numbers odd* — "No, not all prime numbers are odd."
(R33, through the loop, not through the act). *What is a dog* still goes
to definitions and *mary has 3 apples* still goes to v691's world.

**What it cost.** With the maths encoder as the default reader, **1,358 of
1,359 tests pass**. The one that does not is `test_time`'s
`test_a_claim_said_as_a_question_is_confirmed`: *so pigs don't fly, but
this particular pig took a flight on a plane?* is read as two claims about
the particular pig rather than one about pigs and one about it. Four
encoders trained on this corpus fail it and every encoder trained without
mathematics passes it, so it is the corpus and not the seed. The likeliest
reason is in the numbers: 18,873 of the 90,000 maths records have a comma
in them, against 34 in 40,000 of the reader's own corpus, because lists
are said with commas — so a comma is now largely a maths signal. The
shipped `llm/reader` still passes it, and is one environment variable
away (`V689_READER_MODEL`).

On the 114-question common-sense probe, retraining moves four answers
(`can people eat rocks` gains one, `is bleach poisonous` sharpens to
verified, `do people need sleep` and `did people make rivers` fall to
unknown). That is the ordinary movement of a retrained reader — a rebuild
before mathematics existed moved two — and no maths question is among the
114.

## 10. What it does not do

- No proofs. Natural deduction was in the plan and is not built; what is
  here proves nothing, it computes and checks.
- No word problems. *Mary has 3 apples* belongs to v691's world, and the
  encoder is taught to leave it there.
- Kind claims are checked on members in a range, not proved. *Are all primes
  odd* is answered by 2, which is a counterexample; *are all primes greater
  than 1* is answered by not finding one in the range, which is weaker than
  a proof and is said as such.
- Geometry is measures, not figures: there is no diagram, no construction,
  and no reasoning about which shape a thing is.
