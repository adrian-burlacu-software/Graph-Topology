# v693: designing towards a goal

v692 made mathematics a subject every layer works in, but every question
it answers is a **computation**: the answer is what sympy returns when the
question is done. v693 is the other direction. The goal is a
**specification** — *a polynomial with roots 2 and −3 whose value at 0 is
12*, *a sequence whose first 10 terms add to 275* — and the answer is an
object nobody has yet. Getting there is design: proposing what the object
might look like, finding out whether it can be made to fit, and learning
which proposals suit which goals.

That is the part of an LLM's strength this project did not have. An LLM
proposes from a prior learned from text; here the proposals come from a
table of forms, the judge is sympy, and what is learned comes from the
designer's own successes and failures.

## 1. Specifications (`spec.py`)

A specification is a kind of object and clauses it must satisfy. Every
clause is two things: **residuals**, expressions that are zero exactly when
it holds (which is what turns a form with unknowns into equations), and an
exact **check** on a finished object. Some clauses are only checked —
*whole-number coefficients* has no equation.

There is no expected answer anywhere. A design is right when the
specification holds of it; any object that meets it is one. *A sequence
whose 5th term is 48 and 7th is 192* is met by 3·2ⁿ⁻¹ and equally by
72n − 312, and both count.

`features` describes a specification the way v692's solver describes an
equation: what it gives, how much, and the shape of its data (terms of one
sign, evenly stepped, one ratio apart). These are what the learner reads.

## 2. Forms: the proposals (`forms.py`)

A designer does not search all objects. It proposes a **form with holes** —
*a scale times a factor for each root*, *a first term times a ratio to the
n − 1*, *a factor for each root times a polynomial for the rest* — and lets
the specification decide the holes. Ten base forms -- three for
polynomials, three for sequences, four for functions (§9) -- and the
composites made from them (§8). A form's `needs` say what it is built from
(the roots form needs roots), never when it works.

## 3. The designer (`designing.py`)

v691's agent, in a world whose one thing is a draft. The world states the
specification's features; each form is an action declared as bringing
about `designed d`. Doing one is exact: the holes go into every residual,
sympy solves, anything no equation decided is chosen (its default first,
then small whole numbers — which is how *whole-number coefficients* is met
by a scale of 2 for roots ½ and 3), and the result is checked against the
whole specification before the move counts.

A form that cannot fit leaves the draft undesigned, which is v691's
surprise: the move was for `designed d` and did not bring it about. The
learner compares that failure with every time the same move worked, and
what held every time it worked and not this time is what it requires. A
failed form is not proposed again for the same draft; what it taught is
kept for every draft after.

**The order moves are tried in is the proposer.** Equal to the planner,
moves go in the order handed over, so `design(order=...)` is where a
learned guess of which form suits which specification goes first.

## 4. First numbers (`bank.py`)

28 goals written by hand before the designer ran — 16 polynomials, 12
sequences, three of them impossible. Run twice with one learner:

| pass | right | honest | first move worked | moves tried |
|---|---|---|---|---|
| first | 28/28 | 28/28 | 20/28 | 38 |
| second | 28/28 | 28/28 | 22/28 | 34 |

The first run of the first version was 24/28, and the four misses were
each a real flaw, now fixed and tested:

- **A trivial design passed.** The zero polynomial has every root and
  whole coefficients; it is no longer a design.
- **A failed form was proposed again** — eight times, when its failure
  taught nothing yet. It is now taken off the table for that draft.
- **A condition that is only checked could not be met**, because nothing
  chose a free unknown to satisfy it. Free unknowns are now searched.

What it learned, from nothing but its own attempts:

- `fit-geometric requires even-ratios` — true.
- `fit-arithmetic is blocked by even-ratios` — true of everything in the
  bank; a constant sequence is the exception it has not met.
- `fit-roots is blocked by degree-given` — **false**, drawn from one
  failure (a cubic whose roots left too many conditions for a scale). It
  is the learner's known weakness: a lesson that stops a move being tried
  also stops the success that would refute it from happening. Section 6
  lists what to do about it.

## 5. The learned proposer (`generating.py`, `proposer.py`)

**Taught from its own work.** `generating.py` makes an object — a
polynomial from roots and a scale, a geometric sequence, a quadratic in n —
and states true clauses of it: some roots, some values and slopes, a
turning point, some terms, a total. That is a specification the object
certainly meets, and nobody had to write it. Each is labelled by fitting
every form to it (exact) and naming the **simplest** that fits: the fewest
unknowns. *Fits* alone was tried first and taught the wrong thing — the
polynomial form fits any finite data, so a proposer trained on it reached
for a cubic where 2·3ⁿ⁻¹ was wanted. Fewest unknowns is Occam's razor said
exactly.

**The model is small on purpose**: one logistic regression per form over
the specification's features and how many of each clause it gives, a few
seconds to train on 3,000 goals (`llm/designer-proposer`, rebuilt by
`regenerate.py`'s `designer-proposer`). The form and the check are exact;
all it has to learn is which proposal suits which goal.

| where | ordering | first form fits | first form is the simplest |
|---|---|---|---|
| 600 generated goals, a seed it was not trained on | table | 396 | 396 |
| | **proposer** | **571** | **556** |

On the hand-written bank, which it never saw and which was not written
from the generator:

| run | right | first move worked | simplest design | moves tried |
|---|---|---|---|---|
| table order | 28/28 | 20/28 | 26/28 | 38 |
| table order, learner's second pass | 28/28 | 22/28 | 26/28 | 34 |
| **proposer** | 28/28 | **24/28** | **28/28** | 35 |
| proposer + learner, second pass | 28/28 | 24/28 | 28/28 | 33 |

Three of the 28 are impossible, so no first move can work there: 24/28 is
24 of the 25 that can be designed. The one miss is *a root at 1, value 2 at
0 and 6 at 2*, where it tried a scale times the root first and the scale
could not meet two values; the next form did.

**The generator found the designer's blind spots, too.** Every goal it
makes is possible, so a goal no form meets is a gap, not a hard problem:
9 of 600 at first, and every one involved slopes or turning points. A
general form sized by counting conditions gets these wrong — a condition on
the slope says nothing of the constant, so two slopes need a quadratic, not
the line two conditions suggest. Forms that can grow are now tried a degree
or two bigger when the least fails, and the gap is **0 of 600**.

## 6. Exploration: a lesson is not the last word

A lesson drawn from one failure can be wrong, and a lesson that stops a
form being tried stops the success that would refute it. So a form a lesson
holds back is tried anyway **out of curiosity** (first, 15% of the time)
and **as a last resort** (before a goal is called impossible), and a
success refutes the lesson on the spot (`Learner.worked`).

Measured over 400 generated goals with one learner (`bank.py --stream`):

| curiosity | right | moves | lessons held | false lessons |
|---|---|---|---|---|
| none | 400 | 485 | 4 | **3** |
| 0.15 | 400 | 554 | 2 | **0** |

Without it, three of four lessons were false and would have stayed false —
including `fit-geometric requires even-ratios`, which the 28-goal bank had
called true: with two terms, any ratio fits. Curiosity cleared all three
for 14% more moves.

## 7. Goals said in English (`stating.py`, `reading.py`, `page.py`)

The same method as v692's encoder corpus, applied to specifications. The
generator makes a goal from an object and **says it** as a person asks for
one — *find a quadratic whose roots are 2 and minus 3 and whose value at 0
is 12* — every word labelled with its part: the kind (`DKIND`), the word
naming a clause (`CLAUSE`, labelled `root`, `value`, `term`, `start`,
`through`, `monic`, `rate`…), and the values (`AT`, `IS`, `MULT`). The
encoder learns those labels. Putting the parts back together
(`reading.spec_of`) looks at no word: a clause word opens a clause, the
values after it are its values, and a place or a multiplicity right before
a clause word is that clause's (*the 5th term*, *a double root*). Every
record said is read back from its own labels, and kept only when that
gives the specification it was said for.

v693 registers into v692 rather than editing it: `reading.contributes`
(how a design goal's parts are put together) and `doing.contributes` (how
it is done). v690's page imports `v693.page`, and a design goal is then an
act of v692's like any other.

**Three training runs, and what each taught.** The first encoder read the
act right for all 29 English goals (the 5 that are not designs left alone)
and 25 exactly. The four misses shared one cause: a negative number typed
as one word, `-3`, was labelled as its minus sign alone, because the
generator only ever said `minus 3` or `- 3`. Numbers are now sometimes said
as typed, and points as `(0, -2)` with a space. The second run fixed all
four — and broke v692: its bank fell from 98 to 91. The design corpus's
near misses had included maths questions (*solve x^2 - 5x + 6 = 0*)
labelled `none`, and in this task `none` means *no mathematics at all*. They
are out; v692's corpus teaches those acts. The third run fixed v692 and
brought the typed negatives back: MiniLM's tokenizer splits `-3` into the
same pieces as `- 3`, and a word is labelled by its first piece -- so a
typed negative *could not* be told from a spoken minus, and teaching both
only set the two corpora against each other. The fix is in splitting, not
in the model: `words` now splits a minus typed against a number into a word
of its own, as the tokenizer already did, and a word that stands for itself
stands for itself in every piece. The fourth run is the one kept (§10).

## 8. Composition: forms made from forms (`forms.composed`)

A form can be the sum of two, and nothing lists which: every pair of a
kind's forms is summed, except a form built from the specification's own
data (roots, a given derivative), a growing form (a polynomial plus
anything is found by the polynomial growing), a form linear in its unknowns
summed with itself (worked out, not listed: a line plus a line is a line),
and sums of more than five unknowns (sympy's solve of such a system takes
longer than everything else). Occam's rule then prefers a composite exactly
when it says more with less: 2ⁿ + 1 for 3, 5, 9, 17, 33 (four unknowns)
rather than the polynomial through five points (five). Given 1, 1, 2, 3, 5,
8, the designer composed two geometric sequences and came back with
**Binet's formula**.

## 9. Functions (`spec.py`, `forms.py`)

A third kind: functions of x, with clauses a function can meet — its value
and slope somewhere, **its derivative** (`whose derivative is 2x cos(x²)`)
and **an equation it satisfies** (`f'' + f = 0`). Forms: a power series (a
polynomial that grows), an exponential, an oscillation (a sine and a cosine
of one frequency), and the antiderivative — the integral of the derivative
asked for, plus a constant: a design move made of calculus. Composites
follow by §8: two exponentials, an exponential and an oscillation.

A clause that must hold for every x becomes equations by taking it and its
derivatives **at 0**, not at points: e^(kx) and sin(wx) there are
polynomials in k and w, which sympy solves, where at x = 1 they are e^k and
sin(w), which it can search for as long as it likes. Equations are then
solved in stages — the polynomial ones first, the rest with what they fixed
put in — and each clause gives its equations in turn, so a composite solved
from its first few equations is solved from every clause.

Designed: sin(x²) + 1 from its derivative and a value; cos x from
f'' = −f, f(0) = 1, f'(0) = 0; 2e³ˣ from f' = 3f; eˣ + e²ˣ from
f'' − 3f' + 2f = 0 by composing two exponentials; sin 2x from f'' = −4f.

## 10. Numbers after the roadmap

**Reading** (`llm/reader-design4`, now the runtime default):

| | |
|---|---|
| design goals asked in English, hand-written: 24 goals, 5 functions, 5 that are not designs | act read 34/34, understood exactly **34/34**, designed right **34/34** |
| v692's maths bank, 98 | **98/98**: the design data costs mathematics nothing |
| the suite, 1,385 tests | 1,384; the one failure is the compound-sentence test every maths encoder fails (v692 §9) |
| the common-sense probe, 114 | 2 moved (the previous maths encoder moved 4) |

**Designing** (`bank.py`, 41 goals: 16 polynomials, 17 sequences with 5
composites, 8 functions; 3 impossible):

| run | right | first move worked | simplest design | moves |
|---|---|---|---|---|
| table order | 41/41 | 20 | 40 | 92 |
| table order, learner's second pass | 41/41 | 22 | 40 | 84 |
| **proposer** | 41/41 | **35** | 40 | 73 |

Of 600 generated goals it was not trained on, the proposer's first choice
is the simplest fitting form 525 times, against 295 in table order.

**What the proposer cannot do yet, and what does it instead.** It does not
put composites first. Of 400 generated sequence goals only about ten have a
composite as their simplest form (a sum needs five or more terms to beat
the polynomial through them), and no feature of the data describes every
sum: 2ⁿ + 1 has steps one ratio apart, 3·2ⁿ⁻¹ + n has nothing a line or a
ratio test sees. So the rule is at design time instead: a design with an
unknown for every condition only repeats the data it was given, and is not
settled for while a form with fewer unknowns is untried (`_simpler`). It
costs extra moves on exactly those goals (73 against 55 without it), and it
is why the simplest design is used 40 times in 41, all five composites
included.

**On the page**, over HTTP: *find a polynomial with roots 2 and -3 whose
value at 0 is 12*: "The polynomial is minus two times (x minus two) times
(x plus three)". *Find a quadratic through (1, 2), (2, 5) and (3, 10)*:
"The quadratic is x squared plus one". *Find a function whose second
derivative is minus 4 times itself, with value 0 at 0 and slope 2 at 0*:
"The function is sine of the quantity two x". *Give me a sequence that
starts 1, 1, 2, 3, 5, 8*: Binet's formula, said as written, because a value
that long is beyond what the decoder was taught to say; it dropped half of
it the one time it tried, and the check could not read it back to catch
it. *Is there a quadratic with roots 1, 2 and 3*: it could not design one,
and says which forms it tried.

## 11. What is next

- **Proofs.** Everything here is designed and checked; nothing is proved.
  Lemmas as design goals and induction as a move are the next kind of
  search, and the hardest.
- **Richer composition**: products and nesting (a quadratic whose roots
  are themselves designed), and specifications split into sub-goals by
  means-ends.
- **Shapes and optimisation**: a rectangle of area 24 with the least
  perimeter — a clause that is a minimum, not an equation.
