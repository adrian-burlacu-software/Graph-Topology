# bAbI

2026-09-13, on `v689/babi`. Weston et al.'s twenty toy tasks
(`data/babi.SOURCE.md`) put to v689 and to SmolLM3-3B: what v689 could read
of them before, the rules built so it could, and the scores side by side --
with what those scores do and do not show.

## Method

The test set, 1,000 questions per task, 20,000 in all (`babi.py`).

- **v689** hears each story a line at a time in a conversation of its own,
  questions included, exactly as the page would: `Session.say`, v688 behind
  it, no teacher. Nothing is kept between stories.
- **SmolLM3-3B** gets every statement before the question, the question, and
  the form the answer takes (one word; yes or no; a count as a word; a list
  with commas; directions as n, s, e, w). Zero-shot, no thinking, greedy.
- **One scorer for both** (`predict`, tested in `test_babi.py`). Only the
  headline is read -- the first line, up to v689's ` — ` -- so neither system
  is scored on its explanation. A yes/no is the first word, a count the first
  number word, a list a set, a path the directions in order, a word the one
  word of the answer space not in the question. A listing is no answer; a
  short reply that only repeats the question's word is a wrong one. The answer
  space comes from the training and validation splits, never the test.
- `right`, `wrong` and `none` add up to every question. v689 abstains and the
  model almost never does, so the three are reported apart.

Rules were developed against the **training split** only (a scratch driver
over `babi.stories(task, "train")`), and the test split was run once at the
end.

## Before: v689 read none of it

2.4% right, 1.3% wrong, 96.3% no answer. 16,330 of the 20,000 questions went
to v688 as questions about kinds:

| asked | v689 said |
|---|---|
| `Where is John?` | `toilet is found in separate room, bar, bathroom, ...` -- *John* read as a kind the store half-knows |
| `Is John in the kitchen?` | `UNKNOWN (absent, not false)` |
| `How many objects is Mary carrying?` | `not answerable here: this asks for a count of parts or instances` |
| `What is emily afraid of?` | `NO_MATCH` |
| `Is the pink rectangle to the right of the red square?` | `nothing here was said to be pink` |
| `How do you go from the bathroom to the hallway?` | `not answerable here: this asks for a method or a procedure` |

There were no people, no possession, no relation between two individuals,
and a proper noun was a kind. The few right answers were yes/no questions
whose answer happened to be no; qa16's wrong ones were colours inherited from
the wrong swan.

## What was built

The rules are in the README (*People, places and things*); nothing of bAbI is
in the code. In short: names by use and NLTK's gender for pronouns; groups and
`they`; T3 as one place at a time, with `either` as maybe and `no longer` as a
denial; T4 reading possession from VerbNet's `ch_of_poss`, choosing a sense
the story does not contradict; S1–S4, relations along a dimension
(`relations.py`); E1 letting a quality toward something descend; I1
induction; and motives from ConceptNet (`motives.py`).

## After

| task | v689 before right | v689 right | v689 wrong | v689 none | SmolLM3 right | SmolLM3 wrong | SmolLM3 none |
|---|---|---|---|---|---|---|---|
| 1 single supporting fact | 0.0% | 100.0% | 0.0% | 0.0% | 73.8% | 25.5% | 0.7% |
| 2 two supporting facts | 0.0% | 100.0% | 0.0% | 0.0% | 45.4% | 49.1% | 5.5% |
| 3 three supporting facts | 0.0% | 100.0% | 0.0% | 0.0% | 24.6% | 73.2% | 2.2% |
| 4 two argument relations | 0.0% | 100.0% | 0.0% | 0.0% | 39.3% | 56.4% | 4.3% |
| 5 three argument relations | 0.0% | 99.4% | 0.6% | 0.0% | 79.2% | 20.7% | 0.1% |
| 6 yes/no questions | 0.0% | 100.0% | 0.0% | 0.0% | 81.7% | 18.3% | 0.0% |
| 7 counting | 0.0% | 100.0% | 0.0% | 0.0% | 43.7% | 56.3% | 0.0% |
| 8 lists/sets | 0.0% | 100.0% | 0.0% | 0.0% | 52.2% | 47.8% | 0.0% |
| 9 simple negation | 21.9% | 100.0% | 0.0% | 0.0% | 87.3% | 12.7% | 0.0% |
| 10 indefinite knowledge | 12.1% | 100.0% | 0.0% | 0.0% | 52.9% | 47.1% | 0.0% |
| 11 basic coreference | 0.0% | 100.0% | 0.0% | 0.0% | 49.9% | 50.0% | 0.1% |
| 12 conjunction | 0.0% | 100.0% | 0.0% | 0.0% | 58.2% | 41.6% | 0.2% |
| 13 compound coreference | 0.0% | 100.0% | 0.0% | 0.0% | 50.8% | 48.5% | 0.7% |
| 14 time reasoning | 0.0% | 100.0% | 0.0% | 0.0% | 57.2% | 37.1% | 5.7% |
| 15 basic deduction | 0.0% | 100.0% | 0.0% | 0.0% | 39.9% | 55.9% | 4.2% |
| 16 basic induction | 5.2% | 99.5% | 0.5% | 0.0% | 66.0% | 24.4% | 9.6% |
| 17 positional reasoning | 0.0% | 100.0% | 0.0% | 0.0% | 57.7% | 42.3% | 0.0% |
| 18 size reasoning | 8.7% | 93.5% | 0.4% | 6.1% | 91.0% | 9.0% | 0.0% |
| 19 path finding | 0.0% | 100.0% | 0.0% | 0.0% | 3.9% | 96.1% | 0.0% |
| 20 agent's motivations | 0.0% | 67.2% | 0.6% | 32.2% | 43.2% | 15.0% | 41.8% |
| **all** | **2.4%** | **98.0%** | **0.1%** | **1.9%** | **54.9%** | **41.3%** | **3.8%** |

v689 answers the 20,000 in 430 seconds of conversation (21 ms a question).
SmolLM3 took about 80 minutes on the GPU. When v689 answers it is wrong 0.1%
of the time; SmolLM3 is wrong on 43% of what it answers, and worst where the
answer is a chain -- a path (`s,s` came back `n,w`), where something was
before somewhere, how many things someone carries.

## What is still wrong

- **qa20, 322 not told.** `where will Jason go`, after `Jason is thirsty`: the
  store says thirst moves one to `beverage`, and boredom to `orgy`; nothing
  says a kitchen is where a beverage is, and only 69 of 339 training questions
  name the gold room before they are asked. This is knowledge the store lacks,
  and the honest answer is not told.
- **qa18, 61 not told, 4 wrong.** Size questions whose order is not found. In
  the stories looked at, the chain runs through a thing named by its kind --
  `the container`, when a suitcase and a box are both containers and neither
  was introduced as one -- or through a second box told apart from the first.
- **qa5, 6 wrong.** `who received the football` counts picking it up as
  receiving: VerbNet's get-13.5.1 ends with the Agent having it, which is
  what the rule asks. bAbI means received *from someone*.
- **qa16, 5 wrong.** Where the others of a kind disagree, I1 takes the last
  told (`probably white -- the last told, Brian, is white`), and in these five
  the answer was another one's colour. A recency rule is a guess; it is said
  as `probably`, but it is counted wrong.

Changed on purpose, and worth knowing: `yesterday the pig was in an airplane.
today the pig is in a field. is the pig in an airplane` is now **no** (T3,
one place at a time, `test_time.py`); it was not told.

## What these scores do and do not show

- **They show that the rules are right for the stories bAbI generates.**
  v689 answers every form bAbI asks, abstains where its knowledge ends, and
  almost never says something false.
- **They do not show that v689 reads English in general.** bAbI's training
  and test splits come from one template generator with one small vocabulary.
  Developing on training and scoring on test keeps the questions unseen, not
  the kind of sentence. At least one rule follows bAbI's convention rather
  than the language: S3 puts two things *in line* along a direction, so the
  square left of the triangle and the rectangle below it make the rectangle
  right of the square, which is how bAbI draws its shapes.
- **The comparison is not like for like.** v689's rules were written by
  someone who read the training stories; SmolLM3 saw no example. A few-shot
  baseline from the training split, and a benchmark neither was developed on,
  are what would make the gap mean more.
- Every new question form needed a reader and a session method of its own
  (eight readers, about ten methods). The knowledge generalised; the control
  did not. That is what v690 is for.

Other v689 behaviour did not move: the common-sense probe changed 0 of 114
answers and the time probe 2 of 158 turns, both the ones above.
