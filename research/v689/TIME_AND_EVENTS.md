# Time and events

2026-09-13, on `v689/time-and-events`. What the page could not say about
*when* -- what happened first, what happened yesterday, whether the vase was
broken before the cat broke it -- where each broke, the architecture built
for it, and what changed. The common-sense work before this one left time out
on purpose, because there was nothing to hold it in.

## Method

30 conversations, 158 turns, sent to the running page (`/api/say`) in the
example sandbox and forgotten afterwards, in these families: narrative order,
`then` / `before that` / `meanwhile` / `first` ... `finally`, an anchor clause
at the start and in the middle, `while` and `when`, episodes named by day,
`when did`, `who` on a day, `how many times`, breaking, killing, putting and
jumping, a place on one day and another, a quality now and this morning,
doing against being able to, what a kind does against what one of them did,
kinds of verb, the future, the past perfect, E2 across days, the flying pig
as it was, started and stopped, telling order against story order, kinds in
the past, and a state before a change. The probe and its diff are session
scripts (`probe_time.py`, `diff_time.py`); the run before any of this was
built is the baseline.

## What it came to

1. **Time words were read as part of what was said.** `yesterday there was a
   dog` went to v688 as a question about a kind called `yesterday`, so
   nothing after it had a dog to refer to. `it barked again` stored
   `capable_of "bark again"`; `earlier it had run` taught a kind called
   `early`; `it is empty now` stored the property `empty now`.
2. **Sequence and subordinate clauses were listings.** `then it slept`,
   `meanwhile the cat played`, `after the dog chased the cat, it slept` and
   `while the dog slept, the cat ate` came back as v688's listings about
   sleep, cats and dogs, and nothing was stored.
3. **Nothing had an order but the order it was told in.** `what happened
   first` and `what happened last` gave the same list; `did the dog sleep
   before it barked` asked v688 whether dogs sleep before it barked.
4. **Nothing happened by inheritance -- except that it did.** `did the dog
   bark`, with nothing told, was yes, because dogs bark; `did the dog swim`
   yes, because dogs swim.
5. **Changes changed nothing.** After `i closed the door`, `is the door open`
   was yes from `the door was open`; after `the cat killed the mouse`, `is
   the mouse alive` was v688's answer for mice.
6. **Past and present were one.** `is the pig flying` was yes from `he was
   flying`; `it had eaten` was `has_part "eaten"`.

## The architecture

### Memory as events

v689's memory was a set of tables that commands wrote. Nothing now writes a
table: a change is an event appended to a stream, and every table is a fold of
the stream (`events.py`). Domain-driven design names the parts:

| bounded context | aggregate | stream | projections |
| --- | --- | --- | --- |
| conversation | `Session` | the conversation's | episodic tables, salience, the timeline, both tries |
| knowledge | `Knowledge` | `knowledge`, shared; an example keeps its own | kinds, edges, norms |

Commands decide and handlers apply. A command -- `tell`, E2's `withdraw` --
may walk v687 or ask v688, and records what it decided; a handler asks nothing
of anyone. So a replay is deterministic, and the tests replay every page
example and compare what it rebuilds with what the live conversation held.
Conversations on disk come back by replay, and what was kept before there were
events is recorded once as the events that would have written it.

The stream's order is telling time. Story time is a projection of its own.

### Story time, merged with the trie

The trie and its heuristics were built for individuals: predicates as the
alphabet, `adaptive_coverage` ordering, identification by walking down, and
the rules walking the taxonomy up. Each part of time is one of those, applied
to a new kind of node:

| time needs | built as | the existing machinery it is |
| --- | --- | --- |
| something that happened | an **occurrence**: a node with `is_a <verb>` for every verb above it, `subject`, `object`, `place`, `episode`, `aspect` | an individual in the paper's trie, planned by `adaptive_coverage` |
| `did the dog move` after a chase | `is_a move` on the chase, through WordNet's troponymy | R1's closure: `the dog` finds a beagle |
| who, when, what happened, how many times | a description walked down the occurrence trie, one slot read out | `walk_down`, the identification that finds `the black one` |
| before and after | a partial order walked breadth-first | R1's walk along `is_a` |
| a time a story is at | an **episode** on a line of days | a kind an individual is placed under, for what is stage-level |
| a state at a moment | the latest record before it in its episode | R3's nearest-first: what is closest decides |
| nothing inherited | T6 | E1, for doings |

| rule | says |
| --- | --- |
| T1 | told in order, happened in order: a simple past moves the story on, a progressive is in progress at the time it is at, a past perfect is before it; links and anchors say otherwise |
| T2 | before is a partial order, walked like the taxonomy: yes, no, or not told |
| T3 | a state holds within its episode until something ends it; nothing told of one episode answers another |
| T4 | what an occurrence changes, from VerbNet's event structure |
| T5 | an occurrence is an individual of its verb |
| T6 | nothing happens by inheritance |

## The data

| dataset | what it is | taken | why |
| --- | --- | --- | --- |
| VerbNet 3.3 | 331 class files: frames with syntax by role, and semantics over `start(E)`, `during(E)`, `end(E)`, `result(E)` | already in `data/verbnet3.3`; now read for T4 | event calculus with the axioms written out, closed and curated, so a change of state can say what did not hold before |
| WordNet 3.0 verb taxonomy | 13,238 verb hypernym edges in the store | already there | troponymy: chasing is pursuing is travelling |
| WordNet via NLTK | sense keys, adjectives, antonyms | already used for antonyms | joins VerbNet members to the store's verb senses; `open`/`closed`, `alive`/`dead` |
| ATOMIC 2020 | if-then knowledge about everyday events | not taken | free text about kinds of event (`PersonX bakes a cake`), where v689 needed episodes and order about individuals first; the candidate for what an occurrence causes and needs |
| TimeBank, MC-TACO | temporal relations and durations in text | not taken | annotated prose and QA, not knowledge a store can hold |

**T4's joins, measured before use.** Of VerbNet's 601 classes and subclasses,
293 say something at `start`, `end` or `result`; 442 predicates change a
location and 377 a state. The store's verb ids are WordNet 3.0 synsets, and
VerbNet members carry WordNet 3.0 sense keys, so a class is read only when
one of its member's senses is among the first three the store ranks for the
verb: `kill` keeps murder-42.1 and drops amuse-31.1 and pain-40.8.1, `break`
keeps break-45.1, `run` keeps run-51.3.2. A verb-keyed state is kept only
where WordNet has its participle or the verb as an adjective, which drops
`slept` (snooze-40.4) and keeps `broken`.

## Wrong answers found on the way

Each was caught by the probe, the suite or a scratch run, and fixed before it
stood:

- **`did the dog swim` was yes after `tomorrow the dog will swim`.** A past
  question read every episode when none was past. `did` now reads what has
  happened and never what is still to.
- **`there is a dog. it barked.` was two episodes.** An introduction opened
  the present and the bark opened the past, so `what happened` split one
  story in two. An introduction places the story only when it names a time.
- **`does it fly` lost `it wasn't flying`.** T3's filter hid what was told of
  another episode from a habit, which is asked of no moment. It applies only
  to questions that are themselves about a time.
- **`is the pig in an airplane` put an airplane on the table.** A question
  bound its object as a statement does. A question never introduces anyone.
- **`the dog is slept after it`.** snooze-40.4 changes a sleeper's state; a
  state keyed by a verb is kept only where WordNet has the participle as an
  adjective.
- **`earlier it had run` was `has_part "run"`**, because `run` is its own
  participle and the check compared spellings. The tag decides now.
- **A killed mouse would have been amused.** Every class `kill` is a member of
  was read, until the join by WordNet sense.

## What changed

Re-running the 30 conversations on the finished code changed 109 of 158 turns
(108 on the first run after the build; the 11 that moved between the two runs
are the fixes above for the future, `run` and `now`, and wording).

| family | before | after |
| --- | --- | --- |
| `yesterday there was a dog`, `today there is a bird` | a question about a kind called yesterday; `it` then had nothing to refer to | an introduction in the episode yesterday; `what happened yesterday`, `what happened today` answered apart |
| `then it slept`, `meanwhile the cat played`, `first it ate` ... `finally it slept` | v688's listings about sleep, cats and eating | occurrences, each placed by T1 with its reason |
| `after the dog chased the cat, it slept`, `the dog barked after the cat ran`, `while the dog slept, the cat ate` | listings, or `capable_of "bark after the cat ran"` | the anchor told first, the main clause placed against it |
| `what happened first` / `last`, `what did the cat do second` | the same list, in the order told | story order: `before that, it had barked` comes first |
| `did the dog sleep before it barked`, `did it bark before it slept` | v688 asked about dogs | no and yes, by T2, with the walk shown |
| `when did the dog chase the cat` | refused as a date | yesterday |
| `who chased the cat yesterday`, `who chased the dog` | listings about cats, and bears | the dog; the cat |
| `how many times did the dog bark` | refused as a count | 3 times, each quoted |
| `is the vase broken`, `was it broken before the cat broke it` | E1's unknown | yes; no (T4, break-45.1) |
| `is the mouse alive`, `is it dead`, `was it alive before` | E1's unknown | no; yes; yes (T4, murder-42.1) |
| `where is the key` after `i put the key in the drawer`; `was it there before` | nothing said about where | the drawer; no (T4, put-9.1) |
| `is the door open` after `i closed the door` | **yes**, from `the door was open` | no; and `was it open before i closed it` yes (T3) |
| `is the dog barking` after `started` and `stopped barking` | **yes**, dogs bark | no (T4, stop-55.4) |
| `is the pig in an airplane` today, after one yesterday | v688 on pigs | not told of today, and when it was |
| `it is empty now` · `is it full` | `has_property "empty now"`; unknown | `empty`; no |
| `did the dog bark`, nothing told | **yes**, dogs bark | not told, T6, with v688's answer for dogs beside it |
| `did the dog move`, `did the dog pursue the cat` after a chase | v688 on dogs; unknown | yes, T5: chasing is a kind of both |
| `it had eaten` · `did the dog eat before it slept` | `has_part "eaten"`; unknown | an occurrence before the sleep; yes |
| `tomorrow the dog will swim` · `will the dog swim tomorrow` · `what will happen tomorrow` · `did the dog swim` | a listing; unknown; a listing; **yes**, dogs swim | an occurrence to come; yes; the swim; not told as having happened |
| `yesterday it was hungry` · `is it hungry now` | a question about a kind called yesterday; unknown | not told of now, and yesterday's hunger quoted |
| `can the pig fly` after `yesterday it was in an airplane`, `today it was flying` | v688 on pigs | yes: E2 is within one time |

The flying pig, the other page examples and every earlier test answer as
they did.

## Still open

- **Kinds have no tense.** `did dinosaurs fly` is v688's present-tense question
  about dinosaurs, and says yes.
- **What lasts.** Nothing told carries across an episode, a colour included;
  which qualities outlast a day is not in any source here.
- **What an occurrence causes or needs**, beyond VerbNet's states and places:
  `the dog ate` does not end its hunger. ATOMIC 2020 is the candidate.
- **Clock time and duration**: `how long did it sleep`, `at what time`.
- **The other side of a told occurrence**: `did the cat chase the dog`, after
  `the dog chased the cat`, is not told without saying the reverse was.
- **Unique things are individuals** (E1, from the common-sense audit): `will
  the sun rise tomorrow` asks about a sun introduced by `the`.
- **Plural and habitual occurrences**: `the dogs barked`, `it barks every
  morning`.
