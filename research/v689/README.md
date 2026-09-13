# V689 — who is who, and when

v688 answers questions about kinds: *does a beagle swim*. A conversation is
about individuals — *the* beagle, *that* one, *the second* one, *it*, *me* —
and nothing in the words names which. v689 keeps an **episodic memory** of
them, reasons over it with **v687's own rules**, and stores it in **the paper's
trie**, grown live as the conversation goes.

It is also about *when*: what happened yesterday, what came first, whether the
vase was broken before the cat broke it. Memory is kept as **events** — every
change appended to a stream, every table a fold of it — and story time is one
more projection of that stream: **episodes**, **occurrences** stored in a trie
of their own and identified the way individuals are, and six rules, T1 to T6
(below).

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

## Memory as events

Nothing writes a table. A change is an **event** appended to a **stream** and
never altered (`events.py`), and every table is a **projection** of the
stream: the episodic tables, the attention table, the timeline and the tries
planned over them. The model is domain-driven design's, made explicit:

| bounded context | aggregate | stream | projections |
| --- | --- | --- | --- |
| conversation | `Session` | the conversation's own | episodic tables, salience, timeline, both tries |
| knowledge | `Knowledge` | `knowledge`, shared (an example keeps its own) | kinds, edges, norms |

**Commands decide; handlers apply.** `tell`, `introduce`, E2's `withdraw` may
walk v687 or ask v688 before deciding, and what they decided is recorded as an
event. A handler asks nothing of anyone, so replaying a stream is
deterministic: E2's verdict on the airplane is in the log as `withdrawn`, not
re-asked of a v688 that may answer differently after a restart. A conversation
comes back by replaying its stream, and the test suite replays every page
example and checks it rebuilds exactly what the live conversation held.

An event about an individual goes to the conversation's stream; an event about
a kind goes to knowledge's, by the same test `Layered` routes a table write by.

**Two times.** A stream's order is *telling time* — what `what did I tell you
first` asks — and it is not *story time*: `the dog slept; before that it had
barked` was told sleep-first and happened bark-first. Story time is its own
projection.

## Time and events

An **episode** is a stretch of the story at one time: `yesterday`, `this
morning`, `now`, `tomorrow`, each on one line of days, and `then` — the past a
story is told in when no day is named, known only to be before now. A
statement lands in the episode it names, in the present if it is in the present
tense, and in the past the story is already in if it is in the past. The same
dog is in every episode; what happens to it, and what it is like at the time,
is in one.

An **occurrence** is something that happened, kept as an individual is. It is
an individual of its verb, as Davidson's event is: stored under every verb it
is a kind of (`is_a chase`, `is_a pursue`, `is_a travel`, WordNet's troponymy,
walked up as R1 walks a beagle up to dog), under who took part, its episode
and its aspect, planned by `adaptive_coverage` into a trie of its own, and
found by walking that trie down — the same `walk_down` that finds `the black
one`.

Reading comes first (`tense.py`): a time word is taken out before the rest is
read, so `yesterday there was a dog` is an introduction and not a question
about a kind called yesterday. Frames (`yesterday`, `now`) come off either
end; links (`then`, `before that`, `meanwhile`, `finally`) off the start;
`again` off the end; and an anchor clause (`after the dog chased the cat, it
slept`) is split off only when it reads as a statement about someone. Tense
and aspect are Reichenbach's, from the auxiliary and the verb's form: a simple
past, a progressive, a perfect or a future is an occurrence; a habit (`it
barks`), an ability (`it can swim`) and a state (`it was hungry`) are not.

| rule | says | so |
| --- | --- | --- |
| **T1** told in order, happened in order | a simple past moves the story on; a progressive is in progress at the time the story is at; a past perfect is before it; a link or an anchor says otherwise | `it slept. before that, it had barked` puts the bark first, and the `ordered` event says why |
| **T2** before is a partial order, walked like the taxonomy | transitive and asymmetric: yes when one is reached from the other along `before`, no when the other way, and not told when neither — absent, not false. Episodes on the line of days are in order without being walked | `did the dog bark before it slept` |
| **T3** a state holds within its episode until something ends it | what it is like, where it is and what it is doing were told of a time; the latest before the moment asked answers, and nothing told of one episode answers another. What it is, is called, owns and can do are in no episode | `yesterday the pig was in an airplane`: `is the pig in an airplane` today is not told, and the reply says when it was |
| **T4** what an occurrence changes, from VerbNet | a frame's `result(E)`/`end(E)` is what holds after, `start(E)` what held before (`change.py`) | `the cat broke the vase`: broken after, not before; `killed the mouse`: not alive after, alive before; `put the key in the drawer`: in it after, not before; `started` and `stopped barking` |
| **T5** an occurrence is an individual of its verb | who, when, what happened and how many times are identification on the occurrence trie with one slot read out | `did the dog move` finds a chase through `move`, a lemma of travel.v.01 |
| **T6** nothing happens by inheritance | what a kind does is a tendency; that one of them did it is an occurrence — the E1 of events | `did the dog bark`, with nothing told: not told, and v688's answer for dogs beside it |

**T4 is closed data.** VerbNet 3.3 writes each frame's meaning over the phases
of an event, so its axioms are already written out. A frame is read only when
its syntax has the sentence's shape — `the vase broke` meets `NP.Patient VERB`
and changes its subject, `the cat broke the vase` meets `NP.Agent VERB
NP.Patient` and changes its object — and only when its class is joined to the
verb's sense: each VerbNet member carries WordNet 3.0 sense keys, the store's
verb senses are WordNet 3.0 synsets, and a class is read only if they share one
of the first three, which keeps `kill` out of amuse-31.1 (`that joke killed
me`). A change of state is keyed by the verb, whose participle names it (a
broken vase), and kept only where WordNet has that participle or the verb as an
adjective; a state VerbNet names itself (`alive`) is keyed by its word. A
change of state entails the state did not hold before, so T4 can say no.

**E2 is within one episode.** Being in an airplane yesterday explains nothing
about flying today.

| asked | answered by |
| --- | --- |
| `what happened yesterday`, `what happened after the dog chased the cat`, `what happened to the vase`, `what did it do second` | story order, filtered by episode, by T2 against the anchor, by who took part, or by place in the order |
| `when did the dog chase the cat` | the episode, and what came just before and after |
| `how many times did the dog bark` | the occurrences under `is_a bark`, counted |
| `what was the dog doing when the cat ate` | what was in progress during that occurrence |
| `who chased the cat yesterday`, `what did the dog chase`, `where was the pig yesterday` | occurrences and states in that episode |
| `was the door open before i closed it` | T4 and T3 at the moment just before the occurrence |

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

**What is on disk is the events.** The `events` table is the record, one row
per event, and a conversation and the knowledge are replayed from it. The
knowledge rows, each conversation's snapshot and the turns as the page showed
them are read models kept beside it. Rows kept before there were events are
read once and recorded as the events that would have written them — a
snapshot as one `imported` event at the head of its stream. `start over`
deletes a conversation's stream, because it asks for the conversation to be
gone; `unlearn` appends `unlearned`, and the stream keeps what was unlearned.

On the page, **episodic memory** is this conversation's: its episodes in story
order, each with its occurrences, where T1 put them, what T4 changed and the
states told of that time; then who is who; then the event log. **Long-term
memory** is what every conversation shares: the kinds, taxonomy and norms
taught.

## Several claims at once, and opposites

A statement is split into its claims by spaCy's dependency parse
(`clauses.py`), not by the word `and`: the parse says which verbs are
coordinated and what each one's subject is. Each clause is filled in from the
one before it and read as a sentence of its own:

| said | read as |
| --- | --- |
| `testicles shrink in cold temperatures, and they expand in warm ones` | `testicles shrink in cold temperatures` · `testicles expand in warm temperatures` |
| `beagles can't swim but they can run` | `beagles can not swim` · `beagles can run` |
| `there is a beagle and it can't swim` | `there is a beagle` · `it can not swim` |
| `dogs eat meat and bones` | one claim: `bones` hangs off `meat`, not off the verb |

Where the parse has no verb at its root -- spaCy reads `dogs bark and cats
purr` as a noun phrase -- nothing can be split, and if the words still look
like more than one claim they are refused rather than stored as one.

A question that no told or taught fact answers exactly is compared with the
ones that come close. The same predicate with its head word replaced by a
WordNet antonym answers **no**: taught `testicles shrink in cold
temperatures`, `do testicles expand in cold temperatures` is denied, because
doing one under the same condition is not doing the other. An antonym
anywhere else is a different condition and answers nothing. Anything sharing
a word is quoted beside the answer, so `do testicles shrink` says what was
taught even though the qualification keeps it from being a yes.

## Definitions memory

WordNet has a gloss for every noun -- 82,115 of them in the store -- and a
gloss is a definition: a broader kind and what sets this one apart.
`definitions.py` reads each one into facts about the kind it defines, kept in
`state/v689-definitions.sqlite` beside long-term memory and never written into
the store:

| gloss | read as |
| --- | --- |
| `young domestic cat` | a kind of cat; is young; is domestic |
| `one of the two male reproductive glands that produce spermatozoa and secrete androgens` | a kind of gland; is male; is reproductive; can produce spermatozoa; can secrete androgens |
| `a hand tool with a heavy rigid head and a handle; used to deliver an impulsive force by striking` | a kind of hand tool; has a heavy rigid head; has a handle; is used to deliver impulsive force by striking |
| `feline mammal usually having thick soft fur and no ability to roar` | a kind of mammal; is feline; has thick soft fur; cannot roar |

A gloss is a noun phrase with fragments after semicolons, not a sentence, and
spaCy parses it badly whole. So each piece is parsed on its own inside a frame
it handles -- `it is a <piece>`, `it has ...`, `it is used to ...` -- and the
genus is checked against the store taxonomy: a sense of it among the concept
ancestors means the parse found the head it should have. Instances are not
kinds (WordNet says which), names are not properties, alternatives are not
properties (`red or yellow skin` is skin), and verbs joined by `or` share the
object only the last one carries.

**How it is filled.** Every definition a v688 run retrieves -- `what is a
testicle` retrieves five, walking up to organ -- is read as it arrives, and
each fact is put to the teacher as a bare question. In bulk,
`learn_definitions.py` reads every noun gloss (about six minutes on four
processes), asks the teacher about every fact, and reports what it read. A
fact the teacher disputes is kept, so it can be counted, and never read.

**How it is used.** The rules read defined facts after what was told and
before the store rows, so `is a kitten young` is answered from the definition
and says so. `what is a kitten` is retrieved through v688 once and answered
from memory after that. And a definition is not a tendency: told `it is old`
of a kitten, the fact is kept as said and the answer says the definition of
kitten rules it out.

**Measured** (`research/v688/AUDIT.md` section 28). Loaded into a copy of the
store as one more source, the 69,235 defined facts lift coverage by about half
a point in both audit configurations and pair accuracy with it: every gold
answer that rested on a definition was right, and every over-affirmation
measure -- corrupted claims, screened denials, contradicted -- is unchanged.
So the teacher check was not run: there was nothing on any measure for it to
catch. Keeping only glosses whose genus agrees with the taxonomy is as safe and
gains less, so every fact is read.

**More sources** (section 29), each its own definitions memory so it can be
measured and dropped apart from the others:

- **Open English WordNet 2025**, joined to the store's WordNet 3.0 through
  sense keys: the 1,550 noun definitions it rewrote are re-read into
  `state/v689-definitions.sqlite`. No audit measure moves; no gold item is
  about them.
- **Wikipedia lead paragraphs** (`articles.py`, `state/v689-articles.sqlite`):
  a sentence is read only when its subject is the kind itself, unquantified,
  unhedged and present tense, through the same claim reading a person's
  sentence goes through. 485 articles gave 803 facts: 0.15 points of coverage
  and about 0.25 of pair accuracy, no cost measured, though the gold reaches
  only four of the facts.
- **Wiktionary** (`learn_wiktionary.py`, `state/v689-wiktionary.sqlite`): a
  sense is written about a word, not a synset, so it is kept only when the
  broader kind its gloss names sits above exactly one of the word's noun
  synsets in the taxonomy, and no other sense of the word lands there too.
  107,247 senses read, 21,941 synsets matched, 16,918 facts WordNet's glosses
  did not give; 0.29 points of coverage and about 0.13 of pair accuracy, no
  cost measured, with 13 gold answers resting on it and about one fact in five
  wrong by hand.

Neither Wikipedia nor Wiktionary is read by the page yet: v689 reads WordNet's
definitions memory, and the other two are loaded into store copies for the
audit.

## People, places and things

bAbI's twenty tasks (`babi.py`, `BABI.md`) were why, and nothing of bAbI is in
the code: its stories are told in the language below, and every row is a
general rule those stories happen to need.

| said | what it is now |
|---|---|
| `Mary moved to the bathroom` | someone new: a person called Mary, by a capitalised proper noun (or a noun the ontology lacks, `Sumit`); she is in the bathroom after it (T4) |
| `Mary and Daniel went to the kitchen` | told of each, as happening together; `they` is the two of them |
| `then she went to the garden` | the woman last talked about: NLTK's names say whose name is whose, and a person comes before a room |
| `Mary is no longer in the bedroom`, `Fred is either in the school or the park` | a place denied; one place of two, and `maybe` |
| `Mary got the football`, `dropped it`, `gave it to Fred` | VerbNet's `ch_of_poss`, kept as a place: what someone has is where they are, and goes where they go |
| `the kitchen is north of the office`, `the box fits inside the chest` | a relation between two individuals along one dimension (`relations.py`) |
| `mice are afraid of wolves`, `Gertrude is a mouse` | a quality toward something, which descends |
| `Sumit is tired` | a state the store's motivations connect to what one does, and where (`motives.py`) |

The rules added:

- **T3: one place at a time.** Being told somewhere else, or going there,
  ends being here. A story told in two tenses with no day named for either is
  one story: `Mary is in the garden. Daniel went to the kitchen.`
- **T4 reads possession.** VerbNet's `ch_of_poss`, with its `equals` for the
  role a sentence leaves out: the Goal of getting is the Agent. A Theme that
  leaves an unexpressed place (`drop`) or stops touching the Agent (`discard`)
  is let go of, and is where its holder was. A particle verb VerbNet has no
  frame for is read as its head (`put down`, `pick up`). When none of a verb's
  first three senses has a frame of the sentence's shape, any sense that does
  is read: `pass the football to Bill` is giving. What is left, the story
  chooses: a reading whose starting state the story contradicts is dropped --
  `John left the apple`, of a John who has it, is not leaving a place.
- **E1 is about what a thing is like.** `afraid of wolves` is how a mouse is
  toward something else, and descends as what it does does (R3).
- **S1 to S4.** A relation and its converse are one fact; along a dimension,
  a partial order walked like `before`; a direction puts the two in line; the
  compass is a map (`relations.py`).
- **I1: induction.** Nothing told of one individual's colour: the others of
  its kind told of here, all of them where they agree and the last told where
  they do not, said as `probably`.
- **Motives.** A state moves one to what the store says it motivates, needs
  or leads to; a place is for what its kind is `used_for`; the two meet on a
  lemma. `why did Sumit go to the bedroom` is a state told before it that
  meets the place, or `perhaps` the last one; `where will Sumit go` names only
  a place the conversation has been to, and otherwise says not told.

| asked | answered by |
|---|---|
| `where is the football` | where it is, and who has it: `the garden, with Mary` |
| `where was Julie before the school` | the place before she last came to be there, in story order across days (T2) |
| `what is Mary carrying`, `how many objects is Mary carrying` | what is with her -- where VerbNet reads the verb as having something with you (carry-11.4, hold-15.1, keep-15.2) -- listed or counted |
| `who gave the football`, `who received the football`, `who did Fred give it to` | the last occurrence (T1); receiving is ending up with it (obtain-13.5.2), however it got there |
| `what is north of the office`, `how do you go from the kitchen to the garden` | S1; S4 |
| `is the box bigger than the chocolate`, `is the rectangle right of the square` | S2; S3 |
| `what is Gertrude afraid of`, `what color is Greg`, `why did Sumit get the pajamas` | R3 through a taught kind; I1; motives |

Descriptions changed with them: `the blue square`, said first, introduces a
blue square; `the box of chocolates` is a thing of its own and not a box; and
`the container` is the one introduced as a container, not the box that is a
kind of one.

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
- **Time is episodes and order, not clocks.** No durations, dates or times of
  day, and `when` is answered with an episode and its neighbours.
- **Kinds have no tense.** `did dinosaurs fly` is still v688's present-tense
  question about dinosaurs.
- **Nothing told carries across an episode**, a colour included: told
  `yesterday the dog was black`, `is the dog black` today says when it was
  told rather than yes. Which qualities last is not in the data.
- **What an occurrence changes is VerbNet's alone** (T4). `the dog ate` does
  not end its hunger, and a verb VerbNet lacks changes nothing.
- **An anchor clause names at most one occurrence**, the latest told that fits.

## Files

| file | what it does |
| --- | --- |
| `reading.py` | which words pick out an individual, and what the utterance does |
| `events.py` | events, streams and the log: memory as what happened to it |
| `episodic.py` | episodic memory, `EpisodicReasoner` and E1, the live trie and identification, all applied from events |
| `tense.py` | when: frames, links, anchors, tense and aspect, taken out before reading |
| `timeline.py` | story time: episodes, occurrences and their trie, T1 to T3 and T5 |
| `change.py` | T4: VerbNet's event structure, by frame and by sense |
| `story.py` | what a conversation does with time: statements placed, questions about when answered, T6; where things are and who has them |
| `relations.py` | S1 to S4: relations between two individuals along a dimension, their order, and the compass as a map |
| `motives.py` | the store's motivations: what a state moves one to, and what a place is for |
| `discourse.py` | attention: salience, order, focus, names, groups, and resolving a phrase to one individual |
| `goals.py` + `test_goals.py` | what a question asks, as slots -- asked, relation, who, subject, object, verb -- and the operators that answer by composing memory: who or what is at a place, what is with whom, where an occurrence went (`v690/DESIGN.md` P5) |
| `session.py` | one conversation: told facts into memory, questions to v687's walk, the kind to v688 |
| `asker.py` | what a session needs from v687 |
| `clauses.py` | a statement split into its claims by the dependency parse |
| `definitions.py` | glosses read into facts, and definitions memory |
| `learn_definitions.py` | every noun gloss read, checked by the teacher, and reported, offline |
| `articles.py` | Wikipedia lead paragraphs read into facts about the kind, offline |
| `learn_wiktionary.py` | Wiktionary's noun senses matched to one synset each and read, offline |
| `definition_guards.py` | class-level questions v687 must not get wrong, asked of each store copy |
| `longterm.py` | the knowledge every conversation shares, and every conversation, kept on disk |
| `server.py` + `app.html` | the page, over v688's `Service` |
| `test_v689.py` | v687's real reasoner and parser over a nine-concept store built in the test |
| `test_time.py` | episodes, occurrences and T1 to T6 over a store with verbs, and replaying the timeline |
| `test_people.py` | names, groups and pronouns, one place at a time, and possession, over a store with WordNet's own verb senses |
| `test_relations.py` | S1 to S4, fed events by hand and then in conversation |
| `test_kinds.py` | deduction through a taught kind, induction (I1), and motives |
| `babi.py` + `test_babi.py` | bAbI put to v689 and to SmolLM3 the same way, and the one scorer both are read by |
| `TIME_AND_EVENTS.md` | the audit: what could not be said about when, the architecture built for it, and what changed |
| `BABI.md` | the benchmark: the baseline, why v689 could read none of it, the rules built, and the scores beside SmolLM3's |
