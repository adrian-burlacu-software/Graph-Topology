# Questions v689 cannot answer yet

2026-09-12. An audit of *kinds* of question, not of facts: what someone typing
into the v689 page asks that it does not answer, answers about something else,
or answers wrong -- and at which layer each one breaks.

## Method

203 conversations (271 turns), sent to the running page (`/api/say`) in the
example sandbox and forgotten afterwards; long-term memory was checked before
and after and holds nothing from them. They cover about 60 families of
question: every wh-word, quantifiers and connectives, comparison, number, time,
modality, counterfactuals, words about words, named entities, questions about
the conversation's individuals and events, follow-ups, meta-questions, and
malformed input. The 126 single questions were also asked of v688 directly
(`/api/run`), which returns the whole answer rather than the page's summary, so
a question answered below and lost on the page can be told apart from one
nothing answers.

Every classification below rests on what came back, and the cause named is
the code path that produced it.

## What it comes to

1. **Most of what v687 can say that is not yes or no never reaches the page.**
   Lists, profiles, identifications, comparisons, scripts, causes and kind
   counts are computed and then cut: the page is sent v688's summary, whose
   first line is `LISTING — what can a dog do`. About 25 probes are answered
   below and show nothing.
2. **There are no wh-questions about the conversation.** `who chased the
   cat`, `where is the dog`, `what did it do first`, `how many dogs are there`
   -- everything v689 was told is in episodic memory, and only yes/no, `what is
   it` and names are read against it. The rest goes to v688 as a question about
   kinds.
3. **v689's reader turns several openers into something else.** A why-question
   is *taught* (the kind `why` can `dog bark`); `how many legs does a spider
   have` teaches a kind called `how many leg`; `should`, `must`, `doesn't`,
   `if` and imperatives are passed on and come back re-asked as `what is a
   should a dog eat chocolate`.
4. **Constructions R18 exists to refuse slip past it and are answered from the
   part that is understandable**: `how big`, `how long`, `how often`, `which is
   heavier`, `another word for`, `two plus two`, conditionals without `what
   if`, and numbers or objects inside a yes/no question.
5. **Some basic questions come back wrong** (last section): `can a bird fly` is
   headlined NOT SUPPORTED, `is water wet` is a yes about the verb, `do cats eat
   mice` is a yes about meat.

## The catalogue

Each family: what was asked, what happens, and where it breaks.

### 1. Answered below, lost on the page

v688 answers these, with content. The page receives `SUMMARY_KEYS` (`server.py`
-- verdict, lines, trust) and the first line of the summary is the verdict and
the question; the content lives in the headline answer's payload, which is
dropped.

| family | examples | what v687 had |
| --- | --- | --- |
| what a kind does, has, is for, is made of, where it is | `what can a dog do`, `what does a dog have`, `what is a hammer made of`, `what is a key for`, `where does a penguin live`, `what does a dog want`, `what does a car need`, `what is a hammer good for` | a listing of up to 40 facts |
| backwards (R22) | `what has wings`, `what is made of wood`, `what causes fire`, `what does a plant need to grow` | 11–12 concepts each |
| identification (R16) | `what kind of dog has spots`, `what animal has a trunk`, `what kind of animal is a whale` | IDENTIFIED |
| profile (R17) | `tell me about dogs`, `what is a cat like` | 36 stated attributes, 1,514 inherited |
| contrast (R21) | `what do a dog and a cat have in common`, `the difference between a frog and a toad`, `what is similar to a dog` | shared properties and overlap |
| scripts and causes (R23) | `what happens when it rains`, `what happens when a dog barks`, `what explains a fire` | the script or ranked causes |
| counting kinds (R25) | `how many kinds of dog are there` | the count |
| a relation's other end (bridge) | `what does a dog's owner need` | 6 prerequisites of owner |

### 2. Questions about the conversation

v689 answers yes/no about an individual, `what is it`, and names. Everything
else about what it was told falls through to v688 as a question about kinds.
Episodic memory holds facts, not events: no agent and patient beyond one bound
object, no time, no order.

| family | example (after setting it up) | what happens |
| --- | --- | --- |
| who did it | `the dog chased the cat` · `who chased the cat` | kind listing |
| what was done to what | `what did the dog chase` | kind listing |
| where is it | `the dog is in the garden` · `where is the dog` | kind listing |
| what it is on / in | `the cat is on the mat` · `what is the cat on` | kind listing |
| which one | `a black dog`, `a brown dog` · `which dog is black` | AMBIGUOUS (identification over kinds) |
| how many | `a dog`, `another dog` · `how many dogs are there` | taught a kind `how many dog` |
| what kind | `a beagle` · `what kind of dog is it` | UNKNOWN |
| what it can do / has | `a beagle` · `what can it do`, `what does it have` | UNPARSED |
| what can't it do | `a pig` · `what can't it do` | kind listing |
| everything about it | `it is black` · `what do you know about it` | kind listing |
| what happened, in what order | `it barked`, `it slept` · `what happened`, `what did it do first` | kind listing |
| who owns it | `i have a beagle` · `who owns the beagle` | kind listing |
| its parts | `does it have a tail` · `is its tail long` | answered for tails in general |
| a kind with a modifier | `it is black` · `is it a black dog` | UNKNOWN |
| typical, similar | `a penguin` · `is it a typical bird`; `what is it similar to` | UNKNOWN |
| comparing two individuals | `a big dog`, `a small cat` · `is the dog bigger than the cat` | refused as a comparative |
| change over time | `it was full` · `it is empty now` · `is it full` | yes, from `it was full` |
| several at once | `there are two dogs` · `are the dogs black` | the first is read as a question and answered VERIFIED |
| identified by what happened | `the dog chased the cat` · `can the animal that was chased climb` | `which one — the dog or the cat?` |
| why, after an answer | `can it fly` → no · `why not` | UNKNOWN |

### 3. Misread on the page before any reasoning

| family | example | what happens | cause |
| --- | --- | --- | --- |
| why | `why does a dog bark`, `why can't a penguin fly` | **taught**: `capable_of “dog bark”` on a kind `why` | `generic_claim` takes the words before the auxiliary as a kind; `why` is not in `NOT_NAMES` and WordNet has `why.n.01`. v688 alone answers these with R23 |
| how many | `how many legs does a spider have` | taught a kind `how many leg` | same path; v688 alone refuses it by name |
| analogy | `wing is to bird as fin is to what` | taught `has_property` on wing | same path |
| modal and negative openers | `should a dog eat chocolate`, `must a bird have wings`, `doesn't a cat have fur`, `can't a dog swim` | re-asked as `what is a should …`, `does a must …`; `can't` reaches v687 unexpanded and `t` is scored as a property | `AUX` lacks should, must, may, might, shall and the `n't` forms |
| anything unparsed | `define a hammer`, `describe a cat`, `if a dog had wings could it fly`, `dog swim?`, `cna a dog swim`, `and a fish?` | headlined as `what is a <the utterance>` | v688's loop re-asks what v687 cannot parse in that shape, and the re-ask becomes the headline |
| expletive it | `is it safe to eat a mushroom`, `is it likely that a bird can fly` | `nothing has come up yet for “it” to refer to` | `it` is always a referent |
| embedded question | `do you know if a dog can swim` | **no**: asked `does a computer program know if a dog can swim` | `you` is the program |
| a kind named by description | `what is the largest animal` | `nothing here was said to be largest` | `what is the <X>` is read as a referent; v688 would refuse it by name |
| … which bypasses R18 | `what is the capital of france` | `a capital of france, under paris.n.01` | same path |
| pronouns for kinds | `owls can see at night` · `can they fly`; `does a bird have wings` · `can it fly` | UNKNOWN_WORD; nothing for `it` | referents are individuals only |
| plural or taught definitions | `what are kittens`; `a wemble is a kind of animal` · `what is a wemble` | UNKNOWN_WORD | R26 looks the word up as written; the define act does not read taught kinds |
| describing a taught kind | `wembles can fly` · `what can a wemble do` | UNPARSED | listings are v687's, over the store |

### 4. Routed to the wrong reasoning

The question is recognised as *something*, and a different question is
answered without saying so -- the failure R18 was written to stop.

| family | example | what is answered instead |
| --- | --- | --- |
| degree, size, speed | `how big is an elephant`, `how fast can a cheetah run` | a listing of the subject's properties |
| duration, weight, frequency | `how long do dogs live`, `how much does an elephant weigh`, `how often do cats sleep` | a listing |
| manner, procedure | `how does a bird fly`, `how do you make bread` | a listing |
| the value of an attribute | `what color is a banana`, `what shape is a ball` | identification over “color”: AMBIGUOUS |
| backwards within a class | `which animals can swim`, `what mammals lay eggs`, `list animals that fly` | identification AMBIGUOUS, or a listing of animal |
| the exceptions | `which birds cannot fly` | a listing of bird -- though `do all birds fly` names chicken, emu and penguin |
| the kind above | `what is a dog a kind of` | identification AMBIGUOUS (`what is a dog` answers it, through R26) |
| the kinds below, listed | `what kinds of dogs are there`, `name three birds` | NO_MATCH; a listing of bird. `how many kinds of dog are there` works |
| a thing's parts, listed | `what are the parts of a car` | a listing about the subject `part` |
| words, numbers | `what is another word for big`, `what is the plural of mouse`, `what is two plus two` | listings about `word`, `plural`, `two` |
| a comparative as a choice | `which is heavier, a feather or a brick` | identification AMBIGUOUS, not refused |
| a choice between kinds | `is a tomato a fruit or a vegetable` | `is_a` to the concept “fruit or a vegetable” |
| similarity as yes/no | `is a dolphin like a fish` | **VERIFIED**: the norms list “fish” for dolphin |
| evaluation | `is a dog a good pet` | `is_a` “good pet” |
| a question inside a request | `tell me whether a cat has fur` | a listing of what a cat has |
| existence | `are there birds that cannot fly`, `is there such a thing as a flying fish` | property “that cannot fly”; UNKNOWN_WORD |
| location as yes/no | `is a fish found in water` | property “found in water”: UNKNOWN |
| a bare plural class | `are penguins birds` | property “birds”: UNKNOWN |
| a two-word kind in the norms router | `can a fire truck fly`, `does a hot dog bark` | subject truck, target fire; subject dog, target hot |
| analogy (R24 exists) | `fins are to fish as what are to birds` | not recognised: UNKNOWN |

### 5. Structure scored one word at a time

| family | example | what happens |
| --- | --- | --- |
| a number in the question | `does a spider have eight legs`, `does a car have four wheels` | **yes**, on “has eight eyes” and “has four doors” (the page answers the car correctly, from its definition) |
| a relation to a second kind | `do cats eat mice`, `does a cow eat grass` | **yes**, on “eats meat” and “eats leaves”; the object is never checked |
| … with the slots swapped | `can a cat catch a mouse` | subject mouse, target cat |
| a condition | `could a pig fly if it had wings` | **no**: fly, if and wings scored separately, about a pig bed |
| but not | `does a cat have fur but not feathers` | `but` scored as a property |
| a modifier on the kind | `can a small dog swim` | answered about dogs |
| … that the store cannot narrow | `do black cats have fur`, `can a baby bird fly`, `can a dog that is old run`, `does the animal that barks have a tail` | UNRECORDED or UNKNOWN; the last is asked of animal |
| a quantity in the predicate | `can a dog run fifty miles` | UNKNOWN |
| tense | `did dinosaurs fly`; `will the sun rise tomorrow` | yes; `rise tomorrow` as a capability |
| a false premise | `why can a penguin fly`; `when did dogs learn to fly` | a script for flying; refused as a date. The premise is never checked |
| a negative quantifier over nothing | `can no fish walk` | UNRECORDED: 0 of 0 kinds recorded |

### 6. Refused by name, with nothing behind the refusal

R18 names these honestly. Listed because they are questions people ask, and
for some the ingredients are already here.

| construction | example | note |
| --- | --- | --- |
| comparative | `is an elephant bigger than a mouse`, `are cats better than dogs` | no magnitudes in the data |
| superlative | `what is the fastest bird` | same |
| count of parts | `how many legs does a spider have` | no numbers in the data |
| counterfactual | `what if a dog could fly` | |
| time | `when do bears hibernate`, `what year did dogs appear` | `when` is a season as often as a date |
| words | `what does bark mean`, `how do you spell dog` | WordNet lemmas and glosses are in the store |
| antonyms | `what is the opposite of hot` | v689's `Asker.antonyms` already reads WordNet's |
| named individuals | `who invented the telephone` | YAGO 4.6's facts, labels and entities are downloaded in `data/yago`; only its taxonomy is read, by v687's ontology comparison, and the store holds none of it |
| possession | `whose dog is this` | v689 records ownership |

### 7. Not in the data

Honest UNKNOWNs. Listed so the rest of the catalogue can be read against them.

- **Parts of parts**: `does a dog have cells` (and the teacher says yes).
- **Stated only one level down**: `is a violin made of wood` -- `made_of` is not
  inherited and nothing states it of violin.
- **Abstract capacities, maths, safety**: `can happiness be bought`, `is seven
  a prime number`, `is it safe to eat a mushroom` (v688 alone: unsettled).
- **Qualified only**: `can a person live without water` -- the store has
  `live without appendix` (R28).
- **Abduction coverage**: `what explains wet grass`.
- **Typicality**: `is a penguin a typical bird` -- R21 reports the measure is
  unsound on free-listing norms.
- **Unrecorded preferences**: `does a cat like milk`.
- **People**: `who is barack obama`.

### 8. Dialogue about the answers

| family | example | what happens |
| --- | --- | --- |
| justification | `can a bird fly` · `how do you know that` | a listing: read as a new question |
| confidence | `how sure are you` | a listing |
| why, bare | `why` | UNKNOWN |
| what was said | `beagles can't swim` · `what did I tell you` | a listing; the turns are stored |
| ellipsis | `can a dog swim` · `what about a cat`, `and a fish?` | a listing; a junk re-ask |

These work: `what is my name`, `what is your name`, `who are you`.

### 9. Taught knowledge kept apart from the rest

- `wembles can fly` · `what can fly` -- the backwards reading is over the
  store and does not include wembles.
- `is a wemble like a bird` -- asked of animal instead.
- `what is a wemble`, `what can a wemble do` -- section 3.

These work: `do wembles breathe` (asked of animal, yes); `penguins can swim` ·
`can a penguin swim` (answered from what was taught).

## Wrong answers found on the way

Not missing kinds of question, but wrong answers to questions the page claims
to handle:

- **`can a bird fly`** is headlined `NOT SUPPORTED — … the rest of the store
  does not bear it out`, because 1 of the 5 kinds it was put to (chicken)
  denies it. A generic with an exception is read as overturned, and the badge
  beside it still says verified.
- **`is water wet`**: VERIFIED as `is_a`, through `water.v.01`, WordNet's
  seventh sense -- the any-sense rule for taxonomy applied to a property.
- **`do cats eat mice`, `does a cow eat grass`, `does a spider have eight
  legs`, `is a dolphin like a fish`**: yes on a different predicate (section 5
  and 4).
- **`could a pig fly if it had wings`**: no, about `pig bed.n.01`.
- **`did dinosaurs fly`**: yes.
- **`do you know if a dog can swim`**: no, about the program.
- **`there are two dogs`**: answered VERIFIED as a question.
- **`is it full`** after `it is empty now`: yes.

## What works

The control, so the catalogue is not read as everything failing: yes/no
questions about kinds, including plural (`can dogs swim`), mass (`does fire
burn`), abstract (`is love an emotion`) and non-noun subjects (`is red a
color`, `is swimming a sport`); `can a door be opened`; `is a wheel part of a
car`; quantifiers (`do all birds fly` names chicken, emu and penguin; `do any
mammals lay eggs`); `and` and `or`; `what is a kitten` from definitions memory;
names; `did the dog chase the cat` after being told; the E1 answer to `are the
dogs black`; R18's refusals; capitals and punctuation.

## What would buy the most

By how much of the catalogue each covers, cheapest first:

1. **Carry v688's answer content to the page** (section 1). No reasoning
   changes; about 25 probes go from a verdict word to an answer.
2. **Fix the reader's openers** (section 3): question words that cannot be
   kinds, the modal and negative auxiliaries, expletive `it`, `do you know
   if`. About a dozen probes stop being misread, six of which are currently
   taught as facts.
3. **Wh-questions over episodic memory** (section 2): who, what, where, which
   and how many over the individuals and bound objects already stored. Events
   and order need memory to keep them first.
4. **Close R18's gaps** (sections 4, 6): refuse `how big/long/much/often`,
   `which is heavier`, words-about-words and arithmetic by name; answer
   synonyms, plurals and antonyms from WordNet, which is already loaded.
5. **Backwards within a class and the exceptions** (section 4): the quantifier
   path already counts `which birds cannot fly`; the inverse and the kinds-of
   walk already exist separately.
6. **Structure** (section 5): numbers, second kinds and conditions inside a
   question, and modifiers that narrow the kind. Larger, and bounded by the
   data.
7. **Magnitudes, time, events, named individuals** (sections 6, 7): data the
   store does not hold.

## What changed

2026-09-13, on `v689/question-audit`. Re-running the same probes after the
first four commits changed 103 of the 154 main conversations and 31 of the 49
supplementary ones, nearly all from a listing of the wrong thing to an answer or
a refusal by name; the batch after changed 20, 11 and 11 of 126 single
questions more.

| ranked item | what was done | where |
| --- | --- | --- |
| 1. content to the page | each answer shape is read into one line (`content.digest`) and the page shows it | `be5cba8` |
| 2. openers | requests, `should`/`must`, `n't`, expletive `it`, `do you know if`, purpose and `is there such a thing as` are rephrased before reading; question words cannot be kinds; a why is answered from the walk -- where the fact was stated, the requirement that decides it, and whether the premise holds | `be5cba8`, `48292e3` |
| 3. episodic wh-questions | how many, which, who, where, what did it do, what do you know about it, what happened; `how do you know that`, `what did I tell you`, `what about a cat`, and `they` for the last kind named | `9b7a1a5`, `48292e3` |
| 4. R18 | a value is read from what the thing carries (`what color is a banana`: yellow, `how big is an elephant`: big, with no measurement); durations, frequencies, methods, prices, arithmetic, synonyms, deontic and conditional questions are refused by name | `be5cba8`, `1e1211c`, `48292e3` |
| 5. within a class | `which birds cannot fly`, `what mammals lay eggs`, from the quantifier path | `cc4e2f6` |
| 6. structure | a count and its noun are one term, and another count stated of the thing is a no; the predicate cited is the one covering the question; `but`; the kind above, parts, a choice between kinds, likeness | `1e1211c`, `7030197`, `52a26f8` |

The wrong answers: `can a bird fly` holds with exceptions in its family, since
one relative in five denying it is not an overturn; `is water wet` is no
longer answered through the verb; `do cats eat mice` rests on `eats rodents and
mice` and `does a spider have eight legs` on `has eight legs`, and six legs is
a no; `do you know if` asks the question; `could a pig fly if it had wings` is
refused as a conditional. `what does a plant need to grow` is what a plant
needs rather than what a vote does (R22 leaves a question that names its
subject), and `what happens when a dog barks` no longer includes the vomiting
ASCENT++ records of tree bark (R23 keeps a sense-level source's rows only for
something that can happen).

Still open, from this catalogue:

- **Modifiers and tense**: `can a small dog swim` is answered about dogs and
  does not say so; `did dinosaurs fly` is a yes.
- **An activity's prerequisites**: `what do you need to bake a cake` reads
  `cake.n.03` and finds nothing.
- **Terse and misspelt input**: `dog swim?`, `cna a dog swim`.
- **Noise the crawl carries into lists**: `a -4 penalty` while it rains, an
  animal that can `enter the battlefield`.
- **Not revisited**: `there are two dogs` answered as a question, `is it full`
  after `it is empty now`.
- **Section 7 and the data-bound half of section 6**: magnitudes, time, named
  individuals.
