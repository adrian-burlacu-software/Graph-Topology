# V690 design — one graph, one attention, one executive

2026-09-13, on `v690/design`. The plan for rebuilding what v687, v688 and v689
do on one graph-based architecture, before v690 itself (answers written in
English) is started. It says why, what the architecture is and where it comes
from, where every existing rule goes, what is fitted from data and what is
not, how generalisation is measured, and the order the work lands in. The
implementation goes into the experiments whose code it replaces: v687 for
memory, truth and walks, v688 for attention and the executive, v689 for
reading and discourse.

## 1. Why

**The knowledge generalises; the control does not.** bAbI (`v689/BABI.md`)
went from 2.4% to 98.0% right, and every new form of question needed a reader
and a session method of its own. The store, VerbNet, WordNet and ConceptNet
carried over to stories nobody wrote them for. What decides *which* of them to
use, in what order, was written anew for each question.

Control is three dispatch cascades, one per experiment:

| where | how a question finds its reasoning | what it has cost |
| --- | --- | --- |
| v687 `ReasoningEngine.ask` | nine layers tried in a fixed order (`_compare`, `_gated`, `_define`, `_contrast`, `_causal`, `_analogy`, `within`, `attributes`, `shapes`), then backwards, rated, v684, folk | four routing errors in the page audit, each of which looked like a working answer (`v687/README.md`, *The audit that produced these*) |
| v688 `question.py` | `sorted(queue, key=rank)` over five generators | the README said this was enough only because questions were utterance-bound, and that dialogue "is what *would* need" an executive and episodic memory. v689 is dialogue |
| v689 `Session.say` | 28 acts in a dict, chosen by a reader with a pattern for each; `_ask` alone is a 125-line cascade | `session.py` 2,233 lines and 75 private methods, `reading.py` 1,272 lines; bAbI added eight readers and eleven methods |

And 46 named rules — R1–R32, E1–E2, T1–T6, S1–S4, I1 — plus the unnamed
preferences in `discourse.py` and `story.py` (recency, gender, the exact word,
latest first), each with its own code path. Many are the same operation
written again: R1, T2, S2, S4 and R22 are one walk; salience, latest-first and
the pronoun preferences are one retrieval; R5, R19, R20, I1 and the 0.6 floor
are one account of evidence.

About 45,000 lines of Python across the three.

## 2. What does not change

These are what make the experiments worth having, and every phase is checked
against them:

- **Absent is not false.** Silence, denial and assertion stay three answers.
- **Abstain rather than guess.** bAbI's 0.1% wrong is the property to keep.
- **Every answer has its derivation**, drawn on the page, with the rule named.
- **The store is never written.** Taught and told live beside it.
- **Events are the record.** Commands decide, handlers apply, replay is
  deterministic (`v689/TIME_AND_EVENTS.md`).
- **Sources are measured apart**, and a model does not outrank a record.
- **Examples teach nothing that is kept.**

## 3. Where it comes from

### Graph-based cognitive architectures

The architecture is assembled from mechanisms that have been built and
evaluated elsewhere, not invented here. What each contributes:

| architecture | what is taken |
| --- | --- |
| **Common Model of Cognition** (Laird, Lebiere & Rosenbloom 2017 — cited by the trie paper) | the division itself: working memory, declarative long-term memory (semantic and episodic), procedural memory, and a cycle that selects one action at a time |
| **Soar** (Laird 2012) | working memory as a graph; operators proposed by matching and chosen by preferences; an **impasse** — no operator, or a tie — opens a subgoal instead of failing; numeric preferences learned from reward (Soar-RL); episodic and semantic memory as graph stores |
| **ACT-R** (Anderson et al. 2004) | retrieval by **activation**: base-level learning (a power-law of recency and frequency, `B = ln Σ t^-d`), spreading activation from the current goal divided by fan, partial matching, and a retrieval threshold below which nothing is recalled; utility learning for productions |
| **Sigma** (Rosenbloom, Demski & Ustun 2016) | a cognitive architecture compiled into **factor graphs** and solved by message passing, so rules and soft preferences are one inference; learning as gradient descent on factor functions |
| **OpenCog Hyperon** (Goertzel et al. 2023) | one typed, weighted **metagraph** (the AtomSpace) holding every kind of knowledge, links being atoms too; rules as graph rewrites over it; attention as importance spread through the graph (ECAN) |
| **NARS** (Wang 2013) | truth as **evidence**: frequency `f = w⁺/w` and confidence `c = w/(w+k)`, with inference rules (deduction, induction, abduction, revision) that say what each does to both; attention as a budget |
| **Rete** (Forgy 1982), the matcher inside Soar | rule conditions compiled into a discrimination network that shares common condition prefixes -- the trie paper's compression, applied to matching |

Transformers enter where the literature says graphs are weak and they are
strong -- turning text into structure -- and nowhere they would own memory or
truth. Attention in a transformer is associative retrieval (Ramsauer et al.
2021, *Hopfield Networks is All You Need*), which is what ACT-R's activation
computes over an explicit graph; MAC networks (Hudson & Manning 2018) and
neural module networks (Andreas et al. 2016) are the learned versions of the
executive in §4.6, and are not adopted because they learn structure as well
as weights (§5).

### The trie paper

Burlacu & West (2021), *Cognitive Network Topology and Optimization for the
Mental Lexicon*. The repo has used Appendix 3 (predicate order,
`adaptive_coverage`) and "the topographical growth of tries is driven by
allocation". The rest of it bears directly on this design:

| the paper | here |
| --- | --- |
| Access and learning run over **one** router; allocation is the growth signal, and telling a new stimulus from a repeated one is the open problem (§14) | introducing and resolving are one operation on the index: `a beagle` allocates, `the beagle` accesses, and a fresh name is a failed access that allocates (§4.3) |
| **Two tries filter one central answer set** (prefix and suffix; AND is superposition, OR and XOR for partial matches), and other filters can join them (§2.a.i, property 2) | an individual is found through several indexes at once -- kind, name, place, what was told, what happened to it -- intersected on one candidate set |
| A **lower-resolution filter** checked at every node rules negatives out early (Experiment 5) | types before tokens: a relation's signature, VerbNet's selectional restrictions and the episode are checked before any string is compared |
| People take the **unique-answer shortcut** (§14) | identification stops when one candidate is left |
| **Executive memory as an EFSM** whose transitions are classifiers over (state, conditions), with arguments, destination memory and the symbol executed as separate outputs; short-term memory as a stack (§2.a.i); ACT-R criticised for hand-coded productions (§2.a.i); the conclusion asks for "a general neural architecture that supports executive memory" | the executive (§4.6): operators as states, selection learned from outcomes, a goal stack as working memory |
| **Backward error-correction**: a learned router cannot be trained on every negative, so its output is checked by the reverse traversal, which is cheaper than access because each node has one parent (§2.a.i) | nothing a learned component proposes -- a parse, a plan, a retrieval -- is committed until it is traced back to what licenses it (§4.7) |
| **Synthesis is the backward route** from the semantic node to the root (Appendix 2) | v690: an answer is verbalised from the path back from its node, which is its derivation (§4.8) |
| A **stack as a decision procedure for instantiating a concept**, ordering the most common predicates of its subtypes (§14, Appendix 3) | classifying a new individual, and choosing a sense, as identification against the kinds' predicate order |
| Learning is **online and instance-level** on a dynamic topology, against static supervised graphs | fitted parameters live on links and operators and update from use, not in a network trained once (§5) |

## 4. The architecture

### 4.1 Memory: one typed graph

Every memory is one **metagraph** of atoms: nodes (a sense, a kind, an
individual, an occurrence, an episode, a value) and links (typed, and atoms in
their own right, so a link can carry a time, a source, a truth and an
activation, and be the subject of another link).

| layer | holds | written by |
| --- | --- | --- |
| store | the 1.96M facts, WordNet, the norms | nobody -- a read-only view over the SQLite store, loaded lazily |
| knowledge | taught kinds, edges, norms | `knowledge` events |
| conversation | individuals, occurrences, episodes, what was told | the conversation's events |
| working memory | the activated subgraph and the goal stack | the executive, for one turn |

The event log stays the record; the graph is a projection of it, as the
episodic tables are now. The store is not copied into the graph -- a lookup
materialises the atoms it touches, and activation decides how long they stay.

**Link types carry the rules' structure as data.** One table says, for each
link type: its signature (what may be at each end), whether it is transitive,
antisymmetric, its converse, its family, whether it is inherited by kinds and
by individuals, whether it is exclusive (one value at a time), whether it is
scoped to an episode, and whether it may be inferred at all. Most of the named
rules become rows of this table (§6).

### 4.2 Truth: evidence

A link carries evidence, not a verdict: `w⁺` for and `w⁻` against, each with
its sources. Frequency `f = w⁺/(w⁺+w⁻)` and confidence `c = w/(w+k)`.

- **Three answers are read off it**, not stored: stated when `f` is high and
  `c` clears the threshold, denied when `f` is low and `c` clears it, silent
  when `c` does not. R20's Kleene logic is the case `c → 0`.
- **Inference rules say what they do to evidence.** Deduction down the
  taxonomy discounts confidence (R5's decay); induction from a kind's members
  (R19, R11, I1) pools their evidence and cannot be more confident than their
  number allows (R19's "four whales are not evidence" is `k`); revision
  combines sources; a nearer statement is more evidence, not an override (R3,
  R4).
- **Sources keep their own scale.** Ascent++'s confidences, ConceptNet's
  constant and WordNet's 0.95 are mapped to evidence per source, calibrated
  against the audit's gold (`v688/AUDIT.md`), rather than compared to one
  floor.
- A model's judgement is a source with its own small `k`: it can unsettle, it
  cannot outweigh a record.

### 4.3 Index: tries over the graph

Tries stay what the paper made them: indexes over the graph, not the graph.

- **Storage order** is `adaptive_coverage`; **query order** is expected gain
  (`v688/attention.py`). The two orders pull apart and both are kept.
- **Several indexes, one answer set.** Kind, name, place, told predicates and
  occurrences each narrow the same candidates; a description is their
  conjunction.
- **Coarse filters first**: signature and type, then episode, then words.
- **Unique-answer shortcut**: stop at one candidate.
- **Access and allocation are one walk.** A walk that ends on a candidate
  accesses it; one that ends in no candidate allocates, and allocation is the
  novelty signal. Definiteness, names and pronouns raise or lower what counts
  as ending on a candidate; they do not branch the code.
- **Rules are matched through the same structure**: rule conditions shared by
  prefix, as Rete shares them, ordered by the paper's orders.

### 4.4 Attention: activation

One number decides what is retrieved, what a pronoun means, which sense a word
has and what is live enough to be curious about:

    A(i) = B(i) + Σ_j W_j · S_ji − P(i) + ε

- `B(i)`, base level: recency and frequency of use, decaying as a power law
  (ACT-R). What `discourse.py` calls salience, what `story.py` does by taking
  the latest, and what v688's `Activation` decays per cycle.
- `Σ W_j S_ji`, spreading: from what is in focus and from the goal, along
  links, divided by fan. A node with ten thousand descendants spreads almost
  nothing -- R12's breadth gating as ACT-R's fan effect.
- `P(i)`, mismatch: how far the candidate is from the description (gender,
  number, kind, the exact word) -- partial matching rather than a filter
  cascade.
- Retrieval is the best candidate above a threshold `τ`; two within a margin
  is an impasse (`which one?`); none above `τ` is silence.

v688's `rank = salience × gain × urgency` becomes this: salience is `A`, gain
comes from the index, urgency is the goal's priority on the stack.

### 4.5 Rules: graph rewrites with provenance

A rule is a pattern over the graph, a link it derives, the truth function it
applies, and its name. `R2` is: `x is_a y`, `y r o`, `r` inheritable, no
nearer evidence against → `x r o` by deduction. Rules are data, compiled into
the matcher, and every derived link keeps the rule and the links it came from,
so the page's derivation is the graph's own record rather than a separate
trace.

Walks are rules applied along a link type's closure: one walk reads the table
and serves R1 (`is_a`), T2 (`before`), S2 (`bigger`), R22 (any converse), S4
(compass directions as a path) and v688's definition ladder.

### 4.6 Executive: operators, preferences, impasses

Procedural memory is a graph of **operators**. A turn is a cycle: propose the
operators whose conditions match working memory, select one by utility, apply
it, repeat until the goal is answered or nothing applies.

    read · resolve · retrieve · walk · compare · count · exists · induce ·
    abduce · consult (v688, the teacher) · ask back · answer · abstain

- **A question is a goal**, placed on the stack by reading. `where was Julie
  before the school` is `retrieve(place, Julie, before(occurrence(go, Julie,
  school)))`: retrieve, then walk, then retrieve -- a composition, not a
  method.
- **An impasse is a subgoal, not a failure.** No candidate above threshold,
  two tied, a word with no sense, a construction no operator matches: each
  pushes a goal of its own -- ask which one, pin a sense, consult, or refuse by
  name (R18 is the impasse nothing resolves).
- **v688's generators are operators proposed by impasses**: gap, doubt,
  curiosity, sense, require, chain and split.
- **Selection is learned.** Each operator's utility in a context is updated
  from outcomes -- right, wrong, abstained -- on development data (ACT-R's
  utility learning, Soar-RL). The order `ReasoningEngine.ask` hard-codes is
  what those utilities start from.
- **The executive is an EFSM, as the paper proposes**: its state is the goal
  and working memory's contents, its transitions are the selection, and each
  applied operator is recorded as an event -- so how a question was answered
  replays like everything else.

### 4.7 Reading: proposals, traced back

Reading turns text into graph fragments: individuals mentioned, occurrences
with their roles and time, states, relations, and a goal if it is a question.

- **Two proposers.** The symbolic reader emits fragments from what it reads
  now (spaCy's parse, VerbNet frames, `tense.py`), and SmolLM3 proposes
  fragments with its output constrained to the fragment grammar.
- **Nothing is committed untraced.** A proposed fragment is checked backwards
  (§3): every atom must map to words of the sentence, every role to a VerbNet
  frame the sentence has the shape of, every type to WordNet. What does not
  trace back is dropped. Where the two proposers disagree on what traces, it
  is an impasse.
- **Soft choices are joint.** A sense, a referent and a frame constrain each
  other -- `John left the apple`, of a John who has it, is not leaving a place.
  These are factors over one small graph per sentence, solved by message
  passing as Sigma does, with activation and evidence as the factors'
  inputs. What `story._choose` does by hand for one verb, done for all
  choices at once.

### 4.8 Generation: v690

An answer is a node and the path back from it; the path is the derivation.
A decoder verbalises it, and the sentence it writes is read back (§4.7) and
must trace to the same answer before it is said. That is the paper's
synthesis-by-backward-route with its own error correction.

As built (`v690/README.md`), the path is read off a v689 turn into a
**message** -- stance, subject and claim or values, what it rests on, and
v689's own account -- and the pieces are the ones reading already has:

- the **decoder** is small and fine-tuned, SmolLM2-360M-Instruct, not a 3B
  model at run time; SmolLM3 is its offline teacher, as the grammar was the
  encoder's, and nothing it wrote is taught unless it traces;
- the **reader of replies is the encoder**, with two heads of its own
  (a reply's stance, and each word's part: subject, claim, denial, value,
  quote); the one model that reads the user reads the system;
- a reply **traces** when its stance is the message's, every content word of
  the claim or the values is read in it (a WordNet synonym, or a verb it is a
  kind of, says it), it denies the claim in the sentences that say it exactly
  where the answer does, an unknown says it is not known, nothing is read as
  said that the message lacks (save the words replies of that stance say
  around content, counted over the corpus), and it credits no source the
  answer did not come from;
- the first candidate that traces is said; when none does, the least wrong
  is said and **marked untraced** on the page, never passed off -- there is no
  template behind it;
- the decoder **teaches itself** on what it can check: once taught on the
  teacher's replies, it replies to every message in the corpus, and what
  traces is taught again.

## 5. Data-driven, and where it stops

**Fitted from data** -- parameters, each with a name and a home:

| parameter | replaces | fitted on |
| --- | --- | --- |
| base-level decay `d`, spreading weights, fan scaling, mismatch penalties, threshold `τ`, margin | salience decay, recency, the pronoun and exact-word preferences, latest-first | bAbI, ToMi and StepGame **training** splits; the time and people test stories |
| evidence per source, `k`, the answer threshold on `c` | R5's decay, R19's eight kinds, the 0.6 floor, confidence bands | the audit's gold (`v688/audit.py`), development half only |
| operator utilities by context | the three dispatch orders | all development splits |
| link weights `S_ji` | nothing yet: the motives' lemma overlap, the curiosity pool | co-occurrence in use, updated online (instance-level learning, as the paper asks) |
| factor weights | `_choose`, sense order, R6's evidence weighting | development splits; sense-annotated data where there is some |

**Not fitted** -- structure, which stays sourced and named:

- the link-type table (transitivity, converse, signature, inheritance),
  from WordNet, VerbNet, ConceptNet's relation definitions and the rules as
  written;
- the operators and what each does;
- the rules' patterns and truth functions;
- the fragment grammar.

Learning structure from data is what end-to-end memory models did: near
perfect on bAbI when trained on it, and ToMi was built because models that
looked good on the earlier bAbI-style theory-of-mind stories were exploiting
their regularities. Fitting weights on one generator reproduces bAbI's
overfit with numbers instead of rules; so every parameter is fitted on more
than one source, and tested on one it was not fitted on.

## 6. Where each rule goes

| rule | becomes |
| --- | --- |
| R1 subsumption closure | link type: `is_a` transitive (walk) |
| R2 property lift | rule: deduction along `is_a` for inheritable link types |
| R3 exception blocking, R4 specificity | truth: nearer evidence weighs more in revision |
| R5 confidence decay | truth: deduction's confidence function |
| R6 sense scoping, R29 sense-to-sense | memory: atoms are senses; the word-to-sense choice is a factor (§4.7) |
| R7 relation gating | link type: not inferable |
| R8 answer synthesis | truth: the three answers read off `f`, `c` |
| R9 relation families | link type: family |
| R10 redundancy, R11 hoisting | index: storage order; rule: induction, marked |
| R12 breadth gating | attention: fan |
| R13 range typing, R32 only the living | link type: signature, checked first (§4.3) |
| R14 sibling exclusion | rule: inheritance runs up `is_a` only |
| R15 bridging | operator: walk between two subjects |
| R16 identification, R17 retrieval | index: the trie walked down, and back |
| R18 refuse by name | executive: the impasse nothing resolves, named |
| R19 corroboration, I1 induction | truth: induction over a kind's members |
| R20 three-valued logic | truth: composition at `c → 0` |
| R21 contrast | index: lowest common ancestor |
| R22 inverse traversal | link type: converse |
| R23 scripts and abduction | rules over temporal link types; truth: abduction, with specificity as fan |
| R24 analogy | operator over the norms' feature types |
| R25 counting kinds, R26 definition | operators: count, define (genus and gloss) |
| R27 taxonomic exclusion | link type: disjoint top branches |
| R28 qualified claims | reading: a qualifier is part of the atom |
| R30 a sentence recorded of a class, not about it | truth: evidence scoped to the class, not inherited |
| R31 magnitudes | link type: ordered value on a dimension (with S2) |
| E1 a quality does not descend | link type: inherited by kinds, not individuals (unless it takes a target) |
| E2 what carried it did it | rule: a doing re-attributed to the carrier within an episode |
| T1 order from tense | reading |
| T2 before | link type: transitive, antisymmetric |
| T3 a state within its episode; one place at a time | link type: episode-scoped, exclusive |
| T4 what an occurrence changes | rules compiled from VerbNet's frames -- data |
| T5 an occurrence is an individual | index: the occurrence trie |
| T6 nothing happens by inheritance | link type: occurrences are not inherited |
| S1 converse, S2 order | link types |
| S3 in line along a direction | rule, kept named and flagged: bAbI's convention, not English's |
| S4 the compass as a map | operator: path over direction links |
| motives | attention: spreading from the state and from each place, meeting where both are active |
| discourse preferences | attention: base level, spreading from focus, mismatch |
| `story._choose`, sense order | factors (§4.7) |
| v688 salience × gain × urgency | attention, index and goal priority |
| v688's seven generators | operators proposed by impasses |
| v689's 28 acts and readers | goals from reading; the acts disappear |

## 7. How generalisation is measured

**The redesign comes first; the held-out measurement after it.** While the
phases land, the guard is regression (below). When they have landed, v689 as
merged (`a70da1c`, which stays checkable out) and the redesigned system are
each run once on test sets nothing here was developed on, and SmolLM3 beside
them with a few examples from each training split:

| set | why | licence |
| --- | --- | --- |
| **ToMi** (Le, Boureau & Nickel 2019) | bAbI-style stories with new templates and distractors; its reality and memory questions test what bAbI tested, its belief questions what nothing here does | CC BY-NC 4.0 |
| **StepGame** (Shi, Zhang & Lipani 2022) | spatial relations over 1–10 hops; tests S1–S4 and S3's convention | to be checked before download |
| **ProPara** (Mishra et al. 2018) | crowd-written process paragraphs with annotated entity states: the reader on English nobody templated | to be checked before download |
| bAbI (kept) | regression, not evidence of generality | CC BY 3.0 |

**Rules of the measurement.** Each set's test split is run once per phase,
never inspected for failures; development happens on training splits. Right,
wrong and none are reported apart, with selective accuracy (right over
answered). SmolLM3 is given the same few examples v689's parameters are
fitted on.

**Regression, every phase:** the tests, and the common-sense probe (114) and
the time probe (158 turns) against the page. bAbI's test split and the v688
audit's configurations at the same `--limit`, which take the better part of an
hour each, once the phases that change answers have landed.

**Ablation, every component:** each attention term, each truth function and
the learned utilities are switched off one at a time on the development
splits, so what the data supports is measured rather than assumed.

## 8. Order of work

Each phase is a branch, merged when its acceptance holds.

| phase | lands in | what | accepted when |
| --- | --- | --- | --- |
| **P1 graph and link types** | `v687` | the metagraph over the store (lazy, read-only), the link-type table, one table-driven walk serving R1, R9, R22, R27, T2, S1, S2, S4 | every test passes; the probes unchanged |
| **P2 evidence** | `v687`, `v688` | truth as evidence; R2–R5, R19, R20, the floors and bands as truth functions; per-source calibration | tests pass, the probes unchanged; once calibrated, the audit's over-affirmation no worse |
| **P3 activation** | `v688`, `v689` | one activation over the graph for retrieval, reference, latest-first, I1 and curiosity; parameters fitted | tests pass; discourse's cascade deleted; latest-first and I1's last-told are base-level activation, not rules of their own (where qa16's members disagree, the last told was bAbI's answer 59 times in 64: recency is evidence, weighted by activation) |
| **P4 executive** | `v688`, `v689`, `v687` | operators, utilities, impasses and the goal stack; the three dispatch cascades replaced; v688's generators as operators | tests pass; `session.py` and `reasoning.py`'s dispatch gone |
| **P5 reading** | `v689` | fragments; the symbolic reader re-expressed; SmolLM3 as a constrained proposer; backward tracing; joint soft choices | tests pass |
| **P6 rules as data** | `v687`, `v689` | rules compiled into the matcher; T4 from VerbNet and motives by spreading | rules are data; tests pass |
| **held-out** | `v689` harnesses | ToMi, StepGame (and ProPara) harnesses in the shape of `babi.py`; `a70da1c`, the redesign and few-shot SmolLM3, each run once on the test splits | the redesign beats `a70da1c` at no higher wrong rate |
| **v690** | `v690` | generation by the backward route, read back | its own evaluation |

### Where it stands

Each phase so far lands its structure with the parameters set to what the
code did, so that the tests and the probes show nothing moved; fitting is a
step of its own, once there is data to fit on.

**What a phase is accepted on**, from E1 onwards: the suite, and
`research/v689/probe.py` -- 114 ordinary questions answered from the real
store, compared on the outcome and the source, never on the wording. The
suite alone is not enough: 1,027 tests passed either side of a relaxation of
R28 that `v688/audit.py` then measured as tripling over-affirmation, so a
phase that changes an answer has to be seen changing it. The probe was a
scratch script until now and is committed for that reason. Its companion,
the time-and-events probe, is **not** reconstructable -- its log kept the
thirty scenario names and none of their turns -- so `test_time` (39) and
`test_goals` (17) stand in for it, and this is a weaker criterion than the
phase table above was written against.

| phase | branch | what landed |
| --- | --- | --- |
| P1 | `v690/graph` | `v687/links.py`, one table of link types every rule's list is read off; `v687/walks.py`, one closure and one path for R1, T2, S2, S4. 913 tests; both probes unchanged |
| P2 | `v690/evidence` | `v687/truth.py`: R19's sample and floor, R5's decay and logic's three values as evidence. 918 tests |
| P3 | `v690/activation` | `v688/retrieval.py`: discourse's cascade, story's latest-first, T3's latest holding and I1's last-told as one retrieval. 918 tests; both probes unchanged |
| P4 | `v690/executive` | `v687/executive.py` -- in v687 because v687 cannot import v688 -- and v687's eleven layers, v689's 28 acts and v689's `_ask` (fifteen operators over one working memory) as operators. 931 tests |

| P5, first | `v690/goals` | `v689/goals.py`: a question read into slots -- asked, relation, who, subject, object, verb, clause -- and three operators, `located`, `holding` and `occurrence`, that answer every cell their slots make by composing memory. They run before the act operators and decline when they find nothing. 945 tests; both probes unchanged |

**What moved first is what answers.** P1 to P4 put the control on one
footing without changing a verdict, which is what the probes check. P5 is
where it starts to generalise, and the evidence is fifteen questions no
pattern was written for, answered on the real store by composition:

| relation | cells that answer now |
| --- | --- |
| located | who is in the kitchen; is anyone in it; how many people are in it; what is in it; when was Mary in it |
| holding | who has the football; who is carrying it; is anyone carrying it; does Mary have it (yes, or no because it is with John: T3's one place at a time); what does Mary have; how many things does she have |
| occurrence | where did Mary go (first); where did she drop the football; did anyone go to the garden; how many people went to the kitchen |

The second step turned seven of `reading.py`'s patterns -- where, what is
carried, who, what did, to whom, when, how many times -- from acts into the
cells they fill. Each now reads `(asked, relation)` onto its reading, and the
session answers the cell, not the pattern's name:

| cell | answered by |
| --- | --- |
| place of located | `_where`, T3 and T4 |
| object / count of holding | `story.carrying`, T4 |
| subject of occurrence | `_who`, T5 |
| object of occurrence | `_what_did`, T5 |
| recipient of occurrence | `story.to_whom`, T5 |
| time of occurrence | `story.when_asked`, T5 |
| times of occurrence | `story.how_many_times`, T5 |

So the operator table is the coverage matrix: a question is answered by the
cell it fills, whichever reader filled it.

The third step deleted those seven patterns. `goals.cell_reading` is one
grammar for them: the question word says what is asked (when a time, how
many times a count of occurrences, where a place, who the subject, what the
object, how many before a noun a count of things), and what follows says of
which relation (AUX NP VERB-PHRASE an occurrence; COPULA NP, before or after
something, or ending in a place, located; COPULA NP V-ing holding; a
trailing to or from the recipient). The goal reader now runs inside
`reading.read` on the same words, so a reading carries both its cell and its
goal. Checked reading by reading on 1,206 questions -- every question in
the tests and probes, and templates for every pattern family crossed with
eleven noun phrases: none reads differently.

The fourth step moved the rest: `reading.py` has no question patterns left.
`v689/grammar.py` reads every wh-question about this conversation's
individuals into the cell it fills, trying shapes in order after the question
word, and `Session._cells` is the operator table, one handler per cell:

| relation | cells |
| --- | --- |
| occurrence | time, times, subject, recipient, object, verb (`what was it doing`) |
| located | place |
| holding | object, count |
| dimension (S1-S4) | subject (`what is north of X`), object (`what is X north of`), path (`how do you go from X to Y`) |
| attribute | value (`what color is X`), object (`what is X afraid of`) |
| motive | place (`where will X go`) |
| is_a | count (`how many dogs are there`), which (`which dog is black`), kind (`what kind of dog is X`) |
| story / told / future | events (`what happened`, `what did it do first`, `what did i tell you`, `what will happen`) |
| any | facts (`what do you know about X`, `what can X do`) |
| answer | grounds (`how do you know that`) |
| question | again (`what about a dog`) |

Checked against the saved readings: of 1,206, 170 differ, each only in now
naming the cell its act fills, and none in anything else. A test pins that
the cells the grammar reads are exactly the cells the session answers.

The fifth step merged the two readers. A question's reading carries its
**goals**, in the order they are tried: the goal read as slots
(`grammar.slot_goal`, which was `goals.goal_from`), then the goal its own
words' cell states, with the slots read off the reading (`_own_goal`). The
act names are gone: the grammar's readings are `question`, the page shows
each goal's cell, and the session asks the cells, not the name, whether a
turn was `how do you know that` or `what about a cat`. Every cell has one
operator, which a test pins: its own (`Session._cells`), or its relation's
(`goals.COMPOSES`). What is carried was two -- `story.carrying` for `what is
Mary carrying` and the holding operator for `what does Mary have` -- and is
now the holding operator, which resolves a question's own subject as a
question's subject is resolved. `what does Mary have` is proposed twice:
what is with her, composed first, and failing that what was told she has.
Of 1,209 readings, 455 differ, each only in its act being `question`.

The sixth step made the grammar one table (`grammar.SHAPES`): after each
question word, the shapes that read a question straight into slots and the
shapes that read it as a reading of its own, in one order, each proposing a
goal. A question is proposed the first of each, and `reading.read` proposes
before it reads, so the reading a goal states is the question's reading.
Of 1,210 readings, none differs.

The seventh step put every cell on its relation's operator. There is one
operator a relation -- located, holding, occurrence, dimension, attribute,
motive, is_a, story, told, future, any, answer, question -- and it answers
every cell of its relation, trying a question's goals of that relation in
the order they were proposed (`goals.Answering.cells`). The cells memory is
composed for are the operator's own methods; the rest are the handlers
written for them, now its methods for those slots: `where is Mary` is the
located operator's, `who chased the cat` the occurrence operator's.
`Session._cells` is gone, and a test pins that the grammar's cells are
exactly the operators' cells.

**P6 landed as rules as data.** Every named rule is a row of
`v687/rulebook.py` -- R1 to R32, E1, E2, T1 to T6, S1 to S4, I1 and the
motives: what it becomes in the architecture (§6), the link types it reads
and the link-type field that decides it, the truth function it applies, its
parameters, its source and the module that applies it. The code reads its
numbers and its text from the rows: R5's decay and floor, R12's breadth
limit, R14's sibling limit, R19's floor and sample, v688's price for a
derivation by each rule and the rules whose walk is the proof, and the
page's text for every rule, which lived in three dictionaries across two
versions. A test pins that every link type, field and truth function a row
names exists, that every rule v687, v688 and v689 cite has a row, and that
the page's text is the rows'.

What the rows do not do yet is run. R2 is still `reason.py`'s walk, which
reads R2's link-type field and R5's decay from the table; it is not a
pattern compiled into a matcher. T4 already is data: VerbNet's frames, read
at runtime (`change.py`). The motives are flagged: they meet on a shared
lemma, and spreading activation (§4.4) would change which place is chosen,
so it waits for the fitting step and a bAbI run.

**v690 proper, first (2026-09-15, `v690/decoder`): generation read back.**
Answers are said in English by a small decoder and read back by the encoder
before they are said (§4.8, `v690/README.md`). Social acts -- `hello`,
`thanks`, `what can you do` -- became acts of v689's reader, answered from
what the conversation is.

| step | what it came to |
| --- | --- |
| conversations | 1,851 played through v689 over the real store: 16,427 turns, 14,042 distinct messages (noted 7,139, value 2,634, unknown 2,583, yes 902, no 538, refused 162, which 84) |
| teacher | SmolLM3-3B, offline, one greedy reply each to 6,678 stratified messages (3.7 a second once nothing else held the card) |
| label 1 | 5,256 messages (79%) with a reply that reads back: noted 95%, yes 85%, no 83%, unknown 82%, refused 82%, value 63%, social 50%, which 23% |
| decoder v1 | SmolLM2-360M, 4,713 pairs, 3 epochs, 8-bit Adam: held-out loss 1.396 → 0.148 |
| bootstrap | v1 replied four times to the 7,625 messages the teacher never saw |
| label 2 | 12,534 of 14,303 messages (88%) read back: noted 97%, unknown 88%, yes 83%, refused 80%, no 78%, value 70%, social 50%, which 13% |
| label 3 | 12,505 once a denial is judged by sentence and an unknown must say it is not known |
| decoder v2 | 9,829 pairs balanced by stance, 3 epochs: held-out loss 1.604 → 0.123 |
| teacher fixes | the cues v687's parse is taught from stepped over `goldfish` and `minnow` as adjectives -- spaCy's tag after an article -- and read `can a goldfish walk on land` as about walking; an adjective now modifies only if WordNet has it as one (211 parse records moved). Names were taught all known or none, never `Mary gave the football to John` with only Mary known, and `can you V it` kept as said 26 times against 114 participle rewrites; both are taught now |
| encoder | `reader-v690d`, one MiniLM-L6 for every reader, 10 epochs: held-out whole records read 97.4%, ask 99.8%, place 97.7%, parse 97.1%, reply 93.6%, stance 99.4%; all 1,012 tests pass with it. (Retrained on the narration-filtered replies below and shipped as `llm/reader`; `reader-v690d` is kept as the rollback) |
| page probe | every reply read back, and two that should not have: `No, a penguin can't fly. You said so.` of what the norms state -- the check's denial window reached the answer's own `can't` in the sentence before -- and `what can you do` stopped mid-sentence, a habit the decoder learned from teacher replies capped at 64 tokens. A denial now excuses only its own sentence, a reply has to end its sentence, and when nothing traces the speaker reads each reply again with one sentence left out, which never adds a word |
| evaluation | 600 messages of held-out conversations, decoder v2 writing and `reader-v690d` reading back: the first reply reads back 94.7%, one of four 96.0%, one of four or shortened 96.7% (noted 98.8%, unknown 100%, yes 97.9%, no 93.8%, value 85.9%); stance read 98.1%; the encoder agrees with the teacher's labels 97.4%. Corrupted replies caught: an invented word or a rule named 100%, a value swapped 98.4%, a denial added 98.0%, a denial dropped 79.2% |
| decoder v3 | the replies labelled again with those checks: 12,482 messages read back (38 fewer -- 30 `no` replies that said `you said so` of the store, 6 cut-off abilities); `You said so` in training pairs 17 → 2, both of what was said, cut-off replies 3 → 0. 3 epochs, held-out loss 0.146 → 0.125 → 0.128. On the same 600: first reply 95.3%, one of four 96.3%, or shortened 96.7%; a denial added caught 100%, dropped 80.5%. The page probe reads back 17 of 17 with nothing left out: `No, a penguin can't fly.` and the whole of what it can do are what it writes first |

| label 4 | a reply may no longer narrate the exchange (`roundtrip.NARRATING`): `You asked if they can fly, and I said no` of a question the *reader* put again, and `You've told me so` -- one token to spaCy, which is how it walked past a check looking for `you` beside the verb. Framing could not stop either: it counts what replies say around content, and the teacher's own replies said `tell` and `say` often enough to license them for yes and no. 12,482 → 12,389 messages read back, the 93 being that habit (yes −46, unknown −41). Counting `source` and `unfinished` among the reasons shows what had been invisible: 632 replies narrate (value 275, unknown 187, yes 83, no 78, refused 9) and 42 stop mid-sentence |
| decoder v4, encoder again | the same corpus, 9,659 pairs, 3 epochs: held-out loss 1.605 → 0.147 → 0.126 → 0.123. The encoder retrained beside it on the filtered replies, 10 epochs: read 97.2%, ask 99.8%, place 97.7%, parse 97.0%, reply 93.6% — level with `reader-v690d` (±0.16), which is the point: same reading, taught on replies that do not narrate |
| evaluation | the same 600 held-out messages: first reply 95.3%, one of four 96.3%, or shortened **97.3%**; stance read **99.1%**, the encoder agrees with the labels **97.3%**. **Value, the weakest stance, improves most** — first 79.8 → 80.8, one of four 83.8 → 85.9, or shortened 85.9 → **88.9**, stance read 92.6 → **95.8**, agreement 86.9 → **89.1** — and `no` first reply 93.8 → 100%. The price is the stance that had the habit: `yes` first reply 95.8 → 89.6, recovered to 95.8 by leaving a sentence out. Removing a habit helps the stances that never had it |
| bAbI, retrained encoder | the whole test split, 20,000 questions: **97.5% right, 0.2% wrong, 2.3% declined** (SmolLM3 few-shot, re-scored from its kept replies: 54.9% / 41.3% / 3.8%). Fourteen tasks at 100%. Against the same run under `reader-v690d` (98.0% / 0.1% / 1.9%) it is **0.5 points lower, and the loss is one family**: task 2 100 → 93.0 (67 declined), task 3 100 → 98.6, task 6 100 → 99.1 (9 wrong). Every one is `took`/`dropped`/`put down` *X* `there` -- `Where is the football?` answered `the football`, `Is Mary in the hallway?` answered no because `Mary took the apple there` put her *at the apple*. bAbI runs v689 and the reader with no decoder, so the encoder is the only cause. Its held-out `place` score is 0.9774 against v690d's 0.9771: **held-out parity did not mean behavioural parity on this family**, which is the finding worth keeping |
| rebuild, 2026-09-16 | `llm/` was destroyed (a worktree junction) and retrained from empty through `python -m regenerate`, the full cycle `v690/README.md` documents. Reader: read 97.0%, ask 99.6%, place 97.5%, parse 97.0%, reply 92.7%; decoder 9,807 pairs, held-out loss 1.626 → 0.156 → 0.123 → 0.124; 1,027 tests pass. bAbI **97.3% / 0.2% / 2.5%**: seventeen tasks as they were, the loss again tasks 2, 3 and 6 (899, 978, 988 right) |
| `took` *X* `there`, fixed | **The encoder was not the cause; it was the trigger.** `change._frame` gave every bare noun phrase after the verb the position `object`, and this copy of VerbNet spells an adverb, an adjective or a place with no preposition as `NP` in a frame's syntax while its description says what it is: bring-11.3's `NP V NP ADVP` is `NP.Theme NP.Destination`, put_direction-9.4's `dropped it there` likewise -- 90 frames. So a frame of two noun phrases matched a sentence of one, `Daniel took the football there` put Daniel *at the football*, and the football, once dropped, was with itself. A second noun phrase is now a second object only where the description counts two (`NP V NP NP`, give-13.1); otherwise the sentence leaves it unsaid, and what is let go of is where its holder is. `reader-v690d` never happened to choose those readings; the retrained ones did, which is why held-out parity looked like the explanation. bAbI **98.0% / 0.1% / 1.9%** -- tasks 2, 3 and 6 at 1,000 each, the other seventeen identical story for story, level with the best run recorded; 1,029 tests pass |
| E1: a goal stack | Working memory was a plain dict: no goal, nothing beneath it, nowhere for an impasse to go. `executive.Working` is a stack of goal frames and **is the top frame** -- a dict, so every `memory["answer"]` and `memory.get("goals")` reads as before and one frame deep it is the dict it replaced. `push(goal)` opens a subgoal that reads what is beneath it (a Soar substate reading its superstate) and writes only its own frame; `pop` returns what it wrote and restores the goal beneath; `subgoal()` always pops, even when the subgoal raises. A trace records its goal and depth. The three places that ran an executive -- acting on a claim, `_ask`, v687's layers -- pass one with a goal named. Nothing pushes a subgoal yet: that is E2. Behaviour-preserving, measured: the probe's 114 answers exactly as its pre-E1 baseline (itself two runs that agreed), 1,034 tests pass |
| E2: an impasse opens a subgoal | An operator that cannot get past something names it in the slot `impasse`; when nothing else is proposed, the executive pushes the `Subgoal` of that name on E1's stack, runs its operators, pops, writes back the slots it `returns`, and the goal goes on (Soar's impasse, substate and result). Each impasse is subgoaled at most once a run, and a plain dict has nowhere to push one, so nothing else changes. First use: `where will X go`. When no place mentioned is *for* what a state moves one to, the subgoal looks for a place mentioned where what they want is *kept*, following `at_location` up to two steps (`beverage at_location can`, `can at_location kitchen`), and cites every row. **Only what a state moves one to (`motivated_by_goal`, `has_prerequisite`), never what it leads to (`has_subevent`, `causes`)**: the first build followed both and measured 27 right against 32 new wrong -- `tired` leads to `change` (coins in a kitchen drawer), hunger to illness in a body in a bedroom. **A place never mentioned is not guessed** (Adrian's choice): the store names the bedroom, the hotel, the hostel and a railway station as for sleeping, and cannot choose. Developed on task 20's *training* stories (65.9 / 0.8 / 33.3 -> 68.9 / 0.8 / 30.3). Test split: **98.1% / 0.1% / 1.8%**; task 20 declined 322 -> 300, right 672 -> 694, wrong unchanged at 6, the other nineteen tasks identical question for question; the probe as its baseline; 1,042 tests. What task 20 still declines is not control: 221 of its where-questions come before any place is mentioned, and nothing in the store says boredom takes one to a garden |
| E3: what an answer came to, before anything learns | Every executive is named (`act`, `ask about a kind`, `where will`, `kept there`, `v687 layers`); `executive.episode()` records every top-level run of a turn, subgoals within the run that pushed them, and `Session.say` keeps them on `turn.executed` (not in `as_dict`). bAbI records carry them, and `babi.py credit` gives each operator that did something the reward of the answer -- right 1, declined 0, wrong -1 -- in a `Ledger`, with `RATE` still zero; a test shows the ledger's mean is where `reward` takes a utility when learning is on. **Looked at, on bAbI's training split (18,013 answers, all credited): learning would change nothing.** Nearly every operator's mean is +0.99 to +1.00; the one low one, `motive` at +0.150, declines where there is no candidate place or no knowledge, which no other operator could answer instead; and in these executives operators do not compete -- acts are exclusive by `reading.act`, and `_ask` is a fixed pipeline -- so a utility decides nothing. The cascade where utility would decide something, `v687 layers` (nine or more operators proposed together), runs in the v688 engine pool's processes and is not recorded, and bAbI does not exercise it. So learning stays off: it needs conflict sets recorded (what was *proposed*, not only what fired), runs from the pool, and a reward from the audit's gold rather than bAbI. 1,046 tests; the probe as its baseline; task 20's training split exactly as E2 left it |
| E3, measured where a choice is made | Conflict sets recorded (the operators proposed beside the one fired, shown only where there were several); `ReasoningEngine.ask` returns its layers' run on the payload; the audit credits it by the gold's side -- held property positive, corrupted claim negative model-free, screened foil negative. `comps_screened.jsonl` rebuilt: 20,966 pairs (20,925 documented), 55 minutes. **Audit, shipped, screened gold, 250 per rung + 300 corrupted, 2,258 questions:** about 99% right when decided; over-affirmation on the foil ladder 0.0 / 4.8 / 5.6 / 8.9%. **`v687 layers` answered 2 of 2,258** (both `refused by name`); every other layer declined on every question, and the answers came from v684's rules after the layers' impasse -- R17 778, R1 59 -- which are not operators. The conflict sets look full (all ten layers proposed, 2,227 times) because every layer proposes unconditionally: it is an exhaustive try-in-order, not a choice, and order could only matter where two layers would both answer. No reversals. **So nothing in the executive decides what is answered, and there is nothing to learn yet**; the decisions are v684's rule cascade inside `super().ask`. That is E4's first job. Priors must also go on the reward's scale before any learning: they are the order written (12 ... 1) against rewards in [-1, 1] |

Two things were tried against this and **measured, rejected and reverted**, both
recorded where the constant lives. R28 refuses a fact saying more than the
question, so `mammal capable_of "breathe air"` does not answer `does a cat
breathe`; accepting a bare object while still refusing a prepositional tail
bought 2 points of coverage for triple the over-affirmation (`audit.py`:
corrupted asserted 1.1% → 3.1%, relative accuracy 91.6% → 89.3%) and did not
even fix the question, since R19 refuses it anyway. `message.LISTED` 5 → 8, to
license a listing's later items, was worse on every count (value traced
1,865 → 1,860, value added 884 → 892). All 1,027 tests passed through both, so
only the audit and the corpus caught them.

What the check rejects is mostly invention -- `How can I assist you today?`,
`used for grand prix races`, a beagle said of a pig because the teacher's
example had one, `you told me so` of what the store says of stags. `which
one?` is the weak stance: when `the other one` resolves to nothing, the
message has nobody in it to say, and the teacher made someone up.

## 9. Risks

- **Speed.** A graph over 1.96M facts in Python is slow if built eagerly. The
  store stays SQLite; atoms are materialised by lookup; activation bounds
  what is held.
- **The page and the tests describe behaviour, not structure.** Many of the
  899 tests assert wording. Wording that changes because the derivation is
  now the graph's own is updated in the phase that changes it, and said so.
- **Parameters can overfit as rules did.** More than one source for every
  fit; frozen tests; ablations reported.
- **A constrained model can still be confidently wrong.** It proposes; the
  trace decides; its disagreements are counted.
- **Scope.** Each phase has to leave the system working and measured; a phase
  that cannot is split rather than left half-landed.

## 10. What is deliberately not here

- No end-to-end neural memory, and no network that learns the rules.
- No model writing memory without a trace.
- No new knowledge sources: this is about control, and qa20's missing motives
  stay missing until a phase about data.

## References

- Anderson, J. R., Bothell, D., Byrne, M. D., Douglass, S., Lebiere, C., & Qin, Y. (2004). An integrated theory of the mind. *Psychological Review*, 111(4).
- Andreas, J., Rohrbach, M., Darrell, T., & Klein, D. (2016). Neural module networks. *CVPR*.
- Burlacu, A., & West, R. (2021). *Cognitive Network Topology and Optimization for the Mental Lexicon*.
- Forgy, C. L. (1982). Rete: A fast algorithm for the many pattern/many object pattern match problem. *Artificial Intelligence*, 19(1).
- Goertzel, B., et al. (2023). *OpenCog Hyperon: A framework for AGI at the human level and beyond*.
- Hudson, D. A., & Manning, C. D. (2018). Compositional attention networks for machine reasoning. *ICLR*.
- Laird, J. E. (2012). *The Soar Cognitive Architecture*. MIT Press.
- Laird, J. E., Lebiere, C., & Rosenbloom, P. S. (2017). A standard model of the mind. *AI Magazine*, 38(4).
- Le, M., Boureau, Y.-L., & Nickel, M. (2019). Revisiting the evaluation of theory of mind through question answering. *EMNLP*.
- Mishra, B. D., Huang, L., Tandon, N., Yih, W., & Clark, P. (2018). Tracking state changes in procedural text. *NAACL*.
- Ramsauer, H., et al. (2021). Hopfield networks is all you need. *ICLR*.
- Rosenbloom, P. S., Demski, A., & Ustun, V. (2016). The Sigma cognitive architecture and system. *Journal of Artificial General Intelligence*, 7(1).
- Shi, Z., Zhang, Q., & Lipani, A. (2022). StepGame: A new benchmark for robust multi-hop spatial reasoning in texts. *AAAI*.
- Wang, P. (2013). *Non-Axiomatic Logic: A Model of Intelligent Reasoning*. World Scientific.
- Weston, J., Bordes, A., Chopra, S., Rush, A. M., & van Merriënboer, B. (2016). Towards AI-complete question answering. *ICLR*.
