# EntailmentBank — multi-step entailment trees

Gold proof trees over science facts: a hypothesis, the sentences a proof of
it needs, and which of them combine at each step. Read by
`research/v690/entailment.py` for `DESIGN.md` §8c's F3.

## Where it came from

    https://allenai.org/data/entailmentbank

Dalvi, Jansen, Tafjord, Xie, Smith, Schmid & Clark (2021), *Explaining
Answers with Entailment Trees*, EMNLP. 1,840 trees, built over the WorldTree
corpus.

The page redirects to a Google Drive folder holding three dated versions.
`regenerate.py` takes **v3 (`v3_May6_2022`)**, Drive file id
`1kVr-YsUVFisceiIklvpWEe0kHNSIFtNh`, which downloads unauthenticated from
`uc?export=download` while it stays under Drive's virus-scan threshold. It
is 7.8 MB, so it does. There is no S3 or HuggingFace mirror from the
authors; the copies on the HuggingFace hub are third-party re-uploads and
are not used.

## Files pulled in

| file | what it is |
| --- | --- |
| `dataset/task_1/{train,dev,test}.jsonl` | the tree, with **exactly** the sentences its proof needs. **This is what v690 reads.** |
| `dataset/task_2/…` | the same trees, each given 25 sentences of which ~21 are distractors: retrieval as well as structure |
| `dataset/task_3/…` | the same trees against the full corpus |
| `supporting_data/worldtree_corpus_sentences_extended.json` | the corpus the sentences are drawn from |

Splits are 1,313 / 187 / 340. Task 1 dev is what `regenerate` counts.

## Why task 1 first

The point of this dataset here is to test **control**, not knowledge. Every
earlier attempt to exercise the executive died on missing facts — V8 found
39 of 39 unreachable negatives were no data at all — so task 1 is the one
variant where that cannot happen: the facts needed are the facts given, and
anything the system fails to prove it failed to *assemble*.

## What is scored, and what is not

A step's conclusion is new prose (`earth is a planet that rotates on its
tilted axis`). Writing it is a generation problem and is not attempted.
`entailment.py` scores the **structure**: each step named by the set of
original sentences beneath it, so the score does not depend on what the
intermediate conclusions are called.

**The root step is free.** In task 1 every sentence given is needed, so the
root is always all of them, and putting everything into a single step scores
47.9% steps F1 at 100% precision while recovering no structure whatever. The
`inner` numbers drop the root and are the ones that mean anything: the
baseline to beat there is left-to-right chaining at **10.6%**.

## The ceiling

Scoring merges by agreement with the gold tree, binary merges rebuild it
exactly in 135 of the 187 dev trees. So **72%** is the most this framing can
reach, and the remaining 28% need a step of three premises or more.
