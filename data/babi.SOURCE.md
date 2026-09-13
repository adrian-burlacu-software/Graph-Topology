# bAbI

Facebook's twenty toy question-answering tasks: short stories about people
moving between rooms, carrying things, and a few kinds of reasoning over
them, each followed by questions with a one-word answer.

- Paper: Weston, Bordes, Chopra, Rush & van Merriënboer, *Towards AI-Complete
  Question Answering: A Set of Prerequisite Toy Tasks*, ICLR 2016.
  arXiv:1502.05698
- Home: https://research.facebook.com/downloads/babi/
- Downloaded from: https://dl.fbaipublicfiles.com/parlai/babi/babi.tar.gz
  (ParlAI's mirror, 19.2 MB). The original address,
  http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz, answers
  404 now. The archive holds `tasks_1-20_v1-2`, version 1.2.
- Licence: **Creative Commons Attribution 3.0 Unported**
  (`babi/tasks_1-20_v1-2/LICENSE.txt`). Kept under `data/`, which is not
  committed.
- Retrieved: 2026-09-13

## Files

    babi/babi.tar.gz                          the archive as downloaded
    babi/tasks_1-20_v1-2/en/                  1k training stories per task,
                                              and the test set
    babi/tasks_1-20_v1-2/en-valid/            the same, with a validation
                                              split cut from training
    babi/tasks_1-20_v1-2/en-10k/, en-valid-10k/   10k training stories
    babi/tasks_1-20_v1-2/hn*/                 the same tasks in Hindi
    babi/tasks_1-20_v1-2/shuffled*/           the same tasks with the words
                                              shuffled, a control for models
                                              that learn the vocabulary
    babi/tasks_1-20_v1-2/en*-nosf/            without supporting facts

Every `qaN_test.txt` in `en-valid` is byte-identical to its `en` counterpart
(checked 2026-09-13), so there is one test set.

## Format

A story is numbered lines; the number goes back to 1 when a new story starts.
A question line carries a tab, the answer, another tab, and the line numbers
of the facts that support it:

    1 Mary moved to the bathroom.
    2 John went to the hallway.
    3 Where is Mary? 	bathroom	1

Questions are asked partway through a story, and every later line is part of
the same story. Lists (`qa8`) and paths (`qa19`) are comma-separated:
`football,milk`, `s,e`. Counts (`qa7`) are words: `none`, `one`, `two`.

## The twenty tasks

    qa1  single supporting fact      qa11 basic coreference
    qa2  two supporting facts        qa12 conjunction
    qa3  three supporting facts      qa13 compound coreference
    qa4  two argument relations      qa14 time reasoning
    qa5  three argument relations    qa15 basic deduction
    qa6  yes/no questions            qa16 basic induction
    qa7  counting                    qa17 positional reasoning
    qa8  lists/sets                  qa18 size reasoning
    qa9  simple negation             qa19 path finding
    qa10 indefinite knowledge        qa20 agent's motivations

1,000 test questions per task, 20,000 in all.

## Used by

`research/v689/babi.py`, written up in `research/v689/BABI.md`. On the test
set, 2026-09-13: v689 98.0% right, 0.1% wrong, 1.9% no answer (2.4% right
before the rules were built); SmolLM3-3B zero-shot 54.9% right, 41.3% wrong.
Rules were developed on the `en-valid` training split only.
