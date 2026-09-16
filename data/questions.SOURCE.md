# Question datasets — `--external` only

Questions people actually asked, read by the grammar as it stands. Used by
`research/v689/teach_reader.py` (`_natural`, `_qasrl`) and **only** with
`--external`.

Fetch with `python -m regenerate --only questions`.

## Not used by default, on purpose

`teach_reader.sources()` says why:

> The question datasets (`_natural`, `_qasrl`) only with `external`: read by
> this grammar they taught unfamiliar wording as generic -- WikiAnswers'
> `where is the X located` is about no one here -- and cost the held-out
> paraphrases.

So every shipped reader was trained **without** these. They are documented
here because the directory existed and nothing recorded where it came from,
not because the pipeline needs them.

## QA-SRL Bank 2.1

- Paper: Fitzgerald, Michael, He & Zettlemoyer, *Large-Scale QA-SRL
  Parsing*, ACL 2018. arXiv:1805.05377
- Home: https://qasrl.org/ — repository https://github.com/uwnlp/qasrl-bank
- Downloaded from: **https://qasrl.org/data/qasrl-v2_1.tar** (37.7 MB,
  verified 2026-09-16)
- Read as: `questions/qasrl/qasrl-v2_1/expanded/train.jsonl.gz`
  (`teach_reader.py`, `_qasrl`)

Note the repository's own `download.sh` fetches `qasrl-v2.tar` — Bank
**2.0**, 37.9 MB. That is a different archive and unpacks to `qasrl-v2`,
which does not satisfy the `qasrl-v2_1` path the code reads. Take 2.1.

## WikiAnswers

- Corpus: clusters of questions WikiAnswers users tagged as paraphrases;
  30,370,994 clusters, ~25 questions each.
- Paper: Fader, Zettlemoyer & Etzioni, *Paraphrase-Driven Learning for Open
  Question Answering* (Paralex), ACL 2013.
  http://knowitall.cs.washington.edu/paralex/
- Downloaded from:
  https://huggingface.co/datasets/embedding-data/WikiAnswers/resolve/main/WikiAnswers.jsonl.gz
  (jsonl.gz, one JSON object per line, `{"set": [...]}` — exactly the shape
  `_natural` reads)
- Licence: see the dataset card; the corpus is Paralex's.
- Read as: `questions/wikianswers-20k.jsonl`

**Derived, not downloaded whole.** The local file is the **first 20,000
clusters**, one `{"set": [...]}` object per line, which is what
`teach_reader` calls "WikiAnswers' first 20 thousand clusters". The full
corpus is 40 GB decompressed, so the regenerator streams it and stops at
20,000 rather than fetching all of it. The check insists on exactly 20,000
clusters, because a short file would quietly teach less.

## Quora — removed 2026-09-16

`questions/quora-pair-class.parquet` was read by `_natural` alongside
WikiAnswers, taking `sentence1` and `sentence2` from each row. It is
**no longer read, and is not fetched**.

It was dropped rather than re-sourced. The file was lost with the rest of
`data/`, and nothing in this repository or its history recorded where it
came from: several QQP exports carry those same two column names, and
substituting one would have changed what the reader is taught with nothing
to say so. Since `--external` is not used by any shipped reader, the honest
option was to remove it rather than guess.

If it is ever wanted again, it needs a real source recorded here first, and
`_natural` needs its branch back.
