# Question datasets — `--external` only

Questions people actually asked, read by the grammar as it stands. Used by
`research/v689/teach_reader.py` (`_natural`, `_qasrl`) and **only** with
`--external`.

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
  (`teach_reader.py:1559`)

Note the repository's own `download.sh` fetches `qasrl-v2.tar` — Bank
**2.0**, 37.9 MB. That is a different archive and unpacks to `qasrl-v2`,
which does not satisfy the `qasrl-v2_1` path above. Take 2.1.

## WikiAnswers

- Corpus: clusters of questions WikiAnswers users tagged as paraphrases;
  30,370,994 clusters, ~25 questions each.
- Paper: Fader, Zettlemoyer & Etzioni, *Paraphrase-Driven Learning for Open
  Question Answering* (Paralex), ACL 2013.
  http://knowitall.cs.washington.edu/paralex/
- Available as: https://huggingface.co/datasets/embedding-data/WikiAnswers
  (jsonl.gz, one JSON object per line, `{"set": [...]}` — exactly the shape
  `_natural` reads)
- Read as: `questions/wikianswers-20k.jsonl`

**Derived, not downloaded.** The local file is the **first 20,000 clusters**
of that corpus (`teach_reader.py:1501`: "WikiAnswers' first 20 thousand
clusters"), one `{"set": [...]}` object per line. The full corpus is 40 GB
decompressed; do not fetch all of it.

## Quora question pairs

- Read as: `questions/quora-pair-class.parquet`, columns `sentence1` and
  `sentence2` (`teach_reader.py:1522-1527`)
- **Source not established.** The columns match the SetFit/QQP family of
  exports rather than Quora's original release, and no URL in this
  repository or its history records which one was taken.

This is the one file here with no verified provenance. It is `--external`
only, so nothing shipped depends on it, but it should not be guessed at: a
different QQP export would change what the reader is taught without any
error saying so.
