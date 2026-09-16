# Wiktionary — English noun senses

Wiktionary's definitions of English nouns, matched to the store's synsets by
`research/v689/learn_wiktionary.py` and read into a definitions memory of
their own.

- Source: kaikki.org's Wiktextract extract of the English Wiktionary
- Extractor: https://github.com/tatuylonen/wiktextract
- Downloaded from: https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz
  (2.7 GB gzipped, 23.1 GB raw; all languages in one file)
- Licence: **CC BY-SA 4.0**, as Wiktionary itself. Attribution required.
- Retrieved: 2026-09-16 (upstream extracted 2026-08-16 from the enwiktionary
  dump of 2026-08-05)

## This file is derived, not downloaded

`wiktionary/english-nouns.jsonl` is **not** an upstream file. It is built by
filtering and flattening the raw extract, and the script that first built it
was never committed -- which is why it could not be regenerated when `data/`
was lost. It is now step `wiktionary` of `python -m regenerate`.

Upstream rows are one object per (word, part of speech), with glosses nested:

    {"word": "dictionary", "lang_code": "en", "pos": "noun",
     "senses": [{"glosses": ["A reference work listing words ..."],
                 "tags": [...]}, ...], ...}

`learn_wiktionary.usable` wants one **flat** row per sense:

    {"word": "dictionary", "gloss": "A reference work listing words ...",
     "tags": []}

So the derivation is:

1. keep rows with `lang_code == "en"` and `pos == "noun"`;
2. emit one row per sense, taking the sense's first gloss;
3. carry that sense's own `tags` -- **absent on many senses**, so treat a
   missing or null `tags` as `[]` rather than assuming a list;
4. keep `word` verbatim: `usable` drops any word that is not all lowercase,
   which is how proper nouns are excluded.

## What the reader then drops

`usable` leaves a sense when the word is a proper noun, when its tags meet
`USES` (26 tags: figuratively, slang, historical, obsolete, offensive and the
rest -- a use of the word, not the thing), or when the gloss is under two
words or is only a pointer to another entry.

## The trap

A sense read onto the wrong synset is exactly the over-affirmation the audit
counts, so a sense is kept only when the taxonomy picks **exactly one**
candidate synset for it, and no other sense of the same word already claimed
that synset. See `learn_wiktionary.py`'s header for the full rule.
