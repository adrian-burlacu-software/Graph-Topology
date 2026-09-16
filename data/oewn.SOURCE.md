# Open English WordNet — the definitions it changed

WordNet 3.0's maintained successor. `ingestion/oewn.py` reads only the
glosses OEWN *changed*, into the same definitions memory the WordNet glosses
went into, marked `oewn`.

- Home: https://github.com/globalwordnet/english-wordnet
- Downloaded from:
  https://github.com/globalwordnet/english-wordnet/releases/download/2025-edition/english-wordnet-2025-json.zip
  (9.5 MB)
- Licence: CC BY 4.0
- Retrieved: 2026-09-16

## Files

    oewn/english-wordnet-2025-json.zip    read in place, never unpacked
                                          (`ingestion/oewn.py:34`)

The release carries ten assets. This is the **json** one, and the plain
edition rather than `-plus`: `ingestion/oewn.py` opens exactly
`english-wordnet-2025-json.zip` by name, so another asset will not do.

## Why a join by sense key

The store is Princeton WordNet 3.0 (117,659 synsets). The 2025 edition
renumbers every synset, so the two are joined through sense keys: each OEWN
noun sense carries one (`dog%1:05:00::`) and WordNet 3.0 resolves it
(`dog.n.01`). A synset takes the 3.0 synset most of its sense keys resolve
to.

## Numbers this repository depends on

Quoted by `ingestion/oewn.py`:

    noun synsets                      71,864
    same definition                   67,223
    new, no 3.0 sense to hang on       2,803
    changed                            1,838

Many of those changes are punctuation -- `etc` to `etc.`, backticks to curly
quotes -- and only a change in the words is re-read.
