
# V494 — complete semantic memory

V494 removes DialoGLUE completely. The combined semantic stack is now:

```text
SGD
MultiWOZ
Ubuntu
WordNet
VerbNet
ConceptNet
```

## Why these sources?

WordNet provides lexical semantic structure around synsets and relations such
as synonymy, hypernymy, hyponymy and antonymy.

VerbNet adds structured verb knowledge: verb classes, thematic roles,
selectional restrictions, syntactic frames, examples, and semantic predicates.
The official VerbNet documentation explicitly defines roles as semantic
relationships between a predicate and its arguments, and frames as
syntactic/semantic realizations. citeturn828017search12turn828017search0

ConceptNet supplies broad commonsense relations.

The dialogue corpora supply conversational behavior/examples.

## DialoGLUE

DialoGLUE is intentionally **not discovered or ingested** in V494.

## Install WordNet + VerbNet

From Graph-Topology root:

```powershell
python .\research\v494_full_semantic_memory\install_semantics.py --all
```

## ConceptNet

Download the ConceptNet assertions dump:

```powershell
python .\research\v494_full_semantic_memory\download_semantic_data.py `
  --out ".\data\conceptnet-assertions-5.7.0.csv.gz"
```

## Combined ingestion

```powershell
python .\research\v494_full_semantic_memory\combined_ingest.py `
  --data-root "." `
  --conceptnet ".\data\conceptnet-assertions-5.7.0.csv.gz" `
  --wordnet `
  --verbnet `
  --out ".\results\full_semantic_memory.sqlite" `
  --reset
```

The resulting SQLite database preserves source provenance and keeps VerbNet
structured in dedicated tables:

```text
verbnet_classes
verbnet_members
verbnet_roles
verbnet_frames
```

VerbNet member/class relationships are additionally exposed as typed semantic
evidence for retrieval, while frame/role structure remains available to
higher-level reasoning.

## Architecture

Without LLM:

```powershell
python .\research\v494_full_semantic_memory\assistant_cli.py `
  --memory ".\results\full_semantic_memory.sqlite"
```

With LLM:

```powershell
python .\research\v494_full_semantic_memory\assistant_cli.py `
  --memory ".\results\full_semantic_memory.sqlite" `
  --teacher ".\llm\SmolLM2-1.7B-Instruct"
```

No V484/V490/V491/V492 dependency is required.


## V495 SGD robustness

SGD discovery now accepts only JSON/JSONL data files. Empty, malformed,
metadata-only, and unsupported files are skipped rather than aborting the
combined ingestion.

JSON, JSONL, UTF-8 BOM, and gzip-wrapped JSON/JSONL are supported.

Per-file diagnostics identify:

```text
format
valid/bad lines
empty files
unsupported roots
skip reasons
```


## Expected discovery

With the repository layout:

```text
data\
├── dstc8-schema-guided-dialogue\
├── ubuntu\
└── UD_GUM\
```

the discovery stage should report:

```text
sgd: N
multiwoz: N
ubuntu_dataset: 1
ubuntu_vocab: 1
```

UD_GUM is passed separately with `--ud-gum`.


## V498 progress / correctness changes

Ubuntu ingestion now reports every 10,000 pairs with:

```text
processed / total
percent
pairs/sec
ETA
main DB size
WAL size
SHM size
```

Writes are committed every 5,000 pairs and the WAL is passively checkpointed
periodically. This makes database growth visible while keeping the writer
bounded.

The Ubuntu pair key is now:

```text
(source_path, container_index, row_index)
```

because the three encoded containers reuse `row_index` starting at zero.


## V499 fixes

The ingest summary now distinguishes Ubuntu relational pairs from generic
utterances. The console reports:

```text
ubuntu_pairs
ubuntu_decoded_turns
ubuntu_token_rows
```

The warning checks Ubuntu's dedicated relational store instead of requiring
Ubuntu rows in the generic `utterances` table.

The assistant's static retrieval now blocks parser/grammar lemmas such as
`be`, `have`, and `do`, and blocks corpus metadata relations from becoming
answer evidence.

Conversational goals are resolved before static lexical retrieval, preventing a
WordNet definition of `be` from hijacking a question such as "I'm good and you?".

`inspect_memory.py` provides a compact database inventory.


V499 also handles reciprocal elliptical turns using the previous conversational
goal, so a response to:

```text
Assistant: Hello!
User: I'm good and you?
```

is generated from the conversational state rather than from the lexical
meaning of `be`.


## V500 response firewall

The architecture now has a hard public-output boundary.

Internal representations such as:

```text
universe --hypernym--> collection
```

or:

```text
candidate proposition
goal description
architecture / participant / realizer
```

cannot be returned to the user.

Knowledge edges are converted to surface-language candidates before planning,
and the final selected text passes a second public-output firewall after
realization.

The CLI sends cognitive trace to stderr and the actual assistant response to
stdout.

This makes internal diagnostics and conversation output separate streams.
