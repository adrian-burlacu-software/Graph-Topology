
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
