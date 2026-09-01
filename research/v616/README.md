# V616 — Frozen spaCy + DB-Native Entity Resolution

V616 fixes the semantic-chat bottleneck exposed by V615.

The grammar remains completely frozen:

```text
spaCy parser = fixed
grammar training = OFF
```

The new layer is database-native canonical entity resolution:

```text
spaCy entity mention
        ↓
semantic graph entity resolver
        ↓
canonical graph node
        ↓
actual outgoing semantic relations
        ↓
goal candidates
        ↓
goal proof
```

No LLM and no new grammar training variable are introduced.

## Resolution strategy

The resolver uses only the semantic graph database:

```text
1. exact subject match
2. exact object match
3. bounded subject/object substring fallback
```

The canonical subject is retained in persistent context so pronouns can reuse
it on later turns.

## Cognitive stack

```text
frozen spaCy structure
        ↓
canonical entity resolution
        ↓
context-conditioned relation attention
        ↓
semantic goal
        ↓
goal-conditioned path prior
        ↓
bounded graph proof
        ↓
memory + attention learning
```

## Direct proof

Exact goal edges are checked before bridge search:

```text
(subject, requested_relation, object)
        ↓
verified proof
```

A verified one-edge path trains both attention systems.

## Smoke

The smoke test dynamically finds a real graph subject/relation pair, so it does
not depend on a fictional example entity being present in the database.

```powershell
python .\research\v616\v616_semantic_chat_gateway.py --database ".\results\v562_kg_composition_audit.sqlite" --output ".\results\v616_smoke.json" --trace-output ".\results\v616_smoke_traces.jsonl" --prior-output ".\results\v616_smoke_prior.json" --memory-output ".\results\v616_memory.json" --spacy-model en_core_web_sm --mode smoke --max-hypotheses 12 --relation-vocabulary 200 --goal-budget 40 --budget 80 --per-node 60 --max-depth 3 --cache-entries 12000 --prior-decay 0.65 --seed 61600 --progress-every 1
```

The console summary reports entity resolution, direct proofs, relation-attention
updates, path-attention updates, and persistent memory counts.
