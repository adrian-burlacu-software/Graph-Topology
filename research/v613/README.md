# V613 — Semantic Chat Gateway

V613 fixes two silent V612 bugs.

## 1. Split memory/context object

V612 created a second `Context` inside `smoke()`. The smoke path therefore
learned entity/turn state into one object while the saved `memory` object did
not receive those updates.

V613 uses the same `Context` instance throughout the run.

## 2. Inert global attention

V612 persisted relation outcomes but never trained the `Attention` object that
actually controls search. Its output could therefore legitimately show:

```text
attention updates = 0
```

V613 sends every verified successful semantic path through:

```text
goal + prefix + next_relation
        ↓
Attention.update()
```

The learned path is also persisted into semantic memory.

Architecture:

```text
chat
  ↓
spaCy structural parse
  ↓
intent / goal hypotheses
  ↓
persistent context
  ↓
global conditional attention
  ↓
semantic graph search
  ↓
verified result
  ├──> persistent semantic memory
  └──> global attention
```

## Smoke

```powershell
python .\research\v613\v613_semantic_chat_gateway.py --database ".\results\v562_kg_composition_audit.sqlite" --output ".\results\v613_smoke.json" --trace-output ".\results\v613_smoke_traces.jsonl" --prior-output ".\results\v613_smoke_prior.json" --memory-output ".\results\v613_memory.json" --spacy-model en_core_web_sm --mode smoke --max-hypotheses 12 --relation-vocabulary 200 --goal-budget 40 --budget 80 --per-node 60 --max-depth 3 --cache-entries 12000 --prior-decay 0.65 --seed 61300 --progress-every 1
```

The final console summary remains copy/paste friendly.
