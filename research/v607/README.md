# V607 — Semantic Chat Gateway (Boundary-Safe)

V607 fixes a class of API-boundary bugs exposed by V606.

The failure was:

```text
TypeError: 'Namespace' object is not iterable
```

The root cause was positional argument confusion:

```text
hypotheses(parse, context, vocab, max_n)
                       ^
                       Namespace accidentally passed here
```

V607 addresses the class of bug, not just the individual line:

1. `hypotheses()` validates that `vocab` is an iterable of strings.
2. Passing `argparse.Namespace` now raises a precise diagnostic.
3. Internal calls use keyword arguments at the semantic boundary.
4. The smoke path explicitly passes the relation vocabulary.
5. Startup validates the relation vocabulary contract.

Architecture remains:

```text
chat
  ↓
spaCy structural parse
  ↓
generic relation hypotheses
  ↓
global conditional attention
  ↓
bounded semantic graph search
  ↓
graph evidence
  ↓
trace
```

No previous V-version artifact is required.

## Smoke

```powershell
python .\research\v607\v607_semantic_chat_gateway.py --database ".\results\v562_kg_composition_audit.sqlite" --output ".\results\v607_smoke.json" --trace-output ".\results\v607_smoke_traces.jsonl" --prior-output ".\results\v607_smoke_prior.json" --spacy-model en_core_web_sm --mode smoke --max-hypotheses 12 --relation-vocabulary 200 --goal-budget 40 --budget 80 --per-node 60 --max-depth 3 --cache-entries 12000 --prior-decay 0.65 --progress-every 1
```

## Interactive

```powershell
python .\research\v607\v607_semantic_chat_gateway.py --database ".\results\v562_kg_composition_audit.sqlite" --output ".\results\v607_semantic_chat.json" --trace-output ".\results\v607_chat_traces.jsonl" --prior-output ".\results\v607_global_attention_prior.json" --spacy-model en_core_web_sm --mode chat --max-hypotheses 12 --relation-vocabulary 200 --goal-budget 40 --budget 80 --per-node 60 --max-depth 3 --cache-entries 12000 --prior-decay 0.65 --progress-every 1
```

The console summary remains copy/paste friendly.
