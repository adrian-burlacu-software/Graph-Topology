
# V349 — Explicit Latent-State Carrier / State Binding

V348 established that some constructor semantics are not present in ordinary
graph topology:

```text
counterfactual_mode
initial_rule
```

V349 binds those latent environment variables into explicit graph-native state.

```text
Episode latent state
      ↓
LatentStateCarrier
      ↓
latent:* graph nodes
      ↓
semantic state decoder
      ↓
transformation
```

The carrier never binds `answer_bit`.

## Explicit carrier fields

```text
latent:initial_rule
latent:rule_version
latent:counterfactual_mode
latent:active_rule
```

## Modes

```text
carrier_all
carrier_persistent
carrier_context
carrier_disabled
```

The full benchmark compares all four so the contribution of explicit latent
state binding is visible.

## Smoke

```powershell
python .\research\v349\validate.py
python .\research\v349\search.py --seeds 4 --episodes 8 --horizon 9
```

## Full

```powershell
python .\research\v349\validate.py
python .\research\v349\search.py --seeds 12 --episodes 16 --horizon 9
```
