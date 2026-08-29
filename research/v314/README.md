
# V314 — Structured Working Memory / Multi-Variable Latent State

V313 showed that moving hypothesis competition around did not solve the
interference problem.

V314 attacks the deeper representation failure directly.

Instead of:

```text
memory_bit
cue1
cue2
cue3
hypothesis_bit
```

the system maintains:

```text
entity
role
relation
object
context
value
confidence
alternate_state
```

## Architecture

```text
graph
  ↓
persistent memory
  ↓
structured working memory
  ├── goal entity
  ├── relations
  ├── typed cue bindings
  ├── context
  ├── hypothesis
  ├── confidence
  └── alternate state
  ↓
structured decision
  ↓
feedback
```

The critical design choice is:

```text
DO NOT XOR/COLLAPSE ALL INFORMATION EARLY.
```

The decision can use the binding that actually belongs to the current role
and relation.

## Configurations

```text
structured_balanced
structured_goal
structured_context
structured_alternate
```

## Smoke

```powershell
python .\research\v314\validate.py
python .\research\v314\search.py --seeds 4 --episodes 8 --horizon 9
```

## Full

```powershell
python .\research\v314\search.py --seeds 12 --episodes 16 --horizon 9
```
