
# V328 — Explicit Answer Criterion

V327 made goal semantics explicit, but a semantically valid constructed state
could still be treated as a correct answer.

V328 introduces a second explicit object:

```text
AnswerCriterion
    source
    transform
    comparison
    operands
    truth_condition
```

The architecture is now:

```text
Graph
  ↓
StateEstimator
  ↓
CognitiveState
  ↓
GoalPlan
  ↓
semantic state construction
  ↓
GoalSemantics verifier
  ↓
valid constructed state
  ↓
AnswerCriterion evaluator
  ↓
correct answer
```

The crucial distinction is:

```text
state is valid
    ≠
answer is valid
```

For example, a counterfactual state may contain both actual and alternate worlds,
yet still fail unless the final result is explicitly grounded in the alternate
world required by the question.

## Smoke

```powershell
python .\research\v328\validate.py
python .\research\v328\search.py --seeds 4 --episodes 8 --horizon 9
```

## Full

```powershell
python .\research\v328\search.py --seeds 12 --episodes 16 --horizon 9
```
