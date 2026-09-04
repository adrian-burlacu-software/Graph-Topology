# V681.8 self-contained learning runtime

## Normal operation

```powershell
python -m research.v681.run_v681
```

V681 owns its runtime and learning implementation under `research/v681/`.
The runtime starts V681-native chat and offline workers, appends direct chat and
worker events to its SQLite experience store, batches eligible learning,
evaluates candidates, promotes passing candidates atomically, hot-swaps the
current attention policy, and writes `results/v681/runtime_results.json`,
`runtime_report.md`, `experience_manifest.json`, and `current_model_manifest.json`.
It also writes a per-session `v681_session_manifest.json` and the stable
aliases `v681_latest_results.json` and `v681_latest_report.md`.

No path to a chat trace, worker log, dataset, or checkpoint is needed. The
runtime uses `GRAPH_TOPOLOGY_LLM_MODEL` when set, otherwise expects the local
model at `llm/SmolLM3-3B`. If a runtime component is absent, V681 writes its
precise capability failure and continues with the components that are available.

The native runtime is derived from the stable V679 chat/worker implementation;
the native learning package is derived from the stable V680 attention/DAgGER/JEPA
implementation. Their source versions are recorded in artifacts. V681 has no
runtime import or dynamic loading dependency on either historical package.
Chat records are classified as `decision_only` unless actual sequential
transitions exist. V681 never fabricates steps. Worker events remain
knowledge-only and are never attention labels.

Training is batched: V681 bootstraps native DAgGER when fewer than eight
trainable sequential episodes exist, then trains attention and auxiliary JEPA
candidates. A durable learner cursor prevents duplicate training after restart:
no new eligible experience means no retraining or expensive evaluation. A
new live sequential trajectory is copied once into an immutable train dataset
view; its original `live` record is never changed.
candidate is promoted only after valid held-out evaluation passes the
conservative gate; otherwise the current model remains active. Live model
scoring uses only observable state/candidate features and retains
verified-evidence abstention safety.

Attention imitation accepts only sequential `DAGGER`, `SYNTHETIC_CHAT`, and
`CHAT_SEQUENTIAL` records. Offline worker telemetry, held-out evaluation, and
decision-only chat traces cannot become attention labels. JEPA receives a
separate observable transition dataset of `state`, `action`, and `next_state`
records; it remains auxiliary and is not routed directly into live attention
logits.

The runtime reports native chat capabilities at startup. It currently emits
attention and decision-only traces, but does not emit `CHAT_SEQUENTIAL`
transitions: the live graph controller exposes ranked traversal targets and a
final arbitration result, not one bounded policy state/action/next-state tuple
per traversal. The limitation is recorded as
`sequential_attention_capture: unavailable`; no trajectory is reconstructed
from final chat text.

If candidate learning fails for a dataset/configuration version, V681 records
the learner, version, failure, and timestamp, then suppresses repeated
attempts for that running session. A new eligible record, configuration change,
restart, or `--retry-failed-learning` permits another attempt. Runtime reports
separately list bootstrap, candidate training, JEPA, worker telemetry, model
creation/evaluation/promotion, and failures.

## Promotion safety gate

Promotion evaluates the candidate on the held-out structural and adversarial
rollout set. It requires at least 0.75 `overall_action_accuracy`, at least
0.80 `abstain_accuracy`, and no more than 0.10 each of
`false_positive_traverse`, `premature_stop`, and `premature_abstain`. Missing
any required rollout metric rejects the candidate.

The native training corpus also adds train-only matched STOP boundaries:
verified STOP versus abstain, and verified STOP versus useful traversal. Their
candidate order is rotated during generation and cyclically augmented while
fitting; the held-out structural and adversarial suites are unchanged.
Evaluation artifacts report action distributions by source plus STOP confusion,
precision, recall, and F1 alongside the existing rollout metrics.

When a promoted attention model exists, it is evaluated on the identical
held-out set. A candidate may lose at most 0.02 overall accuracy and 0.01 on
`abstain_accuracy`; each of the three safety error rates may increase by at
most 0.01. The first learned model has no relative baseline but must pass every
absolute check. Each candidate provenance artifact records the metrics,
thresholds, individual checks, promotion result, and reason.

```powershell
# One-shot CI/developer lifecycle: discover, collect, learn, evaluate, report, exit.
python -m research.v681.run_v681 --once

# Discovery/report only; does not start chat or learners.
python -m research.v681.run_v681 --dry-run

# Package test discovery from repository root.
python -m unittest discover -s research\v681 -p "test_*.py"

# Native bounded runtime smoke test.
python -m research.v681.run_v681 --smoke

# Retry a failed learner for the current dataset/configuration version.
python -m research.v681.run_v681 --once --retry-failed-learning
```

The remaining options are diagnostics only, not the normal workflow.
