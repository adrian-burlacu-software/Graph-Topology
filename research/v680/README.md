# V680.1 — Learned Attention Policy

V680 is a self-contained learning experiment built from the frozen V679
symbolic controller. It does not modify V679, semantic workers, composition
depth, graph merging, or graph truth. The graph-style evidence environment
offers bounded candidates; its private oracle is used only for reward and
evaluation.

Install the experiment dependency before training:

```powershell
python -m pip install -r .\research\v680\requirements.txt
```

## Architecture

`AttentionObservation` and `CandidateFeatures` are JSON-serializable. At each
state the action set is `TRAVERSE(candidate_i)`, `STOP`, and `ABSTAIN`; no
global node action vocabulary exists. The frozen `V679AttentionTeacher` emits
a softened distribution over every available action. The recurrent PyTorch
student uses state and candidate feature encoders, a GRU state layer, a
candidate-logit head, and a value head.

Distillation uses configurable soft-teacher KL, pairwise rank, hard-label, and
bounded-action masking losses. DAgger retrains after each aggregate round,
records learner-induced states labeled by the frozen teacher, and reports its
held-out no-proof results separately.

`AttentionJEPA` is a separate action-conditioned representation learner:
`encoder(S_t) + action_embedding(A_i) -> predicted target-encoder Z_(t+1)`.
Its target encoder is an EMA-stabilized, gradient-free copy of the context
encoder. It predicts representations, never raw graph observations or hidden
proof fields. PPO collects stochastic rollouts before clipped minibatch updates
with GAE, value loss, entropy, and teacher-KL regularization.

The experiment artifacts are `teacher_trajectories.jsonl`,
`distillation_dataset.jsonl`, `dagger_dataset.jsonl`, `student_checkpoint.pt`,
`ppo_checkpoint.pt`, `evaluation.json`, and `attention_traces.jsonl`. Each
trajectory step records its input state, bounded candidates, teacher
distribution, selected action, resulting state, reward, and terminal outcome.

The built-in corpus has ordinary and held-out structural episodes plus 39
adversarial episodes across disjoint train, validation, and held-out graph
configurations. Categories cover no proof, lexical/wrong-subject/wrong-relation
traps, associations, unsupported candidates, contradiction, longer paths,
worker-only evidence, redundancy, direct-vs-indirect competition, STOP, and
ABSTAIN. The no-proof subset is explicitly partitioned for generalization.

## V680.1 runlines

Run the full matched decision-boundary suite (100 examples/category; descriptive
single-seed output in `v680_1_results.json` and `v680_1_report.txt`):

```powershell
python .\research\v680\run_v680_experiment.py --output-dir ".\results\v680_1\full_seed7" --seed 7 --epochs 8 --rounds 4
```

Run the small smoke suite instead:

```powershell
python .\research\v680\run_v680_experiment.py --output-dir ".\results\v680_1\smoke_seed7" --seed 7 --epochs 2 --rounds 4 --smoke
```

Run the frozen five-seed JEPA causal audit. It writes
`v680_1_jepa_causal_results.json`, `v680_1_jepa_causal_report.md`, and
`v680_1_dagger_report.json`, including matched zero/fixed-random/per-state
random/per-sample-random/action-shuffled/dimension-permuted controls:

```powershell
python .\research\v680\run_v680_jepa_causal.py --output-dir ".\results\v680_1\causal" --seeds 1,2,3,4,5 --epochs 8 --samples-per-category 100 --dagger-rounds 4
```

Use this reduced command only as a wiring check:

```powershell
python .\research\v680\run_v680_jepa_causal.py --output-dir ".\results\v680_1\causal_smoke" --seeds 1 --epochs 1 --dagger-rounds 1 --smoke
```

PPO is blocked unless the recorded gate passes. This command makes an explicitly
labelled smoke-only PPO run:

```powershell
python .\research\v680\run_v680_experiment.py --output-dir ".\results\v680_1\ppo_smoke" --seed 7 --epochs 8 --rounds 4 --phases distillation,dagger,jepa,evaluation,ppo --ppo-smoke --ppo-episodes 2
```

## Individual phases

Generate frozen-teacher trajectories:

```powershell
python .\research\v680\attention_dataset.py --output ".\results\v680\distillation_dataset.jsonl"
```

Add `--database ".\data\v679_focused_semantic.sqlite"` to derive ordinary
direct-proof episodes from the frozen semantic graph; the adversarial no-proof
episode is retained in that dataset.

Train the distilled student:

```powershell
python .\research\v680\attention_distill.py --dataset ".\results\v680\distillation_dataset.jsonl" --checkpoint ".\results\v680\student_checkpoint.pt" --epochs 8 --seed 7
```

Evaluate the student, reporting ordinary, adversarial, and held-out structural results separately:

```powershell
python .\research\v680\attention_evaluate.py --dataset ".\results\v680\distillation_dataset.jsonl" --checkpoint ".\results\v680\student_checkpoint.pt" --output ".\results\v680\evaluation.json"
```

Run iterative DAgger:

```powershell
python .\research\v680\attention_dagger.py --dataset ".\results\v680\distillation_dataset.jsonl" --rounds 2 --epochs 8 --seed 7 --checkpoint-dir ".\results\v680\dagger"
```

Train and validate JEPA dynamics:

```powershell
python .\research\v680\attention_dataset.py --jepa-transitions --output ".\results\v680\jepa_transitions.jsonl"
python .\research\v680\attention_jepa.py --dataset ".\results\v680\jepa_transitions.jsonl" --checkpoint ".\results\v680\jepa.pt" --output ".\results\v680\jepa_evaluation.json" --epochs 8 --seed 7
```

Train/evaluate a student augmented with frozen JEPA predictions:

```powershell
python .\research\v680\attention_distill.py --dataset ".\results\v680\distillation_dataset.jsonl" --checkpoint ".\results\v680\student_jepa.pt" --epochs 8 --seed 7 --use-jepa --jepa-checkpoint ".\results\v680\jepa.pt"
python .\research\v680\attention_evaluate.py --dataset ".\results\v680\distillation_dataset.jsonl" --checkpoint ".\results\v680\student_jepa.pt" --output ".\results\v680\jepa_evaluation.json" --use-jepa --jepa-checkpoint ".\results\v680\jepa.pt"
```

Run only the matched decision-boundary dataset:

```powershell
python .\research\v680\attention_dataset.py --decision-boundary --samples-per-category 100 --output ".\results\v680_1\decision_boundary_teacher.jsonl"
```

Run DAgGER in raw and explicitly inverse-frequency-balanced modes:

```powershell
python .\research\v680\attention_dagger.py --dataset ".\results\v680_1\decision_boundary_teacher.jsonl" --rounds 4 --epochs 8 --seed 7 --checkpoint-dir ".\results\v680_1\dagger_raw" --raw-class-loss
python .\research\v680\attention_dagger.py --dataset ".\results\v680_1\decision_boundary_teacher.jsonl" --rounds 4 --epochs 8 --seed 7 --checkpoint-dir ".\results\v680_1\dagger_balanced"
```

Run leak-free observation ablations:

```powershell
python .\research\v680\attention_ablation.py --dataset ".\results\v680\distillation_dataset.jsonl" --checkpoint ".\results\v680\student_checkpoint.pt" --output ".\results\v680\ablation.json"
```

V680 is a training experiment, not a chat runtime. Use the frozen V679 chat:

```powershell
python .\research\v679\v679_runtime.py --database ".\data\v679_focused_semantic.sqlite" --output ".\results\v679_chat.json" --trace-output ".\results\v679_chat_traces.jsonl" --memory-output ".\results\v679_memory.json" --worker-log-dir ".\results\v679_workers" --shared-memory ".\results\v679_shared_memory.sqlite" --spacy-model en_core_web_sm --llm-model "C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B" --mode chat --worker-count 12 --batch-sleep 0
```

Worker summarization remains the frozen V679 command:

```powershell
python .\research\v679\v679_worker_summary.py --worker-log-dir ".\results\v679_workers" --shared-memory ".\results\v679_shared_memory.sqlite" --output ".\results\v679\worker_summary.jsonl"
```
