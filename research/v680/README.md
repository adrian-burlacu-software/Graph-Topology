# V680 — Learned Attention Policy

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
bounded-action masking losses. DAgger retrains after each aggregate round and
records learner-induced states labeled by the frozen teacher. PPO collects
stochastic rollouts before clipped minibatch updates with GAE, value loss,
entropy, and teacher-KL regularization.

The experiment artifacts are `teacher_trajectories.jsonl`,
`distillation_dataset.jsonl`, `dagger_dataset.jsonl`, `student_checkpoint.pt`,
`ppo_checkpoint.pt`, `evaluation.json`, and `attention_traces.jsonl`. Each
trajectory step records its input state, bounded candidates, teacher
distribution, selected action, resulting state, reward, and terminal outcome.

The built-in episodes include an ordinary direct-proof case, a structurally
held-out bicycle/wheel case, and an adversarial no-proof case containing lexical,
association, generic-relation, contradiction, weak-direct, and unrelated-match
traps. Its correct terminal action is `ABSTAIN`.

## Runlines

Generate frozen-teacher trajectories:

```powershell
python .\research\v680\attention_dataset.py --output ".\results\v680\distillation_dataset.jsonl"
```

Add `--database ".\data\v679_focused_semantic.sqlite"` to derive ordinary
direct-proof episodes from the frozen semantic graph; the adversarial no-proof
episode is retained in that dataset.

Train the distilled student:

```powershell
python .\research\v680\attention_evaluate.py --dataset ".\results\v680\distillation_dataset.jsonl" --checkpoint ".\results\v680\student_checkpoint.pt" --output ".\results\v680\evaluation.json"
```

Evaluate the student, reporting ordinary, adversarial, and held-out structural results separately:

```powershell
python .\research\v680\attention_distill.py --dataset ".\results\v680\distillation_dataset.jsonl" --checkpoint ".\results\v680\student_checkpoint.pt" --epochs 8 --seed 7
```

Run iterative DAgger:

```powershell
python .\research\v680\attention_dagger.py --dataset ".\results\v680\distillation_dataset.jsonl" --rounds 2 --epochs 8 --seed 7 --checkpoint-dir ".\results\v680\dagger"
```

Run PPO:

```powershell
python .\research\v680\attention_ppo.py --student-checkpoint ".\results\v680\student_checkpoint.pt" --checkpoint ".\results\v680\ppo_checkpoint.pt" --episodes 8 --seed 7 --ppo-epochs 4 --teacher-kl-coef 0.05
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
