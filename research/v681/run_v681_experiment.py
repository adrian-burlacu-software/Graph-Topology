"""V681 unified experience collection and offline-learning smoke/full experiment."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V680 = HERE.parent / "v680"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(V680))
from experience import ExperienceSource, ExperienceStore, attention_step_experience, chat_trace_experience
from importers import import_chat_traces, import_worker_logs
from learners import AttentionDistillationLearner, JEPAAuxiliaryLearner
from attention_benchmark import decision_boundary_episodes
from attention_dataset import collect_jepa_transition_episodes, collect_teacher_episodes


def main():
    parser = argparse.ArgumentParser(description="V681 unified learning/experience experiment")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chat-traces", default="")
    parser.add_argument("--worker-logs", default="")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    store = ExperienceStore(output / "v681_experience.sqlite")
    specs = decision_boundary_episodes(5 if args.smoke else 100)
    teacher = collect_teacher_episodes(specs)
    for episode in teacher:
        for step in episode["trajectory"]:
            store.append(attention_step_experience(step))
    chat_count = worker_count = 0
    if args.chat_traces:
        chat_count = import_chat_traces(store, args.chat_traces)
    else:
        # A schema-only fixture, deliberately not a production outcome or attention-training example.
        store.append(chat_trace_experience({"timestamp": 0, "route": {}, "candidate_evidence": []},
                                           source=ExperienceSource.SYNTHETIC_CHAT))
        chat_count = 1
    if args.worker_logs:
        worker_count = import_worker_logs(store, args.worker_logs)
    attention = AttentionDistillationLearner()
    attention_data = attention.prepare(store)
    model, _ = attention.train(attention_data, epochs=args.epochs, seed=args.seed)
    attention_metrics = attention.evaluate(teacher, model)
    # JEPA consumes sequential transitions only; it remains an auxiliary report.
    jepa_records = collect_jepa_transition_episodes(specs)
    for episode in jepa_records:
        for step in episode["trajectory"]:
            store.append(attention_step_experience(step, source=ExperienceSource.JEPA))
    jepa = JEPAAuxiliaryLearner()
    jepa_model, _ = jepa.train(jepa_records, epochs=args.epochs, seed=args.seed)
    results = {
        "experiment": "v681-learning-integration-1", "seed": args.seed, "epochs": args.epochs,
        "attention": attention_metrics, "jepa_auxiliary": jepa.evaluate(jepa_records, jepa_model),
        "source_comparisons": {
            "dagger_only": "evaluated", "chat_only": "unsupported: chat traces do not yet expose V680 observations",
            "offline_only": "unsupported: worker evidence is knowledge, not attention labels",
            "dagger_chat": "pending compatible live attention traces", "dagger_offline": "pending graph-snapshot benchmark",
            "chat_offline": "unsupported: no compatible attention labels", "combined": "pending compatible source data"},
        "promotion": {"promoted": False, "reason": "V681 collects reproducible experience; no model auto-promotion."},
        "jepa_role": "auxiliary predictive representation learner; never a direct attention-logit input",
    }
    manifest = store.manifest(); store.close()
    (output / "v681_learning_integration_results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    (output / "v681_experience_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (output / "v681_learning_integration_report.md").write_text(
        "# V681 learning integration\n\n"
        f"Experience store: `{output / 'v681_experience.sqlite'}`.\n\n"
        f"- Chat records: {chat_count} (synthetic only when no trace file is supplied)\n"
        f"- Offline worker batch evidence: {worker_count}\n"
        "- DAgGER bootstrap is teacher-labelled and remains distinct from verified outcomes.\n"
        "- JEPA trains only on sequential transitions and is reported separately as auxiliary learning.\n"
        "- No online weight update or automatic promotion occurs.\n"
    )


if __name__ == "__main__":
    main()
