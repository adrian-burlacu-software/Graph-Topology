"""Operational V681.2 substrate orchestration; V680 is reached only by adapter."""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from .experience import ExperienceQuality, ExperienceSource, ExperienceStore, attention_step_experience
from .importers import import_chat_traces, import_worker_logs
from .learners import AttentionDistillationLearner, JEPAAuxiliaryLearner, REGISTRY, capability_report
from .v680_adapter import V680_ENGINE_VERSION, V680EngineAdapter

V681_VERSION = "v681.2-learning-substrate-1"


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path, records):
    Path(path).write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))


def _ingest_engine_records(store, records, source, episode_prefix="", heldout_only=False):
    for episode in records:
        for step in episode["trajectory"]:
            if heldout_only and not str(step["split"]).startswith("held_out"):
                continue
            item = attention_step_experience(step, source=source)
            item.episode_id = episode_prefix + item.episode_id
            item.provenance["supervision_source"] = "frozen_v680_teacher"
            store.append(item)


def _dataset_manifest(store, experiences, name, sources):
    selected = [item for item in experiences if item.source in sources]
    episodes = len({item.episode_id for item in selected})
    counts = {}
    for item in selected:
        counts[item.source.value] = counts.get(item.source.value, 0) + 1
    quality_counts, split_counts = {}, {}
    for item in selected:
        quality_counts[item.quality.value] = quality_counts.get(item.quality.value, 0) + 1
        split_counts[item.split] = split_counts.get(item.split, 0) + 1
    return {"dataset_version": f"{V681_VERSION}:{name}", "sources": sorted(source.value for source in sources),
            "records": len(selected), "episodes": episodes, "source_counts": counts,
            "quality_counts": quality_counts, "split_counts": split_counts,
            "graph_versions": sorted({item.provenance.get("graph_version", "unknown") for item in selected}),
            "teacher_versions": sorted({item.provenance.get("teacher_version", "unknown") for item in selected}),
            "store": store.manifest()}


def _run_attention_comparison(name, sources, experiences, engine, output, epochs, seed, min_quality=None):
    learner = AttentionDistillationLearner()
    availability = capability_report(experiences, learner.descriptor, sources)
    if not availability["supported"]:
        return {"supported": False, **availability}
    train, rejected_train = learner.prepare(experiences, sources, min_quality=min_quality)
    evaluation, rejected_eval = learner.prepare(experiences, {ExperienceSource.ATTENTION_EVAL},
                                                allowed_splits=("heldout",), min_quality=min_quality)
    if not train or not evaluation:
        return {"supported": False, "reason": "requires both train and heldout sequential attention experience",
                "records": len(train), "rejected": {"train": rejected_train, "heldout": rejected_eval}}
    dataset = output / f"{name}_train.jsonl"; evaluation_path = output / f"{name}_heldout.jsonl"
    checkpoint, metrics = output / f"{name}.pt", output / f"{name}_metrics.json"
    _write_jsonl(dataset, train); _write_jsonl(evaluation_path, evaluation)
    learner.train(engine, dataset, checkpoint, epochs, seed)
    artifact_provenance = {
        "learner_type": learner.descriptor.learner_type, "artifact_type": learner.descriptor.artifact_type,
        "dataset_version": f"{V681_VERSION}:{name}", "sources": sorted(source.value for source in sources),
        "v680_engine_version": V680_ENGINE_VERSION, "seed": seed, "training_configuration": {"epochs": epochs},
    }
    provenance_path = output / f"{name}.provenance.json"
    provenance_path.write_text(json.dumps(artifact_provenance, indent=2, sort_keys=True))
    return {"supported": True, "dataset": str(dataset), "checkpoint": str(checkpoint),
            "artifact_provenance": str(provenance_path),
            "manifest": {"train_episodes": len(train), "heldout_episodes": len(evaluation),
                         "train_rejected": rejected_train, "heldout_rejected": rejected_eval},
            "metrics": learner.evaluate(engine, evaluation_path, checkpoint, metrics)}


def _inspect(store):
    items = store.load()
    return {"total_records": len(items), "episodes": len({item.episode_id for item in items}),
            "sources": {source.value: sum(item.source is source for item in items) for source in ExperienceSource},
            "qualities": {quality.value: sum(item.quality is quality for item in items) for quality in type(items[0].quality)} if items else {},
            "splits": {split: sum(item.split == split for item in items) for split in ("train", "validation", "heldout", "live")},
            "attention_capable_records": sum(item.sequence_capability == "sequential" for item in items),
            "decision_only_records": sum(item.sequence_capability == "decision_only" for item in items),
            "worker_knowledge_records": sum(item.sequence_capability == "knowledge_only" for item in items)}


def _materialize_live(store, min_quality):
    """Copy reviewed live sequential experience into train; original live data remains immutable."""
    source = store.load(allowed_splits=("live",), min_quality=min_quality)
    created = 0
    for item in source:
        if item.sequence_capability != "sequential":
            continue
        value = item.as_dict(); value["experience_id"] = uuid.uuid4().hex; value["split"] = "train"
        value["provenance"] = {**value["provenance"], "materialized_from": item.experience_id,
                               "materialization_version": V681_VERSION}
        store.append(value); created += 1
    return created


def main():
    parser = argparse.ArgumentParser(description="Run/inspect the self-contained V681.2 learning substrate.")
    parser.add_argument("--output-dir", required=True); parser.add_argument("--v680-engine", default="")
    parser.add_argument("--chat-traces", default=""); parser.add_argument("--worker-logs", default="")
    parser.add_argument("--epochs", type=int, default=8); parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--inspect-experience", action="store_true")
    parser.add_argument("--min-quality", choices=[quality.value for quality in ExperienceQuality])
    parser.add_argument("--materialize-live", action="store_true",
                        help="Copy reviewed sequential live records into a new train version; never mutates live records.")
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    engine = V680EngineAdapter(args.v680_engine or None)
    store = ExperienceStore(output / "v681_experience.sqlite")
    if args.inspect_experience:
        report = _inspect(store); (output / "v681_experience_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(report, indent=2, sort_keys=True)); store.close(); return
    samples = 5 if args.smoke else 100
    raw_teacher = output / "v680_teacher.jsonl"
    engine.generate_teacher_records(raw_teacher, samples)
    records = _read_jsonl(raw_teacher)
    _ingest_engine_records(store, records, ExperienceSource.ATTENTION_EVAL, "evaluation-", heldout_only=True)
    dagger_aggregate = engine.run_dagger(raw_teacher, output / "v680_dagger", 1, args.epochs, args.seed)
    _ingest_engine_records(store, _read_jsonl(dagger_aggregate), ExperienceSource.DAGGER, "dagger-")
    # This is a true sequential synthetic-chat corpus generated by the frozen
    # engine benchmark, never a fabricated production chat outcome.
    _ingest_engine_records(store, records, ExperienceSource.SYNTHETIC_CHAT, "synthetic-chat-")
    chat_records = import_chat_traces(store, args.chat_traces) if args.chat_traces else 0
    worker_records = import_worker_logs(store, args.worker_logs) if args.worker_logs else 0
    materialized = _materialize_live(store, args.min_quality or "unverified") if args.materialize_live else 0
    experiences = store.load()
    source_sets = {
        "dagger_only": {ExperienceSource.DAGGER},
        "chat_only": {ExperienceSource.CHAT_SEQUENTIAL, ExperienceSource.SYNTHETIC_CHAT},
        "dagger_chat": {ExperienceSource.DAGGER, ExperienceSource.CHAT_SEQUENTIAL, ExperienceSource.SYNTHETIC_CHAT},
    }
    comparisons = {name: _run_attention_comparison(name, sources, experiences, engine, output, args.epochs, args.seed,
                                                    args.min_quality)
                   for name, sources in source_sets.items()}
    jepa = JEPAAuxiliaryLearner()
    combined_sources = source_sets["dagger_chat"]
    jepa_records, rejected = jepa.prepare(experiences, combined_sources)
    jepa_data = output / "jepa_sequential.jsonl"; _write_jsonl(jepa_data, jepa_records)
    jepa_report = jepa.train(engine, jepa_data, output / "jepa_auxiliary.pt", output / "jepa_auxiliary_metrics.json",
                             args.epochs, args.seed)
    worker_items = [item for item in experiences if item.source is ExperienceSource.OFFLINE_WORKER]
    knowledge = {"supported": bool(worker_items), "records": len(worker_items),
                 "graph_versions": sorted({item.provenance.get("graph_version", "unknown") for item in worker_items}),
                 "reason": ("worker data is knowledge-only telemetry; frozen V680 does not expose graph-snapshot "
                            "injection, so no attention-label or gain is fabricated.")}
    worker_combinations = {
        name: {"supported": False, "records": len(worker_items),
               "reason": ("requires a V680 graph-snapshot injection API and verified worker knowledge deltas; "
                          + knowledge["reason"])}
        for name in ("offline_only", "dagger_offline", "chat_offline", "dagger_chat_offline")
    }
    manifest = {"v681_version": V681_VERSION, "v680_engine_version": V680_ENGINE_VERSION,
                "teacher_version": records[0]["teacher_version"], "engine_root": str(engine.root),
                "datasets": [_dataset_manifest(store, experiences, name, sources) for name, sources in source_sets.items()],
                "experience": store.manifest(), "inspection": _inspect(store)}
    results = {"v681_version": V681_VERSION, "v680_engine_version": V680_ENGINE_VERSION,
               "teacher_version": records[0]["teacher_version"], "chat_imported": chat_records,
               "worker_imported": worker_records, "materialized_live_records": materialized,
               "source_comparisons": {**comparisons, **worker_combinations},
               "jepa_auxiliary": {"metrics": jepa_report, "rejected": rejected,
                                  "role": "predictive auxiliary; direct attention gain not demonstrated"},
               "knowledge_conditioned": knowledge, "promotion": {"promoted": False, "reason": "explicit evaluation required"}}
    (output / "v681_experience_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (output / "v681_learning_integration_results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    comparison_lines = "\n".join(
        f"- `{name}`: {'supported' if value['supported'] else 'unsupported'}"
        + (f" ({value.get('reason')})" if value.get("reason") else "")
        for name, value in results["source_comparisons"].items()
    )
    (output / "v681_learning_integration_report.md").write_text(
        "# V681.2 learning integration\n\n"
        f"V681 version `{V681_VERSION}` uses explicit frozen engine `{V680_ENGINE_VERSION}` at `{engine.root}`.\n\n"
        "DAgGER and sequential synthetic/live chat traverse the same V681 trajectory adapter. "
        "Decision-only chat and worker knowledge are retained but excluded from attention imitation. "
        "JEPA is auxiliary; V680.1 direct attention gain was not demonstrated. PPO is not run.\n\n"
        "## Source combinations\n" + comparison_lines + "\n\n"
        "## Knowledge-conditioned evaluation\n" + knowledge["reason"] + "\n"
    )
    store.close()


if __name__ == "__main__":
    main()
