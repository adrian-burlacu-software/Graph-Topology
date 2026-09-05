"""File-format adapters at the V681 boundary; V679 processes remain unmodified."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

from .experience import Experience, ExperienceSource, chat_trace_experience, worker_batch_experience
from .native_learning.types import AttentionAction, AttentionObservation


def import_chat_traces(store, path, source=ExperienceSource.CHAT_DECISION_ONLY):
    """Validate and sanitize completed V679 traces; never constructs missing steps."""
    report = {"accepted": [], "rejected": []}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            report["accepted"].append(import_chat_record(store, raw, str(Path(path).resolve()), line_number, source))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            report["rejected"].append({"line": line_number, "reason": str(error)})
    return report


def import_chat_record(store, raw, source_path="runtime", line_number=0,
                       source=ExperienceSource.CHAT_DECISION_ONLY):
    """Append one canonical native transition or decision-only trace with a stable identity."""
    if not isinstance(raw, dict):
        raise ValueError("record is not an object")
    if raw.get("version", "").startswith("v681"):
        item = Experience.from_dict(raw)
        if item.source is not ExperienceSource.CHAT_SEQUENTIAL or item.sequence_capability != "sequential":
            raise ValueError("canonical chat record must declare sequence_capability=sequential")
        _validate_native_chat_transition(item)
        item.source, item.split = ExperienceSource.CHAT_SEQUENTIAL, "live"
    else:
        _validate_v679_chat_trace(raw)
        item = chat_trace_experience(raw, source=source)
    item.experience_id = _source_id("chat", source_path, line_number, raw)
    item.provenance = {**item.provenance, "source_path": source_path, "source_line": line_number}
    store.append(item)
    return _descriptor(item, line_number)


def import_worker_logs(store, directory):
    """Append offline evidence events after worker completion, not action labels."""
    count = 0
    for path in sorted(Path(directory).glob("worker_*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") == "analysis_batch":
                import_worker_event(store, event, str(path.resolve()), line_number)
                count += 1
    return count


def import_worker_event(store, event, source_path="runtime", line_number=0):
    """Append one analysis event as knowledge-only experience."""
    if not isinstance(event, dict) or event.get("event") != "analysis_batch":
        raise ValueError("worker event must be an analysis_batch object")
    item = worker_batch_experience(event)
    item.experience_id = _source_id("worker", source_path, line_number, event)
    item.provenance = {**item.provenance, "source_path": source_path, "source_line": line_number}
    store.append(item)
    return item.experience_id


def _validate_v679_chat_trace(value):
    if not isinstance(value.get("timestamp"), (int, float)):
        raise ValueError("missing numeric timestamp")
    if not isinstance(value.get("route"), dict):
        raise ValueError("missing route object")
    if value["route"].get("mode") == "conversation":
        return
    if "candidate_evidence" not in value or not isinstance(value["candidate_evidence"], list):
        raise ValueError("missing candidate_evidence list")
    if "semantic_decision" not in value or not isinstance(value["semantic_decision"], dict):
        raise ValueError("missing semantic_decision object")


def _validate_native_chat_transition(item):
    view = item.model_view
    state, next_state = view.get("state"), view.get("next_state")
    if not isinstance(state, dict) or not isinstance(next_state, dict):
        raise ValueError("canonical chat transition requires state and next_state")
    if not isinstance(state.get("step"), int) or next_state.get("step") != state["step"] + 1:
        raise ValueError("canonical chat transition steps must be ordered")
    observation = AttentionObservation.from_dict(state)
    AttentionObservation.from_dict(next_state)
    AttentionAction.from_dict(view.get("selected_action"), len(observation.candidate_features))
    candidates = view.get("candidate_actions")
    if not isinstance(candidates, list) or len(candidates) != len(observation.candidate_features):
        raise ValueError("canonical chat transition candidates must match its state")
    selected = view["selected_action"]
    if selected.get("kind") not in {"traverse", "stop", "abstain"}:
        raise ValueError("canonical chat transition must capture a bounded action")
    teacher = item.supervision.get("teacher", {})
    if (not isinstance(teacher, dict) or len(teacher.get("logits", [])) != len(candidates) + 2
            or len(teacher.get("probabilities", [])) != len(candidates) + 2
            or teacher.get("selected_action") != AttentionAction.from_dict(
                selected, len(observation.candidate_features)
            ).index(len(observation.candidate_features))):
        raise ValueError("canonical chat transition requires matching policy supervision")


def _descriptor(item, line_number):
    return {"line": line_number, "experience_id": item.experience_id, "source": item.source.value,
            "sequence_capability": item.sequence_capability, "split": item.split,
            "quality": item.quality.value, "episode_id": item.episode_id,
            "graph_version": item.provenance.get("graph_version", "unknown"),
            "provenance": item.provenance}


def _source_id(kind, source_path, line_number, payload):
    value = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{kind}:{source_path}:{line_number}:{value}".encode("utf-8")).hexdigest()[:32]
