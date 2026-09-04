"""File-format adapters at the V681 boundary; V679 processes remain unmodified."""
from __future__ import annotations

import json
from pathlib import Path

from .experience import Experience, ExperienceSource, chat_trace_experience, worker_batch_experience


def import_chat_traces(store, path, source=ExperienceSource.CHAT_DECISION_ONLY):
    """Validate and sanitize completed V679 traces; never constructs missing steps."""
    report = {"accepted": [], "rejected": []}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError("record is not an object")
            if raw.get("version", "").startswith("v681"):
                item = Experience.from_dict(raw)
                if item.sequence_capability != "sequential":
                    raise ValueError("canonical chat record must declare sequence_capability=sequential")
                item.source, item.split = ExperienceSource.CHAT_SEQUENTIAL, "live"
            else:
                _validate_v679_chat_trace(raw)
                item = chat_trace_experience(raw, source=source)
            store.append(item)
            report["accepted"].append(_descriptor(item, line_number))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            report["rejected"].append({"line": line_number, "reason": str(error)})
    return report


def import_worker_logs(store, directory):
    """Append offline evidence events after worker completion, not action labels."""
    count = 0
    for path in sorted(Path(directory).glob("worker_*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") == "analysis_batch":
                store.append(worker_batch_experience(event))
                count += 1
    return count


def _validate_v679_chat_trace(value):
    if not isinstance(value.get("timestamp"), (int, float)):
        raise ValueError("missing numeric timestamp")
    if not isinstance(value.get("route"), dict):
        raise ValueError("missing route object")
    if "candidate_evidence" not in value or not isinstance(value["candidate_evidence"], list):
        raise ValueError("missing candidate_evidence list")
    if "semantic_decision" not in value or not isinstance(value["semantic_decision"], dict):
        raise ValueError("missing semantic_decision object")


def _descriptor(item, line_number):
    return {"line": line_number, "experience_id": item.experience_id, "source": item.source.value,
            "sequence_capability": item.sequence_capability, "split": item.split,
            "quality": item.quality.value, "episode_id": item.episode_id,
            "graph_version": item.provenance.get("graph_version", "unknown"),
            "provenance": item.provenance}
