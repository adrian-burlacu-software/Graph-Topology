"""File-format adapters at the V681 boundary; V679 processes remain unmodified."""
from __future__ import annotations

import json
from pathlib import Path

from experience import ExperienceSource, chat_trace_experience, worker_batch_experience


def import_chat_traces(store, path, source=ExperienceSource.CHAT):
    """Append V679 JSONL traces after a chat run; never opens a chat runtime."""
    count = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            store.append(chat_trace_experience(json.loads(line), source=source))
            count += 1
    return count


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
