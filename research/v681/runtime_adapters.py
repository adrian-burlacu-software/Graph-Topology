"""Explicit adapters for repository-owned V679 chat and worker runtimes."""
from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepositoryRuntime:
    root: Path
    v679_runtime: Path
    v680_root: Path
    graph_database: Path | None
    llm_model: Path | None
    results_root: Path

    @classmethod
    def discover(cls, root=None):
        root = Path(root) if root else Path(__file__).resolve().parents[2]
        root = root.resolve()
        v679 = root / "research" / "v679" / "v679_runtime.py"
        v680 = root / "research" / "v680"
        graph = _discover_graph(root / "data")
        configured = os.environ.get("GRAPH_TOPOLOGY_LLM_MODEL", "").strip()
        model = Path(configured).expanduser() if configured else root / "llm" / "SmolLM3-3B"
        return cls(root, v679, v680, graph, model if model.is_dir() else None, root / "results" / "v681")

    def capabilities(self):
        return {
            "repository_root": str(self.root),
            "v679_runtime": _capability(self.v679_runtime, "V679 runtime script"),
            "v680_engine": _capability(self.v680_root / "attention_dataset.py", "frozen V680 engine"),
            "graph_database": _capability(self.graph_database, "V679 graph database"),
            "local_llm": _capability(self.llm_model, "local V679 LLM model"),
            "learned_policy_application": {
                "available": False,
                "reason": ("V679 accepts only its relation-bias JSON policy, while frozen V680 checkpoints require "
                           "a V679-to-V680 sequential observation adapter that the upstream runtime does not expose"),
            },
        }


class V679ChatRuntimeAdapter:
    """Runs the real V679 process and tails only its explicitly assigned outputs."""
    def __init__(self, runtime, session_dir, session_id, worker_count=10):
        self.runtime = runtime
        self.session_dir = Path(session_dir)
        self.session_id = session_id
        self.worker_count = worker_count
        self.trace_path = self.session_dir / "v679_chat_traces.jsonl"
        self.worker_dir = self.session_dir / "v679_workers"
        self._process = None
        self._offsets = {}

    def available(self):
        if not self.runtime.v679_runtime.is_file():
            return False, f"V679 runtime missing: expected {self.runtime.v679_runtime}"
        if self.runtime.graph_database is None:
            return False, f"V679 graph database missing: expected a readable *focused_semantic.sqlite in {self.runtime.root / 'data'}"
        if self.runtime.llm_model is None:
            return False, ("V679 local LLM missing: set GRAPH_TOPOLOGY_LLM_MODEL or place the model at "
                           f"{self.runtime.root / 'llm' / 'SmolLM3-3B'}")
        return True, ""

    def start(self):
        available, reason = self.available()
        if not available:
            raise RuntimeError(reason)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.worker_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(self.runtime.v679_runtime),
            "--database", str(self.runtime.graph_database),
            "--output", str(self.session_dir / "v679_chat.json"),
            "--trace-output", str(self.trace_path),
            "--memory-output", str(self.session_dir / "v679_memory.json"),
            "--worker-log-dir", str(self.worker_dir),
            "--shared-memory", str(self.session_dir / "v679_shared_memory.sqlite"),
            "--llm-model", str(self.runtime.llm_model),
            "--worker-count", str(self.worker_count),
            "--v681-session-id", self.session_id,
        ]
        self._process = subprocess.Popen(command, cwd=self.runtime.v679_runtime.parent)

    def running(self):
        return self._process is not None and self._process.poll() is None

    def poll(self):
        return {
            "chat": list(_tail_jsonl(self.trace_path, self._offsets)),
            "worker": [event for path in sorted(self.worker_dir.glob("worker_*.jsonl"))
                       for event in _tail_jsonl(path, self._offsets)],
        }

    def stop(self):
        if not self.running():
            return
        self._process.send_signal(signal.SIGINT)
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=5)


def _tail_jsonl(path, offsets):
    path = Path(path)
    if not path.is_file():
        return
    key = str(path.resolve())
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if line_number <= offsets.get(key, 0) or not line.strip():
                continue
            offsets[key] = line_number
            try:
                yield {"source_path": key, "line": line_number, "value": json.loads(line)}
            except json.JSONDecodeError:
                yield {"source_path": key, "line": line_number, "invalid": "malformed JSONL event"}


def _discover_graph(data_dir):
    candidates = sorted(data_dir.glob("v*_focused_semantic.sqlite"), reverse=True)
    for candidate in candidates:
        try:
            with sqlite3.connect(f"file:{candidate.resolve()}?mode=ro", uri=True) as conn:
                conn.execute("SELECT 1 FROM nodes LIMIT 1")
                conn.execute("SELECT 1 FROM edges LIMIT 1")
            return candidate.resolve()
        except sqlite3.Error:
            continue
    return None


def _capability(path, label):
    if path and Path(path).exists():
        return {"available": True, "path": str(path)}
    return {"available": False, "reason": f"{label} not found"}
