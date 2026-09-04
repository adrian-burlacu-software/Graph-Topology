"""V681-owned live runtime derived from the V679 chat/worker lifecycle."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import threading
from pathlib import Path

from .chat import run_chat_worker
from .memory import SharedCheckpoint
from .live_policy import PolicyProvider
from .workers import worker_main

SOURCE_RUNTIME_VERSION = "v679"
V681_RUNTIME_VERSION = "v681.8-native-sequential-capture-1"
CHAT_CAPABILITIES = {
    "attention_trace": {"available": True},
    "decision_only": {"available": True},
    "sequential_attention_capture": {
        "available": True,
        "reason": "",
    },
}


class NativeRuntime:
    """Starts V681's own chat and worker components with direct event emission."""
    def __init__(self, database, llm_model, session_dir, session_id, worker_count=10, mode="chat", attention_policy=""):
        self.session_dir, self.session_id = Path(session_dir), session_id
        self.events, self.stop_event, self.workers, self.thread = queue.Queue(), mp.Event(), [], None
        self._chat_event_line = 0
        self.policy_provider = PolicyProvider(attention_policy)
        self.args = argparse.Namespace(
            database=str(database), output=str(self.session_dir / "chat.json"),
            trace_output=str(self.session_dir / "chat_traces.jsonl"),
            memory_output=str(self.session_dir / "memory.json"),
            worker_log_dir=str(self.session_dir / "workers"),
            shared_memory=str(self.session_dir / "shared_memory.sqlite"), llm_model=str(llm_model),
            spacy_model="en_core_web_sm", mode=mode, max_hypotheses=12, goal_budget=40,
            per_node=60, max_depth=3, cache_entries=12000, checkpoint_seconds=300,
            worker_count=int(worker_count), total_workers=int(worker_count) + 1, seed=6815,
            batch_sleep=0.0, duration_seconds=0, composition_fanout=4, composition_max=2000,
            composition_max_depth=3, max_no_new_batches=1000, worker_query_batch_subjects=128,
            task_poll_seconds=.25, worker_id=int(worker_count), v681_session_id=session_id,
            attention_policy=str(attention_policy), policy_provider=self.policy_provider,
            stop_event=self.stop_event, experience_sink=self._emit_chat,
        )

    def start(self):
        self.session_dir.mkdir(parents=True, exist_ok=True)
        Path(self.args.worker_log_dir).mkdir(parents=True, exist_ok=True)
        checkpoint = SharedCheckpoint(self.args.shared_memory, self.args.worker_count,
                                      self.args.total_workers, self.args.checkpoint_seconds)
        checkpoint.close()
        worker_args = argparse.Namespace(**vars(self.args))
        del worker_args.experience_sink
        del worker_args.policy_provider
        for worker_id in range(self.args.worker_count):
            process = mp.Process(target=worker_main, args=(worker_args, worker_id, self.stop_event),
                                 name=f"v681-worker-{worker_id:02d}", daemon=True)
            process.start(); self.workers.append(process)
        self.thread = threading.Thread(target=run_chat_worker, args=(self.args,),
                                       name="v681-chat", daemon=True)
        self.thread.start()

    def running(self):
        return bool(self.thread and self.thread.is_alive())

    def poll(self):
        chat = []
        while True:
            try: chat.append(self.events.get_nowait())
            except queue.Empty: break
        return {"chat": chat, "worker": list(self._worker_events())}

    def stop(self):
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=3)
        for process in self.workers: process.join(timeout=2)
        for process in self.workers:
            if process.is_alive(): process.terminate()
        for process in self.workers: process.join(timeout=1)

    def set_attention_policy(self, path):
        self.policy_provider.set(path)

    @staticmethod
    def capabilities():
        return {name: dict(value) for name, value in CHAT_CAPABILITIES.items()}

    def _emit_chat(self, trace):
        self._chat_event_line += 1
        value = trace if str(trace.get("version", "")).startswith("v681") else {
            **trace, "v681_session_id": self.session_id,
        }
        self.events.put({"source_path": "native-chat", "line": self._chat_event_line,
                         "value": value})

    def _worker_events(self):
        for path in sorted(Path(self.args.worker_log_dir).glob("worker_*.jsonl")):
            key = str(path.resolve())
            previous = getattr(self, "_offsets", {}).get(key, 0)
            if not hasattr(self, "_offsets"): self._offsets = {}
            for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if line_number <= previous or not line.strip(): continue
                self._offsets[key] = line_number
                try:
                    yield {"source_path": key, "line": line_number, "value": json.loads(line)}
                except ValueError:
                    yield {"source_path": key, "line": line_number, "invalid": "malformed worker event"}
