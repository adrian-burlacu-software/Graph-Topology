import json
import ast
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent

from research.v681.experience import Experience, ExperienceQuality, ExperienceSource, ExperienceStore, chat_trace_experience, worker_batch_experience
from research.v681.importers import import_chat_traces, import_worker_logs
from research.v681.learners import JEPAAuxiliaryLearner, REGISTRY, capability_report
from research.v681.trajectory import AttentionTrajectoryAdapter, OutcomeTransitionAdapter
from research.v681.coordinator import RuntimePolicy, V681Coordinator, _promotion
from research.v681.native_learning.engine import NativeLearningEngine
from research.v681.native_runtime.chat import preflight_symbol_audit


def sequential_experience(source=ExperienceSource.DAGGER, split="train"):
    state = {"goal_relation": "is_a", "goal_terms": ["goal"], "current_focus": "a", "current_node": "a",
             "relation_features": {}, "candidate_features": [{"relation": "is_a", "target": "b"}],
             "relation_activation": {}, "candidate_activation": {}, "visited_nodes": [], "visited_relations": [],
             "attention_history": [], "step": 0, "remaining_budget": 2}
    return Experience(source, "episode", split=split, quality=ExperienceQuality.TEACHER_LABELLED,
        model_view={"sequence_capability": "sequential", "state": state, "next_state": {**state, "step": 1},
                    "selected_action": {"kind": "traverse", "candidate_id": 0},
                    "candidate_actions": [{"action": {"kind": "traverse", "candidate_id": 0}, "features": state["candidate_features"][0]}]},
        supervision={"teacher": {"logits": [1, 0, 0], "probabilities": [.6, .2, .2], "selected_action": 0, "outcome": "traverse"},
                     "outcome": {"kind": "verified_answer"}, "reward": 1.0},
        diagnostics={"oracle": {"valid_proof_edge": True}}, provenance={"graph_version": "snapshot-a"})


class ExperienceTests(unittest.TestCase):
    def test_promotion_gate_accepts_strong_candidate(self):
        decision = _promotion(_promotion_metrics(),
                              _promotion_metrics(overall_action_accuracy=.88, abstain_accuracy=.88,
                                                 false_positive_traverse=.06, premature_stop=.06,
                                                 premature_abstain=.06),
                              RuntimePolicy())
        self.assertTrue(decision["promoted"])
        self.assertTrue(decision["checks"]["non_regression"])

    def test_promotion_gate_accepts_strong_first_candidate(self):
        decision = _promotion(_promotion_metrics(), None, RuntimePolicy())
        self.assertTrue(decision["promoted"])
        self.assertTrue(decision["checks"]["non_regression"])
        self.assertIsNone(decision["current_metrics"])
        self.assertEqual(decision["safety_metrics"]["candidate"]["overall_action_accuracy"], .90)

    def test_promotion_gate_rejects_weak_candidate(self):
        decision = _promotion(_promotion_metrics(overall_action_accuracy=.5), None, RuntimePolicy())
        self.assertFalse(decision["promoted"])
        self.assertFalse(decision["checks"]["overall_accuracy"])

    def test_promotion_gate_rejects_bad_abstention(self):
        decision = _promotion(_promotion_metrics(abstain_accuracy=.5), None, RuntimePolicy())
        self.assertFalse(decision["promoted"])
        self.assertFalse(decision["checks"]["abstain_accuracy"])

    def test_promotion_gate_rejects_missing_safety_metric(self):
        metrics = _promotion_metrics()
        del metrics["rollout"]["held_out"]["premature_stop"]
        decision = _promotion(metrics, None, RuntimePolicy())
        self.assertFalse(decision["promoted"])
        self.assertIn("premature_stop", decision["reason"])

    def test_promotion_gate_rejects_current_model_safety_regression(self):
        decision = _promotion(_promotion_metrics(false_positive_traverse=.08),
                              _promotion_metrics(false_positive_traverse=.01), RuntimePolicy())
        self.assertFalse(decision["promoted"])
        self.assertFalse(decision["checks"]["non_regression"])

    def test_complete_canonical_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExperienceStore(Path(directory) / "experience.sqlite")
            item = sequential_experience(); item.model_view["graph_context"] = {"graph_version": "a"}
            item.diagnostics["external_raw"] = {"must": "not be model input"}
            store.append(item); store.close()
            store = ExperienceStore(Path(directory) / "experience.sqlite")
            self.assertEqual(store.load()[0].as_dict(), item.as_dict())
            store.close()

    def test_model_view_and_evaluation_safety(self):
        with self.assertRaises(ValueError):
            Experience(ExperienceSource.DAGGER, "x", model_view={"oracle": {}})
        with self.assertRaises(ValueError):
            Experience(ExperienceSource.ATTENTION_EVAL, "x", split="train")
        for forbidden in ("oracle", "ground_truth", "future_state", "teacher_action", "terminal_answer", "future_reward"):
            with self.assertRaises(ValueError, msg=forbidden):
                Experience(ExperienceSource.DAGGER, "x", model_view={"nested": {forbidden: "blocked"}})

    def test_decision_chat_and_worker_are_not_attention_trajectories(self):
        chat = chat_trace_experience({"timestamp": 1, "route": {"success": False}, "candidate_evidence": []})
        worker = worker_batch_experience({"worker_id": 2, "batch": 3, "lane": "synonym_structure"})
        report = capability_report([chat, worker], REGISTRY["attention_distillation"],
                                   {ExperienceSource.CHAT_DECISION_ONLY})
        self.assertFalse(report["supported"])
        self.assertEqual(chat.sequence_capability, "decision_only")
        self.assertEqual(worker.sequence_capability, "knowledge_only")

    def test_one_adapter_extracts_dagger_and_synthetic_chat(self):
        dagger, chat = sequential_experience(), sequential_experience(ExperienceSource.SYNTHETIC_CHAT)
        episodes, rejected = AttentionTrajectoryAdapter().extract(
            [dagger, chat], {ExperienceSource.DAGGER, ExperienceSource.SYNTHETIC_CHAT})
        self.assertEqual(len(episodes), 1)
        self.assertFalse(rejected)
        transition = OutcomeTransitionAdapter().extract([dagger])
        self.assertEqual(len(transition), 1)
        self.assertEqual(REGISTRY["attention_dagger"].training_mode, "bootstrap")
        self.assertEqual(REGISTRY["jepa_auxiliary"].training_mode, "predictive_auxiliary")

    def test_jepa_preparation_returns_episodes_and_rejections(self):
        first, second, third = sequential_experience(), sequential_experience(), sequential_experience()
        first.episode_id, second.episode_id, third.episode_id = "first", "second", "third"
        episodes, rejected = JEPAAuxiliaryLearner().prepare(
            [first, second, third], {ExperienceSource.DAGGER})
        self.assertEqual(len(episodes), 3)
        self.assertEqual(rejected, {})

    def test_file_importers_are_boundary_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); trace = root / "chat.jsonl"
            trace.write_text('{"timestamp":1,"route":{},"candidate_evidence":[],"semantic_decision":{}}\n')
            workers = root / "workers"; workers.mkdir()
            (workers / "worker_00.jsonl").write_text('{"event":"analysis_batch","worker_id":0,"batch":1}\n')
            store = ExperienceStore(root / "experience.sqlite")
            self.assertEqual(len(import_chat_traces(store, trace)["accepted"]), 1)
            self.assertEqual(import_worker_logs(store, workers), 1)
            store.close()

    def test_malformed_chat_trace_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"; path.write_text("not json\n")
            store = ExperienceStore(Path(directory) / "experience.sqlite")
            report = import_chat_traces(store, path)
            self.assertEqual(report["accepted"], [])
            self.assertEqual(len(report["rejected"]), 1)
            store.close()

    def test_worker_graph_context_never_becomes_attention_supervision(self):
        worker = worker_batch_experience({"worker_id": 0, "batch": 1, "lane": "graph_health_sampling",
                                          "before_graph_version": "snapshot-a", "after_graph_version": "snapshot-b"})
        self.assertEqual(worker.model_view["graph_context"]["before_graph_version"], "snapshot-a")
        self.assertNotIn("teacher", worker.supervision)
        self.assertIsNone(worker.model_view["knowledge_delta"])

    def test_native_engine_generates_teacher_records(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "records.jsonl"
            NativeLearningEngine().generate_teacher_records(output, 1)
            self.assertTrue(output.exists())
            self.assertIn("trajectory", json.loads(output.read_text().splitlines()[0]))

    def test_chat_preflight_symbol_audit_allows_local_callbacks(self):
        self.assertTrue(preflight_symbol_audit())

    def test_runtime_has_no_legacy_runtime_imports_or_dynamic_loading(self):
        root = HERE
        forbidden = ("v679", "v680")
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertFalse(any(name.name.startswith(forbidden) for name in node.names), path)
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse(node.module and node.module.startswith(forbidden), path)
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
                    self.assertFalse(node.value.attr == "path" and node.attr == "insert", path)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, {"__import__", "import_module"}, path)

    def test_one_command_coordinator_captures_fake_runtime_and_evaluates_candidate(self):
        class FakeChat:
            policy_path = ""
            def available(self): return True, ""
            def start(self): pass
            def running(self): return False
            def stop(self): pass
            def set_attention_policy(self, path): self.policy_path = str(path)
            def poll(self):
                return {"chat": [{"source_path": "fake-chat", "line": 1, "value": {
                    "timestamp": 1, "route": {"success": True}, "candidate_evidence": [],
                    "semantic_decision": {}, "v681_session_id": "fake-session"}}],
                    "worker": [{"source_path": "fake-worker", "line": 1, "value": {
                        "event": "analysis_batch", "timestamp": 1, "worker_id": 0, "batch": 1}}]}

        class FakeEngine:
            def generate_teacher_records(self, path, _samples):
                records = [{"episode_id": "train", "trajectory": [_step("ordinary")]},
                           {"episode_id": "heldout", "trajectory": [_step("held_out")]}]
                path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
            def run_dagger(self, records_path, directory, *_):
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / "dagger_aggregate_round_0.jsonl"
                path.write_text(json.dumps({"episode_id": "dagger", "trajectory": [_step("ordinary")]}) + "\n")
                return path
            def train_attention(self, _data, checkpoint, *_): checkpoint.write_text("candidate")
            def evaluate_attention(self, _data, _checkpoint, output):
                value = _promotion_metrics()
                output.write_text(json.dumps(value)); return value
            def train_jepa(self, _data, checkpoint, output, *_):
                checkpoint.write_text("jepa"); value = {"status": "ok"}; output.write_text(json.dumps(value)); return value

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repository"; root.mkdir()
            chat = FakeChat()
            result = V681Coordinator(root=root, output_dir=Path(directory) / "v681", engine=FakeEngine(),
                chat_runtime=chat, policy=RuntimePolicy(min_sequential_episodes=1, bootstrap_samples=1, epochs=1)).run(once=True)
            self.assertEqual(result["status"], "once")
            self.assertEqual(result["chat_episodes"], 1)
            self.assertEqual(result["worker_events"], 1)
            self.assertTrue(result["models_created"])
            self.assertTrue(result["training_cycles"][-1]["promotion"]["promoted"])
            self.assertTrue(chat.policy_path.endswith("current_attention.pt"))

    def test_failed_one_command_learning_preserves_captured_experience(self):
        class FakeChat:
            def available(self): return True, ""
            def start(self): pass
            def running(self): return False
            def stop(self): pass
            def poll(self):
                return {"chat": [{"source_path": "fake-chat", "line": 1, "value": {
                    "timestamp": 2, "route": {"success": False}, "candidate_evidence": [],
                    "semantic_decision": {}}}], "worker": []}
        class FailingEngine:
            def generate_teacher_records(self, *_): raise RuntimeError("simulated learner failure")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repository"; root.mkdir()
            result = V681Coordinator(root=root, output_dir=Path(directory) / "v681", engine=FailingEngine(),
                chat_runtime=FakeChat(), policy=RuntimePolicy(min_sequential_episodes=1)).run(once=True)
            self.assertEqual(result["experience_after"]["chat_records"], 1)
            self.assertEqual(result["models_created"], [])
            self.assertEqual(result["failures"][0]["stage"], "bootstrap")

    def test_learning_cursor_prevents_restart_retraining_without_new_experience(self):
        class EmptyChat:
            def start(self): pass
            def running(self): return False
            def stop(self): pass
            def poll(self): return {"chat": [], "worker": []}
        class CountingEngine:
            def __init__(self): self.generated = 0
            def generate_teacher_records(self, path, _):
                self.generated += 1
                path.write_text("\n".join(json.dumps(row) for row in [
                    {"episode_id": "train", "trajectory": [_step("ordinary")]},
                    {"episode_id": "heldout", "trajectory": [_step("held_out")]}]) + "\n")
            def run_dagger(self, _, directory, *__):
                directory.mkdir(parents=True, exist_ok=True); path = directory / "dagger_aggregate_round_0.jsonl"
                path.write_text(json.dumps({"episode_id": "dagger", "trajectory": [_step("ordinary")]}) + "\n"); return path
            def train_attention(self, _, checkpoint, *__): checkpoint.write_text("candidate")
            def evaluate_attention(self, _, __, output):
                value = _promotion_metrics(); output.write_text(json.dumps(value)); return value
            def train_jepa(self, _, checkpoint, output, *__):
                checkpoint.write_text("jepa"); output.write_text("{}"); return {}
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory) / "repository", Path(directory) / "v681"; root.mkdir()
            first = CountingEngine()
            V681Coordinator(root=root, output_dir=output, engine=first, chat_runtime=EmptyChat(),
                            policy=RuntimePolicy(min_sequential_episodes=1, bootstrap_samples=1)).run(once=True)
            second = CountingEngine()
            result = V681Coordinator(root=root, output_dir=output, engine=second, chat_runtime=EmptyChat(),
                                     policy=RuntimePolicy(min_sequential_episodes=1, bootstrap_samples=1)).run(once=True)
            self.assertEqual(first.generated, 1)
            self.assertEqual(second.generated, 0)
            self.assertTrue(any(cycle["status"] == "no_op" for cycle in result["training_cycles"]))


def _step(split):
    state = {"goal_relation": "is_a", "goal_terms": ["goal"], "current_focus": "a", "current_node": "a",
             "relation_features": {}, "candidate_features": [{"relation": "is_a", "target": "b"}],
             "relation_activation": {}, "candidate_activation": {}, "visited_nodes": [], "visited_relations": [],
             "attention_history": [], "step": 0, "remaining_budget": 2}
    return {"episode_id": "episode", "split": split, "state": state, "candidates": [
        {"action": {"kind": "traverse", "candidate_id": 0}, "features": state["candidate_features"][0]}],
        "teacher": {"logits": [1, 0, 0], "probabilities": [.6, .2, .2], "selected_action": 0, "outcome": "traverse"},
        "action": {"kind": "traverse", "candidate_id": 0}, "next_state": {**state, "step": 1},
        "reward": 1.0, "terminal_outcome": "success", "oracle": {"valid_proof_edge": True},
        "teacher_version": "fake", "dataset_version": "fake"}


def _promotion_metrics(**overrides):
    values = {
        "overall_action_accuracy": .90,
        "abstain_accuracy": .90,
        "false_positive_traverse": .05,
        "premature_stop": .05,
        "premature_abstain": .05,
        "decisions": 20,
    }
    values.update(overrides)
    return {"rollout": {"held_out": values}}


if __name__ == "__main__":
    unittest.main()
