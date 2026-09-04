"""Canonical V681 experience adapters for frozen-engine trajectory inputs."""
from __future__ import annotations

from collections import Counter

from .experience import Experience, ExperienceQuality, ExperienceSource
from .native_learning.types import validate_jepa_transition_record


class AttentionTrajectoryAdapter:
    required_capability = "sequential"

    def extract(self, experiences, sources=None, allowed_splits=("train",), min_quality=None):
        episodes = {}
        rejected = Counter()
        selected = set(sources) if sources else None
        minimum = ExperienceQuality(min_quality) if min_quality else None
        quality_order = list(ExperienceQuality)
        for item in experiences:
            if selected and item.source not in selected: rejected["source"] += 1; continue
            if item.split not in allowed_splits: rejected["split"] += 1; continue
            if item.sequence_capability != self.required_capability: rejected[item.sequence_capability] += 1; continue
            if minimum and quality_order.index(item.quality) > quality_order.index(minimum):
                rejected["quality"] += 1; continue
            step = self._v680_step(item)
            episodes.setdefault(item.episode_id, {"episode_id": item.episode_id, "split": step["split"],
                                                   "partition": step.get("partition", ""), "trajectory": []})["trajectory"].append(step)
        return list(episodes.values()), dict(rejected)

    @staticmethod
    def _v680_step(item):
        """V681 owns mapping; no learner reads diagnostics/raw-source payload directly."""
        view, supervision = item.model_view, item.supervision
        teacher = supervision.get("teacher")
        if not teacher:
            raise ValueError("sequential attention experience needs teacher supervision for distillation")
        diagnostic = item.diagnostics.get("raw_v680_step", {})
        return {
            "episode_id": item.episode_id, "split": diagnostic.get("split", "ordinary"),
            "step": diagnostic.get("step", 0), "state": view["state"], "candidates": view["candidate_actions"],
            "teacher": teacher, "action": view["selected_action"], "next_state": view["next_state"],
            "reward": supervision.get("reward", 0.0), "terminal_outcome": diagnostic.get("terminal_outcome", "unknown"),
            "oracle": diagnostic.get("oracle", {}), "provenance": item.provenance,
            "source": item.source.value,
            "partition": diagnostic.get("partition", ""), "category": diagnostic.get("category", ""),
            "no_proof": diagnostic.get("no_proof", False),
            "teacher_version": item.provenance.get("teacher_version", "unknown"),
            "dataset_version": item.provenance.get("dataset_version", "v681"),
            "student_version": "v681-adapter", "jepa_version": "v680.1-action-conditioned-jepa-1",
        }


class SequentialTransitionAdapter:
    """JEPA consumes one observable state/action/next-state transition per step."""
    def extract(self, experiences, sources=None, allowed_splits=("train",)):
        transitions, rejected = [], Counter()
        selected = set(sources) if sources else None
        for item in experiences:
            if selected and item.source not in selected: rejected["source"] += 1; continue
            if item.split not in allowed_splits: rejected["split"] += 1; continue
            if item.sequence_capability != "sequential": rejected[item.sequence_capability] += 1; continue
            view = item.model_view
            try:
                transitions.append(validate_jepa_transition_record({
                    "episode_id": item.episode_id,
                    "step": view["state"]["step"],
                    "state": view["state"],
                    "action": view["selected_action"],
                    "next_state": view["next_state"],
                    "provenance": dict(item.provenance),
                }))
            except (KeyError, TypeError, ValueError):
                rejected["invalid_transition"] += 1
        return transitions, dict(rejected)


class OutcomeTransitionAdapter:
    def extract(self, experiences, allowed_splits=("train",)):
        result = []
        for item in experiences:
            view, supervision = item.model_view, item.supervision
            if item.split in allowed_splits and view.get("state") and view.get("next_state") and view.get("selected_action"):
                result.append((view["state"], view["selected_action"], view["next_state"],
                               supervision.get("reward"), supervision.get("outcome")))
        return result
