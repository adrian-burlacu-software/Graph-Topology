
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class SelectiveConfig:
    name: str
    relevance_decay: float
    cue_bonus: float
    distractor_penalty: float
    path_bonus: float
    threshold: float
    persistence: float


class SelectiveRepresentation:
    """
    Graph-native relevance filter.

    It maintains a persistent relevance score for graph items. At decision
    time, irrelevant/distractor nodes are suppressed before readout/planning.

    The crucial property is that selection is based only on graph structure and
    learned relevance state, not on episode.answer/task metadata.
    """

    def __init__(self, config: SelectiveConfig):
        self.config=config
        self.relevance: Dict[str,float]={}
        self.count=0

    def reset(self):
        self.relevance.clear()
        self.count=0

    def inject_state(self,graph):
        for name,score in self.relevance.items():
            if name in graph.nodes:
                graph.nodes[name].value += (
                    self.config.persistence * score
                )

    def score_graph(self,graph):
        scores={}

        target_names={
            n.name
            for n in graph.nodes.values()
            if n.role=="query_target"
        }

        for node in graph.nodes.values():
            score=0.0

            if node.role=="memory":
                score += 1.0

            if node.role in (
                "cue1",
                "cue2",
                "cue3",
            ):
                score += self.config.cue_bonus

            if node.role=="distractor":
                score -= self.config.distractor_penalty

            if node.name in target_names:
                score += 0.5

            # Structural proximity to the queried target.
            for edge in graph.edges:
                if edge.target==node.name:
                    if edge.target in target_names:
                        score += self.config.path_bonus
                if edge.source==node.name:
                    if edge.target in target_names:
                        score += self.config.path_bonus

            scores[node.name]=score

        return scores

    def apply(
        self,
        graph,
    ):
        scores=self.score_graph(graph)

        for name,score in scores.items():
            old=self.relevance.get(
                name,
                0.0,
            )

            new=(
                self.config.relevance_decay*old
                +(1.0-self.config.relevance_decay)*score
            )

            self.relevance[name]=new

        # Suppress only clearly irrelevant nodes. Preserve topology; we are
        # filtering representation, not deleting the environment.
        for node in graph.nodes.values():
            score=self.relevance.get(
                node.name,
                0.0,
            )

            if (
                node.role=="distractor"
                and score < self.config.threshold
            ):
                node.value*=0.0

        self.count+=1


CONFIGS={
    "weak_filter":SelectiveConfig(
        "weak_filter",
        relevance_decay=0.70,
        cue_bonus=0.20,
        distractor_penalty=0.20,
        path_bonus=0.10,
        threshold=0.10,
        persistence=0.10,
    ),
    "balanced_filter":SelectiveConfig(
        "balanced_filter",
        relevance_decay=0.50,
        cue_bonus=0.50,
        distractor_penalty=0.70,
        path_bonus=0.40,
        threshold=0.20,
        persistence=0.20,
    ),
    "strong_filter":SelectiveConfig(
        "strong_filter",
        relevance_decay=0.30,
        cue_bonus=0.70,
        distractor_penalty=1.00,
        path_bonus=0.60,
        threshold=0.30,
        persistence=0.30,
    ),
    "persistent_filter":SelectiveConfig(
        "persistent_filter",
        relevance_decay=0.85,
        cue_bonus=0.60,
        distractor_penalty=1.00,
        path_bonus=0.70,
        threshold=0.25,
        persistence=0.50,
    ),
}

assert len(CONFIGS)==4
