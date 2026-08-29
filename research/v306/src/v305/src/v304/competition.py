
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class CompetitionConfig:
    name: str
    goal_strength: float
    inhibition: float
    winner_gain: float
    persistence: float
    stale_decay: float
    collision_penalty: float


class GoalCompetition:
    """
    Goal-conditioned competitive working set.

    The selector does not merely suppress nodes globally. It constructs a
    temporary "working set" score around the current goal and allows a winner
    to gate the downstream representation.
    """

    def __init__(self, config: CompetitionConfig):
        self.config=config
        self.goal_state=0.0
        self.winner=0.0
        self.collisions=0
        self.count=0

    def inject_state(self,graph):
        graph.add_node(
            "working_goal",
            "working_goal",
            value=self.goal_state,
            persistent=True,
        )
        graph.add_node(
            "working_winner",
            "working_winner",
            value=self.winner,
            persistent=True,
        )

    def apply(self,graph,episode):
        # Goal comes from generic query structure, not task labels.
        goal_nodes=[
            n for n in graph.nodes.values()
            if n.role=="query_target"
        ]

        candidate_scores=[]

        for node in graph.nodes.values():
            score=0.0

            if node.role=="memory":
                score+=self.config.goal_strength

            if node.role in (
                "cue1",
                "cue2",
                "cue3",
            ):
                score+=0.5*self.config.goal_strength

            if node.role=="distractor":
                score-=self.config.inhibition

            if node.role=="plan_step":
                score+=0.25*self.config.goal_strength

            # Structural relation to current target.
            for edge in graph.edges:
                if (
                    edge.target==node.name
                    and edge.target in {
                        x.name for x in goal_nodes
                    }
                ):
                    score+=self.config.winner_gain

                if (
                    edge.source==node.name
                    and edge.target in {
                        x.name for x in goal_nodes
                    }
                ):
                    score+=self.config.winner_gain

            candidate_scores.append(
                (node.name,score)
            )

        candidate_scores.sort(
            key=lambda x:x[1],
            reverse=True,
        )

        if candidate_scores:
            best_name,best_score=candidate_scores[0]
            self.winner=(
                self.config.persistence*self.winner
                + (1-self.config.persistence)*best_score
            )

            if len(candidate_scores)>1:
                second_score=candidate_scores[1][1]
                if abs(best_score-second_score)<0.05:
                    self.collisions+=1

            # Competitive suppression of non-winning distractors.
            for name,score in candidate_scores:
                node=graph.nodes[name]

                if (
                    node.role=="distractor"
                    and score < self.winner
                ):
                    node.value*=(
                        1.0-self.config.inhibition
                    )

        self.goal_state*=self.config.stale_decay
        self.count+=1


CONFIGS={
    "goal_weak":CompetitionConfig(
        "goal_weak",
        goal_strength=0.4,
        inhibition=0.2,
        winner_gain=0.3,
        persistence=0.7,
        stale_decay=0.95,
        collision_penalty=0.1,
    ),
    "goal_balanced":CompetitionConfig(
        "goal_balanced",
        goal_strength=0.8,
        inhibition=0.5,
        winner_gain=0.6,
        persistence=0.5,
        stale_decay=0.90,
        collision_penalty=0.2,
    ),
    "goal_strong":CompetitionConfig(
        "goal_strong",
        goal_strength=1.2,
        inhibition=0.8,
        winner_gain=0.9,
        persistence=0.3,
        stale_decay=0.80,
        collision_penalty=0.4,
    ),
    "goal_persistent":CompetitionConfig(
        "goal_persistent",
        goal_strength=1.0,
        inhibition=0.7,
        winner_gain=0.8,
        persistence=0.85,
        stale_decay=0.98,
        collision_penalty=0.3,
    ),
}
