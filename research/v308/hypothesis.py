
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HypothesisConfig:
    name: str
    contradiction_gain: float
    confidence_decay: float
    revision_threshold: float
    cooldown: int
    selectivity: float


class HypothesisRevision:
    """
    Explicit current-policy hypothesis.

    hypothesis=0 means keep the frozen base policy.
    hypothesis=1 means invert it.

    Ordinary error forms the first hypothesis.
    A later explicit rule-change marker makes a contradictory outcome revise it.
    """

    def __init__(self, config: HypothesisConfig):
        self.config=config
        self.hypothesis=0
        self.confidence=0.0
        self.contradictions=0
        self.cooldown=0
        self.count=0

    def inject_state(self, graph):
        graph.add_node(
            "hypothesis_state",
            "hypothesis_state",
            value=float(self.hypothesis),
            persistent=True,
        )
        graph.add_node(
            "hypothesis_confidence",
            "hypothesis_confidence",
            value=self.confidence,
            persistent=True,
        )

    def transform_decision(
        self,
        graph,
        decision:int,
        episode,
    )->int:
        if self.confidence>=self.config.revision_threshold:
            return int(decision)^self.hypothesis
        return int(decision)

    def feedback(
        self,
        graph,
        predicted:int,
        answer:int,
        episode,
    ):
        error=int(predicted!=answer)

        changed=(
            episode.task=="rule_change"
            and episode.rule_version==1
        )

        if self.cooldown>0:
            self.cooldown-=1

        # Form the initial hypothesis from ordinary prediction error.
        if not changed:
            if error:
                self.hypothesis=1
                self.confidence=min(
                    1.0,
                    self.confidence
                    +self.config.contradiction_gain,
                )
            else:
                self.confidence=min(
                    1.0,
                    self.confidence+0.20,
                )

        # After explicit regime change, an error is evidence that the stored
        # hypothesis has become stale.
        elif error:
            self.contradictions+=1
            self.confidence=min(
                1.0,
                self.confidence
                +self.config.contradiction_gain,
            )

            if (
                self.contradictions>=1
                and self.cooldown==0
                and self.confidence
                    >=self.config.revision_threshold
            ):
                self.hypothesis^=1
                self.contradictions=0
                self.confidence=1.0
                self.cooldown=self.config.cooldown

        self.count+=1


CONFIGS = {
    "soft_revision": HypothesisConfig(
        name="soft_revision",
        contradiction_gain=0.55,
        confidence_decay=0.70,
        revision_threshold=0.50,
        cooldown=1,
        selectivity=0.10,
    ),
    "fast_revision": HypothesisConfig(
        name="fast_revision",
        contradiction_gain=0.75,
        confidence_decay=0.55,
        revision_threshold=0.50,
        cooldown=0,
        selectivity=0.15,
    ),
    "conservative_revision": HypothesisConfig(
        name="conservative_revision",
        contradiction_gain=0.70,
        confidence_decay=0.85,
        revision_threshold=0.60,
        cooldown=2,
        selectivity=0.05,
    ),
    "sticky_hypothesis": HypothesisConfig(
        name="sticky_hypothesis",
        contradiction_gain=0.60,
        confidence_decay=0.90,
        revision_threshold=0.55,
        cooldown=3,
        selectivity=0.02,
    ),
}
