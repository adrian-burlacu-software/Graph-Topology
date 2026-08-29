
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HybridConfig:
    trace_decay: float
    trace_gain: float
    signal_decay: float
    signal_gain: float
    threshold: float
    baseline_decay: float
    baseline_weight: float


class GlobalLongEligibility:
    """
    Global persistent learning signal fed by a temporal eligibility trace.

    The key distinction from earlier versions:
      * trace controls *when* evidence is retained
      * signal controls *how strongly* that retained evidence enters cognition

    They are independently scaled so long traces are not numerically starved.
    """

    name = "global_long_eligibility"

    def __init__(self, config: HybridConfig):
        self.config = config
        self.trace = 0.0
        self.signal = 0.0
        self.baseline = 0.0
        self.count = 0

    def inject_state(self, graph):
        graph.add_node(
            "credit_global",
            "credit_global",
            value=self.signal,
            persistent=True,
        )
        graph.add_node(
            "credit_trace",
            "credit_trace",
            value=self.trace,
            persistent=True,
        )
        graph.add_node(
            "credit_baseline",
            "credit_baseline",
            value=self.baseline,
            persistent=True,
        )

    def modify_decision(self, graph, decision, path):
        if self.signal >= self.config.threshold:
            return 1 - int(decision)
        return int(decision)

    def feedback(
        self,
        graph,
        predicted,
        answer,
        path,
        episode,
    ):
        error = float(predicted != answer)

        surprise = max(
            0.0,
            error
            - self.config.baseline_weight
            * self.baseline,
        )

        self.trace = min(
            1.0,
            self.config.trace_decay * self.trace
            + self.config.trace_gain * surprise,
        )

        self.baseline = (
            self.config.baseline_decay * self.baseline
            + (1.0 - self.config.baseline_decay) * error
        )

        self.signal = min(
            1.0,
            self.config.signal_decay * self.signal
            + self.config.signal_gain * self.trace,
        )

        self.count += 1


# Deliberately interpretable combinations.
CONFIGS = {
    "fast_global": HybridConfig(
        trace_decay=0.50,
        trace_gain=0.50,
        signal_decay=0.50,
        signal_gain=0.90,
        threshold=0.30,
        baseline_decay=0.90,
        baseline_weight=0.00,
    ),
    "balanced": HybridConfig(
        trace_decay=0.70,
        trace_gain=0.30,
        signal_decay=0.65,
        signal_gain=0.80,
        threshold=0.30,
        baseline_decay=0.90,
        baseline_weight=0.00,
    ),
    "long_trace": HybridConfig(
        trace_decay=0.90,
        trace_gain=0.10,
        signal_decay=0.75,
        signal_gain=0.90,
        threshold=0.25,
        baseline_decay=0.90,
        baseline_weight=0.00,
    ),
    "long_persistent": HybridConfig(
        trace_decay=0.90,
        trace_gain=0.10,
        signal_decay=0.90,
        signal_gain=0.80,
        threshold=0.25,
        baseline_decay=0.90,
        baseline_weight=0.00,
    ),
    "surprise_balanced": HybridConfig(
        trace_decay=0.70,
        trace_gain=0.30,
        signal_decay=0.70,
        signal_gain=0.80,
        threshold=0.25,
        baseline_decay=0.70,
        baseline_weight=0.50,
    ),
    "slow_high_signal": HybridConfig(
        trace_decay=0.85,
        trace_gain=0.15,
        signal_decay=0.80,
        signal_gain=1.00,
        threshold=0.20,
        baseline_decay=0.90,
        baseline_weight=0.00,
    ),
}

assert len(CONFIGS) == 6
