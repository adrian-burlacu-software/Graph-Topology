
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CreditConfig:
    name: str
    trace_decay: float
    trace_gain: float
    signal_decay: float
    signal_gain: float
    threshold: float


class HybridCredit:
    """
    Frozen V299 credit family:
        global persistent signal + eligibility trace

    V300 freezes the credit mechanism itself and only compares its two leading
    operating regimes against each other inside the graph core.
    """

    def __init__(self, config: CreditConfig):
        self.config=config
        self.trace=0.0
        self.signal=0.0
        self.count=0

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

    def modify_decision(
        self,
        graph,
        decision,
        path,
    ):
        if self.signal >= self.config.threshold:
            return 1-int(decision)
        return int(decision)

    def feedback(
        self,
        graph,
        predicted,
        answer,
        path,
        episode,
    ):
        error=float(predicted != answer)

        self.trace=min(
            1.0,
            self.config.trace_decay*self.trace
            +self.config.trace_gain*error,
        )

        self.signal=min(
            1.0,
            self.config.signal_decay*self.signal
            +self.config.signal_gain*self.trace,
        )

        self.count+=1


CONFIGS={
    "slow_high_signal":CreditConfig(
        name="slow_high_signal",
        trace_decay=0.85,
        trace_gain=0.15,
        signal_decay=0.80,
        signal_gain=1.00,
        threshold=0.20,
    ),
    "long_trace":CreditConfig(
        name="long_trace",
        trace_decay=0.90,
        trace_gain=0.10,
        signal_decay=0.75,
        signal_gain=0.90,
        threshold=0.25,
    ),
}

assert len(CONFIGS)==2
