
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class CreditConfig:
    trace_decay:float
    trace_gain:float
    signal_decay:float
    signal_gain:float
    baseline_decay:float
    injection_threshold:float
    update_mode:str


class CombinedGlobalEligibility:
    """
    Hybrid of the two strongest V297 hypotheses.

    Global component:
        converts terminal prediction error into a system-wide learning signal.

    Eligibility component:
        spreads/retains that signal over time with a decaying trace.

    Baseline component is optional:
        learns expected error and uses surprise relative to baseline.

    The important architectural property is that the resulting signal becomes
    persistent cognitive state and directly enters the next action path.
    """

    name="combined_global_eligibility"

    def __init__(
        self,
        config:CreditConfig,
    ):
        self.config=config
        self.trace=0.0
        self.signal=0.0
        self.baseline=0.0
        self.count=0

    def reset(self):
        self.trace=0.0
        self.signal=0.0
        self.baseline=0.0
        self.count=0

    def inject_state(self,graph):
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

    def modify_decision(
        self,
        graph,
        decision:int,
        path:List[Tuple],
    )->int:
        if self.signal>=self.config.injection_threshold:
            return 1-int(decision)

        return int(decision)

    def feedback(
        self,
        graph,
        predicted:int,
        answer:int,
        path,
        episode,
    ):
        error=float(
            predicted!=answer
        )

        baseline_error=(
            error
            -self.baseline
        )

        if self.config.update_mode=="raw":
            teaching=max(
                0.0,
                error,
            )
        elif self.config.update_mode=="surprise":
            teaching=max(
                0.0,
                baseline_error,
            )
        else:
            raise ValueError(
                self.config.update_mode
            )

        self.trace=(
            self.config.trace_decay*self.trace
            +self.config.trace_gain*teaching
        )

        self.baseline=(
            self.config.baseline_decay*self.baseline
            +(1.0-self.config.baseline_decay)*error
        )

        self.signal=(
            self.config.signal_decay*self.signal
            +self.config.signal_gain*self.trace
        )

        self.count+=1


CONFIGS={
    # Strong baseline: global signal with short eligibility memory.
    "global_fast":CreditConfig(
        trace_decay=0.50,
        trace_gain=0.50,
        signal_decay=0.60,
        signal_gain=0.40,
        baseline_decay=0.85,
        injection_threshold=0.50,
        update_mode="raw",
    ),
    # Longer temporal credit assignment.
    "eligibility_balanced":CreditConfig(
        trace_decay=0.70,
        trace_gain=0.30,
        signal_decay=0.75,
        signal_gain=0.25,
        baseline_decay=0.85,
        injection_threshold=0.45,
        update_mode="raw",
    ),
    # High persistence, slower global accumulation.
    "eligibility_long":CreditConfig(
        trace_decay=0.85,
        trace_gain=0.15,
        signal_decay=0.85,
        signal_gain=0.15,
        baseline_decay=0.90,
        injection_threshold=0.40,
        update_mode="raw",
    ),
    # Baseline-subtracted signal.
    "surprise_eligibility":CreditConfig(
        trace_decay=0.70,
        trace_gain=0.30,
        signal_decay=0.75,
        signal_gain=0.25,
        baseline_decay=0.70,
        injection_threshold=0.30,
        update_mode="surprise",
    ),
    # Strong early commitment, strong persistence.
    "global_persistent":CreditConfig(
        trace_decay=0.60,
        trace_gain=0.40,
        signal_decay=0.90,
        signal_gain=0.30,
        baseline_decay=0.85,
        injection_threshold=0.50,
        update_mode="raw",
    ),
}
