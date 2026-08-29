
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CreditConfig:
    name:str
    trace_decay:float
    trace_gain:float
    signal_decay:float
    signal_gain:float
    threshold:float
    reward_decay:float


class BaseCredit:
    def __init__(self,config):
        self.config=config
        self.trace=0.0
        self.signal=0.0
        self.reward_baseline=0.0
        self.count=0

    def inject_state(self,graph):
        graph.add_node(
            "credit_global",
            "credit_global",
            value=self.signal,
        )
        graph.add_node(
            "credit_trace",
            "credit_trace",
            value=self.trace,
        )

    def modify_decision(self,graph,decision,path):
        if self.signal>=self.config.threshold:
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
        raise NotImplementedError


class GlobalPersistent(BaseCredit):
    name="global_persistent"

    def feedback(
        self,
        graph,
        predicted,
        answer,
        path,
        episode,
    ):
        error=float(predicted!=answer)

        self.trace=(
            self.config.trace_decay*self.trace
            +self.config.trace_gain*error
        )

        self.reward_baseline=(
            self.config.reward_decay*self.reward_baseline
            +(1-self.config.reward_decay)*error
        )

        self.signal=min(
            1.0,
            self.config.signal_decay*self.signal
            +self.config.signal_gain*self.trace,
        )

        self.count+=1


class LongEligibility(BaseCredit):
    name="long_eligibility"

    def feedback(
        self,
        graph,
        predicted,
        answer,
        path,
        episode,
    ):
        error=float(predicted!=answer)

        self.trace=(
            self.config.trace_decay*self.trace
            +self.config.trace_gain*error
        )

        self.signal=max(
            self.signal,
            self.trace,
        )

        self.count+=1


class AdaptiveBaseline(BaseCredit):
    name="adaptive_baseline"

    def feedback(
        self,
        graph,
        predicted,
        answer,
        path,
        episode,
    ):
        error=float(predicted!=answer)

        surprise=max(
            0.0,
            error-self.reward_baseline,
        )

        self.reward_baseline=(
            self.config.reward_decay*self.reward_baseline
            +(1-self.config.reward_decay)*error
        )

        self.trace=(
            self.config.trace_decay*self.trace
            +self.config.trace_gain*surprise
        )

        self.signal=min(
            1.0,
            self.config.signal_decay*self.signal
            +self.config.signal_gain*self.trace,
        )

        self.count+=1


class ErrorAccumulator(BaseCredit):
    name="error_accumulator"

    def feedback(
        self,
        graph,
        predicted,
        answer,
        path,
        episode,
    ):
        error=float(predicted!=answer)

        self.trace=min(
            1.0,
            self.trace+0.25*error,
        )

        self.signal=max(
            self.signal,
            self.trace,
        )

        self.count+=1


CONFIGS={
    "global_fast":CreditConfig(
        "global_fast",
        0.65,0.35,
        0.60,0.90,
        0.30,0.90,
    ),
    "global_persistent":CreditConfig(
        "global_persistent",
        0.80,0.20,
        0.90,0.90,
        0.25,0.90,
    ),
    "long_eligibility":CreditConfig(
        "long_eligibility",
        0.90,0.10,
        0.80,0.90,
        0.20,0.90,
    ),
    "adaptive_baseline":CreditConfig(
        "adaptive_baseline",
        0.80,0.20,
        0.80,0.90,
        0.25,0.70,
    ),
    "error_accumulator":CreditConfig(
        "error_accumulator",
        0.95,0.25,
        0.95,0.80,
        0.35,0.90,
    ),
}


CREDIT_CLASSES={
    "global_fast":GlobalPersistent,
    "global_persistent":GlobalPersistent,
    "long_eligibility":LongEligibility,
    "adaptive_baseline":AdaptiveBaseline,
    "error_accumulator":ErrorAccumulator,
}


def make_credit(name):
    return CREDIT_CLASSES[name](
        CONFIGS[name]
    )
