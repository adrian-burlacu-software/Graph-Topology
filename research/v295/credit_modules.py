
from __future__ import annotations


class Credit:
    name="none"

    def __init__(self):
        self.rule_belief=0.0
        self.count=0

    def reset(self):
        self.rule_belief=0.0
        self.count=0

    def update(self,predicted:int,answer:int):
        self.count+=1

    def signal(self)->int:
        return int(
            self.rule_belief>=0.5
        )


class Immediate(Credit):
    name="immediate"

    def update(self,predicted,answer):
        error=int(
            predicted!=answer
        )

        # A single terminal error is decisive evidence that the frozen policy
        # needs inversion. A later correct trial does NOT erase that evidence.
        if error:
            self.rule_belief=1.0

        self.count+=1


class Eligibility(Credit):
    name="eligibility"

    def __init__(self):
        super().__init__()
        self.trace=0.0

    def reset(self):
        super().reset()
        self.trace=0.0

    def update(self,predicted,answer):
        error=float(
            predicted!=answer
        )

        self.trace=(
            0.70*self.trace
            +0.30*error
        )

        # Positive evidence accumulates toward a durable rule hypothesis.
        self.rule_belief=max(
            self.rule_belief,
            self.trace,
        )

        self.count+=1


class DelayedWindow(Credit):
    name="delayed_window"

    def __init__(self):
        super().__init__()
        self.errors=[]

    def reset(self):
        super().reset()
        self.errors=[]

    def update(self,predicted,answer):
        error=int(
            predicted!=answer
        )

        self.errors.append(error)

        if len(self.errors)>4:
            self.errors.pop(0)

        # Once the temporal window contains decisive evidence, preserve the
        # learned rule instead of allowing later correct behavior to erase it.
        window_mean=(
            sum(self.errors)
            /len(self.errors)
        )

        self.rule_belief=max(
            self.rule_belief,
            window_mean,
        )

        self.count+=1


class RuleFlip(Credit):
    name="rule_flip"

    def __init__(self):
        super().__init__()
        self.has_evidence=False

    def reset(self):
        super().reset()
        self.has_evidence=False

    def update(self,predicted,answer):
        error=int(
            predicted!=answer
        )

        if not self.has_evidence:
            self.rule_belief=float(error)
            self.has_evidence=True
        elif error:
            self.rule_belief=1.0

        self.count+=1


CREDITS={
    "none":Credit,
    "immediate":Immediate,
    "eligibility":Eligibility,
    "delayed_window":DelayedWindow,
    "rule_flip":RuleFlip,
}
