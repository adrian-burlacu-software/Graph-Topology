
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SemanticTransform:
    name:str
    task:str
    dependencies:Tuple[str,...]

    def execute(self,values):
        if self.name=="delayed_memory":
            return values["memory"]

        if self.name=="ordered_binding":
            return (
                values["memory"]
                ^values["cue1"]
                ^values["cue2"]
                ^values["cue3"]
            )

        if self.name=="relevant_binding":
            return (
                values["memory"]
                ^values["cue1"]
            )

        if self.name=="planning":
            return (
                values["memory"]
                ^values["cue1"]
                ^values["cue2"]
                ^values["cue3"]
            )

        if self.name=="rule_conditioned":
            return (
                values["memory"]
                ^values["active_rule"]
            )

        if self.name=="counterfactual":
            actual=(
                values["memory"]
                ^values["cue1"]
            )
            if values["counterfactual_mode"]:
                return 1-actual
            return actual

        raise ValueError(self.name)


class SemanticTransformBuilder:
    def build(self,episode):
        if episode.task=="delayed_memory":
            return (
                SemanticTransform(
                    "delayed_memory",
                    episode.task,
                    ("memory",),
                ),
            )

        if episode.task=="sequence_binding":
            return (
                SemanticTransform(
                    "ordered_binding",
                    episode.task,
                    ("memory","cue1","cue2","cue3"),
                ),
            )

        if episode.task=="interference":
            return (
                SemanticTransform(
                    "relevant_binding",
                    episode.task,
                    ("memory","cue1"),
                ),
            )

        if episode.task=="planning":
            return (
                SemanticTransform(
                    "planning",
                    episode.task,
                    ("memory","cue1","cue2","cue3"),
                ),
            )

        if episode.task=="rule_change":
            return (
                SemanticTransform(
                    "rule_conditioned",
                    episode.task,
                    ("memory","active_rule"),
                ),
            )

        if episode.task=="counterfactual":
            return (
                SemanticTransform(
                    "counterfactual",
                    episode.task,
                    (
                        "memory",
                        "cue1",
                        "counterfactual_mode",
                    ),
                ),
            )

        raise ValueError(episode.task)
