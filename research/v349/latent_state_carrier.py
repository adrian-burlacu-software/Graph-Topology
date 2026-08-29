
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class LatentBinding:
    name: str
    value: int
    source: str
    semantics: str


class LatentStateCarrier:
    """
    Makes constructor latent variables explicit graph-native state.

    The carrier NEVER binds answer_bit. It only binds latent environment state
    that the constructor exposes on the Episode object:
      - initial_rule
      - rule_version
      - counterfactual_mode

    Derived active_rule is computed from those explicit latent fields.

    The carrier is intentionally a semantic overlay: nodes and edges are added
    to the graph with stable role names so downstream modules can consume them
    without reaching back into Episode internals.
    """

    def __init__(self, mode="all"):
        if mode not in (
            "all",
            "persistent_only",
            "context_only",
            "no_carrier",
        ):
            raise ValueError(mode)
        self.mode=mode

    def bindings(self, episode):
        out=[]

        if self.mode=="no_carrier":
            return tuple()

        # Initial rule is a latent parameter of the environment, not the answer.
        if self.mode in ("all","persistent_only"):
            out.append(
                LatentBinding(
                    "initial_rule",
                    int(episode.latent_rule),
                    "episode.latent_rule",
                    "initial_rule_parameter",
                )
            )

        # Rule version is an explicit regime/state variable.
        if self.mode in ("all","context_only"):
            out.append(
                LatentBinding(
                    "rule_version",
                    int(episode.rule_version),
                    "episode.rule_version",
                    "regime_indicator",
                )
            )

        # Counterfactual mode is a latent control state.
        if self.mode in ("all","context_only"):
            out.append(
                LatentBinding(
                    "counterfactual_mode",
                    int(episode.counterfactual_bit),
                    "episode.counterfactual_bit",
                    "counterfactual_control",
                )
            )

        return tuple(out)

    def inject(self, graph, episode):
        bindings=self.bindings(episode)

        for binding in bindings:
            graph.add_node(
                f"latent:{binding.name}",
                f"latent_{binding.semantics}",
                value=float(binding.value),
                persistent=True,
            )

        # Derived active rule is explicit graph state, but is computed from the
        # latent rule and regime rather than copied from answer semantics.
        if (
            self.mode!="no_carrier"
            and any(
                b.name=="initial_rule"
                for b in bindings
            )
            and any(
                b.name=="rule_version"
                for b in bindings
            )
        ):
            initial=next(
                b.value for b in bindings
                if b.name=="initial_rule"
            )
            version=next(
                b.value for b in bindings
                if b.name=="rule_version"
            )
            active=int(initial ^ version)

            graph.add_node(
                "latent:active_rule",
                "latent_active_rule",
                value=float(active),
                persistent=True,
            )

        if self.mode!="no_carrier":
            # Bind carrier variables to the semantic structures that consume
            # them. These edges make provenance explicit.
            if "latent:active_rule" in graph.nodes:
                for node in list(graph.nodes.values()):
                    if node.role=="rule_change_marker":
                        graph.add_edge(
                            "latent:active_rule",
                            "controls",
                            node.name,
                        )

            if "latent:counterfactual_mode" in graph.nodes:
                for node in list(graph.nodes.values()):
                    if node.role=="control":
                        graph.add_edge(
                            "latent:counterfactual_mode",
                            "activates",
                            node.name,
                        )

        return bindings

    def read(self, graph):
        out={}

        for name,node in graph.nodes.items():
            if not name.startswith("latent:"):
                continue

            semantic=name.split(":",1)[1]
            out[semantic]=int(
                round(node.value)
            )

        return out


class LatentStateAudit:
    def audit(self, graph, episode):
        expected={
            "initial_rule":int(episode.latent_rule),
            "rule_version":int(episode.rule_version),
            "counterfactual_mode":int(
                episode.counterfactual_bit
            ),
        }

        actual={}
        for key in expected:
            node=graph.nodes.get(
                f"latent:{key}"
            )
            if node is not None:
                actual[key]=int(
                    round(node.value)
                )

        missing=tuple(
            k for k in expected
            if k not in actual
        )
        mismatched=tuple(
            k for k in expected
            if k in actual
            and actual[k]!=expected[k]
        )

        derived_expected=int(
            episode.latent_rule
            ^episode.rule_version
        )

        active_node=graph.nodes.get(
            "latent:active_rule"
        )

        derived_mismatch=(
            active_node is not None
            and int(round(active_node.value))
            !=derived_expected
        )

        return {
            "missing":missing,
            "mismatched":mismatched,
            "active_rule_mismatch":derived_mismatch,
            "pass":not missing
            and not mismatched
            and not derived_mismatch,
        }
