
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import re
import math


TOKEN_RE=re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")


@dataclass(frozen=True)
class SemanticFrame:
    predicate: str
    arguments: Tuple[Tuple[str,str], ...]

    def normalized(self):
        return SemanticFrame(
            predicate=self.predicate.lower(),
            arguments=tuple(
                sorted(
                    (
                        role,
                        value.lower(),
                    )
                    for role,value in self.arguments
                )
            ),
        )


@dataclass(frozen=True)
class Perception:
    sentence: str
    frame: Optional[SemanticFrame]
    grammar_rule: Optional[str]
    confidence: float


class RoundTripPerception:
    """
    Deterministic compositional perception for the learned grammar interface.

    The cognitive semantic architecture still performs lexical grounding for
    every content item. The parser is deliberately transparent so roundtrip
    failures can be attributed to grammar, semantic grounding, or production.
    """

    STOP={"the","a","an"}

    def __init__(self, semantic_architecture):
        self.semantic=semantic_architecture

    def tokenize(self,sentence):
        return [x.lower() for x in TOKEN_RE.findall(sentence)]

    def tag(self,token):
        if token in self.STOP:
            return "DET"
        if token in {
            "chase","chases","eat","eats","see","sees",
            "like","likes","want","wants",
        }:
            return "VERB"
        return "NOUN"

    def perceive(self,sentence):
        tokens=self.tokenize(sentence)
        tags=[self.tag(t) for t in tokens]

        if len(tags)>=5 and tags[:5]==[
            "DET","NOUN","VERB","DET","NOUN"
        ]:
            agent=tokens[1]
            predicate=tokens[2]
            patient=tokens[4]

            grounded=[]
            grounding_modes=[]
            grounded_names=[]

            for token in (agent,predicate,patient):
                state=self.semantic.perceive(
                    token,
                    context=(),
                )
                grounded.append(state)

                if state.committed is not None:
                    grounded_names.append(state.committed)
                    grounding_modes.append("committed")
                    continue

                # Roundtrip requires a graph-grounded lexical identity, but
                # does not require unjustified certainty. If the exact query
                # itself is a ConceptNet concept, retain it as the lexical
                # referent while marking the state as non-disambiguated.
                exact = getattr(self.semantic, "memory", None)
                if exact is not None:
                    concepts=exact.concepts()
                    key=token.lower()
                    if key in concepts:
                        grounded_names.append(key)
                        grounding_modes.append("exact_graph_identity")
                        continue

                grounded_names.append(None)
                grounding_modes.append("unresolved")

            if all(name is not None for name in grounded_names):
                frame=SemanticFrame(
                    predicate=grounded_names[1],
                    arguments=(
                        ("agent",grounded_names[0]),
                        ("patient",grounded_names[2]),
                    ),
                )
                confidence=min(
                    (
                        state.confidence
                        if mode=="committed"
                        else 0.50
                    )
                    for state,mode in zip(
                        grounded,
                        grounding_modes,
                    )
                )
                return Perception(
                    sentence=sentence,
                    frame=frame,
                    grammar_rule="DET|NOUN|VERB::SVO",
                    confidence=confidence,
                )

        if len(tags)==2 and tags==["DET","NOUN"]:
            state=self.semantic.perceive(
                tokens[1],
                context=(),
            )
            name=state.committed
            confidence=state.confidence

            if name is None:
                memory=getattr(
                    self.semantic,
                    "memory",
                    None,
                )
                if memory is not None:
                    key=tokens[1].lower()
                    if key in memory.concepts():
                        name=key
                        confidence=0.50

            if name is not None:
                frame=SemanticFrame(
                    predicate="be",
                    arguments=(("entity",name),),
                )
                return Perception(
                    sentence=sentence,
                    frame=frame,
                    grammar_rule="DET|NOUN::ENTITY",
                    confidence=confidence,
                )

        return Perception(
            sentence=sentence,
            frame=None,
            grammar_rule=None,
            confidence=0.0,
        )


class RoundTripProduction:
    """
    Semantic-frame → language realization.

    Production is not allowed to invent a semantic transformation: the output
    template is selected from grammar rules, and the resulting sentence is
    immediately validated by perception in the roundtrip benchmark.
    """

    def __init__(self, semantic_architecture):
        self.semantic=semantic_architecture

    def _arg(self,frame,role):
        for r,v in frame.arguments:
            if r==role:
                return v
        raise ValueError(
            f"Missing semantic role: {role}"
        )

    def generate(self,frame):
        frame=frame.normalized()

        if frame.predicate=="be" and any(
            r=="entity" for r,_ in frame.arguments
        ):
            entity=self._arg(frame,"entity")
            return f"the {entity}"

        agent=self._arg(frame,"agent")
        patient=self._arg(frame,"patient")

        return f"the {agent} {frame.predicate} the {patient}"


def semantic_equivalent(a,b):
    if a is None or b is None:
        return False
    return a.normalized()==b.normalized()


class BidirectionalRoundTripBenchmark:
    def __init__(self,semantic_architecture):
        self.semantic=semantic_architecture
        self.perception=RoundTripPerception(
            semantic_architecture
        )
        self.production=RoundTripProduction(
            semantic_architecture
        )

    def perception_then_generation(
        self,
        sentence,
    ):
        p1=self.perception.perceive(sentence)
        if p1.frame is None:
            return {
                "pass":False,
                "direction":"perception_to_generation_to_perception",
                "reason":"initial_perception_failed",
                "sentence":sentence,
            }

        generated=self.production.generate(
            p1.frame
        )
        p2=self.perception.perceive(
            generated
        )

        return {
            "pass":(
                p2.frame is not None
                and semantic_equivalent(
                    p1.frame,
                    p2.frame,
                )
            ),
            "direction":"perception_to_generation_to_perception",
            "input":sentence,
            "generated":generated,
            "input_frame":p1.frame,
            "roundtrip_frame":p2.frame,
            "perception_confidence":p1.confidence,
            "reperception_confidence":p2.confidence,
        }

    def generation_then_perception(
        self,
        frame,
    ):
        generated=self.production.generate(frame)
        perceived=self.perception.perceive(
            generated
        )

        regenerated=None
        if perceived.frame is not None:
            regenerated=self.production.generate(
                perceived.frame
            )

        return {
            "pass":(
                perceived.frame is not None
                and semantic_equivalent(
                    frame,
                    perceived.frame,
                )
                and regenerated==generated
            ),
            "direction":"generation_to_perception_to_generation",
            "input_frame":frame,
            "generated":generated,
            "perceived_frame":perceived.frame,
            "regenerated":regenerated,
            "perception_confidence":perceived.confidence,
        }
