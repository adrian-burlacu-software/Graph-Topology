
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class SemanticCase:
    surface: str
    sense_a: str
    sense_b: str
    context_relation: str
    context_target_a: str
    context_target_b: str
 
    @property
    def case_id(self):
        return self.surface

    @property
    def candidates(self):
        return (self.sense_a,self.sense_b)


@dataclass(frozen=True)
class ContextCase:
    case_id: str
    surface: str
    candidates: Tuple[str,...]
    context: Tuple[Tuple[str,str],...]
    expected: Optional[str]
    kind: str


def resolved_cases(pair: SemanticCase):
    return (
        ContextCase(
            case_id=pair.case_id+"-a",
            surface=pair.surface,
            candidates=pair.candidates,
            context=((pair.context_relation,pair.context_target_a),),
            expected=pair.sense_a,
            kind="resolved",
        ),
        ContextCase(
            case_id=pair.case_id+"-b",
            surface=pair.surface,
            candidates=pair.candidates,
            context=((pair.context_relation,pair.context_target_b),),
            expected=pair.sense_b,
            kind="resolved",
        ),
    )


def ambiguous_case(pair: SemanticCase):
    return ContextCase(
        case_id=pair.case_id+"-ambiguous",
        surface=pair.surface,
        candidates=pair.candidates,
        context=(
            (pair.context_relation,pair.context_target_a),
            (pair.context_relation,pair.context_target_b),
        ),
        expected=None,
        kind="ambiguous",
    )


def no_context_case(pair: SemanticCase):
    return ContextCase(
        case_id=pair.case_id+"-no-context",
        surface=pair.surface,
        candidates=pair.candidates,
        context=(),
        expected=None,
        kind="no_context",
    )


def make_synthetic_cases():
    pairs=[
        SemanticCase("bank","bank_finance","bank_river","RelatedTo","money","river"),
        SemanticCase("bat","bat_animal","bat_sports","RelatedTo","animal","ball"),
        SemanticCase("crane","crane_bird","crane_machine","RelatedTo","bird","construction"),
        SemanticCase("seal","seal_animal","seal_stamp","RelatedTo","animal","document"),
        SemanticCase("plant","plant_living","plant_factory","RelatedTo","growth","production"),
        SemanticCase("watch","watch_device","watch_action","RelatedTo","time","observe"),
        SemanticCase("spring","spring_season","spring_mechanism","RelatedTo","weather","mechanics"),
        SemanticCase("match","match_fire","match_game","RelatedTo","flame","competition"),
        SemanticCase("club","club_group","club_weapon","RelatedTo","membership","weapon"),
        SemanticCase("light","light_photon","light_object","RelatedTo","illumination","carry"),
    ]
    out=[]
    for p in pairs:
        out.extend(resolved_cases(p))
        out.append(ambiguous_case(p))
        out.append(no_context_case(p))
    return out
