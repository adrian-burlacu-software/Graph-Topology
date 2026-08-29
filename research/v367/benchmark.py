
from __future__ import annotations
from dataclasses import replace
from richer_cognition import make_sequence


def environment_answer(ep):
    if ep.task=="rule_change" and ep.rule_version==2:
        return int(ep.initial_bit + ep.cue_bits[0])
    if ep.task=="rule_change":
        return int(
            ep.initial_bit
            ^(ep.latent_rule ^ ep.rule_version)
        )
    if ep.task=="counterfactual":
        actual=ep.initial_bit ^ ep.cue_bits[0]
        return int(
            1-actual if ep.counterfactual_bit else actual
        )
    if ep.task in ("sequence_binding","planning"):
        return int(
            ep.initial_bit
            ^ep.cue_bits[0]
            ^ep.cue_bits[1]
            ^ep.cue_bits[2]
        )
    if ep.task=="interference":
        return int(ep.initial_bit ^ ep.cue_bits[0])
    return int(ep.initial_bit)


def curriculum(seed):
    entries=(
        ("R0","rule_change"),
        ("R1","rule_change"),
        ("R0","rule_change"),
        ("R2","rule_change"),
        ("R1","rule_change"),
        ("M0","counterfactual"),
        ("M1","counterfactual"),
        ("M0","counterfactual"),
        ("TRANSFER","sequence_binding"),
        ("COMPOSE","planning"),
    )
    for phase,task in entries:
        ep=make_sequence(seed,task,1,9).episodes[0]
        if phase.startswith("R"):
            ep=replace(
                ep,
                rule_version=int(phase[1:]),
            )
        elif phase.startswith("M"):
            ep=replace(
                ep,
                counterfactual_bit=int(phase[1:]),
            )
        ep=replace(
            ep,
            answer_bit=environment_answer(ep),
        )
        yield phase,task,ep
