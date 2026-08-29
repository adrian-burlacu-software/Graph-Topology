
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class GroundedOperator:
    name: str
    kind: str

    def execute(self,frame,subgoal):
        out=frame.clone()

        if self.name=="read_memory":
            out.registers["memory"]=out.state.memory
            out.trace.append("read_memory")
        elif self.name=="read_cue1":
            out.registers["cue1"]=out.state.cues[0]
            out.trace.append("read_cue1")
        elif self.name=="read_cue2":
            out.registers["cue2"]=out.state.cues[1]
            out.trace.append("read_cue2")
        elif self.name=="read_cue3":
            out.registers["cue3"]=out.state.cues[2]
            out.trace.append("read_cue3")
        elif self.name=="xor_sequence":
            out.registers["result"]=(
                out.registers["memory"]
                ^out.registers["cue1"]
                ^out.registers["cue2"]
            )
            out.registers["result_source"]="memory,cue1,cue2"
            out.trace.append("xor(memory,cue1,cue2)")
        elif self.name=="compute_relevance":
            out.registers["result"]=(
                out.registers["memory"]
                ^out.registers["cue1"]
            )
            out.registers["result_source"]="memory,cue1"
            out.trace.append("compute_relevance")
        elif self.name=="derive_rule_evidence":
            out.registers["result"]=(
                out.registers["memory"]
                ^out.registers["cue1"]
            )
            out.registers["result_source"]="memory,cue1"
            out.registers["rule_phase"]=1
            out.trace.append("derive_rule_evidence")
        elif self.name=="revise_hypothesis":
            out.registers["hypothesis"]=1
            out.trace.append("revise_hypothesis")
        elif self.name=="plan_result":
            out.registers["result"]=(
                out.registers["memory"]
                ^out.registers["cue1"]
                ^out.registers["cue2"]
                ^out.registers["cue3"]
            )
            out.registers["result_source"]="plan"
            out.trace.append("plan_result")
        elif self.name=="set_actual":
            out.registers["actual"]=(
                out.registers["memory"]
                ^out.registers["cue1"]
            )
            out.trace.append("set_actual")
        elif self.name=="set_alternate":
            out.registers["alternate"]=(
                out.registers["memory"]
                ^out.registers["cue2"]
            )
            out.trace.append("set_alternate")
        elif self.name=="evaluate_alternate":
            out.registers["result"] = out.registers["alternate"]
            out.registers["result_source"]="alternate"
            out.trace.append("evaluate_alternate")
        elif self.name=="emit_result":
            out.registers.setdefault(
                "result",
                out.registers.get("memory",0),
            )
            out.trace.append("emit_result")
        else:
            raise ValueError(self.name)

        return out


OPERATORS=(
    GroundedOperator("read_memory","retrieve"),
    GroundedOperator("read_cue1","retrieve"),
    GroundedOperator("read_cue2","retrieve"),
    GroundedOperator("read_cue3","retrieve"),
    GroundedOperator("xor_sequence","transform"),
    GroundedOperator("compute_relevance","transform"),
    GroundedOperator("derive_rule_evidence","transform"),
    GroundedOperator("revise_hypothesis","revise"),
    GroundedOperator("plan_result","transform"),
    GroundedOperator("set_actual","transform"),
    GroundedOperator("set_alternate","transform"),
    GroundedOperator("evaluate_alternate","transform"),
    GroundedOperator("emit_result","decide"),
)
