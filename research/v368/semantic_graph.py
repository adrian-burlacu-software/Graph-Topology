
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class SemanticNode:
    node_id: str
    label: str
    kind: str = "entity"


@dataclass(frozen=True)
class SemanticEdge:
    source: str
    relation: str
    target: str


@dataclass
class SemanticGraph:
    nodes: Dict[str, SemanticNode]
    edges: List[SemanticEdge]

    def add_node(self, node_id: str, label: str, kind: str="entity"):
        self.nodes[node_id]=SemanticNode(node_id,label,kind)

    def add_edge(self, source: str, relation: str, target: str):
        if source not in self.nodes or target not in self.nodes:
            raise KeyError("edge endpoint missing")
        self.edges.append(SemanticEdge(source,relation,target))

    def relations(self):
        return tuple(
            sorted(
                (e.source,e.relation,e.target)
                for e in self.edges
            )
        )

    def validate(self):
        assert self.nodes
        for e in self.edges:
            assert e.source in self.nodes
            assert e.target in self.nodes
            assert e.relation
        return True

    def to_dict(self):
        return {
            "nodes":{
                k:{
                    "label":v.label,
                    "kind":v.kind,
                }
                for k,v in self.nodes.items()
            },
            "edges":[
                {
                    "source":e.source,
                    "relation":e.relation,
                    "target":e.target,
                }
                for e in self.edges
            ],
        }


def build_smoke_semantic_graph() -> SemanticGraph:
    g=SemanticGraph(nodes={},edges=[])
    g.add_node("dog","dog","entity")
    g.add_node("chases","chase","predicate")
    g.add_node("cat","cat","entity")
    g.add_edge("chases","agent","dog")
    g.add_edge("chases","patient","cat")
    return g
