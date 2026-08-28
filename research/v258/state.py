from __future__ import annotations
from dataclasses import dataclass
ACTIONS=('NOOP','REUSE','CREATE','BRANCH','INHIBIT','BIND','COMMIT')
ACTION_TO_ID={a:i for i,a in enumerate(ACTIONS)}
@dataclass
class Node:
    concept:str; activation:float; role:int; persistent:bool=False
@dataclass
class Edge:
    source:str; relation:str; target:str; activation:float; persistent:bool=False
@dataclass
class State:
    nodes:list[Node]; edges:list[Edge]
    def clone(self): return State([Node(n.concept,n.activation,n.role,n.persistent) for n in self.nodes],[Edge(e.source,e.relation,e.target,e.activation,e.persistent) for e in self.edges])
    def node(self,c): return next((n for n in self.nodes if n.concept==c),None)
    def focus(self):
        cs=[n for n in self.nodes if n.role==2]
        return max(cs,key=lambda n:n.activation).concept if cs else None
    def add_node(self,c,a,role,persistent=False):
        n=self.node(c)
        if n is None:self.nodes.append(Node(c,a,role,persistent))
        else:n.activation=max(n.activation,a);n.persistent|=persistent
    def add_edge(self,s,r,t,a=1.,persistent=False):
        for e in self.edges:
            if (e.source,e.relation,e.target)==(s,r,t):e.activation=max(e.activation,a);e.persistent|=persistent;return
        self.edges.append(Edge(s,r,t,a,persistent))
    def has_edge(self,s,r,t,active_only=True):return any(e.source==s and e.relation==r and e.target==t and (not active_only or e.activation>.5) for e in self.edges)
    def signature(self):return {'nodes':[vars(n) for n in self.nodes],'edges':[vars(e) for e in self.edges]}
    def apply(self,aid,source=None,target=None,relation=None):
        s=self.clone();a=ACTIONS[aid]
        if a=='REUSE' and target and s.node(target):
            for n in s.nodes:
                if n.role==2:n.role=1;n.activation=.05
            s.node(target).role=2;s.node(target).activation=1.
        elif a=='CREATE':s.add_node(f'created_{len(s.nodes)}',.85,6)
        elif a=='BRANCH' and source and relation:
            b=f'{source}#branch{len(s.nodes)}';s.add_node(b,.8,7);s.add_edge(source,relation,b,.8)
        elif a=='INHIBIT' and target and s.node(target):s.node(target).activation*=.05
        elif a=='BIND' and source and target and relation:s.add_edge(source,relation,target,1.)
        elif a=='COMMIT':
            for n in s.nodes:
                if n.activation>.5:n.persistent=True
            for e in s.edges:
                if e.activation>.5:e.persistent=True
        return s
