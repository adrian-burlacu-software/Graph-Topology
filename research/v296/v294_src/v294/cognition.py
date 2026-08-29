
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import hashlib
import random


ACTIONS=("NO","YES")
RELATIONS=("r0","r1","r2","r3")

TASKS=(
    "memory",
    "binding",
    "dynamics",
    "credit",
    "planning",
)


@dataclass
class Node:
    name:str
    role:str
    value:float=0.0
    persistent:bool=True


@dataclass
class Edge:
    source:str
    relation:str
    target:str
    weight:float=1.0
    persistent:bool=True


@dataclass
class Graph:
    nodes:Dict[str,Node]=field(default_factory=dict)
    edges:List[Edge]=field(default_factory=list)

    def clone(self)->"Graph":
        return Graph(
            nodes={
                k:Node(
                    n.name,
                    n.role,
                    n.value,
                    n.persistent,
                )
                for k,n in self.nodes.items()
            },
            edges=[
                Edge(
                    e.source,
                    e.relation,
                    e.target,
                    e.weight,
                    e.persistent,
                )
                for e in self.edges
            ],
        )

    def add_node(
        self,
        name:str,
        role:str,
        value:float=0.0,
        persistent:bool=True,
    ):
        self.nodes[name]=Node(
            name,
            role,
            value,
            persistent,
        )

    def ensure_node(
        self,
        name:str,
        role:str="opaque",
    ):
        if name not in self.nodes:
            self.add_node(name,role)

    def add_edge(
        self,
        source:str,
        relation:str,
        target:str,
        weight:float=1.0,
        persistent:bool=True,
    ):
        self.ensure_node(source)
        self.ensure_node(target)

        for edge in self.edges:
            if (
                edge.source==source
                and edge.relation==relation
                and edge.target==target
            ):
                edge.weight=weight
                edge.persistent=(
                    edge.persistent
                    or persistent
                )
                return

        self.edges.append(
            Edge(
                source,
                relation,
                target,
                weight,
                persistent,
            )
        )

    def remove_role(
        self,
        role:str,
    ):
        names={
            n.name
            for n in self.nodes.values()
            if n.role==role
        }
        self.nodes={
            k:v
            for k,v in self.nodes.items()
            if k not in names
        }
        self.edges=[
            e
            for e in self.edges
            if e.source not in names
            and e.target not in names
        ]

    def outgoing(
        self,
        source:str,
        relation:Optional[str]=None,
    )->List[Edge]:
        return [
            e
            for e in self.edges
            if e.source==source
            and e.weight>0.0
            and (
                relation is None
                or e.relation==relation
            )
        ]

    def incoming(
        self,
        target:str,
        relation:Optional[str]=None,
    )->List[Edge]:
        return [
            e
            for e in self.edges
            if e.target==target
            and e.weight>0.0
            and (
                relation is None
                or e.relation==relation
            )
        ]

    def decay(self,factor:float):
        for node in self.nodes.values():
            node.value*=factor
        for edge in self.edges:
            if not edge.persistent:
                edge.weight*=factor


@dataclass(frozen=True)
class Query:
    source:str
    relation_a:str
    relation_b:str
    target:str


@dataclass(frozen=True)
class Episode:
    seed:int
    task:str
    graph:Graph
    query:Query
    cue_order:Tuple[str,...]
    answer_bit:int
    decision_step:int
    latent_rule:int

    @property
    def answer_action(self)->str:
        return ACTIONS[self.answer_bit]


@dataclass(frozen=True)
class Sequence:
    seed:int
    task:str
    episodes:Tuple[Episode,...]


def opaque(prefix:str,seed:int,index:int)->str:
    return (
        prefix+"_"
        +hashlib.sha256(
            f"{prefix}:{seed}:{index}".encode()
        ).hexdigest()[:10]
    )


def make_episode(
    seed:int,
    task:str,
    index:int,
    latent_rule:int,
    horizon:int=7,
)->Episode:
    if task not in TASKS:
        raise ValueError(task)

    rng=random.Random(
        100003*seed
        +7919*TASKS.index(task)
        +101*index
    )

    graph=Graph()

    source=opaque("q",seed,index*5)
    middle=opaque("q",seed,index*5+1)
    target=opaque("q",seed,index*5+2)
    decoy=opaque("d",seed,index*5+3)

    graph.add_node(source,"query_source")
    graph.add_node(middle,"concept")
    graph.add_node(target,"query_target")
    graph.add_node(decoy,"concept")

    rel_a,rel_b=rng.sample(
        RELATIONS,
        2,
    )
    decoy_rel=next(
        r for r in RELATIONS
        if r not in (rel_a,rel_b)
    )

    # Four independent input channels.
    initial=(
        seed
        +index
        +latent_rule
    )%2

    cue=(
        seed//2
        +2*index
    )%2

    delta=(
        seed//3
        +index
    )%2

    operation=(
        seed//5
        +index
    )%2

    # Hidden sensory fact. It is removed after observation.
    fact=opaque("fact",seed,index)
    graph.add_node(
        fact,
        "initial_fact",
        value=float(initial),
        persistent=False,
    )

    # Later cue.
    cue_node=opaque("cue",seed,index)
    graph.add_node(
        cue_node,
        "cue",
        value=float(cue),
        persistent=True,
    )
    graph.add_edge(
        cue_node,
        "modulates",
        target,
    )

    # Delta cue is the signal the dynamics module can transform.
    delta_node=opaque("delta",seed,index)
    graph.add_node(
        delta_node,
        "delta",
        value=float(delta),
        persistent=True,
    )
    graph.add_edge(
        delta_node,
        "updates",
        target,
    )

    # Operation cue for planning.
    op_node=opaque("op",seed,index)
    graph.add_node(
        op_node,
        "operation",
        value=float(operation),
        persistent=True,
    )
    graph.add_edge(
        op_node,
        "operation",
        target,
    )

    # Stable relational environment. It is identical with respect to the
    # hidden fact so topology alone cannot reveal the answer.
    graph.add_edge(
        source,
        rel_a,
        middle,
    )
    graph.add_edge(
        middle,
        rel_b,
        target,
    )

    # Always-present distractor path.
    graph.add_edge(
        source,
        decoy_rel,
        decoy,
    )
    graph.add_edge(
        decoy,
        rel_b,
        target,
    )

    # Planning sequence: ordered generic operations, encoded structurally.
    if task=="planning":
        p1=opaque("plan",seed,index*2)
        p2=opaque("plan",seed,index*2+1)

        graph.add_node(
            p1,
            "plan_step",
            value=1.0,
            persistent=True,
        )
        graph.add_node(
            p2,
            "plan_step",
            value=1.0,
            persistent=True,
        )

        graph.add_edge(
            p1,
            "order",
            p2,
        )
        graph.add_edge(
            p1,
            "applies",
            target,
        )
        graph.add_edge(
            p2,
            "applies",
            target,
        )

    # Counterfactual control is represented generically.
    if task=="binding" and operation:
        control=opaque("control",seed,index)
        marker=opaque("marker",seed,index)

        graph.add_node(
            control,
            "control",
            persistent=True,
        )
        graph.add_node(
            marker,
            "negate",
            persistent=True,
        )
        graph.add_edge(
            control,
            "mode",
            marker,
        )
        graph.add_edge(
            control,
            "applies",
            target,
        )

    # Environment defines the task answer.
    if task=="memory":
        answer=initial
    elif task=="binding":
        answer=initial^cue
        if operation:
            answer=1-answer
    elif task=="dynamics":
        # Requires transforming remembered state using a later update.
        answer=initial^delta
    elif task=="credit":
        # Latent rule must be learned from previous feedback in the sequence.
        answer=(
            initial
            if latent_rule==0
            else 1-initial
        )
    elif task=="planning":
        # Two ordered operations: cue, then operation.
        answer=initial^cue^operation
    else:
        raise AssertionError(task)

    # Random nuisance.
    for i in range(4):
        n=opaque(
            "noise",
            seed+17*index,
            i,
        )
        graph.add_node(
            n,
            "noise",
        )
        graph.add_edge(
            decoy,
            rng.choice(RELATIONS),
            n,
        )

    return Episode(
        seed=seed,
        task=task,
        graph=graph,
        query=Query(
            source,
            rel_a,
            rel_b,
            target,
        ),
        cue_order=("cue","delta","operation"),
        answer_bit=answer,
        decision_step=horizon-1,
        latent_rule=latent_rule,
    )


def make_sequence(
    seed:int,
    task:str,
    episodes:int=10,
    horizon:int=7,
)->Sequence:
    rng=random.Random(
        40009*seed
        +TASKS.index(task)*101
    )
    latent_rule=rng.randrange(2)

    return Sequence(
        seed=seed,
        task=task,
        episodes=tuple(
            make_episode(
                seed,
                task,
                i,
                latent_rule,
                horizon,
            )
            for i in range(episodes)
        ),
    )


# ---------------------------------------------------------------------------
# Memory modules
# ---------------------------------------------------------------------------

class Memory:
    name="none"

    def observe(self,graph,query,step):
        return None

    def maintain(self,graph,query,step):
        return None


class PersistentMemory(Memory):
    name="persistent"

    def observe(self,graph,query,step):
        fact=next(
            (
                n for n in graph.nodes.values()
                if n.role=="initial_fact"
            ),
            None,
        )

        if fact is None:
            return

        graph.add_node(
            "memory",
            "memory",
            value=(
                1.0
                if fact.value>=0.5
                else -1.0
            ),
            persistent=True,
        )

    def maintain(self,graph,query,step):
        node=graph.nodes.get("memory")
        if node is None:
            return

        if node.value>=0:
            node.value=max(
                node.value,
                0.85,
            )
        else:
            node.value=min(
                node.value,
                -0.85,
            )


class EpisodicMemory(Memory):
    name="episodic"

    def observe(self,graph,query,step):
        fact=next(
            (
                n for n in graph.nodes.values()
                if n.role=="initial_fact"
            ),
            None,
        )

        graph.add_node(
            "episode_memory",
            "episode_memory",
            value=(
                1.0
                if fact is not None
                and fact.value>=0.5
                else -1.0
            ),
            persistent=True,
        )

    def maintain(self,graph,query,step):
        node=graph.nodes.get(
            "episode_memory"
        )
        if node is not None:
            if node.value>=0:
                node.value=max(
                    0.80,
                    node.value,
                )
            else:
                node.value=min(
                    -0.80,
                    node.value,
                )


class WorkingMemory(Memory):
    name="working"

    def observe(self,graph,query,step):
        fact=next(
            (
                n for n in graph.nodes.values()
                if n.role=="initial_fact"
            ),
            None,
        )

        graph.add_node(
            "working",
            "working",
            value=(
                0.75
                if fact is not None
                and fact.value>=0.5
                else -0.75
            ),
            persistent=True,
        )

    def maintain(self,graph,query,step):
        node=graph.nodes.get("working")
        if node is not None:
            if node.value>=0:
                node.value=(
                    0.97*node.value
                    +0.02
                )
            else:
                node.value=(
                    0.97*node.value
                    -0.02
                )


# ---------------------------------------------------------------------------
# Dynamics modules
# ---------------------------------------------------------------------------

class Dynamics:
    name="static"

    def step(self,graph,query,step):
        return None


class LeakyDynamics(Dynamics):
    name="leaky"

    def step(self,graph,query,step):
        graph.decay(0.90)


class StabilizingDynamics(Dynamics):
    name="stabilizing"

    def step(self,graph,query,step):
        graph.decay(0.95)

        for name in (
            "memory",
            "episode_memory",
            "working",
            "state",
        ):
            node=graph.nodes.get(name)
            if node is not None:
                if node.value>=0:
                    node.value=max(
                        node.value,
                        0.75,
                    )
                else:
                    node.value=min(
                        node.value,
                        -0.75,
                    )


class TransformDynamics(Dynamics):
    name="transform"

    def step(self,graph,query,step):
        graph.decay(0.97)

        update_node=next(
            (
                n for n in graph.nodes.values()
                if n.role=="delta"
            ),
            None,
        )

        if update_node is None:
            return

        for name in (
            "memory",
            "episode_memory",
            "working",
        ):
            node=graph.nodes.get(name)
            if node is None:
                continue

            if (
                update_node.value>=0.5
                and step>=2
            ):
                node.value*=-1.0


DYNAMICS={
    "static":Dynamics,
    "leaky":LeakyDynamics,
    "stabilizing":StabilizingDynamics,
    "transform":TransformDynamics,
}


# ---------------------------------------------------------------------------
# Readout modules
# ---------------------------------------------------------------------------

class Readout:
    name="null"

    def read(self,graph,query)->Tuple[int,List[Tuple]]:
        return 0,[]


class MemoryReadout(Readout):
    name="memory"

    def read(self,graph,query):
        for name in (
            "memory",
            "episode_memory",
            "working",
        ):
            node=graph.nodes.get(name)
            if node is not None:
                return (
                    int(node.value>=0),
                    [],
                )
        return 0,[]


class RelationalReadout(Readout):
    name="relational"

    def read(self,graph,query):
        path=[]

        for first in graph.outgoing(
            query.source,
            query.relation_a,
        ):
            for second in graph.outgoing(
                first.target,
                query.relation_b,
            ):
                if second.target==query.target:
                    path=[
                        (
                            first.source,
                            first.relation,
                            first.target,
                        ),
                        (
                            second.source,
                            second.relation,
                            second.target,
                        ),
                    ]

        bit=MemoryReadout().read(
            graph,
            query,
        )[0]

        return bit,path


class IntegrativeReadout(Readout):
    name="integrative"

    def read(self,graph,query):
        bit,path=RelationalReadout().read(
            graph,
            query,
        )
        return (
            bit if path else 0,
            path,
        )


class StateReadout(Readout):
    name="state"

    def read(self,graph,query):
        node=graph.nodes.get("working")
        if node is None:
            node=graph.nodes.get("memory")
        if node is None:
            node=graph.nodes.get(
                "episode_memory"
            )

        if node is None:
            return 0,[]

        return (
            int(node.value>=0),
            [],
        )


READOUTS={
    "null":Readout,
    "memory":MemoryReadout,
    "relational":RelationalReadout,
    "integrative":IntegrativeReadout,
    "state":StateReadout,
}


# ---------------------------------------------------------------------------
# Planner modules
# ---------------------------------------------------------------------------

class Planner:
    name="direct"

    def plan(
        self,
        graph,
        query,
        recalled,
    )->int:
        return int(recalled)


def cue_values(
    graph:Graph,
    target:str,
)->List[Tuple[int,int]]:
    values=[]

    for edge in graph.incoming(
        target,
        "modulates",
    ):
        node=graph.nodes.get(
            edge.source
        )
        if node is not None:
            values.append(
                (
                    1,
                    int(node.value>=0.5),
                )
            )

    for edge in graph.incoming(
        target,
        "updates",
    ):
        node=graph.nodes.get(
            edge.source
        )
        if node is not None:
            values.append(
                (
                    2,
                    int(node.value>=0.5),
                )
            )

    for edge in graph.incoming(
        target,
        "operation",
    ):
        node=graph.nodes.get(
            edge.source
        )
        if node is not None:
            values.append(
                (
                    3,
                    int(node.value>=0.5),
                )
            )

    return sorted(values)


class BindingPlanner(Planner):
    name="binding"

    def plan(self,graph,query,recalled):
        value=int(recalled)

        for _,cue in cue_values(
            graph,
            query.target,
        ):
            value^=cue

        return value


class ControlPlanner(Planner):
    name="control"

    def plan(self,graph,query,recalled):
        value=BindingPlanner().plan(
            graph,
            query,
            recalled,
        )

        for node in graph.nodes.values():
            if node.role!="control":
                continue

            applies=any(
                e.source==node.name
                and e.relation=="applies"
                and e.target==query.target
                for e in graph.edges
            )

            if not applies:
                continue

            negate=any(
                graph.nodes.get(e.target) is not None
                and graph.nodes[e.target].role=="negate"
                for e in graph.edges
                if (
                    e.source==node.name
                    and e.relation=="mode"
                )
            )

            if negate:
                value=1-value

        return value


class RolloutPlanner(Planner):
    name="rollout"

    def plan(self,graph,query,recalled):
        value=int(recalled)

        for _,cue in cue_values(
            graph,
            query.target,
        ):
            value^=cue

        # Second operation: ordered plan graph must exist.
        plan_nodes=[
            n for n in graph.nodes.values()
            if n.role=="plan_step"
        ]

        if len(plan_nodes)>=2:
            value^=int(
                all(
                    any(
                        e.source==n.name
                        and e.relation=="applies"
                        and e.target==query.target
                        for e in graph.edges
                    )
                    for n in plan_nodes
                )
            )

        return value


PLANNERS={
    "direct":Planner,
    "binding":BindingPlanner,
    "control":ControlPlanner,
    "rollout":RolloutPlanner,
}


# ---------------------------------------------------------------------------
# Credit
# ---------------------------------------------------------------------------

class Credit:
    name="none"

    def __init__(self):
        self.error_trace=0.0
        self.count=0

    def reset(self):
        self.error_trace=0.0
        self.count=0

    def update(
        self,
        predicted:int,
        answer:int,
    ):
        self.count+=1

    def policy_signal(self)->int:
        return int(
            self.error_trace>=0.5
        )


class ImmediateCredit(Credit):
    name="immediate"

    def update(self,predicted,answer):
        error=float(
            predicted!=answer
        )

        self.error_trace=(
            0.70*self.error_trace
            +0.30*error
        )
        self.count+=1


class EligibilityCredit(Credit):
    name="eligibility"

    def __init__(self):
        super().__init__()
        self.eligibility=0.0

    def reset(self):
        super().reset()
        self.eligibility=0.0

    def update(self,predicted,answer):
        error=float(
            predicted!=answer
        )

        self.eligibility=(
            0.82*self.eligibility
            +0.18*error
        )

        self.error_trace=(
            0.86*self.error_trace
            +0.14*self.eligibility
        )

        self.count+=1


class PathCredit(Credit):
    name="path"

    def __init__(self):
        super().__init__()
        self.path_error=0.0

    def reset(self):
        super().reset()
        self.path_error=0.0

    def update(self,predicted,answer):
        error=float(
            predicted!=answer
        )

        self.path_error=(
            0.80*self.path_error
            +0.20*error
        )

        self.error_trace=(
            0.90*self.error_trace
            +0.10*self.path_error
        )

        self.count+=1


CREDITS={
    "none":Credit,
    "immediate":ImmediateCredit,
    "eligibility":EligibilityCredit,
    "path":PathCredit,
}


# ---------------------------------------------------------------------------
# Cognitive system + architecture search
# ---------------------------------------------------------------------------

MEMORIES={
    "none":Memory,
    "persistent":PersistentMemory,
    "episodic":EpisodicMemory,
    "working":WorkingMemory,
}


@dataclass(frozen=True)
class Architecture:
    memory:str
    dynamics:str
    readout:str
    planner:str
    credit:str

    @property
    def name(self)->str:
        return (
            f"{self.memory}+{self.dynamics}+"
            f"{self.readout}+{self.planner}+"
            f"{self.credit}"
        )


def all_architectures()->List[Architecture]:
    return [
        Architecture(
            m,d,r,p,c
        )
        for m in MEMORIES
        for d in DYNAMICS
        for r in READOUTS
        for p in PLANNERS
        for c in CREDITS
    ]


def architecture_count()->int:
    return (
        len(MEMORIES)
        *len(DYNAMICS)
        *len(READOUTS)
        *len(PLANNERS)
        *len(CREDITS)
    )


class CognitiveSystem:
    def __init__(
        self,
        architecture:Architecture,
    ):
        self.architecture=architecture
        self.memory=MEMORIES[
            architecture.memory
        ]()
        self.dynamics=DYNAMICS[
            architecture.dynamics
        ]()
        self.readout=READOUTS[
            architecture.readout
        ]()
        self.planner=PLANNERS[
            architecture.planner
        ]()
        self.credit=CREDITS[
            architecture.credit
        ]()

    def reset(self):
        self.credit.reset()

    def run(
        self,
        episode:Episode,
        learn:bool=True,
    )->dict:
        graph=episode.graph.clone()

        # Feedback learned on previous episodes is represented as generic
        # internal state. It is never the answer and contains no task label.
        graph.add_node(
            "credit_signal",
            "credit_signal",
            value=float(
                self.credit.policy_signal()
            ),
            persistent=True,
        )

        self.memory.observe(
            graph,
            episode.query,
            0,
        )

        # Initial sensory node expires.
        initial=next(
            (
                name
                for name in list(
                    graph.nodes
                )
                if graph.nodes[name].role
                    =="initial_fact"
            ),
            None,
        )

        if initial is not None:
            graph.nodes.pop(
                initial
            )

        for step in range(
            1,
            episode.decision_step+1,
        ):
            self.dynamics.step(
                graph,
                episode.query,
                step,
            )
            self.memory.maintain(
                graph,
                episode.query,
                step,
            )

        recalled,path=self.readout.read(
            graph,
            episode.query,
        )

        decision=self.planner.plan(
            graph,
            episode.query,
            recalled,
        )

        correct=(
            decision==episode.answer_bit
        )

        if learn:
            self.credit.update(
                decision,
                episode.answer_bit,
            )

        return {
            "correct":correct,
            "decision":decision,
            "answer":episode.answer_bit,
            "path":path,
        }


def evaluate_sequence(
    architecture:Architecture,
    sequence:Sequence,
)->dict:
    system=CognitiveSystem(
        architecture
    )

    rows=[
        system.run(
            episode,
            learn=True,
        )
        for episode in sequence.episodes
    ]

    half=max(
        1,
        len(rows)//2,
    )

    return {
        "accuracy":(
            sum(
                int(r["correct"])
                for r in rows
            )/len(rows)
        ),
        "first_half":(
            sum(
                int(r["correct"])
                for r in rows[:half]
            )/len(rows[:half])
        ),
        "second_half":(
            sum(
                int(r["correct"])
                for r in rows[half:]
            )/len(rows[half:])
        ),
    }


def module_ablation_score(
    architecture:Architecture,
    episodes:List[Episode],
    module:str,
)->float:
    replacement={
        "memory":"none",
        "dynamics":"static",
        "readout":"null",
        "planner":"direct",
        "credit":"none",
    }

    parts={
        "memory":architecture.memory,
        "dynamics":architecture.dynamics,
        "readout":architecture.readout,
        "planner":architecture.planner,
        "credit":architecture.credit,
    }

    parts[module]=replacement[module]

    ablated=CognitiveSystem(
        Architecture(
            parts["memory"],
            parts["dynamics"],
            parts["readout"],
            parts["planner"],
            parts["credit"],
        )
    )

    return (
        sum(
            int(
                ablated.run(
                    episode,
                    learn=False,
                )["correct"]
            )
            for episode in episodes
        )
        /len(episodes)
    )


def causal_profile(
    architecture:Architecture,
    episodes:List[Episode],
)->dict:
    system=CognitiveSystem(
        architecture
    )

    normal=[
        system.run(
            episode,
            learn=False,
        )
        for episode in episodes
    ]

    normal_acc=(
        sum(
            int(r["correct"])
            for r in normal
        )/len(normal)
    )

    drops={}
    for module in (
        "memory",
        "dynamics",
        "readout",
        "planner",
        "credit",
    ):
        ablated=module_ablation_score(
            architecture,
            episodes,
            module,
        )
        drops[module]=(
            normal_acc-ablated
        )

    return {
        "normal":normal_acc,
        "drops":drops,
    }
