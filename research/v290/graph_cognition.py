
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import hashlib
import random


ACTIONS = ("YES", "NO")

RELATIONS = ("r0", "r1", "r2", "r3")

FAMILIES = (
    "chain",
    "diamond",
    "decoy",
    "counterfactual",
)


@dataclass
class Node:
    name: str
    role: str
    activation: float = 0.0
    persistent: bool = True


@dataclass
class Edge:
    source: str
    relation: str
    target: str
    weight: float = 1.0
    persistent: bool = True


@dataclass
class Graph:
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

    def clone(self) -> "Graph":
        return Graph(
            nodes={
                k: Node(
                    n.name,
                    n.role,
                    n.activation,
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
        name: str,
        role: str = "opaque",
        activation: float = 0.0,
        persistent: bool = True,
    ) -> None:
        self.nodes[name]=Node(
            name,
            role,
            activation,
            persistent,
        )

    def ensure_node(
        self,
        name: str,
        role: str = "opaque",
    ) -> None:
        if name not in self.nodes:
            self.add_node(
                name,
                role,
            )

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        weight: float = 1.0,
        persistent: bool = True,
    ) -> None:
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
                    edge.persistent or persistent
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

    def remove_edge(
        self,
        source: str,
        relation: str,
        target: str,
    ) -> None:
        self.edges=[
            e for e in self.edges
            if not (
                e.source==source
                and e.relation==relation
                and e.target==target
            )
        ]

    def outgoing(
        self,
        source: str,
        relation: str|None=None,
    ) -> List[Edge]:
        return [
            e for e in self.edges
            if e.source==source
            and e.weight>0.0
            and (
                relation is None
                or e.relation==relation
            )
        ]

    def incoming(
        self,
        target: str,
        relation: str|None=None,
    ) -> List[Edge]:
        return [
            e for e in self.edges
            if e.target==target
            and e.weight>0.0
            and (
                relation is None
                or e.relation==relation
            )
        ]

    def has_edge(
        self,
        source: str,
        relation: str,
        target: str,
    ) -> bool:
        return any(
            e.source==source
            and e.relation==relation
            and e.target==target
            and e.weight>0.0
            for e in self.edges
        )

    def activate(
        self,
        name: str,
        amount: float,
    ) -> None:
        self.ensure_node(name)
        self.nodes[name].activation += amount

    def decay(self,factor:float)->None:
        for node in self.nodes.values():
            node.activation*=factor

        for edge in self.edges:
            if not edge.persistent:
                edge.weight*=factor


@dataclass(frozen=True)
class Query:
    source: str
    relation_a: str
    relation_b: str
    target: str


@dataclass(frozen=True)
class Episode:
    seed: int
    family: str
    graph: Graph
    query: Query
    answer: bool


def opaque(
    prefix: str,
    seed: int,
    index: int,
) -> str:
    digest=hashlib.sha256(
        f"{seed}:{prefix}:{index}".encode()
    ).hexdigest()[:10]
    return f"{prefix}_{digest}"


def build_episode(
    seed: int,
    family: str,
) -> Episode:
    """
    Anti-shortcut task.

    The algorithm receives only Graph + Query.

    Hidden answer is determined by:
      path existence XOR optional structural negation.

    Node names are opaque.
    Relations are randomized.
    Positive and negative instances are balanced.
    Distractor edges prevent one-hop lookup.
    """

    if family not in FAMILIES:
        raise ValueError(family)

    rng=random.Random(
        seed*1009+
        FAMILIES.index(family)*7919
    )

    graph=Graph()

    source=opaque("node",seed,0)
    middle=opaque("node",seed,1)
    target=opaque("node",seed,2)
    decoy=opaque("node",seed,3)

    for name,role in (
        (source,"query_source"),
        (middle,"concept"),
        (target,"query_target"),
        (decoy,"concept"),
    ):
        graph.add_node(
            name,
            role,
        )

    relations=rng.sample(
        list(RELATIONS),
        3,
    )
    rel_a,rel_b,decoy_rel=relations

    # Balanced base path existence.
    path_exists=(
        (seed+FAMILIES.index(family))%2==0
    )

    if family=="chain":
        graph.add_edge(
            source,
            rel_a,
            middle,
        )
        if path_exists:
            graph.add_edge(
                middle,
                rel_b,
                target,
            )

    elif family=="diamond":
        alt=opaque(
            "node",
            seed,
            4,
        )
        graph.add_node(
            alt,
            "concept",
        )

        graph.add_edge(
            source,
            rel_a,
            middle,
        )

        if path_exists:
            graph.add_edge(
                middle,
                rel_b,
                target,
            )
            graph.add_edge(
                middle,
                rel_b,
                alt,
            )
        else:
            graph.add_edge(
                middle,
                rel_b,
                alt,
            )
            graph.add_edge(
                alt,
                decoy_rel,
                target,
            )

    elif family=="decoy":
        graph.add_edge(
            source,
            rel_a,
            middle,
        )

        # Always create a superficially plausible one-hop distractor.
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

        if path_exists:
            graph.add_edge(
                middle,
                rel_b,
                target,
            )

    elif family=="counterfactual":
        graph.add_edge(
            source,
            rel_a,
            middle,
        )

        if path_exists:
            graph.add_edge(
                middle,
                rel_b,
                target,
            )

        control=opaque(
            "control",
            seed,
            0,
        )
        invert=opaque(
            "mode",
            seed,
            0,
        )

        graph.add_node(
            control,
            "query_control",
        )
        graph.add_node(
            invert,
            "negation_marker",
        )

        # Structural control. The algorithm must infer it by role/edges.
        if (seed//2)%2==1:
            graph.add_edge(
                control,
                "mode",
                invert,
            )
            graph.add_edge(
                control,
                "applies_to",
                target,
            )

    # Independent nuisance structure.
    nuisance=opaque(
        "nuisance",
        seed,
        0,
    )
    graph.add_node(
        nuisance,
        "nuisance",
    )

    graph.add_edge(
        nuisance,
        decoy_rel,
        source,
    )

    negate=(
        any(
            n.role=="query_control"
            for n in graph.nodes.values()
        )
        and any(
            e.source in graph.nodes
            and graph.nodes[e.source].role=="query_control"
            and e.relation=="mode"
            for e in graph.edges
        )
    )

    # Counterfactual mode reverses the path-existence answer. Other families
    # alternate positive/negative via path_exists.
    answer=(
        not path_exists
        if negate
        else path_exists
    )

    return Episode(
        seed=seed,
        family=family,
        graph=graph,
        query=Query(
            source,
            rel_a,
            rel_b,
            target,
        ),
        answer=answer,
    )


def make_dataset(
    seeds:List[int],
    families:List[str]|None=None,
)->List[Episode]:
    families=(
        list(FAMILIES)
        if families is None
        else list(families)
    )
    return [
        build_episode(seed,family)
        for seed in seeds
        for family in families
    ]


class Memory:
    name="structural"
    def write(self,graph,query): raise NotImplementedError
    def maintain(self,graph,query,step): return None


class StructuralMemory(Memory):
    name="structural"
    def write(self,graph,query):
        # No answer is written. Only the query endpoints are marked.
        graph.nodes[query.source].role="query_source"
        graph.nodes[query.target].role="query_target"


class ActivationMemory(Memory):
    name="activated"
    def write(self,graph,query):
        graph.activate(query.source,1.0)
        graph.activate(query.target,1.0)
    def maintain(self,graph,query,step):
        graph.activate(query.source,0.01)
        graph.activate(query.target,0.01)


class RouteMemory(Memory):
    name="route"
    def write(self,graph,query):
        graph.ensure_node(
            "working",
            "working",
        )
        graph.add_edge(
            "working",
            "query_source",
            query.source,
            persistent=False,
        )
        graph.add_edge(
            "working",
            "query_target",
            query.target,
            persistent=False,
        )
    def maintain(self,graph,query,step):
        graph.add_edge(
            "working",
            "query_source",
            query.source,
            persistent=False,
        )
        graph.add_edge(
            "working",
            "query_target",
            query.target,
            persistent=False,
        )


class Credit:
    name="none"
    def update(
        self,
        graph,
        query,
        predicted,
        answer,
        path,
    ):
        return None


class ImmediateCredit(Credit):
    name="immediate"
    def update(self,graph,query,predicted,answer,path):
        graph.activate(
            "credit_signal",
            0.25 if predicted==answer else -0.10,
        )


class EligibilityCredit(Credit):
    name="eligibility"
    def update(self,graph,query,predicted,answer,path):
        amount=0.35 if predicted==answer else -0.15
        graph.activate(
            "eligibility_trace",
            amount,
        )
        for src,rel,dst in path:
            graph.add_edge(
                src,
                "credit",
                dst,
                weight=max(
                    0.1,
                    amount,
                ),
                persistent=False,
            )


class TDCredit(Credit):
    name="td"
    def update(self,graph,query,predicted,answer,path):
        reward=1.0 if predicted==answer else -1.0
        graph.activate(
            "value",
            reward,
        )
        if path:
            graph.activate(
                path[-1][2],
                0.20*reward,
            )


class PathReinforcement(Credit):
    name="path_reinforcement"
    def update(self,graph,query,predicted,answer,path):
        delta=0.10 if predicted==answer else -0.10
        for src,rel,dst in path:
            for edge in graph.edges:
                if (
                    edge.source==src
                    and edge.relation==rel
                    and edge.target==dst
                ):
                    edge.weight=max(
                        0.0,
                        min(
                            1.25,
                            edge.weight+delta,
                        ),
                    )


class Dynamics:
    name="static"
    def pre(self,graph,step): return None
    def post(self,graph,step): return None


class LeakyDynamics(Dynamics):
    name="leaky"
    def pre(self,graph,step): graph.decay(0.90)
    def post(self,graph,step): graph.decay(0.97)


class PersistentDynamics(Dynamics):
    name="persistent"
    def pre(self,graph,step): graph.decay(0.95)
    def post(self,graph,step): graph.decay(0.98)


class GatedDynamics(Dynamics):
    name="gated"
    def pre(self,graph,step):
        graph.decay(0.90)
        for n in graph.nodes.values():
            if n.persistent:
                n.activation=max(
                    n.activation,
                    0.25,
                )
    def post(self,graph,step):
        graph.decay(0.99)


class Readout:
    name="base"
    def read(self,graph,query): raise NotImplementedError


def path_exists(
    graph:Graph,
    query:Query,
)->List[Tuple]:
    paths=[]

    for e1 in graph.outgoing(
        query.source,
        query.relation_a,
    ):
        for e2 in graph.outgoing(
            e1.target,
            query.relation_b,
        ):
            if e2.target==query.target:
                paths.append(
                    (
                        e1.source,
                        e1.relation,
                        e1.target,
                    )
                )
                paths.append(
                    (
                        e2.source,
                        e2.relation,
                        e2.target,
                    )
                )
    return paths


class OneHop(Readout):
    name="one_hop"
    def read(self,graph,query):
        return (
            graph.has_edge(
                query.source,
                query.relation_b,
                query.target,
            ),
            [],
        )


class TwoHop(Readout):
    name="two_hop"
    def read(self,graph,query):
        paths=path_exists(graph,query)
        return bool(paths),paths[:2]


class Consistency(Readout):
    name="consistency"
    def read(self,graph,query):
        paths=path_exists(graph,query)
        return (
            bool(paths)
            and len(paths)>=2,
            paths[:2],
        )


class Voting(Readout):
    name="voting"
    def read(self,graph,query):
        paths=path_exists(graph,query)
        score=float(len(paths))

        for role in (
            "credit_signal",
            "eligibility_trace",
            "value",
        ):
            for node in graph.nodes.values():
                if node.name==role:
                    score+=node.activation

        return score>=1.0,paths[:2]


class Planner:
    name="none"
    def transform(self,graph,query,recalled):
        return bool(recalled)


class ComposePlanner(Planner):
    name="compose"
    def transform(self,graph,query,recalled):
        return bool(recalled)


class ControlPlanner(Planner):
    name="control"
    def transform(self,graph,query,recalled):
        if not recalled:
            return False

        invert=False

        for node in graph.nodes.values():
            if node.role!="query_control":
                continue

            has_mode=any(
                e.source==node.name
                and e.relation=="mode"
                for e in graph.edges
            )
            applies=any(
                e.source==node.name
                and e.relation=="applies_to"
                and e.target==query.target
                for e in graph.edges
            )

            if has_mode and applies:
                invert=True

        return (
            not recalled
            if invert
            else recalled
        )


class TwoStagePlanner(ControlPlanner):
    name="two_stage"


MEMORIES={
    "structural":StructuralMemory,
    "activated":ActivationMemory,
    "route":RouteMemory,
}

CREDITS={
    "none":Credit,
    "immediate":ImmediateCredit,
    "eligibility":EligibilityCredit,
    "td":TDCredit,
    "path_reinforcement":PathReinforcement,
}

DYNAMICS={
    "static":Dynamics,
    "leaky":LeakyDynamics,
    "persistent":PersistentDynamics,
    "gated":GatedDynamics,
}

READOUTS={
    "one_hop":OneHop,
    "two_hop":TwoHop,
    "consistency":Consistency,
    "voting":Voting,
}

PLANNERS={
    "none":Planner,
    "compose":ComposePlanner,
    "control":ControlPlanner,
    "two_stage":TwoStagePlanner,
}


@dataclass(frozen=True)
class Strategy:
    memory:str
    credit:str
    dynamics:str
    readout:str
    planning:str

    @property
    def name(self)->str:
        return (
            f"{self.memory}+{self.credit}+"
            f"{self.dynamics}+{self.readout}+"
            f"{self.planning}"
        )


def strategy_count()->int:
    return (
        len(MEMORIES)
        *len(CREDITS)
        *len(DYNAMICS)
        *len(READOUTS)
        *len(PLANNERS)
    )


def all_strategies()->List[Strategy]:
    return [
        Strategy(
            memory=m,
            credit=c,
            dynamics=d,
            readout=r,
            planning=p,
        )
        for m in MEMORIES
        for c in CREDITS
        for d in DYNAMICS
        for r in READOUTS
        for p in PLANNERS
    ]


class CognitiveAlgorithm:
    def __init__(self,strategy:Strategy):
        self.strategy=strategy
        self.memory=MEMORIES[strategy.memory]()
        self.credit=CREDITS[strategy.credit]()
        self.dynamics=DYNAMICS[strategy.dynamics]()
        self.readout=READOUTS[strategy.readout]()
        self.planner=PLANNERS[strategy.planning]()

    def run(
        self,
        episode:Episode,
        horizon:int=4,
        learn:bool=True,
        trace:bool=False,
    )->dict:
        graph=episode.graph.clone()
        query=episode.query

        self.memory.write(
            graph,
            query,
        )

        trace_rows=[]

        for step in range(1,horizon):
            self.dynamics.pre(
                graph,
                step,
            )
            self.memory.maintain(
                graph,
                query,
                step,
            )
            self.dynamics.post(
                graph,
                step,
            )

            if trace:
                trace_rows.append(
                    {
                        "step":step,
                        "node_count":len(graph.nodes),
                        "edge_count":len(graph.edges),
                    }
                )

        recalled,path=self.readout.read(
            graph,
            query,
        )

        predicted=self.planner.transform(
            graph,
            query,
            recalled,
        )

        correct=(
            predicted==episode.answer
        )

        if learn:
            self.credit.update(
                graph,
                query,
                predicted,
                episode.answer,
                path,
            )

        result={
            "predicted":predicted,
            "answer":episode.answer,
            "correct":correct,
            "path":path,
        }

        if trace:
            result["trace"]=trace_rows

        return result


def causal_probe(
    strategy:Strategy,
    episode:Episode,
    horizon:int=4,
)->dict:
    algo=CognitiveAlgorithm(strategy)

    normal=algo.run(
        episode,
        horizon=horizon,
        learn=False,
    )

    # Remove only the queried source->middle memory route. The graph remains
    # otherwise intact.
    ablated_graph=episode.graph.clone()

    first=next(
        (
            e for e in episode.graph.outgoing(
                episode.query.source,
                episode.query.relation_a,
            )
        ),
        None,
    )

    if first is not None:
        ablated_graph.remove_edge(
            first.source,
            first.relation,
            first.target,
        )

    ablated=algo.run(
        Episode(
            episode.seed,
            episode.family,
            ablated_graph,
            episode.query,
            episode.answer,
        ),
        horizon=horizon,
        learn=False,
    )

    # Swap the middle node for a decoy while keeping opaque graph structure.
    swapped_graph=episode.graph.clone()

    if first is not None:
        replacement=next(
            (
                n for n in swapped_graph.nodes
                if n not in {
                    episode.query.source,
                    first.target,
                    episode.query.target,
                }
            ),
            None,
        )

        if replacement is not None:
            swapped_graph.remove_edge(
                first.source,
                first.relation,
                first.target,
            )
            swapped_graph.add_edge(
                first.source,
                first.relation,
                replacement,
            )

    # Recompute the answer from the modified graph rather than hard-coding
    # False. For counterfactual family this still respects the structural
    # negation control encoded in the graph.
    swapped_path=path_exists(
        swapped_graph,
        episode.query,
    )
    base_exists=bool(swapped_path)

    has_negation=False
    for node in swapped_graph.nodes.values():
        if node.role!="query_control":
            continue

        has_mode=any(
            e.source==node.name
            and e.relation=="mode"
            for e in swapped_graph.edges
        )
        applies=any(
            e.source==node.name
            and e.relation=="applies_to"
            and e.target==episode.query.target
            for e in swapped_graph.edges
        )

        if has_mode and applies:
            has_negation=True

    swapped_answer=(
        (not base_exists)
        if has_negation
        else base_exists
    )

    swapped=algo.run(
        Episode(
            episode.seed,
            episode.family,
            swapped_graph,
            episode.query,
            swapped_answer,
        ),
        horizon=horizon,
        learn=False,
    )

    return {
        "normal_correct":int(normal["correct"]),
        "ablation_correct":int(ablated["correct"]),
        "ablation_drop":(
            int(normal["correct"])
            -int(ablated["correct"])
        ),
        "swap_changed":int(
            normal["predicted"]
            !=swapped["predicted"]
        ),
        "swap_correct":int(
            swapped["predicted"]
            ==swapped_answer
        ),
    }


def evaluate_strategy(
    strategy:Strategy,
    train_seeds:List[int],
    eval_seeds:List[int],
    horizon:int,
)->dict:
    # Training is intentionally separate from held-out evaluation. The
    # current modules are mostly structural, so train/eval separation is a
    # safeguard against accidentally mixing task graphs.
    train_eps=make_dataset(train_seeds)
    eval_eps=make_dataset(eval_seeds)

    algo=CognitiveAlgorithm(strategy)

    train_correct=0
    for ep in train_eps:
        train_correct+=int(
            algo.run(
                ep,
                horizon=horizon,
                learn=True,
            )["correct"]
        )

    eval_rows=[]
    for ep in eval_eps:
        eval_rows.append(
            algo.run(
                ep,
                horizon=horizon,
                learn=False,
            )
        )

    eval_accuracy=(
        sum(
            int(x["correct"])
            for x in eval_rows
        )
        /max(1,len(eval_rows))
    )

    probes=[
        causal_probe(
            strategy,
            ep,
            horizon=horizon,
        )
        for ep in eval_eps
    ]

    return {
        "strategy":strategy,
        "train_accuracy":(
            train_correct/max(
                1,len(train_eps)
            )
        ),
        "eval_accuracy":eval_accuracy,
        "causal_normal":(
            sum(x["normal_correct"] for x in probes)
            /len(probes)
        ),
        "causal_memory_drop":(
            sum(x["ablation_drop"] for x in probes)
            /len(probes)
        ),
        "causal_swap_change":(
            sum(x["swap_changed"] for x in probes)
            /len(probes)
        ),
        "causal_swap_correct":(
            sum(x["swap_correct"] for x in probes)
            /len(probes)
        ),
    }


def result_to_json(row: dict) -> dict:
    strategy=row["strategy"]

    return {
        "name":strategy.name,
        "strategy":{
            "memory":strategy.memory,
            "credit":strategy.credit,
            "dynamics":strategy.dynamics,
            "readout":strategy.readout,
            "planning":strategy.planning,
        },
        "train_accuracy":row["train_accuracy"],
        "eval_accuracy":row["eval_accuracy"],
        "causal_normal":row["causal_normal"],
        "causal_memory_drop":row["causal_memory_drop"],
        "causal_swap_change":row["causal_swap_change"],
        "causal_swap_correct":row["causal_swap_correct"],
    }
