
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import hashlib
import random


ACTIONS=("NO","YES")
RELATIONS=("r0","r1","r2","r3")
TASKS=(
    "recall_bind",
    "relational_query",
    "interference",
    "counterfactual",
    "multi_step",
)


@dataclass
class Node:
    name:str
    role:str
    activation:float=0.0
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
        name:str,
        role:str,
        activation:float=0.0,
        persistent:bool=True,
    ):
        self.nodes[name]=Node(
            name,
            role,
            activation,
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

        for e in self.edges:
            if (
                e.source==source
                and e.relation==relation
                and e.target==target
            ):
                e.weight=weight
                e.persistent=e.persistent or persistent
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
        source:str,
        relation:str,
        target:str,
    ):
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
        source:str,
        relation:Optional[str]=None,
    )->List[Edge]:
        return [
            e for e in self.edges
            if e.source==source
            and e.weight>0.0
            and (
                relation is None
                or e.relation==relation
            )
        ]

    def activate(self,name:str,amount:float):
        self.ensure_node(name)
        self.nodes[name].activation+=amount

    def decay(self,factor:float):
        for n in self.nodes.values():
            n.activation*=factor
        for e in self.edges:
            if not e.persistent:
                e.weight*=factor


@dataclass(frozen=True)
class Query:
    source:str
    first_relation:str
    second_relation:str
    target:str


@dataclass(frozen=True)
class Episode:
    seed:int
    task:str
    graph:Graph
    query:Query
    context_bit:int
    third_bit:int
    answer_bit:int
    decision_step:int

    @property
    def answer_action(self)->str:
        return ACTIONS[self.answer_bit]


@dataclass
class Sequence:
    seed:int
    task:str
    latent_rule:int
    episodes:List[Episode]


def opaque(
    prefix:str,
    seed:int,
    index:int,
)->str:
    digest=hashlib.sha256(
        f"{prefix}:{seed}:{index}".encode()
    ).hexdigest()[:10]
    return f"{prefix}_{digest}"


def make_episode(
    seed:int,
    task:str,
    latent_rule:int,
    episode_index:int,
    horizon:int=6,
)->Episode:
    if task not in TASKS:
        raise ValueError(task)
    if horizon<4:
        raise ValueError(
            "horizon must be >= 4"
        )

    rng=random.Random(
        10007*seed
        +7919*TASKS.index(task)
        +101*episode_index
    )

    graph=Graph()

    source=opaque("q",seed,0)
    middle=opaque("q",seed,1)
    target=opaque("q",seed,2)
    decoy=opaque("d",seed,0)

    graph.add_node(
        source,
        "query_source",
    )
    graph.add_node(
        middle,
        "concept",
    )
    graph.add_node(
        target,
        "query_target",
    )
    graph.add_node(
        decoy,
        "concept",
    )

    rel_a,rel_b=rng.sample(
        RELATIONS,
        2,
    )
    decoy_rel=next(
        r for r in RELATIONS
        if r not in (rel_a,rel_b)
    )

    hidden=(
        seed
        +3*episode_index
    )%2

    context=(
        seed//2
        +episode_index
    )%2

    third=(
        seed//4
        +episode_index
    )%2

    # The hidden bit exists only in a transient sensory node. It is deliberately
    # NOT encoded anywhere in the persistent topology.
    fact=opaque("fact",seed,episode_index)
    graph.add_node(
        fact,
        "initial_fact",
        activation=float(hidden),
        persistent=False,
    )

    # Context arrives later and is represented generically.
    cue=opaque("cue",seed,episode_index)
    graph.add_node(
        cue,
        "context_cue",
        activation=float(context),
        persistent=True,
    )
    graph.add_edge(
        cue,
        "modulates",
        target,
    )

    if task=="multi_step":
        third_node=opaque(
            "third",
            seed,
            episode_index,
        )
        graph.add_node(
            third_node,
            "third_cue",
            activation=float(third),
            persistent=True,
        )
        graph.add_edge(
            third_node,
            "modulates",
            target,
        )

    # Crucially: structural query graph is IDENTICAL with respect to hidden bit.
    # Both hidden=0 and hidden=1 get the same positive relational path.
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

    # Decoy path is independent of the hidden answer.
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

    # Counterfactual control is generic graph structure.
    if task=="counterfactual":
        control=opaque(
            "control",
            seed,
            episode_index,
        )
        marker=opaque(
            "marker",
            seed,
            episode_index,
        )

        graph.add_node(
            control,
            "query_control",
            persistent=True,
        )
        graph.add_node(
            marker,
            "negation_marker",
            persistent=True,
        )

        if context:
            graph.add_edge(
                control,
                "mode",
                marker,
            )
            graph.add_edge(
                control,
                "applies_to",
                target,
            )

    # Nuisance graph changes every episode independently.
    nuisance=opaque(
        "noise",
        seed,
        episode_index,
    )
    graph.add_node(
        nuisance,
        "nuisance",
    )

    for i in range(4):
        n=opaque(
            "n",
            seed+episode_index*13,
            i,
        )
        graph.add_node(
            n,
            "nuisance",
        )
        graph.add_edge(
            nuisance,
            rng.choice(RELATIONS),
            n,
        )

    # Environment answer. Cognitive modules never receive this.
    if task in (
        "recall_bind",
        "relational_query",
        "interference",
    ):
        answer=hidden^context^latent_rule
    elif task=="multi_step":
        answer=hidden^context^third^latent_rule
    else:
        answer=(
            hidden
            ^ latent_rule
            if context==0
            else 1-(hidden^latent_rule)
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
        context_bit=context,
        third_bit=third,
        answer_bit=answer,
        decision_step=horizon-1,
    )


def make_sequence(
    seed:int,
    task:str,
    episodes_per_sequence:int=8,
    horizon:int=6,
)->Sequence:
    rng=random.Random(
        32113*seed
        +TASKS.index(task)
    )
    latent_rule=rng.randrange(2)

    episodes=[
        make_episode(
            seed,
            task,
            latent_rule,
            index,
            horizon=horizon,
        )
        for index in range(
            episodes_per_sequence
        )
    ]

    return Sequence(
        seed=seed,
        task=task,
        latent_rule=latent_rule,
        episodes=episodes,
    )


def make_sequences(
    seeds:List[int],
    episodes_per_sequence:int=8,
    horizon:int=6,
)->List[Sequence]:
    return [
        make_sequence(
            seed,
            task,
            episodes_per_sequence,
            horizon,
        )
        for seed in seeds
        for task in TASKS
    ]


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class Memory:
    name="none"

    def write(self,graph,query):
        return None

    def maintain(self,graph,query,step):
        return None


class PersistentFact(Memory):
    name="persistent_fact"

    def write(self,graph,query):
        sensory=next(
            (
                n for n in graph.nodes.values()
                if n.role=="initial_fact"
            ),
            None,
        )
        hidden_activation=(
            sensory.activation
            if sensory is not None
            else 0.0
        )

        graph.add_node(
            "working_fact",
            "working_fact",
            activation=(
                1.0
                if hidden_activation >= 0.5
                else -1.0
            ),
            persistent=True,
        )

    def maintain(self,graph,query,step):
        node=graph.nodes["working_fact"]

        if node.activation>=0.0:
            node.activation=max(
                node.activation,
                0.80,
            )
        else:
            node.activation=min(
                node.activation,
                -0.80,
            )


class EpisodicMemory(Memory):
    name="episodic"

    def write(self,graph,query):
        sensory=next(
            (
                n for n in graph.nodes.values()
                if n.role=="initial_fact"
            ),
            None,
        )
        hidden_activation=(
            sensory.activation
            if sensory is not None
            else 0.0
        )

        graph.add_node(
            "episode_memory",
            "episode_memory",
            activation=(
                1.0
                if hidden_activation >= 0.5
                else -1.0
            ),
            persistent=True,
        )

    def maintain(self,graph,query,step):
        node=graph.nodes["episode_memory"]

        if node.activation>=0.0:
            node.activation=max(
                node.activation,
                0.70,
            )
        else:
            node.activation=min(
                node.activation,
                -0.70,
            )


class WorkingState(Memory):
    name="working_state"

    def write(self,graph,query):
        sensory=next(
            (
                n for n in graph.nodes.values()
                if n.role=="initial_fact"
            ),
            None,
        )
        hidden_activation=(
            sensory.activation
            if sensory is not None
            else 0.0
        )

        graph.add_node(
            "working_state",
            "working_state",
            activation=(
                0.75
                if hidden_activation >= 0.5
                else -0.75
            ),
            persistent=True,
        )

    def maintain(self,graph,query,step):
        node=graph.nodes["working_state"]

        if node.activation>=0.0:
            node.activation=(
                0.97*node.activation
                +0.03
            )
        else:
            node.activation=(
                0.97*node.activation
                -0.03
            )


# ---------------------------------------------------------------------------
# Credit: state persists across a sequence and can learn the latent rule.
# ---------------------------------------------------------------------------

class Credit:
    name="none"

    def __init__(self):
        self.rule_score=0.0
        self.count=0

    def reset(self):
        self.rule_score=0.0
        self.count=0

    def update(
        self,
        predicted:int,
        answer:int,
        context:int,
    ):
        return None

    def rule_estimate(self)->int:
        return int(
            self.rule_score>=0.30
        )


class ImmediateCredit(Credit):
    name="immediate"

    def update(
        self,
        predicted,
        answer,
        context,
    ):
        # Delayed feedback tells us whether the previous policy was inverted.
        # The error bit is a generic consequence of the action and feedback;
        # no hidden rule is exposed.
        error=int(predicted!=answer)

        self.rule_score=(
            0.75*self.rule_score
            +0.25*error
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

    def update(
        self,
        predicted,
        answer,
        context,
    ):
        error=float(predicted!=answer)

        # Eligibility trace: accumulate the delayed action-error over time
        # with a slow trace before folding it into the latent policy estimate.
        self.eligibility=(
            0.82*self.eligibility
            +0.18*error
        )
        self.rule_score=(
            0.85*self.rule_score
            +0.15*self.eligibility
        )
        self.count+=1


class PathReinforcement(Credit):
    name="path_reinforcement"

    def __init__(self):
        super().__init__()
        self.context_bias={0:0.0,1:0.0}

    def reset(self):
        super().reset()
        self.context_bias={0:0.0,1:0.0}

    def update(
        self,
        predicted,
        answer,
        context,
    ):
        error=(
            1.0
            if predicted!=answer
            else 0.0
        )

        # Maintain credit separately for the current context, then pool it.
        self.context_bias[context]=(
            0.80*self.context_bias[context]
            +0.20*error
        )
        self.rule_score=(
            0.85*self.rule_score
            +0.15*(
                0.5
                *(
                    self.context_bias[0]
                    +self.context_bias[1]
                )
            )
        )
        self.count+=1



MEMORIES={
    "persistent_fact":PersistentFact,
    "episodic":EpisodicMemory,
    "working_state":WorkingState,
}

CREDITS={
    "none":Credit,
    "immediate":ImmediateCredit,
    "eligibility":EligibilityCredit,
    "path_reinforcement":PathReinforcement,
}


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

class Dynamics:
    name="static"

    def pre(self,graph,step):
        return None

    def post(self,graph,step):
        return None


class Leaky(Dynamics):
    name="leaky"

    def pre(self,graph,step):
        graph.decay(0.90)

    def post(self,graph,step):
        graph.decay(0.97)


class Recurrent(Dynamics):
    name="recurrent"

    def pre(self,graph,step):
        graph.decay(0.96)

        for node in graph.nodes.values():
            if node.role in (
                "working_fact",
                "episode_memory",
                "working_state",
            ):
                node.activation=min(
                    1.0,
                    node.activation+0.03,
                )

    def post(self,graph,step):
        graph.decay(0.98)


class Selective(Dynamics):
    name="selective"

    def pre(self,graph,step):
        graph.decay(0.93)

        for node in graph.nodes.values():
            if node.role in (
                "working_fact",
                "episode_memory",
                "working_state",
            ):
                node.activation=max(
                    node.activation,
                    0.72,
                )

    def post(self,graph,step):
        graph.decay(0.99)


DYNAMICS={
    "static":Dynamics,
    "leaky":Leaky,
    "recurrent":Recurrent,
    "selective":Selective,
}


# ---------------------------------------------------------------------------
# Readout
# ---------------------------------------------------------------------------

class Readout:
    name="base"

    def read(self,graph,query):
        raise NotImplementedError


class MemoryReadout(Readout):
    name="memory"

    def read(self,graph,query):
        for name in (
            "working_fact",
            "episode_memory",
            "working_state",
        ):
            node=graph.nodes.get(name)
            if node is not None:
                return int(
                    node.activation>=0.0
                ),[]

        return 0,[]


class RelationalReadout(Readout):
    name="relational"

    def read(self,graph,query):
        # Structural query is not answer-coded. It supplies a path trace for
        # credit attribution, while the actual recalled bit still comes from
        # memory.
        path=[]

        for first in graph.outgoing(
            query.source,
            query.first_relation,
        ):
            for second in graph.outgoing(
                first.target,
                query.second_relation,
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

        bit,_=MemoryReadout().read(
            graph,
            query,
        )

        return bit,path


class IntegrativeReadout(Readout):
    name="integrative"

    def read(self,graph,query):
        bit,path=RelationalReadout().read(
            graph,
            query,
        )

        # Require the query path, but do not use it as the answer itself.
        return (
            bit if path else 0,
            path,
        )


class CreditReadout(Readout):
    name="credit"

    def read(self,graph,query):
        bit,path=RelationalReadout().read(
            graph,
            query,
        )

        credit=graph.nodes.get(
            "credit_signal"
        )

        if credit is not None:
            if credit.activation<-0.5:
                bit=1-bit

        return bit,path


READOUTS={
    "memory":MemoryReadout,
    "relational":RelationalReadout,
    "integrative":IntegrativeReadout,
    "credit":CreditReadout,
}


# ---------------------------------------------------------------------------
# Planner / binding
# ---------------------------------------------------------------------------

class Planner:
    name="none"

    def decide(
        self,
        graph,
        query,
        recalled,
        rule_estimate,
    ):
        return int(recalled)


def cue_values(
    graph:Graph,
    target:str,
)->List[int]:
    values=[]

    for edge in graph.edges:
        if (
            edge.relation=="modulates"
            and edge.target==target
        ):
            node=graph.nodes.get(edge.source)

            if node is None:
                continue

            if node.role=="context_cue":
                order=1
            elif node.role=="third_cue":
                order=2
            else:
                continue

            values.append(
                (
                    order,
                    node.name,
                    int(node.activation>=0.5),
                )
            )

    return [
        value
        for _,_,value in sorted(values)
    ]


class BindPlanner(Planner):
    name="bind"

    def decide(
        self,
        graph,
        query,
        recalled,
        rule_estimate,
    ):
        value=int(recalled)

        for cue in cue_values(
            graph,
            query.target,
        ):
            value^=cue

        return value^int(
            rule_estimate
        )


class ControlPlanner(Planner):
    name="control"

    def decide(
        self,
        graph,
        query,
        recalled,
        rule_estimate,
    ):
        value=BindPlanner().decide(
            graph,
            query,
            recalled,
            rule_estimate,
        )

        invert=False

        for node in graph.nodes.values():
            if node.role!="query_control":
                continue

            applies=any(
                e.source==node.name
                and e.relation=="applies_to"
                and e.target==query.target
                for e in graph.edges
            )

            if not applies:
                continue

            if any(
                graph.nodes.get(e.target) is not None
                and graph.nodes[e.target].role
                    =="negation_marker"
                for e in graph.edges
                if (
                    e.source==node.name
                    and e.relation=="mode"
                )
            ):
                invert=True

        return 1-value if invert else value


class RolloutPlanner(Planner):
    name="rollout"

    def decide(
        self,
        graph,
        query,
        recalled,
        rule_estimate,
    ):
        value=int(recalled)

        for cue in cue_values(
            graph,
            query.target,
        ):
            value^=cue

        value^=int(
            rule_estimate
        )

        # Generic control operation.
        for node in graph.nodes.values():
            if node.role!="query_control":
                continue

            applies=any(
                e.source==node.name
                and e.relation=="applies_to"
                and e.target==query.target
                for e in graph.edges
            )

            if not applies:
                continue

            if any(
                graph.nodes.get(e.target) is not None
                and graph.nodes[e.target].role
                    =="negation_marker"
                for e in graph.edges
                if (
                    e.source==node.name
                    and e.relation=="mode"
                )
            ):
                value=1-value

        return value


PLANNERS={
    "none":Planner,
    "bind":BindPlanner,
    "control":ControlPlanner,
    "rollout":RolloutPlanner,
}


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Strategy:
    memory:str
    credit:str
    dynamics:str
    readout:str
    planner:str

    @property
    def name(self):
        return (
            f"{self.memory}+{self.credit}+"
            f"{self.dynamics}+{self.readout}+"
            f"{self.planner}"
        )


def all_strategies()->List[Strategy]:
    return [
        Strategy(
            m,c,d,r,p
        )
        for m in MEMORIES
        for c in CREDITS
        for d in DYNAMICS
        for r in READOUTS
        for p in PLANNERS
    ]


def strategy_count()->int:
    return (
        len(MEMORIES)
        *len(CREDITS)
        *len(DYNAMICS)
        *len(READOUTS)
        *len(PLANNERS)
    )


class CognitiveAlgorithm:
    def __init__(self,strategy:Strategy):
        self.strategy=strategy
        self.memory=MEMORIES[
            strategy.memory
        ]()
        self.credit=CREDITS[
            strategy.credit
        ]()
        self.dynamics=DYNAMICS[
            strategy.dynamics
        ]()
        self.readout=READOUTS[
            strategy.readout
        ]()
        self.planner=PLANNERS[
            strategy.planner
        ]()

    def reset(self):
        self.credit.reset()

    def run(
        self,
        ep:Episode,
        learn:bool=True,
        trace:bool=False,
    ):
        graph=ep.graph.clone()

        # Delayed credit is available as a generic state signal, never as the
        # hidden answer or task identity.
        signal=self.credit.rule_estimate()
        graph.add_node(
            "credit_signal",
            "credit_signal",
            activation=(
                0.75 if signal else 0.0
            ),
            persistent=True,
        )

        self.memory.write(
            graph,
            ep.query,
        )

        # Sensory fact disappears immediately after perception/memory write.
        initial=next(
            (
                name for name in list(
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

        trace_rows=[]

        for step in range(
            1,
            ep.decision_step+1,
        ):
            self.dynamics.pre(
                graph,
                step,
            )
            self.memory.maintain(
                graph,
                ep.query,
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
                        "memory_nodes":sum(
                            n.role in (
                                "working_fact",
                                "episode_memory",
                                "working_state",
                            )
                            for n in graph.nodes.values()
                        ),
                        "edge_count":len(graph.edges),
                    }
                )

        recalled,path=self.readout.read(
            graph,
            ep.query,
        )

        decision=self.planner.decide(
            graph,
            ep.query,
            recalled,
            self.credit.rule_estimate(),
        )

        action=ACTIONS[decision]

        correct=(
            action==ep.answer_action
        )

        if learn:
            self.credit.update(
                decision,
                ep.answer_bit,
                ep.context_bit,
            )

        result={
            "correct":correct,
            "action":action,
            "answer":ep.answer_action,
            "decision":decision,
            "recalled":recalled,
            "path":path,
        }

        if trace:
            result["trace"]=trace_rows

        return result


def evaluate_sequence(
    strategy:Strategy,
    sequence:Sequence,
    learn:bool=True,
)->dict:
    algo=CognitiveAlgorithm(strategy)

    rows=[]

    for index,episode in enumerate(
        sequence.episodes
    ):
        rows.append(
            algo.run(
                episode,
                learn=learn,
            )
        )

    return {
        "accuracy":(
            sum(
                int(r["correct"])
                for r in rows
            )
            /len(rows)
        ),
        "first_half_accuracy":(
            sum(
                int(r["correct"])
                for r in rows[:len(rows)//2]
            )
            /(len(rows)//2)
        ),
        "second_half_accuracy":(
            sum(
                int(r["correct"])
                for r in rows[len(rows)//2:]
            )
            /(len(rows)//2)
        ),
        "rows":rows,
    }


def causal_probe(
    strategy:Strategy,
    ep:Episode,
)->dict:
    algo=CognitiveAlgorithm(strategy)

    normal=algo.run(
        ep,
        learn=False,
    )

    # Memory ablation: remove the transient sensory fact and any persistent
    # memory carriers. The decision-time graph must therefore lack the initial
    # information.
    ablated_graph=ep.graph.clone()

    for node_name in list(
        ablated_graph.nodes
    ):
        node=ablated_graph.nodes[node_name]

        if node.role in (
            "initial_fact",
            "working_fact",
            "episode_memory",
            "working_state",
        ):
            ablated_graph.nodes.pop(
                node_name,
                None,
            )

    ablated=algo.run(
        Episode(
            ep.seed,
            ep.task,
            ablated_graph,
            ep.query,
            ep.context_bit,
            ep.third_bit,
            ep.answer_bit,
            ep.decision_step,
        ),
        learn=False,
    )

    # Hidden-fact swap: preserve every visible cue and query relation, but flip
    # the initial sensory fact before the memory write phase.
    swapped_graph=ep.graph.clone()

    initial_name=next(
        (
            name for name in swapped_graph.nodes
            if swapped_graph.nodes[name].role
                =="initial_fact"
        ),
        None,
    )

    if initial_name is not None:
        old_value=(
            swapped_graph.nodes[
                initial_name
            ].activation
        )
        swapped_graph.nodes[
            initial_name
        ].activation=1.0-old_value

    swapped_answer=1-ep.answer_bit

    swapped=algo.run(
        Episode(
            ep.seed,
            ep.task,
            swapped_graph,
            ep.query,
            ep.context_bit,
            ep.third_bit,
            swapped_answer,
            ep.decision_step,
        ),
        learn=False,
    )

    return {
        "normal_correct":int(
            normal["correct"]
        ),
        "ablation_correct":int(
            ablated["correct"]
        ),
        "memory_drop":(
            int(normal["correct"])
            -int(ablated["correct"])
        ),
        "swap_changed":int(
            normal["decision"]
            !=swapped["decision"]
        ),
        "swap_correct":int(
            swapped["correct"]
        ),
    }
