
from __future__ import annotations
import json, random
from pathlib import Path
from state import ACTIONS, ACTION_TO_ID, State

RELATIONS=("IsA","PartOf","Causes","UsedFor","RelatedTo","AtLocation","HasProperty","CapableOf")
CONCEPTS=(
    "dog","animal","living_thing","entity","pet","wolf","mammal","organism",
    "creature","thing","cat","vehicle","car","travel","tree","forest","food",
    "water","home","person","tool","place","machine","object","plant","house",
    "road","city","bird","fish","metal","material","energy","motion","weather",
    "space","land","river","stone","sun","rain","wind","fire","earth","sky"
)

def add_edge(s, source, relation, target, activation=0.98):
    s.add_node(source,activation,1)
    s.add_node(target,activation,1)
    s.add_edge(source,relation,target,activation)

def make_example(action,rng,index,reasoning_steps=8):
    if reasoning_steps<8:
        raise ValueError("reasoning_steps must be >= 8")
    pool=list(CONCEPTS); rng.shuffle(pool)
    chain=pool[:reasoning_steps+1]
    relations=[rng.choice(("IsA","PartOf","RelatedTo")) for _ in range(reasoning_steps)]
    initial=State([],[])
    for i,rel in enumerate(relations):
        add_edge(initial,chain[i],rel,chain[i+1])
    initial.node(chain[0]).role=2
    initial.node(chain[0]).activation=1.0

    target_nodes=rng.randint(max(20,len(chain)+8),max(24,len(chain)+12))
    for concept in pool:
        if len(initial.nodes)>=target_nodes: break
        if concept in chain: continue
        initial.add_node(concept,rng.choice((.10,.20,.35,.55,.70)),rng.randint(0,5))
    while len(initial.nodes)<target_nodes:
        initial.add_node(f"distractor_{len(initial.nodes)}",
                         rng.choice((.10,.25,.50)),rng.randint(0,5))

    names=[n.concept for n in initial.nodes]
    for _ in range(rng.randint(20,32)):
        a,b=rng.sample(names,2); rel=rng.choice(RELATIONS)
        if not initial.has_edge(a,rel,b,active_only=False):
            initial.add_edge(a,rel,b,rng.choice((.08,.15,.25,.40,.60)))

    cursor=[
        {"action":"REUSE","source":None,"target":chain[i],"relation":None}
        for i in range(1,len(chain))
    ]
    terminal={
        "BIND":{"action":"BIND","source":chain[-2],"target":chain[-1],"relation":relations[-1]},
        "BRANCH":{"action":"BRANCH","source":chain[-1],"target":None,"relation":relations[-1]},
        "INHIBIT":{"action":"INHIBIT","source":None,"target":chain[-1],"relation":None},
        "REUSE":{"action":"REUSE","source":None,"target":chain[-1],"relation":None},
        "CREATE":{"action":"CREATE","source":None,"target":None,"relation":None},
        "COMMIT":{"action":"COMMIT","source":None,"target":None,"relation":None},
        "NOOP":{"action":"NOOP","source":None,"target":None,"relation":None},
    }[action]

    actions=cursor+[terminal]
    if len(actions)!=reasoning_steps+1:
        raise AssertionError("trajectory length construction failure")

    current=initial.clone()
    trajectory_states=[]; trajectory_attention=[]
    for t,step in enumerate(actions):
        trajectory_states.append(current.signature())
        trajectory_attention.append(chain[:min(t+1,len(chain))])
        current=current.apply(
            ACTION_TO_ID[step["action"]],
            source=step["source"],
            target=step["target"],
            relation=step["relation"],
        )

    return {
        "version":"v236",
        "case_id":f"v236_{index:05d}_{action.lower()}",
        "initial_state":initial.signature(),
        "goal":{
            "source":chain[0],
            "target":chain[-1],
            "relation":relations[-1],
            "depth":reasoning_steps,
        },
        "trajectory_states":trajectory_states,
        "trajectory_attention":trajectory_attention,
        "trajectory_actions":actions,
        "final_action":terminal,
        "reasoning_chain":chain,
        "reasoning_relations":relations,
        "chain_depth":reasoning_steps,
    }

def generate_dataset(samples=500,seed=236,reasoning_steps=8):
    if reasoning_steps<8: raise ValueError("reasoning_steps must be >= 8")
    rng=random.Random(seed)
    q,r=divmod(samples,len(ACTIONS))
    rows=[]
    for i,action in enumerate(ACTIONS):
        for _ in range(q+int(i<r)):
            rows.append(make_example(action,rng,len(rows),reasoning_steps))
    rng.shuffle(rows)
    return rows

def save_dataset(rows,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
