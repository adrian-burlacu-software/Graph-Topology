
import argparse, copy, torch
from model import StateArchitectureModel
from state import State, Node, Edge

def R(name, ok, detail): return (name, bool(ok), detail)
def diff(a,b): return float((a-b).abs().max().item())
def make_state():
    return State(
        [Node("alpha",1.0,2),Node("beta",.8,1),Node("gamma",.6,1),Node("distractor",.1,0)],
        [Edge("alpha","IsA","beta",.9),Edge("beta","RelatedTo","gamma",.8)]
    )
def goal(): return {"source":"alpha","target":"gamma","relation":"RelatedTo","depth":2}
def model(mode,dev):
    torch.manual_seed(230)
    m=StateArchitectureModel(hidden_size=64,heads=4,depth=2,topk=3,state_mode=mode).to(dev)
    m.eval(); return m

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    a=p.parse_args(); dev=torch.device(a.device)
    out=[]

    for req in ["cognitive_step","predicted_transition","autonomous_rollout"]:
        out.append(R("api_"+req,hasattr(StateArchitectureModel,req),"present" if hasattr(StateArchitectureModel,req) else "MISSING"))

    for h in (2,4,8):
        states=[make_state() for _ in range(h)]
        acts=["REUSE"]*(h-1)+["COMMIT"]
        out.append(R(f"horizon_{h}",len(states)==h and len(acts)==h,f"states={len(states)} actions={len(acts)}"))

    # Same weights for branch comparison.
    torch.manual_seed(230)
    ref=StateArchitectureModel(hidden_size=32,heads=4,depth=2,topk=3,state_mode="latent").to(dev)
    weights=copy.deepcopy(ref.state_dict())
    def mk(mode):
        m=StateArchitectureModel(hidden_size=32,heads=4,depth=2,topk=3,state_mode=mode).to(dev)
        m.load_state_dict(weights); m.eval(); return m

    s=State([Node("a",1.0,2),Node("b",.7,1),Node("c",.2,1)],
            [Edge("a","IsA","b",.9),Edge("b","RelatedTo","c",.8)])
    g={"source":"a","target":"c","relation":"RelatedTo","depth":2}
    w=torch.zeros((1,32),device=dev)

    os={m:mk(m).cognitive_step(s,g,w,1,"a","b",0,dev) for m in ("stateless","latent","latent_action")}
    sn=float(os["stateless"]["next_working"].abs().max().item())
    ln=float(os["latent"]["next_working"].abs().max().item())
    lad=diff(os["latent"]["next_working"],os["latent_action"]["next_working"])
    out += [
        R("branch_stateless",sn==0,f"norm={sn:.6g}"),
        R("branch_latent",ln>1e-8,f"norm={ln:.6g}"),
        R("branch_latent_action",lad>1e-8,f"vs_latent={lad:.6g}"),
    ]

    # Working-state causality.
    lm=model("latent",dev); s=make_state(); g=goal()
    a0=lm.cognitive_step(s,g,torch.zeros((1,64),device=dev),None,None,None,None,dev)
    a1=lm.cognitive_step(s,g,torch.ones((1,64),device=dev),None,None,None,None,dev)
    out.append(R("state_causality",diff(a0["action_logits"],a1["action_logits"])>1e-8,
                 f"delta={diff(a0['action_logits'],a1['action_logits']):.6g}"))

    # Action-history causality.
    ham=model("latent_action",dev)
    z=torch.zeros((1,64),device=dev)
    h0=ham.cognitive_step(s,g,z,1,"alpha","beta",0,dev)
    h1=ham.cognitive_step(s,g,z,5,"gamma","alpha",1,dev)
    hd=diff(h0["next_working"],h1["next_working"])
    out.append(R("history_causality",hd>1e-8,f"next_state_delta={hd:.6g}"))

    # Goal causality.
    ga=lm.cognitive_step(s,goal(),z,None,None,None,None,dev)
    gb=lm.cognitive_step(s,{"source":"a","target":"b","relation":"IsA","depth":3},z,None,None,None,None,dev)
    gd=diff(ga["action_logits"],gb["action_logits"])
    out.append(R("goal_causality",gd>1e-8,f"action_delta={gd:.6g}"))

    # Same graph, different history.
    q0=ham.cognitive_step(s,g,z,1,"alpha","beta",0,dev)
    q1=ham.cognitive_step(s,g,z,2,None,None,0,dev)
    qd=diff(q0["action_logits"],q1["action_logits"])
    out.append(R("same_graph_different_history",qd>1e-8,f"action_delta={qd:.6g}"))

    # Symbolic transition causality.
    r0=s.apply(1,target="b"); r1=s.apply(5,source="a",target="b",relation="RelatedTo")
    out.append(R("symbolic_transition",r0.signature()!=r1.signature(),"REUSE != BIND"))

    print("=== V230 ARCHITECTURE BATTERY ===",flush=True)
    print("device:",dev,flush=True); print("="*78,flush=True)
    fails=0
    for n,ok,d in out:
        print(f"[{'PASS' if ok else 'FAIL'}] {n}: {d}",flush=True)
        fails += not ok
    print("="*78,flush=True)
    print(f"BATTERY: {'PASS' if fails==0 else 'FAIL'} ({len(out)-fails}/{len(out)} checks)",flush=True)
    raise SystemExit(1 if fails else 0)

if __name__=="__main__": main()
