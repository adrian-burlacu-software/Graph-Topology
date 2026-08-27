from __future__ import annotations
import argparse,json,os,random,subprocess,sys,time
from collections import Counter
from pathlib import Path
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent; RESEARCH=HERE.parent
if str(RESEARCH) not in sys.path: sys.path.insert(0,str(RESEARCH))
from dataset import generate_dataset,save_dataset
from model import StateArchitectureModel
from state import ACTIONS,ACTION_TO_ID,Node,Edge,State
from v200_graph_transformer_cognitive.long_term_memory import RELATION_TO_ID

def state_from_json(p):
    return State([Node(str(n['concept']),float(n['activation']),int(n['role']),bool(n.get('persistent',False))) for n in p['nodes']], [Edge(str(e['source']),str(e['relation']),str(e['target']),float(e['activation']),bool(e.get('persistent',False))) for e in p['edges']])

def prepare(rows):
    out=[]
    for r in rows:
        states=[state_from_json(s) for s in r['trajectory_states']]; ts=[]
        for t,s in enumerate(states):
            names=[n.concept for n in s.nodes]; a=r['trajectory_actions'][t]; ts.append({'concepts':tuple(r['trajectory_attention'][t]),'action':ACTION_TO_ID[a['action']],'source':a['source'],'target':a['target'],'relation':RELATION_TO_ID.get(a['relation'],0)})
        out.append({'case_id':r['case_id'],'initial':state_from_json(r['initial_state']),'states':states,'targets':ts,'goal':r['goal'],'final_action':ACTION_TO_ID[r['final_action']['action']]})
    return out

def split(rows,seed,f=.15):
    g={a:[] for a in ACTIONS}; rng=random.Random(seed)
    for i,r in enumerate(rows): g[r['final_action']['action']].append(i)
    tr=[];va=[]
    for a in ACTIONS:
        ids=g[a]; rng.shuffle(ids); n=max(1,int(len(ids)*f)); va+=ids[:n]; tr+=ids[n:]
    rng.shuffle(tr);rng.shuffle(va); assert not(set(tr)&set(va)); return tr,va

def sanity(rows,data):
    c=Counter(r['final_action']['action'] for r in rows); assert set(c)==set(ACTIONS),c
    for r,d in zip(rows,data):
        assert len(r['trajectory_states'])==len(r['trajectory_actions'])==len(r['trajectory_attention']),r['case_id']
        assert r['trajectory_actions'][-1]['action']==r['final_action']['action'],r['case_id']
        for t,a in enumerate(r['trajectory_actions']):
            attention=set(r['trajectory_attention'][t])
            if a.get('source') is not None and a['source'] not in attention:
                raise AssertionError((r['case_id'],t,'source_not_attended',a['source']))
            if a.get('target') is not None and a['target'] not in attention:
                raise AssertionError((r['case_id'],t,'target_not_attended',a['target']))
            if t < len(r['trajectory_actions'])-1 and a['action']=='REUSE':
                assert a['target'] is not None
                assert d['states'][t+1].focus()==a['target'],(r['case_id'],t)
    print('preflight: PASS',flush=True)

def align(out,target,state,device):
    names=[n.concept for n in state.nodes]; truth=torch.tensor([float(n in set(target['concepts'])) for n in names],dtype=torch.float32,device=device)
    assert truth.shape==out['attention_logits'].shape,(len(names),truth.shape,out['attention_logits'].shape)
    return truth

def loss(out,target,state,device):
    truth=align(out,target,state,device); names=[n.concept for n in state.nodes]
    la=F.cross_entropy(out['action_logits'][None],torch.tensor([target['action']],device=device)); lh=F.binary_cross_entropy_with_logits(out['attention_logits'],truth)
    ls=F.cross_entropy(out['source_logits'][None],torch.tensor([names.index(target['source'])],device=device)) if target['source'] in names else out['source_logits'].sum()*0
    lt=F.cross_entropy(out['target_logits'][None],torch.tensor([names.index(target['target'])],device=device)) if target['target'] in names else out['target_logits'].sum()*0
    lr=F.cross_entropy(out['relation_logits'][None],torch.tensor([target['relation']],device=device))
    return la+.35*lh+.35*ls+.35*lt+.25*lr

def metric(c,out,target,state):
    c["action"] += int(out["action_logits"].argmax().item()==target["action"])
    names=[n.concept for n in state.nodes]
    if target["source"] is not None:
        c["source_n"] += 1
        c["source"] += int(target["source"] in names and out["source_logits"].argmax().item()==names.index(target["source"]))
    if target["target"] is not None:
        c["target_n"] += 1
        c["target"] += int(target["target"] in names and out["target_logits"].argmax().item()==names.index(target["target"]))
    c["relation"] += int(out["relation_logits"].argmax().item()==target["relation"])
    truth=align(out,target,state,out["attention_hard"].device).bool(); pred=out["attention_hard"]>.5
    c["tp"]+=int((pred&truth).sum()); c["fp"]+=int((pred&~truth).sum()); c["fn"]+=int((~pred&truth).sum())

def finish(c,n):
    p=c["tp"]/max(1,c["tp"]+c["fp"]); r=c["tp"]/max(1,c["tp"]+c["fn"]); f=2*p*r/max(1e-9,p+r)
    return {"action":c["action"]/max(1,n),"source":c["source"]/max(1,c["source_n"]),"target":c["target"]/max(1,c["target_n"]),"relation":c["relation"]/max(1,n),"hard_att_f1":f,"hard_att_p":p,"hard_att_r":r}

def train_epoch(m,data,ids,dev,opt,steps,regime,epoch,epochs,training,every,rng):
    m.train(training); total=0.; c=Counter(); nd=0; start=time.perf_counter(); prob=0 if regime=='teacher' else 1 if regime=='free' else (1 if epochs<=1 else (epoch-1)/max(1,epochs-1))
    for pos,idx in enumerate(ids,1):
        item=data[idx]; current=item['initial'].clone(); working=torch.zeros((1,m.hidden_size),device=dev); prev_a=prev_s=prev_t=prev_r=None; seq=0.; outs=[]
        with torch.set_grad_enabled(training):
            for t in range(min(steps,len(item['states']))):
                if t>0:
                    use= regime=='free' or (regime=='scheduled' and rng.random()<prob)
                    if not use: current=item['states'][t].clone()
                o=m.cognitive_step(current,item['goal'],working,prev_a,prev_s,prev_t,prev_r,dev); tgt=item['targets'][t]
                seq=seq+(1+.15*t)*loss(o,tgt,current,dev); metric(c,o,tgt,current); outs.append((o,tgt)); nd+=1; working=o['next_working']
                if t+1<min(steps,len(item['states'])):
                    if regime=='teacher' or (regime=='scheduled' and rng.random()>=prob):
                        nxt=item['states'][t+1].clone(); prev_a,prev_s,prev_t,prev_r=tgt['action'],tgt['source'],tgt['target'],tgt['relation']
                    else:
                        nxt,aid,src,tar,rid=m.predicted_transition(current,o); current=nxt; prev_a,prev_s,prev_t,prev_r=aid,src,tar,rid; continue
                    current=nxt
            seq=seq/max(1,len(outs))
            if training: opt.zero_grad(set_to_none=True); seq.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        total+=seq.item()
        if pos==1 or pos%every==0 or pos==len(ids):
            el=time.perf_counter()-start; rate=pos/max(el,1e-9); eta=(len(ids)-pos)/max(rate,1e-9); print(f"  [{'train' if training else 'valid'} {pos}/{len(ids)}] loss={total/pos:.4f} rate={rate:.2f}/s eta={eta:.1f}s model_state_p={prob:.2f}",flush=True)
    return {'loss':total/max(1,len(ids)),**finish(c,nd)}

@torch.no_grad()
def autonomous(m,data,ids,dev,steps):
    m.eval(); exact=0; final=0; cases=0; per=[{'cases':0,'action':0,'node_error':0.,'edge_error':0.} for _ in range(steps)]
    for idx in ids:
        item=data[idx]; roll=m.autonomous_rollout(item['initial'],item['goal'],dev,steps)
        pred=[x['action_id'] for x in roll['outputs']]; truth=[x['action'] for x in item['targets'][:steps]]; exact+=int(pred==truth); final+=int(pred==truth and len(pred)==len(truth))
        for t in range(min(steps,len(pred),len(item['states']))):
            names_a={n.concept for n in roll['states'][t].nodes}; names_b={n.concept for n in item['states'][t].nodes}; edges_a={(e.source,e.relation,e.target) for e in roll['states'][t].edges}; edges_b={(e.source,e.relation,e.target) for e in item['states'][t].edges}; nu=len(names_a|names_b); eu=len(edges_a|edges_b); per[t]['cases']+=1; per[t]['action']+=int(pred[t]==truth[t]); per[t]['node_error']+=1-len(names_a&names_b)/nu if nu else 0; per[t]['edge_error']+=1-len(edges_a&edges_b)/eu if eu else 0
        cases+=1
    return {'autonomous_exact_trajectory':exact/max(1,cases),'autonomous_final_action':final/max(1,cases),'per_step':[{'step':i+1,'cases':x['cases'],'action_accuracy':x['action']/max(1,x['cases']),'node_error':x['node_error']/max(1,x['cases']),'edge_error':x['edge_error']/max(1,x['cases'])} for i,x in enumerate(per)]}

def run(args,state_mode,regime,steps,data,tr,va):
    tag=f'v229_{state_mode}_{regime}_d{args.depth}_steps{steps}'; print(f"\n{'='*72}\n{tag}\n{'='*72}",flush=True)
    m=StateArchitectureModel(hidden_size=args.hidden_size,heads=args.heads,depth=args.depth,topk=args.topk,state_mode=state_mode).to(args.device); opt=torch.optim.AdamW(m.parameters(),lr=args.lr,weight_decay=1e-4); best=1e9; best_ep=0; bestm=None; rng=random.Random(args.seed+sum(map(ord,tag)))
    for ep in range(1,args.epochs+1):
        trm=train_epoch(m,data,tr,args.device,opt,steps,regime,ep,args.epochs,True,args.progress_every,rng); vam=train_epoch(m,data,va,args.device,None,steps,regime,ep,args.epochs,False,args.progress_every,rng)
        print(f"EPOCH {ep} train_loss={trm['loss']:.4f} train_action={trm['action']:.4f} valid_loss={vam['loss']:.4f} valid_action={vam['action']:.4f} valid_att={vam['hard_att_f1']:.4f}",flush=True)
        if vam['loss']<best:
            best=vam['loss'];best_ep=ep;bestm=vam; path=args.output_dir/f'{tag}.pt'; torch.save({'version':'v229','experiment':tag,'epoch':ep,'model':m.state_dict(),'valid':vam},path); print('checkpoint_saved:',path,flush=True)
    state=torch.load(path,map_location=args.device,weights_only=False);m.load_state_dict(state['model']); auto=autonomous(m,data,va,args.device,steps)
    print(f"AUTONOMOUS exact={auto['autonomous_exact_trajectory']:.4f}",flush=True)
    return {'experiment':tag,'state_mode':state_mode,'regime':regime,'steps':steps,'transformer_depth':args.depth,'best_epoch':best_ep,'best_valid_loss':best,'best':bestm,'autonomous':auto,'checkpoint':str(path)}

def worker(args):
    payload=json.loads(args.manifest.read_text(encoding='utf-8')); assert payload['version']=='v229'; rows=payload['rows']; data=prepare(rows); sanity(rows,data); tr,va=payload['train_ids'],payload['valid_ids']; args.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); args.output_dir.mkdir(parents=True,exist_ok=True); result=run(args,args.state_modes[0],args.regimes[0],args.steps[0],data,tr,va); (args.output_dir/f"{result['experiment']}.json").write_text(json.dumps(result,indent=2),encoding='utf-8')


def architecture_effect_preflight(device):
    """
    Cheap deterministic check that the three state modes are real architectural
    branches and not merely labels.

    Same weights + same graph + same goal:
      stateless       -> zero next working state
      latent          -> non-zero next working state
      latent_action   -> different next working state
    """
    import copy

    torch.manual_seed(229)

    template = StateArchitectureModel(
        hidden_size=32,
        heads=4,
        depth=2,
        topk=3,
        state_mode="latent",
    ).to(device)
    template.eval()

    weights = copy.deepcopy(template.state_dict())

    def make(mode):
        model = StateArchitectureModel(
            hidden_size=32,
            heads=4,
            depth=2,
            topk=3,
            state_mode=mode,
        ).to(device)
        model.load_state_dict(weights, strict=True)
        model.eval()
        return model

    probe_state = State(
        [
            Node("a", 1.0, 2),
            Node("b", 0.7, 1),
            Node("c", 0.2, 1),
        ],
        [
            Edge("a", "IsA", "b", 0.9),
            Edge("b", "RelatedTo", "c", 0.8),
        ],
    )

    probe_goal = {
        "source": "a",
        "target": "c",
        "relation": "RelatedTo",
        "depth": 2,
    }

    working = torch.zeros(
        (1, 32),
        device=device,
    )

    outputs = {
        mode: make(mode).cognitive_step(
            probe_state,
            probe_goal,
            working,
            1,
            "a",
            "b",
            0,
            device,
        )
        for mode in (
            "stateless",
            "latent",
            "latent_action",
        )
    }

    stateless_norm = float(
        outputs["stateless"]["next_working"]
        .abs()
        .max()
        .item()
    )

    latent_norm = float(
        outputs["latent"]["next_working"]
        .abs()
        .max()
        .item()
    )

    latent_vs_stateless = float(
        (
            outputs["latent"]["next_working"]
            - outputs["stateless"]["next_working"]
        )
        .abs()
        .max()
        .item()
    )

    latent_action_vs_latent = float(
        (
            outputs["latent_action"]["next_working"]
            - outputs["latent"]["next_working"]
        )
        .abs()
        .max()
        .item()
    )

    if stateless_norm != 0.0:
        raise AssertionError(
            "architecture preflight failed: "
            "stateless next_working is non-zero"
        )

    if latent_norm <= 1e-8:
        raise AssertionError(
            "architecture preflight failed: "
            "latent next_working is zero"
        )

    if latent_vs_stateless <= 1e-8:
        raise AssertionError(
            "architecture preflight failed: "
            "latent collapsed to stateless"
        )

    if latent_action_vs_latent <= 1e-8:
        raise AssertionError(
            "architecture preflight failed: "
            "latent_action collapsed to latent"
        )

    print("", flush=True)
    print("=" * 72, flush=True)
    print("V229 ARCHITECTURE EFFECT PREFLIGHT", flush=True)
    print("=" * 72, flush=True)
    print(
        f"stateless_next_state_norm = {stateless_norm:.6f}",
        flush=True,
    )
    print(
        f"latent_next_state_norm = {latent_norm:.6f}",
        flush=True,
    )
    print(
        f"latent_vs_stateless = {latent_vs_stateless:.6f}",
        flush=True,
    )
    print(
        f"latent_action_vs_latent = {latent_action_vs_latent:.6f}",
        flush=True,
    )
    print(
        "architecture_effect_preflight: PASS",
        flush=True,
    )
    print("=" * 72, flush=True)
    print("", flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--worker',action='store_true'); p.add_argument('--manifest',type=Path); p.add_argument('--samples',type=int,default=500); p.add_argument('--epochs',type=int,default=5); p.add_argument('--seed',type=int,default=225); p.add_argument('--lr',type=float,default=2e-4); p.add_argument('--hidden-size',type=int,default=128); p.add_argument('--heads',type=int,default=4); p.add_argument('--topk',type=int,default=5); p.add_argument('--depth',type=int,default=8); p.add_argument('--parallelism',type=int,default=2); p.add_argument('--progress-every',type=int,default=25); p.add_argument('--steps',type=int,nargs='+',default=[2,4,8]); p.add_argument('--state-modes',nargs='+',choices=['stateless','latent','latent_action'],default=['stateless','latent','latent_action']); p.add_argument('--regimes',nargs='+',choices=['teacher','scheduled','free'],default=['free']); p.add_argument('--output-dir',type=Path,default=Path('results/v229')); p.add_argument('--dataset-output',type=Path,default=Path('results/v229_state_architecture_dataset.jsonl')); a=p.parse_args()
    if 6 in a.steps:
        raise ValueError("V229 deliberately excludes the 6-step experiment.")

    if 6 in a.steps: raise ValueError('6-step experiment intentionally removed')
    if a.worker: worker(a); return
    random.seed(a.seed);torch.manual_seed(a.seed);a.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('=== V229 STATE ARCHITECTURE MATRIX ===',flush=True); print('device:',a.device,flush=True); 
    architecture_effect_preflight(a.device)
    if a.device.type=='cuda': print('gpu:',torch.cuda.get_device_name(0),flush=True)
    rows=generate_dataset(a.samples,a.seed); save_dataset(rows,a.dataset_output); data=prepare(rows); sanity(rows,data); tr,va=split(rows,a.seed); assert len(rows)==len(data)
    manifest=a.output_dir/'v229_worker_manifest.json';a.output_dir.mkdir(parents=True,exist_ok=True); manifest.write_text(json.dumps({'version':'v229','rows':rows,'train_ids':tr,'valid_ids':va},indent=2),encoding='utf-8'); check=json.loads(manifest.read_text(encoding='utf-8'));assert len(check['rows'])==a.samples and check['train_ids']==tr and check['valid_ids']==va
    print('dataset_size:',len(rows),flush=True);print('train_size:',len(tr),'valid_size:',len(va),flush=True);print('manifest_sanity: PASS',flush=True);print('state_modes:',a.state_modes,flush=True);print('regimes:',a.regimes,flush=True);print('steps:',a.steps,flush=True);print('parallelism:',a.parallelism,flush=True)
    jobs=[(sm,rg,st) for sm in a.state_modes for rg in a.regimes for st in a.steps]; print('matrix_cells:',len(jobs),flush=True); pending=list(jobs);active=[];done=[]
    while pending or active:
        while pending and len(active)<a.parallelism:
            sm,rg,st=pending.pop(0); cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--manifest',str(manifest),'--samples',str(a.samples),'--epochs',str(a.epochs),'--seed',str(a.seed),'--lr',str(a.lr),'--hidden-size',str(a.hidden_size),'--heads',str(a.heads),'--topk',str(a.topk),'--depth',str(a.depth),'--steps',str(st),'--state-modes',sm,'--regimes',rg,'--output-dir',str(a.output_dir),'--dataset-output',str(a.dataset_output),'--progress-every',str(a.progress_every),'--parallelism','1']; env=os.environ.copy();env['OMP_NUM_THREADS']='2';env['MKL_NUM_THREADS']='2'; print(f'LAUNCH state={sm} regime={rg} steps={st} active={len(active)+1}/{a.parallelism}',flush=True); active.append((subprocess.Popen(cmd,env=env),sm,rg,st))
        nxt=[]
        for proc,sm,rg,st in active:
            code=proc.poll()
            if code is None:nxt.append((proc,sm,rg,st))
            elif code!=0: raise RuntimeError(f'Worker failed: state={sm} regime={rg} steps={st} exit_code={code}')
            else: done.append((sm,rg,st));print(f'COMPLETE state={sm} regime={rg} steps={st} completed={len(done)}/{len(jobs)}',flush=True)
        active=nxt
        if active:time.sleep(.5)
    results=[]
    for sm,rg,st in jobs:
        tag=f'v229_{sm}_{rg}_d{a.depth}_steps{st}';fp=a.output_dir/f'{tag}.json';assert fp.exists(),fp;results.append(json.loads(fp.read_text(encoding='utf-8')))
    summary=a.output_dir/'v229_summary.json';summary.write_text(json.dumps(results,indent=2),encoding='utf-8');print('\n=== V229 SUMMARY ===',flush=True)
    for r in results:print(f"{r['state_mode']:15s} {r['regime']:10s} steps={r['steps']} teacher_action={r['best']['action']:.4f} auto_exact={r['autonomous']['autonomous_exact_trajectory']:.4f}",flush=True)
    print('summary_saved:',summary,flush=True)

if __name__=='__main__': main()
