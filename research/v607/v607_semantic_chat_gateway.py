from __future__ import annotations

import argparse
import heapq
import json
import random
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

TARGET_RELATIONS = (
    'schema:location', 'schema:birthPlace', 'schema:nationality',
    'schema:knowsLanguage', 'birthPlace', 'is_a', 'yago:hasMother'
)

@dataclass(frozen=True)
class Edge:
    subject: str
    relation: str
    object: str

@dataclass(frozen=True)
class Parse:
    text: str
    tokens: list[dict]
    entities: list[dict]
    noun_chunks: list[str]
    root: str
    root_lemma: str
    question: str
    subjects: list[str]
    objects: list[str]

@dataclass(frozen=True)
class Hypothesis:
    subject: str | None
    relation: str
    intent: str
    lexical_score: float
    evidence: dict

class Graph:
    def __init__(self, db: Path, cache_entries: int = 12000):
        self.db = db
        self.cache_entries = cache_entries
        self.conn = sqlite3.connect(str(db), timeout=120.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA query_only=ON')
        self.conn.execute('PRAGMA busy_timeout=120000')
        cols = {str(r['name']) for r in self.conn.execute('PRAGMA table_info(edges)')}
        def pick(opts):
            for x in opts:
                if x in cols: return x
            raise RuntimeError(f'edges schema missing one of {opts}; found {sorted(cols)}')
        self.sc = pick(('subject','source','start'))
        self.rc = pick(('relation','predicate','rel'))
        self.oc = pick(('object','target','end'))
        self.cache = {}
        self.order = []
    def close(self): self.conn.close()
    def _put(self,k,v):
        if self.cache_entries <= 0: return
        if k not in self.cache: self.order.append(k)
        self.cache[k]=v
        while len(self.order)>self.cache_entries:
            self.cache.pop(self.order.pop(0),None)
    def outgoing(self, subject: str, limit: int = 60) -> tuple[Edge,...]:
        k=('o',subject,limit)
        if k in self.cache: return self.cache[k]
        rows=self.conn.execute(f'''SELECT {self.sc} subject,{self.rc} relation,{self.oc} object FROM edges WHERE {self.sc}=? LIMIT ?''',(subject,limit)).fetchall()
        v=tuple(Edge(str(r['subject']),str(r['relation']),str(r['object'])) for r in rows)
        self._put(k,v); return v
    def incoming(self, object_: str, relation: str, limit: int=60) -> tuple[Edge,...]:
        k=('i',object_,relation,limit)
        if k in self.cache: return self.cache[k]
        rows=self.conn.execute(f'''SELECT {self.sc} subject,{self.rc} relation,{self.oc} object FROM edges WHERE {self.oc}=? AND {self.rc}=? LIMIT ?''',(object_,relation,limit)).fetchall()
        v=tuple(Edge(str(r['subject']),str(r['relation']),str(r['object'])) for r in rows)
        self._put(k,v); return v
    def has_edge(self,s,r,o):
        return self.conn.execute(f'''SELECT 1 FROM edges WHERE {self.sc}=? AND {self.rc}=? AND {self.oc}=? LIMIT 1''',(s,r,o)).fetchone() is not None
    def relation_vocab(self, sample_rows=6000, limit=300):
        rows=self.conn.execute(f'''SELECT {self.rc} relation FROM edges WHERE {self.rc} IS NOT NULL LIMIT ?''',(sample_rows,)).fetchall()
        c=Counter(str(r['relation']) for r in rows)
        out=[r for r,_ in c.most_common(limit)]
        for r in TARGET_RELATIONS:
            if r not in out: out.append(r)
        return out

class Attention:
    def __init__(self, decay=.65):
        self.decay=decay
        self.counts=defaultdict(Counter)
        self.goal_counts=defaultdict(Counter)
        self.global_counts=defaultdict(Counter)
        self.updates=0
    def update(self,goal,prefix,next_rel,strength=1.0):
        w=strength*(self.decay**max(0,len(prefix)-1)); d=len(prefix)
        self.counts[(goal,tuple(prefix),d)][next_rel]+=w
        self.goal_counts[(goal,d)][next_rel]+=w
        self.global_counts[d][next_rel]+=w
        self.updates+=1
    def _p(self,c,r):
        if not c:return 0.0
        return (c.get(r,0.0)+.5)/(sum(c.values())+.5*max(1,len(c)))
    def score(self,goal,prefix,r):
        a=self._p(self.counts.get((goal,tuple(prefix),len(prefix)),Counter()),r)
        b=self._p(self.goal_counts.get((goal,len(prefix)),Counter()),r)
        c=self._p(self.global_counts.get(len(prefix),Counter()),r)
        return .6*a+.3*b+.1*c if a else (.7*b+.3*c if b else c)
    def rank(self,goal,prefix,relations): return sorted(((self.score(goal,prefix,r),r) for r in set(relations)),key=lambda x:(-x[0],x[1]))
    def export(self):
        return {'decay':self.decay,'updates':self.updates,'exact_states':len(self.counts),'goal_states':len(self.goal_counts),
                'exact':[{'goal':g,'prefix':list(p),'depth':d,'next':dict(c)} for (g,p,d),c in self.counts.items()],
                'goal':[{'goal':g,'depth':d,'next':dict(c)} for (g,d),c in self.goal_counts.items()]}

class Context:
    def __init__(self): self.active_subject=None; self.turns=[]
    def subject(self,parse: Parse):
        if parse.entities:
            self.active_subject=parse.entities[0]['text']; return self.active_subject
        pron={t['text'].lower() for t in parse.tokens}
        if pron & {'he','she','they','him','her','them','his','their'}: return self.active_subject
        if parse.subjects:
            self.active_subject=parse.subjects[0]; return self.active_subject
        return self.active_subject
    def remember(self,text,goal):
        self.turns.append({'text':text,'goal':asdict(goal) if goal else None})
        self.turns=self.turns[-32:]

class SpaCyParser:
    def __init__(self,model):
        try: import spacy
        except ImportError as e: raise RuntimeError('spaCy is required: python -m pip install spacy') from e
        try: self.nlp=spacy.load(model)
        except OSError as e: raise RuntimeError(f"spaCy model '{model}' missing; install with: python -m spacy download {model}") from e
    def parse(self,text):
        d=self.nlp(text)
        toks=[{'text':t.text,'lemma':t.lemma_,'pos':t.pos_,'tag':t.tag_,'dep':t.dep_,'head':t.head.text} for t in d]
        ents=[{'text':e.text,'label':e.label_} for e in d.ents]
        chunks=[c.text for c in d.noun_chunks]
        roots=[t for t in d if t.dep_=='ROOT']; root=roots[0] if roots else (d[0] if d else None)
        q='DECLARATIVE'
        for t in d:
            if t.text.lower() in {'where','what','who','when','which','how'}:
                q='WH_'+t.text.upper(); break
        return Parse(text,toks,ents,chunks,root.text if root else '',root.lemma_ if root else '',q,[t.text for t in d if 'subj' in t.dep_],[t.text for t in d if t.dep_ in {'obj','dobj','pobj','attr','oprd'}])

def norm(x): return x.lower().replace(':',' ').replace('_',' ')

def hypotheses(
    parse: Parse,
    ctx: Context,
    vocab,
    max_n: int = 12,
):
    """
    Generate structural semantic hypotheses.

    Defensive boundary:
      - `vocab` MUST be an iterable of relation strings.
      - argparse.Namespace / config objects are rejected with a useful error.
    """
    if isinstance(vocab, argparse.Namespace):
        raise TypeError(
            "hypotheses() expected relation vocabulary as `vocab`, "
            "but received argparse.Namespace. "
            "Pass the relation list, not the CLI args object."
        )

    if vocab is None:
        raise TypeError(
            "hypotheses() received vocab=None; "
            "a relation vocabulary is required."
        )

    try:
        vocab = tuple(vocab)
    except TypeError as exc:
        raise TypeError(
            "hypotheses() expected an iterable relation vocabulary."
        ) from exc

    for relation in vocab:
        if not isinstance(relation, str):
            raise TypeError(
                "hypotheses() relation vocabulary must contain strings; "
                f"got {type(relation).__name__}"
            )

    subject = ctx.subject(parse)

    lex = {
        t["text"].lower()
        for t in parse.tokens
    } | {
        t["lemma"].lower()
        for t in parse.tokens
    }

    out = []

    for relation in vocab:
        n = norm(relation)
        score = 0.0

        for word in lex:
            if word and word in n:
                score += 0.15

        # Structural only. Question form is evidence, not a hard semantic map.
        evidence = {
            "root": parse.root_lemma,
            "question": parse.question,
            "lexical_overlap": score,
        }

        out.append(
            Hypothesis(
                subject=subject,
                relation=relation,
                intent="relation_lookup",
                lexical_score=score,
                evidence=evidence,
            )
        )

    out.sort(
        key=lambda h: (
            -h.lexical_score,
            h.relation,
        )
    )

    return out[:max_n]



def search(graph,att,h,args,seed=0):
    if not h.subject:return {'success':False,'steps':0,'path':[],'target':None,'attention':0,'exploration':0}
    rng=random.Random(seed); q=[]; tie=0
    heapq.heappush(q,(0.0,0,tie,h.subject,()))
    seen={(h.subject,())}; steps=0; att_n=0; exp_n=0
    while q and steps<args.goal_budget:
        score,depth,tie,node,prefix=heapq.heappop(q)
        if depth>=args.max_depth: continue
        steps+=1
        edges=list(graph.outgoing(node,args.per_node))
        ranked=att.rank(h.relation,prefix,[e.relation for e in edges if e.relation!=h.relation]); rank={r:s for s,r in ranked}
        edges.sort(key=lambda e:(-rank.get(e.relation,0.0),rng.random()))
        for e in edges:
            if e.relation==h.relation: continue
            np=prefix+(e.relation,); state=(e.object,np)
            if state in seen: continue
            seen.add(state)
            if rank.get(e.relation,0)>0: att_n+=1
            else: exp_n+=1
            # Candidate endpoint: direct graph validation of the proposed relation.
            if len(np)>=1 and graph.has_edge(h.subject,h.relation,e.object):
                return {'success':True,'steps':steps,'path':list(np),'target':e.object,'attention':att_n,'exploration':exp_n}
            if len(np)<args.max_depth:
                tie+=1; heapq.heappush(q,(-rank.get(e.relation,0.0),depth+1,tie,e.object,np))
    return {'success':False,'steps':steps,'path':[],'target':None,'attention':att_n,'exploration':exp_n}

def trace_result(text,parse,hs,selected,search_result,ctx):
    return {'timestamp':time.time(),'text':text,'parse':asdict(parse),'hypotheses':[asdict(h) for h in hs],
            'selected_goal':asdict(selected) if selected else None,'search':search_result,
            'context':{'active_subject':ctx.active_subject,'turns':len(ctx.turns)}}

def smoke(graph, parser, att, vocab, args):
    ctx = Context()
    inputs = [
        'Where was Alice born?',
        'What is her nationality?',
    ]
    traces = []

    print('=== V607 SMOKE ===', flush=True)

    for text in inputs:
        p = parser.parse(text)

        # Explicitly pass vocabulary. Never pass argparse.Namespace here.
        hs = hypotheses(
            parse=p,
            ctx=ctx,
            vocab=vocab,
            max_n=args.max_hypotheses,
        )

        ranked = []

        for h in hs:
            result = search(
                graph,
                att,
                h,
                args,
                args.seed,
            )
            ranked.append(
                (
                    (1.0 if result['success'] else 0.0)
                    + h.lexical_score,
                    h,
                    result,
                )
            )

        ranked.sort(
            key=lambda x: (
                -x[0],
                x[1].relation,
            )
        )

        selected = (
            ranked[0][1]
            if ranked
            else None
        )
        search_result = (
            ranked[0][2]
            if ranked
            else None
        )

        ctx.remember(
            text,
            selected,
        )

        trace = trace_result(
            text,
            p,
            hs,
            selected,
            search_result or {},
            ctx,
        )

        traces.append(trace)

        print(
            f"[SMOKE] {text}",
            flush=True,
        )
        print(
            f"  hypotheses={len(hs)} "
            f"selected={selected.relation if selected else None} "
            f"evidence={search_result}",
            flush=True,
        )

    return traces

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--database',required=True); ap.add_argument('--output',required=True); ap.add_argument('--trace-output',default=''); ap.add_argument('--prior-output',default='')
    ap.add_argument('--spacy-model',default='en_core_web_sm'); ap.add_argument('--mode',choices=('chat','smoke','benchmark'),default='chat')
    ap.add_argument('--max-hypotheses',type=int,default=12); ap.add_argument('--relation-vocabulary',type=int,default=200); ap.add_argument('--goal-budget',type=int,default=40)
    ap.add_argument('--budget',type=int,default=80); ap.add_argument('--per-node',type=int,default=60); ap.add_argument('--max-depth',type=int,default=3); ap.add_argument('--cache-entries',type=int,default=12000); ap.add_argument('--prior-decay',type=float,default=.65); ap.add_argument('--seed',type=int,default=60600); ap.add_argument('--progress-every',type=int,default=1)
    a=ap.parse_args(); db=Path(a.database).resolve(); out=Path(a.output).resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    tr_path=Path(a.trace_output or str(out.with_suffix(''))+'_traces.jsonl').resolve(); pr_path=Path(a.prior_output or str(out.with_suffix(''))+'_prior.json').resolve()
    print('=== V607 SEMANTIC CHAT GATEWAY ===',flush=True); print(f'database              : {db}',flush=True); print('previous artifacts    : NONE',flush=True); print(f'spaCy model           : {a.spacy_model}',flush=True)
    g=Graph(db,a.cache_entries); print('[GATEWAY 1/3] loading spaCy...',flush=True); sp=SpaCyParser(a.spacy_model); print('[GATEWAY 2/3] relation vocabulary...',flush=True); vocab=g.relation_vocab(limit=a.relation_vocabulary)
    if not isinstance(vocab, (list, tuple)):
        raise RuntimeError(
            "Graph.relation_vocab() must return a list/tuple of relation strings."
        )
    if not all(isinstance(rel, str) for rel in vocab):
        raise RuntimeError(
            "Graph.relation_vocab() returned a non-string relation."
        )
    print(f'    relations={len(vocab)}',flush=True); print('[GATEWAY 3/3] global conditional attention...',flush=True); att=Attention(a.prior_decay)
    # Small graph-only bootstrap; no previous file.
    for gi,goal in enumerate(TARGET_RELATIONS):
        for subj in g.sample_subjects(goal,40,a.seed+gi) if hasattr(g,'sample_subjects') else []: pass
    if a.mode=='chat':
        ctx=Context(); traces=[]; print("=== V607 CHAT (type 'exit') ===",flush=True)
        try:
            while True:
                s=input('chat> ').strip()
                if s.lower() in {'exit','quit'}: break
                if not s: continue
                p=sp.parse(s); hs=hypotheses(
                    parse=p,
                    ctx=ctx,
                    vocab=vocab,
                    max_n=a.max_hypotheses,
                ); ranked=[]
                for i,h in enumerate(hs):
                    sr=search(g,att,h,a,a.seed+i); ranked.append((sr['success']+h.lexical_score,h,sr))
                ranked.sort(key=lambda x:(-x[0],x[1].relation)); sel=ranked[0][1] if ranked else None; sr=ranked[0][2] if ranked else None
                if sel and sr and sr['success']:
                    att.update(sel.relation,(),sel.relation,.25)
                ctx.remember(s,sel); tr=trace_result(s,p,hs,sel,sr or {},ctx); traces.append(tr)
                print('goal:',asdict(sel) if sel else None,flush=True); print('evidence:',sr,flush=True); print('alternatives:',len(hs),flush=True)
        finally: pass
    else:
        traces=smoke(g,sp,att,vocab,a) if a.mode=='smoke' else []
    g.close()
    if traces:
        with tr_path.open('w',encoding='utf-8') as f:
            for t in traces:f.write(json.dumps(t,ensure_ascii=False)+'\n')
    pr_path.write_text(json.dumps(att.export(),indent=2,ensure_ascii=False),encoding='utf-8')
    payload={'version':'V607','architecture':{'parser':'spaCy structural parse','hypotheses':'generic alternatives','resolver':'global conditional attention + bounded BFS','context':'lightweight explicit context'},'config':vars(a),'relation_vocabulary':vocab,'traces':len(traces),'prior':att.export(),'trace_output':str(tr_path),'prior_output':str(pr_path)}
    out.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    print(); print('=== V607 COMPLETE ===',flush=True); print(f'mode                  : {a.mode}',flush=True); print(f'traces                : {len(traces)}',flush=True); print(f'prior updates         : {att.updates}',flush=True); print(f'prior exact states    : {len(att.counts)}',flush=True); print(f'prior goal states     : {len(att.goal_counts)}',flush=True); print(f'JSON                  : {out}',flush=True); print(f'TRACE                 : {tr_path}',flush=True); print(f'PRIOR                 : {pr_path}',flush=True)

if __name__=='__main__': main()
