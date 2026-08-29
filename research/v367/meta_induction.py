from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import math
import itertools

@dataclass(frozen=True)
class Term:
    op:str
    args:Tuple['Term',...]=()
    name:str|None=None
    value:int|None=None
    def text(self):
        if self.op=='var': return self.name
        if self.op=='const': return str(self.value)
        if self.op=='not': return f'NOT({self.args[0].text()})'
        if self.op=='xor': return '('+' XOR '.join(x.text() for x in self.args)+')'
        if self.op=='add': return '('+' + '.join(x.text() for x in self.args)+')'
        if self.op=='mul': return '('+' * '.join(x.text() for x in self.args)+')'
        if self.op=='mul': return '('+' * '.join(x.text() for x in self.args)+')'
        return self.op
    def depth(self): return 0 if not self.args else 1+max(x.depth() for x in self.args)
    def leaves(self):
        if self.op=='var': return (self.name,)
        out=[]
        for x in self.args: out.extend(x.leaves())
        return tuple(out)
    def eval(self,v):
        if self.op=='var': return int(v[self.name])
        if self.op=='const': return int(self.value)
        if self.op=='not': return 1-self.args[0].eval(v)
        if self.op=='xor':
            z=0
            for x in self.args: z ^= x.eval(v)
            return z
        if self.op=='add': return sum(x.eval(v) for x in self.args)
        if self.op=='mul':
            z=1
            for x in self.args: z *= x.eval(v)
            return z
        if self.op=='mul':
            z=1
            for x in self.args: z *= x.eval(v)
            return z
        raise ValueError(self.op)

@dataclass(frozen=True)
class Schema:
    name:str
    arity:int
    in_types:Tuple[str,...]
    out_type:str
    evidence:int=0
    learned:bool=True

@dataclass(frozen=True)
class Candidate:
    term:Term
    prior:float
    support:int=0
    contradiction:int=0
    parent:str|None=None
    def score(self):
        like=(self.support+1)/(self.support+self.contradiction+2)
        return math.log(max(self.prior,1e-9))+math.log(max(like,1e-9))-0.15*(1+self.term.depth())-0.03*len(self.term.text())

class MetaInducer:
    def __init__(self,beam=64):
        self.beam=beam
        self.schemas:Dict[str,Schema]={}
        self.evidence=[]
        self.learned_domains=set(['bit'])
    def infer_schema(self,values,observed):
        out_type='integer' if any((not isinstance(x,int)) or x not in (0,1) for x in observed) else 'bit'
        in_types=tuple('bit' if int(values[k]) in (0,1) else 'integer' for k in ('memory','cue1','cue2','cue3') if k in values)
        arity=min(2,len(in_types))
        name=f'INDUCED_{out_type}_{arity}'
        old=self.schemas.get(name)
        self.schemas[name]=Schema(name,arity,in_types[:arity],out_type,(old.evidence+1 if old else 1),True)
        self.learned_domains.add(out_type)
        self.evidence.append(("schema",name,out_type,arity))
        return self.schemas[name]
    def primitive_terms(self,values):
        return [Term('var',name=k) for k in ('memory','cue1','cue2','cue3','actual') if k in values]
    def expand(self,values,extended=False):
        leaves=self.primitive_terms(values)
        pool=list(leaves)
        for x in leaves: pool.append(Term('not',args=(x,)))
        for i,a in enumerate(leaves):
            for b in leaves[i+1:]: pool.append(Term('xor',args=(a,b)))
        if extended:
            ints=list(leaves)
            for i,a in enumerate(ints):
                for b in ints[i+1:]:
                    pool.append(Term('add',args=(a,b)))
                    pool.append(Term('mul',args=(a,b)))
            for a in ints:
                pool.append(Term('mul',args=(a,Term('const',value=2))))
            for a,b in itertools.combinations(ints,2):
                pool.append(Term('add',args=(Term('mul',args=(a,Term('const',value=2))),b)))
                pool.append(Term('add',args=(a,Term('mul',args=(b,Term('const',value=2))))))
        uniq={x.text():x for x in pool}
        # structural prior: sparse/shallow, but let exact short forms win
        scored=sorted(uniq.values(),key=lambda x:(1+x.depth(),len(set(x.leaves())),len(x.text())))
        # Keep the most expressive induced forms available after class failure.
        if extended:
            return tuple(scored[:max(self.beam,64)])
        return tuple(scored[:self.beam])

    def expand_terms(self, values, extended=False):
        leaves=self.primitive_terms(values)
        pool=list(leaves)
        for x in leaves:
            pool.append(Term('not',args=(x,)))
        for i,a in enumerate(leaves):
            for b in leaves[i+1:]:
                pool.append(Term('xor',args=(a,b)))
        if extended:
            for i,a in enumerate(leaves):
                for b in leaves[i+1:]:
                    pool.append(Term('add',args=(a,b)))
                    pool.append(Term('mul',args=(a,b)))
            for a in leaves:
                pool.append(Term('mul',args=(a,Term('const',value=2))))
                pool.append(Term('add',args=(Term('mul',args=(a,Term('const',value=2))),Term('const',value=1))))
            for a,b in itertools.combinations(leaves,2):
                pool.append(Term('add',args=(Term('mul',args=(a,Term('const',value=2))),b)))
        uniq={t.text():t for t in pool}
        return tuple(uniq.values())
    def observe(self,task,regime,values,candidates,intervene_var,intervene_val,observed):
        vv=dict(values); vv[intervene_var]=intervene_val
        entries=[]; survivors=[]
        for c in candidates:
            try: pred=c.term.eval(vv)
            except Exception: continue
            ok=(pred==observed)
            entries.append((c,pred,ok))
            if ok: survivors.append(c)
            self.evidence.append((task,regime,c.term.text(),pred,observed,ok))
        return entries,survivors
    def best_intervention(self,candidates,values):
        best=None
        for var in ('memory','cue1','cue2','cue3'):
            if var not in values: continue
            for val in (0,1):
                vv=dict(values); vv[var]=val
                preds=[]
                for c in candidates:
                    try: preds.append(c.term.eval(vv))
                    except Exception: pass
                if not preds: continue
                groups={p:preds.count(p) for p in set(preds)}
                n=len(preds); score=1-sum((g/n)**2 for g in groups.values())
                if best is None or score>best[0]: best=(score,var,val)
        return None if best is None or best[0] <= 0 else {'score':best[0],'variable':best[1],'value':best[2]}
    def stats(self):
        return {'learned_schemas':sum(int(s.learned) for s in self.schemas.values()),'schema_instances':len(self.schemas),'evidence':len(self.evidence)}
