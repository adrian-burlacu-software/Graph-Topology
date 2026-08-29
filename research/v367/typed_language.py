
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import itertools, math

ValueType=str

@dataclass(frozen=True)
class Term:
    kind:str
    name:str|None=None
    args:Tuple["Term",...]=()
    const:int|None=None

    def text(self):
        if self.kind=="var": return self.name
        if self.kind=="const": return str(self.const)
        if self.kind=="not": return f"NOT({self.args[0].text()})"
        if self.kind=="xor": return "("+" XOR ".join(a.text() for a in self.args)+")"
        if self.kind=="and": return "("+" AND ".join(a.text() for a in self.args)+")"
        if self.kind=="add": return "("+" + ".join(a.text() for a in self.args)+")"
        if self.kind=="eq": return f"EQ({self.args[0].text()},{self.args[1].text()})"
        raise ValueError(self.kind)

    def depth(self):
        return 0 if not self.args else 1+max(a.depth() for a in self.args)

    def leaves(self):
        if self.kind=="var": return (self.name,)
        out=[]
        for a in self.args: out.extend(a.leaves())
        return tuple(out)

class TypeSystem:
    def infer(self,name,value):
        if isinstance(value,bool): return "bit"
        if isinstance(value,int): return "bit" if value in (0,1) else "integer"
        if isinstance(value,tuple): return "tuple"
        return "unknown"
    def environment_types(self,values):
        return {k:self.infer(k,v) for k,v in values.items()}

@dataclass(frozen=True)
class OperatorSchema:
    name:str
    arity:int
    input_types:Tuple[str,...]
    output_type:str
    semantics:str
    learned:bool=False
    evidence:int=0
    contradictions:int=0

class OperatorLibrary:
    def __init__(self):
        self.schemas={}
        self.history=[]
        for s in (
            OperatorSchema("NOT",1,("bit",),"bit","bit_negation"),
            OperatorSchema("XOR",2,("bit","bit"),"bit","bit_xor"),
            OperatorSchema("AND",2,("bit","bit"),"bit","bit_and"),
            OperatorSchema("ADD",2,("integer","integer"),"integer","integer_add"),
            OperatorSchema("EQ",2,("unknown","unknown"),"bit","equality_test"),
        ):
            self.register(s)
    def register(self,schema):
        self.schemas[schema.name]=schema
        self.history.append(("register",schema.name,schema.learned))
    def infer_new(self,arity,input_types,output_type,evidence_hint):
        key=f"INDUCED_{arity}_{'_'.join(input_types)}_{output_type}_{evidence_hint}"
        if key not in self.schemas:
            self.register(OperatorSchema(
                key,arity,tuple(input_types),output_type,
                f"induced:{evidence_hint}",learned=True
            ))
        return self.schemas[key]
    def stats(self):
        return {
            "schemas":len(self.schemas),
            "learned_schemas":sum(int(s.learned) for s in self.schemas.values()),
            "events":len(self.history),
        }

@dataclass(frozen=True)
class TypedCandidate:
    cid:str
    regime:int
    term:Term
    value_type:str
    prior:float
    support:int=0
    contradictions:int=0
    active:bool=True
    parent:str|None=None
    @property
    def complexity(self):
        return 1+0.3*self.term.depth()+0.08*len(self.term.text())+0.15*len(set(self.term.leaves()))
    @property
    def score(self):
        likelihood=(self.support+1)/(self.support+self.contradictions+2)
        return math.log(max(1e-12,self.prior))+math.log(max(1e-12,likelihood))-0.28*self.complexity+0.15*self.support

class TypedProgramGenerator:
    def __init__(self,max_depth=3):
        self.max_depth=max_depth
    def _type_of(self,t,types):
        if t.kind=="var": return types.get(t.name,"unknown")
        if t.kind in ("not","xor","and","eq"): return "bit"
        if t.kind in ("add","const"): return "integer"
        return "unknown"
    def leaves(self,values,types):
        return tuple(
            Term(kind="var",name=k)
            for k in values
            if types.get(k) in ("bit","integer")
        )
    def apply_known(self,terms,types):
        out=list(terms)
        bits=[t for t in terms if self._type_of(t,types)=="bit"]
        ints=[t for t in terms if self._type_of(t,types) in ("integer","bit")]
        for a in bits: out.append(Term(kind="not",args=(a,)))
        for a,b in itertools.combinations(bits,2):
            out.extend((
                Term(kind="xor",args=(a,b)),
                Term(kind="and",args=(a,b)),
                Term(kind="eq",args=(a,b)),
            ))
        for a,b in itertools.combinations(ints,2):
            out.append(Term(kind="add",args=(a,b)))
        unique={t.text():t for t in out}
        return tuple(unique.values())
    def expand(self,values,types,operator_library=None):
        current=list(self.leaves(values,types))
        all_terms=list(current)
        for _ in range(max(1,self.max_depth)):
            current=list(self.apply_known(current,types))
            all_terms.extend(current)
            if len(all_terms)>1200:
                break
        unique={t.text():t for t in all_terms}
        return tuple(unique.values())
    def execute(self,t,values):
        if t.kind=="var": return values[t.name]
        if t.kind=="const": return t.const
        if t.kind=="not": return 1-self.execute(t.args[0],values)
        if t.kind=="xor": return self.execute(t.args[0],values)^self.execute(t.args[1],values)
        if t.kind=="and": return self.execute(t.args[0],values)&self.execute(t.args[1],values)
        if t.kind=="add": return self.execute(t.args[0],values)+self.execute(t.args[1],values)
        if t.kind=="eq": return int(self.execute(t.args[0],values)==self.execute(t.args[1],values))
        raise ValueError(t.kind)
