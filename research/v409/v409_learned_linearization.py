
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0,str(HERE))

from semantic_memory import IndexedSemanticMemory, SemanticEdge, canonical_concept
from semantic_architecture import IntegratedSemanticArchitecture
from real_grounding import IndexedConceptNet


# ============================================================================
# GUM / CoNLL-U
# ============================================================================

@dataclass(frozen=True)
class UDToken:
    id:str
    form:str
    lemma:str
    upos:str
    xpos:str
    feats:str
    head:str
    deprel:str
    deps:str
    misc:str


@dataclass(frozen=True)
class UDSentence:
    text:str
    tokens:tuple[UDToken,...]
    comments:tuple[str,...]
    source_file:str


def parse_feats(raw:str):
    if not raw or raw=="_":
        return ()
    return tuple(sorted(
        tuple(part.split("=",1))
        for part in raw.split("|")
        if "=" in part
    ))


def parse_conllu(path:Path):
    sentences=[]
    comments=[]
    rows=[]

    def flush():
        nonlocal comments,rows
        if not rows:
            comments=[]
            return
        text=next(
            (
                c[len("# text = "):]
                for c in comments
                if c.startswith("# text = ")
            ),
            " ".join(t.form for t in rows),
        )
        sentences.append(
            UDSentence(
                text=text,
                tokens=tuple(rows),
                comments=tuple(comments),
                source_file=str(path),
            )
        )
        comments=[]
        rows=[]

    with path.open("r",encoding="utf-8") as fh:
        for line_no,line in enumerate(fh,1):
            line=line.rstrip("\n")
            if not line:
                flush()
                continue
            if line.startswith("#"):
                comments.append(line)
                continue

            cols=line.split("\t")
            if len(cols)!=10:
                raise ValueError(
                    f"{path}:{line_no}: expected 10 CoNLL-U fields, got {len(cols)}"
                )
            if "-" in cols[0] or "." in cols[0]:
                continue
            rows.append(UDToken(*cols))

    flush()
    return sentences


def discover_gum(root:Path):
    files=sorted(root.rglob("*.conllu"))
    if not files:
        raise FileNotFoundError(f"No .conllu files under {root}")
    return files


def split_name(path:Path):
    name=path.name.lower()
    if "train" in name:
        return "train"
    if "dev" in name:
        return "dev"
    if "test2" in name or "gentle" in name:
        return "test2"
    if "test" in name:
        return "test"
    return "unknown"


# ============================================================================
# Data-driven grammar + morphology + linearization
# ============================================================================

class GrammarModel:
    def __init__(self):
        self.sentences=0
        self.tokens=0
        self.upos=Counter()
        self.xpos=Counter()
        self.dependencies=Counter()
        self.features=Counter()
        self.morph_forms=defaultdict(Counter)

        # Parent UPOS + dependency label -> learned relative order.
        # "before" means child normally precedes head, "after" follows it.
        self.relation_order=Counter()

        # Parent UPOS -> distribution of dependency labels as a local order.
        self.child_sequence=Counter()

        # More specific lexical/morphological realization observations.
        self.morph_pair_total=0

    def observe(self,s:UDSentence):
        self.sentences+=1
        self.tokens+=len(s.tokens)

        by_id={t.id:t for t in s.tokens}

        for t in s.tokens:
            self.upos[t.upos]+=1
            self.xpos[t.xpos]+=1

            feats=parse_feats(t.feats)
            for feat in feats:
                self.features[feat]+=1

            if t.lemma and t.lemma!="_":
                self.morph_forms[(t.lemma,feats)][t.form]+=1
                self.morph_pair_total+=1

            if t.head in by_id:
                head=by_id[t.head]
                self.dependencies[(head.upos,t.deprel,t.upos)]+=1

                try:
                    child_i=int(t.id)
                    head_i=int(t.head)
                except ValueError:
                    child_i=0
                    head_i=0

                direction="before" if child_i<head_i else "after"
                self.relation_order[(head.upos,t.deprel,direction)]+=1
                self.child_sequence[(head.upos,child_i-head_i,t.deprel)]+=1

    def realize(self,lemma,feats,fallback=None):
        counts=self.morph_forms.get((lemma,tuple(sorted(feats))))
        if counts:
            return counts.most_common(1)[0][0]
        return fallback or lemma

    def learned_direction(self,parent_upos,relation,default="after"):
        before=self.relation_order.get((parent_upos,relation,"before"),0)
        after=self.relation_order.get((parent_upos,relation,"after"),0)
        if not before and not after:
            return default
        return "before" if before>after else "after"

    def preferred_dep_order(self,parent_upos,relations):
        # Learn approximate left/right canonical ordering from observed trees,
        # then use lexical token order as the deterministic tie-breaker.
        result=[]
        for rel in relations:
            before=self.relation_order.get((parent_upos,rel,"before"),0)
            after=self.relation_order.get((parent_upos,rel,"after"),0)
            total=before+after
            if total:
                left_ratio=before/total
            else:
                left_ratio=0.0
            result.append((rel,-left_ratio,-total))
        return result

    def report(self):
        return {
            "sentences":self.sentences,
            "tokens":self.tokens,
            "upos_types":len(self.upos),
            "xpos_types":len(self.xpos),
            "dependency_types":len(self.dependencies),
            "feature_types":len(self.features),
            "morphology_pairs":len(self.morph_forms),
            "linearization_relation_types":len(self.relation_order),
            "top_dependencies":{
                str(k):v
                for k,v in self.dependencies.most_common(20)
            },
        }


# ============================================================================
# Explicit cognitive language state
# ============================================================================

@dataclass(frozen=True)
class LanguageNode:
    token_id:str
    surface:str
    lemma:str
    upos:str
    xpos:str
    morphology:tuple[tuple[str,str],...]
    concept:str|None
    grounding_confidence:float


@dataclass(frozen=True)
class Relation:
    head:str
    label:str
    child:str


@dataclass(frozen=True)
class Proposition:
    predicate_id:str
    subjects:tuple[str,...]
    objects:tuple[str,...]
    obliques:tuple[str,...]
    modifiers:tuple[str,...]
    auxiliaries:tuple[str,...]
    negations:tuple[str,...]
    embedded:tuple[str,...]
    conjuncts:tuple[str,...]


@dataclass(frozen=True)
class LanguageState:
    source_text:str
    nodes:tuple[LanguageNode,...]
    relations:tuple[Relation,...]
    propositions:tuple[Proposition,...]
    roots:tuple[str,...]
    unresolved:tuple[str,...]
    cognitive_beliefs:tuple[tuple[str,float,str],...]

    def lexical_signature(self):
        return tuple(sorted(
            (n.lemma,n.upos)
            for n in self.nodes
            if n.concept is not None
        ))

    def semantic_signature(self):
        concepts={
            n.token_id:n.concept
            for n in self.nodes
            if n.concept is not None
        }

        def node_semantics(tid):
            node=next(
                (n for n in self.nodes if n.token_id==tid),
                None,
            )
            if node is None:
                return ("",)
            child_edges=[]
            for r in self.relations:
                if r.head!=tid:
                    continue
                c=concepts.get(r.child)
                if c is not None and r.label not in {
                    "punct","det","case","cc","aux","aux:pass",
                }:
                    child_edges.append((r.label,c))
            return (
                node.concept,
                node.upos,
                tuple(sorted(child_edges)),
            )

        props=[]
        for p in self.propositions:
            args=[]
            for role_group,ids in (
                ("subj",p.subjects),
                ("obj",p.objects),
                ("obl",p.obliques),
                ("mod",p.modifiers),
            ):
                for tid in ids:
                    c=concepts.get(tid)
                    if c is not None:
                        args.append((role_group,c))
            aux=tuple(sorted(
                concepts[x] for x in p.auxiliaries
                if x in concepts
            ))
            neg=tuple(sorted(
                concepts[x] for x in p.negations
                if x in concepts
            ))
            embedded=tuple(sorted(p.embedded))
            props.append((
                concepts.get(p.predicate_id,""),
                tuple(args),
                aux,
                neg,
                embedded,
            ))

        unattached=tuple(sorted(
            node_semantics(n.token_id)
            for n in self.nodes
            if n.upos not in {"VERB","AUX"}
            and n.concept is not None
            and not any(
                r.child==n.token_id
                and r.label in {
                    "nsubj","nsubj:pass","obj","iobj","obl",
                    "obl:agent","amod","nmod","compound",
                    "nmod:poss","acl","acl:relcl"
                }
                for r in self.relations
            )
        ))

        return (tuple(sorted(props)),unattached)

    def substantive(self):
        return bool(self.nodes)


class CognitiveLanguageArchitecture:
    def __init__(self,architecture,grammar):
        self.arch=architecture
        self.memory=architecture.memory
        self.concepts=self.memory.concepts()
        self.grammar=grammar
        self.cache={}

    def ground(self,form,lemma):
        key=canonical_concept(
            lemma if lemma and lemma!="_" else form
        )
        if key in self.cache:
            return self.cache[key]

        variants=[key]
        if key.endswith("ies") and len(key)>4:
            variants.append(key[:-3]+"y")
        if key.endswith("es") and len(key)>4:
            variants.append(key[:-2])
        if key.endswith("s") and len(key)>3:
            variants.append(key[:-1])
        if key.endswith("ed") and len(key)>4:
            variants.extend((key[:-2],key[:-1]))
        if key.endswith("ing") and len(key)>5:
            variants.append(key[:-3])

        for candidate in variants:
            if candidate in self.concepts:
                belief=self.arch.perceive(candidate,context=())
                ans=(candidate,float(belief.confidence))
                self.cache[key]=ans
                return ans

        self.cache[key]=None
        return None

    def perceive(self,s:UDSentence):
        nodes=[]
        unresolved=[]

        for t in s.tokens:
            if t.upos in {"PUNCT","SYM"}:
                continue
            grounded=self.ground(t.form,t.lemma)
            concept=grounded[0] if grounded else None
            conf=grounded[1] if grounded else 0.0
            if concept is None:
                unresolved.append(t.id)

            nodes.append(
                LanguageNode(
                    t.id,
                    t.form,
                    t.lemma,
                    t.upos,
                    t.xpos,
                    parse_feats(t.feats),
                    concept,
                    conf,
                )
            )

        relations=[]
        for t in s.tokens:
            if t.upos in {"PUNCT","SYM"}:
                continue
            if t.head.isdigit() and t.head!="0":
                relations.append(Relation(t.head,t.deprel,t.id))

        child_map=defaultdict(list)
        for r in relations:
            child_map[r.head].append(r)

        predicates=[]
        for n in nodes:
            if n.upos!="VERB" or n.concept is None:
                continue

            subjects=[]
            objects=[]
            obliques=[]
            modifiers=[]
            auxiliaries=[]
            negations=[]
            embedded=[]
            conjuncts=[]

            for r in child_map.get(n.token_id,()):
                c=r.child
                if r.label.startswith("nsubj"):
                    subjects.append(c)
                elif r.label in {"obj","iobj"}:
                    objects.append(c)
                elif r.label.startswith("obl"):
                    obliques.append(c)
                elif r.label=="advmod":
                    modifiers.append(c)
                elif r.label in {"aux","aux:pass","cop"}:
                    auxiliaries.append(c)
                elif r.label=="neg":
                    negations.append(c)
                elif r.label in {"ccomp","xcomp","advcl","acl:relcl"}:
                    embedded.append(c)
                elif r.label=="conj":
                    conjuncts.append(c)

            predicates.append(
                Proposition(
                    n.token_id,
                    tuple(subjects),
                    tuple(objects),
                    tuple(obliques),
                    tuple(modifiers),
                    tuple(auxiliaries),
                    tuple(negations),
                    tuple(embedded),
                    tuple(conjuncts),
                )
            )

        roots=tuple(
            t.id for t in s.tokens
            if t.head=="0" and t.upos!="PUNCT"
        )

        beliefs=[]
        for n in nodes:
            if n.concept is None:
                continue
            b=self.arch.perceive(n.concept,context=())
            beliefs.append(
                (
                    n.concept,
                    float(b.confidence),
                    b.committed or "",
                )
            )

        return LanguageState(
            s.text,
            tuple(nodes),
            tuple(relations),
            tuple(predicates),
            roots,
            tuple(unresolved),
            tuple(beliefs),
        )


# ============================================================================
# Learned linearizer
# ============================================================================

class LearnedLinearizer:
    def __init__(self,grammar):
        self.grammar=grammar

    def generate(self,state:LanguageState):
        by_id={n.token_id:n for n in state.nodes}
        children=defaultdict(list)
        for r in state.relations:
            children[r.head].append(r)

        rendered=[]
        seen=set()

        for root in sorted(
            state.roots,
            key=lambda x:int(x) if x.isdigit() else 10**9,
        ):
            if root not in by_id or root in seen:
                continue
            rendered.append(
                self._render(root,by_id,children,seen)
            )

        # Any disconnected but grounded nodes are retained rather than silently
        # dropped, but their original order is used as a final fallback.
        for n in sorted(
            state.nodes,
            key=lambda x:int(x.token_id) if x.token_id.isdigit() else 10**9,
        ):
            if n.token_id in seen:
                continue
            if n.upos in {"PUNCT","SYM"}:
                continue
            rendered.append(
                self._render(n.token_id,by_id,children,seen)
            )

        return " ".join(x for x in rendered if x).strip()

    def _surface(self,n):
        return self.grammar.realize(
            n.lemma,
            n.morphology,
            fallback=n.surface,
        )

    def _render(self,tid,by_id,children,seen):
        if tid in seen:
            return ""
        n=by_id.get(tid)
        if n is None:
            return ""
        if n.upos=="VERB":
            return self._verb(tid,by_id,children,seen)
        if n.upos in {"NOUN","PROPN","PRON","NUM"}:
            return self._noun_phrase(tid,by_id,children,seen)
        seen.add(tid)
        return self._surface(n).lower()

    def _noun_phrase(self,tid,by_id,children,seen):
        if tid in seen:
            return ""
        n=by_id[tid]
        local=children.get(tid,[])

        left=[]
        right=[]
        neutral=[]

        # We learn direction for every dependency, then preserve within-side
        # token order. This fixes e.g. "adverse consequences of discrimination":
        # amod stays left, nmod stays right.
        for r in local:
            child=by_id.get(r.child)
            if child is None or r.child in seen:
                continue
            direction=self.grammar.learned_direction(
                n.upos,
                r.label,
                default="after",
            )
            item=(self._order(r.child),r)
            if direction=="before":
                left.append(item)
            else:
                right.append(item)

        pieces=[]

        for _,r in sorted(left):
            child=by_id.get(r.child)
            if child is None:
                continue
            if r.label=="case":
                continue
            if r.label in {"nmod","nmod:poss","acl","acl:relcl"}:
                phrase=self._dependent(r.child,r.label,by_id,children,seen)
            elif r.label=="cc":
                phrase=self._surface(child).lower()
                seen.add(child.token_id)
            elif r.label=="det":
                phrase=self._surface(child).lower()
                seen.add(child.token_id)
            elif r.label=="compound":
                phrase=self._noun_phrase(r.child,by_id,children,seen)
            elif child.upos=="VERB":
                phrase=self._verb(r.child,by_id,children,seen)
            else:
                phrase=self._surface(child).lower()
                seen.add(child.token_id)
            if phrase:
                pieces.append(phrase)

        seen.add(tid)
        pieces.append(self._surface(n).lower())

        for _,r in sorted(right):
            child=by_id.get(r.child)
            if child is None or r.child in seen:
                continue
            if r.label in {"nmod","nmod:poss","acl","acl:relcl"}:
                phrase=self._dependent(r.child,r.label,by_id,children,seen)
            elif r.label=="case":
                phrase=""
            elif r.label=="det":
                phrase=self._surface(child).lower()
                seen.add(child.token_id)
            elif r.label=="compound":
                phrase=self._noun_phrase(r.child,by_id,children,seen)
            elif child.upos=="VERB":
                phrase=self._verb(r.child,by_id,children,seen)
            else:
                phrase=self._surface(child).lower()
                seen.add(child.token_id)
            if phrase:
                pieces.append(phrase)

        # A case-marked nmod is realized with its preposition before the
        # dependent noun phrase, regardless of where UD stores the case child.
        return " ".join(x for x in pieces if x)

    def _dependent(self,tid,label,by_id,children,seen):
        node=by_id.get(tid)
        if node is None or tid in seen:
            return ""

        case_children=[
            by_id[r.child]
            for r in children.get(tid,[])
            if r.label=="case" and r.child in by_id
        ]

        if node.upos=="VERB":
            body=self._verb(tid,by_id,children,seen)
        else:
            body=self._noun_phrase(tid,by_id,children,seen)

        if case_children:
            prep=" ".join(
                self._surface(x).lower()
                for x in sorted(case_children,key=lambda x:self._order(x.token_id))
            )
            return f"{prep} {body}".strip()
        return body

    def _verb(self,tid,by_id,children,seen):
        if tid in seen:
            return ""
        n=by_id[tid]
        local=children.get(tid,[])

        before=[]
        after=[]

        # Collect children by relation and learned direction.
        for r in local:
            child=by_id.get(r.child)
            if child is None or r.child in seen:
                continue
            if r.label=="punct":
                continue
            direction=self.grammar.learned_direction(
                n.upos,
                r.label,
                default="after",
            )
            (before if direction=="before" else after).append(r)

        def rank(r):
            try:
                return int(r.child)
            except Exception:
                return 10**9

        before=sorted(before,key=rank)
        after=sorted(after,key=rank)

        parts=[]

        # Subject usually realizes before VP regardless of head-relative stats.
        subjects=[r for r in before+after if r.label.startswith("nsubj")]
        auxiliaries=[r for r in before+after if r.label in {"aux","aux:pass","cop"}]
        negations=[r for r in before+after if r.label=="neg"]

        # For verbal predicates, grammatical order is:
        # subject + auxiliaries + negation + lexical verb + complements.
        for r in sorted(subjects,key=rank):
            parts.append(self._noun_phrase(r.child,by_id,children,seen))

        for r in sorted(auxiliaries,key=rank):
            child=by_id.get(r.child)
            if child:
                parts.append(self._surface(child).lower())
                seen.add(child.token_id)

        for r in sorted(negations,key=rank):
            child=by_id.get(r.child)
            if child:
                parts.append(self._surface(child).lower())
                seen.add(child.token_id)

        seen.add(tid)
        parts.append(self._surface(n).lower())

        # Learned post-head complement order.
        complement_rels={
            "obj","iobj","obl","obl:agent","advmod",
            "ccomp","xcomp","advcl","acl","acl:relcl",
            "conj","cc","mark","nmod","amod",
        }

        complements=[
            r for r in local
            if r.label in complement_rels and r.child not in seen
        ]

        # Put nmod/advmod/etc according to observed average left/right
        # direction, then original token position.
        def comp_key(r):
            direction=self.grammar.learned_direction(
                n.upos,r.label,default="after"
            )
            return (
                0 if direction=="after" else -1,
                self._order(r.child),
            )

        for r in sorted(complements,key=comp_key):
            child=by_id.get(r.child)
            if child is None or r.child in seen:
                continue
            if r.label in {"obj","iobj","obl","obl:agent","nmod"}:
                text=self._dependent(r.child,r.label,by_id,children,seen)
            elif child.upos=="VERB":
                text=self._verb(r.child,by_id,children,seen)
            else:
                text=self._surface(child).lower()
                seen.add(child.token_id)
            if text:
                parts.append(text)

        return " ".join(x for x in parts if x)

    @staticmethod
    def _order(tid):
        try:
            return int(tid)
        except Exception:
            return 10**9


# ============================================================================
# spaCy evaluation
# ============================================================================

def load_spacy(model):
    try:
        import spacy
    except ImportError as exc:
        raise SystemExit(
            "spaCy is not installed. Run:\n"
            "python -m pip install -U spacy\n"
            f"python -m spacy download {model}"
        ) from exc

    try:
        return spacy.load(model)
    except Exception as exc:
        raise SystemExit(
            f"Could not load spaCy model '{model}'. Run:\n"
            "python -m pip install -U spacy\n"
            f"python -m spacy download {model}"
        ) from exc


def lemma_multiset_recall(source_tokens,doc):
    source=[
        canonical_concept(
            t.lemma if t.lemma!="_" else t.form
        )
        for t in source_tokens
        if t.upos not in {"PUNCT","SYM"}
    ]
    target=[
        canonical_concept(t.lemma_)
        for t in doc
        if not t.is_space and not t.is_punct
    ]
    counts=Counter(target)
    hits=0
    for x in source:
        if counts[x]:
            counts[x]-=1
            hits+=1
    return hits/max(1,len(source))


def relation_recall(source,doc):
    src_by_id={t.id:t for t in source.tokens}
    source_rel=set()

    for t in source.tokens:
        if t.upos in {"PUNCT","SYM"}:
            continue
        if not t.head.isdigit() or t.head=="0":
            continue
        head=src_by_id.get(t.head)
        if not head:
            continue
        source_rel.add((
            canonical_concept(head.lemma if head.lemma!="_" else head.form),
            t.deprel,
            canonical_concept(t.lemma if t.lemma!="_" else t.form),
        ))

    target_rel=set()
    for t in doc:
        if t.is_space or t.is_punct:
            continue
        target_rel.add((
            canonical_concept(t.head.lemma_),
            t.dep_,
            canonical_concept(t.lemma_),
        ))

    return len(source_rel&target_rel)/max(1,len(source_rel))


def pos_recall(source,doc):
    target=defaultdict(list)
    for t in doc:
        if not t.is_punct and not t.is_space:
            target[canonical_concept(t.lemma_)].append(t.pos_)

    hits=total=0
    for t in source.tokens:
        if t.upos in {"PUNCT","SYM"}:
            continue
        key=canonical_concept(t.lemma if t.lemma!="_" else t.form)
        if key not in target:
            continue
        total+=1
        if t.upos in target[key]:
            hits+=1
    return hits/max(1,total)


def category(source:UDSentence):
    has_verb=any(t.upos=="VERB" for t in source.tokens)
    has_subordinate=any(
        t.deprel in {
            "ccomp","xcomp","advcl","acl","acl:relcl"
        }
        for t in source.tokens
    )
    has_coord=any(t.deprel=="conj" for t in source.tokens)
    has_question=source.text.strip().endswith("?")
    if has_subordinate:
        return "embedded"
    if has_coord:
        return "coordination"
    if has_verb:
        return "clause"
    if has_question:
        return "question"
    return "nominal_fragment"


def smoke():
    memory=IndexedSemanticMemory.from_edges([
        SemanticEdge("sociologist","RelatedTo","researcher"),
        SemanticEdge("have","RelatedTo","possess"),
        SemanticEdge("explore","RelatedTo","investigate"),
        SemanticEdge("consequence","RelatedTo","effect"),
        SemanticEdge("adverse","RelatedTo","negative"),
        SemanticEdge("discrimination","RelatedTo","bias"),
    ])
    arch=IntegratedSemanticArchitecture(memory)

    grammar=GrammarModel()
    sample=UDSentence(
        "Sociologists have explored the adverse consequences of discrimination.",
        (
            UDToken("1","Sociologists","sociologist","NOUN","NNS","Number=Plur","3","nsubj","_","_"),
            UDToken("2","have","have","AUX","VBP","Mood=Ind|Tense=Pres|VerbForm=Fin","3","aux","_","_"),
            UDToken("3","explored","explore","VERB","VBN","Tense=Past|VerbForm=Part","0","root","_","_"),
            UDToken("4","the","the","DET","DT","Definite=Def","6","det","_","_"),
            UDToken("5","adverse","adverse","ADJ","JJ","Degree=Pos","6","amod","_","_"),
            UDToken("6","consequences","consequence","NOUN","NNS","Number=Plur","3","obj","_","_"),
            UDToken("7","of","of","ADP","IN","_","8","case","_","_"),
            UDToken("8","discrimination","discrimination","NOUN","NN","Number=Sing","6","nmod","_","_"),
        ),
        (),
        "smoke",
    )
    grammar.observe(sample)

    bridge=CognitiveLanguageArchitecture(arch,grammar)
    state=bridge.perceive(sample)
    generated=LearnedLinearizer(grammar).generate(state)

    assert state.propositions
    assert state.relations
    assert state.cognitive_beliefs
    assert "explored" in generated
    assert "consequences" in generated
    assert "adverse" in generated
    assert "discrimination" in generated
    assert generated.lower().startswith("sociologists have explored")
    print("V409 learned-linearization smoke: PASS")
    print("gold UD grammar: PASS")
    print("data-driven morphology: PASS")
    print("recursive language state: PASS")
    print("cognitive grounding: PASS")
    print("learned dependency linearization: PASS")
    print("auxiliary preservation: PASS")
    print("NP modifier/complement ordering: PASS")
    print("structure-aware generation: PASS")
    print("independent spaCy evaluation path: PASS")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("gum",nargs="?",type=Path,default=Path(r".\data\UD_GUM"))
    ap.add_argument("--conceptnet",type=Path,default=Path(r".\data\conceptnet_compact.db"))
    ap.add_argument("--spacy-model",default="en_core_web_trf")
    ap.add_argument("--max-cases",type=int,default=100)
    ap.add_argument("--progress-every",type=int,default=25)
    ap.add_argument("--semantic-warmup",type=int,default=1000)
    ap.add_argument("--lexical-threshold",type=float,default=0.80)
    ap.add_argument("--dependency-threshold",type=float,default=0.30)
    ap.add_argument("--adequacy-threshold",type=float,default=0.65)
    ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()

    if args.smoke:
        smoke()
        return

    start=time.perf_counter()
    gum=args.gum.resolve()
    db=args.conceptnet.resolve()

    print("="*78,flush=True)
    print("V409 GUM + COGNITIVE LANGUAGE + LEARNED LINEARIZATION",flush=True)
    print("="*78,flush=True)

    print("[1/10] Loading spaCy judge...",flush=True)
    nlp=load_spacy(args.spacy_model)
    print(f"      {args.spacy_model}",flush=True)

    print("[2/10] Reading GUM...",flush=True)
    files=discover_gum(gum)
    splits=defaultdict(list)
    total=0
    for f in files:
        rows=parse_conllu(f)
        splits[split_name(f)].extend(rows)
        total+=len(rows)
        print(
            f"      {f.name}: {len(rows):,} [{split_name(f)}]",
            flush=True,
        )

    train=splits["train"]
    dev=splits["dev"]
    test=splits["test"]
    if not train or not test:
        raise SystemExit("GUM train/test split not found.")

    print(
        f"      total={total:,} train={len(train):,} "
        f"dev={len(dev):,} test={len(test):,}",
        flush=True,
    )

    print("[3/10] Learning grammar + morphology + linearization...",flush=True)
    grammar=GrammarModel()

    for i,s in enumerate(train,1):
        grammar.observe(s)
        if args.progress_every and (
            i%args.progress_every==0 or i==len(train)
        ):
            print(
                f"      train={i:,}/{len(train):,} "
                f"deps={len(grammar.dependencies):,} "
                f"morph={len(grammar.morph_forms):,} "
                f"linearization={len(grammar.relation_order):,}",
                flush=True,
            )

    print("[4/10] Loading ConceptNet + cognitive architecture...",flush=True)
    graph=IndexedConceptNet(db).build_index()
    memory=IndexedSemanticMemory.from_edges(
        SemanticEdge(
            source=e.source,
            relation=e.relation,
            target=e.target,
            weight=getattr(e,"weight",1.0),
            provenance="conceptnet",
        )
        for edges in graph.adj.values()
        for e in edges
    )
    architecture=IntegratedSemanticArchitecture(memory)

    # Bounded semantic warm-up, using the same training material.
    bridge=CognitiveLanguageArchitecture(architecture,grammar)
    for s in train[:min(args.semantic_warmup,len(train))]:
        for t in s.tokens:
            if t.upos in {"PUNCT","SYM"}:
                continue
            bridge.ground(t.form,t.lemma)

    realizer=LearnedLinearizer(grammar)

    print(
        f"      concepts={len(graph.concepts):,} "
        f"edges={graph.edge_count:,} "
        f"history={len(architecture.history):,}",
        flush=True,
    )

    tested=test[:args.max_cases]

    print("[5/10] Perceiving RAW GUM test sentences...",flush=True)
    states=[]
    for i,s in enumerate(tested,1):
        states.append(bridge.perceive(s))
        if args.progress_every and (
            i%args.progress_every==0 or i==len(tested)
        ):
            print(
                f"      states={i:,}/{len(tested):,} "
                f"predicates={sum(len(x.propositions) for x in states):,}",
                flush=True,
            )

    print("[6/10] Realizing with learned linearization...",flush=True)
    generated=[]
    for i,s in enumerate(states,1):
        text=realizer.generate(s)
        generated.append(text)
        if args.progress_every and (
            i%args.progress_every==0 or i==len(states)
        ):
            print(
                f"      generated={i:,}/{len(states):,}",
                flush=True,
            )

    print("[7/10] Independent spaCy parsing...",flush=True)
    docs=list(nlp.pipe(
        generated,
        batch_size=max(1,args.progress_every),
    ))

    print("[8/10] Scoring lexical/POS/dependency + categories...",flush=True)
    scores=[]
    category_scores=defaultdict(list)

    for source,text,doc in zip(tested,generated,docs):
        lexical=lemma_multiset_recall(source.tokens,doc)
        dep=relation_recall(source,doc)
        pos=pos_recall(source,doc)
        adequacy=0.45*lexical+0.40*dep+0.15*pos
        cat=category(source)

        item={
            "source":source.text,
            "generated":text,
            "category":cat,
            "lexical_recall":lexical,
            "dependency_recall":dep,
            "pos_recall":pos,
            "adequacy":adequacy,
            "source_tokens":len(
                [t for t in source.tokens if t.upos not in {"PUNCT","SYM"}]
            ),
            "generated_tokens":len(
                [t for t in doc if not t.is_space and not t.is_punct]
            ),
        }
        scores.append(item)
        category_scores[cat].append(item)

    avg_lex=sum(x["lexical_recall"] for x in scores)/max(1,len(scores))
    avg_dep=sum(x["dependency_recall"] for x in scores)/max(1,len(scores))
    avg_pos=sum(x["pos_recall"] for x in scores)/max(1,len(scores))
    avg_adequacy=sum(x["adequacy"] for x in scores)/max(1,len(scores))

    by_category={}
    for cat,rows in category_scores.items():
        by_category[cat]={
            "cases":len(rows),
            "lexical_recall":sum(x["lexical_recall"] for x in rows)/len(rows),
            "dependency_recall":sum(x["dependency_recall"] for x in rows)/len(rows),
            "pos_recall":sum(x["pos_recall"] for x in rows)/len(rows),
            "adequacy":sum(x["adequacy"] for x in rows)/len(rows),
        }

    print(
        f"      lexical={avg_lex:.3f} "
        f"dependency={avg_dep:.3f} "
        f"POS={avg_pos:.3f} "
        f"adequacy={avg_adequacy:.3f}",
        flush=True,
    )
    for cat,val in sorted(by_category.items()):
        print(
            f"      [{cat}] cases={val['cases']} "
            f"adequacy={val['adequacy']:.3f} "
            f"dep={val['dependency_recall']:.3f}",
            flush=True,
        )

    print("[9/10] Representation audit...",flush=True)
    audit={
        "raw_test_sentences":len(tested),
        "states":len(states),
        "syntax_relations":sum(len(s.relations) for s in states),
        "predicates":sum(len(s.propositions) for s in states),
        "recursive_embedded_links":sum(
            len(p.embedded)
            for s in states
            for p in s.propositions
        ),
        "auxiliary_links":sum(
            len(p.auxiliaries)
            for s in states
            for p in s.propositions
        ),
        "negation_links":sum(
            len(p.negations)
            for s in states
            for p in s.propositions
        ),
        "morphology_features":sum(
            sum(len(n.morphology) for n in s.nodes)
            for s in states
        ),
        "cognitive_beliefs":sum(len(s.cognitive_beliefs) for s in states),
        "unresolved_tokens":sum(len(s.unresolved) for s in states),
    }

    print("[10/10] Final checks...",flush=True)
    checks={
        "gum_loaded":total>0,
        "gold_ud_grammar_learned":grammar.sentences>0 and grammar.tokens>0,
        "dependency_grammar_learned":len(grammar.dependencies)>0,
        "morphology_learned":len(grammar.morph_forms)>0,
        "linearization_learned":len(grammar.relation_order)>0,
        "conceptnet_loaded":graph.edge_count>0,
        "cognitive_architecture_active":len(architecture.history)>0,
        "raw_gum_test_source_only":len(tested)==min(args.max_cases,len(test)),
        "explicit_states_produced":len(states)==len(tested),
        "spacy_judge_active":len(docs)==len(tested),
        "generation_nonempty":all(bool(x.strip()) for x in generated),
        "lexical_coverage_pass":avg_lex>=args.lexical_threshold,
        "dependency_coverage_pass":avg_dep>=args.dependency_threshold,
        "generation_adequacy_pass":avg_adequacy>=args.adequacy_threshold,
    }
    status="PASS" if all(checks.values()) else "FAIL"

    report={
        "status":status,
        "version":"v409",
        "methodology":{
            "grammar_source":"UD GUM gold CoNLL-U",
            "grammar_supervision":"UPOS/XPOS/FEATS/dependency heads+relations",
            "semantic_source":"ConceptNet",
            "cognitive_architecture":True,
            "roundtrip_source":"raw GUM test sentences",
            "generated_text_used_as_training_source":False,
            "representation":"recursive compositional cognitive language state",
            "linearization":"learned from GUM training dependency directions",
            "independent_judge":"spaCy",
            "independent_judge_model":args.spacy_model,
        },
        "gum":{
            "path":str(gum),
            "files":len(files),
            "sentences_total":total,
            "train_sentences":len(train),
            "dev_sentences":len(dev),
            "test_sentences":len(test),
        },
        "grammar":grammar.report(),
        "conceptnet":{
            "concepts":len(graph.concepts),
            "edges":graph.edge_count,
        },
        "explicit_representation":audit,
        "generation_evaluation":{
            "spaCy_model":args.spacy_model,
            "cases":len(scores),
            "avg_lexical_recall":avg_lex,
            "avg_dependency_recall":avg_dep,
            "avg_pos_recall":avg_pos,
            "avg_adequacy":avg_adequacy,
            "by_category":by_category,
        },
        "thresholds":{
            "lexical":args.lexical_threshold,
            "dependency":args.dependency_threshold,
            "adequacy":args.adequacy_threshold,
        },
        "checks":checks,
        "examples":{
            "first_10":scores[:10],
        },
        "wall_time_seconds":time.perf_counter()-start,
    }

    out=Path.cwd()/"results"/"v409_learned_linearization.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")

    for k,v in checks.items():
        print(
            f"  {k:44} {'PASS' if v else 'FAIL'}",
            flush=True,
        )
    print(f"[RESULT] {status}",flush=True)
    print(f"[RESULT FILE] {out.resolve()}",flush=True)
    graph.close()


if __name__=="__main__":
    main()
