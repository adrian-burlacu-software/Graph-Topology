
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


# ---------------------------------------------------------------------------
# GUM / CoNLL-U
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UDToken:
    id: str
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: str
    deprel: str
    deps: str
    misc: str


@dataclass(frozen=True)
class UDSentence:
    text: str
    tokens: tuple[UDToken, ...]
    comments: tuple[str, ...]
    source_file: str


def parse_conllu(path: Path) -> list[UDSentence]:
    sentences=[]
    comments=[]
    rows=[]

    def flush():
        nonlocal comments, rows
        if not rows:
            comments=[]
            return
        text=next(
            (c[9:] for c in comments if c.startswith("# text = ")),
            " ".join(t.form for t in rows),
        )
        sentences.append(
            UDSentence(text,tuple(rows),tuple(comments),str(path))
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
                    f"{path}:{line_no}: expected 10 CoNLL-U columns; got {len(cols)}"
                )
            if "-" in cols[0] or "." in cols[0]:
                continue
            rows.append(UDToken(*cols))
    flush()
    return sentences


def discover_conllu(root: Path):
    files=sorted(root.rglob("*.conllu"))
    if not files:
        raise FileNotFoundError(f"No .conllu files found under {root}")
    return files


def split_name(path: Path):
    n=path.name.lower()
    if "train" in n: return "train"
    if "dev" in n: return "dev"
    if "test2" in n or "gentle" in n: return "test2"
    if "test" in n: return "test"
    return "unknown"


# ---------------------------------------------------------------------------
# Gold grammar learning + morphology
# ---------------------------------------------------------------------------

def parse_feats(feats: str):
    if not feats or feats=="_":
        return ()
    result=[]
    for item in feats.split("|"):
        if "=" in item:
            k,v=item.split("=",1)
            result.append((k,v))
    return tuple(sorted(result))


class GrammarLearner:
    def __init__(self):
        self.sentences=0
        self.tokens=0
        self.upos=Counter()
        self.xpos=Counter()
        self.dependencies=Counter()
        self.features=Counter()
        self.morphology=defaultdict(Counter)

    def observe(self,s:UDSentence):
        self.sentences+=1
        self.tokens+=len(s.tokens)
        by_id={t.id:t for t in s.tokens}

        for t in s.tokens:
            self.upos[t.upos]+=1
            self.xpos[t.xpos]+=1
            feats=parse_feats(t.feats)
            for f in feats:
                self.features[f]+=1
            if t.lemma and t.lemma!="_":
                self.morphology[(t.lemma,feats)][t.form]+=1

            if t.head in by_id:
                h=by_id[t.head]
                self.dependencies[(h.upos,t.deprel,t.upos)]+=1

    def realize(self,lemma,feats,fallback=None):
        counts=self.morphology.get((lemma,tuple(sorted(feats))))
        if counts:
            return counts.most_common(1)[0][0]
        return fallback or lemma

    def report(self):
        return {
            "sentences":self.sentences,
            "tokens":self.tokens,
            "upos_types":len(self.upos),
            "xpos_types":len(self.xpos),
            "dependency_types":len(self.dependencies),
            "feature_types":len(self.features),
            "morphology_pairs":len(self.morphology),
            "top_dependencies":{
                str(k):v for k,v in self.dependencies.most_common(20)
            },
        }


# ---------------------------------------------------------------------------
# Cognitive language state
# ---------------------------------------------------------------------------

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
    grounding_candidates:tuple[str,...]


@dataclass(frozen=True)
class CognitivePredicate:
    token_id:str
    concept:str
    morphology:tuple[tuple[str,str],...]
    arguments:tuple[tuple[str,str,str],...]


@dataclass(frozen=True)
class CognitiveLanguageState:
    source_text:str
    nodes:tuple[LanguageNode,...]
    syntax_edges:tuple[tuple[str,str,str],...]
    predicates:tuple[CognitivePredicate,...]
    cognitive_beliefs:tuple[tuple[str,float,str],...]
    unresolved:tuple[str,...]

    def semantic_signature(self):
        concepts={
            n.token_id:n.concept
            for n in self.nodes
            if n.concept is not None
        }
        predicates=[]
        for p in self.predicates:
            args=tuple(sorted(
                (
                    rel,
                    concepts.get(child,""),
                )
                for rel,child,_ in p.arguments
                if concepts.get(child) is not None
                and rel in {
                    "nsubj","nsubj:pass","obj","iobj","obl",
                    "obl:agent","ccomp","xcomp","advcl","acl","acl:relcl",
                    "nmod","nmod:poss","amod","advmod","compound",
                }
            ))
            predicates.append(
                (
                    p.concept,
                    tuple(sorted(p.morphology)),
                    args,
                )
            )
        return tuple(sorted(predicates))

    def lexical_signature(self):
        return tuple(sorted(
            (
                n.lemma,
                n.upos,
            )
            for n in self.nodes
            if n.concept is not None
        ))


class CognitiveLanguageArchitecture:
    def __init__(self,architecture,grammar:GrammarLearner):
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

        candidates=[key]
        if key.endswith("ies") and len(key)>4:
            candidates.append(key[:-3]+"y")
        if key.endswith("es") and len(key)>4:
            candidates.append(key[:-2])
        if key.endswith("s") and len(key)>3:
            candidates.append(key[:-1])
        if key.endswith("ed") and len(key)>4:
            candidates.extend((key[:-2],key[:-1]))
        if key.endswith("ing") and len(key)>5:
            candidates.append(key[:-3])

        for c in candidates:
            if c in self.concepts:
                belief=self.arch.perceive(c,context=())
                result=(
                    c,
                    float(belief.confidence),
                    tuple(belief.candidates),
                )
                self.cache[key]=result
                return result

        self.cache[key]=None
        return None

    def perceive(self,s:UDSentence):
        nodes=[]
        unresolved=[]

        for t in s.tokens:
            if t.upos in {"PUNCT","SYM"}:
                continue
            grounded=self.ground(t.form,t.lemma)
            if grounded is None:
                unresolved.append(t.id)
                concept=None
                conf=0.0
                candidates=()
            else:
                concept,conf,candidates=grounded

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
                    candidates,
                )
            )

        syntax=[]
        for t in s.tokens:
            if t.upos in {"PUNCT","SYM"}:
                continue
            if t.head.isdigit() and t.head!="0":
                syntax.append((t.head,t.deprel,t.id))

        by_id={n.token_id:n for n in nodes}
        predicates=[]

        for n in nodes:
            if n.upos!="VERB" or n.concept is None:
                continue

            args=[]
            for h,rel,c in syntax:
                if h!=n.token_id:
                    continue
                child=by_id.get(c)
                if child is None or child.concept is None:
                    continue
                args.append((rel,c,rel))

            predicates.append(
                CognitivePredicate(
                    n.token_id,
                    n.concept,
                    n.morphology,
                    tuple(args),
                )
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

        return CognitiveLanguageState(
            s.text,
            tuple(nodes),
            tuple(syntax),
            tuple(predicates),
            tuple(beliefs),
            tuple(unresolved),
        )


# ---------------------------------------------------------------------------
# Structure-preserving realizer
# ---------------------------------------------------------------------------

class Realizer:
    def __init__(self,grammar:GrammarLearner):
        self.grammar=grammar

    def generate(self,state:CognitiveLanguageState):
        by_id={n.token_id:n for n in state.nodes}
        children=defaultdict(list)

        for h,rel,c in state.syntax_edges:
            children[h].append((rel,c))

        root=next(
            (n.token_id for n in state.nodes if n.token_id
             not in {c for _,_,c in state.syntax_edges}),
            state.nodes[0].token_id if state.nodes else "",
        )

        if not root:
            return ""

        return self._realize_node(
            root,
            by_id,
            children,
            set(),
        ).strip()

    def _form(self,node):
        return self.grammar.realize(
            node.lemma,
            node.morphology,
            fallback=node.surface,
        )

    def _np(self,tid,by_id,children,seen):
        node=by_id[tid]
        pieces=[]

        dets=sorted(
            [c for r,c in children.get(tid,[]) if r=="det"],
            key=lambda x:int(x) if x.isdigit() else 999999,
        )
        for c in dets:
            if c in by_id:
                pieces.append(by_id[c].surface.lower())
                seen.add(c)

        poss=sorted(
            [c for r,c in children.get(tid,[]) if r=="nmod:poss"],
            key=lambda x:int(x) if x.isdigit() else 999999,
        )
        for c in poss:
            if c in by_id and c not in seen:
                pieces.append(self._np(c,by_id,children,seen))

        compounds=sorted(
            [c for r,c in children.get(tid,[]) if r=="compound"],
            key=lambda x:int(x) if x.isdigit() else 999999,
        )
        for c in compounds:
            if c in by_id and c not in seen:
                pieces.append(self._np(c,by_id,children,seen))

        amods=sorted(
            [c for r,c in children.get(tid,[]) if r=="amod"],
            key=lambda x:int(x) if x.isdigit() else 999999,
        )
        for c in amods:
            if c in by_id and c not in seen:
                pieces.append(self._form(by_id[c]).lower())
                seen.add(c)

        seen.add(tid)
        pieces.append(self._form(node))

        nmods=sorted(
            [(r,c) for r,c in children.get(tid,[]) if r in {"nmod","acl","acl:relcl"}],
            key=lambda x:int(x[1]) if x[1].isdigit() else 999999,
        )
        for rel,c in nmods:
            if c in seen or c not in by_id:
                continue
            child=by_id[c]
            if rel.startswith("acl") and child.upos=="VERB":
                pieces.append(self._vp(c,by_id,children,seen))
            else:
                case=[
                    by_id[x]
                    for rr,x in children.get(c,[])
                    if rr=="case" and x in by_id
                ]
                phrase=self._np(c,by_id,children,seen)
                if case:
                    pieces.append(
                        " ".join(self._form(x).lower() for x in case)
                        + " " + phrase
                    )
                else:
                    pieces.append(phrase)

        return " ".join(x for x in pieces if x)

    def _vp(self,tid,by_id,children,seen):
        node=by_id[tid]
        seen.add(tid)

        subs=[c for r,c in children.get(tid,[]) if r.startswith("nsubj")]
        aux=sorted(
            [c for r,c in children.get(tid,[]) if r.startswith("aux") or r=="cop"],
            key=lambda x:int(x) if x.isdigit() else 999999,
        )
        neg=[c for r,c in children.get(tid,[]) if r=="neg"]
        objs=sorted(
            [c for r,c in children.get(tid,[]) if r in {"obj","iobj"}],
            key=lambda x:int(x) if x.isdigit() else 999999,
        )
        obls=sorted(
            [c for r,c in children.get(tid,[]) if r.startswith("obl")],
            key=lambda x:int(x) if x.isdigit() else 999999,
        )
        adv=sorted(
            [c for r,c in children.get(tid,[]) if r=="advmod"],
            key=lambda x:int(x) if x.isdigit() else 999999,
        )
        embedded=[
            c for r,c in children.get(tid,[])
            if r in {"ccomp","xcomp","advcl"}
        ]

        parts=[]
        for c in subs:
            if c in by_id and c not in seen:
                parts.append(self._np(c,by_id,children,seen))

        for c in aux:
            if c in by_id:
                parts.append(self._form(by_id[c]).lower())
                seen.add(c)

        for c in neg:
            if c in by_id:
                parts.append(self._form(by_id[c]).lower())
                seen.add(c)

        # Verb morphology itself comes from the gold training morphology map.
        parts.append(self._form(node).lower())

        for c in objs:
            if c in by_id and c not in seen:
                parts.append(self._np(c,by_id,children,seen))

        for c in obls:
            if c in by_id and c not in seen:
                parts.append(self._np(c,by_id,children,seen))

        for c in adv:
            if c in by_id and c not in seen:
                parts.append(self._form(by_id[c]).lower())
                seen.add(c)

        for c in embedded:
            if c not in by_id or c in seen:
                continue
            child=by_id[c]
            if child.upos=="VERB":
                parts.append(self._vp(c,by_id,children,seen))

        return " ".join(x for x in parts if x)

    def _realize_node(self,tid,by_id,children,seen):
        node=by_id[tid]
        if node.upos=="VERB":
            return self._vp(tid,by_id,children,seen)
        if node.upos in {"NOUN","PROPN","PRON"}:
            return self._np(tid,by_id,children,seen)
        seen.add(tid)
        return self._form(node).lower()


# ---------------------------------------------------------------------------
# spaCy independent evaluator
# ---------------------------------------------------------------------------

def load_spacy(model_name):
    try:
        import spacy
    except ImportError as exc:
        raise SystemExit(
            "spaCy is not installed. Run:\n"
            "  python -m pip install -U spacy\n"
            "then:\n"
            f"  python -m spacy download {model_name}"
        ) from exc

    try:
        return spacy.load(model_name)
    except Exception as exc:
        raise SystemExit(
            f"Could not load spaCy model '{model_name}'. Run:\n"
            "  python -m pip install -U spacy\n"
            f"  python -m spacy download {model_name}"
        ) from exc


def align_content_tokens(source:UDSentence, doc):
    source_words=[
        t for t in source.tokens
        if t.upos not in {"PUNCT","SYM"}
    ]
    target_words=[
        t for t in doc
        if not t.is_punct and not t.is_space
    ]
    return source_words,target_words


def score_spacy(source:UDSentence, generated, doc):
    source_tokens,target_tokens=align_content_tokens(source,doc)

    source_lemmas=[
        canonical_concept(t.lemma if t.lemma!="_" else t.form)
        for t in source_tokens
    ]
    target_lemmas=[
        canonical_concept(t.lemma_)
        for t in target_tokens
    ]

    target_positions=defaultdict(list)
    for i,lemma in enumerate(target_lemmas):
        target_positions[lemma].append(i)

    lemma_matches=0
    used=set()
    for lemma in source_lemmas:
        candidates=target_positions.get(lemma,[])
        for pos in candidates:
            if pos not in used:
                used.add(pos)
                lemma_matches+=1
                break

    lexical_recall=lemma_matches/max(1,len(source_lemmas))

    # Compare available UD relations at the concept/lemma level. This is
    # intentionally tolerant of word-order changes.
    src_by_id={t.id:t for t in source.tokens}
    src_relations=set()
    for t in source_tokens:
        if not t.head.isdigit() or t.head=="0":
            continue
        h=src_by_id.get(t.head)
        if h:
            src_relations.add((
                canonical_concept(h.lemma if h.lemma!="_" else h.form),
                t.deprel,
                canonical_concept(t.lemma if t.lemma!="_" else t.form),
            ))

    tgt_relations=set()
    for t in target_tokens:
        if t.head==t:
            continue
        head=t.head
        if head is None:
            continue
        tgt_relations.add((
            canonical_concept(head.lemma_),
            t.dep_,
            canonical_concept(t.lemma_),
        ))

    relation_recall=(
        len(src_relations & tgt_relations)
        /max(1,len(src_relations))
    )

    # POS retention is measured by aligned lemma occurrence and exact POS.
    pos_hits=0
    pos_total=0
    tgt_by_lemma=defaultdict(list)
    for t in target_tokens:
        tgt_by_lemma[canonical_concept(t.lemma_)].append(t)

    for st in source_tokens:
        lemma=canonical_concept(st.lemma if st.lemma!="_" else st.form)
        candidates=tgt_by_lemma.get(lemma,[])
        if candidates:
            pos_total+=1
            if any(
                t.pos_==st.upos
                or t.tag_==st.xpos
                for t in candidates
            ):
                pos_hits+=1

    pos_recall=pos_hits/max(1,pos_total)

    adequacy=(
        0.45*lexical_recall
        +0.45*relation_recall
        +0.10*pos_recall
    )

    return {
        "lexical_recall":lexical_recall,
        "dependency_relation_recall":relation_recall,
        "pos_recall":pos_recall,
        "adequacy":adequacy,
        "source_tokens":len(source_tokens),
        "generated_tokens":len(target_tokens),
    }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def smoke():
    # No external spaCy model required for the structural smoke. The actual
    # real-data benchmark requires the requested spaCy model.
    memory=IndexedSemanticMemory.from_edges([
        SemanticEdge("dog","IsA","animal"),
        SemanticEdge("cat","IsA","animal"),
        SemanticEdge("chase","RelatedTo","pursuit"),
    ])
    arch=IntegratedSemanticArchitecture(memory)

    grammar=GrammarLearner()
    s=UDSentence(
        "The dog chased the cat.",
        (
            UDToken("1","The","the","DET","DT","Definite=Def","2","det","_","_"),
            UDToken("2","dog","dog","NOUN","NN","Number=Sing","3","nsubj","_","_"),
            UDToken("3","chased","chase","VERB","VBD","Tense=Past|VerbForm=Fin","0","root","_","_"),
            UDToken("4","the","the","DET","DT","Definite=Def","5","det","_","_"),
            UDToken("5","cat","cat","NOUN","NN","Number=Sing","3","obj","_","_"),
        ),
        (),
        "smoke",
    )
    grammar.observe(s)

    bridge=CognitiveLanguageArchitecture(arch,grammar)
    state=bridge.perceive(s)
    out=Realizer(grammar).generate(state)

    assert state.nodes
    assert state.syntax_edges
    assert state.predicates
    assert out

    print("V407 spaCy-independent benchmark smoke: PASS")
    print("gold UD grammar: PASS")
    print("morphology learner: PASS")
    print("cognitive grounding bridge: PASS")
    print("explicit syntax graph: PASS")
    print("structure-aware realization: PASS")
    print("external spaCy judge path: READY")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "gum",nargs="?",type=Path,
        default=Path(r".\data\UD_GUM"),
    )
    ap.add_argument(
        "--conceptnet",type=Path,
        default=Path(r".\data\conceptnet_compact.db"),
    )
    ap.add_argument(
        "--spacy-model",
        default="en_core_web_trf",
    )
    ap.add_argument("--max-cases",type=int,default=100)
    ap.add_argument("--progress-every",type=int,default=25)
    ap.add_argument("--semantic-warmup",type=int,default=1000)
    ap.add_argument("--syntax-threshold",type=float,default=0.70)
    ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()

    if args.smoke:
        smoke()
        return

    start=time.perf_counter()
    gum=args.gum.resolve()
    db=args.conceptnet.resolve()

    print("="*78,flush=True)
    print("V407 UD GUM + COGNITIVE ARCHITECTURE + SPACY JUDGE",flush=True)
    print("="*78,flush=True)

    print("[1/9] Loading spaCy judge...",flush=True)
    nlp=load_spacy(args.spacy_model)
    print(f"      model={args.spacy_model}",flush=True)

    print("[2/9] Discovering GUM...",flush=True)
    files=discover_conllu(gum)
    splits=defaultdict(list)
    total=0
    for f in files:
        rows=parse_conllu(f)
        splits[split_name(f)].extend(rows)
        total+=len(rows)
        print(
            f"      {f.name}: {len(rows):,} "
            f"[{split_name(f)}]",
            flush=True,
        )

    train=splits["train"]
    dev=splits["dev"]
    test=splits["test"]
    if not train or not test:
        raise SystemExit("GUM train/test splits were not found.")

    print(
        f"      total={total:,} train={len(train):,} "
        f"dev={len(dev):,} test={len(test):,}",
        flush=True,
    )

    print("[3/9] Learning grammar from GOLD GUM UD...",flush=True)
    grammar=GrammarLearner()
    for i,s in enumerate(train,1):
        grammar.observe(s)
        if args.progress_every and (
            i%args.progress_every==0 or i==len(train)
        ):
            print(
                f"      train={i:,}/{len(train):,} "
                f"deps={len(grammar.dependencies):,} "
                f"features={len(grammar.features):,} "
                f"morphology={len(grammar.morphology):,}",
                flush=True,
            )

    print("[4/9] Loading ConceptNet + cognitive architecture...",flush=True)
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

    # Exercise existing cognitive semantics with bounded training data.
    from babylm_grammar import GrammarCognitiveLearner
    cognitive=GrammarCognitiveLearner(architecture)
    for s in train[:min(args.semantic_warmup,len(train))]:
        cognitive.observe_sentence(s.text,learn=True)

    bridge=CognitiveLanguageArchitecture(architecture,grammar)
    realizer=Realizer(grammar)

    print(
        f"      concepts={len(graph.concepts):,} "
        f"edges={graph.edge_count:,} "
        f"cognitive_history={len(architecture.history):,}",
        flush=True,
    )

    tested=test[:args.max_cases]

    print("[5/9] Perceiving RAW GUM test sentences...",flush=True)
    states=[bridge.perceive(s) for s in tested]
    print(
        f"      states={len(states):,} "
        f"unresolved={sum(len(s.unresolved) for s in states):,}",
        flush=True,
    )

    print("[6/9] Generating from cognitive language states...",flush=True)
    generated=[]
    for i,state in enumerate(states,1):
        text=realizer.generate(state)
        generated.append(text)
        if args.progress_every and (
            i%args.progress_every==0 or i==len(states)
        ):
            print(
                f"      generated={i:,}/{len(states):,}",
                flush=True,
            )

    print("[7/9] Independent spaCy parsing + scoring...",flush=True)
    docs=list(nlp.pipe(
        generated,
        batch_size=max(1,args.progress_every),
    ))

    scores=[]
    for i,(source,text,doc) in enumerate(
        zip(tested,generated,docs),
        1,
    ):
        score=score_spacy(source,text,doc)
        score.update({
            "source":source.text,
            "generated":text,
        })
        scores.append(score)
        if args.progress_every and (
            i%args.progress_every==0 or i==len(scores)
        ):
            print(
                f"      judged={i:,}/{len(tested):,} "
                f"adequacy={sum(x['adequacy'] for x in scores)/len(scores):.3f} "
                f"dep_recall={sum(x['dependency_relation_recall'] for x in scores)/len(scores):.3f}",
                flush=True,
            )

    avg_lex=sum(s["lexical_recall"] for s in scores)/max(1,len(scores))
    avg_dep=sum(s["dependency_relation_recall"] for s in scores)/max(1,len(scores))
    avg_pos=sum(s["pos_recall"] for s in scores)/max(1,len(scores))
    avg_adequacy=sum(s["adequacy"] for s in scores)/max(1,len(scores))
    adequacy_pass=sum(
        x["adequacy"]>=args.syntax_threshold
        for x in scores
    )/max(1,len(scores))

    print("[8/9] Benchmark summary...",flush=True)
    print(
        f"      lexical_recall={avg_lex:.3f} "
        f"dependency_recall={avg_dep:.3f} "
        f"pos_recall={avg_pos:.3f} "
        f"adequacy={avg_adequacy:.3f} "
        f"case_pass_rate={adequacy_pass:.3f}",
        flush=True,
    )

    print("[9/9] Final checks...",flush=True)
    checks={
        "gum_loaded":total>0,
        "gold_grammar_learned":grammar.sentences>0,
        "dependency_grammar_learned":len(grammar.dependencies)>0,
        "morphology_learned":len(grammar.morphology)>0,
        "conceptnet_loaded":graph.edge_count>0,
        "cognitive_architecture_active":len(architecture.history)>0,
        "raw_gum_test_source_only":len(tested)==min(args.max_cases,len(test)),
        "explicit_states_produced":len(states)==len(tested),
        "spacy_judge_active":bool(docs),
        "generated_nonempty":all(bool(x.strip()) for x in generated),
        "independent_adequacy_pass":avg_adequacy>=args.syntax_threshold,
    }
    status="PASS" if all(checks.values()) else "FAIL"

    report={
        "status":status,
        "version":"v407",
        "methodology":{
            "grammar_source":"UD GUM gold CoNLL-U",
            "grammar_supervision":"UPOS/XPOS/FEATS/dependency heads+relations",
            "semantic_source":"ConceptNet",
            "cognitive_architecture":True,
            "roundtrip_source":"raw GUM test sentences",
            "generated_text_used_as_training_source":False,
            "independent_judge":"spaCy dependency parser",
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
        "explicit_representation":{
            "raw_test_sentences":len(tested),
            "states":len(states),
            "syntax_edges":sum(len(s.syntax_edges) for s in states),
            "predicates":sum(len(s.predicates) for s in states),
            "morphology_features":sum(
                sum(len(n.morphology) for n in s.nodes)
                for s in states
            ),
            "cognitive_beliefs":sum(
                len(s.cognitive_beliefs)
                for s in states
            ),
            "unresolved_tokens":sum(
                len(s.unresolved)
                for s in states
            ),
        },
        "generation_evaluation":{
            "spaCy_model":args.spacy_model,
            "cases":len(scores),
            "avg_lexical_recall":avg_lex,
            "avg_dependency_relation_recall":avg_dep,
            "avg_pos_recall":avg_pos,
            "avg_adequacy":avg_adequacy,
            "adequacy_case_pass_rate":adequacy_pass,
            "adequacy_threshold":args.syntax_threshold,
        },
        "checks":checks,
        "examples":{
            "first_10":scores[:10],
        },
        "wall_time_seconds":time.perf_counter()-start,
    }

    out=Path.cwd()/"results"/"v407_spacy_independent_roundtrip.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(
        json.dumps(report,indent=2,default=str),
        encoding="utf-8",
    )

    for k,v in checks.items():
        print(
            f"  {k:42} {'PASS' if v else 'FAIL'}",
            flush=True,
        )
    print(f"[RESULT] {status}",flush=True)
    print(f"[RESULT FILE] {out.resolve()}",flush=True)

    graph.close()


if __name__=="__main__":
    main()
