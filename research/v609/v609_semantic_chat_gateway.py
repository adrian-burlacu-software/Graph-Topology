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
    """
    Entity-aware conversational memory.

    No domain-specific semantic rules are stored here. The memory records:
      - entity mentions and their spaCy labels;
      - active subject;
      - recent goals and outcomes;
      - aliases introduced by the conversation;
      - relation evidence learned from successful graph resolutions.
    """

    PRONOUNS = {
        "he", "she", "they", "him", "her", "them",
        "his", "their", "its", "it"
    }

    def __init__(self, path: Path | None = None):
        self.path = path
        self.active_subject = None
        self.active_entity_label = None
        self.entities = []
        self.turns = []
        self.goal_memory = defaultdict(Counter)
        self.relation_memory = defaultdict(Counter)

        if path and path.exists():
            self.load()

    def _remember_entity(self, text, label):
        if not text:
            return
        key = text.lower()
        if not any(
            item["text"].lower() == key
            for item in self.entities
        ):
            self.entities.append({
                "text": text,
                "label": label or "UNKNOWN",
                "mentions": 1,
            })
        else:
            for item in self.entities:
                if item["text"].lower() == key:
                    item["mentions"] += 1

        self.entities = sorted(
            self.entities,
            key=lambda x: (
                -x["mentions"],
                x["text"].lower(),
            ),
        )[:128]

    def subject(self, parse: Parse):
        if parse.entities:
            ent = parse.entities[0]
            self.active_subject = ent["text"]
            self.active_entity_label = ent.get(
                "label",
                "UNKNOWN",
            )

            for item in parse.entities:
                self._remember_entity(
                    item["text"],
                    item.get("label"),
                )
            return self.active_subject

        pronouns = {
            token["text"].lower()
            for token in parse.tokens
            if token["text"]
        }

        if pronouns & self.PRONOUNS:
            return self.active_subject

        if parse.subjects:
            self.active_subject = parse.subjects[0]
            return self.active_subject

        # Noun chunks provide a better fallback than raw token position.
        if parse.noun_chunks:
            candidate = parse.noun_chunks[0]
            if candidate.lower() not in {
                "who",
                "what",
                "where",
                "when",
                "which",
                "how",
            }:
                self.active_subject = candidate
                return self.active_subject

        return self.active_subject

    def remember(self, text, goal, result=None, parse=None):
        if goal and goal.subject:
            self.active_subject = goal.subject

        entry = {
            "text": text,
            "goal": (
                asdict(goal)
                if goal
                else None
            ),
            "result": (
                result
                if result is not None
                else None
            ),
            "active_subject": self.active_subject,
            "entity_label": self.active_entity_label,
        }

        self.turns.append(entry)
        self.turns = self.turns[-64:]

        if goal and goal.relation:
            self.goal_memory[
                goal.relation
            ][
                goal.intent
            ] += 1

        if goal and goal.relation and result:
            if result.get("success"):
                self.relation_memory[
                    goal.relation
                ]["success"] += 1
            else:
                self.relation_memory[
                    goal.relation
                ]["miss"] += 1

        self.save()

    def learn_relation(
        self,
        goal,
        relation,
        strength=1.0,
    ):
        if not goal or not relation:
            return

        self.relation_memory[
            goal
        ][relation] += float(strength)

        self.save()

    def candidate_relation_bias(
        self,
        goal,
        relation,
    ):
        counter = self.relation_memory.get(
            goal,
            Counter(),
        )

        if not counter:
            return 0.0

        total = sum(
            counter.values()
        )
        if total <= 0:
            return 0.0

        return (
            counter.get(
                relation,
                0.0,
            )
            / total
        )

    def save(self):
        if not self.path:
            return

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "active_subject": self.active_subject,
            "active_entity_label": (
                self.active_entity_label
            ),
            "entities": self.entities,
            "turns": self.turns,
            "goal_memory": {
                key: dict(value)
                for key, value
                in self.goal_memory.items()
            },
            "relation_memory": {
                key: dict(value)
                for key, value
                in self.relation_memory.items()
            },
        }

        tmp = self.path.with_suffix(
            self.path.suffix + ".tmp"
        )
        tmp.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def load(self):
        try:
            payload = json.loads(
                self.path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            return

        self.active_subject = payload.get(
            "active_subject"
        )
        self.active_entity_label = payload.get(
            "active_entity_label"
        )
        self.entities = list(
            payload.get(
                "entities",
                [],
            )
        )
        self.turns = list(
            payload.get(
                "turns",
                [],
            )
        )[-64:]

        for key, value in payload.get(
            "goal_memory",
            {},
        ).items():
            self.goal_memory[key].update(
                value
            )

        for key, value in payload.get(
            "relation_memory",
            {},
        ).items():
            self.relation_memory[key].update(
                value
            )



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


def _intent_hypotheses(parse, max_n=8):
    tokens = parse.tokens
    pos = {t["pos"] for t in tokens}
    deps = {t["dep"] for t in tokens}
    has_entity = bool(parse.entities)
    has_subject = bool(parse.subjects)
    has_object = bool(parse.objects)
    has_question = parse.question != "DECLARATIVE" or parse.text.strip().endswith("?")

    scores = {
        "conversation": 0.0,
        "relation_lookup": 0.0,
        "entity_profile": 0.0,
        "identity_lookup": 0.0,
        "statement": 0.0,
    }

    if "INTJ" in pos:
        scores["conversation"] += 0.45
    if has_question:
        scores["relation_lookup"] += 0.30
        scores["identity_lookup"] += 0.20
    if has_entity:
        scores["entity_profile"] += 0.25
        scores["identity_lookup"] += 0.15
    if has_subject:
        scores["relation_lookup"] += 0.10
    if has_object:
        scores["relation_lookup"] += 0.10
    if parse.root_lemma.lower() in {"be", "seem", "become"}:
        scores["identity_lookup"] += 0.08
    if not has_entity and not has_subject and not has_object:
        scores["conversation"] += 0.25

    total = sum(max(v, 0.0) for v in scores.values()) or 1.0
    ranked = sorted(
        ((v / total, k) for k, v in scores.items() if v > 0.0),
        key=lambda x: (-x[0], x[1]),
    )

    return [
        {
            "intent": intent,
            "score": score,
            "evidence": {
                "root": parse.root_lemma,
                "question": parse.question,
                "has_entity": has_entity,
                "has_subject": has_subject,
                "has_object": has_object,
                "has_question_mark": has_question,
                "pos": sorted(pos),
                "deps": sorted(deps),
            },
        }
        for score, intent in ranked[:max_n]
    ]


def _relation_local_score(relation, parse):
    relation_norm = norm(relation)
    lexical = {t["text"].lower() for t in parse.tokens}
    lexical.update(t["lemma"].lower() for t in parse.tokens)
    score = 0.0
    for word in lexical:
        if not word:
            continue
        if word in relation_norm:
            score += 0.08
        for part in relation_norm.split():
            if len(word) >= 4 and len(part) >= 4 and (word in part or part in word):
                score += 0.025
    return score


def hypotheses(parse, ctx, vocab, max_n=12, graph=None, memory=None):
    if isinstance(vocab, argparse.Namespace):
        raise TypeError("hypotheses() expected relation vocabulary, not argparse.Namespace")
    if vocab is None:
        raise TypeError("hypotheses() requires a relation vocabulary")

    vocab = tuple(str(x) for x in vocab)
    subject = ctx.subject(parse)
    intents = _intent_hypotheses(parse, min(8, max_n))
    out = []

    for intent_item in intents:
        intent = intent_item["intent"]

        # Intent-only hypotheses are valid. They never get turned into an
        # arbitrary ontology relation.
        if intent in {"conversation", "statement"}:
            out.append(
                Hypothesis(
                    subject=subject,
                    relation="",
                    intent=intent,
                    lexical_score=intent_item["score"],
                    evidence={
                        "intent_score": intent_item["score"],
                        **intent_item["evidence"],
                    },
                )
            )
            continue

        candidate_relations = set(vocab)

        # Graph-local expansion makes the semantic vocabulary much larger than
        # a tiny sampled global list, while remaining bounded.
        if graph is not None and subject:
            try:
                local_edges = graph.outgoing(
                    subject,
                    max(80, min(160, len(vocab) * 4)),
                )
                candidate_relations.update(
                    edge.relation
                    for edge in local_edges
                )
            except Exception:
                pass

        for relation in candidate_relations:
            lexical = _relation_local_score(
                relation,
                parse,
            )

            memory_score = (
                memory.candidate_relation_bias(
                    relation,
                    relation,
                )
                if memory is not None
                else 0.0
            )

            score = (
                intent_item["score"]
                + lexical
                + 0.50 * memory_score
            )

            out.append(
                Hypothesis(
                    subject=subject,
                    relation=relation,
                    intent=intent,
                    lexical_score=score,
                    evidence={
                        "intent_score": intent_item["score"],
                        "intent_evidence": intent_item["evidence"],
                        "relation_score": lexical,
                        "memory_score": memory_score,
                    },
                )
            )

    out.sort(
        key=lambda h: (
            -h.lexical_score,
            h.intent,
            h.relation,
        )
    )

    # Keep a mixture of intent families so one weak heuristic cannot crowd out
    # every alternative.
    selected = []
    counts = Counter()

    for h in out:
        if len(selected) >= max_n:
            break
        if counts[h.intent] >= max(1, max_n // 3):
            continue
        selected.append(h)
        counts[h.intent] += 1

    if len(selected) < max_n:
        used = {
            (h.intent, h.relation, h.subject)
            for h in selected
        }
        for h in out:
            key = (h.intent, h.relation, h.subject)
            if key in used:
                continue
            selected.append(h)
            used.add(key)
            if len(selected) >= max_n:
                break

    return selected


def rank_goal_hypotheses(hypotheses_list, results):
    ranked = []
    for h, result in zip(hypotheses_list, results):
        score = h.lexical_score
        if h.relation:
            score += 2.0 if result.get("success") else 0.0
            score += min(
                1.0,
                float(result.get("attention", 0)) / 4.0,
            )
        ranked.append((score, h, result))

    ranked.sort(
        key=lambda x: (
            -x[0],
            x[1].intent,
            x[1].relation,
        )
    )
    return ranked


def resolve_hypothesis(graph, att, h, args, seed=0):
    if not h.relation:
        return {
            "success": h.intent == "conversation",
            "intent_only": True,
            "steps": 0,
            "path": [],
            "target": None,
            "attention": 0,
            "exploration": 0,
        }

    return search(
        graph,
        att,
        h,
        args,
        seed,
    )



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


def trace_result(text, parse, hs, selected, search_result, ctx):
    return {
        "timestamp": time.time(),
        "text": text,
        "parse": asdict(parse),
        "intent_hypotheses": _intent_hypotheses(
            parse,
            max_n=8,
        ),
        "hypotheses": [
            asdict(h) for h in hs
        ],
        "selected_goal": (
            asdict(selected)
            if selected else None
        ),
        "search": search_result,
        "context": {
            "active_subject": ctx.active_subject,
            "turns": len(ctx.turns),
            "known_entities": len(ctx.entities),
        },
    }


def _learn_from_resolution(selected, result, memory):
    if not selected or not selected.relation:
        return
    if result.get("success"):
        memory.learn_relation(
            selected.relation,
            selected.relation,
            strength=1.0,
        )
    else:
        memory.learn_relation(
            selected.relation,
            "__miss__",
            strength=0.05,
        )


def smoke(graph, parser, att, vocab, args, memory):
    ctx = Context(path=memory.path)
    inputs = [
        "hello",
        "who was Albert Einstein?",
        "Where was he born?",
        "What was his nationality?",
    ]
    traces = []

    print("=== V609 SMOKE ===", flush=True)

    for index, text in enumerate(inputs, 1):
        parse = parser.parse(text)
        hs = hypotheses(
            parse=parse,
            ctx=ctx,
            vocab=vocab,
            max_n=args.max_hypotheses,
            graph=graph,
            memory=memory,
        )

        results = [
            resolve_hypothesis(
                graph,
                att,
                h,
                args,
                args.seed + index * 100 + offset,
            )
            for offset, h in enumerate(hs)
        ]

        ranked = rank_goal_hypotheses(
            hs,
            results,
        )

        selected = ranked[0][1] if ranked else None
        result = (
            ranked[0][2]
            if ranked
            else {
                "success": False,
                "steps": 0,
                "path": [],
            }
        )

        if selected:
            _learn_from_resolution(
                selected,
                result,
                memory,
            )

        ctx.remember(
            text,
            selected,
            result,
            parse,
        )

        trace = trace_result(
            text,
            parse,
            hs,
            selected,
            result,
            ctx,
        )

        traces.append(trace)

        print(
            f"[SMOKE {index}/{len(inputs)}] {text}",
            flush=True,
        )
        print(
            f"  selected_intent="
            f"{selected.intent if selected else None} "
            f"subject="
            f"{selected.subject if selected else None!r} "
            f"relation="
            f"{selected.relation if selected else None!r}",
            flush=True,
        )
        print(
            f"  alternatives={len(hs)} "
            f"success={result.get('success', False)} "
            f"steps={result.get('steps', 0)}",
            flush=True,
        )
        print(
            "  intents:",
            ", ".join(
                f"{x['intent']}={x['score']:.3f}"
                for x in trace["intent_hypotheses"]
            ),
            flush=True,
        )

    return traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--trace-output", default="")
    ap.add_argument("--prior-output", default="")
    ap.add_argument("--memory-output", default="")
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    ap.add_argument(
        "--mode",
        choices=("chat", "smoke"),
        default="chat",
    )
    ap.add_argument("--max-hypotheses", type=int, default=12)
    ap.add_argument("--relation-vocabulary", type=int, default=200)
    ap.add_argument("--goal-budget", type=int, default=40)
    ap.add_argument("--budget", type=int, default=80)
    ap.add_argument("--per-node", type=int, default=60)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--cache-entries", type=int, default=12000)
    ap.add_argument("--prior-decay", type=float, default=.65)
    ap.add_argument("--seed", type=int, default=60900)
    ap.add_argument("--progress-every", type=int, default=1)
    args = ap.parse_args()

    db = Path(args.database).resolve()
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    base = out.with_suffix("")
    trace_path = Path(
        args.trace_output
        or f"{base}_traces.jsonl"
    ).resolve()
    prior_path = Path(
        args.prior_output
        or f"{base}_prior.json"
    ).resolve()
    memory_path = Path(
        args.memory_output
        or f"{base}_memory.json"
    ).resolve()

    print(
        "=== V609 SEMANTIC CHAT GATEWAY ===",
        flush=True,
    )
    print(
        f"database              : {db}",
        flush=True,
    )
    print(
        "previous artifacts    : NONE",
        flush=True,
    )
    print(
        f"spaCy model           : {args.spacy_model}",
        flush=True,
    )
    print(
        "intent model           : structural hypotheses",
        flush=True,
    )
    print(
        f"persistent memory     : {memory_path}",
        flush=True,
    )

    graph = Graph(
        db,
        args.cache_entries,
    )

    try:
        print(
            "[GATEWAY 1/4] loading spaCy...",
            flush=True,
        )
        parser = SpaCyParser(
            args.spacy_model
        )

        print(
            "[GATEWAY 2/4] relation vocabulary...",
            flush=True,
        )
        vocab = graph.relation_vocab(
            limit=args.relation_vocabulary
        )

        if not isinstance(
            vocab,
            (list, tuple),
        ):
            raise RuntimeError(
                "relation vocabulary contract violated"
            )

        if not all(
            isinstance(x, str)
            for x in vocab
        ):
            raise RuntimeError(
                "relation vocabulary contains non-string value"
            )

        print(
            f"    relations={len(vocab)}",
            flush=True,
        )

        print(
            "[GATEWAY 3/4] global conditional attention...",
            flush=True,
        )
        att = Attention(
            args.prior_decay
        )

        print(
            "[GATEWAY 4/4] persistent semantic memory...",
            flush=True,
        )
        memory = Context(
            path=memory_path
        )

        print(
            f"    known_entities={len(memory.entities)} "
            f"known_turns={len(memory.turns)}",
            flush=True,
        )

        traces = []

        if args.mode == "smoke":
            traces = smoke(
                graph,
                parser,
                att,
                vocab,
                args,
                memory,
            )
        else:
            ctx = memory
            print(
                "=== V609 CHAT (type 'exit') ===",
                flush=True,
            )

            while True:
                try:
                    text = input(
                        "chat> "
                    ).strip()
                except (
                    EOFError,
                    KeyboardInterrupt,
                ):
                    print()
                    break

                if not text:
                    continue

                if text.lower() in {
                    "exit",
                    "quit",
                }:
                    break

                parse = parser.parse(
                    text
                )

                hs = hypotheses(
                    parse=parse,
                    ctx=ctx,
                    vocab=vocab,
                    max_n=args.max_hypotheses,
                    graph=graph,
                    memory=memory,
                )

                results = [
                    resolve_hypothesis(
                        graph,
                        att,
                        h,
                        args,
                        args.seed + i,
                    )
                    for i, h in enumerate(hs)
                ]

                ranked = rank_goal_hypotheses(
                    hs,
                    results,
                )

                selected = (
                    ranked[0][1]
                    if ranked else None
                )

                result = (
                    ranked[0][2]
                    if ranked
                    else {
                        "success": False,
                        "steps": 0,
                        "path": [],
                    }
                )

                if selected:
                    _learn_from_resolution(
                        selected,
                        result,
                        memory,
                    )

                ctx.remember(
                    text,
                    selected,
                    result,
                    parse,
                )

                trace = trace_result(
                    text,
                    parse,
                    hs,
                    selected,
                    result,
                    ctx,
                )
                traces.append(trace)

                print(
                    "intent:",
                    selected.intent
                    if selected else None,
                    flush=True,
                )
                print(
                    "goal:",
                    asdict(selected)
                    if selected else None,
                    flush=True,
                )
                print(
                    "evidence:",
                    result,
                    flush=True,
                )
                print(
                    "alternatives:",
                    len(hs),
                    flush=True,
                )
                print(
                    "intent hypotheses:",
                    trace["intent_hypotheses"],
                    flush=True,
                )

        if traces:
            trace_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            with trace_path.open(
                "w",
                encoding="utf-8",
            ) as handle:
                for trace in traces:
                    handle.write(
                        json.dumps(
                            trace,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        prior_path.write_text(
            json.dumps(
                att.export(),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        memory.save()

        payload = {
            "version": "V609",
            "architecture": {
                "parser": "spaCy structural parse",
                "intent": "structural intent hypotheses",
                "goal_generation": (
                    "generic graph-aware alternatives"
                ),
                "resolver": (
                    "global conditional attention + bounded BFS"
                ),
                "memory": (
                    "persistent entity-aware context"
                ),
            },
            "config": vars(args),
            "relation_vocabulary": vocab,
            "traces": len(traces),
            "attention": att.export(),
            "memory": {
                "path": str(memory_path),
                "entities": len(memory.entities),
                "turns": len(memory.turns),
            },
            "trace_output": str(trace_path),
            "prior_output": str(prior_path),
        }

        out.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print(
            "=== V609 COMPLETE ===",
            flush=True,
        )
        print(
            f"mode                  : {args.mode}",
            flush=True,
        )
        print(
            f"traces                : {len(traces)}",
            flush=True,
        )
        print(
            f"memory entities       : {len(memory.entities)}",
            flush=True,
        )
        print(
            f"memory turns          : {len(memory.turns)}",
            flush=True,
        )
        print(
            f"attention updates     : {att.updates}",
            flush=True,
        )
        print(
            f"JSON                  : {out}",
            flush=True,
        )
        print(
            f"TRACE                 : {trace_path}",
            flush=True,
        )
        print(
            f"PRIOR                 : {prior_path}",
            flush=True,
        )
        print(
            f"MEMORY                : {memory_path}",
            flush=True,
        )
    finally:
        graph.close()


if __name__ == "__main__":
    main()
