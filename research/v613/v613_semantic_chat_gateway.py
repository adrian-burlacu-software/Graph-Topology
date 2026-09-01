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
    Persistent episodic + semantic memory.

    Memory is deliberately generic:
      - entities and aliases;
      - conversation turns;
      - structural signatures -> intent outcomes;
      - structural signatures + context -> successful relation goals;
      - relation/path traces.

    No ontology-specific language->relation dictionary is hardcoded.
    """

    PRONOUNS = {
        "he", "she", "they", "him", "her", "them",
        "his", "their", "its", "it",
    }

    def __init__(self, path=None):
        self.path = path
        self.active_subject = None
        self.active_entity_label = None
        self.entities = {}
        self.turns = []
        self.intent_memory = defaultdict(Counter)
        self.goal_memory = defaultdict(Counter)
        self.path_memory = defaultdict(Counter)
        self.relation_outcomes = defaultdict(Counter)

        if path and path.exists():
            self.load()

    @staticmethod
    def structural_key(parse):
        lemmas = tuple(
            token["lemma"].lower()
            for token in parse.tokens
            if token["lemma"]
        )
        pos = tuple(
            token["pos"]
            for token in parse.tokens
        )
        deps = tuple(
            token["dep"]
            for token in parse.tokens
        )

        return json.dumps(
            {
                "question": parse.question,
                "root": parse.root_lemma.lower(),
                "lemmas": lemmas[:24],
                "pos": pos[:24],
                "deps": deps[:24],
                "entity_labels": tuple(
                    entity["label"]
                    for entity in parse.entities
                ),
            },
            sort_keys=True,
        )

    def subject(self, parse):
        if parse.entities:
            entity = parse.entities[0]
            self.active_subject = entity["text"]
            self.active_entity_label = entity.get(
                "label"
            )

            for item in parse.entities:
                key = item["text"].lower()
                record = self.entities.setdefault(
                    key,
                    {
                        "text": item["text"],
                        "label": item.get(
                            "label",
                            "UNKNOWN",
                        ),
                        "mentions": 0,
                    },
                )
                record["mentions"] += 1

            return self.active_subject

        tokens = {
            token["text"].lower()
            for token in parse.tokens
        }

        if tokens & self.PRONOUNS:
            return self.active_subject

        if parse.subjects:
            self.active_subject = parse.subjects[0]
            return self.active_subject

        if parse.noun_chunks:
            candidate = parse.noun_chunks[0]
            if candidate.lower() not in {
                "who", "what", "where",
                "when", "which", "how",
            }:
                self.active_subject = candidate
                return candidate

        return self.active_subject

    def record_turn(
        self,
        text,
        parse,
        selected,
        search_result,
        hypotheses_list,
    ):
        structural = self.structural_key(parse)

        if selected:
            self.intent_memory[
                structural
            ][selected.intent] += 1

            if selected.relation:
                self.goal_memory[
                    (
                        structural,
                        selected.intent,
                    )
                ][selected.relation] += (
                    1
                    if search_result.get(
                        "success",
                        False,
                    )
                    else 0.05
                )

                self.relation_outcomes[
                    selected.relation
                ][
                    "success"
                    if search_result.get(
                        "success",
                        False,
                    )
                    else "miss"
                ] += 1

        turn = {
            "timestamp": time.time(),
            "text": text,
            "structural_key": structural,
            "selected": (
                asdict(selected)
                if selected
                else None
            ),
            "search": search_result,
            "hypotheses": [
                asdict(item)
                for item in hypotheses_list
            ],
            "active_subject": self.active_subject,
        }

        self.turns.append(turn)
        self.turns = self.turns[-256:]

        self.save()

    def intent_bias(self, parse, intent):
        key = self.structural_key(parse)
        counter = self.intent_memory.get(
            key,
            Counter(),
        )

        if not counter:
            return 0.0

        total = sum(counter.values())
        return (
            counter.get(intent, 0.0)
            / max(total, 1e-9)
        )

    def goal_bias(
        self,
        parse,
        intent,
        relation,
    ):
        key = self.structural_key(parse)

        counter = self.goal_memory.get(
            (
                key,
                intent,
            ),
            Counter(),
        )

        if not counter:
            return 0.0

        total = sum(counter.values())
        return (
            counter.get(
                relation,
                0.0,
            )
            / max(total, 1e-9)
        )

    def candidate_relation_bias(
        self,
        goal,
        relation,
    ):
        counter = self.relation_outcomes.get(
            goal,
            Counter(),
        )

        success = counter.get(
            "success",
            0.0,
        )
        miss = counter.get(
            "miss",
            0.0,
        )

        if (
            success <= 0
            and miss <= 0
        ):
            return 0.0

        return (
            success
            / (
                success
                + miss
                + 1.0
            )
        )

    def remember_path(
        self,
        goal,
        path,
        successful,
        strength=1.0,
    ):
        if not path:
            return

        key = (
            goal,
            tuple(path),
        )

        self.path_memory[key][
            "success"
            if successful
            else "miss"
        ] += strength

        self.save()

    def save(self):
        if not self.path:
            return

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "version": "semantic-memory-v613",
            "active_subject": self.active_subject,
            "active_entity_label": (
                self.active_entity_label
            ),
            "entities": list(
                self.entities.values()
            ),
            "turns": self.turns,
            "intent_memory": {
                key: dict(value)
                for key, value
                in self.intent_memory.items()
            },
            "goal_memory": {
                json.dumps(key): dict(value)
                for key, value
                in self.goal_memory.items()
            },
            "path_memory": {
                json.dumps(key): dict(value)
                for key, value
                in self.path_memory.items()
            },
            "relation_outcomes": {
                key: dict(value)
                for key, value
                in self.relation_outcomes.items()
            },
        }

        temporary = self.path.with_suffix(
            self.path.suffix + ".tmp"
        )

        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        temporary.replace(
            self.path
        )

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

        self.entities = {
            item["text"].lower(): item
            for item in payload.get(
                "entities",
                [],
            )
        }

        self.turns = list(
            payload.get(
                "turns",
                [],
            )
        )[-256:]

        for key, value in payload.get(
            "intent_memory",
            {},
        ).items():
            self.intent_memory[key].update(
                value
            )

        for encoded, value in payload.get(
            "goal_memory",
            {},
        ).items():
            try:
                key = tuple(
                    json.loads(
                        encoded
                    )
                )
                self.goal_memory[key].update(
                    value
                )
            except Exception:
                continue

        for encoded, value in payload.get(
            "path_memory",
            {},
        ).items():
            try:
                key_json = json.loads(
                    encoded
                )
                key = (
                    key_json[0],
                    tuple(key_json[1]),
                )
                self.path_memory[key].update(
                    value
                )
            except Exception:
                continue

        for key, value in payload.get(
            "relation_outcomes",
            {},
        ).items():
            self.relation_outcomes[key].update(
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



def _intent_hypotheses(
    parse,
    ctx,
    max_n=8,
):
    tokens = parse.tokens

    if not tokens:
        return []

    pos = {
        token["pos"]
        for token in tokens
    }

    has_entity = bool(
        parse.entities
    )
    has_subject = bool(
        parse.subjects
    )
    has_object = bool(
        parse.objects
    )
    has_question = (
        parse.question
        != "DECLARATIVE"
        or parse.text.strip().endswith("?")
    )

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

    if not (
        has_entity
        or has_subject
        or has_object
    ):
        scores["conversation"] += 0.25

    # Learned intent preference.
    for intent in tuple(scores):
        scores[intent] += (
            0.50
            * ctx.intent_bias(
                parse,
                intent,
            )
        )

    total = sum(
        max(
            0.0,
            value,
        )
        for value in scores.values()
    ) or 1.0

    ranked = sorted(
        (
            (
                value / total,
                intent,
            )
            for intent, value
            in scores.items()
            if value > 0
        ),
        key=lambda x: (
            -x[0],
            x[1],
        ),
    )

    return [
        {
            "intent": intent,
            "score": score,
            "evidence": {
                "question": parse.question,
                "root": parse.root_lemma,
                "has_entity": has_entity,
                "has_subject": has_subject,
                "has_object": has_object,
                "has_question_mark": has_question,
                "pos": sorted(pos),
                "learned_intent_bias": ctx.intent_bias(
                    parse,
                    intent,
                ),
            },
        }
        for score, intent in ranked[:max_n]
    ]


def _relation_local_score(
    relation,
    parse,
):
    relation_norm = norm(relation)
    lexical = {
        token["text"].lower()
        for token in parse.tokens
    }
    lexical.update(
        token["lemma"].lower()
        for token in parse.tokens
    )

    score = 0.0

    for word in lexical:
        if not word:
            continue

        if word in relation_norm:
            score += 0.08

        for part in relation_norm.split():
            if (
                len(word) >= 4
                and len(part) >= 4
                and (
                    word in part
                    or part in word
                )
            ):
                score += 0.025

    return score


def hypotheses(
    parse,
    ctx,
    vocab,
    max_n=12,
    graph=None,
):
    if isinstance(
        vocab,
        argparse.Namespace,
    ):
        raise TypeError(
            "hypotheses() expected relation vocabulary, "
            "not argparse.Namespace"
        )

    if vocab is None:
        raise TypeError(
            "hypotheses() requires a relation vocabulary"
        )

    vocab = tuple(
        str(x)
        for x in vocab
    )

    subject = ctx.subject(
        parse
    )

    intents = _intent_hypotheses(
        parse,
        ctx,
        min(8, max_n),
    )

    candidates = set(
        vocab
    )

    # The graph gives us actual relation candidates around the active entity.
    # This is bounded and generic; no relation semantics are hardcoded.
    if graph is not None and subject:
        try:
            local_edges = graph.outgoing(
                subject,
                max(
                    80,
                    min(
                        200,
                        len(vocab) * 5,
                    ),
                ),
            )
            candidates.update(
                edge.relation
                for edge in local_edges
            )
        except Exception:
            pass

    output = []

    for intent_item in intents:
        intent = intent_item["intent"]

        if intent in {
            "conversation",
            "statement",
        }:
            output.append(
                Hypothesis(
                    subject=subject,
                    relation="",
                    intent=intent,
                    lexical_score=(
                        intent_item["score"]
                    ),
                    evidence={
                        "intent_score": (
                            intent_item["score"]
                        ),
                        **intent_item["evidence"],
                    },
                )
            )
            continue

        for relation in candidates:
            lexical = _relation_local_score(
                relation,
                parse,
            )

            goal_memory_bias = (
                ctx.goal_bias(
                    parse,
                    intent,
                    relation,
                )
            )

            relation_outcome_bias = (
                ctx.candidate_relation_bias(
                    relation,
                    relation,
                )
            )

            score = (
                intent_item["score"]
                + lexical
                + 0.85 * goal_memory_bias
                + 0.20 * relation_outcome_bias
            )

            output.append(
                Hypothesis(
                    subject=subject,
                    relation=relation,
                    intent=intent,
                    lexical_score=score,
                    evidence={
                        "intent_score": (
                            intent_item["score"]
                        ),
                        "intent_evidence": (
                            intent_item["evidence"]
                        ),
                        "relation_score": lexical,
                        "goal_memory_bias": (
                            goal_memory_bias
                        ),
                        "relation_outcome_bias": (
                            relation_outcome_bias
                        ),
                    },
                )
            )

    output.sort(
        key=lambda h: (
            -h.lexical_score,
            h.intent,
            h.relation,
        )
    )

    selected = []
    per_intent = Counter()

    # Preserve diversity of interpretations.
    for hypothesis in output:
        if len(selected) >= max_n:
            break

        if (
            hypothesis.relation == ""
            or per_intent[
                hypothesis.intent
            ] < max(
                1,
                max_n // 3,
            )
        ):
            selected.append(
                hypothesis
            )
            per_intent[
                hypothesis.intent
            ] += 1

    if len(selected) < max_n:
        used = {
            (
                item.intent,
                item.relation,
                item.subject,
            )
            for item in selected
        }

        for hypothesis in output:
            key = (
                hypothesis.intent,
                hypothesis.relation,
                hypothesis.subject,
            )

            if key in used:
                continue

            selected.append(
                hypothesis
            )
            used.add(key)

            if len(selected) >= max_n:
                break

    return selected




def rank_goal_hypotheses(
    hypotheses_list,
    results,
):
    ranked = []

    for hypothesis, result in zip(
        hypotheses_list,
        results,
    ):
        score = hypothesis.lexical_score

        if hypothesis.relation:
            if result.get(
                "success",
                False,
            ):
                score += 2.0

            score += min(
                1.0,
                float(
                    result.get(
                        "attention",
                        0,
                    )
                ) / 4.0,
            )

        ranked.append(
            (
                score,
                hypothesis,
                result,
            )
        )

    ranked.sort(
        key=lambda item: (
            -item[0],
            item[1].intent,
            item[1].relation,
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
    # V613: keep the trace serializer on the current context-aware API.
    return {
        "timestamp": time.time(),
        "text": text,
        "parse": asdict(parse),
        "intent_hypotheses": _intent_hypotheses(
            parse,
            ctx,
            max_n=8,
        ),
        "hypotheses": [
            asdict(h)
            for h in hs
        ],
        "selected_goal": (
            asdict(selected)
            if selected
            else None
        ),
        "search": search_result,
        "context": {
            "active_subject": ctx.active_subject,
            "turns": len(ctx.turns),
            "known_entities": len(ctx.entities),
        },
    }


def _learn_from_resolution(
    selected,
    result,
    memory,
    attention,
):
    """Consolidate one semantic decision into persistent memory + attention."""
    if not selected:
        return

    relation = selected.relation
    if not relation:
        return

    success = bool(
        result.get("success", False)
    )

    memory.relation_outcomes[
        relation
    ][
        "success" if success else "miss"
    ] += (
        1.0 if success else 0.05
    )

    path = tuple(
        result.get("path", [])
    )

    if success and path:
        memory.remember_path(
            relation,
            path,
            True,
            strength=1.0,
        )

        prefix = ()
        for next_relation in path:
            attention.update(
                relation,
                prefix,
                next_relation,
                strength=1.0,
            )
            prefix = (
                prefix
                + (next_relation,)
            )


def smoke(graph, parser, att, vocab, args, memory):
    ctx = memory

    inputs = [
        "hello",
        "who was Albert Einstein?",
        "Where was he born?",
        "What was his nationality?",
        "Who was Albert Einstein?",
        "Where was he born?",
    ]

    traces = []

    print(
        "=== V613 SMOKE ===",
        flush=True,
    )

    for index, text in enumerate(
        inputs,
        1,
    ):
        parse = parser.parse(
            text
        )

        hs = hypotheses(
            parse=parse,
            ctx=ctx,
            vocab=vocab,
            max_n=args.max_hypotheses,
            graph=graph,
        )

        results = [
            resolve_hypothesis(
                graph,
                att,
                h,
                args,
                args.seed
                + index * 100
                + offset,
            )
            for offset, h
            in enumerate(hs)
        ]

        ranked = rank_goal_hypotheses(
            hs,
            results,
        )

        selected = (
            ranked[0][1]
            if ranked
            else None
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
                att,
            )

        ctx.record_turn(
            text,
            parse,
            selected,
            result,
            hs,
        )

        trace = trace_result(
            text,
            parse,
            hs,
            selected,
            result,
            ctx,
        )

        traces.append(
            trace
        )

        print(
            f"[SMOKE {index}/{len(inputs)}] "
            f"{text}",
            flush=True,
        )
        print(
            f"  intent={selected.intent if selected else None} "
            f"subject={selected.subject if selected else None!r} "
            f"relation={selected.relation if selected else None!r}",
            flush=True,
        )
        print(
            f"  alternatives={len(hs)} "
            f"success={result.get('success', False)} "
            f"steps={result.get('steps', 0)}",
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
    ap.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
    )
    ap.add_argument(
        "--mode",
        choices=("chat", "smoke"),
        default="chat",
    )
    ap.add_argument(
        "--max-hypotheses",
        type=int,
        default=12,
    )
    ap.add_argument(
        "--relation-vocabulary",
        type=int,
        default=200,
    )
    ap.add_argument(
        "--goal-budget",
        type=int,
        default=40,
    )
    ap.add_argument(
        "--budget",
        type=int,
        default=80,
    )
    ap.add_argument(
        "--per-node",
        type=int,
        default=60,
    )
    ap.add_argument(
        "--max-depth",
        type=int,
        default=3,
    )
    ap.add_argument(
        "--cache-entries",
        type=int,
        default=12000,
    )
    ap.add_argument(
        "--prior-decay",
        type=float,
        default=.65,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=61000,
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=1,
    )

    args = ap.parse_args()

    db = Path(
        args.database
    ).resolve()
    output = Path(
        args.output
    ).resolve()
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    base = output.with_suffix("")

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
        "=== V613 SEMANTIC CHAT GATEWAY ===",
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
        "intent model           : structural + learned memory",
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
            f"known_turns={len(memory.turns)} "
            f"learned_goal_patterns={len(memory.goal_memory)}",
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
                "=== V613 CHAT (type 'exit') ===",
                flush=True,
            )

            while True:
                try:
                    user_text = input(
                        "chat> "
                    ).strip()
                except (
                    EOFError,
                    KeyboardInterrupt,
                ):
                    print()
                    break

                if not user_text:
                    continue

                if user_text.lower() in {
                    "exit",
                    "quit",
                }:
                    break

                parse = parser.parse(
                    user_text
                )

                hs = hypotheses(
                    parse=parse,
                    ctx=ctx,
                    vocab=vocab,
                    max_n=args.max_hypotheses,
                    graph=graph,
                )

                results = [
                    resolve_hypothesis(
                        graph,
                        att,
                        h,
                        args,
                        args.seed + index,
                    )
                    for index, h
                    in enumerate(hs)
                ]

                ranked = rank_goal_hypotheses(
                    hs,
                    results,
                )

                selected = (
                    ranked[0][1]
                    if ranked
                    else None
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

                    if result.get(
                        "success",
                        False,
                    ):
                        memory.remember_path(
                            selected.relation,
                            result.get(
                                "path",
                                [],
                            ),
                            True,
                            strength=1.0,
                        )

                ctx.record_turn(
                    user_text,
                    parse,
                    selected,
                    result,
                    hs,
                )

                trace = trace_result(
                    user_text,
                    parse,
                    hs,
                    selected,
                    result,
                    ctx,
                )

                traces.append(
                    trace
                )

                print(
                    "intent:",
                    selected.intent
                    if selected
                    else None,
                    flush=True,
                )
                print(
                    "goal:",
                    asdict(selected)
                    if selected
                    else None,
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
                    "active subject:",
                    ctx.active_subject,
                    flush=True,
                )
                print(
                    "memory turns:",
                    len(ctx.turns),
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
            "version": "V613",
            "architecture": {
                "parser": "spaCy structural parse",
                "intent": (
                    "structural intent hypotheses + learned intent memory"
                ),
                "goal_generation": (
                    "graph-local + global generic relation candidates "
                    "+ learned goal memory"
                ),
                "resolver": (
                    "global conditional attention + bounded BFS"
                ),
                "memory": (
                    "persistent episodic/entity/goal/path memory"
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
                "intent_patterns": len(
                    memory.intent_memory
                ),
                "goal_patterns": len(
                    memory.goal_memory
                ),
                "path_patterns": len(
                    memory.path_memory
                ),
            },
            "trace_output": str(
                trace_path
            ),
            "prior_output": str(
                prior_path
            ),
        }

        output.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print(
            "=== V613 COMPLETE ===",
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
            f"learned intent patterns: "
            f"{len(memory.intent_memory)}",
            flush=True,
        )
        print(
            f"learned goal patterns  : "
            f"{len(memory.goal_memory)}",
            flush=True,
        )
        print(
            f"learned path patterns  : "
            f"{len(memory.path_memory)}",
            flush=True,
        )
        print(
            f"JSON                  : {output}",
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
