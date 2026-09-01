from __future__ import annotations

import argparse
import heapq
import json
import random
import re
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

    @staticmethod
    def _norm_entity(value):
        return " ".join(
            str(value).strip().lower().split()
        )

    def resolve_entity(
        self,
        mention,
        limit=24,
    ):
        """
        Exact-first graph entity resolution.

        Approximate substring hits are returned only for diagnostics and have
        accepted=False. They must never become the canonical search subject.
        """
        if mention is None:
            return []

        mention=str(mention).strip()
        if not mention:
            return []

        norm=self._norm_entity(mention)
        candidates=[]
        seen=set()

        for row in self.conn.execute(
            f"""
            SELECT {self.sc} AS node
            FROM edges
            WHERE lower({self.sc})=?
            LIMIT ?
            """,
            (norm,limit),
        ).fetchall():
            node=row["node"]
            if node is not None and node not in seen:
                seen.add(node)
                candidates.append({
                    "node":str(node),
                    "kind":"subject_exact",
                    "score":1.0,
                    "accepted":True,
                })

        for row in self.conn.execute(
            f"""
            SELECT DISTINCT {self.oc} AS node
            FROM edges
            WHERE lower({self.oc})=?
            LIMIT ?
            """,
            (norm,limit),
        ).fetchall():
            node=row["node"]
            if node is not None and node not in seen:
                seen.add(node)
                candidates.append({
                    "node":str(node),
                    "kind":"object_exact",
                    "score":0.95,
                    "accepted":True,
                })

        if candidates:
            return candidates[:limit]

        pattern=f"%{mention}%"

        for row in self.conn.execute(
            f"""
            SELECT DISTINCT {self.sc} AS node
            FROM edges
            WHERE {self.sc} LIKE ? COLLATE NOCASE
            LIMIT ?
            """,
            (pattern,limit),
        ).fetchall():
            node=row["node"]
            if node is not None and node not in seen:
                seen.add(node)
                candidates.append({
                    "node":str(node),
                    "kind":"subject_contains",
                    "score":0.60,
                    "accepted":False,
                })

        for row in self.conn.execute(
            f"""
            SELECT DISTINCT {self.oc} AS node
            FROM edges
            WHERE {self.oc} LIKE ? COLLATE NOCASE
            LIMIT ?
            """,
            (pattern,limit),
        ).fetchall():
            node=row["node"]
            if node is not None and node not in seen:
                seen.add(node)
                candidates.append({
                    "node":str(node),
                    "kind":"object_contains",
                    "score":0.55,
                    "accepted":False,
                })

        return candidates[:limit]

    def resolve_entity_strict(
        self,
        mention,
        limit=24,
    ):
        candidates=self.resolve_entity(
            mention,
            limit,
        )
        accepted=[
            item
            for item in candidates
            if item.get("accepted") is True
        ]

        if len(accepted)==1:
            return {
                "status":"resolved",
                "mention":str(mention),
                "canonical":accepted[0]["node"],
                "confidence":float(accepted[0]["score"]),
                "candidates":candidates,
            }

        if len(accepted)>1:
            return {
                "status":"ambiguous",
                "mention":str(mention),
                "canonical":None,
                "confidence":0.0,
                "candidates":candidates,
            }

        return {
            "status":"unresolved",
            "mention":str(mention),
            "canonical":None,
            "confidence":0.0,
            "candidates":candidates,
        }



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


class ContextRelationAttention:
    """
    Query-context -> relation attention.

    spaCy is a fixed parser. This module learns only the compatibility between
    the frozen structural representation and relations actually observed in
    successful graph resolutions.
    """
    def __init__(self, decay=0.65):
        self.decay = decay
        self.counts = defaultdict(Counter)
        self.global_counts = Counter()
        self.updates = 0

    @staticmethod
    def features(parse, ctx):
        features = [
            f"question:{parse.question}",
            f"root:{parse.root_lemma.lower()}",
        ]
        if parse.subjects:
            features.append("has_subject")
        if parse.objects:
            features.append("has_object")
        if parse.entities:
            features.append("has_entity")
        if parse.noun_chunks:
            features.append("has_noun_chunk")

        for entity in parse.entities:
            label = entity.get("label")
            if label:
                features.append(f"entity:{label}")

        for token in parse.tokens[:32]:
            if token.get("pos"):
                features.append(f"pos:{token['pos']}")
            if token.get("dep"):
                features.append(f"dep:{token['dep']}")

        lowered = {token["text"].lower() for token in parse.tokens}
        if lowered & Context.PRONOUNS:
            features.append("context:pronoun")
        if ctx.active_subject:
            features.append("context:active_subject")

        return tuple(dict.fromkeys(features))[:96]

    def score(self, features, relation):
        counter = self.counts.get(tuple(features), Counter())
        total = sum(max(0.0, v) for v in counter.values())
        local = (
            max(0.0, counter.get(relation, 0.0)) / total
            if total > 0 else 0.0
        )

        global_total = sum(
            max(0.0, v) for v in self.global_counts.values()
        )
        global_score = (
            max(0.0, self.global_counts.get(relation, 0.0))
            / global_total
            if global_total > 0 else 0.0
        )

        return 0.85 * local + 0.15 * global_score if local else global_score

    def rank(self, features, relations):
        return sorted(
            (
                (self.score(features, relation), relation)
                for relation in set(relations)
            ),
            key=lambda x: (-x[0], x[1]),
        )

    def update(self, features, relation, strength=1.0):
        if not relation:
            return
        self.counts[tuple(features)][relation] += strength
        self.global_counts[relation] += strength * 0.25
        self.updates += 1

    def export(self):
        top = []
        for features, counter in self.counts.items():
            for relation, weight in counter.most_common(12):
                top.append({
                    "features": list(features),
                    "relation": relation,
                    "weight": round(weight, 6),
                })
        top.sort(key=lambda x: -x["weight"])
        return {
            "decay": self.decay,
            "updates": self.updates,
            "context_states": len(self.counts),
            "global_relations": len(self.global_counts),
            "top_context_relations": top[:50],
        }

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
        self.current_resolution = {
            "status": "unresolved",
            "mention": None,
            "canonical": None,
            "confidence": 0.0,
            "candidates": [],
        }
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

    def subject(self, parse, graph=None):
        mention=entity_mention_from_parse(parse)

        self.current_resolution={
            "status":"unresolved",
            "mention":mention,
            "canonical":None,
            "confidence":0.0,
            "candidates":[],
        }

        if mention:
            if graph is not None:
                resolution=graph.resolve_entity_strict(
                    mention,
                    16,
                )
                self.current_resolution=resolution

                if resolution["status"]=="resolved":
                    self.active_subject=resolution["canonical"]

                    key=mention.lower()
                    record=self.entities.setdefault(
                        key,
                        {
                            "text":mention,
                            "label":(
                                parse.entities[0].get(
                                    "label",
                                    "GRAPH_ID",
                                )
                                if parse.entities
                                else "GRAPH_ID"
                            ),
                            "mentions":0,
                            "canonical":None,
                        },
                    )
                    record["mentions"]+=1
                    record["canonical"]=resolution["canonical"]
                    record["resolution_status"]="resolved"
                    record["resolution_confidence"]=resolution["confidence"]

                    return self.active_subject

                # Critical: no fuzzy candidate is promoted.
                return self.active_subject

            self.active_subject=mention
            return mention

        lowered={
            token["text"].lower()
            for token in parse.tokens
        }

        if lowered & self.PRONOUNS:
            self.current_resolution={
                "status":(
                    "context_resolved"
                    if self.active_subject
                    else "unresolved"
                ),
                "mention":None,
                "canonical":self.active_subject,
                "confidence":(
                    0.75 if self.active_subject else 0.0
                ),
                "candidates":[],
            }
            return self.active_subject

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
            "version": "semantic-memory-v617",
            "active_subject": self.active_subject,
            "active_entity_label": (
                self.active_entity_label
            ),
            "current_resolution": self.current_resolution,
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
        self.current_resolution = payload.get(
            "current_resolution",
            {
                "status": "unresolved",
                "mention": None,
                "canonical": None,
                "confidence": 0.0,
                "candidates": [],
            },
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




def extract_graph_mentions(text):
    patterns=(
        r"https?://[^\s<>\"']+",
        r"www\.[^\s<>\"']+",
    )
    found=[]
    for pattern in patterns:
        found.extend(
            re.findall(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
        )

    result=[]
    seen=set()
    for value in found:
        value=value.rstrip(".,!?;:)]}").strip()
        if not value:
            continue
        key=value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def entity_mention_from_parse(parse):
    uri=extract_graph_mentions(
        parse.text
    )
    if uri:
        return uri[0]
    if parse.entities:
        return parse.entities[0]["text"]
    if parse.objects:
        return parse.objects[-1]
    if parse.subjects:
        return parse.subjects[0]
    return None


def _intent_hypotheses(parse, ctx, max_n=8):
    tokens = parse.tokens
    if not tokens:
        return []

    pos = {t["pos"] for t in tokens}
    has_entity = bool(parse.entities)
    has_subject = bool(parse.subjects)
    has_object = bool(parse.objects)
    has_question = (
        parse.question != "DECLARATIVE"
        or parse.text.strip().endswith("?")
    )

    scores = {
        "conversation": 0.0,
        "relation_lookup": 0.0,
        "entity_profile": 0.0,
        "identity_lookup": 0.0,
        "statement": 0.0,
    }

    # Generic structural evidence from the frozen spaCy parser.
    if "INTJ" in pos:
        scores["conversation"] += 0.60
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
    if not (has_entity or has_subject or has_object):
        scores["conversation"] += 0.20

    for intent in scores:
        scores[intent] += 0.50 * ctx.intent_bias(parse, intent)

    total = sum(max(0.0, v) for v in scores.values()) or 1.0

    ranked = sorted(
        ((v / total, intent) for intent, v in scores.items() if v > 0),
        key=lambda x: (-x[0], x[1]),
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
                "pos": sorted(pos),
                "learned_intent_bias": ctx.intent_bias(parse, intent),
            },
        }
        for score, intent in ranked[:max_n]
    ]


def _candidate_relation_set(graph, vocab, subject):
    candidates = set(vocab)
    if graph is not None and subject:
        try:
            local_edges = graph.outgoing(
                subject,
                max(100, min(200, max(1, len(vocab)) * 5)),
            )
            candidates.update(
                edge.relation
                for edge in local_edges
            )
        except Exception:
            pass
    return candidates



def hypotheses(
    parse,
    ctx,
    vocab,
    max_n=12,
    graph=None,
    relation_attention=None,
):
    if isinstance(vocab, argparse.Namespace):
        raise TypeError(
            "hypotheses() expected relation vocabulary, not argparse.Namespace"
        )
    if vocab is None:
        raise TypeError(
            "hypotheses() requires a relation vocabulary"
        )

    vocab = tuple(str(item) for item in vocab)
    mention = entity_mention_from_parse(parse)
    subject = ctx.subject(parse, graph=graph)
    resolution = getattr(ctx, "current_resolution", {})

    intents = _intent_hypotheses(
        parse,
        ctx,
        min(8, max_n),
    )

    candidates = _candidate_relation_set(
        graph,
        vocab,
        subject,
    )

    features = (
        relation_attention.features(parse, ctx)
        if relation_attention
        else ()
    )

    resolution_candidates = resolution.get("candidates", [])

    direct_relations = set()
    if graph is not None and subject:
        try:
            direct_relations = {
                edge.relation
                for edge in graph.outgoing(
                    subject,
                    min(
                        200,
                        max(100, len(candidates) * 5),
                    ),
                )
            }
        except Exception:
            pass

    output = []

    resolution_ok = (
        not mention
        or resolution.get("status")
        in {"resolved", "context_resolved"}
    )

    for intent_item in intents:
        intent = intent_item["intent"]

        if intent in {"conversation", "statement"}:
            output.append(
                Hypothesis(
                    subject=subject,
                    relation="",
                    intent=intent,
                    lexical_score=intent_item["score"],
                    evidence={
                        "intent_score": intent_item["score"],
                        "structural_features": list(features),
                        "direct_graph_relations": len(direct_relations),
                        "entity_mention": mention,
                        "canonical_subject": subject,
                        "entity_resolution": resolution_candidates[:8],
                        **intent_item["evidence"],
                    },
                )
            )
            continue

        if not resolution_ok:
            output.append(
                Hypothesis(
                    subject=None,
                    relation="",
                    intent="entity_unresolved",
                    lexical_score=0.0,
                    evidence={
                        "entity_resolution": resolution,
                        "structural_features": list(features),
                    },
                )
            )
            continue

        ranked_relations = (
            relation_attention.rank(
                features,
                candidates,
            )
            if relation_attention
            else [(0.0, relation) for relation in candidates]
        )

        for relation_score, relation in ranked_relations:
            goal_bias = ctx.goal_bias(
                parse,
                intent,
                relation,
            )

            outcome_bias = ctx.candidate_relation_bias(
                relation,
                relation,
            )

            affordance = (
                0.12
                if relation in direct_relations
                else 0.0
            )

            score = (
                intent_item["score"]
                + relation_score
                + affordance
                + 0.80 * goal_bias
                + 0.10 * outcome_bias
            )

            output.append(
                Hypothesis(
                    subject=subject,
                    relation=relation,
                    intent=intent,
                    lexical_score=score,
                    evidence={
                        "intent_score": intent_item["score"],
                        "relation_attention": relation_score,
                        "direct_graph_affordance": affordance,
                        "entity_mention": mention,
                        "canonical_subject": subject,
                        "entity_resolution": resolution_candidates[:8],
                        "goal_memory_bias": goal_bias,
                        "relation_outcome_bias": outcome_bias,
                        "structural_features": list(features),
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

    for item in output:
        if len(selected) >= max_n:
            break
        if (
            item.relation == ""
            or per_intent[item.intent] < max(1, max_n // 3)
        ):
            selected.append(item)
            per_intent[item.intent] += 1

    if len(selected) < max_n:
        used = {
            (item.intent, item.relation, item.subject)
            for item in selected
        }

        for item in output:
            key = (
                item.intent,
                item.relation,
                item.subject,
            )
            if key in used:
                continue
            selected.append(item)
            used.add(key)
            if len(selected) >= max_n:
                break

    return selected



def rank_goal_hypotheses(hypotheses_list, results):
    ranked = []

    for hypothesis, result in zip(
        hypotheses_list,
        results,
    ):
        score = hypothesis.lexical_score

        if hypothesis.relation:
            if result.get("success", False):
                score += 4.0
                if result.get("direct_proof", False):
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
            (score, hypothesis, result)
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




def search(graph, att, h, args, seed=0):
    empty = {
        "success": False,
        "steps": 0,
        "path": [],
        "target": None,
        "attention": 0,
        "exploration": 0,
        "direct_proof": False,
        "goal_relation": h.relation,
    }

    if not h.subject:
        return empty

    if not h.relation:
        if h.intent == "conversation":
            return {
                **empty,
                "success": True,
                "intent_only": True,
            }
        return empty

    # PHASE 1: exact requested relation from current subject.
    for edge in list(
        graph.outgoing(
            h.subject,
            args.per_node,
        )
    ):
        if edge.relation == h.relation:
            return {
                "success": True,
                "steps": 1,
                "path": [h.relation],
                "target": edge.object,
                "attention": 0,
                "exploration": 0,
                "direct_proof": True,
                "goal_relation": h.relation,
            }

    if args.max_depth <= 1:
        return empty

    rng = random.Random(seed)
    queue = []
    tie = 0

    heapq.heappush(
        queue,
        (0.0, 0, tie, h.subject, ()),
    )

    seen = {
        (h.subject, ())
    }

    expanded = 0
    attention_hits = 0
    exploration = 0

    while queue and expanded < args.goal_budget:
        _, depth, _, node, prefix = heapq.heappop(queue)

        if depth >= args.max_depth - 1:
            continue

        expanded += 1

        edges = list(
            graph.outgoing(
                node,
                args.per_node,
            )
        )

        ranked = att.rank(
            h.relation,
            prefix,
            [edge.relation for edge in edges],
        )
        scores = dict(ranked)

        edges.sort(
            key=lambda edge: (
                -scores.get(edge.relation, 0.0),
                rng.random(),
            )
        )

        for edge in edges:
            next_prefix = prefix + (edge.relation,)
            state = (edge.object, next_prefix)

            if state in seen:
                continue

            seen.add(state)

            if scores.get(edge.relation, 0.0) > 0:
                attention_hits += 1
            else:
                exploration += 1

            # The requested relation is allowed as a bridge step.
            # Proof is always checked against the endpoint.
            goal_edges = list(
                graph.outgoing(
                    edge.object,
                    args.per_node,
                )
            )

            for goal_edge in goal_edges:
                if goal_edge.relation == h.relation:
                    return {
                        "success": True,
                        "steps": expanded,
                        "path": list(next_prefix) + [h.relation],
                        "target": goal_edge.object,
                        "attention": attention_hits,
                        "exploration": exploration,
                        "direct_proof": False,
                        "goal_relation": h.relation,
                    }

            if len(next_prefix) < args.max_depth - 1:
                tie += 1
                heapq.heappush(
                    queue,
                    (
                        -scores.get(edge.relation, 0.0),
                        len(next_prefix),
                        tie,
                        edge.object,
                        next_prefix,
                    ),
                )

    return {
        **empty,
        "steps": expanded,
        "attention": attention_hits,
        "exploration": exploration,
    }


def trace_result(text, parse, hs, selected, search_result, ctx):
    # V617: keep the trace serializer on the current context-aware API.
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
    path_attention,
    relation_attention,
    parse,
    ctx,
):
    if not selected or not selected.relation:
        return

    resolution=getattr(
        ctx,
        "current_resolution",
        {},
    )

    if resolution.get("status") not in {
        "resolved",
        "context_resolved",
    }:
        return

    canonical=resolution.get(
        "canonical"
    )

    if (
        canonical
        and selected.subject
        and selected.subject != canonical
    ):
        return

    if not result.get(
        "success",
        False,
    ):
        return

    path=tuple(
        result.get("path", [])
    )

    if not path:
        return

    relation=selected.relation

    memory.relation_outcomes[
        relation
    ]["success"] += 1.0

    memory.remember_path(
        relation,
        path,
        True,
        strength=1.0,
    )

    features=relation_attention.features(
        parse,
        ctx,
    )

    relation_attention.update(
        features,
        path[0],
        strength=1.0,
    )

    prefix=()
    for next_relation in path:
        path_attention.update(
            relation,
            prefix,
            next_relation,
            strength=1.0,
        )
        prefix += (next_relation,)


def smoke(
    graph,
    parser,
    path_attention,
    relation_attention,
    vocab,
    args,
    memory,
):
    ctx=memory

    row=None
    target_relation=None

    for candidate_relation in vocab:
        row=graph.conn.execute(
            f"""
            SELECT {graph.sc} AS subject
            FROM edges
            WHERE {graph.rc}=?
            LIMIT 1
            """,
            (candidate_relation,),
        ).fetchone()
        if row and row["subject"]:
            target_relation=candidate_relation
            break

    if not row or not row["subject"]:
        raise RuntimeError(
            "Smoke could not find graph subject"
        )

    canonical=str(row["subject"])

    inputs=[
        "hello",
        f"What is the {target_relation} of {canonical}?",
        "What can you tell me about it?",
        "not-a-real-graph-entity-v617",
    ]

    traces=[]

    print("=== V617 SMOKE ===",flush=True)
    print(
        f"graph-grounded subject : {canonical}",
        flush=True,
    )
    print(
        f"graph-grounded relation: {target_relation}",
        flush=True,
    )

    for index,user_text in enumerate(inputs,1):
        parse=parser.parse(user_text)

        hs=hypotheses(
            parse=parse,
            ctx=ctx,
            vocab=vocab,
            max_n=args.max_hypotheses,
            graph=graph,
            relation_attention=relation_attention,
        )

        results=[
            resolve_hypothesis(
                graph,
                path_attention,
                h,
                args,
                args.seed+index*100+offset,
            )
            for offset,h in enumerate(hs)
        ]

        ranked=rank_goal_hypotheses(
            hs,
            results,
        )

        selected=ranked[0][1] if ranked else None
        result=(
            ranked[0][2]
            if ranked
            else {
                "success":False,
                "steps":0,
                "path":[],
            }
        )

        _learn_from_resolution(
            selected,
            result,
            memory,
            path_attention,
            relation_attention,
            parse,
            ctx,
        )

        ctx.record_turn(
            user_text,
            parse,
            selected,
            result,
            hs,
        )

        resolution=getattr(
            ctx,
            "current_resolution",
            {},
        )

        trace=trace_result(
            user_text,
            parse,
            hs,
            selected,
            result,
            ctx,
        )
        trace["entity_resolution"]=resolution
        trace["relation_attention"]=relation_attention.export()
        trace["path_attention"]=path_attention.export()
        traces.append(trace)

        print(
            f"[SMOKE {index}/{len(inputs)}] {user_text}",
            flush=True,
        )
        print(
            f"  intent={selected.intent if selected else None} "
            f"subject={selected.subject if selected else None!r} "
            f"relation={selected.relation if selected else None!r}",
            flush=True,
        )
        print(
            f"  success={result.get('success',False)} "
            f"direct={result.get('direct_proof',False)} "
            f"steps={result.get('steps',0)}",
            flush=True,
        )
        print(
            "  entity resolution:",
            resolution,
            flush=True,
        )
        print(
            f"  relation_attention_updates={relation_attention.updates} "
            f"path_attention_updates={path_attention.updates}",
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
    ap.add_argument("--mode", choices=("chat", "smoke"), default="chat")
    ap.add_argument("--max-hypotheses", type=int, default=12)
    ap.add_argument("--relation-vocabulary", type=int, default=200)
    ap.add_argument("--goal-budget", type=int, default=40)
    ap.add_argument("--budget", type=int, default=80)
    ap.add_argument("--per-node", type=int, default=60)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--cache-entries", type=int, default=12000)
    ap.add_argument("--prior-decay", type=float, default=.65)
    ap.add_argument("--seed", type=int, default=61400)
    ap.add_argument("--progress-every", type=int, default=1)
    args = ap.parse_args()

    db = Path(args.database).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    base = output.with_suffix("")
    trace_path = Path(
        args.trace_output or f"{base}_traces.jsonl"
    ).resolve()
    prior_path = Path(
        args.prior_output or f"{base}_prior.json"
    ).resolve()
    memory_path = Path(
        args.memory_output or f"{base}_memory.json"
    ).resolve()

    print("=== V617 SEMANTIC CHAT GATEWAY ===", flush=True)
    print(f"database              : {db}", flush=True)
    print("grammar               : FROZEN spaCy", flush=True)
    print("grammar training      : OFF", flush=True)
    print("goal layer            : contextual relation attention", flush=True)
    print("search                : goal-conditioned path prior + BFS", flush=True)
    print(f"persistent memory     : {memory_path}", flush=True)

    graph = Graph(db, args.cache_entries)

    try:
        print("[GATEWAY 1/5] loading spaCy...", flush=True)
        parser = SpaCyParser(args.spacy_model)

        print("[GATEWAY 2/5] relation vocabulary...", flush=True)
        vocab = graph.relation_vocab(limit=args.relation_vocabulary)
        if not isinstance(vocab, (list, tuple)):
            raise RuntimeError("relation vocabulary contract violated")
        if not all(isinstance(x, str) for x in vocab):
            raise RuntimeError("relation vocabulary contains non-string value")
        print(f"    relations={len(vocab)}", flush=True)

        print("[GATEWAY 3/5] global path attention...", flush=True)
        path_attention = Attention(args.prior_decay)

        print("[GATEWAY 4/5] contextual relation attention...", flush=True)
        relation_attention = ContextRelationAttention(args.prior_decay)

        print("[GATEWAY 5/5] persistent semantic memory...", flush=True)
        memory = Context(path=memory_path)
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
                path_attention,
                relation_attention,
                vocab,
                args,
                memory,
            )
        else:
            ctx = memory
            print("=== V617 CHAT (type 'exit') ===", flush=True)

            while True:
                try:
                    user_text = input("chat> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

                if not user_text:
                    continue
                if user_text.lower() in {"exit", "quit"}:
                    break

                parse = parser.parse(user_text)
                hs = hypotheses(
                    parse=parse,
                    ctx=ctx,
                    vocab=vocab,
                    max_n=args.max_hypotheses,
                    graph=graph,
                    relation_attention=relation_attention,
                )

                results = [
                    resolve_hypothesis(
                        graph,
                        path_attention,
                        h,
                        args,
                        args.seed + i,
                    )
                    for i, h in enumerate(hs)
                ]

                ranked = rank_goal_hypotheses(hs, results)
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

                _learn_from_resolution(
                    selected,
                    result,
                    memory,
                    path_attention,
                    relation_attention,
                    parse,
                    ctx,
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
                traces.append(trace)

                print("intent:", selected.intent if selected else None, flush=True)
                print("goal:", asdict(selected) if selected else None, flush=True)
                print("evidence:", result, flush=True)
                print("alternatives:", len(hs), flush=True)
                print("active subject:", ctx.active_subject, flush=True)
                print("relation attention updates:", relation_attention.updates, flush=True)
                print("path attention updates:", path_attention.updates, flush=True)

        if traces:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("w", encoding="utf-8") as f:
                for trace in traces:
                    f.write(
                        json.dumps(
                            trace,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        prior_path.write_text(
            json.dumps(
                path_attention.export(),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        relation_prior_path = prior_path.with_name(
            prior_path.stem + "_relation_attention.json"
        )
        relation_prior_path.write_text(
            json.dumps(
                relation_attention.export(),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        memory.save()

        output.write_text(
            json.dumps(
                {
                    "version": "V617",
                    "architecture": {
                        "grammar": "frozen spaCy",
                        "relation_attention": "learned context-conditioned relation attention",
                        "path_attention": "goal-conditioned path prior",
                        "search": "bounded cognitive graph search",
                        "memory": "persistent entity/intent/goal/path memory",
                    },
                    "config": vars(args),
                    "relation_vocabulary": vocab,
                    "traces": len(traces),
                    "relation_attention": relation_attention.export(),
                    "path_attention": path_attention.export(),
                    "memory": {
                        "path": str(memory_path),
                        "entities": len(memory.entities),
                        "turns": len(memory.turns),
                        "intent_patterns": len(memory.intent_memory),
                        "goal_patterns": len(memory.goal_memory),
                        "path_patterns": len(memory.path_memory),
                    },
                    "trace_output": str(trace_path),
                    "prior_output": str(prior_path),
                    "relation_prior_output": str(relation_prior_path),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print("=== V617 COMPLETE ===", flush=True)
        print(f"mode                  : {args.mode}", flush=True)
        print(f"traces                : {len(traces)}", flush=True)
        print(f"memory entities       : {len(memory.entities)}", flush=True)
        print(f"memory turns          : {len(memory.turns)}", flush=True)
        print(
            f"learned intent patterns: {len(memory.intent_memory)}",
            flush=True,
        )
        print(
            f"learned goal patterns  : {len(memory.goal_memory)}",
            flush=True,
        )
        print(
            f"learned path patterns  : {len(memory.path_memory)}",
            flush=True,
        )
        print(
            f"relation attention updates: {relation_attention.updates}",
            flush=True,
        )
        print(
            f"relation attention contexts: {len(relation_attention.counts)}",
            flush=True,
        )
        print(
            f"path attention updates: {path_attention.updates}",
            flush=True,
        )
        print(
            f"path attention exact states: {len(path_attention.counts)}",
            flush=True,
        )
        print(f"JSON                  : {output}", flush=True)
        print(f"TRACE                 : {trace_path}", flush=True)
        print(f"PRIOR                 : {prior_path}", flush=True)
        print(f"MEMORY                : {memory_path}", flush=True)
    finally:
        graph.close()


if __name__ == "__main__":
    main()
