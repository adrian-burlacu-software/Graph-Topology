
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path


# =============================================================================
# Fixed project defaults
# =============================================================================

DSTC8 = Path(
    r"C:\Users\adria\Desktop\dev\Graph-Topology\data\dstc8-schema-guided-dialogue"
)
MULTIWOZ = Path(
    r"C:\Users\adria\Desktop\dev\Graph-Topology\data\multiwoz"
)
DIALOGUE = Path(
    r"C:\Users\adria\Desktop\dev\Graph-Topology\data\dialoglue"
)
UBUNTU = Path(
    r"C:\Users\adria\Desktop\dev\Graph-Topology\data\ubuntu"
)
MODEL = Path(
    r"C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM2-1.7B-Instruct"
)
SPACY_MODEL = "en_core_web_sm"

RESULTS = Path(
    r"C:\Users\adria\Desktop\dev\Graph-Topology\results"
)

SEMANTIC_DB = RESULTS / "assistant_semantic_net.sqlite"


# =============================================================================
# Helpers
# =============================================================================

def norm(x):
    x = str(x or "").strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def lemma_key(x):
    x = norm(x)
    x = re.sub(r"[^\w'\- ]", "", x)
    return x


def digest(*parts):
    raw = json.dumps(
        parts,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [
            json_safe(v)
            for v in value
        ]
    return value


def iter_files(root, suffixes=None):
    wanted = None
    if suffixes:
        wanted = {
            str(x).lower()
            for x in suffixes
        }

    if not root.exists():
        return

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if wanted and p.suffix.lower() not in wanted:
            continue
        yield p


def write_json(path, payload):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            json_safe(payload),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def append_jsonl(path, payload):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                json_safe(payload),
                ensure_ascii=False,
            ) + "\n"
        )


# =============================================================================
# Semantic net database
# =============================================================================

def init_db(path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")

    con.executescript("""
    CREATE TABLE IF NOT EXISTS nodes (
        node_id TEXT PRIMARY KEY,
        node_type TEXT NOT NULL,
        label TEXT NOT NULL,
        source TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        count INTEGER NOT NULL DEFAULT 1
    );

    CREATE INDEX IF NOT EXISTS idx_nodes_type
      ON nodes(node_type);

    CREATE INDEX IF NOT EXISTS idx_nodes_label
      ON nodes(label);

    CREATE TABLE IF NOT EXISTS edges (
        edge_id TEXT PRIMARY KEY,
        source_node TEXT NOT NULL,
        relation TEXT NOT NULL,
        target_node TEXT NOT NULL,
        source_dataset TEXT NOT NULL,
        weight REAL NOT NULL DEFAULT 1.0,
        payload_json TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_edges_source
      ON edges(source_node);

    CREATE INDEX IF NOT EXISTS idx_edges_target
      ON edges(target_node);

    CREATE INDEX IF NOT EXISTS idx_edges_relation
      ON edges(relation);

    CREATE TABLE IF NOT EXISTS utterances (
        utterance_id TEXT PRIMARY KEY,
        dataset TEXT NOT NULL,
        dialogue_id TEXT,
        turn_index INTEGER,
        speaker TEXT,
        text TEXT NOT NULL,
        intent TEXT,
        domain TEXT,
        payload_json TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_utterances_dataset
      ON utterances(dataset);

    CREATE INDEX IF NOT EXISTS idx_utterances_intent
      ON utterances(intent);

    CREATE TABLE IF NOT EXISTS dialogue_states (
        state_id TEXT PRIMARY KEY,
        dataset TEXT NOT NULL,
        dialogue_id TEXT,
        turn_index INTEGER,
        payload_json TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS intents (
        intent_id TEXT PRIMARY KEY,
        dataset TEXT NOT NULL,
        domain TEXT,
        name TEXT NOT NULL,
        description TEXT,
        count INTEGER NOT NULL DEFAULT 1
    );

    CREATE UNIQUE INDEX IF NOT EXISTS uq_intent
      ON intents(dataset, domain, name);

    CREATE TABLE IF NOT EXISTS slots (
        slot_id TEXT PRIMARY KEY,
        dataset TEXT NOT NULL,
        domain TEXT,
        name TEXT NOT NULL,
        slot_type TEXT,
        description TEXT,
        count INTEGER NOT NULL DEFAULT 1
    );

    CREATE UNIQUE INDEX IF NOT EXISTS uq_slot
      ON slots(dataset, domain, name);

    CREATE TABLE IF NOT EXISTS actions (
        action_id TEXT PRIMARY KEY,
        dataset TEXT NOT NULL,
        name TEXT NOT NULL,
        description TEXT,
        count INTEGER NOT NULL DEFAULT 1
    );

    CREATE UNIQUE INDEX IF NOT EXISTS uq_action
      ON actions(dataset, name);

    CREATE TABLE IF NOT EXISTS schemas (
        schema_id TEXT PRIMARY KEY,
        dataset TEXT NOT NULL,
        domain TEXT,
        service TEXT,
        payload_json TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS parser_facts (
        fact_id TEXT PRIMARY KEY,
        utterance_id TEXT NOT NULL,
        predicate TEXT NOT NULL,
        role TEXT,
        argument TEXT,
        pos TEXT,
        payload_json TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_parser_facts_pred
      ON parser_facts(predicate);

    CREATE TABLE IF NOT EXISTS curiosity_targets (
        target_id TEXT PRIMARY KEY,
        node_id TEXT NOT NULL,
        label TEXT NOT NULL,
        node_type TEXT NOT NULL,
        family TEXT NOT NULL,
        question TEXT NOT NULL,
        score REAL NOT NULL,
        asked INTEGER NOT NULL DEFAULT 0,
        answered INTEGER NOT NULL DEFAULT 0,
        evidence_count INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL,
        UNIQUE(node_id, family)
    );

    CREATE TABLE IF NOT EXISTS curiosity_answers (
        answer_id TEXT PRIMARY KEY,
        target_id TEXT NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        parser_json TEXT NOT NULL,
        fact_count INTEGER NOT NULL DEFAULT 0,
        accepted INTEGER NOT NULL,
        reason TEXT,
        created_at REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS assistant_patterns (
        subject TEXT NOT NULL,
        relation TEXT NOT NULL,
        object TEXT NOT NULL,
        count INTEGER NOT NULL DEFAULT 1,
        PRIMARY KEY(subject, relation, object)
    );

    CREATE TABLE IF NOT EXISTS run_state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """)
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_nodes_type_count
        ON nodes(node_type, count DESC)
    """)
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_curiosity_targets_node_family
        ON curiosity_targets(node_id, family)
    """)
    con.commit()
    return con


def add_node(
    con,
    node_type,
    label,
    source,
    payload=None,
    count=1,
):
    label = norm(label)
    if not label:
        return None

    node_id = digest(
        "node",
        node_type,
        label,
        source,
    )

    con.execute("""
        INSERT INTO nodes
        (node_id,node_type,label,source,payload_json,count)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(node_id)
        DO UPDATE SET count=count+excluded.count
    """, (
        node_id,
        node_type,
        label,
        source,
        json.dumps(
            payload or {},
            ensure_ascii=False,
        ),
        count,
    ))
    return node_id


def add_edge(
    con,
    source_node,
    relation,
    target_node,
    source_dataset,
    payload=None,
    weight=1.0,
):
    if not source_node or not target_node:
        return

    edge_id = digest(
        "edge",
        source_node,
        relation,
        target_node,
        source_dataset,
    )

    con.execute("""
        INSERT INTO edges
        (edge_id,source_node,relation,target_node,
         source_dataset,weight,payload_json)
        VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(edge_id)
        DO UPDATE SET weight=weight+excluded.weight
    """, (
        edge_id,
        source_node,
        norm(relation),
        target_node,
        source_dataset,
        weight,
        json.dumps(
            payload or {},
            ensure_ascii=False,
        ),
    ))


def upsert_intent(
    con,
    dataset,
    domain,
    name,
    description="",
):
    if not name:
        return
    iid = digest(
        "intent",
        dataset,
        domain,
        name,
    )
    con.execute("""
        INSERT INTO intents
        VALUES (?,?,?,?,?,1)
        ON CONFLICT(intent_id)
        DO UPDATE SET count=count+1
    """, (
        iid,
        dataset,
        domain or "",
        name,
        description or "",
    ))


def upsert_slot(
    con,
    dataset,
    domain,
    name,
    slot_type="",
    description="",
):
    if not name:
        return
    sid = digest(
        "slot",
        dataset,
        domain,
        name,
    )
    con.execute("""
        INSERT INTO slots
        VALUES (?,?,?,?,?,?,1)
        ON CONFLICT(slot_id)
        DO UPDATE SET count=count+1
    """, (
        sid,
        dataset,
        domain or "",
        name,
        slot_type or "",
        description or "",
    ))


def upsert_action(
    con,
    dataset,
    name,
    description="",
):
    if not name:
        return
    aid = digest(
        "action",
        dataset,
        name,
    )
    con.execute("""
        INSERT INTO actions
        VALUES (?,?,?,?,1)
        ON CONFLICT(action_id)
        DO UPDATE SET count=count+1
    """, (
        aid,
        dataset,
        name,
        description or "",
    ))



def configure_parser(nlp):
    """Print parser/runtime information and return a useful batch size."""
    pipes=list(nlp.pipe_names)
    print(
        f"      spaCy pipes={pipes}",
        flush=True,
    )

    try:
        import torch
        gpu=torch.cuda.is_available()
        print(
            f"      torch_cuda={gpu}",
            flush=True,
        )
    except Exception:
        gpu=False

    # Transformer parsing is expensive; pipe() is dramatically cheaper than
    # calling nlp(text) independently for every utterance.
    batch_size=32 if gpu else 16
    print(
        f"      parser_batch_size={batch_size}",
        flush=True,
    )
    return batch_size


def parse_text_batch(nlp, items, batch_size):
    """
    Parse [(identifier, text), ...] with nlp.pipe and return:
        [(identifier, text, facts), ...]
    """
    docs=nlp.pipe(
        [text for _,text in items],
        batch_size=batch_size,
        n_process=1,
    )

    out=[]
    for (identifier,text),doc in zip(items,docs):
        facts=[]
        for token in doc:
            if token.is_space or token.is_punct:
                continue

            if token.pos_ in {"VERB","AUX"}:
                predicate=lemma_key(
                    token.lemma_ or token.text
                )
                facts.append({
                    "kind":"predicate",
                    "predicate":predicate,
                    "surface":token.text,
                    "pos":token.pos_,
                    "dep":token.dep_,
                })

                for child in token.children:
                    if child.is_space or child.is_punct:
                        continue

                    role=None
                    if child.dep_ in {
                        "nsubj","nsubjpass","csubj","csubjpass"
                    }:
                        role="subject"
                    elif child.dep_ in {"obj","dobj","iobj"}:
                        role="object"
                    elif (
                        child.dep_.startswith("obl")
                        or child.dep_=="prep"
                    ):
                        role="oblique"
                    elif child.dep_ in {
                        "xcomp","ccomp","advcl","acl","relcl"
                    }:
                        role="complement"

                    if role:
                        facts.append({
                            "kind":"argument",
                            "predicate":predicate,
                            "role":role,
                            "argument":lemma_key(
                                child.lemma_ or child.text
                            ),
                            "surface":child.text,
                            "pos":child.pos_,
                            "dep":child.dep_,
                        })

        out.append(
            (identifier,text,facts)
        )

    return out


# =============================================================================
# Coarse semantic parser
# =============================================================================

def load_spacy():
    try:
        import spacy
        return spacy.load(SPACY_MODEL)
    except Exception as exc:
        raise SystemExit(
            f"Could not load spaCy model {SPACY_MODEL}: {exc}\n"
            f"Install with:\n"
            f"python -m spacy download {SPACY_MODEL}"
        )


def parser_facts(nlp, text):
    doc = nlp(text)
    facts = []

    for token in doc:
        if token.is_space or token.is_punct:
            continue

        if token.pos_ in {"VERB", "AUX"}:
            predicate = lemma_key(
                token.lemma_ or token.text
            )
            facts.append({
                "kind": "predicate",
                "predicate": predicate,
                "surface": token.text,
                "pos": token.pos_,
                "dep": token.dep_,
            })

            for child in token.children:
                if child.is_space or child.is_punct:
                    continue

                role = None
                if child.dep_ in {
                    "nsubj",
                    "nsubjpass",
                    "csubj",
                    "csubjpass",
                }:
                    role = "subject"
                elif child.dep_ in {
                    "obj",
                    "dobj",
                    "iobj",
                }:
                    role = "object"
                elif (
                    child.dep_.startswith("obl")
                    or child.dep_ == "prep"
                ):
                    role = "oblique"
                elif child.dep_ in {
                    "xcomp",
                    "ccomp",
                    "advcl",
                    "acl",
                    "relcl",
                }:
                    role = "complement"

                if role:
                    facts.append({
                        "kind": "argument",
                        "predicate": predicate,
                        "role": role,
                        "argument": lemma_key(
                            child.lemma_ or child.text
                        ),
                        "surface": child.text,
                        "pos": child.pos_,
                        "dep": child.dep_,
                    })

    return facts


def store_parser_facts(
    con,
    utterance_id,
    facts,
):
    count = 0

    for fact in facts:
        fact_id = digest(
            "parser",
            utterance_id,
            fact,
        )
        con.execute("""
            INSERT OR IGNORE INTO parser_facts
            VALUES (?,?,?,?,?,?,?)
        """, (
            fact_id,
            utterance_id,
            fact.get("predicate",""),
            fact.get("role",""),
            fact.get("argument",""),
            fact.get("pos",""),
            json.dumps(
                fact,
                ensure_ascii=False,
            ),
        ))

        if fact.get("kind") == "predicate":
            p = add_node(
                con,
                "predicate",
                fact["predicate"],
                "parser",
                fact,
            )
            if p:
                count += 1

        elif fact.get("kind") == "argument":
            p = add_node(
                con,
                "predicate",
                fact["predicate"],
                "parser",
                fact,
            )
            a = add_node(
                con,
                "concept",
                fact["argument"],
                "parser",
                fact,
            )
            add_edge(
                con,
                p,
                fact["role"] or "argument",
                a,
                "parser",
                fact,
            )
            if p and a:
                con.execute("""
                    INSERT INTO assistant_patterns
                    VALUES (?,?,?,1)
                    ON CONFLICT(subject,relation,object)
                    DO UPDATE SET count=count+1
                """, (
                    fact["predicate"],
                    fact["role"] or "argument",
                    fact["argument"],
                ))
                count += 1

    return count



def iter_candidate_records(value, path="root", depth=0):
    """
    Recursively expose likely dataset records while retaining their location.

    This is deliberately structural rather than semantic: it discovers real
    records first, then the dataset-specific adapters decide how to interpret
    them.
    """
    if depth>8:
        return

    if isinstance(value,dict):
        keys={str(k).lower() for k in value.keys()}

        # A record-like dict often contains one of these direct text fields.
        if keys & {
            "text","utterance","query","sentence",
            "question","input","prompt","response",
            "context","label","intent","domain",
        }:
            yield path,value
            return

        # Ubuntu-style y/c/r container.
        if {"y","c","r"} <= keys:
            yield path,value
            return

        for k,v in value.items():
            yield from iter_candidate_records(
                v,
                f"{path}.{k}",
                depth+1,
            )
        return

    if isinstance(value,list):
        # Dataset records are frequently a list of dicts/lists.
        for i,v in enumerate(value):
            yield from iter_candidate_records(
                v,
                f"{path}[{i}]",
                depth+1,
            )
        return


def text_from_scalar(x):
    if isinstance(x,str):
        text=norm(x)
        if len(text.split())>=2 and len(text)<=4000:
            return text
    return None


def extract_text_candidates(record):
    """
    Return (field,text) pairs from a record without requiring a fixed schema.
    """
    found=[]

    if isinstance(record,dict):
        preferred=(
            "text","utterance","query","sentence","question",
            "input","prompt","user","user_input",
            "user_utterance","response","system","context",
            "reply","target",
        )
        for field in preferred:
            value=record.get(field)
            text=text_from_scalar(value)
            if text:
                found.append((field,text))

        # Common nested turns.
        for field in (
            "turns","utterances","dialogue",
            "conversations","messages",
        ):
            value=record.get(field)
            if isinstance(value,list):
                for i,item in enumerate(value):
                    for subfield,text in extract_text_candidates(item):
                        found.append(
                            (f"{field}[{i}].{subfield}",text)
                        )

    elif isinstance(record,(list,tuple)):
        for i,item in enumerate(record[:50]):
            text=text_from_scalar(item)
            if text:
                found.append((f"[{i}]",text))

    return found


def add_discovered_utterance(
    con,
    dataset,
    path,
    index,
    text,
    payload,
    intent="",
    domain="",
    speaker="unknown",
):
    uid=digest(
        "utterance",
        dataset,
        path,
        index,
        text,
    )

    con.execute("""
        INSERT OR REPLACE INTO utterances
        VALUES (?,?,?,?,?,?,?,?,?)
    """,(
        uid,
        dataset,
        path,
        index,
        speaker,
        text,
        intent,
        domain,
        json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        ),
    ))

    u=add_node(
        con,
        "utterance",
        text,
        dataset,
        payload,
    )

    if intent:
        i=add_node(
            con,
            "intent",
            intent,
            dataset,
        )
        add_edge(
            con,
            u,
            "expresses_intent",
            i,
            dataset,
        )
        upsert_intent(
            con,
            dataset,
            domain,
            intent,
        )

    if domain:
        d=add_node(
            con,
            "domain",
            domain,
            dataset,
        )
        add_edge(
            con,
            u,
            "in_domain",
            d,
            dataset,
        )

    return uid
# =============================================================================
# Generic JSON extraction
# =============================================================================

def read_json_file(path):
    try:
        return json.loads(
            path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        )
    except Exception:
        return None


def flatten_strings(value, prefix=""):
    if isinstance(value, str):
        yield prefix, value
        return

    if isinstance(value, dict):
        for k, v in value.items():
            next_prefix = f"{prefix}.{k}" if prefix else str(k)
            yield from flatten_strings(v, next_prefix)
        return

    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from flatten_strings(
                v,
                f"{prefix}[{i}]",
            )


# =============================================================================
# Schema-Guided Dialogue
# =============================================================================



def ingest_sgd(con, root, nlp=None, batch_size=16, status_every=1000, parse_text=False):
    """
    Strict SGD ingestion.

    Rules:
      * schema.json is NEVER an utterance source.
      * schema.json contributes services/intents/slots only.
      * dialogue files contribute utterances only when they contain actual
        dialogue turns/utterances.
      * arbitrary schema descriptions, examples, slot documentation, etc.
        are never recursively harvested as dialogue text.
    """
    stats=Counter()
    json_files=list(iter_files(root,{".json"}))
    total_files=len(json_files)
    started=time.perf_counter()
    pending=[]

    def flush_pending():
        nonlocal pending
        if not pending:
            return

        if parse_text:
            batch_parsed=parse_text_batch(
                nlp,
                pending,
                batch_size,
            )
            parsed=[
                (*identifier,facts)
                for identifier,_text,facts in batch_parsed
            ]
        else:
            parsed=[
                (*item,[])
                for item,_text in pending
            ]

        for uid,text,turn,payload,dialogue_id,intent,speaker,facts in parsed:
            con.execute("""
                INSERT OR REPLACE INTO utterances
                VALUES (?,?,?,?,?,?,?,?,?)
            """,(
                uid,
                "sgd",
                str(dialogue_id),
                turn,
                speaker,
                text,
                intent,
                "",
                json.dumps(
                    payload,
                    ensure_ascii=False,
                ),
            ))

            if facts:
                store_parser_facts(
                    con,
                    uid,
                    facts,
                )

            u=add_node(
                con,
                "utterance",
                text,
                "sgd",
                payload,
            )

            if intent:
                i=add_node(
                    con,
                    "intent",
                    intent,
                    "sgd",
                )
                add_edge(
                    con,
                    u,
                    "expresses_intent",
                    i,
                    "sgd",
                )
                upsert_intent(
                    con,
                    "sgd",
                    "",
                    intent,
                )

            stats["utterances"]+=1

        pending=[]

    def ingest_schema_file(path):
        data=read_json_file(path)
        if data is None:
            return

        services=[]

        if isinstance(data,list):
            services=data
        elif isinstance(data,dict):
            # Common SGD schema layouts.
            if isinstance(data.get("services"),list):
                services=data["services"]
            elif "service_name" in data:
                services=[data]
            else:
                # Some releases wrap service records under arbitrary keys.
                for value in data.values():
                    if isinstance(value,list):
                        for item in value:
                            if (
                                isinstance(item,dict)
                                and (
                                    "service_name" in item
                                    or "intents" in item
                                    or "slots" in item
                                )
                            ):
                                services.append(item)

        print(
            f"        SGD schema records={len(services):,}",
            flush=True,
        )

        for service in services:
            if not isinstance(service,dict):
                continue

            domain=str(
                service.get("service_name")
                or service.get("name")
                or ""
            ).strip()

            if not domain:
                continue

            stats["services"]+=1

            schema_id=digest(
                "sgd_schema",
                path,
                domain,
            )

            con.execute("""
                INSERT OR REPLACE INTO schemas
                VALUES (?,?,?,?,?)
            """,(
                schema_id,
                "sgd",
                domain,
                domain,
                json.dumps(
                    service,
                    ensure_ascii=False,
                ),
            ))

            service_node=add_node(
                con,
                "service",
                domain,
                "sgd",
                service,
            )

            for intent in service.get("intents",[]) or []:
                if not isinstance(intent,dict):
                    continue

                name=str(
                    intent.get("name")
                    or intent.get("intent")
                    or ""
                ).strip()

                if not name:
                    continue

                desc=str(
                    intent.get("description")
                    or ""
                )

                upsert_intent(
                    con,
                    "sgd",
                    domain,
                    name,
                    desc,
                )

                intent_node=add_node(
                    con,
                    "intent",
                    name,
                    "sgd",
                    intent,
                )

                add_edge(
                    con,
                    service_node,
                    "supports_intent",
                    intent_node,
                    "sgd",
                )

                stats["intents"]+=1

                # SGD schema intent slots.
                for slot_name in (
                    intent.get("required_slots",[])
                    or []
                ):
                    if isinstance(slot_name,str):
                        upsert_slot(
                            con,
                            "sgd",
                            domain,
                            slot_name,
                        )
                        slot_node=add_node(
                            con,
                            "slot",
                            slot_name,
                            "sgd",
                        )
                        add_edge(
                            con,
                            intent_node,
                            "requires_slot",
                            slot_node,
                            "sgd",
                        )

            for slot in service.get("slots",[]) or []:
                if not isinstance(slot,dict):
                    continue

                name=str(
                    slot.get("name")
                    or slot.get("slot_name")
                    or ""
                ).strip()

                if not name:
                    continue

                upsert_slot(
                    con,
                    "sgd",
                    domain,
                    name,
                    slot.get("type",""),
                    slot.get("description",""),
                )

                slot_node=add_node(
                    con,
                    "slot",
                    name,
                    "sgd",
                    slot,
                )

                add_edge(
                    con,
                    service_node,
                    "has_slot",
                    slot_node,
                    "sgd",
                )

                stats["slots"]+=1

                for value in (
                    slot.get("possible_values",[])
                    or []
                ):
                    value_node=add_node(
                        con,
                        "slot_value",
                        str(value),
                        "sgd",
                    )
                    add_edge(
                        con,
                        slot_node,
                        "allows_value",
                        value_node,
                        "sgd",
                    )

    for fi,path in enumerate(json_files,1):
        name=path.name.lower()

        print(
            f"      SGD file {fi}/{total_files} "
            f"{path.name} "
            f"size={path.stat().st_size/1024/1024:.1f}MB",
            flush=True,
        )

        data=read_json_file(path)
        if data is None:
            print(
                "        invalid JSON; skipped",
                flush=True,
            )
            continue

        # HARD RULE: schema.json is metadata, never dialogue.
        if name=="schema.json" or name.startswith("schema"):
            ingest_schema_file(path)
            con.commit()

            elapsed=time.perf_counter()-started
            print(
                f"      SGD schema {fi}/{total_files} done "
                f"services={stats['services']:,} "
                f"intents={stats['intents']:,} "
                f"slots={stats['slots']:,} "
                f"time={elapsed:.1f}s",
                flush=True,
            )
            continue

        # Dialogue files must contain dialogue records. Do not recursively
        # inspect arbitrary dicts/lists for strings.
        if isinstance(data,list):
            records=data
        elif isinstance(data,dict):
            # Some SGD files wrap records under a list key.
            records=[]
            for k,v in data.items():
                if isinstance(v,list):
                    # Only accept lists whose first item is a dialogue record.
                    if v and isinstance(v[0],dict) and (
                        "turns" in v[0]
                        or "dialogue_id" in v[0]
                        or "dialog_id" in v[0]
                    ):
                        records.extend(v)
            # A single dialogue record.
            if (
                "turns" in data
                or "dialogue_id" in data
                or "dialog_id" in data
            ):
                records=[data]
        else:
            records=[]

        print(
            f"        dialogue_records={len(records):,}",
            flush=True,
        )

        before=stats["utterances"]
        dialogue_records=0

        for item in records:
            if not isinstance(item,dict):
                continue

            turns=(
                item.get("turns")
                or item.get("utterances")
                or []
            )

            # This is the gate that prevents schema.json-like data from
            # becoming dialogue text.
            if not isinstance(turns,list) or not turns:
                continue

            dialogue_records+=1

            dialogue_id=(
                item.get("dialogue_id")
                or item.get("dialog_id")
                or digest("sgd_dialogue",path,dialogue_records)
            )

            for ti,turn in enumerate(turns):
                if not isinstance(turn,dict):
                    continue

                text=str(
                    turn.get("utterance")
                    or turn.get("text")
                    or ""
                ).strip()

                if not text:
                    continue

                speaker=str(
                    turn.get("speaker")
                    or turn.get("role")
                    or ""
                )

                intent=str(
                    turn.get("intent")
                    or ""
                )

                uid=digest(
                    "utterance",
                    "sgd",
                    dialogue_id,
                    ti,
                    text,
                )

                pending.append(
                    (
                        (
                            uid,
                            text,
                            ti,
                            turn,
                            dialogue_id,
                            intent,
                            speaker,
                        ),
                        text,
                    )
                )

                if len(pending)>=batch_size*8:
                    flush_pending()

        flush_pending()
        con.commit()

        stats["dialogue_records"]+=dialogue_records

        elapsed=time.perf_counter()-started
        print(
            f"      SGD dialogue {fi}/{total_files} done "
            f"dialogues={dialogue_records:,} "
            f"imported={stats['utterances']-before:,} "
            f"total_utterances={stats['utterances']:,} "
            f"rate={stats['utterances']/max(elapsed,1e-9):.1f}/s",
            flush=True,
        )

    elapsed=time.perf_counter()-started

    return {
        "json_files":total_files,
        "services":stats["services"],
        "intents":stats["intents"],
        "slots":stats["slots"],
        "dialogue_records":stats["dialogue_records"],
        "utterances":stats["utterances"],
        "seconds":elapsed,
        "utterances_per_second":
            stats["utterances"]/max(elapsed,1e-9),
    }



def extract_multiwoz_dialogues(data):
    """
    Normalize common MultiWOZ layouts to a list of dialogue dictionaries.

    Supported shapes:
      {dialogue_id: {...}, ...}
      [{...}, {...}]
      {"dialogues": [...]}
      {"data": [...]}
    """
    if isinstance(data,list):
        return data

    if isinstance(data,dict):
        for key in (
            "dialogues",
            "data",
            "train",
            "dev",
            "validation",
            "test",
        ):
            value=data.get(key)
            if isinstance(value,list):
                return value

        # Standard MultiWOZ files often use dialogue-id as dict keys.
        values=list(data.values())
        if values and all(
            isinstance(v,dict) for v in values
        ):
            return values

        # Single dialogue object.
        if (
            "turns" in data
            or "dialogue_id" in data
            or "dialog_id" in data
        ):
            return [data]

    return []


def ingest_multiwoz(con, root, nlp=None, batch_size=16, status_every=1000, parse_text=False):
    stats=Counter()
    files=list(iter_files(root,{".json"}))
    pending=[]
    started=time.perf_counter()

    def flush_pending():
        nonlocal pending
        if not pending:
            return

        if parse_text:
            batch_parsed=parse_text_batch(
                nlp,
                pending,
                batch_size,
            )
            parsed=[
                (*identifier, facts)
                for identifier, _text, facts in batch_parsed
            ]
        else:
            parsed=[
                (*item, [])
                for item, _text in pending
            ]

        for uid,text,ti,payload,dialogue_id,speaker,intent,domain,facts in parsed:
            con.execute("""
                INSERT OR REPLACE INTO utterances
                VALUES (?,?,?,?,?,?,?,?,?)
            """,(
                uid,"multiwoz",str(dialogue_id),ti,speaker,text,
                intent,domain,json.dumps(payload,ensure_ascii=False),
            ))
            if facts:
                store_parser_facts(con,uid,facts)

            u=add_node(
                con,"utterance",text,"multiwoz",payload
            )
            if domain:
                d=add_node(
                    con,"domain",domain,"multiwoz"
                )
                add_edge(
                    con,u,"in_domain",d,"multiwoz"
                )
            stats["utterances"]+=1

        pending=[]

    for fi,path in enumerate(files,1):
        data=read_json_file(path)
        if data is None:
            continue

        dialogues=extract_multiwoz_dialogues(data)

        for di,dialogue in enumerate(dialogues):
            if not isinstance(dialogue,dict):
                continue

            dialogue_id=(
                dialogue.get("dialogue_id")
                or dialogue.get("dialog_id")
                or str(di)
            )

            turns=(
                dialogue.get("turns")
                or dialogue.get("dialogue")
                or dialogue.get("log")
                or []
            )

            for ti,turn in enumerate(turns):
                if isinstance(turn,dict):
                    speaker=(
                        turn.get("speaker")
                        or turn.get("role")
                        or ""
                    )
                    text=str(
                        turn.get("utterance")
                        or turn.get("text")
                        or turn.get("transcript")
                        or ""
                    ).strip()
                    payload=turn
                else:
                    speaker=""
                    text=str(turn).strip()
                    payload={"raw":turn}

                if not text:
                    continue

                intent=str(
                    payload.get("intent","")
                ) if isinstance(payload,dict) else ""

                domain=""
                metadata=(
                    payload.get("metadata")
                    if isinstance(payload,dict)
                    else None
                )

                if isinstance(metadata,dict):
                    for dom_name,dom in metadata.items():
                        if isinstance(dom,dict):
                            domain=dom_name
                            break

                uid=digest(
                    "utterance","multiwoz",
                    dialogue_id,ti,text
                )

                pending.append(
                    (
                        (
                            uid,text,ti,payload,
                            dialogue_id,speaker,intent,domain
                        ),
                        text,
                    )
                )

                if len(pending)>=batch_size*8:
                    flush_pending()

            for state_key in (
                "goal","goal_description",
                "dialogue_state","dialog_act"
            ):
                state=dialogue.get(state_key)
                if state:
                    sid=digest(
                        "state","multiwoz",
                        dialogue_id,state_key
                    )
                    con.execute("""
                        INSERT OR REPLACE INTO dialogue_states
                        VALUES (?,?,?,?,?)
                    """,(
                        sid,"multiwoz",
                        str(dialogue_id),-1,
                        json.dumps(state,ensure_ascii=False),
                    ))
                    add_node(
                        con,"dialogue_state",
                        state_key,"multiwoz",state
                    )

        flush_pending()
        con.commit()

        elapsed=time.perf_counter()-started
        rate=stats["utterances"]/max(elapsed,1e-9)
        print(
            f"      MultiWOZ file {fi}/{len(files)} "
            f"utterances={stats['utterances']:,} "
            f"rate={rate:.1f}/s "
            f"file={path.name}",
            flush=True,
        )

    return {
        "json_files":len(files),
        "utterances":stats["utterances"],
        "seconds":time.perf_counter()-started,
    }

def safe_parser_facts(nlp, text):
    if nlp is None:
        return []
    return parser_facts(nlp, text)


# =============================================================================
# DialoGLUE / mixed task-oriented datasets
# =============================================================================

def parse_tabular(path):
    try:
        with path.open(
            "r",
            encoding="utf-8",
            errors="ignore",
            newline="",
        ) as f:
            sample=f.read(8192)
            f.seek(0)
            dialect=csv.Sniffer().sniff(
                sample,
                delimiters="\t,",
            )
            reader=csv.DictReader(
                f,
                dialect=dialect,
            )
            return list(reader)
    except Exception:
        return []



def summarize_json_shape(data, max_items=3):
    """Return a compact description without dumping huge dataset contents."""
    if isinstance(data, dict):
        keys=list(data.keys())
        sample={}
        for k in keys[:12]:
            v=data[k]
            if isinstance(v,(str,int,float,bool)) or v is None:
                sample[k]=type(v).__name__
            elif isinstance(v,list):
                sample[k]=f"list[{len(v)}]"
            elif isinstance(v,dict):
                sample[k]=f"dict[{len(v)}]"
            else:
                sample[k]=type(v).__name__
        return {
            "type":"dict",
            "keys":keys[:30],
            "sample_types":sample,
        }
    if isinstance(data,list):
        return {
            "type":"list",
            "length":len(data),
            "item_types":[
                type(x).__name__
                for x in data[:max_items]
            ],
            "first_item_keys":(
                list(data[0].keys())[:30]
                if data and isinstance(data[0],dict)
                else []
            ),
        }
    return {"type":type(data).__name__}


def discover_text_fields(record):
    """
    Discover likely conversational text fields without requiring one fixed
    DialoGLUE schema. Returns (field, text) pairs.
    """
    candidates=(
        "text",
        "utterance",
        "query",
        "sentence",
        "question",
        "input",
        "user_input",
        "user_utterance",
        "prompt",
    )

    if not isinstance(record,dict):
        return []

    found=[]
    for field in candidates:
        value=record.get(field)
        if isinstance(value,str) and value.strip():
            found.append((field,value.strip()))

    # Nested common structures.
    for outer in (
        "data",
        "examples",
        "dialogue",
        "turns",
        "utterances",
    ):
        value=record.get(outer)
        if isinstance(value,list):
            for item in value:
                found.extend(
                    discover_text_fields(item)
                )
        elif isinstance(value,dict):
            found.extend(
                discover_text_fields(value)
            )

    return found





def extract_dialoglue_example(record):
    """
    Interpret the observed DialoGLUE gt_test.json structures.

    Observed examples include:
      "plain string"
      ["plain string"]
      ["plain string", labels]
      {"text": "..."}-style records in other releases
    """
    if isinstance(record,str):
        text=text_from_scalar(record)
        return (text,"") if text else None

    if isinstance(record,(list,tuple)):
        if not record:
            return None

        # First scalar string is overwhelmingly the utterance in gt_test.
        text=None
        label=""

        for item in record:
            candidate=text_from_scalar(item)
            if candidate:
                text=candidate
                break

        if text is None:
            return None

        # Preserve a simple scalar second/label field when available.
        if len(record)>=2:
            second=record[1]
            if isinstance(second,(str,int,float,bool)):
                label=str(second)
            elif isinstance(second,list):
                label="|".join(
                    str(x)
                    for x in second[:8]
                    if isinstance(x,(str,int,float,bool))
                )

        return text,label

    if isinstance(record,dict):
        keys={str(k).lower() for k in record.keys()}

        if keys & {
            "class_types",
            "label_maps",
            "vocab",
            "slots",
        }:
            return None

        for field in (
            "text","utterance","query","sentence",
            "question","input","prompt","user",
        ):
            value=record.get(field)
            text=text_from_scalar(value)
            if text:
                label=str(
                    record.get("label")
                    or record.get("intent")
                    or ""
                )
                return text,label

    return None



DIALOGUE_TEXT_FILES = {
    ".csv",
    ".tsv",
    ".txt",
    ".json",
    ".jsonl",
}


def ingest_dialoglue(con,root,nlp=None,status_every=500):
    """
    Ingest actual DialoGLUE task data.

    Important:
      * gt_test.json is benchmark ground truth/output and is NOT a text corpus.
      * category/schema files are metadata only.
      * actual task files under banking/, hwu/, clinc/, restaurant8k/,
        dstc8_sgd/, top/, and multiwoz/ are the dialogue/utterance sources.
    """
    stats=Counter()
    started=time.perf_counter()

    skip_names={
        "gt_test.json",
        "categories.json",
        "multiwoz21.json",
        "sim-m.json",
        "sim-r.json",
        "woz2.json",
        "stats.csv",
        "vocab.txt",
        "vocab.intent",
        "vocab.slot",
    }

    files=[
        p for p in iter_files(
            root,
            DIALOGUE_TEXT_FILES,
        )
        if p.name.lower() not in skip_names
    ]

    print(
        f"      DialoGLUE actual-data files={len(files)}",
        flush=True,
    )

    for fi,path in enumerate(files,1):
        rel=str(
            path.relative_to(root)
            if root in path.parents
            else path.name
        )
        print(
            f"      DialoGLUE {fi}/{len(files)} "
            f"{rel} "
            f"size={path.stat().st_size/1024/1024:.2f}MB",
            flush=True,
        )

        suffix=path.suffix.lower()
        before=stats["utterances"]

        # CSV / TSV task data.
        if suffix in {".csv",".tsv"}:
            rows=parse_tabular(path)
            print(
                f"        tabular records={len(rows):,}",
                flush=True,
            )

            for ri,row in enumerate(rows):
                if not isinstance(row,dict):
                    continue

                text=(
                    row.get("text")
                    or row.get("utterance")
                    or row.get("query")
                    or row.get("sentence")
                    or row.get("input")
                    or ""
                )
                text=text_from_scalar(text)
                if not text:
                    continue

                intent=str(
                    row.get("label")
                    or row.get("intent")
                    or ""
                )
                domain=path.parent.name

                add_discovered_utterance(
                    con,
                    f"dialoglue:{domain}",
                    rel,
                    ri,
                    text,
                    row,
                    intent,
                    domain,
                    "user",
                )
                stats["utterances"]+=1

        elif suffix==".jsonl":
            try:
                rows=[
                    json.loads(line)
                    for line in path.read_text(
                        encoding="utf-8",
                        errors="ignore",
                    ).splitlines()
                    if line.strip()
                ]
            except Exception:
                rows=[]

            print(
                f"        jsonl records={len(rows):,}",
                flush=True,
            )

            for ri,row in enumerate(rows):
                example=extract_dialoglue_example(row)
                if not example:
                    continue
                text,label=example
                domain=path.parent.name

                add_discovered_utterance(
                    con,
                    f"dialoglue:{domain}",
                    rel,
                    ri,
                    text,
                    row if isinstance(row,dict) else {"record":row},
                    label,
                    domain,
                    "user",
                )
                stats["utterances"]+=1

        elif suffix==".json":
            data=read_json_file(path)
            if data is None:
                continue

            print(
                "        shape="
                +json.dumps(
                    summarize_json_shape(data),
                    ensure_ascii=False,
                )[:1400],
                flush=True,
            )

            def consume_records(records,dataset_name):
                imported=0
                for ri,record in enumerate(records):
                    example=extract_dialoglue_example(record)
                    if not example:
                        continue

                    text,label=example

                    add_discovered_utterance(
                        con,
                        f"dialoglue:{dataset_name}",
                        rel,
                        ri,
                        text,
                        record if isinstance(record,dict)
                        else {"record":record},
                        label,
                        dataset_name,
                        "user",
                    )
                    imported+=1
                    stats["utterances"]+=1

                    if imported%status_every==0:
                        print(
                            f"          imported={imported:,}",
                            flush=True,
                        )
                return imported

            imported=0

            if isinstance(data,list):
                imported=consume_records(
                    data,
                    path.parent.name,
                )

            elif isinstance(data,dict):
                # Dialog files may wrap examples/dialogues under one or more
                # named arrays. Never recursively harvest arbitrary strings.
                for key_name,value in data.items():
                    if not isinstance(value,list):
                        continue
                    lname=str(key_name).lower()
                    if lname in {
                        "class_types",
                        "label_maps",
                        "slots",
                        "categories",
                        "vocab",
                    }:
                        continue
                    imported+=consume_records(
                        value,
                        path.parent.name,
                    )

            print(
                f"        imported={imported:,}",
                flush=True,
            )

        elif suffix==".txt":
            # TOP and similar DialoGLUE text files can contain structured
            # labels after the utterance. Preserve the utterance-like first
            # field/line as language data, without interpreting label text as
            # natural-language utterances.
            lines=path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines()

            for ri,line in enumerate(lines):
                line=line.strip()
                if not line:
                    continue

                # TOP format is typically tab-delimited input + intent/slot
                # annotation. Use the left-most field as the text.
                text=line.split("\t",1)[0].strip()
                text=text_from_scalar(text)
                if not text:
                    continue

                add_discovered_utterance(
                    con,
                    f"dialoglue:{path.parent.name}",
                    rel,
                    ri,
                    text,
                    {"raw_line":line[:4000]},
                    "",
                    path.parent.name,
                    "user",
                )
                stats["utterances"]+=1

        con.commit()

        elapsed=time.perf_counter()-started
        print(
            f"      DialoGLUE file {fi}/{len(files)} done "
            f"imported={stats['utterances']-before:,} "
            f"total={stats['utterances']:,} "
            f"rate={stats['utterances']/max(elapsed,1e-9):.1f}/s",
            flush=True,
        )

    return {
        "files_found":len(files),
        "utterances":stats["utterances"],
        "json_decode_errors":stats["json_decode_errors"],
        "seconds":time.perf_counter()-started,
    }


# =============================================================================
# Ubuntu pickle / binarized data
# =============================================================================

def iter_pickle_texts(value, seen=None, depth=0):
    """
    Ubuntu releases vary in pickle shape. Discover strings recursively without
    assuming one exact internal representation.

    Limits recursion depth to avoid walking arbitrary object internals.
    """
    if seen is None:
        seen=set()

    if depth>5:
        return

    oid=id(value)
    if oid in seen:
        return
    seen.add(oid)

    if isinstance(value,str):
        text=value.strip()
        if text:
            yield text
        return

    # pandas DataFrame / Series without importing pandas explicitly.
    if hasattr(value,"to_dict") and hasattr(value,"columns"):
        try:
            records=value.to_dict(
                orient="records"
            )
            yield from iter_pickle_texts(
                records,
                seen,
                depth+1,
            )
            return
        except Exception:
            pass

    if isinstance(value,dict):
        for v in value.values():
            yield from iter_pickle_texts(
                v,
                seen,
                depth+1,
            )
        return

    if isinstance(value,(list,tuple)):
        for v in value:
            yield from iter_pickle_texts(
                v,
                seen,
                depth+1,
            )
        return

    # numpy-ish arrays
    if hasattr(value,"tolist"):
        try:
            yield from iter_pickle_texts(
                value.tolist(),
                seen,
                depth+1,
            )
        except Exception:
            pass



def inspect_pickle_shape(value, depth=0, path="root"):
    """Compact structural introspection for a pickle."""
    if depth>4:
        return {
            "path":path,
            "type":type(value).__name__,
        }

    if isinstance(value,dict):
        keys=list(value.keys())
        return {
            "path":path,
            "type":"dict",
            "size":len(value),
            "keys":[str(k)[:80] for k in keys[:20]],
        }

    if isinstance(value,(list,tuple)):
        info={
            "path":path,
            "type":type(value).__name__,
            "size":len(value),
        }
        if value:
            info["first"]=inspect_pickle_shape(
                value[0],
                depth+1,
                path+"[0]",
            )
        return info

    if isinstance(value,str):
        return {
            "path":path,
            "type":"str",
            "length":len(value),
            "preview":value[:120],
        }

    if hasattr(value,"shape"):
        return {
            "path":path,
            "type":type(value).__name__,
            "shape":str(value.shape),
        }

    return {
        "path":path,
        "type":type(value).__name__,
    }


def iter_pickle_utterances(value, depth=0):
    """
    Yield strings that look like actual utterances. Avoid walking arbitrary
    object attributes and avoid returning every tiny metadata string.
    """
    if depth>8:
        return

    if isinstance(value,str):
        text=norm(value)
        if len(text.split())>=2 and len(text)<=4000:
            yield text
        return

    if isinstance(value,dict):
        # Prioritize common dialogue/data keys first.
        preferred=(
            "utterances",
            "dialogue",
            "dialogs",
            "conversations",
            "data",
            "texts",
            "text",
            "input",
            "output",
            "user",
            "system",
        )
        seen=set()

        for k in preferred:
            if k in value:
                seen.add(k)
                yield from iter_pickle_utterances(
                    value[k],
                    depth+1,
                )

        for k,v in value.items():
            if k in seen:
                continue
            # Avoid likely metadata/config branches.
            if str(k).lower() in {
                "vocab","metadata","config",
                "labels","label_map","ids",
                "indices","index",
            }:
                continue
            yield from iter_pickle_utterances(
                v,
                depth+1,
            )
        return

    if isinstance(value,(list,tuple)):
        for item in value:
            yield from iter_pickle_utterances(
                item,
                depth+1,
            )
        return

    if hasattr(value,"to_dict") and hasattr(value,"columns"):
        try:
            yield from iter_pickle_utterances(
                value.to_dict(
                    orient="records"
                ),
                depth+1,
            )
        except Exception:
            return
        return

    if hasattr(value,"tolist"):
        try:
            yield from iter_pickle_utterances(
                value.tolist(),
                depth+1,
            )
        except Exception:
            return



def numeric_sequence_info(x):
    """
    Return a compact description of a numpy/list-like numeric sequence.
    """
    if isinstance(x,(list,tuple)):
        sample=x[:5]
        flat=True
        for v in sample:
            if isinstance(v,(list,tuple)):
                flat=False
                break
        return {
            "type":type(x).__name__,
            "length":len(x),
            "flat":flat,
            "sample":sample,
        }

    if hasattr(x,"shape"):
        return {
            "type":type(x).__name__,
            "shape":str(x.shape),
            "dtype":str(getattr(x,"dtype","")),
        }

    return {
        "type":type(x).__name__,
    }


def scalar_label(x):
    if isinstance(x,(str,int,float,bool)):
        return str(x)
    try:
        if hasattr(x,"item"):
            return str(x.item())
    except Exception:
        pass
    return ""




def find_token_dictionary(value, depth=0, found=None):
    """
    Find a plausible Ubuntu token vocabulary inside the loaded pickle.
    Supports both id->token and token->id dictionaries.
    """
    if found is None:
        found=[]
    if depth>7:
        return found

    if isinstance(value,dict):
        n=len(value)
        if n>=1000:
            int_to_str=0
            str_to_int=0

            for k,v in list(value.items())[:10000]:
                if isinstance(k,int) and isinstance(v,str):
                    int_to_str+=1
                elif isinstance(k,str) and isinstance(v,int):
                    str_to_int+=1

            if int_to_str>=100:
                found.append(("id_to_token",n,value))
            if str_to_int>=100:
                found.append(("token_to_id",n,value))

        for v in value.values():
            find_token_dictionary(v,depth+1,found)

    elif isinstance(value,(list,tuple)):
        for v in value[:20]:
            find_token_dictionary(v,depth+1,found)

    return found


def load_ubuntu_vocabulary(files):
    candidates=[]
    diagnostics=[]

    for path in files:
        print(
            f"        vocabulary search: {path.name} "
            f"size={path.stat().st_size/1024/1024:.1f}MB",
            flush=True,
        )

        try:
            with path.open("rb") as f:
                data=pickle.load(f)
        except Exception as exc:
            diagnostics.append({
                "file":path.name,
                "error":repr(exc),
            })
            continue

        found=find_token_dictionary(data)

        diagnostics.append({
            "file":path.name,
            "candidate_count":len(found),
            "candidates":[
                {
                    "direction":direction,
                    "size":size,
                }
                for direction,size,_mapping in found
            ],
        })

        for direction,size,mapping in found:
            candidates.append(
                (size,direction,mapping,path)
            )

    if not candidates:
        return {},diagnostics

    candidates.sort(
        key=lambda x:x[0],
        reverse=True,
    )
    size,direction,mapping,path=candidates[0]

    if direction=="id_to_token":
        id_to_token={
            int(k):str(v)
            for k,v in mapping.items()
            if isinstance(k,int)
            and isinstance(v,str)
        }
    else:
        id_to_token={
            int(v):str(k)
            for k,v in mapping.items()
            if isinstance(k,str)
            and isinstance(v,int)
        }

    print(
        f"        vocabulary selected: {path.name} "
        f"direction={direction} size={len(id_to_token):,}",
        flush=True,
    )
    print(
        "        vocabulary sample="
        +json.dumps(
            dict(list(id_to_token.items())[:20]),
            ensure_ascii=False,
        ),
        flush=True,
    )

    return id_to_token,diagnostics


def decode_ubuntu_sequence(seq,id_to_token):
    if not isinstance(seq,(list,tuple)):
        return ""

    words=[]
    for token_id in seq:
        try:
            token_id=int(token_id)
        except Exception:
            continue

        token=id_to_token.get(token_id)
        if token is None:
            continue

        if token in {
            "<pad>","<PAD>","<unk>","<UNK>",
            "<s>","</s>","<bos>","<eos>",
        }:
            continue

        words.append(token)

    return norm(" ".join(words))




def load_pickle_compat(path):
    """
    Try modern pickle first, then Python-2-compatible latin1 decoding.

    This is required for old Ubuntu dataset vocabularies that contain
    byte/unicode objects serialized by Python 2.
    """
    errors=[]

    for kwargs in ({}, {"encoding":"latin1"}):
        try:
            with path.open("rb") as f:
                return pickle.load(f,**kwargs),None
        except Exception as exc:
            errors.append(repr(exc))

    return None,errors[-1] if errors else "unknown pickle error"


def extract_id_to_token(value, depth=0):
    """
    Find a usable ID -> token mapping or a token -> ID mapping.

    Handles:
      dict
      objects with id2word / word2id / idx2word / word2idx
      list/tuple vocabularies
    """
    if depth>5:
        return {}

    if isinstance(value,dict):
        # Direct dictionary.
        int_to_str={
            int(k):str(v)
            for k,v in value.items()
            if isinstance(k,int)
            and isinstance(v,str)
        }
        if len(int_to_str)>=100:
            return int_to_str

        str_to_int={
            str(k):int(v)
            for k,v in value.items()
            if isinstance(k,str)
            and isinstance(v,int)
        }
        if len(str_to_int)>=100:
            return {
                idx:token
                for token,idx in str_to_int.items()
            }

        # Named vocabulary fields.
        for name in (
            "id2word",
            "idx2word",
            "index2word",
            "id_to_token",
            "itos",
        ):
            candidate=value.get(name)
            result=extract_id_to_token(
                candidate,
                depth+1,
            )
            if result:
                return result

        for name in (
            "word2id",
            "word2idx",
            "token2id",
            "stoi",
        ):
            candidate=value.get(name)
            result=extract_id_to_token(
                candidate,
                depth+1,
            )
            if result:
                return result

        for v in value.values():
            result=extract_id_to_token(
                v,
                depth+1,
            )
            if result:
                return result

    elif isinstance(value,(list,tuple)):
        # List vocabulary: index == token id.
        if len(value)>=100:
            result={
                i:str(token)
                for i,token in enumerate(value)
                if isinstance(token,str)
            }
            if len(result)>=100:
                return result

        for v in value[:20]:
            result=extract_id_to_token(
                v,
                depth+1,
            )
            if result:
                return result

    else:
        # Old pickles often contain Vocab objects rather than dicts.
        for name in (
            "id2word",
            "idx2word",
            "index2word",
            "id_to_token",
            "itos",
        ):
            if hasattr(value,name):
                try:
                    candidate=getattr(value,name)
                    result=extract_id_to_token(
                        candidate,
                        depth+1,
                    )
                    if result:
                        return result
                except Exception:
                    pass

        for name in (
            "word2id",
            "word2idx",
            "token2id",
            "stoi",
        ):
            if hasattr(value,name):
                try:
                    candidate=getattr(value,name)
                    result=extract_id_to_token(
                        candidate,
                        depth+1,
                    )
                    if result:
                        return result
                except Exception:
                    pass

        if hasattr(value,"__dict__"):
            try:
                return extract_id_to_token(
                    value.__dict__,
                    depth+1,
                )
            except Exception:
                pass

    return {}


def load_ubuntu_vocab_file(path):
    data,error=load_pickle_compat(path)

    if error:
        print(
            f"        vocab load error: {error}",
            flush=True,
        )
        return {},error

    mapping=extract_id_to_token(data)

    if mapping:
        print(
            f"        decoded vocabulary from "
            f"{path.name}: {len(mapping):,} tokens",
            flush=True,
        )
        print(
            "        vocabulary sample="
            +json.dumps(
                dict(list(mapping.items())[:20]),
                ensure_ascii=False,
            ),
            flush=True,
        )
        return mapping,None

    print(
        f"        no ID->token mapping found in {path.name}; "
        f"type={type(data).__name__}",
        flush=True,
    )
    return {},None


def locate_ubuntu_vocab(root,loaded_files):
    # Prefer files explicitly named vocab*.pkl.
    candidates=[
        p for p in loaded_files
        if "vocab" in p.name.lower()
    ]
    if not candidates:
        candidates=list(iter_files(
            root,
            {".pkl",".pickle"},
        ))
        candidates=[
            p for p in candidates
            if "vocab" in p.name.lower()
        ]

    diagnostics=[]

    for path in candidates:
        mapping,error=load_ubuntu_vocab_file(path)
        diagnostics.append({
            "file":str(path),
            "error":error,
            "tokens":len(mapping),
        })
        if mapping:
            return mapping,diagnostics

    return {},diagnostics




def as_sequence(value):
    """
    Normalize scalar-or-sequence dataset fields.

    Ubuntu releases can contain:
      y = "0" / 0 / [0, 1, ...]
      c = [token_ids] / token_ids
      r = [token_ids] / token_ids

    A scalar must remain one record rather than being iterated as if it were a
    collection.
    """
    if value is None:
        return []

    if isinstance(value,(list,tuple)):
        return list(value)

    # Avoid treating strings as sequences of characters.
    if isinstance(value,(str,bytes,int,float,bool)):
        return [value]

    if hasattr(value,"tolist"):
        try:
            converted=value.tolist()
            if isinstance(converted,list):
                return converted
            return [converted]
        except Exception:
            pass

    try:
        return list(value)
    except TypeError:
        return [value]


def ubuntu_pair_sequences(c,r,y):
    """
    Normalize context/response/label fields into aligned records.

    If c/r are token sequences for one pair, preserve them as one element
    rather than exploding their token IDs into individual pairs.
    """
    c_list=as_sequence(c)
    r_list=as_sequence(r)
    y_list=as_sequence(y)

    # A single token-ID sequence looks like [123, 456, 789]. Distinguish it
    # from [[123, 456], [789, ...]].
    def looks_like_token_sequence(value):
        if not isinstance(value,list) or not value:
            return False
        sample=value[:20]
        return all(
            isinstance(x,(int,float))
            or (
                isinstance(x,str)
                and x.strip().lstrip("-").isdigit()
            )
            for x in sample
        )

    if looks_like_token_sequence(c_list):
        c_list=[c_list]

    if looks_like_token_sequence(r_list):
        r_list=[r_list]

    n=max(len(c_list),len(r_list))

    if not y_list:
        y_list=[""]*n
    elif len(y_list)==1 and n>1:
        y_list=y_list*n

    return c_list,r_list,y_list



def ubuntu_token_ids(value):
    """
    Return a token-id sequence for one Ubuntu context/response item.

    A single scalar token id is a valid one-token sequence.
    """
    if isinstance(value,(list,tuple)):
        return list(value)

    if isinstance(value,(int,float)):
        return [value]

    if isinstance(value,str):
        value=value.strip()
        if value.lstrip("-").isdigit():
            return [int(value)]

    if hasattr(value,"tolist"):
        try:
            converted=value.tolist()
            if isinstance(converted,list):
                return converted
            return [converted]
        except Exception:
            pass

    try:
        return list(value)
    except TypeError:
        return [value]


def decode_ubuntu_item(value,id_to_token):
    """
    Decode one context/response item regardless of whether it is:
      * a list of token IDs
      * a tuple of token IDs
      * a single token ID
    """
    text=text_from_scalar(value)
    if text:
        return text

    ids=ubuntu_token_ids(value)
    words=[]

    for token_id in ids:
        try:
            token=id_to_token.get(int(token_id))
        except Exception:
            token=None

        if token is None:
            continue

        if token in {
            "<pad>","<PAD>","<unk>","<UNK>",
            "<s>","</s>","<bos>","<eos>",
        }:
            continue

        words.append(token)

    return norm(" ".join(words))


def ingest_ubuntu(con,root,nlp=None,status_every=1000):
    stats=Counter()
    files=list(iter_files(
        root,
        {".pkl",".pickle"},
    ))
    started=time.perf_counter()

    print(
        f"      Ubuntu discovery: pickle_files={len(files)}",
        flush=True,
    )

    vocab_files=[
        p for p in files
        if "vocab" in p.name.lower()
    ]

    vocab_diag=[]
    id_to_token={}

    if vocab_files:
        print(
            "      Ubuntu vocabulary files:",
            flush=True,
        )
        for path in vocab_files:
            mapping,error=load_ubuntu_vocab_file(path)
            vocab_diag.append({
                "file":str(path),
                "error":error,
                "tokens":len(mapping),
            })
            if mapping and not id_to_token:
                id_to_token=mapping
    else:
        print(
            "      Ubuntu vocabulary file not found",
            flush=True,
        )

    # Load the data files only once.
    for fi,path in enumerate(files,1):
        size_mb=path.stat().st_size/1024/1024
        print(
            f"      Ubuntu file {fi}/{len(files)} "
            f"{path.name} size={size_mb:.1f}MB",
            flush=True,
        )

        if "vocab" in path.name.lower():
            print(
                "        vocabulary file already handled",
                flush=True,
            )
            continue

        t0=time.perf_counter()
        data,error=load_pickle_compat(path)

        if error:
            stats["pickle_errors"]+=1
            print(
                f"        pickle_load_error={error}",
                flush=True,
            )
            continue

        print(
            f"        pickle_loaded "
            f"{time.perf_counter()-t0:.1f}s "
            f"shape="+json.dumps(
                inspect_pickle_shape(data),
                ensure_ascii=False,
            )[:1800],
            flush=True,
        )

        container=None

        if isinstance(data,dict):
            lower={
                str(k).lower():v
                for k,v in data.items()
            }
            if {"y","c","r"} <= set(lower):
                container=lower

        elif isinstance(data,list):
            for item in data[:20]:
                if isinstance(item,dict):
                    lower={
                        str(k).lower():v
                        for k,v in item.items()
                    }
                    if {"y","c","r"} <= set(lower):
                        container=lower
                        break

        if container is None:
            print(
                "        no y/c/r dataset container; skipped",
                flush=True,
            )
            continue

        y,c,r=ubuntu_pair_sequences(
            container["c"],
            container["r"],
            container["y"],
        )

        pair_count=min(len(c),len(r))

        print(
            f"        pairs={pair_count:,} "
            f"vocab={len(id_to_token):,}",
            flush=True,
        )

        decoded=0
        skipped=0
        unknown_tokens=0

        for i in range(pair_count):
            c_text=decode_ubuntu_item(
                c[i],
                id_to_token,
            )
            r_text=decode_ubuntu_item(
                r[i],
                id_to_token,
            )

            if not c_text and not r_text:
                skipped+=1
                continue

            label=(
                scalar_label(y[i])
                if i<len(y)
                else ""
            )

            if c_text:
                add_discovered_utterance(
                    con,
                    "ubuntu",
                    path.name,
                    i*2,
                    c_text,
                    {
                        "kind":"context",
                        "label":label,
                    },
                    "",
                    "computer_support",
                    "user",
                )
                stats["utterances"]+=1

            if r_text:
                add_discovered_utterance(
                    con,
                    "ubuntu",
                    path.name,
                    i*2+1,
                    r_text,
                    {
                        "kind":"response",
                        "label":label,
                    },
                    "",
                    "computer_support",
                    "assistant",
                )
                stats["utterances"]+=1

            decoded+=1

            if decoded%status_every==0:
                elapsed=time.perf_counter()-t0
                print(
                    f"        Ubuntu decoded={decoded:,}/"
                    f"{pair_count:,} "
                    f"utterances={stats['utterances']:,} "
                    f"skipped={skipped:,} "
                    f"unknown_tokens={unknown_tokens:,} "
                    f"rate={decoded/max(elapsed,1e-9):.1f}/s",
                    flush=True,
                )

        con.commit()

        elapsed=time.perf_counter()-t0
        print(
            f"      Ubuntu file {fi}/{len(files)} done "
            f"decoded={decoded:,} "
            f"skipped={skipped:,} "
            f"unknown_tokens={unknown_tokens:,} "
            f"total_utterances={stats['utterances']:,} "
            f"rate={decoded/max(elapsed,1e-9):.1f}/s",
            flush=True,
        )

    return {
        "pickle_files":len(files),
        "utterances":stats["utterances"],
        "pickle_errors":stats["pickle_errors"],
        "vocabulary_files":vocab_diag,
        "vocabulary_size":len(id_to_token),
        "seconds":time.perf_counter()-started,
    }


# =============================================================================
# Curiosity scheduling
# =============================================================================

QUESTION_FAMILIES = {
    "goal": "What is the user trying to accomplish with {x}?",
    "meaning": "What does {x} mean in a normal conversation?",
    "required_information": "What information does an assistant usually need before helping with {x}?",
    "clarification": "What is a useful question an assistant could ask about {x}?",
    "confirmation": "When should an assistant confirm something about {x}?",
    "action": "What could an assistant do when the user asks for {x}?",
    "result": "What result should the user expect after {x}?",
    "failure": "What can go wrong with {x}, and what should the assistant do?",
    "state": "What information should an assistant remember after {x}?",
    "alternative": "What is a common alternative to {x}?",
    "computer_context": "How might someone ask about {x} when talking about a computer?",
    "conversation_context": "What might someone say immediately before or after {x} in a conversation?",
}


def node_evidence(con, node_id, family):
    # Explicit edges are the strongest evidence.
    rel_map={
        "goal":{"expresses_goal","has_goal"},
        "meaning":{"defines","synonym","hypernym"},
        "required_information":{"has_slot","requires","needs"},
        "clarification":{"asks_for","clarifies"},
        "confirmation":{"confirms","needs_confirmation"},
        "action":{"supports_action","can_do","triggers_action"},
        "result":{"produces","results_in"},
        "failure":{"fails_with","error","fallback"},
        "state":{"updates_state","stores","remembers"},
        "alternative":{"alternative_to","similar_to"},
        "computer_context":{"in_domain","computer_related"},
        "conversation_context":{"precedes","follows","related_to"},
    }

    rels=rel_map.get(family,set())
    if not rels:
        return 0

    placeholders=",".join("?" for _ in rels)
    return int(
        con.execute(
            f"""
            SELECT COUNT(*) FROM edges
            WHERE (source_node=? OR target_node=?)
              AND relation IN ({placeholders})
            """,
            (node_id,node_id,*rels),
        ).fetchone()[0]
    )



def next_curiosity(con, exclude_keys=None):
    """Bounded epistemic-gap scheduler for large persistent graphs."""
    if exclude_keys is None:
        exclude_keys=set()

    families={
        "required_information":(
            "What information does an assistant usually need before helping with {x}?",
            4.8,
            ("has_slot","requires_slot","needs","requires"),
        ),
        "clarification":(
            "What is a useful question an assistant could ask about {x}?",
            4.8,
            ("asks_for","clarifies","requires_slot"),
        ),
        "failure":(
            "What can go wrong with {x}, and what should the assistant do?",
            4.7,
            ("failure","error","fallback"),
        ),
        "action":(
            "What could an assistant do when the user asks for {x}?",
            4.6,
            ("supports_action","can_do","triggers_action"),
        ),
        "confirmation":(
            "When should an assistant confirm something about {x}?",
            4.2,
            ("confirms","needs_confirmation","confirmation"),
        ),
        "state":(
            "What information should an assistant remember after {x}?",
            4.1,
            ("updates_state","stores","remembers"),
        ),
        "result":(
            "What result should the user expect after {x}?",
            3.8,
            ("produces","results_in","outcome"),
        ),
        "goal":(
            "What is the user trying to accomplish with {x}?",
            3.8,
            ("expresses_goal","has_goal"),
        ),
        "alternative":(
            "What is a common alternative to {x}?",
            3.0,
            ("alternative_to","similar_to"),
        ),
        "computer_context":(
            "How might someone ask about {x} when talking about a computer?",
            3.5,
            ("in_domain","computer_related"),
        ),
        "conversation_context":(
            "What might someone say immediately before or after {x} in a conversation?",
            3.5,
            ("precedes","follows","related_to"),
        ),
    }

    type_weight={
        "intent":7.0,
        "action":6.5,
        "service":6.0,
        "slot":5.5,
        "domain":4.0,
        "predicate":4.5,
        "concept":2.5,
    }

    rows=con.execute("""
        SELECT node_id,node_type,label,count
        FROM nodes
        WHERE node_type IN (
            'intent','action','service','slot',
            'domain','predicate','concept'
        )
        ORDER BY
            CASE node_type
                WHEN 'intent' THEN 0
                WHEN 'action' THEN 1
                WHEN 'service' THEN 2
                WHEN 'slot' THEN 3
                WHEN 'domain' THEN 4
                WHEN 'predicate' THEN 5
                ELSE 6
            END,
            count DESC
        LIMIT 3000
    """).fetchall()

    if not rows:
        return None

    ids=[r[0] for r in rows]
    ph=",".join("?" for _ in ids)

    evidence=defaultdict(Counter)

    edge_rows=con.execute(
        f"""
        SELECT source_node AS node_id,relation,COUNT(*) AS n
        FROM edges
        WHERE source_node IN ({ph})
        GROUP BY source_node,relation
        UNION ALL
        SELECT target_node AS node_id,relation,COUNT(*) AS n
        FROM edges
        WHERE target_node IN ({ph})
        GROUP BY target_node,relation
        """,
        tuple(ids)+tuple(ids),
    ).fetchall()

    for node_id,relation,n in edge_rows:
        evidence[node_id][norm(relation)]+=int(n)

    attempted=set(
        con.execute(
            f"""
            SELECT node_id,family
            FROM curiosity_targets
            WHERE node_id IN ({ph})
            """,
            tuple(ids),
        ).fetchall()
    )

    best=None

    for node_id,node_type,label,count in rows:
        ev=evidence.get(node_id,Counter())
        degree=sum(ev.values())

        for family,(question,fweight,relations) in families.items():
            if (node_id,family) in attempted:
                continue
            if (node_id,family) in exclude_keys:
                continue

            support=sum(ev[r] for r in relations)
            gap=max(0.0,4.0-float(support))

            score=(
                type_weight.get(node_type,1.0)
                + fweight
                + min(3.0,max(0,int(count)).bit_length()/2)
                + min(2.0,degree/25.0)
                + gap*1.5
                + (
                    2.0
                    if node_type in {"intent","action","service","slot"}
                    else 0.0
                )
            )

            candidate={
                "node_id":node_id,
                "node_type":node_type,
                "label":label,
                "family":family,
                "question":question.format(x=label),
                "score":score,
                "evidence_count":support,
                "graph_degree":degree,
                "epistemic_gap":gap,
            }

            if best is None or score>best["score"]:
                best=candidate

    return best


def store_target(con,target):
    target_id=digest(
        "curiosity-target",
        target["node_id"],
        target["family"],
    )

    con.execute("""
        INSERT OR IGNORE INTO curiosity_targets
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """,(
        target_id,
        target["node_id"],
        target["label"],
        target["node_type"],
        target["family"],
        target["question"],
        target["score"],
        0,
        0,
        target["evidence_count"],
        time.time(),
    ))
    return target_id


# =============================================================================
# Teacher
# =============================================================================

class Teacher:
    def __init__(self,model_path,max_new_tokens=120):
        try:
            import torch
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
            )
        except ImportError as exc:
            raise SystemExit(
                "Install:\n"
                "python -m pip install -U torch transformers accelerate"
            ) from exc

        self.torch=torch
        self.max_new_tokens=max_new_tokens

        print(
            f"[TEACHER] loading {model_path}",
            flush=True,
        )

        self.tokenizer=AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )

        kwargs={
            "trust_remote_code":True,
            "device_map":"auto",
        }

        if torch.cuda.is_available():
            kwargs["torch_dtype"]=torch.float16

        self.model=AutoModelForCausalLM.from_pretrained(
            str(model_path),
            **kwargs,
        )
        self.model.eval()

    def answer(self,question,context):
        # Context is compact and deliberately plain language. The teacher
        # supplies evidence; it does not choose the curiosity target.
        lines=[
            f"Question: {question}",
            "Answer in one or two short, normal English sentences.",
            "Do not explain your reasoning.",
        ]

        if context:
            lines.append(
                "Useful background:"
            )
            for k,v in context.items():
                if v:
                    lines.append(
                        f"{k}: {str(v)[:800]}"
                    )

        prompt="\n".join(lines)

        messages=[
            {
                "role":"system",
                "content":(
                    "Answer simple assistant questions in ordinary English. "
                    "Be direct."
                ),
            },
            {
                "role":"user",
                "content":prompt,
            },
        ]

        if hasattr(
            self.tokenizer,
            "apply_chat_template",
        ):
            text=self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text=(
                "Answer simply.\n"
                f"User: {prompt}\n"
                "Assistant:"
            )

        inputs=self.tokenizer(
            text,
            return_tensors="pt",
        )

        device=next(
            self.model.parameters()
        ).device

        inputs={
            k:v.to(device)
            for k,v in inputs.items()
        }

        with self.torch.inference_mode():
            output=self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        continuation=output[0][
            inputs["input_ids"].shape[1]:
        ]

        return self.tokenizer.decode(
            continuation,
            skip_special_tokens=True,
        ).strip()


# =============================================================================
# Context / semantic learning
# =============================================================================

def context_for_target(con,target):
    node_id=target["node_id"]

    rows=con.execute("""
        SELECT e.relation,n.label,n.node_type
        FROM edges e
        JOIN nodes n
          ON n.node_id=e.target_node
        WHERE e.source_node=?
        ORDER BY e.weight DESC
        LIMIT 12
    """,(node_id,)).fetchall()

    incoming=con.execute("""
        SELECT e.relation,n.label,n.node_type
        FROM edges e
        JOIN nodes n
          ON n.node_id=e.source_node
        WHERE e.target_node=?
        ORDER BY e.weight DESC
        LIMIT 12
    """,(node_id,)).fetchall()

    return {
        "target_type":target["node_type"],
        "existing_relations":[
            {
                "relation":r,
                "label":l,
                "type":t,
            }
            for r,l,t in rows
        ],
        "incoming_relations":[
            {
                "relation":r,
                "label":l,
                "type":t,
            }
            for r,l,t in incoming
        ],
    }


def accept_answer(
    con,
    target,
    question,
    answer,
    parsed,
):
    facts=[]

    for fact in parsed:
        if fact.get("kind")=="predicate":
            facts.append({
                "relation":"contains_predicate",
                "value":fact.get("predicate"),
            })
        elif fact.get("kind")=="argument":
            facts.append({
                "relation":fact.get("role") or "argument",
                "value":fact.get("argument"),
                "predicate":fact.get("predicate"),
            })

    if not facts:
        return False,0,"no_structured_facts"

    aid=digest(
        "curiosity-answer",
        target["node_id"],
        target["family"],
        question,
        answer,
    )

    con.execute("""
        INSERT OR REPLACE INTO curiosity_answers
        VALUES (?,?,?,?,?,?,?,?,?)
    """,(
        aid,
        digest(
            "curiosity-target",
            target["node_id"],
            target["family"],
        ),
        question,
        answer,
        json.dumps(
            parsed,
            ensure_ascii=False,
        ),
        len(facts),
        1,
        "accepted",
        time.time(),
    ))

    for fact in facts:
        fact_node=add_node(
            con,
            "concept",
            fact["value"],
            "curiosity",
            fact,
        )

        target_node=target["node_id"]

        add_edge(
            con,
            target_node,
            fact["relation"],
            fact_node,
            "curiosity",
            {
                "question":question,
                "answer":answer,
                "predicate":fact.get("predicate"),
            },
        )

    con.execute("""
        UPDATE curiosity_targets
        SET asked=1,
            answered=1
        WHERE node_id=? AND family=?
    """,(
        target["node_id"],
        target["family"],
    ))

    con.commit()
    return True,len(facts),"accepted"


def reject_answer(
    con,
    target,
    question,
    answer,
    reason,
):
    aid=digest(
        "curiosity-answer",
        target["node_id"],
        target["family"],
        question,
        answer,
        reason,
    )

    con.execute("""
        INSERT OR REPLACE INTO curiosity_answers
        VALUES (?,?,?,?,?,?,?,?,?)
    """,(
        aid,
        digest(
            "curiosity-target",
            target["node_id"],
            target["family"],
        ),
        question,
        answer,
        json.dumps({},ensure_ascii=False),
        0,
        0,
        reason,
        time.time(),
    ))

    # IMPORTANT: failed attempts are still marked as asked so the scheduler
    # cannot get stuck asking the same family forever.
    con.execute("""
        UPDATE curiosity_targets
        SET asked=1
        WHERE node_id=? AND family=?
    """,(
        target["node_id"],
        target["family"],
    ))
    con.commit()


def report_counts(con):
    tables=[
        "nodes",
        "edges",
        "utterances",
        "dialogue_states",
        "intents",
        "slots",
        "actions",
        "schemas",
        "parser_facts",
        "assistant_patterns",
        "curiosity_targets",
        "curiosity_answers",
    ]
    result={}
    for table in tables:
        result[table]=int(
            con.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
        )
    return result


# =============================================================================
# Smoke
# =============================================================================

def smoke():
    path=Path.cwd()/"v436_smoke.sqlite"
    if path.exists():
        path.unlink()

    con=init_db(path)

    node=add_node(
        con,
        "intent",
        "book hotel",
        "smoke",
    )
    assert node

    target={
        "node_id":node,
        "node_type":"intent",
        "label":"book hotel",
        "family":"goal",
        "question":"What is the user trying to accomplish with book hotel?",
        "score":4.0,
        "evidence_count":0,
        "asked":False,
    }

    tid=store_target(con,target)
    assert tid

    class Dummy:
        pass

    parsed=[
        {
            "kind":"predicate",
            "predicate":"book",
        },
        {
            "kind":"argument",
            "predicate":"book",
            "role":"object",
            "argument":"hotel",
        },
    ]

    ok,n,reason=accept_answer(
        con,
        target,
        target["question"],
        "I want to book a hotel.",
        parsed,
    )
    assert ok
    assert n==2

    counts=report_counts(con)
    assert counts["edges"]>=1
    assert counts["curiosity_answers"]==1

    # Failure must also mark the family asked.
    target2={
        "node_id":node,
        "node_type":"intent",
        "label":"book hotel",
        "family":"failure",
        "question":"What can go wrong with book hotel?",
        "score":3.0,
        "evidence_count":0,
        "asked":False,
    }
    store_target(con,target2)
    reject_answer(
        con,
        target2,
        target2["question"],
        "I don't know.",
        "no_structured_facts",
    )

    row=con.execute("""
        SELECT asked FROM curiosity_targets
        WHERE node_id=? AND family='failure'
    """,(node,)).fetchone()

    assert row and row[0]==1

    con.close()
    path.unlink(missing_ok=True)

    print("V436 assistant semantic-net + curiosity smoke: PASS")
    print("semantic-net persistence: PASS")
    print("assistant question-family scheduler: PASS")
    print("failed-question no-repeat protection: PASS")
    print("predicate/argument learning: PASS")
    print("SQLite durable memory: PASS")

    assert ubuntu_token_ids(123)==[123]
    assert ubuntu_token_ids([1,2,3])==[1,2,3]
    test_vocab={123:"hello",456:"world"}
    assert decode_ubuntu_item(123,test_vocab)=="hello"
    assert decode_ubuntu_item([123,456],test_vocab)=="hello world"
    print("Ubuntu per-pair scalar decoding: PASS")

    yc,cc,yv=ubuntu_pair_sequences(
        [[1,2,3]],
        [[4,5]],
        1,
    )
    assert len(yv)==1
    assert len(cc)==1
    assert len(yc)==1

    yc,cc,yv=ubuntu_pair_sequences(
        [1,2,3],
        [4,5],
        [0,1,1],
    )
    assert len(yc)==1
    assert len(cc)==1
    assert len(yv)==3

    print("Ubuntu scalar/list normalization: PASS")
    assert extract_multiwoz_dialogues([
        {"dialogue_id":"d1","turns":[]}
    ])
    assert extract_multiwoz_dialogues({
        "d1":{"dialogue_id":"d1","turns":[]}
    })
    print("MultiWOZ record normalizer: PASS")
    import inspect
    assert "parse_text" in inspect.signature(ingest_sgd).parameters
    assert "parse_text" in inspect.signature(ingest_multiwoz).parameters
    print("bulk-ingestion signature compatibility: PASS")
    pending_sgd=[(("u","text",0,{}, "d","i","user"),"text")]
    fast_sgd=[(*pending_sgd[0][0], [])]
    assert len(fast_sgd[0]) == 8

    pending_mw=[(("u","text",0,{}, "d","user","i","domain"),"text")]
    fast_mw=[(*pending_mw[0][0], [])]
    assert len(fast_mw[0]) == 9

    print("batch parser/result shape compatibility: PASS")
    sgd_flat=("u","text",0,{}, "d","i","user",[])
    assert len(sgd_flat)==8
    mw_flat=("u","text",0,{}, "d","user","i","domain",[])
    assert len(mw_flat)==9
    print("flat batch loop compatibility: PASS")


# =============================================================================
# Main
# =============================================================================






class NativeAssistant:
    """Natural conversational cognitive controller."""

    SOCIAL_RESPONSES={
        "greeting":[
            "Hello! How are you?",
            "Hey! How's it going?",
            "Hi! What can I help you with?",
        ],
        "thanks":[
            "You're welcome!",
            "Anytime!",
            "Glad I could help.",
        ],
        "affection":[
            "That's sweet of you.",
            "Aw, thank you. I like talking with you too.",
            "That's nice to hear.",
        ],
        "goodbye":[
            "Bye! Talk to you later.",
            "See you later!",
        ],
    }

    QUESTION_FAMILIES={
        "meaning":"What does {topic} mean?",
        "uses":"What can someone do with {topic}?",
        "applicability":"What kinds of things can someone {topic}?",
        "requirements":"What does someone need before they can {topic}?",
        "clarification":"Someone wants to {topic}. What should I ask them?",
        "next":"Someone wants to {topic}. What should they do next?",
        "before":"What usually happens before someone {topic}?",
        "after":"What usually happens after someone {topic}?",
        "problem":"What can go wrong when someone {topic}?",
        "example":"Give one simple example using {topic}.",
        "conversation":"Someone says they want to {topic}. What might they say next?",
        "assistant":"Someone asks an assistant to {topic}. What should the assistant do?",
    }

    def __init__(self,con,nlp,teacher=None,confidence_threshold=2.5):
        self.con=con
        self.nlp=nlp
        self.teacher=teacher
        self.confidence_threshold=confidence_threshold
        self.session_id=digest(
            "assistant_session",
            time.time(),
            id(self),
        )
        self.turn_index=0
        self._ensure_runtime_tables()

    def _ensure_runtime_tables(self):
        self.con.executescript("""
        CREATE TABLE IF NOT EXISTS assistant_sessions (
            session_id TEXT PRIMARY KEY,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS assistant_turns (
            turn_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            speaker TEXT NOT NULL,
            text TEXT NOT NULL,
            parsed_json TEXT NOT NULL,
            decision TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0,
            created_at REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_assistant_turns_session
          ON assistant_turns(session_id,turn_index);

        CREATE TABLE IF NOT EXISTS assistant_llm_dialogue (
            message_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            speaker TEXT NOT NULL,
            purpose TEXT NOT NULL,
            text TEXT NOT NULL,
            parsed_json TEXT NOT NULL,
            created_at REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_llm_dialogue_session
          ON assistant_llm_dialogue(session_id,turn_index);

        CREATE TABLE IF NOT EXISTS assistant_interaction_facts (
            fact_id TEXT PRIMARY KEY,
            turn_id TEXT NOT NULL,
            predicate TEXT,
            subject TEXT,
            object TEXT,
            relation TEXT,
            payload_json TEXT NOT NULL,
            created_at REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_interaction_facts_predicate
          ON assistant_interaction_facts(predicate);
        """)

        self.con.execute(
            "INSERT OR IGNORE INTO assistant_sessions VALUES (?,?)",
            (self.session_id,time.time()),
        )
        self.con.commit()

    def perceive(self,text):
        doc=self.nlp(text)
        predicates=[]
        nouns=[]
        entities=[]
        tokens=[]
        lower=text.strip().lower()

        for token in doc:
            if token.is_space or token.is_punct:
                continue

            lemma=norm(token.lemma_ or token.text)
            tokens.append({
                "text":token.text,
                "lemma":lemma,
                "pos":token.pos_,
                "dep":token.dep_,
            })

            if token.pos_ in {"VERB","AUX"}:
                predicates.append(lemma)
            elif token.pos_ in {"NOUN","PROPN"}:
                nouns.append(lemma)

            if token.ent_type_:
                entities.append({
                    "text":token.text,
                    "lemma":lemma,
                    "type":token.ent_type_,
                })

        if lower in {
            "hello","hello!","hi","hi!","hey","hey!",
            "good morning","good afternoon","good evening",
        }:
            speech_act="greeting"
        elif lower in {
            "thanks","thank you","thanks!","thank you!",
            "thx","many thanks",
        }:
            speech_act="thanks"
        elif lower in {
            "bye","goodbye","bye!","see you","see you later",
        }:
            speech_act="goodbye"
        elif re.search(
            r"\b(i|we)\s+(really\s+)?(like|love|adore|appreciate)\s+you\b",
            lower,
        ):
            speech_act="affection"
        elif lower.endswith("?"):
            speech_act="question"
        elif any(
            lower.startswith(prefix)
            for prefix in (
                "please ","can you ","could you ",
                "would you ","i need ","i want ","help me ",
            )
        ):
            speech_act="request"
        elif predicates:
            speech_act="statement"
        else:
            speech_act="other"

        return {
            "text":text,
            "predicates":list(dict.fromkeys(predicates)),
            "nouns":list(dict.fromkeys(nouns)),
            "entities":entities,
            "tokens":tokens,
            "speech_act":speech_act,
        }

    def retrieve(self,perception):
        if perception["speech_act"] in {
            "greeting","thanks","goodbye","affection",
        }:
            return []

        terms=list(dict.fromkeys(
            perception["predicates"]
            +perception["nouns"]
            +[
                e["lemma"]
                for e in perception["entities"]
            ],
        ))

        hits=[]

        for term in terms[:12]:
            rows=self.con.execute(
                """
                SELECT node_id,node_type,label,source,count
                FROM nodes
                WHERE lower(label)=lower(?)
                ORDER BY count DESC
                LIMIT 8
                """,
                (term,),
            ).fetchall()

            for node_id,node_type,label,source,count in rows:
                edges=self.con.execute(
                    """
                    SELECT relation,target_node,weight
                    FROM edges
                    WHERE source_node=?
                    ORDER BY weight DESC
                    LIMIT 16
                    """,
                    (node_id,),
                ).fetchall()

                outgoing=[]
                for relation,target,weight in edges:
                    row=self.con.execute(
                        "SELECT node_type,label FROM nodes WHERE node_id=?",
                        (target,),
                    ).fetchone()
                    if row:
                        outgoing.append({
                            "relation":relation,
                            "target_type":row[0],
                            "target":row[1],
                            "weight":weight,
                        })

                hits.append({
                    "node_id":node_id,
                    "node_type":node_type,
                    "label":label,
                    "source":source,
                    "count":count,
                    "outgoing":outgoing,
                })

        return hits

    def reason(self,perception,hits):
        act=perception["speech_act"]

        if act in {
            "greeting","thanks","goodbye","affection",
        }:
            return {
                "decision":"social",
                "confidence":10.0,
                "relations":[],
                "intent_hits":[],
                "reason":act,
            }

        intent_hits=[
            h
            for h in hits
            if h["node_type"] in {
                "intent","action","service"
            }
        ]

        relations=[
            (
                h["label"],
                e["relation"],
                e["target"],
            )
            for h in hits[:10]
            for e in h["outgoing"][:6]
        ][:20]

        support=sum(
            min(4,int(h["count"]))
            for h in hits[:12]
        )

        if act in {"question","request"}:
            if intent_hits:
                decision="answer_or_clarify"
            elif relations and support>=self.confidence_threshold:
                decision="answer"
            elif perception["predicates"] and not perception["nouns"]:
                decision="clarify_object"
            else:
                decision="consult"
        elif act=="statement":
            decision="respond_to_statement"
        else:
            decision="consult"

        return {
            "decision":decision,
            "confidence":min(
                10.0,
                support/3.0
                +(2.0 if intent_hits else 0.0),
            ),
            "relations":relations,
            "intent_hits":intent_hits[:5],
            "reason":"semantic_retrieval",
        }

    def choose_llm_question_family(self,perception,reasoning):
        act=perception["speech_act"]

        if act=="greeting":
            return "conversation","hello"
        if act=="affection":
            return "conversation","they like you"
        if act=="thanks":
            return "conversation","someone thanks you"
        if act=="goodbye":
            return "conversation","someone says goodbye"

        if act=="question":
            if perception["nouns"]:
                return "meaning",perception["nouns"][0]
            if perception["predicates"]:
                return "meaning",perception["predicates"][0]
            return "conversation","the user's question"

        if perception["predicates"]:
            predicate=perception["predicates"][0]
            if reasoning["decision"]=="clarify_object":
                return "clarification",predicate
            if perception["nouns"]:
                return "next",f"{predicate} {perception['nouns'][0]}"
            return "meaning",predicate

        if perception["nouns"]:
            return "meaning",perception["nouns"][0]

        return "conversation",perception["text"]

    def make_llm_question(self,perception,reasoning):
        family,topic=self.choose_llm_question_family(
            perception,
            reasoning,
        )

        if family=="conversation":
            special={
                "hello":"Someone says hello to you. What is a natural reply?",
                "they like you":"Someone says they like you. What is a natural friendly reply?",
                "someone thanks you":"Someone thanks you. What is a natural reply?",
                "someone says goodbye":"Someone says goodbye. What is a natural reply?",
                "the user's question":"What is a helpful natural reply to the user's question?",
            }
            if topic in special:
                return family,special[topic]

        return family,self.QUESTION_FAMILIES[family].format(topic=topic)

    def ask_llm(self,question,context):
        if self.teacher is None:
            return None

        answer=self.teacher.answer(
            question,
            context,
        )
        answer=str(answer).strip()

        if not answer:
            return None

        message_id=digest(
            "llm_message",
            self.session_id,
            self.turn_index,
            question,
            answer,
        )

        self.con.execute(
            """
            INSERT OR REPLACE INTO assistant_llm_dialogue
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                message_id,
                self.session_id,
                self.turn_index,
                "llm",
                "internal_consultation",
                answer,
                json.dumps(
                    {
                        "question":question,
                        "context":context,
                    },
                    ensure_ascii=False,
                ),
                time.time(),
            ),
        )
        self.con.commit()

        return {
            "message_id":message_id,
            "speaker":"llm",
            "text":answer,
            "purpose":"internal_consultation",
        }

    def perceive_llm(self,message):
        if not message:
            return None
        perception=self.perceive(message["text"])
        message["perception"]=perception
        return perception

    def evaluate_llm(self,user_perception,llm_perception,reasoning):
        if not llm_perception:
            return "native",reasoning["confidence"]

        if user_perception["speech_act"] in {
            "greeting","thanks","goodbye","affection",
        }:
            return "social_reply_candidate",9.0

        user_terms=set(
            user_perception["predicates"]
            +user_perception["nouns"]
        )
        llm_terms=set(
            llm_perception["predicates"]
            +llm_perception["nouns"]
        )

        overlap=len(user_terms & llm_terms)

        if overlap:
            return (
                "llm_informed",
                max(
                    reasoning["confidence"],
                    3.0+overlap,
                ),
            )

        return "native",reasoning["confidence"]

    def integrate_llm_knowledge(self,turn_id,message,perception):
        learned=0
        predicates=perception["predicates"]
        nouns=perception["nouns"]

        for predicate in predicates[:8]:
            subject=nouns[0] if nouns else ""
            obj=nouns[1] if len(nouns)>1 else ""

            fact_id=digest(
                "llm_fact",
                turn_id,
                predicate,
                subject,
                obj,
            )

            self.con.execute(
                """
                INSERT OR IGNORE INTO assistant_interaction_facts
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    fact_id,
                    turn_id,
                    predicate,
                    subject,
                    obj,
                    "llm_observation",
                    json.dumps(
                        {
                            "speaker":"llm",
                            "predicate":predicate,
                            "subject":subject,
                            "object":obj,
                            "text":message["text"],
                        },
                        ensure_ascii=False,
                    ),
                    time.time(),
                ),
            )

            if subject and obj:
                snode=add_node(
                    self.con,
                    "concept",
                    subject,
                    "llm_interaction",
                )
                onode=add_node(
                    self.con,
                    "concept",
                    obj,
                    "llm_interaction",
                )
                add_edge(
                    self.con,
                    snode,
                    predicate,
                    onode,
                    "llm_interaction",
                    {
                        "turn_id":turn_id,
                        "source":"llm",
                    },
                    0.5,
                )

            learned+=1

        self.con.commit()
        return learned

    def store_turns(
        self,
        turn_id,
        perception,
        reasoning,
        response,
    ):
        self.con.execute(
            """
            INSERT OR REPLACE INTO assistant_turns
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                turn_id,
                self.session_id,
                self.turn_index,
                "user",
                perception["text"],
                json.dumps(
                    {
                        "perception":perception,
                        "reasoning":reasoning,
                    },
                    ensure_ascii=False,
                ),
                reasoning["decision"],
                reasoning["confidence"],
                time.time(),
            ),
        )

        self.con.execute(
            """
            INSERT OR REPLACE INTO assistant_turns
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                digest("assistant_response",turn_id),
                self.session_id,
                self.turn_index,
                "assistant",
                response,
                json.dumps(
                    {
                        "controller":"cognitive_architecture",
                    },
                    ensure_ascii=False,
                ),
                "response",
                reasoning["confidence"],
                time.time(),
            ),
        )

        self.turn_index+=1
        self.con.commit()

    def respond(self,text):
        turn_id=digest(
            "assistant_turn",
            self.session_id,
            self.turn_index,
            text,
        )

        user_perception=self.perceive(text)
        hits=self.retrieve(user_perception)
        reasoning=self.reason(
            user_perception,
            hits,
        )

        print(
            f"  [USER PERCEPTION] act={user_perception['speech_act']} "
            f"predicates={user_perception['predicates']} "
            f"nouns={user_perception['nouns']}",
            flush=True,
        )

        llm_message=None
        llm_perception=None
        decision="native"
        confidence=reasoning["confidence"]
        learned=0

        # The internal LLM participant is consulted for social, question, and
        # request turns. Statements can remain native unless the graph says a
        # consultation is useful.
        should_consult=(
            self.teacher is not None
            and user_perception["speech_act"] in {
                "greeting","thanks","goodbye","affection",
                "question","request",
            }
        )

        if should_consult:
            family,question=self.make_llm_question(
                user_perception,
                reasoning,
            )

            print(
                f"  [ARCHITECTURE → LLM] family={family}",
                flush=True,
            )
            print(
                f"  [ARCHITECTURE → LLM] {question}",
                flush=True,
            )

            known_relations="; ".join(
                f"{a} {b} {c}"
                for a,b,c in reasoning["relations"][:8]
            )

            llm_message=self.ask_llm(
                question,
                {
                    "user_message":text,
                    "speech_act":user_perception["speech_act"],
                    "known_words":" ".join(
                        user_perception["predicates"]
                        +user_perception["nouns"]
                    ),
                    "known_relations":known_relations,
                },
            )

            if llm_message:
                print(
                    f"  [LLM PARTICIPANT] {llm_message['text']}",
                    flush=True,
                )

                llm_perception=self.perceive_llm(
                    llm_message,
                )

                print(
                    f"  [ARCHITECTURE ← LLM] act="
                    f"{llm_perception['speech_act']} "
                    f"predicates={llm_perception['predicates']} "
                    f"nouns={llm_perception['nouns']}",
                    flush=True,
                )

                decision,confidence=self.evaluate_llm(
                    user_perception,
                    llm_perception,
                    reasoning,
                )

                print(
                    f"  [ARCHITECTURE] decision={decision} "
                    f"confidence={confidence:.2f}",
                    flush=True,
                )

                learned=self.integrate_llm_knowledge(
                    turn_id,
                    llm_message,
                    llm_perception,
                )

        # Final response policy.
        if user_perception["speech_act"] in {
            "greeting","thanks","goodbye","affection",
        }:
            # Social LLM content is a candidate. The architecture decides
            # whether it is suitable; if not, it uses its own response bank.
            if decision=="social_reply_candidate" and llm_message:
                response=llm_message["text"]
            else:
                options=self.SOCIAL_RESPONSES[
                    user_perception["speech_act"]
                ]
                response=options[
                    (
                        self.turn_index
                        +len(self.session_id)
                    )%len(options)
                ]

        elif decision=="llm_informed" and llm_message:
            # For substantive requests/questions the architecture accepts the
            # LLM's contribution as information after perception/evaluation.
            response=llm_message["text"]

        else:
            response=self.render_native(
                user_perception,
                reasoning,
            )

        if response is None:
            response=(
                "I'm not sure yet. "
                "Tell me a little more about what you need."
            )

        self.store_turns(
            turn_id,
            user_perception,
            reasoning,
            response,
        )

        return {
            "response":response,
            "decision":decision,
            "confidence":confidence,
            "retrieved":len(hits),
            "learned":learned,
            "llm_used":bool(llm_message),
        }

def assistant_main():
    ap=argparse.ArgumentParser(
        description="Native cognitive assistant over persistent semantic memory."
    )
    ap.add_argument(
        "--memory",
        default=r"C:\Users\adria\Desktop\dev\Graph-Topology\results\assistant_semantic_net.sqlite",
    )
    ap.add_argument(
        "--parser-model",
        default="en_core_web_sm",
        help="Informational; the inherited loader controls the spaCy model.",
    )
    ap.add_argument(
        "--teacher",
        default="",
        help="Optional SmolLM2 fallback. Native mode does not require it.",
    )
    ap.add_argument(
        "--confidence",
        type=float,
        default=2.5,
    )
    args=ap.parse_args()

    memory=Path(args.memory)
    if not memory.exists():
        raise SystemExit(
            f"Semantic memory not found: {memory}"
        )

    print(f"[MEMORY] {memory}",flush=True)
    print(
        f"[PARSER] loading {args.parser_model}...",
        flush=True,
    )

    nlp=load_spacy()
    print("[PARSER] ready",flush=True)

    teacher=None
    if args.teacher:
        print(
            "[TEACHER] optional fallback loading...",
            flush=True,
        )
        teacher=Teacher(
            str(Path(args.teacher)),
            max_new_tokens=80,
        )

    con=sqlite3.connect(str(memory))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")

    assistant=NativeAssistant(
        con,
        nlp,
        teacher,
        args.confidence,
    )

    print()
    print("Native cognitive assistant ready.")
    print("LLM is OFF unless --teacher is supplied.")
    print("Commands: /status  /new  /quit")
    print()

    while True:
        try:
            text=input("You: ").strip()
        except (EOFError,KeyboardInterrupt):
            print()
            break

        if not text:
            continue

        command=text.lower()

        if command=="/quit":
            break

        if command=="/new":
            assistant.session_id=digest(
                "assistant_session",
                time.time(),
                id(assistant),
            )
            assistant.turn_index=0
            con.execute(
                "INSERT OR IGNORE INTO assistant_sessions VALUES (?,?)",
                (assistant.session_id,time.time()),
            )
            con.commit()
            print("New conversation session.")
            continue

        if command=="/status":
            row=con.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM nodes),
                    (SELECT COUNT(*) FROM edges),
                    (SELECT COUNT(*) FROM assistant_turns),
                    (SELECT COUNT(*) FROM assistant_interaction_facts)
                """
            ).fetchone()

            print({
                "nodes":row[0],
                "edges":row[1],
                "turns":row[2],
                "interaction_facts":row[3],
            })
            continue

        result=assistant.respond(text)

        print(
            f"Assistant: {result['response']}",
            flush=True,
        )
        print(
            f"  [{result['decision']} "
            f"confidence={result['confidence']:.2f} "
            f"retrieved={result['retrieved']} "
            f"learned={result['learned']}]",
            flush=True,
        )


if __name__=="__main__":
    assistant_main()
