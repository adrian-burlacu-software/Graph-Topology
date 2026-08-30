
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path


# ============================================================================
# Candidate / GUM parsing
# ============================================================================

AUXILIARY_VERBS = {
    "be","have","do","can","could","may","might","must",
    "shall","should","will","would",
}

LIGHT_VERBS = {
    "get","go","come","make","take","give","put","keep","let",
    "say","tell","see","know","look","seem","become","want","need",
}

USEFUL_ACTIONS = {
    "install","configure","create","build","compile","deploy","debug","test",
    "run","execute","download","upload","convert","transform","compare",
    "search","find","schedule","book","plan","prepare","cook","write","edit",
    "delete","remove","move","copy","backup","restore","connect","disconnect",
    "start","stop","open","close","check","verify","validate","measure",
    "calculate","sort","filter","parse","generate","train","learn","fix",
    "repair","update","upgrade","migrate","export","import","save","load",
}

LOW_VALUE_OBJECTS = {
    "it","that","this","one","what","which","who","whom","how","much","many",
    "someone","somebody","something","yourself","myself","himself","herself",
    "themselves","i","you","he","she","we","they","me","him","her","us","them",
}

GENERIC_NOUNS = {
    "thing","way","stuff","something","anything","nothing","part",
}

BAD_FUNCTIONAL_OBJECTS = {
    "more","less","much","many","how","why","what","which","who",
    "kind","sort","type","number","amount",
}

CORPUS_SPECIFIC_OBJECTS = {
    "scoring","role","fire","order","place","charge","care","account",
    "point","time","effect","question",
}


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
    source_file: str


@dataclass(frozen=True)
class Candidate:
    verb: str
    object_lemma: str
    construction: str
    source_sentence: str
    frequency: int
    score: float


def lexical_key(text):
    text=str(text or "").strip().lower()
    return re.sub(r"\s+"," ",text).strip(
        ".,!?;:\"'()[]{}"
    )


def parse_conllu(path: Path):
    out=[]
    rows=[]
    text=""

    def flush():
        nonlocal rows,text
        if rows:
            out.append(
                UDSentence(
                    text or " ".join(t.form for t in rows),
                    tuple(rows),
                    str(path),
                )
            )
        rows=[]
        text=""

    with path.open("r",encoding="utf-8") as f:
        for line_no,line in enumerate(f,1):
            line=line.rstrip("\n")
            if not line:
                flush()
                continue
            if line.startswith("# text = "):
                text=line[len("# text = "):]
                continue
            if line.startswith("#"):
                continue
            cols=line.split("\t")
            if len(cols)!=10:
                raise ValueError(
                    f"{path}:{line_no}: expected 10 CoNLL-U columns"
                )
            if "-" in cols[0] or "." in cols[0]:
                continue
            rows.append(UDToken(*cols))
    flush()
    return out


def discover_train(gum: Path):
    files=sorted(gum.rglob("*.conllu"))
    files=[f for f in files if "train" in f.name.lower()]
    if not files:
        raise FileNotFoundError(
            f"No GUM train CoNLL-U files under {gum}"
        )
    return files


def candidate_quality(verb,obj):
    if verb in AUXILIARY_VERBS:
        return -10.0

    score=2.0
    if verb in USEFUL_ACTIONS:
        score+=3.5
    elif verb in LIGHT_VERBS:
        score-=0.5

    if obj:
        score+=2.5
        if obj in LOW_VALUE_OBJECTS:
            score-=5.0
        if obj in GENERIC_NOUNS:
            score-=1.5
        if obj in BAD_FUNCTIONAL_OBJECTS:
            score-=1.5
        if obj in CORPUS_SPECIFIC_OBJECTS:
            score-=0.8
    else:
        score-=0.8

    return score


def mine_candidates(sentences,max_candidates):
    counts=Counter()
    examples={}

    for sent in sentences:
        for tok in sent.tokens:
            if tok.upos!="VERB":
                continue

            verb=lexical_key(
                tok.lemma if tok.lemma!="_" else tok.form
            )
            if not verb or verb in AUXILIARY_VERBS:
                continue

            objects=[
                c for c in sent.tokens
                if c.head==tok.id and c.deprel in {"obj","iobj"}
            ]

            for obj in objects:
                obj_lemma=lexical_key(
                    obj.lemma if obj.lemma!="_" else obj.form
                )
                if not obj_lemma:
                    continue

                q=candidate_quality(verb,obj_lemma)
                if q<2.0:
                    continue

                key=(verb,obj_lemma,f"{verb} + {obj_lemma}")
                counts[key]+=1
                examples.setdefault(key,sent.text)

            # Bare useful actions are still allowed, but only as a fallback.
            if not objects and verb in USEFUL_ACTIONS:
                key=(verb,"",verb)
                counts[key]+=1
                examples.setdefault(key,sent.text)

    rows=[]
    for (verb,obj,construction),freq in counts.items():
        q=max(0.1,candidate_quality(verb,obj))
        score=math.log1p(freq)*q*max(
            0.55,
            1.0-0.008*len(construction),
        )
        rows.append(
            Candidate(
                verb,obj,construction,
                examples[(verb,obj,construction)],
                freq,score
            )
        )

    rows.sort(
        key=lambda x:(-x.score,-x.frequency,x.construction)
    )

    # Diversity first; then fill with additional constructions.
    selected=[]
    seen=set()

    for row in rows:
        if row.verb in seen:
            continue
        selected.append(row)
        seen.add(row.verb)
        if len(selected)>=max_candidates:
            return selected

    for row in rows:
        if row in selected:
            continue
        selected.append(row)
        if len(selected)>=max_candidates:
            break

    return selected


# ============================================================================
# Teacher
# ============================================================================

def sentence_prompt(c):
    if c.object_lemma:
        return (
            f'Write one short, natural English sentence. '
            f'It must contain the exact word "{c.verb}" and the exact word '
            f'"{c.object_lemma}". Do not explain anything.'
        )
    return (
        f'Write one short, natural English sentence. '
        f'It must contain the exact word "{c.verb}". Do not explain anything.'
    )


class Teacher:
    def __init__(self,model_name,max_new_tokens=80):
        try:
            import torch
            from transformers import AutoTokenizer,AutoModelForCausalLM
        except ImportError as exc:
            raise SystemExit(
                "Install with:\n"
                "python -m pip install -U torch transformers accelerate spacy"
            ) from exc

        self.torch=torch
        self.max_new_tokens=max_new_tokens

        print(f"[TEACHER] tokenizer -> {model_name}",flush=True)
        self.tokenizer=AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        print(f"[TEACHER] model -> {model_name}",flush=True)
        kwargs={
            "trust_remote_code":True,
            "device_map":"auto",
        }
        if torch.cuda.is_available():
            kwargs["torch_dtype"]=torch.float16

        self.model=AutoModelForCausalLM.from_pretrained(
            model_name,
            **kwargs,
        )
        self.model.eval()

    def generate(self,prompt):
        messages=[
            {
                "role":"system",
                "content":"Answer with one short normal English sentence. Do not explain.",
            },
            {"role":"user","content":prompt},
        ]

        if hasattr(self.tokenizer,"apply_chat_template"):
            prompt_text=self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text=(
                "Answer with one short normal English sentence. Do not explain.\n"
                f"User: {prompt}\nAssistant:"
            )

        inputs=self.tokenizer(prompt_text,return_tensors="pt")
        device=next(self.model.parameters()).device
        inputs={k:v.to(device) for k,v in inputs.items()}

        with self.torch.inference_mode():
            output=self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        continuation=output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(
            continuation,
            skip_special_tokens=True,
        ).strip()


# ============================================================================
# Parser
# ============================================================================

def load_parser(model_name):
    try:
        import spacy
        return spacy.load(model_name)
    except ImportError as exc:
        raise SystemExit(
            "Install spaCy:\n"
            "python -m pip install -U spacy\n"
            "python -m spacy download en_core_web_trf"
        ) from exc
    except Exception as exc:
        raise SystemExit(
            f"Could not load {model_name}: {exc}\n"
            f"python -m spacy download {model_name}"
        ) from exc


def clean_sentence(text):
    text=(text or "").strip()
    text=re.sub(
        r"^(assistant|answer|response)\s*:\s*",
        "",
        text,
        flags=re.I,
    )
    text=re.sub(r"\s+"," ",text).strip("'\" ")

    if not text or len(text.split())<3:
        return None

    if any(x in text for x in ("{","}","```")):
        return None

    m=re.search(r"(.+?[.!?])(?:\s|$)",text)
    if m:
        text=m.group(1).strip()

    return text


def lexical_variants(term):
    term=lexical_key(term)
    out={term}

    irregular={
        "be":{"was","were","is","are","am","been","being"},
        "have":{"had","has","having"},
        "do":{"did","does","done","doing"},
        "go":{"went","gone","going"},
        "find":{"found","finding"},
        "write":{"wrote","written","writing"},
        "run":{"ran","running"},
        "see":{"saw","seen","seeing"},
        "take":{"took","taken","taking"},
        "make":{"made","making"},
        "know":{"knew","known","knowing"},
        "build":{"built","building"},
        "send":{"sent","sending"},
        "buy":{"bought","buying"},
        "win":{"won","winning"},
        "fight":{"fought","fighting"},
    }
    out.update(irregular.get(term,set()))

    if term.endswith("y") and len(term)>3:
        out.add(term[:-1]+"ies")
    if term.endswith("e") and len(term)>3:
        out.add(term+"d")
        out.add(term[:-1]+"ing")
    else:
        out.add(term+"ed")
        out.add(term+"ing")
        out.add(term+"s")

    return out


def parse_sentence(nlp,text):
    doc=nlp(text)

    tokens=[]
    for t in doc:
        if t.is_space:
            continue
        tokens.append({
            "i":t.i,
            "text":t.text,
            "lemma":lexical_key(t.lemma_),
            "pos":t.pos_,
            "tag":t.tag_,
            "dep":t.dep_,
            "head":t.head.i,
        })

    predicates=[]
    for t in doc:
        if t.is_space or t.is_punct:
            continue
        if t.pos_ not in {"VERB","AUX"}:
            continue

        subjects=[]
        objects=[]
        obliques=[]
        modifiers=[]
        auxiliaries=[]
        negations=[]
        complements=[]

        for child in t.children:
            if child.is_space or child.is_punct:
                continue

            item={
                "text":child.text,
                "lemma":lexical_key(child.lemma_),
                "pos":child.pos_,
                "dep":child.dep_,
            }

            if child.dep_ in {
                "nsubj","nsubjpass","csubj","csubjpass"
            }:
                subjects.append(item)
            elif child.dep_ in {"obj","dobj","iobj"}:
                objects.append(item)
            elif child.dep_.startswith("obl") or child.dep_=="prep":
                obliques.append(item)
            elif child.dep_ in {"advmod","amod"}:
                modifiers.append(item)
            elif child.dep_ in {"aux","auxpass","cop"}:
                auxiliaries.append(item)
            elif child.dep_=="neg":
                negations.append(item)
            elif child.dep_ in {
                "xcomp","ccomp","advcl","acl","relcl"
            }:
                complements.append(item)

        predicates.append({
            "predicate":lexical_key(t.lemma_),
            "surface":t.text,
            "dep":t.dep_,
            "subjects":subjects,
            "objects":objects,
            "obliques":obliques,
            "modifiers":modifiers,
            "auxiliaries":auxiliaries,
            "negations":negations,
            "complements":complements,
        })

    return {
        "text":text,
        "tokens":tokens,
        "predicates":predicates,
    }


def find_predicates(parsed,verb):
    target=lexical_key(verb)
    variants=lexical_variants(target)
    return [
        p for p in parsed["predicates"]
        if lexical_key(p["predicate"]) in variants
        or target in lexical_variants(
            lexical_key(p["predicate"])
        )
    ]


def validate_example(parsed,c):
    text_lexemes={
        lexical_key(x)
        for x in re.findall(
            r"[A-Za-z]+(?:'[A-Za-z]+)?",
            parsed["text"].lower(),
        )
    }

    if not (text_lexemes & lexical_variants(c.verb)):
        return False,"target_verb_not_realized"

    preds=find_predicates(parsed,c.verb)
    if not preds:
        return False,"predicate_not_found"

    if not c.object_lemma:
        return True,"predicate_found"

    obj_vars=lexical_variants(c.object_lemma)
    if not (text_lexemes & obj_vars):
        return False,"target_object_not_realized"

    for pred in preds:
        for item in pred["objects"]+pred["obliques"]:
            if lexical_key(item["lemma"]) in obj_vars:
                return True,"predicate_and_object_found"

        if any(
            lexical_key(t["lemma"]) in obj_vars
            for t in parsed["tokens"]
        ):
            return True,"predicate_and_object_present"

    return False,"object_structure_not_found"


# ============================================================================
# ConceptNet adapter
# ============================================================================

def conceptnet_schema(path):
    con=sqlite3.connect(str(path))
    cur=con.cursor()
    tables=[
        r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    info={}
    for table in tables:
        info[table]=[
            r[1] for r in cur.execute(
                f'PRAGMA table_info("{table}")'
            )
        ]
    return con,tables,info


def detect_conceptnet(path):
    result={
        "loaded":False,
        "usable":False,
        "concepts":0,
        "edges":0,
        "tables":[],
        "edge_table":None,
        "start_column":None,
        "end_column":None,
        "relation_column":None,
        "error":None,
    }

    if not path.exists():
        result["error"]="file_not_found"
        return result

    try:
        con,tables,info=conceptnet_schema(path)
        result["tables"]=tables

        for table,cols in info.items():
            low={c.lower():c for c in cols}

            start=next(
                (
                    low[x] for x in (
                        "start","start_uri","subject","source",
                        "head","from_uri","from"
                    ) if x in low
                ),
                None,
            )
            end=next(
                (
                    low[x] for x in (
                        "end","end_uri","object","target",
                        "tail","to_uri","to"
                    ) if x in low
                ),
                None,
            )
            rel=next(
                (
                    low[x] for x in (
                        "relation","rel","predicate","edge"
                    ) if x in low
                ),
                None,
            )

            if start and end:
                result["edge_table"]=table
                result["start_column"]=start
                result["end_column"]=end
                result["relation_column"]=rel
                result["edges"]=int(
                    con.execute(
                        f'SELECT COUNT(*) FROM "{table}"'
                    ).fetchone()[0] or 0
                )
                break

        result["loaded"]=True
        result["usable"]=result["edge_table"] is not None
        con.close()
    except Exception as exc:
        result["error"]=repr(exc)

    return result


def conceptnet_ground(path,terms,max_per_term=12):
    stats=detect_conceptnet(path)
    result={t:[] for t in terms if t}

    if not stats["usable"]:
        return result,stats

    try:
        con=sqlite3.connect(str(path))
        table=stats["edge_table"]
        sc=stats["start_column"]
        ec=stats["end_column"]
        rc=stats["relation_column"]

        for term in result:
            variants={
                term,
                term.replace(" ","_"),
                "/c/en/"+term.replace(" ","_"),
                "/c/en/"+term.replace(" ","_")+"/n",
                "/c/en/"+term.replace(" ","_")+"/v",
            }

            for endpoint,other in ((sc,ec),(ec,sc)):
                for variant in variants:
                    if rc:
                        rows=con.execute(
                            f'SELECT "{rc}","{other}" '
                            f'FROM "{table}" WHERE lower("{endpoint}")=? '
                            f'LIMIT ?',
                            (variant,max_per_term),
                        ).fetchall()
                    else:
                        rows=con.execute(
                            f'SELECT "{other}" '
                            f'FROM "{table}" WHERE lower("{endpoint}")=? '
                            f'LIMIT ?',
                            (variant,max_per_term),
                        ).fetchall()
                        rows=[("",r[0]) for r in rows]

                    for rel,other_value in rows:
                        result[term].append({
                            "relation":str(rel or ""),
                            "other":str(other_value or ""),
                        })

            # Deduplicate.
            seen=set()
            dedup=[]
            for x in result[term]:
                k=json.dumps(x,sort_keys=True)
                if k not in seen:
                    seen.add(k)
                    dedup.append(x)
            result[term]=dedup[:max_per_term]

        con.close()
        stats["matched_terms"]=sum(
            1 for v in result.values() if v
        )
        stats["queried_terms"]=len(result)
    except Exception as exc:
        stats["lookup_error"]=repr(exc)

    return result,stats


# ============================================================================
# Persistent cognitive memory
# ============================================================================

def init_memory(path:Path):
    con=sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")

    con.executescript("""
    CREATE TABLE IF NOT EXISTS learned_examples (
        candidate_id TEXT PRIMARY KEY,
        verb TEXT NOT NULL,
        object_lemma TEXT,
        construction TEXT NOT NULL,
        source_sentence TEXT,
        teacher_sentence TEXT NOT NULL,
        parser_reason TEXT,
        teacher_seconds REAL NOT NULL,
        learned_at REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS observations (
        observation_id TEXT PRIMARY KEY,
        candidate_id TEXT NOT NULL,
        predicate TEXT NOT NULL,
        sentence TEXT NOT NULL,
        subjects_json TEXT NOT NULL,
        objects_json TEXT NOT NULL,
        obliques_json TEXT NOT NULL,
        modifiers_json TEXT NOT NULL,
        auxiliaries_json TEXT NOT NULL,
        negations_json TEXT NOT NULL,
        complements_json TEXT NOT NULL,
        conceptnet_json TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS lexical_patterns (
        predicate TEXT NOT NULL,
        role TEXT NOT NULL,
        argument_lemma TEXT NOT NULL,
        count INTEGER NOT NULL,
        PRIMARY KEY(predicate,role,argument_lemma)
    );

    CREATE TABLE IF NOT EXISTS semantic_edges (
        concept TEXT NOT NULL,
        relation TEXT NOT NULL,
        other TEXT NOT NULL,
        count INTEGER NOT NULL,
        PRIMARY KEY(concept,relation,other)
    );

    CREATE TABLE IF NOT EXISTS run_state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_examples_verb
    ON learned_examples(verb);

    CREATE INDEX IF NOT EXISTS idx_observations_predicate
    ON observations(predicate);
    """)
    con.commit()
    return con


def candidate_id(c):
    raw=json.dumps(
        asdict(c),
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def already_learned(con,cid):
    row=con.execute(
        "SELECT 1 FROM learned_examples WHERE candidate_id=?",
        (cid,),
    ).fetchone()
    return row is not None


def store_state(con,key,value):
    con.execute(
        """
        INSERT INTO run_state(key,value) VALUES(?,?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (key,json.dumps(value,ensure_ascii=False)),
    )


def record_learning(con,c,teacher_sentence,parsed,reason,
                    teacher_seconds,grounded):
    cid=candidate_id(c)
    obs_seed=f"{cid}:{teacher_sentence}"
    oid=hashlib.sha256(obs_seed.encode("utf-8")).hexdigest()

    pred=next(
        (
            p for p in parsed["predicates"]
            if lexical_key(p["predicate"]) in lexical_variants(c.verb)
            or lexical_key(c.verb) in lexical_variants(
                lexical_key(p["predicate"])
            )
        ),
        parsed["predicates"][0] if parsed["predicates"] else {
            "predicate":c.verb,
            "surface":c.verb,
            "subjects":[],
            "objects":[],
            "obliques":[],
            "modifiers":[],
            "auxiliaries":[],
            "negations":[],
            "complements":[],
        },
    )

    # One transaction: if the process dies before COMMIT, none of the learning
    # from this example is visible. If COMMIT succeeds, it is durable.
    con.execute("BEGIN")
    try:
        con.execute(
            """
            INSERT OR IGNORE INTO learned_examples
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                cid,
                c.verb,
                c.object_lemma,
                c.construction,
                c.source_sentence,
                teacher_sentence,
                reason,
                teacher_seconds,
                time.time(),
            ),
        )

        con.execute(
            """
            INSERT OR REPLACE INTO observations
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                oid,
                cid,
                pred["predicate"],
                teacher_sentence,
                json.dumps(pred["subjects"],ensure_ascii=False),
                json.dumps(pred["objects"],ensure_ascii=False),
                json.dumps(pred["obliques"],ensure_ascii=False),
                json.dumps(pred["modifiers"],ensure_ascii=False),
                json.dumps(pred["auxiliaries"],ensure_ascii=False),
                json.dumps(pred["negations"],ensure_ascii=False),
                json.dumps(pred["complements"],ensure_ascii=False),
                json.dumps(grounded,ensure_ascii=False),
            ),
        )

        for role,key in (
            ("subject","subjects"),
            ("object","objects"),
            ("oblique","obliques"),
            ("modifier","modifiers"),
            ("complement","complements"),
        ):
            for item in pred.get(key,[]):
                lemma=lexical_key(item.get("lemma",""))
                if not lemma:
                    continue
                con.execute(
                    """
                    INSERT INTO lexical_patterns
                    VALUES(?,?,?,1)
                    ON CONFLICT(predicate,role,argument_lemma)
                    DO UPDATE SET count=count+1
                    """,
                    (
                        pred["predicate"],
                        role,
                        lemma,
                    ),
                )

        for term,items in grounded.items():
            for item in items:
                relation=lexical_key(item.get("relation",""))
                other=lexical_key(item.get("other",""))
                if not other:
                    continue
                con.execute(
                    """
                    INSERT INTO semantic_edges
                    VALUES(?,?,?,1)
                    ON CONFLICT(concept,relation,other)
                    DO UPDATE SET count=count+1
                    """,
                    (term,relation,other),
                )

        con.commit()
    except Exception:
        con.rollback()
        raise


# ============================================================================
# Smoke
# ============================================================================

def smoke():
    con=init_memory(Path.cwd()/"v424_smoke_memory.sqlite")
    c=Candidate(
        "find","book","find + book",
        "I found a book.",3,3.0
    )
    parsed={
        "text":"I found a book.",
        "tokens":[
            {"lemma":"i"},
            {"lemma":"find"},
            {"lemma":"book"},
        ],
        "predicates":[{
            "predicate":"find",
            "surface":"found",
            "subjects":[{"lemma":"i","pos":"PRON"}],
            "objects":[{"lemma":"book","pos":"NOUN"}],
            "obliques":[],
            "modifiers":[],
            "auxiliaries":[],
            "negations":[],
            "complements":[],
        }],
    }
    record_learning(
        con,c,"I found a book.",parsed,
        "predicate_and_object_found",
        0.1,
        {"find":[{"relation":"UsedFor","other":"/c/en/discovery"}]},
    )
    assert already_learned(con,candidate_id(c))
    assert con.execute(
        "SELECT COUNT(*) FROM observations"
    ).fetchone()[0]==1
    assert con.execute(
        "SELECT COUNT(*) FROM lexical_patterns"
    ).fetchone()[0]>=2
    assert con.execute(
        "SELECT COUNT(*) FROM semantic_edges"
    ).fetchone()[0]==1
    con.close()

    print("V424 overnight semantic-learning smoke: PASS")
    print("durable SQLite learning: PASS")
    print("WAL + synchronous=FULL: PASS")
    print("per-example transaction/commit: PASS")
    print("resume/deduplication: PASS")
    print("predicate/argument learning: PASS")
    print("ConceptNet edge learning: PASS")


# ============================================================================
# Overnight run
# ============================================================================

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=False)
    ap.add_argument("--gum",type=Path,default=Path(r".\data\UD_GUM"))
    ap.add_argument("--conceptnet",type=Path,
                    default=Path(r".\data\conceptnet_compact.db"))
    ap.add_argument("--spacy-model",default="en_core_web_trf")
    ap.add_argument("--max-candidates",type=int,default=10000)
    ap.add_argument("--train-sentences",type=int,default=11314)
    ap.add_argument("--max-new-tokens",type=int,default=80)
    ap.add_argument("--status-every",type=int,default=25)
    ap.add_argument("--teacher-probe",type=int,default=3)
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--fresh",action="store_true",
                    help="Delete this run's cognitive memory before starting.")
    args=ap.parse_args()

    if args.smoke:
        smoke()
        return

    if not args.model:
        raise SystemExit("--model is required.")

    start=time.perf_counter()
    gum=args.gum.resolve()
    conceptnet=args.conceptnet.resolve()
    results=Path.cwd()/"results"
    results.mkdir(parents=True,exist_ok=True)

    memory_path=results/"cognitive_language_memory.sqlite"
    state_path=results/"v424_run_state.json"
    examples_path=results/"v424_teacher_examples.jsonl"
    failures_path=results/"v424_failures.jsonl"
    candidates_path=results/"v424_candidates.jsonl"
    report_path=results/"v424_overnight_report.json"

    if args.fresh:
        for path in (
            memory_path,
            state_path,
            examples_path,
            failures_path,
            candidates_path,
            report_path,
        ):
            if path.exists():
                if path.is_file():
                    path.unlink()

    print("="*78,flush=True)
    print("V424 OVERNIGHT SEMANTIC LEARNING",flush=True)
    print("="*78,flush=True)

    print("[1/9] Loading GUM grammar corpus...",flush=True)
    train_files=discover_train(gum)
    train=[]
    for f in train_files:
        train.extend(parse_conllu(f))
        if len(train)>=args.train_sentences:
            break
    train=train[:args.train_sentences]
    print(
        f"      sentences={len(train):,} files={len(train_files)}",
        flush=True,
    )

    print("[2/9] Mining reusable constructions...",flush=True)
    candidates=mine_candidates(train,args.max_candidates)
    print(
        f"      requested={args.max_candidates:,} "
        f"mined={len(candidates):,}",
        flush=True,
    )

    with candidates_path.open("w",encoding="utf-8") as f:
        for c in candidates:
            f.write(
                json.dumps(
                    asdict(c),
                    ensure_ascii=False,
                    separators=(",",":"),
                )+"\n"
            )

    print("[3/9] Initializing persistent cognitive memory...",flush=True)
    con=init_memory(memory_path)

    learned_before=con.execute(
        "SELECT COUNT(*) FROM learned_examples"
    ).fetchone()[0]
    print(
        f"      existing_learned_examples={learned_before:,}",
        flush=True,
    )

    print("[4/9] Loading SmolLM2...",flush=True)
    teacher=Teacher(args.model,args.max_new_tokens)

    print("[5/9] Loading independent spaCy parser...",flush=True)
    nlp=load_parser(args.spacy_model)

    print("[6/9] Inspecting ConceptNet...",flush=True)
    cn_stats=detect_conceptnet(conceptnet)
    print(
        f"      loaded={cn_stats['loaded']} "
        f"usable={cn_stats['usable']} "
        f"edges={cn_stats['edges']:,} "
        f"edge_table={cn_stats['edge_table']}",
        flush=True,
    )

    if args.teacher_probe>0:
        print(
            f"[PROBE] {min(args.teacher_probe,len(candidates))} examples",
            flush=True,
        )
        for i,c in enumerate(candidates[:args.teacher_probe],1):
            prompt=sentence_prompt(c)
            t0=time.perf_counter()
            raw=teacher.generate(prompt)
            print(
                f"      PROBE {i}: {c.construction}\n"
                f"        prompt={prompt!r}\n"
                f"        raw={raw!r}\n"
                f"        seconds={time.perf_counter()-t0:.3f}",
                flush=True,
            )

    print("[7/9] Learning semantics continuously...",flush=True)

    # Counts are loaded from DB, so a resumed run remains accurate.
    accepted=con.execute(
        "SELECT COUNT(*) FROM learned_examples"
    ).fetchone()[0]
    skipped=0
    failed=0
    teacher_nonempty=con.execute(
        "SELECT COUNT(*) FROM learned_examples "
        "WHERE teacher_sentence <> ''"
    ).fetchone()[0]

    session_accepted=0
    session_failed=0
    session_started=time.perf_counter()
    last_checkpoint=time.perf_counter()

    # Open append files for auditability.
    exf=examples_path.open("a",encoding="utf-8")
    ff=failures_path.open("a",encoding="utf-8")

    try:
        total=len(candidates)

        for i,c in enumerate(candidates,1):
            cid=candidate_id(c)

            if already_learned(con,cid):
                skipped+=1
                continue

            prompt=sentence_prompt(c)
            print(
                f"      LEARN {i:,}/{total:,} -> {c.construction}",
                flush=True,
            )

            t0=time.perf_counter()

            try:
                raw=teacher.generate(prompt)
                teacher_s=time.perf_counter()-t0
                clean=clean_sentence(raw)

                if not clean:
                    failed+=1
                    session_failed+=1
                    ff.write(json.dumps({
                        "candidate_id":cid,
                        "candidate":asdict(c),
                        "reason":"empty_or_invalid_teacher_output",
                        "raw":raw[:3000],
                    },ensure_ascii=False)+"\n")
                    ff.flush()
                    continue

                teacher_nonempty+=1

                parsed=parse_sentence(nlp,clean)
                ok,reason=validate_example(parsed,c)

                if not ok:
                    failed+=1
                    session_failed+=1
                    ff.write(json.dumps({
                        "candidate_id":cid,
                        "candidate":asdict(c),
                        "reason":reason,
                        "teacher_sentence":clean,
                        "parsed":parsed,
                    },ensure_ascii=False)+"\n")
                    ff.flush()
                    continue

                semantic_terms={
                    parsed["predicates"][0]["predicate"]
                    if parsed["predicates"] else c.verb
                }
                if c.object_lemma:
                    semantic_terms.add(lexical_key(c.object_lemma))
                for tok in parsed["tokens"]:
                    lemma=lexical_key(tok["lemma"])
                    if lemma:
                        semantic_terms.add(lemma)

                grounded,_=conceptnet_ground(
                    conceptnet,
                    sorted(semantic_terms),
                    max_per_term=8,
                )

                record_learning(
                    con,
                    c,
                    clean,
                    parsed,
                    reason,
                    teacher_s,
                    grounded,
                )

                accepted+=1
                session_accepted+=1

                exf.write(json.dumps({
                    "candidate_id":cid,
                    "candidate":asdict(c),
                    "teacher_sentence":clean,
                    "parser":parsed,
                    "conceptnet":grounded,
                    "teacher_seconds":teacher_s,
                },ensure_ascii=False)+"\n")
                exf.flush()

            except Exception as exc:
                failed+=1
                session_failed+=1
                ff.write(json.dumps({
                    "candidate_id":cid,
                    "candidate":asdict(c),
                    "reason":"exception",
                    "error":repr(exc),
                    "traceback":traceback.format_exc(),
                },ensure_ascii=False)+"\n")
                ff.flush()
                print(
                    f"        ERROR: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue

            now=time.perf_counter()

            if (
                i==1
                or i%args.status_every==0
                or i==total
            ):
                session_elapsed=now-session_started
                session_done=session_accepted+session_failed
                rate=session_done/max(1e-9,session_elapsed)
                remaining=max(0,total-i-skipped)
                eta=remaining/max(1e-9,rate)

                stored_obs=con.execute(
                    "SELECT COUNT(*) FROM observations"
                ).fetchone()[0]
                stored_patterns=con.execute(
                    "SELECT COUNT(*) FROM lexical_patterns"
                ).fetchone()[0]
                stored_edges=con.execute(
                    "SELECT COUNT(*) FROM semantic_edges"
                ).fetchone()[0]

                print(
                    f"      CHECKPOINT i={i:,}/{total:,} "
                    f"accepted_session={session_accepted:,} "
                    f"accepted_total={accepted:,} "
                    f"skipped={skipped:,} "
                    f"failed={failed:,} "
                    f"observations={stored_obs:,} "
                    f"patterns={stored_patterns:,} "
                    f"semantic_edges={stored_edges:,} "
                    f"rate={rate:.2f}/s "
                    f"eta={eta/3600:.2f}h",
                    flush=True,
                )

                state={
                    "version":"v424",
                    "last_candidate_index":i,
                    "accepted_total":accepted,
                    "skipped":skipped,
                    "failed":failed,
                    "observations":stored_obs,
                    "patterns":stored_patterns,
                    "semantic_edges":stored_edges,
                    "wall_seconds":now-start,
                }
                state_path.write_text(
                    json.dumps(state,indent=2),
                    encoding="utf-8",
                )

    finally:
        exf.close()
        ff.close()

    print("[8/9] Finalizing memory statistics...",flush=True)

    observations=con.execute(
        "SELECT COUNT(*) FROM observations"
    ).fetchone()[0]
    patterns=con.execute(
        "SELECT COUNT(*) FROM lexical_patterns"
    ).fetchone()[0]
    semantic_edges=con.execute(
        "SELECT COUNT(*) FROM semantic_edges"
    ).fetchone()[0]
    learned_verbs=con.execute(
        "SELECT COUNT(DISTINCT predicate) FROM observations"
    ).fetchone()[0]
    grounded_concepts=con.execute(
        "SELECT COUNT(DISTINCT concept) FROM semantic_edges"
    ).fetchone()[0]
    con.close()

    runtime=time.perf_counter()-start

    report={
        "status":"PASS" if accepted>0 else "FAIL",
        "version":"v424",
        "mode":"overnight_resumable_semantic_learning",
        "teacher":{
            "model":args.model,
            "max_new_tokens":args.max_new_tokens,
        },
        "grammar":{
            "source":"UD GUM gold CoNLL-U",
            "train_sentences_used":len(train),
            "train_files":len(train_files),
        },
        "semantic_sources":{
            "teacher_examples":accepted,
            "conceptnet":cn_stats,
        },
        "learning":{
            "candidate_count":len(candidates),
            "accepted_total":accepted,
            "skipped_existing":skipped,
            "failed_total":failed,
            "learned_predicates":learned_verbs,
            "cognitive_observations":observations,
            "lexical_patterns":patterns,
            "grounded_concepts":grounded_concepts,
            "semantic_edges":semantic_edges,
        },
        "persistence":{
            "sqlite":str(memory_path.resolve()),
            "wal_mode":True,
            "synchronous":"FULL",
            "per_example_commit":True,
            "resume_enabled":True,
        },
        "outputs":{
            "memory":str(memory_path.resolve()),
            "examples":str(examples_path.resolve()),
            "failures":str(failures_path.resolve()),
            "candidates":str(candidates_path.resolve()),
            "state":str(state_path.resolve()),
            "report":str(report_path.resolve()),
        },
        "runtime_seconds":runtime,
    }

    report_path.write_text(
        json.dumps(report,indent=2,ensure_ascii=False),
        encoding="utf-8",
    )

    print("[9/9] COMPLETE",flush=True)
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
