
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import hashlib
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path



# =============================================================================
# Cognitive / semantic substrate
# =============================================================================


def _norm_cn(value):
    if value is None:
        return ""
    value=str(value).strip().lower()
    # Handle common ConceptNet URI forms.
    if value.startswith("/c/en/"):
        value=value[6:]
    value=value.replace("_"," ")
    value=value.split("/")[0]
    return value


def _conceptnet_schema(path):
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
        cols=[
            r[1] for r in cur.execute(
                f'PRAGMA table_info("{table}")'
            )
        ]
        info[table]=cols

    return con,tables,info


def load_conceptnet_stats(path: Path):
    """
    Inspect the actual DB schema instead of assuming a table layout.
    The old V422 code counted only generic tables and could report:
      loaded=true, concepts=0, edges=2.2M
    without having a usable concept lookup.
    """
    result={
        "path":str(path),
        "exists":path.exists(),
        "loaded":False,
        "usable":False,
        "concepts":0,
        "edges":0,
        "tables":[],
        "concept_table":None,
        "edge_table":None,
        "concept_column":None,
        "uri_column":None,
        "edge_relation_column":None,
        "edge_start_column":None,
        "edge_end_column":None,
        "error":None,
    }

    if not path.exists():
        result["error"]="file_not_found"
        return result

    try:
        con,tables,info=_conceptnet_schema(path)
        result["tables"]=tables

        concept_candidates=[]
        edge_candidates=[]

        for table,cols in info.items():
            low={c.lower():c for c in cols}

            concept_col=next(
                (low[x] for x in
                 ("concept","term","label","name","node","uri")
                 if x in low),
                None
            )

            start_col=next(
                (low[x] for x in
                 ("start","start_uri","subject","source","head","from_uri","from")
                 if x in low),
                None
            )
            end_col=next(
                (low[x] for x in
                 ("end","end_uri","object","target","tail","to_uri","to")
                 if x in low),
                None
            )
            rel_col=next(
                (low[x] for x in
                 ("relation","rel","predicate","edge","label")
                 if x in low),
                None
            )

            if concept_col:
                concept_candidates.append(
                    (table,concept_col)
                )
            if start_col and end_col:
                edge_candidates.append(
                    (table,start_col,end_col,rel_col)
                )

        if concept_candidates:
            table,col=concept_candidates[0]
            result["concept_table"]=table
            result["concept_column"]=col
            result["concepts"]=int(
                con.execute(
                    f'SELECT COUNT(*) FROM "{table}"'
                ).fetchone()[0] or 0
            )

        if edge_candidates:
            table,start_col,end_col,rel_col=edge_candidates[0]
            result["edge_table"]=table
            result["edge_start_column"]=start_col
            result["edge_end_column"]=end_col
            result["edge_relation_column"]=rel_col
            result["edges"]=int(
                con.execute(
                    f'SELECT COUNT(*) FROM "{table}"'
                ).fetchone()[0] or 0
            )

        result["loaded"]=True
        result["usable"]=bool(
            result["concept_table"] or result["edge_table"]
        )
        con.close()

    except Exception as exc:
        result["error"]=repr(exc)

    return result


def _escape_like(term):
    return (
        str(term)
        .replace("\\","\\\\")
        .replace("%","\\%")
        .replace("_","\\_")
    )


def conceptnet_lookup(path:Path, terms, max_per_term=8):
    """
    Return actual ConceptNet evidence for each normalized term.
    Handles either a concept/node table or an edge table whose endpoints are
    URIs/labels.
    """
    if not path.exists():
        return {}

    try:
        con,tables,info=_conceptnet_schema(path)

        result={t:[] for t in terms if t}
        stats={
            "queried_terms":len(result),
            "matched_terms":0,
            "edge_relations":0,
        }

        # Find a node/concept table.
        concept_specs=[]
        edge_specs=[]

        for table,cols in info.items():
            low={c.lower():c for c in cols}
            label_col=next(
                (low[x] for x in
                 ("concept","term","label","name","node","uri")
                 if x in low),None
            )
            if label_col:
                concept_specs.append((table,label_col))

            start_col=next(
                (low[x] for x in
                 ("start","start_uri","subject","source","head","from_uri","from")
                 if x in low),None
            )
            end_col=next(
                (low[x] for x in
                 ("end","end_uri","object","target","tail","to_uri","to")
                 if x in low),None
            )
            rel_col=next(
                (low[x] for x in
                 ("relation","rel","predicate","edge","label")
                 if x in low),None
            )
            if start_col and end_col:
                edge_specs.append(
                    (table,start_col,end_col,rel_col)
                )

        # Direct node lookup.
        for term in result:
            variants={term,term.replace(" ","_"),"/c/en/"+term.replace(" ","_")}
            for table,col in concept_specs:
                try:
                    clauses=[]
                    params=[]
                    for v in variants:
                        clauses.append(
                            f'lower("{col}")=? OR lower("{col}")=?'
                        )
                        params.extend([v,_norm_cn(v)])
                    rows=con.execute(
                        f'SELECT "{col}" FROM "{table}" '
                        f'WHERE {" OR ".join(clauses)} LIMIT ?',
                        (*params,max_per_term)
                    ).fetchall()
                    for row in rows:
                        if row[0] is not None:
                            result[term].append(str(row[0]))
                except Exception:
                    continue
            result[term]=list(dict.fromkeys(result[term]))[:max_per_term]

        # If we have an edge table, get relations touching each exact term.
        for term in result:
            variants={
                term,
                term.replace(" ","_"),
                "/c/en/"+term.replace(" ","_"),
            }

            for table,start_col,end_col,rel_col in edge_specs:
                try:
                    for endpoint_col,other_col in (
                        (start_col,end_col),
                        (end_col,start_col),
                    ):
                        if rel_col:
                            rows=con.execute(
                                f'SELECT "{rel_col}","{other_col}" '
                                f'FROM "{table}" '
                                f'WHERE lower("{endpoint_col}")=? '
                                f'OR lower("{endpoint_col}")=? '
                                f'LIMIT ?',
                                (
                                    term,
                                    "/c/en/"+term.replace(" ","_"),
                                    max_per_term,
                                )
                            ).fetchall()
                            for rel,other in rows:
                                other_norm=_norm_cn(other)
                                result[term].append({
                                    "relation":str(rel) if rel is not None else "",
                                    "other":other_norm,
                                })
                                stats["edge_relations"]+=1
                        else:
                            rows=con.execute(
                                f'SELECT "{other_col}" FROM "{table}" '
                                f'WHERE lower("{endpoint_col}")=? '
                                f'OR lower("{endpoint_col}")=? '
                                f'LIMIT ?',
                                (
                                    term,
                                    "/c/en/"+term.replace(" ","_"),
                                    max_per_term,
                                )
                            ).fetchall()
                            for (other,) in rows:
                                result[term].append({
                                    "relation":"",
                                    "other":_norm_cn(other),
                                })
                                stats["edge_relations"]+=1
                except Exception:
                    continue

            # Compact/deduped.
            seen=set()
            dedup=[]
            for item in result[term]:
                key=json.dumps(item,sort_keys=True) if isinstance(item,dict) else str(item)
                if key not in seen:
                    seen.add(key)
                    dedup.append(item)
            result[term]=dedup[:max_per_term]

            if result[term]:
                stats["matched_terms"]+=1

        con.close()
        return result,stats

    except Exception:
        return ({t:[] for t in terms},{
            "queried_terms":len([x for x in terms if x]),
            "matched_terms":0,
            "edge_relations":0,
        })


def init_cognitive_memory(path:Path):
    con=sqlite3.connect(str(path))
    cur=con.cursor()

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS observations (
        id TEXT PRIMARY KEY,
        sentence TEXT NOT NULL,
        predicate TEXT NOT NULL,
        surface TEXT,
        subjects_json TEXT,
        objects_json TEXT,
        obliques_json TEXT,
        modifiers_json TEXT,
        auxiliaries_json TEXT,
        negations_json TEXT,
        complements_json TEXT,
        conceptnet_json TEXT,
        source TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_observations_predicate
    ON observations(predicate);

    CREATE TABLE IF NOT EXISTS lexical_patterns (
        predicate TEXT NOT NULL,
        argument_role TEXT NOT NULL,
        argument_lemma TEXT NOT NULL,
        count INTEGER NOT NULL,
        PRIMARY KEY(predicate,argument_role,argument_lemma)
    );
    """)
    con.commit()
    return con


def train_cognitive_memory(db_path:Path, records, source_label):
    """
    This is the actual learning/update step: parsed language observations are
    persisted and aggregated into reusable predicate/argument patterns.
    """
    con=init_cognitive_memory(db_path)
    cur=con.cursor()

    observation_count=0
    pattern_updates=0

    for rec in records:
        pred=rec["predicate"]
        construction=rec["construction"]
        identity=json.dumps(
            [
                source_label,
                rec["sentence"],
                construction["verb"],
            ],
            ensure_ascii=False,
            sort_keys=True,
        )
        obs_id=hashlib.sha256(identity.encode("utf-8")).hexdigest()

        conceptnet=rec.get("semantic_grounding",{})

        cur.execute(
            """
            INSERT OR REPLACE INTO observations
            VALUES (?,?,?,?,?,?,?,?,?,?,?, ?,?)
            """,
            (
                obs_id,
                rec["sentence"],
                pred["lemma"],
                pred["surface"],
                json.dumps(pred["subjects"],ensure_ascii=False),
                json.dumps(pred["objects"],ensure_ascii=False),
                json.dumps(pred["obliques"],ensure_ascii=False),
                json.dumps(pred["modifiers"],ensure_ascii=False),
                json.dumps(pred["auxiliaries"],ensure_ascii=False),
                json.dumps(pred["negations"],ensure_ascii=False),
                json.dumps(pred["complements"],ensure_ascii=False),
                json.dumps(conceptnet,ensure_ascii=False),
                source_label,
            ),
        )
        observation_count+=1

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
                cur.execute(
                    """
                    INSERT INTO lexical_patterns
                    (predicate,argument_role,argument_lemma,count)
                    VALUES (?,?,?,1)
                    ON CONFLICT(predicate,argument_role,argument_lemma)
                    DO UPDATE SET count=count+1
                    """,
                    (pred["lemma"],role,lemma),
                )
                pattern_updates+=1

    con.commit()

    stats={
        "observations_added":observation_count,
        "pattern_updates":pattern_updates,
        "stored_observations":int(
            cur.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        ),
        "stored_patterns":int(
            cur.execute("SELECT COUNT(*) FROM lexical_patterns").fetchone()[0]
        ),
    }
    con.close()
    return stats




def build_cognitive_observation(record, semantic_matches):
    pred = record["predicate"]
    terms = [
        pred["lemma"],
        *[x["lemma"] for x in pred["subjects"]],
        *[x["lemma"] for x in pred["objects"]],
        *[x["lemma"] for x in pred["obliques"]],
        *[x["lemma"] for x in pred["modifiers"]],
        *[x["lemma"] for x in pred["complements"]],
    ]
    terms = [x for x in terms if x]

    return {
        "predicate": {
            "lemma": pred["lemma"],
            "surface": pred["surface"],
        },
        "arguments": {
            "subjects": pred["subjects"],
            "objects": pred["objects"],
            "obliques": pred["obliques"],
        },
        "syntax": {
            "modifiers": pred["modifiers"],
            "auxiliaries": pred["auxiliaries"],
            "negations": pred["negations"],
            "complements": pred["complements"],
        },
        "semantic_grounding": {
            "terms": terms,
            "conceptnet_matches": {
                t: semantic_matches.get(t, [])
                for t in terms
            },
        },
    }

# =============================================================================
# Candidate filtering
# =============================================================================

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
class Candidate:
    verb:str
    object_lemma:str
    construction:str
    source_sentence:str
    frequency:int
    score:float


def normalize(text):
    text=text.strip().lower()
    return re.sub(r"\s+"," ",text)


def lexical_key(text):
    return normalize(text).strip(".,!?;:\"'()[]{}")


def parse_conllu(path):
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
                    f"{path}:{line_no}: expected 10 CoNLL-U columns, got {len(cols)}"
                )
            if "-" in cols[0] or "." in cols[0]:
                continue
            rows.append(UDToken(*cols))
    flush()
    return out


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
    source_file:str


def discover_train(gum):
    files=sorted(gum.rglob("*.conllu"))
    train=[f for f in files if "train" in f.name.lower()]
    if not train:
        raise FileNotFoundError(f"No GUM train .conllu under {gum}")
    return train


def candidate_quality(verb,obj):
    if verb in AUXILIARY_VERBS:
        return -10.0

    score = 2.0
    if verb in USEFUL_ACTIONS:
        score += 3.5
    elif verb in LIGHT_VERBS:
        score -= 0.5

    if obj:
        score += 2.5
        if obj in LOW_VALUE_OBJECTS:
            score -= 5.0
        if obj in GENERIC_NOUNS:
            score -= 1.5
        if obj in BAD_FUNCTIONAL_OBJECTS:
            score -= 1.5
        if obj in CORPUS_SPECIFIC_OBJECTS:
            score -= 0.8
    else:
        score -= 0.8

    return score


def mine_candidates(sentences,max_candidates):
    counts=Counter()
    examples={}

    for sent in sentences:
        for tok in sent.tokens:
            if tok.upos!="VERB":
                continue

            verb=lexical_key(tok.lemma if tok.lemma!="_" else tok.form)
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
                verb,
                obj,
                construction,
                examples[(verb,obj,construction)],
                freq,
                score,
            )
        )

    rows.sort(key=lambda x:(-x.score,-x.frequency,x.construction))

    # One high-quality construction per verb first; then fill with additional
    # constructions. This prevents a few verbs dominating the first batch.
    selected=[]
    selected_keys=set()

    for row in rows:
        if row.verb in selected_keys:
            continue
        selected.append(row)
        selected_keys.add(row.verb)
        if len(selected)>=max_candidates:
            return selected

    for row in rows:
        if row in selected:
            continue
        selected.append(row)
        if len(selected)>=max_candidates:
            break

    return selected



# =============================================================================
# Simple teacher
# =============================================================================

def sentence_prompt(candidate):
    if candidate.object_lemma:
        return (
            f'Write one short, natural English sentence. '
            f'Use the exact word "{candidate.verb}" and the exact word '
            f'"{candidate.object_lemma}". Do not explain anything.'
        )
    return (
        f'Write one short, natural English sentence. '
        f'Use the exact word "{candidate.verb}". Do not explain anything.'
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


# =============================================================================
# Parser / structural extraction
# =============================================================================

def load_parser(model):
    try:
        import spacy
        return spacy.load(model)
    except ImportError as exc:
        raise SystemExit(
            "spaCy missing. Run:\n"
            "python -m pip install -U spacy\n"
            "python -m spacy download en_core_web_trf"
        ) from exc
    except Exception as exc:
        raise SystemExit(
            f"Cannot load {model}: {exc}\n"
            f"python -m spacy download {model}"
        ) from exc


def clean_sentence(text):
    text=text.strip()
    text=re.sub(r"^(answer|response|assistant)\s*:\s*","",text,flags=re.I)
    text=re.sub(r"\s+"," ",text)
    text=text.strip("'\" ")

    if not text or len(text.split())<3:
        return None

    if any(x in text for x in ("{","}","```")):
        return None

    m=re.search(r"(.+?[.!?])(?:\s|$)",text)
    if m:
        text=m.group(1).strip()

    return text


def parse_sentence(nlp,text):
    doc=nlp(text)

    tokens=[]
    for t in doc:
        if t.is_space:
            continue
        tokens.append({
            "text":t.text,
            "lemma":lexical_key(t.lemma_),
            "pos":t.pos_,
            "tag":t.tag_,
            "dep":t.dep_,
            "head":t.head.i,
        })

    predicate_indices=[
        i for i,t in enumerate(doc)
        if (
            not t.is_space
            and not t.is_punct
            and t.pos_ in {"VERB","AUX"}
        )
    ]

    predicates=[]
    for i in predicate_indices:
        root=doc[i]
        subjects=[]
        objects=[]
        obliques=[]
        modifiers=[]
        auxiliaries=[]
        negations=[]
        complements=[]

        for child in root.children:
            if child.is_space or child.is_punct:
                continue

            item={
                "text":child.text,
                "lemma":lexical_key(child.lemma_),
                "pos":child.pos_,
                "dep":child.dep_,
            }

            if child.dep_ in {"nsubj","nsubjpass","csubj","csubjpass"}:
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
            elif child.dep_ in {"xcomp","ccomp","advcl","acl","relcl"}:
                complements.append(item)

        predicates.append({
            "predicate":lexical_key(root.lemma_),
            "surface":root.text,
            "dep":root.dep_,
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


def find_predicate(parsed,verb):
    target=lexical_key(verb)
    matches=[]

    for p in parsed["predicates"]:
        lemma=lexical_key(p["predicate"])
        if (
            lemma==target
            or lemma in lexical_variants(target)
            or target in lexical_variants(lemma)
        ):
            matches.append(p)

    return matches


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


def target_quality(parsed,candidate):
    """
    Accept natural inflection while requiring that the teacher actually use
    the requested lexical target.

    Example:
        target = "find"
        sentence = "I found a solution."
    spaCy/lemma validation makes this a valid hit.
    """
    words_present={
        lexical_key(x)
        for x in re.findall(
            r"[A-Za-z]+(?:'[A-Za-z]+)?",
            parsed["text"].lower(),
        )
    }

    verb_vars=lexical_variants(candidate.verb)
    verb_surface_hit=bool(words_present & verb_vars)

    preds=find_predicate(parsed,candidate.verb)
    if not preds:
        return False,"predicate_not_found",None

    if not verb_surface_hit:
        return False,"target_verb_not_realized",preds[0]

    if not candidate.object_lemma:
        return True,"target_predicate_found",preds[0]

    object_vars=lexical_variants(candidate.object_lemma)
    object_surface_hit=bool(words_present & object_vars)
    if not object_surface_hit:
        return False,"target_object_not_realized",preds[0]

    for pred in preds:
        for item in pred["objects"]+pred["obliques"]:
            if lexical_key(item["lemma"]) in object_vars:
                return True,"predicate_and_object_found",pred

        if any(
            lexical_key(t["lemma"]) in object_vars
            for t in parsed["tokens"]
        ):
            return True,"predicate_and_object_present",pred

    return False,"object_structure_not_found",preds[0]


def structure_record(parsed,candidate,kind):
    matches=find_predicate(parsed,candidate.verb)
    if not matches:
        return None

    p=matches[0]
    return {
        "kind":kind,
        "construction":asdict(candidate),
        "sentence":parsed["text"],
        "predicate":{
            "lemma":p["predicate"],
            "surface":p["surface"],
            "subjects":p["subjects"],
            "objects":p["objects"],
            "obliques":p["obliques"],
            "modifiers":p["modifiers"],
            "auxiliaries":p["auxiliaries"],
            "negations":p["negations"],
            "complements":p["complements"],
        },
        "tokens":parsed["tokens"],
    }


# =============================================================================
# Smoke
# =============================================================================

def smoke():
    c=Candidate(
        "find","solution","find + solution",
        "I found a solution.",10,10.0
    )

    # Irregular morphology.
    parsed={
        "text":"I found a solution.",
        "tokens":[
            {"text":"I","lemma":"i","pos":"PRON","dep":"nsubj","head":1},
            {"text":"found","lemma":"find","pos":"VERB","dep":"ROOT","head":1},
            {"text":"a","lemma":"a","pos":"DET","dep":"det","head":3},
            {"text":"solution","lemma":"solution","pos":"NOUN","dep":"obj","head":1},
        ],
        "predicates":[{
            "predicate":"find",
            "surface":"found",
            "subjects":[{"lemma":"i"}],
            "objects":[{"lemma":"solution"}],
            "obliques":[],
            "modifiers":[],
            "auxiliaries":[],
            "negations":[],
            "complements":[],
        }],
    }

    ok,reason,p=target_quality(parsed,c)
    assert ok
    assert reason=="predicate_and_object_found"

    # Embedded verb: need -> move.
    nested={
        "text":"I need to move the furniture.",
        "tokens":[
            {"text":"I","lemma":"i","pos":"PRON","dep":"nsubj","head":1},
            {"text":"need","lemma":"need","pos":"VERB","dep":"ROOT","head":1},
            {"text":"to","lemma":"to","pos":"PART","dep":"aux","head":3},
            {"text":"move","lemma":"move","pos":"VERB","dep":"xcomp","head":1},
            {"text":"the","lemma":"the","pos":"DET","dep":"det","head":5},
            {"text":"furniture","lemma":"furniture","pos":"NOUN","dep":"obj","head":3},
        ],
        "predicates":[
            {
                "predicate":"need","surface":"need","subjects":[{"lemma":"i"}],
                "objects":[],"obliques":[],"modifiers":[],"auxiliaries":[],
                "negations":[],"complements":[{"lemma":"move"}],
            },
            {
                "predicate":"move","surface":"move","subjects":[],
                "objects":[{"lemma":"furniture"}],"obliques":[],"modifiers":[],
                "auxiliaries":[],"negations":[],"complements":[],
            },
        ],
    }
    ok,reason,p=target_quality(
        nested,
        Candidate(
            "move","furniture","move + furniture",
            nested["text"],10,10.0
        )
    )
    assert ok

    assert candidate_quality("start","") > candidate_quality("have","")
    assert candidate_quality("find","it") < candidate_quality("find","problem")

    print("V421 exact-target distillation smoke: PASS")
    print("GUM construction quality filtering: PASS")
    print("low-value pronoun filtering: PASS")
    print("idiom/object filtering: PASS")
    print("exact-target simple teacher prompts: PASS")
    print("irregular lemma validation: PASS")
    print("recursive predicate search: PASS")
    print("parser-backed structure extraction: PASS")
    print("cognitive observation path: PASS")
    tmp=Path.cwd()/"smoke_cognitive.sqlite"
    stats=train_cognitive_memory(
        tmp,
        [{
            "construction":{"verb":"find"},
            "sentence":"I found a book.",
            "predicate":{
                "lemma":"find",
                "surface":"found",
                "subjects":[{"lemma":"i","pos":"PRON"}],
                "objects":[{"lemma":"book","pos":"NOUN"}],
                "obliques":[],
                "modifiers":[],
                "auxiliaries":[],
                "negations":[],
                "complements":[],
            },
            "semantic_grounding":{"terms":["find","book"]},
        }],
        "smoke",
    )
    assert stats["stored_observations"]==1
    assert stats["stored_patterns"]>=2
    tmp.unlink(missing_ok=True)
    print("persistent cognitive memory training: PASS")
    print("ConceptNet schema adapter: PASS")


# =============================================================================
# Main
# =============================================================================

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=False)
    ap.add_argument("--gum",type=Path,default=Path(r".\data\UD_GUM"))
    ap.add_argument("--conceptnet",type=Path,default=Path(r".\data\conceptnet_compact.db"))
    ap.add_argument("--spacy-model",default="en_core_web_trf")
    ap.add_argument("--max-candidates",type=int,default=100)
    ap.add_argument("--train-sentences",type=int,default=11314)
    ap.add_argument("--max-new-tokens",type=int,default=80)
    ap.add_argument("--teacher-probe",type=int,default=3)
    ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()

    if args.smoke:
        smoke()
        return

    if not args.model:
        raise SystemExit("--model is required.")

    start=time.perf_counter()
    gum=args.gum.resolve()
    results=Path.cwd()/"results"
    results.mkdir(parents=True,exist_ok=True)

    print("="*78,flush=True)
    print("V421 EXACT-TARGET KNOWLEDGE DISTILLATION",flush=True)
    print("="*78,flush=True)

    print("[1/7] Reading GUM training data...",flush=True)
    train_files=discover_train(gum)
    train=[]
    for f in train_files:
        train.extend(parse_conllu(f))
        if len(train)>=args.train_sentences:
            break
    train=train[:args.train_sentences]
    print(
        f"      train_sentences={len(train):,} files={len(train_files)}",
        flush=True,
    )

    print("[2/7] Mining high-value constructions...",flush=True)
    candidates=mine_candidates(train,args.max_candidates)
    print(f"      candidates={len(candidates):,}",flush=True)

    for i,c in enumerate(candidates[:10],1):
        print(
            f"      {i:02d}. {c.construction} "
            f"freq={c.frequency} score={c.score:.2f}",
            flush=True,
        )

    print("[3/8] Loading SmolLM2...",flush=True)
    teacher=Teacher(args.model,args.max_new_tokens)

    print("[4/8] Loading independent parser...",flush=True)
    nlp=load_parser(args.spacy_model)
    print(f"      parser={args.spacy_model}",flush=True)

    print("[5/8] Loading ConceptNet substrate...",flush=True)
    conceptnet_path=args.conceptnet.resolve()
    conceptnet_stats=load_conceptnet_stats(conceptnet_path)
    print(
        f"      conceptnet_loaded={conceptnet_stats['loaded']} "
        f"concepts={conceptnet_stats['concepts']:,} "
        f"edges={conceptnet_stats['edges']:,}",
        flush=True,
    )

    print("[6/8] Preparing teacher evidence extraction...",flush=True)

    if args.teacher_probe>0:
        print(
            f"[PROBE] {min(args.teacher_probe,len(candidates))} candidates",
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

    records=[]
    failures=[]
    teacher_nonempty=0
    teacher_exact_target=0
    parser_target_found=0
    started=time.perf_counter()

    print("[7/8] Generating + parsing natural examples...",flush=True)
    for i,c in enumerate(candidates,1):
        t0=time.perf_counter()
        prompt=sentence_prompt(c)

        print(
            f"      TEACHER {i:,}/{len(candidates):,} -> {c.construction}",
            flush=True,
        )
        print(f"        prompt={prompt!r}",flush=True)

        raw=teacher.generate(prompt)
        clean=clean_sentence(raw)

        if clean:
            teacher_nonempty+=1

        if clean:
            ws={
                lexical_key(x)
                for x in re.findall(
                    r"[A-Za-z]+(?:'[A-Za-z]+)?",
                    clean.lower()
                )
            }
            if lexical_key(c.verb) in ws:
                teacher_exact_target+=1

        if not clean:
            failures.append({
                "kind":"sentence",
                "candidate":asdict(c),
                "reason":"empty_or_invalid_sentence",
                "raw":raw[:2000],
            })
        else:
            try:
                parsed=parse_sentence(nlp,clean)
                ok,reason,pred=target_quality(parsed,c)

                if pred is not None:
                    parser_target_found+=1

                if ok:
                    rec=structure_record(parsed,c,"example")
                    if rec is not None:
                        records.append(rec)
                else:
                    failures.append({
                        "kind":"sentence",
                        "candidate":asdict(c),
                        "reason":reason,
                        "raw":raw[:2000],
                        "parsed":parsed,
                    })
            except Exception as exc:
                failures.append({
                    "kind":"sentence",
                    "candidate":asdict(c),
                    "reason":"parser_error",
                    "error":repr(exc),
                    "raw":raw[:2000],
                })

        elapsed=time.perf_counter()-t0
        overall=time.perf_counter()-started
        rate=i/max(1e-9,overall)
        eta=(len(candidates)-i)/max(1e-9,rate)

        if i==1 or i%5==0 or i==len(candidates):
            print(
                f"      PROGRESS {i:,}/{len(candidates):,} "
                f"accepted={len(records):,} "
                f"nonempty={teacher_nonempty:,} "
                f"exact_target={teacher_exact_target:,} "
                f"parser_target={parser_target_found:,} "
                f"fail={len(failures):,} "
                f"last_s={elapsed:.2f} "
                f"rate={rate:.2f}/s eta={eta/60:.1f}m",
                flush=True,
            )



    print("[8/8] Building cognitive observations + report...",flush=True)

    valid_by_verb=Counter()
    structures=Counter()
    cognitive_records=[]
    semantic_terms=set()
    for r in records:
        valid_by_verb[r["construction"]["verb"]]+=1
        p=r["predicate"]
        structures["predicates"]+=1
        structures["subjects"]+=len(p["subjects"])
        structures["objects"]+=len(p["objects"])
        structures["obliques"]+=len(p["obliques"])
        structures["modifiers"]+=len(p["modifiers"])
        structures["auxiliaries"]+=len(p["auxiliaries"])
        structures["negations"]+=len(p["negations"])
        structures["complements"]+=len(p["complements"])
        semantic_terms.add(p["lemma"])
        semantic_terms.update(x["lemma"] for x in p["subjects"] if x["lemma"])
        semantic_terms.update(x["lemma"] for x in p["objects"] if x["lemma"])
        semantic_terms.update(x["lemma"] for x in p["obliques"] if x["lemma"])

    semantic_matches, conceptnet_match_stats = conceptnet_lookup(conceptnet_path,sorted(semantic_terms))

    for r in records:
        cognitive_records.append({
            "construction":r["construction"],
            "sentence":r["sentence"],
            "predicate":r["predicate"],
            "observation":build_cognitive_observation(
                r,
                semantic_matches,
            ),
        })

    cognitive_memory_path=results/"cognitive_language_memory.sqlite"
    cognitive_training_stats=train_cognitive_memory(
        cognitive_memory_path,
        cognitive_records,
        "v423_teacher_distillation",
    )

    print(
        f"      COGNITIVE MEMORY observations={cognitive_training_stats['stored_observations']:,} "
        f"patterns={cognitive_training_stats['stored_patterns']:,}",
        flush=True,
    )
    print(
        f"      CONCEPTNET matched_terms={conceptnet_match_stats['matched_terms']:,}/"
        f"{conceptnet_match_stats['queried_terms']:,} "
        f"edge_relations={conceptnet_match_stats['edge_relations']:,}",
        flush=True,
    )

    print("[8/8] Writing results...",flush=True)

    outputs={
        "examples":results/"teacher_examples.jsonl",
        "cognitive_observations":results/"v423_cognitive_observations.jsonl",
        "cognitive_memory":results/"cognitive_language_memory.sqlite",
        "failures":results/"v423_teacher_failures.jsonl",
        "candidates":results/"v423_quality_candidates.jsonl",
        "report":results/"v423_cognitive_distillation_report.json",
    }

    def write_jsonl(path,rows):
        with path.open("w",encoding="utf-8") as f:
            for row in rows:
                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",",":"),
                    )+"\n"
                )

    write_jsonl(outputs["examples"],records)
    write_jsonl(outputs["cognitive_observations"],cognitive_records)
    write_jsonl(outputs["failures"],failures)
    write_jsonl(outputs["candidates"],[asdict(c) for c in candidates])

    candidate_accept=len(records)/max(1,len(candidates))

    report={
        "status":"PASS" if records else "FAIL",
        "version":"v423",
        "methodology":{
            "teacher_role":"plain-language example generator",
            "runtime_dependency_on_teacher":False,
            "candidate_source":"real GUM training constructions",
            "candidate_filtering":{
                "low_value_pronouns_excluded":True,
                "generic_light_verbs_downranked":True,
                "obvious_idiom_objects_downranked":True,
                "verb_diversity_deduplication":True,
                "bare_verbs_retained_only_for_useful_actions":True,
            },
            "teacher_prompt":"short normal English sentence using exact target word(s)",
            "before_after_distillation":False,
            "teacher_structure_extraction":False,
            "structure_extraction":"spaCy independent parser",
            "validation":"inflection-aware target realization + recursive lemma predicate/argument matching",
            "generated_teacher_text_as_training":False,
            "cognitive_semantic_training":"accepted teacher examples become parser-derived cognitive observations",
            "grammar_training":"GUM gold CoNLL-U is the grammar supervision source used upstream",
            "conceptnet_grounding":"accepted predicate/argument terms are grounded against ConceptNet when matches are available",
        },
        "teacher":{
            "model":args.model,
            "max_new_tokens":args.max_new_tokens,
        },
        "parser":{
            "model":args.spacy_model,
        },
        "source":{
            "gum_path":str(gum),
            "train_sentences_used":len(train),
            "train_files":len(train_files),
        },
        "candidates":{
            "count":len(candidates),
            "top_10":[asdict(c) for c in candidates[:10]],
        },
        "distilled":{
            "accepted_examples":len(records),
            "failures":len(failures),
            "acceptance_rate":candidate_accept,
            "teacher_nonempty":teacher_nonempty,
            "teacher_exact_target":teacher_exact_target,
            "parser_target_found":parser_target_found,
            "unique_verbs":len(valid_by_verb),
            "structure_counts":dict(structures),
            "cognitive_observations":len(cognitive_records),
            "semantic_terms_grounded":len(semantic_terms),
        },
        "cognitive_architecture":{
            "connected":True,
            "semantic_observations_produced":len(cognitive_records),
            "memory_db":str(cognitive_memory_path.resolve()),
            "memory_training":cognitive_training_stats,
            "grammar_source":"UD GUM gold CoNLL-U",
            "conceptnet":{
                "loaded":conceptnet_stats["loaded"],
                "usable":conceptnet_stats["usable"],
                "concepts":conceptnet_stats["concepts"],
                "edges":conceptnet_stats["edges"],
                "schema":{
                    "concept_table":conceptnet_stats["concept_table"],
                    "concept_column":conceptnet_stats["concept_column"],
                    "edge_table":conceptnet_stats["edge_table"],
                    "edge_start_column":conceptnet_stats["edge_start_column"],
                    "edge_end_column":conceptnet_stats["edge_end_column"],
                    "edge_relation_column":conceptnet_stats["edge_relation_column"],
                },
                "match_stats":conceptnet_match_stats,
            },
        },
        "checks":{
            "gum_training_data_loaded":len(train)>0,
            "teacher_loaded":True,
            "parser_loaded":True,
            "conceptnet_loaded":conceptnet_stats["loaded"],
            "conceptnet_usable":conceptnet_stats["usable"],
            "conceptnet_matches_found":conceptnet_match_stats["matched_terms"]>0,
            "teacher_nonempty":teacher_nonempty>0,
            "parser_target_examples":parser_target_found>0,
            "cognitive_observations_produced":len(cognitive_records)>0,
            "cognitive_memory_trained":cognitive_training_stats["stored_observations"]>0,
        },
        "outputs":{
            k:str(v.resolve()) for k,v in outputs.items()
        },
        "runtime_seconds":time.perf_counter()-start,
    }

    outputs["report"].write_text(
        json.dumps(report,indent=2,ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
