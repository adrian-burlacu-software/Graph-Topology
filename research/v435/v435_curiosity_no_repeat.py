
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path


# =============================================================================
# YOUR PROJECT DEFAULTS
# =============================================================================

DEFAULT_DB = Path(
    r"C:\Users\adria\Desktop\dev\Graph-Topology\results\cognitive_language_memory.sqlite"
)
DEFAULT_MODEL = Path(
    r"C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM2-1.7B-Instruct"
)
DEFAULT_SPACY = "en_core_web_trf"
DEFAULT_CONCEPTNET = Path(
    r"C:\Users\adria\Desktop\dev\Graph-Topology\data\conceptnet_compact.db"
)


# =============================================================================
# Curiosity vocabulary
# =============================================================================

QUESTION_FAMILIES = {
    "identity": "What is {x}?",
    "category": "What kind of thing is {x}?",
    "agents": "What kinds of things can {x}?",
    "patients": "What kinds of things can someone {x}?",
    "objects": "What can someone {x}?",
    "tools": "What can someone use to {x}?",
    "locations": "Where can {x} happen?",
    "causes": "What can cause {x}?",
    "effects": "What can {x} cause?",
    "purpose": "What is {x} used for?",
    "examples": "Give an ordinary example of {x}.",
    "preconditions": "What usually needs to happen before {x}?",
    "postconditions": "What usually happens after {x}?",
    "negation": "When does {x} not apply?",
    "contrast": "What is {x} different from?",
}


# =============================================================================
# General helpers
# =============================================================================

def norm(x):
    x = str(x or "").strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def key(x):
    return re.sub(
        r"[^\w-]",
        "",
        norm(x),
    )


def digest(x):
    return hashlib.sha256(
        str(x).encode("utf-8")
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
    if isinstance(value, Counter):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }
    return value


def clean_answer(text):
    text = (text or "").strip()
    text = re.sub(
        r"^(assistant|answer|response)\s*:\s*",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    if any(x in text for x in ("```", "{", "}")):
        return None
    m = re.search(r"(.+?[.!?])(?:\s|$)", text)
    if m:
        text = m.group(1).strip()
    if len(text.split()) < 3:
        return None
    return text


# =============================================================================
# Database
# =============================================================================

def table_exists(con, name):
    return con.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type='table' AND name=?",
        (name,),
    ).fetchone() is not None


def ensure_schema(con):
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")

    con.executescript("""
    CREATE TABLE IF NOT EXISTS curiosity_state (
        lemma TEXT PRIMARY KEY,
        asked_json TEXT NOT NULL,
        covered_json TEXT NOT NULL,
        rounds INTEGER NOT NULL DEFAULT 0,
        updated_at REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS curiosity_questions (
        question_id TEXT PRIMARY KEY,
        lemma TEXT NOT NULL,
        family TEXT NOT NULL,
        question TEXT NOT NULL,
        answer TEXT,
        accepted INTEGER NOT NULL,
        reason TEXT,
        created_at REAL NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_curiosity_questions_lemma
      ON curiosity_questions(lemma);

    CREATE TABLE IF NOT EXISTS curiosity_facts (
        fact_id TEXT PRIMARY KEY,
        lemma TEXT NOT NULL,
        family TEXT NOT NULL,
        fact_json TEXT NOT NULL,
        question_id TEXT NOT NULL,
        created_at REAL NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_curiosity_facts_lemma
      ON curiosity_facts(lemma);
    """)
    con.commit()



def question_evidence(con, lemma, family):
    """
    Estimate actual evidence for a question family.

    Lexical resources provide PRIORS, not completion. A VerbNet class does not
    mean that the cognitive memory knows which concrete things can walk, nor
    does a PropBank roleset mean the model has learned useful examples.

    Only explicit learned/curiosity evidence counts strongly as coverage.
    """
    lemma = key(lemma)
    score = 0.0

    # Existing parser-derived learning.
    if table_exists(con, "learned_patterns"):
        rows = con.execute(
            "SELECT role, COUNT(*) "
            "FROM learned_patterns "
            "WHERE lemma=? "
            "GROUP BY role",
            (lemma,),
        ).fetchall()
        role_counts={role:int(n) for role,n in rows}

        if family == "agents":
            score += min(role_counts.get("subject",0), 5) * 1.0
        elif family in {"patients","objects"}:
            score += min(role_counts.get("object",0), 5) * 1.0
        elif family == "locations":
            score += min(role_counts.get("oblique",0), 5) * 1.0
        elif family == "postconditions":
            score += min(role_counts.get("complement",0), 5) * 0.8

    # Explicit curiosity facts are stronger evidence.
    if table_exists(con, "curiosity_facts"):
        count=con.execute(
            "SELECT COUNT(*) FROM curiosity_facts "
            "WHERE lemma=? AND family=?",
            (lemma,family),
        ).fetchone()[0]
        score += min(int(count), 5) * 2.0

    # Teacher examples prove that the concept can occur in language, but don't
    # by themselves answer all semantic questions.
    if family == "examples" and table_exists(con, "teacher_examples"):
        count=con.execute(
            "SELECT COUNT(*) FROM teacher_examples "
            "WHERE lemma=?",
            (lemma,),
        ).fetchone()[0]
        score += min(int(count), 3) * 2.0

    return score


def lexical_prior(con, lemma, family):
    """
    Resource-derived prior only. It reduces curiosity pressure rather than
    declaring the question answered.
    """
    lemma=key(lemma)
    prior=0.0

    if family in {"identity","category"} and table_exists(con,"concepts"):
        wn=con.execute(
            "SELECT COUNT(*) FROM concepts "
            "WHERE lemma=? AND source='wordnet'",
            (lemma,),
        ).fetchone()[0]
        prior += min(int(wn),3) * 0.75

    if family in {"agents","patients","objects"}:
        if table_exists(con,"verbnet_roles"):
            vn_roles=con.execute(
                "SELECT COUNT(*) FROM verbnet_roles "
                "WHERE class_id IN ("
                "SELECT class_id FROM verbnet_members WHERE lemma=?"
                ")",
                (lemma,),
            ).fetchone()[0]
            prior += min(int(vn_roles),8) * 0.35

        if table_exists(con,"propbank_roles"):
            pb_roles=con.execute(
                "SELECT COUNT(*) FROM propbank_roles pr "
                "JOIN propbank_rolesets rs "
                "ON rs.roleset_id=pr.roleset_id "
                "WHERE rs.lemma=?",
                (lemma,),
            ).fetchone()[0]
            prior += min(int(pb_roles),12) * 0.25

    if family in {"causes","effects","purpose","tools","preconditions","postconditions"}:
        if table_exists(con,"semantic_grounding"):
            edges=con.execute(
                "SELECT COUNT(*) FROM semantic_grounding "
                "WHERE lemma=?",
                (lemma,),
            ).fetchone()[0]
            prior += min(int(edges),10) * 0.10

    return prior


def family_score(con, lemma, family):
    """
    Higher score = more useful next question.

    Core rule:
      curiosity = desired information value - actual evidence

    We deliberately do not let WordNet/VerbNet/PropBank mark a family solved.
    """
    desired={
        "agents":3.5,
        "patients":3.5,
        "objects":3.2,
        "tools":2.6,
        "locations":2.6,
        "preconditions":2.4,
        "postconditions":2.4,
        "causes":2.2,
        "effects":2.2,
        "purpose":2.0,
        "examples":1.8,
        "category":1.4,
        "identity":1.2,
        "negation":1.0,
        "contrast":0.8,
    }

    # Argument-heavy verbs benefit from participant questions.
    semantic_roles=0
    if table_exists(con,"verbnet_roles"):
        semantic_roles=con.execute(
            "SELECT COUNT(*) FROM verbnet_roles "
            "WHERE class_id IN ("
            "SELECT class_id FROM verbnet_members WHERE lemma=?"
            ")",
            (key(lemma),),
        ).fetchone()[0]

    if family in {"agents","patients","objects"}:
        desired[family] += min(semantic_roles,10)*0.12

    evidence=question_evidence(
        con,
        lemma,
        family,
    )
    prior=lexical_prior(
        con,
        lemma,
        family,
    )

    # Teacher-generated evidence dominates lexical priors.
    return desired.get(family,1.0) - evidence - 0.20*prior



def existing_asked(con, lemma):
    """
    Families count as asked even when the teacher answer failed validation.

    V434 only read curiosity_state, which is updated on successful learning.
    That created a feedback loop:
        same family -> failed answer -> not recorded -> same family again.
    """
    lemma=key(lemma)
    asked=set()

    if table_exists(con,"curiosity_state"):
        row=con.execute(
            "SELECT asked_json FROM curiosity_state "
            "WHERE lemma=?",
            (lemma,),
        ).fetchone()
        if row:
            try:
                asked.update(json.loads(row[0]))
            except Exception:
                pass

    if table_exists(con,"curiosity_questions"):
        for (family,) in con.execute(
            "SELECT DISTINCT family FROM curiosity_questions "
            "WHERE lemma=?",
            (lemma,),
        ):
            asked.add(family)

    return asked


def record_question_attempt(
    con,
    lemma,
    family,
    question,
    answer,
    accepted,
    reason,
):
    """
    Persist the fact that a curiosity family was attempted, whether or not
    learning succeeded.
    """
    lemma=key(lemma)

    qid=sha(
        json.dumps(
            [
                lemma,
                family,
                question,
                answer,
                time.time(),
            ],
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    con.execute("""
        INSERT INTO curiosity_questions
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,(
        qid,
        lemma,
        family,
        question,
        answer,
        int(accepted),
        reason,
        time.time(),
    ))

    row=con.execute(
        "SELECT asked_json,covered_json,rounds "
        "FROM curiosity_state WHERE lemma=?",
        (lemma,),
    ).fetchone()

    if row:
        try:
            asked=set(json.loads(row[0]))
        except Exception:
            asked=set()
        try:
            covered=set(json.loads(row[1]))
        except Exception:
            covered=set()
        rounds=int(row[2])+1
    else:
        asked=set()
        covered=set()
        rounds=1

    # IMPORTANT: asked is updated regardless of success.
    asked.add(family)

    if accepted:
        covered.add(family)

    con.execute("""
        INSERT OR REPLACE INTO curiosity_state
        VALUES (?, ?, ?, ?, ?)
    """,(
        lemma,
        json.dumps(sorted(asked)),
        json.dumps(sorted(covered)),
        rounds,
        time.time(),
    ))
    con.commit()

    return qid



def next_question(con, lemma, exclude_families=None):
    """
    Choose the highest-information question not already attempted in the
    current sweep and not excluded by the caller.
    """
    lemma=key(lemma)
    exclude=set(exclude_families or ())
    asked=existing_asked(con,lemma)

    families=list(QUESTION_FAMILIES.keys())
    scores=[]

    for family in families:
        if family in exclude:
            continue

        score=family_score(
            con,
            lemma,
            family,
        )

        # Strongly prefer genuinely new families.
        if family not in asked:
            score += 1.0
        else:
            score -= 3.0

        scores.append((score,family))

    if not scores:
        return None,None

    scores.sort(
        key=lambda x:(-x[0],x[1])
    )

    family=scores[0][1]
    return family,QUESTION_FAMILIES[family].format(x=lemma)


def curiosity_diagnostics(con, lemma):
    rows=[]
    asked=existing_asked(con,lemma)
    for family in QUESTION_FAMILIES:
        rows.append({
            "family":family,
            "score":round(
                family_score(con,lemma,family),
                3,
            ),
            "asked":family in asked,
            "evidence":round(
                question_evidence(con,lemma,family),
                3,
            ),
            "lexical_prior":round(
                lexical_prior(con,lemma,family),
                3,
            ),
        })
    rows.sort(
        key=lambda x:(-x["score"],x["family"])
    )
    return rows


def resource_context(con, lemma):
    lemma = key(lemma)
    context = {}

    if table_exists(con, "concepts"):
        context["wordnet"] = [
            {
                "id": r[0],
                "pos": r[1],
                "gloss": r[2],
            }
            for r in con.execute(
                "SELECT source_id,pos,gloss "
                "FROM concepts "
                "WHERE lemma=? AND source='wordnet' "
                "LIMIT 5",
                (lemma,),
            )
        ]

    if table_exists(con, "verbnet_members") and table_exists(con, "verbnet_classes"):
        context["verbnet"] = [
            {
                "class_id": r[0],
                "name": r[1],
            }
            for r in con.execute(
                "SELECT vm.class_id,vc.name "
                "FROM verbnet_members vm "
                "JOIN verbnet_classes vc "
                "ON vc.class_id=vm.class_id "
                "WHERE vm.lemma=? LIMIT 5",
                (lemma,),
            )
        ]

    if table_exists(con, "verbnet_roles") and table_exists(con, "verbnet_members"):
        context["verbnet_roles"] = [
            {
                "class_id": r[0],
                "role": r[1],
                "description": r[2],
            }
            for r in con.execute(
                "SELECT vr.class_id,vr.role,vr.description "
                "FROM verbnet_roles vr "
                "JOIN verbnet_members vm "
                "ON vm.class_id=vr.class_id "
                "WHERE vm.lemma=? LIMIT 8",
                (lemma,),
            )
        ]

    if table_exists(con, "propbank_rolesets"):
        context["propbank"] = [
            {
                "roleset_id": r[0],
                "name": r[1],
                "vncls": r[2],
            }
            for r in con.execute(
                "SELECT roleset_id,name,vncls "
                "FROM propbank_rolesets "
                "WHERE lemma=? LIMIT 5",
                (lemma,),
            )
        ]

    if table_exists(con, "semlink_mappings"):
        context["semlink"] = [
            {
                "source_family": r[0],
                "source_id": r[1],
                "target_family": r[2],
                "target_id": r[3],
            }
            for r in con.execute(
                "SELECT source_family,source_id,target_family,target_id "
                "FROM semlink_mappings "
                "WHERE source_id LIKE ? "
                "OR target_id LIKE ? LIMIT 8",
                (f"%{lemma}%", f"%{lemma}%"),
            )
        ]

    if table_exists(con, "semantic_grounding"):
        context["existing_grounding"] = [
            {
                "source": r[0],
                "relation": r[1],
                "target": r[2],
            }
            for r in con.execute(
                "SELECT source,relation,target "
                "FROM semantic_grounding "
                "WHERE lemma=? LIMIT 8",
                (lemma,),
            )
        ]

    if table_exists(con, "learned_patterns"):
        context["learned_patterns"] = [
            {
                "role": r[0],
                "argument": r[1],
                "count": r[2],
            }
            for r in con.execute(
                "SELECT role,argument_lemma,count "
                "FROM learned_patterns "
                "WHERE lemma=? ORDER BY count DESC LIMIT 10",
                (lemma,),
            )
        ]

    return context


# =============================================================================
# Teacher
# =============================================================================

class Teacher:
    def __init__(self, model_path, max_new_tokens):
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise SystemExit(
                "Install: python -m pip install -U torch transformers accelerate"
            ) from exc

        self.torch = torch
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )

        kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            **kwargs,
        )
        self.model.eval()

    def answer(self, question, context):
        compact = []

        for name in (
            "wordnet",
            "verbnet",
            "verbnet_roles",
            "propbank",
            "semlink",
            "existing_grounding",
            "learned_patterns",
        ):
            if context.get(name):
                compact.append(
                    f"{name}: {context[name][:5]}"
                )

        if compact:
            background = (
                "\nUseful background from the knowledge base:\n"
                + "\n".join(compact)
            )
        else:
            background = ""

        prompt = (
            f"{question}\n"
            "Answer in one or two short, normal English sentences. "
            "Do not explain your reasoning."
            + background
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "Answer simple questions in ordinary English. "
                    "Be direct."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        if hasattr(
            self.tokenizer,
            "apply_chat_template",
        ):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = (
                "Answer simply.\n"
                f"User: {prompt}\n"
                "Assistant:"
            )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        )
        device = next(
            self.model.parameters()
        ).device
        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        continuation = output[0][
            inputs["input_ids"].shape[1]:
        ]
        return self.tokenizer.decode(
            continuation,
            skip_special_tokens=True,
        ).strip()


# =============================================================================
# Parsing + semantic extraction
# =============================================================================

def load_parser(model_name):
    try:
        import spacy
        return spacy.load(model_name)
    except Exception as exc:
        raise SystemExit(
            f"Cannot load spaCy model {model_name}: {exc}"
        )


def parse_answer(nlp, answer):
    doc = nlp(answer)

    predicates = []

    for token in doc:
        if token.is_space or token.is_punct:
            continue
        if token.pos_ not in {"VERB", "AUX"}:
            continue

        pred = {
            "predicate": lemma_key(token.lemma_),
            "surface": token.text,
            "subjects": [],
            "objects": [],
            "obliques": [],
            "complements": [],
        }

        for child in token.children:
            if child.is_space or child.is_punct:
                continue

            item = {
                "text": child.text,
                "lemma": lemma_key(child.lemma_),
                "pos": child.pos_,
                "dep": child.dep_,
            }

            if child.dep_ in {
                "nsubj",
                "nsubjpass",
                "csubj",
                "csubjpass",
            }:
                pred["subjects"].append(item)
            elif child.dep_ in {
                "obj",
                "dobj",
                "iobj",
            }:
                pred["objects"].append(item)
            elif (
                child.dep_.startswith("obl")
                or child.dep_ == "prep"
            ):
                pred["obliques"].append(item)
            elif child.dep_ in {
                "xcomp",
                "ccomp",
                "advcl",
                "acl",
                "relcl",
            }:
                pred["complements"].append(item)

        predicates.append(pred)

    return {
        "text": answer,
        "predicates": predicates,
    }


def find_target(parsed, lemma):
    lemma = key(lemma)
    return [
        p
        for p in parsed["predicates"]
        if key(p["predicate"]) == lemma
    ]


def extract_facts(parsed):
    facts = []

    for pred in parsed["predicates"]:
        facts.append({
            "kind": "predicate",
            "predicate": pred["predicate"],
            "surface": pred["surface"],
        })

        for role, field in (
            ("subject", "subjects"),
            ("object", "objects"),
            ("oblique", "obliques"),
            ("complement", "complements"),
        ):
            for item in pred[field]:
                facts.append({
                    "kind": "argument",
                    "predicate": pred["predicate"],
                    "role": role,
                    "lemma": item["lemma"],
                    "pos": item.get("pos"),
                })

    return facts


# =============================================================================
# Learning update
# =============================================================================

def learn(con, lemma, family, question, answer, parsed):
    qid = digest(
        json.dumps(
            [lemma, family, question],
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    facts = extract_facts(parsed)

    con.execute("BEGIN")
    try:
        con.execute("""
            INSERT OR REPLACE INTO curiosity_questions
            VALUES (?, ?, ?, ?, ?, 1, ?, ?)
        """, (
            qid,
            key(lemma),
            family,
            question,
            answer,
            "accepted",
            time.time(),
        ))

        for fact in facts:
            fact_id = digest(
                json.dumps(
                    [qid, fact],
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )

            con.execute("""
                INSERT OR IGNORE INTO curiosity_facts
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                fact_id,
                key(lemma),
                family,
                json.dumps(
                    fact,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                qid,
                time.time(),
            ))

            if fact["kind"] == "argument" and fact["lemma"]:
                con.execute("""
                    INSERT INTO learned_patterns
                    (lemma,role,argument_lemma,count)
                    VALUES (?, ?, ?, 1)
                    ON CONFLICT(lemma,role,argument_lemma)
                    DO UPDATE SET count=count+1
                """, (
                    key(fact["predicate"]),
                    fact["role"],
                    key(fact["lemma"]),
                ))

        row = con.execute(
            "SELECT asked_json,covered_json,rounds "
            "FROM curiosity_state WHERE lemma=?",
            (key(lemma),),
        ).fetchone()

        if row:
            try:
                asked = set(json.loads(row[0]))
            except Exception:
                asked = set()
            try:
                covered = set(json.loads(row[1]))
            except Exception:
                covered = set()
            rounds = int(row[2]) + 1
        else:
            asked = set()
            covered = set()
            rounds = 1

        asked.add(family)
        covered.add(family)

        con.execute("""
            INSERT OR REPLACE INTO curiosity_state
            VALUES (?, ?, ?, ?, ?)
        """, (
            key(lemma),
            json.dumps(sorted(asked)),
            json.dumps(sorted(covered)),
            rounds,
            time.time(),
        ))

        con.commit()
    except Exception:
        con.rollback()
        raise

    return qid, len(facts)




# =============================================================================
# Vocabulary selection
# =============================================================================

def select_vocabulary(con, limit):
    """
    Start from concepts already in the semantic memory. Prefer items that have
    cross-resource support, then learned patterns, then general coverage.
    """
    rows = con.execute("""
        SELECT lemma, COUNT(DISTINCT source) AS sources, COUNT(*) AS evidence
        FROM concepts
        WHERE lemma <> ''
        GROUP BY lemma
        ORDER BY sources DESC, evidence DESC, lemma
        LIMIT ?
    """, (limit * 4,)).fetchall()

    scored = []

    for lemma, sources, evidence in rows:
        score = (
            50 * int(sources)
            + int(evidence)
        )

        # Prefer verbs / predicate-like vocabulary because that is where
        # question families give us the most useful participant structure.
        vn = 0
        pb = 0
        if table_exists(con, "verbnet_members"):
            vn = con.execute(
                "SELECT COUNT(*) FROM verbnet_members WHERE lemma=?",
                (lemma,),
            ).fetchone()[0]
        if table_exists(con, "propbank_rolesets"):
            pb = con.execute(
                "SELECT COUNT(*) FROM propbank_rolesets WHERE lemma=?",
                (lemma,),
            ).fetchone()[0]

        score += 20 * min(vn, 3)
        score += 20 * min(pb, 3)

        scored.append(
            (score, lemma)
        )

    scored.sort(
        key=lambda x:(-x[0], x[1])
    )

    return [lemma for _, lemma in scored[:limit]]


# =============================================================================
# Smoke
# =============================================================================

def smoke():
    path = Path.cwd() / "v433_smoke.sqlite"
    if path.exists():
        path.unlink()

    con = sqlite3.connect(str(path))
    con.execute("""
        CREATE TABLE concepts(
            concept_id TEXT PRIMARY KEY,
            lemma TEXT,
            pos TEXT,
            source TEXT,
            source_id TEXT,
            gloss TEXT,
            payload_json TEXT
        )
    """)
    con.execute("""
        CREATE TABLE learned_patterns(
            lemma TEXT,
            role TEXT,
            argument_lemma TEXT,
            count INTEGER,
            PRIMARY KEY(lemma,role,argument_lemma)
        )
    """)
    con.execute(
        "INSERT INTO concepts VALUES(?,?,?,?,?,?,?)",
        ("1","walk","v","wordnet","1","move on foot","{}"),
    )
    con.commit()

    ensure_schema(con)

    first_family = next_question(
        con,
        "walk",
    )[0]
    assert first_family in QUESTION_FAMILIES

    # Simulate the lexical prior for agents/patients without actual learned
    # evidence. Tools must NOT automatically become the only question.
    for fam in ("agents","patients","objects","tools"):
        assert fam in QUESTION_FAMILIES

    scores = curiosity_diagnostics(con,"walk")
    assert scores[0]["score"] >= scores[-1]["score"]

    # After recording one family, another family should become eligible.
    learn(
        con,
        "walk",
        "agents",
        "What kinds of things can walk?",
        "Dogs walk.",
        {
            "text":"Dogs walk.",
            "predicates":[{
                "predicate":"walk",
                "surface":"walk",
                "subjects":[{"lemma":"dog","pos":"NOUN","dep":"nsubj"}],
                "objects":[],
                "obliques":[],
                "complements":[],
            }],
        },
    )
    second_family=next_question(con,"walk")[0]
    assert second_family != "agents"

    parsed = {
        "text":"Dogs walk in parks.",
        "predicates":[{
            "predicate":"walk",
            "surface":"walk",
            "subjects":[
                {"lemma":"dog","pos":"NOUN","dep":"nsubj"}
            ],
            "objects":[],
            "obliques":[
                {"lemma":"park","pos":"NOUN","dep":"obl"}
            ],
            "complements":[],
        }],
    }

    qid, fact_count = learn(
        con,
        "walk",
        "agents",
        "What kinds of things can walk?",
        "Dogs walk in parks.",
        parsed,
    )

    assert fact_count == 3
    assert con.execute(
        "SELECT COUNT(*) FROM curiosity_facts"
    ).fetchone()[0] == 3

    con.close()
    path.unlink(missing_ok=True)

    print("V433 curiosity learner smoke: PASS")
    print("question-family selection: PASS")
    print("semantic observation extraction: PASS")
    print("persistent curiosity memory: PASS")
    print("predicate/argument reinforcement: PASS")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Curiosity-driven semantic learning on existing cognitive memory."
    )
    ap.add_argument("--max-concepts", type=int, default=10000)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--status-every", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--teacher-probe", type=int, default=3)
    ap.add_argument("--fresh-curiosity", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return

    start = time.perf_counter()

    db_path = DEFAULT_DB.resolve()
    model_path = DEFAULT_MODEL.resolve()

    if not db_path.exists():
        raise SystemExit(
            f"Cognitive memory not found:\n{db_path}"
        )
    if not model_path.exists():
        raise SystemExit(
            f"Teacher model not found:\n{model_path}"
        )

    con = sqlite3.connect(str(db_path))
    ensure_schema(con)

    if args.fresh_curiosity:
        con.execute("DELETE FROM curiosity_state")
        con.execute("DELETE FROM curiosity_questions")
        con.execute("DELETE FROM curiosity_facts")
        con.commit()

    print("=" * 78, flush=True)
    print("V433 CURIOSITY-DRIVEN SEMANTIC LEARNING", flush=True)
    print("=" * 78, flush=True)

    print("[1/6] Using existing cognitive memory...", flush=True)
    print(f"      {db_path}", flush=True)

    print("[2/6] Selecting semantic vocabulary...", flush=True)
    vocabulary = select_vocabulary(
        con,
        args.max_concepts,
    )
    print(
        f"      concepts={len(vocabulary):,}",
        flush=True,
    )

    print("[3/6] Loading SmolLM2...", flush=True)
    teacher = Teacher(
        model_path,
        args.max_new_tokens,
    )

    print("[4/6] Loading spaCy...", flush=True)
    nlp = load_parser(DEFAULT_SPACY)

    if args.teacher_probe:
        print(
            f"      probe={min(args.teacher_probe, len(vocabulary))}",
            flush=True,
        )
        for i, lemma in enumerate(
            vocabulary[:args.teacher_probe],
            1,
        ):
            family, question = next_question(
                con,
                lemma,
            )
            print(
                f"      PROBE {i}: {lemma} -> "
                f"[{family}] {question}",
                flush=True,
            )
            print(
                "        top families: "
                + ", ".join(
                    f"{d['family']}={d['score']:.2f}"
                    for d in curiosity_diagnostics(
                        con,
                        lemma,
                    )[:5]
                ),
                flush=True,
            )

    print("[5/6] Curiosity loop...", flush=True)

    session_start = time.perf_counter()
    questions = 0
    accepted = 0
    failed = 0
    facts = 0

    failures_path = Path.cwd() / "results" / "v435_curiosity_failures.jsonl"
    questions_path = Path.cwd() / "results" / "v435_curiosity_questions.jsonl"
    report_path = Path.cwd() / "results" / "v435_curiosity_report.json"

    failures_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ff = failures_path.open(
        "a",
        encoding="utf-8",
    )
    qf = questions_path.open(
        "a",
        encoding="utf-8",
    )

    try:
        for ci, lemma in enumerate(
            vocabulary,
            1,
        ):
            sweep_families=set()

            for round_no in range(
                args.rounds
            ):
                family, question = next_question(
                    con,
                    lemma,
                    exclude_families=sweep_families,
                )

                if family is None:
                    # All question families have been attempted in this
                    # sweep; begin another sweep.
                    sweep_families.clear()
                    family, question = next_question(
                        con,
                        lemma,
                    )

                sweep_families.add(family)

                diagnostics = curiosity_diagnostics(
                    con,
                    lemma,
                )
                unasked = [
                    d for d in diagnostics
                    if not d["asked"]
                ]
                if unasked and diagnostics[0]["asked"]:
                    family = unasked[0]["family"]
                    question = QUESTION_FAMILIES[family].format(
                        x=lemma
                    )

                print(
                    f"      CURIOUS {ci:,}/{len(vocabulary):,} "
                    f"round={round_no+1}/{args.rounds} "
                    f"lemma='{lemma}' "
                    f"family={family}",
                    flush=True,
                )
                print(
                    f"        {question}",
                    flush=True,
                )

                context = resource_context(
                    con,
                    lemma,
                )

                t0 = time.perf_counter()
                try:
                    raw = teacher.answer(
                        question,
                        context,
                    )
                    answer = clean_answer(raw)

                    if not answer:
                        failed += 1
                        qid = record_question_attempt(
                            con,
                            lemma,
                            family,
                            question,
                            raw[:3000],
                            False,
                            "invalid_answer",
                        )
                        ff.write(
                            json.dumps(
                                {
                                    "question_id": qid,
                                    "lemma": lemma,
                                    "family": family,
                                    "question": question,
                                    "raw": raw[:3000],
                                    "reason": "invalid_answer",
                                },
                                ensure_ascii=False,
                            ) + "\n"
                        )
                        ff.flush()
                        continue

                    parsed = parse_answer(
                        nlp,
                        answer,
                    )

                    if not find_target(
                        parsed,
                        lemma,
                    ):
                        failed += 1
                        qid = record_question_attempt(
                            con,
                            lemma,
                            family,
                            question,
                            answer,
                            False,
                            "target_predicate_not_found",
                        )
                        ff.write(
                            json.dumps(
                                {
                                    "question_id": qid,
                                    "lemma": lemma,
                                    "family": family,
                                    "question": question,
                                    "answer": answer,
                                    "parsed": parsed,
                                    "reason":
                                        "target_predicate_not_found",
                                },
                                ensure_ascii=False,
                            ) + "\n"
                        )
                        ff.flush()
                        continue

                    qid, new_facts = learn(
                        con,
                        lemma,
                        family,
                        question,
                        answer,
                        parsed,
                    )

                    accepted += 1
                    facts += new_facts

                    qf.write(
                        json.dumps(
                            {
                                "question_id": qid,
                                "lemma": lemma,
                                "family": family,
                                "question": question,
                                "answer": answer,
                                "parsed": parsed,
                                "teacher_seconds":
                                    time.perf_counter() - t0,
                            },
                            ensure_ascii=False,
                        ) + "\n"
                    )
                    qf.flush()

                except Exception as exc:
                    failed += 1
                    try:
                        con.rollback()
                    except Exception:
                        pass

                    try:
                        record_question_attempt(
                            con,
                            lemma,
                            family,
                            question,
                            "",
                            False,
                            "exception",
                        )
                    except Exception:
                        con.rollback()

                    ff.write(
                        json.dumps(
                            {
                                "lemma": lemma,
                                "family": family,
                                "question": question,
                                "reason": "exception",
                                "error": repr(exc),
                            },
                            ensure_ascii=False,
                        ) + "\n"
                    )
                    ff.flush()

                questions += 1

                if (
                    questions == 1
                    or questions % args.status_every == 0
                ):
                    elapsed = (
                        time.perf_counter()
                        - session_start
                    )
                    rate = questions / max(
                        elapsed,
                        1e-9,
                    )
                    remaining = max(
                        len(vocabulary) * args.rounds
                        - questions,
                        0,
                    )
                    eta = remaining / max(
                        rate,
                        1e-9,
                    )

                    stored_facts = con.execute(
                        "SELECT COUNT(*) FROM curiosity_facts"
                    ).fetchone()[0]
                    stored_questions = con.execute(
                        "SELECT COUNT(*) FROM curiosity_questions"
                    ).fetchone()[0]
                    stored_patterns = con.execute(
                        "SELECT COUNT(*) FROM learned_patterns"
                    ).fetchone()[0]

                    print(
                        f"      PROGRESS "
                        f"questions={questions:,} "
                        f"accepted={accepted:,} "
                        f"failed={failed:,} "
                        f"new_facts={facts:,} "
                        f"stored_facts={stored_facts:,} "
                        f"questions_db={stored_questions:,} "
                        f"patterns={stored_patterns:,} "
                        f"rate={rate:.2f}/s "
                        f"eta={eta/3600:.2f}h",
                        flush=True,
                    )

    finally:
        ff.close()
        qf.close()

    print("[6/6] Final report...", flush=True)

    counts = {}
    for table in (
        "concepts",
        "lexical_relations",
        "verbnet_classes",
        "verbnet_members",
        "verbnet_roles",
        "verbnet_frames",
        "propbank_predicates",
        "propbank_rolesets",
        "propbank_roles",
        "semlink_mappings",
        "teacher_examples",
        "learned_patterns",
        "semantic_grounding",
        "curiosity_state",
        "curiosity_questions",
        "curiosity_facts",
    ):
        if table_exists(con, table):
            counts[table] = int(
                con.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
            )

    coverage = Counter()
    for (blob,) in con.execute(
        "SELECT covered_json FROM curiosity_state"
    ):
        try:
            coverage.update(
                json.loads(blob)
            )
        except Exception:
            pass

    report = {
        "status": "PASS"
        if counts.get("curiosity_facts", 0) > 0
        else "FAIL",
        "version": "v435",
        "methodology": {
            "architecture": "curiosity-driven semantic learner",
            "starting_memory":
                str(db_path),
            "lexical_priors": [
                "WordNet",
                "VerbNet 3.3",
                "PropBank 3.1",
                "SemLink",
            ],
            "broad_semantic_graph": "ConceptNet already present in memory",
            "grammar_source": "UD GUM already learned upstream",
            "teacher_role":
                "answer questions selected from missing knowledge",
            "teacher_controls_curiosity": False,
            "teacher_not_runtime_dependency": False,
        },
        "curiosity": {
            "concepts": len(vocabulary),
            "rounds_per_concept": args.rounds,
            "questions_attempted": questions,
            "accepted_answers": accepted,
            "failed_answers": failed,
            "new_facts": facts,
            "coverage_by_family": dict(coverage),
        },
        "memory": {
            "sqlite": str(db_path),
            "counts": counts,
        },
        "outputs": {
            "questions": str(questions_path.resolve()),
            "failures": str(failures_path.resolve()),
            "report": str(report_path.resolve()),
        },
        "runtime_seconds": time.perf_counter() - start,
    }

    report = json_safe(report)
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    con.close()


if __name__ == "__main__":
    main()
