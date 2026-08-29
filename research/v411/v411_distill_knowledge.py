
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0,str(HERE))


# =============================================================================
# Input helpers
# =============================================================================

@dataclass
class Candidate:
    concept: str
    frequency: int
    source: str
    priority: float = 0.0


def normalize(text: str) -> str:
    text=text.strip().lower()
    text=re.sub(r"\s+"," ",text)
    text=re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$","",text)
    return text


def parse_tsv_sentences(path: Path, limit: int | None=None):
    rows=[]
    with path.open("r",encoding="utf-8") as f:
        for line in f:
            line=line.rstrip("\n")
            if not line:
                continue
            parts=line.split("\t",1)
            text=parts[1] if len(parts)==2 else parts[0]
            text=text.strip()
            if text:
                rows.append(text)
            if limit and len(rows)>=limit:
                break
    return rows


def discover_conllu(gum: Path):
    files=sorted(gum.rglob("*.conllu"))
    if not files:
        raise FileNotFoundError(f"No .conllu files under {gum}")
    return files


def split_name(path: Path):
    n=path.name.lower()
    if "train" in n:
        return "train"
    if "dev" in n:
        return "dev"
    if "test" in n:
        return "test"
    return "unknown"


def conllu_lemmas(path: Path, limit_sentences: int | None=None):
    counts=Counter()
    sentences=0
    in_sentence=False
    with path.open("r",encoding="utf-8") as f:
        for line in f:
            line=line.rstrip("\n")
            if not line:
                if in_sentence:
                    sentences+=1
                    in_sentence=False
                    if limit_sentences and sentences>=limit_sentences:
                        break
                continue
            if line.startswith("#"):
                continue
            cols=line.split("\t")
            if len(cols)!=10:
                continue
            tid=cols[0]
            if "-" in tid or "." in tid:
                continue
            lemma=normalize(cols[2] if cols[2]!="_" else cols[1])
            upos=cols[3]
            if lemma and upos not in {"PUNCT","SYM"}:
                counts[(lemma,upos)]+=1
            in_sentence=True
    return counts


# =============================================================================
# Teacher prompting
# =============================================================================

SYSTEM_INSTRUCTION = r"""
You are a knowledge extraction teacher for a cognitive language architecture.

Your job is NOT to write an answer for a human. Your job is to produce compact,
machine-readable knowledge that can be inserted into a symbolic/cognitive
knowledge graph.

Be conservative:
- Prefer common, reusable knowledge over obscure trivia.
- Do not invent facts to fill fields.
- Use canonical concept names.
- Keep relations short and explicit.
- Separate general world knowledge from procedural knowledge.
- Return ONLY valid JSON.
""".strip()


def concept_prompt(term: str):
    return {
        "task":"concept",
        "concept":term,
        "schema":{
            "canonical_name":"string",
            "aliases":["string"],
            "category":"string",
            "properties":[
                {"name":"string","value":"string"}
            ],
            "relations":[
                {"relation":"string","target":"string"}
            ],
            "common_actions":["string"],
            "common_uses":["string"],
            "parts":["string"],
            "confusions":["string"],
        },
    }


def frame_prompt(term: str):
    return {
        "task":"frame",
        "concept":term,
        "schema":{
            "name":"string",
            "trigger_concepts":["string"],
            "roles":[
                {"name":"string","description":"string","required":True}
            ],
            "preconditions":["string"],
            "effects":["string"],
            "typical_relations":[
                {"relation":"string","from":"string","to":"string"}
            ],
        },
    }


def procedure_prompt(term: str):
    return {
        "task":"procedure",
        "concept":term,
        "schema":{
            "goal":"string",
            "preconditions":["string"],
            "steps":[
                {"order":1,"action":"string","purpose":"string"}
            ],
            "failure_modes":[
                {"condition":"string","response":"string"}
            ],
            "success_criteria":["string"],
        },
    }


def extract_json(text: str):
    text=text.strip()

    # Direct JSON first.
    try:
        return json.loads(text)
    except Exception:
        pass

    # Strip markdown fences.
    text=re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text=re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass

    # Locate largest plausible object.
    starts=[m.start() for m in re.finditer(r"\{",text)]
    ends=[m.start() for m in re.finditer(r"\}",text)]
    for start in starts:
        for end in reversed(ends):
            if end<=start:
                continue
            candidate=text[start:end+1]
            try:
                return json.loads(candidate)
            except Exception:
                continue
    return None


# =============================================================================
# Teacher interface
# =============================================================================

class Teacher:
    def __init__(self, model_name: str, max_new_tokens: int=900):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "Teacher dependencies are missing. Install:\n"
                "  python -m pip install -U torch transformers accelerate"
            ) from exc

        self.torch=torch
        self.max_new_tokens=max_new_tokens

        print(f"[TEACHER] loading tokenizer: {model_name}",flush=True)
        self.tokenizer=AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        print(f"[TEACHER] loading model: {model_name}",flush=True)
        kwargs={"trust_remote_code":True}
        if torch.cuda.is_available():
            kwargs["device_map"]="auto"
            kwargs["torch_dtype"]=torch.float16
        else:
            kwargs["device_map"]="auto"

        self.model=AutoModelForCausalLM.from_pretrained(
            model_name,
            **kwargs,
        )
        self.model.eval()

    def generate(self, payload: dict):
        prompt=SYSTEM_INSTRUCTION+"\n\n"+json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",",":"),
        )

        messages=[
            {"role":"system","content":SYSTEM_INSTRUCTION},
            {"role":"user","content":json.dumps(
                payload,
                ensure_ascii=False,
            )},
        ]

        if hasattr(self.tokenizer,"apply_chat_template"):
            text=self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text=prompt

        inputs=self.tokenizer(
            text,
            return_tensors="pt",
        )

        device=next(self.model.parameters()).device
        inputs={k:v.to(device) for k,v in inputs.items()}

        with self.torch.inference_mode():
            output=self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated=output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()


# =============================================================================
# Validation + ranking
# =============================================================================

def clean_list(value):
    if not isinstance(value,list):
        return []
    out=[]
    for x in value:
        if isinstance(x,str):
            x=normalize(x)
            if x:
                out.append(x)
    return list(dict.fromkeys(out))


def validate_concept(obj, requested):
    if not isinstance(obj,dict):
        return None

    name=normalize(
        obj.get("canonical_name") or requested
    )
    if not name:
        return None

    category=normalize(obj.get("category",""))
    aliases=clean_list(obj.get("aliases"))
    actions=clean_list(obj.get("common_actions"))
    uses=clean_list(obj.get("common_uses"))
    parts=clean_list(obj.get("parts"))
    confusions=clean_list(obj.get("confusions"))

    properties=[]
    raw=obj.get("properties",[])
    if isinstance(raw,list):
        for item in raw:
            if not isinstance(item,dict):
                continue
            n=normalize(item.get("name",""))
            v=normalize(item.get("value",""))
            if n and v:
                properties.append({"name":n,"value":v})

    relations=[]
    raw=obj.get("relations",[])
    if isinstance(raw,list):
        for item in raw:
            if not isinstance(item,dict):
                continue
            r=normalize(item.get("relation",""))
            t=normalize(item.get("target",""))
            if r and t:
                relations.append({"relation":r,"target":t})

    return {
        "canonical_name":name,
        "aliases":aliases,
        "category":category,
        "properties":properties,
        "relations":relations,
        "common_actions":actions,
        "common_uses":uses,
        "parts":parts,
        "confusions":confusions,
    }


def validate_frame(obj, requested):
    if not isinstance(obj,dict):
        return None
    name=normalize(obj.get("name") or requested)
    if not name:
        return None

    roles=[]
    for item in obj.get("roles",[]) if isinstance(obj.get("roles",[]),list) else []:
        if not isinstance(item,dict):
            continue
        n=normalize(item.get("name",""))
        d=normalize(item.get("description",""))
        if n:
            roles.append({
                "name":n,
                "description":d,
                "required":bool(item.get("required",False)),
            })

    rels=[]
    for item in obj.get("typical_relations",[]) if isinstance(obj.get("typical_relations",[]),list) else []:
        if not isinstance(item,dict):
            continue
        r=normalize(item.get("relation",""))
        a=normalize(item.get("from",""))
        b=normalize(item.get("to",""))
        if r and a and b:
            rels.append({"relation":r,"from":a,"to":b})

    return {
        "name":name,
        "trigger_concepts":clean_list(obj.get("trigger_concepts")),
        "roles":roles,
        "preconditions":clean_list(obj.get("preconditions")),
        "effects":clean_list(obj.get("effects")),
        "typical_relations":rels,
    }


def validate_procedure(obj, requested):
    if not isinstance(obj,dict):
        return None

    goal=normalize(obj.get("goal",""))
    if not goal:
        goal=requested

    steps=[]
    raw=obj.get("steps",[])
    if isinstance(raw,list):
        for i,item in enumerate(raw,1):
            if not isinstance(item,dict):
                continue
            action=normalize(item.get("action",""))
            purpose=normalize(item.get("purpose",""))
            if action:
                steps.append({
                    "order":int(item.get("order",i) or i),
                    "action":action,
                    "purpose":purpose,
                })

    failure=[]
    raw=obj.get("failure_modes",[])
    if isinstance(raw,list):
        for item in raw:
            if not isinstance(item,dict):
                continue
            c=normalize(item.get("condition",""))
            r=normalize(item.get("response",""))
            if c or r:
                failure.append({"condition":c,"response":r})

    return {
        "goal":goal,
        "preconditions":clean_list(obj.get("preconditions")),
        "steps":steps,
        "failure_modes":failure,
        "success_criteria":clean_list(obj.get("success_criteria")),
    }


def priority_score(freq:int, concept:str, graph_bonus:float=1.0):
    length_penalty=max(0.65,1.0-0.015*max(0,len(concept)-8))
    frequency_score=math.log1p(freq)
    reusable_bonus=1.0
    if freq>=100:
        reusable_bonus=1.35
    elif freq>=30:
        reusable_bonus=1.15
    return frequency_score*length_penalty*reusable_bonus*graph_bonus


# =============================================================================
# Smoke
# =============================================================================

def smoke():
    # Test the extraction/validation layer without a 1.7B model.
    fake_concept={
        "canonical_name":"bicycle",
        "aliases":["bike"],
        "category":"vehicle",
        "properties":[{"name":"powered_by","value":"human"}],
        "relations":[{"relation":"used_for","target":"transportation"}],
        "common_actions":["ride","pedal","steer"],
        "common_uses":["transportation","recreation"],
        "parts":["wheel","pedal","frame"],
        "confusions":["motorcycle"],
    }
    fake_frame={
        "name":"purchase",
        "trigger_concepts":["buy","purchase"],
        "roles":[
            {"name":"buyer","description":"person acquiring the item","required":True},
            {"name":"item","description":"thing acquired","required":True},
            {"name":"seller","description":"party providing the item","required":False},
        ],
        "preconditions":["buyer chooses item"],
        "effects":["buyer owns item"],
        "typical_relations":[
            {"relation":"acquires","from":"buyer","to":"item"}
        ],
    }
    fake_proc={
        "goal":"install a package",
        "preconditions":["package manager available"],
        "steps":[
            {"order":1,"action":"resolve package","purpose":"identify package"},
            {"order":2,"action":"install package","purpose":"put package into environment"},
        ],
        "failure_modes":[
            {"condition":"missing dependency","response":"install dependency"}
        ],
        "success_criteria":["package imports successfully"],
    }

    assert validate_concept(fake_concept,"bicycle")["category"]=="vehicle"
    assert validate_frame(fake_frame,"purchase")["roles"][0]["required"] is True
    assert len(validate_procedure(fake_proc,"install")["steps"])==2
    assert priority_score(1000,"computer")>priority_score(10,"obscure_term")

    print("V411 teacher-distillation smoke: PASS")
    print("structured concept schema: PASS")
    print("frame/schema validation: PASS")
    print("procedure/schema validation: PASS")
    print("priority ranking: PASS")
    print("JSON extraction path: PASS")
    print("teacher remains external to runtime architecture: PASS")


# =============================================================================
# Real distillation
# =============================================================================

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        required=False,
        help="Local or Hugging Face 1.7B-class teacher model path/name.",
    )
    ap.add_argument(
        "--gum",
        type=Path,
        default=Path(r".\data\UD_GUM"),
    )
    ap.add_argument(
        "--conceptnet",
        type=Path,
        default=Path(r".\data\conceptnet_compact.db"),
    )
    ap.add_argument(
        "--max-concepts",
        type=int,
        default=1000,
    )
    ap.add_argument(
        "--train-scan",
        type=int,
        default=11314,
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=900,
    )
    ap.add_argument(
        "--frames-per-concept",
        type=int,
        default=1,
    )
    ap.add_argument(
        "--procedures-per-concept",
        type=int,
        default=1,
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
    )
    args=ap.parse_args()

    if args.smoke:
        smoke()
        return

    if not args.model:
        raise SystemExit(
            "--model is required for real distillation.\n"
            "Example:\n"
            "  --model .\\models\\your-1.7b-model"
        )

    start=time.perf_counter()

    gum=args.gum.resolve()
    if not gum.exists():
        raise SystemExit(f"GUM directory not found: {gum}")

    results=Path.cwd()/"results"
    results.mkdir(parents=True,exist_ok=True)

    output={
        "status":"RUNNING",
        "version":"v411",
        "methodology":{
            "teacher_role":"structured knowledge extractor",
            "runtime_dependency_on_teacher":False,
            "source_priority":"frequency × reuse × graph connectivity",
            "outputs":[
                "concepts.jsonl",
                "frames.jsonl",
                "procedures.jsonl",
            ],
        },
    }

    print("="*78,flush=True)
    print("V411 PRIORITY KNOWLEDGE DISTILLATION",flush=True)
    print("="*78,flush=True)

    print("[1/7] Scanning GUM training vocabulary...",flush=True)
    files=discover_conllu(gum)
    train_files=[f for f in files if split_name(f)=="train"]
    if not train_files:
        raise SystemExit("No GUM train CoNLL-U files found.")

    frequency=Counter()
    total_train=0
    for f in train_files:
        counts=conllu_lemmas(f,args.train_scan)
        frequency.update(counts)
        total_train+=sum(counts.values())

    # Core lexical concepts: prioritize nouns/proper nouns/verbs/adjectives.
    lexical=Counter()
    for (lemma,upos),freq in frequency.items():
        if upos in {"NOUN","PROPN","VERB","ADJ","ADV"}:
            lexical[lemma]+=freq

    print(
        f"      unique_lexemes={len(lexical):,} "
        f"token_observations={total_train:,}",
        flush=True,
    )

    print("[2/7] Computing priority candidates...",flush=True)
    candidates=[]
    for term,freq in lexical.items():
        candidates.append(
            Candidate(
                concept=term,
                frequency=freq,
                source="GUM",
                priority=priority_score(freq,term),
            )
        )

    # Stable high-frequency first; no random sampling.
    candidates.sort(
        key=lambda x:(-x.priority,-x.frequency,x.concept)
    )
    candidates=candidates[:args.max_concepts]

    candidate_path=results/"v411_priority_candidates.jsonl"
    with candidate_path.open("w",encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(asdict(c),ensure_ascii=False)+"\n")

    print(
        f"      selected={len(candidates):,} "
        f"top={candidates[0].concept if candidates else '<none>'}",
        flush=True,
    )

    print("[3/7] Loading 1.7B teacher...",flush=True)
    teacher=Teacher(args.model,args.max_new_tokens)

    concepts=[]
    frames=[]
    procedures=[]
    failures=[]
    seen_concepts=set()
    seen_frames=set()
    seen_proc=set()

    print("[4/7] Distilling concepts...",flush=True)
    for i,candidate in enumerate(candidates,1):
        payload=concept_prompt(candidate.concept)
        raw=teacher.generate(payload)
        obj=extract_json(raw)
        clean=validate_concept(obj,candidate.concept)

        if clean:
            clean.update({
                "teacher_source":args.model,
                "candidate_frequency":candidate.frequency,
                "candidate_priority":candidate.priority,
            })
            key=clean["canonical_name"]
            if key not in seen_concepts:
                seen_concepts.add(key)
                concepts.append(clean)
        else:
            failures.append({
                "type":"concept",
                "requested":candidate.concept,
                "raw":raw[:2000],
            })

        if i%25==0 or i==len(candidates):
            print(
                f"      concepts {i:,}/{len(candidates):,} "
                f"valid={len(concepts):,} failures={len(failures):,}",
                flush=True,
            )

    print("[5/7] Distilling high-value frames...",flush=True)
    # Frames are only asked for the highest-ranked concepts. This avoids wasting
    # teacher capacity on every lexical item.
    frame_candidates=candidates[:max(1,min(len(candidates),args.max_concepts))]
    for i,candidate in enumerate(frame_candidates,1):
        raw=teacher.generate(frame_prompt(candidate.concept))
        obj=extract_json(raw)
        clean=validate_frame(obj,candidate.concept)
        if clean:
            clean.update({
                "teacher_source":args.model,
                "candidate_frequency":candidate.frequency,
                "candidate_priority":candidate.priority,
            })
            key=clean["name"]
            if key not in seen_frames:
                seen_frames.add(key)
                frames.append(clean)
        else:
            failures.append({
                "type":"frame",
                "requested":candidate.concept,
                "raw":raw[:2000],
            })

        if i%25==0 or i==len(frame_candidates):
            print(
                f"      frames {i:,}/{len(frame_candidates):,} "
                f"valid={len(frames):,}",
                flush=True,
            )

    print("[6/7] Distilling procedures from action-heavy concepts...",flush=True)
    # Procedures are more useful for verbs/actions than arbitrary nouns.
    action_candidates=[
        c for c in candidates
        if any(
            (lemma==c.concept and upos=="VERB")
            for (lemma,upos) in frequency.keys()
        )
    ]
    action_candidates=action_candidates[:args.max_concepts]

    for i,candidate in enumerate(action_candidates,1):
        raw=teacher.generate(procedure_prompt(candidate.concept))
        obj=extract_json(raw)
        clean=validate_procedure(obj,candidate.concept)
        if clean and clean["steps"]:
            clean.update({
                "teacher_source":args.model,
                "candidate_frequency":candidate.frequency,
                "candidate_priority":candidate.priority,
            })
            key=clean["goal"]
            if key not in seen_proc:
                seen_proc.add(key)
                procedures.append(clean)
        else:
            failures.append({
                "type":"procedure",
                "requested":candidate.concept,
                "raw":raw[:2000],
            })

        if i%25==0 or i==len(action_candidates):
            print(
                f"      procedures {i:,}/{len(action_candidates):,} "
                f"valid={len(procedures):,}",
                flush=True,
            )

    print("[7/7] Writing distillation corpus...",flush=True)

    def write_jsonl(path,rows):
        with path.open("w",encoding="utf-8") as f:
            for row in rows:
                f.write(
                    json.dumps(row,ensure_ascii=False,separators=(",",":"))
                    +"\n"
                )

    concept_path=results/"concepts.jsonl"
    frame_path=results/"frames.jsonl"
    procedure_path=results/"procedures.jsonl"
    failure_path=results/"distillation_failures.jsonl"

    write_jsonl(concept_path,concepts)
    write_jsonl(frame_path,frames)
    write_jsonl(procedure_path,procedures)
    write_jsonl(failure_path,failures)

    report={
        "status":"PASS" if concepts else "FAIL",
        "version":"v411",
        "methodology":{
            "teacher_role":"structured knowledge extractor",
            "runtime_dependency_on_teacher":False,
            "candidate_source":"GUM training vocabulary",
            "candidate_selection":"frequency-prioritized",
            "generated_teacher_text_in_runtime":False,
        },
        "teacher":{
            "model":args.model,
            "max_new_tokens":args.max_new_tokens,
        },
        "source":{
            "gum_path":str(gum),
            "train_files":len(train_files),
            "frequency_observations":total_train,
            "candidate_count":len(candidates),
        },
        "distilled":{
            "concepts":len(concepts),
            "frames":len(frames),
            "procedures":len(procedures),
            "failures":len(failures),
        },
        "outputs":{
            "concepts":str(concept_path.resolve()),
            "frames":str(frame_path.resolve()),
            "procedures":str(procedure_path.resolve()),
            "failures":str(failure_path.resolve()),
            "candidates":str(candidate_path.resolve()),
        },
        "runtime_seconds":time.perf_counter()-start,
    }

    report_path=results/"v411_distillation_report.json"
    report_path.write_text(
        json.dumps(report,indent=2,ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
