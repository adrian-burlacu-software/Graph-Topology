from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
import sqlite3

from v665_semantic_core import (
    Graph,
    DistilledMemory,
    candidate_senses,
    Context,
    Attention,
    Hypothesis,
    SpaCyParser,
    relation_hypotheses,
    search,
    structural_question_frame,
)


def append_trace(
    path,
    payload,
):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                payload,
                ensure_ascii=False,
            )
            + "\n"
        )
        handle.flush()


class MemoryContext(Context):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)

    def load(self):
        if not self.path.exists():
            return
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
        self.turns = payload.get(
            "turns",
            [],
        )

    def save(self):
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.path.write_text(
            json.dumps(
                {
                    "active_subject": self.active_subject,
                    "turns": self.turns[-256:],
                    "entities": self.entities,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


class LocalLLMRuntime:
    """One shared SmolLM tokenizer/model for teacher and realizer."""
    def __init__(self, model_path):
        self.model_path = str(model_path)
        self.tokenizer = None
        self.model = None
        self.load_count = 0

    def load(self):
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("[LLM] loading shared SmolLM3 runtime...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            clean_up_tokenization_spaces=False,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            device_map="auto",
        )
        self.load_count += 1

    def generate(self, prompt, temperature=0.10, max_new_tokens=96):
        self.load()
        encoded = self.tokenizer(prompt, return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is not None:
            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }
        import torch
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else 1.0,
                top_p=0.90,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = encoded["input_ids"].shape[1]
        return self.tokenizer.decode(
            output[0, prompt_len:],
            skip_special_tokens=True,
        ).strip()


class Realizer:
    def __init__(self, runtime):
        self.runtime = runtime

    def load(self):
        self.runtime.load()

    def generate(self, prompt, temperature=0.10, max_new_tokens=96):
        return self.runtime.generate(
            prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        ).split("\n",1)[0].strip()

    def grounded_prompt(
        self,
        question,
        hypothesis,
        result,
        definition=None,
    ):
        subject_label = result.get(
            "subject_label",
            hypothesis.subject,
        )
        target_label = result.get(
            "target_label",
            result.get("target"),
        )

        return (
            "Write one short natural answer.\n"
            "The semantic result below is verified.\n"
            "Use only the verified result.\n"
            "Do not output URIs, internal IDs, relation names, "
            "or graph notation unless the user asked for them.\n"
            "Do not introduce another entity.\n\n"
            f"QUESTION: {question}\n"
            f"SUBJECT: {subject_label}\n"
            f"RELATION: {hypothesis.relation}\n"
            f"VERIFIED TARGET: {target_label}\n"
            + (
                f"DEFINITION: {definition}\n"
                if definition
                else ""
            )
            + "ANSWER:"
        )

    def conversation_prompt(self, question, history):
        recent=[]
        for turn in history[-6:]:
            if isinstance(turn, dict):
                recent.append(
                    "USER: "
                    + str(turn.get("question",turn.get("text","")))[:240]
                )
                recent.append(
                    "ASSISTANT: " + str(turn.get("answer",""))[:280]
                )
        return (
            "You are the conversational assistant.\n"
            "Be natural and concise.\n"
            "You may greet the user, tell jokes, and explain general concepts.\n"
            "Do not claim that the semantic graph verified facts it did not verify.\n\n"
            "RECENT:\n"
            + ("\n".join(recent) if recent else "none")
            + f"\nUSER: {question}\nASSISTANT:"
        )




class LiveSemanticTeacher:
    """Candidate-constrained semantic teacher using the shared LLM runtime."""
    def __init__(self, runtime, temperature=0.05):
        self.runtime = runtime
        self.temperature = float(temperature)

    def choose_goal(
        self,
        question,
        candidates,
        descriptions=None,
        frame=None,
    ):
        import json
        import re

        items=[
            {
                "id":i,
                "goal":str(candidate),
                "meaning":str(
                    (descriptions or {}).get(
                        candidate,
                        "",
                    )
                ),
            }
            for i,candidate
            in enumerate(candidates,1)
        ]

        prompt=(
            "Choose the USER'S SEMANTIC GOAL.\n"
            "You are not answering the user's question.\n"
            "The candidate goals are the complete allowed vocabulary.\n"
            "Do not mention or infer database relation names.\n"
            "Use the grammatical structure and question wording.\n"
            "For 'what is X?', choose definition.\n"
            "For 'is X Y?' where Y is a characteristic, choose property.\n"
            "For 'is X a Y?' choose type.\n"
            "For parts/components choose part.\n"
            "For abilities/actions choose capability.\n"
            "For place/location choose location.\n"
            "Choose the goal that represents what information is requested, "
            "not a goal that merely has a graph edge to a matching word.\n"
            'Return only JSON: {"selected_id": 1, "confidence": 0.0}\n\n'
            f"QUESTION: {question}\n"
            f"FRAME: {json.dumps(frame or {},ensure_ascii=False,sort_keys=True,separators=(',',':'))}\n"
            f"GOALS: {json.dumps(items,ensure_ascii=False,separators=(',',':'))}\n"
            "JSON:"
        )

        raw=self.runtime.generate(
            prompt,
            temperature=min(
                self.temperature,
                0.05,
            ),
        )

        match=re.search(
            r"\{.*\}",
            str(raw),
            re.DOTALL,
        )
        if not match:
            return None

        try:
            data=json.loads(match.group(0))
            selected_id=int(data.get("selected_id",0))
            confidence=float(data.get("confidence",0.0))
        except Exception:
            return None

        if not 1 <= selected_id <= len(items):
            return None

        return {
            "selected":items[
                selected_id-1
            ]["goal"],
            "confidence":max(
                0.0,
                min(
                    1.0,
                    confidence,
                ),
            ),
        }


    def choose(self, question, candidates, decision_type, descriptions=None, frame=None):
        import json
        import re

        if not candidates:
            return None

        items=[]
        for index,candidate in enumerate(candidates,1):
            item={"id":index,"candidate":str(candidate)}
            if descriptions:
                item["description"]=str(descriptions.get(candidate,""))
            items.append(item)

        if decision_type == "relation_from_frame_v662":
            prompt=(
                "Interpret the user's question as a semantic relation query.\n"
                "Do NOT answer the user.\n"
                "Do NOT choose a relation only because its example contains "
                "the same words.\n"
                "Use the grammatical argument structure and the meaning "
                "demonstrated by the graph examples.\n"
                "A relation is appropriate only if its examples express the "
                "same kind of relationship between the subject and the "
                "requested property, type, part, action, place, or other "
                "semantic target.\n"
                "Choose exactly one supplied relation.\n"
                'Return only JSON: {"selected_id": 1, "confidence": 0.0}\n\n'
                f"QUESTION: {question}\n"
                f"STRUCTURE: {json.dumps(frame or {}, sort_keys=True, separators=(',', ':'))}\n"
                f"RELATIONS: {json.dumps(items, ensure_ascii=False, separators=(',', ':'))}\n"
                "JSON:"
            )

        else:
            prompt=(
                "Choose ONE graph fact that directly answers the user's question.\n"
                "Use the relation meaning and target together.\n"
                "Do not choose a fact merely because it is associated with the target.\n"
                'Return only JSON: {"selected_id": 1, "confidence": 0.0}\n\n'
                f"QUESTION: {question}\n"
                f"FACTS: {json.dumps(items, ensure_ascii=False, separators=(',', ':'))}\n"
                "JSON:"
            )


        raw=self.runtime.generate(
            prompt,
            temperature=self.temperature,
            max_new_tokens=32,
        ).strip()

        match=re.search(
            r"\{.*?\}",
            raw,
            flags=re.DOTALL,
        )
        if not match:
            return None

        try:
            payload=json.loads(match.group(0))
            selected_id=int(payload["selected_id"])
            confidence=float(payload.get("confidence",0.5))
        except Exception:
            return None

        if not 1 <= selected_id <= len(candidates):
            return None

        confidence=max(0.0,min(1.0,confidence))
        return {
            "selected":str(candidates[selected_id-1]),
            "confidence":confidence,
            "raw":raw,
            "selected_id":selected_id,
            "format":"json",
        }



def normalize_graph_fact_candidates(
    candidates,
    subject=None,
):
    normalized = []
    seen = set()

    for item in candidates or []:
        if not isinstance(item, dict):
            continue

        relation = item.get("relation")
        target = item.get(
            "target",
            item.get("object"),
        )

        if not relation or target is None:
            continue

        candidate_subject = item.get(
            "subject",
            subject,
        )

        normalized_item = {
            "id": str(
                item.get(
                    "id",
                    f"{relation}|{target}",
                )
            ),
            "subject": str(
                candidate_subject
                if candidate_subject is not None
                else ""
            ),
            "relation": str(relation),
            "relation_meaning": str(
                item.get(
                    "relation_meaning",
                    str(relation).replace("_", " "),
                )
            ),
            "target": str(target),
            "label": str(
                item.get("label", target)
            ),
            "score": float(
                item.get("score", 0.0) or 0.0
            ),
        }

        key = (
            normalized_item["subject"],
            normalized_item["relation"],
            normalized_item["target"],
        )

        if key in seen:
            continue

        seen.add(key)
        normalized.append(normalized_item)

    return normalized



def question_argument_frame(parse):
    tokens=[
        item
        for item in (parse.tokens or [])
        if isinstance(item,dict)
    ]

    predicates=[
        str(item.get("text",""))
        for item in tokens
        if item.get("dep") in {
            "acomp",
            "attr",
            "dobj",
            "obj",
            "pobj",
            "ROOT",
        }
        and item.get("text")
    ]

    predicate_lemmas=[
        str(item.get("lemma",""))
        for item in tokens
        if item.get("dep") in {
            "acomp",
            "attr",
            "dobj",
            "obj",
            "pobj",
        }
        and item.get("lemma")
    ]

    return {
        "question":str(
            parse.question or ""
        ),
        "root":str(
            parse.root_lemma or ""
        ),
        "subjects":[
            str(x)
            for x in (parse.subjects or [])
        ],
        "objects":[
            str(x)
            for x in (parse.objects or [])
        ],
        "predicate_tokens":predicates,
        "predicate_lemmas":predicate_lemmas,
        "pronoun_subject":any(
            item.get("pos")=="PRON"
            and item.get("dep")=="nsubj"
            for item in tokens
        ),
    }


def classify_turn_mode(parse):
    """
    Classify the speech act before semantic relation inference.

    Semantic grounding is for explicit information-seeking questions.
    Ordinary conversation/request utterances (greetings, imperatives,
    social requests such as "tell me a joke") bypass the graph entirely.

    This is structural rather than concept/word based: interrogative/WH
    structure routes to semantic lookup; non-interrogative utterances route
    to conversation. Embedded WH clauses remain eligible for grounding.
    """
    tokens=[item for item in (parse.tokens or []) if isinstance(item,dict)]
    text=str(parse.text or "").strip()

    wh_questions={
        "WH_WHAT","WH_WHO","WH_WHERE","WH_WHEN",
        "WH_WHY","WH_HOW","WH_WHICH",
    }
    question_kind=str(getattr(parse,"question","") or "")
    has_wh=any(
        str(item.get("pos","")).upper()=="PRON"
        and str(item.get("text","")).lower() in
        {"what","who","where","when","why","how","which"}
        for item in tokens
    )
    has_question_mark=text.endswith("?")

    if question_kind in wh_questions or has_wh or has_question_mark:
        return "semantic"

    # Imperatives and ordinary non-interrogative utterances are conversational
    # requests/statements unless they contain an embedded WH question above.
    return "conversation"


def question_target_terms(parse):
    """
    Extract explicit non-subject semantic arguments from the frozen parse.

    This is structural, not relation-specific: nouns/adjectives/objects from
    the question become constraints on which graph object may satisfy the
    already-selected semantic goal.
    """
    tokens=[
        item
        for item in (parse.tokens or [])
        if isinstance(item,dict)
    ]

    stop={
        "a","an","the","it","he","she","they","them",
        "this","that","these","those","what","who","where",
        "when","why","how","which","is","are","was","were",
        "be","can","could","do","does","did","of","to",
        "for","with","and","or","on","in","at","from",
    }

    terms=[]
    for item in tokens:
        pos=str(item.get("pos","")).upper()
        dep=str(item.get("dep","")).lower()
        lemma=str(item.get("lemma") or item.get("text") or "").lower().strip()

        if not lemma or lemma in stop:
            continue
        if pos not in {"NOUN","PROPN","ADJ","ADV"}:
            continue
        if dep in {"nsubj","nsubjpass","det","aux","auxpass"}:
            continue
        if lemma not in terms:
            terms.append(lemma)

    # Keep the explicit object/predicate ordering visible for the trace.
    return terms[:8]


def choose_semantic_goal(
    teacher,
    graph,
    parse,
    subject,
):
    goals=[
        item
        for item in graph.semantic_goal_candidates()
        if item.get("available_relations")
    ]

    if not goals:
        return None,{
            "source":"no_available_goal",
            "candidates":[],
        }

    descriptions={
        item["goal"]:item["meaning"]
        for item in goals
    }

    syntax_frame=structural_question_frame(parse)
    argument_frame=question_argument_frame(parse)

    choice=teacher.choose_goal(
        parse.text,
        [item["goal"] for item in goals],
        descriptions=descriptions,
        frame={
            "syntax":syntax_frame,
            "arguments":argument_frame,
        },
    )

    if not choice:
        return None,{
            "source":"teacher_rejected",
            "candidates":[item["goal"] for item in goals],
            "candidate_details":goals,
        }

    selected=str(
        choice.get("selected") or ""
    )
    valid={
        item["goal"]
        for item in goals
    }

    if selected not in valid:
        return None,{
            "source":"invalid_teacher_selection",
            "selected":selected,
            "candidates":[item["goal"] for item in goals],
            "candidate_details":goals,
        }

    return selected,{
        "source":"llm_teacher",
        "confidence":choice.get("confidence",0.0),
        "candidates":[item["goal"] for item in goals],
        "candidate_details":goals,
        "frame":argument_frame,
        "syntax_frame":syntax_frame,
    }



def choose_relation_from_frame(
    teacher,
    graph,
    parse,
    hypotheses,
    relation_limit=32,
):
    subject=(
        hypotheses[0].subject
        if hypotheses
        else None
    )

    if not subject:
        return None,{
            "source":"none",
            "candidates":[],
        }

    evidence_candidates=graph.semantic_relation_candidates(
        subject=subject,
        query_text=parse.text,
        local_relations=[
            str(h.relation)
            for h in hypotheses
            if h.relation
        ],
        limit=relation_limit,
        examples_per_relation=4,
    )

    if not evidence_candidates:
        return None,{
            "source":"none",
            "candidates":[],
        }

    relation_names=[
        item["relation"]
        for item in evidence_candidates
    ]

    frame=structural_question_frame(parse)
    argument_frame=question_argument_frame(parse)
    frame_key=graph.relation_frame_key(
        frame
    )

    remembered=graph.relation_frame_lookup(
        frame_key,
        relation_names,
        min_confidence=0.85,
        min_count=2,
    )

    if remembered:
        return remembered["selected"],{
            "source":remembered["source"],
            "confidence":remembered["confidence"],
            "count":remembered["count"],
            "frame_key":frame_key,
            "frame":frame,
            "argument_frame":argument_frame,
            "candidates":relation_names,
            "candidate_details":evidence_candidates,
        }

    descriptions={
        item["relation"]:json.dumps(
            {
                "relation":item["relation"],
                "direct_question_matches":
                    item.get(
                        "direct_question_matches",
                        0,
                    ),
                "examples":item.get(
                    "examples",
                    [],
                ),
            },
            ensure_ascii=False,
            separators=(",",":"),
        )
        for item in evidence_candidates
    }

    choice=teacher.choose(
        parse.text,
        relation_names,
        "relation_from_frame_v662",
        descriptions=descriptions,
        frame={
            **frame,
            "argument_frame":
                argument_frame,
        },
    )

    if not choice:
        return None,{
            "source":"teacher_rejected",
            "confidence":0.0,
            "count":0,
            "frame_key":frame_key,
            "frame":frame,
            "argument_frame":argument_frame,
            "candidates":relation_names,
            "candidate_details":evidence_candidates,
        }

    selected=str(
        choice.get("selected") or ""
    )

    if selected not in relation_names:
        return None,{
            "source":"invalid_teacher_selection",
            "selected":selected,
            "confidence":0.0,
            "count":0,
            "frame_key":frame_key,
            "frame":frame,
            "argument_frame":argument_frame,
            "candidates":relation_names,
            "candidate_details":evidence_candidates,
        }

    graph.relation_frame_learn(
        frame_key,
        relation_names,
        selected,
        choice.get("confidence",0.0),
    )

    return selected,{
        "source":"llm_teacher",
        "confidence":choice.get("confidence",0.0),
        "count":1,
        "frame_key":frame_key,
        "frame":frame,
        "argument_frame":argument_frame,
        "candidates":relation_names,
        "candidate_details":evidence_candidates,
    }









def choose_graph_fact(
    teacher,
    distilled,
    graph,
    question,
    subject,
    candidates,
):
    candidates=normalize_graph_fact_candidates(
        candidates,
        subject=subject,
    )

    if not candidates:
        return None,{
            "source":"none",
            "selected":None,
            "confidence":0.0,
            "candidates":[],
        }

    ordered=sorted(
        candidates,
        key=lambda item:(
            -float(
                item.get(
                    "score",
                    0.0,
                ) or 0.0
            ),
            item["target"].lower(),
            item["relation"],
        ),
    )

    chosen=ordered[0]

    return chosen,{
        "source":"graph_deterministic",
        "selected":chosen["id"],
        "confidence":1.0,
        "candidates":[
            item["id"]
            for item in candidates
        ],
    }







def choose_best(
    ranked,
):
    if not ranked:
        return (
            Hypothesis(
                None,
                "",
                "conversation",
                0.0,
                {},
            ),
            {
                "success": False,
                "intent_only": True,
                "steps": 0,
                "path": [],
                "target": None,
                "attention": 0,
                "exploration": 0,
                "direct_proof": False,
                "proof_kind": None,
            },
        )

    ranked.sort(
        key=lambda row: (
            -row[0],
            row[1].relation,
            row[1].subject or "",
        )
    )

    return (
        ranked[0][1],
        ranked[0][2],
    )



def distilled_choice(
    teacher,
    distilled,
    decision_type,
    surface,
    context_text,
    candidates,
    descriptions=None,
):
    if not candidates:
        return None, "none", 0.0

    remembered = distilled.lookup(
        decision_type,
        surface,
        context_text,
        candidates,
    )

    if remembered:
        return (
            remembered["selected"],
            "distilled_memory",
            remembered["confidence"],
        )

    choice = teacher.choose(
        context_text,
        candidates,
        decision_type,
        descriptions=descriptions,
    )

    if not choice:
        return None, "teacher_rejected", 0.0

    distilled.learn(
        decision_type,
        surface,
        context_text,
        candidates,
        choice["selected"],
        choice["confidence"],
    )

    return (
        choice["selected"],
        "llm_teacher",
        choice["confidence"],
    )



def apply_live_distillation(
    question,
    parse,
    hypotheses,
    graph,
    distilled,
    teacher,
):
    if not hypotheses:
        return hypotheses,{
            "source":"none",
            "decision":None,
        }

    base=hypotheses[0]

    if base.intent not in {
        "concept_lookup",
        "relation_lookup",
    }:
        return hypotheses,{
            "source":"none",
            "decision":None,
        }

    goal,info=choose_semantic_goal(
        teacher,
        graph,
        parse,
        base.subject,
    )

    if not goal:
        return hypotheses,{
            "source":info.get(
                "source",
                "teacher_rejected",
            ),
            "decision":None,
        }

    evidence={
        **base.evidence,
        "semantic_goal":goal,
        "semantic_goal_source":info[
            "source"
        ],
        "semantic_goal_confidence":info.get(
            "confidence",
            0.0,
        ),
        "semantic_goal_candidates":info[
            "candidates"
        ],
        "semantic_goal_details":info.get(
            "candidate_details",
            [],
        ),
        "argument_frame":info.get(
            "frame",
            {},
        ),
        "syntax_frame":info.get(
            "syntax_frame",
            {},
        ),
        "target_terms":question_target_terms(parse),
    }

    # Keep the clean goal as the hypothesis relation. Raw graph relations are
    # deliberately not exposed to the teacher.
    hypotheses=[
        Hypothesis(
            base.subject,
            goal,
            "relation_lookup",
            base.lexical_score,
            evidence,
        )
    ]

    # Context resolution + goal-specific fact lookup. This adapter is entirely
    # internal to the graph and cannot be selected by the language model.
    if (
        base.subject
        and base.evidence.get(
            "entity_resolution",
            {},
        ).get("status")
        == "context_resolved"
    ):
        target_terms=question_target_terms(parse)
        facts=graph.find_goal_facts(
            base.subject,
            goal,
            query_text=question,
            target_terms=target_terms,
            limit=24,
        )

        evidence["target_terms"]=target_terms

        if facts:
            chosen=sorted(
                facts,
                key=lambda item:(
                    item["relation"],
                    item["label"].lower(),
                ),
            )[0]

            hypotheses=[
                Hypothesis(
                    base.subject,
                    goal,
                    "relation_lookup",
                    base.lexical_score,
                    {
                        **evidence,
                        "raw_relation":
                            chosen["relation"],
                        "selected_fact_target":
                            chosen["object"],
                        "selected_fact_label":
                            chosen["label"],
                        "fact_source":
                            "semantic_goal_adapter",
                        "fact_confidence":1.0,
                        "fact_target_terms":question_target_terms(parse),
                    },
                )
            ]

    return hypotheses,{
        "source":info["source"],
        "decision":goal,
    }









def validate_grounded_answer(
    answer,
    hypothesis,
    result,
):
    text = str(answer or "").strip()
    if not text:
        return False

    allowed = {
        str(hypothesis.subject or "").rstrip(".,"),
        str(result.get("target") or "").rstrip(".,"),
    }

    entities = re.findall(
        r"https?://\S+|en:[A-Za-z0-9_.:-]+",
        text,
    )

    return all(
        item.rstrip(".,") in allowed
        for item in entities
    )



def clean_surface_answer(answer):
    text=str(answer or "").strip()
    text=text.replace("```","").strip()
    text=re.sub(
        r"^(answer|response)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text



def handle_turn(
    question,
    graph,
    parser,
    memory,
    attention,
    realizer,
    teacher,
    distilled,
    args,
):
    started = time.perf_counter()

    t0 = time.perf_counter()
    parse = parser.parse(
        question
    )
    parse_seconds = (
        time.perf_counter()
        - t0
    )

    # Speech-act routing happens before graph hypothesis generation. This
    # prevents conversational requests such as "tell me a joke" from being
    # interpreted as semantic definitions merely because a concept named
    # "joke" exists in the graph.
    turn_mode = classify_turn_mode(parse)
    if turn_mode == "conversation":
        t_conv = time.perf_counter()
        answer = realizer.generate(
            realizer.conversation_prompt(
                question,
                memory.turns,
            ),
            temperature=0.15,
        )
        llm_seconds = time.perf_counter() - t_conv
        total_seconds = time.perf_counter() - started
        result = {
            "success": False,
            "intent_only": True,
            "steps": 0,
            "path": [],
            "target": None,
            "attention": 0,
            "exploration": 0,
            "direct_proof": False,
            "proof_kind": "conversation",
        }
        trace = {
            "question": question,
            "answer": answer,
            "route": {
                "intent": "conversation",
                "subject": None,
                "relation": None,
                "mode": "conversation",
                "success": False,
                **result,
            },
            "parse": {
                "text": parse.text,
                "question": parse.question,
                "root": parse.root,
                "root_lemma": parse.root_lemma,
                "subjects": list(parse.subjects or []),
                "objects": list(parse.objects or []),
            },
            "goal_selection": [],
            "hypotheses": [],
            "search": result,
            "distillation": {
                "source": "none",
                "decision": None,
            },
            "realization_cache": None,
            "realization_guard": "not_applicable",
            "timing": {
                "parse_seconds": parse_seconds,
                "hypothesis_seconds": 0.0,
                "distillation_seconds": 0.0,
                "search_seconds": 0.0,
                "llm_seconds": llm_seconds,
                "total_seconds": total_seconds,
                "realization_seconds": llm_seconds,
            },
        }
        memory.turns.append({"question": question, "answer": answer})
        memory.save()
        return answer, trace

    t0 = time.perf_counter()
    hypotheses = relation_hypotheses(
        parse,
        graph,
        memory,
        max_n=args.max_hypotheses,
    )
    hypothesis_seconds = (
        time.perf_counter()
        - t0
    )

    t0 = time.perf_counter()
    distill_info = apply_live_distillation(
        question,
        parse,
        hypotheses,
        graph,
        distilled,
        teacher,
    )
    distill_seconds = (
        time.perf_counter()
        - t0
    )
    hypotheses = distill_info[0]
    distill_info = distill_info[1]

    t0 = time.perf_counter()

    ranked = []

    runtime_hypotheses = []

    for hypothesis in hypotheses:
        semantic_goal = (
            hypothesis.evidence.get(
                "semantic_goal"
            )
            if isinstance(
                hypothesis.evidence,
                dict,
            )
            else None
        )

        if semantic_goal:
            raw_relations=graph.semantic_relations_for_goal(
                semantic_goal
            )

            for raw_relation in raw_relations:
                runtime_hypotheses.append(
                    Hypothesis(
                        hypothesis.subject,
                        raw_relation,
                        hypothesis.intent,
                        hypothesis.lexical_score,
                        {
                            **hypothesis.evidence,
                            "semantic_goal":
                                semantic_goal,
                            "semantic_goal_runtime_relation":
                                raw_relation,
                        },
                    )
                )
        else:
            runtime_hypotheses.append(
                hypothesis
            )

    for index,hypothesis in enumerate(
        runtime_hypotheses
    ):
        selected_fact = (
            hypothesis.evidence.get(
                "selected_fact_target"
            )
            if isinstance(
                hypothesis.evidence,
                dict,
            )
            else None
        )
        raw_relation = (
            hypothesis.evidence.get(
                "raw_relation"
            )
            if isinstance(
                hypothesis.evidence,
                dict,
            )
            else None
        )

        if (
            selected_fact
            and raw_relation
            and hypothesis.evidence.get(
                "entity_resolution",
                {},
            ).get("status")
            == "context_resolved"
        ):
            result=graph.prove_edge(
                hypothesis.subject,
                raw_relation,
                selected_fact,
            )
        else:
            result=search(
                graph,
                attention,
                hypothesis,
                budget=args.goal_budget,
                per_node=args.per_node,
                max_depth=args.max_depth,
            )

        semantic_match=float(
            hypothesis.lexical_score
        )

        score=(
            4.0*semantic_match
            +(
                1.0
                if result.get(
                    "success",
                    False,
                )
                else 0.0
            )
            +(
                0.15
                if result.get(
                    "direct_proof",
                    False,
                )
                else 0.0
            )
        )

        ranked.append(
            (
                score,
                hypothesis,
                result,
            )
        )


    search_seconds = (
        time.perf_counter()
        - t0
    )

    selected, result = choose_best(
        ranked
    )

    # Keep the public semantic goal stable even though graph search uses
    # internal raw relations.
    if (
        isinstance(selected.evidence, dict)
        and selected.evidence.get(
            "semantic_goal"
        )
    ):
        result["semantic_goal"] = selected.evidence[
            "semantic_goal"
        ]
        result["semantic_relation"] = selected.evidence[
            "semantic_goal"
        ]

    # Resolve conversational fallback only after the semantic attempt.
    t0 = time.perf_counter()

    entity_resolution = (
        selected.evidence.get(
            "entity_resolution",
            {},
        )
        if isinstance(
            selected.evidence,
            dict,
        )
        else {}
    )

    if not isinstance(
        entity_resolution,
        dict,
    ):
        entity_resolution = {}

    realization_cache = None
    realization_guard = "not_applicable"

    if result.get("success", False):
        result["subject_label"] = graph.node_label(
            selected.subject
        )
        if result.get("target") is not None:
            result["target_label"] = (
                graph.node_label(
                    result["target"]
                )
                if (
                    isinstance(
                        result.get("target"),
                        str,
                    )
                    and (
                        result.get("target","").startswith("en:")
                        or result.get("target","").startswith("http")
                    )
                )
                else result.get("target")
            )

    if result.get(
        "success",
        False,
    ):
        realization_cache = graph.get_realized_answer(
            question,
            selected.subject,
            selected.evidence.get(
                "semantic_goal",
                selected.relation,
            )
            if isinstance(
                selected.evidence,
                dict,
            )
            else selected.relation,
            result.get("target"),
            result.get("path", []),
        )

        if realization_cache:
            answer = realization_cache["answer"]
            mode = "grounded_cache"
        else:
            distilled_sense = (
                selected.evidence.get(
                    "distilled_sense"
                )
                if isinstance(
                    selected.evidence,
                    dict,
                )
                else None
            )

            sense_node = (
                distilled_sense.get("node")
                if isinstance(
                    distilled_sense,
                    dict,
                )
                else None
            )

            definition = graph.definition(
                selected.subject,
                sense_node=sense_node,
            )

            answer = clean_surface_answer(
                realizer.generate(
                    realizer.grounded_prompt(
                        question,
                        selected,
                        result,
                        definition,
                    ),
                    temperature=0.05,
                )
            )

            if not validate_grounded_answer(
                answer,
                selected,
                result,
            ):
                answer = (
                    f"{selected.subject} "
                    f"{selected.relation.replace('_', ' ')} "
                    f"{result.get('target')}."
                )
                realization_guard = "fallback_exact_fact"
            else:
                realization_guard = "accepted"

            graph.save_realized_answer(
                question,
                selected.subject,
                selected.relation,
                result.get("target"),
                result.get("path", []),
                answer,
            )

            mode = "grounded"

    elif selected.intent == "entity_unresolved":
        answer = (
            "I couldn't verify that entity "
            "in the semantic graph."
        )
        mode = "unresolved"

    elif selected.intent == "relation_lookup":
        # A semantic/factual request is never allowed to fall through to
        # conversational generation merely because graph proof failed.
        # Doing so would let the LLM answer from pretrained world knowledge
        # and violate the graph-authoritative grounding boundary.
        goal = (
            selected.evidence.get("semantic_goal")
            if isinstance(selected.evidence, dict)
            else None
        )
        target_terms = (
            selected.evidence.get("target_terms", [])
            if isinstance(selected.evidence, dict)
            else []
        )
        if goal and target_terms:
            target_text = ", ".join(str(x) for x in target_terms)
            answer = (
                "I couldn't verify that "
                f"the subject has the requested {goal} ({target_text}) "
                "in the semantic graph."
            )
        else:
            answer = (
                "I couldn't verify that requested fact "
                "in the semantic graph."
            )
        mode = "unverified"

    else:
        answer = realizer.generate(
            realizer.conversation_prompt(
                question,
                memory.turns,
            ),
            temperature=0.15,
        )
        mode = "conversation"


    llm_seconds = (
        time.perf_counter()
        - t0
    )

    total_seconds = (
        time.perf_counter()
        - started
    )

    route = {
        "intent": selected.intent,
        "subject": selected.subject,
        "relation": selected.relation,
        "mode": mode,
        "success": bool(
            result.get(
                "success",
                False,
            )
        ),
        "direct_proof": bool(
            result.get(
                "direct_proof",
                False,
            )
        ),
        "proof_kind": result.get(
            "proof_kind"
        ),
        "steps": result.get(
            "steps",
            0,
        ),
        "path": list(
            result.get(
                "path",
                [],
            )
        ),
        "target": result.get(
            "target"
        ),
        "selected_fact_target": (
            selected.evidence.get(
                "selected_fact_target"
            )
            if isinstance(
                selected.evidence,
                dict,
            )
            else None
        ),
        "attention": result.get(
            "attention",
            0,
        ),
        "exploration": result.get(
            "exploration",
            0,
        ),
    }

    trace = {
        "timestamp": time.time(),
        "question": question,
        "answer": answer,
        "entity_resolution_kind": (
            entity_resolution.get("candidates", [{}])[0].get("kind")
            if entity_resolution.get("candidates")
            else None
        ),
        "context": {
            "active_subject_before_turn": memory.active_subject,
            "selected_subject": selected.subject,
        },
        "route": route,
        "entity_resolution": entity_resolution,
        "goal_selection": [
            {
                "relation": h.relation,
                "intent": h.intent,
                "lexical_score": h.lexical_score,
                "verified": bool(r.get("success", False)),
                "direct_proof": bool(r.get("direct_proof", False)),
                "steps": r.get("steps", 0),
            }
            for _, h, r in ranked[:args.max_hypotheses]
        ],
        "parse": asdict(parse),
        "selected": asdict(selected),
        "hypotheses": [
            asdict(
                h
            )
            for _, h, _
            in ranked[:args.max_hypotheses]
        ],
        "search": result,
        "distillation": distill_info,
        "realization_cache": realization_cache,
        "realization_guard": realization_guard,
        "timing": {
            "parse_seconds": parse_seconds,
            "hypothesis_seconds": hypothesis_seconds,
            "distillation_seconds": distill_seconds,
            "search_seconds": search_seconds,
            "llm_seconds": (
                0.0
                if realization_cache
                else llm_seconds
            ),
            "total_seconds": total_seconds,
            "realization_seconds": llm_seconds,
        },
    }

    memory.active_subject = selected.subject
    memory.turns.append(
        trace
    )
    memory.turns = memory.turns[-256:]
    memory.save()

    return answer, trace


def preflight_symbol_audit():
    import builtins
    import ast as _ast
    from pathlib import Path as _Path

    tree = _ast.parse(
        _Path(__file__).read_text(
            encoding="utf-8"
        )
    )

    defined = {
        node.name
        for node in _ast.walk(tree)
        if isinstance(
            node,
            (
                _ast.FunctionDef,
                _ast.AsyncFunctionDef,
                _ast.ClassDef,
            ),
        )
    }

    imported = set()

    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            imported.update(
                alias.asname or alias.name.split(".")[0]
                for alias in node.names
            )
        elif isinstance(node, _ast.ImportFrom):
            imported.update(
                alias.asname or alias.name
                for alias in node.names
            )

    builtins_set = set(
        dir(builtins)
    )

    calls = {
        node.func.id
        for node in _ast.walk(tree)
        if isinstance(node, _ast.Call)
        and isinstance(node.func, _ast.Name)
    }

    missing = sorted(
        name
        for name in calls
        if name not in defined
        and name not in imported
        and name not in builtins_set
    )

    # These are deliberately resolved dynamically inside the shared LLM
    # runtime after startup.
    allowed = {
        "torch",
        "transformers",
        "AutoTokenizer",
        "AutoModelForCausalLM",
    }

    missing = [
        name
        for name in missing
        if name not in allowed
    ]

    if missing:
        raise RuntimeError(
            "Gateway symbol audit failed: "
            + ", ".join(missing)
        )

    return True



def main():
    preflight_symbol_audit()
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--database",
        required=True,
    )
    ap.add_argument(
        "--output",
        default="",
    )
    ap.add_argument(
        "--trace-output",
        required=True,
    )
    ap.add_argument(
        "--memory-output",
        required=True,
    )
    ap.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
    )
    ap.add_argument(
        "--llm-model",
        default=r"C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B",
    )
    ap.add_argument(
        "--mode",
        choices=("chat","smoke"),
        default="chat",
    )
    ap.add_argument(
        "--max-hypotheses",
        type=int,
        default=12,
    )
    ap.add_argument(
        "--goal-budget",
        type=int,
        default=40,
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
        "--seed",
        type=int,
        default=63300,
    )

    args = ap.parse_args()

    database = Path(
        args.database
    ).resolve()
    trace_path = Path(
        args.trace_output
    ).resolve()
    memory_path = Path(
        args.memory_output
    ).resolve()

    print(
        "=== V665 FULL SEMANTIC CHAT ===",
        flush=True,
    )
    print(
        f"database : {database}",
        flush=True,
    )
    stats_probe = time.perf_counter()
    try:
        with sqlite3.connect(
            str(database),
            timeout=30.0,
        ) as stats_conn:
            node_count = stats_conn.execute(
                "SELECT COUNT(*) FROM nodes"
            ).fetchone()[0]
            edge_count = stats_conn.execute(
                "SELECT COUNT(*) FROM edges"
            ).fetchone()[0]
        print(
            f"graph stats: nodes={node_count:,} edges={edge_count:,} "
            f"probe={time.perf_counter()-stats_probe:.3f}s",
            flush=True,
        )
    except Exception as exc:
        print(
            f"graph stats: unavailable ({exc})",
            flush=True,
        )
    print(
        "knowledge: ALL WordNet + ALL English ConceptNet",
        flush=True,
    )
    print(
        "grammar  : frozen spaCy",
        flush=True,
    )
    print(
        "search   : contextual path attention + bounded BFS",
        flush=True,
    )

    graph = Graph(
        database,
        args.cache_entries,
    )
    parser = SpaCyParser(
        args.spacy_model
    )

    memory = MemoryContext(
        memory_path
    )
    memory.load()

    attention = Attention(
        0.65
    )

    distilled = DistilledMemory(
        graph
    )

    llm_runtime = LocalLLMRuntime(
        args.llm_model
    )

    teacher = LiveSemanticTeacher(
        llm_runtime,
        temperature=0.05,
    )

    realizer = Realizer(
        llm_runtime
    )

    print(
        "\nTOPICS / EXAMPLES",
        flush=True,
    )

    distilled_stats = distilled.counts()
    print(
        "distilled semantic memory: "
        f"decisions={distilled_stats['decisions']} "
        f"observations={distilled_stats['observations']}",
        flush=True,
    )

    realized_stats = graph.realized_answer_stats()
    print(
        "semantic answer cache: "
        f"entries={realized_stats['entries']} "
        f"hits={realized_stats['hits']}",
        flush=True,
    )

    topics = [
        "What is a dog?",
        "What is an animal?",
        "What is a house?",
        "What is water?",
        "What is a person?",
        "What can a dog do?",
        "What parts does a dog have?",
        "What is a bird?",
        "What is a tree?",
        "What is food?",
    ]

    for index, question in enumerate(
        topics,
        1,
    ):
        print(
            f"  {index}. {question}",
            flush=True,
        )

    print(
        "\nCommands: help, exit",
        flush=True,
    )

    if args.mode == "smoke":
        questions = topics[:7]
    else:
        questions = None

    if questions:
        for question in questions:
            answer, trace = handle_turn(
                question,
                graph,
                parser,
                memory,
                attention,
                realizer,
                teacher,
                distilled,
                args,
            )
            print(
                f"\nQ: {question}",
                flush=True,
            )
            print(
                f"A: {answer}",
                flush=True,
            )
            print(
                f"  mode={trace['route']['mode']} "
                f"route="
                + (
                    " -> ".join(
                        trace["route"]["path"]
                    )
                    if trace["route"]["path"]
                    else "conversation"
                ),
                flush=True,
            )
            print(
                f"  result="
                f"{'VERIFIED' if trace['route']['success'] else 'NOT VERIFIED'} "
                f"steps={trace['route']['steps']} "
                f"attention={trace['route']['attention']} "
                f"exploration={trace['route']['exploration']}",
                flush=True,
            )
            print(
                f"  time={trace['timing']['total_seconds']:.3f}s "
                f"(search={trace['timing']['search_seconds']:.3f}s "
                f"llm={trace['timing']['llm_seconds']:.3f}s)",
                flush=True,
            )
            append_trace(
                trace_path,
                trace,
            )
    else:
        while True:
            try:
                question = input(
                    "chat> "
                ).strip()
            except (
                EOFError,
                KeyboardInterrupt,
            ):
                print()
                break

            if not question:
                continue

            if question.lower() in {
                "exit",
                "quit",
            }:
                break

            if question.lower() in {
                "help",
                "?",
            }:
                for index, item in enumerate(
                    topics,
                    1,
                ):
                    print(
                        f"  {index}. {item}",
                        flush=True,
                    )
                continue

            answer, trace = handle_turn(
                question,
                graph,
                parser,
                memory,
                attention,
                realizer,
                teacher,
                distilled,
                args,
            )

            print(
                f"answer: {answer}",
                flush=True,
            )
            print(
                f"  mode={trace['route']['mode']} "
                f"intent={trace['route']['intent']} "
                f"relation={trace['route']['relation']!r}",
                flush=True,
            )
            print(
                "  route="
                + (
                    " -> ".join(
                        trace["route"]["path"]
                    )
                    if trace["route"]["path"]
                    else "conversation"
                ),
                flush=True,
            )
            print(
                f"  result="
                f"{'VERIFIED' if trace['route']['success'] else 'NOT VERIFIED'} "
                f"steps={trace['route']['steps']} "
                f"attention={trace['route']['attention']} "
                f"exploration={trace['route']['exploration']}",
                flush=True,
            )
            print(
                f"  time={trace['timing']['total_seconds']:.3f}s "
                f"(search={trace['timing']['search_seconds']:.3f}s "
                f"llm={trace['timing']['llm_seconds']:.3f}s)",
                flush=True,
            )

            append_trace(
                trace_path,
                trace,
            )

    print(
        "\n=== V665 COMPLETE ===",
        flush=True,
    )
    print(
        f"TRACE  : {trace_path}",
        flush=True,
    )
    distilled_stats = distilled.counts()
    realized_stats = graph.realized_answer_stats()
    print(
        f"DISTILLED DB: decisions={distilled_stats['decisions']} "
        f"observations={distilled_stats['observations']}",
        flush=True,
    )
    print(
        f"REALIZED CACHE: entries={realized_stats['entries']} "
        f"hits={realized_stats['hits']}",
        flush=True,
    )
    print(
        f"MEMORY : {memory_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
