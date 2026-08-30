
from __future__ import annotations

import gzip
import json
import pickle
import re
from pathlib import Path


SKIP_NAMES={
    "schema.json","schemas.json","ontology.json",
    "requirements.txt","README.md",
}


def clean_text(text):
    return re.sub(r"\s+"," ",str(text or "").strip())[:4000]


def load_json(path):
    path=Path(path)
    opener=gzip.open if path.name.lower().endswith(".gz") else open
    with opener(
        path,"rt",encoding="utf-8-sig",errors="replace"
    ) as f:
        raw=f.read().strip()

    if not raw:
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"malformed JSON: {path} "
            f"(line={exc.lineno}, column={exc.colno})"
        ) from exc


def load_jsonl(path):
    path=Path(path)
    opener=gzip.open if path.name.lower().endswith(".gz") else open

    with opener(
        path,"rt",encoding="utf-8-sig",errors="replace"
    ) as f:
        for line_no,line in enumerate(f,1):
            text=line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                yield line_no,json.loads(text)
            except json.JSONDecodeError as exc:
                yield line_no,exc


def load_pickle_robust(path):
    errors=[]
    for kwargs in (
        {},
        {"encoding":"latin1"},
        {"encoding":"bytes"},
    ):
        try:
            with Path(path).open("rb") as f:
                return pickle.load(f,**kwargs)
        except Exception as exc:
            errors.append(exc)

    raise RuntimeError(
        f"pickle load failed for {path}: "
        +"; ".join(type(e).__name__ for e in errors)
    )


def _walk(obj,depth=0,max_depth=8):
    if depth>max_depth:
        return
    yield obj
    if isinstance(obj,dict):
        for v in obj.values():
            yield from _walk(v,depth+1,max_depth)
    elif isinstance(obj,(list,tuple)):
        for v in obj:
            yield from _walk(v,depth+1,max_depth)


def decode_vocab(obj):
    """
    Robustly detect:
      id -> token
      token -> id
      flat token list
      nested vocabulary objects
    """
    for node in _walk(obj):
        if isinstance(node,(list,tuple)):
            sample=list(node[:min(1000,len(node))])
            if sample and sum(isinstance(x,(str,bytes)) for x in sample) >= max(
                5,int(len(sample)*0.8)
            ):
                out={}
                for i,x in enumerate(node):
                    if isinstance(x,bytes):
                        x=x.decode("utf-8","replace")
                    elif not isinstance(x,str):
                        continue
                    if str(x).strip():
                        out[i]=str(x)
                if len(out)>=5:
                    return out

        if isinstance(node,dict):
            id_to_token={}
            token_to_id={}
            for k,v in node.items():
                try:
                    i=int(k)
                except Exception:
                    i=None

                if i is not None:
                    if isinstance(v,bytes):
                        token=v.decode("utf-8","replace")
                    elif isinstance(v,str):
                        token=v
                    else:
                        token=None
                    if token and token.strip():
                        id_to_token[i]=token

                if isinstance(k,(str,bytes)):
                    key=k.decode("utf-8","replace") if isinstance(k,bytes) else k
                    try:
                        j=int(v)
                    except Exception:
                        j=None
                    if j is not None and key.strip():
                        token_to_id[j]=key

            if len(id_to_token)>=5:
                return id_to_token
            if len(token_to_id)>=5:
                return token_to_id

    return {}



def normalize_ubuntu_vocab(obj):
    """
    Return id -> token while preserving the original token IDs.

    Supports:
      id->token dict
      token->id dict
      flat token list
      nested legacy containers
    """
    direct=decode_vocab(obj)
    if direct:
        return direct

    # Search nested containers explicitly and score candidates by size.
    best={}
    for node in _walk(obj):
        candidate={}
        if isinstance(node,dict):
            for k,v in node.items():
                try:
                    i=int(k)
                except Exception:
                    continue
                if isinstance(v,bytes):
                    v=v.decode("utf-8","replace")
                if isinstance(v,str) and v.strip():
                    candidate[i]=v
        elif isinstance(node,(list,tuple)):
            candidate={
                i:(x.decode("utf-8","replace") if isinstance(x,bytes) else str(x))
                for i,x in enumerate(node)
                if isinstance(x,(str,bytes)) and str(x).strip()
            }

        if len(candidate)>len(best):
            best=candidate

    return best


def ubuntu_container_summary(obj):
    summaries=[]
    for node in _walk(obj):
        if not isinstance(node,dict):
            continue
        c=node.get("c")
        r=node.get("r")
        y=node.get("y")
        if isinstance(c,(list,tuple)) and isinstance(r,(list,tuple)):
            summaries.append({
                "y_type":type(y).__name__,
                "y_len":len(y) if isinstance(y,(list,tuple)) else None,
                "c_len":len(c),
                "r_len":len(r),
                "c_first_type":type(c[0]).__name__ if c else None,
                "r_first_type":type(r[0]).__name__ if r else None,
            })
    return summaries


def extract_ubuntu_relational(dataset,vocab):
    """
    Preserve BOTH representations:
      1. raw encoded token IDs per pair
      2. decoded text when vocabulary coverage is sufficient

    This is important because Ubuntu's original representation carries
    relational structure in y/c/r rather than merely a text corpus.
    """
    rows=[]
    containers=_find_ubuntu_containers(dataset)

    for container_index,container in enumerate(containers):
        c=container["c"]
        r=container["r"]
        y=container.get("y")
        count=min(len(c),len(r))

        for i in range(count):
            cseq=c[i]
            rseq=r[i]

            # Only accept sequence-like rows; scalar rows are metadata/labels.
            if not isinstance(cseq,(list,tuple)):
                continue
            if not isinstance(rseq,(list,tuple)):
                continue

            user=decode_sequence(cseq,vocab)
            reply=decode_sequence(rseq,vocab)

            label=None
            if isinstance(y,(list,tuple)) and i<len(y):
                label=y[i]

            rows.append({
                "container_index":container_index,
                "row_index":i,
                "label":label,
                "context_ids":[int(x) for x in cseq if isinstance(x,(int,float))],
                "response_ids":[int(x) for x in rseq if isinstance(x,(int,float))],
                "user":user,
                "reply":reply,
            })

    return rows
def _token_to_text(token_id,vocab):
    try:
        idx=int(token_id)
    except Exception:
        return None

    token=vocab.get(idx)
    if token is None:
        return None

    if isinstance(token,bytes):
        return token.decode("utf-8","replace")
    return str(token)


def decode_sequence(seq,vocab):
    if isinstance(seq,(str,bytes)):
        return (
            seq.decode("utf-8","replace")
            if isinstance(seq,bytes)
            else seq
        )

    if not isinstance(seq,(list,tuple)):
        return None

    out=[]
    for token in seq:
        text=_token_to_text(token,vocab)
        if text is None:
            # Some legacy pickle layouts use nested singleton token arrays.
            if isinstance(token,(list,tuple)) and len(token)==1:
                text=_token_to_text(token[0],vocab)
        if text is None:
            return None
        out.append(text)

    text=clean_text(" ".join(out))
    return text or None


def _find_ubuntu_containers(dataset):
    """
    Locate containers containing compatible c/r arrays anywhere in the pickle.
    This handles the observed [dict(y,c,r), ...] layout without assuming that
    root[0] is always the only useful container.
    """
    found=[]

    for node in _walk(dataset):
        if not isinstance(node,dict):
            continue

        c=node.get("c")
        r=node.get("r")
        y=node.get("y")

        if isinstance(c,(list,tuple)) and isinstance(r,(list,tuple)):
            found.append({
                "c":c,
                "r":r,
                "y":y,
            })

    return found


def extract_ubuntu_pairs(dataset,vocab):
    rows=[]

    for container in _find_ubuntu_containers(dataset):
        c=container["c"]
        r=container["r"]
        n=min(len(c),len(r))

        for i in range(n):
            user=decode_sequence(c[i],vocab)
            reply=decode_sequence(r[i],vocab)

            # Fallback: c/r can occasionally be a single token sequence rather
            # than a list of sequences.
            if user is None and i==0:
                user=decode_sequence(c,vocab)
            if reply is None and i==0:
                reply=decode_sequence(r,vocab)

            if user and reply:
                rows.append({
                    "user":user,
                    "reply":reply,
                    "dialogue_id":f"ubuntu-{len(rows)}",
                    "turn_index":0,
                })

    # Deduplicate pairs while preserving order.
    seen=set()
    out=[]
    for row in rows:
        key=(row["user"],row["reply"])
        if key in seen:
            continue
        seen.add(key)
        row["dialogue_id"]=f"ubuntu-{len(out)}"
        out.append(row)

    return out


def parse_sgd_file(path):
    path=Path(path)

    try:
        data=load_json(path)
    except ValueError as exc:
        valid=[]
        bad=0
        for line_no,item in load_jsonl(path):
            if isinstance(item,Exception):
                bad+=1
                if bad<=3:
                    print(
                        f"      [SGD] {path.name}:{line_no} "
                        f"invalid JSONL: {item}",
                        flush=True,
                    )
            else:
                valid.append(item)

        if not valid:
            print(
                f"      [SGD] SKIP {path.name}: {exc}",
                flush=True,
            )
            return

        print(
            f"      [SGD] {path.name} treated as JSONL "
            f"valid={len(valid)} bad={bad}",
            flush=True,
        )
        data=valid

    if data is None:
        return

    records=data.get("dialogues",data.get("data",data)) if isinstance(data,dict) else data
    if not isinstance(records,list):
        records=[records]

    yielded=0

    for rec_index,record in enumerate(records):
        if not isinstance(record,dict):
            continue

        did=(
            record.get("dialogue_id")
            or record.get("dialogueId")
            or f"{path.stem}-{rec_index}"
        )

        turns=record.get("turns")
        if not isinstance(turns,list):
            continue

        for i,turn in enumerate(turns):
            if not isinstance(turn,dict):
                continue

            text=turn.get("utterance") or turn.get("text") or turn.get("value")
            if not isinstance(text,str) or not text.strip():
                continue

            intent=None
            frames=turn.get("frames")
            if isinstance(frames,list):
                for frame in frames:
                    if not isinstance(frame,dict):
                        continue
                    acts=frame.get("actions")
                    if isinstance(acts,list):
                        for action in acts:
                            if isinstance(action,dict):
                                intent=action.get("act") or action.get("intent")
                                if intent:
                                    break
                    if intent:
                        break

            yield {
                "text":clean_text(text),
                "speaker":turn.get("speaker"),
                "dialogue_id":str(did),
                "turn_index":i,
                "intent":intent,
                "domain":turn.get("service"),
            }
            yielded+=1

    if yielded==0:
        print(
            f"      [SGD] {path.name}: valid JSON but no dialogue turns",
            flush=True,
        )


def parse_multiwoz_file(path):
    data=load_json(path)
    if not isinstance(data,(dict,list)):
        return

    items=data.items() if isinstance(data,dict) else enumerate(data)

    for did,dialogue in items:
        if not isinstance(dialogue,dict):
            continue

        turns=dialogue.get("log") or dialogue.get("turns")
        if not isinstance(turns,list):
            continue

        for i,turn in enumerate(turns):
            if not isinstance(turn,dict):
                continue
            text=turn.get("text") or turn.get("utterance") or turn.get("content")
            if isinstance(text,str) and text.strip():
                yield {
                    "text":clean_text(text),
                    "speaker":turn.get("speaker") or turn.get("role"),
                    "dialogue_id":str(did),
                    "turn_index":i,
                    "intent":None,
                    "domain":None,
                }


def classify_path(path):
    path=Path(path)
    name=path.name.lower()
    parts=[x.lower() for x in path.parts]

    if name in SKIP_NAMES:
        return None
    if "schema" in name or "ontology" in name:
        return None

    # Deliberately exclude the unusable source.
    if any(x in part for part in parts for x in ("dialoglue","dialo_glue")):
        return None

    suffixes=name

    # SGD / DSTC8: the actual directory name supplied by the user is
    # "dstc8-schema-guided-dialogue".
    sgd_dirs={
        "dstc8-schema-guided-dialogue",
        "schema-guided-dialogue",
        "schema_guided_dialogue",
        "schema-guided",
        "sgd",
    }
    if any(part in sgd_dirs for part in parts):
        if suffixes.endswith(
            (".json",".jsonl",".json.gz",".jsonl.gz")
        ):
            return "sgd"
        return None

    multiwoz_dirs={"multiwoz","multiwoz2.1","multiwoz21"}
    if any(part in multiwoz_dirs for part in parts):
        if suffixes.endswith(
            (".json",".jsonl",".json.gz",".jsonl.gz")
        ):
            return "multiwoz"
        return None

    # Ubuntu is explicitly scoped to a directory called ubuntu.
    if "ubuntu" in parts:
        if suffixes=="dataset.pkl":
            return "ubuntu_dataset"
        if suffixes=="vocab.pkl":
            return "ubuntu_vocab"

    return None


def discover(root):
    root=Path(root)
    found={
        "sgd":[],
        "multiwoz":[],
        "ubuntu_dataset":[],
        "ubuntu_vocab":[],
    }

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        kind=classify_path(path)
        if kind:
            found[kind].append(path)

    for key in found:
        found[key]=sorted(set(found[key]))

    return found
