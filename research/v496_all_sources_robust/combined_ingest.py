
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from schema import SCHEMA
from udgum_source import iter_udgum
from verbnet_source import iter_verbnet, ensure_verbnet
from semantic_sources import iter_conceptnet, iter_wordnet
from ingest_adapters import (
    discover,
    load_json,
    load_pickle_robust,
    decode_vocab,
    extract_ubuntu_pairs,
    parse_sgd_file,
    parse_multiwoz_file,
)


def canonical(text):
    return " ".join(str(text or "").lower().split())


def get_concept(con,canonical_text,display=None,concept_type="concept"):
    row=con.execute(
        """
        SELECT concept_id
        FROM concepts
        WHERE canonical=? AND concept_type=?
        """,
        (canonical_text,concept_type),
    ).fetchone()
    if row:
        return row[0]

    cur=con.execute(
        """
        INSERT INTO concepts(canonical,display,concept_type)
        VALUES(?,?,?)
        """,
        (
            canonical_text,
            display or canonical_text,
            concept_type,
        ),
    )
    return cur.lastrowid


def source_id(con,dataset,path,record_key,content_type,metadata=None):
    cur=con.execute(
        """
        INSERT OR IGNORE INTO sources(
            dataset,source_path,record_key,content_type,metadata_json
        ) VALUES(?,?,?,?,?)
        """,
        (
            dataset,
            str(path),
            str(record_key) if record_key is not None else "",
            content_type,
            json.dumps(metadata or {},ensure_ascii=False),
        ),
    )
    row=con.execute(
        """
        SELECT source_id FROM sources
        WHERE dataset=? AND source_path=? AND record_key=? AND content_type=?
        """,
        (
            dataset,str(path),
            str(record_key) if record_key is not None else "",
            content_type,
        ),
    ).fetchone()
    return row[0]


def add_utterance(con,dataset,record,path,src_id):
    con.execute(
        """
        INSERT INTO utterances(
            dataset,text,speaker,dialogue_id,turn_index,
            intent,domain,source_id,metadata_json
        ) VALUES(?,?,?,?,?,?,?,?,?)
        """,
        (
            dataset,
            record["text"],
            record.get("speaker"),
            record.get("dialogue_id"),
            record.get("turn_index"),
            record.get("intent"),
            record.get("domain"),
            src_id,
            json.dumps(
                {
                    "source_path":str(path),
                    "dialogue_id":record.get("dialogue_id"),
                },
                ensure_ascii=False,
            ),
        ),
    )


def add_dialogue_pair(con,dataset,user,reply,path,pair_index):
    src=source_id(
        con,
        dataset,
        path,
        pair_index,
        "dialogue_pair",
        {"pair_index":pair_index},
    )
    add_utterance(
        con,
        dataset,
        {
            "text":user,
            "speaker":"user",
            "dialogue_id:fallback":f"{dataset}-{pair_index}",
            "dialogue_id":f"{dataset}-{pair_index}",
            "turn_index":0,
            "intent":None,
            "domain":None,
        },
        path,
        src,
    )
    add_utterance(
        con,
        dataset,
        {
            "text":reply,
            "speaker":"assistant",
            "dialogue_id":f"{dataset}-{pair_index}",
            "turn_index":1,
            "intent":None,
            "domain":None,
        },
        path,
        src,
    )


def add_fact(
    con,
    dataset,
    path,
    subject,
    predicate,
    object_text,
    fact_type,
    domain=None,
    confidence=1.0,
    frequency=1.0,
    answerable=True,
    record_key="",
):
    subj_id=get_concept(con,canonical(subject),str(subject),"concept")
    obj_id=None
    if object_text is not None:
        obj_id=get_concept(
            con,
            canonical(object_text),
            str(object_text),
            "concept",
        )

    sid=source_id(
        con,
        dataset,
        path,
        record_key,
        "fact",
        {"fact_type":fact_type},
    )

    con.execute(
        """
        INSERT OR IGNORE INTO facts(
            subject_id,predicate,object_id,object_text,
            fact_type,domain,source_id,confidence,frequency,
            answerable,metadata_json
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            subj_id,
            predicate,
            obj_id,
            None if obj_id else object_text,
            fact_type,
            domain,
            sid,
            confidence,
            frequency,
            int(answerable),
            json.dumps(
                {"record_key":record_key},
                ensure_ascii=False,
            ),
        ),
    )


def add_semantic_fact(con,dataset,path,fact):
    subj=fact["subject"]
    pred=fact["predicate"]
    obj=fact["object"]
    ftype=fact.get("fact_type","semantic")

    subj_id=get_concept(con,canonical(subj),subj,"concept")
    if pred=="defined_as":
        obj_id=None
        object_text=obj
    else:
        obj_id=get_concept(con,canonical(obj),obj,"concept")
        object_text=None

    sid=source_id(
        con,dataset,path,
        fact.get("record_key",""),
        "semantic_fact",
        fact.get("metadata",{}),
    )

    con.execute(
        """
        INSERT OR IGNORE INTO fact_evidence(
            subject_id,predicate,object_id,object_text,
            fact_type,source_id,confidence,weight
        ) VALUES(?,?,?,?,?,?,?,?)
        """,
        (
            subj_id,pred,obj_id,object_text,ftype,sid,
            float(fact.get("confidence",1.0)),
            float(fact.get("frequency",1.0)),
        ),
    )

    con.execute(
        """
        INSERT OR IGNORE INTO facts(
            subject_id,predicate,object_id,object_text,
            fact_type,domain,source_id,confidence,frequency,
            answerable,metadata_json
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            subj_id,pred,obj_id,object_text,ftype,
            fact.get("domain"),
            sid,
            float(fact.get("confidence",1.0)),
            float(fact.get("frequency",1.0)),
            int(bool(fact.get("answerable",True))),
            json.dumps(fact.get("metadata",{}),ensure_ascii=False),
        ),
    )


def merge_semantic_evidence(con):
    rows=con.execute(
        """
        SELECT subject_id,predicate,object_id,object_text,fact_type,
               COUNT(DISTINCT source_id),MAX(confidence),SUM(weight)
        FROM fact_evidence
        GROUP BY subject_id,predicate,object_id,object_text,fact_type
        """
    ).fetchall()

    for (
        subject_id,predicate,object_id,object_text,
        fact_type,source_count,max_conf,total_weight
    ) in rows:
        con.execute(
            """
            UPDATE facts
            SET frequency=MIN(20.0,?),
                confidence=MAX(confidence,?)
            WHERE subject_id=?
              AND predicate=?
              AND fact_type=?
              AND (
                object_id=?
                OR (object_id IS NULL AND object_text=?)
              )
            """,
            (
                float(total_weight),
                min(1.0,float(max_conf)+0.05*max(0,source_count-1)),
                subject_id,predicate,fact_type,object_id,object_text,
            ),
        )
    con.commit()


def ingest_conceptnet(con,path):
    count=0
    print(
        f"      ConceptNet {path.name} "
        f"size={path.stat().st_size/1024/1024:.1f}MB",
        flush=True,
    )
    for fact in iter_conceptnet(path):
        add_semantic_fact(con,"conceptnet",path,fact)
        count+=1
        if count%100000==0:
            con.commit()
            print(
                f"        ConceptNet facts={count}",
                flush=True,
            )
    con.commit()
    print(f"        ConceptNet imported={count}",flush=True)
    return count


def ingest_wordnet(con):
    path=Path("WordNet")
    count=0
    print("      WordNet via NLTK",flush=True)
    for fact in iter_wordnet():
        add_semantic_fact(con,"wordnet",path,fact)
        count+=1
        if count%50000==0:
            con.commit()
            print(
                f"        WordNet facts={count}",
                flush=True,
            )
    con.commit()
    print(f"        WordNet imported={count}",flush=True)
    return count


def ingest_dialogues(con,dataset,paths,parser):
    imported=0
    files=0
    skipped=0

    for path in paths:
        files+=1
        src=source_id(
            con,
            dataset,
            path,
            "",
            "utterances",
            {"file_size":path.stat().st_size},
        )

        count=0
        try:
            records=parser(path)
            for record in records or ():
                if not record.get("text"):
                    continue
                add_utterance(
                    con,
                    dataset,
                    record,
                    path,
                    src,
                )
                count+=1
                imported+=1
        except Exception as exc:
            skipped+=1
            print(
                f"      [{dataset.upper()}] SKIP {path.name}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

        con.commit()
        print(
            f"      {dataset} {files}/{len(paths)} "
            f"{path.name} utterances={count} total={imported}",
            flush=True,
        )

    print(
        f"      {dataset} summary files={files} "
        f"skipped={skipped} utterances={imported}",
        flush=True,
    )
    return files,imported


def ingest_ubuntu(con,paths_dataset,paths_vocab):
    if not paths_dataset:
        return 0,0

    vocab={}
    if paths_vocab:
        path=paths_vocab[0]
        print(
            f"      Ubuntu vocabulary {path.name} "
            f"size={path.stat().st_size/1024/1024:.1f}MB",
            flush=True,
        )
        obj=load_pickle_robust(path)
        vocab=decode_vocab(obj)
        print(
            f"        vocabulary_tokens={len(vocab)}",
            flush=True,
        )

    imported=0
    for index,path in enumerate(paths_dataset,1):
        print(
            f"      Ubuntu dataset {index}/{len(paths_dataset)} "
            f"{path.name} size={path.stat().st_size/1024/1024:.1f}MB",
            flush=True,
        )
        dataset=load_pickle_robust(path)
        pairs=extract_ubuntu_pairs(dataset,vocab)
        print(
            f"        decoded_pairs={len(pairs)}",
            flush=True,
        )

        for pair_index,pair in enumerate(pairs):
            if pair["user"] and pair["reply"]:
                add_dialogue_pair(
                    con,
                    "ubuntu",
                    pair["user"],
                    pair["reply"],
                    path,
                    pair_index,
                )

                # Dialogue examples become dialogue knowledge, not semantic facts.
                add_fact(
                    con,
                    "ubuntu",
                    path,
                    pair["user"][:200],
                    "next_reply",
                    pair["reply"][:400],
                    "dialogue",
                    domain="ubuntu",
                    confidence=0.65,
                    frequency=1.0,
                    answerable=True,
                    record_key=f"pair-{pair_index}",
                )
                imported+=2

        con.commit()
    return len(paths_dataset),imported


def ingest_verbnet(con,download_missing=False):
    from pathlib import Path

    ensure_verbnet(download_missing)

    path=Path("VerbNet")
    class_seen=set()
    members=roles=frames=0

    for item in iter_verbnet(path):
        kind=item["kind"]

        if item["class_id"] not in class_seen:
            con.execute(
                """
                INSERT OR IGNORE INTO verbnet_classes(
                    class_id,parent_class,source_id,metadata_json
                ) VALUES(?,?,?,?)
                """,
                (
                    item["class_id"],
                    None,
                    source_id(
                        con,"verbnet",path,item["class_id"],
                        "verbnet_class",{}
                    ),
                    json.dumps({},ensure_ascii=False),
                ),
            )
            class_seen.add(item["class_id"])

        sid=source_id(
            con,
            "verbnet",
            path,
            f"{item['kind']}:{item['class_id']}:{item.get('frame_index','')}:"
            f"{item.get('verb','')}",
            "verbnet",
            {},
        )

        if kind=="member":
            con.execute(
                """
                INSERT INTO verbnet_members(
                    class_id,verb,wn_refs,fn_refs,source_id
                ) VALUES(?,?,?,?,?)
                """,
                (
                    item["class_id"],
                    item["verb"],
                    item["metadata"].get("wn"),
                    item["metadata"].get("fn") or item["metadata"].get("framenet"),
                    sid,
                ),
            )
            # Also expose the member-class relationship to typed semantic
            # retrieval, but as provenance-aware procedural/linguistic data.
            add_semantic_fact(
                con,
                "verbnet",
                path,
                {
                    "subject":item["verb"],
                    "predicate":"member_of_verb_class",
                    "object":item["class_id"],
                    "fact_type":"procedural",
                    "domain":"verbnet",
                    "confidence":1.0,
                    "frequency":1.0,
                    "answerable":True,
                    "record_key":sid,
                    "metadata":item["metadata"],
                },
            )
            members+=1

        elif kind=="role":
            con.execute(
                """
                INSERT INTO verbnet_roles(
                    class_id,role,restrictions_json,source_id
                ) VALUES(?,?,?,?)
                """,
                (
                    item["class_id"],
                    item["role"],
                    json.dumps(item["restrictions"],ensure_ascii=False),
                    sid,
                ),
            )
            roles+=1

        elif kind=="frame":
            con.execute(
                """
                INSERT INTO verbnet_frames(
                    class_id,frame_index,description,example,
                    syntax_json,semantics_json,source_id
                ) VALUES(?,?,?,?,?,?,?)
                """,
                (
                    item["class_id"],
                    item["frame_index"],
                    item.get("description"),
                    item.get("example"),
                    json.dumps(item.get("syntax",[]),ensure_ascii=False),
                    json.dumps(item.get("semantics",[]),ensure_ascii=False),
                    sid,
                ),
            )
            frames+=1

        if (members+roles+frames)%50000==0:
            con.commit()

    con.commit()
    print(
        f"      VerbNet classes={len(class_seen)} "
        f"members={members} roles={roles} frames={frames}",
        flush=True,
    )
    return members,roles,frames


def ingest_udgum(con,path):
    count=0
    tokens=0
    path=Path(path)

    print(
        f"      UD_GUM root={path}",
        flush=True,
    )

    for item in iter_udgum(path):
        cur=con.execute(
            """
            INSERT INTO udgum_sentences(
                source_path,sent_id,text,metadata_json
            ) VALUES(?,?,?,?)
            """,
            (
                str(item["path"]),
                item.get("sent_id"),
                item["text"],
                json.dumps(
                    {"dataset":"UD_GUM"},
                    ensure_ascii=False,
                ),
            ),
        )
        sentence_id=cur.lastrowid

        for token in item["tokens"]:
            con.execute(
                """
                INSERT INTO udgum_tokens(
                    sentence_id,position,form,lemma,upos,xpos,
                    feats,head,deprel,deps,misc
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    sentence_id,
                    token["id"],
                    token["form"],
                    token["lemma"],
                    token["upos"],
                    token["xpos"],
                    token["feats"],
                    token["head"],
                    token["deprel"],
                    token["deps"],
                    token["misc"],
                ),
            )
            tokens+=1

        count+=1

        if count%5000==0:
            con.commit()
            print(
                f"        UD_GUM sentences={count} tokens={tokens}",
                flush=True,
            )

    con.commit()
    print(
        f"      UD_GUM imported sentences={count} tokens={tokens}",
        flush=True,
    )
    return count,tokens


def print_discovery(found):
    print("[DISCOVERY] supported sources",flush=True)
    for key in (
        "sgd","multiwoz",
        "ubuntu_dataset","ubuntu_vocab"
    ):
        paths=found.get(key,[])
        print(
            f"  {key}: {len(paths)}",
            flush=True,
        )
        for path in paths[:12]:
            print(
                f"    - {path}",
                flush=True,
            )
        if len(paths)>12:
            print(
                f"    ... +{len(paths)-12} more",
                flush=True,
            )


def main():
    ap=argparse.ArgumentParser(
        description="Combined typed cognitive-memory ingestion"
    )
    ap.add_argument("--data-root",default=".")
    ap.add_argument(
        "--out",
        default=r".\results\combined_cognitive_memory.sqlite",
    )
    ap.add_argument("--reset",action="store_true")
    ap.add_argument("--ud-gum",default="")
    ap.add_argument("--conceptnet",default="")
    ap.add_argument("--wordnet",action="store_true")
    ap.add_argument("--verbnet",action="store_true")
    args=ap.parse_args()

    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)

    if args.reset and out.exists():
        out.unlink()

    con=sqlite3.connect(str(out))
    con.executescript(SCHEMA)

    print(f"[COMBINED INGEST] root={Path(args.data_root).resolve()}")
    found=discover(args.data_root)
    print_discovery(found)

    for key,paths in found.items():
        print(
            f"  {key}: {len(paths)} file(s)",
            flush=True,
        )

    totals={
        "sgd_files":0,
        "sgd_utterances":0,
        "multiwoz_files":0,
        "multiwoz_utterances":0,
        "ubuntu_files":0,
        "ubuntu_utterances":0,
    }

    if found["sgd"]:
        f,u=ingest_dialogues(
            con,"sgd",found["sgd"],parse_sgd_file
        )
        totals["sgd_files"]+=f
        totals["sgd_utterances"]+=u

    if found["multiwoz"]:
        f,u=ingest_dialogues(
            con,"multiwoz",found["multiwoz"],parse_multiwoz_file
        )
        totals["multiwoz_files"]+=f
        totals["multiwoz_utterances"]+=u

    if args.conceptnet:
        totals["conceptnet_facts"]=ingest_conceptnet(
            con,Path(args.conceptnet)
        )

    if args.wordnet:
        totals["wordnet_facts"]=ingest_wordnet(con)

    if args.verbnet:
        (
            totals["verbnet_members"],
            totals["verbnet_roles"],
            totals["verbnet_frames"],
        )=ingest_verbnet(con,download_missing=True)

    merge_semantic_evidence(con)

    f,u=ingest_ubuntu(
        con,
        found["ubuntu_dataset"],
        found["ubuntu_vocab"],
    )
    totals["ubuntu_files"]+=f
    totals["ubuntu_utterances"]+=u

    if args.ud_gum:
        (
            totals["udgum_sentences"],
            totals["udgum_tokens"],
        )=ingest_udgum(con,Path(args.ud_gum))

    # Summary diagnostics.
    rows=con.execute(
        "SELECT dataset,COUNT(*) FROM utterances GROUP BY dataset"
    ).fetchall()
    facts=con.execute(
        "SELECT fact_type,COUNT(*) FROM facts GROUP BY fact_type"
    ).fetchall()

    print("\n[COMBINED INGEST] SUMMARY")
    print(json.dumps({
        "files":totals,
        "utterances_by_dataset":dict(rows),
        "facts_by_type":dict(facts),
        "concepts":con.execute("SELECT COUNT(*) FROM concepts").fetchone()[0],
        "facts":con.execute("SELECT COUNT(*) FROM facts").fetchone()[0],
        "utterances":con.execute("SELECT COUNT(*) FROM utterances").fetchone()[0],
    },indent=2))

    con.commit()

    expected=("sgd","multiwoz","","ubuntu")
    actual={r[0] for r in rows}
    missing=[
        name for name in expected
        if name not in actual
    ]
    if missing:
        print(
            f"[COMBINED INGEST] WARNING missing usable utterances: "
            f"{', '.join(missing)}",
            flush=True,
        )

    con.close()


if __name__=="__main__":
    main()
