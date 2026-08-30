
from __future__ import annotations

import uuid

import json
import sqlite3
import time
from pathlib import Path

from schema import SCHEMA

# These are parser/grammar artifacts, not useful standalone semantic query
# concepts. In particular, WordNet has many entries for "be", "have", etc.
PREDICATE_RETRIEVAL_BLOCKLIST = {
    "be","have","do","get","go","make","take","come","say",
    "tell","ask","want","need","can","could","should","would",
    "will","may","might","must","is","are","was","were",
    "am","been","being",
}


class TypedMemory:
    def __init__(self,path_or_con,session_id="default"):
        if isinstance(path_or_con,sqlite3.Connection):
            self.con=path_or_con
        else:
            self.con=sqlite3.connect(str(Path(path_or_con)))
        self.con.row_factory=sqlite3.Row
        self.con.executescript(SCHEMA)
        self.session_id=session_id

        self.topic=None
        self.goal=None
        self.previous_goal=None
        self.turn_index=0
        self.last_user=""
        self.last_assistant=""
        self.freeze_knowledge=False

    def set_knowledge_frozen(self,frozen=True):
        self.freeze_knowledge=bool(frozen)

    def knowledge_is_frozen(self):
        return bool(self.freeze_knowledge)

    def reset_session(self):
        old_session=self.session_id

        for table in ("live_turns","live_facts","live_entities"):
            try:
                self.con.execute(
                    f"DELETE FROM {table} WHERE session_id=?",
                    (old_session,),
                )
            except sqlite3.OperationalError:
                pass

        self.session_id=f"session-{uuid.uuid4().hex}"
        self.topic=None
        self.goal=None
        self.previous_goal=None
        self.turn_index=0
        self.last_user=""
        self.last_assistant=""
        self.last_answer_source=None
        self.last_answer_content=None
        self.con.commit()


    def set_context(self,text,goal,topic=None):
        self.last_user=text
        self.previous_goal=self.goal
        self.goal=goal
        if topic:
            self.topic=topic

    def add_live_turn(self,speaker,text):
        self.con.execute(
            """
            INSERT OR REPLACE INTO live_turns(
                session_id,turn_index,speaker,text,timestamp
            ) VALUES(?,?,?,?,?)
            """,
            (
                self.session_id,
                self.turn_index,
                speaker,
                text,
                time.time(),
            ),
        )


    def entity_instances(self,canonical):
        return [
            dict(r)
            for r in self.con.execute(
                """
                SELECT entity_id,canonical,ordinal,mention_turn
                FROM live_entities
                WHERE session_id=? AND canonical=? AND active=1
                ORDER BY ordinal
                """,
                (self.session_id,canonical.lower()),
            )
        ]

    def mention_entity(self,canonical,new_entity=False):
        canonical=canonical.lower().strip()
        if not canonical:
            return None

        current=self.entity_instances(canonical)

        if current and not new_entity:
            return current[-1]["entity_id"]

        ordinal=(
            self.con.execute(
                "SELECT COALESCE(MAX(ordinal),0)+1 FROM live_entities WHERE session_id=?",
                (self.session_id,),
            ).fetchone()[0]
        )

        cur=self.con.execute(
            """
            INSERT INTO live_entities(
                session_id,canonical,ordinal,mention_turn,active
            ) VALUES(?,?,?,?,1)
            """,
            (
                self.session_id,
                canonical,
                ordinal,
                self.turn_index,
            ),
        )
        self.con.commit()
        return cur.lastrowid

    def entity_count(self,canonical):
        return self.con.execute(
            """
            SELECT COUNT(*)
            FROM live_entities
            WHERE session_id=? AND canonical=? AND active=1
            """,
            (self.session_id,canonical.lower()),
        ).fetchone()[0]

    def last_entity(self,canonical):
        rows=self.entity_instances(canonical)
        return rows[-1] if rows else None

    def add_live_fact(self,p,source_turn=None):
        self.con.execute(
            """
            INSERT INTO live_facts(
                session_id,subject,predicate,object_text,
                fact_type,negated,confidence,turn_index,active
            ) VALUES(?,?,?,?,?,?,?,?,1)
            """,
            (
                self.session_id,
                p["subject"].lower(),
                p["predicate"].lower(),
                str(p.get("object","")).lower(),
                p.get("fact_type","state"),
                int(bool(p.get("negated",False))),
                float(p.get("certainty",1.0)),
                self.turn_index if source_turn is None else source_turn,
            ),
        )
        self.con.commit()

    def facts(self,subject=None,predicate=None):
        sql="""
            SELECT subject,predicate,object_text,fact_type,
                   negated,confidence,turn_index
            FROM live_facts
            WHERE session_id=? AND active=1
        """
        params=[self.session_id]
        if subject:
            sql+=" AND lower(subject)=lower(?)"
            params.append(subject)
        if predicate:
            sql+=" AND lower(predicate)=lower(?)"
            params.append(predicate)
        sql+=" ORDER BY turn_index DESC,confidence DESC"
        return [dict(r) for r in self.con.execute(sql,params)]

    def add_assistant_turn(self,text):
        self.last_assistant=text
        self.add_live_turn("assistant",text)
        self.con.commit()

    def remember_answer(self,source,content):
        self.last_answer_source=source
        self.last_answer_content=content


    def recent_turns(self,limit=8):
        rows=self.con.execute(
            """
            SELECT speaker,text,turn_index
            FROM live_turns
            WHERE session_id=?
            ORDER BY turn_index DESC
            LIMIT ?
            """,
            (self.session_id,limit),
        ).fetchall()
        return list(reversed([dict(r) for r in rows]))

    def context(self,limit=12):
        lines=[]
        if self.topic:
            lines.append(f"TOPIC: {self.topic}")
        if self.goal:
            lines.append(f"GOAL: {self.goal}")
        for fact in self.facts()[:limit]:
            sign="not " if fact["negated"] else ""
            lines.append(
                f"FACT: {fact['subject']} "
                f"{fact['predicate']} {sign}{fact['object_text']}"
            )
        for turn in self.recent_turns(8):
            lines.append(
                f"{turn['speaker'].upper()}: {turn['text']}"
            )
        return "\n".join(lines)

    # ---- static typed knowledge ----

    def static_facts(self,subject_terms,goal_name,domains=None,limit=24):
        if not subject_terms:
            return []

        prefs={
            "request_information":{
                "semantic":1.0,"lexical":0.9,
                "dialogue":0.3,"procedural":0.4,
            },
            "request_explanation":{
                "semantic":1.2,"lexical":0.8,
                "dialogue":0.3,"procedural":0.5,
            },
            "request_generation":{
                "procedural":1.1,"dialogue":0.8,
                "semantic":0.3,"lexical":0.5,
            },
            "challenge_claim":{
                "semantic":1.1,"state":1.4,
                "dialogue":0.5,"lexical":0.5,
            },
            "explore_assistant":{
                "semantic":0.3,"lexical":0.3,
                "dialogue":0.7,
            },
        }.get(
            goal_name,
            {
                "semantic":1.0,"lexical":0.8,
                "dialogue":0.5,"procedural":0.5,
            },
        )

        clean_terms=[]
        for term in subject_terms:
            t=str(term or "").strip().lower()
            if not t or t in PREDICATE_RETRIEVAL_BLOCKLIST:
                continue
            if len(t)<=2 and t not in {"ai","us"}:
                continue
            clean_terms.append(t)

        if not clean_terms:
            return []

        rows=[]

        for term in list(dict.fromkeys(clean_terms))[:12]:
            sql="""
                SELECT
                    c.canonical AS subject,
                    f.predicate,
                    COALESCE(o.canonical,f.object_text) AS object_text,
                    f.fact_type,
                    f.domain,
                    MAX(f.confidence) AS confidence,
                    SUM(f.frequency) AS frequency,
                    COUNT(DISTINCT f.source_id) AS source_count,
                    GROUP_CONCAT(DISTINCT s.dataset) AS datasets
                FROM facts f
                JOIN concepts c ON c.concept_id=f.subject_id
                LEFT JOIN concepts o ON o.concept_id=f.object_id
                LEFT JOIN sources s ON s.source_id=f.source_id
                WHERE lower(c.canonical)=lower(?)
                  AND f.answerable=1
                  AND f.predicate NOT IN (
                      'in_domain','domain','source','provenance',
                      'dataset','node_type','type','label',
                      'nsubj','nsubjpass','obj','dobj','iobj',
                      'ccomp','xcomp','amod','advmod','nmod',
                      'obl','oblique','root','dep','aux','auxpass',
                      'cop','det','case','mark','punct','conj','cc',
                      'compound','appos','acl','advcl'
                  )
            """
            params=[term]
            if domains:
                sql+=" AND (f.domain IS NULL OR f.domain IN ({0}))".format(
                    ",".join("?" for _ in domains)
                )
                params.extend(domains)

            sql+="""
                GROUP BY
                    c.canonical,
                    f.predicate,
                    COALESCE(o.canonical,f.object_text),
                    f.fact_type,
                    f.domain
                ORDER BY MAX(f.confidence) DESC,SUM(f.frequency) DESC
                LIMIT ?
            """
            params.append(limit)

            rows.extend(
                dict(r)
                for r in self.con.execute(sql,params)
            )

        for row in rows:
            pref=prefs.get(row["fact_type"],0.1)
            source_bonus=min(
                3.0,
                0.75*max(0,int(row["source_count"])-1)
            )
            row["relevance"]=(
                3.0
                *pref
                *max(0.1,float(row["confidence"]))
                *(1.0+min(3.0,float(row["frequency"]))/3.0)
                +source_bonus
            )

            # Generic parser concepts are intentionally weak evidence unless
            # the user actually named the concept explicitly.
            if row["subject"] not in clean_terms:
                row["relevance"]*=0.35

        # De-duplicate equivalent grouped claims.
        dedup={}
        for row in rows:
            key=(
                row["subject"],
                row["predicate"],
                row["object_text"],
                row["fact_type"],
                row["domain"],
            )
            old=dedup.get(key)
            if old is None or row["relevance"]>old["relevance"]:
                dedup[key]=row

        rows=list(dedup.values())
        rows.sort(key=lambda x:x["relevance"],reverse=True)
        return rows[:limit]
