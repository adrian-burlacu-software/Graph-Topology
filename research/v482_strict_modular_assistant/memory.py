
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Any


def digest(*parts: Any) -> str:
    raw=json.dumps(
        parts,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class ConversationState:
    topic: str | None = None
    goal: str | None = None
    last_user_act: str | None = None
    last_assistant_act: str | None = None
    last_user_text: str | None = None
    last_assistant_text: str | None = None
    unresolved_question: str | None = None


class ConversationMemory:
    """
    Dynamic conversation memory.

    This is deliberately separate from the large static semantic graph.
    `freeze_learning` does not disable working memory. It only prevents
    learned policies/facts from becoming permanent training signals.
    """

    def __init__(
        self,
        con: sqlite3.Connection,
        session_id: str,
        recent_turns: int = 8,
        memory_facts: int = 24,
        freeze_learning: bool = True,
    ):
        self.con=con
        self.session_id=session_id
        self.recent_turns=recent_turns
        self.memory_facts=memory_facts
        self.freeze_learning=freeze_learning
        self.ensure_tables()

    def ensure_tables(self) -> None:
        self.con.executescript("""
        CREATE TABLE IF NOT EXISTS assistant_sessions(
            session_id TEXT PRIMARY KEY,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS assistant_turns(
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

        CREATE TABLE IF NOT EXISTS assistant_llm_dialogue(
            message_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            speaker TEXT NOT NULL,
            purpose TEXT NOT NULL,
            text TEXT NOT NULL,
            parsed_json TEXT NOT NULL,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS conversation_state(
            session_id TEXT PRIMARY KEY,
            topic TEXT,
            goal TEXT,
            last_user_act TEXT,
            last_assistant_act TEXT,
            last_user_text TEXT,
            last_assistant_text TEXT,
            unresolved_question TEXT,
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS conversation_memory(
            memory_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            subject TEXT,
            predicate TEXT,
            object TEXT,
            value TEXT,
            salience REAL NOT NULL DEFAULT 0,
            source_turn INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            last_accessed REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_conversation_memory_session
        ON conversation_memory(session_id,salience,last_accessed);

        CREATE TABLE IF NOT EXISTS dialogue_policies(
            policy_key TEXT PRIMARY KEY,
            speech_act TEXT NOT NULL,
            signature TEXT NOT NULL,
            response TEXT NOT NULL,
            evidence INTEGER NOT NULL DEFAULT 0,
            llm_uses INTEGER NOT NULL DEFAULT 0,
            last_used REAL NOT NULL,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS assistant_interaction_facts(
            fact_id TEXT PRIMARY KEY,
            turn_id TEXT NOT NULL,
            predicate TEXT,
            subject TEXT,
            object TEXT,
            relation TEXT,
            payload_json TEXT NOT NULL,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS assistant_metrics(
            metric_key TEXT PRIMARY KEY,
            metric_value REAL NOT NULL DEFAULT 0
        );
        """)

        self.con.execute(
            "INSERT OR IGNORE INTO assistant_sessions VALUES (?,?)",
            (self.session_id,time.time()),
        )

        self.con.execute(
            """
            INSERT OR IGNORE INTO conversation_state(
                session_id,topic,goal,last_user_act,last_assistant_act,
                last_user_text,last_assistant_text,unresolved_question,
                updated_at
            )
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                self.session_id,
                None,None,None,None,None,None,None,
                time.time(),
            ),
        )

        for key in (
            "turns","llm_participant_calls","llm_realizer_calls",
            "participant_messages","realizer_messages",
            "memory_turns","memory_facts",
            "candidate_count","candidate_selected",
        ):
            self.con.execute(
                "INSERT OR IGNORE INTO assistant_metrics VALUES (?,0)",
                (key,),
            )
        self.con.commit()

    def inc(self,key: str,value: float = 1) -> None:
        self.con.execute(
            """
            INSERT INTO assistant_metrics(metric_key,metric_value)
            VALUES(?,?)
            ON CONFLICT(metric_key)
            DO UPDATE SET metric_value=metric_value+excluded.metric_value
            """,
            (key,float(value)),
        )

    def state(self) -> ConversationState:
        row=self.con.execute(
            """
            SELECT topic,goal,last_user_act,last_assistant_act,
                   last_user_text,last_assistant_text,
                   unresolved_question
            FROM conversation_state
            WHERE session_id=?
            """,
            (self.session_id,),
        ).fetchone()

        if not row:
            return ConversationState()

        return ConversationState(*row)

    def update_user(
        self,
        text: str,
        act: str,
        topic: str | None,
        goal: str | None = None,
        turn_index: int = 0,
    ) -> None:
        previous=self.state()
        topic=topic or previous.topic

        unresolved=text if act=="question" else previous.unresolved_question

        self.con.execute(
            """
            UPDATE conversation_state
            SET topic=?,
                goal=COALESCE(?,goal),
                last_user_act=?,
                last_user_text=?,
                unresolved_question=?,
                updated_at=?
            WHERE session_id=?
            """,
            (
                topic,
                goal,
                act,
                text,
                unresolved,
                time.time(),
                self.session_id,
            ),
        )

        if not self.freeze_learning and topic:
            self.remember(
                "topic",
                "conversation",
                "about",
                topic,
                salience=6.0,
                turn_index=turn_index,
            )

        self.inc("memory_turns")
        self.con.commit()

    def update_assistant(
        self,
        text: str,
        act: str | None,
    ) -> None:
        self.con.execute(
            """
            UPDATE conversation_state
            SET last_assistant_act=?,
                last_assistant_text=?,
                updated_at=?
            WHERE session_id=?
            """,
            (
                act,
                text,
                time.time(),
                self.session_id,
            ),
        )
        self.con.commit()

    def remember(
        self,
        kind: str,
        subject: str,
        predicate: str,
        value: str,
        salience: float = 1.0,
        turn_index: int = 0,
    ) -> None:
        now=time.time()
        mid=digest(
            "conversation_memory",
            self.session_id,
            kind,
            subject,
            predicate,
            value,
        )
        self.con.execute(
            """
            INSERT INTO conversation_memory(
                memory_id,session_id,kind,subject,predicate,
                object,value,salience,source_turn,
                created_at,last_accessed
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(memory_id) DO UPDATE SET
                salience=max(salience,excluded.salience),
                source_turn=excluded.source_turn,
                last_accessed=excluded.last_accessed
            """,
            (
                mid,
                self.session_id,
                kind,
                subject,
                predicate,
                value,
                value,
                salience,
                turn_index,
                now,
                now,
            ),
        )
        self.inc("memory_facts")
        self.con.commit()

    def recent_dialogue(self) -> list[tuple[str,str]]:
        rows=self.con.execute(
            """
            SELECT speaker,text
            FROM assistant_turns
            WHERE session_id=?
            ORDER BY turn_index DESC,created_at DESC
            LIMIT ?
            """,
            (
                self.session_id,
                self.recent_turns*2,
            ),
        ).fetchall()
        return list(reversed(rows))

    def facts(self) -> list[tuple]:
        return self.con.execute(
            """
            SELECT kind,subject,predicate,value,salience
            FROM conversation_memory
            WHERE session_id=?
            ORDER BY salience DESC,last_accessed DESC
            LIMIT ?
            """,
            (
                self.session_id,
                self.memory_facts,
            ),
        ).fetchall()

    def resolve_reference(self,text: str) -> str | None:
        if not any(
            x in text.lower().split()
            for x in ("it","this","that","they","them","there")
        ):
            return None

        state=self.state()
        if state.topic:
            return state.topic

        for kind,subject,predicate,value,salience in self.facts():
            if kind=="topic" and value:
                return value

        return None

    def store_turn(
        self,
        turn_id: str,
        turn_index: int,
        speaker: str,
        text: str,
        parsed: dict,
        decision: str,
        confidence: float,
    ) -> None:
        self.con.execute(
            """
            INSERT OR REPLACE INTO assistant_turns
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                turn_id,
                self.session_id,
                turn_index,
                speaker,
                text,
                json.dumps(parsed,ensure_ascii=False),
                decision,
                confidence,
                time.time(),
            ),
        )
        self.con.commit()

    def store_llm_message(
        self,
        turn_index: int,
        purpose: str,
        text: str,
        payload: dict,
    ) -> None:
        self.inc("participant_messages" if purpose=="participant"
                 else "realizer_messages")
        mid=digest(
            "assistant_llm_dialogue",
            self.session_id,
            turn_index,
            purpose,
            text,
        )
        self.con.execute(
            """
            INSERT OR REPLACE INTO assistant_llm_dialogue
            VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                mid,
                self.session_id,
                turn_index,
                "llm",
                purpose,
                text,
                json.dumps(payload,ensure_ascii=False),
                time.time(),
            ),
        )
        self.con.commit()
