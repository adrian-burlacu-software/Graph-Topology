"""Long-term memory: what the conversations taught, kept after they end.

Until now everything a conversation was told or taught lived in the server's
memory and went when the server did. A conversation is how this layer learns
anything at all, so every restart unlearned it. This keeps two things on
disk, in one sqlite file, and keeps them differently because they are
different kinds of memory:

    said                           belongs to          survives
    a wemble is a kind of animal   every conversation  a restart, start over
    beagles can't swim             every conversation  a restart, start over
    there is a beagle              this conversation   a restart
    its name is Rex                this conversation   a restart

**Knowledge** is what was taught about kinds: taught kinds, taxonomy, norms.
It is about no one in particular, so it is no one conversation's. A norm
taught this morning answers this afternoon in a new conversation, and the
answer says it was taught earlier. It is still never written into the store:
v687's store is what the corpora say, and this is what someone said.

**A conversation** is everything else: its individuals, what was told of
them, who is salient, what each phrase meant. It is saved after every turn,
so a page that comes back after the server restarted finds its conversation
where it left it, transcript and all. `start over` forgets the conversation
and keeps the knowledge; `unlearn` forgets the knowledge.

**Examples teach nothing that is kept.** The page's examples teach on purpose
-- `beagles can't swim` is there to show R3 blocking what dogs do -- and a
demonstration that wrote that into what every later conversation starts from
would be a false norm with a permanent address. An example conversation has
knowledge of its own, saved with it and read by nothing else.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path

from .episodic import Knowledge
from .session import Session

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

#: Out of the source tree and out of git: it is what someone said to a
#: running server, not something the repository builds.
DEFAULT_PATH = REPOSITORY_ROOT / "state" / "v689-memory.sqlite"

#: Conversations held in memory at once. Past this the least recently used
#: leaves memory only; it is still on disk, and comes back when it is asked.
KEPT = 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS kinds (word TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS edges (
    node TEXT NOT NULL, parent TEXT NOT NULL, said TEXT NOT NULL,
    conversation TEXT NOT NULL, taught REAL NOT NULL,
    PRIMARY KEY (node, parent));
CREATE TABLE IF NOT EXISTS norms (
    node TEXT NOT NULL, relation TEXT NOT NULL, object TEXT NOT NULL,
    said TEXT NOT NULL, mode TEXT NOT NULL, against TEXT,
    conversation TEXT NOT NULL, taught REAL NOT NULL,
    PRIMARY KEY (node, relation, object));
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY, example INTEGER NOT NULL, saved REAL NOT NULL,
    state TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS turns (
    conversation TEXT NOT NULL, number INTEGER NOT NULL, turn TEXT NOT NULL,
    PRIMARY KEY (conversation, number));
"""


class Archive:
    """One sqlite file: the knowledge, and every conversation.

    Its own connection and its own lock. It shares nothing with the store,
    which is opened read-only by every engine and must stay that way.
    """

    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.connection = sqlite3.connect(str(self.path),
                                          check_same_thread=False)
        self.connection.executescript(SCHEMA)

    def close(self) -> None:
        with self.lock:
            self.connection.close()

    # -- knowledge ---------------------------------------------------------
    def knowledge(self) -> Knowledge:
        with self.lock:
            read = self.connection.execute
            return Knowledge.from_state({
                "kinds": [row[0] for row in read(
                    "SELECT word FROM kinds ORDER BY rowid")],
                "edges": [list(row) for row in read(
                    "SELECT node, parent, said, conversation, taught "
                    "FROM edges ORDER BY rowid")],
                "norms": [list(row) for row in read(
                    "SELECT node, relation, object, said, mode, against, "
                    "conversation, taught FROM norms ORDER BY rowid")]})

    def keep(self, knowledge: Knowledge) -> None:
        """Write the knowledge as it stands, whole, in one transaction.

        It is small, and rewriting it is simpler than tracking what changed
        -- a correction replaces a norm, `unlearn` empties it -- and a
        half-written norm is worse than one that was never saved.
        """
        state = knowledge.as_state()
        with self.lock, self.connection:
            write = self.connection.execute
            for table in ("kinds", "edges", "norms"):
                write(f"DELETE FROM {table}")
            self.connection.executemany("INSERT INTO kinds VALUES (?)",
                                        [(word,) for word in state["kinds"]])
            self.connection.executemany(
                "INSERT OR REPLACE INTO edges VALUES (?, ?, ?, ?, ?)",
                state["edges"])
            self.connection.executemany(
                "INSERT OR REPLACE INTO norms VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                state["norms"])

    # -- conversations -----------------------------------------------------
    def save(self, session: Session, turn: dict | None = None) -> None:
        """The conversation as it is after a turn, and that turn as the page
        showed it."""
        state = json.dumps(session.snapshot())
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO conversations VALUES (?, ?, ?, ?)",
                (session.conversation, int(session.example), time.time(),
                 state))
            if turn is not None:
                self.connection.execute(
                    "INSERT OR REPLACE INTO turns VALUES (?, ?, ?)",
                    (session.conversation, int(turn.get("number") or 0),
                     json.dumps(turn)))

    def state(self, conversation: str) -> dict | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT state FROM conversations WHERE id = ?",
                (conversation,)).fetchone()
        return json.loads(row[0]) if row else None

    def turns(self, conversation: str) -> list[dict]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT turn FROM turns WHERE conversation = ? "
                "ORDER BY number", (conversation,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def forget(self, conversation: str) -> None:
        with self.lock, self.connection:
            self.connection.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation,))
            self.connection.execute(
                "DELETE FROM turns WHERE conversation = ?", (conversation,))


class Keeper:
    """Every conversation on the page, and the knowledge they share.

    Nothing here is thread-safe on purpose: a turn reasons over the store's
    one sqlite connection, so the server calls this with its engines lock
    held, and one caller at a time is the contract.
    """

    def __init__(self, asker, archive: Archive | None = None,
                 kept: int = KEPT) -> None:
        self.asker = asker
        self.archive = archive
        self.kept = kept
        self.knowledge = (archive.knowledge() if archive is not None
                          else Knowledge())
        self.held: OrderedDict[str, Session] = OrderedDict()

    def _known(self, conversation: str) -> bool:
        return conversation in self.held or (
            self.archive is not None
            and self.archive.state(conversation) is not None)

    def session(self, conversation: str, example: bool = False) -> Session:
        """The conversation held, or the one on disk, or a new one.

        `example` only decides what a *new* conversation is. One that already
        exists stays what it was made as.
        """
        found = self.held.pop(conversation, None)
        if found is None and self.archive is not None:
            state = self.archive.state(conversation)
            if state is not None:
                found = Session.resume(self.asker, state, self.knowledge)
        if found is None:
            found = Session(self.asker, None if example else self.knowledge,
                            conversation, example)
        self.held[conversation] = found
        while len(self.held) > self.kept:
            self.held.popitem(last=False)
        return found

    def say(self, conversation: str, text: str, example: bool = False,
            trim=None) -> dict:
        session = self.session(conversation, example)
        turn = session.say(text).as_dict()
        if trim is not None:
            turn = trim(turn)
        if self.archive is not None:
            self.archive.save(session, turn)
            if not session.example:
                self.archive.keep(self.knowledge)
        return turn

    def history(self, conversation: str) -> dict:
        """What the page needs to show a conversation it did not just have:
        every turn, and memory as it is now rather than as the last turn
        left it -- knowledge taught elsewhere, or unlearned, since."""
        session = (self.session(conversation) if self._known(conversation)
                   else None)
        if self.archive is not None:
            turns = self.archive.turns(conversation)
        else:
            turns = ([turn.as_dict() for turn in session.turns]
                     if session else [])
        return {"turns": turns,
                "example": bool(session and session.example),
                "memory": session.memory.as_dict() if session else None,
                "discourse": session.discourse.as_dict() if session else None,
                "knowledge": self.summary()}

    def forget(self, conversation: str) -> None:
        self.held.pop(conversation, None)
        if self.archive is not None:
            self.archive.forget(conversation)

    def unlearn(self) -> None:
        """Forget every kind, edge and norm that was taught. Conversations
        keep their individuals; an individual of a kind that is gone keeps
        the word it was introduced by, and nothing above it."""
        self.knowledge.clear()
        if self.archive is not None:
            self.archive.keep(self.knowledge)
        for session in self.held.values():
            if not session.example:
                session.memory.store("unlearned")

    def summary(self) -> dict:
        state = self.knowledge.as_state()
        return {"kinds": len(state["kinds"]), "edges": len(state["edges"]),
                "norms": len(state["norms"])}
