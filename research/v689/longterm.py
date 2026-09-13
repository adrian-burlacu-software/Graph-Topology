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

## Events are what is kept

Both memories are event streams (`events.py`), and the `events` table is the
record of them: append-only, one row per event, in the order it happened. A
conversation comes back by replaying its stream, and knowledge by replaying
`knowledge`. The other tables are read models kept beside the log -- the
knowledge as rows, a snapshot of each conversation, and every turn as the page
showed it -- so sqlite can be read without folding a stream by hand.

Rows kept before there were events are read once and recorded as the events
that would have written them: a conversation's snapshot as one `imported`
event at the head of its stream, and knowledge rows as `kind_coined`,
`related`, `told` and `judged`. `start over` deletes a conversation's stream,
because it asks for the conversation to be gone; `unlearn` appends
`unlearned`, and the stream keeps what was unlearned.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path

from .episodic import Knowledge
from .events import KNOWLEDGE, Event, Stream
from .session import Session

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

#: Out of the source tree and out of git: it is what someone said to a
#: running server, not something the repository builds.
DEFAULT_PATH = REPOSITORY_ROOT / "state" / "v689-memory.sqlite"

#: Definitions memory: glosses read into facts, as v688 retrieves them and in
#: bulk by `learn_definitions.py`.
DEFINITIONS_PATH = REPOSITORY_ROOT / "state" / "v689-definitions.sqlite"

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
CREATE TABLE IF NOT EXISTS events (
    stream TEXT NOT NULL, seq INTEGER NOT NULL, type TEXT NOT NULL,
    turn INTEGER NOT NULL, recorded REAL NOT NULL, data TEXT NOT NULL,
    PRIMARY KEY (stream, seq));
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

    # -- events ------------------------------------------------------------
    def append(self, stream: Stream) -> int:
        """Write what a stream has that the archive does not. Returns how
        many events were written."""
        rows = [event.as_row() for event in stream.unsaved()]
        if rows:
            with self.lock, self.connection:
                self.connection.executemany(
                    "INSERT OR REPLACE INTO events VALUES (?, ?, ?, ?, ?, ?)",
                    rows)
        stream.mark_saved()
        return len(rows)

    def events(self, stream: str) -> list[Event]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT stream, seq, type, turn, recorded, data FROM events "
                "WHERE stream = ? ORDER BY seq", (stream,)).fetchall()
        return [Event.from_row(row) for row in rows]

    # -- knowledge ---------------------------------------------------------
    def knowledge(self, name: str = KNOWLEDGE) -> Knowledge:
        """Knowledge replayed from its stream, or -- kept before there were
        events -- read from its rows and recorded as events."""
        events = self.events(name)
        if events or name != KNOWLEDGE:
            return Knowledge.replayed(events, name)
        with self.lock:
            read = self.connection.execute
            state = {
                "kinds": [row[0] for row in read(
                    "SELECT word FROM kinds ORDER BY rowid")],
                "edges": [list(row) for row in read(
                    "SELECT node, parent, said, conversation, taught "
                    "FROM edges ORDER BY rowid")],
                "norms": [list(row) for row in read(
                    "SELECT node, relation, object, said, mode, against, "
                    "conversation, taught FROM norms ORDER BY rowid")]}
        knowledge = Knowledge.from_state(state, name)
        self.append(knowledge.stream)
        return knowledge

    def keep(self, knowledge: Knowledge) -> None:
        """Append the knowledge's new events, and write the rows it folds to.

        The rows are a read model, rewritten whole in one transaction: they
        are small, a correction replaces a norm and `unlearn` empties them,
        and a half-written norm is worse than one that was never saved.
        """
        self.append(knowledge.stream)
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
        """What the conversation's stream gained in a turn, and beside it a
        snapshot and the turn as the page showed it. An example's own
        knowledge is its own stream, kept with it."""
        self.append(session.memory.log.conversation)
        if session.example:
            self.append(session.memory.knowledge.stream)
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
            self.connection.execute(
                "DELETE FROM events WHERE stream IN (?, ?)",
                (conversation, f"{KNOWLEDGE}:{conversation}"))


class Keeper:
    """Every conversation on the page, and the knowledge they share.

    Nothing here is thread-safe on purpose: a turn reasons over the store's
    one sqlite connection, so the server calls this with its engines lock
    held, and one caller at a time is the contract.
    """

    def __init__(self, asker, archive: Archive | None = None,
                 kept: int = KEPT, definitions=None) -> None:
        self.asker = asker
        self.archive = archive
        self.definitions = definitions
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
            found = self._restore(conversation)
        if found is None:
            found = Session(self.asker, None if example else self.knowledge,
                            conversation, example, self.definitions)
        self.held[conversation] = found
        while len(self.held) > self.kept:
            self.held.popitem(last=False)
        return found

    def _restore(self, conversation: str) -> Session | None:
        """A conversation on disk: replayed from its stream, or, kept before
        there were streams, resumed from its snapshot."""
        state = self.archive.state(conversation)
        events = self.archive.events(conversation)
        if events:
            example = bool(state and state.get("example"))
            knowledge = (self.archive.knowledge(f"{KNOWLEDGE}:{conversation}")
                         if example else self.knowledge)
            return Session.rebuild(self.asker, conversation, events,
                                   knowledge, self.definitions, example)
        if state is not None:
            return Session.resume(self.asker, state, self.knowledge,
                                  self.definitions)
        return None

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
                "memory": session.memory_view() if session else None,
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
        self.knowledge.unlearn()
        if self.archive is not None:
            self.archive.keep(self.knowledge)
        for session in self.held.values():
            if not session.example:
                session.memory.store("unlearned")

    def summary(self) -> dict:
        state = self.knowledge.as_state()
        out = {"kinds": len(state["kinds"]), "edges": len(state["edges"]),
               "norms": len(state["norms"])}
        if self.definitions is not None:
            out.update(self.definitions.summary())
        return out
