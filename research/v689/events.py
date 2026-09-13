"""Event sourcing: memory as what happened to it.

Until now v689 kept memory the way a program keeps a variable: `tell` changed
the tables, and the tables were saved. What a conversation *did* to memory --
told this, then took that back, then learned the word -- was gone the moment
the tables moved on, and so were two things a conversation about time needs:
what memory held at an earlier turn, and the order things were said in.

So nothing writes a table any more. A change is an **event**, appended to a
**stream** and never altered, and every table is a **projection**: a fold of
the stream from the first event to the last. The episodic tables, the
attention table, the timeline and the trie planned over them are all read
models of the one log.

## Domain-driven design: the model this makes explicit

    bounded context   aggregate      stream              projections
    conversation      Session        the conversation's  episodic tables,
                                                         attention, timeline,
                                                         the tries
    knowledge         Knowledge      `knowledge`, shared kinds, edges, norms
                                     (one per example)

An **aggregate** decides; a **projection** only applies. A command -- `tell`,
`introduce`, E2's `withdraw` -- may walk v687 or ask v688 before it decides,
and what it decided is recorded as an event. Applying an event asks nothing of
anyone, so replaying a stream is deterministic: E2's verdict on the airplane is
in the log as `withdrawn`, not re-asked of v688 on a restart that may answer
differently.

Events about an individual belong to the conversation; events about a kind
belong to knowledge, because what was taught about kinds is no one
conversation's (`longterm.py`). The stream is chosen by the same test
`episodic.Layered` routes a table write by, so the two can never disagree.

## Two times

A stream's order is **telling time**: the order things were said in, which is
what `what did I tell you first` asks about. It is not **story time** -- `the
dog slept; before that it had barked` is told sleep-first and happened
bark-first. Story time is a projection like any other (`timeline.py`), built
from what the events say about when.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

#: The stream every conversation's knowledge is kept in, unless it is an
#: example's, which keeps its own.
KNOWLEDGE = "knowledge"

#: Every kind of event, and what it records. The one place to read to know
#: what can happen to memory.
TYPES = {
    # -- attention (discourse.py) ---------------------------------------
    "turn_began": "a turn started; salience decays",
    "referent_added": "an individual came up, in this order",
    "attended": "a mention refreshed an individual's salience",
    "named": "an individual was told its name",
    "owned": "an individual was said to be yours",
    # -- episodic memory (episodic.py) ----------------------------------
    "placed": "an individual was put under a kind, or moved under a narrower",
    "told": "a fact about one node, replacing what it contradicts",
    "judged": "v688's answer for the kind, beside a told fact",
    "withdrawn": "E2 took a doing back: what carried it did it",
    "restored": "E2 gave a doing back: nothing carrying it does it",
    "imported": "a conversation kept before events were, as it was saved",
    # -- knowledge (episodic.Knowledge) ---------------------------------
    "kind_coined": "a word the store has no sense for became a kind",
    "related": "a taught edge between two kinds",
    "unlearned": "everything taught was forgotten",
    # -- time (timeline.py) ---------------------------------------------
    "episode_opened": "the story moved to a time it had not been at",
    "episode_entered": "the story went back to a time it had been at",
    "occurred": "something happened: an occurrence, with who took part",
    "ordered": "T1: one occurrence comes before, or during, another",
    "changed": "T4: an occurrence began or ended a state (VerbNet)",
}


@dataclass(frozen=True)
class Event:
    """One thing that happened to memory. Never changed once appended."""

    stream: str
    seq: int
    type: str
    data: dict = field(default_factory=dict)
    turn: int = 0
    recorded: float = 0.0

    def as_row(self) -> tuple:
        return (self.stream, self.seq, self.type, self.turn, self.recorded,
                json.dumps(self.data, sort_keys=True))

    @classmethod
    def from_row(cls, row) -> "Event":
        stream, seq, kind, turn, recorded, data = row
        return cls(stream, int(seq), kind, json.loads(data), int(turn or 0),
                   float(recorded or 0.0))

    def as_dict(self) -> dict:
        return {"stream": self.stream, "seq": self.seq, "type": self.type,
                "turn": self.turn, "data": dict(self.data)}


class Stream:
    """An append-only list of events, and how much of it is already on disk."""

    def __init__(self, name: str, events=()) -> None:
        self.name = name
        self.events: list[Event] = list(events)
        #: events before this index have been written by the archive
        self.saved = len(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    def append(self, kind: str, data: dict, turn: int = 0) -> Event:
        if kind not in TYPES:
            raise ValueError(f"no such event: {kind}")
        event = Event(self.name, len(self.events) + 1, kind, dict(data), turn,
                      time.time())
        self.events.append(event)
        return event

    def unsaved(self) -> list[Event]:
        return self.events[self.saved:]

    def mark_saved(self) -> None:
        self.saved = len(self.events)

    def tail(self, count: int = 40) -> list[dict]:
        return [event.as_dict() for event in self.events[-count:]]


class Log:
    """The two streams one conversation writes to, and who applies each event.

    Projections register a handler per event type (`on`); `record` appends
    an event to the conversation's stream or to the knowledge stream and
    hands it to every handler for its type, in the order they registered.
    `replay` hands over events already in a stream, and says so while it
    does, so a projection can put off work it only needs once at the end --
    re-planning a trie after every event of a long log, say.
    """

    def __init__(self, conversation: Stream, knowledge: Stream) -> None:
        self.conversation = conversation
        self.knowledge = knowledge
        self.handlers: dict[str, list] = {}
        #: the turn events are stamped with
        self.turn = 0
        self.replaying = False

    def on(self, kind: str, handler) -> None:
        if kind not in TYPES:
            raise ValueError(f"no such event: {kind}")
        self.handlers.setdefault(kind, []).append(handler)

    def record(self, kind: str, data: dict, knowledge: bool = False):
        """Append and apply. Returns what the first handler returned."""
        stream = self.knowledge if knowledge else self.conversation
        return self.apply(stream.append(kind, data, self.turn))

    def apply(self, event: Event):
        found = None
        for index, handler in enumerate(self.handlers.get(event.type, ())):
            result = handler(event)
            if index == 0:
                found = result
        return found

    def replay(self, events) -> int:
        """Apply events that are already in a stream. Returns how many."""
        count = 0
        self.replaying = True
        try:
            for event in events:
                self.turn = max(self.turn, event.turn)
                self.apply(event)
                count += 1
        finally:
            self.replaying = False
        return count
