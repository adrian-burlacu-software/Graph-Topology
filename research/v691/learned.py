"""What the agent works out about acting, kept between conversations.

`verbs.py` reads what verbs mean, and `DESIGN.md` §9 lists what VerbNet does
not say: that a door cannot be open and closed at once, that you have to be
holding a thing to put it down, that opening an open door is not an action.
None of those is a fact about a verb -- they are facts about doors and
hands and rooms -- so they are not in VerbNet and never will be.

They are, however, exactly the kind of thing a person says while you are
getting something wrong. This is where that goes, and it is a file in
`state/` rather than in the repository because it is what someone told a
running server, not something the repository builds.

## Three things, and the first buys the third

    excludes   two states of a thing that cannot both hold: `open` and
               `closed`, `alive` and `dead`, `full` and `empty`
    requires   something else that has to be true to do a thing: `you can
               only put it down if you are holding it`
    brings     an effect VerbNet did not mention
    carries    a doing done by being carried: a pig flies on a plane that
               flies -- learned from what the story showed (E2), not from
               being told (`DESIGN.md` §10b)
    blocks     a state that stops a thing being done: a door that is
               locked does not open -- a negative precondition, learned
               from a surprise (`lessons.py`)
    tried      what held when an action was done, and whether it worked:
               the evidence `lessons.py` compares, kept so a lesson can
               be drawn from a failure and a success a conversation apart
    prefers    which verb brings a fact about, the way people do it: seen
               in what they said happened, and in their corrections

**Negative preconditions come free with exclusion**, which is why exclusion
is worth learning first. `needs` in the executive is a list of slots that
must be *present*, so *the door is not already open* cannot be said. But
once `open` and `closed` are known to exclude each other, *the door is
closed* says the same thing and is positive. One thing learned closes two
of §9's gaps.

## How it is learned

**Being corrected.** The scene holds `closed door`; the person says *the
door is open now*. Two states of one thing, one right after the other, and
the person marking the second as a change: that is what `open` and `closed`
being incompatible looks like from the inside. Recorded with what was said,
so it can be read back and argued with.

**Being told.** `you can only pour it if the glass is empty`,
`a door cannot be open and closed`, `you must be holding it to drop it`.

Nothing here is inferred from a single co-occurrence: two facts holding at
once is evidence that they *do not* exclude, and never evidence that they
do. So exclusion is only ever learned from a correction or from being told,
and `forget` is a plain part of the interface because both can be wrong.
"""
from __future__ import annotations

import re
import sqlite3
import threading
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

#: Out of the source tree and out of git, beside v689's memory, and for the
#: same reason: it is what someone said to a running server.
DEFAULT_PATH = REPOSITORY_ROOT / "state" / "v691-learned.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS excludes (
    one TEXT NOT NULL, other TEXT NOT NULL, said TEXT NOT NULL,
    learned REAL NOT NULL, PRIMARY KEY (one, other));
CREATE TABLE IF NOT EXISTS requires (
    verb TEXT NOT NULL, literal TEXT NOT NULL, said TEXT NOT NULL,
    learned REAL NOT NULL, PRIMARY KEY (verb, literal));
CREATE TABLE IF NOT EXISTS brings (
    verb TEXT NOT NULL, literal TEXT NOT NULL, gone INTEGER NOT NULL,
    said TEXT NOT NULL, learned REAL NOT NULL, PRIMARY KEY (verb, literal));
CREATE TABLE IF NOT EXISTS carries (
    verb TEXT NOT NULL, thing TEXT NOT NULL, carrier TEXT NOT NULL,
    named TEXT NOT NULL, way TEXT NOT NULL, own INTEGER NOT NULL,
    said TEXT NOT NULL,
    learned REAL NOT NULL, PRIMARY KEY (verb, thing, carrier));
CREATE TABLE IF NOT EXISTS blocks (
    verb TEXT NOT NULL, literal TEXT NOT NULL, said TEXT NOT NULL,
    learned REAL NOT NULL, PRIMARY KEY (verb, literal));
CREATE TABLE IF NOT EXISTS prefers (
    predicate TEXT NOT NULL, verb TEXT NOT NULL, count REAL NOT NULL,
    said TEXT NOT NULL, learned REAL NOT NULL,
    PRIMARY KEY (predicate, verb));
CREATE TABLE IF NOT EXISTS tried (
    id INTEGER PRIMARY KEY AUTOINCREMENT, verb TEXT NOT NULL,
    context TEXT NOT NULL, worked INTEGER NOT NULL, learned REAL NOT NULL);
"""

#: How many tries of one verb are kept, newest first: enough to compare,
#: and a bound so a long-running server does not grow without limit.
TRIED = 40


class Learned:
    """One sqlite file, its own connection and its own lock.

    In memory when no path is given, so a test or a REPL does not write to
    what a server is using.
    """

    def __init__(self, path: Path | str | None = DEFAULT_PATH) -> None:
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.connection = sqlite3.connect(
            str(self.path) if self.path is not None else ":memory:",
            check_same_thread=False)
        self.connection.executescript(SCHEMA)
        self._migrate()
        self.connection.commit()

    #: Columns added after a table first shipped, so a store written by an
    #: older server gains them rather than failing: what a lesson is scoped
    #: to (a kind, or everything), the kinds it does not hold of, and the
    #: kind of thing a try was about.
    ADDED = {"requires": (("scope", "TEXT NOT NULL DEFAULT ''"),
                          ("unless", "TEXT NOT NULL DEFAULT ''")),
             "blocks": (("scope", "TEXT NOT NULL DEFAULT ''"),
                        ("unless", "TEXT NOT NULL DEFAULT ''")),
             "tried": (("kind", "TEXT NOT NULL DEFAULT ''"),)}

    def _migrate(self) -> None:
        for table, columns in self.ADDED.items():
            have = {row[1] for row in self.connection.execute(
                f"PRAGMA table_info({table})")}
            for name, kind in columns:
                if name not in have:
                    self.connection.execute(
                        f"ALTER TABLE {table} ADD COLUMN {name} {kind}")

    # -- states that cannot both hold --------------------------------------
    def exclude(self, one: str, other: str, said: str = "") -> bool:
        """Learn that two states of a thing are incompatible.

        Kept both ways round, because exclusion is symmetric and a lookup
        that had to try both orders would eventually forget to.
        """
        if one == other or not one or not other:
            return False
        if other in self.excluded(one):
            # Already known. Said again -- and a compound utterance is said
            # once per claim -- it is not news, and a reply that reported
            # it twice would be telling the person what they just said.
            return False
        with self.lock:
            for first, second in ((one, other), (other, one)):
                self.connection.execute(
                    "INSERT OR REPLACE INTO excludes VALUES (?, ?, ?, ?)",
                    (first, second, said, time.time()))
            self.connection.commit()
        return True

    def excluded(self, one: str) -> frozenset:
        with self.lock:
            return frozenset(row[0] for row in self.connection.execute(
                "SELECT other FROM excludes WHERE one = ?", (one,)))

    def exclusions(self) -> list:
        with self.lock:
            return sorted((row[0], row[1], row[2]) for row in
                          self.connection.execute(
                              "SELECT one, other, said FROM excludes")
                          if row[0] < row[1])

    # -- what else has to be true ------------------------------------------
    def require(self, verb: str, literal: str, said: str = "") -> bool:
        """Learn a precondition VerbNet does not give.

        The literal is over an action's *positions* rather than over its
        roles -- `with ?object ?subject` is *you are holding it* -- because
        what a person says is about the thing and the doer, and which
        thematic role those are is different for every verb.
        """
        if not verb or not literal or literal in self.required(verb):
            return False
        with self.lock:
            self.connection.execute(
                "INSERT OR REPLACE INTO requires (verb, literal, said, "
                "learned) VALUES (?, ?, ?, ?)",
                (verb, literal, said, time.time()))
            self.connection.commit()
        return True

    def required(self, verb: str) -> list:
        with self.lock:
            return [row[0] for row in self.connection.execute(
                "SELECT literal FROM requires WHERE verb = ?", (verb,))]

    def requirements(self) -> list:
        with self.lock:
            return sorted((row[0], row[1], row[2]) for row in
                          self.connection.execute(
                              "SELECT verb, literal, said FROM requires"))

    # -- effects it was not told about -------------------------------------
    def bring(self, verb: str, literal: str, gone: bool = False,
              said: str = "") -> bool:
        if not verb or not literal:
            return False
        with self.lock:
            self.connection.execute(
                "INSERT OR REPLACE INTO brings VALUES (?, ?, ?, ?, ?)",
                (verb, literal, int(gone), said, time.time()))
            self.connection.commit()
        return True

    def brought(self, verb: str) -> list:
        with self.lock:
            return [(row[0], bool(row[1])) for row in
                    self.connection.execute(
                        "SELECT literal, gone FROM brings WHERE verb = ?",
                        (verb,))]

    # -- doing a thing by being carried -----------------------------------
    def carry(self, verb: str, thing: str, carrier: str, named: str = "",
              way: str = "", own: bool = False, said: str = "") -> bool:
        """Learn that a `thing` does `verb` by being aboard a `carrier` that
        does it -- E2, seen happen: a pig put on a plane that flew, flew.

        Kinds, not individuals: what was seen of one pig and one plane is
        kept of pigs and planes, the way a person comes away from it. `way`
        is the verb that put the thing aboard, when the story said, so the
        plan can do it the way it was seen done; `own` is whether the thing
        could have done it by itself anyway. `thing` and `carrier` are
        senses where the story had them, and `named` is the word the
        carrier was called, for saying it. False when it was known.
        """
        if not verb or not thing or not carrier or thing == carrier:
            return False
        with self.lock:
            known = self.connection.execute(
                "SELECT way FROM carries WHERE verb = ? AND thing = ? AND "
                "carrier = ?", (verb, thing, carrier)).fetchone()
            if known is not None and (known[0] == way or not way):
                return False
            self.connection.execute(
                "INSERT OR REPLACE INTO carries VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?)",
                (verb, thing, carrier, named or carrier, way, int(own), said,
                 time.time()))
            self.connection.commit()
        return True

    def carriers(self, verb: str) -> list:
        """(thing, carrier, named, way, said) for every way of doing
        `verb` by being carried that has been seen."""
        with self.lock:
            return [tuple(row) for row in self.connection.execute(
                "SELECT thing, carrier, named, way, said FROM carries "
                "WHERE verb = ? ORDER BY learned", (verb,))]

    def carryings(self) -> list:
        with self.lock:
            return sorted(tuple(row) for row in self.connection.execute(
                "SELECT verb, thing, carrier, way, said FROM carries"))

    # -- what stops a thing being done -------------------------------------
    def block(self, verb: str, literal: str, said: str = "") -> bool:
        """Learn a negative precondition: while `literal` holds, `verb`
        cannot be done. Over positions, like `require`."""
        if not verb or not literal or literal in self.blockers(verb):
            return False
        with self.lock:
            self.connection.execute(
                "INSERT OR REPLACE INTO blocks (verb, literal, said, "
                "learned) VALUES (?, ?, ?, ?)",
                (verb, literal, said, time.time()))
            self.connection.commit()
        return True

    def blockers(self, verb: str) -> list:
        with self.lock:
            return [row[0] for row in self.connection.execute(
                "SELECT literal FROM blocks WHERE verb = ?", (verb,))]

    def blockings(self) -> list:
        with self.lock:
            return sorted((row[0], row[1], row[2]) for row in
                          self.connection.execute(
                              "SELECT verb, literal, said FROM blocks"))

    # -- the evidence ------------------------------------------------------
    def tried_it(self, verb: str, context, worked: bool,
                 kind: str = "") -> int:
        """Record what held when `verb` was done and whether it worked, and
        the kind of thing it was done to. Returns the row, so a success
        later reported as a failure can be taken back (`untry`)."""
        with self.lock:
            row = self.connection.execute(
                "INSERT INTO tried (verb, context, worked, learned, kind) "
                "VALUES (?, ?, ?, ?, ?)",
                (verb, ";".join(sorted(context)), int(worked),
                 time.time(), kind)).lastrowid
            self.connection.execute(
                "DELETE FROM tried WHERE verb = ? AND id NOT IN (SELECT id "
                "FROM tried WHERE verb = ? ORDER BY id DESC LIMIT ?)",
                (verb, verb, TRIED))
            self.connection.commit()
        return row

    def untry(self, row: int) -> None:
        with self.lock:
            self.connection.execute("DELETE FROM tried WHERE id = ?", (row,))
            self.connection.commit()

    def tries(self, verb: str, kinds: bool = False) -> list:
        """(context, worked) for every try of `verb` kept, oldest first --
        (context, worked, kind) with `kinds`."""
        with self.lock:
            rows = self.connection.execute(
                "SELECT context, worked, kind FROM tried WHERE verb = ? "
                "ORDER BY id", (verb,)).fetchall()
        out = [(frozenset(one for one in row[0].split(";") if one),
                bool(row[1]), row[2]) for row in rows]
        return out if kinds else [one[:2] for one in out]

    # -- which verb, when several would do --------------------------------
    def prefer(self, predicate: str, verb: str, said: str = "",
               weight: float = 1.0) -> float:
        """Count one more time `verb` was seen bringing `predicate` about,
        or was asked for instead of what was done. Returns the count."""
        if not predicate or not verb:
            return 0.0
        with self.lock:
            row = self.connection.execute(
                "SELECT count FROM prefers WHERE predicate = ? AND verb = ?",
                (predicate, verb)).fetchone()
            count = (row[0] if row else 0.0) + weight
            self.connection.execute(
                "INSERT OR REPLACE INTO prefers VALUES (?, ?, ?, ?, ?)",
                (predicate, verb, count, said, time.time()))
            self.connection.commit()
        return count

    def preferred(self, predicate: str) -> list:
        """Verbs for `predicate`, most seen first; of two seen as often,
        the one seen last."""
        with self.lock:
            return [row[0] for row in self.connection.execute(
                "SELECT verb FROM prefers WHERE predicate = ? "
                "ORDER BY count DESC, learned DESC", (predicate,))]

    def preferences(self) -> list:
        with self.lock:
            return sorted(tuple(row) for row in self.connection.execute(
                "SELECT predicate, verb, count, said FROM prefers"))

    # -- where a lesson holds ----------------------------------------------
    def scoped(self, table: str, verb: str, literal: str) -> tuple:
        """(scope, unless) of a lesson: the kind it holds of ("" for
        everything), and the kinds it does not."""
        with self.lock:
            row = self.connection.execute(
                f"SELECT scope, unless FROM {table} WHERE verb = ? AND "
                f"literal = ?", (verb, literal)).fetchone()
        if row is None:
            return ("", ())
        return (row[0], tuple(one for one in row[1].split(",") if one))

    def rescope(self, table: str, verb: str, literal: str, scope=None,
                unless=None) -> None:
        with self.lock:
            if scope is not None:
                self.connection.execute(
                    f"UPDATE {table} SET scope = ? WHERE verb = ? AND "
                    f"literal = ?", (scope, verb, literal))
            if unless is not None:
                self.connection.execute(
                    f"UPDATE {table} SET unless = ? WHERE verb = ? AND "
                    f"literal = ?", (",".join(sorted(set(unless))), verb,
                                     literal))
            self.connection.commit()

    # -- taking it back ----------------------------------------------------
    def forget(self, what: str, one: str, other: str = "") -> int:
        """Unlearn. Part of the interface and not an afterthought: being
        corrected is how most of this is learned, and a correction can be
        wrong in its turn."""
        with self.lock:
            if what == "excludes":
                done = self.connection.execute(
                    "DELETE FROM excludes WHERE (one = ? AND other = ?) "
                    "OR (one = ? AND other = ?)",
                    (one, other, other, one)).rowcount
            elif what == "carries":
                done = self.connection.execute(
                    "DELETE FROM carries WHERE verb = ? AND thing = ?",
                    (one, other)).rowcount
            elif what == "requires":
                done = self.connection.execute(
                    "DELETE FROM requires WHERE verb = ? AND literal = ?",
                    (one, other)).rowcount
            elif what == "prefers":
                done = self.connection.execute(
                    "DELETE FROM prefers WHERE predicate = ? AND verb = ?",
                    (one, other)).rowcount
            elif what == "blocks":
                done = self.connection.execute(
                    "DELETE FROM blocks WHERE verb = ? AND literal = ?",
                    (one, other)).rowcount
            else:
                done = self.connection.execute(
                    "DELETE FROM brings WHERE verb = ? AND literal = ?",
                    (one, other)).rowcount
            self.connection.commit()
        return done

    def count(self) -> dict:
        with self.lock:
            return {name: self.connection.execute(
                f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                for name in ("excludes", "requires", "brings",
                             "carries", "blocks", "prefers")}

    def close(self) -> None:
        self.connection.close()


def applied(actions: list, learned: "Learned | None", is_a=None) -> list:
    """The actions, with what has been learned folded in.

    Done at grounding rather than in the schema, for two reasons. The
    exclusions are about *this thing* -- `closed door` is a fact and
    `closed ?Patient` is not something the person said -- and what has been
    learned changes while the server runs, so a schema built once would go
    stale the moment somebody taught it something.
    """
    if learned is None:
        return actions
    out = []
    for action in actions:
        verb = action.name.split()[0]
        needs = set(action.needs)
        adds = set(action.adds)
        deletes = set(action.deletes)
        # A state that excludes one this action brings about was true
        # before and is not after. **This is where a negative precondition
        # becomes a positive one**: to open a door it has to be closed.
        for literal in list(adds):
            parts = literal.split()
            if len(parts) != 2:
                continue
            for other in learned.excluded(parts[0]):
                needs.add(f"{other} {parts[1]}")
                deletes.add(f"{other} {parts[1]}")
            # What WordNet says is the opposite stops holding -- unlocking
            # a door ends its being locked -- but is not demanded first:
            # nobody said the door was closed, and opening it should not
            # wait on being told.
            for other in _opposites(parts[0]):
                deletes.add(f"{other} {parts[1]}")
        for literal in learned.required(verb):
            filled = _fill(literal, action.name, action.adds)
            if filled and _holds(learned, "requires", verb, literal,
                                 action, is_a):
                needs.add(filled)
        for literal, gone in learned.brought(verb):
            filled = _fill(literal, action.name, action.adds)
            if filled:
                (deletes if gone else adds).add(filled)
        forbids = set(getattr(action, "forbids", ()))
        for literal in learned.blockers(verb):
            filled = _fill(literal, action.name, action.adds)
            if filled and _holds(learned, "blocks", verb, literal, action,
                                 is_a):
                forbids.add(filled)
        out.append(type(action)(action.name, frozenset(needs),
                                frozenset(adds),
                                frozenset(deletes) - frozenset(adds),
                                frozenset(forbids)))
    return out


def _holds(learned, table: str, verb: str, literal: str, action,
           is_a) -> bool:
    """Whether a lesson holds of the thing this action changes: inside its
    scope and outside its exceptions. Everything, when nothing can say what
    kind a thing is."""
    scope, unless = learned.scoped(table, verb, literal)
    if is_a is None or not (scope or unless):
        return True
    thing = _fill(IT, action.name, action.adds)
    if not thing:
        return not scope
    if scope and not is_a(thing, scope):
        return False
    return not any(is_a(thing, one) for one in unless)


def _opposites(state: str) -> frozenset:
    try:
        from research.v691.verbs import opposites
        return opposites(state)
    except Exception:                              # noqa: BLE001
        return frozenset()


#: The thing an action changes, in a lesson learned from acting: found by
#: what the action brings about rather than by where it is named, so a
#: lesson about doors finds the door in every reading of `open`.
IT = "?it"


def changed(adds) -> str:
    """The one thing an action's one-place effects are about, or empty:
    `open door` changes the door, `fetch book shop` (`carrying book`) the
    book. Empty when there are none or several."""
    found = {literal.split()[1] for literal in adds
             if len(literal.split()) == 2}
    return next(iter(found)) if len(found) == 1 else ""


def _fill(literal: str, name: str, adds=()) -> str:
    """A learned literal over `?subject`/`?object`/`?place`/`?it`, ground
    against one action. Empty when the action has no such position, which
    is how `you must be holding it` leaves an intransitive verb alone."""
    parts = name.split()
    where = {"?subject": parts[1] if len(parts) > 1 else "",
             "?object": parts[2] if len(parts) > 2 else "",
             "?place": parts[-1] if len(parts) > 3 else "",
             IT: changed(adds) or (parts[2] if len(parts) > 2 else "")}
    out = []
    for word in literal.split():
        if word.startswith("?"):
            if not where.get(word):
                return ""
            out.append(where[word])
        else:
            out.append(word)
    return " ".join(out)


# -- being told -------------------------------------------------------------

#: `a door cannot be open and closed`, `nothing can be full and empty`.
CANNOT = re.compile(r"(?:cannot|can ?not|can't|could not|never)\s+be\s+"
                    r"(\w+)\s+and\s+(\w+)")
#: `nothing can be open and closed`, `no door can be open and closed`.
NOTHING = re.compile(r"(?:nothing|no \w+)\s+(?:can|could|may)\s+be\s+"
                     r"(\w+)\s+and\s+(\w+)")
#: `open and closed are opposites`, `full is the opposite of empty`.
OPPOSITE = re.compile(r"\b(\w+)\s+and\s+(\w+)\s+are\s+opposites?\b"
                      r"|\b(\w+)\s+is\s+the\s+opposite\s+of\s+(\w+)")
#: `you can only put it down if you are holding it`.
ONLY_IF = re.compile(r"\byou\s+can\s+only\s+(\w+)\b.*?\bif\s+(.+)")
#: `you must be holding it to drop it`, `it must be closed before you can
#: open it`.
MUST = re.compile(r"\b(.+?)\s+(?:to|before\s+you\s+can)\s+(\w+)\b")

#: How a condition names the doer and the thing, when a person states one.
#: `you`/`I` is whoever is acting and `it` is what is being acted on, which
#: is what makes a taught condition apply to a verb rather than to one
#: particular door.
#: `must be`, `have to be` and a bare `are` all say the same thing about a
#: condition, and a person uses whichever the sentence around it wants.
BEING = r"(?:must\s+be\s+|has\s+to\s+be\s+|have\s+to\s+be\s+|should\s+be\s+" \
        r"|are\s+|is\s+|'re\s+|'s\s+)?"
HOLDING = re.compile(rf"\byou\s+{BEING}(?:holding|carrying|have)\s+it\b")
IT_IS = re.compile(rf"\bit\s+{BEING}(\w+)")
THING_IS = re.compile(rf"\bthe\s+(\w+)\s+{BEING}(\w+)")


def condition(text: str) -> str:
    """One taught condition as a literal over positions, or empty.

    Over `?subject` and `?object` rather than over thematic roles, because
    a person says *you* and *it* and which role those are is different for
    every verb.
    """
    plain = " ".join(text.lower().split())
    if HOLDING.search(plain):
        return "with ?object ?subject"
    found = IT_IS.search(plain)
    if found:
        # `it` is the thing acted on, wherever a reading of the verb names
        # it: `open box` has no object position and still opens the box.
        return f"{found.group(1)} {IT}"
    found = THING_IS.search(plain)
    if found:
        return f"{found.group(2)} {found.group(1)}"
    return ""


def teaching(text: str) -> list:
    """What a sentence teaches about acting, as ("exclude", a, b) or
    ("require", verb, literal). Empty when it teaches nothing."""
    plain = " ".join(text.lower().replace(",", " ").split())
    out: list = []
    for pattern in (CANNOT, NOTHING):
        for found in pattern.finditer(plain):
            out.append(("exclude", found.group(1), found.group(2)))
    for found in OPPOSITE.finditer(plain):
        pair = [one for one in found.groups() if one]
        if len(pair) == 2:
            out.append(("exclude", pair[0], pair[1]))
    if out:
        return out
    found = ONLY_IF.search(plain)
    if found:
        literal = condition(found.group(2))
        if literal:
            return [("require", found.group(1), literal)]
    found = MUST.search(plain)
    if found and ("must" in found.group(1) or "have to" in found.group(1)):
        literal = condition(found.group(1))
        if literal:
            return [("require", found.group(2), literal)]
    return out
