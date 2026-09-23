"""Designing on the page: a design goal asked in English, designed.

Importing this registers two things with v692, which knows nothing of
design: how a design goal's parts are put together once the encoder has
read them (`reading.contributes`, with `reading.spec_of`), and how the
act is done (`doing.contributes`). v692's page does the rest -- the act
is proposed when the encoder reads *find a quadratic with roots 2 and −3*
as `design`, and what it comes to is said like any other result.

The designer keeps what it learns for as long as the page runs: one
learner (`lessons.Learner`), and the proposer taught offline
(`proposer.py`) ordering the forms.
"""
from __future__ import annotations

from research.v692 import doing, reading as v692_reading
from research.v693 import designing
from research.v693.reading import ROLES, spec_of

ACT = "design"

_STATE: dict = {}


def _learner():
    if "learner" not in _STATE:
        from research.v691 import learned, lessons
        _STATE["learner"] = lessons.Learner(learned.Learned(None))
    return _STATE["learner"]


def _proposer():
    if "proposer" not in _STATE:
        from research.v693.proposer import Proposer
        _STATE["proposer"] = Proposer.load()
    return _STATE["proposer"]


def _assemble(said, roles, labels) -> dict:
    spec = spec_of(said, roles, labels)
    return {} if spec is None else {"SPEC": spec}


def _design(get, parts) -> doing.Result:
    spec = parts.get("SPEC")
    if spec is None or not spec.clauses:
        return doing.Result(ACT, "unknown", text="I did not catch what it "
                                                 "should be like")
    proposer = _proposer()
    found = designing.design(spec, _learner(), order=(
        proposer.order(spec) if proposer is not None else None))
    if found.design is None:
        return doing.Result(ACT, "unknown", text=found.said(),
                            about=spec.said())
    # Nothing `about`: a design said plainly is the design and what it is
    # made of, not the whole goal said back first.
    return doing.Result(ACT, "value", found.design,
                        designing.shown(found.design, spec),
                        because=found.form.said)


v692_reading.contributes(ACT, ROLES, _assemble)
doing.contributes(ACT, _design)
