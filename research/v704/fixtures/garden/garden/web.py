"""What the nursery answers on the web."""
from __future__ import annotations

from garden import plants as P

RELEASE = "2.0.1"


def _plants() -> dict:
    return {"plants": [plant.name for plant in P.read_plants()]}


def _about() -> dict:
    return {"nursery": "garden", "release": RELEASE}


ROUTES = {"/plants": _plants, "/about": _about}


def serve(path: str) -> dict:
    """The answer to a request for `path`."""
    answer = ROUTES.get(path)
    if answer is None:
        return {"error": f"not found: {path}"}
    return answer()
