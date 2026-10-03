"""The semantic graph a turn touched, drawn from what the turn kept.

Every node and edge says where it came from (`via`), so the drawing is an
account and not a picture:

    walk       v687's walk: the chain it climbed (is_a), the evidence it
               found (concept -relation-> object, with its source and how
               sure), each node's rules
    memory     the conversation's individuals and what they are (is_a)
    said       what a phrase was resolved to mean (refers to)
    run        v688's question loop: how active each word's node was, and
               what it doubted (concept -relation-> predicate)
"""
from __future__ import annotations


def label(node: str) -> str:
    """`beagle.n.01` as `beagle`; anything else as it is."""
    parts = str(node).rsplit(".", 2)
    if len(parts) == 3 and parts[1] in "nvasr" and parts[2].isdigit():
        return parts[0]
    return str(node)


class _Graph:
    def __init__(self) -> None:
        self.nodes: dict = {}
        self.edges: list = []
        self._seen: set = set()

    def node(self, key, via: str, **said) -> str:
        key = str(key)
        found = self.nodes.setdefault(key, {"id": key, "label": label(key),
                                            "via": []})
        if via not in found["via"]:
            found["via"].append(via)
        for name, value in said.items():
            if value is not None:
                found[name] = value
        return key

    def edge(self, start, end, relation: str, via: str, **said) -> None:
        if not start or not end:
            return
        start, end = self.node(start, via), self.node(end, via)
        key = (start, end, relation, via)
        if key in self._seen:
            return
        self._seen.add(key)
        self.edges.append({"from": start, "to": end,
                           "relation": relation, "via": via,
                           **{k: v for k, v in said.items() if v is not None}})


def _resolutions(turn: dict) -> list:
    found = turn.get("resolution")
    if isinstance(found, dict):
        return [found]
    return [one for one in found or () if isinstance(one, dict)]


def graph_of(turn: dict) -> dict:
    """{nodes: [...], edges: [...]} of what the turn touched."""
    graph = _Graph()
    walk = turn.get("walk") or {}
    chain = walk.get("chain") or []
    for below, above in zip(chain, chain[1:]):
        graph.edge(below, above, "is_a", "walk")
    for step in walk.get("steps") or ():
        concept = step.get("concept")
        if concept:
            rules = graph.node(concept, "walk")
            ruled = graph.nodes[rules].setdefault("rules", [])
            if step.get("rule") and step["rule"] not in ruled:
                ruled.append(step["rule"])
    for one in walk.get("evidence") or ():
        graph.edge(one.get("concept"), one.get("object"),
                   one.get("relation", "?"), "walk",
                   source=one.get("source"),
                   confidence=one.get("confidence"))
    for one in (turn.get("memory") or {}).get("individuals") or ():
        graph.node(one.get("id"), "memory", individual=True)
        graph.edge(one.get("id"), one.get("parent"), "is_a", "memory")
    for one in _resolutions(turn):
        if one.get("referent"):
            graph.node(one["referent"], "said", individual=True)
            graph.edge(f"“{one.get('expression')}”", one["referent"],
                       "refers to", "said", how=one.get("how"))
    run = turn.get("run") or {}
    table = ((run.get("buffer") or {}).get("activation") or {}).get(
        "table") or {}
    for word, value in table.items():
        # a word's node, where the walk named it by its sense
        match = next((key for key, one in graph.nodes.items()
                      if one["label"] == word), None)
        graph.node(match or word, "run", activation=round(float(value), 3))
    for one in (run.get("summary") or {}).get("doubts") or ():
        graph.edge(one.get("concept"), one.get("predicate"),
                   one.get("relation", "?"), "run", doubt=one.get("reason"),
                   confidence=one.get("confidence"))
    return {"nodes": list(graph.nodes.values()), "edges": graph.edges}
