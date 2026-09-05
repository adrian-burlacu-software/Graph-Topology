"""Focused semantic-graph loading, empirical rule discovery, and inference."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from math import sqrt
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable


REQUIRED_TABLES = {"nodes", "edges", "relations", "metadata"}
REQUIRED_COLUMNS = {
    "nodes": {"node", "normalized", "label", "definition", "source_mask", "node_type"},
    "edges": {"subject", "relation", "object", "source"},
    "relations": {"relation", "phrases"},
    "metadata": {"key", "value"},
}
TRAINING_FOLDS = 5
MAX_INFERRED_FACTS = 5_000
MAX_INFERENCE_DEPTH = 2


@dataclass(frozen=True, order=True)
class Fact:
    subject: str
    relation: str
    object: str

    def as_dict(self) -> dict[str, str]:
        return {"subject": self.subject, "relation": self.relation, "object": self.object}


@dataclass(frozen=True)
class Edge:
    fact: Fact
    source: str


@dataclass(frozen=True)
class Proof:
    fact: Fact
    kind: str
    source: str | None = None
    rule: dict[str, Any] | None = None
    premises: tuple["Proof", ...] = ()

    @property
    def depth(self) -> int:
        return 0 if not self.premises else 1 + max(premise.depth for premise in self.premises)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "fact": self.fact.as_dict(),
            "kind": self.kind,
            "proof_depth": self.depth,
        }
        if self.source:
            result["source"] = self.source
        if self.rule:
            result["rule"] = {
                key: self.rule[key]
                for key in ("left_relation", "right_relation", "result_relation", "confidence")
            }
        if self.premises:
            result["premises"] = [premise.as_dict() for premise in self.premises]
        return result


def _stable_fold(values: Iterable[str]) -> int:
    return int(sha256("\0".join(values).encode("utf-8")).hexdigest(), 16) % TRAINING_FOLDS


def _pair_overlap(left: set[tuple[str, str]], right: set[tuple[str, str]]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _name_similarity(left: str, right: str) -> float:
    left_words = set(re.findall(r"[a-z0-9]+", left.lower()))
    right_words = set(re.findall(r"[a-z0-9]+", right.lower()))
    return len(left_words & right_words) / len(left_words | right_words) if left_words | right_words else 0.0


class SemanticGraph:
    """An immutable in-memory projection of every direct edge in a focused SQLite graph."""

    def __init__(self, database: Path) -> None:
        self.database = Path(database).resolve()
        if not self.database.is_file():
            raise FileNotFoundError(f"Focused semantic database not found: {self.database}")
        self.tables: dict[str, dict[str, Any]] = {}
        self.metadata: dict[str, str] = {}
        self.nodes: dict[str, dict[str, Any]] = {}
        self.relation_phrases: dict[str, str] = {}
        self.edges: list[Edge] = []
        self.direct: dict[Fact, Proof] = {}
        self.outgoing: dict[str, list[Edge]] = defaultdict(list)
        self.by_relation: dict[str, list[Edge]] = defaultdict(list)
        self.pair_relations: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._load()

    def _load(self) -> None:
        try:
            connection = sqlite3.connect(f"file:{self.database}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
        except sqlite3.Error as error:
            raise RuntimeError(f"Unable to open focused semantic database {self.database}: {error}") from error
        with connection:
            found = {
                str(row["name"])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            missing = REQUIRED_TABLES - found
            if missing:
                raise RuntimeError(f"{self.database} is not a focused semantic graph; missing tables: {sorted(missing)}")
            for table in sorted(REQUIRED_TABLES):
                columns = {
                    str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")
                }
                missing_columns = REQUIRED_COLUMNS[table] - columns
                if missing_columns:
                    raise RuntimeError(f"{table} is missing required columns: {sorted(missing_columns)}")
                self.tables[table] = {
                    "columns": sorted(columns),
                    "rows": int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]),
                }
            self.metadata = {
                str(row["key"]): str(row["value"])
                for row in connection.execute("SELECT key,value FROM metadata ORDER BY key")
            }
            self.nodes = {
                str(row["node"]): {
                    "id": str(row["node"]),
                    "label": str(row["label"] or row["normalized"] or row["node"]),
                    "normalized": str(row["normalized"] or ""),
                    "definition": row["definition"],
                    "node_type": str(row["node_type"]),
                    "source_mask": int(row["source_mask"]),
                }
                for row in connection.execute(
                    "SELECT node,normalized,label,definition,source_mask,node_type FROM nodes"
                )
            }
            self.relation_phrases = {
                str(row["relation"]): str(row["phrases"])
                for row in connection.execute("SELECT relation,phrases FROM relations")
            }
            for row in connection.execute("SELECT subject,relation,object,source FROM edges"):
                fact = Fact(str(row["subject"]), str(row["relation"]), str(row["object"]))
                edge = Edge(fact, str(row["source"]))
                self.edges.append(edge)
                self.direct[fact] = Proof(fact, "DIRECT", source=edge.source)
                self.outgoing[fact.subject].append(edge)
                self.by_relation[fact.relation].append(edge)
                self.pair_relations[fact.subject, fact.object].add(fact.relation)
        for collection in (self.edges,):
            collection.sort(key=lambda edge: edge.fact)
        for edges in self.outgoing.values():
            edges.sort(key=lambda edge: edge.fact)
        for edges in self.by_relation.values():
            edges.sort(key=lambda edge: edge.fact)

    @property
    def relations(self) -> tuple[str, ...]:
        return tuple(sorted(self.by_relation))

    def graph_stats(self) -> dict[str, Any]:
        return {
            "database": str(self.database),
            "tables_used": self.tables,
            "rows_examined": sum(value["rows"] for value in self.tables.values()),
            "entities": len(self.nodes),
            "edges": len(self.edges),
            "unique_relations": len(self.relations),
            "metadata": self.metadata,
            "relationship_representation": "edges(subject, relation, object, source)",
            "entity_identifier_column": "nodes.node",
            "edge_direction": "subject -> object",
        }

    def relation_models(self, discovery: dict[str, Any]) -> list[dict[str, Any]]:
        pairs = {
            relation: {(edge.fact.subject, edge.fact.object) for edge in edges}
            for relation, edges in self.by_relation.items()
        }
        subjects = {
            relation: {edge.fact.subject for edge in edges}
            for relation, edges in self.by_relation.items()
        }
        objects = {
            relation: {edge.fact.object for edge in edges}
            for relation, edges in self.by_relation.items()
        }
        equivalence_candidates: list[dict[str, Any]] = []
        for index, left in enumerate(self.relations):
            for right in self.relations[index + 1:]:
                overlap = _pair_overlap(pairs[left], pairs[right])
                neighborhood = (_pair_overlap(subjects[left], subjects[right]) +
                                _pair_overlap(objects[left], objects[right])) / 2
                confidence = round(0.75 * overlap + 0.20 * neighborhood +
                                   0.05 * _name_similarity(left, right), 4)
                if len(pairs[left] & pairs[right]) >= 3 or confidence >= 0.25:
                    equivalence_candidates.append({
                        "left_relation": left, "right_relation": right,
                        "shared_direct_pairs": len(pairs[left] & pairs[right]),
                        "pair_overlap": round(overlap, 4),
                        "neighborhood_overlap": round(neighborhood, 4),
                        "name_similarity": round(_name_similarity(left, right), 4),
                        "confidence": confidence,
                        "status": "CANDIDATE" if confidence < 0.65 else "ACCEPTED",
                    })
        models = []
        for relation in self.relations:
            relation_pairs = pairs[relation]
            inverse_scores = [
                (len(relation_pairs & {(object_, subject) for subject, object_ in pairs[other]}) /
                 len(relation_pairs), other)
                for other in self.relations if relation_pairs
            ]
            inverse_score, inverse = max(inverse_scores, default=(0.0, None))
            symmetric = len(relation_pairs & {(object_, subject) for subject, object_ in relation_pairs})
            transitivity = discovery["pair_metrics"].get((relation, relation), {})
            models.append({
                "canonical_relation": relation,
                "source_relations": [relation],
                "support": len(self.by_relation[relation]),
                "phrases": self.relation_phrases.get(relation, ""),
                "direction": "subject_to_object",
                "candidate_inverse": inverse,
                "inverse_confidence": round(inverse_score, 4),
                "candidate_symmetry": {
                    "support": symmetric,
                    "confidence": round(symmetric / len(relation_pairs), 4) if relation_pairs else 0.0,
                },
                "candidate_transitivity": {
                    "support": transitivity.get("outcomes", 0),
                    "two_hop_paths": transitivity.get("paths", 0),
                    "precision": transitivity.get("precision", 0.0),
                },
                "equivalence_candidates": [
                    item for item in equivalence_candidates
                    if relation in (item["left_relation"], item["right_relation"])
                ],
            })
        return models

    def discover_rules(self) -> dict[str, Any]:
        """Examine every observed R1,R2 two-hop path and every direct R3 outcome."""
        paths = Counter()
        train_paths = Counter()
        validation_paths = Counter()
        outcomes = Counter()
        train_outcomes = Counter()
        validation_outcomes = Counter()
        unique_outcomes: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
        samples: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for first in self.edges:
            first_fact = first.fact
            for second in self.outgoing.get(first_fact.object, ()):
                second_fact = second.fact
                pair = first_fact.relation, second_fact.relation
                paths[pair] += 1
                held_out = _stable_fold((
                    first_fact.subject, first_fact.relation, first_fact.object,
                    second_fact.relation, second_fact.object,
                )) == 0
                (validation_paths if held_out else train_paths)[pair] += 1
                for result_relation in self.pair_relations.get((first_fact.subject, second_fact.object), ()):
                    key = first_fact.relation, second_fact.relation, result_relation
                    outcomes[key] += 1
                    unique_outcomes[key].add((first_fact.subject, second_fact.object))
                    (validation_outcomes if held_out else train_outcomes)[key] += 1
                    if len(samples[key]) < 3:
                        samples[key].append({
                            "premise_1": {**first_fact.as_dict(), "source": first.source},
                            "premise_2": {**second_fact.as_dict(), "source": second.source},
                            "observed_outcome": {
                                "subject": first_fact.subject, "relation": result_relation,
                                "object": second_fact.object,
                            },
                        })
        pair_metrics = {
            pair: {
                "paths": total,
                "outcomes": sum(count for key, count in outcomes.items() if key[:2] == pair),
                "precision": round(
                    max((count / total for key, count in outcomes.items() if key[:2] == pair), default=0.0), 4
                ),
            }
            for pair, total in paths.items()
        }
        rules = []
        for key, support in outcomes.items():
            left, right, result = key
            total = paths[left, right]
            train_support = train_outcomes[key]
            validation_support = validation_outcomes[key]
            train_precision = train_support / train_paths[left, right] if train_paths[left, right] else 0.0
            validation_precision = (
                validation_support / validation_paths[left, right]
                if validation_paths[left, right] else 0.0
            )
            precision = support / total
            accepted = (
                train_support >= 8
                and validation_support >= 2
                and precision >= 0.10
                and validation_precision >= 0.10
            )
            confidence = min(
                1.0,
                precision * (0.65 + 0.35 * min(1.0, sqrt(support) / 10))
                * (0.75 + 0.25 * min(1.0, validation_precision / max(precision, 0.0001))),
            )
            rules.append({
                "left_relation": left,
                "right_relation": right,
                "result_relation": result,
                "status": "ACCEPTED" if accepted else "REJECTED",
                "support": support,
                "coverage": round(len(unique_outcomes[key]) / len(self.by_relation[result]), 4),
                "precision": round(precision, 4),
                "contradictions": total - support,
                "observed_two_hop_paths": total,
                "matching_direct_outcomes": support,
                "confidence": round(confidence, 4),
                "sample_size": total,
                "testing": {
                    "training_paths": train_paths[left, right],
                    "training_matches": train_support,
                    "training_precision": round(train_precision, 4),
                    "validation_paths": validation_paths[left, right],
                    "validation_matches": validation_support,
                    "validation_precision": round(validation_precision, 4),
                },
                "samples": samples[key],
            })
        rules.sort(key=lambda item: (
            item["status"] != "ACCEPTED", -item["confidence"], -item["support"],
            item["left_relation"], item["right_relation"], item["result_relation"],
        ))
        return {
            "discovery_method": (
                "All direct edges are joined on first.object = second.subject. "
                "A candidate is emitted only when the corresponding direct R3 outcome exists."
            ),
            "certification_method": (
                "A deterministic 80/20 path split requires at least 8 training matches, "
                "2 held-out matches, and >=0.10 training-independent held-out precision."
            ),
            "relation_pairs_examined": len(paths),
            "observed_two_hop_paths": sum(paths.values()),
            "candidate_rules": len(rules),
            "accepted_rules": sum(rule["status"] == "ACCEPTED" for rule in rules),
            "rejected_rules": sum(rule["status"] == "REJECTED" for rule in rules),
            "rules": rules,
            "pair_metrics": pair_metrics,
        }

    def infer(self, accepted_rules: Iterable[dict[str, Any]]) -> dict[Fact, Proof]:
        """Apply empirical rules generically without mutating the source graph."""
        rules = tuple(accepted_rules)
        known = dict(self.direct)
        inferred: dict[Fact, Proof] = {}
        for _ in range(MAX_INFERENCE_DEPTH):
            by_relation: dict[str, list[Proof]] = defaultdict(list)
            outgoing: dict[tuple[str, str], list[Proof]] = defaultdict(list)
            for proof in known.values():
                by_relation[proof.fact.relation].append(proof)
                outgoing[proof.fact.subject, proof.fact.relation].append(proof)
            additions: dict[Fact, Proof] = {}
            for rule in rules:
                for first in by_relation.get(rule["left_relation"], ()):
                    for second in outgoing.get((first.fact.object, rule["right_relation"]), ()):
                        fact = Fact(first.fact.subject, rule["result_relation"], second.fact.object)
                        if fact in known or fact in additions:
                            continue
                        additions[fact] = Proof(fact, "INFERRED", rule=rule, premises=(first, second))
                        if len(inferred) + len(additions) >= MAX_INFERRED_FACTS:
                            break
                    if len(inferred) + len(additions) >= MAX_INFERRED_FACTS:
                        break
                if len(inferred) + len(additions) >= MAX_INFERRED_FACTS:
                    break
            if not additions:
                break
            known.update(additions)
            inferred.update(additions)
            if len(inferred) >= MAX_INFERRED_FACTS:
                break
        return inferred

    def prove(self, fact: Fact, accepted_rules: Iterable[dict[str, Any]], depth: int = 3,
              active: frozenset[Fact] = frozenset()) -> Proof | None:
        """Back-chain an individual real-graph query when it is outside the visualization LOD."""
        if fact in self.direct:
            return self.direct[fact]
        if depth <= 0 or fact in active:
            return None
        next_active = active | {fact}
        for rule in accepted_rules:
            if rule["result_relation"] != fact.relation:
                continue
            for first in self.by_relation.get(rule["left_relation"], ()):
                if first.fact.subject != fact.subject:
                    continue
                premise_1 = self.direct[first.fact]
                premise_2 = self.prove(
                    Fact(first.fact.object, rule["right_relation"], fact.object),
                    accepted_rules, depth - 1, next_active,
                )
                if premise_2:
                    return Proof(fact, "INFERRED", rule=rule, premises=(premise_1, premise_2))
        return None

    def resolve_node(self, text: str) -> str | None:
        cleaned = str(text).strip().lower()
        candidates = (
            cleaned, f"en:{cleaned}", cleaned.replace(" ", "_"), f"en:{cleaned.replace(' ', '_')}",
        )
        for candidate in candidates:
            if candidate in self.nodes:
                return candidate
        matches = [
            node_id for node_id, node in self.nodes.items()
            if node["label"].lower() == cleaned or node["normalized"].lower() == cleaned
        ]
        return sorted(matches)[0] if matches else None

    def query_natural_language(self, text: str, accepted_rules: Iterable[dict[str, Any]]) -> dict[str, Any]:
        cleaned = re.sub(
            r"^(?:so|then|and)\s*,?\s*", "", str(text).lower().strip()
        ).rstrip("?.! ")
        match = re.fullmatch(r"is\s+(.+?)\s+(?:a|an)\s+(.+)", cleaned)
        if not match:
            return {"status": "UNVERIFIED", "reason": "Unsupported grounded semantic query."}
        subject, object_ = (self.resolve_node(value) for value in match.groups())
        if not subject or not object_:
            return {"status": "UNVERIFIED", "reason": "A query term is absent from the focused graph."}
        relations = [
            relation for relation, phrases in self.relation_phrases.items()
            if "is a" in phrases.lower()
        ] or list(self.relations)
        for relation in sorted(relations):
            fact = Fact(subject, relation, object_)
            proof = self.prove(fact, accepted_rules)
            if proof:
                return {
                    "status": "VERIFIED", "evidence_kind": proof.kind,
                    "query": fact.as_dict(), "proof": proof.as_dict(),
                }
        return {
            "status": "UNVERIFIED",
            "query": {"subject": subject, "relation_candidates": sorted(relations), "object": object_},
        }


def facts_document(facts: dict[Fact, Proof]) -> list[dict[str, Any]]:
    return [{**fact.as_dict(), "proof": proof.as_dict()} for fact, proof in sorted(facts.items())]
