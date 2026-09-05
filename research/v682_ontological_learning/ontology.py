"""A small canonical ontology with explicit facts and on-demand derivations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import re


CANONICAL_RELATIONS = {
    "type": {
        "aliases": ("type", "instance_of", "type_of"),
        "description": "an entity instantiates a class",
    },
    "is_a": {
        "aliases": ("is_a", "subclass_of", "hypernym"),
        "description": "a class is a subclass of another class",
    },
    "has_property": {
        "aliases": ("has_property", "has_attribute"),
        "description": "an entity or class has an inheritable property",
    },
}

RULES = {
    "is_a_transitivity": "is_a + is_a -> is_a",
    "type_through_is_a": "type + is_a -> type",
    "property_inheritance": "type + has_property -> has_property",
}


def normalize_term(value: str) -> str:
    """Normalize a graph identifier, not a relation's semantic meaning."""
    text = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    if not text:
        raise ValueError("ontology terms must not be empty")
    return text


class RelationNormalizer:
    """Semantic aliases are declared by relation meaning, never string similarity."""

    def __init__(self) -> None:
        self._aliases = {
            alias: canonical
            for canonical, definition in CANONICAL_RELATIONS.items()
            for alias in definition["aliases"]
        }

    def canonicalize(self, relation: str) -> str:
        normalized = normalize_term(relation)
        try:
            return self._aliases[normalized]
        except KeyError as error:
            raise ValueError(f"unsupported ontology relation: {relation!r}") from error

    @property
    def aliases(self) -> dict[str, str]:
        return dict(sorted(self._aliases.items()))


@dataclass(frozen=True, order=True)
class Fact:
    subject: str
    relation: str
    object: str

    def as_dict(self) -> dict[str, str]:
        return {"subject": self.subject, "relation": self.relation, "object": self.object}

    def text(self) -> str:
        return f"{self.subject} --{self.relation}--> {self.object}"


@dataclass(frozen=True)
class Proof:
    fact: Fact
    kind: str
    rule: str | None = None
    premises: tuple["Proof", ...] = ()
    source_relation: str | None = None

    @property
    def depth(self) -> int:
        return 0 if not self.premises else 1 + max(premise.depth for premise in self.premises)

    def as_dict(self) -> dict:
        result = {
            "fact": self.fact.as_dict(),
            "kind": self.kind,
            "proof_depth": self.depth,
        }
        if self.source_relation:
            result["source_relation"] = self.source_relation
        if self.rule:
            result["rule"] = self.rule
        if self.premises:
            result["premises"] = [premise.as_dict() for premise in self.premises]
        return result

    def lines(self, indent: int = 0) -> list[str]:
        prefix = "  " * indent
        if self.kind == "DIRECT":
            alias = f" (normalized from {self.source_relation})" if self.source_relation != self.fact.relation else ""
            return [f"{prefix}{self.fact.text()} [DIRECT]{alias}"]
        lines: list[str] = []
        for premise in self.premises:
            lines.extend(premise.lines(indent + 1))
        lines.append(f"{prefix}{self.rule}; therefore {self.fact.text()} [INFERRED]")
        return lines


@dataclass(frozen=True)
class QueryResult:
    fact: Fact
    status: str
    proof: Proof | None

    def as_dict(self) -> dict:
        result = {"query": self.fact.as_dict(), "status": self.status}
        if self.proof:
            result["proof"] = self.proof.as_dict()
        return result


class Ontology:
    """Stores only canonical explicit facts; all facts returned by closure are ephemeral."""

    def __init__(self) -> None:
        self.normalizer = RelationNormalizer()
        self._explicit: dict[Fact, Proof] = {}
        self._negative: set[Fact] = set()
        self.entities: set[str] = set()
        self.types: set[str] = set()
        self.properties: set[str] = set()

    def _fact(self, subject: str, relation: str, object_: str) -> Fact:
        return Fact(normalize_term(subject), self.normalizer.canonicalize(relation), normalize_term(object_))

    def _record_roles(self, fact: Fact) -> None:
        if fact.relation == "type":
            self.entities.add(fact.subject)
            self.types.add(fact.object)
        elif fact.relation == "is_a":
            self.types.update((fact.subject, fact.object))
        elif fact.relation == "has_property":
            self.properties.add(fact.object)
            if fact.subject not in self.entities:
                self.types.add(fact.subject)

    def add_fact(self, subject: str, relation: str, object_: str) -> Fact:
        source_relation = normalize_term(relation)
        fact = self._fact(subject, relation, object_)
        self._record_roles(fact)
        self._explicit.setdefault(
            fact, Proof(fact=fact, kind="DIRECT", source_relation=source_relation)
        )
        return fact

    def add_negative_fact(self, subject: str, relation: str, object_: str) -> Fact:
        fact = self._fact(subject, relation, object_)
        self._negative.add(fact)
        return fact

    @property
    def explicit_facts(self) -> tuple[Fact, ...]:
        return tuple(sorted(self._explicit))

    def derive(self) -> dict[Fact, Proof]:
        """Build a least fixed point without adding inferred facts to the ontology."""
        known = dict(self._explicit)
        changed = True
        while changed:
            changed = False
            facts = tuple(sorted(known))
            is_a = [fact for fact in facts if fact.relation == "is_a"]
            typed = [fact for fact in facts if fact.relation == "type"]
            properties = [fact for fact in facts if fact.relation == "has_property"]
            candidates: list[tuple[Fact, str, tuple[Proof, ...]]] = []
            for left in is_a:
                for right in is_a:
                    if left.object == right.subject:
                        candidates.append((
                            Fact(left.subject, "is_a", right.object),
                            RULES["is_a_transitivity"],
                            (known[left], known[right]),
                        ))
            for instance in typed:
                for parent in is_a:
                    if instance.object == parent.subject:
                        candidates.append((
                            Fact(instance.subject, "type", parent.object),
                            RULES["type_through_is_a"],
                            (known[instance], known[parent]),
                        ))
            for instance in typed:
                for property_fact in properties:
                    if instance.object == property_fact.subject:
                        candidates.append((
                            Fact(instance.subject, "has_property", property_fact.object),
                            RULES["property_inheritance"],
                            (known[instance], known[property_fact]),
                        ))
            for fact, rule, premises in candidates:
                candidate = Proof(fact=fact, kind="INFERRED", rule=rule, premises=premises)
                previous = known.get(fact)
                if previous is None or (previous.kind != "DIRECT" and candidate.depth < previous.depth):
                    known[fact] = candidate
                    changed = True
        return {fact: proof for fact, proof in known.items() if proof.kind == "INFERRED"}

    def all_proofs(self) -> dict[Fact, Proof]:
        return {**self._explicit, **self.derive()}

    def query(self, subject: str, relation: str, object_: str) -> QueryResult:
        fact = self._fact(subject, relation, object_)
        proof = self.all_proofs().get(fact)
        if proof and fact in self._negative:
            return QueryResult(fact, "CONFLICTED", proof)
        if proof:
            return QueryResult(fact, proof.kind, proof)
        if fact in self._negative:
            return QueryResult(fact, "CONFLICTED", None)
        return QueryResult(fact, "UNVERIFIED", None)

    def query_natural_language(self, text: str) -> QueryResult:
        cleaned = str(text).lower().strip()
        cleaned = re.sub(r"^(?:so|then|and)\s*,?\s*", "", cleaned)
        cleaned = cleaned.rstrip("?.! ").strip()
        match = re.fullmatch(r"is\s+(.+?)\s+(?:a|an)\s+(.+)", cleaned)
        if not match:
            raise ValueError(f"unsupported grounded semantic query: {text!r}")
        return self.query(match.group(1), "type", match.group(2))

    def stats(self) -> dict[str, int | float]:
        inferred = self.derive()
        depths = [proof.depth for proof in inferred.values()]
        return {
            "entities": len(self.entities),
            "types": len(self.types),
            "properties": len(self.properties),
            "canonical_relations": len(CANONICAL_RELATIONS),
            "aliases": sum(len(value["aliases"]) - 1 for value in CANONICAL_RELATIONS.values()),
            "explicit_facts": len(self._explicit),
            "inferred_facts": len(inferred),
            "rules": len(RULES),
            "average_proof_depth": round(sum(depths) / len(depths), 2) if depths else 0.0,
        }

    def ontology_document(self) -> dict:
        return {
            "representation": "canonical explicit facts; inference is derived on demand",
            "node_kinds": {
                "entities": sorted(self.entities),
                "types": sorted(self.types),
                "properties": sorted(self.properties),
            },
            "relations": {
                name: {"aliases": list(value["aliases"]), "description": value["description"]}
                for name, value in CANONICAL_RELATIONS.items()
            },
            "explicit_facts": [
                {**fact.as_dict(), "source_relation": self._explicit[fact].source_relation}
                for fact in self.explicit_facts
            ],
            "negative_facts": [fact.as_dict() for fact in sorted(self._negative)],
        }


def build_demo_ontology() -> Ontology:
    """A compact source graph deliberately uses semantic aliases from prior graph vocabularies."""
    ontology = Ontology()
    ontology.add_fact("dog", "instance_of", "mammal")
    ontology.add_fact("mammal", "hypernym", "vertebrate")
    ontology.add_fact("vertebrate", "subclass_of", "animal")
    ontology.add_fact("animal", "is_a", "organism")
    ontology.add_fact("mammal", "has_attribute", "warm_blooded")
    ontology.add_fact("organism", "has_property", "living")
    ontology.add_fact("plant", "is_a", "organism")
    ontology.add_fact("mineral", "is_a", "physical_object")
    return ontology


def facts_to_documents(facts: Iterable[tuple[Fact, Proof]]) -> list[dict]:
    return [
        {**fact.as_dict(), "proof": proof.as_dict()}
        for fact, proof in sorted(facts, key=lambda item: item[0])
    ]
