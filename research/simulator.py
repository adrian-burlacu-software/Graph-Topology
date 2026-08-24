from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

CREATE = "CREATE"
REUSE = "REUSE"
BRANCH = "BRANCH"
INHIBIT = "INHIBIT"
EXCITE = "EXCITE"


@dataclass
class Config:
    designer_learning_rate: float = 0.05
    vocabulary_learning_rate: float = 0.05
    spike_threshold: float = 1.0
    leak: float = 0.90
    excite_weight: float = 1.0
    inhibit_weight: float = 0.6
    reward_correct_reuse: float = 1.0
    reward_correct_branch: float = 1.0
    reward_wrong_reuse: float = -1.0
    reward_wrong_branch: float = -0.25
    branch_cost: float = -0.25
    feedback_weight: float = 0.10
    max_designer_cells: int = 32


@dataclass
class Synapse:
    source: int
    target: int
    kind: str = EXCITE
    weight: float = 1.0
    learning: float = 0.0


@dataclass
class Cell:
    id: int
    kind: str
    symbol: Optional[str] = None
    parent: Optional[int] = None
    order: int = 0
    incoming: list[int] = field(default_factory=list)
    outgoing: list[int] = field(default_factory=list)
    potential: float = 0.0
    spikes: int = 0
    inhibition: float = 0.0
    signal: Optional[str] = None

    # Compatibility with older experiments that called this field activation.
    @property
    def activation(self) -> float:
        return self.potential

    @activation.setter
    def activation(self, value: float) -> None:
        self.potential = value


class Network:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.cells: dict[int, Cell] = {}
        self.synapses: dict[tuple[int, int], Synapse] = {}
        self.next_cell_id = 0

        self.total_reward = 0.0
        self.total_reuse = 0
        self.total_create = 0
        self.correct_reuse = 0
        self.correct_branch = 0
        self.wrong_reuse = 0
        self.wrong_branch = 0
        self.action_reuse = 0
        self.action_branch = 0
        self.designer_spikes = 0

        self.designer_root = self.create_cell("designer")
        self.reuse_cell = self.create_cell("designer")
        self.branch_cell = self.create_cell("designer")
        self.output_cell = self.create_cell("designer")

        self.connect(self.designer_root, self.reuse_cell, EXCITE, 1.0)
        self.connect(self.designer_root, self.branch_cell, EXCITE, 1.0)
        self.connect(self.reuse_cell, self.branch_cell, INHIBIT, self.config.inhibit_weight)
        self.connect(self.branch_cell, self.reuse_cell, INHIBIT, self.config.inhibit_weight)
        self.connect(self.reuse_cell, self.output_cell, EXCITE, 1.0)
        self.connect(self.branch_cell, self.output_cell, EXCITE, 1.0)

        self.designer_signals = {CREATE: 0.0, REUSE: 0.0, BRANCH: 0.0, INHIBIT: 0.0}

    def create_cell(self, kind: str, symbol: Optional[str] = None,
                    parent: Optional[int] = None, order: int = 0) -> int:
        cell_id = self.next_cell_id
        self.next_cell_id += 1
        self.cells[cell_id] = Cell(cell_id, kind, symbol, parent, order)
        return cell_id

    def connect(self, source: int, target: int, kind: str = EXCITE,
                weight: float = 1.0) -> Synapse:
        key = (source, target)
        if key in self.synapses:
            return self.synapses[key]
        syn = Synapse(source, target, kind, weight)
        self.synapses[key] = syn
        self.cells[source].outgoing.append(target)
        self.cells[target].incoming.append(source)
        return syn

    def create_vocabulary_cell(self, symbol: str, parent: Optional[int], order: int) -> int:
        cid = self.create_cell("vocabulary", symbol, parent, order)
        if parent is not None:
            self.connect(parent, cid, EXCITE, self.config.excite_weight)

        # Every vocabulary neuron can feed the designer.  This is the missing
        # feedback path that caused the recent experiments to become blind to
        # the vocabulary graph.
        self.connect(cid, self.designer_root, EXCITE, self.config.feedback_weight)
        return cid

    def find_child(self, parent_id: Optional[int], symbol: str) -> Optional[int]:
        if parent_id is None:
            return self.find_root_symbol(symbol)
        parent = self.cells.get(parent_id)
        if parent is None:
            return None
        for cid in parent.outgoing:
            child = self.cells[cid]
            if child.kind == "vocabulary" and child.symbol == symbol:
                return cid
        return None

    def find_children(self, parent_id: Optional[int]) -> list[int]:
        if parent_id is None:
            return []
        return [cid for cid in self.cells[parent_id].outgoing
                if self.cells[cid].kind == "vocabulary"]

    def find_root_symbol(self, symbol: str) -> Optional[int]:
        for cell in self.vocabulary_cells():
            if cell.parent is None and cell.symbol == symbol:
                return cell.id
        return None

    def designer_features(self, current_id: Optional[int], symbol: str) -> dict[str, float]:
        children = self.find_children(current_id)
        existing = self.find_child(current_id, symbol)
        depth = 0 if current_id is None else self.cells[current_id].order
        return {
            "reuse_available": 1.0 if existing is not None else 0.0,
            "has_children": 1.0 if children else 0.0,
            "branch_count": float(len(children)),
            "depth": float(depth),
            "vocabulary_size": float(len(self.vocabulary_cells())),
        }

    def _reset_designer_input(self) -> None:
        # Keep the circuit state bounded.  We intentionally do not erase
        # learned weights, only transient membrane state.
        for cid in (self.designer_root, self.reuse_cell, self.branch_cell, self.output_cell):
            self.cells[cid].potential *= 0.5
            self.cells[cid].inhibition *= 0.5

    def spike_designer(self, current_id: Optional[int], symbol: str) -> None:
        features = self.designer_features(current_id, symbol)
        available = features["reuse_available"] > 0.5

        root = self.cells[self.designer_root]
        reuse = self.cells[self.reuse_cell]
        branch = self.cells[self.branch_cell]

        # Structural input.  The designer receives the same fact a biological
        # circuit would have to infer: does this edge already exist?
        root.potential += 0.75
        if available:
            reuse.potential += 1.50
            branch.potential += 0.15
        else:
            branch.potential += 1.50
            reuse.potential += 0.15

        # Existing vocabulary neurons provide weak recurrent/context feedback.
        if current_id is not None:
            self.cells[current_id].spikes += 1
            self.designer_spikes += 1
            root.potential += self.config.feedback_weight

        # Root spike drives the two competing populations.
        if root.potential >= self.config.spike_threshold:
            root.potential = 0.0
            root.spikes += 1
            self.designer_spikes += 1
            reuse.potential += self.synapses[(self.designer_root, self.reuse_cell)].weight
            branch.potential += self.synapses[(self.designer_root, self.branch_cell)].weight

        # Winner-take-most inhibition.
        if reuse.potential > self.config.spike_threshold:
            branch.inhibition += self.config.inhibit_weight
            branch.potential -= self.config.inhibit_weight
            reuse.spikes += 1
            self.designer_spikes += 1
        if branch.potential > self.config.spike_threshold:
            reuse.inhibition += self.config.inhibit_weight
            reuse.potential -= self.config.inhibit_weight
            branch.spikes += 1
            self.designer_spikes += 1

        reuse.potential *= self.config.leak
        branch.potential *= self.config.leak

    def designer_signal(self, current_id: Optional[int], symbol: str) -> str:
        f = self.designer_features(current_id, symbol)
        available = f["reuse_available"] > 0.5

        # Structural signal is deliberately the strongest component. Learned
        # membrane state can bias it, but cannot permanently erase the basic
        # distinction between an existing edge and a missing edge.
        reuse_drive = (2.0 if available else 0.0) + self.cells[self.reuse_cell].potential
        branch_drive = (2.0 if not available else 0.0) + self.cells[self.branch_cell].potential
        if f["has_children"]:
            reuse_drive += 0.10
        branch_drive -= 0.05 * f["branch_count"]

        signal = REUSE if reuse_drive >= branch_drive else BRANCH
        self.designer_signals[signal] += 1.0
        self.cells[self.output_cell].signal = signal
        return signal

    def learn_designer(self, action: str, correct_action: str, reward: float) -> None:
        self.total_reward += reward
        if action == REUSE:
            self.action_reuse += 1
            if correct_action == REUSE:
                self.correct_reuse += 1
            else:
                self.wrong_reuse += 1
        elif action == BRANCH:
            self.action_branch += 1
            if correct_action == BRANCH:
                self.correct_branch += 1
            else:
                self.wrong_branch += 1

        lr = self.config.designer_learning_rate
        delta = lr * reward
        target = self.reuse_cell if action == REUSE else self.branch_cell
        opposite = self.branch_cell if action == REUSE else self.reuse_cell
        self.cells[target].potential = max(-5.0, min(5.0, self.cells[target].potential + delta))
        self.cells[opposite].potential = max(-5.0, min(5.0, self.cells[opposite].potential - delta * 0.5))

        for source in (self.reuse_cell, self.branch_cell):
            syn = self.synapses.get((self.designer_root, source))
            if syn is None:
                continue
            if source == target:
                change = delta * 0.10
                syn.weight += change
                syn.learning += change
            else:
                syn.weight -= delta * 0.05
            syn.weight = max(0.05, min(2.0, syn.weight))

    def _apply_decision(self, current_id: Optional[int], symbol: str, order: int,
                        action: str) -> tuple[int, int, int, float]:
        existing = self.find_child(current_id, symbol)
        correct = REUSE if existing is not None else BRANCH

        if existing is not None and action == REUSE:
            return existing, 0, 1, self.config.reward_correct_reuse

        if existing is None and action == BRANCH:
            cid = self.create_vocabulary_cell(symbol, current_id, order)
            return cid, 1, 0, self.config.reward_correct_branch + self.config.branch_cost

        if existing is not None and action == BRANCH:
            # Wrong branch must never duplicate an existing edge.
            return existing, 0, 0, self.config.reward_wrong_branch + self.config.branch_cost

        # Wrong reuse: structural repair creates the missing edge.
        cid = self.create_vocabulary_cell(symbol, current_id, order)
        return cid, 1, 0, self.config.reward_wrong_reuse

    def process_word(self, word: str) -> dict:
        current_id: Optional[int] = None
        created = reused = branched = 0

        for order, symbol in enumerate(word):
            existing = self.find_child(current_id, symbol)
            correct = REUSE if existing is not None else BRANCH
            self._reset_designer_input()
            self.spike_designer(current_id, symbol)
            action = self.designer_signal(current_id, symbol)

            new_id, made, reused_now, reward = self._apply_decision(current_id, symbol, order, action)
            current_id = new_id
            created += made
            reused += reused_now
            if action == BRANCH:
                branched += 1
            if made:
                self.total_create += made
            if reused_now:
                self.total_reuse += reused_now
            self.learn_designer(action, correct, reward)

        return {"word": word, "created": created, "reused": reused, "branched": branched}

    def train(self, words: list[str], epochs: int = 5):
        print()
        print("=== PLASTICITY EXPERIMENT ===")
        print(f"epochs : {epochs}")
        print(f"words  : {len(words)}")
        print()
        for epoch in range(1, epochs + 1):
            rb, ub, cb = self.total_reward, self.total_reuse, self.total_create
            for word in words:
                self.process_word(word)
            print(f"epoch={epoch:3d} cells={len(self.cells):3d} reuse={self.total_reuse-ub:3d} create={self.total_create-cb:3d} reward={self.total_reward-rb:8.2f}")

    def vocabulary_cells(self) -> list[Cell]:
        return [c for c in self.cells.values() if c.kind == "vocabulary"]

    def designer_cells(self) -> list[Cell]:
        return [c for c in self.cells.values() if c.kind == "designer"]

    def reusable_cells_count(self) -> int:
        return sum(1 for c in self.vocabulary_cells() if any(self.cells[s].kind == "vocabulary" for s in c.incoming))

    def vocabulary_feedback_count(self) -> int:
        return sum(1 for s in self.synapses.values()
                   if self.cells[s.source].kind == "vocabulary"
                   and self.cells[s.target].kind == "designer")

    def print_summary(self):
        print()
        print("=== FINAL ===")
        print(f"cells               : {len(self.cells)}")
        print(f"designer_cells      : {len(self.designer_cells())}")
        print(f"vocabulary_cells    : {len(self.vocabulary_cells())}")
        print(f"synapses            : {len(self.synapses)}")
        print(f"reusable_cells      : {self.reusable_cells_count()}")
        print(f"designer_spikes     : {self.designer_spikes}")
        print(f"vocabulary_feedback_synapses: {self.vocabulary_feedback_count()}")
        print(f"total_reward        : {self.total_reward:.1f}")
        print(f"total_reuse         : {self.total_reuse}")
        print(f"total_create        : {self.total_create}")
        print(f"correct_reuse       : {self.correct_reuse}")
        print(f"correct_branch      : {self.correct_branch}")
        print(f"wrong_reuse         : {self.wrong_reuse}")
        print(f"wrong_branch        : {self.wrong_branch}")
        print(f"action_reuse        : {self.action_reuse}")
        print(f"action_branch       : {self.action_branch}")
        print()
        print("=== DESIGNER OUTPUT ===")
        print(f"reuse_output : {self.cells[self.reuse_cell].potential:.4f}")
        print(f"branch_output: {self.cells[self.branch_cell].potential:.4f}")
        print()
        print("=== DECISION MATRIX ===")
        print("reuse available:")
        print(f"    REUSE  : {self.correct_reuse}")
        print(f"    BRANCH : {self.wrong_branch}")
        print("reuse unavailable:")
        print(f"    REUSE  : {self.wrong_reuse}")
        print(f"    BRANCH : {self.correct_branch}")
        print()
        print("=== DESIGNER SYNAPSES ===")
        for s in self.synapses.values():
            a, b = self.cells[s.source], self.cells[s.target]
            if a.kind == b.kind == "designer":
                print(f"{s.source} -> {s.target} {s.kind:7s} weight={s.weight:.4f} learning={s.learning:.4f}")

    def print_topology(self):
        print()
        print("=== TOPOLOGY ===")
        for c in self.cells.values():
            parent = "None" if c.parent is None else str(c.parent)
            symbol = "None" if c.symbol is None else c.symbol
            print(f"{c.id:3d} {c.kind:10s} {symbol:>4s} parent={parent:>4s} in={len(c.incoming)} out={len(c.outgoing)} order={c.order} pot={c.potential:.2f} spikes={c.spikes} inh={c.inhibition:.2f}")

    def print_vocabulary_tree(self):
        print()
        print("=== VOCABULARY GRAPH ===")
        roots = sorted((c for c in self.vocabulary_cells() if c.parent is None), key=lambda c: c.id)
        def walk(cid: int, prefix: str = ""):
            c = self.cells[cid]
            print(f"{prefix}{c.symbol} [{c.id}]")
            children = sorted((self.cells[x] for x in c.outgoing if self.cells[x].kind == "vocabulary"), key=lambda x: x.order)
            for child in children:
                walk(child.id, prefix + "  ")
        for root in roots:
            walk(root.id)
