from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

try:
    from genome import GENOME
except ImportError:
    from .genome import GENOME


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

    # Genome-driven experimental controls.
    genome: dict = field(default_factory=lambda: GENOME)


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

    @property
    def activation(self) -> float:
        return self.potential

    @activation.setter
    def activation(self, value: float) -> None:
        self.potential = value


class Network:
    """
    Plastic topology simulator.

    Important experimental change:

    The designer no longer receives a privileged boolean saying
    "reuse_available".

    Instead, the current vocabulary graph generates local activity:
      - matching vocabulary cells become active;
      - parent/context activity becomes active;
      - the designer integrates those signals.

    The designer therefore has to learn that a strong matching signal
    corresponds to REUSE while weak/no matching activity corresponds
    to BRANCH.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

        self.genome = self.config.genome

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

        self.designer_signals = {
            CREATE: 0.0,
            REUSE: 0.0,
            BRANCH: 0.0,
            INHIBIT: 0.0,
        }

        self.designer_root = self.create_cell("designer")
        self.reuse_cell = self.create_cell("designer")
        self.branch_cell = self.create_cell("designer")
        self.output_cell = self.create_cell("designer")

        self.connect(
            self.designer_root,
            self.reuse_cell,
            EXCITE,
            1.0,
        )

        self.connect(
            self.designer_root,
            self.branch_cell,
            EXCITE,
            1.0,
        )

        self.connect(
            self.reuse_cell,
            self.branch_cell,
            INHIBIT,
            self.config.inhibit_weight,
        )

        self.connect(
            self.branch_cell,
            self.reuse_cell,
            INHIBIT,
            self.config.inhibit_weight,
        )

        self.connect(
            self.reuse_cell,
            self.output_cell,
            EXCITE,
            1.0,
        )

        self.connect(
            self.branch_cell,
            self.output_cell,
            EXCITE,
            1.0,
        )

    # ------------------------------------------------------------------
    # Genome helpers
    # ------------------------------------------------------------------

    @property
    def designer_genome(self):
        return self.genome["designer"]

    @property
    def growth_genome(self):
        return self.genome["growth"]

    @property
    def connection_genome(self):
        return self.genome["connection"]

    @property
    def ordering_genome(self):
        return self.genome["ordering"]

    @property
    def inhibition_genome(self):
        return self.genome["inhibition"]

    @property
    def reuse_genome(self):
        return self.genome["reuse"]

    @property
    def plasticity_genome(self):
        return self.genome["plasticity"]

    # ------------------------------------------------------------------
    # Cells / graph
    # ------------------------------------------------------------------

    def create_cell(
        self,
        kind: str,
        symbol: Optional[str] = None,
        parent: Optional[int] = None,
        order: int = 0,
    ) -> int:
        cell_id = self.next_cell_id
        self.next_cell_id += 1

        self.cells[cell_id] = Cell(
            id=cell_id,
            kind=kind,
            symbol=symbol,
            parent=parent,
            order=order,
        )

        return cell_id

    def connect(
        self,
        source: int,
        target: int,
        kind: str = EXCITE,
        weight: float = 1.0,
    ) -> Synapse:
        key = (source, target)

        if key in self.synapses:
            return self.synapses[key]

        syn = Synapse(
            source=source,
            target=target,
            kind=kind,
            weight=weight,
        )

        self.synapses[key] = syn

        self.cells[source].outgoing.append(target)
        self.cells[target].incoming.append(source)

        return syn

    def vocabulary_cells(self) -> list[Cell]:
        return [
            c for c in self.cells.values()
            if c.kind == "vocabulary"
        ]

    def designer_cells(self) -> list[Cell]:
        return [
            c for c in self.cells.values()
            if c.kind == "designer"
        ]

    def find_root_symbol(self, symbol: str) -> Optional[int]:
        for cell in self.vocabulary_cells():
            if cell.parent is None and cell.symbol == symbol:
                return cell.id

        return None

    def find_child(
        self,
        parent_id: Optional[int],
        symbol: str,
    ) -> Optional[int]:
        if parent_id is None:
            return self.find_root_symbol(symbol)

        parent = self.cells.get(parent_id)

        if parent is None:
            return None

        for cid in parent.outgoing:
            child = self.cells[cid]

            if (
                child.kind == "vocabulary"
                and child.symbol == symbol
            ):
                return cid

        return None

    def find_children(
        self,
        parent_id: Optional[int],
    ) -> list[int]:
        if parent_id is None:
            return []

        parent = self.cells[parent_id]

        return [
            cid
            for cid in parent.outgoing
            if self.cells[cid].kind == "vocabulary"
        ]

    def create_vocabulary_cell(
        self,
        symbol: str,
        parent: Optional[int],
        order: int,
    ) -> int:
        cid = self.create_cell(
            "vocabulary",
            symbol,
            parent,
            order,
        )

        if parent is not None:
            self.connect(
                parent,
                cid,
                EXCITE,
                self.config.excite_weight,
            )

        # Vocabulary cells provide local recurrent feedback.
        self.connect(
            cid,
            self.designer_root,
            EXCITE,
            self.config.feedback_weight,
        )

        return cid

    # ------------------------------------------------------------------
    # Local sensory/context signal
    # ------------------------------------------------------------------

    def _matching_children(
        self,
        current_id: Optional[int],
        symbol: str,
    ) -> list[int]:
        """
        Return matching local vocabulary cells.

        This is deliberately represented as a list of active cells rather
        than being converted into a boolean "reuse_available" feature.
        """
        if current_id is None:
            roots = [
                c.id
                for c in self.vocabulary_cells()
                if c.parent is None and c.symbol == symbol
            ]
            return roots

        return [
            cid
            for cid in self.find_children(current_id)
            if self.cells[cid].symbol == symbol
        ]

    def _stimulate_local_context(
        self,
        current_id: Optional[int],
        symbol: str,
    ) -> tuple[float, float]:
        """
        Produce local graph activity.

        Returns:
            match_activity
            context_activity

        The designer never receives a reuse boolean.
        """

        matching = self._matching_children(
            current_id,
            symbol,
        )

        match_activity = 0.0

        for cid in matching:
            cell = self.cells[cid]

            cell.potential += self.designer_genome["match_gain"]

            match_activity += 1.0

            # Matching cells spike locally.
            if cell.potential >= self.designer_genome["threshold"]:
                cell.potential = 0.0
                cell.spikes += 1

        context_activity = 0.0

        if current_id is not None:
            current = self.cells[current_id]

            current.potential += (
                self.designer_genome["context_gain"]
            )

            context_activity += min(
                1.0,
                max(0.0, current.order * 0.1),
            )

            current.spikes += 1

        return match_activity, context_activity

    # ------------------------------------------------------------------
    # Designer
    # ------------------------------------------------------------------

    def _reset_designer_input(self) -> None:
        leak = self.designer_genome["leak"]

        for cid in (
            self.designer_root,
            self.reuse_cell,
            self.branch_cell,
            self.output_cell,
        ):
            cell = self.cells[cid]

            cell.potential *= leak
            cell.inhibition *= leak

    def spike_designer(
        self,
        current_id: Optional[int],
        symbol: str,
    ) -> None:
        match_activity, context_activity = (
            self._stimulate_local_context(
                current_id,
                symbol,
            )
        )

        root = self.cells[self.designer_root]
        reuse = self.cells[self.reuse_cell]
        branch = self.cells[self.branch_cell]

        input_gain = self.designer_genome["input_gain"]

        root.potential += input_gain

        # Matching local activity excites reuse.
        reuse.potential += (
            match_activity
            * self.designer_genome["match_gain"]
        )

        # Context gives the branch population a weak baseline.
        branch.potential += (
            self.designer_genome["branch_bias"]
            + context_activity
            * self.designer_genome["context_gain"]
        )

        # Root distributes the generic input.
        if root.potential >= self.designer_genome["threshold"]:
            root.potential = 0.0

            root.spikes += 1
            self.designer_spikes += 1

            reuse.potential += (
                self.synapses[
                    (self.designer_root, self.reuse_cell)
                ].weight
            )

            branch.potential += (
                self.synapses[
                    (self.designer_root, self.branch_cell)
                ].weight
            )

        # Winner-take-most competition.
        threshold = self.designer_genome["threshold"]

        if reuse.potential >= threshold:
            branch.inhibition += self.inhibition_genome["strength"]

            branch.potential -= self.inhibition_genome["strength"]

            reuse.spikes += 1
            self.designer_spikes += 1

        if branch.potential >= threshold:
            reuse.inhibition += self.inhibition_genome["strength"]

            reuse.potential -= self.inhibition_genome["strength"]

            branch.spikes += 1
            self.designer_spikes += 1

        reuse.potential *= self.designer_genome["leak"]
        branch.potential *= self.designer_genome["leak"]

    def designer_signal(
        self,
        current_id: Optional[int],
        symbol: str,
    ) -> str:
        """
        Make the decision from learned membrane activity.

        There is NO structural lookup here.
        """

        reuse_drive = self.cells[self.reuse_cell].potential
        branch_drive = self.cells[self.branch_cell].potential

        reuse_drive += self.designer_genome["reuse_bias"]
        branch_drive += self.designer_genome["branch_bias"]

        margin = self.designer_genome["decision_margin"]

        if reuse_drive > branch_drive + margin:
            signal = REUSE
        elif branch_drive > reuse_drive + margin:
            signal = BRANCH
        else:
            # Deterministic tie-break.
            signal = (
                REUSE
                if reuse_drive >= branch_drive
                else BRANCH
            )

        self.designer_signals[signal] += 1.0

        self.cells[self.output_cell].signal = signal

        return signal

    # ------------------------------------------------------------------
    # Plasticity
    # ------------------------------------------------------------------

    def learn_designer(
        self,
        action: str,
        correct_action: str,
        reward: float,
    ) -> None:
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

        lr = (
            self.config.designer_learning_rate
            * self.plasticity_genome[
                "reward_learning_rate"
            ]
        )

        delta = lr * reward

        target = (
            self.reuse_cell
            if action == REUSE
            else self.branch_cell
        )

        opposite = (
            self.branch_cell
            if action == REUSE
            else self.reuse_cell
        )

        target_cell = self.cells[target]
        opposite_cell = self.cells[opposite]

        target_cell.potential = max(
            -5.0,
            min(5.0, target_cell.potential + delta),
        )

        opposite_cell.potential = max(
            -5.0,
            min(
                5.0,
                opposite_cell.potential - delta * 0.5,
            ),
        )

        weight_lr = self.plasticity_genome[
            "weight_learning_rate"
        ]

        for source in (
            self.reuse_cell,
            self.branch_cell,
        ):
            syn = self.synapses.get(
                (self.designer_root, source)
            )

            if syn is None:
                continue

            if source == target:
                change = delta * weight_lr

                syn.weight += change
                syn.learning += change
            else:
                syn.weight -= delta * weight_lr * 0.5

            syn.weight = max(
                0.05,
                min(2.0, syn.weight),
            )

    # ------------------------------------------------------------------
    # Structural action
    # ------------------------------------------------------------------

    def _apply_decision(
        self,
        current_id: Optional[int],
        symbol: str,
        order: int,
        action: str,
    ) -> tuple[int, int, int, float]:
        existing = self.find_child(
            current_id,
            symbol,
        )

        correct = (
            REUSE
            if existing is not None
            else BRANCH
        )

        if existing is not None and action == REUSE:
            return (
                existing,
                0,
                1,
                self.config.reward_correct_reuse,
            )

        if existing is None and action == BRANCH:
            cid = self.create_vocabulary_cell(
                symbol,
                current_id,
                order,
            )

            return (
                cid,
                1,
                0,
                self.config.reward_correct_branch
                + self.config.branch_cost,
            )

        if existing is not None and action == BRANCH:
            # Never duplicate an existing edge.
            return (
                existing,
                0,
                0,
                self.config.reward_wrong_branch
                + self.config.branch_cost,
            )

        # Wrong reuse: repair structurally.
        cid = self.create_vocabulary_cell(
            symbol,
            current_id,
            order,
        )

        return (
            cid,
            1,
            0,
            self.config.reward_wrong_reuse,
        )

    # ------------------------------------------------------------------
    # Word processing
    # ------------------------------------------------------------------

    def process_word(
        self,
        word: str,
        learn: bool = True,
    ) -> dict:
        current_id: Optional[int] = None

        created = 0
        reused = 0
        branched = 0

        for order, symbol in enumerate(word):
            # Ground truth is captured BEFORE mutation.
            existing = self.find_child(
                current_id,
                symbol,
            )

            correct = (
                REUSE
                if existing is not None
                else BRANCH
            )

            self._reset_designer_input()

            self.spike_designer(
                current_id,
                symbol,
            )

            action = self.designer_signal(
                current_id,
                symbol,
            )

            new_id, made, reused_now, reward = (
                self._apply_decision(
                    current_id,
                    symbol,
                    order,
                    action,
                )
            )

            current_id = new_id

            created += made
            reused += reused_now

            if action == BRANCH:
                branched += 1

            if made:
                self.total_create += made

            if reused_now:
                self.total_reuse += reused_now

            if learn:
                self.learn_designer(
                    action,
                    correct,
                    reward,
                )

        return {
            "word": word,
            "created": created,
            "reused": reused,
            "branched": branched,
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        words: list[str],
        epochs: int = 5,
    ):
        print()
        print("=== PLASTICITY EXPERIMENT ===")
        print(f"epochs : {epochs}")
        print(f"words  : {len(words)}")
        print()

        for epoch in range(1, epochs + 1):
            rb = self.total_reward
            ub = self.total_reuse
            cb = self.total_create

            for word in words:
                self.process_word(
                    word,
                    learn=True,
                )

            print(
                f"epoch={epoch:3d} "
                f"cells={len(self.cells):3d} "
                f"reuse={self.total_reuse - ub:3d} "
                f"create={self.total_create - cb:3d} "
                f"reward={self.total_reward - rb:8.2f}"
            )

    # ------------------------------------------------------------------
    # Metrics / output
    # ------------------------------------------------------------------

    def reusable_cells_count(self) -> int:
        return sum(
            1
            for c in self.vocabulary_cells()
            if any(
                self.cells[s].kind == "vocabulary"
                for s in c.incoming
            )
        )

    def vocabulary_feedback_count(self) -> int:
        return sum(
            1
            for s in self.synapses.values()
            if (
                self.cells[s.source].kind == "vocabulary"
                and self.cells[s.target].kind == "designer"
            )
        )

    def print_summary(self):
        print()
        print("=== FINAL ===")
        print(f"cells               : {len(self.cells)}")
        print(
            f"designer_cells      : "
            f"{len(self.designer_cells())}"
        )
        print(
            f"vocabulary_cells    : "
            f"{len(self.vocabulary_cells())}"
        )
        print(
            f"synapses            : "
            f"{len(self.synapses)}"
        )
        print(
            f"reusable_cells      : "
            f"{self.reusable_cells_count()}"
        )
        print(
            f"designer_spikes     : "
            f"{self.designer_spikes}"
        )
        print(
            f"vocabulary_feedback_synapses: "
            f"{self.vocabulary_feedback_count()}"
        )
        print(
            f"total_reward        : "
            f"{self.total_reward:.1f}"
        )
        print(
            f"total_reuse         : "
            f"{self.total_reuse}"
        )
        print(
            f"total_create        : "
            f"{self.total_create}"
        )
        print(
            f"correct_reuse       : "
            f"{self.correct_reuse}"
        )
        print(
            f"correct_branch      : "
            f"{self.correct_branch}"
        )
        print(
            f"wrong_reuse         : "
            f"{self.wrong_reuse}"
        )
        print(
            f"wrong_branch        : "
            f"{self.wrong_branch}"
        )
        print(
            f"action_reuse        : "
            f"{self.action_reuse}"
        )
        print(
            f"action_branch       : "
            f"{self.action_branch}"
        )

        print()
        print("=== DESIGNER OUTPUT ===")
        print(
            f"reuse_output : "
            f"{self.cells[self.reuse_cell].potential:.4f}"
        )
        print(
            f"branch_output: "
            f"{self.cells[self.branch_cell].potential:.4f}"
        )

        print()
        print("=== DECISION MATRIX ===")
        print("reuse available:")
        print(
            f"    REUSE  : "
            f"{self.correct_reuse}"
        )
        print(
            f"    BRANCH : "
            f"{self.wrong_branch}"
        )
        print("reuse unavailable:")
        print(
            f"    REUSE  : "
            f"{self.wrong_reuse}"
        )
        print(
            f"    BRANCH : "
            f"{self.correct_branch}"
        )

        print()
        print("=== DESIGNER SYNAPSES ===")

        for s in self.synapses.values():
            a = self.cells[s.source]
            b = self.cells[s.target]

            if (
                a.kind == "designer"
                and b.kind == "designer"
            ):
                print(
                    f"{s.source} -> {s.target} "
                    f"{s.kind:7s} "
                    f"weight={s.weight:.4f} "
                    f"learning={s.learning:.4f}"
                )

    def print_topology(self):
        print()
        print("=== TOPOLOGY ===")

        for c in self.cells.values():
            parent = (
                "None"
                if c.parent is None
                else str(c.parent)
            )

            symbol = (
                "None"
                if c.symbol is None
                else c.symbol
            )

            print(
                f"{c.id:3d} "
                f"{c.kind:10s} "
                f"{symbol:>4s} "
                f"parent={parent:>4s} "
                f"in={len(c.incoming)} "
                f"out={len(c.outgoing)} "
                f"order={c.order} "
                f"pot={c.potential:.2f} "
                f"spikes={c.spikes} "
                f"inh={c.inhibition:.2f}"
            )

    def print_vocabulary_tree(self):
        print()
        print("=== VOCABULARY GRAPH ===")

        roots = sorted(
            (
                c
                for c in self.vocabulary_cells()
                if c.parent is None
            ),
            key=lambda c: c.id,
        )

        def walk(
            cid: int,
            prefix: str = "",
        ):
            c = self.cells[cid]

            print(
                f"{prefix}{c.symbol} [{c.id}]"
            )

            children = sorted(
                (
                    self.cells[x]
                    for x in c.outgoing
                    if self.cells[x].kind == "vocabulary"
                ),
                key=lambda x: x.order,
            )

            for child in children:
                walk(
                    child.id,
                    prefix + "  ",
                )

        for root in roots:
            walk(root.id)