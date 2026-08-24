from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# CONFIG
# ============================================================

CREATE = "CREATE"
REUSE = "REUSE"
BRANCH = "BRANCH"
INHIBIT = "INHIBIT"
EXCITE = "EXCITE"


@dataclass
class Config:
    # Keep this small while developing.
    designer_learning_rate: float = 0.05
    vocabulary_learning_rate: float = 0.05

    spike_threshold: float = 1.0
    leak: float = 0.90

    excite_weight: float = 1.0
    inhibit_weight: float = 0.6

    # Reward shaping
    reward_correct_reuse: float = 1.0
    reward_correct_branch: float = 1.0
    reward_wrong_reuse: float = -1.0
    reward_wrong_branch: float = -0.25

    # Penalize unnecessary growth.
    branch_cost: float = -0.25

    # Prevent runaway designer growth.
    max_designer_cells: int = 32


# ============================================================
# DATA STRUCTURES
# ============================================================

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

    # Last design signal emitted by this cell.
    signal: Optional[str] = None


# ============================================================
# NETWORK
# ============================================================

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

        # The one original cell.
        self.designer_root = self.create_cell("designer")

        # Build a tiny designer circuit.
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

        # Reciprocal inhibition gives the designer a
        # competition mechanism.
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

        # Statistics for designer decisions.
        self.designer_signals = {
            CREATE: 0.0,
            REUSE: 0.0,
            BRANCH: 0.0,
            INHIBIT: 0.0,
        }

    # ========================================================
    # CELL / SYNAPSE CREATION
    # ========================================================

    def create_cell(
        self,
        kind: str,
        symbol: Optional[str] = None,
        parent: Optional[int] = None,
        order: int = 0,
    ) -> int:

        cell_id = self.next_cell_id
        self.next_cell_id += 1

        cell = Cell(
            id=cell_id,
            kind=kind,
            symbol=symbol,
            parent=parent,
            order=order,
        )

        self.cells[cell_id] = cell

        return cell_id

    def connect(
        self,
        source: int,
        target: int,
        kind: str = EXCITE,
        weight: float = 1.0,
    ) -> Synapse:

        key = (source, target)

        existing = self.synapses.get(key)

        if existing is not None:
            return existing

        synapse = Synapse(
            source=source,
            target=target,
            kind=kind,
            weight=weight,
        )

        self.synapses[key] = synapse

        self.cells[source].outgoing.append(target)
        self.cells[target].incoming.append(source)

        return synapse

    # ========================================================
    # VOCABULARY GRAPH
    # ========================================================

    def create_vocabulary_cell(
        self,
        symbol: str,
        parent: Optional[int],
        order: int,
    ) -> int:

        cell_id = self.create_cell(
            kind="vocabulary",
            symbol=symbol,
            parent=parent,
            order=order,
        )

        # IMPORTANT:
        #
        # parent metadata is NOT enough.
        #
        # The parent relationship is an actual synapse.
        if parent is not None:
            self.connect(
                parent,
                cell_id,
                EXCITE,
                self.config.excite_weight,
            )

        return cell_id

    def find_child(
        self,
        parent_id: Optional[int],
        symbol: str,
    ) -> Optional[int]:

        if parent_id is None:
            return None

        parent = self.cells.get(parent_id)

        if parent is None:
            return None

        for child_id in parent.outgoing:
            child = self.cells[child_id]

            if child.kind != "vocabulary":
                continue

            if child.symbol == symbol:
                return child_id

        return None

    def find_children(
        self,
        parent_id: Optional[int],
    ) -> list[int]:

        if parent_id is None:
            return []

        parent = self.cells[parent_id]

        return [
            child_id
            for child_id in parent.outgoing
            if self.cells[child_id].kind == "vocabulary"
        ]

    def reusable_symbols(
        self,
        parent_id: Optional[int],
    ) -> list[str]:

        return [
            self.cells[cell_id].symbol
            for cell_id in self.find_children(parent_id)
        ]

    # ========================================================
    # DESIGNER INPUT
    # ========================================================

    def designer_features(
        self,
        current_id: Optional[int],
        symbol: str,
    ) -> dict[str, float]:

        children = self.find_children(current_id)

        reusable = self.find_child(current_id, symbol)

        depth = 0

        if current_id is not None:
            depth = self.cells[current_id].order

        return {
            # Main signal:
            # 1 means the desired symbol already exists.
            "reuse_available": 1.0 if reusable is not None else 0.0,

            # Whether there are already branches here.
            "has_children": 1.0 if children else 0.0,

            # How many choices already exist.
            "branch_count": float(len(children)),

            # Current depth.
            "depth": float(depth),

            # Number of vocabulary cells.
            "vocabulary_size": float(
                sum(
                    1
                    for cell in self.cells.values()
                    if cell.kind == "vocabulary"
                )
            ),
        }

    # ========================================================
    # DESIGNER
    # ========================================================

    def designer_signal(
        self,
        current_id: Optional[int],
        symbol: str,
    ) -> str:

        features = self.designer_features(
            current_id,
            symbol,
        )

        reuse_available = features["reuse_available"]

        # ----------------------------------------------------
        # Initial learning bias
        #
        # The designer must understand the structural
        # difference between:
        #
        #   matching existing edge
        #
        # and
        #
        #   missing edge.
        #
        # We intentionally keep this simple.
        # Plasticity will modify the decision over time.
        # ----------------------------------------------------

        reuse_drive = 0.0
        branch_drive = 0.0

        if reuse_available:
            reuse_drive += 2.0
        else:
            branch_drive += 2.0

        # Existing branching structure makes reuse attractive
        # because we want the network to exploit existing paths.
        if features["has_children"]:
            reuse_drive += 0.25

        # Too many branches slightly discourage more branching.
        branch_drive -= (
            features["branch_count"] * 0.10
        )

        # Read designer cell potentials.
        reuse_drive += self.cells[self.reuse_cell].potential
        branch_drive += self.cells[self.branch_cell].potential

        if reuse_drive >= branch_drive:
            signal = REUSE
        else:
            signal = BRANCH

        self.designer_signals[signal] += 1.0

        return signal

    # ========================================================
    # DESIGNER PLASTICITY
    # ========================================================

    def learn_designer(
        self,
        action: str,
        correct_action: str,
        reward: float,
    ):

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

        # -----------------------------------------------
        # Plasticity
        # -----------------------------------------------

        if action == REUSE:
            target = self.reuse_cell
            opposite = self.branch_cell
        else:
            target = self.branch_cell
            opposite = self.reuse_cell

        delta = (
            self.config.designer_learning_rate
            * reward
        )

        self.cells[target].potential += delta
        self.cells[opposite].potential -= delta * 0.5

        # Clamp potentials so they don't explode.
        self.cells[target].potential = max(
            -5.0,
            min(5.0, self.cells[target].potential),
        )

        self.cells[opposite].potential = max(
            -5.0,
            min(5.0, self.cells[opposite].potential),
        )

        # Plasticity also changes the relevant synapses.
        for source in (self.reuse_cell, self.branch_cell):
            syn = self.synapses.get(
                (self.designer_root, source)
            )

            if syn is None:
                continue

            if source == target:
                syn.weight += delta * 0.1
                syn.learning += delta * 0.1
            else:
                syn.weight -= delta * 0.05

            syn.weight = max(
                0.05,
                min(2.0, syn.weight),
            )

    # ========================================================
    # SPIKE DESIGNER
    # ========================================================

    def spike_designer(
        self,
        current_id: Optional[int],
        symbol: str,
    ):

        features = self.designer_features(
            current_id,
            symbol,
        )

        # Feed the structural state into the designer root.
        input_drive = 1.0

        if features["reuse_available"]:
            input_drive += 1.0

        self.cells[self.designer_root].potential += input_drive

        if (
            self.cells[self.designer_root].potential
            >= self.config.spike_threshold
        ):
            self.cells[self.designer_root].potential = 0.0
            self.cells[self.designer_root].spikes += 1
            self.designer_spikes += 1

            # Competing populations.
            self.cells[self.reuse_cell].potential += (
                self.synapses[
                    (self.designer_root, self.reuse_cell)
                ].weight
            )

            self.cells[self.branch_cell].potential += (
                self.synapses[
                    (self.designer_root, self.branch_cell)
                ].weight
            )

        # Apply reciprocal inhibition.
        reuse_potential = self.cells[
            self.reuse_cell
        ].potential

        branch_potential = self.cells[
            self.branch_cell
        ].potential

        if branch_potential > self.config.spike_threshold:
            self.cells[
                self.reuse_cell
            ].potential -= self.config.inhibit_weight

            self.cells[
                self.branch_cell
            ].spikes += 1

            self.designer_spikes += 1

        if reuse_potential > self.config.spike_threshold:
            self.cells[
                self.branch_cell
            ].potential -= self.config.inhibit_weight

            self.cells[
                self.reuse_cell
            ].spikes += 1

            self.designer_spikes += 1

        # Leak.
        self.cells[self.reuse_cell].potential *= (
            self.config.leak
        )

        self.cells[self.branch_cell].potential *= (
            self.config.leak
        )

    # ========================================================
    # WORD PROCESSING
    # ========================================================

    def process_word(
        self,
        word: str,
    ) -> dict:

        current_id: Optional[int] = None

        created = 0
        reused = 0
        branched = 0

        for order, symbol in enumerate(word):

            # ------------------------------------------------
            # First character
            # ------------------------------------------------
            if current_id is None:

                # At the root there is no reusable child unless
                # one was previously created.
                existing = self.find_child(
                    None,
                    symbol,
                )

                # Root-level vocabulary cells are tracked
                # separately because there is no vocabulary root.
                existing = self.find_root_symbol(symbol)

                correct_action = (
                    REUSE
                    if existing is not None
                    else BRANCH
                )

                self.spike_designer(
                    None,
                    symbol,
                )

                action = self.designer_signal(
                    None,
                    symbol,
                )

                if existing is not None and action == REUSE:
                    current_id = existing
                    reused += 1
                    self.total_reuse += 1

                    reward = (
                        self.config.reward_correct_reuse
                    )

                elif existing is None and action == BRANCH:
                    current_id = self.create_vocabulary_cell(
                        symbol=symbol,
                        parent=None,
                        order=order,
                    )

                    created += 1
                    branched += 1
                    self.total_create += 1

                    reward = (
                        self.config.reward_correct_branch
                        + self.config.branch_cost
                    )

                elif existing is not None and action == BRANCH:
                    # Branching at root means a new root path.
                    current_id = self.create_vocabulary_cell(
                        symbol=symbol,
                        parent=None,
                        order=order,
                    )

                    created += 1
                    branched += 1
                    self.total_create += 1

                    reward = (
                        self.config.reward_wrong_branch
                        + self.config.branch_cost
                    )

                else:
                    # REUSE was selected but unavailable.
                    # Correct response is still to create.
                    current_id = self.create_vocabulary_cell(
                        symbol=symbol,
                        parent=None,
                        order=order,
                    )

                    created += 1
                    self.total_create += 1

                    reward = self.config.reward_wrong_reuse

                self.learn_designer(
                    action,
                    correct_action,
                    reward,
                )

                continue

            # ------------------------------------------------
            # Subsequent characters
            # ------------------------------------------------

            existing = self.find_child(
                current_id,
                symbol,
            )

            correct_action = (
                REUSE
                if existing is not None
                else BRANCH
            )

            self.spike_designer(
                current_id,
                symbol,
            )

            action = self.designer_signal(
                current_id,
                symbol,
            )

            # ------------------------------------------------
            # Correct reuse
            # ------------------------------------------------
            if existing is not None and action == REUSE:

                current_id = existing

                reused += 1
                self.total_reuse += 1

                reward = (
                    self.config.reward_correct_reuse
                )

            # ------------------------------------------------
            # Correct branch
            # ------------------------------------------------
            elif existing is None and action == BRANCH:

                current_id = self.create_vocabulary_cell(
                    symbol=symbol,
                    parent=current_id,
                    order=order,
                )

                created += 1
                branched += 1
                self.total_create += 1

                reward = (
                    self.config.reward_correct_branch
                    + self.config.branch_cost
                )

            # ------------------------------------------------
            # Wrong branch when reuse existed
            # ------------------------------------------------
            elif existing is not None and action == BRANCH:

                # IMPORTANT:
                #
                # We DO NOT create a duplicate.
                #
                # The network should remain structurally
                # efficient even if the designer makes a
                # bad decision.
                current_id = existing

                branched += 1

                reward = (
                    self.config.reward_wrong_branch
                    + self.config.branch_cost
                )

            # ------------------------------------------------
            # Wrong reuse when unavailable
            # ------------------------------------------------
            else:

                # Safety fallback:
                # create the missing edge.
                current_id = self.create_vocabulary_cell(
                    symbol=symbol,
                    parent=current_id,
                    order=order,
                )

                created += 1
                self.total_create += 1

                reward = (
                    self.config.reward_wrong_reuse
                )

            self.learn_designer(
                action,
                correct_action,
                reward,
            )

        return {
            "word": word,
            "created": created,
            "reused": reused,
            "branched": branched,
        }

    # ========================================================
    # ROOT LOOKUP
    # ========================================================

    def find_root_symbol(
        self,
        symbol: str,
    ) -> Optional[int]:

        for cell in self.cells.values():

            if cell.kind != "vocabulary":
                continue

            if cell.parent is not None:
                continue

            if cell.symbol == symbol:
                return cell.id

        return None

    # ========================================================
    # TRAINING
    # ========================================================

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

            epoch_reward_before = self.total_reward
            epoch_reuse_before = self.total_reuse
            epoch_create_before = self.total_create

            for word in words:
                self.process_word(word)

            epoch_reward = (
                self.total_reward
                - epoch_reward_before
            )

            epoch_reuse = (
                self.total_reuse
                - epoch_reuse_before
            )

            epoch_create = (
                self.total_create
                - epoch_create_before
            )

            print(
                f"epoch={epoch:3d} "
                f"cells={len(self.cells):3d} "
                f"reuse={epoch_reuse:3d} "
                f"create={epoch_create:3d} "
                f"reward={epoch_reward:8.2f}"
            )

    # ========================================================
    # STATISTICS
    # ========================================================

    def vocabulary_cells(self) -> list[Cell]:

        return [
            cell
            for cell in self.cells.values()
            if cell.kind == "vocabulary"
        ]

    def designer_cells(self) -> list[Cell]:

        return [
            cell
            for cell in self.cells.values()
            if cell.kind == "designer"
        ]

    def reusable_cells_count(self) -> int:

        # A reusable cell is one which participates in an
        # actual vocabulary synapse.
        #
        # This deliberately does NOT mean "all vocabulary
        # cells".
        count = 0

        for cell in self.vocabulary_cells():

            if cell.incoming:
                count += 1

        return count

    # ========================================================
    # PRINTING
    # ========================================================

    def print_summary(self):

        vocabulary = self.vocabulary_cells()
        designer = self.designer_cells()

        feedback_synapses = 0

        for synapse in self.synapses.values():

            if (
                self.cells[synapse.source].kind
                == "vocabulary"
                and self.cells[synapse.target].kind
                == "designer"
            ):
                feedback_synapses += 1

        print()
        print("=== FINAL ===")
        print(
            f"cells               : {len(self.cells)}"
        )
        print(
            f"designer_cells      : {len(designer)}"
        )
        print(
            f"vocabulary_cells    : {len(vocabulary)}"
        )
        print(
            f"synapses            : {len(self.synapses)}"
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
            f"{feedback_synapses}"
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

        for synapse in self.synapses.values():

            source = self.cells[synapse.source]
            target = self.cells[synapse.target]

            if (
                source.kind == "designer"
                and target.kind == "designer"
            ):

                print(
                    f"{synapse.source} -> "
                    f"{synapse.target} "
                    f"{synapse.kind:7s} "
                    f"weight={synapse.weight:.4f} "
                    f"learning={synapse.learning:.4f}"
                )

    def print_topology(self):

        print()
        print("=== TOPOLOGY ===")

        for cell in self.cells.values():

            parent = (
                "None"
                if cell.parent is None
                else str(cell.parent)
            )

            symbol = (
                "None"
                if cell.symbol is None
                else cell.symbol
            )

            print(
                f"{cell.id:3d} "
                f"{cell.kind:10s} "
                f"{symbol:>4s} "
                f"parent={parent:>4s} "
                f"in={len(cell.incoming)} "
                f"out={len(cell.outgoing)} "
                f"order={cell.order} "
                f"pot={cell.potential:.2f} "
                f"spikes={cell.spikes} "
                f"inh={cell.inhibition:.2f}"
            )

    # ========================================================
    # VOCABULARY TREE
    # ========================================================

    def print_vocabulary_tree(self):

        print()
        print("=== VOCABULARY GRAPH ===")

        roots = [
            cell
            for cell in self.vocabulary_cells()
            if cell.parent is None
        ]

        roots.sort(key=lambda c: c.id)

        def walk(cell_id: int, prefix: str = ""):

            cell = self.cells[cell_id]

            print(
                f"{prefix}{cell.symbol}"
                f" [{cell.id}]"
            )

            children = [
                self.cells[child_id]
                for child_id in cell.outgoing
                if self.cells[child_id].kind == "vocabulary"
            ]

            children.sort(
                key=lambda child: child.order
            )

            for child in children:
                walk(
                    child.id,
                    prefix + "  ",
                )

        for root in roots:
            walk(root.id)