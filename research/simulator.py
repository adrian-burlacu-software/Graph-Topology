from dataclasses import dataclass, field
from typing import Optional
import random


@dataclass
class Synapse:
    source: int
    target: int
    weight: float = 1.0
    inhibitory: bool = False


@dataclass
class Cell:
    id: int
    parent: Optional[int] = None
    symbol: Optional[str] = None

    # Internal developmental state
    activation: float = 0.0
    order_state: float = 0.0
    inhibition: float = 0.0

    children: list[int] = field(default_factory=list)
    incoming: list[int] = field(default_factory=list)
    outgoing: list[int] = field(default_factory=list)

    activity_total: float = 0.0


class DevelopmentalNetwork:

    def __init__(self, genome, seed=42):

        self.genome = genome
        self.rng = random.Random(seed)

        self.cells: dict[int, Cell] = {}
        self.synapses: dict[int, Synapse] = {}

        self.next_cell_id = 0

        # --------------------------------------------------
        # Everything starts with ONE CELL.
        # --------------------------------------------------

        self.root = self._create_cell()

        # The root initially acts as the designer.
        self.designer_cells = {self.root}

        # Vocabulary network starts empty.
        self.vocabulary_cells = set()

    # ======================================================
    # CELL CREATION
    # ======================================================

    def _create_cell(self, parent=None, symbol=None):

        cell = Cell(
            id=self.next_cell_id,
            parent=parent,
            symbol=symbol
        )

        self.cells[cell.id] = cell
        self.next_cell_id += 1

        if parent is not None:
            self.cells[parent].children.append(cell.id)

        return cell.id

    # ======================================================
    # SYNAPSES
    # ======================================================

    def _connect(
        self,
        source,
        target,
        weight=1.0,
        inhibitory=False
    ):

        # Don't create duplicate connections.
        for sid in self.cells[source].outgoing:

            synapse = self.synapses[sid]

            if synapse.target == target:
                return sid

        sid = len(self.synapses)

        self.synapses[sid] = Synapse(
            source=source,
            target=target,
            weight=weight,
            inhibitory=inhibitory
        )

        self.cells[source].outgoing.append(sid)
        self.cells[target].incoming.append(sid)

        return sid

    # ======================================================
    # DESIGNER
    # ======================================================

    def _designer_decision(self, symbol, position):

        """
        The designer decides what should happen.

        This is deliberately primitive in Experiment 1.

        Importantly, the vocabulary network is NOT searched
        globally for a matching symbol.

        The designer receives the current symbol and produces
        a local construction decision.
        """

        # First symbol of a sequence:
        if position == 0:

            return {
                "action": "CREATE_ROOT",
                "symbol": symbol
            }

        # Later symbols:
        return {
            "action": "EXTEND",
            "symbol": symbol
        }

    # ======================================================
    # VOCABULARY CONSTRUCTION
    # ======================================================

    def _create_vocabulary_cell(self, parent, symbol):

        cell_id = self._create_cell(
            parent=parent,
            symbol=symbol
        )

        self.vocabulary_cells.add(cell_id)

        self._connect(parent, cell_id)

        return cell_id

    # ======================================================
    # PROCESS ONE WORD
    # ======================================================

    def process_word(self, word):

        previous = None

        for position, symbol in enumerate(word):

            decision = self._designer_decision(
                symbol,
                position
            )

            if decision["action"] == "CREATE_ROOT":

                current = self._create_vocabulary_cell(
                    parent=self.root,
                    symbol=symbol
                )

            elif decision["action"] == "EXTEND":

                current = self._create_vocabulary_cell(
                    parent=previous,
                    symbol=symbol
                )

            else:

                raise RuntimeError(
                    f"Unknown designer action: "
                    f"{decision['action']}"
                )

            self.cells[current].activation += 1.0
            self.cells[current].activity_total += 1.0
            self.cells[current].order_state = position

            previous = current

    # ======================================================
    # TRAINING
    # ======================================================

    def train(self, vocabulary):

        for word in vocabulary:

            self.process_word(word)

            # Small decay.
            for cell in self.cells.values():

                cell.activation *= 0.95
                cell.inhibition *= 0.95

    # ======================================================
    # STATISTICS
    # ======================================================

    def stats(self):

        reusable_cells = sum(
            1
            for cell_id in self.vocabulary_cells
            if len(self.cells[cell_id].incoming) > 1
        )

        designer_count = len(self.designer_cells)

        vocabulary_count = len(self.vocabulary_cells)

        return {
            "cells": len(self.cells),
            "designer_cells": designer_count,
            "vocabulary_cells": vocabulary_count,
            "synapses": len(self.synapses),
            "reusable_cells": reusable_cells
        }

    # ======================================================
    # TOPOLOGY
    # ======================================================

    def topology(self):

        result = []

        for cell in self.cells.values():

            role = "designer"

            if cell.id in self.vocabulary_cells:
                role = "vocabulary"

            result.append({
                "id": cell.id,
                "role": role,
                "symbol": cell.symbol,
                "parent": cell.parent,
                "children": cell.children,
                "incoming": len(cell.incoming),
                "outgoing": len(cell.outgoing),
                "order": cell.order_state
            })

        return result