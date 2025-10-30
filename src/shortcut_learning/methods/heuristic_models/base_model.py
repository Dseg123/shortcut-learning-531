"""Base policy interface for improvisational approaches."""

from __future__ import annotations

from abc import ABC, abstractmethod
from shortcut_learning.methods.graph_utils import PlanningGraphNode


class HeuristicModel(ABC):
    """Base class for heuristic models."""

    def __init__(self, hyper_params) -> None:
        """Initialize policy with environment."""
        self._hyper_params = hyper_params

    @abstractmethod
    def get_distance(self, shortcut_pair: tuple[PlanningGraphNode, PlanningGraphNode]) -> float:
        """ Given a shortcut pair, return the predicted path distance between the two nodes."""

    @abstractmethod
    def train(
        self,
        train_data: list[tuple[PlanningGraphNode, PlanningGraphNode, float]] | None,
    ) -> None:
        """Train the policy if needed.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Save policy to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load policy from disk."""
