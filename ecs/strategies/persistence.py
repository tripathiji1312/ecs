"""Strategy library persistence: save/load learning state across sessions."""

import os
import time
import pickle
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class StrategyLibraryState:
    """Picklable state for strategy library."""
    usage_history: List[Dict]
    learned_success_rates: Dict[str, float]
    co_occurrence_wins: Dict[str, int]
    co_occurrence_success: Dict[str, int]
    retired_strategies: List[str]
    custom_strategies: List[Dict]
    saved_at: float


class StrategyPersistence:
    """Handles saving/loading strategy library state."""

    def __init__(self, filepath: str = "ecs/data/strategy_state.pkl"):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def save(self, library) -> None:
        """Save strategy library learning state."""
        state = StrategyLibraryState(
            usage_history=library.usage_history[-500:],
            learned_success_rates={
                name: s.success_rate
                for name, s in library.strategies.items()
            },
            co_occurrence_wins=library.co_occurrence_wins,
            co_occurrence_success=library.co_occurrence_success,
            retired_strategies=library.retired_strategies,
            custom_strategies=library.custom_strategies_data,
            saved_at=time.time()
        )

        with open(self.filepath, "wb") as f:
            pickle.dump(state, f)

    def load(self, library) -> bool:
        """Load strategy library state. Returns True if loaded."""
        if not os.path.exists(self.filepath):
            return False

        with open(self.filepath, "rb") as f:
            state = pickle.load(f)

        library.usage_history = state.usage_history
        library.co_occurrence_wins = state.co_occurrence_wins
        library.co_occurrence_success = state.co_occurrence_success
        library.retired_strategies = state.retired_strategies
        library.custom_strategies_data = state.custom_strategies

        for name, rate in state.learned_success_rates.items():
            if name in library.strategies:
                library.strategies[name].success_rate = rate

        return True
