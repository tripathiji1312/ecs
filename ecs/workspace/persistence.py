"""Self-model persistence: save/load workspace state across sessions."""

import os
import time
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

from ecs.workspace.global_workspace import GlobalWorkspace, ModuleType


@dataclass
class SerializableSelfModel:
    """Picklable version of self-model + workspace state."""
    strengths: Dict[str, float]
    weaknesses: Dict[str, float]
    current_goal: Optional[str]
    active_strategy: Optional[str]
    confidence_level: float
    uncertainty_level: float
    free_energy_level: float
    tasks_attempted: int
    tasks_succeeded: int
    stuck_counter: int
    patterns_noticed: List[str]
    attention_weights: Dict[str, float]
    saved_at: float


class SelfModelPersistence:
    """Handles saving/loading self-model state."""

    def __init__(self, filepath: str = "ecs/data/self_model.pkl"):
        self.filepath = filepath

    def save(self, workspace: GlobalWorkspace) -> None:
        sm = workspace.self_model
        state = SerializableSelfModel(
            strengths=dict(sm.strengths),
            weaknesses=dict(sm.weaknesses),
            current_goal=sm.current_goal,
            active_strategy=sm.active_strategy,
            confidence_level=sm.confidence_level,
            uncertainty_level=sm.uncertainty_level,
            free_energy_level=sm.free_energy_level,
            tasks_attempted=sm.tasks_attempted,
            tasks_succeeded=sm.tasks_succeeded,
            stuck_counter=sm.stuck_counter,
            patterns_noticed=list(sm.patterns_noticed),
            attention_weights={
                k.value: v for k, v in workspace.attention_weights.items()
            },
            saved_at=time.time()
        )

        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, "wb") as f:
            pickle.dump(state, f)

    def load(self, workspace: GlobalWorkspace) -> bool:
        """Load self-model state. Returns True if loaded."""
        if not os.path.exists(self.filepath):
            return False

        with open(self.filepath, "rb") as f:
            state = pickle.load(f)

        sm = workspace.self_model
        sm.strengths = state.strengths
        sm.weaknesses = state.weaknesses
        sm.current_goal = state.current_goal
        sm.active_strategy = state.active_strategy
        sm.confidence_level = state.confidence_level
        sm.uncertainty_level = state.uncertainty_level
        sm.free_energy_level = state.free_energy_level
        sm.tasks_attempted = state.tasks_attempted
        sm.tasks_succeeded = state.tasks_succeeded
        sm.stuck_counter = state.stuck_counter
        sm.patterns_noticed = state.patterns_noticed

        for type_str, weight in state.attention_weights.items():
            try:
                mtype = ModuleType(type_str)
                workspace.attention_weights[mtype] = weight
            except ValueError:
                continue

        return True
