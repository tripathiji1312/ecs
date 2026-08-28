"""Unified session state persistence — atomic save/load of all components."""

import os
import json
import shutil
import tempfile

from ecs.workspace.persistence import SelfModelPersistence
from ecs.strategies.persistence import StrategyPersistence


class SessionPersistence:
    """Unified session state persistence."""

    def __init__(self, base_dir: str = "ecs/data/session"):
        self.base_dir = base_dir

    def save(self, orchestrator) -> bool:
        """Save all state atomically."""
        temp_dir = tempfile.mkdtemp()

        try:
            orchestrator.memory.save(f"{temp_dir}/memory.pkl")

            workspace_p = SelfModelPersistence(f"{temp_dir}/workspace.pkl")
            workspace_p.save(orchestrator.workspace)

            strategy_p = StrategyPersistence(f"{temp_dir}/strategies.pkl")
            strategy_p.save(orchestrator.strategies)

            with open(f"{temp_dir}/stats.json", "w") as f:
                json.dump(orchestrator.session_stats, f, indent=2, default=str)

            if os.path.exists(self.base_dir):
                shutil.rmtree(self.base_dir)
            os.rename(temp_dir, self.base_dir)

            return True

        except Exception:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return False

    def load(self, orchestrator) -> bool:
        """Load all state. Returns True if successful."""
        if not os.path.exists(self.base_dir):
            return False

        success = True

        try:
            orchestrator.memory.load(f"{self.base_dir}/memory.pkl")
        except Exception:
            success = False

        try:
            workspace_p = SelfModelPersistence(f"{self.base_dir}/workspace.pkl")
            workspace_p.load(orchestrator.workspace)
        except Exception:
            success = False

        try:
            strategy_p = StrategyPersistence(f"{self.base_dir}/strategies.pkl")
            strategy_p.load(orchestrator.strategies)
        except Exception:
            success = False

        try:
            with open(f"{self.base_dir}/stats.json", "r") as f:
                orchestrator.session_stats = json.load(f)
        except Exception:
            pass

        return success
