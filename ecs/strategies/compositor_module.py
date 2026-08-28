"""Compositor as a workspace specialist — bids with composite strategies."""

from typing import Dict, List

from ecs.workspace.global_workspace import ModuleType, WorkspaceBid
from ecs.strategies.library import StrategyLibrary, StrategyCompositor


class CompositorModule:
    """Bids with composite strategies that compete against solo strategies."""

    def __init__(self, strategy_library: StrategyLibrary):
        self.library = strategy_library
        self.compositor = strategy_library.compositor
        self.name = "CompositorModule"

    def generate_bid(self, context: Dict) -> WorkspaceBid:
        """Generate bid proposing a composite strategy."""
        problem_features = context.get("problem_features", [])
        if not problem_features:
            return WorkspaceBid(
                module_type=ModuleType.STRATEGY,
                module_name=self.name,
                content="No features to compose for",
                payload=None,
                free_energy_reduction=0.0
            )

        applicable = self.library.find_applicable(problem_features)

        if len(applicable) < 2:
            return WorkspaceBid(
                module_type=ModuleType.STRATEGY,
                module_name=self.name,
                content="Not enough strategies to compose",
                payload=None,
                free_energy_reduction=0.0
            )

        best_composition = None
        best_score = 0.0

        for i, s1 in enumerate(applicable):
            for s2 in applicable[i + 1:]:
                relation = self.library.dag.check_composition(
                    s1["name"], s2["name"]
                )

                if relation and "Synergistic" in relation:
                    composition = self.compositor.compose(s1["name"], s2["name"])

                    score = (s1["success_rate"] + s2["success_rate"]) / 2
                    if composition.get("confidence", 0.5) > 0.7:
                        score += 0.1

                    if score > best_score:
                        best_score = score
                        best_composition = composition

        if not best_composition:
            return WorkspaceBid(
                module_type=ModuleType.STRATEGY,
                module_name=self.name,
                content="No synergistic compositions found",
                payload=None,
                free_energy_reduction=0.1
            )

        strategies = best_composition.get("strategies", [])
        content = f"COMPOSITE: {' + '.join(strategies)} (score={best_score:.2f})"

        return WorkspaceBid(
            module_type=ModuleType.STRATEGY,
            module_name=self.name,
            content=content,
            payload={
                "composition": best_composition,
                "is_composite": True
            },
            free_energy_reduction=0.7 * best_score,
            confidence=best_score
        )
