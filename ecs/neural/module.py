"""
Neural interface as a workspace specialist module.
Bids with language understanding, code generation proposals,
and problem analysis.
"""

import ast
from typing import Dict, List, Optional

from ecs.workspace.global_workspace import ModuleType, WorkspaceBid
from ecs.neural.interface import NeuralInterface


class NeuralModule:
    """Neural interface as workspace participant."""

    def __init__(self, neural_interface: NeuralInterface):
        self.interface = neural_interface
        self.name = "NeuralModule"
        self.last_generation: Optional[Dict] = None
        self.generation_count: int = 0

    def generate_bid(self, context: Dict) -> WorkspaceBid:
        """Generate bid based on neural analysis of the problem."""
        if not self.interface.is_available():
            return WorkspaceBid(
                module_type=ModuleType.NEURAL,
                module_name=self.name,
                content="Neural interface offline",
                payload=None,
                free_energy_reduction=0.0
            )

        problem = context.get("current_problem", "")
        if not problem:
            return WorkspaceBid(
                module_type=ModuleType.NEURAL,
                module_name=self.name,
                content="No problem to analyze",
                payload=None,
                free_energy_reduction=0.0
            )

        understanding = self.interface.understand_problem(problem)
        confidence = 0.7 if understanding["type"] != "other" else 0.4

        content = (
            f"Problem analysis: type={understanding['type']}, "
            f"difficulty={understanding['difficulty']}, "
            f"features={understanding['features']}"
        )

        return WorkspaceBid(
            module_type=ModuleType.NEURAL,
            module_name=self.name,
            content=content,
            payload={
                "understanding": understanding,
                "features": understanding["features"],
                "sub_problems": understanding["sub_problems"]
            },
            free_energy_reduction=0.6,
            confidence=confidence
        )

    def generate_code_bid(self, context: Dict) -> WorkspaceBid:
        """Generate bid with code proposal."""
        problem = context.get("current_problem", "")
        strategy_context = context.get("strategy_context")
        retrieved_knowledge = context.get("retrieved_knowledge", [])

        if not problem:
            return WorkspaceBid(
                module_type=ModuleType.NEURAL,
                module_name=self.name,
                content="No problem to solve",
                payload=None,
                free_energy_reduction=0.0
            )

        response = self.interface.generate_code(
            specification=problem,
            strategy_context=strategy_context,
            retrieved_knowledge=retrieved_knowledge
        )

        if not response.code:
            return WorkspaceBid(
                module_type=ModuleType.NEURAL,
                module_name=self.name,
                content="Code generation failed",
                payload=None,
                free_energy_reduction=0.1
            )

        syntax_valid = self._validate_syntax(response.code)
        confidence = response.confidence if syntax_valid else 0.2

        self.last_generation = {
            "code": response.code,
            "syntax_valid": syntax_valid,
            "latency_ms": response.latency_ms
        }
        self.generation_count += 1

        return WorkspaceBid(
            module_type=ModuleType.NEURAL,
            module_name=self.name,
            content=f"Code proposal ({'valid' if syntax_valid else 'SYNTAX ERROR'})",
            payload={
                "code": response.code,
                "syntax_valid": syntax_valid,
                "latency_ms": response.latency_ms
            },
            free_energy_reduction=0.5 if syntax_valid else 0.2,
            confidence=confidence
        )

    def calibrate_confidence(self, test_results: Dict) -> float:
        """Calibrate confidence based on verification results."""
        base_confidence = 0.5

        if test_results.get("syntax_valid"):
            base_confidence += 0.1

        if test_results.get("tests_passed"):
            base_confidence += 0.3

        test_count = test_results.get("test_count", 0)
        if test_count > 3:
            base_confidence += 0.1

        return min(1.0, base_confidence)

    def _validate_syntax(self, code: str) -> bool:
        """Validate Python syntax using ast.parse."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
