"""
Unified Synthesis Pipeline:
1. Strategy selection -> composition template
2. Z3 fills logical holes (conditions, bounds, expressions)
3. Neural fills creative holes (statements, code blocks)
4. Sandbox executes and verifies
5. CEGIS refines with counterexamples
6. Confidence computed from verification depth
"""

from typing import Dict, List, Optional, Any

from ecs.verification.sketch_synth import (
    SketchSynthesizer, SketchGenerator, HoleType, ProgramSketch
)
from ecs.verification.sandbox import SafeExecutor
from ecs.strategies.library import StrategyLibrary
from ecs.neural.interface import NeuralInterface


class UnifiedSynthesizer:
    """The complete synthesis pipeline."""

    def __init__(self, neural_interface: NeuralInterface,
                 strategy_library: StrategyLibrary):
        self.neural = neural_interface
        self.strategies = strategy_library
        self.sketch_gen = SketchGenerator()
        self.z3_synth = SketchSynthesizer()
        self.sandbox = SafeExecutor(timeout=5.0)
        self.synthesis_log: List[Dict] = []

    def synthesize(self, specification: str,
                   strategy_name: str,
                   problem_features: List[str],
                   retrieved_knowledge: Optional[List[str]] = None,
                   test_cases: Optional[List[Dict]] = None) -> Dict:
        """
        Full synthesis pipeline. Returns result dict with:
        success, code, confidence, verification_level,
        strategy_used, synthesis_method, test_results,
        counterexamples, iterations
        """
        # Phase 1: Generate sketch if strategy has a template
        sketch = None
        if strategy_name in self.sketch_gen.TEMPLATES:
            sketch = self.sketch_gen.generate_sketch(strategy_name, specification)

        # Phase 2: Try Z3-first synthesis (if sketch exists and all holes are Z3-solvable)
        if sketch and self._all_holes_z3_solvable(sketch):
            z3_result = self._try_z3_synthesis(sketch, specification, test_cases)
            if z3_result["success"]:
                return self._finalize_result(
                    z3_result["code"], specification, strategy_name,
                    "z3_only", "formal_proof_test_cases", test_cases
                )

        # Phase 3: Hybrid synthesis (Z3 + neural)
        if sketch:
            hybrid_result = self._try_hybrid_synthesis(
                sketch, specification, retrieved_knowledge, test_cases
            )
            if hybrid_result["success"]:
                result = self._finalize_result(
                    hybrid_result["code"], specification, strategy_name,
                    "hybrid", "unit_tests", test_cases
                )
                if not result["success"] and self.neural.is_available():
                    # CEGIS refinement
                    refined = self._cegis_refine(
                        hybrid_result["code"], specification, test_cases
                    )
                    if refined:
                        return self._finalize_result(
                            refined, specification, strategy_name,
                            "hybrid_cegis", "unit_tests", test_cases
                        )
                return result

        # Phase 4: Neural-only fallback
        if self.neural.is_available():
            neural_result = self._neural_fallback(
                specification, strategy_name, retrieved_knowledge
            )
            if neural_result:
                return self._finalize_result(
                    neural_result, specification, strategy_name,
                    "neural_only", "unit_tests", test_cases
                )

        return {
            "success": False,
            "code": None,
            "confidence": 0.0,
            "verification_level": "none",
            "strategy_used": strategy_name,
            "synthesis_method": "failed",
            "test_results": {"passed": False, "message": "All methods failed"},
            "counterexamples": [],
            "iterations": 0
        }

    def _all_holes_z3_solvable(self, sketch: ProgramSketch) -> bool:
        return all(
            h.hole_type in (HoleType.CONDITION, HoleType.LOOP_BOUND, HoleType.EXPRESSION)
            for h in sketch.holes
        )

    def _try_z3_synthesis(self, sketch: ProgramSketch, specification: str,
                          test_cases: Optional[List[Dict]]) -> Dict:
        """Try pure Z3 synthesis for all holes."""
        filled_code = sketch.template

        for hole in sketch.holes:
            solution = None

            if hole.hole_type == HoleType.CONDITION and test_cases:
                comparison_cases = [
                    tc for tc in test_cases
                    if "a" in tc and "b" in tc and "result" in tc
                ]
                if comparison_cases:
                    solution = self.z3_synth.synthesize_comparison(comparison_cases)

            elif hole.hole_type == HoleType.LOOP_BOUND:
                solution = str(self.z3_synth.synthesize_loop_bound(10, True))

            elif hole.hole_type == HoleType.EXPRESSION and test_cases:
                expr_cases = [
                    tc for tc in test_cases
                    if "input" in tc and "output" in tc
                ]
                if expr_cases:
                    solution = self.z3_synth.synthesize_linear_expression(expr_cases)

            if solution:
                filled_code = filled_code.replace(f"{{{hole.id}}}", solution)
            else:
                return {"success": False}

        return {"success": True, "code": filled_code}

    def _try_hybrid_synthesis(self, sketch: ProgramSketch, specification: str,
                              retrieved_knowledge: Optional[List[str]],
                              test_cases: Optional[List[Dict]]) -> Dict:
        """Z3 for logical holes, neural for statement holes."""
        filled_code = sketch.template

        for hole in sketch.holes:
            solution = None

            if hole.hole_type in (HoleType.CONDITION, HoleType.LOOP_BOUND,
                                  HoleType.EXPRESSION):
                if test_cases and hole.hole_type == HoleType.CONDITION:
                    comp_cases = [tc for tc in test_cases
                                  if "a" in tc and "b" in tc and "result" in tc]
                    if comp_cases:
                        solution = self.z3_synth.synthesize_comparison(comp_cases)
                elif hole.hole_type == HoleType.LOOP_BOUND:
                    solution = str(self.z3_synth.synthesize_loop_bound(10, True))

            if not solution and self.neural.is_available():
                solution = self._neural_fill_hole(hole, specification)

            if solution:
                filled_code = filled_code.replace(f"{{{hole.id}}}", solution)
            else:
                return {"success": False}

        return {"success": True, "code": filled_code}

    def _neural_fill_hole(self, hole, specification: str) -> Optional[str]:
        """Fill a single hole via neural interface."""
        prompt = (
            f"Fill in this code hole for: {specification}\n"
            f"Hole: {hole.id} (type: {hole.hole_type.value})\n"
            f"Context: {hole.context}\n\n"
            f"Provide ONLY the code expression (no explanation):"
        )

        result = self.neural._call_ollama(prompt, temperature=0.2, max_tokens=256)
        code = result["content"].strip()

        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1]) if len(lines) > 2 else lines[-1]

        return code if code else None

    def _neural_fallback(self, specification: str, strategy_name: str,
                         retrieved_knowledge: Optional[List[str]]) -> Optional[str]:
        """Pure neural generation."""
        response = self.neural.generate_code(
            specification=specification,
            strategy_context=strategy_name,
            retrieved_knowledge=retrieved_knowledge
        )
        return response.code

    def _cegis_refine(self, code: str, specification: str,
                      test_cases: Optional[List[Dict]]) -> Optional[str]:
        """Refine code using counterexamples from failed tests."""
        if not test_cases:
            return None

        _, _, details = self.sandbox.execute_tests(code, test_cases)
        failed = [d for d in details if not d.get("passed")]

        if not failed:
            return code

        for attempt in range(3):
            failure = failed[0]
            error_msg = (
                f"Input: {failure.get('inputs')}, "
                f"Expected: {failure.get('expected')}, "
                f"Got: {failure.get('actual')}"
            )

            refined = self.neural.refine_code(
                previous_code=code,
                error_message=error_msg,
                original_specification=specification
            )

            if refined.code:
                passed, _, new_details = self.sandbox.execute_tests(
                    refined.code, test_cases
                )
                if passed:
                    return refined.code
                code = refined.code
                failed = [d for d in new_details if not d.get("passed")]

        return None

    def _finalize_result(self, code: str, specification: str,
                         strategy_name: str, synthesis_method: str,
                         verification_level: str,
                         test_cases: Optional[List[Dict]]) -> Dict:
        """Verify code and compute final result."""
        if test_cases:
            passed, message, details = self.sandbox.execute_tests(code, test_cases)
        else:
            valid, err = self.sandbox.verify_syntax(code)
            passed = valid
            message = "Syntax valid" if valid else err
            details = []
            verification_level = "syntax_only" if valid else "none"

        confidence = self.z3_synth.compute_synthesis_confidence(
            code, verification_level, details
        )

        result = {
            "success": passed,
            "code": code,
            "confidence": confidence,
            "verification_level": verification_level,
            "strategy_used": strategy_name,
            "synthesis_method": synthesis_method,
            "test_results": {"passed": passed, "message": message},
            "counterexamples": [d for d in details if not d.get("passed", True)],
            "iterations": 1
        }

        self.synthesis_log.append(result)
        return result
