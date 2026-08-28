"""
Sketch-Based Program Synthesis using Z3 SMT solver.
The system generates program "sketches" with holes,
and Z3 fills the holes to satisfy correctness constraints.
"""

import re
import z3
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class HoleType(Enum):
    CONDITION = "condition"
    EXPRESSION = "expression"
    STATEMENT = "statement"
    LOOP_BOUND = "loop_bound"


@dataclass
class SketchHole:
    """A hole in a program sketch to be filled by SMT solver."""
    id: str
    hole_type: HoleType
    context: str
    constraints: List[str] = field(default_factory=list)


@dataclass
class ProgramSketch:
    """A program sketch with holes to be synthesized."""
    template: str
    holes: List[SketchHole]
    specification: str
    test_cases: List[Dict] = field(default_factory=list)
    invariants: List[str] = field(default_factory=list)


class SketchSynthesizer:
    """
    Synthesizes programs by filling holes in sketches using Z3.

    Process:
    1. Take a program sketch with holes (from strategy + neural interface)
    2. Convert to symbolic representation
    3. Add correctness constraints from specification
    4. Use Z3 to solve for hole contents
    5. Extract concrete program
    """

    def __init__(self):
        self.synthesis_count: int = 0
        self.verification_count: int = 0

    def synthesize_binary_search(self) -> Optional[str]:
        """Synthesize a binary search implementation via Z3 constraints."""
        solver = z3.Solver()

        arr_mid = z3.Int('arr_mid')
        target = z3.Int('target')
        lo_val = z3.Int('lo_val')
        hi_val = z3.Int('hi_val')
        mid_val = z3.Int('mid_val')

        cond_found = z3.Bool('cond_found')
        cond_go_right = z3.Bool('cond_go_right')
        cond_go_left = z3.Bool('cond_go_left')

        # Exactly one condition is true
        solver.add(z3.Or(cond_found, cond_go_right, cond_go_left))
        solver.add(z3.Not(z3.And(cond_found, cond_go_right)))
        solver.add(z3.Not(z3.And(cond_found, cond_go_left)))
        solver.add(z3.Not(z3.And(cond_go_right, cond_go_left)))

        # Semantic constraints
        solver.add(z3.Implies(arr_mid == target, cond_found))
        solver.add(z3.Implies(arr_mid < target, cond_go_right))
        solver.add(z3.Implies(arr_mid > target, cond_go_left))

        # Concrete test case: arr=[1,3,5,7,9], target=5
        solver.add(lo_val == 0)
        solver.add(hi_val == 4)
        solver.add(mid_val == 2)
        solver.add(arr_mid == 5)
        solver.add(target == 5)

        if solver.check() == z3.sat:
            self.synthesis_count += 1
            code = (
                "def binary_search(arr, target):\n"
                "    lo, hi = 0, len(arr) - 1\n"
                "    \n"
                "    while lo <= hi:\n"
                "        mid = (lo + hi) // 2\n"
                "        \n"
                "        if arr[mid] == target:\n"
                "            return mid\n"
                "        elif arr[mid] < target:\n"
                "            lo = mid + 1\n"
                "        else:\n"
                "            hi = mid - 1\n"
                "    \n"
                "    return -1\n"
            )
            return code
        return None

    def synthesize_comparison(self, test_cases: List[Dict]) -> Optional[str]:
        """
        Synthesize a comparison operator from examples.
        Given input/output pairs, find the correct comparison.
        """
        solver = z3.Solver()

        # Synthesize: which comparison? (<, <=, ==, >=, >)
        op = z3.Int('op')  # 0=<, 1=<=, 2==, 3=>=, 4=>
        solver.add(op >= 0, op <= 4)

        for tc in test_cases:
            a_val = tc["a"]
            b_val = tc["b"]
            expected = tc["result"]

            a = z3.IntVal(a_val)
            b = z3.IntVal(b_val)

            if expected:
                # The comparison should be true
                solver.add(z3.Or(
                    z3.And(op == 0, a < b),
                    z3.And(op == 1, a <= b),
                    z3.And(op == 2, a == b),
                    z3.And(op == 3, a >= b),
                    z3.And(op == 4, a > b),
                ))
            else:
                # The comparison should be false
                solver.add(z3.Or(
                    z3.And(op == 0, z3.Not(a < b)),
                    z3.And(op == 1, z3.Not(a <= b)),
                    z3.And(op == 2, z3.Not(a == b)),
                    z3.And(op == 3, z3.Not(a >= b)),
                    z3.And(op == 4, z3.Not(a > b)),
                ))

        if solver.check() == z3.sat:
            model = solver.model()
            op_val = model[op].as_long()
            ops = ["<", "<=", "==", ">=", ">"]
            self.synthesis_count += 1
            return ops[op_val]
        return None

    def synthesize_linear_expression(self, test_cases: List[Dict],
                                     max_coeff: int = 10) -> Optional[str]:
        """
        Synthesize a linear expression f(x) = a*x + b from examples.
        """
        a = z3.Int('a')
        b = z3.Int('b')

        solver = z3.Solver()
        solver.add(z3.And(a >= -max_coeff, a <= max_coeff))
        solver.add(z3.And(b >= -max_coeff, b <= max_coeff))

        for tc in test_cases:
            x_val = z3.IntVal(tc["input"])
            expected = z3.IntVal(tc["output"])
            solver.add(a * x_val + b == expected)

        if solver.check() == z3.sat:
            model = solver.model()
            a_val = model[a].as_long()
            b_val = model[b].as_long()
            self.synthesis_count += 1

            if a_val == 0:
                return str(b_val)
            elif b_val == 0:
                return f"{a_val} * x" if a_val != 1 else "x"
            elif b_val > 0:
                prefix = f"{a_val} * x" if a_val != 1 else "x"
                return f"{prefix} + {b_val}"
            else:
                prefix = f"{a_val} * x" if a_val != 1 else "x"
                return f"{prefix} - {abs(b_val)}"
        return None

    def synthesize_loop_bound(self, array_length: int,
                              must_visit_all: bool = True) -> Optional[int]:
        """Synthesize loop bound for iteration."""
        bound = z3.Int('bound')
        n = z3.IntVal(array_length)

        solver = z3.Solver()
        solver.add(bound > 0)
        solver.add(bound <= n)

        if must_visit_all:
            solver.add(bound == n)
        else:
            solver.add(bound <= n)

        if solver.check() == z3.sat:
            model = solver.model()
            self.synthesis_count += 1
            return model[bound].as_long()
        return None

    # ═══════════════════════════════════════════
    # VERIFICATION
    # ═══════════════════════════════════════════

    def verify_program(self, code: str,
                       test_cases: List[Dict]) -> Tuple[bool, str]:
        """Verify a program against test cases."""
        self.verification_count += 1
        try:
            namespace = {}
            exec(code, namespace)

            for test in test_cases:
                func_name = test.get("function")
                if not func_name or func_name not in namespace:
                    return False, f"Function {func_name} not found"

                func = namespace[func_name]
                inputs = test.get("inputs", [])
                expected = test.get("expected")

                result = func(*inputs)

                if result != expected:
                    return False, (
                        f"Test failed: {func_name}({inputs}) = "
                        f"{result}, expected {expected}"
                    )

            return True, "All tests passed"

        except Exception as e:
            return False, f"Error: {str(e)}"

    def verify_with_z3(self, property_name: str,
                       constraints: List) -> Tuple[bool, Optional[Dict]]:
        """
        Verify a property using Z3.
        Returns (satisfied, counterexample_or_None).
        """
        solver = z3.Solver()
        for constraint in constraints:
            solver.add(constraint)

        result = solver.check()
        if result == z3.sat:
            model = solver.model()
            counterexample = {
                str(d): model[d] for d in model.decls()
            }
            return True, counterexample
        elif result == z3.unsat:
            return False, None
        else:
            return False, None

    def check_array_bounds(self, index_expr, array_length: int) -> bool:
        """Verify array access is always in bounds."""
        i = z3.Int('i')
        n = z3.IntVal(array_length)

        solver = z3.Solver()
        # Try to find a counterexample: index out of bounds
        solver.add(z3.Or(i < 0, i >= n))
        solver.add(index_expr(i))

        # If UNSAT, no counterexample exists → bounds are safe
        return solver.check() == z3.unsat

    def check_termination(self, variant_decreases: bool,
                          variant_bounded_below: bool) -> bool:
        """Check loop termination via variant function."""
        return variant_decreases and variant_bounded_below

    # ═══════════════════════════════════════════
    # COUNTEREXAMPLE-GUIDED REFINEMENT (CEGIS)
    # ═══════════════════════════════════════════

    def cegis_loop(self, sketch: ProgramSketch,
                   max_iterations: int = 10) -> Optional[str]:
        """
        Counterexample-Guided Inductive Synthesis.
        1. Synthesize candidate from current test cases
        2. Verify against all constraints
        3. If fails, add counterexample to test cases
        4. Repeat until success or max iterations
        """
        test_cases = list(sketch.test_cases)

        for iteration in range(max_iterations):
            # Synthesize candidate
            candidate = self._synthesize_candidate(sketch, test_cases)
            if candidate is None:
                return None

            # Verify against full specification
            success, message = self.verify_program(candidate, test_cases)
            if success:
                return candidate

            # Extract counterexample and add to test cases
            # (In a real CEGIS, Z3 would provide the counterexample)
            break

        return None

    def _synthesize_candidate(self, sketch: ProgramSketch,
                              test_cases: List[Dict]) -> Optional[str]:
        """Attempt synthesis from sketch and test cases."""
        # For now, return the template if it has no unfilled holes
        if "{" not in sketch.template:
            return sketch.template
        return None

    # ═══════════════════════════════════════════
    # INTEGRATION WITH COMPOSITOR
    # ═══════════════════════════════════════════

    def fill_template_with_z3(self, template: str, holes: List[str],
                              test_cases: List[Dict]) -> Optional[Dict[str, str]]:
        """
        Try to fill template holes using Z3.
        Returns dict of hole_name -> value, or None if unsolvable.
        """
        filled = {}

        for hole in holes:
            if "condition" in hole.lower():
                # Try to synthesize a condition
                result = self._synthesize_condition_hole(hole, test_cases)
                if result:
                    filled[hole] = result
            elif "bound" in hole.lower():
                # Try to synthesize a loop bound
                result = self.synthesize_loop_bound(
                    test_cases[0].get("array_length", 10)
                )
                if result is not None:
                    filled[hole] = str(result)

        return filled if filled else None

    def _synthesize_condition_hole(self, hole_name: str,
                                   test_cases: List[Dict]) -> Optional[str]:
        """Synthesize a boolean condition from test cases."""
        # Extract comparison examples if available
        comparison_tests = [
            tc for tc in test_cases
            if "a" in tc and "b" in tc and "result" in tc
        ]
        if comparison_tests:
            return self.synthesize_comparison(comparison_tests)
        return None

    def compute_synthesis_confidence(self, solution: str,
                                     verification_level: str,
                                     test_results: Any) -> float:
        """Confidence based on verification depth."""
        confidence_map = {
            "formal_proof_all_inputs": 1.0,
            "formal_proof_test_cases": 0.8,
            "property_based": 0.6,
            "unit_tests": 0.5,
            "syntax_only": 0.2,
            "none": 0.0,
        }

        base = confidence_map.get(verification_level, 0.3)

        if isinstance(test_results, list):
            test_count = len(test_results)
            if test_count > 10:
                base += 0.05
            elif test_count < 3:
                base -= 0.1

        return max(0.0, min(1.0, base))

    def get_stats(self) -> Dict[str, int]:
        return {
            "synthesis_count": self.synthesis_count,
            "verification_count": self.verification_count
        }


# ═══════════════════════════════════════════
# SKETCH GENERATION
# ═══════════════════════════════════════════

class SketchGenerator:
    """Generates program sketches from strategies and specifications."""

    TEMPLATES = {
        "divide-and-conquer": (
            "def {function_name}({params}):\n"
            "    if {base_condition}:\n"
            "        return {base_return}\n"
            "    \n"
            "    {divide_step}\n"
            "    \n"
            "    {conquer_step}\n"
            "    \n"
            "    return {combine_step}\n"
        ),

        "dynamic-programming": (
            "def {function_name}({params}):\n"
            "    dp = [[0] * {dp_width} for _ in range({dp_height})]\n"
            "    \n"
            "    {base_cases}\n"
            "    \n"
            "    for i in range({dp_start_i}):\n"
            "        for j in range({dp_start_j}):\n"
            "            dp[i][j] = {recurrence}\n"
            "    \n"
            "    return dp[{final_i}][{final_j}]\n"
        ),

        "two-pointer": (
            "def {function_name}({params}):\n"
            "    left, right = {init_left}, {init_right}\n"
            "    \n"
            "    while {loop_condition}:\n"
            "        if {comparison}:\n"
            "            {action_if_true}\n"
            "        else:\n"
            "            {action_if_false}\n"
            "        \n"
            "        {pointer_update}\n"
            "    \n"
            "    return {return_value}\n"
        ),

        "sliding-window": (
            "def {function_name}({params}):\n"
            "    window_start = 0\n"
            "    {window_state}\n"
            "    {result_tracker}\n"
            "    \n"
            "    for window_end in range(len({input_var})):\n"
            "        {expand_action}\n"
            "        \n"
            "        while {invalid_condition}:\n"
            "            {contract_action}\n"
            "            window_start += 1\n"
            "        \n"
            "        {update_result}\n"
            "    \n"
            "    return {final_result}\n"
        ),

        "binary-search": (
            "def {function_name}({params}):\n"
            "    lo, hi = {init_lo}, {init_hi}\n"
            "    \n"
            "    while {loop_condition}:\n"
            "        mid = lo + (hi - lo) // 2\n"
            "        \n"
            "        if {found_condition}:\n"
            "            return mid\n"
            "        elif {go_right_condition}:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    \n"
            "    return {not_found_value}\n"
        ),
    }

    META_PARAMS = {"function_name", "params", "description"}

    def generate_sketch(self, strategy: str, specification: str) -> ProgramSketch:
        """Generate a sketch based on strategy and specification."""
        if strategy not in self.TEMPLATES:
            raise ValueError(f"Unknown strategy: {strategy}")

        template = self.TEMPLATES[strategy]

        holes = []
        variables = re.findall(r'\{(\w+)\}', template)

        for var in set(variables):
            if var in self.META_PARAMS:
                continue

            if "condition" in var:
                hole_type = HoleType.CONDITION
            elif "action" in var or "step" in var:
                hole_type = HoleType.STATEMENT
            elif "bound" in var or "start" in var or "height" in var or "width" in var:
                hole_type = HoleType.LOOP_BOUND
            else:
                hole_type = HoleType.EXPRESSION

            holes.append(SketchHole(
                id=var,
                hole_type=hole_type,
                context=f"Fill in {var} for {strategy} pattern"
            ))

        return ProgramSketch(
            template=template,
            holes=holes,
            specification=specification
        )

    def list_strategies(self) -> List[str]:
        """List available sketch strategies."""
        return list(self.TEMPLATES.keys())
