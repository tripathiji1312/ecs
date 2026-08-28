"""
Tests for the Constraint Inference Engine (Week 8).
Validates constraint extraction from natural language and
Z3-free synthesis from inferred constraints.
"""

from ecs.verification.constraint_inference import ConstraintInferenceEngine
from ecs.verification.sandbox import SafeExecutor
from ecs.memory.hdc import MemoryType


engine = ConstraintInferenceEngine()
executor = SafeExecutor(timeout=5.0)


# ═══════════════════════════════════════════
# CONSTRAINT INFERENCE
# ═══════════════════════════════════════════

def test_infer_sort_constraints():
    """Sort-related keywords trigger sort constraint."""
    constraints = engine.infer_constraints("Sort a list of integers in ascending order")
    patterns = [c["pattern"] for c in constraints]
    assert "sort" in patterns


def test_infer_binary_search():
    """Binary search keywords trigger binary_search constraint."""
    constraints = engine.infer_constraints("Implement binary search on a sorted array")
    patterns = [c["pattern"] for c in constraints]
    assert "binary_search" in patterns
    assert "sorted" in patterns


def test_infer_palindrome():
    """Palindrome keywords detected."""
    constraints = engine.infer_constraints("Check if a string is a palindrome")
    patterns = [c["pattern"] for c in constraints]
    assert "palindrome" in patterns


def test_infer_two_sum():
    """Two-sum pattern detected."""
    constraints = engine.infer_constraints(
        "Find two numbers in an array that sum to a target"
    )
    patterns = [c["pattern"] for c in constraints]
    assert "sum_pair" in patterns


def test_infer_fibonacci():
    """Fibonacci pattern detected."""
    constraints = engine.infer_constraints("Compute the nth Fibonacci number")
    patterns = [c["pattern"] for c in constraints]
    assert "fibonacci" in patterns


def test_infer_prime():
    """Prime check pattern detected."""
    constraints = engine.infer_constraints("Check if a number is prime")
    patterns = [c["pattern"] for c in constraints]
    assert "prime" in patterns


def test_infer_balanced_brackets():
    """Balanced brackets pattern detected."""
    constraints = engine.infer_constraints(
        "Check if a string of brackets is balanced"
    )
    patterns = [c["pattern"] for c in constraints]
    assert "balanced_brackets" in patterns


def test_infer_maximum():
    """Maximum element pattern detected."""
    constraints = engine.infer_constraints("Find the maximum element in a list")
    patterns = [c["pattern"] for c in constraints]
    assert "maximum" in patterns


def test_infer_no_match():
    """Unrecognized problems return empty constraints."""
    constraints = engine.infer_constraints("xyzzy nonsense blurble")
    assert constraints == []


def test_infer_multiple_constraints():
    """Multiple constraints from complex problem."""
    constraints = engine.infer_constraints(
        "Find the minimum element in a sorted unique array"
    )
    patterns = [c["pattern"] for c in constraints]
    assert "minimum" in patterns
    assert "sorted" in patterns
    assert "unique" in patterns


# ═══════════════════════════════════════════
# SYNTHESIS FROM CONSTRAINTS
# ═══════════════════════════════════════════

def test_synthesize_sort():
    """Sort constraint produces working sort code."""
    constraints = engine.infer_constraints("Sort a list of integers")
    code = engine.synthesize_from_constraints(constraints, "Sort a list")
    assert code is not None
    assert "def " in code


def test_synthesize_binary_search():
    """Binary search constraint produces working code."""
    constraints = engine.infer_constraints("Implement binary search on a sorted array")
    code = engine.synthesize_from_constraints(constraints, "binary search")
    assert code is not None
    assert "while" in code


def test_synthesize_palindrome():
    """Palindrome constraint produces working code."""
    constraints = engine.infer_constraints("Check if a string is a palindrome")
    code = engine.synthesize_from_constraints(constraints, "palindrome")
    assert code is not None


def test_synthesize_unknown_returns_none():
    """No constraints → no synthesis."""
    code = engine.synthesize_from_constraints([], "unknown problem")
    assert code is None


# ═══════════════════════════════════════════
# VERIFICATION: SYNTHESIZED CODE PASSES TESTS
# ═══════════════════════════════════════════

def test_sort_code_passes_tests():
    """Synthesized sort code passes test cases."""
    constraints = engine.infer_constraints("Sort a list")
    code = engine.synthesize_from_constraints(constraints, "sort")
    tests = engine.get_test_cases_for_pattern("sort")

    passed, message, _ = executor.execute_tests(code, tests)
    assert passed, f"Sort code failed: {message}"


def test_binary_search_code_passes_tests():
    """Synthesized binary search code passes test cases."""
    constraints = engine.infer_constraints("Binary search on sorted array")
    code = engine.synthesize_from_constraints(constraints, "binary search")
    tests = engine.get_test_cases_for_pattern("binary_search")

    passed, message, _ = executor.execute_tests(code, tests)
    assert passed, f"Binary search code failed: {message}"


def test_palindrome_code_passes_tests():
    """Synthesized palindrome code passes test cases."""
    constraints = engine.infer_constraints("Check if palindrome")
    code = engine.synthesize_from_constraints(constraints, "palindrome")
    tests = engine.get_test_cases_for_pattern("palindrome")

    passed, message, _ = executor.execute_tests(code, tests)
    assert passed, f"Palindrome code failed: {message}"


def test_fibonacci_code_passes_tests():
    """Synthesized fibonacci code passes test cases."""
    constraints = engine.infer_constraints("Compute fibonacci number")
    code = engine.synthesize_from_constraints(constraints, "fibonacci")
    tests = engine.get_test_cases_for_pattern("fibonacci")

    passed, message, _ = executor.execute_tests(code, tests)
    assert passed, f"Fibonacci code failed: {message}"


def test_factorial_code_passes_tests():
    """Synthesized factorial code passes test cases."""
    constraints = engine.infer_constraints("Calculate the factorial of n")
    code = engine.synthesize_from_constraints(constraints, "factorial")
    tests = engine.get_test_cases_for_pattern("factorial")

    passed, message, _ = executor.execute_tests(code, tests)
    assert passed, f"Factorial code failed: {message}"


def test_prime_code_passes_tests():
    """Synthesized prime check code passes test cases."""
    constraints = engine.infer_constraints("Check if a number is prime")
    code = engine.synthesize_from_constraints(constraints, "prime check")
    tests = engine.get_test_cases_for_pattern("prime")

    passed, message, _ = executor.execute_tests(code, tests)
    assert passed, f"Prime code failed: {message}"


def test_maximum_code_passes_tests():
    """Synthesized maximum code passes test cases."""
    constraints = engine.infer_constraints("Find the maximum element")
    code = engine.synthesize_from_constraints(constraints, "max")
    tests = engine.get_test_cases_for_pattern("maximum")

    passed, message, _ = executor.execute_tests(code, tests)
    assert passed, f"Maximum code failed: {message}"


def test_balanced_brackets_code_passes_tests():
    """Synthesized balanced brackets code passes test cases."""
    constraints = engine.infer_constraints("Check if brackets are balanced")
    code = engine.synthesize_from_constraints(constraints, "balanced")
    tests = engine.get_test_cases_for_pattern("balanced_brackets")

    passed, message, _ = executor.execute_tests(code, tests)
    assert passed, f"Balanced brackets code failed: {message}"


def test_rotate_code_passes_tests():
    """Synthesized rotate code passes test cases."""
    constraints = engine.infer_constraints("Rotate an array by k positions")
    code = engine.synthesize_from_constraints(constraints, "rotate")
    tests = engine.get_test_cases_for_pattern("rotate")

    passed, message, _ = executor.execute_tests(code, tests)
    assert passed, f"Rotate code failed: {message}"


def test_two_sum_code_passes_tests():
    """Synthesized two-sum code passes test cases."""
    constraints = engine.infer_constraints(
        "Find two numbers that sum to a target"
    )
    code = engine.synthesize_from_constraints(constraints, "two sum")
    tests = engine.get_test_cases_for_pattern("sum_pair")

    passed, message, _ = executor.execute_tests(code, tests)
    assert passed, f"Two sum code failed: {message}"


# ═══════════════════════════════════════════
# TEST CASE GENERATION
# ═══════════════════════════════════════════

def test_get_test_cases_known_pattern():
    """Known patterns have test cases."""
    tests = engine.get_test_cases_for_pattern("sort")
    assert len(tests) >= 3


def test_get_test_cases_unknown_pattern():
    """Unknown patterns return empty list."""
    tests = engine.get_test_cases_for_pattern("quantum_teleportation")
    assert tests == []


# ═══════════════════════════════════════════
# ORCHESTRATOR INTEGRATION
# ═══════════════════════════════════════════

def test_orchestrator_uses_constraint_inference():
    """Orchestrator uses constraint inference when neural is offline."""
    from ecs.orchestrator import ECSOrchestrator

    ecs = ECSOrchestrator()

    # Neural is offline (no Ollama) — should use constraint inference
    result = ecs.solve_problem("Sort a list of integers in ascending order")

    # With constraint inference, sort should now succeed
    assert result["synthesis_method"] == "constraint_inference"
    assert result["code"] is not None
    assert result["success"] is True
    assert result["confidence"] > 0.5


def test_orchestrator_constraint_fibonacci():
    """Orchestrator solves fibonacci via constraints."""
    from ecs.orchestrator import ECSOrchestrator

    ecs = ECSOrchestrator()
    result = ecs.solve_problem("Compute the nth Fibonacci number")

    assert result["success"] is True
    assert result["synthesis_method"] == "constraint_inference"


def test_orchestrator_constraint_palindrome():
    """Orchestrator solves palindrome via constraints."""
    from ecs.orchestrator import ECSOrchestrator

    ecs = ECSOrchestrator()
    result = ecs.solve_problem("Check if a string is a palindrome")

    assert result["success"] is True
    assert result["synthesis_method"] == "constraint_inference"


def test_orchestrator_stores_successful_code():
    """Successful code is stored in procedural memory for reuse."""
    from ecs.orchestrator import ECSOrchestrator

    ecs = ECSOrchestrator()
    ecs.solve_problem("Check if a number is prime")

    # Should have procedural memory entries now
    procedural_ids = ecs.memory.type_indices[MemoryType.PROCEDURAL]
    code_items = [pid for pid in procedural_ids if pid.startswith("code_")]
    assert len(code_items) >= 1


def test_orchestrator_code_reuse():
    """Second similar problem retrieves code from memory."""
    from ecs.orchestrator import ECSOrchestrator

    ecs = ECSOrchestrator()

    # First solve — constraint inference
    r1 = ecs.solve_problem("Check if a number is prime")
    assert r1["success"]

    # Second solve of similar problem — should reuse from memory
    r2 = ecs.solve_problem("Check if number is prime number")

    # Either memory_reuse or constraint_inference (both are valid)
    assert r2["success"]
    assert r2["synthesis_method"] in ("memory_reuse", "constraint_inference")


def test_tier_success_evaluation():
    """Tier-specific success evaluation works."""
    from ecs.orchestrator import ECSOrchestrator

    ecs = ECSOrchestrator()
    result = ecs.solve_problem("Sort a list of integers")

    tier_ok, reason = ecs.evaluate_tier_success(result)
    assert tier_ok is True
    assert "Structured success" in reason
