"""Tests for Sketch Synthesis."""

from ecs.verification.sketch_synth import (
    SketchSynthesizer, SketchGenerator, ProgramSketch,
    SketchHole, HoleType
)


# ═══════════════════════════════════════════
# BINARY SEARCH SYNTHESIS
# ═══════════════════════════════════════════

def test_binary_search_synthesis():
    """Test synthesizing binary search via Z3 constraints."""
    synth = SketchSynthesizer()

    code = synth.synthesize_binary_search()
    assert code is not None, "Synthesis failed"
    assert "def binary_search" in code

    test_cases = [
        {"function": "binary_search", "inputs": [[1, 3, 5, 7, 9], 5], "expected": 2},
        {"function": "binary_search", "inputs": [[1, 3, 5, 7, 9], 1], "expected": 0},
        {"function": "binary_search", "inputs": [[1, 3, 5, 7, 9], 9], "expected": 4},
        {"function": "binary_search", "inputs": [[1, 3, 5, 7, 9], 4], "expected": -1},
    ]

    success, message = synth.verify_program(code, test_cases)
    assert success, f"Verification failed: {message}"


def test_binary_search_edge_cases():
    synth = SketchSynthesizer()
    code = synth.synthesize_binary_search()

    test_cases = [
        {"function": "binary_search", "inputs": [[1], 1], "expected": 0},
        {"function": "binary_search", "inputs": [[1], 2], "expected": -1},
        {"function": "binary_search", "inputs": [[], 1], "expected": -1},
    ]

    success, message = synth.verify_program(code, test_cases)
    assert success, f"Edge case failed: {message}"


# ═══════════════════════════════════════════
# COMPARISON SYNTHESIS
# ═══════════════════════════════════════════

def test_synthesize_less_than():
    synth = SketchSynthesizer()
    test_cases = [
        {"a": 1, "b": 5, "result": True},
        {"a": 5, "b": 1, "result": False},
        {"a": 3, "b": 3, "result": False},
    ]
    result = synth.synthesize_comparison(test_cases)
    assert result == "<"


def test_synthesize_less_equal():
    synth = SketchSynthesizer()
    test_cases = [
        {"a": 1, "b": 5, "result": True},
        {"a": 3, "b": 3, "result": True},
        {"a": 5, "b": 1, "result": False},
    ]
    result = synth.synthesize_comparison(test_cases)
    assert result == "<="


def test_synthesize_equality():
    synth = SketchSynthesizer()
    test_cases = [
        {"a": 3, "b": 3, "result": True},
        {"a": 1, "b": 5, "result": False},
        {"a": 5, "b": 1, "result": False},
    ]
    result = synth.synthesize_comparison(test_cases)
    assert result == "=="


def test_synthesize_greater_than():
    synth = SketchSynthesizer()
    test_cases = [
        {"a": 5, "b": 1, "result": True},
        {"a": 1, "b": 5, "result": False},
        {"a": 3, "b": 3, "result": False},
    ]
    result = synth.synthesize_comparison(test_cases)
    assert result == ">"


def test_synthesize_unsatisfiable():
    synth = SketchSynthesizer()
    # Contradictory: 3 < 3 AND 3 > 3
    test_cases = [
        {"a": 3, "b": 3, "result": True},
        {"a": 3, "b": 3, "result": False},
    ]
    result = synth.synthesize_comparison(test_cases)
    assert result is None


# ═══════════════════════════════════════════
# LINEAR EXPRESSION SYNTHESIS
# ═══════════════════════════════════════════

def test_synthesize_double():
    synth = SketchSynthesizer()
    test_cases = [
        {"input": 1, "output": 2},
        {"input": 2, "output": 4},
        {"input": 3, "output": 6},
    ]
    result = synth.synthesize_linear_expression(test_cases)
    assert result == "2 * x"


def test_synthesize_identity():
    synth = SketchSynthesizer()
    test_cases = [
        {"input": 0, "output": 0},
        {"input": 5, "output": 5},
        {"input": -3, "output": -3},
    ]
    result = synth.synthesize_linear_expression(test_cases)
    assert result == "x"


def test_synthesize_constant():
    synth = SketchSynthesizer()
    test_cases = [
        {"input": 0, "output": 7},
        {"input": 5, "output": 7},
        {"input": -3, "output": 7},
    ]
    result = synth.synthesize_linear_expression(test_cases)
    assert result == "7"


def test_synthesize_offset():
    synth = SketchSynthesizer()
    test_cases = [
        {"input": 0, "output": 3},
        {"input": 1, "output": 4},
        {"input": 2, "output": 5},
    ]
    result = synth.synthesize_linear_expression(test_cases)
    assert result == "x + 3"


def test_synthesize_negative_slope():
    synth = SketchSynthesizer()
    test_cases = [
        {"input": 0, "output": 5},
        {"input": 1, "output": 2},
        {"input": 2, "output": -1},
    ]
    result = synth.synthesize_linear_expression(test_cases)
    assert result == "-3 * x + 5"


def test_linear_unsatisfiable():
    """Non-linear relationship can't be captured."""
    synth = SketchSynthesizer()
    test_cases = [
        {"input": 1, "output": 1},
        {"input": 2, "output": 4},
        {"input": 3, "output": 9},
    ]
    result = synth.synthesize_linear_expression(test_cases)
    assert result is None


# ═══════════════════════════════════════════
# LOOP BOUND SYNTHESIS
# ═══════════════════════════════════════════

def test_loop_bound_visit_all():
    synth = SketchSynthesizer()
    result = synth.synthesize_loop_bound(10, must_visit_all=True)
    assert result == 10


def test_loop_bound_partial():
    synth = SketchSynthesizer()
    result = synth.synthesize_loop_bound(10, must_visit_all=False)
    assert result is not None
    assert 1 <= result <= 10


# ═══════════════════════════════════════════
# PROGRAM VERIFICATION
# ═══════════════════════════════════════════

def test_verify_correct_program():
    synth = SketchSynthesizer()
    code = "def add(a, b):\n    return a + b\n"
    tests = [
        {"function": "add", "inputs": [2, 3], "expected": 5},
        {"function": "add", "inputs": [0, 0], "expected": 0},
        {"function": "add", "inputs": [-1, 1], "expected": 0},
    ]
    success, msg = synth.verify_program(code, tests)
    assert success


def test_verify_incorrect_program():
    synth = SketchSynthesizer()
    code = "def add(a, b):\n    return a - b\n"
    tests = [{"function": "add", "inputs": [2, 3], "expected": 5}]
    success, msg = synth.verify_program(code, tests)
    assert not success
    assert "expected 5" in msg


def test_verify_missing_function():
    synth = SketchSynthesizer()
    code = "def foo(): pass\n"
    tests = [{"function": "bar", "inputs": [], "expected": None}]
    success, msg = synth.verify_program(code, tests)
    assert not success
    assert "not found" in msg


def test_verify_runtime_error():
    synth = SketchSynthesizer()
    code = "def divide(a, b):\n    return a / b\n"
    tests = [{"function": "divide", "inputs": [1, 0], "expected": None}]
    success, msg = synth.verify_program(code, tests)
    assert not success
    assert "Error" in msg


# ═══════════════════════════════════════════
# Z3 VERIFICATION
# ═══════════════════════════════════════════

def test_verify_with_z3_sat():
    import z3
    synth = SketchSynthesizer()

    x = z3.Int('x')
    constraints = [x > 0, x < 10, x == 5]
    sat, model = synth.verify_with_z3("test", constraints)
    assert sat
    assert model is not None


def test_verify_with_z3_unsat():
    import z3
    synth = SketchSynthesizer()

    x = z3.Int('x')
    constraints = [x > 10, x < 5]
    sat, model = synth.verify_with_z3("test", constraints)
    assert not sat
    assert model is None


def test_check_array_bounds_safe():
    import z3
    synth = SketchSynthesizer()

    # index i where 0 <= i < 5 — always in bounds for array of size 5
    safe = synth.check_array_bounds(
        lambda i: z3.And(i >= 0, i < 5),
        array_length=5
    )
    assert safe


def test_check_array_bounds_unsafe():
    import z3
    synth = SketchSynthesizer()

    # index i where 0 <= i <= 5 — i=5 is out of bounds
    safe = synth.check_array_bounds(
        lambda i: z3.And(i >= 0, i <= 5),
        array_length=5
    )
    assert not safe


# ═══════════════════════════════════════════
# SKETCH GENERATION
# ═══════════════════════════════════════════

def test_sketch_generation_dc():
    gen = SketchGenerator()
    sketch = gen.generate_sketch("divide-and-conquer", "Sort using merge sort")

    assert sketch is not None
    assert len(sketch.holes) > 0
    assert sketch.specification == "Sort using merge sort"

    hole_ids = [h.id for h in sketch.holes]
    assert "base_condition" in hole_ids
    assert "combine_step" in hole_ids


def test_sketch_generation_dp():
    gen = SketchGenerator()
    sketch = gen.generate_sketch("dynamic-programming", "Longest common subsequence")

    hole_ids = [h.id for h in sketch.holes]
    assert "recurrence" in hole_ids
    assert "base_cases" in hole_ids


def test_sketch_generation_binary_search():
    gen = SketchGenerator()
    sketch = gen.generate_sketch("binary-search", "Find target in sorted array")

    hole_ids = [h.id for h in sketch.holes]
    assert "found_condition" in hole_ids
    assert "loop_condition" in hole_ids


def test_sketch_generation_unknown_strategy():
    gen = SketchGenerator()
    try:
        gen.generate_sketch("nonexistent", "anything")
        assert False, "Should have raised"
    except ValueError:
        pass


def test_sketch_hole_types():
    gen = SketchGenerator()
    sketch = gen.generate_sketch("two-pointer", "Two sum")

    condition_holes = [h for h in sketch.holes if h.hole_type == HoleType.CONDITION]
    statement_holes = [h for h in sketch.holes if h.hole_type == HoleType.STATEMENT]

    assert len(condition_holes) > 0
    assert len(statement_holes) > 0


def test_list_strategies():
    gen = SketchGenerator()
    strategies = gen.list_strategies()
    assert "divide-and-conquer" in strategies
    assert "binary-search" in strategies
    assert len(strategies) >= 5


# ═══════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════

def test_synthesis_stats():
    synth = SketchSynthesizer()
    synth.synthesize_binary_search()
    synth.verify_program("def f(): pass", [])

    stats = synth.get_stats()
    assert stats["synthesis_count"] == 1
    assert stats["verification_count"] == 1
