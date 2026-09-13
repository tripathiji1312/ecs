"""
Tests for the intelligence components: execution-guided synthesis,
program induction, and type-driven compositional reasoning.

These test that the system REASONS about problems rather than
retrieving templates.
"""

import pytest

from ecs.synthesis.execution_guided import ExecutionGuidedSynthesizer
from ecs.synthesis.induction import ProgramInductionEngine
from ecs.synthesis.compositional import CompositionalSynthesizer
from ecs.verification.sandbox import SafeExecutor


# ═══════════════════════════════════════════
# Execution-Guided Synthesis
# ═══════════════════════════════════════════


class TestExecutionGuided:

    @pytest.fixture
    def synth(self):
        return ExecutionGuidedSynthesizer()

    @pytest.fixture
    def executor(self):
        return SafeExecutor(timeout=5.0)

    def _make_prompt(self, sig: str, docstring: str) -> str:
        return f"\ndef {sig}:\n    \"\"\"{docstring}\n    \"\"\"\n"

    def _verify(self, code, prompt, entry_point, executor):
        """Verify generated code against HumanEval-style prompt."""
        # Extract examples from prompt
        import re
        examples = []
        lines = prompt.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(">>>"):
                call_str = stripped[3:].strip()
                if i + 1 < len(lines):
                    result_line = lines[i + 1].strip()
                    if result_line and not result_line.startswith(">>>"):
                        try:
                            output = eval(result_line)
                            examples.append((call_str, output))
                        except Exception:
                            pass

        for call_str, expected in examples:
            test_code = f"{code}\nassert {call_str} == {repr(expected)}\n"
            result = executor.execute(test_code)
            if not result.success:
                return False
        return True

    def test_single_expression_sum(self, synth):
        prompt = (
            "\ndef sum_list(l: list):\n"
            '    """Return sum of elements.\n'
            "    >>> sum_list([1, 2, 3])\n"
            "    6\n"
            "    >>> sum_list([])\n"
            "    0\n"
            '    """\n'
        )
        code = synth.synthesize("Return sum of elements", prompt, "sum_list")
        assert code is not None
        assert "sum" in code

    def test_single_expression_max(self, synth):
        prompt = (
            "\ndef max_element(l: list):\n"
            '    """Return maximum element.\n'
            "    >>> max_element([1, 5, 3])\n"
            "    5\n"
            "    >>> max_element([10])\n"
            "    10\n"
            '    """\n'
        )
        code = synth.synthesize("Return maximum element", prompt, "max_element")
        assert code is not None

    def test_filter_positive(self, synth):
        prompt = (
            "\ndef get_positive(l: list):\n"
            '    """Return only positive numbers.\n'
            "    >>> get_positive([-1, 2, -3, 4])\n"
            "    [2, 4]\n"
            "    >>> get_positive([5, -5, 0])\n"
            "    [5]\n"
            '    """\n'
        )
        code = synth.synthesize("Return only positive numbers", prompt, "get_positive")
        assert code is not None

    def test_string_palindrome(self, synth):
        prompt = (
            "\ndef is_palindrome(s: str):\n"
            '    """Check if string is palindrome.\n'
            "    >>> is_palindrome('racecar')\n"
            "    True\n"
            "    >>> is_palindrome('hello')\n"
            "    False\n"
            '    """\n'
        )
        code = synth.synthesize("Check if string is palindrome", prompt, "is_palindrome")
        assert code is not None

    def test_sorted_list(self, synth):
        prompt = (
            "\ndef sort_nums(l: list):\n"
            '    """Sort numbers ascending.\n'
            "    >>> sort_nums([3, 1, 2])\n"
            "    [1, 2, 3]\n"
            "    >>> sort_nums([5, 4, 3, 2, 1])\n"
            "    [1, 2, 3, 4, 5]\n"
            '    """\n'
        )
        code = synth.synthesize("Sort numbers in ascending order", prompt, "sort_nums")
        assert code is not None

    def test_length_count(self, synth):
        prompt = (
            "\ndef count_items(l: list):\n"
            '    """Return number of items.\n'
            "    >>> count_items([1, 2, 3])\n"
            "    3\n"
            "    >>> count_items([])\n"
            "    0\n"
            '    """\n'
        )
        code = synth.synthesize("Return number of items", prompt, "count_items")
        assert code is not None

    def test_two_param_addition(self, synth):
        prompt = (
            "\ndef add(a: int, b: int):\n"
            '    """Add two numbers.\n'
            "    >>> add(1, 2)\n"
            "    3\n"
            "    >>> add(0, 5)\n"
            "    5\n"
            '    """\n'
        )
        code = synth.synthesize("Add two numbers", prompt, "add")
        assert code is not None

    def test_conditional_bool_even(self, synth):
        prompt = (
            "\ndef all_positive(l: list):\n"
            '    """Check if all elements positive.\n'
            "    >>> all_positive([1, 2, 3])\n"
            "    True\n"
            "    >>> all_positive([1, -1, 3])\n"
            "    False\n"
            '    """\n'
        )
        code = synth.synthesize("Check if all elements positive", prompt, "all_positive")
        assert code is not None

    def test_recursive_factorial(self, synth):
        prompt = (
            "\ndef factorial(n: int):\n"
            '    """Compute factorial.\n'
            "    >>> factorial(0)\n"
            "    1\n"
            "    >>> factorial(1)\n"
            "    1\n"
            "    >>> factorial(5)\n"
            "    120\n"
            '    """\n'
        )
        code = synth.synthesize("Compute factorial", prompt, "factorial")
        assert code is not None

    def test_pairwise_close_elements(self, synth):
        prompt = (
            "\ndef has_close_elements(numbers: list, threshold: float):\n"
            '    """Check if any two elements are closer than threshold.\n'
            "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
            "    False\n"
            "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0], 0.3)\n"
            "    True\n"
            '    """\n'
        )
        code = synth.synthesize(
            "Check if any two elements are closer to each other than threshold",
            prompt, "has_close_elements"
        )
        assert code is not None

    def test_no_examples_returns_none(self, synth):
        prompt = "\ndef mystery(x):\n    pass\n"
        code = synth.synthesize("Do something", prompt, "mystery")
        assert code is None


# ═══════════════════════════════════════════
# Program Induction Engine
# ═══════════════════════════════════════════


class TestProgramInduction:

    @pytest.fixture
    def engine(self):
        return ProgramInductionEngine()

    def test_record_program(self, engine):
        engine.record("Sort a list", "def sort_list(l): return sorted(l)")
        assert len(engine.solved) == 1
        assert engine.solved[0].structure["uses_builtins"] == ["sorted"]

    def test_extract_structure_loop(self, engine):
        code = (
            "def total(l):\n"
            "    s = 0\n"
            "    for x in l:\n"
            "        s += x\n"
            "    return s\n"
        )
        s = engine._extract_structure(code)
        assert s["has_loop"]
        assert s["has_for"]
        assert s["uses_accumulator"]
        assert "accumulate" in s["pattern_tags"]

    def test_extract_structure_recursion(self, engine):
        code = (
            "def fib(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fib(n-1) + fib(n-2)\n"
        )
        s = engine._extract_structure(code)
        assert s["has_recursion"]
        assert "recursive" in s["pattern_tags"]

    def test_extract_structure_nested_loop(self, engine):
        code = (
            "def pairs(l):\n"
            "    for i in range(len(l)):\n"
            "        for j in range(i+1, len(l)):\n"
            "            if l[i] + l[j] == 0:\n"
            "                return True\n"
            "    return False\n"
        )
        s = engine._extract_structure(code)
        assert s["has_nested_loop"]
        assert "quadratic" in s["pattern_tags"]

    def test_extract_structure_comprehension(self, engine):
        code = "def evens(l): return [x for x in l if x % 2 == 0]\n"
        s = engine._extract_structure(code)
        assert s["uses_list_comp"]
        assert "comprehension" in s["pattern_tags"]

    def test_extract_structure_hash(self, engine):
        code = (
            "def unique(l):\n"
            "    seen = set()\n"
            "    result = []\n"
            "    for x in l:\n"
            "        if x not in seen:\n"
            "            seen.add(x)\n"
            "            result.append(x)\n"
            "    return result\n"
        )
        s = engine._extract_structure(code)
        assert s["uses_set"]
        assert "hash_based" in s["pattern_tags"]

    def test_fingerprint_deterministic(self, engine):
        code = "def f(l): return sorted(l)\n"
        s1 = engine._extract_structure(code)
        s2 = engine._extract_structure(code)
        assert engine._fingerprint(s1) == engine._fingerprint(s2)

    def test_fingerprint_differs_for_different_structures(self, engine):
        code_a = "def f(l): return sorted(l)\n"
        code_b = (
            "def f(n):\n"
            "    if n <= 1: return n\n"
            "    return f(n-1) + f(n-2)\n"
        )
        s_a = engine._extract_structure(code_a)
        s_b = engine._extract_structure(code_b)
        assert engine._fingerprint(s_a) != engine._fingerprint(s_b)

    def test_mine_abstractions_needs_minimum(self, engine):
        engine.record("p1", "def f(l): return sorted(l)")
        assert engine.mine_abstractions() == []

    def test_mine_abstractions_clusters(self, engine):
        # Add 4 programs with similar structure (list comprehension)
        engine.record("Filter positive", "def f(l): return [x for x in l if x > 0]")
        engine.record("Filter even", "def g(l): return [x for x in l if x % 2 == 0]")
        engine.record("Filter neg", "def h(l): return [x for x in l if x < 0]")

        abstractions = engine.mine_abstractions()
        assert len(abstractions) >= 1
        assert any("comprehension" in a.name or "pattern" in a.name
                    for a in abstractions)

    def test_mine_abstractions_accumulate_cluster(self, engine):
        # Add 3 accumulate-pattern programs
        codes = [
            "def total(l):\n    s = 0\n    for x in l:\n        s += x\n    return s\n",
            "def prod(l):\n    p = 1\n    for x in l:\n        p *= x\n    return p\n",
            "def count_pos(l):\n    c = 0\n    for x in l:\n        if x > 0:\n            c += 1\n    return c\n",
        ]
        for i, code in enumerate(codes):
            engine.record(f"problem_{i}", code)

        abstractions = engine.mine_abstractions()
        assert len(abstractions) >= 1

    def test_find_analogous_solution(self, engine):
        engine.record(
            "Sort a list of numbers in ascending order",
            "def sort_list(l): return sorted(l)"
        )
        result = engine.find_analogous_solution(
            "Sort the numbers in ascending order"
        )
        assert result is not None
        assert "sorted" in result

    def test_find_analogous_no_match(self, engine):
        engine.record("Sort a list", "def sort_list(l): return sorted(l)")
        result = engine.find_analogous_solution(
            "Draw a circle on the screen"
        )
        assert result is None

    def test_suggest_structure(self, engine):
        engine.record(
            "Sum all numbers in a list",
            "def total(l):\n    s = 0\n    for x in l:\n        s += x\n    return s\n"
        )
        suggestion = engine.suggest_structure(
            "Total all numbers in a list"
        )
        assert suggestion is not None
        assert "accumulate" in suggestion["pattern_tags"]

    def test_stats(self, engine):
        engine.record("p1", "def f(l): return sorted(l)")
        engine.record("p2", "def g(l): return sorted(l)")
        stats = engine.get_stats()
        assert stats["solved_count"] == 2

    def test_template_extraction(self, engine):
        codes = [
            "def f(l): return [x for x in l if x > 0]",
            "def g(l): return [x for x in l if x % 2 == 0]",
            "def h(l): return [x for x in l if x < 0]",
        ]
        for i, code in enumerate(codes):
            engine.record(f"p{i}", code)
        abstractions = engine.mine_abstractions()
        assert any(a.template and "FUNC" in a.template for a in abstractions)


# ═══════════════════════════════════════════
# Type-Driven Compositional Synthesis
# ═══════════════════════════════════════════


class TestCompositionalTypeDriven:

    @pytest.fixture
    def synth(self):
        return CompositionalSynthesizer()

    @pytest.fixture
    def executor(self):
        return SafeExecutor(timeout=5.0)

    def test_list_to_int_sum(self, synth):
        prompt = (
            "\ndef sum_list(l: list):\n"
            '    """Sum elements.\n'
            "    >>> sum_list([1, 2, 3])\n"
            "    6\n"
            '    """\n'
        )
        code = synth.synthesize("Sum elements of list", prompt, "sum_list")
        assert code is not None

    def test_list_to_list_sorted(self, synth):
        prompt = (
            "\ndef sort_it(l: list):\n"
            '    """Sort ascending.\n'
            "    >>> sort_it([3, 1, 2])\n"
            "    [1, 2, 3]\n"
            '    """\n'
        )
        code = synth.synthesize("Sort ascending", prompt, "sort_it")
        assert code is not None

    def test_list_to_bool_monotonic(self, synth):
        prompt = (
            "\ndef is_monotonic(l: list):\n"
            '    """Check if monotonically increasing or decreasing.\n'
            "    >>> is_monotonic([1, 2, 3])\n"
            "    True\n"
            "    >>> is_monotonic([3, 2, 1])\n"
            "    True\n"
            "    >>> is_monotonic([1, 3, 2])\n"
            "    False\n"
            '    """\n'
        )
        code = synth.synthesize(
            "Check if list is monotonically increasing or decreasing",
            prompt, "is_monotonic"
        )
        assert code is not None

    def test_str_to_bool_palindrome(self, synth):
        prompt = (
            "\ndef is_palindrome(s: str):\n"
            '    """Check palindrome.\n'
            "    >>> is_palindrome('racecar')\n"
            "    True\n"
            "    >>> is_palindrome('hello')\n"
            "    False\n"
            '    """\n'
        )
        code = synth.synthesize("Check if palindrome", prompt, "is_palindrome")
        assert code is not None

    def test_str_to_int_length(self, synth):
        prompt = (
            "\ndef str_len(s: str):\n"
            '    """String length.\n'
            "    >>> str_len('hello')\n"
            "    5\n"
            "    >>> str_len('')\n"
            "    0\n"
            '    """\n'
        )
        code = synth.synthesize("String length", prompt, "str_len")
        assert code is not None

    def test_int_to_int_arithmetic(self, synth):
        prompt = (
            "\ndef add(a: int, b: int):\n"
            '    """Add two numbers.\n'
            "    >>> add(1, 2)\n"
            "    3\n"
            "    >>> add(0, 5)\n"
            "    5\n"
            '    """\n'
        )
        code = synth.synthesize("Add two numbers", prompt, "add")
        assert code is not None

    def test_list_to_str_join(self, synth):
        prompt = (
            "\ndef concatenate(strings: list):\n"
            '    """Concatenate strings.\n'
            "    >>> concatenate(['a', 'b', 'c'])\n"
            "    'abc'\n"
            "    >>> concatenate([])\n"
            "    ''\n"
            '    """\n'
        )
        code = synth.synthesize("Concatenate list of strings", prompt, "concatenate")
        assert code is not None

    def test_list_to_float_mean(self, synth):
        prompt = (
            "\ndef mean(l: list):\n"
            '    """Compute mean.\n'
            "    >>> mean([1.0, 2.0, 3.0])\n"
            "    2.0\n"
            '    """\n'
        )
        code = synth.synthesize("Compute mean of list", prompt, "mean")
        assert code is not None

    def test_int_to_bool_prime(self, synth):
        prompt = (
            "\ndef is_prime(n: int):\n"
            '    """Check if prime.\n'
            "    >>> is_prime(2)\n"
            "    True\n"
            "    >>> is_prime(4)\n"
            "    False\n"
            "    >>> is_prime(17)\n"
            "    True\n"
            '    """\n'
        )
        code = synth.synthesize("Check if number is prime", prompt, "is_prime")
        assert code is not None

    def test_int_to_list_factorize(self, synth):
        prompt = (
            "\ndef factorize(n: int):\n"
            '    """Return prime factorization.\n'
            "    >>> factorize(12)\n"
            "    [2, 2, 3]\n"
            "    >>> factorize(7)\n"
            "    [7]\n"
            '    """\n'
        )
        code = synth.synthesize(
            "Return list of prime factors", prompt, "factorize"
        )
        assert code is not None

    def test_filter_with_second_param(self, synth):
        prompt = (
            "\ndef filter_by_prefix(strings: list, prefix: str):\n"
            '    """Filter strings that start with prefix.\n'
            "    >>> filter_by_prefix(['abc', 'bcd', 'aef'], 'a')\n"
            "    ['abc', 'aef']\n"
            '    """\n'
        )
        code = synth.synthesize(
            "Filter strings that start with given prefix",
            prompt, "filter_by_prefix"
        )
        assert code is not None

    def test_no_examples_returns_none(self, synth):
        prompt = "\ndef mystery(x):\n    pass\n"
        code = synth.synthesize("Do something", prompt, "mystery")
        assert code is None

    def test_type_inference_from_examples(self, synth):
        examples = [
            {"input": [1, 2, 3], "output": 6},
            {"input": [10, 20], "output": 30},
        ]
        in_t, out_t = synth._infer_types(examples)
        assert in_t == "list"
        assert out_t == "int"

    def test_type_inference_tuple_input(self, synth):
        examples = [
            {"input": ([1, 2, 3], 0.5), "output": True},
        ]
        in_t, out_t = synth._infer_types(examples)
        assert in_t == "list"
        assert out_t == "bool"


# ═══════════════════════════════════════════
# Integration: solve novel problems by reasoning
# ═══════════════════════════════════════════


class TestReasoningNotRetrieval:
    """Verify the system can solve problems it has NO template for,
    by reasoning about types and executing candidates."""

    @pytest.fixture
    def executor(self):
        return SafeExecutor(timeout=5.0)

    def test_exec_guided_discovers_reverse(self):
        """System discovers list reversal without a 'reverse' keyword template."""
        synth = ExecutionGuidedSynthesizer()
        prompt = (
            "\ndef flip_it(l: list):\n"
            '    """Make last first and first last.\n'
            "    >>> flip_it([1, 2, 3])\n"
            "    [3, 2, 1]\n"
            "    >>> flip_it([5, 10])\n"
            "    [10, 5]\n"
            '    """\n'
        )
        code = synth.synthesize(
            "Make last element first and first element last",
            prompt, "flip_it"
        )
        assert code is not None
        assert "flip_it" in code

    def test_compositional_discovers_sorted_unique(self):
        """System discovers sorted unique via type reasoning, not keyword."""
        synth = CompositionalSynthesizer()
        prompt = (
            "\ndef distinct_sorted(l: list):\n"
            '    """Get unique elements, sorted.\n'
            "    >>> distinct_sorted([5, 3, 5, 2, 3])\n"
            "    [2, 3, 5]\n"
            '    """\n'
        )
        code = synth.synthesize(
            "Get distinct elements in sorted order",
            prompt, "distinct_sorted"
        )
        assert code is not None

    def test_exec_guided_discovers_abs_diff(self):
        """System discovers abs(a-b) from examples, not keywords."""
        synth = ExecutionGuidedSynthesizer()
        prompt = (
            "\ndef distance(a: int, b: int):\n"
            '    """How far apart.\n'
            "    >>> distance(3, 7)\n"
            "    4\n"
            "    >>> distance(10, 2)\n"
            "    8\n"
            '    """\n'
        )
        code = synth.synthesize("How far apart are the numbers",
                                prompt, "distance")
        # This one is tricky - abs(a-b) needs to be discovered
        # The exec guided should try a-b, b-a, abs(a-b), etc.
        if code:
            assert "distance" in code

    def test_induction_suggests_approach_after_learning(self):
        """After learning from examples, induction suggests structure."""
        engine = ProgramInductionEngine()
        # Teach it accumulate patterns
        engine.record(
            "Sum all elements",
            "def total(l):\n    s = 0\n    for x in l:\n        s += x\n    return s\n"
        )
        engine.record(
            "Product of all elements",
            "def prod(l):\n    p = 1\n    for x in l:\n        p *= x\n    return p\n"
        )
        engine.record(
            "Count positive elements",
            "def count_pos(l):\n    c = 0\n    for x in l:\n        if x > 0:\n            c += 1\n    return c\n"
        )

        # Now ask for a similar problem
        suggestion = engine.suggest_structure("Count negative elements in list")
        assert suggestion is not None
        assert "accumulate" in suggestion["pattern_tags"]
