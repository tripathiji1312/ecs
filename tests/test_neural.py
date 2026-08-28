"""Tests for Neural Interface (mocked — no Ollama dependency)."""

import json
from unittest.mock import patch, MagicMock
from ecs.neural.interface import NeuralInterface, NeuralResponse
from ecs.neural.module import NeuralModule
from ecs.neural.prompts import PromptRegistry


def _mock_ollama_response(content, eval_count=50):
    """Create a mock requests.Response for Ollama API."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "message": {"content": content},
        "eval_count": eval_count
    }
    return mock_resp


# ═══════════════════════════════════════════
# CODE GENERATION
# ═══════════════════════════════════════════

@patch("ecs.neural.interface.requests.post")
def test_generate_code(mock_post):
    code_response = '```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```'
    mock_post.return_value = _mock_ollama_response(code_response, eval_count=20)

    iface = NeuralInterface()
    result = iface.generate_code("Write a function that reverses a string")

    assert result.code == 'def reverse_string(s: str) -> str:\n    return s[::-1]'
    assert result.confidence == 0.7
    assert result.tokens_used == 20
    assert result.latency_ms >= 0

    call_args = mock_post.call_args
    payload = call_args.kwargs["json"] if "json" in call_args.kwargs else call_args[1]["json"]
    assert payload["options"]["temperature"] == 0.3


@patch("ecs.neural.interface.requests.post")
def test_generate_code_with_strategy_and_knowledge(mock_post):
    mock_post.return_value = _mock_ollama_response("```python\ndef bs(arr, t):\n    pass\n```")

    iface = NeuralInterface()
    result = iface.generate_code(
        specification="Binary search on sorted array",
        strategy_context="Use divide-and-conquer, halve search space each step",
        retrieved_knowledge=["Binary search requires sorted input", "O(log n) time"]
    )

    assert result.code is not None
    prompt_sent = mock_post.call_args.kwargs["json"]["messages"][1]["content"]
    assert "Relevant knowledge from memory" in prompt_sent
    assert "Approach to use:" in prompt_sent
    assert "divide-and-conquer" in prompt_sent


@patch("ecs.neural.interface.requests.post")
def test_extract_code_no_markers(mock_post):
    mock_post.return_value = _mock_ollama_response("def foo():\n    return 42")

    iface = NeuralInterface()
    result = iface.generate_code("Write a function returning 42")
    assert "return 42" in result.code


@patch("ecs.neural.interface.requests.post")
def test_extract_code_generic_block(mock_post):
    mock_post.return_value = _mock_ollama_response("Here:\n```\ndef bar(): pass\n```")

    iface = NeuralInterface()
    result = iface.generate_code("Write bar")
    assert result.code == "def bar(): pass"


# ═══════════════════════════════════════════
# PROBLEM UNDERSTANDING
# ═══════════════════════════════════════════

@patch("ecs.neural.interface.requests.post")
def test_understand_problem(mock_post):
    response_text = (
        "TYPE: sorting\n"
        "FEATURES: divisible, recursion, ordered\n"
        "DIFFICULTY: medium\n"
        "SUB_PROBLEMS: partition array; recurse on halves"
    )
    mock_post.return_value = _mock_ollama_response(response_text)

    iface = NeuralInterface()
    analysis = iface.understand_problem("Sort an array using quicksort")

    assert analysis["type"] == "sorting"
    assert "divisible" in analysis["features"]
    assert "recursion" in analysis["features"]
    assert analysis["difficulty"] == "medium"
    assert len(analysis["sub_problems"]) == 2
    assert "latency_ms" in analysis


@patch("ecs.neural.interface.requests.post")
def test_understand_problem_partial_response(mock_post):
    """Handle model returning incomplete format."""
    mock_post.return_value = _mock_ollama_response("TYPE: graph\nFEATURES: graph, tree\n")

    iface = NeuralInterface()
    analysis = iface.understand_problem("Find shortest path in graph")

    assert analysis["type"] == "graph"
    assert "graph" in analysis["features"]
    assert analysis["difficulty"] == "medium"  # default
    assert analysis["sub_problems"] == []


# ═══════════════════════════════════════════
# CODE EXPLANATION
# ═══════════════════════════════════════════

@patch("ecs.neural.interface.requests.post")
def test_explain_code(mock_post):
    explanation = (
        "1. Reverses a string\n"
        "2. Uses Python slicing\n"
        "3. O(n) time, O(n) space"
    )
    mock_post.return_value = _mock_ollama_response(explanation)

    iface = NeuralInterface()
    result = iface.explain_code("def rev(s): return s[::-1]")

    assert "Reverses" in result.text
    assert result.confidence == 0.8


# ═══════════════════════════════════════════
# TEST CASE GENERATION
# ═══════════════════════════════════════════

@patch("ecs.neural.interface.requests.post")
def test_generate_test_cases(mock_post):
    response_text = (
        "INPUT: [1, 2, 3], 2\n"
        "EXPECTED: 1\n"
        "TYPE: normal\n"
        "INPUT: [], 5\n"
        "EXPECTED: -1\n"
        "TYPE: edge_case\n"
        "INPUT: [1], 1\n"
        "EXPECTED: 0\n"
        "TYPE: edge_case\n"
    )
    mock_post.return_value = _mock_ollama_response(response_text)

    iface = NeuralInterface()
    cases = iface.generate_test_cases(
        "def binary_search(arr: List[int], target: int) -> int",
        "Search for target in sorted array, return index or -1"
    )

    assert len(cases) == 3
    assert cases[0]["type"] == "normal"
    assert cases[1]["type"] == "edge_case"
    assert cases[1]["input"] == "[], 5"
    assert cases[1]["expected"] == "-1"


@patch("ecs.neural.interface.requests.post")
def test_generate_test_cases_empty_response(mock_post):
    mock_post.return_value = _mock_ollama_response("I can't generate tests for that.")

    iface = NeuralInterface()
    cases = iface.generate_test_cases("def foo()", "does something")
    assert cases == []


# ═══════════════════════════════════════════
# AVAILABILITY & ERROR HANDLING
# ═══════════════════════════════════════════

@patch("ecs.neural.interface.requests.get")
def test_is_available_true(mock_get):
    mock_get.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value={
            "models": [{"name": "qwen2.5-coder:0.5b"}]
        })
    )
    iface = NeuralInterface()
    assert iface.is_available()


@patch("ecs.neural.interface.requests.get")
def test_is_available_no_model(mock_get):
    mock_get.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value={"models": [{"name": "llama3:7b"}]})
    )
    iface = NeuralInterface()
    assert not iface.is_available()


@patch("ecs.neural.interface.requests.get")
def test_is_available_connection_error(mock_get):
    import requests as req
    mock_get.side_effect = req.ConnectionError("refused")
    iface = NeuralInterface()
    assert not iface.is_available()


@patch("ecs.neural.interface.requests.post")
def test_api_error_raises(mock_post):
    mock_post.return_value = MagicMock(status_code=500)

    iface = NeuralInterface()
    try:
        iface.generate_code("anything")
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "500" in str(e)


# ═══════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════

@patch("ecs.neural.interface.requests.post")
def test_stats_tracking(mock_post):
    mock_post.return_value = _mock_ollama_response("```python\npass\n```", eval_count=10)

    iface = NeuralInterface()
    iface.generate_code("a")
    iface.generate_code("b")

    stats = iface.get_stats()
    assert stats["total_calls"] == 2
    assert stats["total_tokens"] == 20
    assert stats["model"] == "qwen2.5-coder:0.5b"


# ═══════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════

@patch("ecs.neural.interface.requests.post")
def test_system_prompt_sent(mock_post):
    mock_post.return_value = _mock_ollama_response("```python\npass\n```")

    iface = NeuralInterface()
    iface.generate_code("test")

    payload = mock_post.call_args.kwargs["json"]
    messages = payload["messages"]
    assert messages[0]["role"] == "system"
    assert "language interface" in messages[0]["content"]
    assert messages[1]["role"] == "user"


# ═══════════════════════════════════════════
# WORKSPACE INTEGRATION
# ═══════════════════════════════════════════

@patch("ecs.neural.interface.requests.post")
def test_neural_module_bid(mock_post):
    """Test NeuralModule generates bids for the workspace."""
    from ecs.workspace.global_workspace import ModuleType, WorkspaceBid

    code_response = '```python\ndef solve(n): return n * 2\n```'
    mock_post.return_value = _mock_ollama_response(code_response)

    iface = NeuralInterface()
    result = iface.generate_code("Double the input")

    bid = WorkspaceBid(
        module_type=ModuleType.NEURAL,
        module_name="NeuralInterface",
        content=f"Generated code: {result.code[:50]}",
        payload={"code": result.code, "latency_ms": result.latency_ms},
        free_energy_reduction=result.confidence,
        confidence=result.confidence
    )

    assert bid.module_type == ModuleType.NEURAL
    assert bid.free_energy_reduction == 0.7
    assert "solve" in bid.payload["code"]


# ═══════════════════════════════════════════
# NEURAL MODULE (SPECIALIST BIDDER)
# ═══════════════════════════════════════════

@patch("ecs.neural.interface.requests.get")
@patch("ecs.neural.interface.requests.post")
def test_neural_module_understanding_bid(mock_post, mock_get):
    """NeuralModule bids with problem understanding."""
    mock_get.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value={"models": [{"name": "qwen2.5-coder:0.5b"}]})
    )
    mock_post.return_value = _mock_ollama_response(
        "TYPE: sorting\nFEATURES: divisible, recursion\nDIFFICULTY: medium\nSUB_PROBLEMS: none"
    )

    iface = NeuralInterface()
    module = NeuralModule(iface)

    context = {"current_problem": "Sort an array using merge sort"}
    bid = module.generate_bid(context)

    assert bid.free_energy_reduction == 0.6
    assert bid.confidence == 0.7
    assert "sorting" in bid.content
    assert "divisible" in bid.payload["features"]


@patch("ecs.neural.interface.requests.get")
def test_neural_module_offline(mock_get):
    """NeuralModule returns zero bid when offline."""
    import requests as req
    mock_get.side_effect = req.ConnectionError("refused")

    iface = NeuralInterface()
    module = NeuralModule(iface)

    bid = module.generate_bid({"current_problem": "anything"})
    assert bid.free_energy_reduction == 0.0
    assert "offline" in bid.content


@patch("ecs.neural.interface.requests.post")
def test_neural_module_code_bid_valid(mock_post):
    """NeuralModule generates code bid with syntax validation."""
    mock_post.return_value = _mock_ollama_response(
        '```python\ndef add(a, b):\n    return a + b\n```'
    )

    iface = NeuralInterface()
    module = NeuralModule(iface)

    context = {"current_problem": "Add two numbers"}
    bid = module.generate_code_bid(context)

    assert bid.payload["syntax_valid"] is True
    assert bid.confidence == 0.7
    assert bid.free_energy_reduction == 0.5
    assert "valid" in bid.content


@patch("ecs.neural.interface.requests.post")
def test_neural_module_code_bid_invalid_syntax(mock_post):
    """NeuralModule detects syntax errors and lowers confidence."""
    mock_post.return_value = _mock_ollama_response(
        '```python\ndef broken(:\n    return\n```'
    )

    iface = NeuralInterface()
    module = NeuralModule(iface)

    context = {"current_problem": "Something"}
    bid = module.generate_code_bid(context)

    assert bid.payload["syntax_valid"] is False
    assert bid.confidence == 0.2
    assert bid.free_energy_reduction == 0.2
    assert "SYNTAX ERROR" in bid.content


def test_neural_module_calibrate_confidence():
    iface = NeuralInterface()
    module = NeuralModule(iface)

    conf = module.calibrate_confidence({
        "syntax_valid": True,
        "tests_passed": True,
        "test_count": 5
    })
    assert abs(conf - 1.0) < 0.01

    conf = module.calibrate_confidence({
        "syntax_valid": True,
        "tests_passed": False,
        "test_count": 1
    })
    assert abs(conf - 0.6) < 0.01


# ═══════════════════════════════════════════
# REFINE CODE
# ═══════════════════════════════════════════

@patch("ecs.neural.interface.requests.post")
def test_refine_code(mock_post):
    mock_post.return_value = _mock_ollama_response(
        '```python\ndef add(a, b):\n    return a + b\n```'
    )

    iface = NeuralInterface()
    result = iface.refine_code(
        previous_code="def add(a, b):\n    return a - b",
        error_message="Expected 5 but got -1 for add(2, 3)",
        original_specification="Add two numbers"
    )

    assert result.code is not None
    assert result.confidence == 0.6
    prompt_sent = mock_post.call_args.kwargs["json"]["messages"][1]["content"]
    assert "Fix the bug" in prompt_sent
    assert "a - b" in prompt_sent


# ═══════════════════════════════════════════
# TOKEN BUDGET
# ═══════════════════════════════════════════

def test_estimate_complexity():
    iface = NeuralInterface()

    assert iface.estimate_complexity("reverse a string") == "easy"
    assert iface.estimate_complexity("sort and search the array") == "medium"
    assert iface.estimate_complexity(
        "find the shortest path in a recursive graph traversal"
    ) == "hard"


def test_token_budget():
    iface = NeuralInterface()
    assert iface.get_token_budget("easy") == 512
    assert iface.get_token_budget("medium") == 1024
    assert iface.get_token_budget("hard") == 2048
    assert iface.get_token_budget("unknown") == 1024


# ═══════════════════════════════════════════
# PROMPT REGISTRY
# ═══════════════════════════════════════════

def test_prompt_registry_strategy():
    registry = PromptRegistry()

    prompt = registry.get_strategy_prompt("greedy")
    assert prompt is not None
    assert "locally optimal" in prompt

    assert registry.get_strategy_prompt("nonexistent") is None


def test_prompt_registry_build_prompt():
    registry = PromptRegistry()

    prompt = registry.build_generation_prompt(
        specification="Sort the array",
        strategy_name="greedy",
        retrieved_knowledge=["Arrays can be sorted in O(n log n)"]
    )

    assert "locally optimal" in prompt
    assert "Sort the array" in prompt
    assert "O(n log n)" in prompt


def test_prompt_registry_no_strategy():
    registry = PromptRegistry()

    prompt = registry.build_generation_prompt(
        specification="Do something",
        strategy_name=None,
        retrieved_knowledge=None
    )

    assert "Do something" in prompt
    assert "locally optimal" not in prompt


def test_prompt_registry_custom_template():
    registry = PromptRegistry()
    registry.register_template("my-custom", "Use a custom approach:\n- Step 1\n- Step 2")

    prompt = registry.get_strategy_prompt("my-custom")
    assert "custom approach" in prompt
