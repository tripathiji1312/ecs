"""Tests for Global Workspace System."""

import time
from ecs.workspace.global_workspace import (
    GlobalWorkspace, MemoryModule, StrategyModule,
    VerificationModule, ModuleType, WorkspaceBid, SelfModel
)
from ecs.memory.hdc import HDCMemory, MemoryType


def _create_test_memory():
    """Create an HDC memory with test data."""
    mem = HDCMemory(dimension=10000)

    test_items = [
        ("binary_search", "binary search algorithm sorted array", MemoryType.SEMANTIC,
         "Binary search: divide sorted array, check middle, recurse"),
        ("quick_sort", "quicksort algorithm partition pivot", MemoryType.SEMANTIC,
         "Quicksort: partition around pivot, recurse on halves"),
        ("retry_pattern", "retry with exponential backoff", MemoryType.PROCEDURAL,
         "Retry pattern: catch exception, wait, retry with exponential delay"),
        ("debug_strategy", "isolate problem reproduce minimize", MemoryType.STRATEGIC,
         "Debug strategy: isolate, reproduce, minimize test case"),
    ]

    for item_id, text, mtype, content in test_items:
        vec = mem.encode_text(text)
        mem.store(item_id, vec, mtype, content)

    return mem


# ═══════════════════════════════════════════
# AUCTION TESTS
# ═══════════════════════════════════════════

def test_workspace_auction():
    """Test that the workspace auction selects relevant information."""
    workspace = GlobalWorkspace()
    memory = _create_test_memory()

    memory_module = MemoryModule(memory)
    workspace.register_module("memory", ModuleType.MEMORY,
                              memory_module.generate_bid)

    context = {
        "current_problem": "sort array efficiently",
        "problem_features": ["sorting", "comparison"]
    }

    bids = [module["generate_bid"](context)
            for module in workspace.modules.values()]
    winners = workspace.conduct_auction(bids)

    memory_won = any(b.module_type == ModuleType.MEMORY for b in winners)
    assert memory_won
    assert len(winners) <= workspace.capacity


def test_auction_respects_capacity():
    """Test that auction never exceeds workspace capacity."""
    workspace = GlobalWorkspace()

    bids = [
        WorkspaceBid(
            module_type=ModuleType.MEMORY,
            module_name=f"mod_{i}",
            content=f"Bid {i}",
            payload=None,
            free_energy_reduction=0.5 + i * 0.01
        )
        for i in range(20)
    ]

    winners = workspace.conduct_auction(bids)
    assert len(winners) == workspace.CAPACITY


def test_auction_priority_ordering():
    """Test that higher free-energy bids win over lower ones."""
    workspace = GlobalWorkspace()

    low_bid = WorkspaceBid(
        module_type=ModuleType.MEMORY, module_name="low",
        content="Low priority", payload=None, free_energy_reduction=0.2
    )
    high_bid = WorkspaceBid(
        module_type=ModuleType.VERIFICATION, module_name="high",
        content="High priority", payload=None, free_energy_reduction=0.9
    )

    winners = workspace.conduct_auction([low_bid, high_bid])
    assert winners[0].module_name == "high"


def test_attention_weights():
    """Test that attention weights modulate bid effectiveness."""
    workspace = GlobalWorkspace()

    # Verification has 1.5x weight, self_model has 0.8x
    verification_bid = WorkspaceBid(
        module_type=ModuleType.VERIFICATION, module_name="verifier",
        content="Verification result", payload=None,
        free_energy_reduction=0.5  # effective: 0.5 * 1.5 = 0.75
    )
    selfmodel_bid = WorkspaceBid(
        module_type=ModuleType.SELF_MODEL, module_name="self",
        content="Self insight", payload=None,
        free_energy_reduction=0.7  # effective: 0.7 * 0.8 = 0.56
    )

    winners = workspace.conduct_auction([verification_bid, selfmodel_bid])
    assert winners[0].module_name == "verifier"


def test_ignition_threshold():
    """Test that weak bids don't reach consciousness."""
    workspace = GlobalWorkspace()

    weak_bids = [
        WorkspaceBid(
            module_type=ModuleType.MEMORY, module_name=f"weak_{i}",
            content="Weak signal", payload=None,
            free_energy_reduction=0.05
        )
        for i in range(10)
    ]

    winners = workspace.conduct_auction(weak_bids)
    assert len(winners) == 0

    # One strong bid among weak ones
    strong_bid = WorkspaceBid(
        module_type=ModuleType.VERIFICATION, module_name="strong",
        content="Strong signal", payload=None,
        free_energy_reduction=0.9
    )
    winners = workspace.conduct_auction(weak_bids + [strong_bid])
    assert len(winners) == 1
    assert winners[0].module_name == "strong"


def test_urgency_boost():
    """Test that starved modules get urgency boost next round."""
    workspace = GlobalWorkspace()

    # Round 1: module_a wins, module_b is starved
    bid_a = WorkspaceBid(
        module_type=ModuleType.VERIFICATION, module_name="mod_a",
        content="A wins", payload=None, free_energy_reduction=0.5
    )
    bid_b = WorkspaceBid(
        module_type=ModuleType.MEMORY, module_name="mod_b",
        content="B loses", payload=None, free_energy_reduction=0.3,
        urgency=0.8
    )

    # Only 1 slot for this test
    workspace.capacity = 1
    workspace.conduct_auction([bid_a, bid_b])
    assert "mod_b" in workspace._starved_modules

    # Round 2: mod_b gets urgency boost (0.3 * 1.0 * (1.0 + 0.8*0.5) = 0.42)
    # mod_a has no boost: 0.5 * 1.5 = 0.75 (still wins, but gap closes)
    # Let's make them closer so urgency flips the result
    bid_a2 = WorkspaceBid(
        module_type=ModuleType.MEMORY, module_name="mod_a",
        content="A bids", payload=None, free_energy_reduction=0.11
    )
    bid_b2 = WorkspaceBid(
        module_type=ModuleType.MEMORY, module_name="mod_b",
        content="B bids with urgency", payload=None,
        free_energy_reduction=0.11, urgency=0.9
    )

    winners = workspace.conduct_auction([bid_a2, bid_b2])
    # mod_b should win: 0.11 * 1.0 * 1.45 = 0.1595 vs mod_a: 0.11 * 1.0 = 0.11
    assert winners[0].module_name == "mod_b"

    workspace.capacity = GlobalWorkspace.CAPACITY


# ═══════════════════════════════════════════
# SELF-MODEL TESTS
# ═══════════════════════════════════════════

def test_self_model_tracking():
    """Test that self-model tracks performance."""
    workspace = GlobalWorkspace()

    for _ in range(5):
        workspace.update_self_model({"success": True, "domain": "sorting"})

    for _ in range(2):
        workspace.update_self_model({"success": False, "domain": "graphs"})

    sm = workspace.self_model
    assert sm.tasks_attempted == 7
    assert sm.tasks_succeeded == 5
    assert sm.strengths["sorting"] > 0.3
    assert sm.weaknesses["graphs"] > 0.1
    assert sm.stuck_counter == 2


def test_self_model_confidence_update():
    """Test Bayesian confidence update."""
    sm = SelfModel()
    assert sm.confidence_level == 0.5

    sm.update_confidence(1.0)
    assert sm.confidence_level > 0.5

    sm.update_confidence(0.0)
    assert sm.confidence_level < 0.65


def test_self_model_decay():
    """Test that strengths/weaknesses decay over time."""
    workspace = GlobalWorkspace()

    for _ in range(10):
        workspace.update_self_model({"success": True, "domain": "sorting"})

    initial_strength = workspace.self_model.strengths["sorting"]

    for _ in range(50):
        workspace.update_self_model({"success": True, "domain": "graphs"})

    decayed_strength = workspace.self_model.strengths.get("sorting", 0)
    assert decayed_strength < initial_strength, \
        f"Should decay: {initial_strength:.4f} -> {decayed_strength:.4f}"


# ═══════════════════════════════════════════
# METACOGNITION TESTS
# ═══════════════════════════════════════════

def test_metacognition_stuck():
    """Test metacognitive insight when stuck."""
    workspace = GlobalWorkspace()
    for _ in range(5):
        workspace.update_self_model({"success": False, "domain": "hard_problem"})

    insight = workspace.self_model.reflect()
    assert insight is not None
    assert "stuck" in insight.lower()


def test_metacognition_confident():
    """Test metacognitive insight when confident."""
    workspace = GlobalWorkspace()
    workspace.self_model.confidence_level = 0.9
    workspace.self_model.active_strategy = "divide-and-conquer"

    insight = workspace.self_model.reflect()
    assert insight is not None
    assert "confident" in insight.lower()


def test_metacognition_uncertain():
    """Test metacognitive insight when uncertain."""
    workspace = GlobalWorkspace()
    workspace.self_model.confidence_level = 0.1

    insight = workspace.self_model.reflect()
    assert insight is not None
    assert "low confidence" in insight.lower()


# ═══════════════════════════════════════════
# BROADCAST TESTS
# ═══════════════════════════════════════════

def test_broadcast():
    """Test global broadcast mechanism."""
    workspace = GlobalWorkspace()
    memory = _create_test_memory()

    memory_module = MemoryModule(memory)
    workspace.register_module("memory", ModuleType.MEMORY,
                              memory_module.generate_bid)

    workspace.self_model.current_goal = "Implement efficient sorting"
    workspace.self_model.active_strategy = "divide-and-conquer"

    context = {"current_problem": "sort large array", "problem_features": ["sorting"]}
    bids = [module["generate_bid"](context)
            for module in workspace.modules.values()]
    workspace.conduct_auction(bids)

    broadcast = workspace.broadcast()

    assert "conscious_contents" in broadcast
    assert "self_state" in broadcast
    assert broadcast["self_state"]["goal"] == "Implement efficient sorting"
    assert "timestamp" in broadcast


def test_broadcast_subscribers():
    """Test that subscribers receive broadcast data."""
    workspace = GlobalWorkspace()
    received = []

    workspace.subscribe_to_broadcast(lambda data: received.append(data))

    bids = [WorkspaceBid(
        module_type=ModuleType.MEMORY, module_name="test",
        content="Test bid", payload=None, free_energy_reduction=0.5
    )]
    workspace.conduct_auction(bids)
    workspace.broadcast()

    assert len(received) == 1
    assert "conscious_contents" in received[0]


def test_broadcast_subscriber_error_isolated():
    """Test that a failing subscriber doesn't crash the broadcast."""
    workspace = GlobalWorkspace()

    def bad_subscriber(data):
        raise RuntimeError("subscriber failure")

    workspace.subscribe_to_broadcast(bad_subscriber)

    bids = [WorkspaceBid(
        module_type=ModuleType.MEMORY, module_name="test",
        content="Test", payload=None, free_energy_reduction=0.5
    )]
    workspace.conduct_auction(bids)

    # Should not raise
    broadcast = workspace.broadcast()
    assert broadcast is not None


def test_broadcast_history():
    """Test that broadcast history is recorded."""
    workspace = GlobalWorkspace()

    bids = [WorkspaceBid(
        module_type=ModuleType.MEMORY, module_name="test",
        content="Test bid", payload=None, free_energy_reduction=0.5
    )]

    workspace.conduct_auction(bids)
    workspace.conduct_auction(bids)

    assert len(workspace.broadcast_history) == 2
    assert "winners" in workspace.broadcast_history[0]
    assert "total_reduction" in workspace.broadcast_history[0]


def test_history_cap():
    """Test that history doesn't grow unbounded."""
    workspace = GlobalWorkspace()

    bids = [WorkspaceBid(
        module_type=ModuleType.MEMORY, module_name="test",
        content="Test", payload=None, free_energy_reduction=0.5
    )]

    for _ in range(1100):
        workspace.conduct_auction(bids)

    assert len(workspace.broadcast_history) <= GlobalWorkspace.MAX_HISTORY


# ═══════════════════════════════════════════
# STUCK DETECTION TESTS
# ═══════════════════════════════════════════

def test_stuck_detection():
    """Test that workspace detects when system is stuck."""
    workspace = GlobalWorkspace()

    low_bids = [WorkspaceBid(
        module_type=ModuleType.MEMORY, module_name="test",
        content="Low relevance", payload=None,
        free_energy_reduction=0.02
    )]

    for _ in range(5):
        workspace.conduct_auction(low_bids)

    assert workspace.self_model.stuck_counter >= 3


def test_stuck_recovery():
    """Test that high-value auctions reduce stuck counter."""
    workspace = GlobalWorkspace()

    low_bids = [WorkspaceBid(
        module_type=ModuleType.MEMORY, module_name="t",
        content="Low", payload=None, free_energy_reduction=0.01
    )]
    for _ in range(5):
        workspace.conduct_auction(low_bids)

    stuck_before = workspace.self_model.stuck_counter

    high_bids = [WorkspaceBid(
        module_type=ModuleType.VERIFICATION, module_name="t",
        content="High", payload=None, free_energy_reduction=0.5
    )]
    workspace.conduct_auction(high_bids)
    assert workspace.self_model.stuck_counter < stuck_before


# ═══════════════════════════════════════════
# ATTENTION ADAPTATION TESTS
# ═══════════════════════════════════════════

def test_attention_adaptation():
    """Test that attention weights adapt based on outcomes."""
    workspace = GlobalWorkspace()

    for _ in range(10):
        workspace.update_attention_weights({
            ModuleType.VERIFICATION: True,
            ModuleType.MEMORY: False
        })

    v_weight = workspace.attention_weights[ModuleType.VERIFICATION]
    m_weight = workspace.attention_weights[ModuleType.MEMORY]

    assert v_weight > 1.5
    assert m_weight < 1.0
    assert v_weight <= 2.5
    assert m_weight >= 0.5


def test_attention_bounds():
    """Test that weights stay within bounds even under extreme updates."""
    workspace = GlobalWorkspace()

    for _ in range(100):
        workspace.update_attention_weights({ModuleType.MEMORY: True})

    assert workspace.attention_weights[ModuleType.MEMORY] <= 2.5

    for _ in range(200):
        workspace.update_attention_weights({ModuleType.MEMORY: False})

    assert workspace.attention_weights[ModuleType.MEMORY] >= 0.5


def test_mean_reversion():
    """Test that mean reversion pulls weights back toward defaults."""
    workspace = GlobalWorkspace()

    # Push memory weight high
    for _ in range(20):
        workspace.update_attention_weights({ModuleType.MEMORY: True})

    high_weight = workspace.attention_weights[ModuleType.MEMORY]

    # Apply mean reversion many times
    for _ in range(1000):
        workspace.apply_mean_reversion()

    reverted_weight = workspace.attention_weights[ModuleType.MEMORY]
    assert reverted_weight < high_weight


# ═══════════════════════════════════════════
# BID OUTCOME TRACKING
# ═══════════════════════════════════════════

def test_bid_outcome_tracking():
    """Test that bid outcomes feed into reliability metric."""
    workspace = GlobalWorkspace()

    winning_bid = WorkspaceBid(
        module_type=ModuleType.MEMORY, module_name="MemoryModule",
        content="Relevant memory", payload=None, free_energy_reduction=0.8
    )

    for _ in range(5):
        workspace.record_bid_outcome([winning_bid], problem_success=True)

    reliability = workspace.get_module_reliability()
    assert "MemoryModule" in reliability
    assert reliability["MemoryModule"] == 1.0


def test_bid_outcome_mixed():
    """Test reliability with mixed outcomes."""
    workspace = GlobalWorkspace()

    bid = WorkspaceBid(
        module_type=ModuleType.STRATEGY, module_name="StrategyModule",
        content="Strategy", payload=None, free_energy_reduction=0.7
    )

    for _ in range(3):
        workspace.record_bid_outcome([bid], problem_success=True)
    for _ in range(2):
        workspace.record_bid_outcome([bid], problem_success=False)

    reliability = workspace.get_module_reliability()
    assert abs(reliability["StrategyModule"] - 0.6) < 0.01


def test_bid_outcome_log_cap():
    """Test that bid outcome log doesn't grow unbounded."""
    workspace = GlobalWorkspace()

    bid = WorkspaceBid(
        module_type=ModuleType.MEMORY, module_name="test",
        content="Test", payload=None, free_energy_reduction=0.5
    )

    for _ in range(600):
        workspace.record_bid_outcome([bid], problem_success=True)

    assert len(workspace.bid_outcome_log) <= 500


# ═══════════════════════════════════════════
# MODULE TESTS
# ═══════════════════════════════════════════

def test_memory_module_with_empty_context():
    """Test memory module handles empty context gracefully."""
    memory = _create_test_memory()
    module = MemoryModule(memory)

    bid = module.generate_bid({})
    assert bid.free_energy_reduction == 0.0
    assert bid.payload is None


def test_verification_module_no_result():
    """Test verification module before any verification."""
    module = VerificationModule()
    bid = module.generate_bid({})
    assert bid.free_energy_reduction == 0.2
    assert "No verification" in bid.content


def test_verification_module_with_result():
    """Test verification module after success and failure."""
    module = VerificationModule()

    module.last_result = {"success": True, "message": "All tests pass", "confidence": 0.95}
    bid = module.generate_bid({})
    assert bid.free_energy_reduction == 0.9
    assert "PASSED" in bid.content

    module.last_result = {"success": False, "message": "Assertion error"}
    bid = module.generate_bid({})
    assert bid.free_energy_reduction == 0.8
    assert "FAILED" in bid.content


def test_multiple_modules_compete():
    """Test auction with multiple module types competing."""
    workspace = GlobalWorkspace()
    memory = _create_test_memory()

    memory_module = MemoryModule(memory)
    verification_module = VerificationModule()
    verification_module.last_result = {
        "success": False, "message": "Test failed", "confidence": 0.9
    }

    workspace.register_module("memory", ModuleType.MEMORY,
                              memory_module.generate_bid)
    workspace.register_module("verification", ModuleType.VERIFICATION,
                              verification_module.generate_bid)

    context = {"current_problem": "sort array", "problem_features": ["sorting"]}
    bids = [module["generate_bid"](context)
            for module in workspace.modules.values()]
    winners = workspace.conduct_auction(bids)

    # Verification (0.8 * 1.5 = 1.2) should beat memory
    assert winners[0].module_type == ModuleType.VERIFICATION


def test_workspace_summary():
    """Test human-readable workspace summary."""
    workspace = GlobalWorkspace()
    workspace.self_model.current_goal = "Test goal"
    workspace.self_model.active_strategy = "test-strategy"

    bids = [WorkspaceBid(
        module_type=ModuleType.MEMORY, module_name="mem",
        content="Retrieved something", payload=None,
        free_energy_reduction=0.6
    )]
    workspace.conduct_auction(bids)

    summary = workspace.get_workspace_summary()
    assert "GLOBAL WORKSPACE STATE" in summary
    assert "Test goal" in summary
    assert "test-strategy" in summary
    assert "Retrieved something" in summary


def test_pattern_tracking():
    """Test that patterns are accumulated in self-model."""
    workspace = GlobalWorkspace()

    workspace.update_self_model({
        "success": True, "domain": "sorting",
        "pattern": "Divide-and-conquer works well for sorting"
    })
    workspace.update_self_model({
        "success": True, "domain": "sorting",
        "pattern": "Recursive solutions benefit from memoization"
    })

    assert len(workspace.self_model.patterns_noticed) == 2


# ═══════════════════════════════════════════
# PERSISTENCE TESTS
# ═══════════════════════════════════════════

def test_self_model_persistence(tmp_path):
    """Test save and load of self-model state."""
    from ecs.workspace.persistence import SelfModelPersistence

    filepath = str(tmp_path / "self_model.pkl")
    persistence = SelfModelPersistence(filepath)

    workspace = GlobalWorkspace()
    workspace.self_model.current_goal = "Solve sorting problems"
    workspace.self_model.confidence_level = 0.85
    workspace.self_model.tasks_attempted = 42
    workspace.self_model.tasks_succeeded = 35

    for _ in range(5):
        workspace.update_attention_weights({ModuleType.VERIFICATION: True})

    persistence.save(workspace)

    # Load into new workspace
    workspace2 = GlobalWorkspace()
    loaded = persistence.load(workspace2)

    assert loaded is True
    assert workspace2.self_model.current_goal == "Solve sorting problems"
    assert workspace2.self_model.confidence_level == 0.85
    assert workspace2.self_model.tasks_attempted == 42
    assert workspace2.self_model.tasks_succeeded == 35
    assert workspace2.attention_weights[ModuleType.VERIFICATION] > 1.5
