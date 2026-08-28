"""Tests for Strategy Library."""

from ecs.strategies.library import (
    StrategyLibrary, MetaStrategy, ProblemFeature, STRATEGIES,
    StrategyDAG, StrategyRelation, CompositionMode,
    build_strategy_dag, StrategyCompositor,
)


# ═══════════════════════════════════════════
# BASIC MATCHING
# ═══════════════════════════════════════════

def test_strategy_matching_exact():
    """Test that strategies match when all required features are present."""
    lib = StrategyLibrary()
    features = ["divisible", "recursion"]
    applicable = lib.find_applicable(features)
    names = [s["name"] for s in applicable]
    assert "divide-and-conquer" in names


def test_strategy_matching_sorting():
    lib = StrategyLibrary()
    features = ["sorting", "divisible", "recursion"]
    applicable = lib.find_applicable(features)
    names = [s["name"] for s in applicable]
    assert "divide-and-conquer" in names


def test_strategy_matching_graph():
    lib = StrategyLibrary()
    features = ["graph", "tree"]
    applicable = lib.find_applicable(features)
    names = [s["name"] for s in applicable]
    assert "graph-traversal" in names


def test_strategy_matching_no_match():
    lib = StrategyLibrary()
    features = ["nonexistent_feature"]
    applicable = lib.find_applicable(features)
    assert isinstance(applicable, list)


# ═══════════════════════════════════════════
# PARTIAL MATCHING
# ═══════════════════════════════════════════

def test_partial_match():
    lib = StrategyLibrary()
    features = ["recursion"]
    matches = lib.find_partial_matches(features, min_score=0.4)
    names = [m["name"] for m in matches]
    assert "divide-and-conquer" in names

    dc = next(m for m in matches if m["name"] == "divide-and-conquer")
    assert dc["match_score"] == 0.5


def test_partial_match_score():
    lib = StrategyLibrary()
    dc = lib.strategies["divide-and-conquer"]

    score = lib.partial_match_score(dc, {ProblemFeature.DIVISIBLE, ProblemFeature.RECURSION})
    assert score == 1.0

    score = lib.partial_match_score(dc, {ProblemFeature.RECURSION})
    assert score == 0.5

    score = lib.partial_match_score(dc, {ProblemFeature.STRING})
    assert score == 0.0


# ═══════════════════════════════════════════
# COMPATIBILITY & COMPOSITION
# ═══════════════════════════════════════════

def test_strategy_compatibility():
    lib = StrategyLibrary()
    assert not lib.check_compatibility("divide-and-conquer", "greedy")
    assert not lib.check_compatibility("greedy", "divide-and-conquer")
    assert not lib.check_compatibility("dynamic-programming", "greedy")
    assert lib.check_compatibility("divide-and-conquer", "graph-traversal")
    assert lib.check_compatibility("two-pointer", "binary-search")


def test_strategy_compatibility_nonexistent():
    lib = StrategyLibrary()
    assert not lib.check_compatibility("divide-and-conquer", "nonexistent")


def test_strategy_composition():
    lib = StrategyLibrary()
    features = ["searching", "ordered"]
    combos = lib.get_compatible_combinations("binary-search", features)
    combo_names = [combo[1] for combo in combos]
    assert "two-pointer" in combo_names


def test_compositor_known_template():
    """Test that known compositions use templates."""
    lib = StrategyLibrary()
    result = lib.compositor.compose("binary-search", "two-pointer")
    assert result["mode"] == CompositionMode.HIERARCHICAL
    assert result["confidence"] == 0.8
    assert "holes" in result


def test_compositor_unknown_pair():
    """Test fallback composition for unknown pairs."""
    lib = StrategyLibrary()
    result = lib.compositor.compose("graph-traversal", "backtracking")
    assert result["mode"] == CompositionMode.HIERARCHICAL
    assert result["confidence"] == 0.5
    assert "strategies" in result


# ═══════════════════════════════════════════
# STRATEGY DAG
# ═══════════════════════════════════════════

def test_strategy_dag_ordering():
    """Test topological ordering of strategies."""
    dag = build_strategy_dag()
    order = dag.get_preferred_order(["greedy", "dynamic-programming"])
    assert order.index("greedy") < order.index("dynamic-programming")


def test_strategy_dag_synergistic():
    dag = build_strategy_dag()
    result = dag.check_composition("binary-search", "two-pointer")
    assert result is not None
    assert "Synergistic" in result


def test_strategy_dag_antagonistic():
    dag = build_strategy_dag()
    result = dag.check_composition("greedy", "dynamic-programming")
    assert result is not None
    assert "Cannot" in result


def test_strategy_dag_no_relation():
    dag = build_strategy_dag()
    result = dag.check_composition("graph-traversal", "exchange-argument")
    assert result is None


def test_dag_integrated_with_library():
    """Test that DAG blocks antagonistic compositions via check_compatibility."""
    lib = StrategyLibrary()
    # greedy and DP are antagonistic in DAG
    assert not lib.check_compatibility("greedy", "dynamic-programming")


# ═══════════════════════════════════════════
# LEARNING & STATISTICS
# ═══════════════════════════════════════════

def test_strategy_learning():
    lib = StrategyLibrary()
    initial_rate = lib.strategies["greedy"].success_rate
    for _ in range(5):
        lib.record_usage("greedy", False, ["optimization"])
    assert lib.strategies["greedy"].success_rate < initial_rate


def test_strategy_learning_success():
    lib = StrategyLibrary()
    initial_rate = lib.strategies["two-pointer"].success_rate
    for _ in range(10):
        lib.record_usage("two-pointer", True, ["searching", "ordered"])
    assert lib.strategies["two-pointer"].success_rate > initial_rate


def test_strategy_statistics():
    lib = StrategyLibrary()
    lib.record_usage("greedy", True, ["optimization"])
    lib.record_usage("greedy", True, ["optimization"])
    lib.record_usage("greedy", False, ["optimization"])
    lib.record_usage("binary-search", True, ["searching"])

    stats = lib.get_strategy_statistics()
    assert stats["greedy"]["uses"] == 3
    assert stats["greedy"]["successes"] == 2
    assert abs(stats["greedy"]["win_rate"] - 2 / 3) < 0.01
    assert stats["binary-search"]["uses"] == 1
    assert stats["divide-and-conquer"]["win_rate"] is None


def test_usage_history_tracking():
    lib = StrategyLibrary()
    lib.record_usage("greedy", True, ["optimization", "monotonic"])
    lib.record_usage("greedy", False, ["optimization"])
    assert len(lib.usage_history) == 2


# ═══════════════════════════════════════════
# GAP DETECTION
# ═══════════════════════════════════════════

def test_gap_detection():
    """Test detecting strategy coverage gaps."""
    lib = StrategyLibrary()
    for _ in range(10):
        lib.record_usage("greedy", False, ["geometric", "optimization"])

    gaps = lib.detect_strategy_gaps()
    assert len(gaps) > 0
    assert any(g["feature"] == "geometric" for g in gaps)


def test_suggest_new_strategy():
    lib = StrategyLibrary()
    for _ in range(10):
        lib.record_usage("greedy", False, ["geometric", "optimization"])

    suggestion = lib.suggest_new_strategy()
    assert suggestion is not None
    assert "geometric" in suggestion["suggestion"]


def test_no_gaps_when_all_succeed():
    lib = StrategyLibrary()
    for _ in range(10):
        lib.record_usage("binary-search", True, ["searching", "ordered"])
    gaps = lib.detect_strategy_gaps()
    assert len(gaps) == 0


# ═══════════════════════════════════════════
# HDC ENCODING
# ═══════════════════════════════════════════

def test_strategy_hdc_encoding():
    """Test encoding strategies as HDC vectors."""
    from ecs.memory.hdc import HDCMemory

    lib = StrategyLibrary()
    mem = HDCMemory(dimension=10000)

    dc_vec = lib.encode_strategy_to_hdc(mem, "divide-and-conquer")
    dp_vec = lib.encode_strategy_to_hdc(mem, "dynamic-programming")
    tp_vec = lib.encode_strategy_to_hdc(mem, "two-pointer")

    # All should be binary vectors of the right size
    assert dc_vec.shape == (10000,)
    assert dp_vec.shape == (10000,)
    assert tp_vec.shape == (10000,)

    # Vectors should be valid binary (all 0s and 1s)
    import numpy as np
    assert set(np.unique(dc_vec)).issubset({0, 1})

    # Different strategies should NOT be identical
    dc_dp_sim = mem._similarity(dc_vec, dp_vec)
    assert dc_dp_sim < 1.0

    # Encoding produces meaningful vectors (not all zeros or all ones)
    assert 0.3 < np.mean(dc_vec) < 0.7


def test_find_similar_strategies():
    """Test finding similar strategies via HDC."""
    from ecs.memory.hdc import HDCMemory

    lib = StrategyLibrary()
    mem = HDCMemory(dimension=10000)

    # binary-search and two-pointer both apply to ordered+searching
    similar = lib.find_similar_strategies(mem, "binary-search", threshold=0.50)
    similar_names = [name for name, _ in similar]

    # Should find at least one similar strategy
    assert len(similar) > 0


# ═══════════════════════════════════════════
# CUSTOM STRATEGIES
# ═══════════════════════════════════════════

def test_custom_strategy():
    lib = StrategyLibrary()
    custom = MetaStrategy(
        name="my-custom-strategy",
        description="A custom strategy for testing",
        applicability={ProblemFeature.STRING, ProblemFeature.ITERATION},
        difficulty=0.3,
        success_rate=0.9,
        steps=["1. Do step A", "2. Do step B"]
    )
    lib.add_custom_strategy(custom)
    assert "my-custom-strategy" in lib.strategies

    features = ["string", "iteration"]
    applicable = lib.find_applicable(features)
    names = [s["name"] for s in applicable]
    assert "my-custom-strategy" in names


def test_score_ranking():
    lib = StrategyLibrary()
    features = ["searching", "ordered"]
    applicable = lib.find_applicable(features)
    assert len(applicable) >= 2
    scores = [s["score"] for s in applicable]
    assert scores == sorted(scores, reverse=True)


def test_all_strategies_have_steps():
    for name, strategy in STRATEGIES.items():
        assert len(strategy.steps) > 0, f"{name} has no steps"
        assert len(strategy.examples) > 0, f"{name} has no examples"


# ═══════════════════════════════════════════
# WORKSPACE INTEGRATION
# ═══════════════════════════════════════════

def test_integration_with_workspace_exact():
    """Test StrategyModule with exact match."""
    from ecs.workspace.global_workspace import StrategyModule

    lib = StrategyLibrary()
    module = StrategyModule(lib)

    context = {"problem_features": ["divisible", "recursion"]}
    bid = module.generate_bid(context)

    assert bid.free_energy_reduction > 0.5
    assert "[EXACT]" in bid.content
    assert "divide-and-conquer" in bid.content
    assert bid.payload["match_type"] == "exact"


def test_integration_with_workspace_partial():
    """Test StrategyModule with partial match fallback."""
    from ecs.workspace.global_workspace import StrategyModule

    lib = StrategyLibrary()
    module = StrategyModule(lib)

    # Only one feature of a two-feature strategy
    context = {"problem_features": ["recursion"]}
    bid = module.generate_bid(context)

    assert "[PARTIAL]" in bid.content
    assert bid.payload["match_type"] == "partial"
    assert bid.confidence < 0.8


def test_integration_no_match():
    """Test StrategyModule when nothing matches well."""
    from ecs.workspace.global_workspace import StrategyModule

    lib = StrategyLibrary()
    module = StrategyModule(lib)

    context = {"problem_features": ["nonexistent_feature"]}
    bid = module.generate_bid(context)

    assert bid.free_energy_reduction == 0.1
    assert bid.payload.get("fallback") == "neural_only"


# ═══════════════════════════════════════════
# SELF-MODEL BID & TEMPERATURE
# ═══════════════════════════════════════════

def test_self_model_bid_when_stuck():
    """Test that self-model generates bid when stuck."""
    from ecs.workspace.global_workspace import GlobalWorkspace

    workspace = GlobalWorkspace()
    for _ in range(4):
        workspace.update_self_model({"success": False, "domain": "test"})

    bid = workspace.generate_self_model_bid()
    assert bid is not None
    assert bid.free_energy_reduction >= 0.7
    assert bid.payload["action"] == "switch_strategy"


def test_self_model_bid_not_generated_when_fine():
    """Test no self-model bid when not stuck."""
    from ecs.workspace.global_workspace import GlobalWorkspace

    workspace = GlobalWorkspace()
    workspace.update_self_model({"success": True, "domain": "test"})

    bid = workspace.generate_self_model_bid()
    assert bid is None


def test_run_auction_cycle_includes_self_bid():
    """Test that run_auction_cycle includes self-model bid."""
    from ecs.workspace.global_workspace import GlobalWorkspace, WorkspaceBid, ModuleType

    workspace = GlobalWorkspace()
    for _ in range(4):
        workspace.update_self_model({"success": False, "domain": "test"})

    module_bids = [WorkspaceBid(
        module_type=ModuleType.MEMORY, module_name="mem",
        content="Memory recall", payload=None, free_energy_reduction=0.3
    )]

    winners = workspace.run_auction_cycle(module_bids)
    # Self-model bid (0.7 * 0.8 weight = 0.56) should beat memory (0.3 * 1.0 = 0.3)
    self_won = any(b.module_type == ModuleType.SELF_MODEL for b in winners)
    assert self_won


def test_attention_temperature_annealing():
    """Test temperature decreases with experience."""
    from ecs.workspace.global_workspace import GlobalWorkspace

    workspace = GlobalWorkspace()
    assert workspace.attention_temperature == 1.0

    workspace.anneal_temperature(50)
    assert workspace.attention_temperature < 1.0

    workspace.anneal_temperature(200)
    temp_200 = workspace.attention_temperature
    assert temp_200 < 0.5


# ═══════════════════════════════════════════
# CO-OCCURRENCE & RETIREMENT
# ═══════════════════════════════════════════

def test_co_occurrence_tracking():
    lib = StrategyLibrary()
    lib.record_co_occurrence(["binary-search", "two-pointer"], True)
    lib.record_co_occurrence(["binary-search", "two-pointer"], True)
    lib.record_co_occurrence(["binary-search", "two-pointer"], False)

    key = "binary-search|two-pointer"
    assert lib.co_occurrence_wins[key] == 3
    assert lib.co_occurrence_success[key] == 2


def test_empirical_synergies():
    lib = StrategyLibrary()
    for _ in range(5):
        lib.record_co_occurrence(["binary-search", "two-pointer"], True)

    synergies = lib.get_empirical_synergies(min_samples=3)
    assert len(synergies) == 1
    assert synergies[0]["success_rate"] == 1.0
    assert synergies[0]["strategies"] == ("binary-search", "two-pointer")


def test_empirical_synergies_below_threshold():
    lib = StrategyLibrary()
    for _ in range(5):
        lib.record_co_occurrence(["greedy", "backtracking"], False)

    synergies = lib.get_empirical_synergies(min_samples=3)
    assert len(synergies) == 0


def test_strategy_retirement():
    lib = StrategyLibrary()
    lib.strategies["greedy"].success_rate = 0.15

    for _ in range(25):
        lib.usage_history.append({"strategy": "greedy", "success": False, "features": []})

    newly_retired = lib.check_retirement()
    assert "greedy" in newly_retired
    assert "greedy" in lib.retired_strategies


def test_retired_strategy_excluded_from_find():
    lib = StrategyLibrary()
    lib.retired_strategies.append("greedy")

    features = ["optimization", "monotonic"]
    applicable = lib.find_applicable(features)
    names = [s["name"] for s in applicable]
    assert "greedy" not in names


def test_retirement_not_triggered_below_min_uses():
    lib = StrategyLibrary()
    lib.strategies["greedy"].success_rate = 0.1

    for _ in range(5):
        lib.usage_history.append({"strategy": "greedy", "success": False, "features": []})

    newly_retired = lib.check_retirement()
    assert "greedy" not in newly_retired


# ═══════════════════════════════════════════
# STRATEGY PERSISTENCE
# ═══════════════════════════════════════════

def test_strategy_persistence(tmp_path):
    from ecs.strategies.persistence import StrategyPersistence

    filepath = str(tmp_path / "strat.pkl")
    persistence = StrategyPersistence(filepath=filepath)

    lib = StrategyLibrary()
    lib.record_usage("greedy", True, ["optimization"])
    lib.record_usage("greedy", False, ["optimization"])
    lib.record_co_occurrence(["greedy", "binary-search"], True)
    lib.retired_strategies.append("backtracking")

    persistence.save(lib)

    lib2 = StrategyLibrary()
    loaded = persistence.load(lib2)
    assert loaded is True
    assert len(lib2.usage_history) == 2
    assert "backtracking" in lib2.retired_strategies
    assert lib2.co_occurrence_wins.get("binary-search|greedy") == 1


def test_strategy_persistence_no_file(tmp_path):
    from ecs.strategies.persistence import StrategyPersistence

    persistence = StrategyPersistence(filepath=str(tmp_path / "nope.pkl"))
    lib = StrategyLibrary()
    assert persistence.load(lib) is False


# ═══════════════════════════════════════════
# COMPOSITOR MODULE
# ═══════════════════════════════════════════

def test_compositor_module_synergistic():
    from ecs.strategies.compositor_module import CompositorModule

    lib = StrategyLibrary()
    module = CompositorModule(lib)

    # binary-search and two-pointer are synergistic and both match ordered+searching
    context = {"problem_features": ["ordered", "searching"]}
    bid = module.generate_bid(context)

    assert "COMPOSITE" in bid.content
    assert bid.payload["is_composite"] is True
    assert bid.free_energy_reduction > 0


def test_compositor_module_no_synergy():
    from ecs.strategies.compositor_module import CompositorModule

    lib = StrategyLibrary()
    module = CompositorModule(lib)

    # Only one strategy matches "combinatorial" alone (backtracking needs recursion too)
    context = {"problem_features": ["sorting", "divisible", "recursion"]}
    bid = module.generate_bid(context)
    # divide-and-conquer is the only exact match, < 2 applicable
    # so "Not enough" or "No synergistic"
    assert bid.free_energy_reduction <= 0.1


# ═══════════════════════════════════════════
# TEMPERATURE COUPLING
# ═══════════════════════════════════════════

def test_temperature_affects_auction():
    """High temperature lowers ignition threshold, allowing weaker bids through."""
    from ecs.workspace.global_workspace import GlobalWorkspace, WorkspaceBid, ModuleType

    workspace = GlobalWorkspace()

    weak_bid = WorkspaceBid(
        module_type=ModuleType.MEMORY, module_name="mem",
        content="Weak memory", payload=None, free_energy_reduction=0.08
    )

    # At default temp=1.0, ignition=0.1 → 0.08 fails
    workspace.attention_temperature = 1.0
    winners = workspace.conduct_auction([weak_bid])
    assert len(winners) == 0

    # At high temp=2.0, ignition threshold drops → 0.08 passes
    workspace.attention_temperature = 2.0
    winners = workspace.conduct_auction([weak_bid])
    assert len(winners) == 1
