"""
Meta-Strategy Library: Explicit reasoning strategies for problem-solving.
These are patterns of thought, not code patterns.
Inspired by Polya's "How to Solve It" and competitive programming techniques.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter, defaultdict


class ProblemFeature(Enum):
    SORTING = "sorting"
    SEARCHING = "searching"
    GRAPH = "graph"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    OPTIMIZATION = "optimization"
    RECURSION = "recursion"
    ITERATION = "iteration"
    DIVISIBLE = "divisible"
    MONOTONIC = "monotonic"
    ORDERED = "ordered"
    OVERLAPPING = "overlapping"
    STATE_MACHINE = "state_machine"
    COMBINATORIAL = "combinatorial"
    GEOMETRIC = "geometric"
    STRING = "string"
    NUMERIC = "numeric"
    TREE = "tree"
    LINKED_LIST = "linked_list"
    CONCURRENCY = "concurrency"
    ERROR_HANDLING = "error_handling"


@dataclass
class MetaStrategy:
    """A reasoning strategy for approaching problems."""
    name: str
    description: str
    applicability: Set[ProblemFeature]
    conflicts_with: Set[str] = field(default_factory=set)
    typical_complexity: str = "O(n)"
    difficulty: float = 0.5
    success_rate: float = 0.5
    examples: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════
# STRATEGY DEFINITIONS
# ═══════════════════════════════════════════

STRATEGIES = {
    "divide-and-conquer": MetaStrategy(
        name="divide-and-conquer",
        description="Split problem into independent subproblems, solve recursively, combine",
        applicability={ProblemFeature.DIVISIBLE, ProblemFeature.RECURSION},
        conflicts_with={"single-pass", "greedy"},
        typical_complexity="O(n log n)",
        difficulty=0.6,
        success_rate=0.7,
        examples=["Merge sort", "Quick sort", "Binary search", "Closest pair of points"],
        steps=[
            "1. Identify how to divide the problem into smaller instances",
            "2. Define base case(s) where problem is trivially solvable",
            "3. Solve subproblems recursively",
            "4. Combine solutions from subproblems",
            "5. Verify: are subproblems truly independent?"
        ]
    ),

    "dynamic-programming": MetaStrategy(
        name="dynamic-programming",
        description="Break into overlapping subproblems, memoize solutions",
        applicability={ProblemFeature.OVERLAPPING, ProblemFeature.OPTIMIZATION},
        conflicts_with={"greedy"},
        typical_complexity="O(n^2) or O(n*m)",
        difficulty=0.7,
        success_rate=0.6,
        examples=["Fibonacci", "Knapsack", "Edit distance", "Longest common subsequence"],
        steps=[
            "1. Define the recurrence relation: how does f(n) relate to f(n-1), f(n-2)...?",
            "2. Identify base cases",
            "3. Choose memoization (top-down) or tabulation (bottom-up)",
            "4. Determine state space and order of computation",
            "5. Extract solution from computed table"
        ]
    ),

    "greedy": MetaStrategy(
        name="greedy",
        description="Make locally optimal choice at each step",
        applicability={ProblemFeature.OPTIMIZATION, ProblemFeature.MONOTONIC},
        conflicts_with={"dynamic-programming"},
        typical_complexity="O(n log n)",
        difficulty=0.4,
        success_rate=0.5,
        examples=["Huffman coding", "Dijkstra", "Kruskal MST", "Activity selection"],
        steps=[
            "1. Identify the greedy choice property",
            "2. Prove greedy choice is safe (exchange argument)",
            "3. Sort or order by greedy criterion",
            "4. Iterate, making locally optimal choices",
            "5. Verify: does greedy always lead to global optimum?"
        ]
    ),

    "two-pointer": MetaStrategy(
        name="two-pointer",
        description="Use two pointers moving through data to find solution",
        applicability={ProblemFeature.ORDERED, ProblemFeature.SEARCHING},
        conflicts_with=set(),
        typical_complexity="O(n)",
        difficulty=0.3,
        success_rate=0.8,
        examples=["Two sum on sorted array", "Container with most water", "Palindrome check"],
        steps=[
            "1. Initialize two pointers (usually start and end)",
            "2. Define condition for moving each pointer",
            "3. Move pointers based on comparison",
            "4. Track best answer found",
            "5. Terminate when pointers meet or cross"
        ]
    ),

    "sliding-window": MetaStrategy(
        name="sliding-window",
        description="Maintain a window over data, slide efficiently",
        applicability={ProblemFeature.ITERATION, ProblemFeature.STRING},
        conflicts_with=set(),
        typical_complexity="O(n)",
        difficulty=0.4,
        success_rate=0.7,
        examples=["Longest substring without repeating", "Min window substring",
                  "Max sum subarray of size k"],
        steps=[
            "1. Define window properties (what does the window represent?)",
            "2. Initialize window at start",
            "3. Expand right edge, checking if window is valid",
            "4. Contract left edge while invalid or to optimize",
            "5. Track best window found"
        ]
    ),

    "binary-search": MetaStrategy(
        name="binary-search",
        description="Search sorted data by halving search space",
        applicability={ProblemFeature.ORDERED, ProblemFeature.SEARCHING},
        conflicts_with={"linear-scan"},
        typical_complexity="O(log n)",
        difficulty=0.3,
        success_rate=0.9,
        examples=["Standard binary search", "Search in rotated array",
                  "Find first bad version"],
        steps=[
            "1. Verify input is sorted or sortable",
            "2. Define search space (lo, hi)",
            "3. Compute mid, compare with target",
            "4. Eliminate half based on comparison",
            "5. Handle edge cases (empty, not found)"
        ]
    ),

    "graph-traversal": MetaStrategy(
        name="graph-traversal",
        description="Systematically visit nodes in a graph (BFS/DFS)",
        applicability={ProblemFeature.GRAPH, ProblemFeature.TREE},
        conflicts_with=set(),
        typical_complexity="O(V + E)",
        difficulty=0.5,
        success_rate=0.8,
        examples=["Connected components", "Shortest path (BFS)",
                  "Cycle detection", "Topological sort"],
        steps=[
            "1. Choose BFS (shortest path, level-order) or DFS (deep exploration)",
            "2. Initialize visited set/array",
            "3. Start from source node",
            "4. Explore neighbors, marking visited",
            "5. Process node when visited (pre-order, post-order, etc.)"
        ]
    ),

    "backtracking": MetaStrategy(
        name="backtracking",
        description="Explore solution space, backtrack when dead end",
        applicability={ProblemFeature.COMBINATORIAL, ProblemFeature.RECURSION},
        conflicts_with=set(),
        typical_complexity="O(2^n) or O(n!)",
        difficulty=0.7,
        success_rate=0.6,
        examples=["N-Queens", "Sudoku", "Permutations", "Subset sum"],
        steps=[
            "1. Define state space and choices at each step",
            "2. Write recursive function with state",
            "3. Check if current state is valid",
            "4. If valid and complete: record solution",
            "5. If invalid: backtrack (undo choice, try next)"
        ]
    ),

    "invariant-discovery": MetaStrategy(
        name="invariant-discovery",
        description="Find what stays constant to guide solution",
        applicability={ProblemFeature.STATE_MACHINE, ProblemFeature.ITERATION},
        conflicts_with=set(),
        typical_complexity="varies",
        difficulty=0.8,
        success_rate=0.5,
        examples=["Loop invariants", "Conservation laws", "Monotonic invariants"],
        steps=[
            "1. Identify what should be true at each iteration",
            "2. Formally state the invariant",
            "3. Prove invariant holds initially",
            "4. Prove invariant is maintained by each iteration",
            "5. Use invariant to prove correctness"
        ]
    ),

    "reduction": MetaStrategy(
        name="reduction",
        description="Transform problem into a known solved problem",
        applicability={ProblemFeature.OPTIMIZATION},
        conflicts_with=set(),
        typical_complexity="depends on target",
        difficulty=0.9,
        success_rate=0.6,
        examples=["Reduce to max-flow", "Reduce to matching",
                  "Reduce to linear programming"],
        steps=[
            "1. Identify structure of current problem",
            "2. Search for known problems with similar structure",
            "3. Construct mapping (isomorphism) to known problem",
            "4. Apply known solution",
            "5. Map solution back to original problem"
        ]
    ),

    "probabilistic-method": MetaStrategy(
        name="probabilistic-method",
        description="Use randomness to prove existence or find solution",
        applicability={ProblemFeature.COMBINATORIAL},
        conflicts_with=set(),
        typical_complexity="varies",
        difficulty=0.8,
        success_rate=0.5,
        examples=["Randomized quicksort", "Monte Carlo methods",
                  "Probabilistic counting"],
        steps=[
            "1. Define probability space",
            "2. Show random choice succeeds with positive probability",
            "3. If existence proof: done",
            "4. If algorithm needed: derandomize or accept randomness",
            "5. Analyze expected performance"
        ]
    ),

    "exchange-argument": MetaStrategy(
        name="exchange-argument",
        description="Prove optimality by showing swaps don't help",
        applicability={ProblemFeature.OPTIMIZATION, ProblemFeature.ORDERED},
        conflicts_with=set(),
        typical_complexity="O(n log n) usually",
        difficulty=0.7,
        success_rate=0.7,
        examples=["Optimal scheduling", "Huffman optimality", "MST cut property"],
        steps=[
            "1. Assume you have optimal solution",
            "2. Consider swapping two elements",
            "3. Show swap doesn't improve (or contradicts optimality)",
            "4. Conclude original is optimal",
            "5. Use insight to construct algorithm"
        ]
    ),
}


# ═══════════════════════════════════════════
# STRATEGY DAG
# ═══════════════════════════════════════════

class StrategyRelation(Enum):
    PREFER_BEFORE = "prefer_before"
    SYNERGISTIC = "synergistic"
    REDUNDANT = "redundant"
    ANTAGONISTIC = "antagonistic"


@dataclass
class StrategyEdge:
    source: str
    target: str
    relation: StrategyRelation
    rationale: str


class StrategyDAG:
    """Directed graph of strategy relationships."""

    def __init__(self):
        self.edges: List[StrategyEdge] = []
        self.adjacency: Dict[str, List[StrategyEdge]] = {}

    def add_edge(self, source: str, target: str,
                 relation: StrategyRelation, rationale: str) -> None:
        edge = StrategyEdge(source, target, relation, rationale)
        self.edges.append(edge)
        self.adjacency.setdefault(source, []).append(edge)

        if relation in (StrategyRelation.SYNERGISTIC,
                        StrategyRelation.REDUNDANT,
                        StrategyRelation.ANTAGONISTIC):
            reverse = StrategyEdge(target, source, relation, rationale)
            self.edges.append(reverse)
            self.adjacency.setdefault(target, []).append(reverse)

    def get_preferred_order(self, strategies: List[str]) -> List[str]:
        """Topological sort based on prefer_before relationships."""
        from collections import deque

        in_degree: Dict[str, int] = defaultdict(int)
        graph: Dict[str, List[str]] = defaultdict(list)

        for s in strategies:
            for edge in self.adjacency.get(s, []):
                if (edge.relation == StrategyRelation.PREFER_BEFORE and
                        edge.target in strategies):
                    graph[s].append(edge.target)
                    in_degree[edge.target] += 1

        queue = deque([s for s in strategies if in_degree[s] == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Add any remaining (cycles or disconnected)
        for s in strategies:
            if s not in result:
                result.append(s)

        return result

    def check_composition(self, strategy_a: str, strategy_b: str) -> Optional[str]:
        """Check relationship between two strategies."""
        for edge in self.adjacency.get(strategy_a, []):
            if edge.target == strategy_b:
                if edge.relation == StrategyRelation.ANTAGONISTIC:
                    return f"Cannot compose: {edge.rationale}"
                elif edge.relation == StrategyRelation.SYNERGISTIC:
                    return f"Synergistic: {edge.rationale}"
                elif edge.relation == StrategyRelation.REDUNDANT:
                    return f"Redundant: {edge.rationale}"
        return None


def build_strategy_dag() -> StrategyDAG:
    """Build the default strategy relationship graph."""
    dag = StrategyDAG()

    dag.add_edge("greedy", "dynamic-programming",
                 StrategyRelation.PREFER_BEFORE,
                 "Greedy is O(n log n), DP is O(n^2) - try fast first")

    dag.add_edge("two-pointer", "binary-search",
                 StrategyRelation.PREFER_BEFORE,
                 "Two-pointer O(n) full scan before binary search O(log n) per query")

    dag.add_edge("sliding-window", "dynamic-programming",
                 StrategyRelation.PREFER_BEFORE,
                 "Sliding window is O(n), DP is usually O(n^2) - try linear first")

    dag.add_edge("binary-search", "two-pointer",
                 StrategyRelation.SYNERGISTIC,
                 "Binary search finds region, two-pointer scans within it")

    dag.add_edge("divide-and-conquer", "invariant-discovery",
                 StrategyRelation.SYNERGISTIC,
                 "D&C structures the problem, invariants prove correctness")

    dag.add_edge("two-pointer", "sliding-window",
                 StrategyRelation.REDUNDANT,
                 "Both are linear-scan patterns - use one, not both")

    dag.add_edge("greedy", "dynamic-programming",
                 StrategyRelation.ANTAGONISTIC,
                 "Greedy makes local choices, DP considers all - philosophically opposed")

    return dag


# ═══════════════════════════════════════════
# STRATEGY COMPOSITOR
# ═══════════════════════════════════════════

class CompositionMode(Enum):
    SEQUENTIAL = "sequential"
    MERGED = "merged"
    HIERARCHICAL = "hierarchical"


class StrategyCompositor:
    """Executes strategy compositions with known templates."""

    COMPOSITION_TEMPLATES: Dict[Tuple[str, str], Dict] = {
        ("binary-search", "two-pointer"): {
            "mode": CompositionMode.HIERARCHICAL,
            "description": "Binary search finds region, two-pointer scans within",
            "holes": ["binary_search_condition", "two_pointer_condition"],
        },
        ("divide-and-conquer", "invariant-discovery"): {
            "mode": CompositionMode.HIERARCHICAL,
            "description": "D&C with invariant-guided correctness",
            "holes": ["invariant_statement", "base_case", "base_solution"],
        },
        ("dynamic-programming", "invariant-discovery"): {
            "mode": CompositionMode.MERGED,
            "description": "DP with invariant-guided state definition",
            "holes": ["invariant_meaning", "recurrence_relation"],
        },
    }

    def __init__(self, library: 'StrategyLibrary'):
        self.library = library

    def compose(self, strategy_a: str, strategy_b: str,
                problem: str = "") -> Dict[str, Any]:
        """Compose two strategies for execution."""
        key = (min(strategy_a, strategy_b), max(strategy_a, strategy_b))
        if key in self.COMPOSITION_TEMPLATES:
            template = self.COMPOSITION_TEMPLATES[key]
            return {
                "mode": template["mode"],
                "description": template["description"],
                "holes": template["holes"],
                "strategies": list(key),
                "confidence": 0.8,
            }

        return self._compose_unknown_pair(strategy_a, strategy_b)

    def _compose_unknown_pair(self, strategy_a: str, strategy_b: str) -> Dict[str, Any]:
        sa = self.library.strategies.get(strategy_a)
        sb = self.library.strategies.get(strategy_b)

        if not sa or not sb:
            return {"mode": CompositionMode.SEQUENTIAL, "confidence": 0.3,
                    "strategies": [strategy_a, strategy_b]}

        if sa.difficulty <= sb.difficulty:
            structuring, solving = sa, sb
        else:
            structuring, solving = sb, sa

        return {
            "mode": CompositionMode.HIERARCHICAL,
            "structuring_strategy": structuring.name,
            "solving_strategy": solving.name,
            "strategies": [structuring.name, solving.name],
            "confidence": 0.5,
            "instructions": (
                f"Use {structuring.name} to decompose, "
                f"then apply {solving.name} to solve subproblems"
            ),
        }


# ═══════════════════════════════════════════
# STRATEGY LIBRARY
# ═══════════════════════════════════════════

class StrategyLibrary:
    """Manages the collection of meta-strategies."""

    RETIREMENT_THRESHOLD = 0.2
    RETIREMENT_MIN_USES = 20

    def __init__(self):
        self.strategies: Dict[str, MetaStrategy] = {k: v for k, v in STRATEGIES.items()}
        self.usage_history: List[Dict] = []
        self.dag = build_strategy_dag()
        self.compositor = StrategyCompositor(self)

        self.co_occurrence_wins: Dict[str, int] = {}
        self.co_occurrence_success: Dict[str, int] = {}
        self.retired_strategies: List[str] = []
        self.custom_strategies_data: List[Dict] = []

    def find_applicable(self, problem_features: List[str]) -> List[Dict]:
        """Find strategies applicable to given problem features (exact match)."""
        feature_set = set()
        for f in problem_features:
            try:
                feature_set.add(ProblemFeature(f))
            except ValueError:
                continue

        applicable = []
        for name, strategy in self.strategies.items():
            if name in self.retired_strategies:
                continue
            if strategy.applicability.issubset(feature_set):
                applicable.append(self._strategy_to_dict(strategy, 1.0))

        applicable.sort(key=lambda x: x["score"], reverse=True)
        return applicable

    def find_partial_matches(self, problem_features: List[str],
                             min_score: float = 0.3) -> List[Dict]:
        """Find strategies with partial feature overlap."""
        feature_set = set()
        for f in problem_features:
            try:
                feature_set.add(ProblemFeature(f))
            except ValueError:
                continue

        if not feature_set:
            return []

        matches = []
        for name, strategy in self.strategies.items():
            score = self.partial_match_score(strategy, feature_set)
            if score >= min_score:
                matches.append(self._strategy_to_dict(strategy, score))

        matches.sort(key=lambda x: x["match_score"], reverse=True)
        return matches

    def partial_match_score(self, strategy: MetaStrategy,
                            problem_features: Set[ProblemFeature]) -> float:
        """Score how well a problem matches a strategy (0-1)."""
        required = strategy.applicability
        if not required:
            return 0.5
        matched = required & problem_features
        return len(matched) / len(required)

    def _strategy_to_dict(self, strategy: MetaStrategy, match_score: float) -> Dict:
        return {
            "name": strategy.name,
            "description": strategy.description,
            "steps": strategy.steps,
            "complexity": strategy.typical_complexity,
            "difficulty": strategy.difficulty,
            "success_rate": strategy.success_rate,
            "examples": strategy.examples,
            "score": strategy.success_rate / (strategy.difficulty + 0.1),
            "match_score": match_score,
        }

    def check_compatibility(self, strategy_a: str, strategy_b: str) -> bool:
        """Check if two strategies can be composed (legacy + DAG)."""
        if strategy_a not in self.strategies or strategy_b not in self.strategies:
            return False

        sa = self.strategies[strategy_a]
        sb = self.strategies[strategy_b]

        if strategy_b in sa.conflicts_with:
            return False
        if strategy_a in sb.conflicts_with:
            return False

        # Also check DAG for antagonistic relationships
        result = self.dag.check_composition(strategy_a, strategy_b)
        if result and "Cannot" in result:
            return False

        return True

    def get_compatible_combinations(self, base_strategy: str,
                                    problem_features: List[str]) -> List[List[str]]:
        """Find compatible strategy combinations."""
        applicable = self.find_applicable(problem_features)
        applicable_names = [s["name"] for s in applicable]

        combinations = []
        for other in applicable_names:
            if other != base_strategy and self.check_compatibility(base_strategy, other):
                combinations.append([base_strategy, other])

        return combinations

    def record_usage(self, strategy_name: str, success: bool,
                     problem_features: List[str]) -> None:
        """Record strategy usage for learning."""
        self.usage_history.append({
            "strategy": strategy_name,
            "success": success,
            "features": problem_features
        })

        strategy = self.strategies.get(strategy_name)
        if strategy:
            old_rate = strategy.success_rate
            new_obs = 1.0 if success else 0.0
            strategy.success_rate = 0.9 * old_rate + 0.1 * new_obs

    def get_strategy_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics for all strategies."""
        stats: Dict[str, Dict[str, Any]] = {}
        usage_counts: Counter = Counter()
        success_counts: Counter = Counter()

        for entry in self.usage_history:
            name = entry["strategy"]
            usage_counts[name] += 1
            if entry["success"]:
                success_counts[name] += 1

        for name, strategy in self.strategies.items():
            uses = usage_counts[name]
            successes = success_counts[name]
            stats[name] = {
                "uses": uses,
                "successes": successes,
                "success_rate": strategy.success_rate,
                "difficulty": strategy.difficulty,
                "win_rate": successes / uses if uses > 0 else None,
            }

        return stats

    def detect_strategy_gaps(self) -> List[Dict]:
        """Find problem types with poor strategy coverage."""
        feature_success: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"attempts": 0, "successes": 0}
        )

        for entry in self.usage_history:
            for feature in entry.get("features", []):
                feature_success[feature]["attempts"] += 1
                if entry["success"]:
                    feature_success[feature]["successes"] += 1

        gaps = []
        for feature, stats in feature_success.items():
            if stats["attempts"] >= 5:
                success_rate = stats["successes"] / stats["attempts"]
                if success_rate < 0.3:
                    try:
                        pf = ProblemFeature(feature)
                    except ValueError:
                        continue
                    targeted = any(
                        pf in s.applicability
                        for s in self.strategies.values()
                    )
                    if not targeted:
                        gaps.append({
                            "feature": feature,
                            "attempts": stats["attempts"],
                            "success_rate": success_rate,
                        })

        return gaps

    def suggest_new_strategy(self) -> Optional[Dict]:
        """Suggest where a new strategy is needed based on usage gaps."""
        gaps = self.detect_strategy_gaps()
        if not gaps:
            return None

        worst_gap = min(gaps, key=lambda g: g["success_rate"])

        attempted_strategies = []
        for entry in self.usage_history:
            if worst_gap["feature"] in entry.get("features", []):
                attempted_strategies.append(entry["strategy"])

        strategy_counts = Counter(attempted_strategies)
        most_attempted = strategy_counts.most_common(1)[0] if strategy_counts else ("none", 0)

        return {
            "gap": worst_gap,
            "most_attempted_strategy": most_attempted[0],
            "suggestion": (
                f"Feature '{worst_gap['feature']}' has "
                f"{worst_gap['success_rate']:.0%} success rate. "
                f"Most attempted: {most_attempted[0]}. "
                f"Need alternative approach."
            ),
        }

    # ═══════════════════════════════════════════
    # CO-OCCURRENCE & RETIREMENT
    # ═══════════════════════════════════════════

    def record_co_occurrence(self, strategies: List[str], success: bool) -> None:
        """Record that these strategies co-won an auction."""
        for i, s1 in enumerate(strategies):
            for s2 in strategies[i + 1:]:
                key = "|".join(sorted([s1, s2]))
                self.co_occurrence_wins[key] = self.co_occurrence_wins.get(key, 0) + 1
                if success:
                    self.co_occurrence_success[key] = (
                        self.co_occurrence_success.get(key, 0) + 1
                    )

    def get_empirical_synergies(self, min_samples: int = 3) -> List[Dict]:
        """Find strategy pairs that empirically work well together."""
        synergies = []

        for key, wins in self.co_occurrence_wins.items():
            if wins < min_samples:
                continue

            successes = self.co_occurrence_success.get(key, 0)
            success_rate = successes / wins

            if success_rate > 0.6:
                s1, s2 = key.split("|")
                synergies.append({
                    "strategies": (s1, s2),
                    "success_rate": success_rate,
                    "sample_size": wins
                })

        return sorted(synergies, key=lambda x: x["success_rate"], reverse=True)

    def check_retirement(self) -> List[str]:
        """Check and retire strategies with consistently poor performance."""
        newly_retired = []
        for name, strategy in self.strategies.items():
            if name in self.retired_strategies:
                continue

            uses = sum(1 for entry in self.usage_history
                       if entry["strategy"] == name)

            if uses >= self.RETIREMENT_MIN_USES:
                if strategy.success_rate < self.RETIREMENT_THRESHOLD:
                    self.retired_strategies.append(name)
                    newly_retired.append(name)

        return newly_retired

    # ═══════════════════════════════════════════
    # HDC ENCODING
    # ═══════════════════════════════════════════

    def encode_strategy_to_hdc(self, hdc_memory, strategy_name: str):
        """Encode a strategy as a hypervector (name + description + steps + features)."""
        import numpy as np
        strategy = self.strategies[strategy_name]

        name_vec = hdc_memory.get_atomic_vector(f"strategy_{strategy_name}")
        desc_vec = hdc_memory.encode_text(strategy.description)

        step_vectors = []
        for i, step in enumerate(strategy.steps):
            step_vec = hdc_memory.encode_text(step)
            step_vectors.append(hdc_memory._permute(step_vec, positions=i))

        steps_bundle = hdc_memory._majority_bundle(step_vectors)

        feature_vecs = [
            hdc_memory.get_atomic_vector(f.value)
            for f in strategy.applicability
        ]
        features_bundle = hdc_memory._majority_bundle(feature_vecs)

        weighted_components = [
            (name_vec, 0.5),
            (desc_vec, 1.0),
            (steps_bundle, 2.0),
            (features_bundle, 1.5),
        ]

        return hdc_memory._weighted_bundle(weighted_components)

    def find_similar_strategies(self, hdc_memory, strategy_name: str,
                                threshold: float = 0.55) -> List[Tuple[str, float]]:
        """Find strategies with similar HDC encodings."""
        query_vec = self.encode_strategy_to_hdc(hdc_memory, strategy_name)

        similar = []
        for other_name in self.strategies:
            if other_name == strategy_name:
                continue
            other_vec = self.encode_strategy_to_hdc(hdc_memory, other_name)
            similarity = hdc_memory._similarity(query_vec, other_vec)
            if similarity > threshold:
                similar.append((other_name, similarity))

        return sorted(similar, key=lambda x: x[1], reverse=True)

    def add_custom_strategy(self, strategy: MetaStrategy) -> None:
        """Add a new strategy discovered by the abstraction engine."""
        self.strategies[strategy.name] = strategy
