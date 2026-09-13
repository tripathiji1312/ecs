"""
Ablation Study: Which components contribute most to ECS performance?

Conditions:
1. Full ECS (baseline: all components)
2. No constraint inference (disable pattern matching)
3. No memory reuse (disable procedural memory retrieval)
4. No attention adaptation (freeze attention weights)
5. No cross-domain bridges (no seeded bridges)
6. No temperature feedback (freeze temperature)
7. No strategy module (disable strategy matching)

Run: uv run python scripts/ablation.py
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ecs.orchestrator import ECSOrchestrator
from ecs.memory.hdc import MemoryType


TEST_PROBLEMS = [
    "Sort a list of integers in ascending order",
    "Implement merge sort",
    "Implement quicksort with median-of-three pivot selection",
    "Implement binary search on a sorted array",
    "Find the first occurrence of a target in a sorted array with duplicates",
    "Find the peak element in a mountain array",
    "Check if a string is a palindrome",
    "Reverse a string without using built-in functions",
    "Find the longest substring without repeating characters",
    "Find the maximum element in a list",
    "Find two numbers in an array that sum to a target",
    "Rotate an array by k positions",
    "Compute the nth Fibonacci number",
    "Calculate the factorial of n",
    "Generate all permutations of a list",
    "Check if a number is prime",
    "Check if a string of brackets is balanced",
    "Implement a stack with push, pop, and min operations",
    "Find the longest common prefix of a list of strings",
    "Remove duplicates from a sorted array in-place",
]


def run_condition(name: str, setup_fn, problems: List[str]) -> Dict:
    """Run one ablation condition."""
    ecs = ECSOrchestrator()
    setup_fn(ecs)

    results = []
    for problem in problems:
        result = ecs.solve_problem(problem)
        results.append({
            "problem": problem,
            "success": result["success"],
            "confidence": result["confidence"],
            "method": result.get("synthesis_method", "failed"),
            "time": result["time_taken"],
        })

    success_rate = sum(1 for r in results if r["success"]) / len(results)
    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    avg_time = sum(r["time"] for r in results) / len(results)

    methods = {}
    for r in results:
        m = r["method"]
        methods[m] = methods.get(m, 0) + 1

    return {
        "name": name,
        "success_rate": success_rate,
        "avg_confidence": avg_confidence,
        "avg_time_ms": avg_time * 1000,
        "methods": methods,
        "results": results,
    }


def setup_full(ecs):
    """Full ECS (baseline)."""
    pass


def setup_no_constraints(ecs):
    """Disable constraint inference."""
    ecs.constraint_engine.CONSTRAINT_PATTERNS = {}


def setup_no_memory_reuse(ecs):
    """Disable procedural memory retrieval."""
    original_retrieve = ecs._retrieve_code_from_memory
    ecs._retrieve_code_from_memory = lambda problem: None


def setup_no_attention(ecs):
    """Freeze attention weights (no adaptation)."""
    original_update = ecs.workspace.update_attention_weights
    ecs.workspace.update_attention_weights = lambda outcomes: None


def setup_no_bridges(ecs):
    """Remove cross-domain bridge memories."""
    bridge_ids = [k for k in ecs.memory.items if k.startswith("bridge_")]
    for bid in bridge_ids:
        del ecs.memory.items[bid]
    for mtype in ecs.memory.type_indices:
        ecs.memory.type_indices[mtype] = [
            x for x in ecs.memory.type_indices[mtype]
            if not x.startswith("bridge_")
        ]


def setup_no_temperature(ecs):
    """Freeze temperature (no confidence-temperature feedback)."""
    ecs._update_confidence_temperature = lambda: None


def setup_no_strategy(ecs):
    """Disable strategy module."""
    ecs.workspace.modules.pop("strategy", None)


def setup_constraints_only(ecs):
    """Only constraint inference, no memory, no strategies."""
    ecs._retrieve_code_from_memory = lambda problem: None
    ecs.workspace.modules.pop("strategy", None)
    ecs.workspace.modules.pop("memory", None)


def main():
    conditions = [
        ("Full ECS (baseline)", setup_full),
        ("No constraint inference", setup_no_constraints),
        ("No memory reuse", setup_no_memory_reuse),
        ("No attention adaptation", setup_no_attention),
        ("No cross-domain bridges", setup_no_bridges),
        ("No temperature feedback", setup_no_temperature),
        ("No strategy module", setup_no_strategy),
        ("Constraints only (no memory/strat)", setup_constraints_only),
    ]

    print("=" * 70)
    print("  ABLATION STUDY: ECS v3 Component Contributions")
    print("=" * 70)
    print(f"  Problems: {len(TEST_PROBLEMS)}")
    print(f"  Conditions: {len(conditions)}")
    print()

    all_results = {}

    for i, (name, setup_fn) in enumerate(conditions, 1):
        print(f"[{i}/{len(conditions)}] {name}...")
        result = run_condition(name, setup_fn, TEST_PROBLEMS)
        all_results[name] = result
        print(f"       Success: {result['success_rate']:.0%}, "
              f"Confidence: {result['avg_confidence']:.3f}, "
              f"Time: {result['avg_time_ms']:.0f}ms")

    # Report
    print("\n" + "=" * 70)
    print("  ABLATION RESULTS")
    print("=" * 70)

    baseline = all_results["Full ECS (baseline)"]
    baseline_rate = baseline["success_rate"]

    print(f"\n{'Condition':<35} {'Success':<9} {'Delta':<8} {'Conf':<7} {'Time':<8}")
    print("-" * 70)

    for name, result in all_results.items():
        delta = result["success_rate"] - baseline_rate
        delta_str = f"{delta:+.0%}" if name != "Full ECS (baseline)" else "---"
        print(f"  {name:<33} {result['success_rate']:>5.0%}    "
              f"{delta_str:<6}  "
              f"{result['avg_confidence']:.3f}   "
              f"{result['avg_time_ms']:.0f}ms")

    # Component importance ranking
    print(f"\n{'=' * 70}")
    print("  COMPONENT IMPORTANCE (by success rate drop)")
    print(f"{'=' * 70}")

    drops = []
    for name, result in all_results.items():
        if name == "Full ECS (baseline)":
            continue
        drop = baseline_rate - result["success_rate"]
        drops.append((name, drop))

    drops.sort(key=lambda x: -x[1])

    for i, (name, drop) in enumerate(drops, 1):
        bar = "█" * int(drop * 50) if drop > 0 else ""
        print(f"  {i}. {name:<35} -{drop:.0%}  {bar}")

    # Save
    output_path = Path(__file__).parent.parent / "evaluation_ablation.json"
    with open(output_path, "w") as f:
        # Remove per-problem results for cleaner output
        save_data = {}
        for name, result in all_results.items():
            save_data[name] = {k: v for k, v in result.items() if k != "results"}
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
