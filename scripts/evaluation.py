"""
Baseline Evaluation: Does ECS v3 architecture actually improve
over plain Qwen2.5-Coder-0.5B?

Conditions:
1. qwen_only: Direct generation with Qwen2.5-Coder-0.5B
2. qwen_rag: Qwen + vector RAG (retrieve similar problems, include as context)
3. ecs_v3: Full ECS system

Measures:
- Success rate (code passes tests)
- Confidence calibration (does confidence predict success?)
- Latency
- Code quality (lines, complexity)

Run: uv run python scripts/evaluation.py
"""

import sys
import time
import json
from typing import List, Dict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ecs.neural.interface import NeuralInterface
from ecs.memory.hdc import HDCMemory, MemoryType
from ecs.verification.sandbox import SafeExecutor
from ecs.orchestrator import ECSOrchestrator


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


class BaselineEvaluator:
    """Evaluates ECS v3 against baselines."""

    def __init__(self):
        self.results: Dict[str, List[Dict]] = {
            "qwen_only": [],
            "qwen_rag": [],
            "ecs_v3": [],
            "ecs_structured": [],
        }

    def evaluate_qwen_only(self, problems: List[str]):
        """Baseline 1: Direct Qwen generation."""
        interface = NeuralInterface()

        if not interface.is_available():
            print("  [SKIP] Ollama not available")
            return

        executor = SafeExecutor(timeout=5.0)

        for i, problem in enumerate(problems, 1):
            start = time.time()

            response = interface.generate_code(
                specification=problem,
                strategy_context=None,
                retrieved_knowledge=None
            )

            tests = interface.generate_test_cases(
                "def solve(input):", problem
            )
            formatted_tests = [
                {"function": "solve", "inputs": [t.get("input")],
                 "expected": t.get("expected")}
                for t in tests[:5]
            ]

            verified = False
            message = "No code generated"
            if response.code and formatted_tests:
                verified, message, _ = executor.execute_tests(
                    response.code, formatted_tests
                )

            elapsed = time.time() - start

            self.results["qwen_only"].append({
                "problem": problem,
                "success": verified,
                "confidence": response.confidence,
                "time": elapsed,
                "code_length": len(response.code or ""),
            })

            status = "PASS" if verified else "FAIL"
            print(f"  [{i:2d}/{len(problems)}] {status} {problem[:45]}... ({elapsed:.1f}s)")

    def evaluate_qwen_rag(self, problems: List[str]):
        """Baseline 2: Qwen + simple vector RAG."""
        interface = NeuralInterface()

        if not interface.is_available():
            print("  [SKIP] Ollama not available")
            return

        memory = HDCMemory(dimension=10000)
        executor = SafeExecutor(timeout=5.0)

        seed_data = [
            ("sort a list", "Use sorted() or implement quicksort with pivot"),
            ("binary search", "Divide search space in half, compare with mid"),
            ("palindrome", "Compare string with its reverse"),
            ("fibonacci", "Use DP: fib[i] = fib[i-1] + fib[i-2]"),
            ("find maximum", "Iterate, track maximum seen so far"),
            ("two sum pair", "Use hashmap for O(n) complement lookup"),
            ("merge sort divide", "Split array, recursively sort, merge halves"),
            ("factorial recursion", "Base case n<=1, multiply n * factorial(n-1)"),
            ("prime number check", "Check divisibility up to sqrt(n)"),
            ("balanced brackets", "Use a stack, push open, pop on close"),
        ]

        for text, solution in seed_data:
            vec = memory.encode_text(text)
            memory.store(text.replace(" ", "_"), vec,
                         MemoryType.SEMANTIC, solution)

        for i, problem in enumerate(problems, 1):
            start = time.time()

            results = memory.query_by_content(problem, top_k=3)
            retrieved = [item.content for item, sim in results if sim > 0.3]

            response = interface.generate_code(
                specification=problem,
                strategy_context=None,
                retrieved_knowledge=retrieved
            )

            tests = interface.generate_test_cases(
                "def solve(input):", problem
            )
            formatted_tests = [
                {"function": "solve", "inputs": [t.get("input")],
                 "expected": t.get("expected")}
                for t in tests[:5]
            ]

            verified = False
            if response.code and formatted_tests:
                verified, _, _ = executor.execute_tests(
                    response.code, formatted_tests
                )

            elapsed = time.time() - start

            self.results["qwen_rag"].append({
                "problem": problem,
                "success": verified,
                "confidence": response.confidence,
                "time": elapsed,
                "code_length": len(response.code or ""),
                "retrieved_count": len(retrieved),
            })

            status = "PASS" if verified else "FAIL"
            print(f"  [{i:2d}/{len(problems)}] {status} {problem[:45]}... ({elapsed:.1f}s)")

    def evaluate_ecs_v3(self, problems: List[str]):
        """Condition 3: Full ECS v3 (neural online)."""
        orchestrator = ECSOrchestrator()

        if not orchestrator.neural.is_available():
            print("  [SKIP] Ollama not available for full tier")
            return

        for i, problem in enumerate(problems, 1):
            result = orchestrator.solve_problem(problem)

            self.results["ecs_v3"].append({
                "problem": problem,
                "success": result["success"],
                "confidence": result["confidence"],
                "time": result["time_taken"],
                "code_length": len(result.get("code") or ""),
                "strategy": result.get("strategy_used"),
                "method": result.get("synthesis_method"),
                "emergence": result.get("emergence_findings", []),
            })

            status = "PASS" if result["success"] else "FAIL"
            print(f"  [{i:2d}/{len(problems)}] {status} {problem[:45]}... ({result['time_taken']:.1f}s)")

    def evaluate_ecs_structured(self, problems: List[str]):
        """Condition 4: ECS v3 structured tier (no neural, constraint inference)."""
        orchestrator = ECSOrchestrator()

        for i, problem in enumerate(problems, 1):
            result = orchestrator.solve_problem(problem)

            tier_success, tier_reason = orchestrator.evaluate_tier_success(result)

            self.results["ecs_structured"].append({
                "problem": problem,
                "success": result["success"],
                "tier_success": tier_success,
                "tier_reason": tier_reason,
                "confidence": result["confidence"],
                "time": result["time_taken"],
                "code_length": len(result.get("code") or ""),
                "strategy": result.get("strategy_used"),
                "method": result.get("synthesis_method"),
                "emergence": result.get("emergence_findings", []),
            })

            code_status = "PASS" if result["success"] else "FAIL"
            tier_status = "TIER-OK" if tier_success else "TIER-NO"
            print(f"  [{i:2d}/{len(problems)}] {code_status}/{tier_status} "
                  f"{problem[:40]}... ({result['time_taken']*1000:.0f}ms)")

    def analyze_results(self) -> Dict:
        """Analyze and compare results."""
        analysis = {}

        for condition, results in self.results.items():
            if not results:
                continue

            successes = [r["success"] for r in results]
            confidences = [r["confidence"] for r in results]
            times = [r["time"] for r in results]

            success_rate = sum(successes) / len(successes) if successes else 0

            high_conf = [r for r in results if r["confidence"] > 0.6]
            low_conf = [r for r in results if r["confidence"] <= 0.6]

            high_conf_success = (
                sum(1 for r in high_conf if r["success"]) / len(high_conf)
                if high_conf else 0
            )
            low_conf_success = (
                sum(1 for r in low_conf if r["success"]) / len(low_conf)
                if low_conf else 0
            )

            analysis[condition] = {
                "success_rate": success_rate,
                "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
                "avg_time": sum(times) / len(times) if times else 0,
                "total_problems": len(results),
                "confidence_calibration": {
                    "high_conf_count": len(high_conf),
                    "high_conf_success_rate": high_conf_success,
                    "low_conf_count": len(low_conf),
                    "low_conf_success_rate": low_conf_success,
                    "calibrated": high_conf_success > low_conf_success
                },
            }

            # Tier-specific metrics for structured
            if condition == "ecs_structured":
                tier_successes = [r.get("tier_success", False) for r in results]
                analysis[condition]["tier_success_rate"] = (
                    sum(tier_successes) / len(tier_successes) if tier_successes else 0
                )
                emergence_count = sum(
                    len(r.get("emergence", [])) for r in results
                )
                analysis[condition]["emergence_events"] = emergence_count

        return analysis

    def print_report(self):
        """Print evaluation report."""
        analysis = self.analyze_results()

        print("\n" + "=" * 70)
        print("  EVALUATION REPORT: ECS v3 vs Baselines")
        print("=" * 70)

        header = f"{'Condition':<18} {'Success':<9} {'Avg Conf':<10} {'Avg Time':<10} {'Calibrated?':<12}"
        print(f"\n{header}")
        print("-" * 65)

        for condition, stats in analysis.items():
            calibrated = "Yes" if stats["confidence_calibration"]["calibrated"] else "No"
            time_str = f"{stats['avg_time']:.2f}s" if stats['avg_time'] > 0.1 else f"{stats['avg_time']*1000:.0f}ms"
            print(
                f"{condition:<18} "
                f"{stats['success_rate']:>5.1%}    "
                f"{stats['avg_confidence']:>6.3f}    "
                f"{time_str:>8}   "
                f"{calibrated:<12}"
            )

        # Structured tier detail
        if "ecs_structured" in analysis:
            structured = analysis["ecs_structured"]
            print(f"\n--- ECS Structured Tier Detail ---")
            print(f"  Code success rate: {structured['success_rate']:.1%}")
            print(f"  Tier success rate: {structured['tier_success_rate']:.1%}")
            print(f"  Emergence events:  {structured['emergence_events']}")

        # Confidence calibration
        print(f"\n{'=' * 70}")
        print("CONFIDENCE CALIBRATION")
        print(f"{'=' * 70}")

        for condition, stats in analysis.items():
            cal = stats["confidence_calibration"]
            print(f"\n  {condition}:")
            print(f"    High confidence (>0.6): {cal['high_conf_count']} problems, "
                  f"{cal['high_conf_success_rate']:.1%} success")
            print(f"    Low confidence (<=0.6): {cal['low_conf_count']} problems, "
                  f"{cal['low_conf_success_rate']:.1%} success")
            print(f"    Well-calibrated: {'Yes' if cal['calibrated'] else 'No'}")

        # Learning analysis for structured tier
        if "ecs_structured" in self.results and self.results["ecs_structured"]:
            results = self.results["ecs_structured"]
            print(f"\n{'=' * 70}")
            print("LEARNING ACROSS PROBLEMS (ECS Structured)")
            print(f"{'=' * 70}")

            first_half = results[:len(results)//2]
            second_half = results[len(results)//2:]

            first_success = sum(1 for r in first_half if r["success"]) / len(first_half)
            second_success = sum(1 for r in second_half if r["success"]) / len(second_half)

            print(f"  First half success:  {first_success:.1%}")
            print(f"  Second half success: {second_success:.1%}")
            if second_success > first_success:
                print(f"  Learning effect: +{second_success - first_success:.1%}")
            else:
                print(f"  No learning effect detected")

            # Method distribution
            methods = {}
            for r in results:
                m = r.get("method", "unknown")
                methods[m] = methods.get(m, 0) + 1
            print(f"\n  Synthesis methods: {methods}")

        print(f"\n{'=' * 70}\n")


def run_structured_only():
    """Run just the structured tier evaluation (no Ollama needed)."""
    evaluator = BaselineEvaluator()

    print("=" * 70)
    print("  ECS v3 STRUCTURED TIER EVALUATION")
    print("  (No Ollama required — constraint inference + Z3)")
    print("=" * 70)
    print(f"\n  Problems: {len(TEST_PROBLEMS)}")
    print()

    print("[1/1] Evaluating ECS v3 (structured tier)...")
    evaluator.evaluate_ecs_structured(TEST_PROBLEMS)

    evaluator.print_report()

    output_path = Path(__file__).parent.parent / "evaluation_structured.json"
    with open(output_path, "w") as f:
        json.dump(evaluator.results, f, indent=2, default=str)
    print(f"Raw results saved to {output_path}")


def run_full_evaluation():
    """Run the complete evaluation (requires Ollama)."""
    evaluator = BaselineEvaluator()

    print("=" * 70)
    print("  ECS v3 FULL EVALUATION")
    print("=" * 70)
    print(f"\n  Problems: {len(TEST_PROBLEMS)}")
    print(f"  Conditions: qwen_only, qwen_rag, ecs_v3, ecs_structured")
    print()

    print("[1/4] Evaluating qwen_only baseline...")
    evaluator.evaluate_qwen_only(TEST_PROBLEMS)

    print("\n[2/4] Evaluating qwen_rag baseline...")
    evaluator.evaluate_qwen_rag(TEST_PROBLEMS)

    print("\n[3/4] Evaluating ECS v3 (full tier)...")
    evaluator.evaluate_ecs_v3(TEST_PROBLEMS)

    print("\n[4/4] Evaluating ECS v3 (structured tier)...")
    evaluator.evaluate_ecs_structured(TEST_PROBLEMS)

    evaluator.print_report()

    output_path = Path(__file__).parent.parent / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(evaluator.results, f, indent=2, default=str)
    print(f"Raw results saved to {output_path}")


if __name__ == "__main__":
    if "--structured" in sys.argv:
        run_structured_only()
    else:
        run_full_evaluation()
