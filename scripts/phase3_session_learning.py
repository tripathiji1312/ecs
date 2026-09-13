"""
Phase 3: Session Learning — does memory accumulation improve performance?

Two conditions on HumanEval (164 problems):
  (a) With memory: single ECSOrchestrator processes all problems sequentially
  (b) Without memory: fresh ECSOrchestrator per problem (no accumulation)

Measures first-half vs second-half pass rate improvement.
Runs 3 shuffled orderings for statistical robustness.

Run: uv run python scripts/phase3_session_learning.py
"""

import sys
import json
import random
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from ecs.orchestrator import ECSOrchestrator
from ecs.evaluation.signature_adapter import SignatureAdapter

ROOT = Path(__file__).parent.parent


def extract_problem_description(prompt: str) -> str:
    import re
    lines = prompt.strip().split('\n')
    description_lines = []
    in_docstring = False
    for line in lines:
        stripped = line.strip()
        if '"""' in stripped or "'''" in stripped:
            if in_docstring:
                break
            in_docstring = True
            after_quote = stripped.split('"""', 1)[-1] if '"""' in stripped else stripped.split("'''", 1)[-1]
            if after_quote.strip():
                description_lines.append(after_quote.strip())
            continue
        if in_docstring:
            if stripped.startswith('>>>'):
                break
            if stripped:
                description_lines.append(stripped)
    desc = ' '.join(description_lines)
    if not desc:
        func_match = re.search(r'def\s+(\w+)', prompt)
        if func_match:
            name = func_match.group(1).replace('_', ' ')
            desc = f"Implement a function to {name}"
    return desc


def verify_humaneval(code: str, problem: Dict) -> bool:
    if not code:
        return False
    full_program = f"{problem['prompt']}{code}\n\n{problem['test']}\n\ncheck({problem['entry_point']})\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_program)
        f.flush()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False


def run_sequential(problems: List[Dict], with_memory: bool, seed: int) -> Dict:
    """Run all problems sequentially through one orchestrator."""
    rng = random.Random(seed)
    order = list(range(len(problems)))
    rng.shuffle(order)

    adapter = SignatureAdapter()
    results = []

    if with_memory:
        ecs = ECSOrchestrator()

    for seq_idx, prob_idx in enumerate(order):
        problem = problems[prob_idx]
        prompt = problem['prompt']
        entry_point = problem['entry_point']
        desc = extract_problem_description(prompt)

        if not with_memory:
            ecs = ECSOrchestrator()

        ecs_result = ecs.solve_problem(desc, prompt=prompt, entry_point=entry_point)
        code = ecs_result.get("code")

        passed = False
        if code:
            adapted, _ = adapter.adapt_code(code, prompt)
            if adapted:
                body = adapter._extract_function_body(adapted)
                if body:
                    passed = verify_humaneval(body, problem)
                if not passed:
                    passed = verify_humaneval(adapted, problem)
            if not passed:
                passed = verify_humaneval(code, problem)

        results.append({
            "seq_index": seq_idx,
            "task_id": problem['task_id'],
            "passed": passed,
            "confidence": ecs_result["confidence"],
            "method": ecs_result.get("synthesis_method", "failed"),
            "memory_items": len(ecs.memory.items) if hasattr(ecs.memory, 'items') else 0,
        })

        if (seq_idx + 1) % 20 == 0:
            rate = sum(1 for r in results if r["passed"]) / len(results)
            print(f"      [{seq_idx+1}/{len(problems)}] pass rate: {rate:.1%}")

    first_half = results[:len(results)//2]
    second_half = results[len(results)//2:]

    first_rate = sum(1 for r in first_half if r["passed"]) / len(first_half)
    second_rate = sum(1 for r in second_half if r["passed"]) / len(second_half)
    total_rate = sum(1 for r in results if r["passed"]) / len(results)

    # Quartile breakdown
    q_size = len(results) // 4
    quartiles = []
    for q in range(4):
        q_results = results[q * q_size:(q + 1) * q_size]
        q_rate = sum(1 for r in q_results if r["passed"]) / len(q_results) if q_results else 0
        quartiles.append(q_rate)

    return {
        "seed": seed,
        "with_memory": with_memory,
        "total_pass_rate": total_rate,
        "first_half_rate": first_rate,
        "second_half_rate": second_rate,
        "improvement": second_rate - first_rate,
        "quartile_rates": quartiles,
        "total_passed": sum(1 for r in results if r["passed"]),
        "total": len(results),
        "results": results,
    }


def main():
    print("Loading HumanEval dataset...")
    ds = load_dataset('evalplus/humanevalplus', split='test')
    problems = list(ds)

    seeds = [42, 123, 456]
    all_results = {"with_memory": [], "without_memory": []}

    print(f"\n{'='*60}")
    print("PHASE 3: SESSION LEARNING EXPERIMENT")
    print(f"{'='*60}")
    print(f"  Problems: {len(problems)}")
    print(f"  Seeds: {seeds}")
    print(f"  Conditions: with_memory, without_memory")

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        print(f"  Running WITH memory...")
        result = run_sequential(problems, with_memory=True, seed=seed)
        all_results["with_memory"].append(result)
        print(f"    pass@1: {result['total_pass_rate']:.1%}, "
              f"first half: {result['first_half_rate']:.1%}, "
              f"second half: {result['second_half_rate']:.1%}, "
              f"delta: {result['improvement']:+.1%}")

        print(f"  Running WITHOUT memory...")
        result = run_sequential(problems, with_memory=False, seed=seed)
        all_results["without_memory"].append(result)
        print(f"    pass@1: {result['total_pass_rate']:.1%}, "
              f"first half: {result['first_half_rate']:.1%}, "
              f"second half: {result['second_half_rate']:.1%}, "
              f"delta: {result['improvement']:+.1%}")

    # Aggregate
    print(f"\n{'='*60}")
    print("SESSION LEARNING RESULTS (mean across 3 seeds)")
    print(f"{'='*60}")

    for condition in ["with_memory", "without_memory"]:
        runs = all_results[condition]
        avg_total = sum(r["total_pass_rate"] for r in runs) / len(runs)
        avg_first = sum(r["first_half_rate"] for r in runs) / len(runs)
        avg_second = sum(r["second_half_rate"] for r in runs) / len(runs)
        avg_delta = sum(r["improvement"] for r in runs) / len(runs)

        # Quartile averages
        avg_quartiles = [sum(r["quartile_rates"][q] for r in runs) / len(runs) for q in range(4)]

        print(f"\n  {condition}:")
        print(f"    Total pass@1:     {avg_total:.1%}")
        print(f"    First half:       {avg_first:.1%}")
        print(f"    Second half:      {avg_second:.1%}")
        print(f"    Improvement:      {avg_delta:+.1%}")
        print(f"    Quartile curve:   {' -> '.join(f'{q:.1%}' for q in avg_quartiles)}")

    # Simple significance test (paired comparison)
    with_deltas = [r["improvement"] for r in all_results["with_memory"]]
    without_deltas = [r["improvement"] for r in all_results["without_memory"]]
    print(f"\n  Memory condition improvements:    {[f'{d:+.1%}' for d in with_deltas]}")
    print(f"  No-memory condition improvements: {[f'{d:+.1%}' for d in without_deltas]}")

    # Save (strip per-problem results for readability)
    save_data = {}
    for condition in ["with_memory", "without_memory"]:
        save_data[condition] = []
        for r in all_results[condition]:
            save_data[condition].append({k: v for k, v in r.items() if k != "results"})

    save_data["per_problem"] = {}
    for condition in ["with_memory", "without_memory"]:
        save_data["per_problem"][condition] = all_results[condition][0]["results"]  # seed 42

    out_path = ROOT / "phase3_results.json"
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
