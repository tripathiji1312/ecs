"""
HumanEval Evaluation for ECS v3.

Evaluates ECS structured tier on HumanEval (164 problems).
Measures: pass@1, confidence calibration, method distribution.

Run: uv run python scripts/humaneval_eval.py
"""

import sys
import time
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from ecs.orchestrator import ECSOrchestrator
from ecs.verification.sandbox import SafeExecutor
from ecs.evaluation.signature_adapter import SignatureAdapter


def extract_problem_description(prompt: str) -> str:
    """Extract natural language description from HumanEval prompt."""
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
    """Verify generated code against HumanEval test suite."""
    if not code:
        return False

    full_program = f"{problem['prompt']}{code}\n\n{problem['test']}\n\ncheck({problem['entry_point']})\n"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_program)
        f.flush()

        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False


def extract_function_body(code: str, prompt: str) -> Optional[str]:
    """Extract just the function body from generated code."""
    if not code:
        return None

    # If the code starts with the function signature, extract body only
    func_match = re.search(r'def\s+\w+\s*\(', prompt)
    if func_match:
        entry_point_match = re.search(r'def\s+(\w+)', prompt)
        if entry_point_match:
            func_name = entry_point_match.group(1)
            # If generated code contains the function def, extract body
            body_match = re.search(
                rf'def\s+{func_name}\s*\([^)]*\)[^:]*:(.*)',
                code, re.DOTALL
            )
            if body_match:
                return body_match.group(1)

    # Otherwise return code as-is (it might be just the body)
    return code


def run_evaluation(max_problems: int = 164):
    """Run HumanEval evaluation."""
    print("Loading HumanEval dataset...")
    ds = load_dataset('evalplus/humanevalplus', split='test')
    problems = list(ds)[:max_problems]
    print(f"Loaded {len(problems)} problems")

    orchestrator = ECSOrchestrator()
    adapter = SignatureAdapter()
    results = []

    print(f"\nEvaluating ECS v3 (structured tier) on HumanEval...")
    print(f"  SignatureAdapter: ENABLED")
    print("-" * 70)

    for i, problem in enumerate(problems):
        task_id = problem['task_id']
        entry_point = problem['entry_point']
        prompt = problem['prompt']

        # Extract problem description
        description = extract_problem_description(prompt)

        start = time.time()

        # Solve with ECS (pass prompt + entry_point for compositional synthesis)
        ecs_result = orchestrator.solve_problem(
            description, prompt=prompt, entry_point=entry_point
        )

        elapsed = time.time() - start

        # Get the generated code
        generated_code = ecs_result.get("code")

        # Try to verify against HumanEval tests
        passed = False
        adapted_info = {}
        if generated_code:
            # Strategy 1: Use SignatureAdapter to align function signature
            adapted_code, adapted_info = adapter.adapt_code(
                generated_code, prompt
            )
            if adapted_code:
                # Extract body from adapted code for HumanEval format
                body = adapter._extract_function_body(adapted_code)
                if body:
                    passed = verify_humaneval(body, problem)

                # If body extraction didn't work, try full adapted code
                if not passed:
                    passed = verify_humaneval(adapted_code, problem)

            # Strategy 2: Fallback to original approaches
            if not passed:
                body = extract_function_body(generated_code, prompt)
                if body:
                    passed = verify_humaneval(body, problem)

            if not passed:
                passed = verify_humaneval(generated_code, problem)

        results.append({
            "task_id": task_id,
            "entry_point": entry_point,
            "description": description[:80],
            "passed": passed,
            "ecs_success": ecs_result["success"],
            "confidence": ecs_result["confidence"],
            "method": ecs_result.get("synthesis_method", "failed"),
            "strategy": ecs_result.get("strategy_used", "none"),
            "time": elapsed,
            "code_length": len(generated_code) if generated_code else 0,
            "emergence": ecs_result.get("emergence_findings", []),
            "adapted": adapted_info.get("adapted", False),
            "adaptation_details": adapted_info,
        })

        status = "PASS" if passed else "FAIL"
        ecs_status = "ecs-ok" if ecs_result["success"] else "ecs-no"
        adapt_flag = "A" if adapted_info.get("adapted") else " "
        print(f"  [{i+1:3d}/{len(problems)}] {status} ({ecs_status}) {adapt_flag} "
              f"[{ecs_result.get('synthesis_method', 'none'):21}] "
              f"{task_id}: {description[:35]}...")

    # Analysis
    print("\n" + "=" * 70)
    print("  HUMANEVAL RESULTS (with SignatureAdapter)")
    print("=" * 70)

    passed_count = sum(1 for r in results if r["passed"])
    ecs_success_count = sum(1 for r in results if r["ecs_success"])
    adapted_count = sum(1 for r in results if r.get("adapted"))
    total = len(results)

    print(f"\n  HumanEval pass@1: {passed_count}/{total} ({passed_count/total:.1%})")
    print(f"  ECS internal success: {ecs_success_count}/{total} ({ecs_success_count/total:.1%})")
    print(f"  Signature adaptations: {adapted_count}/{total} ({adapted_count/total:.1%})")
    print(f"  Adapter success rate: {adapter.success_count}/{adapter.adaptation_count} "
          f"({adapter.success_count/max(1,adapter.adaptation_count):.1%})")

    # Method breakdown
    methods = {}
    method_passes = {}
    for r in results:
        m = r["method"]
        methods[m] = methods.get(m, 0) + 1
        if r["passed"]:
            method_passes[m] = method_passes.get(m, 0) + 1

    print(f"\n  Synthesis method distribution:")
    for m, count in sorted(methods.items(), key=lambda x: -x[1]):
        passes = method_passes.get(m, 0)
        print(f"    {m:25} {count:3} problems, {passes:3} passed ({passes/count:.0%} if count > 0)")

    # Confidence calibration
    high_conf = [r for r in results if r["confidence"] > 0.6]
    low_conf = [r for r in results if r["confidence"] <= 0.6]

    high_pass = sum(1 for r in high_conf if r["passed"]) / len(high_conf) if high_conf else 0
    low_pass = sum(1 for r in low_conf if r["passed"]) / len(low_conf) if low_conf else 0

    print(f"\n  Confidence calibration:")
    print(f"    High confidence (>0.6): {len(high_conf)} problems, {high_pass:.1%} pass rate")
    print(f"    Low confidence (<=0.6): {len(low_conf)} problems, {low_pass:.1%} pass rate")
    print(f"    Calibrated: {'Yes' if high_pass > low_pass else 'No'}")

    # Learning over time
    first_half = results[:len(results)//2]
    second_half = results[len(results)//2:]
    first_pass = sum(1 for r in first_half if r["passed"]) / len(first_half)
    second_pass = sum(1 for r in second_half if r["passed"]) / len(second_half)

    print(f"\n  Learning effect:")
    print(f"    First half:  {first_pass:.1%}")
    print(f"    Second half: {second_pass:.1%}")

    # Emergence
    total_emergence = sum(len(r["emergence"]) for r in results)
    print(f"\n  Emergence events: {total_emergence}")

    print(f"\n  Average latency: {sum(r['time'] for r in results)/len(results)*1000:.0f}ms")
    print("=" * 70)

    # Save results
    output_path = Path(__file__).parent.parent / "evaluation_humaneval.json"
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "total": total,
                "humaneval_passed": passed_count,
                "humaneval_pass_rate": passed_count / total,
                "ecs_success": ecs_success_count,
                "ecs_success_rate": ecs_success_count / total,
                "avg_latency_ms": sum(r['time'] for r in results) / len(results) * 1000,
                "signature_adapter": {
                    "adaptations_attempted": adapter.adaptation_count,
                    "adaptations_succeeded": adapter.success_count,
                    "adapted_problems": adapted_count,
                },
                "confidence_calibration": {
                    "high_conf_count": len(high_conf),
                    "high_conf_pass_rate": high_pass,
                    "low_conf_count": len(low_conf),
                    "low_conf_pass_rate": low_pass,
                },
                "methods": methods,
                "method_passes": method_passes,
                "emergence_events": total_emergence,
            },
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 164
    run_evaluation(max_problems=n)
