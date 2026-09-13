"""
Full 3-Condition Comparison: The experiment that determines the paper's story.

Conditions:
1. qwen_only:    Raw Qwen2.5-Coder-0.5B on HumanEval (no architecture)
2. ecs_no_neural: ECS structured tier (constraints + memory, no neural)
3. ecs_full:     ECS with neural integration + confidence arbitration

The relationship between these numbers determines the paper framing:
- ecs_full > qwen_only → "Architecture amplifies small models" (STRONG)
- ecs_full ≈ qwen_only → "Architecture adds calibration, not pass rate" (HONEST)
- ecs_full < qwen_only → "Architecture interferes" (DEBUG NEEDED)

Requires: ollama serve running with qwen2.5-coder:0.5b pulled.

Run: uv run python scripts/full_comparison.py
"""

import sys
import time
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from ecs.orchestrator import ECSOrchestrator
from ecs.neural.interface import NeuralInterface
from ecs.evaluation.signature_adapter import SignatureAdapter


def verify_humaneval(code: str, problem: Dict) -> bool:
    """Verify generated code against HumanEval test suite."""
    if not code:
        return False
    full_program = (
        f"{problem['prompt']}{code}\n\n"
        f"{problem['test']}\n\ncheck({problem['entry_point']})\n"
    )
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


def extract_problem_description(prompt: str) -> str:
    """Extract NL description from HumanEval prompt."""
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


def run_condition_qwen_only(problems: List[Dict]) -> List[Dict]:
    """Condition 1: Raw Qwen2.5-Coder-0.5B, no architecture."""
    interface = NeuralInterface()
    adapter = SignatureAdapter()

    if not interface.is_available():
        print("  [SKIP] Ollama not available")
        return []

    results = []
    for i, problem in enumerate(problems):
        prompt = problem['prompt']
        generation_prompt = (
            f"Complete the following Python function. "
            f"Provide ONLY the function body.\n\n{prompt}\n"
        )
        start = time.time()
        try:
            response = interface.generate_code(specification=generation_prompt)
            code = response.code
        except Exception:
            code = None
        elapsed = time.time() - start

        passed = False
        if code:
            passed = verify_humaneval(code, problem)
            if not passed:
                adapted, _ = adapter.adapt_code(code, prompt)
                if adapted:
                    body = adapter._extract_function_body(adapted)
                    if body:
                        passed = verify_humaneval(body, problem)

        results.append({
            "task_id": problem['task_id'],
            "passed": passed,
            "time": elapsed,
            "method": "qwen_only",
        })

        if (i + 1) % 20 == 0:
            rate = sum(1 for r in results if r["passed"]) / len(results)
            print(f"    [{i+1}/{len(problems)}] running pass rate: {rate:.1%}")

    return results


def run_condition_ecs_no_neural(problems: List[Dict]) -> List[Dict]:
    """Condition 2: ECS structured tier (no neural)."""
    orchestrator = ECSOrchestrator()
    adapter = SignatureAdapter()
    results = []

    for i, problem in enumerate(problems):
        prompt = problem['prompt']
        entry_point = problem['entry_point']
        desc = extract_problem_description(prompt)

        start = time.time()
        ecs_result = orchestrator.solve_problem(desc, prompt=prompt, entry_point=entry_point)
        elapsed = time.time() - start

        code = ecs_result.get("code")
        passed = False
        if code:
            adapted, info = adapter.adapt_code(code, prompt)
            if adapted:
                body = adapter._extract_function_body(adapted)
                if body:
                    passed = verify_humaneval(body, problem)
                if not passed:
                    passed = verify_humaneval(adapted, problem)
            if not passed:
                passed = verify_humaneval(code, problem)

        results.append({
            "task_id": problem['task_id'],
            "passed": passed,
            "ecs_success": ecs_result["success"],
            "confidence": ecs_result["confidence"],
            "time": elapsed,
            "method": ecs_result.get("synthesis_method", "failed"),
        })

        if (i + 1) % 20 == 0:
            rate = sum(1 for r in results if r["passed"]) / len(results)
            print(f"    [{i+1}/{len(problems)}] running pass rate: {rate:.1%}")

    return results


def run_condition_ecs_full(problems: List[Dict]) -> List[Dict]:
    """Condition 3: ECS with neural integration."""
    orchestrator = ECSOrchestrator()
    adapter = SignatureAdapter()

    if not orchestrator.neural.is_available():
        print("  [SKIP] Ollama not available — cannot run full tier")
        return []

    results = []
    for i, problem in enumerate(problems):
        prompt = problem['prompt']
        entry_point = problem['entry_point']
        desc = extract_problem_description(prompt)

        start = time.time()
        ecs_result = orchestrator.solve_problem(desc, prompt=prompt, entry_point=entry_point)
        elapsed = time.time() - start

        code = ecs_result.get("code")
        passed = False
        if code:
            adapted, info = adapter.adapt_code(code, prompt)
            if adapted:
                body = adapter._extract_function_body(adapted)
                if body:
                    passed = verify_humaneval(body, problem)
                if not passed:
                    passed = verify_humaneval(adapted, problem)
            if not passed:
                passed = verify_humaneval(code, problem)

        results.append({
            "task_id": problem['task_id'],
            "passed": passed,
            "ecs_success": ecs_result["success"],
            "confidence": ecs_result["confidence"],
            "time": elapsed,
            "method": ecs_result.get("synthesis_method", "failed"),
        })

        if (i + 1) % 20 == 0:
            rate = sum(1 for r in results if r["passed"]) / len(results)
            print(f"    [{i+1}/{len(problems)}] running pass rate: {rate:.1%}")

    return results


def print_comparison(all_results: Dict[str, List[Dict]]):
    """Print the comparison table."""
    print("\n" + "=" * 70)
    print("  3-CONDITION COMPARISON")
    print("=" * 70)

    for name, results in all_results.items():
        if not results:
            print(f"\n  {name}: SKIPPED (Ollama not available)")
            continue
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        avg_time = sum(r["time"] for r in results) / total * 1000
        print(f"\n  {name}:")
        print(f"    pass@1: {passed}/{total} ({passed/total:.1%})")
        print(f"    avg latency: {avg_time:.0f}ms")

    # Analysis
    qwen = all_results.get("qwen_only", [])
    ecs_no = all_results.get("ecs_no_neural", [])
    ecs_full = all_results.get("ecs_full", [])

    if qwen and ecs_no:
        qwen_rate = sum(1 for r in qwen if r["passed"]) / len(qwen)
        ecs_no_rate = sum(1 for r in ecs_no if r["passed"]) / len(ecs_no)

        print(f"\n  {'=' * 60}")
        print(f"  STRUCTURAL vs NEURAL:")
        print(f"    ECS structured ({ecs_no_rate:.1%}) vs Qwen alone ({qwen_rate:.1%})")
        if ecs_no_rate > qwen_rate:
            print(f"    → Templates BEAT raw model by {ecs_no_rate - qwen_rate:.1%}")
        else:
            print(f"    → Raw model beats templates by {qwen_rate - ecs_no_rate:.1%}")

    if qwen and ecs_full:
        qwen_rate = sum(1 for r in qwen if r["passed"]) / len(qwen)
        ecs_full_rate = sum(1 for r in ecs_full if r["passed"]) / len(ecs_full)
        delta = ecs_full_rate - qwen_rate

        print(f"\n  ARCHITECTURE VALUE:")
        print(f"    ECS full ({ecs_full_rate:.1%}) vs Qwen alone ({qwen_rate:.1%})")
        if delta > 0.05:
            print(f"    → ARCHITECTURE AMPLIFIES MODEL (+{delta:.1%})")
            print(f"    → PAPER: 'Cognitive Architecture Amplifies Small LMs'")
        elif delta > -0.02:
            print(f"    → ARCHITECTURE MATCHES MODEL (delta={delta:+.1%})")
            print(f"    → PAPER: 'Architecture adds calibration, not pass rate'")
        else:
            print(f"    → ARCHITECTURE INTERFERES ({delta:+.1%})")
            print(f"    → DEBUG: find where architecture loses neural performance")

    # Per-problem Venn analysis
    if qwen and ecs_no:
        qwen_passed = {r["task_id"] for r in qwen if r["passed"]}
        ecs_no_passed = {r["task_id"] for r in ecs_no if r["passed"]}

        only_qwen = qwen_passed - ecs_no_passed
        only_ecs = ecs_no_passed - qwen_passed
        both = qwen_passed & ecs_no_passed

        print(f"\n  COMPLEMENTARITY (Qwen vs ECS structured):")
        print(f"    Both solve: {len(both)}")
        print(f"    Only Qwen: {len(only_qwen)}")
        print(f"    Only ECS:  {len(only_ecs)}")
        print(f"    Union:     {len(qwen_passed | ecs_no_passed)}")
        print(f"    → Perfect oracle would get {len(qwen_passed | ecs_no_passed)}/164 "
              f"({len(qwen_passed | ecs_no_passed)/164:.1%})")

    print("=" * 70)


def main():
    print("Loading HumanEval dataset...")
    ds = load_dataset('evalplus/humanevalplus', split='test')
    problems = list(ds)
    max_n = int(sys.argv[1]) if len(sys.argv) > 1 else 164
    problems = problems[:max_n]
    print(f"Loaded {len(problems)} problems\n")

    all_results = {}

    # Condition 1: Qwen only
    print("[1/3] Qwen2.5-Coder-0.5B alone...")
    all_results["qwen_only"] = run_condition_qwen_only(problems)

    # Condition 2: ECS without neural
    print("\n[2/3] ECS structured tier (no neural)...")
    all_results["ecs_no_neural"] = run_condition_ecs_no_neural(problems)

    # Condition 3: ECS full
    print("\n[3/3] ECS full tier (with neural)...")
    all_results["ecs_full"] = run_condition_ecs_full(problems)

    # Report
    print_comparison(all_results)

    # Save
    output_path = Path(__file__).parent.parent / "evaluation_full_comparison.json"
    with open(output_path, "w") as f:
        save_data = {}
        for name, results in all_results.items():
            if results:
                total = len(results)
                passed = sum(1 for r in results if r["passed"])
                save_data[name] = {
                    "total": total,
                    "passed": passed,
                    "pass_rate": passed / total,
                    "results": results,
                }
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
