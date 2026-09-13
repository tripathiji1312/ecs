"""
Qwen2.5-Coder-0.5B baseline on HumanEval.

The critical missing number: what does the raw model achieve WITHOUT ECS?
This determines the paper's entire framing.

Requires: ollama serve running with qwen2.5-coder:0.5b pulled.

Run: uv run python scripts/qwen_baseline_humaneval.py
"""

import sys
import time
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
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


def extract_function_body(code: str, prompt: str) -> str:
    """Extract just the function body from generated code."""
    import re

    if not code:
        return ""

    entry_match = re.search(r'def\s+(\w+)', prompt)
    if not entry_match:
        return code

    func_name = entry_match.group(1)
    body_match = re.search(
        rf'def\s+{func_name}\s*\([^)]*\)[^:]*:(.*)',
        code, re.DOTALL
    )
    if body_match:
        return body_match.group(1)

    return code


def run_qwen_baseline(max_problems: int = 164):
    """Run Qwen2.5-Coder-0.5B directly on HumanEval without ECS."""
    print("Loading HumanEval dataset...")
    ds = load_dataset('evalplus/humanevalplus', split='test')
    problems = list(ds)[:max_problems]
    print(f"Loaded {len(problems)} problems")

    interface = NeuralInterface()
    adapter = SignatureAdapter()

    if not interface.is_available():
        print("\nERROR: Ollama is not running or qwen2.5-coder:0.5b is not pulled.")
        print("Start Ollama and pull the model:")
        print("  ollama serve &")
        print("  ollama pull qwen2.5-coder:0.5b")
        return None

    results = []

    print(f"\nEvaluating Qwen2.5-Coder-0.5B ALONE on HumanEval...")
    print("(No ECS architecture — raw model generation)")
    print("-" * 70)

    for i, problem in enumerate(problems):
        task_id = problem['task_id']
        entry_point = problem['entry_point']
        prompt = problem['prompt']

        start = time.time()

        # Direct generation: give the model the HumanEval prompt
        # and ask it to complete the function
        generation_prompt = (
            f"Complete the following Python function. "
            f"Provide ONLY the function body (the code after the def line and docstring). "
            f"Do not repeat the function signature.\n\n"
            f"{prompt}\n"
        )

        try:
            response = interface.generate_code(
                specification=generation_prompt,
                strategy_context=None,
                retrieved_knowledge=None
            )
            generated_code = response.code
        except Exception as e:
            generated_code = None

        elapsed = time.time() - start

        passed = False
        if generated_code:
            # Try direct submission
            passed = verify_humaneval(generated_code, problem)

            # Try extracting body
            if not passed:
                body = extract_function_body(generated_code, prompt)
                if body:
                    passed = verify_humaneval(body, problem)

            # Try with signature adapter
            if not passed:
                adapted, info = adapter.adapt_code(generated_code, prompt)
                if adapted:
                    body = adapter._extract_function_body(adapted)
                    if body:
                        passed = verify_humaneval(body, problem)
                    if not passed:
                        passed = verify_humaneval(adapted, problem)

        results.append({
            "task_id": task_id,
            "entry_point": entry_point,
            "passed": passed,
            "time": elapsed,
            "code_length": len(generated_code) if generated_code else 0,
        })

        status = "PASS" if passed else "FAIL"
        print(f"  [{i+1:3d}/{len(problems)}] {status} {task_id}: "
              f"{entry_point[:35]}...")

    # Results
    print("\n" + "=" * 70)
    print("  QWEN2.5-CODER-0.5B BASELINE RESULTS")
    print("=" * 70)

    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)

    print(f"\n  HumanEval pass@1: {passed_count}/{total} ({passed_count/total:.1%})")
    print(f"  Average latency: {sum(r['time'] for r in results)/len(results)*1000:.0f}ms")
    print("=" * 70)

    # Save
    output_path = Path(__file__).parent.parent / "evaluation_qwen_baseline.json"
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "total": total,
                "passed": passed_count,
                "pass_rate": passed_count / total,
                "avg_latency_ms": sum(r['time'] for r in results) / len(results) * 1000,
                "model": "qwen2.5-coder:0.5b",
                "method": "direct_generation",
            },
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return {
        "passed": passed_count,
        "total": total,
        "pass_rate": passed_count / total,
    }


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 164
    run_qwen_baseline(max_problems=n)
