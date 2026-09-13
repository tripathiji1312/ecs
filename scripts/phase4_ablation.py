"""
Phase 4: Selection-focused ablation on HumanEval (164 problems).

Ablation around SELECTION, not synthesis:
  1. Full ECS (baseline)
  2. No HDC memory (disable memory retrieval)
  3. No compositional synthesis
  4. No constraint inference
  5. No confidence gating (skip verification)
  6. No workspace auction (skip auction phase)

Run: uv run python scripts/phase4_ablation.py
"""

import sys
import json
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


def run_ablation_condition(name: str, setup_fn, problems: List[Dict]) -> Dict:
    """Run one ablation condition on HumanEval."""
    ecs = ECSOrchestrator()
    setup_fn(ecs)
    adapter = SignatureAdapter()

    results = []
    for i, problem in enumerate(problems):
        prompt = problem['prompt']
        entry_point = problem['entry_point']
        desc = extract_problem_description(prompt)

        start = time.time()
        ecs_result = ecs.solve_problem(desc, prompt=prompt, entry_point=entry_point)
        elapsed = time.time() - start

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
            "task_id": problem['task_id'],
            "passed": passed,
            "confidence": ecs_result["confidence"],
            "method": ecs_result.get("synthesis_method", "failed"),
            "time": elapsed,
        })

        if (i + 1) % 40 == 0:
            rate = sum(1 for r in results if r["passed"]) / len(results)
            print(f"      [{i+1}/{len(problems)}] pass rate: {rate:.1%}")

    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)

    methods = {}
    for r in results:
        m = r["method"]
        methods[m] = methods.get(m, 0) + 1

    method_passes = {}
    for r in results:
        if r["passed"]:
            m = r["method"]
            method_passes[m] = method_passes.get(m, 0) + 1

    return {
        "name": name,
        "passed": passed_count,
        "total": total,
        "pass_rate": passed_count / total,
        "avg_confidence": sum(r["confidence"] for r in results) / total,
        "avg_time_ms": sum(r["time"] for r in results) / total * 1000,
        "methods": methods,
        "method_passes": method_passes,
        "results": results,
    }


# Ablation setup functions
def setup_full(ecs):
    pass

def setup_no_memory(ecs):
    ecs._retrieve_code_from_memory = lambda problem: None

def setup_no_compositional(ecs):
    ecs._try_compositional_synthesis = lambda problem: None

def setup_no_constraints(ecs):
    ecs.constraint_engine.CONSTRAINT_PATTERNS = {}

def setup_no_confidence_gate(ecs):
    original_verify = ecs._verify_with_confidence_gate
    def skip_verify(synth_result, problem):
        return synth_result
    ecs._verify_with_confidence_gate = skip_verify

def setup_no_auction(ecs):
    ecs.workspace.conduct_auction = lambda bids: bids[:1] if bids else []


def main():
    print("Loading HumanEval dataset...")
    ds = load_dataset('evalplus/humanevalplus', split='test')
    problems = list(ds)

    conditions = [
        ("Full ECS", setup_full),
        ("No HDC memory", setup_no_memory),
        ("No compositional", setup_no_compositional),
        ("No constraint inference", setup_no_constraints),
        ("No confidence gating", setup_no_confidence_gate),
        ("No workspace auction", setup_no_auction),
    ]

    print(f"\n{'='*70}")
    print("  PHASE 4: SELECTION-FOCUSED ABLATION ON HUMANEVAL")
    print(f"{'='*70}")
    print(f"  Problems: {len(problems)}")
    print(f"  Conditions: {len(conditions)}")

    all_results = {}
    for i, (name, setup_fn) in enumerate(conditions, 1):
        print(f"\n[{i}/{len(conditions)}] {name}...")
        result = run_ablation_condition(name, setup_fn, problems)
        all_results[name] = result
        print(f"    pass@1: {result['passed']}/{result['total']} ({result['pass_rate']:.1%}), "
              f"avg_conf: {result['avg_confidence']:.3f}, "
              f"avg_time: {result['avg_time_ms']:.0f}ms")

    # Report
    print(f"\n{'='*70}")
    print("  ABLATION RESULTS (HumanEval, 164 problems)")
    print(f"{'='*70}")

    baseline = all_results["Full ECS"]
    baseline_rate = baseline["pass_rate"]

    print(f"\n  {'Condition':<30} {'Passed':<10} {'Rate':<8} {'Delta':<8} {'Conf':<7}")
    print(f"  {'-'*65}")

    for name, result in all_results.items():
        delta = result["pass_rate"] - baseline_rate
        delta_str = f"{delta:+.1%}" if name != "Full ECS" else "---"
        print(f"  {name:<30} {result['passed']:>3}/{result['total']:<5} "
              f"{result['pass_rate']:<8.1%} {delta_str:<8} {result['avg_confidence']:.3f}")

    # Component importance
    print(f"\n  Component Importance (by pass rate drop):")
    drops = []
    for name, result in all_results.items():
        if name == "Full ECS":
            continue
        drop = baseline_rate - result["pass_rate"]
        drops.append((name, drop, result["pass_rate"]))

    drops.sort(key=lambda x: -x[1])
    for i, (name, drop, rate) in enumerate(drops, 1):
        bar = "#" * max(1, int(drop * 200))
        print(f"    {i}. {name:<30} -{drop:.1%}  ({rate:.1%}) {bar}")

    # Save (strip per-problem for top-level)
    save_data = {}
    for name, result in all_results.items():
        save_data[name] = {k: v for k, v in result.items() if k != "results"}
    save_data["per_problem"] = {name: result["results"] for name, result in all_results.items()}

    out_path = ROOT / "phase4_results.json"
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
