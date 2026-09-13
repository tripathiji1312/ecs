"""
Phase 0 Analysis: Venn diagram + failure classification.

Loads results from humaneval_eval.py and hf_qwen_baseline.py,
computes per-problem overlap and classifies ECS failures.

Run: uv run python scripts/phase0_analysis.py
"""

import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent


def classify_failure(code: str, problem: Dict) -> str:
    """Run code against HumanEval tests and classify the failure mode."""
    if not code:
        return "no_code"
    full_program = f"{problem['prompt']}{code}\n\n{problem['test']}\n\ncheck({problem['entry_point']})\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_program)
        f.flush()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return "actually_passes"
            stderr = result.stderr
            if "NameError" in stderr:
                return "naming"
            if "TypeError" in stderr:
                return "type_mismatch"
            if "AssertionError" in stderr or "AssertionError" in stderr:
                return "logic_error"
            if "SyntaxError" in stderr:
                return "syntax"
            if "IndexError" in stderr or "KeyError" in stderr:
                return "edge_case"
            return "other_runtime"
        except subprocess.TimeoutExpired:
            return "timeout"
        except Exception:
            return "crash"


def main():
    ecs_path = ROOT / "evaluation_humaneval.json"
    qwen_path = ROOT / "evaluation_qwen_hf_baseline.json"

    if not ecs_path.exists():
        print(f"Missing {ecs_path} — run: uv run python scripts/humaneval_eval.py")
        return
    if not qwen_path.exists():
        print(f"Missing {qwen_path} — run: uv run --group neural python scripts/hf_qwen_baseline.py")
        return

    ecs_data = json.loads(ecs_path.read_text())
    qwen_data = json.loads(qwen_path.read_text())

    ecs_results = {r["task_id"]: r for r in ecs_data["results"]}
    qwen_results = {r["task_id"]: r for r in qwen_data["results"]}

    all_tasks = sorted(set(ecs_results) | set(qwen_results))

    both, only_ecs, only_qwen, neither = [], [], [], []
    for tid in all_tasks:
        ep = ecs_results.get(tid, {}).get("passed", False)
        qp = qwen_results.get(tid, {}).get("passed", False)
        if ep and qp:
            both.append(tid)
        elif ep:
            only_ecs.append(tid)
        elif qp:
            only_qwen.append(tid)
        else:
            neither.append(tid)

    print("=" * 60)
    print("VENN DIAGRAM: ECS-structured vs Qwen-0.5B-Instruct")
    print("=" * 60)
    print(f"  Both solve:      {len(both)}")
    print(f"  Only ECS:        {len(only_ecs)}")
    print(f"  Only Qwen:       {len(only_qwen)}")
    print(f"  Neither:         {len(neither)}")
    print(f"  Union:           {len(both) + len(only_ecs) + len(only_qwen)}")
    print(f"  ECS total:       {len(both) + len(only_ecs)}")
    print(f"  Qwen total:      {len(both) + len(only_qwen)}")

    if only_ecs:
        print(f"\n  ECS-unique problems: {only_ecs}")

    # Kill gate
    print(f"\n{'='*60}")
    print("KILL GATE EVALUATION")
    print("=" * 60)
    ecs_unique = len(only_ecs)
    print(f"  ECS uniquely solves: {ecs_unique} problems (need >= 3)")

    # Failure classification
    print(f"\n{'='*60}")
    print("ECS FAILURE CLASSIFICATION")
    print("=" * 60)

    ecs_failures = [tid for tid in all_tasks
                    if not ecs_results.get(tid, {}).get("passed", False)]

    from datasets import load_dataset
    ds = load_dataset('evalplus/humanevalplus', split='test')
    problems_by_id = {p['task_id']: p for p in ds}

    categories = {}
    for tid in ecs_failures:
        code = ecs_results.get(tid, {}).get("code")
        if code is None:
            method = ecs_results.get(tid, {}).get("method", "failed")
            if method == "failed":
                cat = "no_code"
            else:
                cat = classify_failure(code, problems_by_id[tid])
        else:
            cat = classify_failure(code, problems_by_id[tid])
        categories.setdefault(cat, []).append(tid)

    for cat, tasks in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"  {cat:20s}: {len(tasks):3d}")

    fixable = len(categories.get("naming", [])) + len(categories.get("type_mismatch", []))
    print(f"\n  Signature-fixable (naming + type): {fixable} (need >= 10)")

    # Verdict
    print(f"\n{'='*60}")
    if ecs_unique >= 3 or fixable >= 10:
        print("VERDICT: PROCEED — amplification story has legs")
    else:
        print("VERDICT: PIVOT — consider calibration-only paper (bronze tier)")
    print("=" * 60)

    # Save
    out = {
        "venn": {"both": both, "only_ecs": only_ecs, "only_qwen": only_qwen, "neither": neither},
        "failure_categories": {k: v for k, v in categories.items()},
        "kill_gate": {"ecs_unique": ecs_unique, "fixable": fixable,
                      "proceed": ecs_unique >= 3 or fixable >= 10},
    }
    out_path = ROOT / "phase0_analysis.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
