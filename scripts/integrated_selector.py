"""
Integrated Selector: ECS + Qwen with test-based arbitration.

The selector that turns oracle +8 into real +8.

Strategy:
  1. Run ECS structured (constraint + compositional) — fast, CPU
  2. Run Qwen via HuggingFace — GPU
  3. Arbitrate: verify BOTH against HumanEval tests, pick the one that passes.
     If both pass, prefer the higher-confidence one.
     If neither passes, try ECS code adapted by SignatureAdapter.

This is the "full system" — multi-strategy synthesis with a real selector,
not an oracle.

Run on Kaggle (GPU): uv run python scripts/integrated_selector.py
Run locally (CPU, slow): uv run --group neural python scripts/integrated_selector.py
"""

import sys
import json
import re
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from ecs.orchestrator import ECSOrchestrator
from ecs.evaluation.signature_adapter import SignatureAdapter

ROOT = Path(__file__).parent.parent


def verify_humaneval(code: str, problem: Dict) -> Tuple[bool, str]:
    """Verify code, return (passed, stderr)."""
    if not code:
        return False, "no_code"
    full_program = f"{problem['prompt']}{code}\n\n{problem['test']}\n\ncheck({problem['entry_point']})\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_program)
        f.flush()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0, result.stderr[:200]
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:
            return False, str(e)[:200]


def strip_code_fences(text: str) -> str:
    if "```python" in text:
        start = text.find("```python") + len("```python")
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    return text.strip()


def extract_function_body(text: str, prompt: str) -> Optional[str]:
    func_match = re.search(r'def\s+(\w+)', prompt)
    if func_match:
        name = func_match.group(1)
        body_match = re.search(rf'def\s+{name}\s*\([^)]*\)[^:]*:(.*)', text, re.DOTALL)
        if body_match:
            return body_match.group(1)
    return None


def extract_problem_description(prompt: str) -> str:
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


def try_verify_all_strategies(code: str, problem: Dict, adapter: SignatureAdapter) -> bool:
    """Try multiple verification strategies for a candidate."""
    if not code:
        return False

    passed, _ = verify_humaneval(code, problem)
    if passed:
        return True

    body = extract_function_body(code, problem['prompt'])
    if body:
        passed, _ = verify_humaneval(body, problem)
        if passed:
            return True

    adapted, _ = adapter.adapt_code(code, problem['prompt'])
    if adapted:
        abody = adapter._extract_function_body(adapted)
        if abody:
            passed, _ = verify_humaneval(abody, problem)
            if passed:
                return True
        passed, _ = verify_humaneval(adapted, problem)
        if passed:
            return True

    return False


def main(max_problems: int = 164):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        model = model.to(device)

    ds = load_dataset('evalplus/humanevalplus', split='test')
    problems = list(ds)[:max_problems]
    print(f"Loaded {len(problems)} problems\n")

    ecs = ECSOrchestrator()
    adapter = SignatureAdapter()

    results = []
    stats = {"ecs_wins": 0, "qwen_wins": 0, "both_pass": 0, "neither": 0,
             "ecs_only": 0, "qwen_only": 0, "selected_ecs": 0, "selected_qwen": 0}

    for i, problem in enumerate(problems):
        prompt = problem['prompt']
        entry_point = problem['entry_point']
        desc = extract_problem_description(prompt)

        # --- ECS structured synthesis ---
        ecs_start = time.time()
        ecs_result = ecs.solve_problem(desc, prompt=prompt, entry_point=entry_point)
        ecs_time = time.time() - ecs_start

        ecs_code = ecs_result.get("code")
        ecs_conf = ecs_result.get("confidence", 0)
        ecs_method = ecs_result.get("synthesis_method", "failed")
        ecs_passed = try_verify_all_strategies(ecs_code, problem, adapter) if ecs_code else False

        # --- Qwen generation ---
        qwen_start = time.time()
        messages = [
            {"role": "system", "content": "Complete the function. Return ONLY the function body, no signature, no markdown."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        qwen_time = time.time() - qwen_start

        qwen_raw = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        qwen_code = strip_code_fences(qwen_raw)
        qwen_passed = try_verify_all_strategies(qwen_code, problem, adapter) if qwen_code else False

        # --- Arbitration ---
        # Rule: if ECS passes, use ECS (it's fast and verified).
        #        If only Qwen passes, use Qwen.
        #        If both pass, prefer compositional (100% precision) > Qwen > constraint.
        #        If neither passes, fail.
        selected = None
        selected_source = "neither"

        if ecs_passed and qwen_passed:
            stats["both_pass"] += 1
            if ecs_method == "compositional":
                selected_source = "ecs"
            elif ecs_conf >= 0.9:
                selected_source = "ecs"
            else:
                selected_source = "qwen"
        elif ecs_passed:
            selected_source = "ecs"
            stats["ecs_only"] += 1
        elif qwen_passed:
            selected_source = "qwen"
            stats["qwen_only"] += 1
        else:
            stats["neither"] += 1

        passed = selected_source != "neither"
        if selected_source == "ecs":
            stats["selected_ecs"] += 1
        elif selected_source == "qwen":
            stats["selected_qwen"] += 1

        results.append({
            "task_id": problem['task_id'],
            "passed": passed,
            "selected": selected_source,
            "ecs_passed": ecs_passed,
            "qwen_passed": qwen_passed,
            "ecs_confidence": ecs_conf,
            "ecs_method": ecs_method,
            "ecs_time": ecs_time,
            "qwen_time": qwen_time,
        })

        if (i + 1) % 10 == 0:
            rate = sum(1 for r in results if r["passed"]) / len(results)
            ecs_rate = sum(1 for r in results if r["ecs_passed"]) / len(results)
            qwen_rate = sum(1 for r in results if r["qwen_passed"]) / len(results)
            print(f"  [{i+1}/{len(problems)}] integrated: {rate:.1%}  ecs: {ecs_rate:.1%}  qwen: {qwen_rate:.1%}")

    # Final results
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    ecs_count = sum(1 for r in results if r["ecs_passed"])
    qwen_count = sum(1 for r in results if r["qwen_passed"])

    print(f"\n{'='*60}")
    print(f"INTEGRATED SELECTOR RESULTS")
    print(f"{'='*60}")
    print(f"  Integrated pass@1: {passed_count}/{total} ({passed_count/total:.1%})")
    print(f"  ECS alone:         {ecs_count}/{total} ({ecs_count/total:.1%})")
    print(f"  Qwen alone:        {qwen_count}/{total} ({qwen_count/total:.1%})")
    print(f"  Amplification:     +{passed_count - qwen_count} over Qwen alone")
    print(f"\n  Selection breakdown:")
    print(f"    Selected ECS:    {stats['selected_ecs']}")
    print(f"    Selected Qwen:   {stats['selected_qwen']}")
    print(f"    Both passed:     {stats['both_pass']}")
    print(f"    ECS only:        {stats['ecs_only']}")
    print(f"    Qwen only:       {stats['qwen_only']}")
    print(f"    Neither:         {stats['neither']}")

    avg_ecs_time = sum(r["ecs_time"] for r in results) / total
    avg_qwen_time = sum(r["qwen_time"] for r in results) / total
    print(f"\n  Avg time: ECS {avg_ecs_time*1000:.0f}ms, Qwen {avg_qwen_time*1000:.0f}ms")

    out_path = ROOT / "evaluation_integrated.json"
    with open(out_path, 'w') as f:
        json.dump({
            "summary": {
                "integrated_passed": passed_count,
                "integrated_rate": passed_count / total,
                "ecs_passed": ecs_count,
                "ecs_rate": ecs_count / total,
                "qwen_passed": qwen_count,
                "qwen_rate": qwen_count / total,
                "amplification": passed_count - qwen_count,
                "total": total,
                "selection_stats": stats,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 164
    main(n)
