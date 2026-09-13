"""
HuggingFace Qwen-0.5B-Instruct baseline on HumanEval.

Standalone — no ECS architecture, no Ollama.
Auto-detects CUDA (FP16) or falls back to CPU (FP32).

Run locally:  uv run --group neural python scripts/hf_qwen_baseline.py
Run on Kaggle: see notebooks/kaggle_phase0.ipynb
"""

import sys
import json
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset


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


def extract_function_body(text: str, prompt: str) -> str | None:
    import re
    func_match = re.search(r'def\s+(\w+)', prompt)
    if func_match:
        name = func_match.group(1)
        body_match = re.search(rf'def\s+{name}\s*\([^)]*\)[^:]*:(.*)', text, re.DOTALL)
        if body_match:
            return body_match.group(1)
    return None


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
        model_name, torch_dtype=dtype, device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        model = model.to(device)

    ds = load_dataset('evalplus/humanevalplus', split='test')
    problems = list(ds)[:max_problems]
    print(f"Loaded {len(problems)} problems\n")

    results = []
    for i, problem in enumerate(problems):
        prompt = problem['prompt']
        entry_point = problem['entry_point']

        messages = [
            {"role": "system", "content": "You are a helpful coding assistant. Complete the function body only."},
            {"role": "user", "content": f"Complete the following Python function. Provide ONLY the function body (the code after the def line and docstring). Do not repeat the function signature.\n\n{prompt}\n"},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        start = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - start

        generated = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        passed = False
        if generated:
            passed = verify_humaneval(generated, problem)
            if not passed:
                body = extract_function_body(generated, prompt)
                if body:
                    passed = verify_humaneval(body, problem)

        results.append({
            "task_id": problem['task_id'],
            "passed": passed,
            "time": elapsed,
            "generated_code": generated[:500] if generated else None,
        })

        if (i + 1) % 20 == 0:
            rate = sum(1 for r in results if r["passed"]) / len(results)
            print(f"  [{i+1}/{len(problems)}] pass rate: {rate:.1%} ({elapsed:.1f}s)")

    passed_count = sum(1 for r in results if r["passed"])
    print(f"\n{'='*50}")
    print(f"Qwen2.5-Coder-0.5B-Instruct (HuggingFace, {device})")
    print(f"pass@1: {passed_count}/{len(results)} ({passed_count/len(results):.1%})")

    out_path = Path(__file__).parent.parent / "evaluation_qwen_hf_baseline.json"
    with open(out_path, 'w') as f:
        json.dump({"model": model_name, "device": device, "pass_at_1": passed_count / len(results),
                    "passed": passed_count, "total": len(results), "results": results}, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 164
    main(n)
