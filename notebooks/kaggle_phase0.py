"""
Kaggle notebook script for Phase 0: Qwen-0.5B-Instruct baseline on HumanEval.

Usage on Kaggle:
  1. Create a new Kaggle notebook with GPU T4 x2 accelerator
  2. Upload this as a script or paste into a cell
  3. Run — takes ~20-30 min on T4

The script clones the repo, installs deps, runs the baseline, and prints results.
"""

# %% [markdown]
# # ECS Phase 0: Qwen-0.5B-Instruct HumanEval Baseline
# GPU: T4 | ~20-30 min

# %%
import subprocess, os

# Clone repo (fill in your URL)
REPO_URL = "https://github.com/tripathiji1312/ecs.git"  # <-- UPDATE THIS
REPO_DIR = "/kaggle/working/ecs"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)

os.chdir(REPO_DIR)

# %%
# Install dependencies
subprocess.run(["pip", "install", "-e", "."], check=True)
subprocess.run(["pip", "install", "transformers", "torch", "accelerate"], check=True)

# %%
# Run the baseline
subprocess.run(["python", "scripts/hf_qwen_baseline.py"], check=True)

# %%
# Show results
import json
with open("evaluation_qwen_hf_baseline.json") as f:
    data = json.load(f)
print(f"pass@1: {data['passed']}/{data['total']} ({data['pass_at_1']:.1%})")
print(f"Device: {data['device']}")
