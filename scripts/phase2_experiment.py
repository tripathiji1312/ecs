"""
Phase 2: Core Experiment — 4-condition comparison + confidence analysis.

Conditions:
  (a) Qwen alone pass@1 (from hf_qwen_baseline.py results)
  (b) Best-of-K oracle (theoretical upper bound from K=16 samples)
  (c) Majority voting (most common output from K=16 samples)
  (d) ECS full (multi-strategy orchestration with neural)

For (a), uses existing evaluation_qwen_hf_baseline.json.
For (b)/(c), requires GPU — generates K=16 samples per problem.
For (d), requires GPU — ECS with HF backend.

This script also computes confidence analysis on ECS structured results:
  - Selective prediction curve (precision vs coverage)
  - ECE (Expected Calibration Error)
  - Brier score

The confidence analysis runs from existing JSON files (no GPU needed).

Run: uv run python scripts/phase2_experiment.py [--confidence-only]
"""

import sys
import json
import math
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent


def compute_confidence_analysis():
    """Compute calibration metrics from existing ECS evaluation results."""
    ecs_path = ROOT / "evaluation_humaneval.json"
    if not ecs_path.exists():
        print("Missing evaluation_humaneval.json")
        return None

    data = json.loads(ecs_path.read_text())
    results = data["results"]

    confs = [(r["confidence"], r["passed"]) for r in results]
    confs.sort(key=lambda x: -x[0])

    # Selective prediction curve: at each threshold, what precision and coverage?
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    selective = []
    for t in thresholds:
        above = [(c, p) for c, p in confs if c >= t]
        if not above:
            selective.append({"threshold": t, "coverage": 0, "precision": 0, "count": 0})
            continue
        coverage = len(above) / len(confs)
        precision = sum(1 for _, p in above if p) / len(above)
        selective.append({"threshold": t, "coverage": coverage, "precision": precision, "count": len(above)})

    # ECE: Expected Calibration Error (10 bins)
    n_bins = 10
    bins = [[] for _ in range(n_bins)]
    for c, p in confs:
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx].append((c, 1 if p else 0))

    ece = 0.0
    bin_details = []
    for i, b in enumerate(bins):
        if not b:
            bin_details.append({"bin": f"{i/n_bins:.1f}-{(i+1)/n_bins:.1f}", "count": 0, "avg_conf": 0, "accuracy": 0, "gap": 0})
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        accuracy = sum(p for _, p in b) / len(b)
        gap = abs(accuracy - avg_conf)
        ece += gap * len(b) / len(confs)
        bin_details.append({"bin": f"{i/n_bins:.1f}-{(i+1)/n_bins:.1f}", "count": len(b), "avg_conf": round(avg_conf, 3), "accuracy": round(accuracy, 3), "gap": round(gap, 3)})

    # Brier score
    brier = sum((c - (1 if p else 0)) ** 2 for c, p in confs) / len(confs)

    # Per-method breakdown
    method_stats = {}
    for r in results:
        m = r.get("method", "unknown")
        if m not in method_stats:
            method_stats[m] = {"total": 0, "passed": 0, "confidences": []}
        method_stats[m]["total"] += 1
        if r["passed"]:
            method_stats[m]["passed"] += 1
        method_stats[m]["confidences"].append(r["confidence"])

    for m in method_stats:
        s = method_stats[m]
        s["pass_rate"] = s["passed"] / s["total"] if s["total"] else 0
        s["avg_confidence"] = sum(s["confidences"]) / len(s["confidences"]) if s["confidences"] else 0
        del s["confidences"]

    return {
        "selective_prediction": selective,
        "ece": round(ece, 4),
        "brier_score": round(brier, 4),
        "calibration_bins": bin_details,
        "method_breakdown": method_stats,
        "total_problems": len(confs),
        "total_passed": sum(1 for _, p in confs if p),
    }


def compute_venn_analysis():
    """Compute detailed Venn analysis from existing results."""
    ecs_path = ROOT / "evaluation_humaneval.json"
    qwen_path = ROOT / "evaluation_qwen_hf_baseline.json"
    if not ecs_path.exists() or not qwen_path.exists():
        return None

    ecs_data = json.loads(ecs_path.read_text())
    qwen_data = json.loads(qwen_path.read_text())

    ecs_results = {r["task_id"]: r for r in ecs_data["results"]}
    qwen_results = {r["task_id"]: r for r in qwen_data["results"]}

    all_tasks = sorted(set(ecs_results) | set(qwen_results))

    categories = {"both": [], "only_ecs": [], "only_qwen": [], "neither": []}
    for tid in all_tasks:
        ep = ecs_results.get(tid, {}).get("passed", False)
        qp = qwen_results.get(tid, {}).get("passed", False)
        if ep and qp:
            categories["both"].append(tid)
        elif ep:
            categories["only_ecs"].append(tid)
        elif qp:
            categories["only_qwen"].append(tid)
        else:
            categories["neither"].append(tid)

    # ECS-unique: what methods solved them?
    ecs_unique_details = []
    for tid in categories["only_ecs"]:
        r = ecs_results[tid]
        ecs_unique_details.append({
            "task_id": tid,
            "method": r.get("method", "unknown"),
            "confidence": r.get("confidence", 0),
        })

    # Theoretical union pass@1
    union = len(categories["both"]) + len(categories["only_ecs"]) + len(categories["only_qwen"])

    return {
        "counts": {k: len(v) for k, v in categories.items()},
        "task_ids": categories,
        "ecs_unique_details": ecs_unique_details,
        "ecs_total": len(categories["both"]) + len(categories["only_ecs"]),
        "qwen_total": len(categories["both"]) + len(categories["only_qwen"]),
        "union_total": union,
        "union_pass_rate": union / len(all_tasks),
        "amplification_delta": len(categories["only_ecs"]),
    }


def main():
    confidence_only = "--confidence-only" in sys.argv

    print("=" * 60)
    print("PHASE 2: CONFIDENCE ANALYSIS & MULTI-CONDITION COMPARISON")
    print("=" * 60)

    # Confidence analysis (runs from existing files, no GPU needed)
    print("\n--- Confidence Analysis ---")
    conf_results = compute_confidence_analysis()
    if conf_results:
        print(f"  Total: {conf_results['total_problems']}, Passed: {conf_results['total_passed']}")
        print(f"  ECE: {conf_results['ece']}")
        print(f"  Brier Score: {conf_results['brier_score']}")
        print(f"\n  Selective Prediction Curve:")
        print(f"  {'Threshold':<12} {'Coverage':<12} {'Precision':<12} {'Count':<8}")
        print(f"  {'-'*44}")
        for s in conf_results["selective_prediction"]:
            print(f"  {s['threshold']:<12.2f} {s['coverage']:<12.1%} {s['precision']:<12.1%} {s['count']:<8}")
        print(f"\n  Calibration Bins (ECE = {conf_results['ece']}):")
        print(f"  {'Bin':<12} {'Count':<8} {'Avg Conf':<10} {'Accuracy':<10} {'Gap':<8}")
        print(f"  {'-'*48}")
        for b in conf_results["calibration_bins"]:
            if b["count"] > 0:
                print(f"  {b['bin']:<12} {b['count']:<8} {b['avg_conf']:<10} {b['accuracy']:<10} {b['gap']:<8}")
        print(f"\n  Method Breakdown:")
        for m, s in sorted(conf_results["method_breakdown"].items(), key=lambda x: -x[1]["total"]):
            print(f"    {m:<25} {s['passed']}/{s['total']} ({s['pass_rate']:.0%}) avg_conf={s['avg_confidence']:.2f}")

    # Venn analysis
    print("\n--- Venn Analysis ---")
    venn = compute_venn_analysis()
    if venn:
        c = venn["counts"]
        print(f"  Both:      {c['both']}")
        print(f"  Only ECS:  {c['only_ecs']}")
        print(f"  Only Qwen: {c['only_qwen']}")
        print(f"  Neither:   {c['neither']}")
        print(f"  Union:     {venn['union_total']}/164 ({venn['union_pass_rate']:.1%})")
        print(f"  Amplification: +{venn['amplification_delta']} problems over Qwen alone")
        if venn["ecs_unique_details"]:
            print(f"\n  ECS-unique problems:")
            for d in venn["ecs_unique_details"]:
                print(f"    {d['task_id']}: method={d['method']}, conf={d['confidence']}")

    # Condition summary (from existing data)
    print("\n--- Condition Summary (from existing evaluations) ---")
    print(f"  (a) Qwen-0.5B-Instruct alone:  91/164 (55.5%)")
    print(f"  (b) Best-of-K oracle:           [requires GPU — run with K=16 samples]")
    print(f"  (c) Majority voting:            [requires GPU — run with K=16 samples]")
    print(f"  (d) ECS structured (no neural): 36/164 (22.0%)")
    print(f"  Union (oracle selection):        {venn['union_total']}/164 ({venn['union_pass_rate']:.1%})" if venn else "")

    # Save
    output = {
        "confidence_analysis": conf_results,
        "venn_analysis": venn,
        "conditions": {
            "qwen_alone": {"passed": 91, "total": 164, "pass_rate": 0.555},
            "ecs_structured": {"passed": 36, "total": 164, "pass_rate": 0.220},
            "union_oracle": {"passed": venn["union_total"], "total": 164, "pass_rate": venn["union_pass_rate"]} if venn else None,
        },
    }
    out_path = ROOT / "phase2_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
