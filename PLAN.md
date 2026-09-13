# ECS Research Plan: Verified, Stress-Tested, Evidence-Backed

## 1. Objective & Success Criteria

**Objective:** Transform ECS from a template-based synthesis system into a publishable contribution on **multi-strategy orchestration with calibrated confidence for test-time program synthesis**.

**Success criteria (tiered):**
- **Gold (NeurIPS main):** ECS + Qwen-0.5B-Instruct achieves ≥8 point HumanEval improvement over 0.5B alone, with calibrated selective prediction (≥95% precision at ≥30% coverage).
- **Silver (AAAI/EMNLP main):** ECS provides measurably better calibration/selective prediction than uncalibrated baselines, even if pass@1 delta is small (3-5 points).
- **Bronze (NeurIPS Workshop):** Novel analysis of multi-strategy synthesis showing complementary methods + confidence calibration, even with modest empirical gains.

---

## 2. Locked Scope & Constraints

**IN scope:**
- Integrating Qwen2.5-Coder-0.5B-Instruct (frozen, no fine-tuning) as a generation source
- Multi-strategy candidate generation (neural + constraint + compositional + execution-guided)
- Cognitive architecture for candidate selection and confidence calibration
- Evaluation on HumanEval/HumanEval+ (164 problems)
- Within-session learning via HDC memory accumulation

**OUT of scope:**
- Training or fine-tuning any model (user's hard constraint)
- MCTS-based search (S* paper explicitly rejects MCTS for code; PRM for code is unsolved)
- MBPP evaluation (no existing harness, insufficient time to build + validate)
- Comparison against GPT-4/GPT-3.5/Claude (different compute regime, not a fair comparison)

**Constraints:**
- Compute: Kaggle T4 GPU (~30h/week), laptop CPU for prototyping
- Timeline: 10-12 weeks for paper-ready results (see Phase timeline)
- Zero training: Frozen pretrained model only, all adaptation via architecture

---

## 3. Research Findings (Evidence-Backed)

### The Other AI's Proposal: Critical Factual Errors

| Claim | Reality | Source |
|-------|---------|--------|
| "ReST-MCTS* evaluates on code with 559 citations" | Does NOT evaluate on code (math only). Citation count UNVERIFIED. | arXiv:2406.03816 Limitation section |
| "S* uses MCTS for code" | S* explicitly rejects MCTS. Uses parallel sampling + debugging + pairwise selection. | arXiv:2502.14382 Section 2 |
| "Qwen-0.5B is ~15-20% on HumanEval" | **28.0%** (base), **61.6%** (instruct) | arXiv:2409.12186v3 Tables 5 & 16 |
| "Expected 40-50% with architecture" | FABRICATED. Zero evidence. S* takes 0.5B from 1.2% → 10.9% on LiveCodeBench (harder benchmark). | arXiv:2502.14382 Table 1 |
| "Nobody has combined VSA with program synthesis" | Doug paper (arXiv:2510.16533, ICCM 2025) directly does this. | arXiv:2510.16533 |
| "Nobody has published GWT at ML venues" | Bengio et al. published GWT at ICLR 2022. | arXiv:2103.01197 |

### What the Research Actually Shows

**The real opportunity:**
1. **The selection bottleneck IS the key unsolved problem.** (arXiv:2608.18931: reward models correlate ρ≈0.12 with true quality on open-ended tasks)
2. **No published system uses calibrated confidence to gate synthesis strategy selection.** (Gap verified: 0 results across arXiv, Semantic Scholar)
3. **HDC for code pattern encoding is novel at ML venues.** (Doug paper exists at cognitive science venue only)
4. **ECS's multi-strategy synthesis produces complementary results.** (RESULTS.md: 0 overlap between constraint and compositional paths; compositional achieves 100% precision)
5. **Every "training-free" NeurIPS paper uses a frozen pretrained model.** Zero-neural-parameter systems don't exist at top venues.

### Real Baselines

| Model | HumanEval pass@1 | Source |
|-------|-------------------|--------|
| Qwen2.5-Coder-0.5B base | 28.0% | arXiv:2409.12186v3 |
| Qwen2.5-Coder-0.5B-Instruct | 61.6% | arXiv:2409.12186v3 |
| ECS structured (no neural) | 22.0% | evaluation_humaneval.json |
| ECS internal success | 68.9% | evaluation_humaneval.json |
| CodeT + code-davinci-002 | 65.8% | arXiv:2207.10397 |
| Qwen2.5-Coder-3B-Instruct | 84.1% | arXiv:2409.12186v3 |

---

## 4. Chosen Approach & Why

### Approach: Multi-Strategy Orchestration with Calibrated Selective Prediction

**NOT "selection oracle" (plan-reviewer caught this mismatch).** The architecture doesn't rank N copies of the same thing — it orchestrates QUALITATIVELY DIFFERENT synthesis methods and provides calibrated confidence.

**The primary paper contribution is a trifecta:**

1. **Multi-strategy fusion:** Show that constraint templates, compositional reasoning, and neural generation are COMPLEMENTARY (verified: zero overlap, different problem types). The UNION of methods solves problems no single method can.

2. **Calibrated confidence without training:** Show that the architecture's confidence score is monotonically predictive of correctness (NOT "100% at >0.6" — the plan-reviewer caught this; it's 100% on custom benchmark but 35% on HumanEval). Reframe as selective prediction: at threshold T, what precision does the system achieve? Show the precision-coverage tradeoff is better than random.

3. **Within-session learning:** Show measurable pass@1 improvement over a sequence of problems through HDC memory accumulation, without weight updates.

### Why not alternatives?

| Rejected Approach | Reason |
|---|---|
| MCTS-based search | S* explicitly rejects MCTS for code. PRM for code is unsolved (ρ≈0.12). No viable PRM to learn or replace. |
| HDC as process reward model | Plan-reviewer identified: no code path exists for HDC to RANK candidates. HDC is for retrieval, not ranking. Would require major new implementation with uncertain payoff. |
| Pure "selection oracle" framing | Architecture orchestrates different methods, not ranks same-method candidates. Framing mismatch. |
| Competing on raw pass@1 against Qwen-0.5B-Instruct (61.6%) | ECS's 36 solved problems are likely a SUBSET of what 0.5B-Instruct solves. Amplification may be <5 points. |

---

## 5. Implementation Plan

### PHASE 0 — "Kill/Pivot" Gate (Week 1)
**Purpose:** Answer the ONE question that determines whether the amplification story works.

**Tasks:**
1. **[BUG FIX]** Patch `scripts/full_comparison.py` to pass `prompt` and `entry_point` to `solve_problem()` (currently missing — disables compositional synthesis, losing 41.6% of HumanEval passes).
   - File: `scripts/full_comparison.py`, around line 192
   - Fix: Match the calling convention in `scripts/humaneval_eval.py` lines 130-131

2. **Run Qwen-0.5B-Instruct baseline on HumanEval.**
   - Use HuggingFace Transformers with `torch_dtype=torch.float16` (NOT Ollama, to avoid quantization loss)
   - Run on Kaggle T4
   - Record per-problem pass/fail

3. **Compute the Venn diagram:** How many problems does ECS-structured solve that 0.5B-Instruct misses? If <5, the "amplification" story is dead → pivot to selective prediction only.

4. **Classify the 77 HumanEval failures** (68.9% internal success, 22% actual pass): categorize as (a) function naming, (b) type mismatch, (c) logic error, (d) edge case error. Quantify fixable vs unfixable.

**Kill criteria:** If ECS-structured solves <3 problems that 0.5B-Instruct misses, AND the failure classification shows <10 are signature-fixable → pivot to Paper B (calibration-only paper, bronze tier).

**Verification:** Venn diagram plotted, failure classification table produced.

### PHASE 1 — Infrastructure (Week 2-3)
**Purpose:** Build the missing engineering pieces.

**Tasks:**
1. **Add HuggingFace Transformers backend** to `ecs/neural/interface.py` as an alternative to Ollama. Must support:
   - FP16 on T4
   - Temperature parameter for varied generation
   - Multiple samples per call (N generations)
   - Qwen chat template formatting

2. **Build candidate pool infrastructure:**
   - New module: `ecs/synthesis/candidate_pool.py`
   - Collects candidates from ALL synthesis methods
   - For neural: generate K samples at different temperatures
   - For deterministic methods (constraint, compositional): generate 1 candidate each
   - Total candidates per problem: K + 1-4 (neural + structured methods)

3. **Add dependencies:** `scikit-learn`, `matplotlib` to `pyproject.toml`

4. **Fix signature adaptation** based on Phase 0 failure analysis (if applicable).

**Verification:** Can generate 10+ candidates per problem, collect them in a pool, run all on HumanEval.

### PHASE 2 — Core Experiment: Multi-Strategy Fusion (Week 4-5)
**Purpose:** Run the primary experiment.

**Tasks:**
1. **Generate candidate pools** for all 164 HumanEval problems:
   - Qwen-0.5B-Instruct: K=16 samples at temperatures [0.2, 0.4, 0.6, 0.8]
   - Constraint templates: 0-1 candidate per problem
   - Compositional synthesis: 0-1 candidate per problem
   - Execution-guided: 0-1 candidate per problem
   - Induction: 0-1 candidate per problem

2. **Selection comparison (4 conditions):**
   - (a) Qwen-0.5B alone (pass@1, single temperature=0)
   - (b) Qwen-0.5B best-of-K (K=16, oracle selection = upper bound)
   - (c) Qwen-0.5B + majority voting (self-consistency among K=16)
   - (d) ECS full architecture (workspace selects from mixed candidate pool)

3. **Confidence analysis:**
   - For condition (d), record ECS confidence for each problem
   - Compute selective prediction curve: at threshold T, what fraction solved? What precision?
   - Compare: ECS confidence vs random baseline vs neural log-probability

4. **Run 3 times** with different random seeds. Report mean ± std.

**Verification:** 4-condition comparison table with error bars. Selective prediction curve plotted.

### PHASE 3 — Session Learning (Week 6)
**Purpose:** Demonstrate within-session improvement through memory.

**Tasks:**
1. **Sequential HumanEval run (2 conditions):**
   - (a) ECS with memory (problems solved early inform later solutions)
   - (b) ECS without memory (fresh orchestrator per problem)
   - Both with Qwen-0.5B-Instruct as neural backend

2. **Measure:**
   - Pass@1 for first 82 problems vs last 82 problems
   - Number of memory retrievals over time
   - Does accumulated experience improve selection quality?

3. **Run 3 times** with shuffled problem order.

**Verification:** Learning curve plot. Statistical test for first-half vs second-half difference.

### PHASE 4 — Ablation (Week 7)
**Purpose:** Show each architecture component contributes on HumanEval.

**Tasks:**
1. **Ablate around the SELECTION task** (not the synthesis task — plan-reviewer's fix):
   - Full ECS (baseline)
   - No HDC memory (random selection from candidate pool)
   - No compositional synthesis (neural + constraint only)
   - No confidence gating (accept highest-scoring candidate regardless)
   - No workspace auction (round-robin selection)

2. **Measure:** pass@1, precision-at-K, calibration error for each condition.

**Verification:** Ablation table where ≥2 components show >0% contribution.

### PHASE 5 — Paper Writing (Week 8-10)
**Purpose:** Write the paper.

**Structure:**
1. Introduction: The selection bottleneck in test-time code synthesis
2. Related Work: Test-time scaling (S*, CodeT), cognitive architectures (GWT at ICLR 2022), program synthesis (DreamCoder, Reflexion)
3. Architecture: Multi-strategy generation + workspace selection + HDC memory + confidence calibration
4. Experiments: Multi-strategy fusion, selective prediction, session learning, ablation
5. Analysis: Which problems does multi-strategy solve that single-strategy misses? Why?
6. Discussion: What cognitive architecture provides that pure sampling doesn't

**Target venues (with deadlines):**
- ICLR 2027: ~Oct 2026 (VERY tight — only if Phase 0-2 results are strong by end of September)
- AAAI 2027: ~Aug-Sep 2026 (may have already passed)
- ICML 2027: ~Jan 2027 (comfortable timeline)
- NeurIPS 2027: ~May 2027 (safest, allows iteration)

### BUFFER (Week 11-12)
For debugging, re-running experiments, addressing reviewer-style objections.

---

## 6. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ECS solves a pure subset of what 0.5B-Instruct solves (zero additive value) | Medium-High | Fatal for "amplification" story | Phase 0 kill gate. Pivot to calibration paper. |
| HDC similarity is too coarse for meaningful selection | Medium | Weakens selection experiment | HDC is only ONE signal. Combine with execution feedback + confidence. |
| Qwen-0.5B-Instruct on Ollama loses 3-5 points to quantization | High | Unfair baseline comparison | Use HuggingFace Transformers FP16, not Ollama. |
| 30h/week Kaggle quota insufficient for all experiments | Medium | Delays timeline | Prototype on CPU, run final experiments on GPU. Batch efficiently with vLLM. |
| Confidence calibration doesn't hold on HumanEval | Low | Weakens paper contribution | Already partially demonstrated (high-conf has 35% pass vs low-conf 0%). Threshold recalibration. |
| Ablation still shows 0% delta for GWT/memory/attention | Medium-High | Weakens architecture story | Ablate around selection (not synthesis). Show memory helps in session learning. |

---

## 7. Verification Checklist

Run these top-to-bottom:

- [ ] `full_comparison.py` passes `prompt` and `entry_point` to `solve_problem()`
- [ ] `uv run pytest` still passes all 268 tests after any changes
- [ ] Qwen-0.5B-Instruct baseline runs on Kaggle T4 with HuggingFace Transformers (FP16)
- [ ] Venn diagram computed: ECS-structured vs Qwen-0.5B-Instruct per-problem overlap
- [ ] 77 HumanEval failures classified by failure mode
- [ ] Candidate pool generates 10+ candidates per problem
- [ ] 4-condition comparison (alone, best-of-K, majority, ECS) produces results with error bars
- [ ] Selective prediction curve plotted (precision vs coverage at varying thresholds)
- [ ] Session learning experiment shows measurable first-half vs second-half difference (p<0.05)
- [ ] Ablation table shows ≥2 components with >0% contribution

---

## 8. Open Questions (Decisions for You)

1. **Timeline pressure:** ICLR 2027 deadline is ~October 2026 — almost certainly too tight. ICML 2027 (January 2027) is reachable with 10-12 weeks of focused work. NeurIPS 2027 (May 2027) is comfortable. **Which do you want to target?**

2. **If Phase 0 kills the "amplification" story:** Do you want to pivot to (a) calibration-only paper (smaller contribution, easier to execute), or (b) redesign the architecture for a stronger experiment (more work, higher ceiling)?

3. **Compute access:** Can you get your college GPU allocation confirmed before starting Phase 2? This is the most compute-intensive phase.

4. **Scope of "no training":** Do you accept using a frozen pretrained model (Qwen-0.5B-Instruct) as a generation component? This is standard in the literature ("training-free" papers at NeurIPS all use frozen pretrained models), but it means the system is not truly "zero neural parameters" anymore. **Is this acceptable?**

---

## What the Other AI Got Right vs Wrong — Summary

| Area | Assessment |
|------|-----------|
| "Test-time compute scaling is the hot topic" | **RIGHT** — verified as the dominant 2024-2026 research direction |
| "Use MCTS guided by HDC" | **WRONG** — S* explicitly rejects MCTS for code. No viable PRM. |
| "ReST-MCTS* is the comparison target for code" | **WRONG** — paper doesn't evaluate on code at all |
| "Expected 40-50% with architecture" | **FABRICATED** — zero evidence, not even a reasonable estimate |
| "Nobody has done HDC + program synthesis" | **PARTIALLY WRONG** — Doug paper (ICCM 2025) exists |
| "Run the neural comparison first" | **RIGHT** — this is Phase 0 of our plan |
| "Reframe 'emergent' as 'adaptive'" | **RIGHT** — essential for surviving review |
| "Qwen-0.5B is ~15-20%" | **WRONG** — actual score is 28.0% (base), 61.6% (instruct) |
| "The code snippets provided" | **BUGGY** — missing closing paren, arbitrary weights, naive quality function |
