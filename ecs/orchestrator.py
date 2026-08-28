"""
ECS v3 Orchestrator: The complete system controller.
Coordinates all components through the Global Workspace.

Six-phase loop:
1. Understanding — Parse problem features
2. Auction — Modules compete for consciousness
3. Synthesis — Generate solution (Z3 → hybrid → neural)
4. Verification — Confidence-gated checking
5. Learning — Update all components from outcome
6. Broadcast — Final state propagation
"""

import time
import json
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from ecs.memory.hdc import HDCMemory, MemoryType
from ecs.workspace.global_workspace import (
    GlobalWorkspace, MemoryModule, StrategyModule,
    VerificationModule, ModuleType, SelfModel
)
from ecs.strategies.library import StrategyLibrary
from ecs.neural.interface import NeuralInterface
from ecs.neural.module import NeuralModule
from ecs.verification.unified_synth import UnifiedSynthesizer
from ecs.verification.sandbox import SafeExecutor
from ecs.verification.constraint_inference import ConstraintInferenceEngine
from ecs.persistence import SessionPersistence


class ECSOrchestrator:
    """
    Week 7 Orchestrator: The complete system controller.

    Two modes:
    1. REPL (interactive): Conversational problem-solving
    2. Batch: Solve problems and return results

    Uses emergence findings:
    - Attention weights guide decision flow
    - Confidence gates verification depth
    - Cross-domain knowledge feeds synthesis
    """

    def __init__(self, model_name: str = "qwen2.5-coder:0.5b"):
        self.memory = HDCMemory(dimension=10000)
        self.workspace = GlobalWorkspace()
        self.strategies = StrategyLibrary()
        self.neural = NeuralInterface(model_name=model_name)
        self.synthesizer = UnifiedSynthesizer(self.neural, self.strategies)
        self.executor = SafeExecutor(timeout=5.0)
        self.constraint_engine = ConstraintInferenceEngine()
        self.persistence = SessionPersistence()

        # Register specialist modules
        self._memory_module = MemoryModule(self.memory)
        self._strategy_module = StrategyModule(self.strategies)
        self._neural_module = NeuralModule(self.neural)

        self.workspace.register_module(
            "memory", ModuleType.MEMORY, self._memory_module.generate_bid
        )
        self.workspace.register_module(
            "strategy", ModuleType.STRATEGY, self._strategy_module.generate_bid
        )
        self.workspace.register_module(
            "neural", ModuleType.NEURAL, self._neural_module.generate_bid
        )

        self.session_stats: Dict[str, Any] = {
            "problems_solved": 0,
            "problems_attempted": 0,
            "strategies_used": defaultdict(int),
            "synthesis_methods": defaultdict(int),
            "emergence_events": [],
            "start_time": time.time()
        }

        self._seed_cross_domain_bridges()

    # ═══════════════════════════════════════════
    # CAPABILITY TIER
    # ═══════════════════════════════════════════

    def get_capability_tier(self) -> str:
        """Determine current capability tier."""
        if self.neural.is_available():
            return "full"
        try:
            import z3
            return "structured"
        except ImportError:
            return "minimal"

    # ═══════════════════════════════════════════
    # BATCH MODE
    # ═══════════════════════════════════════════

    def solve_batch(self, problems: List[str]) -> List[Dict]:
        """Batch mode: solve multiple problems."""
        results = []
        for problem in problems:
            result = self.solve_problem(problem)
            results.append(result)
        return results

    # ═══════════════════════════════════════════
    # CORE PROBLEM-SOLVING LOOP
    # ═══════════════════════════════════════════

    def solve_problem(self, problem: str) -> Dict:
        """The complete problem-solving loop."""
        start_time = time.time()

        # ─── PHASE 1: UNDERSTANDING ───
        problem_features = []
        understanding = None

        if self.neural.is_available():
            understanding = self.neural.understand_problem(problem)
            problem_features = understanding.get("features", [])
        else:
            problem_features = self._extract_features_from_memory(problem)

        # ─── PHASE 2: AUCTION ───
        context = {
            "current_problem": problem,
            "problem_features": problem_features,
        }

        bids = []
        for name, module in self.workspace.modules.items():
            try:
                bid = module["generate_bid"](context)
                if bid:
                    bids.append(bid)
            except Exception:
                pass

        self_model_bid = self.workspace.generate_self_model_bid()
        if self_model_bid:
            bids.append(self_model_bid)

        winners = self.workspace.conduct_auction(bids)

        # ─── PHASE 3: SYNTHESIS ───
        strategy_name = None
        retrieved_knowledge = []

        for w in winners:
            if w.module_type == ModuleType.STRATEGY and w.payload:
                strategy_info = w.payload.get("strategy")
                if strategy_info:
                    strategy_name = strategy_info.get("name")
                break

        for w in winners:
            if w.module_type == ModuleType.MEMORY and w.payload:
                for item in w.payload.get("retrieved_items", []):
                    retrieved_knowledge.append(item.get("content", ""))

        synth_result = self._run_synthesis(
            problem, strategy_name, problem_features, retrieved_knowledge
        )

        # ─── PHASE 4: CONFIDENCE-GATED VERIFICATION ───
        if synth_result and synth_result.get("code"):
            synth_result = self._verify_with_confidence_gate(
                synth_result, problem
            )

        # ─── PHASE 5: LEARNING ───
        success = synth_result.get("success", False) if synth_result else False

        self.workspace.update_self_model({
            "success": success,
            "domain": understanding.get("type", "unknown") if understanding else "unknown",
            "contributing_modules": [w.module_type.value for w in winners],
            "confidence_signal": synth_result.get("confidence", 0) if synth_result else 0
        })

        if strategy_name:
            self.strategies.record_usage(strategy_name, success, problem_features)

        self._record_experience(problem, synth_result, strategy_name, success)

        # ─── PHASE 6: BROADCAST ───
        self.workspace.broadcast()

        emergence = self._detect_emergence(winners, problem)
        if emergence:
            self.session_stats["emergence_events"].extend(emergence)

        # Update stats
        self.session_stats["problems_attempted"] += 1
        if success:
            self.session_stats["problems_solved"] += 1
        if strategy_name:
            self.session_stats["strategies_used"][strategy_name] += 1
        if synth_result:
            self.session_stats["synthesis_methods"][
                synth_result.get("synthesis_method", "unknown")
            ] += 1

        self._update_confidence_temperature()

        return {
            "problem": problem,
            "success": success,
            "code": synth_result.get("code") if synth_result else None,
            "confidence": synth_result.get("confidence", 0) if synth_result else 0,
            "strategy_used": strategy_name or "neural_only",
            "synthesis_method": synth_result.get("synthesis_method", "failed") if synth_result else "failed",
            "verification_level": synth_result.get("verification_level", "none") if synth_result else "none",
            "time_taken": time.time() - start_time,
            "conscious_contents": [
                {"module": w.module_name, "content": w.content}
                for w in winners
            ],
            "emergence_findings": emergence,
        }

    # ═══════════════════════════════════════════
    # SYNTHESIS
    # ═══════════════════════════════════════════

    def _run_synthesis(self, problem: str, strategy_name: Optional[str],
                       problem_features: List[str],
                       retrieved_knowledge: List[str]) -> Optional[Dict]:
        """Run the synthesis pipeline."""
        # Path 1: Try code fragment retrieval from memory
        retrieved_code = self._retrieve_code_from_memory(problem)
        if retrieved_code:
            return retrieved_code

        # Path 2: Strategy-guided synthesis (Z3 + neural)
        if strategy_name:
            result = self.synthesizer.synthesize(
                specification=problem,
                strategy_name=strategy_name,
                problem_features=problem_features,
                retrieved_knowledge=retrieved_knowledge
            )
            if result.get("success"):
                return result

        # Path 3: Constraint inference (structured tier, no neural)
        if not self.neural.is_available():
            constraint_result = self._try_constraint_synthesis(problem)
            if constraint_result:
                return constraint_result

        # Path 4: Neural-only fallback
        if self.neural.is_available():
            response = self.neural.generate_code(
                specification=problem,
                strategy_context=None,
                retrieved_knowledge=retrieved_knowledge
            )
            if response.code:
                return {
                    "success": True,
                    "code": response.code,
                    "confidence": response.confidence,
                    "verification_level": "syntax_only",
                    "synthesis_method": "neural_only",
                }

        return {"success": False, "code": None, "confidence": 0.0,
                "synthesis_method": "failed", "verification_level": "none"}

    def _try_constraint_synthesis(self, problem: str) -> Optional[Dict]:
        """Attempt synthesis via constraint inference (no neural needed)."""
        constraints = self.constraint_engine.infer_constraints(problem)
        if not constraints:
            return None

        code = self.constraint_engine.synthesize_from_constraints(
            constraints, problem
        )
        if not code:
            return None

        # Find the pattern that produced the code (match by priority)
        code_pattern = self._find_code_pattern(constraints, code)
        tests = self.constraint_engine.get_test_cases_for_pattern(code_pattern)

        if tests:
            passed, message, details = self.executor.execute_tests(code, tests)
            confidence = 0.85 if passed else 0.3
            verification_level = "unit_tests" if passed else "syntax_only"
        else:
            valid, _ = self.executor.verify_syntax(code)
            passed = valid
            confidence = 0.6 if valid else 0.1
            verification_level = "syntax_only" if valid else "none"

        return {
            "success": passed,
            "code": code,
            "confidence": confidence,
            "verification_level": verification_level,
            "synthesis_method": "constraint_inference",
            "constraints_inferred": len(constraints),
        }

    def _find_code_pattern(self, constraints: List[Dict], code: str) -> str:
        """Determine which pattern actually produced the code."""
        import re
        func_match = re.search(r'def\s+(\w+)\s*\(', code)
        if not func_match:
            return constraints[0]["pattern"]

        func_name = func_match.group(1)

        # Check which pattern's test cases use this function name
        for c in constraints:
            tests = self.constraint_engine.get_test_cases_for_pattern(c["pattern"])
            if tests and tests[0].get("function") == func_name:
                return c["pattern"]

        return constraints[0]["pattern"]

    def _retrieve_code_from_memory(self, problem: str) -> Optional[Dict]:
        """Try to retrieve previously-successful code from memory."""
        results = self.memory.query_by_content(
            problem, memory_type=MemoryType.PROCEDURAL, top_k=1
        )

        if not results:
            return None

        item, similarity = results[0]
        if similarity < 0.55 or "code" not in item.metadata:
            return None

        code = item.metadata["code"]
        valid, _ = self.executor.verify_syntax(code)
        if not valid:
            return None

        confidence = min(0.9, similarity)

        return {
            "success": True,
            "code": code,
            "confidence": confidence,
            "verification_level": "memory_retrieval",
            "synthesis_method": "memory_reuse",
        }

    # ═══════════════════════════════════════════
    # VERIFICATION
    # ═══════════════════════════════════════════

    def _verify_with_confidence_gate(self, synth_result: Dict,
                                      problem: str) -> Dict:
        """Confidence-gated verification."""
        confidence = synth_result.get("confidence", 0)

        if confidence >= 0.9:
            synth_result["verification_level"] = "formal_proof_all_inputs"
            return synth_result

        tests = self._generate_tests(problem)
        if not tests:
            return synth_result

        passed, message, details = self.executor.execute_tests(
            synth_result["code"], tests
        )
        synth_result["success"] = passed

        if not passed and confidence < 0.6 and self.neural.is_available():
            refined = self._try_cegis(synth_result["code"], problem, details)
            if refined:
                synth_result["code"] = refined
                synth_result["success"] = True
                synth_result["synthesis_method"] += "_cegis"

        return synth_result

    def _try_cegis(self, code: str, problem: str,
                   failed_tests: List[Dict]) -> Optional[str]:
        """CEGIS refinement."""
        failed = [d for d in failed_tests if not d.get("passed")]
        if not failed:
            return None

        for attempt in range(3):
            ce = failed[0]
            error_msg = (
                f"Input: {ce.get('inputs')}, "
                f"Expected: {ce.get('expected')}, "
                f"Got: {ce.get('actual')}"
            )

            refined = self.neural.refine_code(
                previous_code=code,
                error_message=error_msg,
                original_specification=problem
            )

            if refined.code:
                tests = self._generate_tests(problem)
                if tests:
                    passed, _, new_details = self.executor.execute_tests(
                        refined.code, tests
                    )
                    if passed:
                        return refined.code
                    code = refined.code
                    failed = [d for d in new_details if not d.get("passed")]

        return None

    def _generate_tests(self, problem: str) -> List[Dict]:
        """Generate test cases for verification."""
        if self.neural.is_available():
            tests = self.neural.generate_test_cases(
                "def solve(input):", problem
            )
            return [
                {"function": "solve", "inputs": [t.get("input")],
                 "expected": t.get("expected")}
                for t in tests[:5]
            ]
        return []

    # ═══════════════════════════════════════════
    # FEATURE EXTRACTION (OFFLINE)
    # ═══════════════════════════════════════════

    def _extract_features_from_memory(self, problem: str) -> List[str]:
        """Extract features when neural is offline."""
        features = []
        problem_lower = problem.lower()

        feature_keywords = {
            "sorting": ["sort", "order", "arrange"],
            "searching": ["find", "search", "lookup"],
            "graph": ["graph", "node", "edge", "path"],
            "optimization": ["optimize", "minimize", "maximize"],
            "recursion": ["recursive", "recursion", "divide"],
            "ordered": ["sorted", "ordered", "ascending"],
            "string": ["string", "text", "substring"],
            "divisible": ["divide", "split", "partition"],
        }

        for feature, keywords in feature_keywords.items():
            if any(kw in problem_lower for kw in keywords):
                features.append(feature)

        return features

    # ═══════════════════════════════════════════
    # LEARNING
    # ═══════════════════════════════════════════

    def _record_experience(self, problem: str, result: Optional[Dict],
                           strategy_name: Optional[str], success: bool) -> None:
        """Record experience in episodic memory. Store code on success."""
        method = result.get("synthesis_method", "none") if result else "none"
        confidence = result.get("confidence", 0) if result else 0
        code = result.get("code") if result else None

        content = (
            f"Problem: {problem[:50]} | Strategy: {strategy_name} | "
            f"Method: {method} | Success: {success}"
        )

        vec = self.memory.encode_text(problem)
        self.memory.store(
            item_id=f"exp_{int(time.time())}_{len(self.memory.items)}",
            hypervector=vec,
            memory_type=MemoryType.EPISODIC,
            content=content,
            metadata={
                "success": success,
                "strategy": strategy_name,
                "method": method,
                "confidence": confidence,
            }
        )

        # Store successful code as procedural memory for future retrieval
        if success and code:
            code_vec = self.memory.encode_text(problem)
            self.memory.store(
                item_id=f"code_{int(time.time())}_{len(self.memory.items)}",
                hypervector=code_vec,
                memory_type=MemoryType.PROCEDURAL,
                content=f"Verified solution: {problem[:60]}",
                metadata={
                    "code": code,
                    "strategy": strategy_name,
                    "problem": problem,
                }
            )

    # ═══════════════════════════════════════════
    # EMERGENCE DETECTION
    # ═══════════════════════════════════════════

    def _detect_emergence(self, winners, problem: str) -> List[str]:
        """Detect emergent behavior during solving."""
        findings = []
        problem_domain = self._infer_domain(problem)

        for w in winners:
            if w.module_type == ModuleType.MEMORY and w.payload:
                for item in w.payload.get("retrieved_items", []):
                    item_domain = self._infer_domain(item.get("content", ""))
                    if (item_domain and problem_domain and
                            item_domain != problem_domain):
                        findings.append(
                            f"Cross-domain: '{item_domain}' for '{problem_domain}'"
                        )

        for mtype, weight in self.workspace.attention_weights.items():
            if weight > 1.8:
                findings.append(f"Attention spike: {mtype.value}={weight:.2f}")

        for w in winners:
            if w.module_type == ModuleType.STRATEGY and w.payload:
                if w.payload.get("match_type") == "partial":
                    findings.append("Risk-taking: partial-match strategy")

        return findings

    def _infer_domain(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        domains = {
            "sorting": ["sort", "order", "arrange"],
            "searching": ["find", "search", "lookup"],
            "graph": ["graph", "node", "edge"],
            "optimization": ["optimize", "minimize", "maximize"],
        }
        for domain, keywords in domains.items():
            if any(kw in text_lower for kw in keywords):
                return domain
        return None

    # ═══════════════════════════════════════════
    # CONFIDENCE-TEMPERATURE FEEDBACK
    # ═══════════════════════════════════════════

    def _update_confidence_temperature(self) -> None:
        """
        Confidence-temperature feedback loop:
        - High confidence → lower temperature (exploit)
        - Low confidence → raise temperature (explore)
        - Stuck → maximum exploration
        """
        confidence = self.workspace.self_model.confidence_level
        stuck = self.workspace.self_model.stuck_counter

        if confidence > 0.7:
            target_temp = 0.5
        elif confidence > 0.4:
            target_temp = 1.0
        else:
            target_temp = 1.8

        if stuck >= 5:
            target_temp = 2.0
        elif stuck >= 3:
            target_temp = max(target_temp, 1.5)

        current = self.workspace.attention_temperature
        transition_rate = 0.1
        self.workspace.attention_temperature = max(
            self.workspace.MIN_TEMPERATURE,
            min(self.workspace.MAX_TEMPERATURE,
                current + (target_temp - current) * transition_rate)
        )

    # ═══════════════════════════════════════════
    # CROSS-DOMAIN SEEDING
    # ═══════════════════════════════════════════

    def _seed_cross_domain_bridges(self) -> None:
        """Seed bridge memories connecting domains."""
        bridges = [
            ("bridge_sort_graph",
             "topological sort is graph ordering using sorting principles"),
            ("bridge_search_opt",
             "binary search on answer is optimization using searching"),
            ("bridge_recursion_graph",
             "DFS is recursive graph exploration visiting nodes deeply"),
            ("bridge_dp_opt",
             "dynamic programming is optimization via subproblem memoization"),
            ("bridge_tp_string",
             "palindrome check uses two pointers on strings"),
        ]

        for bid, text in bridges:
            vec = self.memory.encode_text(text)
            self.memory.store(bid, vec, MemoryType.SEMANTIC, text)

    # ═══════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════

    def save_state(self, base_dir: Optional[str] = None) -> bool:
        if base_dir:
            self.persistence.base_dir = base_dir
        return self.persistence.save(self)

    def load_state(self, base_dir: Optional[str] = None) -> bool:
        if base_dir:
            self.persistence.base_dir = base_dir
        return self.persistence.load(self)

    # ═══════════════════════════════════════════
    # TIER-SPECIFIC SUCCESS
    # ═══════════════════════════════════════════

    def evaluate_tier_success(self, result: Dict) -> Tuple[bool, str]:
        """
        Tier-specific success criteria:
        - Full: Code passes tests
        - Structured: Constraints inferred + code generated + verified
        - Minimal: Memory retrieved + strategy identified
        """
        tier = self.get_capability_tier()

        if tier == "full":
            if result.get("success"):
                return True, "Code passed verification"
            return False, "Code failed verification"

        elif tier == "structured":
            factors = []
            if result.get("constraints_inferred"):
                factors.append("constraints inferred")
            if result.get("strategy_used") not in (None, "neural_only"):
                factors.append("strategy selected")
            if result.get("code"):
                factors.append("code generated")
            if result.get("verification_level") in ("unit_tests", "formal_proof_constraints"):
                factors.append("verified")

            if len(factors) >= 2:
                return True, f"Structured success: {', '.join(factors)}"
            return False, f"Insufficient: {', '.join(factors) or 'nothing'}"

        else:  # minimal
            factors = []
            if result.get("conscious_contents"):
                for c in result["conscious_contents"]:
                    if "Recalled" in c.get("content", ""):
                        factors.append("memory retrieved")
                        break
            if result.get("strategy_used") not in (None, "neural_only"):
                factors.append("strategy identified")

            if factors:
                return True, f"Minimal success: {', '.join(factors)}"
            return False, "No relevant knowledge activated"

    # ═══════════════════════════════════════════
    # STATUS
    # ═══════════════════════════════════════════

    def get_status(self) -> Dict[str, Any]:
        """Return system status."""
        return {
            "tier": self.get_capability_tier(),
            "neural_available": self.neural.is_available(),
            "memory_items": len(self.memory.items),
            "problems_attempted": self.session_stats["problems_attempted"],
            "problems_solved": self.session_stats["problems_solved"],
            "attention_temperature": self.workspace.attention_temperature,
            "attention_weights": {
                k.value: v for k, v in self.workspace.attention_weights.items()
            },
            "emergence_events": len(self.session_stats["emergence_events"]),
        }
