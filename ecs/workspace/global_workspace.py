"""
Global Workspace Implementation based on Global Workspace Theory (GWT).
Specialist modules compete for access through a free-energy auction.
Winning information is broadcast to all modules.
"""

import time
import numpy as np
from typing import Dict, List, Set, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class ModuleType(Enum):
    MEMORY = "memory"
    STRATEGY = "strategy"
    NEURAL = "neural"
    VERIFICATION = "verification"
    SELF_MODEL = "self_model"
    ABSTRACTION = "abstraction"


@dataclass
class WorkspaceBid:
    """A bid from a specialist module for workspace access."""
    module_type: ModuleType
    module_name: str
    content: str
    payload: Any
    free_energy_reduction: float
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.5
    urgency: float = 0.5
    coalition_members: List[str] = field(default_factory=list)
    coalition_boost: float = 1.0


@dataclass
class SelfModel:
    """Metacognitive self-model: the system's model of itself."""
    STRENGTH_DECAY = 0.99

    strengths: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    weaknesses: Dict[str, float] = field(default_factory=lambda: defaultdict(float))

    current_goal: Optional[str] = None
    active_strategy: Optional[str] = None
    confidence_level: float = 0.5
    uncertainty_level: float = 0.5
    free_energy_level: float = 1.0

    tasks_attempted: int = 0
    tasks_succeeded: int = 0
    stuck_counter: int = 0
    last_n_actions: List[str] = field(default_factory=list)

    patterns_noticed: List[str] = field(default_factory=list)
    learning_rate_estimate: float = 0.1

    def _decay_all(self):
        for d in list(self.strengths):
            self.strengths[d] *= self.STRENGTH_DECAY
        for d in list(self.weaknesses):
            self.weaknesses[d] *= self.STRENGTH_DECAY

    def record_success(self, domain: str):
        self.tasks_attempted += 1
        self.tasks_succeeded += 1
        self._decay_all()
        self.strengths[domain] = self.strengths.get(domain, 0) + 0.1
        self.stuck_counter = 0

    def record_failure(self, domain: str):
        self.tasks_attempted += 1
        self._decay_all()
        self.weaknesses[domain] = self.weaknesses.get(domain, 0) + 0.1
        self.stuck_counter += 1

    def update_confidence(self, new_evidence: float):
        alpha = 0.3
        self.confidence_level = (1 - alpha) * self.confidence_level + alpha * new_evidence

    def reflect(self) -> Optional[str]:
        if self.stuck_counter > 3:
            return (f"I'm stuck. I've failed {self.stuck_counter} times. "
                    f"Consider trying a different strategy.")

        if self.confidence_level > 0.8:
            return (f"I'm confident (P={self.confidence_level:.2f}). "
                    f"Strategy '{self.active_strategy}' is working.")

        if self.confidence_level < 0.2:
            return (f"Low confidence (P={self.confidence_level:.2f}). "
                    f"Uncertainty is high. Need more information.")

        return None


class GlobalWorkspace:
    """
    The Global Workspace: limited-capacity bottleneck where specialist
    modules compete for access. Implements free-energy auction.
    """

    CAPACITY = 7
    IGNITION_THRESHOLD = 0.1
    MAX_HISTORY = 1000
    ATTENTION_LEARNING_RATE = 0.05
    ATTENTION_BOUNDS = (0.5, 2.5)
    ATTENTION_MEAN_REVERSION = 0.001
    MIN_TEMPERATURE = 0.1
    MAX_TEMPERATURE = 2.0

    def __init__(self):
        self.capacity = self.CAPACITY
        self.conscious_contents: List[WorkspaceBid] = []
        self.self_model = SelfModel()

        self.modules: Dict[str, Dict[str, Any]] = {}
        self.attention_temperature: float = 1.0

        self.broadcast_history: List[Dict] = []
        self.broadcast_subscribers: List[Callable] = []

        self.attention_weights: Dict[ModuleType, float] = {
            ModuleType.MEMORY: 1.0,
            ModuleType.STRATEGY: 1.0,
            ModuleType.NEURAL: 1.0,
            ModuleType.VERIFICATION: 1.5,
            ModuleType.SELF_MODEL: 0.8,
            ModuleType.ABSTRACTION: 0.9,
        }
        self.default_weights: Dict[ModuleType, float] = self.attention_weights.copy()

        self._starved_modules: Set[str] = set()

        self.bid_outcome_log: List[Dict] = []

    def register_module(self, name: str, module_type: ModuleType,
                        bid_generator: Callable) -> None:
        """Register a specialist module that can bid for workspace access."""
        self.modules[name] = {
            "type": module_type,
            "generate_bid": bid_generator
        }

    def subscribe_to_broadcast(self, callback: Callable) -> None:
        """Register a module to receive global broadcasts."""
        self.broadcast_subscribers.append(callback)

    # ═══════════════════════════════════════════
    # AUCTION
    # ═══════════════════════════════════════════

    def conduct_auction(self, bids: List[WorkspaceBid]) -> List[WorkspaceBid]:
        """
        Free-energy auction with temperature-controlled behavior.
        High temp (exploration) → flattened weights, lower threshold, noise.
        Low temp (exploitation) → amplified weights, higher threshold, deterministic.
        """
        t = self.attention_temperature

        # Temperature-adjusted weights
        adjusted_weights = {}
        for mtype, weight in self.attention_weights.items():
            if t >= 1.0:
                adjusted = 1.0 + (weight - 1.0) / t
            else:
                adjusted = 1.0 + (weight - 1.0) * (1.0 / max(t, 0.1))
            adjusted_weights[mtype] = max(0.1, min(3.0, adjusted))

        # Temperature-adjusted ignition threshold
        ignition_threshold = self.IGNITION_THRESHOLD * (1.0 / max(t, 0.1))
        ignition_threshold = max(0.05, min(0.3, ignition_threshold))

        competing_bids = []
        for bid in bids:
            weight = adjusted_weights.get(bid.module_type, 1.0)

            # Urgency boost for previously-starved modules
            if bid.module_name in self._starved_modules:
                boost = 1.0 + (bid.urgency * 0.5)
            else:
                boost = 1.0

            raw_energy = bid.free_energy_reduction

            # Exploration noise at high temperature
            if t > 1.5:
                noise_scale = (t - 1.0) * 0.1
                noise = 1.0 + np.random.uniform(-noise_scale, noise_scale)
                raw_energy = raw_energy * noise

            effective_bid = raw_energy * weight * boost

            if effective_bid >= ignition_threshold:
                competing_bids.append((effective_bid, bid))

        if not competing_bids:
            self.conscious_contents = []
            self._update_stuck_counter(0.0)
            # Track starved (all modules starved this round)
            self._starved_modules = {bid.module_name for bid in bids}
            self._record_history([], 0.0)
            return []

        competing_bids.sort(key=lambda x: x[0], reverse=True)
        winners = [bid for _, bid in competing_bids[:self.capacity]]
        self.conscious_contents = winners

        total_reduction = sum(bid.free_energy_reduction for bid in winners)
        self._update_stuck_counter(total_reduction)

        # Track starved modules
        winner_names = {bid.module_name for bid in winners}
        self._starved_modules = {
            bid.module_name for bid in bids if bid.module_name not in winner_names
        }

        self._record_history(winners, total_reduction)
        return winners

    def _update_stuck_counter(self, total_reduction: float) -> None:
        if total_reduction < 0.1:
            self.self_model.stuck_counter += 1
        else:
            self.self_model.stuck_counter = max(0, self.self_model.stuck_counter - 1)

    def _record_history(self, winners: List[WorkspaceBid], total_reduction: float) -> None:
        self.broadcast_history.append({
            "timestamp": time.time(),
            "winners": [
                {
                    "module": bid.module_name,
                    "type": bid.module_type.value,
                    "content": bid.content,
                    "free_energy_reduction": bid.free_energy_reduction
                }
                for bid in winners
            ],
            "total_reduction": total_reduction
        })

        if len(self.broadcast_history) > self.MAX_HISTORY:
            self.broadcast_history = self.broadcast_history[-self.MAX_HISTORY:]

    # ═══════════════════════════════════════════
    # BROADCAST
    # ═══════════════════════════════════════════

    def broadcast(self) -> Dict[str, Any]:
        """Broadcast conscious contents to all subscribers."""
        insight = self.self_model.reflect()

        broadcast_content = {
            "conscious_contents": [
                {
                    "module": bid.module_name,
                    "type": bid.module_type.value,
                    "content": bid.content,
                    "payload": bid.payload,
                    "confidence": bid.confidence,
                    "free_energy_reduction": bid.free_energy_reduction
                }
                for bid in self.conscious_contents
            ],
            "self_state": {
                "goal": self.self_model.current_goal,
                "active_strategy": self.self_model.active_strategy,
                "confidence": self.self_model.confidence_level,
                "uncertainty": self.self_model.uncertainty_level,
                "stuck": self.self_model.stuck_counter > 0,
                "metacognitive_insight": insight
            },
            "timestamp": time.time()
        }

        for subscriber in self.broadcast_subscribers:
            try:
                subscriber(broadcast_content)
            except Exception:
                pass

        return broadcast_content

    # ═══════════════════════════════════════════
    # ATTENTION ADAPTATION
    # ═══════════════════════════════════════════

    def update_attention_weights(self, outcomes: Dict[ModuleType, bool]) -> None:
        """Update attention weights based on module success/failure correlation."""
        for module_type, success in outcomes.items():
            if module_type not in self.attention_weights:
                continue

            current = self.attention_weights[module_type]
            if success:
                new_weight = current + self.ATTENTION_LEARNING_RATE
            else:
                new_weight = current - self.ATTENTION_LEARNING_RATE

            new_weight = max(self.ATTENTION_BOUNDS[0],
                             min(self.ATTENTION_BOUNDS[1], new_weight))
            self.attention_weights[module_type] = new_weight

    def apply_mean_reversion(self) -> None:
        """Pull attention weights back toward defaults."""
        for mtype in self.attention_weights:
            default = self.default_weights[mtype]
            current = self.attention_weights[mtype]
            self.attention_weights[mtype] = (
                current + (default - current) * self.ATTENTION_MEAN_REVERSION
            )

    def anneal_temperature(self, problems_solved: int) -> None:
        """Reduce temperature over time: explore early, exploit later."""
        target_temp = max(
            self.MIN_TEMPERATURE * 3,
            1.0 * (0.99 ** problems_solved)
        )
        self.attention_temperature = max(
            self.MIN_TEMPERATURE,
            min(self.MAX_TEMPERATURE, target_temp)
        )

    # ═══════════════════════════════════════════
    # SELF-MODEL BID GENERATION
    # ═══════════════════════════════════════════

    def generate_self_model_bid(self) -> Optional[WorkspaceBid]:
        """Self-model generates a bid when stuck — competes to be heard."""
        sm = self.self_model

        if sm.stuck_counter < 2:
            return None

        urgency = min(1.0, sm.stuck_counter / 5.0)

        if sm.stuck_counter >= 5:
            content = ("CRITICAL: Stuck 5+ times. "
                       "Recommend radical strategy change or fallback.")
            free_energy = 0.9
        elif sm.stuck_counter >= 3:
            content = f"Stuck {sm.stuck_counter} times. Recommend switching strategy."
            free_energy = 0.7
        else:
            content = "Making slow progress. Consider alternative approach."
            free_energy = 0.4

        return WorkspaceBid(
            module_type=ModuleType.SELF_MODEL,
            module_name="SelfModel",
            content=content,
            payload={
                "action": "switch_strategy" if sm.stuck_counter >= 3 else "note_progress",
                "stuck_count": sm.stuck_counter
            },
            free_energy_reduction=free_energy,
            confidence=0.9,
            urgency=urgency
        )

    def run_auction_cycle(self, module_bids: List[WorkspaceBid]) -> List[WorkspaceBid]:
        """Full auction cycle including self-model bid."""
        self_model_bid = self.generate_self_model_bid()
        if self_model_bid:
            all_bids = module_bids + [self_model_bid]
        else:
            all_bids = module_bids

        return self.conduct_auction(all_bids)

    # ═══════════════════════════════════════════
    # SELF-MODEL & OUTCOMES
    # ═══════════════════════════════════════════

    def update_self_model(self, outcome: Dict[str, Any]) -> None:
        """Update self-model and optionally attention weights."""
        success = outcome.get("success", False)
        domain = outcome.get("domain", "general")

        if success:
            self.self_model.record_success(domain)
        else:
            self.self_model.record_failure(domain)

        if "confidence_signal" in outcome:
            self.self_model.update_confidence(outcome["confidence_signal"])

        if "pattern" in outcome:
            self.self_model.patterns_noticed.append(outcome["pattern"])

        if "contributing_modules" in outcome:
            module_outcomes = {}
            for mtype_str in outcome["contributing_modules"]:
                try:
                    mtype = ModuleType(mtype_str)
                    module_outcomes[mtype] = success
                except ValueError:
                    continue
            self.update_attention_weights(module_outcomes)

    # ═══════════════════════════════════════════
    # BID OUTCOME TRACKING
    # ═══════════════════════════════════════════

    def record_bid_outcome(self, auction_winners: List[WorkspaceBid],
                           problem_success: bool) -> None:
        """Record which modules won auctions and whether the problem was solved."""
        self.bid_outcome_log.append({
            "timestamp": time.time(),
            "winner_types": [bid.module_type.value for bid in auction_winners],
            "winner_modules": [bid.module_name for bid in auction_winners],
            "success": problem_success
        })

        if len(self.bid_outcome_log) > 500:
            self.bid_outcome_log = self.bid_outcome_log[-500:]

    def get_module_reliability(self) -> Dict[str, float]:
        """Compute per-module reliability from bid outcomes."""
        module_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"wins": 0, "successes": 0}
        )

        for entry in self.bid_outcome_log:
            for module in entry["winner_modules"]:
                module_stats[module]["wins"] += 1
                if entry["success"]:
                    module_stats[module]["successes"] += 1

        return {
            module: stats["successes"] / stats["wins"]
            for module, stats in module_stats.items()
            if stats["wins"] >= 3
        }

    # ═══════════════════════════════════════════
    # COALITIONS (STUB FOR WEEK 6)
    # ═══════════════════════════════════════════

    def detect_coalitions(self, bids: List[WorkspaceBid]) -> List[WorkspaceBid]:
        """Detect modules bidding on related content. Week 6 feature."""
        return bids

    # ═══════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════

    def get_workspace_summary(self) -> str:
        """Human-readable summary of current conscious state."""
        lines = ["=== GLOBAL WORKSPACE STATE ===\n"]

        lines.append("Conscious Contents:")
        for i, bid in enumerate(self.conscious_contents, 1):
            lines.append(
                f"  {i}. [{bid.module_type.value}] {bid.module_name}: "
                f"{bid.content} (dF={bid.free_energy_reduction:.3f})"
            )

        lines.append("\nSelf-Model:")
        sm = self.self_model
        lines.append(f"  Current Goal: {sm.current_goal}")
        lines.append(f"  Active Strategy: {sm.active_strategy}")
        lines.append(f"  Confidence: {sm.confidence_level:.2f}")
        lines.append(f"  Uncertainty: {sm.uncertainty_level:.2f}")
        lines.append(f"  Stuck Counter: {sm.stuck_counter}")

        if sm.strengths:
            lines.append(f"  Strengths: {dict(sm.strengths)}")
        if sm.weaknesses:
            lines.append(f"  Weaknesses: {dict(sm.weaknesses)}")

        lines.append(f"\nAttention Weights:")
        for mtype, weight in self.attention_weights.items():
            lines.append(f"  {mtype.value}: {weight:.2f}")

        return "\n".join(lines)


# ═══════════════════════════════════════════
# SPECIALIST MODULE IMPLEMENTATIONS
# ═══════════════════════════════════════════

class MemoryModule:
    """Specialist module that retrieves from HDC memory."""

    def __init__(self, hdc_memory):
        self.memory = hdc_memory
        self.name = "MemoryModule"

    def generate_bid(self, context: Dict) -> WorkspaceBid:
        query = context.get("current_problem", "")
        if not query:
            return WorkspaceBid(
                module_type=ModuleType.MEMORY,
                module_name=self.name,
                content="No query to retrieve for",
                payload=None,
                free_energy_reduction=0.0
            )

        results = self.memory.batch_query(
            self.memory.encode_text(query), top_k=3
        )

        if not results:
            return WorkspaceBid(
                module_type=ModuleType.MEMORY,
                module_name=self.name,
                content="No relevant memories found",
                payload=None,
                free_energy_reduction=0.05
            )

        best_item, similarity = results[0]
        content = f"Recalled: {best_item.content} (relevance={similarity:.3f})"
        free_energy_reduction = similarity * min(best_item.access_count + 1, 10) * 0.1

        return WorkspaceBid(
            module_type=ModuleType.MEMORY,
            module_name=self.name,
            content=content,
            payload={
                "retrieved_items": [
                    {"id": item.id, "content": item.content,
                     "similarity": sim, "type": item.memory_type.value}
                    for item, sim in results
                ]
            },
            free_energy_reduction=free_energy_reduction,
            confidence=similarity
        )


class StrategyModule:
    """Specialist module that recommends strategies with hybrid matching."""

    def __init__(self, strategy_library):
        self.library = strategy_library
        self.name = "StrategyModule"

    def generate_bid(self, context: Dict) -> WorkspaceBid:
        problem_features = context.get("problem_features", [])

        # Strict matching first
        strict_matches = self.library.find_applicable(problem_features)

        if strict_matches:
            best = strict_matches[0]
            match_type = "exact"
            match_confidence = 1.0
        else:
            # Partial matching fallback
            partial_matches = self.library.find_partial_matches(problem_features)
            if partial_matches and partial_matches[0]["match_score"] >= 0.5:
                best = partial_matches[0]
                match_type = "partial"
                match_confidence = partial_matches[0]["match_score"]
            else:
                return WorkspaceBid(
                    module_type=ModuleType.STRATEGY,
                    module_name=self.name,
                    content="No applicable strategies found",
                    payload={"fallback": "neural_only"},
                    free_energy_reduction=0.1,
                    confidence=0.3
                )

        historical_rate = best.get("success_rate", 0.5)

        if match_type == "exact":
            confidence = 0.8 * match_confidence + 0.2 * historical_rate
            free_energy = 0.7 * confidence
        else:
            confidence = 0.5 * match_confidence + 0.2 * historical_rate
            free_energy = 0.4 * confidence

        tag = "[EXACT]" if match_type == "exact" else "[PARTIAL]"
        content = (f"{tag} Recommend: {best['name']} "
                   f"(match={match_confidence:.2f}, hist={historical_rate:.2f})")

        return WorkspaceBid(
            module_type=ModuleType.STRATEGY,
            module_name=self.name,
            content=content,
            payload={
                "strategy": best,
                "match_type": match_type,
                "match_confidence": match_confidence,
            },
            free_energy_reduction=free_energy,
            confidence=confidence
        )


class VerificationModule:
    """Specialist module that reports verification results."""

    def __init__(self):
        self.name = "VerificationModule"
        self.last_result: Optional[Dict] = None

    def generate_bid(self, context: Dict) -> WorkspaceBid:
        if not self.last_result:
            return WorkspaceBid(
                module_type=ModuleType.VERIFICATION,
                module_name=self.name,
                content="No verification performed yet",
                payload=None,
                free_energy_reduction=0.2
            )

        result = self.last_result

        if result["success"]:
            content = f"Verification PASSED: {result['message']}"
            free_energy = 0.9
        else:
            content = f"Verification FAILED: {result['message']}"
            free_energy = 0.8

        return WorkspaceBid(
            module_type=ModuleType.VERIFICATION,
            module_name=self.name,
            content=content,
            payload=result,
            free_energy_reduction=free_energy,
            confidence=result.get("confidence", 0.5)
        )
