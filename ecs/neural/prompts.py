"""Strategy-specific prompt templates for neural code generation."""

from typing import Dict, List, Optional


STRATEGY_PROMPTS: Dict[str, str] = {
    "greedy": (
        "When generating code for a greedy approach:\n"
        "- Make locally optimal choices at each step\n"
        "- Sort or order by greedy criterion first\n"
        "- The solution should be O(n log n) or better\n"
        "- Each choice should be irrevocable"
    ),

    "divide-and-conquer": (
        "When generating code for divide-and-conquer:\n"
        "- Clearly identify how to divide the problem\n"
        "- Define explicit base cases\n"
        "- Subproblems should be independent\n"
        "- Combine step must merge subproblem solutions\n"
        "- Think about the recurrence: T(n) = a*T(n/b) + f(n)"
    ),

    "dynamic-programming": (
        "When generating code for dynamic programming:\n"
        "- Define the DP state clearly (what does dp[i][j] represent?)\n"
        "- Write the recurrence relation explicitly\n"
        "- Identify base cases\n"
        "- Choose top-down (memoization) or bottom-up (tabulation)\n"
        "- Space optimization if possible (rolling arrays)"
    ),

    "binary-search": (
        "When generating code for binary search:\n"
        "- Input must be sorted or have monotonic property\n"
        "- Define the search space [lo, hi] carefully\n"
        "- Compute mid safely: mid = lo + (hi - lo) // 2\n"
        "- Handle termination: lo <= hi vs lo < hi\n"
        "- Consider what to return when not found"
    ),

    "two-pointer": (
        "When generating code for two-pointer:\n"
        "- Define what each pointer represents\n"
        "- Clarify the movement condition for each pointer\n"
        "- Think about termination (pointers meet or cross)\n"
        "- Often used on sorted arrays or linked lists"
    ),

    "backtracking": (
        "When generating code for backtracking:\n"
        "- Define state space and choices at each step\n"
        "- Implement: choose -> explore -> un-choose\n"
        "- Prune early if state is invalid\n"
        "- Collect all valid solutions"
    ),

    "sliding-window": (
        "When generating code for sliding window:\n"
        "- Define window validity condition\n"
        "- Expand right edge, contract left edge\n"
        "- Track window state incrementally (add/remove)\n"
        "- Update best answer at each valid state"
    ),

    "graph-traversal": (
        "When generating code for graph traversal:\n"
        "- Choose BFS (shortest path) or DFS (exhaustive)\n"
        "- Always maintain a visited set\n"
        "- Build adjacency list from edges\n"
        "- Handle disconnected components if needed"
    ),
}


class PromptRegistry:
    """Registry of strategy-specific prompt augmentations."""

    def __init__(self):
        self.templates = STRATEGY_PROMPTS.copy()

    def get_strategy_prompt(self, strategy_name: str) -> Optional[str]:
        """Get prompt augmentation for a strategy."""
        return self.templates.get(strategy_name)

    def build_generation_prompt(self, specification: str,
                                strategy_name: Optional[str] = None,
                                retrieved_knowledge: Optional[List[str]] = None) -> str:
        """Build a complete generation prompt with strategy context."""
        parts = []

        if strategy_name and strategy_name in self.templates:
            parts.append(self.templates[strategy_name])
            parts.append("")

        if retrieved_knowledge:
            parts.append("Relevant knowledge:")
            for k in retrieved_knowledge[:3]:
                parts.append(f"  - {k}")
            parts.append("")

        parts.append(f"Specification: {specification}")
        parts.append("")
        parts.append("Provide the Python implementation:")

        return "\n".join(parts)

    def register_template(self, strategy_name: str, template: str) -> None:
        """Register a new strategy prompt template."""
        self.templates[strategy_name] = template
