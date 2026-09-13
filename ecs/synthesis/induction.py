"""
Program Induction Engine: Learning Abstractions from Experience.

Instead of hand-coded templates, this system DISCOVERS solution
patterns from programs it has solved:

1. Record every solved program's AST structure
2. Cluster programs by structural similarity
3. When a cluster has 3+ members, extract the shared pattern
4. Name it, store it, and reuse it for new problems

This is how the system becomes creative: it discovers that
"iterate + accumulate + conditional update" is a reusable concept
across many problems, without anyone programming that in.

The induction engine also maintains an analogy map:
if problem A was solved with structure X, and new problem B
is structurally similar to A, try structure X for B.
"""

import ast
import hashlib
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field


@dataclass
class ProgramRecord:
    """A solved program with its structural fingerprint."""
    problem: str
    code: str
    entry_point: str
    structure: Dict[str, Any]
    fingerprint: str


@dataclass
class LearnedAbstraction:
    """A pattern discovered from multiple solved programs."""
    name: str
    description: str
    structure: Dict[str, Any]
    example_codes: List[str]
    example_problems: List[str]
    usage_count: int = 0
    template: Optional[str] = None


class ProgramInductionEngine:

    def __init__(self):
        self.solved: List[ProgramRecord] = []
        self.abstractions: Dict[str, LearnedAbstraction] = {}
        self._fingerprint_cache: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Record solved programs
    # ------------------------------------------------------------------

    def record(self, problem: str, code: str,
               entry_point: str = "") -> None:
        """Record a solved program for structural analysis."""
        structure = self._extract_structure(code)
        fp = self._fingerprint(structure)
        self.solved.append(ProgramRecord(
            problem=problem,
            code=code,
            entry_point=entry_point,
            structure=structure,
            fingerprint=fp,
        ))

    # ------------------------------------------------------------------
    # Structure extraction (the core intelligence)
    # ------------------------------------------------------------------

    def _extract_structure(self, code: str) -> Dict[str, Any]:
        """Extract structural skeleton from code, ignoring surface names."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {"parse_error": True}

        structure = {
            "has_loop": False,
            "has_for": False,
            "has_while": False,
            "has_recursion": False,
            "has_nested_loop": False,
            "has_conditional": False,
            "has_early_return": False,
            "loop_depth": 0,
            "return_count": 0,
            "uses_accumulator": False,
            "uses_dict": False,
            "uses_set": False,
            "uses_list_comp": False,
            "uses_builtins": set(),
            "param_count": 0,
            "operation_sequence": [],
            "pattern_tags": set(),
        }

        func_names = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_names.add(node.name)
                structure["param_count"] = len(node.args.args)

        self._analyze_node(tree, structure, func_names, depth=0)

        # Derive high-level pattern tags
        self._tag_patterns(structure)

        # Convert sets to sorted lists for fingerprinting
        structure["uses_builtins"] = sorted(structure["uses_builtins"])
        structure["pattern_tags"] = sorted(structure["pattern_tags"])

        return structure

    def _analyze_node(self, node: ast.AST, structure: Dict,
                       func_names: Set[str], depth: int) -> None:
        """Recursively analyze AST nodes."""
        loop_depth = 0

        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While)):
                structure["has_loop"] = True
                if isinstance(child, ast.For):
                    structure["has_for"] = True
                else:
                    structure["has_while"] = True
                structure["operation_sequence"].append("LOOP")
                new_depth = depth + 1
                structure["loop_depth"] = max(
                    structure["loop_depth"], new_depth
                )
                if new_depth >= 2:
                    structure["has_nested_loop"] = True

            elif isinstance(child, ast.If):
                structure["has_conditional"] = True
                structure["operation_sequence"].append("COND")

            elif isinstance(child, ast.Return):
                structure["return_count"] += 1
                if depth > 0:
                    structure["has_early_return"] = True
                structure["operation_sequence"].append("RETURN")

            elif isinstance(child, ast.AugAssign):
                structure["uses_accumulator"] = True
                structure["operation_sequence"].append("ACCUM")

            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    name = child.func.id
                    if name in func_names:
                        structure["has_recursion"] = True
                        structure["operation_sequence"].append("RECURSE")
                    elif name in ("sorted", "sum", "max", "min", "len",
                                  "abs", "range", "enumerate", "zip",
                                  "map", "filter", "set", "list",
                                  "all", "any", "reversed", "dict"):
                        structure["uses_builtins"].add(name)
                    if name == "set":
                        structure["uses_set"] = True
                    if name == "dict":
                        structure["uses_dict"] = True

            elif isinstance(child, ast.ListComp):
                structure["uses_list_comp"] = True
                structure["operation_sequence"].append("LISTCOMP")

            elif isinstance(child, ast.Dict):
                structure["uses_dict"] = True

            elif isinstance(child, ast.Set):
                structure["uses_set"] = True

            self._analyze_node(child, structure, func_names,
                               depth + (1 if isinstance(child, (ast.For, ast.While)) else 0))

    def _tag_patterns(self, structure: Dict) -> None:
        """Assign high-level pattern tags based on structure."""
        tags = structure["pattern_tags"]

        if structure["has_recursion"]:
            tags.add("recursive")
        if structure["has_loop"] and structure["uses_accumulator"]:
            tags.add("accumulate")
        if structure["has_loop"] and structure["has_early_return"]:
            tags.add("search")
        if structure["has_nested_loop"]:
            tags.add("quadratic")
        if structure["uses_list_comp"]:
            tags.add("comprehension")
        if structure["uses_dict"] or structure["uses_set"]:
            tags.add("hash_based")
        if structure["has_loop"] and not structure["uses_accumulator"]:
            if structure["has_conditional"]:
                tags.add("filter_scan")
        if "sorted" in structure.get("uses_builtins", []):
            tags.add("sort_first")
        if structure["has_while"] and not structure["has_for"]:
            tags.add("convergence")

    def _fingerprint(self, structure: Dict) -> str:
        """Create a hashable fingerprint from structure."""
        key_parts = [
            f"loop={structure.get('has_loop')}",
            f"rec={structure.get('has_recursion')}",
            f"nest={structure.get('has_nested_loop')}",
            f"acc={structure.get('uses_accumulator')}",
            f"lc={structure.get('uses_list_comp')}",
            f"hash={structure.get('uses_dict') or structure.get('uses_set')}",
            f"early={structure.get('has_early_return')}",
            f"depth={structure.get('loop_depth')}",
            f"tags={','.join(sorted(structure.get('pattern_tags', [])))}",
        ]
        raw = "|".join(key_parts)
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    # ------------------------------------------------------------------
    # Abstraction mining
    # ------------------------------------------------------------------

    def mine_abstractions(self) -> List[LearnedAbstraction]:
        """Discover new abstractions from accumulated solved programs."""
        if len(self.solved) < 3:
            return []

        clusters = self._cluster_by_fingerprint()
        new_abstractions = []

        for fp, members in clusters.items():
            if len(members) < 3:
                continue

            name = self._generate_name(members)
            if name in self.abstractions:
                self.abstractions[name].usage_count = len(members)
                continue

            desc = self._describe_cluster(members)
            template = self._extract_template(members)

            abstraction = LearnedAbstraction(
                name=name,
                description=desc,
                structure=members[0].structure,
                example_codes=[m.code for m in members[:5]],
                example_problems=[m.problem for m in members[:5]],
                usage_count=len(members),
                template=template,
            )
            self.abstractions[name] = abstraction
            new_abstractions.append(abstraction)

        return new_abstractions

    def _cluster_by_fingerprint(self) -> Dict[str, List[ProgramRecord]]:
        """Group solved programs by structural fingerprint."""
        clusters: Dict[str, List[ProgramRecord]] = {}
        for record in self.solved:
            fp = record.fingerprint
            if fp not in clusters:
                clusters[fp] = []
            clusters[fp].append(record)
        return clusters

    def _generate_name(self, members: List[ProgramRecord]) -> str:
        """Generate a meaningful name for a cluster."""
        tags = set()
        for m in members:
            for t in m.structure.get("pattern_tags", []):
                tags.add(t)

        if "recursive" in tags:
            return "recursive_decomposition"
        if "accumulate" in tags and "search" in tags:
            return "scan_and_track"
        if "accumulate" in tags:
            return "accumulate_pattern"
        if "comprehension" in tags and "hash_based" in tags:
            return "hash_filter_pattern"
        if "comprehension" in tags:
            return "comprehension_pattern"
        if "search" in tags:
            return "early_exit_search"
        if "quadratic" in tags:
            return "pairwise_comparison"
        if "hash_based" in tags:
            return "hash_lookup_pattern"
        if "sort_first" in tags:
            return "sort_then_process"
        if "convergence" in tags:
            return "convergence_loop"

        return f"structural_pattern_{members[0].fingerprint[:6]}"

    def _describe_cluster(self, members: List[ProgramRecord]) -> str:
        """Generate a description of what this cluster represents."""
        s = members[0].structure
        parts = []

        if s.get("has_recursion"):
            parts.append("breaks problem into subproblems recursively")
        elif s.get("has_loop"):
            if s.get("uses_accumulator"):
                parts.append("iterates while accumulating a result")
            elif s.get("has_early_return"):
                parts.append("scans until a condition is found")
            else:
                parts.append("iterates over input")

        if s.get("uses_list_comp"):
            parts.append("uses list comprehension for transformation")
        if s.get("uses_dict") or s.get("uses_set"):
            parts.append("uses hash-based lookup for efficiency")
        if s.get("has_nested_loop"):
            parts.append("compares pairs of elements")

        return "; ".join(parts) if parts else "general computation"

    def _extract_template(self, members: List[ProgramRecord]) -> Optional[str]:
        """Extract a generalized template from cluster members.

        Replace specific names with placeholders to create a reusable skeleton.
        """
        if not members:
            return None

        # Use the shortest member as the template base
        shortest = min(members, key=lambda m: len(m.code))
        code = shortest.code

        # Replace the function name with FUNC
        code = re.sub(
            r'def\s+\w+\s*\(',
            'def FUNC(',
            code,
            count=1
        )

        return code

    # ------------------------------------------------------------------
    # Analogy-based synthesis
    # ------------------------------------------------------------------

    def find_analogous_solution(self, problem: str,
                                 examples: Optional[List[Dict]] = None
                                 ) -> Optional[str]:
        """Find a solved program structurally similar to the new problem.

        Uses problem text similarity as a proxy: if a new problem uses
        similar vocabulary to a past problem, its solution structure
        is likely similar too.
        """
        if not self.solved:
            return None

        problem_tokens = set(problem.lower().split())
        best_match = None
        best_score = 0

        for record in self.solved:
            record_tokens = set(record.problem.lower().split())
            overlap = len(problem_tokens & record_tokens)
            union = len(problem_tokens | record_tokens)
            if union == 0:
                continue
            jaccard = overlap / union
            if jaccard > best_score and jaccard > 0.3:
                best_score = jaccard
                best_match = record

        if best_match:
            return best_match.code

        return None

    def suggest_structure(self, problem: str) -> Optional[Dict]:
        """Suggest what structural approach might work for a problem.

        Returns pattern tags and a description — not code.
        """
        analog = self.find_analogous_solution(problem)
        if not analog:
            return None

        structure = self._extract_structure(analog)
        return {
            "pattern_tags": structure.get("pattern_tags", []),
            "description": self._describe_cluster(
                [ProgramRecord(problem="", code=analog, entry_point="",
                               structure=structure, fingerprint="")]
            ),
            "analogous_code": analog,
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        return {
            "solved_count": len(self.solved),
            "abstractions_count": len(self.abstractions),
            "abstraction_names": list(self.abstractions.keys()),
            "fingerprint_distribution": {
                fp: len(members)
                for fp, members in self._cluster_by_fingerprint().items()
            },
        }
