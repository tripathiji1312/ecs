"""
Execution-Guided Synthesis: Thinking by Doing.

Instead of generating complete code and checking if it works,
this synthesizer builds solutions incrementally:

1. Parse examples to understand input/output contract
2. Generate a skeleton (def + return placeholder)
3. Try candidate expressions, EXECUTE each, observe results
4. Keep expressions that produce correct output on examples
5. Compose verified sub-expressions into a complete solution

The key insight: by executing partial programs against concrete
examples, the system DISCOVERS what works rather than GUESSING
from keywords. A filter condition is found by trying candidate
predicates and seeing which one produces the right output — not
by matching "even" → "x % 2 == 0".
"""

import re
import ast
import itertools
from typing import Dict, List, Optional, Tuple, Any

from ecs.verification.sandbox import SafeExecutor


class ExecutionGuidedSynthesizer:

    def __init__(self):
        self.executor = SafeExecutor(timeout=3.0)

    def synthesize(self, problem: str, prompt: str,
                   entry_point: str) -> Optional[str]:
        """Attempt execution-guided synthesis from examples.

        Returns verified function code or None.
        """
        examples = self._extract_examples(prompt, entry_point)
        params = self._extract_params(prompt, entry_point)
        if not examples or not params:
            return None

        in_type, out_type = self._infer_types(examples)
        param_str = ", ".join(params)

        # Strategy 1: Single-expression synthesis
        # Try to find one expression that maps inputs → outputs
        code = self._try_single_expression(
            entry_point, params, param_str, examples, in_type, out_type, problem
        )
        if code:
            return code

        # Strategy 2: Conditional synthesis
        # For bool outputs or mixed outputs, try if/else structures
        code = self._try_conditional(
            entry_point, params, param_str, examples, in_type, out_type, problem
        )
        if code:
            return code

        # Strategy 3: Loop-accumulate synthesis
        # For list→value or list→list, try iteration patterns
        if in_type == "list":
            code = self._try_loop_accumulate(
                entry_point, params, param_str, examples, out_type, problem
            )
            if code:
                return code

        # Strategy 4: Recursive decomposition
        # Try breaking the problem into base case + recursive step
        code = self._try_recursive(
            entry_point, params, param_str, examples, in_type, out_type
        )
        if code:
            return code

        return None

    # ------------------------------------------------------------------
    # Strategy 1: Single-expression search
    # ------------------------------------------------------------------

    def _try_single_expression(self, entry_point: str, params: List[str],
                                param_str: str, examples: List[Dict],
                                in_type: str, out_type: str,
                                problem: str) -> Optional[str]:
        """Try to find a single expression that solves all examples."""
        first = params[0]
        candidates = []

        # Generate candidate expressions based on type signature
        if in_type == "list" and out_type in ("int", "float"):
            candidates.extend([
                f"sum({first})",
                f"len({first})",
                f"max({first})" if out_type != "float" else f"max({first})",
                f"min({first})",
                f"sum({first}) / len({first})",
                f"sum(abs(x - sum({first})/len({first})) for x in {first}) / len({first})",
                f"len(set({first}))",
                f"len([x for x in {first} if x > 0])",
                f"len([x for x in {first} if x % 2 == 0])",
            ])
            # Median
            candidates.append(
                f"(lambda s, n: s[n//2] if n % 2 else (s[n//2-1]+s[n//2])/2)"
                f"(sorted({first}), len({first}))"
            )

        if in_type == "list" and out_type == "list":
            candidates.extend([
                f"sorted({first})",
                f"sorted(set({first}))",
                f"list(set({first}))",
                f"list(reversed({first}))",
                f"{first}[::-1]",
                f"[x for x in {first} if x > 0]",
                f"[x for x in {first} if x % 2 == 0]",
                f"[x for x in {first} if x % 2 != 0]",
                f"[x * 2 for x in {first}]",
                f"[x + 1 for x in {first}]",
                f"[abs(x) for x in {first}]",
            ])

        if in_type == "list" and out_type == "bool":
            candidates.extend([
                f"all(x > 0 for x in {first})",
                f"any(x > 0 for x in {first})",
                f"len({first}) == len(set({first}))",
                f"{first} == sorted({first})",
                f"{first} == sorted({first}, reverse=True)",
            ])

        if in_type == "str" and out_type == "bool":
            candidates.extend([
                f"{first} == {first}[::-1]",
                f"{first}.isdigit()",
                f"{first}.isalpha()",
                f"len({first}) > 0",
                f"all(c in '()' for c in {first})",
            ])

        if in_type == "str" and out_type == "int":
            candidates.extend([
                f"len({first})",
                f"len(set({first}))",
                f"len(set({first}.lower()))",
                f"sum(1 for c in {first} if c.isupper())",
                f"sum(1 for c in {first} if c.isdigit())",
            ])

        if in_type == "str" and out_type == "str":
            candidates.extend([
                f"{first}[::-1]",
                f"{first}.lower()",
                f"{first}.upper()",
                f"{first}.strip()",
                f"''.join(sorted({first}))",
                f"''.join(c for c in {first} if c.isalpha())",
            ])

        if in_type == "str" and out_type == "list":
            candidates.extend([
                f"list({first})",
                f"{first}.split()",
                f"[{first}[:i+1] for i in range(len({first}))]",
            ])

        if in_type in ("int", "float") and out_type in ("int", "float"):
            candidates.extend([
                f"{first}",
                f"abs({first})",
                f"{first} * {first}",
                f"{first} * 2",
                f"{first} + 1",
                f"{first} - 1",
            ])
            if len(params) >= 2:
                second = params[1]
                candidates.extend([
                    f"{first} + {second}",
                    f"{first} - {second}",
                    f"{first} * {second}",
                    f"{first} // {second}" if out_type == "int" else f"{first} / {second}",
                    f"{first} % {second}",
                    f"max({first}, {second})",
                    f"min({first}, {second})",
                    f"{first} ** {second}",
                ])

        if in_type in ("int", "float") and out_type == "bool":
            candidates.extend([
                f"{first} > 0",
                f"{first} < 0",
                f"{first} == 0",
                f"{first} % 2 == 0",
                f"{first} >= 0",
            ])
            if len(params) >= 2:
                second = params[1]
                candidates.extend([
                    f"{first} == {second}",
                    f"{first} > {second}",
                    f"{first} < {second}",
                    f"{first} % {second} == 0",
                ])

        if in_type in ("int", "float") and out_type == "list":
            candidates.extend([
                f"list(range({first}))",
                f"list(range(1, {first} + 1))",
                f"list(range({first}, 0, -1))",
            ])

        # Multi-arg list operations
        if len(params) >= 2:
            second = params[1]
            if in_type == "list":
                candidates.extend([
                    f"[x for x in {first} if {second} in str(x)]",
                    f"[x for x in {first} if x.startswith({second})]"
                    if out_type == "list" else "",
                    f"[x for x in {first} if {second} in x]"
                    if out_type == "list" else "",
                ])

        # Filter empty strings
        candidates = [c for c in candidates if c]

        # Execute each candidate and check against examples
        for expr in candidates:
            code = (
                f"def {entry_point}({param_str}):\n"
                f"    return {expr}\n"
            )
            if self._verify_all(code, examples, entry_point):
                return code

        return None

    # ------------------------------------------------------------------
    # Strategy 2: Conditional synthesis
    # ------------------------------------------------------------------

    def _try_conditional(self, entry_point: str, params: List[str],
                          param_str: str, examples: List[Dict],
                          in_type: str, out_type: str,
                          problem: str) -> Optional[str]:
        """Try if/else patterns."""
        first = params[0]

        # Group examples by output to find condition boundaries
        if out_type == "bool":
            true_exs = [e for e in examples if e["output"] is True]
            false_exs = [e for e in examples if e["output"] is False]

            if not true_exs or not false_exs:
                return None

            # Try to find a condition that separates true from false
            conditions = self._generate_conditions(
                params, in_type, true_exs, false_exs
            )

            for cond in conditions:
                code = (
                    f"def {entry_point}({param_str}):\n"
                    f"    return {cond}\n"
                )
                if self._verify_all(code, examples, entry_point):
                    return code

        # Empty-input guard pattern (common in HumanEval)
        if in_type == "list":
            empty_exs = [e for e in examples
                         if self._get_first_arg(e["input"]) == []
                         or self._get_first_arg(e["input"]) is None]
            if empty_exs:
                empty_ret = repr(empty_exs[0]["output"])
                non_empty = [e for e in examples if e not in empty_exs]

                # Try expressions for the non-empty case
                exprs = self._guess_expressions_for(
                    params, in_type, out_type, non_empty
                )
                for expr in exprs:
                    code = (
                        f"def {entry_point}({param_str}):\n"
                        f"    if not {first}:\n"
                        f"        return {empty_ret}\n"
                        f"    return {expr}\n"
                    )
                    if self._verify_all(code, examples, entry_point):
                        return code

        return None

    def _generate_conditions(self, params: List[str], in_type: str,
                              true_exs: List[Dict],
                              false_exs: List[Dict]) -> List[str]:
        """Generate candidate boolean conditions."""
        first = params[0]
        conditions = []

        if in_type == "list":
            conditions.extend([
                f"len({first}) > 0",
                f"len({first}) == len(set({first}))",
                f"all({first}[i] <= {first}[i+1] for i in range(len({first})-1)) if len({first}) > 1 else True",
                f"{first} == sorted({first})",
            ])
            if len(params) >= 2:
                second = params[1]
                conditions.extend([
                    f"any(abs({first}[i] - {first}[j]) < {second} "
                    f"for i in range(len({first})) "
                    f"for j in range(i+1, len({first})))",
                ])

        if in_type in ("int", "float"):
            conditions.extend([
                f"{first} > 0",
                f"{first} >= 0",
                f"{first} % 2 == 0",
                f"{first} > 1",
            ])
            if len(params) >= 2:
                second = params[1]
                conditions.extend([
                    f"{first} % {second} == 0",
                    f"{first} > {second}",
                    f"{first} == {second}",
                ])

        if in_type == "str":
            conditions.extend([
                f"{first} == {first}[::-1]",
                f"len({first}) > 0",
                f"{first}.isdigit()",
                f"{first}.isalpha()",
            ])

        return conditions

    # ------------------------------------------------------------------
    # Strategy 3: Loop-accumulate
    # ------------------------------------------------------------------

    def _try_loop_accumulate(self, entry_point: str, params: List[str],
                              param_str: str, examples: List[Dict],
                              out_type: str,
                              problem: str) -> Optional[str]:
        """Try loop-based accumulation patterns."""
        first = params[0]
        templates = []

        if out_type in ("int", "float"):
            # Running max/min with tracking
            templates.extend([
                # Running sum
                (f"def {entry_point}({param_str}):\n"
                 f"    total = 0\n"
                 f"    for x in {first}:\n"
                 f"        total += x\n"
                 f"    return total\n"),
                # Count with condition
                (f"def {entry_point}({param_str}):\n"
                 f"    count = 0\n"
                 f"    for x in {first}:\n"
                 f"        if x > 0:\n"
                 f"            count += 1\n"
                 f"    return count\n"),
                # Running max
                (f"def {entry_point}({param_str}):\n"
                 f"    best = {first}[0]\n"
                 f"    for x in {first}[1:]:\n"
                 f"        if x > best:\n"
                 f"            best = x\n"
                 f"    return best\n"),
            ])

        if out_type == "list":
            # Build output list by accumulation
            templates.extend([
                # Running max list
                (f"def {entry_point}({param_str}):\n"
                 f"    result = []\n"
                 f"    current_max = float('-inf')\n"
                 f"    for x in {first}:\n"
                 f"        current_max = max(current_max, x)\n"
                 f"        result.append(current_max)\n"
                 f"    return result\n"),
                # Running sum list
                (f"def {entry_point}({param_str}):\n"
                 f"    result = []\n"
                 f"    total = 0\n"
                 f"    for x in {first}:\n"
                 f"        total += x\n"
                 f"        result.append(total)\n"
                 f"    return result\n"),
                # Prefix strings
                (f"def {entry_point}({param_str}):\n"
                 f"    result = []\n"
                 f"    for i in range(1, len({first}) + 1):\n"
                 f"        result.append({first}[:i])\n"
                 f"    return result\n"),
                # Deduplicate preserving order
                (f"def {entry_point}({param_str}):\n"
                 f"    seen = set()\n"
                 f"    result = []\n"
                 f"    for x in {first}:\n"
                 f"        if x not in seen:\n"
                 f"            seen.add(x)\n"
                 f"            result.append(x)\n"
                 f"    return result\n"),
            ])

        if out_type == "bool":
            # Iterate and check condition
            templates.extend([
                # Balance check (below zero)
                (f"def {entry_point}({param_str}):\n"
                 f"    balance = 0\n"
                 f"    for x in {first}:\n"
                 f"        balance += x\n"
                 f"        if balance < 0:\n"
                 f"            return True\n"
                 f"    return False\n"),
                # All unique
                (f"def {entry_point}({param_str}):\n"
                 f"    return len({first}) == len(set({first}))\n"),
                # Monotonic increasing
                (f"def {entry_point}({param_str}):\n"
                 f"    for i in range(len({first}) - 1):\n"
                 f"        if {first}[i] > {first}[i+1]:\n"
                 f"            return False\n"
                 f"    return True\n"),
            ])

        if out_type == "str":
            templates.extend([
                # Join
                (f"def {entry_point}({param_str}):\n"
                 f"    return ''.join({first})\n"),
                (f"def {entry_point}({param_str}):\n"
                 f"    return ' '.join({first})\n"),
                (f"def {entry_point}({param_str}):\n"
                 f"    return ', '.join(str(x) for x in {first})\n"),
            ])

        for code in templates:
            if self._verify_all(code, examples, entry_point):
                return code

        # Pairwise iteration (for problems needing comparison of pairs)
        if len(params) >= 2 and out_type == "bool":
            second = params[1]
            pairwise = [
                (f"def {entry_point}({param_str}):\n"
                 f"    for i in range(len({first})):\n"
                 f"        for j in range(i + 1, len({first})):\n"
                 f"            if abs({first}[i] - {first}[j]) < {second}:\n"
                 f"                return True\n"
                 f"    return False\n"),
                (f"def {entry_point}({param_str}):\n"
                 f"    for i in range(len({first})):\n"
                 f"        for j in range(i + 1, len({first})):\n"
                 f"            if {first}[i] + {first}[j] == {second}:\n"
                 f"                return True\n"
                 f"    return False\n"),
            ]
            for code in pairwise:
                if self._verify_all(code, examples, entry_point):
                    return code

        return None

    # ------------------------------------------------------------------
    # Strategy 4: Recursive decomposition
    # ------------------------------------------------------------------

    def _try_recursive(self, entry_point: str, params: List[str],
                        param_str: str, examples: List[Dict],
                        in_type: str, out_type: str) -> Optional[str]:
        """Try recursive patterns for numeric inputs."""
        if in_type not in ("int", "float") or len(params) < 1:
            return None

        first = params[0]

        # Find base cases: examples with small inputs
        base_cases = []
        recursive_cases = []
        for ex in examples:
            inp = self._get_first_arg(ex["input"])
            if isinstance(inp, (int, float)) and abs(inp) <= 2:
                base_cases.append(ex)
            else:
                recursive_cases.append(ex)

        if not base_cases:
            return None

        # Generate base case code from examples
        base_lines = []
        for ex in base_cases:
            inp = self._get_first_arg(ex["input"])
            base_lines.append(
                f"    if {first} == {repr(inp)}:\n"
                f"        return {repr(ex['output'])}"
            )
        base_code = "\n".join(base_lines)

        # Try common recursive patterns
        rec_exprs = [
            f"{first} * {entry_point}({first} - 1)",
            f"{entry_point}({first} - 1) + {entry_point}({first} - 2)",
            f"{first} + {entry_point}({first} - 1)",
            f"1 + {entry_point}({first} - 1)",
        ]

        for rec in rec_exprs:
            code = (
                f"def {entry_point}({param_str}):\n"
                f"{base_code}\n"
                f"    return {rec}\n"
            )
            if self._verify_all(code, examples, entry_point):
                return code

        return None

    # ------------------------------------------------------------------
    # Expression search for non-empty case
    # ------------------------------------------------------------------

    def _guess_expressions_for(self, params: List[str], in_type: str,
                                out_type: str,
                                examples: List[Dict]) -> List[str]:
        """Generate candidate expressions for non-empty inputs."""
        first = params[0]
        exprs = []

        if in_type == "list" and out_type in ("int", "float"):
            exprs.extend([
                f"max({first})",
                f"min({first})",
                f"sum({first})",
                f"len({first})",
                f"sum({first}) / len({first})",
                f"max({first}, key=len)",
            ])

        if in_type == "list" and out_type == "str":
            exprs.extend([
                f"max({first}, key=len)",
                f"min({first}, key=len)",
                f"max({first})",
                f"''.join({first})",
            ])

        if in_type == "list" and out_type == "list":
            exprs.extend([
                f"sorted({first})",
                f"sorted(set({first}))",
                f"{first}[::-1]",
            ])

        return exprs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_first_arg(self, inp: Any) -> Any:
        if isinstance(inp, tuple):
            return inp[0] if inp else None
        return inp

    def _infer_types(self, examples: List[Dict]) -> Tuple[str, str]:
        if not examples:
            return "unknown", "unknown"
        ex = examples[0]
        inp = ex["input"]
        if isinstance(inp, tuple):
            inp = inp[0] if inp else None
        return self._type_of(inp), self._type_of(ex["output"])

    def _type_of(self, val: Any) -> str:
        if isinstance(val, bool):
            return "bool"
        if isinstance(val, int):
            return "int"
        if isinstance(val, float):
            return "float"
        if isinstance(val, str):
            return "str"
        if isinstance(val, list):
            return "list"
        if isinstance(val, tuple):
            return "tuple"
        if val is None:
            return "none"
        return "unknown"

    def _extract_examples(self, prompt: str,
                          entry_point: str) -> List[Dict]:
        examples = []
        lines = prompt.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(">>>"):
                call_str = stripped[3:].strip()
                if i + 1 < len(lines):
                    result_line = lines[i + 1].strip()
                    if result_line and not result_line.startswith(">>>"):
                        try:
                            output = eval(result_line)
                        except Exception:
                            continue
                        inp = self._parse_call_args(call_str, entry_point)
                        if inp is not None:
                            examples.append({"input": inp, "output": output})
        return examples

    def _parse_call_args(self, call_str: str,
                         entry_point: str) -> Any:
        match = re.match(rf'{re.escape(entry_point)}\s*\((.+)\)\s*$',
                         call_str, re.DOTALL)
        if not match:
            return None
        args_str = match.group(1)
        try:
            result = eval(f"({args_str},)")
            if len(result) == 1:
                return result[0]
            return result
        except Exception:
            return None

    def _extract_params(self, prompt: str,
                        entry_point: str) -> List[str]:
        match = re.search(
            rf'def\s+{re.escape(entry_point)}\s*\(([^)]*)\)',
            prompt
        )
        if not match:
            return ["x"]
        params = []
        for part in match.group(1).split(","):
            part = part.strip()
            name_match = re.match(r'(\w+)', part)
            if name_match:
                params.append(name_match.group(1))
        return params if params else ["x"]

    def _verify_all(self, code: str, examples: List[Dict],
                     entry_point: str) -> bool:
        """Verify code against ALL examples by executing."""
        if not examples:
            return False
        for ex in examples:
            inp = ex["input"]
            expected = ex["output"]
            if isinstance(inp, tuple):
                call_args = ", ".join(repr(a) for a in inp)
            else:
                call_args = repr(inp)
            test_code = (
                f"{code}\n"
                f"_result = {entry_point}({call_args})\n"
                f"assert _result == {repr(expected)}, "
                f"repr(_result)\n"
            )
            result = self.executor.execute(test_code)
            if not result.success:
                return False
        return True
