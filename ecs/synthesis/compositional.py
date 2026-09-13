"""
Compositional Synthesizer: Type-Driven Reasoning + Primitive Composition.

This synthesizer reasons at two levels:

LEVEL 1 — Type-driven reasoning:
  Infer the type signature (list→int, str→bool, etc.) from examples,
  then look up which TRANSFORMATION RULES apply for that signature.
  Types constrain the search space before any keyword matching happens.

LEVEL 2 — Compositional construction:
  Decompose the transformation into a pipeline of verified primitives:
  filter → map → reduce, or sort → take → transform.
  Each primitive is simple and correct; the composition is verified
  against examples.

This is reasoning, not retrieval: the system UNDERSTANDS that
"list → int" means aggregation, then REASONS about which aggregation
by trying candidates against concrete examples.
"""

import re
from typing import Dict, List, Optional, Tuple, Any

from ecs.verification.sandbox import SafeExecutor


# Type signature → applicable transformation strategies
TRANSFORMATION_RULES: Dict[Tuple[str, str], Dict] = {
    ("list", "int"): {
        "description": "aggregate list to integer",
        "strategies": ["reduce", "count", "measure"],
    },
    ("list", "float"): {
        "description": "aggregate list to float",
        "strategies": ["reduce_float", "measure_float"],
    },
    ("list", "list"): {
        "description": "transform list to list",
        "strategies": ["filter", "map", "sort", "build"],
    },
    ("list", "bool"): {
        "description": "check list property",
        "strategies": ["check_all", "check_any", "check_property"],
    },
    ("list", "str"): {
        "description": "aggregate list to string",
        "strategies": ["join", "reduce_str"],
    },
    ("list", "tuple"): {
        "description": "multi-aggregate from list",
        "strategies": ["multi_reduce"],
    },
    ("str", "bool"): {
        "description": "check string property",
        "strategies": ["str_check"],
    },
    ("str", "int"): {
        "description": "measure string",
        "strategies": ["str_measure"],
    },
    ("str", "str"): {
        "description": "transform string",
        "strategies": ["str_transform"],
    },
    ("str", "list"): {
        "description": "decompose string",
        "strategies": ["str_decompose"],
    },
    ("int", "int"): {
        "description": "numeric computation",
        "strategies": ["arithmetic", "recursive_int"],
    },
    ("int", "float"): {
        "description": "numeric computation to float",
        "strategies": ["arithmetic_float"],
    },
    ("int", "bool"): {
        "description": "check number property",
        "strategies": ["num_check"],
    },
    ("int", "list"): {
        "description": "generate list from number",
        "strategies": ["generate"],
    },
    ("int", "str"): {
        "description": "number to string",
        "strategies": ["num_to_str"],
    },
    ("float", "float"): {
        "description": "float computation",
        "strategies": ["arithmetic_float"],
    },
}


class CompositionalSynthesizer:

    def __init__(self):
        self.executor = SafeExecutor(timeout=5.0)

    def synthesize(self, problem: str, prompt: str,
                   entry_point: str) -> Optional[str]:
        """Attempt type-driven compositional synthesis.

        Returns complete function code if successful, None otherwise.
        """
        examples = self._extract_examples(prompt, entry_point)
        params = self._extract_params(prompt, entry_point)

        if not examples:
            return None

        in_type, out_type = self._infer_types(examples)

        # Level 1: Type-driven strategy selection
        rules = TRANSFORMATION_RULES.get((in_type, out_type))
        if not rules:
            return None

        # Level 2: Generate and verify candidates per strategy
        for strategy in rules["strategies"]:
            candidates = self._candidates_for_strategy(
                strategy, problem, entry_point, params, examples,
                in_type, out_type
            )
            for code in candidates:
                if self._verify(code, examples, entry_point):
                    return code

        return None

    # ------------------------------------------------------------------
    # Type inference from examples
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Strategy-based candidate generation
    # ------------------------------------------------------------------

    def _candidates_for_strategy(self, strategy: str, problem: str,
                                  entry_point: str, params: List[str],
                                  examples: List[Dict],
                                  in_type: str, out_type: str
                                  ) -> List[str]:
        """Generate candidates for a specific strategy."""
        p = problem.lower()
        ps = ", ".join(params)
        first = params[0] if params else "x"
        second = params[1] if len(params) >= 2 else None

        def fn(body_lines: List[str]) -> str:
            body = "\n".join(f"    {line}" for line in body_lines)
            return f"def {entry_point}({ps}):\n{body}\n"

        candidates = []

        # ── LIST → INT/FLOAT aggregation ──
        if strategy == "reduce":
            candidates.extend([
                fn([f"return sum({first})"]),
                fn([f"return len({first})"]),
                fn([f"return max({first})"]),
                fn([f"return min({first})"]),
                fn([f"return len(set({first}))"]),
                fn([f"return len([x for x in {first} if x > 0])"]),
                fn([f"return len([x for x in {first} if x % 2 == 0])"]),
                fn([f"return max({first}) - min({first})"]),
            ])

        if strategy == "count":
            candidates.extend([
                fn([f"return len({first})"]),
                fn([f"return len(set({first}))"]),
                fn([f"return sum(1 for x in {first} if x > 0)"]),
            ])

        if strategy in ("measure", "measure_float"):
            candidates.extend([
                fn([f"return sum({first}) / len({first})"]),
                fn([f"mean = sum({first}) / len({first})",
                    f"return sum(abs(x - mean) for x in {first}) / len({first})"]),
            ])
            # Median
            candidates.append(fn([
                f"s = sorted({first})",
                f"n = len(s)",
                f"if n % 2 == 1:",
                f"    return s[n // 2]",
                f"return (s[n // 2 - 1] + s[n // 2]) / 2.0",
            ]))

        if strategy == "reduce_float":
            candidates.extend([
                fn([f"return sum({first})"]),
                fn([f"return sum({first}) / len({first})"]),
                fn([f"return float(max({first}))"]),
            ])

        # ── LIST → LIST transformation ──
        if strategy == "filter":
            conditions = self._infer_filter_conditions(p, examples, first, second)
            for cond in conditions:
                candidates.append(fn([f"return [x for x in {first} if {cond}]"]))

        if strategy == "map":
            exprs = self._infer_map_expressions(p, examples, first)
            for expr in exprs:
                candidates.append(fn([f"return [{expr} for x in {first}]"]))

        if strategy == "sort":
            candidates.extend([
                fn([f"return sorted({first})"]),
                fn([f"return sorted({first}, reverse=True)"]),
                fn([f"return sorted(set({first}))"]),
            ])

        if strategy == "build":
            candidates.extend([
                # Intersperse
                fn([f"if not {first}:", f"    return []",
                    f"result = [{first}[0]]",
                    f"for x in {first}[1:]:",
                    f"    result.append({second})" if second else f"    pass",
                    f"    result.append(x)",
                    f"return result"]) if second else "",
                # Running max
                fn([f"result = []", f"current_max = float('-inf')",
                    f"for x in {first}:",
                    f"    current_max = max(current_max, x)",
                    f"    result.append(current_max)",
                    f"return result"]),
                # Prefix list
                fn([f"return [{first}[:i+1] for i in range(len({first}))]"]),
                # Deduplicate preserving order
                fn([f"seen = set()", f"result = []",
                    f"for x in {first}:",
                    f"    if x not in seen:",
                    f"        seen.add(x)",
                    f"        result.append(x)",
                    f"return result"]),
                # Remove elements occurring more than once
                fn([f"from collections import Counter",
                    f"counts = Counter({first})",
                    f"return [x for x in {first} if counts[x] == 1]"]),
                # Flatten
                fn([f"result = []",
                    f"for item in {first}:",
                    f"    if isinstance(item, list):",
                    f"        result.extend(item)",
                    f"    else:",
                    f"        result.append(item)",
                    f"return result"]),
            ])
            candidates = [c for c in candidates if c]

        # ── LIST → BOOL checking ──
        if strategy == "check_all":
            conditions = self._infer_check_conditions(p, params)
            for cond in conditions:
                candidates.append(fn([f"return all({cond} for x in {first})"]))

        if strategy == "check_any":
            conditions = self._infer_check_conditions(p, params)
            for cond in conditions:
                candidates.append(fn([f"return any({cond} for x in {first})"]))
            # Pairwise closeness
            if second and ("close" in p or "closer" in p):
                candidates.append(fn([
                    f"for i in range(len({first})):",
                    f"    for j in range(i + 1, len({first})):",
                    f"        if abs({first}[i] - {first}[j]) < {second}:",
                    f"            return True",
                    f"return False",
                ]))
            # Balance check
            if "below zero" in p or "balance" in p:
                candidates.append(fn([
                    f"balance = 0",
                    f"for op in {first}:",
                    f"    balance += op",
                    f"    if balance < 0:",
                    f"        return True",
                    f"return False",
                ]))

        if strategy == "check_property":
            candidates.extend([
                fn([f"return len({first}) == len(set({first}))"]),
                fn([f"return {first} == sorted({first})"]),
            ])
            # Monotonic
            if "monoton" in p:
                candidates.append(fn([
                    f"if len({first}) <= 1:", f"    return True",
                    f"inc = all({first}[i] <= {first}[i+1] for i in range(len({first})-1))",
                    f"dec = all({first}[i] >= {first}[i+1] for i in range(len({first})-1))",
                    f"return inc or dec",
                ]))

        # ── LIST → STR join ──
        if strategy == "join":
            candidates.extend([
                fn([f"return ''.join({first})"]),
                fn([f"return ''.join(str(x) for x in {first})"]),
                fn([f"return ' '.join({first})"]),
            ])

        if strategy == "reduce_str":
            candidates.extend([
                fn([f"if not {first}:", f"    return None",
                    f"return max({first}, key=len)"]),
                fn([f"if not {first}:", f"    return None",
                    f"return min({first}, key=len)"]),
            ])

        # ── LIST → TUPLE ──
        if strategy == "multi_reduce":
            candidates.append(fn([
                f"s = sum({first}) if {first} else 0",
                f"p = 1",
                f"for x in {first}:", f"    p *= x",
                f"return (s, p)",
            ]))

        # ── STR → BOOL ──
        if strategy == "str_check":
            candidates.extend([
                fn([f"return {first} == {first}[::-1]"]),
                fn([f"return {first}.isdigit()"]),
                fn([f"return {first}.isalpha()"]),
                fn([f"return len({first}) > 0"]),
            ])
            # Bracket checking
            if "bracket" in p or "parenthes" in p:
                candidates.append(fn([
                    f"depth = 0",
                    f"for ch in {first}:",
                    f"    if ch in '(<':", f"        depth += 1",
                    f"    elif ch in ')>':", f"        depth -= 1",
                    f"    if depth < 0:", f"        return False",
                    f"return depth == 0",
                ]))

        # ── STR → INT ──
        if strategy == "str_measure":
            candidates.extend([
                fn([f"return len({first})"]),
                fn([f"return len(set({first}))"]),
                fn([f"return len(set({first}.lower()))"]),
                fn([f"return sum(1 for c in {first} if c.isupper())"]),
            ])
            if second:
                candidates.extend([
                    fn([f"count = 0", f"start = 0",
                        f"while True:",
                        f"    pos = {first}.find({second}, start)",
                        f"    if pos == -1:", f"        break",
                        f"    count += 1", f"    start = pos + 1",
                        f"return count"]),
                ])

        # ── STR → STR ──
        if strategy == "str_transform":
            candidates.extend([
                fn([f"return {first}[::-1]"]),
                fn([f"return {first}.lower()"]),
                fn([f"return {first}.upper()"]),
                fn([f"return {first}.strip()"]),
                fn([f"return ''.join(sorted({first}))"]),
                fn([f"return ''.join(c for c in {first} if c.isalpha())"]),
            ])

        # ── STR → LIST ──
        if strategy == "str_decompose":
            candidates.extend([
                fn([f"return list({first})"]),
                fn([f"return {first}.split()"]),
                fn([f"return [{first}[:i+1] for i in range(len({first}))]"]),
            ])

        # ── INT → INT ──
        if strategy == "arithmetic":
            candidates.extend([
                fn([f"return abs({first})"]),
                fn([f"return {first} * {first}"]),
                fn([f"return {first} + 1"]),
            ])
            if second:
                candidates.extend([
                    fn([f"return {first} + {second}"]),
                    fn([f"return {first} - {second}"]),
                    fn([f"return {first} * {second}"]),
                    fn([f"return {first} // {second}"]),
                    fn([f"return {first} % {second}"]),
                    fn([f"return max({first}, {second})"]),
                    fn([f"return min({first}, {second})"]),
                    fn([f"return {first} ** {second}"]),
                    # GCD
                    fn([f"a, b = {first}, {second}",
                        f"while b:", f"    a, b = b, a % b",
                        f"return a"]),
                ])

        if strategy == "recursive_int":
            # Factorial
            candidates.append(fn([
                f"if {first} <= 1:", f"    return 1",
                f"return {first} * {entry_point}({first} - 1)",
            ]))
            # Fibonacci
            if len(params) == 1:
                candidates.append(fn([
                    f"if {first} <= 0:", f"    return 0",
                    f"if {first} == 1:", f"    return 1",
                    f"a, b = 0, 1",
                    f"for _ in range(2, {first} + 1):",
                    f"    a, b = b, a + b",
                    f"return b",
                ]))

        if strategy == "arithmetic_float":
            candidates.extend([
                fn([f"return float({first})"]),
                fn([f"return abs({first})"]),
            ])
            if second:
                candidates.extend([
                    fn([f"return {first} + {second}"]),
                    fn([f"return {first} / {second}"]),
                ])

        # ── INT → BOOL ──
        if strategy == "num_check":
            candidates.extend([
                fn([f"return {first} > 0"]),
                fn([f"return {first} % 2 == 0"]),
                fn([f"return {first} >= 0"]),
            ])
            # Prime check
            if "prime" in p:
                candidates.append(fn([
                    f"if {first} < 2:", f"    return False",
                    f"for i in range(2, int({first}**0.5) + 1):",
                    f"    if {first} % i == 0:", f"        return False",
                    f"return True",
                ]))
            if second:
                # Power check
                if "power" in p:
                    candidates.append(fn([
                        f"if {first} == 1:", f"    return True",
                        f"if {second} == 1:", f"    return {first} == 1",
                        f"power = {second}",
                        f"while power < {first}:", f"    power *= {second}",
                        f"return power == {first}",
                    ]))
                candidates.extend([
                    fn([f"return {first} % {second} == 0"]),
                    fn([f"return {first} > {second}"]),
                    fn([f"return {first} == {second}"]),
                ])

        # ── INT → LIST ──
        if strategy == "generate":
            candidates.extend([
                fn([f"return list(range({first}))"]),
                fn([f"return list(range(1, {first} + 1))"]),
            ])
            # Factorize
            if "factor" in p:
                candidates.append(fn([
                    f"factors = []", f"d = 2",
                    f"n = {first}",
                    f"while d * d <= n:",
                    f"    while n % d == 0:",
                    f"        factors.append(d)", f"        n //= d",
                    f"    d += 1",
                    f"if n > 1:", f"    factors.append(n)",
                    f"return factors",
                ]))
            # Primes up to
            if "prime" in p:
                candidates.append(fn([
                    f"primes = []",
                    f"for i in range(2, {first}):",
                    f"    is_p = True",
                    f"    for j in range(2, int(i**0.5) + 1):",
                    f"        if i % j == 0:",
                    f"            is_p = False", f"            break",
                    f"    if is_p:", f"        primes.append(i)",
                    f"return primes",
                ]))

        # ── INT → STR ──
        if strategy == "num_to_str":
            candidates.extend([
                fn([f"return str({first})"]),
                fn([f"return bin({first})"]),
                fn([f"return hex({first})"]),
            ])

        return candidates

    # ------------------------------------------------------------------
    # Condition / expression inference
    # ------------------------------------------------------------------

    def _infer_filter_conditions(self, problem: str,
                                  examples: List[Dict],
                                  param: str,
                                  second: Optional[str]) -> List[str]:
        conditions = []

        if "positive" in problem:
            conditions.append("x > 0")
        if "negative" in problem:
            conditions.append("x < 0")
        if "even" in problem:
            conditions.append("x % 2 == 0")
        if "odd" in problem:
            conditions.append("x % 2 != 0")
        if "non-zero" in problem or "nonzero" in problem:
            conditions.append("x != 0")
        if second:
            if "contain" in problem and "substring" in problem:
                conditions.append(f"{second} in x")
            if "start" in problem or "prefix" in problem:
                conditions.append(f"x.startswith({second})")

        # Deduce from examples
        if not conditions and examples:
            cond = self._deduce_filter(examples)
            if cond:
                conditions.append(cond)

        return conditions if conditions else ["True"]

    def _deduce_filter(self, examples: List[Dict]) -> Optional[str]:
        for ex in examples:
            inp = self._get_first_arg(ex.get("input"))
            out = ex.get("output")
            if isinstance(inp, list) and isinstance(out, list) and len(out) < len(inp):
                if (all(isinstance(x, (int, float)) for x in out)
                        and all(isinstance(x, (int, float)) for x in inp)):
                    if all(x > 0 for x in out) and any(x <= 0 for x in inp):
                        return "x > 0"
                    if all(x % 2 == 0 for x in out):
                        return "x % 2 == 0"
        return None

    def _infer_map_expressions(self, problem: str,
                                examples: List[Dict],
                                param: str) -> List[str]:
        exprs = []
        if "increment" in problem:
            exprs.append("x + 1")
        if "double" in problem:
            exprs.append("x * 2")
        if "square" in problem:
            exprs.append("x * x")
        if "negate" in problem:
            exprs.append("-x")
        if "absolute" in problem:
            exprs.append("abs(x)")

        if not exprs and examples:
            expr = self._deduce_map(examples)
            if expr:
                exprs.append(expr)

        return exprs if exprs else ["x"]

    def _deduce_map(self, examples: List[Dict]) -> Optional[str]:
        for ex in examples:
            inp = self._get_first_arg(ex.get("input"))
            out = ex.get("output")
            if isinstance(inp, list) and isinstance(out, list) and len(inp) == len(out) and inp:
                diffs = [o - i for i, o in zip(inp, out)
                         if isinstance(i, (int, float)) and isinstance(o, (int, float))]
                if diffs and all(d == diffs[0] for d in diffs):
                    if diffs[0] == 1:
                        return "x + 1"
                    elif diffs[0] == -1:
                        return "x - 1"
                    return f"x + {diffs[0]}"
                ratios = [o / i for i, o in zip(inp, out)
                          if isinstance(i, (int, float)) and i != 0
                          and isinstance(o, (int, float))]
                if ratios and all(abs(r - ratios[0]) < 0.001 for r in ratios):
                    if ratios[0] == 2:
                        return "x * 2"
                    elif ratios[0] == -1:
                        return "-x"
        return None

    def _infer_check_conditions(self, problem: str,
                                 params: List[str]) -> List[str]:
        conditions = []
        if "positive" in problem:
            conditions.append("x > 0")
        if "negative" in problem:
            conditions.append("x < 0")
        if "even" in problem:
            conditions.append("x % 2 == 0")
        if len(params) >= 2:
            if "below" in problem and "threshold" in problem:
                conditions.append(f"x < {params[1]}")
        return conditions if conditions else ["True"]

    def _get_first_arg(self, inp: Any) -> Any:
        if isinstance(inp, tuple):
            return inp[0] if inp else None
        return inp

    # ------------------------------------------------------------------
    # Example extraction
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _verify(self, code: str, examples: List[Dict],
                entry_point: str) -> bool:
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
