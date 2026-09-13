"""
Signature Adapter: Bridges ECS's generic code output to HumanEval's
exact function signatures.

The gap: ECS generates correct algorithms but with different function
names, parameter names, and sometimes different structure. HumanEval
tests call the exact function name with exact parameters.

Strategy:
1. Parse expected signature from HumanEval prompt
2. Parse generated code's signature
3. Rename function + parameters to match
4. Handle common structural mismatches (wrapper vs inline, extra helpers)
"""

import ast
import re
from typing import Dict, List, Optional, Tuple


class SignatureAdapter:

    def __init__(self):
        self.adaptation_count = 0
        self.success_count = 0

    def extract_expected_signature(self, prompt: str) -> Optional[Dict]:
        """Extract function signature from HumanEval prompt."""
        match = re.search(
            r'def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*([^:]+))?:',
            prompt
        )
        if not match:
            return None

        func_name = match.group(1)
        params_str = match.group(2)
        return_type = match.group(3).strip() if match.group(3) else None

        params = []
        if params_str.strip():
            for part in self._split_params(params_str):
                part = part.strip()
                if not part:
                    continue
                name_match = re.match(r'(\w+)', part)
                if name_match:
                    params.append({"name": name_match.group(1)})

        return {
            "function_name": func_name,
            "parameters": params,
            "return_type": return_type,
        }

    def _split_params(self, params_str: str) -> List[str]:
        """Split parameter string handling nested brackets."""
        parts = []
        depth = 0
        current = ""
        for ch in params_str:
            if ch in "([{":
                depth += 1
                current += ch
            elif ch in ")]}":
                depth -= 1
                current += ch
            elif ch == "," and depth == 0:
                parts.append(current)
                current = ""
            else:
                current += ch
        if current.strip():
            parts.append(current)
        return parts

    def extract_generated_signature(self, code: str) -> Optional[Dict]:
        """Extract signature from ECS-generated code."""
        if not code:
            return None
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    params = []
                    for arg in node.args.args:
                        params.append({"name": arg.arg})
                    return {
                        "function_name": node.name,
                        "parameters": params,
                    }
        except SyntaxError:
            pass
        return None

    def adapt_code(self, generated_code: str,
                   humaneval_prompt: str) -> Tuple[str, Dict]:
        """
        Full adaptation: rename function and parameters to match
        HumanEval's expected signature.

        Returns (adapted_code, info_dict).
        """
        if not generated_code:
            return "", {"adapted": False, "reason": "no_code"}

        expected = self.extract_expected_signature(humaneval_prompt)
        if not expected:
            return generated_code, {"adapted": False, "reason": "no_expected_sig"}

        generated_sig = self.extract_generated_signature(generated_code)
        if not generated_sig:
            return generated_code, {"adapted": False, "reason": "no_gen_sig"}

        self.adaptation_count += 1
        adapted = generated_code

        # Step 1: Rename function
        old_name = generated_sig["function_name"]
        new_name = expected["function_name"]
        if old_name != new_name:
            adapted = self._rename_function(adapted, old_name, new_name)

        # Step 2: Rename parameters (positional mapping)
        gen_params = [p["name"] for p in generated_sig["parameters"]]
        exp_params = [p["name"] for p in expected["parameters"]]

        if len(gen_params) == len(exp_params):
            adapted = self._rename_params(adapted, gen_params, exp_params)
        elif len(gen_params) > len(exp_params):
            # Generated has extra params — try mapping first N
            adapted = self._rename_params(
                adapted, gen_params[:len(exp_params)], exp_params
            )

        # Step 3: If code is just the body (no def), wrap it
        if f"def {new_name}" not in adapted:
            adapted = self._wrap_as_function(adapted, expected)

        self.success_count += 1
        return adapted, {
            "adapted": True,
            "old_name": old_name,
            "new_name": new_name,
            "param_mapping": dict(zip(gen_params[:len(exp_params)], exp_params)),
        }

    def _rename_function(self, code: str, old_name: str,
                         new_name: str) -> str:
        """Rename function throughout code (def + recursive calls)."""
        # Use word boundary to avoid partial replacements
        pattern = r'\b' + re.escape(old_name) + r'\b'
        return re.sub(pattern, new_name, code)

    def _rename_params(self, code: str, old_params: List[str],
                       new_params: List[str]) -> str:
        """Rename parameters using temporary placeholders to avoid collisions."""
        if old_params == new_params:
            return code

        # Phase 1: old → temp
        temps = [f"__ecs_param_{i}__" for i in range(len(old_params))]
        for old, temp in zip(old_params, temps):
            if old == new_params[old_params.index(old)]:
                continue
            pattern = r'\b' + re.escape(old) + r'\b'
            code = re.sub(pattern, temp, code)

        # Phase 2: temp → new
        for temp, new in zip(temps, new_params):
            code = code.replace(temp, new)

        return code

    def _wrap_as_function(self, code: str, expected: Dict) -> str:
        """Wrap bare code in the expected function signature."""
        params = ", ".join(p["name"] for p in expected["parameters"])
        ret_type = f" -> {expected['return_type']}" if expected.get("return_type") else ""
        header = f"def {expected['function_name']}({params}){ret_type}:\n"

        # Indent the code
        indented = "\n".join(f"    {line}" for line in code.split("\n"))
        return header + indented

    def create_adapted_submission(self, generated_code: str,
                                  humaneval_prompt: str) -> str:
        """
        Create a complete submission: the function body that completes
        the HumanEval prompt.

        HumanEval format: prompt contains the function header + docstring.
        We need to return JUST the function body (indented).
        """
        if not generated_code:
            return "    pass\n"

        adapted, info = self.adapt_code(generated_code, humaneval_prompt)

        if not info.get("adapted"):
            # Couldn't adapt — try to use code as function body
            return self._extract_body_or_indent(generated_code)

        # Extract just the function body from the adapted code
        return self._extract_function_body(adapted)

    def _extract_function_body(self, code: str) -> str:
        """Extract the body of the first function definition."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Get source lines of the body
                    lines = code.split("\n")
                    # Find where body starts (after def line)
                    body_start = node.body[0].lineno - 1
                    body_lines = lines[body_start:]
                    return "\n".join(body_lines) + "\n"
        except SyntaxError:
            pass
        return self._extract_body_or_indent(code)

    def _extract_body_or_indent(self, code: str) -> str:
        """If code isn't a function, indent it as a body."""
        lines = code.strip().split("\n")

        # Skip the def line if present
        start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                start = i + 1
                break

        if start > 0:
            body_lines = lines[start:]
        else:
            body_lines = ["    " + line for line in lines]

        return "\n".join(body_lines) + "\n"
