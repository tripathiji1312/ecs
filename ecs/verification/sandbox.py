"""
Sandboxed code execution with:
- Time limits (configurable, default 5s)
- Memory limits (configurable, default 512MB)
- Output truncation (default 10KB)
- Temporary file cleanup
"""

import ast
import os
import json
import subprocess
import tempfile
import resource
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of sandboxed execution."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    execution_time_ms: float = 0.0
    timed_out: bool = False
    crashed: bool = False


class SafeExecutor:
    """Sandboxed code execution with resource limits."""

    def __init__(self,
                 timeout: float = 5.0,
                 memory_limit_mb: int = 512,
                 max_output_chars: int = 10000):
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self.max_output_chars = max_output_chars

    def execute(self, code: str,
                function_name: Optional[str] = None,
                test_inputs: Optional[List] = None) -> ExecutionResult:
        """Execute code in sandbox. Optionally call a function with inputs."""
        if function_name and test_inputs is not None:
            script = (
                f"{code}\n\n"
                f"import json\n"
                f"try:\n"
                f"    result = {function_name}(*{test_inputs!r})\n"
                f"    print(json.dumps({{\"result\": result, \"success\": True}}))\n"
                f"except Exception as e:\n"
                f"    print(json.dumps({{\"error\": str(e), \"success\": False}}))\n"
            )
        else:
            script = code

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as f:
            f.write(script)
            filepath = f.name

        try:
            import time
            start = time.time()

            def limit_resources():
                memory_bytes = self.memory_limit_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

            result = subprocess.run(
                ['python3', filepath],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                preexec_fn=limit_resources,
                env={'PATH': os.environ.get('PATH', '')}
            )

            elapsed = (time.time() - start) * 1000

            stdout = result.stdout[:self.max_output_chars]
            stderr = result.stderr[:self.max_output_chars]

            return ExecutionResult(
                success=result.returncode == 0,
                stdout=stdout,
                stderr=stderr,
                return_code=result.returncode,
                execution_time_ms=elapsed,
                timed_out=False
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stderr=f"Execution timed out after {self.timeout}s",
                timed_out=True
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stderr=f"Execution error: {str(e)}",
                crashed=True
            )
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def execute_tests(self, code: str,
                      test_cases: List[Dict]) -> Tuple[bool, str, List[Dict]]:
        """Execute multiple test cases. Returns (all_passed, summary, details)."""
        results = []

        for i, test in enumerate(test_cases):
            func_name = test.get("function")
            inputs = test.get("inputs", [])
            expected = test.get("expected")

            exec_result = self.execute(code, func_name, inputs)

            test_result = {
                "test_index": i,
                "inputs": inputs,
                "expected": expected,
                "success": exec_result.success,
                "timed_out": exec_result.timed_out,
            }

            if exec_result.success:
                try:
                    last_line = exec_result.stdout.strip().split('\n')[-1]
                    output_data = json.loads(last_line)
                    actual = output_data.get("result")
                    test_result["actual"] = actual
                    test_result["passed"] = (actual == expected)
                except (json.JSONDecodeError, IndexError):
                    test_result["passed"] = False
                    test_result["actual"] = "UNPARSEABLE_OUTPUT"
            else:
                test_result["passed"] = False
                test_result["actual"] = f"ERROR: {exec_result.stderr[:200]}"

            results.append(test_result)

        all_passed = all(r.get("passed", False) for r in results)

        passed_count = sum(1 for r in results if r.get("passed"))
        total_count = len(results)
        summary = f"{passed_count}/{total_count} tests passed"

        if not all_passed:
            failed = [r for r in results if not r.get("passed")]
            for f in failed[:3]:
                summary += (
                    f"\n  Test {f['test_index']}: "
                    f"inputs={f['inputs']}, expected={f['expected']}, "
                    f"got={f.get('actual', 'ERROR')}"
                )

        return all_passed, summary, results

    def verify_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Verify code is syntactically valid Python."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
