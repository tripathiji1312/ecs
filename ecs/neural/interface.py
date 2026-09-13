"""
Neural Interface: Wraps Qwen2.5-Coder-0.5B as a frozen language interface.
The neural model handles ONLY language understanding and generation.
All knowledge comes from HDC memory; all reasoning from strategies.
"""

import time
import requests
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class NeuralResponse:
    """Response from the neural interface."""
    text: str
    code: Optional[str] = None
    confidence: float = 0.5
    tokens_used: int = 0
    latency_ms: float = 0.0


class NeuralInterface:
    """
    Interface to the frozen Qwen2.5-Coder-0.5B model.
    Uses Ollama's API for simplicity.
    """

    SYSTEM_PROMPT = (
        "You are the language interface component of a larger cognitive system.\n\n"
        "Your role is ONLY to:\n"
        "1. Understand natural language descriptions of coding problems\n"
        "2. Generate code when given a clear specification\n"
        "3. Explain code and concepts in natural language\n\n"
        "You do NOT make decisions. You do NOT reason about strategies.\n"
        "The system tells you what to generate; you generate it well.\n\n"
        "When generating code:\n"
        "- Be precise and syntactically correct\n"
        "- Include type hints when possible\n"
        "- Add brief comments for clarity\n"
        "- Handle edge cases mentioned in the specification\n"
    )

    def __init__(self, model_name: str = "qwen2.5-coder:0.5b",
                 ollama_url: str = "http://localhost:11434",
                 backend: str = "ollama"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.backend = backend
        self._hf_model = None
        self._hf_tokenizer = None
        self.conversation_history: List[Dict] = []
        self.total_tokens: int = 0
        self.total_calls: int = 0

    def is_available(self) -> bool:
        """Check if the configured backend is available."""
        if self.backend == "hf":
            try:
                self._init_hf()
                return True
            except Exception:
                return False
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            if resp.status_code != 200:
                return False
            models = resp.json().get("models", [])
            return any(m.get("name", "").startswith(self.model_name.split(":")[0])
                       for m in models)
        except (requests.ConnectionError, requests.Timeout):
            return False

    def _init_hf(self):
        """Lazy-init HuggingFace model + tokenizer."""
        if self._hf_model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None
        )
        if device == "cpu":
            self._hf_model = self._hf_model.to(device)

    def _call_hf(self, prompt: str, system: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 2048) -> Dict[str, Any]:
        """Generate via HuggingFace Transformers. Same return format as _call_ollama."""
        import torch
        self._init_hf()
        messages = []
        if system or self.SYSTEM_PROMPT:
            messages.append({"role": "system", "content": system or self.SYSTEM_PROMPT})
        messages.append({"role": "user", "content": prompt})

        text = self._hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._hf_tokenizer(text, return_tensors="pt").to(self._hf_model.device)

        start = time.time()
        with torch.no_grad():
            output = self._hf_model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
                pad_token_id=self._hf_tokenizer.eos_token_id,
            )
        latency = (time.time() - start) * 1000
        new_tokens = output[0][inputs['input_ids'].shape[1]:]
        content = self._hf_tokenizer.decode(new_tokens, skip_special_tokens=True)
        tokens = len(new_tokens)

        self.total_tokens += tokens
        self.total_calls += 1

        return {"content": content, "tokens": tokens, "latency_ms": latency}

    def _call_backend(self, prompt: str, system: Optional[str] = None,
                      temperature: float = 0.7, max_tokens: int = 2048) -> Dict[str, Any]:
        """Route to the configured backend."""
        if self.backend == "hf":
            return self._call_hf(prompt, system, temperature, max_tokens)
        return self._call_ollama(prompt, system, temperature, max_tokens)

    def _call_ollama(self, prompt: str,
                     system: Optional[str] = None,
                     temperature: float = 0.7,
                     max_tokens: int = 2048) -> Dict[str, Any]:
        """Make API call to Ollama. Returns dict with content and metadata."""
        messages = []

        if system or self.SYSTEM_PROMPT:
            messages.append({
                "role": "system",
                "content": system or self.SYSTEM_PROMPT
            })

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        start = time.time()
        response = requests.post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=60
        )
        latency = (time.time() - start) * 1000

        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error: {response.status_code}")

        data = response.json()
        content = data["message"]["content"]
        tokens = data.get("eval_count", len(content.split()))

        self.total_tokens += tokens
        self.total_calls += 1

        return {
            "content": content,
            "tokens": tokens,
            "latency_ms": latency
        }

    # ═══════════════════════════════════════════
    # CODE GENERATION
    # ═══════════════════════════════════════════

    def generate_code(self, specification: str,
                      strategy_context: Optional[str] = None,
                      retrieved_knowledge: Optional[List[str]] = None) -> NeuralResponse:
        """Generate code from a specification, guided by strategy and memory."""
        prompt_parts = []

        if retrieved_knowledge:
            prompt_parts.append("Relevant knowledge from memory:")
            for i, knowledge in enumerate(retrieved_knowledge, 1):
                prompt_parts.append(f"  {i}. {knowledge}")
            prompt_parts.append("")

        if strategy_context:
            prompt_parts.append(f"Approach to use: {strategy_context}")
            prompt_parts.append("")

        prompt_parts.append("Write Python code for the following:")
        prompt_parts.append(f"Specification: {specification}")
        prompt_parts.append("")
        prompt_parts.append("Provide ONLY the code, wrapped in ```python ... ```")

        prompt = "\n".join(prompt_parts)

        result = self._call_backend(prompt, temperature=0.3)
        code = self._extract_code(result["content"])

        return NeuralResponse(
            text=result["content"],
            code=code,
            confidence=0.7,
            tokens_used=result["tokens"],
            latency_ms=result["latency_ms"]
        )

    def generate_code_multi(self, specification: str, k: int = 16,
                            temperature: float = 0.8,
                            strategy_context: Optional[str] = None,
                            retrieved_knowledge: Optional[List[str]] = None) -> List[NeuralResponse]:
        """Generate K code samples for the same specification."""
        prompt_parts = []
        if retrieved_knowledge:
            prompt_parts.append("Relevant knowledge from memory:")
            for i, knowledge in enumerate(retrieved_knowledge, 1):
                prompt_parts.append(f"  {i}. {knowledge}")
            prompt_parts.append("")
        if strategy_context:
            prompt_parts.append(f"Approach to use: {strategy_context}")
            prompt_parts.append("")
        prompt_parts.append("Write Python code for the following:")
        prompt_parts.append(f"Specification: {specification}")
        prompt_parts.append("")
        prompt_parts.append("Provide ONLY the code, wrapped in ```python ... ```")
        prompt = "\n".join(prompt_parts)

        responses = []
        for _ in range(k):
            result = self._call_backend(prompt, temperature=temperature)
            code = self._extract_code(result["content"])
            responses.append(NeuralResponse(
                text=result["content"], code=code, confidence=0.7,
                tokens_used=result["tokens"], latency_ms=result["latency_ms"]
            ))
        return responses

    def _extract_code(self, text: str) -> Optional[str]:
        """Extract code block from response."""
        if "```python" in text:
            start = text.find("```python") + len("```python")
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        return text.strip()

    # ═══════════════════════════════════════════
    # PROBLEM UNDERSTANDING
    # ═══════════════════════════════════════════

    def understand_problem(self, problem_description: str) -> Dict[str, Any]:
        """Use the neural model to understand and structure a problem."""
        prompt = (
            "Analyze this coding problem and extract its features:\n\n"
            f"Problem: {problem_description}\n\n"
            "Respond in this exact format:\n"
            "TYPE: [sorting/searching/graph/dynamic_programming/"
            "optimization/string/numeric/other]\n"
            "FEATURES: [comma-separated list of features like: "
            "divisible, ordered, overlapping, monotonic, "
            "state_machine, combinatorial, recursion, iteration]\n"
            "DIFFICULTY: [easy/medium/hard]\n"
            "SUB_PROBLEMS: [semicolon-separated list of sub-problems, if any]"
        )

        result = self._call_backend(prompt, temperature=0.1)
        response = result["content"]

        features = []
        problem_type = "other"
        difficulty = "medium"
        sub_problems = []

        for line in response.split("\n"):
            if line.startswith("TYPE:"):
                problem_type = line.split(":", 1)[1].strip().lower()
            elif line.startswith("FEATURES:"):
                features_str = line.split(":", 1)[1].strip()
                features = [f.strip().lower() for f in features_str.split(",")]
            elif line.startswith("DIFFICULTY:"):
                difficulty = line.split(":", 1)[1].strip().lower()
            elif line.startswith("SUB_PROBLEMS:"):
                subs = line.split(":", 1)[1].strip()
                sub_problems = [s.strip() for s in subs.split(";") if s.strip()]

        return {
            "type": problem_type,
            "features": features,
            "difficulty": difficulty,
            "sub_problems": sub_problems,
            "raw_response": response,
            "latency_ms": result["latency_ms"]
        }

    # ═══════════════════════════════════════════
    # EXPLANATION
    # ═══════════════════════════════════════════

    def explain_code(self, code: str) -> NeuralResponse:
        """Explain what code does in natural language."""
        prompt = (
            "Explain what this Python code does, concisely:\n\n"
            f"```python\n{code}\n```\n\n"
            "Explain:\n"
            "1. What it does overall\n"
            "2. The algorithm/technique used\n"
            "3. Time and space complexity"
        )

        result = self._call_backend(prompt, temperature=0.5)

        return NeuralResponse(
            text=result["content"],
            confidence=0.8,
            tokens_used=result["tokens"],
            latency_ms=result["latency_ms"]
        )

    # ═══════════════════════════════════════════
    # TEST GENERATION
    # ═══════════════════════════════════════════

    def generate_test_cases(self, function_signature: str,
                            description: str) -> List[Dict[str, Any]]:
        """Generate test cases for a function."""
        prompt = (
            f"Generate test cases for this function:\n\n"
            f"Function: {function_signature}\n"
            f"Description: {description}\n\n"
            "Provide 5 test cases in this format:\n"
            "INPUT: [input arguments]\n"
            "EXPECTED: [expected output]\n"
            "TYPE: [normal/edge_case/error_case]"
        )

        result = self._call_backend(prompt, temperature=0.5)
        response = result["content"]

        test_cases = []
        current_input = None
        current_expected = None
        current_type = "normal"

        for line in response.split("\n"):
            if line.startswith("INPUT:"):
                current_input = line.split(":", 1)[1].strip()
            elif line.startswith("EXPECTED:"):
                current_expected = line.split(":", 1)[1].strip()
            elif line.startswith("TYPE:"):
                current_type = line.split(":", 1)[1].strip()

                if current_input is not None:
                    test_cases.append({
                        "input": current_input,
                        "expected": current_expected,
                        "type": current_type
                    })
                    current_input = None
                    current_expected = None

        return test_cases

    # ═══════════════════════════════════════════
    # ITERATIVE REFINEMENT
    # ═══════════════════════════════════════════

    def refine_code(self, previous_code: str, error_message: str,
                    original_specification: str) -> NeuralResponse:
        """Iterative refinement: fix code based on error feedback."""
        prompt = (
            f"Fix the bug in this code.\n\n"
            f"Original specification: {original_specification}\n\n"
            f"Previous code:\n```python\n{previous_code}\n```\n\n"
            f"Error encountered:\n{error_message}\n\n"
            "Provide the corrected code, fixing ONLY the bug. "
            "Keep everything else the same."
        )

        result = self._call_backend(prompt, temperature=0.2)
        code = self._extract_code(result["content"])

        return NeuralResponse(
            text=result["content"],
            code=code,
            confidence=0.6,
            tokens_used=result["tokens"],
            latency_ms=result["latency_ms"]
        )

    # ═══════════════════════════════════════════
    # TOKEN BUDGET
    # ═══════════════════════════════════════════

    def estimate_complexity(self, specification: str) -> str:
        """Estimate problem complexity for token budgeting."""
        spec_lower = specification.lower()

        hard_indicators = [
            "optimize", "efficient", "minimum", "maximum", "shortest",
            "longest", "dynamic", "recursive", "graph", "tree"
        ]
        medium_indicators = [
            "sort", "search", "find", "reverse", "merge", "parse"
        ]

        hard_count = sum(1 for ind in hard_indicators if ind in spec_lower)
        medium_count = sum(1 for ind in medium_indicators if ind in spec_lower)

        if hard_count >= 2:
            return "hard"
        elif hard_count >= 1 or medium_count >= 2:
            return "medium"
        return "easy"

    def get_token_budget(self, complexity: str) -> int:
        """Get appropriate token budget for complexity level."""
        budgets = {"easy": 512, "medium": 1024, "hard": 2048}
        return budgets.get(complexity, 1024)

    # ═══════════════════════════════════════════
    # STATISTICS
    # ═══════════════════════════════════════════

    def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics."""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "model": self.model_name,
        }
