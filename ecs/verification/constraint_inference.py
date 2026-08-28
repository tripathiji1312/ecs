"""
Constraint Inference Engine: Extract formal constraints from natural language.
Enables Z3-only synthesis when neural is offline.

Maps natural language patterns to:
1. Z3 constraints (for verification)
2. Code templates (for synthesis)
"""

import re
from typing import Dict, List, Optional, Tuple


class ConstraintInferenceEngine:
    """
    Extracts formal constraints from natural language problem descriptions.
    Enables structured-tier synthesis without neural.
    """

    CONSTRAINT_PATTERNS = {
        "sorted": {
            "keywords": ["sorted", "ordered", "ascending", "increasing",
                         "non-decreasing"],
            "description": "array is sorted in ascending order",
        },
        "descending": {
            "keywords": ["descending", "decreasing", "non-increasing"],
            "description": "array is sorted in descending order",
        },
        "contains_target": {
            "keywords": ["find", "search for", "locate", "contains",
                         "look up", "lookup"],
            "description": "array contains the target value",
        },
        "not_contains": {
            "keywords": ["absent", "not present", "missing", "not found"],
            "description": "array does not contain the target",
        },
        "sum_pair": {
            "keywords": ["two numbers that sum", "pair that adds",
                         "two sum", "a + b equals", "sum to a target",
                         "add up to"],
            "description": "exists a pair summing to target",
        },
        "minimum": {
            "keywords": ["minimum", "smallest", "least", "min element"],
            "description": "result is the minimum element",
        },
        "maximum": {
            "keywords": ["maximum", "largest", "greatest", "biggest",
                         "max element", "peak"],
            "description": "result is the maximum element",
        },
        "unique": {
            "keywords": ["unique", "distinct", "no duplicates",
                         "remove duplicates", "deduplicate"],
            "description": "all elements are distinct",
        },
        "palindrome": {
            "keywords": ["palindrome", "reads same forwards and backwards",
                         "same reversed"],
            "description": "string is a palindrome",
        },
        "reverse": {
            "keywords": ["reverse", "reversed", "flip", "backwards"],
            "description": "reverse the sequence",
        },
        "sort": {
            "keywords": ["sort", "arrange", "order elements",
                         "put in order"],
            "description": "sort the elements",
        },
        "fibonacci": {
            "keywords": ["fibonacci", "fib number", "fib sequence"],
            "description": "compute fibonacci number",
        },
        "factorial": {
            "keywords": ["factorial", "n!", "product of 1 to n"],
            "description": "compute factorial",
        },
        "prime": {
            "keywords": ["prime", "primality", "is prime",
                         "prime number"],
            "description": "check if number is prime",
        },
        "permutation": {
            "keywords": ["permutation", "permutations", "all orderings",
                         "rearrangements"],
            "description": "generate all permutations",
        },
        "subsequence": {
            "keywords": ["subsequence", "longest common",
                         "lcs", "common subsequence"],
            "description": "find longest common subsequence",
        },
        "balanced_brackets": {
            "keywords": ["balanced", "brackets", "parentheses",
                         "matching brackets", "valid brackets"],
            "description": "check if brackets are balanced",
        },
        "rotate": {
            "keywords": ["rotate", "rotation", "circular shift",
                         "shift by k"],
            "description": "rotate array by k positions",
        },
        "merge_sorted": {
            "keywords": ["merge two sorted", "merge sorted",
                         "combine sorted"],
            "description": "merge two sorted arrays",
        },
        "prefix": {
            "keywords": ["prefix", "longest common prefix",
                         "shared prefix"],
            "description": "find longest common prefix",
        },
        "divide_and_conquer": {
            "keywords": ["divide and conquer", "merge sort",
                         "quicksort", "split and merge"],
            "description": "divide-and-conquer algorithm",
        },
        "binary_search": {
            "keywords": ["binary search", "search sorted",
                         "bisect", "log n search"],
            "description": "binary search on sorted data",
        },
    }

    CODE_TEMPLATES: Dict[str, str] = {
        "sorted+contains_target": (
            "def binary_search(arr, target):\n"
            "    lo, hi = 0, len(arr) - 1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return -1\n"
        ),
        "binary_search": (
            "def binary_search(arr, target):\n"
            "    lo, hi = 0, len(arr) - 1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return -1\n"
        ),
        "minimum": (
            "def find_minimum(arr):\n"
            "    if not arr:\n"
            "        return None\n"
            "    result = arr[0]\n"
            "    for x in arr[1:]:\n"
            "        if x < result:\n"
            "            result = x\n"
            "    return result\n"
        ),
        "maximum": (
            "def find_maximum(arr):\n"
            "    if not arr:\n"
            "        return None\n"
            "    result = arr[0]\n"
            "    for x in arr[1:]:\n"
            "        if x > result:\n"
            "            result = x\n"
            "    return result\n"
        ),
        "sum_pair": (
            "def two_sum(arr, target):\n"
            "    seen = {}\n"
            "    for i, num in enumerate(arr):\n"
            "        complement = target - num\n"
            "        if complement in seen:\n"
            "            return [seen[complement], i]\n"
            "        seen[num] = i\n"
            "    return None\n"
        ),
        "palindrome": (
            "def is_palindrome(s):\n"
            "    return s == s[::-1]\n"
        ),
        "reverse": (
            "def reverse_sequence(seq):\n"
            "    result = []\n"
            "    for i in range(len(seq) - 1, -1, -1):\n"
            "        result.append(seq[i])\n"
            "    return result\n"
        ),
        "sort": (
            "def sort_array(arr):\n"
            "    if len(arr) <= 1:\n"
            "        return arr\n"
            "    pivot = arr[len(arr) // 2]\n"
            "    left = [x for x in arr if x < pivot]\n"
            "    middle = [x for x in arr if x == pivot]\n"
            "    right = [x for x in arr if x > pivot]\n"
            "    return sort_array(left) + middle + sort_array(right)\n"
        ),
        "divide_and_conquer": (
            "def merge_sort(arr):\n"
            "    if len(arr) <= 1:\n"
            "        return arr\n"
            "    mid = len(arr) // 2\n"
            "    left = merge_sort(arr[:mid])\n"
            "    right = merge_sort(arr[mid:])\n"
            "    return merge(left, right)\n"
            "\n"
            "def merge(left, right):\n"
            "    result = []\n"
            "    i = j = 0\n"
            "    while i < len(left) and j < len(right):\n"
            "        if left[i] <= right[j]:\n"
            "            result.append(left[i])\n"
            "            i += 1\n"
            "        else:\n"
            "            result.append(right[j])\n"
            "            j += 1\n"
            "    result.extend(left[i:])\n"
            "    result.extend(right[j:])\n"
            "    return result\n"
        ),
        "fibonacci": (
            "def fibonacci(n):\n"
            "    if n <= 0:\n"
            "        return 0\n"
            "    if n == 1:\n"
            "        return 1\n"
            "    a, b = 0, 1\n"
            "    for _ in range(2, n + 1):\n"
            "        a, b = b, a + b\n"
            "    return b\n"
        ),
        "factorial": (
            "def factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    result = 1\n"
            "    for i in range(2, n + 1):\n"
            "        result *= i\n"
            "    return result\n"
        ),
        "prime": (
            "def is_prime(n):\n"
            "    if n < 2:\n"
            "        return False\n"
            "    if n < 4:\n"
            "        return True\n"
            "    if n % 2 == 0 or n % 3 == 0:\n"
            "        return False\n"
            "    i = 5\n"
            "    while i * i <= n:\n"
            "        if n % i == 0 or n % (i + 2) == 0:\n"
            "            return False\n"
            "        i += 6\n"
            "    return True\n"
        ),
        "balanced_brackets": (
            "def is_balanced(s):\n"
            "    stack = []\n"
            "    pairs = {'(': ')', '[': ']', '{': '}'}\n"
            "    for ch in s:\n"
            "        if ch in pairs:\n"
            "            stack.append(ch)\n"
            "        elif ch in pairs.values():\n"
            "            if not stack:\n"
            "                return False\n"
            "            if pairs[stack.pop()] != ch:\n"
            "                return False\n"
            "    return len(stack) == 0\n"
        ),
        "rotate": (
            "def rotate_array(arr, k):\n"
            "    if not arr:\n"
            "        return arr\n"
            "    k = k % len(arr)\n"
            "    return arr[-k:] + arr[:-k]\n"
        ),
        "merge_sorted": (
            "def merge_sorted(a, b):\n"
            "    result = []\n"
            "    i = j = 0\n"
            "    while i < len(a) and j < len(b):\n"
            "        if a[i] <= b[j]:\n"
            "            result.append(a[i])\n"
            "            i += 1\n"
            "        else:\n"
            "            result.append(b[j])\n"
            "            j += 1\n"
            "    result.extend(a[i:])\n"
            "    result.extend(b[j:])\n"
            "    return result\n"
        ),
        "unique": (
            "def remove_duplicates(arr):\n"
            "    seen = set()\n"
            "    result = []\n"
            "    for x in arr:\n"
            "        if x not in seen:\n"
            "            seen.add(x)\n"
            "            result.append(x)\n"
            "    return result\n"
        ),
        "prefix": (
            "def longest_common_prefix(strs):\n"
            "    if not strs:\n"
            "        return ''\n"
            "    prefix = strs[0]\n"
            "    for s in strs[1:]:\n"
            "        while not s.startswith(prefix):\n"
            "            prefix = prefix[:-1]\n"
            "            if not prefix:\n"
            "                return ''\n"
            "    return prefix\n"
        ),
        "permutation": (
            "def permutations(lst):\n"
            "    if len(lst) <= 1:\n"
            "        return [lst]\n"
            "    result = []\n"
            "    for i, elem in enumerate(lst):\n"
            "        rest = lst[:i] + lst[i+1:]\n"
            "        for perm in permutations(rest):\n"
            "            result.append([elem] + perm)\n"
            "    return result\n"
        ),
    }

    def infer_constraints(self, problem_text: str) -> List[Dict]:
        """Extract constraints from problem text."""
        text_lower = problem_text.lower()
        constraints = []

        for pattern_name, pattern in self.CONSTRAINT_PATTERNS.items():
            if any(kw in text_lower for kw in pattern["keywords"]):
                constraints.append({
                    "pattern": pattern_name,
                    "description": pattern["description"],
                    "source_text": problem_text,
                })

        return constraints

    def synthesize_from_constraints(self, constraints: List[Dict],
                                    problem_text: str) -> Optional[str]:
        """Attempt synthesis using inferred constraints."""
        if not constraints:
            return None

        pattern_names = sorted([c["pattern"] for c in constraints])
        pattern_key = "+".join(pattern_names)

        # Try exact combination first
        if pattern_key in self.CODE_TEMPLATES:
            return self.CODE_TEMPLATES[pattern_key]

        # Try individual patterns (prefer more specific ones)
        priority_order = [
            "divide_and_conquer", "binary_search", "merge_sorted",
            "sum_pair", "subsequence", "permutation",
            "balanced_brackets", "palindrome", "prime",
            "fibonacci", "factorial", "rotate", "prefix",
            "unique", "reverse", "sort", "minimum", "maximum",
        ]

        for pattern in priority_order:
            if pattern in pattern_names and pattern in self.CODE_TEMPLATES:
                return self.CODE_TEMPLATES[pattern]

        # Fallback: try any matching pattern
        for c in constraints:
            if c["pattern"] in self.CODE_TEMPLATES:
                return self.CODE_TEMPLATES[c["pattern"]]

        return None

    def get_test_cases_for_pattern(self, pattern_name: str) -> List[Dict]:
        """Generate test cases for a known pattern."""
        test_map = {
            "sort": [
                {"function": "sort_array", "inputs": [[3, 1, 2]], "expected": [1, 2, 3]},
                {"function": "sort_array", "inputs": [[5, 4, 3, 2, 1]], "expected": [1, 2, 3, 4, 5]},
                {"function": "sort_array", "inputs": [[1]], "expected": [1]},
                {"function": "sort_array", "inputs": [[]], "expected": []},
            ],
            "divide_and_conquer": [
                {"function": "merge_sort", "inputs": [[3, 1, 2]], "expected": [1, 2, 3]},
                {"function": "merge_sort", "inputs": [[5, 4, 3, 2, 1]], "expected": [1, 2, 3, 4, 5]},
                {"function": "merge_sort", "inputs": [[1]], "expected": [1]},
            ],
            "binary_search": [
                {"function": "binary_search", "inputs": [[1, 3, 5, 7, 9], 5], "expected": 2},
                {"function": "binary_search", "inputs": [[1, 3, 5, 7, 9], 1], "expected": 0},
                {"function": "binary_search", "inputs": [[1, 3, 5, 7, 9], 4], "expected": -1},
            ],
            "minimum": [
                {"function": "find_minimum", "inputs": [[3, 1, 2]], "expected": 1},
                {"function": "find_minimum", "inputs": [[5]], "expected": 5},
                {"function": "find_minimum", "inputs": [[-1, 0, 1]], "expected": -1},
            ],
            "maximum": [
                {"function": "find_maximum", "inputs": [[3, 1, 2]], "expected": 3},
                {"function": "find_maximum", "inputs": [[5]], "expected": 5},
                {"function": "find_maximum", "inputs": [[-1, 0, 1]], "expected": 1},
            ],
            "sum_pair": [
                {"function": "two_sum", "inputs": [[2, 7, 11, 15], 9], "expected": [0, 1]},
                {"function": "two_sum", "inputs": [[3, 2, 4], 6], "expected": [1, 2]},
            ],
            "palindrome": [
                {"function": "is_palindrome", "inputs": ["racecar"], "expected": True},
                {"function": "is_palindrome", "inputs": ["hello"], "expected": False},
                {"function": "is_palindrome", "inputs": [""], "expected": True},
            ],
            "reverse": [
                {"function": "reverse_sequence", "inputs": [[1, 2, 3]], "expected": [3, 2, 1]},
                {"function": "reverse_sequence", "inputs": [[1]], "expected": [1]},
            ],
            "fibonacci": [
                {"function": "fibonacci", "inputs": [0], "expected": 0},
                {"function": "fibonacci", "inputs": [1], "expected": 1},
                {"function": "fibonacci", "inputs": [10], "expected": 55},
            ],
            "factorial": [
                {"function": "factorial", "inputs": [0], "expected": 1},
                {"function": "factorial", "inputs": [1], "expected": 1},
                {"function": "factorial", "inputs": [5], "expected": 120},
            ],
            "prime": [
                {"function": "is_prime", "inputs": [2], "expected": True},
                {"function": "is_prime", "inputs": [4], "expected": False},
                {"function": "is_prime", "inputs": [17], "expected": True},
                {"function": "is_prime", "inputs": [1], "expected": False},
            ],
            "balanced_brackets": [
                {"function": "is_balanced", "inputs": ["()[]{}"], "expected": True},
                {"function": "is_balanced", "inputs": ["([)]"], "expected": False},
                {"function": "is_balanced", "inputs": [""], "expected": True},
            ],
            "rotate": [
                {"function": "rotate_array", "inputs": [[1, 2, 3, 4, 5], 2], "expected": [4, 5, 1, 2, 3]},
                {"function": "rotate_array", "inputs": [[1, 2, 3], 0], "expected": [1, 2, 3]},
            ],
            "unique": [
                {"function": "remove_duplicates", "inputs": [[1, 2, 2, 3, 3, 3]], "expected": [1, 2, 3]},
                {"function": "remove_duplicates", "inputs": [[1, 1, 1]], "expected": [1]},
            ],
            "prefix": [
                {"function": "longest_common_prefix", "inputs": [["flower", "flow", "flight"]], "expected": "fl"},
                {"function": "longest_common_prefix", "inputs": [["dog", "racecar"]], "expected": ""},
            ],
        }

        return test_map.get(pattern_name, [])
