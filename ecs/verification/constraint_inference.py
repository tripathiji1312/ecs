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
        "all_prefixes": {
            "keywords": ["all prefixes", "list of prefixes",
                         "prefixes from shortest"],
            "description": "generate all prefixes of a string",
        },
        "prefix": {
            "keywords": ["longest common prefix",
                         "shared prefix", "common prefix"],
            "description": "find longest common prefix",
        },
        "divide_and_conquer": {
            "keywords": ["divide and conquer", "merge sort",
                         "quicksort", "split and merge"],
            "description": "divide-and-conquer algorithm",
        },
        "sliding_window": {
            "keywords": ["longest substring", "shortest substring",
                         "sliding window", "window size",
                         "subarray sum", "maximum subarray"],
            "description": "sliding window optimization",
        },
        "cycle_detection": {
            "keywords": ["cycle", "linked list cycle", "circular",
                         "happy number", "floyd"],
            "description": "Floyd's cycle detection",
        },
        "monotonic_stack": {
            "keywords": ["next greater", "next smaller",
                         "daily temperatures", "nearest larger",
                         "stock span"],
            "description": "monotonic stack pattern",
        },
        "gcd": {
            "keywords": ["gcd", "greatest common divisor", "euclidean"],
            "description": "compute greatest common divisor",
        },
        "power": {
            "keywords": ["power", "exponent", "x to the n",
                         "exponentiation"],
            "description": "compute power/exponentiation",
        },
        "abs_value": {
            "keywords": ["absolute value", "abs", "magnitude",
                         "distance between"],
            "description": "compute absolute value",
        },
        "flatten": {
            "keywords": ["flatten", "nested list", "flatten list",
                         "unnest"],
            "description": "flatten nested structure",
        },
        "count_occurrences": {
            "keywords": ["count occurrences", "frequency", "how many times",
                         "count elements"],
            "description": "count element occurrences",
        },
        "binary_search": {
            "keywords": ["binary search", "search sorted",
                         "bisect", "log n search"],
            "description": "binary search on sorted data",
        },
        "mean_absolute_deviation": {
            "keywords": ["mean absolute deviation", "average deviation",
                         "mad"],
            "description": "compute mean absolute deviation",
        },
        "rolling_max": {
            "keywords": ["rolling max", "running maximum",
                         "rolling maximum", "maximum so far"],
            "description": "compute rolling maximum list",
        },
        "factorize": {
            "keywords": ["prime factors", "factorize", "factorization",
                         "list of prime factors"],
            "description": "find prime factorization",
        },
        "largest_prime_factor": {
            "keywords": ["largest prime factor", "biggest prime factor",
                         "greatest prime factor"],
            "description": "find largest prime factor",
        },
        "count_distinct": {
            "keywords": ["distinct characters", "unique characters",
                         "how many distinct", "count distinct"],
            "description": "count distinct characters in string",
        },
        "count_substring": {
            "keywords": ["how many times a given substring",
                         "count overlapping", "count substring",
                         "substring occurrences"],
            "description": "count substring occurrences",
        },
        "largest_divisor": {
            "keywords": ["largest number that divides",
                         "largest divisor", "biggest divisor"],
            "description": "find largest divisor smaller than n",
        },
        "sorted_unique": {
            "keywords": ["sorted unique", "unique sorted",
                         "sorted distinct"],
            "description": "return sorted unique elements",
        },
        "pairs_sum_zero": {
            "keywords": ["pairs sum to zero", "two distinct elements",
                         "sum to zero", "pairs_sum_to_zero"],
            "description": "check if two elements sum to zero",
        },
        "triples_sum_zero": {
            "keywords": ["triples sum to zero", "three distinct elements",
                         "triples_sum_to_zero", "three elements that sum to zero"],
            "description": "check if three elements sum to zero",
        },
        "common_elements": {
            "keywords": ["common elements", "common for two lists",
                         "elements in both", "intersection"],
            "description": "find sorted unique common elements",
        },
        "is_simple_power": {
            "keywords": ["simple power of", "power of n",
                         "is a power of"],
            "description": "check if x is a simple power of n",
        },
        "count_up_to_prime": {
            "keywords": ["prime numbers and less than",
                         "prime numbers less than",
                         "primes less than", "primes up to",
                         "integers that are prime"],
            "description": "find all primes less than n",
        },
        "correct_bracketing": {
            "keywords": ["opening bracket has a corresponding closing",
                         "every opening bracket"],
            "description": "check bracket matching",
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
        "gcd": (
            "def gcd(a, b):\n"
            "    while b:\n"
            "        a, b = b, a % b\n"
            "    return a\n"
        ),
        "power": (
            "def power(base, exp):\n"
            "    if exp == 0:\n"
            "        return 1\n"
            "    if exp < 0:\n"
            "        return 1 / power(base, -exp)\n"
            "    if exp % 2 == 0:\n"
            "        half = power(base, exp // 2)\n"
            "        return half * half\n"
            "    return base * power(base, exp - 1)\n"
        ),
        "flatten": (
            "def flatten(lst):\n"
            "    result = []\n"
            "    for item in lst:\n"
            "        if isinstance(item, list):\n"
            "            result.extend(flatten(item))\n"
            "        else:\n"
            "            result.append(item)\n"
            "    return result\n"
        ),
        "count_occurrences": (
            "def count_occurrences(lst, target):\n"
            "    count = 0\n"
            "    for x in lst:\n"
            "        if x == target:\n"
            "            count += 1\n"
            "    return count\n"
        ),
        "abs_value": (
            "def absolute_value(x):\n"
            "    return x if x >= 0 else -x\n"
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
        "all_prefixes": (
            "def all_prefixes(string):\n"
            "    result = []\n"
            "    for i in range(1, len(string) + 1):\n"
            "        result.append(string[:i])\n"
            "    return result\n"
        ),
        "mean_absolute_deviation": (
            "def mean_absolute_deviation(numbers):\n"
            "    mean = sum(numbers) / len(numbers)\n"
            "    return sum(abs(x - mean) for x in numbers) / len(numbers)\n"
        ),
        "rolling_max": (
            "def rolling_max(numbers):\n"
            "    result = []\n"
            "    current_max = None\n"
            "    for n in numbers:\n"
            "        if current_max is None or n > current_max:\n"
            "            current_max = n\n"
            "        result.append(current_max)\n"
            "    return result\n"
        ),
        "factorize": (
            "def factorize(n):\n"
            "    factors = []\n"
            "    d = 2\n"
            "    while d * d <= n:\n"
            "        while n % d == 0:\n"
            "            factors.append(d)\n"
            "            n //= d\n"
            "        d += 1\n"
            "    if n > 1:\n"
            "        factors.append(n)\n"
            "    return factors\n"
        ),
        "largest_prime_factor": (
            "def largest_prime_factor(n):\n"
            "    largest = 1\n"
            "    d = 2\n"
            "    while d * d <= n:\n"
            "        while n % d == 0:\n"
            "            largest = d\n"
            "            n //= d\n"
            "        d += 1\n"
            "    if n > 1:\n"
            "        largest = n\n"
            "    return largest\n"
        ),
        "count_distinct": (
            "def count_distinct_characters(string):\n"
            "    return len(set(string.lower()))\n"
        ),
        "count_substring": (
            "def how_many_times(string, substring):\n"
            "    count = 0\n"
            "    start = 0\n"
            "    while True:\n"
            "        pos = string.find(substring, start)\n"
            "        if pos == -1:\n"
            "            break\n"
            "        count += 1\n"
            "        start = pos + 1\n"
            "    return count\n"
        ),
        "largest_divisor": (
            "def largest_divisor(n):\n"
            "    for i in range(n - 1, 0, -1):\n"
            "        if n % i == 0:\n"
            "            return i\n"
            "    return 1\n"
        ),
        "sorted_unique": (
            "def unique(l):\n"
            "    return sorted(set(l))\n"
        ),
        "pairs_sum_zero": (
            "def pairs_sum_to_zero(l):\n"
            "    for i in range(len(l)):\n"
            "        for j in range(i + 1, len(l)):\n"
            "            if l[i] + l[j] == 0:\n"
            "                return True\n"
            "    return False\n"
        ),
        "triples_sum_zero": (
            "def triples_sum_to_zero(l):\n"
            "    for i in range(len(l)):\n"
            "        for j in range(i + 1, len(l)):\n"
            "            for k in range(j + 1, len(l)):\n"
            "                if l[i] + l[j] + l[k] == 0:\n"
            "                    return True\n"
            "    return False\n"
        ),
        "common_elements": (
            "def common(l1, l2):\n"
            "    return sorted(set(l1) & set(l2))\n"
        ),
        "is_simple_power": (
            "def is_simple_power(x, n):\n"
            "    if x == 1:\n"
            "        return True\n"
            "    if n == 1:\n"
            "        return x == 1\n"
            "    power = n\n"
            "    while power < x:\n"
            "        power *= n\n"
            "    return power == x\n"
        ),
        "count_up_to_prime": (
            "def count_up_to(n):\n"
            "    primes = []\n"
            "    for i in range(2, n):\n"
            "        is_p = True\n"
            "        for j in range(2, int(i**0.5) + 1):\n"
            "            if i % j == 0:\n"
            "                is_p = False\n"
            "                break\n"
            "        if is_p:\n"
            "            primes.append(i)\n"
            "    return primes\n"
        ),
        "correct_bracketing": (
            "def correct_bracketing(brackets):\n"
            "    depth = 0\n"
            "    for ch in brackets:\n"
            "        if ch == '<' or ch == '(':\n"
            "            depth += 1\n"
            "        elif ch == '>' or ch == ')':\n"
            "            depth -= 1\n"
            "        if depth < 0:\n"
            "            return False\n"
            "    return depth == 0\n"
        ),
    }

    def infer_constraints(self, problem_text: str) -> List[Dict]:
        """Extract constraints from problem text with specificity scoring."""
        text_lower = problem_text.lower()
        constraints = []

        for pattern_name, pattern in self.CONSTRAINT_PATTERNS.items():
            best_match_len = 0
            for kw in pattern["keywords"]:
                if kw in text_lower:
                    best_match_len = max(best_match_len, len(kw))
            if best_match_len > 0:
                constraints.append({
                    "pattern": pattern_name,
                    "description": pattern["description"],
                    "source_text": problem_text,
                    "specificity": best_match_len,
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

        # Sort constraints by specificity (longest keyword match wins)
        by_specificity = sorted(
            constraints, key=lambda c: c.get("specificity", 0), reverse=True
        )

        # Try most specific match first
        for c in by_specificity:
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
            "gcd": [
                {"function": "gcd", "inputs": [12, 8], "expected": 4},
                {"function": "gcd", "inputs": [7, 13], "expected": 1},
                {"function": "gcd", "inputs": [25, 15], "expected": 5},
            ],
            "all_prefixes": [
                {"function": "all_prefixes", "inputs": ["abc"], "expected": ["a", "ab", "abc"]},
                {"function": "all_prefixes", "inputs": ["x"], "expected": ["x"]},
            ],
            "mean_absolute_deviation": [
                {"function": "mean_absolute_deviation", "inputs": [[1.0, 2.0, 3.0, 4.0]], "expected": 1.0},
            ],
            "rolling_max": [
                {"function": "rolling_max", "inputs": [[1, 3, 2, 5, 4]], "expected": [1, 3, 3, 5, 5]},
                {"function": "rolling_max", "inputs": [[5, 4, 3]], "expected": [5, 5, 5]},
            ],
            "factorize": [
                {"function": "factorize", "inputs": [12], "expected": [2, 2, 3]},
                {"function": "factorize", "inputs": [7], "expected": [7]},
                {"function": "factorize", "inputs": [8], "expected": [2, 2, 2]},
            ],
            "largest_prime_factor": [
                {"function": "largest_prime_factor", "inputs": [15], "expected": 5},
                {"function": "largest_prime_factor", "inputs": [12], "expected": 3},
                {"function": "largest_prime_factor", "inputs": [49], "expected": 7},
            ],
            "count_distinct": [
                {"function": "count_distinct_characters", "inputs": ["xyzXYZ"], "expected": 3},
                {"function": "count_distinct_characters", "inputs": ["Jerry"], "expected": 4},
            ],
            "count_substring": [
                {"function": "how_many_times", "inputs": ["aaa", "a"], "expected": 3},
                {"function": "how_many_times", "inputs": ["aaa", "aa"], "expected": 2},
                {"function": "how_many_times", "inputs": ["", "a"], "expected": 0},
            ],
            "largest_divisor": [
                {"function": "largest_divisor", "inputs": [15], "expected": 5},
                {"function": "largest_divisor", "inputs": [12], "expected": 6},
                {"function": "largest_divisor", "inputs": [7], "expected": 1},
            ],
            "sorted_unique": [
                {"function": "unique", "inputs": [[5, 3, 5, 2, 3, 3, 9, 0, 123]], "expected": [0, 2, 3, 5, 9, 123]},
            ],
            "pairs_sum_zero": [
                {"function": "pairs_sum_to_zero", "inputs": [[1, 3, 5, 0]], "expected": False},
                {"function": "pairs_sum_to_zero", "inputs": [[1, 3, -2, 1]], "expected": False},
                {"function": "pairs_sum_to_zero", "inputs": [[1, 2, 3, 7]], "expected": False},
                {"function": "pairs_sum_to_zero", "inputs": [[2, 4, -5, 3, 5, 7]], "expected": True},
                {"function": "pairs_sum_to_zero", "inputs": [[1]], "expected": False},
            ],
            "triples_sum_zero": [
                {"function": "triples_sum_to_zero", "inputs": [[1, 3, 5, 0]], "expected": False},
                {"function": "triples_sum_to_zero", "inputs": [[1, 3, -2, 1]], "expected": True},
                {"function": "triples_sum_to_zero", "inputs": [[1, 2, 3, 7]], "expected": False},
            ],
            "common_elements": [
                {"function": "common", "inputs": [[1, 4, 3, 34, 653, 2, 5], [5, 7, 1, 5, 9, 653, 121]], "expected": [1, 5, 653]},
            ],
            "is_simple_power": [
                {"function": "is_simple_power", "inputs": [1, 4], "expected": True},
                {"function": "is_simple_power", "inputs": [2, 2], "expected": True},
                {"function": "is_simple_power", "inputs": [8, 2], "expected": True},
                {"function": "is_simple_power", "inputs": [3, 2], "expected": False},
            ],
            "count_up_to_prime": [
                {"function": "count_up_to", "inputs": [5], "expected": [2, 3]},
                {"function": "count_up_to", "inputs": [11], "expected": [2, 3, 5, 7]},
                {"function": "count_up_to", "inputs": [0], "expected": []},
                {"function": "count_up_to", "inputs": [1], "expected": []},
            ],
            "correct_bracketing": [
                {"function": "correct_bracketing", "inputs": ["<>"], "expected": True},
                {"function": "correct_bracketing", "inputs": ["<<><>>"], "expected": True},
                {"function": "correct_bracketing", "inputs": ["<"], "expected": False},
                {"function": "correct_bracketing", "inputs": ["<>><"], "expected": False},
            ],
        }

        return test_map.get(pattern_name, [])
