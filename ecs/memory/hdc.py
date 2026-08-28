"""
Hyperdimensional Computing Memory System
Based on Vector Symbolic Architectures (VSA)
Uses Binary Spatter Codes with 10,000-dimensional vectors.
"""

import math
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import pickle


# Pre-computed popcount lookup table for uint8
POPCOUNT_TABLE = np.array([
    bin(i).count('1') for i in range(256)
], dtype=np.uint8)


class MemoryType(Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    STRATEGIC = "strategic"


@dataclass
class HDCItem:
    """A single item stored in HDC memory."""
    id: str
    hypervector: np.ndarray
    memory_type: MemoryType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


class HDCMemory:
    """
    Hyperdimensional Memory using Binary Spatter Codes.

    Operations:
    - BIND (XOR):     Combine two vectors (role-filler binding)
    - BUNDLE (MAJORITY): Superpose vectors (set membership)
    - PERMUTE (ROLL):  Sequence encoding
    - SIMILARITY (HAMMING): Measure closeness
    """

    DIMENSION = 10000
    DECAY_HALF_LIFE = 86400.0
    CONSOLIDATION_THRESHOLD = 10

    STOPWORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "in", "on",
        "at", "to", "for", "of", "with", "and", "or", "not", "but",
        "algorithm", "function", "code", "write", "implement", "create",
    })

    def __init__(self, dimension: int = 10000):
        self.dimension = dimension
        self.items: Dict[str, HDCItem] = {}
        self.type_indices: Dict[MemoryType, List[str]] = {
            mtype: [] for mtype in MemoryType
        }
        self.atomic_vectors: Dict[str, np.ndarray] = {}

        # Document frequency tracking for IDF
        self.document_frequencies: Counter = Counter()
        self.total_documents: int = 0

        # Packed matrix cache
        self._packed_matrix_cache: Optional[np.ndarray] = None
        self._cached_item_ids: Optional[List[str]] = None

    # ═══════════════════════════════════════════
    # CORE HDC OPERATIONS
    # ═══════════════════════════════════════════

    def _xor_bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """BIND operation: XOR two binary vectors."""
        return np.logical_xor(a, b).astype(np.int8)

    def _majority_bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """BUNDLE operation: Element-wise majority vote."""
        if not vectors:
            raise ValueError("Cannot bundle empty list")

        stacked = np.stack(vectors)
        sums = np.sum(stacked, axis=0)
        threshold = len(vectors) / 2
        result = (sums > threshold).astype(np.int8)

        ties = (sums == threshold)
        if np.any(ties):
            result[ties] = np.random.randint(0, 2, size=np.sum(ties)).astype(np.int8)

        return result

    def _weighted_bundle(self, weighted_vectors: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """Weighted bundling: vectors with higher weight contribute more via repetition."""
        if not weighted_vectors:
            raise ValueError("Cannot bundle empty list")

        min_weight = min(w for _, w in weighted_vectors)
        if min_weight <= 0:
            min_weight = 0.1

        repetitions = []
        for vec, weight in weighted_vectors:
            rep_count = max(1, int(round(weight / min_weight)))
            repetitions.extend([vec] * rep_count)

        return self._majority_bundle(repetitions)

    def _permute(self, vec: np.ndarray, positions: int = 1) -> np.ndarray:
        """PERMUTE operation: Cyclic shift for sequence encoding."""
        return np.roll(vec, positions)

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """SIMILARITY: Normalized Hamming similarity.
        1.0 = identical, 0.0 = opposite, 0.5 = unrelated."""
        hamming = np.sum(np.logical_xor(a, b))
        return 1.0 - (hamming / self.dimension)

    # ═══════════════════════════════════════════
    # VECTOR GENERATION
    # ═══════════════════════════════════════════

    def generate_random_vector(self) -> np.ndarray:
        """Generate a random binary hypervector."""
        return np.random.randint(0, 2, size=self.dimension, dtype=np.int8)

    def get_atomic_vector(self, token: str) -> np.ndarray:
        """Get or create a deterministic atomic vector for a token.
        Uses 128-bit hash seed with collision detection."""
        if token not in self.atomic_vectors:
            hasher = hashlib.sha256(token.encode())
            seed_bytes = hasher.digest()[:16]
            seed = int.from_bytes(seed_bytes, 'big') % (2**32)
            rng = np.random.RandomState(seed)
            vec = rng.randint(0, 2, size=self.dimension, dtype=np.int8)

            if len(self.atomic_vectors) < 1000:
                for existing_token, existing_vec in self.atomic_vectors.items():
                    if self._similarity(vec, existing_vec) > 0.51:
                        perturbation = rng.randint(0, 2, size=self.dimension, dtype=np.int8)
                        vec = np.logical_xor(vec, perturbation).astype(np.int8)
                        break

            self.atomic_vectors[token] = vec
        return self.atomic_vectors[token]

    # ═══════════════════════════════════════════
    # IDF
    # ═══════════════════════════════════════════

    def _compute_idf(self, token: str) -> float:
        """Compute inverse document frequency for a token."""
        if self.total_documents == 0:
            return 1.0
        df = self.document_frequencies.get(token, 0)
        return math.log(self.total_documents / (df + 1)) + 1

    # ═══════════════════════════════════════════
    # ENCODING
    # ═══════════════════════════════════════════

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text with TF-IDF weighting."""
        words = text.lower().split()
        if not words:
            return self.generate_random_vector()

        term_freq = Counter(words)
        weighted_vectors: List[Tuple[np.ndarray, float]] = []

        for i, word in enumerate(words):
            word_vec = self.get_atomic_vector(word)

            tf = term_freq[word] / len(words)
            idf = self._compute_idf(word)

            if word in self.STOPWORDS:
                idf *= 0.1

            weight = tf * idf

            position_vec = self._permute(word_vec, positions=i)
            weighted_vectors.append((position_vec, weight * 0.5))

            if i > 0:
                prev_word = words[i - 1]
                prev_vec = self.get_atomic_vector(prev_word)
                bigram_vec = self._xor_bind(prev_vec, word_vec)
                bigram_idf = (self._compute_idf(prev_word) + self._compute_idf(word)) / 2
                weighted_vectors.append((bigram_vec, weight * bigram_idf))

        return self._weighted_bundle(weighted_vectors)

    def encode_code_pattern(self, code: str) -> np.ndarray:
        """Encode code structure into hypervector."""
        features = []

        if "for " in code or "while " in code:
            features.append("loop")
        if "if " in code or "else" in code:
            features.append("condition")
        if "def " in code or "function" in code:
            features.append("function")
        if "return" in code:
            features.append("return")
        if "class " in code:
            features.append("class")
        if "import" in code:
            features.append("import")
        if "try:" in code:
            features.append("try")
        if "except" in code:
            features.append("catch")
        if "[" in code and "]" in code:
            features.append("array")
        if "{" in code and "}" in code:
            features.append("dict")
        if "(" in code and ")" in code:
            features.append("call")
        if code.count("for ") > 1:
            features.append("nested_loop")
        if "recursive" in code.lower() or code.count("def ") > 1:
            features.append("recursion")

        if not features:
            return self.encode_text(code)

        feature_vectors = [self.get_atomic_vector(f) for f in features]
        return self._majority_bundle(feature_vectors)

    def encode_relationship(self, subject: str, relation: str,
                            object_: str) -> np.ndarray:
        """Encode a triple relationship: (subject, relation, object)."""
        subj_role = self.get_atomic_vector("ROLE_SUBJECT")
        rel_role = self.get_atomic_vector("ROLE_RELATION")
        obj_role = self.get_atomic_vector("ROLE_OBJECT")

        subj_vec = self.get_atomic_vector(subject.lower())
        rel_vec = self.get_atomic_vector(relation.lower())
        obj_vec = self.get_atomic_vector(object_.lower())

        bound_subj = self._xor_bind(subj_role, subj_vec)
        bound_rel = self._xor_bind(rel_role, rel_vec)
        bound_obj = self._xor_bind(obj_role, obj_vec)

        return self._majority_bundle([bound_subj, bound_rel, bound_obj])

    # ═══════════════════════════════════════════
    # STORAGE & RETRIEVAL
    # ═══════════════════════════════════════════

    def store(self, item_id: str, hypervector: np.ndarray,
              memory_type: MemoryType, content: str,
              metadata: Optional[Dict] = None) -> HDCItem:
        """Store an item in memory and update document frequencies."""
        if len(hypervector) != self.dimension:
            raise ValueError(f"Hypervector must be {self.dimension}-dimensional")

        words = content.lower().split()
        for word in set(words):
            self.document_frequencies[word] += 1
        self.total_documents += 1

        item = HDCItem(
            id=item_id,
            hypervector=hypervector,
            memory_type=memory_type,
            content=content,
            metadata=metadata or {},
        )

        self.items[item_id] = item
        self.type_indices[memory_type].append(item_id)
        self._invalidate_matrix_cache()

        return item

    def _apply_decay(self, item: HDCItem) -> float:
        """Compute decayed similarity multiplier for an item."""
        age = time.time() - item.created_at
        time_decay = math.exp(-age / self.DECAY_HALF_LIFE)

        access_consolidation = 1.0 - math.exp(
            -item.access_count / self.CONSOLIDATION_THRESHOLD
        )

        return (0.3 + 0.7 * access_consolidation) * (0.5 + 0.5 * time_decay)

    def query(self, query_vector: np.ndarray,
              memory_type: Optional[MemoryType] = None,
              top_k: int = 5,
              apply_decay: bool = False) -> List[Tuple[HDCItem, float]]:
        """Query memory for similar items."""
        if memory_type:
            item_ids = self.type_indices[memory_type]
        else:
            item_ids = list(self.items.keys())

        if not item_ids:
            return []

        candidates = []
        for item_id in item_ids:
            item = self.items[item_id]
            sim = self._similarity(query_vector, item.hypervector)
            if apply_decay:
                sim *= self._apply_decay(item)
            candidates.append((item, sim))

        candidates.sort(key=lambda x: x[1], reverse=True)

        now = time.time()
        for item, _ in candidates[:top_k]:
            item.access_count += 1
            item.last_accessed = now

        return candidates[:top_k]

    def query_by_content(self, text: str,
                         memory_type: Optional[MemoryType] = None,
                         top_k: int = 5) -> List[Tuple[HDCItem, float]]:
        """Convenience method: encode text and query."""
        query_vec = self.encode_text(text)
        return self.query(query_vec, memory_type, top_k)

    # ═══════════════════════════════════════════
    # BATCH QUERY (PACKED BITS)
    # ═══════════════════════════════════════════

    def _invalidate_matrix_cache(self) -> None:
        """Call when items are added/removed."""
        self._packed_matrix_cache = None
        self._cached_item_ids = None

    def _build_packed_matrix(self, item_ids: List[str]) -> np.ndarray:
        """Build packed bit matrix for fast batch similarity."""
        vectors = [self.items[iid].hypervector for iid in item_ids]
        matrix = np.stack(vectors).astype(np.uint8)
        return np.packbits(matrix, axis=1)

    def batch_query(self, query_vector: np.ndarray,
                    memory_type: Optional[MemoryType] = None,
                    top_k: int = 5) -> List[Tuple[HDCItem, float]]:
        """Fast batch query using packed bit operations."""
        if memory_type:
            item_ids = self.type_indices[memory_type]
        else:
            item_ids = list(self.items.keys())

        if not item_ids:
            return []

        packed_query = np.packbits(query_vector.astype(np.uint8))
        packed_matrix = self._build_packed_matrix(item_ids)

        xored = np.bitwise_xor(packed_matrix, packed_query)
        hamming_distances = POPCOUNT_TABLE[xored].sum(axis=1).astype(np.float64)
        similarities = 1.0 - (hamming_distances / self.dimension)

        top_k_clamped = min(top_k, len(item_ids))
        top_indices = np.argsort(similarities)[-top_k_clamped:][::-1]

        now = time.time()
        results = []
        for idx in top_indices:
            item_id = item_ids[idx]
            item = self.items[item_id]
            item.access_count += 1
            item.last_accessed = now
            results.append((item, float(similarities[idx])))

        return results

    # ═══════════════════════════════════════════
    # ASSOCIATIVE OPERATIONS
    # ═══════════════════════════════════════════

    def associate(self, item_a_id: str, item_b_id: str,
                  relation: str = "related_to") -> None:
        """Create an associative link between two items."""
        if item_a_id not in self.items or item_b_id not in self.items:
            raise KeyError("Both items must exist in memory")

        item_a = self.items[item_a_id]
        item_b = self.items[item_b_id]

        if "associations" not in item_a.metadata:
            item_a.metadata["associations"] = []
        item_a.metadata["associations"].append({
            "target": item_b_id,
            "relation": relation
        })

        if "associations" not in item_b.metadata:
            item_b.metadata["associations"] = []
        item_b.metadata["associations"].append({
            "target": item_a_id,
            "relation": relation
        })

    def spread_activation(self, seed_item_id: str,
                          max_depth: int = 2) -> List[Tuple[str, float]]:
        """Spreading activation: find items connected through associations."""
        if seed_item_id not in self.items:
            raise KeyError(f"Item {seed_item_id} not found")

        visited = {seed_item_id}
        frontier = [(seed_item_id, 1.0)]
        results = []

        for _ in range(max_depth):
            next_frontier = []
            for item_id, strength in frontier:
                if item_id in self.items:
                    item = self.items[item_id]
                    for assoc in item.metadata.get("associations", []):
                        target = assoc["target"]
                        if target not in visited:
                            decayed_strength = strength * 0.7
                            next_frontier.append((target, decayed_strength))
                            results.append((target, decayed_strength))
                            visited.add(target)
            frontier = next_frontier

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ═══════════════════════════════════════════
    # MEMORY CONSOLIDATION
    # ═══════════════════════════════════════════

    def consolidate_memories(self) -> int:
        """Prune old episodic memories that were never accessed."""
        items_to_remove = []
        now = time.time()

        for item_id, item in self.items.items():
            age = now - item.created_at
            if (age > 7 * 86400 and
                    item.access_count == 0 and
                    item.memory_type == MemoryType.EPISODIC):
                items_to_remove.append(item_id)

        for item_id in items_to_remove:
            item = self.items[item_id]
            self.type_indices[item.memory_type].remove(item_id)
            del self.items[item_id]

        if items_to_remove:
            self._invalidate_matrix_cache()

        return len(items_to_remove)

    # ═══════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════

    def save(self, filepath: str) -> None:
        """Save memory to disk."""
        data = {
            "dimension": self.dimension,
            "items": self.items,
            "atomic_vectors": self.atomic_vectors,
            "document_frequencies": self.document_frequencies,
            "total_documents": self.total_documents,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, filepath: str) -> None:
        """Load memory from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.dimension = data["dimension"]
        self.items = data["items"]
        self.atomic_vectors = data["atomic_vectors"]
        self.document_frequencies = data.get("document_frequencies", Counter())
        self.total_documents = data.get("total_documents", 0)

        self.type_indices = {mtype: [] for mtype in MemoryType}
        for item_id, item in self.items.items():
            self.type_indices[item.memory_type].append(item_id)

        self._invalidate_matrix_cache()
