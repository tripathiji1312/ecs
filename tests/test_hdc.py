"""Tests for HDC Memory System."""

import time
import numpy as np
from ecs.memory.hdc import HDCMemory, MemoryType


def test_basic_operations():
    """Test fundamental HDC operations."""
    mem = HDCMemory(dimension=10000)

    a = mem.generate_random_vector()
    b = mem.generate_random_vector()
    bound = mem._xor_bind(a, b)

    recovered_a = mem._xor_bind(bound, b)
    sim = mem._similarity(recovered_a, a)
    assert sim > 0.99, f"Unbind failed: {sim}"

    vectors = [mem.generate_random_vector() for _ in range(10)]
    bundled = mem._majority_bundle(vectors)

    for v in vectors:
        sim = mem._similarity(bundled, v)
        assert sim > 0.4, f"Bundle similarity too low: {sim}"

    random_vec = mem.generate_random_vector()
    random_sim = mem._similarity(bundled, random_vec)
    assert random_sim < 0.6, f"Too similar to random: {random_sim}"


def test_text_encoding():
    """Test text encoding and retrieval."""
    mem = HDCMemory(dimension=10000)

    mem.store(
        "concept_1",
        mem.encode_text("binary search algorithm"),
        MemoryType.SEMANTIC,
        "Binary search: O(log n) search on sorted arrays"
    )

    mem.store(
        "concept_2",
        mem.encode_text("quicksort algorithm"),
        MemoryType.SEMANTIC,
        "Quicksort: divide and conquer sorting"
    )

    mem.store(
        "concept_3",
        mem.encode_text("hash table lookup"),
        MemoryType.SEMANTIC,
        "Hash table: O(1) average lookup"
    )

    results = mem.query_by_content("binary search algorithm")
    assert results[0][0].id == "concept_1", \
        f"Expected concept_1, got {results[0][0].id}"

    results = mem.query_by_content("quicksort algorithm")
    assert results[0][0].id == "concept_2", \
        f"Expected concept_2, got {results[0][0].id}"


def test_idf_semantic_retrieval():
    """Test that IDF weighting improves semantic retrieval."""
    mem = HDCMemory(dimension=10000)

    mem.store("c1", mem.encode_text("binary search sorted logarithmic divide"),
              MemoryType.SEMANTIC, "Binary search on sorted arrays")
    mem.store("c2", mem.encode_text("quicksort partition pivot recursive divide"),
              MemoryType.SEMANTIC, "Quicksort partition sort")
    mem.store("c3", mem.encode_text("hash table lookup constant average"),
              MemoryType.SEMANTIC, "Hash table lookup")

    results = mem.query_by_content("search sorted logarithmic")
    assert results[0][0].id == "c1", \
        f"Expected c1, got {results[0][0].id}"

    results = mem.query_by_content("partition pivot recursive")
    assert results[0][0].id == "c2", \
        f"Expected c2, got {results[0][0].id}"


def test_memory_types():
    """Test memory type filtering."""
    mem = HDCMemory(dimension=10000)

    mem.store("ep_1", mem.generate_random_vector(),
              MemoryType.EPISODIC, "Fixed bug in parser")
    mem.store("se_1", mem.generate_random_vector(),
              MemoryType.SEMANTIC, "Python GIL prevents true parallelism")
    mem.store("pr_1", mem.generate_random_vector(),
              MemoryType.PROCEDURAL, "How to write a retry decorator")
    mem.store("st_1", mem.generate_random_vector(),
              MemoryType.STRATEGIC, "Use divide-and-conquer for sorting")

    query_vec = mem.generate_random_vector()
    results = mem.query(query_vec, memory_type=MemoryType.EPISODIC)
    assert all(item.memory_type == MemoryType.EPISODIC for item, _ in results)


def test_relationships():
    """Test relationship encoding."""
    mem = HDCMemory(dimension=10000)

    rel_vec = mem.encode_relationship("python", "has_feature", "dynamic_typing")
    mem.store("rel_1", rel_vec, MemoryType.SEMANTIC,
              "Python has dynamic typing")

    subj_role = mem.get_atomic_vector("ROLE_SUBJECT")
    query_vec = mem._xor_bind(rel_vec, subj_role)

    python_vec = mem.get_atomic_vector("python")
    sim = mem._similarity(query_vec, python_vec)
    assert sim > 0.5, f"Role-filler unbinding failed: {sim}"


def test_noise_robustness():
    """Test that HDC is robust to noise."""
    mem = HDCMemory(dimension=10000)

    original = mem.encode_text("important coding pattern")
    mem.store("noisy_1", original, MemoryType.PROCEDURAL,
              "Important pattern")

    noisy_query = original.copy()
    flip_indices = np.random.choice(10000, size=4000, replace=False)
    noisy_query[flip_indices] = 1 - noisy_query[flip_indices]

    results = mem.query(noisy_query)
    assert results[0][0].id == "noisy_1", "Noise robustness failed"
    assert results[0][1] > 0.55, f"Similarity too low with noise: {results[0][1]}"


def test_associations():
    """Test associative links and spreading activation."""
    mem = HDCMemory(dimension=10000)

    mem.store("a", mem.generate_random_vector(), MemoryType.SEMANTIC, "Item A")
    mem.store("b", mem.generate_random_vector(), MemoryType.SEMANTIC, "Item B")
    mem.store("c", mem.generate_random_vector(), MemoryType.SEMANTIC, "Item C")

    mem.associate("a", "b", "related_to")
    mem.associate("b", "c", "related_to")

    results = mem.spread_activation("a", max_depth=2)
    target_ids = [item_id for item_id, _ in results]
    assert "b" in target_ids
    assert "c" in target_ids


def test_batch_query():
    """Test batch query correctness and that it matches linear query."""
    mem = HDCMemory(dimension=10000)

    for i in range(100):
        vec = mem.generate_random_vector()
        mem.store(f"item_{i}", vec, MemoryType.SEMANTIC, f"Item {i}")

    query_vec = mem.items["item_50"].hypervector.copy()

    results_batch = mem.batch_query(query_vec, top_k=5)
    assert results_batch[0][0].id == "item_50"
    assert results_batch[0][1] > 0.99

    # Reset access counts to compare fairly
    for item in mem.items.values():
        item.access_count = 0

    results_linear = mem.query(query_vec, top_k=5)
    assert results_linear[0][0].id == results_batch[0][0].id


def test_memory_decay():
    """Test memory decay and consolidation boost."""
    mem = HDCMemory(dimension=10000)

    vec = mem.encode_text("test memory decay item")
    mem.store("decay_test", vec, MemoryType.EPISODIC, "Test memory")

    item = mem.items["decay_test"]
    item.created_at = time.time() - (10 * 86400)
    item.access_count = 0

    results_decayed = mem.query(vec, apply_decay=True)
    decayed_sim = results_decayed[0][1]

    item.access_count = 20
    # Reset the access count bump from the query above
    item.access_count = 20

    results_consolidated = mem.query(vec, apply_decay=True)
    consolidated_sim = results_consolidated[0][1]

    assert consolidated_sim > decayed_sim, \
        f"Consolidated ({consolidated_sim:.3f}) should beat decayed ({decayed_sim:.3f})"


def test_consolidate_memories():
    """Test that old unused episodic memories get pruned."""
    mem = HDCMemory(dimension=10000)

    mem.store("old_unused", mem.generate_random_vector(),
              MemoryType.EPISODIC, "Old unused episode")
    mem.store("old_used", mem.generate_random_vector(),
              MemoryType.EPISODIC, "Old but accessed episode")
    mem.store("semantic", mem.generate_random_vector(),
              MemoryType.SEMANTIC, "Semantic knowledge")

    mem.items["old_unused"].created_at = time.time() - (10 * 86400)
    mem.items["old_unused"].access_count = 0

    mem.items["old_used"].created_at = time.time() - (10 * 86400)
    mem.items["old_used"].access_count = 5

    pruned = mem.consolidate_memories()
    assert pruned == 1
    assert "old_unused" not in mem.items
    assert "old_used" in mem.items
    assert "semantic" in mem.items


def test_fuzzy_retrieval():
    """Test that queries with high token overlap retrieve the best match."""
    mem = HDCMemory(dimension=10000)

    mem.store("sort", mem.encode_text("quicksort partition pivot divide"),
              MemoryType.SEMANTIC, "quicksort partition pivot divide")
    mem.store("search", mem.encode_text("binary search sorted array log"),
              MemoryType.SEMANTIC, "binary search sorted array log")
    mem.store("hash", mem.encode_text("hash table key value bucket"),
              MemoryType.SEMANTIC, "hash table key value bucket")

    results = mem.query_by_content("quicksort partition pivot")
    assert results[0][0].id == "sort", \
        f"Expected sort, got {results[0][0].id}"

    results = mem.query_by_content("hash table key value")
    assert results[0][0].id == "hash", \
        f"Expected hash, got {results[0][0].id}"


def test_persistence(tmp_path):
    """Test save and load."""
    mem = HDCMemory(dimension=10000)
    mem.store("p1", mem.encode_text("persistent item"),
              MemoryType.SEMANTIC, "Should survive save/load")

    filepath = str(tmp_path / "test_memory.pkl")
    mem.save(filepath)

    mem2 = HDCMemory(dimension=10000)
    mem2.load(filepath)

    assert "p1" in mem2.items
    assert mem2.items["p1"].content == "Should survive save/load"
    assert len(mem2.type_indices[MemoryType.SEMANTIC]) == 1
    assert mem2.total_documents == 1


def test_timestamps():
    """Test that items get real timestamps."""
    mem = HDCMemory(dimension=10000)
    before = time.time()
    mem.store("ts_test", mem.generate_random_vector(),
              MemoryType.SEMANTIC, "Timestamp test")
    after = time.time()

    item = mem.items["ts_test"]
    assert before <= item.created_at <= after
    assert before <= item.last_accessed <= after
