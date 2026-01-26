"""Tests for RRF merge functionality."""

import pytest
from collections import defaultdict


def rrf_merge(
    bm25_results: list[dict],
    vector_results: list[dict],
    k: int = 60,
    bm25_weight: float = 1.0,
    vector_weight: float = 1.0,
) -> list[dict]:
    """RRF merge implementation for testing."""
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, dict] = {}

    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc.get("doc_id")
        if doc_id:
            scores[doc_id] += bm25_weight / (k + rank)
            doc_map[doc_id] = doc

    for rank, doc in enumerate(vector_results, start=1):
        doc_id = doc.get("doc_id")
        if doc_id:
            scores[doc_id] += vector_weight / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = scores[doc_id]
        results.append(doc)

    return results


class TestRRFMerge:
    """Tests for Reciprocal Rank Fusion."""

    def test_basic_merge(self):
        """Test basic RRF merge with overlapping results."""
        bm25 = [
            {"doc_id": "a", "name": "Doc A"},
            {"doc_id": "b", "name": "Doc B"},
            {"doc_id": "c", "name": "Doc C"},
        ]
        vector = [
            {"doc_id": "b", "name": "Doc B"},
            {"doc_id": "d", "name": "Doc D"},
            {"doc_id": "a", "name": "Doc A"},
        ]

        merged = rrf_merge(bm25, vector, k=60)

        # All unique docs should be present
        doc_ids = [d["doc_id"] for d in merged]
        assert set(doc_ids) == {"a", "b", "c", "d"}

        # Doc B should be ranked highest (rank 1 in both)
        assert merged[0]["doc_id"] == "b"

    def test_rrf_scores_calculated(self):
        """Test that RRF scores are calculated correctly."""
        bm25 = [{"doc_id": "a"}]
        vector = [{"doc_id": "a"}]

        merged = rrf_merge(bm25, vector, k=60)

        # Score should be 1/(60+1) + 1/(60+1) = 2/61
        expected_score = 2 / 61
        assert merged[0]["rrf_score"] == pytest.approx(expected_score)

    def test_empty_results(self):
        """Test merge with empty results."""
        merged = rrf_merge([], [], k=60)
        assert merged == []

        merged = rrf_merge([{"doc_id": "a"}], [], k=60)
        assert len(merged) == 1

    def test_weight_impact(self):
        """Test that weights affect ranking."""
        bm25 = [{"doc_id": "a"}, {"doc_id": "b"}]
        vector = [{"doc_id": "b"}, {"doc_id": "a"}]

        # Equal weights: b should be first (rank 1+2 vs 2+1)
        merged_equal = rrf_merge(bm25, vector, bm25_weight=1.0, vector_weight=1.0)

        # Heavy BM25 weight: a should be first
        merged_bm25 = rrf_merge(bm25, vector, bm25_weight=10.0, vector_weight=1.0)

        # Heavy vector weight: b should be first
        merged_vector = rrf_merge(bm25, vector, bm25_weight=1.0, vector_weight=10.0)

        # With equal weights and symmetric ranks, scores are equal
        assert merged_equal[0]["rrf_score"] == merged_equal[1]["rrf_score"]

        # With BM25 weight, 'a' should score higher
        assert merged_bm25[0]["doc_id"] == "a"

        # With vector weight, 'b' should score higher
        assert merged_vector[0]["doc_id"] == "b"

    def test_k_parameter_impact(self):
        """Test that k parameter affects score distribution."""
        docs = [{"doc_id": "a"}, {"doc_id": "b"}, {"doc_id": "c"}]

        # Low k: bigger score differences
        merged_low_k = rrf_merge(docs, [], k=1)

        # High k: smaller score differences
        merged_high_k = rrf_merge(docs, [], k=100)

        # Score difference should be larger with low k
        diff_low = merged_low_k[0]["rrf_score"] - merged_low_k[1]["rrf_score"]
        diff_high = merged_high_k[0]["rrf_score"] - merged_high_k[1]["rrf_score"]

        assert diff_low > diff_high
