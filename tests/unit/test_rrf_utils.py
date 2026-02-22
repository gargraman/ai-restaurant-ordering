"""Tests for RRF utilities - 3-way and generic merge."""

import pytest

from src.search.rrf_utils import (
    rrf_merge_3way,
    rrf_merge_simple,
    compute_rrf_scores,
    format_rrf_results,
)


class TestRRFMerge3Way:
    """Tests for rrf_merge_3way function."""

    def test_merge_three_sources(self):
        """Test merging results from three sources."""
        bm25 = [
            {"doc_id": "doc-1", "source": "bm25"},
            {"doc_id": "doc-2", "source": "bm25"},
            {"doc_id": "doc-3", "source": "bm25"},
        ]
        vector = [
            {"doc_id": "doc-2", "source": "vector"},
            {"doc_id": "doc-4", "source": "vector"},
            {"doc_id": "doc-1", "source": "vector"},
        ]
        graph = [
            {"doc_id": "doc-5", "source": "graph"},
            {"doc_id": "doc-2", "source": "graph"},
            {"doc_id": "doc-6", "source": "graph"},
        ]

        merged = rrf_merge_3way(bm25, vector, graph, k=60)

        # All unique docs should be present
        doc_ids = [d["doc_id"] for d in merged]
        assert set(doc_ids) == {"doc-1", "doc-2", "doc-3", "doc-4", "doc-5", "doc-6"}

        # doc-2 should be ranked highest (appears in all three)
        assert merged[0]["doc_id"] == "doc-2"

    def test_merge_with_equal_weights(self):
        """Test merging with equal weights for all sources."""
        bm25 = [{"doc_id": "doc-1"}, {"doc_id": "doc-2"}]
        vector = [{"doc_id": "doc-2"}, {"doc_id": "doc-1"}]
        graph = [{"doc_id": "doc-3"}]

        merged = rrf_merge_3way(
            bm25, vector, graph,
            k=60,
            bm25_weight=1.0,
            vector_weight=1.0,
            graph_weight=1.0,
        )

        # doc-1 and doc-2 should be ranked higher than doc-3
        doc_ids = [d["doc_id"] for d in merged[:2]]
        assert "doc-1" in doc_ids
        assert "doc-2" in doc_ids

    def test_merge_with_custom_weights(self):
        """Test merging with custom weights."""
        bm25 = [{"doc_id": "doc-1"}, {"doc_id": "doc-2"}]
        vector = [{"doc_id": "doc-2"}, {"doc_id": "doc-1"}]
        graph = [{"doc_id": "doc-3"}, {"doc_id": "doc-1"}]

        # Heavy graph weight should boost doc-3
        # doc-1: rank 1 in bm25, rank 2 in vector, rank 2 in graph
        # doc-2: rank 2 in bm25, rank 1 in vector
        # doc-3: rank 1 in graph
        # With graph_weight=10, doc-3 gets 10/(60+1) = 0.164
        # doc-1 gets 0.5/61 + 0.5/62 + 10/62 = 0.008 + 0.008 + 0.161 = 0.177
        # So doc-1 should still be first
        merged = rrf_merge_3way(
            bm25, vector, graph,
            k=60,
            bm25_weight=0.5,
            vector_weight=0.5,
            graph_weight=10.0,
        )

        # doc-1 appears in all three sources, should be ranked first
        assert merged[0]["doc_id"] == "doc-1"

    def test_merge_empty_sources(self):
        """Test merging when some sources are empty."""
        bm25 = [{"doc_id": "doc-1"}, {"doc_id": "doc-2"}]
        vector = []
        graph = [{"doc_id": "doc-3"}]

        merged = rrf_merge_3way(bm25, vector, graph)

        # Should still merge available results
        doc_ids = [d["doc_id"] for d in merged]
        assert "doc-1" in doc_ids
        assert "doc-2" in doc_ids
        assert "doc-3" in doc_ids

    def test_merge_all_empty(self):
        """Test merging when all sources are empty."""
        merged = rrf_merge_3way([], [], [])

        assert merged == []

    def test_merge_sets_source_indicators(self):
        """Test that merge sets correct source indicators."""
        bm25 = [{"doc_id": "doc-1"}]
        vector = [{"doc_id": "doc-2"}]
        graph = [{"doc_id": "doc-3"}]

        merged = rrf_merge_3way(bm25, vector, graph)

        # Each doc should have correct source indicators
        doc_map = {d["doc_id"]: d for d in merged}

        assert doc_map["doc-1"]["in_bm25"] is True
        assert doc_map["doc-1"]["in_vector"] is False
        assert doc_map["doc-1"]["in_graph"] is False

        assert doc_map["doc-2"]["in_bm25"] is False
        assert doc_map["doc-2"]["in_vector"] is True
        assert doc_map["doc-2"]["in_graph"] is False

        assert doc_map["doc-3"]["in_bm25"] is False
        assert doc_map["doc-3"]["in_vector"] is False
        assert doc_map["doc-3"]["in_graph"] is True

    def test_merge_overlapping_results(self):
        """Test merging with overlapping results across sources."""
        bm25 = [
            {"doc_id": "doc-1", "name": "BM25-1"},
            {"doc_id": "doc-2", "name": "BM25-2"},
        ]
        vector = [
            {"doc_id": "doc-2", "name": "Vector-2"},
            {"doc_id": "doc-3", "name": "Vector-3"},
        ]
        graph = [
            {"doc_id": "doc-1", "name": "Graph-1"},
            {"doc_id": "doc-3", "name": "Graph-3"},
        ]

        merged = rrf_merge_3way(bm25, vector, graph)

        # Should have 3 unique docs
        assert len(merged) == 3

        # All docs should appear in multiple sources
        for doc in merged:
            sources_count = sum([
                doc.get("in_bm25", False),
                doc.get("in_vector", False),
                doc.get("in_graph", False),
            ])
            assert sources_count >= 2

    def test_merge_preserves_document_data(self):
        """Test that merge preserves original document data."""
        bm25 = [
            {
                "doc_id": "doc-1",
                "item_name": "Pasta",
                "restaurant_name": "Italian Place",
                "display_price": 89.99,
            }
        ]
        vector = []
        graph = []

        merged = rrf_merge_3way(bm25, vector, graph)

        # Should preserve all original fields
        assert merged[0]["item_name"] == "Pasta"
        assert merged[0]["restaurant_name"] == "Italian Place"
        assert merged[0]["display_price"] == 89.99
        assert "rrf_score" in merged[0]

    def test_merge_k_parameter_effect(self):
        """Test that k parameter affects score distribution."""
        # Use asymmetric rankings to ensure different scores
        bm25 = [{"doc_id": "doc-1"}, {"doc_id": "doc-2"}, {"doc_id": "doc-3"}]
        vector = [{"doc_id": "doc-1"}, {"doc_id": "doc-2"}]
        graph = []

        # Low k: bigger score differences
        merged_low_k = rrf_merge_3way(bm25, vector, graph, k=1)

        # High k: smaller score differences
        merged_high_k = rrf_merge_3way(bm25, vector, graph, k=100)

        # Score difference between doc-1 and doc-2 should be larger with low k
        # doc-1: rank 1 in both, doc-2: rank 2 in both
        # With k=1: doc-1 = 1/2 + 1/2 = 1.0, doc-2 = 1/3 + 1/3 = 0.67, diff = 0.33
        # With k=100: doc-1 = 1/101 + 1/101 = 0.02, doc-2 = 1/102 + 1/102 = 0.02, diff ≈ 0.0004
        diff_low = merged_low_k[0]["rrf_score"] - merged_low_k[1]["rrf_score"]
        diff_high = merged_high_k[0]["rrf_score"] - merged_high_k[1]["rrf_score"]

        assert diff_low > diff_high
        assert diff_low > 0  # Ensure there's actually a difference

    def test_merge_sources_field(self):
        """Test that merge includes sources field."""
        bm25 = [{"doc_id": "doc-1"}]
        vector = [{"doc_id": "doc-1"}]
        graph = [{"doc_id": "doc-1"}]

        merged = rrf_merge_3way(bm25, vector, graph)

        assert "sources" in merged[0]
        assert "bm25" in merged[0]["sources"]
        assert "vector" in merged[0]["sources"]
        assert "graph" in merged[0]["sources"]


class TestRRFMergeSimple:
    """Tests for rrf_merge_simple function."""

    def test_merge_two_sources(self):
        """Test merging two sources with generic function."""
        source1 = [{"doc_id": "doc-1"}, {"doc_id": "doc-2"}]
        source2 = [{"doc_id": "doc-2"}, {"doc_id": "doc-3"}]

        merged = rrf_merge_simple(source1, source2)

        assert len(merged) == 3
        doc_ids = [d["doc_id"] for d in merged]
        assert "doc-1" in doc_ids
        assert "doc-2" in doc_ids
        assert "doc-3" in doc_ids

    def test_merge_four_sources(self):
        """Test merging four sources with generic function."""
        source1 = [{"doc_id": "doc-1"}]
        source2 = [{"doc_id": "doc-2"}]
        source3 = [{"doc_id": "doc-3"}]
        source4 = [{"doc_id": "doc-4"}]

        merged = rrf_merge_simple(source1, source2, source3, source4)

        assert len(merged) == 4

    def test_merge_with_custom_weights(self):
        """Test merging with custom weights."""
        source1 = [{"doc_id": "doc-1"}, {"doc_id": "doc-2"}]
        source2 = [{"doc_id": "doc-2"}, {"doc_id": "doc-1"}]

        merged = rrf_merge_simple(
            source1, source2,
            weights=[10.0, 1.0],
        )

        # doc-1 should be ranked first due to higher weight on source1
        assert merged[0]["doc_id"] == "doc-1"

    def test_merge_with_custom_source_names(self):
        """Test merging with custom source names."""
        source1 = [{"doc_id": "doc-1"}]
        source2 = [{"doc_id": "doc-1"}]

        merged = rrf_merge_simple(
            source1, source2,
            source_names=["custom1", "custom2"],
        )

        assert "custom1" in merged[0]["sources"]
        assert "custom2" in merged[0]["sources"]

    def test_merge_weight_count_mismatch(self):
        """Test that weight count must match source count."""
        source1 = [{"doc_id": "doc-1"}]
        source2 = [{"doc_id": "doc-2"}]

        with pytest.raises(ValueError, match="Number of weights"):
            rrf_merge_simple(source1, source2, weights=[1.0, 2.0, 3.0])

    def test_merge_source_names_count_mismatch(self):
        """Test that source names count must match source count."""
        source1 = [{"doc_id": "doc-1"}]
        source2 = [{"doc_id": "doc-2"}]

        with pytest.raises(ValueError, match="Number of source names"):
            rrf_merge_simple(source1, source2, source_names=["only_one"])

    def test_merge_empty_input(self):
        """Test merging with empty input."""
        merged = rrf_merge_simple()

        assert merged == []

    def test_merge_some_empty_sources(self):
        """Test merging when some sources are empty."""
        source1 = [{"doc_id": "doc-1"}]
        source2 = []
        source3 = [{"doc_id": "doc-2"}]

        merged = rrf_merge_simple(source1, source2, source3)

        assert len(merged) == 2


class TestComputeRRFScores:
    """Tests for compute_rrf_scores function."""

    def test_compute_scores_basic(self):
        """Test basic score computation."""
        results_list = [
            [{"doc_id": "doc-1"}, {"doc_id": "doc-2"}],
            [{"doc_id": "doc-2"}, {"doc_id": "doc-3"}],
        ]
        weights = [1.0, 1.0]
        source_names = ["source1", "source2"]

        scores, doc_map, sources = compute_rrf_scores(
            results_list, weights, source_names, k=60
        )

        # doc-2 should have highest score (rank 2 in source1, rank 1 in source2)
        assert scores["doc-2"] > scores["doc-1"]
        assert scores["doc-2"] > scores["doc-3"]

    def test_compute_scores_doc_map(self):
        """Test that doc map preserves first occurrence."""
        results_list = [
            [{"doc_id": "doc-1", "name": "First"}],
            [{"doc_id": "doc-1", "name": "Second"}],
        ]
        weights = [1.0, 1.0]
        source_names = ["source1", "source2"]

        scores, doc_map, sources = compute_rrf_scores(
            results_list, weights, source_names, k=60
        )

        # Should preserve first occurrence
        assert doc_map["doc-1"]["name"] == "First"

    def test_compute_scores_sources(self):
        """Test that sources are tracked correctly."""
        results_list = [
            [{"doc_id": "doc-1"}],
            [{"doc_id": "doc-1"}],
            [{"doc_id": "doc-2"}],
        ]
        weights = [1.0, 1.0, 1.0]
        source_names = ["bm25", "vector", "graph"]

        scores, doc_map, sources = compute_rrf_scores(
            results_list, weights, source_names, k=60
        )

        assert "bm25" in sources["doc-1"]
        assert "vector" in sources["doc-1"]
        assert "graph" not in sources["doc-1"]

        assert "graph" in sources["doc-2"]

    def test_compute_scores_empty_results(self):
        """Test score computation with empty results."""
        results_list = [[], []]
        weights = [1.0, 1.0]
        source_names = ["source1", "source2"]

        scores, doc_map, sources = compute_rrf_scores(
            results_list, weights, source_names, k=60
        )

        assert scores == {}
        assert doc_map == {}
        assert sources == {}

    def test_compute_scores_weighted(self):
        """Test score computation with different weights."""
        results_list = [
            [{"doc_id": "doc-1"}],
            [{"doc_id": "doc-1"}],
        ]
        weights = [10.0, 1.0]
        source_names = ["heavy", "light"]

        scores, doc_map, sources = compute_rrf_scores(
            results_list, weights, source_names, k=60
        )

        # Score should be weighted sum
        expected_score = 10.0 / (60 + 1) + 1.0 / (60 + 1)
        assert scores["doc-1"] == pytest.approx(expected_score)


class TestFormatRRFResults:
    """Tests for format_rrf_results function."""

    def test_format_basic(self):
        """Test basic result formatting."""
        scores = {"doc-1": 0.03, "doc-2": 0.02}
        doc_map = {
            "doc-1": {"doc_id": "doc-1", "name": "Doc 1"},
            "doc-2": {"doc_id": "doc-2", "name": "Doc 2"},
        }
        sources = {
            "doc-1": {"source1"},
            "doc-2": {"source2"},
        }

        results = format_rrf_results(scores, doc_map, sources)

        # Should be sorted by score
        assert results[0]["doc_id"] == "doc-1"
        assert results[1]["doc_id"] == "doc-2"

        # Should include RRF score
        assert results[0]["rrf_score"] == 0.03
        assert results[1]["rrf_score"] == 0.02

    def test_format_includes_source_indicators(self):
        """Test that formatting includes source indicators."""
        scores = {"doc-1": 0.03}
        doc_map = {"doc-1": {"doc_id": "doc-1"}}
        sources = {"doc-1": {"bm25", "vector"}}

        results = format_rrf_results(scores, doc_map, sources)

        assert results[0]["in_bm25"] is True
        assert results[0]["in_vector"] is True
        assert results[0]["in_graph"] is False

    def test_format_preserves_document_data(self):
        """Test that formatting preserves document data."""
        scores = {"doc-1": 0.03}
        doc_map = {
            "doc-1": {
                "doc_id": "doc-1",
                "item_name": "Pasta",
                "restaurant_name": "Italian Place",
            }
        }
        sources = {"doc-1": {"bm25"}}

        results = format_rrf_results(scores, doc_map, sources)

        assert results[0]["item_name"] == "Pasta"
        assert results[0]["restaurant_name"] == "Italian Place"

    def test_format_empty(self):
        """Test formatting empty results."""
        results = format_rrf_results({}, {}, {})

        assert results == []

    def test_format_copies_document(self):
        """Test that formatting creates a copy of document."""
        scores = {"doc-1": 0.03}
        doc_map = {"doc-1": {"doc_id": "doc-1", "original": True}}
        sources = {"doc-1": {"source1"}}

        results = format_rrf_results(scores, doc_map, sources)

        # Modify result
        results[0]["modified"] = True

        # Original should be unchanged
        assert "modified" not in doc_map["doc-1"]
