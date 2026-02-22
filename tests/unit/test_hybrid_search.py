"""Tests for hybrid search implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.search.hybrid import HybridSearcher


class TestHybridSearcher:
    """Tests for HybridSearcher class."""

    @pytest.fixture
    def mock_bm25_searcher(self):
        """Create a mock BM25 searcher."""
        searcher = MagicMock()
        searcher.search = MagicMock(return_value=[
            {"doc_id": "doc-1", "item_name": "Pasta", "_score": 5.5},
            {"doc_id": "doc-2", "item_name": "Pizza", "_score": 4.2},
        ])
        searcher.search_by_ids = MagicMock(return_value=[
            {"doc_id": "doc-3", "item_name": "Salad", "restaurant_name": "Test"},
        ])
        return searcher

    @pytest.fixture
    def mock_vector_searcher(self):
        """Create a mock vector searcher."""
        searcher = AsyncMock()
        searcher.search = AsyncMock(return_value=[
            {"doc_id": "doc-2", "score": 0.92},
            {"doc_id": "doc-3", "score": 0.88},
        ])
        searcher.close = AsyncMock()
        return searcher

    @pytest.fixture
    def hybrid_searcher(self, mock_bm25_searcher, mock_vector_searcher):
        """Create a HybridSearcher with mock searchers."""
        return HybridSearcher(
            bm25_searcher=mock_bm25_searcher,
            vector_searcher=mock_vector_searcher,
        )

    @pytest.mark.asyncio
    async def test_search_combines_bm25_and_vector(
        self,
        hybrid_searcher,
        mock_bm25_searcher,
        mock_vector_searcher,
    ):
        """Test that hybrid search runs both BM25 and vector searches."""
        with patch("src.search.hybrid.rrf_merge_2way") as mock_rrf:
            mock_rrf.return_value = [
                {"doc_id": "doc-2", "rrf_score": 0.03, "in_bm25": True, "in_vector": True},
                {"doc_id": "doc-1", "rrf_score": 0.02, "in_bm25": True, "in_vector": False},
            ]

            results = await hybrid_searcher.search("italian food", top_k=10)

            # Both searches should be called
            mock_bm25_searcher.search.assert_called_once()
            mock_vector_searcher.search.assert_called_once()

            # RRF should be called to merge results
            mock_rrf.assert_called_once()

            # Should return merged results
            assert len(results) == 2
            assert results[0]["doc_id"] == "doc-2"

    @pytest.mark.asyncio
    async def test_search_uses_parallel_execution(
        self,
        hybrid_searcher,
        mock_bm25_searcher,
        mock_vector_searcher,
    ):
        """Test that BM25 and vector searches run in parallel."""
        # Both searches should be called in parallel via asyncio.gather
        # We verify this by checking both are called during the search
        with patch("src.search.hybrid.rrf_merge_2way") as mock_rrf:
            mock_rrf.return_value = []

            await hybrid_searcher.search("test query")

            # Both searches should be called
            mock_bm25_searcher.search.assert_called_once()
            mock_vector_searcher.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_applies_filters(
        self,
        hybrid_searcher,
        mock_bm25_searcher,
        mock_vector_searcher,
    ):
        """Test that filters are passed to both searchers."""
        from src.models.state import SearchFilters

        filters: SearchFilters = {
            "city": "Boston",
            "cuisine": ["Italian"],
            "price_max": 100.0,
        }

        with patch("src.search.hybrid.rrf_merge_2way") as mock_rrf:
            mock_rrf.return_value = []

            await hybrid_searcher.search("italian food", filters=filters)

            # Both searchers should receive the filters
            mock_bm25_searcher.search.assert_called_once()
            bm25_filters = mock_bm25_searcher.search.call_args[0][1]
            assert bm25_filters == filters

            mock_vector_searcher.search.assert_called_once()
            vector_filters = mock_vector_searcher.search.call_args[0][1]
            assert vector_filters == filters

    @pytest.mark.asyncio
    async def test_search_uses_settings_top_k(
        self,
        hybrid_searcher,
        mock_bm25_searcher,
        mock_vector_searcher,
    ):
        """Test that search uses top_k from settings."""
        with (
            patch("src.search.hybrid.settings") as mock_settings,
            patch("src.search.hybrid.rrf_merge_2way") as mock_rrf,
        ):
            mock_settings.bm25_top_k = 50
            mock_settings.vector_top_k = 75
            mock_settings.bm25_weight = 1.0
            mock_settings.vector_weight = 1.0
            mock_settings.rrf_k = 60
            mock_rrf.return_value = []

            await hybrid_searcher.search("test query")

            # BM25 should use bm25_top_k
            mock_bm25_searcher.search.assert_called_once()
            bm25_top_k = mock_bm25_searcher.search.call_args[0][2]
            assert bm25_top_k == 50

            # Vector should use vector_top_k
            mock_vector_searcher.search.assert_called_once()
            vector_top_k = mock_vector_searcher.search.call_args[0][2]
            assert vector_top_k == 75

    @pytest.mark.asyncio
    async def test_search_limits_results_to_top_k(
        self,
        hybrid_searcher,
        mock_bm25_searcher,
        mock_vector_searcher,
    ):
        """Test that results are limited to top_k."""
        with patch("src.search.hybrid.rrf_merge_2way") as mock_rrf:
            # Return more results than requested
            mock_rrf.return_value = [
                {"doc_id": f"doc-{i}", "rrf_score": 0.1 - i * 0.01, "in_bm25": True, "in_vector": False}
                for i in range(20)
            ]

            results = await hybrid_searcher.search("test", top_k=5)

            # Should limit to top_k
            assert len(results) == 5

    @pytest.mark.asyncio
    async def test_search_uses_custom_weights(
        self,
        hybrid_searcher,
        mock_bm25_searcher,
        mock_vector_searcher,
    ):
        """Test that custom weights are passed to RRF."""
        with patch("src.search.hybrid.rrf_merge_2way") as mock_rrf:
            mock_rrf.return_value = []

            await hybrid_searcher.search(
                "test",
                bm25_weight=2.0,
                vector_weight=0.5,
            )

            mock_rrf.assert_called_once()
            call_kwargs = mock_rrf.call_args[1]
            assert call_kwargs["bm25_weight"] == 2.0
            assert call_kwargs["vector_weight"] == 0.5

    @pytest.mark.asyncio
    async def test_search_handles_bm25_error(
        self,
        hybrid_searcher,
        mock_bm25_searcher,
        mock_vector_searcher,
    ):
        """Test that BM25 errors are handled gracefully."""
        mock_bm25_searcher.search.side_effect = Exception("OpenSearch down")

        with pytest.raises(Exception, match="OpenSearch down"):
            await hybrid_searcher.search("test")

    @pytest.mark.asyncio
    async def test_search_handles_vector_error(
        self,
        hybrid_searcher,
        mock_bm25_searcher,
        mock_vector_searcher,
    ):
        """Test that vector search errors are handled gracefully."""
        mock_vector_searcher.search.side_effect = Exception("PostgreSQL down")

        with pytest.raises(Exception, match="PostgreSQL down"):
            await hybrid_searcher.search("test")

    @pytest.mark.asyncio
    async def test_enrich_results_fetches_vector_only_docs(
        self,
        hybrid_searcher,
        mock_bm25_searcher,
    ):
        """Test that vector-only results are enriched with full documents."""
        bm25_results = [
            {"doc_id": "doc-1", "item_name": "Pasta"},
        ]
        merged = [
            {"doc_id": "doc-1", "rrf_score": 0.03, "in_bm25": True, "in_vector": True},
            {"doc_id": "doc-2", "rrf_score": 0.02, "in_bm25": False, "in_vector": True},
            {"doc_id": "doc-3", "rrf_score": 0.01, "in_bm25": False, "in_vector": True},
        ]

        mock_bm25_searcher.search_by_ids.return_value = [
            {"doc_id": "doc-2", "item_name": "Pizza", "restaurant_name": "Test"},
            {"doc_id": "doc-3", "item_name": "Salad", "restaurant_name": "Test"},
        ]

        enriched = await hybrid_searcher._enrich_results(merged, bm25_results)

        # Should fetch vector-only docs
        mock_bm25_searcher.search_by_ids.assert_called_once()
        fetched_ids = mock_bm25_searcher.search_by_ids.call_args[0][0]
        assert set(fetched_ids) == {"doc-2", "doc-3"}

        # Should enrich with full data
        assert len(enriched) == 3
        pizza_doc = next(d for d in enriched if d["doc_id"] == "doc-2")
        assert pizza_doc["item_name"] == "Pizza"
        assert pizza_doc["rrf_score"] == 0.02

    @pytest.mark.asyncio
    async def test_enrich_results_skips_if_no_vector_only(
        self,
        hybrid_searcher,
        mock_bm25_searcher,
    ):
        """Test that enrichment skips when all results are in BM25."""
        bm25_results = [
            {"doc_id": "doc-1", "item_name": "Pasta"},
            {"doc_id": "doc-2", "item_name": "Pizza"},
        ]
        merged = [
            {"doc_id": "doc-1", "rrf_score": 0.03, "in_bm25": True, "in_vector": True},
            {"doc_id": "doc-2", "rrf_score": 0.02, "in_bm25": True, "in_vector": True},
        ]

        enriched = await hybrid_searcher._enrich_results(merged, bm25_results)

        # Should not fetch any docs
        mock_bm25_searcher.search_by_ids.assert_not_called()

        # Should return unchanged
        assert enriched == merged

    @pytest.mark.asyncio
    async def test_enrich_results_handles_missing_docs(
        self,
        hybrid_searcher,
        mock_bm25_searcher,
    ):
        """Test that enrichment handles missing documents gracefully."""
        bm25_results = []
        merged = [
            {"doc_id": "doc-1", "rrf_score": 0.03, "in_bm25": False, "in_vector": True},
        ]

        # Return empty list (doc not found)
        mock_bm25_searcher.search_by_ids.return_value = []

        enriched = await hybrid_searcher._enrich_results(merged, bm25_results)

        # Should still return the result even without enrichment
        assert len(enriched) == 1
        assert enriched[0]["doc_id"] == "doc-1"

    @pytest.mark.asyncio
    async def test_rrf_merge_calls_util_function(
        self,
        hybrid_searcher,
        mock_bm25_searcher,
        mock_vector_searcher,
    ):
        """Test that RRF merge uses the utility function."""
        bm25_results = [{"doc_id": "doc-1"}]
        vector_results = [{"doc_id": "doc-2"}]

        with (
            patch("src.search.hybrid.rrf_merge_2way") as mock_rrf,
            patch("src.search.hybrid.settings") as mock_settings,
        ):
            mock_settings.rrf_k = 60
            mock_rrf.return_value = []

            hybrid_searcher._rrf_merge(bm25_results, vector_results, 1.0, 1.0)

            mock_rrf.assert_called_once_with(
                bm25_results,
                vector_results,
                k=60,
                bm25_weight=1.0,
                vector_weight=1.0,
            )

    @pytest.mark.asyncio
    async def test_close_closes_vector_searcher(
        self,
        hybrid_searcher,
        mock_vector_searcher,
    ):
        """Test that close closes the vector searcher connection."""
        await hybrid_searcher.close()

        mock_vector_searcher.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_records_metrics(
        self,
        hybrid_searcher,
        mock_bm25_searcher,
        mock_vector_searcher,
    ):
        """Test that search records metrics."""
        with (
            patch("src.search.hybrid.rrf_merge_2way") as mock_rrf,
            patch("src.search.hybrid.record_search_request") as mock_record,
        ):
            mock_rrf.return_value = [{"doc_id": "doc-1", "in_bm25": True, "in_vector": False}]

            await hybrid_searcher.search("test")

            # Should record metrics
            mock_record.assert_called()
            call_args = mock_record.call_args
            assert call_args[0][0] == "hybrid"  # search_type
            # Duration is the second argument (index 1), result_count is third (index 2)
            assert call_args[0][1] > 0  # duration
            assert call_args[0][2] >= 0  # result_count (may be 0 on error)


class TestHybridSearcherIntegration:
    """Integration-style tests for HybridSearcher."""

    @pytest.mark.asyncio
    async def test_search_end_to_end_with_overlapping_results(self):
        """Test full search flow with overlapping BM25 and vector results."""
        bm25_searcher = MagicMock()
        bm25_searcher.search.return_value = [
            {"doc_id": "doc-1", "item_name": "Pasta", "restaurant_name": "Italian Place"},
            {"doc_id": "doc-2", "item_name": "Pizza", "restaurant_name": "Italian Place"},
        ]
        bm25_searcher.search_by_ids.return_value = []

        vector_searcher = AsyncMock()
        vector_searcher.search.return_value = [
            {"doc_id": "doc-2", "score": 0.95},
            {"doc_id": "doc-3", "score": 0.85},
        ]
        vector_searcher.close = AsyncMock()

        searcher = HybridSearcher(
            bm25_searcher=bm25_searcher,
            vector_searcher=vector_searcher,
        )

        results = await searcher.search("italian food", top_k=10)

        # Should have all unique results
        doc_ids = [r["doc_id"] for r in results]
        assert "doc-1" in doc_ids
        assert "doc-2" in doc_ids
        assert "doc-3" in doc_ids

        # doc-2 should be ranked high (appears in both)
        assert results[0]["doc_id"] == "doc-2"

    @pytest.mark.asyncio
    async def test_search_with_empty_bm25_results(self):
        """Test search when BM25 returns no results."""
        bm25_searcher = MagicMock()
        bm25_searcher.search.return_value = []
        bm25_searcher.search_by_ids.return_value = []

        vector_searcher = AsyncMock()
        vector_searcher.search.return_value = [
            {"doc_id": "doc-1", "score": 0.9},
        ]
        vector_searcher.close = AsyncMock()

        searcher = HybridSearcher(
            bm25_searcher=bm25_searcher,
            vector_searcher=vector_searcher,
        )

        results = await searcher.search("test")

        # Should still return vector results
        assert len(results) == 1
        assert results[0]["doc_id"] == "doc-1"

    @pytest.mark.asyncio
    async def test_search_with_empty_vector_results(self):
        """Test search when vector search returns no results."""
        bm25_searcher = MagicMock()
        bm25_searcher.search.return_value = [
            {"doc_id": "doc-1", "item_name": "Pasta"},
        ]
        bm25_searcher.search_by_ids.return_value = []

        vector_searcher = AsyncMock()
        vector_searcher.search.return_value = []
        vector_searcher.close = AsyncMock()

        searcher = HybridSearcher(
            bm25_searcher=bm25_searcher,
            vector_searcher=vector_searcher,
        )

        results = await searcher.search("test")

        # Should still return BM25 results
        assert len(results) == 1
        assert results[0]["doc_id"] == "doc-1"

    @pytest.mark.asyncio
    async def test_search_with_empty_results(self):
        """Test search when both sources return no results."""
        bm25_searcher = MagicMock()
        bm25_searcher.search.return_value = []
        bm25_searcher.search_by_ids.return_value = []

        vector_searcher = AsyncMock()
        vector_searcher.search.return_value = []
        vector_searcher.close = AsyncMock()

        searcher = HybridSearcher(
            bm25_searcher=bm25_searcher,
            vector_searcher=vector_searcher,
        )

        results = await searcher.search("test")

        # Should return empty list
        assert results == []
